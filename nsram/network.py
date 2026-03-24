"""nsram.network — Vectorized NS-RAM Network Simulation

Supports three fidelity levels:
  Level 1 (physics): scipy ODE solver, accurate but slow (1-100 neurons)
  Level 2 (compact): Euler integration, medium fidelity (100-10K neurons)
  Level 3 (dimensionless): GPU-accelerated AdEx-LIF (10K-1M+ neurons)

Backend detection:
  - PyTorch: CUDA (NVIDIA) or ROCm (AMD) GPUs, or CPU fallback
  - NumPy: Pure CPU, no GPU dependencies
  - Auto: tries torch GPU → torch CPU → numpy
"""

import numpy as np
from typing import Optional, Literal, Dict
from nsram.physics import DimensionlessParams, PRESETS


def _get_backend():
    """Detect best available backend."""
    try:
        import torch
        if torch.cuda.is_available():
            # Works for both NVIDIA CUDA and AMD ROCm (HIP)
            return torch, 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch, 'mps'  # Apple Silicon
        else:
            return torch, 'cpu'
    except ImportError:
        return None, 'cpu'


class NSRAMNetwork:
    """Vectorized NS-RAM spiking network.

    Creates N AdEx-LIF neurons with:
      - Per-neuron parameter heterogeneity (die-to-die variability)
      - Dale's law excitatory/inhibitory balance
      - Sparse/small-world/dense recurrent connectivity
      - Tsodyks-Markram STP from charge trapping physics
      - Multi-timescale state readout

    Example:
        >>> net = NSRAMNetwork(128)
        >>> result = net.run(np.random.randn(3000))
        >>> print(result['states'].shape)  # (128, 3000)
    """

    def __init__(self, N: int = 128, n_inputs: int = 1,
                 connectivity: Literal['sparse', 'small_world', 'dense',
                                       'scale_free', 'distance_dependent'] = 'sparse',
                 params: Optional[DimensionlessParams] = None,
                 backend: Literal['numpy', 'torch', 'auto'] = 'auto',
                 seed: int = 42,
                 dist_lambda: float = 2.0, dist_p0: float = 0.3):
        self.N = N
        self.n_inputs = n_inputs
        self.seed = seed

        if params is None:
            params = DimensionlessParams()
        self.params = params

        # Backend
        if backend == 'auto':
            self._torch, self.device = _get_backend()
            self.backend = 'torch' if self._torch is not None else 'numpy'
        elif backend == 'torch':
            self._torch, self.device = _get_backend()
            if self._torch is None:
                raise ImportError("PyTorch not installed. Use pip install nsram[gpu]")
            self.backend = 'torch'
        else:
            self._torch = None
            self.device = 'cpu'
            self.backend = 'numpy'

        # Generate per-neuron parameters with variability
        rng = np.random.RandomState(seed)
        v = params.variability

        def vary(base, frac=v):
            a = base * (1 + frac * rng.randn(N))
            return np.clip(a, base * 0.3, base * 3.0).astype(np.float32)

        self._tau = vary(params.tau, 0.15)
        self._theta = vary(params.theta, 0.05)
        self._t_ref = vary(params.t_refrac, 0.10)
        self._dT = vary(params.delta_T, 0.15)
        self._I_bg = vary(params.bg_frac, 0.10) * self._theta

        # STP parameters — heterogeneous VG2 mapped to U
        # Each neuron gets a different VG2 → different STP behavior
        self._Vg2 = (0.35 + 0.12 * rng.rand(N)).astype(np.float32)
        from nsram.physics import charge_capture_rate
        k_cap_arr = charge_capture_rate(self._Vg2, 1000.0, 0.40, 0.05)
        self._U_base = np.clip(k_cap_arr / 1000.0 * params.U * 2, 0.01, 0.95).astype(np.float32)
        self._tau_rec = vary(params.tau_rec, 0.30)
        self._tau_fac = ((1.0 - self._U_base) * params.tau_fac).astype(np.float32)
        self._alpha_theta = params.alpha_theta
        self._alpha_weight = params.alpha_weight

        # Input weights (per-neuron projection)
        self._W_in = (rng.randn(N, n_inputs) * params.input_scale).astype(np.float32)

        # Recurrent weight matrix with Dale's law
        N_exc = int(N * params.exc_frac)
        self._neuron_sign = np.ones(N, dtype=np.float32)
        self._neuron_sign[N_exc:] = -1.0

        self._W = self._build_weights(rng, connectivity,
                                        params.spectral_radius,
                                        params.connection_prob,
                                        dist_lambda, dist_p0)

        # Move to GPU if available
        if self.backend == 'torch':
            self._to_device()

    def _build_weights(self, rng, connectivity, sr, p_conn,
                       dist_lambda=2.0, dist_p0=0.3):
        N = self.N
        if connectivity == 'sparse':
            mask = rng.rand(N, N) < p_conn
            W = rng.randn(N, N).astype(np.float32) * mask
        elif connectivity == 'small_world':
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for k in [1, 2, 3, 4]:
                    W[i, (i+k)%N] = rng.randn() * 0.5
                    W[(i+k)%N, i] = rng.randn() * 0.5
                if rng.rand() < 0.10:
                    W[i, rng.randint(N)] = rng.randn()
        elif connectivity == 'dense':
            W = (rng.randn(N, N) / np.sqrt(N)).astype(np.float32)
        elif connectivity == 'scale_free':
            # Barabási-Albert preferential attachment
            m = max(1, int(p_conn * N * 0.5))
            m = min(m, N - 1)
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(min(m + 1, N)):
                for j in range(i + 1, min(m + 1, N)):
                    W[i, j] = rng.randn()
                    W[j, i] = rng.randn()
            degree = np.abs(W).sum(axis=1) + 1e-6
            for new in range(m + 1, N):
                probs = degree[:new] / degree[:new].sum()
                targets = rng.choice(new, size=min(m, new), replace=False, p=probs)
                for tgt in targets:
                    W[new, tgt] = rng.randn()
                    W[tgt, new] = rng.randn()
                degree[new] = len(targets)
                degree[targets] += 1
        elif connectivity == 'distance_dependent':
            # Neurons on 2D grid, p(i,j) = p0 * exp(-d/lambda)
            side = int(np.ceil(np.sqrt(N)))
            pos = np.array([(i % side, i // side) for i in range(N)], dtype=np.float32)
            diff = pos[:, None, :] - pos[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            p_matrix = dist_p0 * np.exp(-dist / max(dist_lambda, 0.01))
            np.fill_diagonal(p_matrix, 0)
            mask = rng.rand(N, N) < p_matrix
            W = rng.randn(N, N).astype(np.float32) * mask
            self._neuron_positions = pos
        else:
            raise ValueError(f"Unknown connectivity: {connectivity}")
        np.fill_diagonal(W, 0)
        W = np.abs(W) * self._neuron_sign[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W = (W * sr / eigs.max()).astype(np.float32)
        return W

    def _to_device(self):
        """Move all parameter arrays to GPU tensor."""
        torch = self._torch
        d = self.device
        self._t_tau = torch.tensor(self._tau, device=d)
        self._t_theta = torch.tensor(self._theta, device=d)
        self._t_tref = torch.tensor(self._t_ref, device=d)
        self._t_dT = torch.tensor(self._dT, device=d)
        self._t_Ibg = torch.tensor(self._I_bg, device=d)
        self._t_U = torch.tensor(self._U_base, device=d)
        self._t_trec = torch.tensor(self._tau_rec, device=d)
        self._t_tfac = torch.tensor(self._tau_fac, device=d)
        self._t_W = torch.tensor(self._W, device=d)
        self._t_Win = torch.tensor(self._W_in, device=d)

    def run(self, inputs: np.ndarray, noise_sigma: float = 0.01,
            syn_scale: float = 0.30, record_full: bool = False) -> Dict:
        """Run simulation.

        Args:
            inputs: (T,) or (T, n_inputs) input signal
            noise_sigma: Noise amplitude (0 = deterministic)
            syn_scale: Synaptic weight scale factor
            record_full: Record Vm, Q, x_stp, u_stp traces (slower)

        Returns:
            Dict with 'states' (N,T) and 'spikes' (N,T).
            If record_full: also 'Vm', 'Q', 'x_stp', 'u_stp'.
        """
        if self.backend == 'torch':
            return self._run_torch(inputs, noise_sigma, syn_scale, record_full)
        return self._run_numpy(inputs, noise_sigma, syn_scale, record_full)

    def _run_torch(self, inputs, noise_sigma, syn_scale, record_full):
        torch = self._torch
        d = self.device
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        T = len(inputs)
        N = self.N

        inp = torch.tensor(inputs, dtype=torch.float32, device=d)
        W = self._t_W; Win = self._t_Win
        tau = self._t_tau; theta_base = self._t_theta
        t_ref = self._t_tref; dT = self._t_dT; I_bg = self._t_Ibg
        U_base = self._t_U; t_rec = self._t_trec; t_fac = self._t_tfac

        Vm = torch.zeros(N, device=d)
        syn = torch.zeros(N, device=d)
        x_stp = torch.ones(N, device=d)
        u_stp = U_base.clone()
        Q = torch.zeros(N, device=d)
        refrac = torch.zeros(N, device=d)
        rate_est = torch.zeros(N, device=d)
        ft = torch.zeros(N, device=d); st = torch.zeros(N, device=d)

        states_out = torch.zeros(N, T, device=d)
        spk_out = torch.zeros(N, T, device=d)

        if record_full:
            Vm_out = torch.zeros(N, T, device=d)
            Q_out = torch.zeros(N, T, device=d)
            xstp_out = torch.zeros(N, T, device=d)
            ustp_out = torch.zeros(N, T, device=d)

        a_th = self._alpha_theta
        a_w = self._alpha_weight

        with torch.no_grad():
            for t in range(T):
                u = inp[t % T]
                I_in = Win @ u

                # STP-modulated synaptic current
                stp_mod = u_stp * x_stp * (1.0 + a_w * Q)
                I_syn = syn_scale * (W.T @ (syn * stp_mod))

                # Charge trapping → threshold shift
                dQ = U_base * (1.0 - Q) * rate_est * 0.01 - Q * 0.01
                Q = torch.clamp(Q + dQ, 0, 1)
                theta_eff = torch.clamp(theta_base - a_th * Q, min=0.1)

                # AdEx-LIF
                active = (refrac <= 0).float()
                leak = -Vm / tau
                exp_term = dT * torch.exp(torch.clamp((Vm - theta_eff) / dT.clamp(min=1e-6), -10, 5))
                Vm = Vm + active * (leak + I_bg + I_in + I_syn + exp_term)
                if noise_sigma > 0:
                    Vm = Vm + active * noise_sigma * torch.randn(N, device=d)
                Vm = torch.clamp(Vm, -2.0, 5.0)

                # Spike
                spiked = (Vm >= theta_eff) & (refrac <= 0)
                if spiked.any():
                    Vm[spiked] = 0.0
                    refrac[spiked] = t_ref[spiked]
                    syn[spiked] += 1.0
                    rate_est[spiked] += 5.0
                    spk_out[spiked, t] = 1.0
                    # STP on spike
                    u_stp[spiked] += U_base[spiked] * (1.0 - u_stp[spiked])
                    x_stp[spiked] -= u_stp[spiked] * x_stp[spiked]

                # Recovery
                syn *= 0.9
                rate_est *= 0.95
                x_stp += (1.0 - x_stp) / t_rec.clamp(min=0.5)
                u_stp += (U_base - u_stp) / t_fac.clamp(min=0.5)
                refrac = torch.clamp(refrac - 1.0, min=0.0)

                ft = 0.8 * ft + 0.2 * Vm
                st = 0.98 * st + 0.02 * Vm

                states_out[:, t] = (Vm + spk_out[:, t]
                                    + 0.3 * ft + 0.1 * st
                                    + 0.1 * x_stp + 0.05 * u_stp)

                if record_full:
                    Vm_out[:, t] = Vm; Q_out[:, t] = Q
                    xstp_out[:, t] = x_stp; ustp_out[:, t] = u_stp

        result = {
            'states': states_out.cpu().numpy(),
            'spikes': spk_out.cpu().numpy(),
        }
        if record_full:
            result.update({
                'Vm': Vm_out.cpu().numpy(),
                'Q': Q_out.cpu().numpy(),
                'x_stp': xstp_out.cpu().numpy(),
                'u_stp': ustp_out.cpu().numpy(),
            })
        return result

    def _run_numpy(self, inputs, noise_sigma, syn_scale, record_full):
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        T = len(inputs); N = self.N
        tau = self._tau; theta_base = self._theta
        t_ref = self._t_ref; dT_arr = self._dT; I_bg = self._I_bg
        U_base = self._U_base; t_rec = self._tau_rec; t_fac = self._tau_fac
        W = self._W; Win = self._W_in

        Vm = np.zeros(N, np.float32)
        syn = np.zeros(N, np.float32)
        x_stp = np.ones(N, np.float32)
        u_stp = U_base.copy()
        Q = np.zeros(N, np.float32)
        refrac = np.zeros(N, np.float32)
        rate_est = np.zeros(N, np.float32)
        ft = np.zeros(N, np.float32); st = np.zeros(N, np.float32)

        states_out = np.zeros((N, T), np.float32)
        spk_out = np.zeros((N, T), np.float32)

        a_th = self._alpha_theta; a_w = self._alpha_weight

        for t in range(T):
            u = inputs[t % T]
            I_in = Win @ u
            stp_mod = u_stp * x_stp * (1.0 + a_w * Q)
            I_syn = syn_scale * (W.T @ (syn * stp_mod))

            dQ = U_base * (1-Q) * rate_est * 0.01 - Q * 0.01
            Q = np.clip(Q + dQ, 0, 1)
            theta_eff = np.maximum(theta_base - a_th * Q, 0.1)

            active = (refrac <= 0).astype(np.float32)
            leak = -Vm / np.maximum(tau, 0.01)
            exp_term = dT_arr * np.exp(np.clip((Vm - theta_eff) / np.maximum(dT_arr, 1e-6), -10, 5))
            Vm += active * (leak + I_bg + I_in + I_syn + exp_term)
            if noise_sigma > 0:
                Vm += active * noise_sigma * np.random.randn(N).astype(np.float32)
            Vm = np.clip(Vm, -2.0, 5.0)

            spiked = (Vm >= theta_eff) & (refrac <= 0)
            if spiked.any():
                Vm[spiked] = 0.0
                refrac[spiked] = t_ref[spiked]
                syn[spiked] += 1.0; rate_est[spiked] += 5.0
                spk_out[spiked, t] = 1.0
                u_stp[spiked] += U_base[spiked] * (1 - u_stp[spiked])
                x_stp[spiked] -= u_stp[spiked] * x_stp[spiked]

            syn *= 0.9; rate_est *= 0.95
            x_stp += (1 - x_stp) / np.maximum(t_rec, 0.5)
            u_stp += (U_base - u_stp) / np.maximum(t_fac, 0.5)
            refrac = np.maximum(refrac - 1, 0)
            ft = 0.8*ft + 0.2*Vm; st = 0.98*st + 0.02*Vm

            states_out[:, t] = (Vm + spk_out[:, t]
                                + 0.3*ft + 0.1*st + 0.1*x_stp + 0.05*u_stp)

        return {'states': states_out, 'spikes': spk_out}

    @property
    def info(self) -> str:
        """Summary string."""
        n_conn = (self._W != 0).sum()
        return (f"NSRAMNetwork(N={self.N}, backend={self.backend}, "
                f"device={self.device}, connections={n_conn}, "
                f"sr={self.params.spectral_radius:.2f})")
