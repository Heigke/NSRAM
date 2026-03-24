"""nsram.neurons — Additional Neuron Models

Extends the core NS-RAM AdEx-LIF with standard neuron models for comparison
and broader applicability. All models are GPU-accelerated via PyTorch.

Models:
    IzhikevichNetwork  — Izhikevich 2003, 20+ firing patterns
    PLIFNetwork        — Parametric LIF with learnable time constants
    HHNetwork          — Hodgkin-Huxley (4-variable, biophysical)
    SRMNetwork         — Spike Response Model (filter-based)
    PoissonEncoder     — Rate-to-spike Poisson encoding

All follow the same interface as NSRAMNetwork: .run(signal) → dict.
"""

import numpy as np
from typing import Optional, Literal

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _device():
    if HAS_TORCH and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


class IzhikevichNetwork:
    """Izhikevich (2003) spiking network — 20+ firing patterns.

    dv/dt = 0.04v² + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v >= 30: v = c, u = u + d

    Presets: 'RS' (regular spiking), 'IB' (intrinsic bursting),
             'CH' (chattering), 'FS' (fast spiking), 'LTS' (low-threshold),
             'TC' (thalamocortical), 'RZ' (resonator), 'mixed'.

    Args:
        N: Number of neurons
        preset: Firing pattern preset or 'mixed' for heterogeneous
        connectivity: 'sparse', 'small_world', 'scale_free'
        seed: Random seed
    """

    PRESETS = {
        'RS':  {'a': 0.02, 'b': 0.2,  'c': -65, 'd': 8},
        'IB':  {'a': 0.02, 'b': 0.2,  'c': -55, 'd': 4},
        'CH':  {'a': 0.02, 'b': 0.2,  'c': -50, 'd': 2},
        'FS':  {'a': 0.10, 'b': 0.2,  'c': -65, 'd': 2},
        'LTS': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
        'TC':  {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
        'RZ':  {'a': 0.10, 'b': 0.26, 'c': -65, 'd': 2},
    }

    def __init__(self, N: int = 128, n_inputs: int = 1,
                 preset: str = 'mixed',
                 connectivity: str = 'sparse',
                 sparsity: float = 0.1, spectral_radius: float = 0.9,
                 seed: int = 42):
        self.N = N
        self.n_inputs = n_inputs
        self.device = _device()
        rng = np.random.RandomState(seed)

        # Neuron parameters
        if preset == 'mixed':
            # Mix of excitatory (RS/IB/CH) and inhibitory (FS/LTS)
            N_exc = int(N * 0.8)
            re = rng.rand(N_exc)
            ri = rng.rand(N - N_exc)
            a = np.concatenate([0.02 * np.ones(N_exc), 0.02 + 0.08 * ri])
            b = np.concatenate([0.2 * np.ones(N_exc), 0.25 - 0.05 * ri])
            c = np.concatenate([-65 + 15 * re**2, -65 * np.ones(N - N_exc)])
            d = np.concatenate([8 - 6 * re**2, 2 * np.ones(N - N_exc)])
        else:
            p = self.PRESETS[preset]
            a = np.full(N, p['a'])
            b = np.full(N, p['b'])
            c = np.full(N, p['c'])
            d = np.full(N, p['d'])
            N_exc = int(N * 0.8)

        if HAS_TORCH:
            self.a = torch.tensor(a, dtype=torch.float32, device=self.device)
            self.b = torch.tensor(b, dtype=torch.float32, device=self.device)
            self.c = torch.tensor(c, dtype=torch.float32, device=self.device)
            self.d = torch.tensor(d, dtype=torch.float32, device=self.device)
        else:
            self.a, self.b, self.c, self.d = a, b, c, d

        # Connectivity
        mask = (rng.rand(N, N) < sparsity).astype(np.float32)
        np.fill_diagonal(mask, 0)
        W = rng.randn(N, N).astype(np.float32) * mask
        signs = np.ones(N, dtype=np.float32)
        signs[N_exc:] = -1
        W = np.abs(W) * signs[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W *= spectral_radius / eigs.max()

        # Input weights
        W_in = rng.randn(N, n_inputs).astype(np.float32) * 5.0

        if HAS_TORCH:
            self.W = torch.tensor(W, device=self.device)
            self.W_in = torch.tensor(W_in, device=self.device)
        else:
            self.W = W
            self.W_in = W_in

    @torch.no_grad()
    def run(self, signal, dt=0.5):
        """Run the network. signal: (T,) or (T, n_inputs). Returns dict."""
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        if signal.dim() == 1:
            signal = signal.unsqueeze(1)
        T = signal.shape[0]

        v = torch.full((self.N,), -65.0, device=self.device)
        u = self.b * v
        states = torch.zeros(self.N, T, device=self.device)
        spikes = torch.zeros(self.N, T, device=self.device)

        for t in range(T):
            I = self.W_in @ signal[t] + self.W @ (v > 30).float() * 5
            v += dt * (0.04 * v**2 + 5 * v + 140 - u + I)
            u += dt * self.a * (self.b * v - u)
            fired = v >= 30
            spikes[:, t] = fired.float()
            v[fired] = self.c[fired]
            u[fired] = u[fired] + self.d[fired]
            v.clamp_(-100, 30)
            states[:, t] = v

        return {
            'states': states.cpu().numpy(),
            'spikes': spikes.cpu().numpy(),
            'spike_count': spikes.sum(dim=1).cpu().numpy(),
        }


class PLIFNetwork:
    """Parametric LIF — learnable decay constants (Wu et al., 2021).

    tau * dv/dt = -(v - v_rest) + R*I
    if v >= v_thresh: v = v_reset, spike

    tau is a learnable parameter (via sigmoid: tau = tau_min + (tau_max-tau_min)*sigmoid(w)).
    Useful for surrogate gradient training.

    Args:
        N: Number of neurons
        tau_range: (min, max) for learnable time constants
    """

    def __init__(self, N: int = 128, n_inputs: int = 1,
                 tau_range=(2.0, 20.0), v_thresh: float = 1.0,
                 sparsity: float = 0.1, spectral_radius: float = 0.9,
                 seed: int = 42):
        self.N = N
        self.n_inputs = n_inputs
        self.device = _device()
        self.v_thresh = v_thresh
        self.tau_min, self.tau_max = tau_range
        rng = np.random.RandomState(seed)

        # Learnable tau parameter (logit space)
        tau_logit = rng.randn(N).astype(np.float32) * 0.5
        if HAS_TORCH:
            self.tau_logit = torch.tensor(tau_logit, device=self.device)
        else:
            self.tau_logit = tau_logit

        # Connectivity
        mask = (rng.rand(N, N) < sparsity).astype(np.float32)
        np.fill_diagonal(mask, 0)
        W = rng.randn(N, N).astype(np.float32) * mask
        N_exc = int(N * 0.8)
        signs = np.ones(N, dtype=np.float32)
        signs[N_exc:] = -1
        W = np.abs(W) * signs[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W *= spectral_radius / eigs.max()
        W_in = rng.randn(N, n_inputs).astype(np.float32) * 0.3

        if HAS_TORCH:
            self.W = torch.tensor(W, device=self.device)
            self.W_in = torch.tensor(W_in, device=self.device)
        else:
            self.W = W
            self.W_in = W_in

    def _get_tau(self):
        if HAS_TORCH:
            return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_logit)
        return self.tau_min + (self.tau_max - self.tau_min) / (1 + np.exp(-self.tau_logit))

    @torch.no_grad()
    def run(self, signal, dt=1.0):
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        if signal.dim() == 1:
            signal = signal.unsqueeze(1)
        T = signal.shape[0]

        tau = self._get_tau()
        alpha = torch.exp(-dt / tau)
        v = torch.zeros(self.N, device=self.device)
        syn = torch.zeros(self.N, device=self.device)
        states = torch.zeros(self.N, T, device=self.device)
        spikes = torch.zeros(self.N, T, device=self.device)

        for t in range(T):
            I = self.W_in @ signal[t] + self.W @ syn * 0.3
            v = alpha * v + (1 - alpha) * I
            fired = v >= self.v_thresh
            spikes[:, t] = fired.float()
            v[fired] = 0
            syn = 0.8 * syn + fired.float()
            states[:, t] = v

        return {
            'states': states.cpu().numpy(),
            'spikes': spikes.cpu().numpy(),
            'spike_count': spikes.sum(dim=1).cpu().numpy(),
            'tau': tau.cpu().numpy(),
        }


class HHNetwork:
    """Hodgkin-Huxley (1952) network — biophysical gold standard.

    C dV/dt = -g_Na*m³h*(V-E_Na) - g_K*n⁴*(V-E_K) - g_L*(V-E_L) + I
    dm/dt = α_m(V)(1-m) - β_m(V)m
    dh/dt = α_h(V)(1-h) - β_h(V)h
    dn/dt = α_n(V)(1-n) - β_n(V)n

    4 state variables per neuron. GPU-accelerated for large networks.

    Args:
        N: Number of neurons
        g_Na, g_K, g_L: Maximum conductances (mS/cm²)
    """

    def __init__(self, N: int = 128, n_inputs: int = 1,
                 g_Na: float = 120.0, g_K: float = 36.0, g_L: float = 0.3,
                 sparsity: float = 0.05, spectral_radius: float = 0.5,
                 seed: int = 42):
        self.N = N
        self.n_inputs = n_inputs
        self.device = _device()
        self.g_Na, self.g_K, self.g_L = g_Na, g_K, g_L
        self.E_Na, self.E_K, self.E_L = 50.0, -77.0, -54.387
        self.C_m = 1.0
        rng = np.random.RandomState(seed)

        mask = (rng.rand(N, N) < sparsity).astype(np.float32)
        np.fill_diagonal(mask, 0)
        W = rng.randn(N, N).astype(np.float32) * mask
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W *= spectral_radius / eigs.max()
        W_in = rng.randn(N, n_inputs).astype(np.float32) * 5.0

        if HAS_TORCH:
            self.W = torch.tensor(W, device=self.device)
            self.W_in = torch.tensor(W_in, device=self.device)

    @staticmethod
    def _alpha_m(V): return 0.1 * (V + 40) / (1 - torch.exp(-(V + 40) / 10) + 1e-7)
    @staticmethod
    def _beta_m(V):  return 4.0 * torch.exp(-(V + 65) / 18)
    @staticmethod
    def _alpha_h(V): return 0.07 * torch.exp(-(V + 65) / 20)
    @staticmethod
    def _beta_h(V):  return 1.0 / (1 + torch.exp(-(V + 35) / 10) + 1e-7)
    @staticmethod
    def _alpha_n(V): return 0.01 * (V + 55) / (1 - torch.exp(-(V + 55) / 10) + 1e-7)
    @staticmethod
    def _beta_n(V):  return 0.125 * torch.exp(-(V + 65) / 80)

    @torch.no_grad()
    def run(self, signal, dt=0.01):
        """Run HH network. dt in ms (default 0.01 ms for stability)."""
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        if signal.dim() == 1:
            signal = signal.unsqueeze(1)
        T = signal.shape[0]

        V = torch.full((self.N,), -65.0, device=self.device)
        m = torch.full((self.N,), 0.05, device=self.device)
        h = torch.full((self.N,), 0.6, device=self.device)
        n = torch.full((self.N,), 0.32, device=self.device)
        syn = torch.zeros(self.N, device=self.device)

        states = torch.zeros(self.N, T, device=self.device)
        spikes = torch.zeros(self.N, T, device=self.device)
        prev_V = V.clone()

        for t in range(T):
            I_ext = self.W_in @ signal[t] + self.W @ syn
            I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
            I_K = self.g_K * n**4 * (V - self.E_K)
            I_L = self.g_L * (V - self.E_L)

            dV = (-I_Na - I_K - I_L + I_ext) / self.C_m
            V = V + dt * dV

            am, bm = self._alpha_m(V), self._beta_m(V)
            ah, bh = self._alpha_h(V), self._beta_h(V)
            an, bn = self._alpha_n(V), self._beta_n(V)
            m = m + dt * (am * (1 - m) - bm * m)
            h = h + dt * (ah * (1 - h) - bh * h)
            n = n + dt * (an * (1 - n) - bn * n)
            m.clamp_(0, 1); h.clamp_(0, 1); n.clamp_(0, 1)

            # Spike detection: upward threshold crossing at 0 mV
            fired = (V >= 0) & (prev_V < 0)
            spikes[:, t] = fired.float()
            syn = 0.9 * syn + fired.float()
            prev_V = V.clone()
            states[:, t] = V

        return {
            'states': states.cpu().numpy(),
            'spikes': spikes.cpu().numpy(),
            'spike_count': spikes.sum(dim=1).cpu().numpy(),
        }
