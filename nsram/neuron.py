"""nsram.neuron — Single NS-RAM Neuron Simulation

For detailed single-cell studies, parameter extraction, and validation
against SPICE/TCAD. Uses scipy ODE integration for accuracy.

For network-level simulation, use nsram.network instead (vectorized).

Example:
    >>> from nsram.neuron import NSRAMNeuron
    >>> neuron = NSRAMNeuron(Vg1=0.35, Vg2=0.40)
    >>> trace = neuron.simulate(duration=1e-3, I_ext=1e-9)
    >>> print(f"Spikes: {trace['n_spikes']}")
"""

import numpy as np
from typing import Optional, Callable
from nsram.physics import (DeviceParams, breakdown_voltage, avalanche_current,
                            vcb_self_oscillation, charge_capture_rate,
                            srh_trapping_ode, threshold_modulation, thermal_voltage)


class NSRAMNeuron:
    """Single NS-RAM 2T cell with full device physics.

    Simulates the complete ODE system:
      C × dVm/dt = I_aval(Vcb, Vg1, T) + I_leak(Vm) + I_ext
      dQ/dt = k_cap(Vg2) × (1-Q) × rate - k_em × Q

    With spike detection, refractory period, and charge trapping.
    """

    def __init__(self, Vg1: float = 0.35, Vg2: float = 0.40,
                 device: Optional[DeviceParams] = None,
                 temperature: float = 300.0):
        self.device = device or DeviceParams()
        self.Vg1 = Vg1
        self.Vg2 = Vg2
        self.T = temperature
        self.k_cap = float(charge_capture_rate(Vg2))
        self.reset()

    def reset(self):
        """Reset all state variables."""
        self.Vm = 0.0
        self.Q = 0.0
        self.refrac_timer = 0.0
        self.spike_times = []
        self.total_energy = 0.0

    def simulate(self, duration: float = 1e-3, dt: float = 1e-7,
                 I_ext: float = 0.0,
                 I_ext_fn: Optional[Callable] = None,
                 noise_sigma: float = 0.0) -> dict:
        """Run time-domain simulation.

        Args:
            duration: Simulation time (seconds)
            dt: Timestep (seconds). Default 100ns.
            I_ext: Constant external current (A)
            I_ext_fn: Time-varying current function(t) → Amps
            noise_sigma: Current noise std (A)

        Returns:
            dict with: t, Vm, Q, spikes, n_spikes, spike_times, energy
        """
        p = self.device
        n_steps = int(duration / dt)
        t_arr = np.linspace(0, duration, n_steps)

        Vm_arr = np.zeros(n_steps)
        Q_arr = np.zeros(n_steps)
        spike_arr = np.zeros(n_steps, dtype=bool)

        Vm = self.Vm
        Q = self.Q
        refrac = self.refrac_timer
        spike_rate = 0.0

        for i in range(n_steps):
            t = t_arr[i]

            # External current
            I_e = I_ext
            if I_ext_fn is not None:
                I_e = I_ext_fn(t)

            # Vcb self-oscillation
            Vcb = float(vcb_self_oscillation(t, amplitude=p.Vmem_linear_high,
                                               frequency=200e3))

            # Avalanche current
            I_aval = float(avalanche_current(Vcb, self.Vg1, self.T, p.Is))

            # Leak
            I_leak = -p.g_leak * (Vm - p.V_leak_rest)

            # Noise
            I_noise = noise_sigma * np.random.randn() if noise_sigma > 0 else 0.0

            # Charge trapping
            dQ = float(srh_trapping_ode(Q, spike_rate, self.k_cap, 200.0))
            Q = np.clip(Q + dQ * dt, 0, 1)
            delta_Vth = float(threshold_modulation(Q, 0.5))
            V_th = max(p.V_thresh + delta_Vth, 0.1)

            # Integration
            if refrac > 0:
                refrac -= dt
            else:
                dVm = (I_aval + I_leak + I_e + I_noise) / p.C_mem
                Vm += dVm * dt
                Vm = np.clip(Vm, -1.0, 5.0)

                # Spike detection
                if Vm >= V_th:
                    spike_arr[i] = True
                    self.spike_times.append(t)
                    Vm *= 0.2  # Partial reset
                    refrac = p.t_refrac
                    spike_rate += 100.0
                    self.total_energy += p.E_spike

            spike_rate *= np.exp(-dt / 1e-3)
            Vm_arr[i] = Vm
            Q_arr[i] = Q

        self.Vm = Vm
        self.Q = Q
        self.refrac_timer = refrac

        return {
            't': t_arr,
            'Vm': Vm_arr,
            'Q': Q_arr,
            'spikes': spike_arr,
            'n_spikes': int(spike_arr.sum()),
            'spike_times': np.array(self.spike_times),
            'energy_J': self.total_energy,
        }

    def iv_curve(self, Vds_range=(0, 3.5), n_points=200) -> dict:
        """Compute static I-V curve.

        Returns:
            dict with Vds, Id arrays
        """
        Vds = np.linspace(Vds_range[0], Vds_range[1], n_points)
        Id = avalanche_current(Vds, self.Vg1, self.T, self.device.Is)
        return {'Vds': Vds, 'Id': Id}

    def __repr__(self):
        bvpar = float(breakdown_voltage(self.Vg1, self.T))
        return (f"NSRAMNeuron(Vg1={self.Vg1:.3f}V, Vg2={self.Vg2:.3f}V, "
                f"BVpar={bvpar:.3f}V, T={self.T:.0f}K)")
