"""nsram.reservoir — High-Level NS-RAM Reservoir Computing API

One-line interface for creating and benchmarking NS-RAM reservoirs.
Handles parameter configuration, STP mode selection, and readout.

Example:
    >>> from nsram import NSRAMReservoir, rc_benchmark
    >>> res = NSRAMReservoir(128, stp='heterogeneous')
    >>> results = rc_benchmark(res)
"""

import numpy as np
from typing import Literal, Optional
from nsram.physics import DimensionlessParams, PRESETS
from nsram.network import NSRAMNetwork


class NSRAMReservoir:
    """NS-RAM Reservoir Computer.

    Args:
        N: Number of neurons (16 to 1M+)
        stp: Short-term plasticity mode:
            'none' — No STP, standard AdEx-LIF
            'std' — Short-term depression (low VG2, synapse mode)
            'stf' — Short-term facilitation (high VG2, neuron mode)
            'heterogeneous' — Mixed STD/STF via VG2 distribution (novel!)
        preset: Named parameter preset (overrides stp). See PRESETS dict.
        connectivity: 'sparse' (15%), 'small_world', 'dense'
        spectral_radius: Recurrent weight spectral radius (0.8-1.1)
        variability: Die-to-die variability (0=uniform, 0.1=typical, 0.2=high)
        bg_frac: Background drive as fraction of threshold (0.9-1.0)
        noise_sigma: Default noise amplitude
        seed: Random seed
        backend: 'auto', 'torch', 'numpy'
    """

    def __init__(self, N: int = 128, n_inputs: int = 1,
                 stp: Literal['none', 'std', 'stf', 'heterogeneous'] = 'heterogeneous',
                 preset: Optional[str] = None,
                 connectivity: str = 'sparse',
                 spectral_radius: float = 0.90,
                 variability: float = 0.10,
                 bg_frac: float = 0.95,
                 noise_sigma: float = 0.01,
                 seed: int = 42,
                 backend: str = 'auto'):

        self.noise_sigma = noise_sigma
        self.stp = stp

        # Start from preset or default
        if preset and preset in PRESETS:
            params = PRESETS[preset]
        else:
            params = DimensionlessParams()

        # Apply user overrides
        params.spectral_radius = spectral_radius
        params.variability = variability
        params.bg_frac = bg_frac
        params.connection_prob = 0.15 if connectivity == 'sparse' else 0.30

        # STP mode
        if stp == 'none':
            params.U = 0.0
            params.alpha_theta = 0.0
            params.alpha_weight = 0.0
        elif stp == 'std':
            params.U = 0.70
            params.tau_rec = 5.0
            params.tau_fac = 1.0
        elif stp == 'stf':
            params.U = 0.15
            params.tau_rec = 15.0
            params.tau_fac = 10.0
        # 'heterogeneous' uses default (VG2-mapped per neuron)

        self.net = NSRAMNetwork(N=N, n_inputs=n_inputs,
                                 connectivity=connectivity,
                                 params=params, backend=backend, seed=seed)
        self.N = N
        self.params = params

    def transform(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Pass input through reservoir, return (N, T) state matrix."""
        result = self.net.run(inputs, noise_sigma=self.noise_sigma, **kwargs)
        return result['states']

    def run(self, inputs: np.ndarray, **kwargs) -> dict:
        """Full run with all recorded data."""
        return self.net.run(inputs, noise_sigma=self.noise_sigma, **kwargs)

    def __repr__(self):
        return (f"NSRAMReservoir(N={self.N}, stp='{self.stp}', "
                f"backend='{self.net.backend}', device='{self.net.device}')")
