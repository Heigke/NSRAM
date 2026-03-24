"""nsram — Neuro-Synaptic RAM Simulator

First open-source Python library for NS-RAM floating-body transistor
neuron simulation with charge-trapping synaptic plasticity.

Based on: Pazos et al., "Synaptic and neural behaviours in a standard
silicon transistor", Nature 640, 69-76 (2025).

Quick start:
    >>> from nsram import NSRAMReservoir, rc_benchmark
    >>> res = NSRAMReservoir(128, stp='heterogeneous')
    >>> results = rc_benchmark(res)

Three fidelity levels:
    Level 1: NSRAMNeuron — single cell, full physics ODE (scipy)
    Level 2: NSRAMNetwork — vectorized, compact model (numpy/torch)
    Level 3: NSRAMReservoir — high-level RC API with benchmarks

Backends: NumPy (CPU), PyTorch (NVIDIA CUDA / AMD ROCm / Apple MPS)
"""

__version__ = "0.5.0"

# Core
from nsram.neuron import NSRAMNeuron
from nsram.network import NSRAMNetwork
from nsram.reservoir import NSRAMReservoir

# Benchmarks
from nsram.benchmarks import (
    rc_benchmark, xor_accuracy, memory_capacity, narma_prediction,
    waveform_classification, mackey_glass, kernel_rank,
    generalization_rank, nonlinear_memory_capacity,
)

# Learning rules
from nsram.learning import STDP, RewardSTDP, VoltageSTDP, HomeostaticPlasticity
from nsram.onchip_learning import ForwardForward, EquilibriumPropagation, Eprop

# Alternative neuron models
from nsram.neurons import IzhikevichNetwork, PLIFNetwork, HHNetwork

# Encoding / Decoding
from nsram.encoding import (
    rate_encode, latency_encode, delta_encode, population_encode, phase_encode,
    rate_decode, ttfs_decode, population_decode,
)

# Analysis
from nsram.analysis import (
    firing_rate, isi_statistics, fano_factor, correlation_matrix,
    synchrony_index, spike_entropy, transfer_entropy,
    avalanche_analysis, effective_dimension, lyapunov_estimate,
    raster_plot, isi_histogram,
)

# Vision
from nsram.vision import NSRAMClassifier

# Export
from nsram.export import to_brian2, to_nestml, to_spice_subcircuit
from nsram.spice import NSRAMCell, simulate_iv_curve

# Physics
from nsram.physics import (
    DeviceParams, DimensionlessParams, PRESETS,
    breakdown_voltage, avalanche_current,
    charge_capture_rate, srh_trapping_ode,
    vcb_self_oscillation, thermal_voltage,
    body_charge_ode,
)
