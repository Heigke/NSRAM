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

__version__ = "0.11.1"

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

# Device Characterization
from nsram.characterize import (
    simulate_pulse_response, firing_frequency_map,
    simulate_ltp_ltd, paired_pulse_ratio,
    simulate_retention, simulate_endurance,
    energy_per_spike, energy_comparison_table,
    simulate_voltage_ramp, sweep_rate_dependence,
    bulk_current_polynomial, fit_bulk_polynomial,
    deep_nwell_iv, ei_input_neuron,
    frequency_encode_image,
)

# Fitting & Comparison
from nsram.fitting import (
    fit_iv_curve, fit_iv_family, fit_retention,
    fit_bsim4_impact, fit_bsim4_family,
    monte_carlo, technology_comparison, compare_model_experiment,
    TECH_COMPARISON,
)

# BSIM4 — native impact-ionization + body-bias 2T floating-body model
from nsram.bsim4 import (
    BSIM4Params, BSIM4_PRESETS,
    vth_bsim4, drain_current_bsim,
    impact_ionization_bsim4,
    gidl_current, gisl_current, body_diode_current,
    body_charge_ode_bsim4,
    TwoTransistorCell,
    from_device_params,
)

# BEAM — Byte-level Embodied Associative Memory
from nsram.beam import BEAMModel

# Physics
from nsram.physics import (
    DeviceParams, DimensionlessParams, PRESETS,
    breakdown_voltage, avalanche_current,
    charge_capture_rate, srh_trapping_ode,
    vcb_self_oscillation, thermal_voltage,
    body_charge_ode,
)
