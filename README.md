# nsram — Neuro-Synaptic RAM Simulator

[![PyPI](https://img.shields.io/pypi/v/nsram)](https://pypi.org/project/nsram/)
[![Python](https://img.shields.io/pypi/pyversions/nsram)](https://pypi.org/project/nsram/)
[![License](https://img.shields.io/pypi/l/nsram)](https://github.com/enimble/nsram/blob/main/LICENSE)

The first open-source Python library for simulating NS-RAM floating-body transistor neurons with charge-trapping synaptic plasticity.

Based on: Pazos et al., *"Synaptic and neural behaviours in a standard silicon transistor"*, Nature 640, 69-76 (2025).

### New in v0.11 — BSIM4-native 2T floating-body model

The firing mechanism can now be driven directly from **BSIM4 impact ionization** (§6.1, `ALPHA0` / `BETA0`) and **body-bias** `Vth(Vbs)` (§2.2, `K1` / `K2`), with no breakdown-voltage (`BVpar`) control parameter. Drop in measured I-V curves, extract `ALPHA0` / `BETA0`, and the 2T spiking dynamics follow. See the *BSIM4-Native 2T Model* section below. The legacy Chynoweth / `BVpar` path is unchanged.

## Installation

```bash
pip install nsram              # CPU only (numpy + scipy)
pip install nsram[gpu]         # GPU acceleration (PyTorch CUDA/ROCm/MPS)
pip install nsram[all]         # Everything (GPU + plotting)
```

## Quickstart

```python
from nsram import NSRAMReservoir, rc_benchmark

# Create 128-neuron reservoir with heterogeneous short-term plasticity
res = NSRAMReservoir(N=128, stp='heterogeneous')

# Run reservoir computing benchmarks
results = rc_benchmark(res)
# XOR-1: 90.3%, MC: 2.29, NARMA-10: 0.52, Wave-4: 92.5%
```

## Key Results

| Benchmark | NS-RAM (N=2000) | ESN (tanh) | Izhikevich | PLIF |
|-----------|-----------------|------------|------------|------|
| **XOR-1** | **97.0%** | 50.5% | 50.2% | 50.5% |
| **Waveform-4** | **97.6%** | 69.8% | 34.2% | 42.6% |
| **NARMA-10** | **0.358** | 0.000 | 0.000 | 0.000 |
| **Memory Capacity** | 3.38 | **4.13** | 0.02 | 1.35 |
| **Kernel Rank** | **230** | 12 | 213 | 3 |
| **MNIST** (5K neurons) | **96.75%** | — | — | — |

## What is NS-RAM?

NS-RAM (Neuro-Synaptic Random Access Memory) is a standard CMOS floating-body transistor that exhibits both neuron-like spiking and synapse-like plasticity from device physics alone — no special materials or processes required.

**Key device physics simulated:**
- Impact ionization avalanche (Chynoweth model): `I_aval = I0 × exp((Vcb - BVpar) / Vt)`
- Breakdown voltage control: `BVpar = 3.5 - 1.5 × Vg1` (gate-tunable, from Pazos SPICE)
- Temperature dependence: `BVpar(T) = BVpar × (1 - 21.3μ × ΔT)`
- SRH charge trapping: `dQ/dt = k_cap(Vg2) × (1-Q) × rate - k_em × Q`
- VG2-controlled mode switching (neuron ↔ synapse)
- **BSIM4-native mode (v0.11+):** impact ionization `Iii ∝ (Vds−Vdseff)·exp(−BETA0/(Vds−Vdseff))·Ids` (§6.1) and `Vth(Vbs) = VTH0 + K1·(√(Φs−Vbs) − √Φs) − K2·Vbs` (§2.2), with no `BVpar` control parameter

## Library Overview (v0.11.0)

**16 modules, 4.4K+ lines, 100+ public exports.**

### Core Simulation

| Module | Description |
|--------|-------------|
| `nsram.physics` | Full device physics — SPICE-matched avalanche, SRH trapping, body-charge ODE |
| `nsram.neuron` | Single-cell ODE simulation (scipy), IV curves, parameter extraction |
| `nsram.network` | GPU-accelerated NS-RAM spiking network (10K+ neurons) |
| `nsram.reservoir` | High-level reservoir computing API |
| `nsram.vision` | Batch GPU image classifier — 96.75% MNIST at 5K neurons |

### Alternative Neuron Models

| Model | Class | Description |
|-------|-------|-------------|
| Izhikevich | `IzhikevichNetwork` | 20+ firing patterns (RS, IB, CH, FS, LTS, TC, RZ, mixed) |
| Parametric LIF | `PLIFNetwork` | Learnable time constants (Wu et al. 2021) |
| Hodgkin-Huxley | `HHNetwork` | 4-variable biophysical gold standard |

```python
from nsram import IzhikevichNetwork, PLIFNetwork, HHNetwork

iz = IzhikevichNetwork(N=1000, preset='mixed')
result = iz.run(signal)  # → dict with 'states', 'spikes'
```

### Benchmarks

| Benchmark | Function | Reference |
|-----------|----------|-----------|
| Temporal XOR | `xor_accuracy()` | Delay-τ nonlinear memory |
| Memory Capacity | `memory_capacity()` | Jaeger 2001 |
| NARMA-N | `narma_prediction()` | Atiya & Parlos 2000 |
| Waveform Classification | `waveform_classification()` | N-class temporal |
| **Mackey-Glass** | `mackey_glass()` | Chaotic time series prediction |
| **Kernel Rank** | `kernel_rank()` | Nonlinear transformation capacity |
| **Generalization Rank** | `generalization_rank()` | Generalization vs memorization |
| **Nonlinear Memory** | `nonlinear_memory_capacity()` | Memory × nonlinearity tradeoff |

```python
from nsram import mackey_glass, kernel_rank, nonlinear_memory_capacity

mg_r2 = mackey_glass(states, washout=500, tau=17)
kr = kernel_rank(states, washout=500)
nmc = nonlinear_memory_capacity(states, inputs, washout=500)
```

### Spike Encoding & Decoding

| Encoder | Function | Description |
|---------|----------|-------------|
| Rate (Poisson) | `rate_encode()` | Firing probability ∝ input value |
| Latency | `latency_encode()` | Time-to-first-spike |
| Delta | `delta_encode()` | DVS-style change detection |
| Population | `population_encode()` | Gaussian receptive fields |
| Phase | `phase_encode()` | Phase-of-firing coding |

```python
from nsram import rate_encode, latency_encode, population_encode

spikes = rate_encode(image, n_steps=100, gain=1.0)
ttfs = latency_encode(features, n_steps=50, tau=5.0)
pop = population_encode(signal, n_neurons=20)
```

### Analysis & Visualization

| Tool | Function | Description |
|------|----------|-------------|
| Firing rate | `firing_rate()` | Per-neuron, sliding window |
| ISI statistics | `isi_statistics()` | Mean, CV, burst ratio |
| Fano factor | `fano_factor()` | Spike count variability |
| Correlation | `correlation_matrix()` | Pairwise spike correlation |
| Synchrony | `synchrony_index()` | Population synchrony measure |
| Entropy | `spike_entropy()` | Shannon entropy of spike trains |
| Transfer entropy | `transfer_entropy()` | Directed information flow |
| Avalanche analysis | `avalanche_analysis()` | Power-law criticality test |
| Effective dimension | `effective_dimension()` | State space dimensionality |
| Lyapunov exponent | `lyapunov_estimate()` | Chaos detection |
| **Raster plot** | `raster_plot()` | Publication-quality spike raster |
| **ISI histogram** | `isi_histogram()` | ISI distribution visualization |

```python
from nsram import raster_plot, avalanche_analysis, effective_dimension

raster_plot(spikes, title='NS-RAM Network', save_path='raster.png')
aval = avalanche_analysis(spikes)
print(f"Critical: {aval['is_critical']}, α = {aval['size_exponent']:.2f}")
ed = effective_dimension(states)
```

### Learning Rules

**Hardware-realistic (tapeout-compatible):**

| Rule | Class | Reference |
|------|-------|-----------|
| STDP | `STDP` | Bi & Poo 1998 |
| Reward-STDP | `RewardSTDP` | Three-factor rule, eligibility traces |
| Voltage-STDP | `VoltageSTDP` | Clopath et al. 2010, uses body potential |
| Homeostatic | `HomeostaticPlasticity` | Intrinsic plasticity via charge trapping |
| Forward-Forward | `ForwardForward` | Hinton 2022, goodness = spike rate² |
| Equilibrium Propagation | `EquilibriumPropagation` | Scellier & Bengio 2017 |
| e-prop | `Eprop` | Bellec et al. 2020, eligibility propagation |

### Export & Integration

```python
from nsram import to_brian2, to_nestml, to_spice_subcircuit

brian2_eqs = to_brian2()           # Brian2 NeuronGroup equations
nestml_model = to_nestml()         # NESTML model for NEST HPC
spice_subckt = to_spice_subcircuit()  # SPICE .subckt for any simulator
```

### SPICE Bridge

```python
from nsram import NSRAMCell, simulate_iv_curve

cell = NSRAMCell()
cell.generate_netlist('nsram_1t.spice', Vg1=0.3)
iv_data = simulate_iv_curve(Vg1=0.3)  # Runs ngspice
```

## Novel Finding: Charge Trapping = Tsodyks-Markram STP

This library implements the analytical mapping between NS-RAM charge trapping and the Tsodyks-Markram short-term plasticity model:

| NS-RAM physics | TM-STP neuroscience |
|---|---|
| Q (trapped charge) | 1 − x (depleted resources) |
| k_cap(VG2) | U (utilization) |
| 1/k_em | τ_rec (recovery time) |
| ΔVth = −αQ | PSP amplitude modulation |

VG2 voltage controls the STP type: low VG2 → depression, high VG2 → facilitation.

## BSIM4-Native 2T Model (v0.11+)

An alternative path that drops the Zenodo avalanche-diode subcircuit and builds the floating-body dynamics directly on **BSIM4.3.0** equations: impact ionization (§6.1), `Vth(Vbs)` body effect (§2.2), optional GIDL/GISL (§6.2), and source/drain-body junction diodes (§10.1). The body charge balance

```
Cb · dVB/dt = Iii(Vds, Vgs, Vbs) + IGIDL − Ibs(Vbs) − Ibd(Vbd) − VB/Rb
```

is integrated externally — BSIM4 has no native floating-body mode (that's BSIMSOI), so `TwoTransistorCell` wraps the BSIM4 current sources with a Cb / Rb integrator and Vg2→Rb mode switch.

```python
from nsram import (
    BSIM4Params, BSIM4_PRESETS,
    TwoTransistorCell, impact_ionization_bsim4,
    fit_bsim4_impact,
)

# 1. Simulate a 2T cell — no BVpar, firing emerges from Iii + Vth(Vbs)
cell = TwoTransistorCell(bsim=BSIM4_PRESETS["ns_ram_180nm"])
res = cell.simulate(Vg1=0.9, Vg2=0.2, Vds=3.8, t_end=20e-6, dt=1e-8)
print(f"spikes: {len(res['spikes'])}, peak Iii: {res['Iii'].max():.2e} A")

# 2. Extract ALPHA0 / BETA0 from a measured I-V sweep
fit = fit_bsim4_impact(Vds_array, Isub_array, Vgs=1.0, Vbs=0.0)
# → fit['params'] is a BSIM4Params ready to drop into TwoTransistorCell
```

Presets: `ns_ram_180nm` (default), `ns_ram_180nm_hot`, `generic_65nm`. Override any BSIM4 parameter in `BSIM4Params(ALPHA0=..., BETA0=..., K1=..., ...)`.

### BSIM4 parameters (selected — full set in `nsram.bsim4.BSIM4Params`)

| Parameter | Default | BSIM4 §   | Role |
|-----------|---------|-----------|------|
| `ALPHA0`  | 6e-6    | §6.1 / A.5| first Iii coefficient (A·m/V) |
| `ALPHA1`  | 0       | §6.1 / A.5| length-scaling Iii coefficient |
| `BETA0`   | 22      | §6.1 / A.5| Chynoweth exponential coefficient (V) |
| `VTH0`    | 0.432   | §2.2 / A.3| long-channel Vth at Vbs=0 |
| `K1`      | 0.55    | §2.2 / A.3| first-order body coefficient (V^½) |
| `K2`      | 0.03    | §2.2 / A.3| second-order body coefficient |
| `AGIDL`   | 1e-10   | §6.2 / A.6| GIDL pre-exponential (mho) |
| `BGIDL`   | 2.3e9   | §6.2 / A.6| GIDL exponential (V/m) |
| `JSS`/`JSD` | 1e-4  | §10.1 / A.12 | body-diode reverse sat current density (A/m²) |
| `NJS`/`NJD` | 1.0   | §10.1 / A.12 | body-diode ideality factor |
| `Cb`      | 1 pF    | external  | floating-body capacitance |
| `Rb`      | 1 MΩ    | external  | bulk leakage resistance (Vg2-modulated in `TwoTransistorCell`) |

Round-trip fit of synthetic Iii data recovers ALPHA0/BETA0 to <1% (R² ≈ 0.998). Measured I-V CSVs drop straight into `fit_bsim4_impact(Vds, Isub, Vgs)`.

## Device Parameters

Default parameters from Pazos et al. SPICE model (Zenodo: 13843362):

| Parameter | Value | Source |
|---|---|---|
| BVpar₀ | 3.5 V | BJTparams.txt |
| dBVpar/dVg1 | -1.5 V/V | BJTparams.txt |
| Tbv1 | -21.3 μ/K | Davalanche.txt |
| Is | 1×10⁻¹⁶ A | BJTparams.txt |
| Bf | 50 | BJTparams.txt |
| Vth₀ (NMOS) | 0.432 V | PTM130bulk_lite.txt |
| Energy/spike | 21 fJ | Nature 640 |
| Cell area (1T) | 8 μm² | Nature 640 |
| Cell area (2T) | 17 μm² | Nature 640 |

## Backends

| Backend | Devices | Detection |
|---------|---------|-----------|
| NumPy | CPU (any) | Always available |
| PyTorch CUDA | NVIDIA GPUs | `torch.cuda.is_available()` |
| PyTorch ROCm | AMD GPUs (gfx1100, gfx1151, ...) | `torch.cuda.is_available()` via HIP |
| PyTorch MPS | Apple Silicon | `torch.backends.mps.is_available()` |

## Examples

See `examples/` for runnable scripts:

| Script | Description |
|--------|-------------|
| `quickstart.py` | Minimal hello-world |
| `bsim4_2t_floating_body.py` | **BSIM4-native 2T cell** — Iii sweep, neuron/synapse spiking, ALPHA0/BETA0 fit |
| `architecture_comparison.py` | 5 neuron models × 10 benchmarks × 3 scales |
| `mnist_scaling.py` | MNIST accuracy vs neuron count (96.75% at 8K) |
| `brain_arena.py` | 90K-neuron cortical brain foraging in 2D arena |
| `brain_plays_pong.py` | 50K-neuron brain playing Pong |
| `cortical_breakout.py` | 200K cortical brain on Breakout |
| `learning_benchmark.py` | STDP, R-STDP, V-STDP comparison |
| `onchip_mnist.py` | Forward-Forward vs EP vs e-prop on MNIST |
| `large_scale_test.py` | 8 publication-quality plots |
| `hero_figure.py` | Single hero image for publication |

## Citation

If you use this library, please cite:

```bibtex
@article{pazos2025nsram,
  title={Synaptic and neural behaviours in a standard silicon transistor},
  author={Pazos, Sebastian and others},
  journal={Nature},
  volume={640},
  pages={69--76},
  year={2025}
}
```

## License

Apache 2.0 — [Enimble Solutions AB](https://enimble.se)

**GitHub**: [github.com/Heigke/NSRAM](https://github.com/Heigke/NSRAM) | **PyPI**: [pypi.org/project/nsram](https://pypi.org/project/nsram/)
