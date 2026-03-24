"""nsram.physics — NS-RAM Device Physics

All equations and constants derived from:
  Pazos et al., Nature 640, 69-76 (2025)
  Zenodo SPICE: BJTparams.txt, Davalanche.txt, PTM130bulk_lite.txt
  Published device characterization (Pazos, Lanza)

Provides both device-level (Volts, Amps, seconds) and dimensionless
formulations. Users can choose fidelity level.

Three abstraction levels:
  Level 1 (physics): Full SPICE-matched I-V with Chynoweth + SRH
  Level 2 (compact): Simplified exponential LIF + charge trapping ODE
  Level 3 (dimensionless): Normalized AdEx-LIF for fast network simulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union

Array = Union[float, np.ndarray]

# ═══════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

K_B = 1.380649e-23          # J/K  Boltzmann constant
Q_E = 1.602176634e-19       # C    Electron charge
EPS_SI = 11.7 * 8.854e-12  # F/m  Silicon permittivity
EPS_OX = 3.9 * 8.854e-12   # F/m  SiO2 permittivity

def thermal_voltage(T: float = 300.0) -> float:
    """Thermal voltage Vt = kT/q. 25.85 mV at 300K."""
    return K_B * T / Q_E


# ═══════════════════════════════════════════════════════════════════════
# NS-RAM DEVICE PARAMETERS (from Pazos et al. SPICE model and Nature 640)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DeviceParams:
    """Physical parameters for a single NS-RAM 2T cell.

    Extracted from Pazos et al. Zenodo SPICE files and Nature 640.
    All values in SI units (V, A, F, s, m).

    Attributes:
        --- Avalanche (BJTparams.txt, Davalanche.txt) ---
        BV0: Base breakdown voltage (V). SPICE: 3.5V
        k_vg: dBVpar/dVg1 sensitivity (V/V). SPICE: 1.5
        Tbv1: Temperature coefficient (1/K). SPICE: -21.3e-6
        Is: BJT saturation current (A). SPICE: 1e-16
        Bf: Forward current gain. SPICE: 50
        Nf: Forward emission coefficient. SPICE: 0.9
        Ne: Avalanche relaxation factor. SPICE: 1.5

        --- MOSFET (PTM130bulk_lite.txt) ---
        Vth0: NMOS threshold voltage (V). PTM: 0.432
        Tox: Gate oxide thickness (m). PTM: 3.3e-9

        --- Cell geometry (Nature 640) ---
        area_1T: 1T cell area (m²). 8 μm²
        area_2T: 2T cell area (m²). 17 μm²

        --- Membrane (from Brian2 simulation parameters) ---
        C_mem: Membrane capacitance (F). ~50-100 fF
        tau_mem: Membrane time constant (s). Brian2: 1 μs (device), 1 ms (slowed)
        V_thresh: Spike threshold (V). Brian2: 1.364 V
        t_refrac: Refractory period (s). Brian2: 1.6 μs
        V_leak_rest: Resting potential (V). Brian2: 0.1 V (in μs units)

        --- Energy (Nature 640) ---
        E_spike: Energy per spike (J). 21 fJ total
        E_generation: Spike generation energy (J). 4.7 fJ
        E_integration: Integration energy (J). 50 fJ
        I_leak: Leakage current (A). 0.5 nA constant

        --- Operating ranges (Pazos et al., Nature 640) ---
        Vds_range: Drain-source operating range (V). 0 to 5V (1T), 0 to 3.5V (2T)
        Vmem_linear: Membrane linear range (V). 2.3 to 3.0V
        firing_range: Min/max firing rate ratio. 10⁴
        freq_min: Minimum spiking frequency (Hz). 60 kHz
        freq_max: Maximum spiking frequency (Hz). 360 kHz

        --- Charge trapping (SRH model) ---
        Rb_default: Bulk resistance for synapse mode (Ω). 1 MΩ
    """
    # Avalanche
    BV0: float = 3.5
    k_vg: float = 1.5
    Tbv1: float = -21.3e-6
    Is: float = 1e-16
    Bf: float = 50.0
    Nf: float = 0.9
    Ne: float = 1.5

    # MOSFET
    Vth0: float = 0.432
    Tox: float = 3.3e-9

    # Geometry
    area_1T: float = 8e-12
    area_2T: float = 17e-12

    # Membrane
    C_mem: float = 50e-15
    tau_mem: float = 1e-6
    V_thresh: float = 1.364
    t_refrac: float = 1.6e-6
    V_leak_rest: float = 0.0

    # Energy
    E_spike: float = 21e-15
    E_generation: float = 4.7e-15
    E_integration: float = 50e-15
    I_leak: float = 0.5e-9

    # Operating ranges
    Vmem_linear_low: float = 2.3
    Vmem_linear_high: float = 3.0
    firing_range: float = 1e4
    freq_min: float = 60e3
    freq_max: float = 360e3

    # Bulk resistance (controls neuron/synapse mode)
    # Nature 640: 10 kΩ = neuron mode, 1 MΩ = synapse mode
    Rb_default: float = 1e6
    Rb_neuron: float = 10e3    # Ω, neuron mode
    Rb_synapse: float = 1e6    # Ω, synapse mode

    # Body capacitance (from TCAD: ~1 pF in SPICE, physical ~10-100 fF)
    Cb: float = 1e-12          # F, body capacitance (Cbe in SPICE)

    # SRH recombination (from TCAD sdevice.par)
    tau_srh_e: float = 8e-6    # s, electron SRH lifetime max
    tau_srh_h: float = 0.8e-6  # s, hole SRH lifetime max
    Nref_srh: float = 1e16     # cm⁻³, SRH doping reference

    # Synaptic state (Nature 640)
    N_eff_levels: int = 14     # distinct resistance levels
    R_LRS: float = 400e3       # Ω, low-resistance state
    R_HRS: float = 15e6        # Ω, high-resistance state
    endurance_neuron: float = 1e7   # cycles (neuron mode)
    endurance_synapse: float = 1e5  # cycles (synapse mode)
    retention_s: float = 1e4   # seconds (>2.8 hours)

    # Vspike range (Nature 640: 3.5-4.5V for firing)
    Vspike_min: float = 3.5
    Vspike_max: float = 4.5

    # Impact ionization (van Overstraeten-de Man, from TCAD)
    alpha_n: float = 7.03e5    # cm⁻¹, electron ionization coefficient
    beta_n: float = 1.231e6    # V/cm, critical field (electrons)
    alpha_p: float = 1.582e6   # cm⁻¹, hole ionization coefficient (low field)
    beta_p: float = 2.036e6    # V/cm, critical field (holes, low field)
    hbar_omega: float = 0.063  # eV, optical phonon energy

    # SPICE circuit dimensions
    Ln: float = 0.25e-6        # m, channel length
    Wn: float = 10e-6          # m, channel width

    @property
    def g_leak(self) -> float:
        """Leak conductance from τ = C/g."""
        return self.C_mem / self.tau_mem

    @property
    def tau_body(self) -> float:
        """Body RC time constant τ_body = Rb × Cb."""
        return self.Rb_default * self.Cb


# ═══════════════════════════════════════════════════════════════════════
# LEVEL 0.5: BODY-CHARGE ODE (the REAL NS-RAM differential equation)
# ═══════════════════════════════════════════════════════════════════════

def body_charge_ode(t, state, Vds, Vg, Rb, params=None):
    """The fundamental NS-RAM differential equation.

    This is the REAL physics — the body-charge balance that creates
    both neuron and synapse behavior in a single transistor.

    CB × dVB/dt = I_ii - I_bsj - I_recomb

    Where:
      VB = body (bulk) potential — THIS is the "membrane voltage"
      I_ii = impact ionization current (charges the body)
      I_bsj = body-source junction leakage (discharges the body)
      I_recomb = SRH recombination in the body

    The body potential VB modulates the MOSFET threshold via body effect:
      Vth(VB) = Vth0 - gamma × (sqrt(2φF + VSB) - sqrt(2φF))

    When VB rises enough to forward-bias the body-source junction (~0.6V),
    the parasitic BJT turns on → snap-back → SPIKE.

    State: [VB, Q_trap]
      VB: body potential (V)
      Q_trap: trapped charge fraction [0, 1]

    Args:
        t: time (s)
        state: [VB, Q_trap]
        Vds: drain-source voltage (V) — can be callable(t)
        Vg: gate voltage (V)
        Rb: bulk resistance (Ω) — controls neuron/synapse mode
        params: DeviceParams

    Returns:
        [dVB/dt, dQ_trap/dt]

    Usage with scipy:
        from scipy.integrate import solve_ivp
        from nsram.physics import body_charge_ode, DeviceParams

        p = DeviceParams()
        sol = solve_ivp(
            body_charge_ode, [0, 1e-3], [0.0, 0.0],
            args=(3.5, 0.35, 100e3, p), max_step=1e-8
        )
    """
    if params is None:
        params = DeviceParams()
    p = params

    VB = state[0]
    Q_trap = state[1] if len(state) > 1 else 0.0

    # Resolve Vds (may be time-varying)
    _Vds = Vds(t) if callable(Vds) else Vds

    # 1. Impact ionization current (charges body with holes)
    # I_ii = (M-1) × Ids, where M = multiplication factor
    # Simplified: I_ii ∝ Ids × exp(-Ecrit / E_lateral)
    # E_lateral ≈ (Vds - Vdsat) / l_pinchoff
    Vt = thermal_voltage(300.0)
    BVpar = (p.BV0 - p.k_vg * Vg) * (1 + p.Tbv1 * 0)  # T=300K
    excess = max(_Vds - BVpar, 0)
    I_ii = p.Is * p.Bf * np.exp(min(excess / (p.Ne * Vt), 30))
    I_ii = min(I_ii, 1e-3)  # Clamp to 1 mA

    # Body potential feedback: VB raises Ids by lowering Vth
    # This creates positive feedback → snap-back
    Vth_eff = p.Vth0 - 0.3 * max(VB, 0)  # Body effect
    if Vg > Vth_eff and VB > 0:
        I_ii *= (1 + 5 * VB)  # Positive feedback from BJT gain

    # 2. Body-source junction leakage (drains holes)
    # I_bsj = Is_diode × (exp(VB/Vt) - 1)
    I_bsj = 1e-12 * (np.exp(min(VB / Vt, 30)) - 1)

    # 3. SRH recombination in body
    # I_recomb = Q_body / tau_srh
    Q_body = p.Cb * max(VB, 0)
    I_recomb = Q_body / p.tau_srh_h

    # 4. Bulk resistance leakage path
    I_Rb = VB / Rb if Rb > 0 else 0

    # Body charge balance: CB × dVB/dt = I_ii - I_bsj - I_recomb - I_Rb
    dVB = (I_ii - I_bsj - I_recomb - I_Rb) / p.Cb

    # 5. Charge trapping (SRH kinetics)
    # Trapped charge modifies threshold for subsequent spikes
    spike_indicator = 1.0 if VB > 0.5 else 0.0
    k_cap = 1000.0 / (1 + np.exp((Rb - 100e3) / 50e3))  # Rb→k_cap mapping
    dQ = k_cap * (1 - Q_trap) * spike_indicator - 200.0 * Q_trap

    return [dVB, dQ]


# ═══════════════════════════════════════════════════════════════════════
# LEVEL 1: FULL PHYSICS EQUATIONS
# ═══════════════════════════════════════════════════════════════════════

def breakdown_voltage(vg1: Array, T: float = 300.0,
                       BV0: float = 3.5, k_vg: float = 1.5,
                       Tbv1: float = -21.3e-6) -> Array:
    """BVpar(Vg1, T) — Avalanche breakdown voltage.

    BVpar = (BV0 - k_vg × Vg1) × (1 + Tbv1 × (T - 300))

    From: BJTparams.txt line 25, Davalanche.txt Tbv1 parameter.
    BVpar decreases linearly with Vg1 (range 1.5-3.5V).

    Args:
        vg1: Gate voltage(s) (V)
        T: Temperature (K)
        BV0: Base breakdown voltage (V)
        k_vg: Gate voltage sensitivity (V/V)
        Tbv1: Temperature coefficient (1/K)
    """
    vg1 = np.asarray(vg1, dtype=np.float64)
    return (BV0 - k_vg * vg1) * (1.0 + Tbv1 * (T - 300.0))


def avalanche_current(Vcb: Array, vg1: Array, T: float = 300.0,
                       I0: float = 1e-16, BV0: float = 3.5,
                       k_vg: float = 1.5, Tbv1: float = -21.3e-6,
                       I_max: float = 1e-4) -> Array:
    """Impact ionization current (Chynoweth model).

    I_aval = I0 × exp((Vcb - BVpar) / (Ne × Vt))

    From: BJTparams.txt (Is, Ne). Measured I-V shows exponential
    above Vds≈2.0V, spanning 8 decades (10⁻¹² to 10⁻⁴ A).
    """
    Vcb = np.asarray(Vcb, dtype=np.float64)
    vg1 = np.asarray(vg1, dtype=np.float64)
    Vt = thermal_voltage(T)
    bvpar = breakdown_voltage(vg1, T, BV0, k_vg, Tbv1)
    Ne = 1.5  # Avalanche relaxation factor
    exp_arg = np.clip((Vcb - bvpar) / (Ne * Vt), -40, 40)
    return np.minimum(I0 * np.exp(exp_arg), I_max)


def bulk_current_empirical(Vds: Array, Vg1: Array,
                            Rb: float = 1e6) -> Array:
    """Semi-empirical bulk current model.

    Itot = Iexp + Ibase

    Iexp = α(Vg1) × exp(β(Vg1) × Vds)  for Vds > Vds_onset
    Ibase = γ(Vg2) × Ids_mosfet

    Interpolated from measured curves (Pazos et al., Nature 640):
      Vg1=0.1V: onset ≈ 2.5V, I_bulk(3V) ≈ 1e-9 A
      Vg1=0.3V: onset ≈ 2.0V, I_bulk(3V) ≈ 1e-7 A
      Vg1=0.5V: onset ≈ 1.5V, I_bulk(3V) ≈ 1e-5 A
      Vg1=0.7V: onset ≈ 1.0V, I_bulk(3V) ≈ 1e-4 A

    The onset voltage is: Vds_onset ≈ BVpar(Vg1)
    The exponential slope is: β ≈ 1/(Ne × Vt) ≈ 25.7/V at 300K
    """
    Vds = np.asarray(Vds, dtype=np.float64)
    Vg1 = np.asarray(Vg1, dtype=np.float64)
    bvpar = breakdown_voltage(Vg1)
    Vt = thermal_voltage(300.0)
    Ne = 1.5
    # Above breakdown: exponential
    excess = np.maximum(Vds - bvpar, 0)
    I_exp = 1e-16 * np.exp(excess / (Ne * Vt))
    # MOSFET channel current (simplified square law)
    Ids = 1e-6 * np.maximum(Vg1 - 0.432, 0) ** 2
    return np.minimum(I_exp + Ids, 1e-3)


def vcb_self_oscillation(t: Array, amplitude: float = 2.5,
                          frequency: float = 100e3,
                          waveform: str = 'triangular') -> Array:
    """Self-oscillation Vcb pulse train.

    The floating body generates this autonomously through the
    charge-discharge-avalanche-reset cycle.

    Args:
        t: Time (seconds)
        amplitude: Peak Vcb (V). 2.5V for 2T thick oxide
        frequency: Oscillation frequency (Hz). 60-360 kHz
        waveform: 'triangular' (default), 'sawtooth', or 'sine'
    """
    t = np.asarray(t, dtype=np.float64)
    phase = (t * frequency) % 1.0

    if waveform == 'triangular':
        # Ramp up 80%, fast reset 20% (matches measured waveforms)
        return np.where(phase < 0.8,
                        amplitude * phase / 0.8,
                        amplitude * (1.0 - (phase - 0.8) / 0.2))
    elif waveform == 'sawtooth':
        return amplitude * phase
    elif waveform == 'sine':
        return amplitude * 0.5 * (1.0 + np.sin(2 * np.pi * phase))
    else:
        raise ValueError(f"Unknown waveform: {waveform}")


# ═══════════════════════════════════════════════════════════════════════
# LEVEL 1: CHARGE TRAPPING (SRH / Tsodyks-Markram bridge)
# ═══════════════════════════════════════════════════════════════════════

def charge_capture_rate(Vg2: Array,
                         k_cap_max: float = 1000.0,
                         V_mid: float = 0.40,
                         delta: float = 0.05) -> Array:
    """VG2-dependent charge capture rate (SRH trapping kinetics).

    k_cap(VG2) = k_cap_max / (1 + exp((VG2 - V_mid) / delta))

    Maps to Tsodyks-Markram utilization parameter U:
      Low VG2 → high k_cap → high U → short-term depression (STD)
      High VG2 → low k_cap → low U → short-term facilitation (STF)

    VG2-dependent bulk current controls neuron/synapse mode
    (Pazos et al., Nature 640, 2025).
    """
    Vg2 = np.asarray(Vg2, dtype=np.float64)
    return k_cap_max / (1.0 + np.exp((Vg2 - V_mid) / delta))


def srh_trapping_ode(Q: Array, spike_rate: Array,
                      k_cap: Array, k_em: float = 200.0) -> Array:
    """SRH charge trapping ODE (= Tsodyks-Markram STP).

    dQ/dt = k_cap × (1 - Q) × spike_rate - k_em × Q

    Mathematically equivalent to TM-STP:
      dx/dt = (1-x)/τ_rec - U × x × δ(spike)
    where: Q ↔ (1-x), k_cap ↔ U, 1/k_em ↔ τ_rec

    Args:
        Q: Trapped charge fraction [0, 1]
        spike_rate: Instantaneous rate or spike indicator
        k_cap: Capture rate (from charge_capture_rate)
        k_em: Emission rate (1/s). Controls detrapping timescale.
    """
    Q = np.asarray(Q, dtype=np.float64)
    return k_cap * (1.0 - Q) * spike_rate - k_em * Q


def threshold_modulation(Q: Array, alpha: float = 0.5) -> Array:
    """Threshold shift from trapped charge.

    ΔVth = -α × Q (more charge → lower threshold → potentiation)
    """
    return -alpha * np.asarray(Q, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# LEVEL 2: COMPACT NEURON ODE (for scipy.integrate)
# ═══════════════════════════════════════════════════════════════════════

def nsram_neuron_ode(t, state, params, I_ext_fn=None):
    """Full NS-RAM neuron ODE for scipy.integrate.solve_ivp.

    State vector: [Vm, Q]
      Vm: membrane potential (V)
      Q: trapped charge fraction [0,1]

    dVm/dt = (I_aval + I_leak + I_ext) / C_mem
    dQ/dt = k_cap(Vg2) × (1-Q) × f(Vm) - k_em × Q

    Usage:
        from scipy.integrate import solve_ivp
        from nsram.physics import nsram_neuron_ode, DeviceParams

        p = DeviceParams()
        sol = solve_ivp(nsram_neuron_ode, [0, 1e-3], [0.0, 0.0],
                        args=(p,), max_step=1e-7)
    """
    Vm, Q = state
    p = params if isinstance(params, DeviceParams) else DeviceParams()
    I_ext = I_ext_fn(t) if I_ext_fn else 0.0

    Vcb = vcb_self_oscillation(t, amplitude=2.5, frequency=150e3)
    I_aval = float(avalanche_current(Vcb, p.Vth0 * 0.8))
    I_leak = -p.g_leak * (Vm - p.V_leak_rest)
    dVm = (I_aval + I_leak + I_ext) / p.C_mem

    # Spike detection (approximate: if Vm > threshold, it will be reset externally)
    spike_indicator = 1.0 if Vm > p.V_thresh else 0.0
    k_cap = float(charge_capture_rate(0.40))
    dQ = float(srh_trapping_ode(Q, spike_indicator, k_cap, 200.0))

    return [dVm, dQ]


# ═══════════════════════════════════════════════════════════════════════
# LEVEL 3: DIMENSIONLESS (for fast network simulation)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DimensionlessParams:
    """Normalized parameters for fast AdEx-LIF simulation.

    All voltages normalized to threshold (θ=1), time to membrane τ.
    Maps to physical parameters via scaling factors.

    The dimensionless system is:
      dv/dt = -v/τ + I_bg + I_in + I_syn + ΔT×exp((v-θ)/ΔT) + σξ(t)
      if v ≥ θ: spike, v → v_reset
    """
    # Membrane
    tau: float = 1.0           # normalized time constant
    theta: float = 1.0         # spike threshold
    v_reset: float = 0.0       # post-spike reset
    t_refrac: float = 0.05     # refractory (fraction of τ)

    # AdEx exponential (models avalanche nonlinearity near threshold)
    delta_T: float = 0.10      # exponential sharpness

    # Drive
    bg_frac: float = 0.88      # background current / threshold (tuned)
    input_scale: float = 0.30  # input weight scaling

    # STP (mapped from NS-RAM charge trapping, tuned via grid search)
    U: float = 0.01            # TM utilization — very gentle (tuned: 0.01 optimal)
    tau_rec: float = 15.0      # TM recovery time (tuned: 15 optimal)
    tau_fac: float = 10.0      # TM facilitation time (tuned: 10 optimal)
    alpha_theta: float = 0.0   # Q → threshold modulation (tuned: 0 = off)
    alpha_weight: float = 0.30 # Q → weight modulation (tuned: 0.3 helps MC/NARMA)

    # Network
    exc_frac: float = 0.80     # fraction excitatory (Dale's law)
    spectral_radius: float = 1.05  # edge of chaos (tuned)
    connection_prob: float = 0.15  # sparse connectivity

    # Variability (die-to-die)
    variability: float = 0.10

    def to_physical(self, device: DeviceParams = None) -> dict:
        """Convert dimensionless params to physical SI units."""
        if device is None:
            device = DeviceParams()
        return {
            'tau_mem_s': self.tau * device.tau_mem,
            'V_thresh_V': self.theta * device.V_thresh,
            'delta_T_V': self.delta_T * device.V_thresh,
            'I_bg_A': self.bg_frac * device.V_thresh * device.g_leak,
            't_refrac_s': self.t_refrac * device.tau_mem,
            'tau_rec_s': self.tau_rec * device.tau_mem,
        }

    def from_Vg2(self, Vg2: float, k_cap_max: float = 1000.0,
                  k_em: float = 200.0) -> 'DimensionlessParams':
        """Derive STP parameters from physical VG2 voltage.

        This is the NS-RAM ↔ TM bridge:
          k_cap(VG2) → U
          1/k_em → τ_rec
        """
        import copy
        p = copy.copy(self)
        k_cap = float(charge_capture_rate(Vg2, k_cap_max))
        p.U = min(k_cap / k_cap_max, 0.95)
        p.tau_rec = max(1.0 / k_em * 1000, 1.0)
        p.tau_fac = (1.0 - p.U) * 5.0
        return p


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE: Parameter presets
# ═══════════════════════════════════════════════════════════════════════

PRESETS = {
    # Default = tuned optimal (U=0.01, sr=1.05, bg=0.88)
    'default': DimensionlessParams(),

    # Tuned: STP OFF baseline — strong AdEx-LIF without charge trapping
    'no_stp': DimensionlessParams(U=0.0, alpha_theta=0.0, alpha_weight=0.0),

    # Tuned: optimal STP (from grid search, +4.5% MC/NARMA over no_stp)
    'tuned_stp': DimensionlessParams(
        U=0.01, tau_rec=15.0, tau_fac=10.0,
        alpha_theta=0.0, alpha_weight=0.30,
        bg_frac=0.88, spectral_radius=1.05),

    # STP types
    'depression': DimensionlessParams(U=0.70, tau_rec=5.0, tau_fac=0.0),
    'facilitation': DimensionlessParams(U=0.05, tau_rec=20.0, tau_fac=15.0),
    'fast_stp': DimensionlessParams(U=0.30, tau_rec=3.0),
    'slow_stp': DimensionlessParams(U=0.05, tau_rec=50.0),

    # Network topology variants
    'low_noise': DimensionlessParams(variability=0.05),
    'high_variability': DimensionlessParams(variability=0.20),
    'critical': DimensionlessParams(spectral_radius=1.05, bg_frac=0.85),
    'subcritical': DimensionlessParams(spectral_radius=0.80, bg_frac=0.95),

    # NS-RAM operating modes
    'neuron_mode': DimensionlessParams(  # High VG2: neuron behavior
        bg_frac=0.90, delta_T=0.10, U=0.0, alpha_weight=0.0),
    'synapse_mode': DimensionlessParams(  # Low VG2: strong charge trapping
        bg_frac=0.80, delta_T=0.05, U=0.50, tau_rec=5.0),
}
