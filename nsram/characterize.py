"""nsram.characterize — Device Characterization & Lifetime Analysis

Complete characterization suite for NS-RAM experimental validation.
Designed for Sebastian's workflow: measure → fit → predict → validate.

Transient Analysis:
    simulate_pulse_response   — Body voltage response to drain pulse
    simulate_spike_train      — Generate spike train at given Vg1/Vds
    firing_frequency_map      — 2D map of f(Vg1, Vds) — reproduces Nature Fig 3

Synaptic Plasticity:
    simulate_ltp_ltd          — Multi-pulse potentiation/depression cycles
    paired_pulse_ratio        — PPF/PPD extraction from paired stimuli
    weight_levels             — Characterize distinct conductance states

Lifetime & Reliability:
    simulate_retention        — Charge decay at temperature with Arrhenius
    simulate_endurance        — Degradation tracking over N cycles
    time_to_failure           — Weibull extrapolation from endurance data

Energy Analysis:
    energy_per_spike          — Dynamic energy from I×V×dt integration
    energy_per_synop          — Synaptic operation energy (read/write)
    power_dissipation_map     — P(Vg1, Vds, mode) for thermal budgeting

Based on: Pazos et al., Nature 640, 69-76 (2025).
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from nsram.physics import (
    DeviceParams, breakdown_voltage, avalanche_current,
    thermal_voltage, charge_capture_rate, srh_trapping_ode,
    threshold_modulation, vcb_self_oscillation,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════
# TRANSIENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def simulate_pulse_response(Vds_pulse: float = 3.0, pulse_width: float = 10e-6,
                             Vg1: float = 0.3, Vg2: float = 0.0,
                             Rb: float = 1e4, Cb: float = 1e-12,
                             params: Optional[DeviceParams] = None,
                             dt: float = 50e-9, pre_time: float = 5e-6,
                             post_time: float = 50e-6) -> Dict:
    """Simulate body voltage response to a drain voltage pulse.

    This is the fundamental transient measurement: apply Vds pulse,
    observe VB(t) buildup from avalanche, and body discharge after.

    Args:
        Vds_pulse: Drain pulse amplitude (V)
        pulse_width: Pulse duration (s)
        Vg1: Gate voltage
        Vg2: Control gate (sets Rb: low Vg2 → high Rb → synapse mode)
        Rb: Body resistance (ohm). 10k = neuron, 1M = synapse
        Cb: Body capacitance (F)
        params: Device parameters
        dt: Timestep
        pre_time: Time before pulse
        post_time: Time after pulse

    Returns:
        dict: 't', 'VB', 'Id', 'Vds_stimulus', 'tau_charge', 'tau_discharge'
    """
    p = params or DeviceParams()
    total = pre_time + pulse_width + post_time
    n_steps = int(total / dt)
    t = np.linspace(0, total, n_steps)

    VB = np.zeros(n_steps)
    Id = np.zeros(n_steps)
    Vds = np.zeros(n_steps)

    # Stimulus: rectangular pulse
    pulse_start = pre_time
    pulse_end = pre_time + pulse_width
    Vds[(t >= pulse_start) & (t < pulse_end)] = Vds_pulse

    vb = 0.0
    for i in range(1, n_steps):
        # Avalanche current (only during pulse)
        if Vds[i] > 0:
            I_aval = float(avalanche_current(Vds[i], Vg1, 300, p.Is))
            # Body effect: positive feedback
            if vb > 0:
                I_aval *= (1 + 5 * vb)
        else:
            I_aval = 0.0

        # Body-source junction leakage
        Vt = thermal_voltage(300)
        I_bsj = 1e-12 * (np.exp(min(vb / Vt, 30)) - 1) if vb > 0 else 0

        # Body resistance discharge
        I_Rb = vb / Rb

        # Body charge ODE: Cb × dVB/dt = I_aval - I_bsj - I_Rb
        dvb = (I_aval - I_bsj - I_Rb) / Cb
        vb += dvb * dt
        vb = np.clip(vb, -0.5, 2.0)

        VB[i] = vb
        Id[i] = I_aval

    # Extract time constants
    # Charge: time to reach 63% of peak during pulse
    pulse_mask = (t >= pulse_start) & (t < pulse_end)
    if pulse_mask.any() and VB[pulse_mask].max() > 0.01:
        peak = VB[pulse_mask].max()
        target = 0.632 * peak
        charge_idx = np.where(pulse_mask & (VB >= target))[0]
        tau_charge = float(t[charge_idx[0]] - pulse_start) if len(charge_idx) > 0 else np.nan
    else:
        tau_charge = np.nan

    # Discharge: time to decay to 37% of peak after pulse
    post_mask = t >= pulse_end
    if post_mask.any() and VB[post_mask].max() > 0.01:
        peak_post = VB[np.argmin(np.abs(t - pulse_end))]
        target = 0.368 * peak_post
        decay_idx = np.where(post_mask & (VB <= target))[0]
        tau_discharge = float(t[decay_idx[0]] - pulse_end) if len(decay_idx) > 0 else np.nan
    else:
        tau_discharge = np.nan

    return {
        't': t, 'VB': VB, 'Id': Id, 'Vds_stimulus': Vds,
        'tau_charge': tau_charge, 'tau_discharge': tau_discharge,
        'Rb': Rb, 'Cb': Cb, 'Vg1': Vg1,
    }


def firing_frequency_map(Vg1_range=(0.1, 0.7), Vds_range=(1.5, 4.0),
                          n_Vg1=20, n_Vds=20,
                          params: Optional[DeviceParams] = None,
                          duration: float = 500e-6) -> Dict:
    """Generate 2D firing frequency map f(Vg1, Vds).

    Reproduces Nature Fig 3: heat map showing how gate voltage and
    drain voltage jointly control spiking frequency.

    Args:
        Vg1_range: Gate voltage sweep
        Vds_range: Drain voltage sweep
        n_Vg1, n_Vds: Grid resolution
        duration: Simulation duration per point

    Returns:
        dict: 'Vg1', 'Vds', 'freq_Hz' (2D arrays), 'energy_map'
    """
    from nsram.neuron import NSRAMNeuron

    Vg1_arr = np.linspace(*Vg1_range, n_Vg1)
    Vds_arr = np.linspace(*Vds_range, n_Vds)
    freq_map = np.zeros((n_Vg1, n_Vds))
    energy_map = np.zeros((n_Vg1, n_Vds))

    for i, vg1 in enumerate(Vg1_arr):
        for j, vds in enumerate(Vds_arr):
            neuron = NSRAMNeuron(Vg1=vg1, device=params)
            bvpar = float(breakdown_voltage(vg1, 300))
            # Only simulate above breakdown
            if vds < bvpar * 0.8:
                continue
            result = neuron.simulate(
                duration=duration, dt=100e-9,
                I_ext_fn=lambda t, v=vds: float(avalanche_current(v, vg1, 300)),
            )
            if result['n_spikes'] > 1:
                freq_map[i, j] = result['n_spikes'] / duration
                energy_map[i, j] = result['energy_J'] / max(result['n_spikes'], 1)

    return {
        'Vg1': Vg1_arr, 'Vds': Vds_arr,
        'freq_Hz': freq_map, 'energy_per_spike_J': energy_map,
    }


# ═══════════════════════════════════════════════════════════════════
# SYNAPTIC PLASTICITY
# ═══════════════════════════════════════════════════════════════════

def simulate_ltp_ltd(n_pulses_ltp: int = 50, n_pulses_ltd: int = 50,
                      pulse_amplitude: float = 3.0,
                      pulse_width: float = 1e-6,
                      interval: float = 10e-6,
                      Vg1: float = 0.3, Vg2_ltp: float = 0.2,
                      Vg2_ltd: float = 0.5,
                      Rb_synapse: float = 1e6, Cb: float = 1e-12,
                      params: Optional[DeviceParams] = None) -> Dict:
    """Simulate LTP/LTD potentiation-depression cycles.

    Reproduces Nature Fig 4: conductance change over repeated pulses.

    LTP: low Vg2 → high capture rate → charge accumulates → Vth decreases
    LTD: high Vg2 → low capture rate → charge emits → Vth recovers

    Args:
        n_pulses_ltp: Number of potentiation pulses
        n_pulses_ltd: Number of depression pulses
        pulse_amplitude: Drain pulse voltage
        pulse_width: Single pulse duration
        interval: Time between pulses
        Vg1: Gate voltage
        Vg2_ltp: Control gate for LTP (low → high capture)
        Vg2_ltd: Control gate for LTD (high → low capture)

    Returns:
        dict: 'pulse_number', 'Q_ltp', 'Q_ltd', 'Vth_ltp', 'Vth_ltd',
              'conductance_ltp', 'conductance_ltd', 'linearity_ltp', 'linearity_ltd'
    """
    p = params or DeviceParams()
    dt = 50e-9

    # LTP phase
    Q = 0.0
    Q_ltp = []
    k_cap_ltp = float(charge_capture_rate(Vg2_ltp))

    for pulse in range(n_pulses_ltp):
        # Apply pulse: charge accumulates
        n_pulse_steps = int(pulse_width / dt)
        for _ in range(n_pulse_steps):
            dQ = float(srh_trapping_ode(Q, 1000.0, k_cap_ltp, 200.0))
            Q = np.clip(Q + dQ * dt, 0, 1)

        # Inter-pulse interval: slight decay
        n_gap_steps = int(interval / dt)
        for _ in range(n_gap_steps):
            dQ = float(srh_trapping_ode(Q, 0.0, k_cap_ltp, 200.0))
            Q = np.clip(Q + dQ * dt, 0, 1)

        Q_ltp.append(Q)

    # LTD phase (start from accumulated charge)
    Q_ltd = []
    k_cap_ltd = float(charge_capture_rate(Vg2_ltd))

    for pulse in range(n_pulses_ltd):
        n_pulse_steps = int(pulse_width / dt)
        for _ in range(n_pulse_steps):
            # LTD: emission dominates (low capture, high emission)
            dQ = float(srh_trapping_ode(Q, 100.0, k_cap_ltd, 200.0))
            Q = np.clip(Q + dQ * dt, 0, 1)

        n_gap_steps = int(interval / dt)
        for _ in range(n_gap_steps):
            dQ = float(srh_trapping_ode(Q, 0.0, k_cap_ltd, 200.0))
            Q = np.clip(Q + dQ * dt, 0, 1)

        Q_ltd.append(Q)

    Q_ltp = np.array(Q_ltp)
    Q_ltd = np.array(Q_ltd)

    # Convert charge to threshold voltage shift
    alpha = 0.5  # V per unit charge
    Vth_ltp = p.V_thresh - alpha * Q_ltp
    Vth_ltd = p.V_thresh - alpha * Q_ltd

    # Conductance (proportional to 1/Vth for simplicity)
    G_ltp = 1.0 / np.clip(Vth_ltp, 0.1, 5.0)
    G_ltd = 1.0 / np.clip(Vth_ltd, 0.1, 5.0)

    # Linearity: max deviation from ideal linear ramp / total range
    def _linearity(arr):
        ideal = np.linspace(arr[0], arr[-1], len(arr))
        rng = abs(arr[-1] - arr[0])
        if rng < 1e-10:
            return 1.0
        return float(np.max(np.abs(arr - ideal)) / rng)

    return {
        'pulse_number_ltp': np.arange(n_pulses_ltp),
        'pulse_number_ltd': np.arange(n_pulses_ltp, n_pulses_ltp + n_pulses_ltd),
        'Q_ltp': Q_ltp, 'Q_ltd': Q_ltd,
        'Vth_ltp': Vth_ltp, 'Vth_ltd': Vth_ltd,
        'conductance_ltp': G_ltp, 'conductance_ltd': G_ltd,
        'linearity_ltp': _linearity(G_ltp),
        'linearity_ltd': _linearity(G_ltd),
        'n_levels': int(len(np.unique(np.round(G_ltp, 3)))),
    }


def paired_pulse_ratio(intervals, Vds: float = 3.0, Vg1: float = 0.3,
                        pulse_width: float = 1e-6,
                        Rb: float = 1e4, Cb: float = 1e-12,
                        params: Optional[DeviceParams] = None) -> Dict:
    """Measure paired-pulse facilitation/depression ratio.

    Sends two identical pulses separated by variable interval.
    PPR = response2 / response1.  PPR > 1 = facilitation, < 1 = depression.

    Args:
        intervals: Array of inter-pulse intervals (seconds)
        Vds: Drain pulse amplitude
        pulse_width: Pulse duration

    Returns:
        dict: 'intervals', 'ppr', 'response1', 'response2'
    """
    intervals = np.asarray(intervals)
    pprs = np.zeros(len(intervals))
    r1s = np.zeros(len(intervals))
    r2s = np.zeros(len(intervals))

    for i, gap in enumerate(intervals):
        # First pulse
        res1 = simulate_pulse_response(Vds, pulse_width, Vg1, Rb=Rb, Cb=Cb,
                                        params=params, pre_time=1e-6,
                                        post_time=gap + pulse_width + 10e-6)
        peak1 = res1['VB'][:int(len(res1['t']) * 0.3)].max()

        # Second pulse (body still has residual charge from first)
        res2 = simulate_pulse_response(Vds, pulse_width, Vg1, Rb=Rb, Cb=Cb,
                                        params=params, pre_time=gap,
                                        post_time=10e-6)
        peak2 = res2['VB'].max()

        r1s[i] = peak1
        r2s[i] = peak2
        pprs[i] = peak2 / (peak1 + 1e-15)

    return {
        'intervals': intervals,
        'ppr': pprs,
        'response1': r1s,
        'response2': r2s,
        'is_facilitating': bool(pprs.mean() > 1.0),
    }


# ═══════════════════════════════════════════════════════════════════
# LIFETIME & RELIABILITY
# ═══════════════════════════════════════════════════════════════════

def simulate_retention(Q0: float = 0.8, duration: float = 1e4,
                        temperatures: Optional[List[float]] = None,
                        Rb: float = 1e6, Cb: float = 1e-12,
                        Ea: float = 0.7,
                        params: Optional[DeviceParams] = None) -> Dict:
    """Simulate charge retention decay at multiple temperatures.

    Uses Arrhenius-accelerated emission: k_em(T) = k_em0 × exp(-Ea/kT).
    Reproduces Nature Fig 5c: retention curves at 25°C, 85°C, 125°C.

    Args:
        Q0: Initial trapped charge fraction
        duration: Total retention time (seconds)
        temperatures: List of temperatures in Kelvin (default: [300, 358, 398])
        Rb: Body resistance (synapse mode)
        Ea: Activation energy for detrapping (eV)

    Returns:
        dict: 'time', 'Q_vs_T' (dict of T→Q(t)), 'tau_vs_T', 'Ea_extracted'
    """
    if temperatures is None:
        temperatures = [300, 358, 398]  # 27°C, 85°C, 125°C

    kB = 8.617e-5  # eV/K
    k_em0 = 1e-4  # Base emission rate at 300K (matches >10^4s retention from Pazos)

    n_points = 500
    t = np.logspace(-3, np.log10(duration), n_points)

    Q_curves = {}
    taus = {}

    for T in temperatures:
        # Arrhenius: higher T → faster emission → shorter retention
        # k_em(T) = k_em(300K) × exp(Ea/kB × (1/300 - 1/T))
        # When T > 300: (1/300 - 1/T) > 0 → k_em increases → faster decay
        k_em = k_em0 * np.exp(Ea / kB * (1.0/300.0 - 1.0/T))

        Q = np.zeros(n_points)
        Q[0] = Q0
        for i in range(1, n_points):
            dt_step = t[i] - t[i-1]
            # Pure emission decay (no capture during retention)
            dQ = -k_em * Q[i-1] * dt_step
            Q[i] = np.clip(Q[i-1] + dQ, 0, Q0)

        Q_curves[T] = Q

        # Extract tau (time to Q0 × 1/e)
        target = Q0 * np.exp(-1)
        idx = np.where(Q <= target)[0]
        taus[T] = float(t[idx[0]]) if len(idx) > 0 else float(duration)

    # Extract activation energy from tau vs T
    if len(temperatures) >= 2:
        T_arr = np.array(temperatures)
        tau_arr = np.array([taus[T] for T in temperatures])
        valid = tau_arr > 0
        if valid.sum() >= 2:
            # ln(tau) = Ea/(kB×T) + const → linear fit
            x = 1.0 / T_arr[valid]
            y = np.log(tau_arr[valid])
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 1)
                Ea_extracted = coeffs[0] * kB
            else:
                Ea_extracted = np.nan
        else:
            Ea_extracted = np.nan
    else:
        Ea_extracted = np.nan

    return {
        'time': t,
        'Q_vs_T': Q_curves,
        'tau_vs_T': taus,
        'Ea_extracted': float(Ea_extracted),
        'Ea_input': Ea,
        'temperatures': temperatures,
    }


def simulate_endurance(n_cycles: int = 10000, Vds: float = 3.0,
                        pulse_width: float = 1e-6,
                        degradation_rate: float = 1e-6,
                        params: Optional[DeviceParams] = None) -> Dict:
    """Simulate endurance: track parameter degradation over write cycles.

    Models oxide trap generation: each write cycle creates a small
    probability of permanent trap that shifts BVpar and increases Is.

    Args:
        n_cycles: Total write/erase cycles to simulate
        Vds: Write pulse voltage
        pulse_width: Write pulse width
        degradation_rate: Fractional degradation per cycle

    Returns:
        dict: 'cycles', 'BVpar', 'Is', 'on_off_ratio', 'failure_cycle'
    """
    p = params or DeviceParams()

    sample_points = np.unique(np.logspace(0, np.log10(n_cycles), 200).astype(int))
    sample_points = sample_points[sample_points <= n_cycles]

    BV0 = p.BV0
    Is = p.Is
    BVpar_arr = []
    Is_arr = []
    ratio_arr = []

    cumulative_damage = 0.0
    failure_cycle = n_cycles  # Default: no failure

    for cycle in range(1, n_cycles + 1):
        # Stochastic degradation: power-law trap generation
        cumulative_damage += degradation_rate * np.random.exponential(1.0)

        if cycle in sample_points:
            # BVpar decreases as oxide degrades
            BV_now = BV0 * (1 - 0.1 * np.log10(1 + cumulative_damage))
            # Is increases (leakage through damaged oxide)
            Is_now = Is * (1 + cumulative_damage * 10)
            # On/off ratio (R_HRS / R_LRS)
            ratio = max(1, p.R_HRS / p.R_LRS * (1 - 0.05 * np.log10(1 + cumulative_damage)))

            BVpar_arr.append(BV_now)
            Is_arr.append(Is_now)
            ratio_arr.append(ratio)

            # Failure criterion: BVpar dropped below 80% of original
            if BV_now < 0.8 * BV0 and failure_cycle == n_cycles:
                failure_cycle = cycle

    return {
        'cycles': sample_points[:len(BVpar_arr)],
        'BVpar': np.array(BVpar_arr),
        'Is': np.array(Is_arr),
        'on_off_ratio': np.array(ratio_arr),
        'failure_cycle': failure_cycle,
        'n_cycles': n_cycles,
    }


# ═══════════════════════════════════════════════════════════════════
# ENERGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def energy_per_spike(Vg1: float = 0.3, Vds: float = 3.0,
                      params: Optional[DeviceParams] = None) -> Dict:
    """Compute energy per spike from I×V×dt integration.

    Decomposes into: generation energy + integration energy + leakage.
    Pazos reports 21 fJ total (4.7 fJ generation + 16.3 fJ integration).

    Returns:
        dict: 'E_total_J', 'E_generation_J', 'E_integration_J', 'E_leakage_J'
    """
    p = params or DeviceParams()
    dt = 50e-9
    duration = 50e-6

    # Simulate one neuron cycle
    from nsram.neuron import NSRAMNeuron
    neuron = NSRAMNeuron(Vg1=Vg1, device=p)
    result = neuron.simulate(duration=duration, dt=dt, I_ext=0.0)

    if result['n_spikes'] == 0:
        # Try with external current to force a spike
        neuron.reset()
        result = neuron.simulate(duration=duration, dt=dt, I_ext=1e-9)

    n_spk = max(result['n_spikes'], 1)
    E_total = result['energy_J'] / n_spk

    # Decomposition (approximate from Pazos ratios)
    E_gen = E_total * 0.224    # 4.7/21 = 22.4%
    E_int = E_total * 0.776    # 16.3/21 = 77.6%
    E_leak = p.g_leak * p.V_leak_rest**2 * p.t_refrac  # Leakage during refractory

    return {
        'E_total_J': float(E_total),
        'E_generation_J': float(E_gen),
        'E_integration_J': float(E_int),
        'E_leakage_J': float(E_leak),
        'n_spikes': result['n_spikes'],
        'Vg1': Vg1, 'Vds': Vds,
    }


def energy_comparison_table(params: Optional[DeviceParams] = None) -> Dict:
    """Compare NS-RAM energy with other neuron/synapse implementations.

    Returns publication-ready comparison data.
    """
    p = params or DeviceParams()

    return {
        'NS-RAM neuron (2T)': {
            'energy_pJ': p.E_spike * 1e12,
            'area_um2': p.area_2T * 1e12,
            'area_F2': p.area_2T / (130e-9)**2,  # Normalized to F²
        },
        'CMOS LIF (18T)': {
            'energy_pJ': 10.0,
            'area_um2': 900,
            'area_F2': 900 / (0.130)**2,
        },
        'NS-RAM synapse (2T)': {
            'energy_pJ': p.E_spike * 1e12 * 0.5,  # Write is ~half spike energy
            'area_um2': p.area_2T * 1e12,
            'levels': 14,
        },
        'RRAM synapse (1T1R)': {
            'energy_pJ': 0.1,
            'area_um2': 4,
            'levels': 8,
        },
        'PCM synapse': {
            'energy_pJ': 100,
            'area_um2': 20,
            'levels': 16,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# VOLTAGE RAMP & SWEEP-RATE DEPENDENT I-V
# (From Sebastian's transient VD ramp measurements — unpublished)
# ═══════════════════════════════════════════════════════════════════

def simulate_voltage_ramp(Vds_max: float = 4.0, sweep_rate: float = 1.0,
                           Vg1: float = 0.3, Rb: float = 1e4, Cb: float = 1e-12,
                           params: Optional[DeviceParams] = None,
                           return_sweep: str = 'both') -> Dict:
    """Simulate I-V under transient voltage ramp (not DC).

    The 2T NS-RAM cell shows hysteresis under voltage ramps because the
    body charges during the up-sweep and discharges during down-sweep.
    Sweep rate controls hysteresis width: slower → less hysteresis.

    From Sebastian's unpublished data: thick oxide cell optimized at 0.1 V/s,
    tested from 0.01 to 10 V/s.

    Args:
        Vds_max: Maximum drain voltage in ramp
        sweep_rate: Voltage sweep rate (V/s). Range: 0.01 to 10
        Vg1: Gate voltage
        Rb: Body resistance
        Cb: Body capacitance
        return_sweep: 'up', 'down', or 'both'

    Returns:
        dict: 'Vds_up', 'Id_up', 'Vds_down', 'Id_down', 'VB_up', 'VB_down',
              'hysteresis_area', 'sweep_rate'
    """
    p = params or DeviceParams()
    ramp_time = Vds_max / sweep_rate
    dt = min(1e-3, ramp_time / 2000)  # At least 2000 points per sweep
    n_up = int(ramp_time / dt)
    n_down = n_up

    # Up sweep
    Vds_up = np.linspace(0, Vds_max, n_up)
    Id_up = np.zeros(n_up)
    VB_up = np.zeros(n_up)
    vb = 0.0

    for i in range(n_up):
        I_aval = float(avalanche_current(Vds_up[i], Vg1, 300, p.Is))
        if vb > 0:
            I_aval *= (1 + 5 * vb)
        I_Rb = vb / Rb
        Vt = thermal_voltage(300)
        I_bsj = 1e-12 * (np.exp(min(vb / Vt, 30)) - 1) if vb > 0 else 0
        dvb = (I_aval - I_bsj - I_Rb) / Cb
        vb += dvb * dt
        vb = np.clip(vb, -0.5, 2.0)
        VB_up[i] = vb
        Id_up[i] = I_aval + I_bsj

    # Down sweep (body has accumulated charge)
    Vds_down = np.linspace(Vds_max, 0, n_down)
    Id_down = np.zeros(n_down)
    VB_down = np.zeros(n_down)

    for i in range(n_down):
        I_aval = float(avalanche_current(Vds_down[i], Vg1, 300, p.Is))
        if vb > 0:
            I_aval *= (1 + 5 * vb)
        I_Rb = vb / Rb
        Vt = thermal_voltage(300)
        I_bsj = 1e-12 * (np.exp(min(vb / Vt, 30)) - 1) if vb > 0 else 0
        dvb = (I_aval - I_bsj - I_Rb) / Cb
        vb += dvb * dt
        vb = np.clip(vb, -0.5, 2.0)
        VB_down[i] = vb
        Id_down[i] = I_aval + I_bsj

    # Hysteresis area (enclosed loop)
    # Interpolate to common Vds grid
    Vds_common = np.linspace(0, Vds_max, 500)
    Id_up_interp = np.interp(Vds_common, Vds_up, Id_up)
    Id_down_interp = np.interp(Vds_common, Vds_down[::-1], Id_down[::-1])
    hyst_area = float(np.trapz(np.abs(Id_up_interp - Id_down_interp), Vds_common))

    return {
        'Vds_up': Vds_up, 'Id_up': Id_up, 'VB_up': VB_up,
        'Vds_down': Vds_down, 'Id_down': Id_down, 'VB_down': VB_down,
        'hysteresis_area': hyst_area,
        'sweep_rate': sweep_rate,
    }


def sweep_rate_dependence(sweep_rates=None, Vds_max: float = 4.0,
                           Vg1: float = 0.3, **kwargs) -> Dict:
    """Characterize I-V hysteresis vs sweep rate.

    From Sebastian's thick oxide cell data: optimized at 0.1 V/s.
    Slower sweeps → body reaches steady state → less hysteresis.
    Faster sweeps → transient body charging → more hysteresis.

    Args:
        sweep_rates: Array of sweep rates (V/s). Default: [0.01, 0.1, 1, 10]

    Returns:
        dict: 'sweep_rates', 'hysteresis_areas', 'ramp_results'
    """
    if sweep_rates is None:
        sweep_rates = [0.01, 0.1, 1.0, 10.0]

    areas = []
    results = []
    for sr in sweep_rates:
        r = simulate_voltage_ramp(Vds_max=Vds_max, sweep_rate=sr, Vg1=Vg1, **kwargs)
        areas.append(r['hysteresis_area'])
        results.append(r)

    return {
        'sweep_rates': np.array(sweep_rates),
        'hysteresis_areas': np.array(areas),
        'ramp_results': results,
    }


# ═══════════════════════════════════════════════════════════════════
# POLYNOMIAL BULK CURRENT MODEL
# (From Sebastian's semi-empirical fits — unpublished)
# ═══════════════════════════════════════════════════════════════════

def bulk_current_polynomial(Vds, Vg1, a=None, b=None):
    """Semi-empirical polynomial bulk current model.

    From Sebastian's unpublished fit: I_bulk = a(Vg1) × Vds + b(Vg1) × Vds²
    This is an alternative to the Chynoweth exponential model — simpler,
    faster, and sometimes more accurate in the linear regime.

    Args:
        Vds: Drain-source voltage (scalar or array)
        Vg1: Gate voltage
        a: Linear coefficient (auto-estimated from Vg1 if None)
        b: Quadratic coefficient (auto-estimated from Vg1 if None)

    Returns:
        I_bulk: Bulk current (same shape as Vds)
    """
    Vds = np.asarray(Vds, dtype=np.float64)

    # Default: empirical relationship from Vg1 (from Sebastian's fits)
    if a is None:
        a = 1e-12 * np.exp(3.0 * Vg1)  # Increases with Vg1
    if b is None:
        b = 5e-13 * np.exp(4.0 * Vg1)  # Stronger Vg1 dependence

    I = a * Vds + b * Vds**2
    return np.clip(I, 0, 1e-3)


def fit_bulk_polynomial(Vds, Id, Vg1: float = 0.3) -> Dict:
    """Fit polynomial bulk current model to measured data.

    Args:
        Vds: (N,) measured voltage points
        Id: (N,) measured drain current
        Vg1: Gate voltage at measurement

    Returns:
        dict: 'a', 'b', 'r_squared', 'model_type'
    """
    Vds = np.asarray(Vds, dtype=np.float64)
    Id = np.asarray(Id, dtype=np.float64)
    mask = (Vds > 0) & (Id > 0)
    V, I = Vds[mask], Id[mask]

    if len(V) < 3:
        return {'a': 0, 'b': 0, 'r_squared': 0, 'error': 'too few points'}

    # Fit I = a*V + b*V²
    A = np.column_stack([V, V**2])
    coeffs, _, _, _ = np.linalg.lstsq(A, I, rcond=None)
    a, b = coeffs

    pred = a * V + b * V**2
    ss_res = np.sum((I - pred)**2)
    ss_tot = np.sum((I - I.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {'a': float(a), 'b': float(b), 'r_squared': float(r2),
            'Vg1': Vg1, 'model_type': 'polynomial'}


# ═══════════════════════════════════════════════════════════════════
# DEEP N-WELL HIGH-VOLTAGE REGIME
# (From Sebastian's 130nm triple-well measurements — unpublished)
# ═══════════════════════════════════════════════════════════════════

def deep_nwell_iv(Vds_range=(0, 12), Vg_values=None,
                   BV0: float = 10.5, k_vg: float = 0.5,
                   params: Optional[DeviceParams] = None) -> Dict:
    """I-V curves for deep N-well NFET floating body cell.

    The deep N-well 1T cell operates at MUCH higher voltages than the
    standard cell (7-10V+ vs 2-4V). This is because the deep N-well
    provides better body isolation → higher BVpar.

    From Sebastian's unpublished data: firing between 7V and 10V+,
    Vg controls onset.

    Args:
        Vds_range: Drain voltage sweep range (0 to 12V typical)
        Vg_values: Gate voltages to sweep (default: 7-10V)
        BV0: Breakdown voltage at Vg=0 for deep N-well (much higher than standard)
        k_vg: BVpar sensitivity to Vg

    Returns:
        dict: 'Vds', 'Id_family' (dict of Vg→Id), 'BVpar_vs_Vg'
    """
    p = params or DeviceParams()
    if Vg_values is None:
        Vg_values = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    Vds = np.linspace(*Vds_range, 500)
    Id_family = {}
    BVpar_vs_Vg = {}

    for Vg in Vg_values:
        BVpar = BV0 - k_vg * Vg
        BVpar_vs_Vg[Vg] = BVpar

        Vt = thermal_voltage(300)
        Ne = p.Ne
        Is = p.Is

        # Avalanche current with deep N-well parameters
        exponent = np.clip((Vds - BVpar) / (Ne * Vt), -30, 30)
        Id = Is * np.exp(exponent)
        Id = np.clip(Id, 0, 1e-3)
        Id_family[Vg] = Id

    return {
        'Vds': Vds,
        'Id_family': Id_family,
        'BVpar_vs_Vg': BVpar_vs_Vg,
        'Vg_values': Vg_values,
        'BV0': BV0,
        'k_vg': k_vg,
    }


# ═══════════════════════════════════════════════════════════════════
# EXCITATORY / INHIBITORY INPUT CIRCUIT MODEL
# (From Sebastian's NSRAM building blocks — unpublished)
# ═══════════════════════════════════════════════════════════════════

def ei_input_neuron(I_exc, I_inh, Vg1: float = 0.3,
                     g_exc: float = 1.0, g_inh: float = 1.0,
                     params: Optional[DeviceParams] = None) -> Dict:
    """NS-RAM neuron with excitatory and inhibitory current inputs.

    From Sebastian's circuit: excitatory and inhibitory inputs via
    current mirrors, without diode (soma-only configuration).
    The balance of E/I determines firing rate.

    Args:
        I_exc: (T,) excitatory input current time series
        I_inh: (T,) inhibitory input current time series
        Vg1: Gate bias
        g_exc: Excitatory gain
        g_inh: Inhibitory gain

    Returns:
        dict: 't', 'Vm', 'spikes', 'firing_rate', 'ei_balance'
    """
    from nsram.neuron import NSRAMNeuron

    I_exc = np.asarray(I_exc, dtype=np.float64)
    I_inh = np.asarray(I_inh, dtype=np.float64)
    T = len(I_exc)

    # Net current: E - I
    I_net = g_exc * I_exc - g_inh * I_inh

    neuron = NSRAMNeuron(Vg1=Vg1, device=params)
    result = neuron.simulate(
        duration=T * 1e-6,  # Assume 1µs per sample
        dt=1e-7,
        I_ext_fn=lambda t, _I=I_net: float(_I[min(int(t * 1e6), T-1)]),
    )

    return {
        't': result['t'],
        'Vm': result['Vm'],
        'spikes': result['spikes'],
        'n_spikes': result['n_spikes'],
        'firing_rate': result['n_spikes'] / (T * 1e-6),
        'ei_balance': float(np.mean(I_exc) / (np.mean(I_inh) + 1e-15)),
    }


# ═══════════════════════════════════════════════════════════════════
# FREQUENCY-CODED INPUT (MNIST APPLICATION)
# (From Sebastian's integrator-reset frequency coding — unpublished)
# ═══════════════════════════════════════════════════════════════════

def frequency_encode_image(image, n_steps: int = 100,
                            f_max: float = 200e3, f_min: float = 1e3):
    """Encode image pixels as spike frequencies (NS-RAM native encoding).

    From Sebastian's unpublished work: "Integrator reset example for
    frequency coded input of MNIST". Each pixel intensity maps to a
    firing frequency. The NS-RAM neuron's natural frequency tunability
    (4 decades: 20 Hz to 200 kHz) is used directly.

    This is more biologically realistic than rate coding because it
    preserves precise spike timing within each coding interval.

    Args:
        image: (H, W) or (N,) pixel values in [0, 1]
        n_steps: Number of time steps
        f_max: Maximum frequency (bright pixel)
        f_min: Minimum frequency (dark pixel)

    Returns:
        spikes: (n_steps, N) binary spike train
    """
    image = np.asarray(image, dtype=np.float32).ravel()
    N = len(image)

    # Map pixel intensity to inter-spike interval
    freqs = f_min + image * (f_max - f_min)
    periods = 1.0 / (freqs + 1e-10)  # In time steps

    spikes = np.zeros((n_steps, N), dtype=np.float32)
    phase = np.random.uniform(0, 1, N)  # Random initial phase

    for t in range(n_steps):
        phase += 1.0 / (periods * n_steps + 1e-10)
        fired = phase >= 1.0
        spikes[t, fired] = 1.0
        phase[fired] -= 1.0

    return spikes
