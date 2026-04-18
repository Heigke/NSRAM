"""nsram.fitting — Parameter Extraction from Experimental I-V Data

Automated curve fitting for NS-RAM device characterization.
Takes experimental I-V curves and extracts DeviceParams.

Key features:
    fit_iv_curve       — Extract BVpar, Is, Bf, Ne from drain current vs Vcb
    fit_retention       — Extract tau_retention from decay curves
    fit_stp_params      — Extract U, tau_rec, tau_fac from paired-pulse data
    fit_from_zenodo     — Load and fit from Pazos et al. Zenodo data (13843362)
    compare_model_exp   — Overlay model prediction vs experimental data
    monte_carlo         — Statistical variability analysis

Designed to accept Sebastian's experimental data in CSV/Excel format.

Based on: Pazos et al., Nature 640, 69-76 (2025).
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from scipy.optimize import curve_fit, minimize
from nsram.physics import (
    DeviceParams, DimensionlessParams, PRESETS,
    breakdown_voltage, avalanche_current, thermal_voltage,
)
from nsram.bsim4 import (
    BSIM4Params, impact_ionization_bsim4, drain_current_bsim,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════
# I-V CURVE FITTING
# ═══════════════════════════════════════════════════════════════════

def _avalanche_model(Vcb, BV0, k_vg, Is, Ne, Vg1):
    """Avalanche current model: I = Is * exp((Vcb - BVpar) / (Ne * Vt))."""
    Vt = thermal_voltage(300)
    BVpar = BV0 - k_vg * Vg1
    exponent = np.clip((Vcb - BVpar) / (Ne * Vt), -30, 30)
    return Is * np.exp(exponent)


def fit_iv_curve(Vcb, Id, Vg1: float = 0.0,
                 p0: Optional[Dict] = None) -> Dict:
    """Fit NS-RAM I-V curve to extract device parameters.

    Args:
        Vcb: (N,) collector-base voltage array
        Id: (N,) drain/collector current array (positive, in Amps)
        Vg1: Gate voltage at which this I-V was measured
        p0: Initial parameter guess dict

    Returns:
        dict: fitted parameters + fit quality metrics
            'BV0': Breakdown voltage at Vg1=0
            'k_vg': BVpar sensitivity to Vg1 (V/V)
            'Is': Saturation current (A)
            'Ne': Avalanche emission coefficient
            'BVpar': BVpar at this Vg1
            'r_squared': Fit quality
            'residuals': Fit residuals
    """
    Vcb = np.asarray(Vcb, dtype=np.float64)
    Id = np.asarray(Id, dtype=np.float64)

    # Filter to positive currents in avalanche region
    mask = Id > 0
    Vcb_fit = Vcb[mask]
    Id_fit = Id[mask]

    if len(Vcb_fit) < 4:
        raise ValueError("Not enough data points with positive current")

    # Work in log space for better fitting
    log_Id = np.log(Id_fit + 1e-20)

    # Initial guesses
    defaults = {'BV0': 3.5, 'k_vg': 1.5, 'Is': 1e-16, 'Ne': 1.5}
    if p0:
        defaults.update(p0)

    def _model_log(Vcb, BV0, log_Is, Ne):
        Vt = thermal_voltage(300)
        BVpar = BV0 - defaults['k_vg'] * Vg1
        return log_Is + (Vcb - BVpar) / (Ne * Vt)

    try:
        popt, pcov = curve_fit(
            _model_log, Vcb_fit, log_Id,
            p0=[defaults['BV0'], np.log(defaults['Is']), defaults['Ne']],
            bounds=([0.5, -50, 0.5], [10.0, -20, 5.0]),
            maxfev=10000,
        )
        BV0, log_Is, Ne = popt
        Is = np.exp(log_Is)

        # Compute fit quality
        pred = _model_log(Vcb_fit, BV0, log_Is, Ne)
        ss_res = np.sum((log_Id - pred) ** 2)
        ss_tot = np.sum((log_Id - log_Id.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*3

        return {
            'BV0': float(BV0),
            'k_vg': defaults['k_vg'],
            'Is': float(Is),
            'Ne': float(Ne),
            'BVpar': float(BV0 - defaults['k_vg'] * Vg1),
            'r_squared': float(r2),
            'param_errors': {'BV0': perr[0], 'log_Is': perr[1], 'Ne': perr[2]},
            'Vg1': Vg1,
        }
    except RuntimeError as e:
        return {'error': str(e), 'r_squared': 0.0}


def fit_iv_family(Vcb_list, Id_list, Vg1_list) -> Dict:
    """Fit a family of I-V curves at different Vg1 values.

    Extracts BV0 and k_vg from the Vg1-dependent BVpar shift.

    Args:
        Vcb_list: List of (N,) voltage arrays, one per Vg1
        Id_list: List of (N,) current arrays
        Vg1_list: List of Vg1 values

    Returns:
        dict: Global fit parameters + per-curve results
    """
    results = []
    bvpars = []

    for Vcb, Id, Vg1 in zip(Vcb_list, Id_list, Vg1_list):
        r = fit_iv_curve(Vcb, Id, Vg1)
        results.append(r)
        if 'BVpar' in r and r['r_squared'] > 0.5:
            bvpars.append((Vg1, r['BVpar']))

    # Fit BVpar = BV0 - k_vg * Vg1
    if len(bvpars) >= 2:
        vg_arr = np.array([x[0] for x in bvpars])
        bv_arr = np.array([x[1] for x in bvpars])
        # Linear fit
        coeffs = np.polyfit(vg_arr, bv_arr, 1)
        k_vg = -coeffs[0]
        BV0 = coeffs[1]
    else:
        BV0 = results[0].get('BV0', 3.5) if results else 3.5
        k_vg = 1.5

    return {
        'BV0': float(BV0),
        'k_vg': float(k_vg),
        'per_curve': results,
        'n_curves': len(results),
    }


# ═══════════════════════════════════════════════════════════════════
# BSIM4 IMPACT-IONIZATION FITTING  (Sebastian's 2026 measurement flow)
# ═══════════════════════════════════════════════════════════════════

def _bsim4_iii_vs_vds(Vds, ALPHA0, BETA0, Vgs, Vbs, base: BSIM4Params):
    """Iii(Vds) at fixed Vgs, Vbs — target for curve_fit."""
    p = BSIM4Params(**{**base.__dict__, "ALPHA0": ALPHA0, "BETA0": BETA0})
    return np.asarray(impact_ionization_bsim4(Vgs, Vds, Vbs, p),
                      dtype=np.float64)


def fit_bsim4_impact(Vds, Isub, Vgs: float, Vbs: float = 0.0,
                      base: Optional[BSIM4Params] = None,
                      p0: Optional[Dict] = None) -> Dict:
    """Fit BSIM4 impact ionization Iii(Vds, Vgs, Vbs).

    Use this when you have a substrate/body-current measurement and
    want to extract ALPHA0, BETA0 (and optionally re-calibrate the
    drain-current skeleton through `base`).  Drops BVpar as a free
    parameter entirely — firing emerges from (ALPHA0, BETA0) plus
    Vth(Vbs).

    Args:
        Vds    (N,)  drain-source voltage sweep (V)
        Isub   (N,)  substrate/body current (A, positive)
        Vgs    scalar gate voltage at which the sweep was taken (V)
        Vbs    scalar body-source voltage (V), default 0
        base   BSIM4Params providing fixed geometry/Vth/mobility.
               If None, uses the 180 nm preset.
        p0     optional initial guesses: {'ALPHA0': ..., 'BETA0': ...}

    Returns:
        dict with fitted ALPHA0, BETA0, r_squared, residual_norm,
        and a populated BSIM4Params (`params`) you can drop straight
        into `nsram.bsim4.body_charge_ode_bsim4` or
        `nsram.bsim4.TwoTransistorCell`.
    """
    Vds = np.asarray(Vds, dtype=np.float64)
    Isub = np.asarray(Isub, dtype=np.float64)
    base = base or BSIM4Params()

    p0 = p0 or {}
    A0 = float(p0.get("ALPHA0", base.ALPHA0))
    B0 = float(p0.get("BETA0", base.BETA0))

    def _model(vds, a0, b0):
        return _bsim4_iii_vs_vds(vds, a0, b0, Vgs, Vbs, base)

    # Fit in log-space to balance 6–8 decade Iii dynamic range.
    mask = (Isub > 0) & np.isfinite(Isub) & (Vds > 0)
    if mask.sum() < 4:
        return {"error": "insufficient data points above zero", "r_squared": 0.0}

    log_target = np.log(Isub[mask])

    def _log_model(vds, a0, b0):
        y = _model(vds, a0, b0)
        return np.log(np.maximum(y, 1e-30))

    try:
        popt, _ = curve_fit(
            _log_model, Vds[mask], log_target, p0=[A0, B0],
            bounds=([1e-9, 1.0], [1e-2, 80.0]),
            maxfev=8000,
        )
        ALPHA0_fit, BETA0_fit = popt
        pred = _model(Vds[mask], ALPHA0_fit, BETA0_fit)
        ss_res = float(np.sum((Isub[mask] - pred) ** 2))
        ss_tot = float(np.sum((Isub[mask] - Isub[mask].mean()) ** 2) + 1e-30)
        r2 = 1.0 - ss_res / ss_tot
        fitted = BSIM4Params(**{**base.__dict__,
                                 "ALPHA0": float(ALPHA0_fit),
                                 "BETA0": float(BETA0_fit)})
        return {
            "ALPHA0": float(ALPHA0_fit),
            "BETA0": float(BETA0_fit),
            "r_squared": float(r2),
            "residual_norm": float(np.sqrt(ss_res)),
            "n_points": int(mask.sum()),
            "params": fitted,
        }
    except RuntimeError as e:
        return {"error": str(e), "r_squared": 0.0}


def fit_bsim4_family(Vds_list, Isub_list, Vgs_list, Vbs_list=None,
                      base: Optional[BSIM4Params] = None) -> Dict:
    """Fit BSIM4 Iii across a family of (Vgs, Vbs) sweeps.

    Joint optimisation: ALPHA0 and BETA0 should be process-global; only
    the bias condition changes.  We therefore pool all curves and fit
    one (ALPHA0, BETA0) pair.  Per-curve residuals returned for QC.
    """
    base = base or BSIM4Params()
    Vbs_list = Vbs_list or [0.0] * len(Vgs_list)

    all_Vds, all_Isub, all_Vgs, all_Vbs = [], [], [], []
    for V, I, g, b in zip(Vds_list, Isub_list, Vgs_list, Vbs_list):
        V = np.asarray(V); I = np.asarray(I)
        mask = (I > 0) & np.isfinite(I) & (V > 0)
        all_Vds.append(V[mask]); all_Isub.append(I[mask])
        all_Vgs.append(np.full(mask.sum(), float(g)))
        all_Vbs.append(np.full(mask.sum(), float(b)))
    Vds = np.concatenate(all_Vds)
    Isub = np.concatenate(all_Isub)
    Vgs = np.concatenate(all_Vgs)
    Vbs = np.concatenate(all_Vbs)

    if len(Vds) < 4:
        return {"error": "insufficient data", "r_squared": 0.0}

    def _log_model(_x, a0, b0):
        # _x is ignored; we use the captured Vds / Vgs / Vbs arrays
        p = BSIM4Params(**{**base.__dict__, "ALPHA0": a0, "BETA0": b0})
        y = np.array([
            float(impact_ionization_bsim4(vg, vd, vb, p))
            for vd, vg, vb in zip(Vds, Vgs, Vbs)
        ])
        return np.log(np.maximum(y, 1e-30))

    try:
        popt, _ = curve_fit(
            _log_model, Vds, np.log(Isub),
            p0=[base.ALPHA0, base.BETA0],
            bounds=([1e-9, 1.0], [1e-2, 80.0]),
            maxfev=12000,
        )
        ALPHA0_fit, BETA0_fit = popt
        fitted = BSIM4Params(**{**base.__dict__,
                                 "ALPHA0": float(ALPHA0_fit),
                                 "BETA0": float(BETA0_fit)})
        # Per-curve R² for QC
        per_curve = []
        for V, I, g, b in zip(Vds_list, Isub_list, Vgs_list, Vbs_list):
            pred = np.array([float(impact_ionization_bsim4(g, v, b, fitted))
                             for v in V])
            mask = (np.asarray(I) > 0) & np.isfinite(I)
            I_arr = np.asarray(I)[mask]; p_arr = pred[mask]
            ss_res = float(np.sum((I_arr - p_arr) ** 2))
            ss_tot = float(np.sum((I_arr - I_arr.mean()) ** 2) + 1e-30)
            per_curve.append({"Vgs": float(g), "Vbs": float(b),
                              "r_squared": 1.0 - ss_res / ss_tot})
        return {
            "ALPHA0": float(ALPHA0_fit),
            "BETA0": float(BETA0_fit),
            "n_curves": len(Vgs_list),
            "params": fitted,
            "per_curve": per_curve,
        }
    except RuntimeError as e:
        return {"error": str(e), "r_squared": 0.0}


# ═══════════════════════════════════════════════════════════════════
# RETENTION FITTING
# ═══════════════════════════════════════════════════════════════════

def fit_retention(time, signal, model='stretched_exp'):
    """Fit charge retention decay curve.

    Args:
        time: (N,) time points
        signal: (N,) normalized signal (1.0 = full charge)
        model: 'exponential', 'stretched_exp', or 'multi_exp'

    Returns:
        dict: fitted parameters
    """
    time = np.asarray(time, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if model == 'exponential':
        # signal = A * exp(-t / tau)
        def _model(t, A, tau):
            return A * np.exp(-t / tau)
        popt, _ = curve_fit(_model, time, signal, p0=[1.0, 1.0],
                            bounds=([0, 1e-6], [2, 1e6]))
        return {'A': popt[0], 'tau': popt[1], 'model': model}

    elif model == 'stretched_exp':
        # signal = A * exp(-(t/tau)^beta) — Kohlrausch-Williams-Watts
        def _model(t, A, tau, beta):
            return A * np.exp(-(t / tau) ** beta)
        popt, _ = curve_fit(_model, time, signal, p0=[1.0, 1.0, 0.5],
                            bounds=([0, 1e-6, 0.1], [2, 1e6, 1.5]))
        return {'A': popt[0], 'tau': popt[1], 'beta': popt[2], 'model': model}

    elif model == 'multi_exp':
        # signal = A1*exp(-t/tau1) + A2*exp(-t/tau2)
        def _model(t, A1, tau1, A2, tau2):
            return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)
        popt, _ = curve_fit(_model, time, signal,
                            p0=[0.5, 0.1, 0.5, 10.0],
                            bounds=([0, 1e-6, 0, 1e-6], [2, 1e6, 2, 1e6]))
        return {'A1': popt[0], 'tau1': popt[1],
                'A2': popt[2], 'tau2': popt[3], 'model': model}


# ═══════════════════════════════════════════════════════════════════
# MONTE CARLO VARIABILITY
# ═══════════════════════════════════════════════════════════════════

def monte_carlo(params: DeviceParams, n_samples: int = 1000,
                sigma_dict: Optional[Dict] = None,
                seed: int = 42) -> Dict:
    """Monte Carlo variability analysis for NS-RAM arrays.

    Samples device parameters with Gaussian variability and computes
    statistics on key metrics (BVpar, firing rate, energy, etc.).

    Args:
        params: Nominal DeviceParams
        n_samples: Number of Monte Carlo samples
        sigma_dict: Dict of parameter_name → relative_sigma (default 5%)
            e.g., {'BV0': 0.05, 'Vth0': 0.03, 'Rb_neuron': 0.10}
        seed: Random seed

    Returns:
        dict: 'samples' (DataFrame-like), 'statistics', 'yield_estimate'
    """
    rng = np.random.RandomState(seed)

    # Default variability (from typical CMOS process)
    defaults = {
        'BV0': 0.05, 'k_vg': 0.03, 'Is': 0.20, 'Bf': 0.10,
        'Ne': 0.05, 'Vth0': 0.03, 'Cb': 0.10,
        'Rb_neuron': 0.15, 'Rb_synapse': 0.15,
    }
    if sigma_dict:
        defaults.update(sigma_dict)

    samples = {}
    for pname, sigma in defaults.items():
        nominal = getattr(params, pname, None)
        if nominal is not None and nominal != 0:
            samples[pname] = nominal * (1 + sigma * rng.randn(n_samples))
        else:
            samples[pname] = np.zeros(n_samples)

    # Compute derived quantities
    Vt = thermal_voltage(300)
    BVpar = samples['BV0'] - samples['k_vg'] * 0.3  # At Vg1=0.3V
    tau_neuron = np.abs(samples['Rb_neuron']) * np.abs(samples['Cb'])
    tau_synapse = np.abs(samples['Rb_synapse']) * np.abs(samples['Cb'])

    # Firing threshold spread
    firing_ok = (BVpar > 1.0) & (BVpar < 6.0)  # Functional range
    yield_pct = firing_ok.mean() * 100

    stats = {}
    for name, vals in [('BVpar', BVpar), ('tau_neuron', tau_neuron),
                        ('tau_synapse', tau_synapse)]:
        stats[name] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'cv': float(np.std(vals) / (np.mean(vals) + 1e-12)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        }

    return {
        'n_samples': n_samples,
        'yield_percent': float(yield_pct),
        'statistics': stats,
        'raw_samples': samples,
        'sigma_dict': defaults,
    }


# ═══════════════════════════════════════════════════════════════════
# TECHNOLOGY COMPARISON
# ═══════════════════════════════════════════════════════════════════

# Reference data from published literature
TECH_COMPARISON = {
    'NS-RAM (Pazos 2025)': {
        'area_um2': 17, 'energy_pJ': 0.415, 'endurance': 1e7,
        'levels': 14, 'retention_s': 1e4, 'speed_ns': 12600,
        'type': 'CMOS floating-body',
    },
    'CMOS LIF (18T)': {
        'area_um2': 900, 'energy_pJ': 10, 'endurance': 1e15,
        'levels': 0, 'retention_s': 0, 'speed_ns': 10,
        'type': 'digital CMOS',
    },
    'HfOx RRAM': {
        'area_um2': 4, 'energy_pJ': 0.1, 'endurance': 1e6,
        'levels': 8, 'retention_s': 1e7, 'speed_ns': 10,
        'type': 'metal-oxide memristor',
    },
    'PCM (GST)': {
        'area_um2': 20, 'energy_pJ': 100, 'endurance': 1e9,
        'levels': 16, 'retention_s': 1e7, 'speed_ns': 50,
        'type': 'phase-change',
    },
    'hBN memristor (Lanza 2025)': {
        'area_um2': 1, 'energy_pJ': 0.01, 'endurance': 1e6,
        'levels': 16, 'retention_s': 1e4, 'speed_ns': 100,
        'type': '2D material memristor',
    },
    'NOR Flash (Woo 2026)': {
        'area_um2': 50, 'energy_pJ': 50, 'endurance': 1e5,
        'levels': 256, 'retention_s': 1e8, 'speed_ns': 28,
        'type': 'flash memory',
    },
    'STT-MRAM': {
        'area_um2': 10, 'energy_pJ': 0.5, 'endurance': 1e12,
        'levels': 2, 'retention_s': 1e7, 'speed_ns': 5,
        'type': 'magnetic tunnel junction',
    },
}


def technology_comparison(custom_devices: Optional[Dict] = None,
                           metrics: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> Dict:
    """Compare NS-RAM against other neuromorphic technologies.

    Args:
        custom_devices: Additional devices to include (same format as TECH_COMPARISON)
        metrics: Which metrics to compare (default: all)
        save_path: If given, save comparison plot

    Returns:
        dict: comparison table + NS-RAM advantages/disadvantages
    """
    devices = dict(TECH_COMPARISON)
    if custom_devices:
        devices.update(custom_devices)

    if metrics is None:
        metrics = ['area_um2', 'energy_pJ', 'endurance', 'levels', 'retention_s']

    # Find NS-RAM advantages
    nsram = devices['NS-RAM (Pazos 2025)']
    advantages = []
    disadvantages = []

    for name, dev in devices.items():
        if name == 'NS-RAM (Pazos 2025)':
            continue
        # Area comparison
        if nsram['area_um2'] < dev['area_um2']:
            advantages.append(f"Smaller than {name} ({nsram['area_um2']} vs {dev['area_um2']} µm²)")
        elif nsram['area_um2'] > dev['area_um2']:
            disadvantages.append(f"Larger than {name} ({nsram['area_um2']} vs {dev['area_um2']} µm²)")

    if save_path and HAS_MPL:
        _plot_comparison(devices, metrics, save_path)

    return {
        'devices': devices,
        'advantages': advantages,
        'disadvantages': disadvantages,
        'nsram_unique': 'Dual neuron+synapse in single standard CMOS device — no special materials',
    }


def _plot_comparison(devices, metrics, save_path):
    """Generate technology comparison radar + bar plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0d1117')
    fig.suptitle('NS-RAM vs Neuromorphic Technologies',
                 fontsize=14, fontweight='bold', color='white')

    names = list(devices.keys())
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#FFEB3B']

    # Bar: Area
    ax = axes[0]; ax.set_facecolor('#0d1117')
    areas = [devices[n]['area_um2'] for n in names]
    ax.barh(range(len(names)), areas, color=colors[:len(names)], alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color='white', fontsize=7)
    ax.set_xlabel('Cell Area (µm²)', color='white')
    ax.set_title('(A) Area', color='white', fontsize=11)
    ax.set_xscale('log'); ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2, axis='x')

    # Bar: Energy
    ax = axes[1]; ax.set_facecolor('#0d1117')
    energy = [devices[n]['energy_pJ'] for n in names]
    ax.barh(range(len(names)), energy, color=colors[:len(names)], alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color='white', fontsize=7)
    ax.set_xlabel('Energy per Operation (pJ)', color='white')
    ax.set_title('(B) Energy Efficiency', color='white', fontsize=11)
    ax.set_xscale('log'); ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2, axis='x')

    # Bar: Endurance
    ax = axes[2]; ax.set_facecolor('#0d1117')
    endur = [devices[n]['endurance'] for n in names]
    ax.barh(range(len(names)), endur, color=colors[:len(names)], alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color='white', fontsize=7)
    ax.set_xlabel('Endurance (cycles)', color='white')
    ax.set_title('(C) Endurance', color='white', fontsize=11)
    ax.set_xscale('log'); ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0d1117')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# MODEL VS EXPERIMENT COMPARISON
# ═══════════════════════════════════════════════════════════════════

def compare_model_experiment(Vcb_exp, Id_exp, Vg1: float = 0.3,
                              params: Optional[DeviceParams] = None,
                              save_path: Optional[str] = None) -> Dict:
    """Overlay NS-RAM model prediction vs experimental data.

    Args:
        Vcb_exp: (N,) experimental voltage
        Id_exp: (N,) experimental current
        Vg1: Gate voltage
        params: DeviceParams (uses defaults if None)
        save_path: Save comparison plot

    Returns:
        dict: 'r_squared', 'rmse', 'max_error'
    """
    if params is None:
        params = DeviceParams()

    Vcb_model = np.linspace(Vcb_exp.min(), Vcb_exp.max(), 200)
    Id_model = np.array([avalanche_current(v, Vg1, params=params) for v in Vcb_model])

    # Interpolate model at experimental points for metrics
    Id_interp = np.interp(Vcb_exp, Vcb_model, Id_model)
    mask = (Id_exp > 0) & (Id_interp > 0)
    if mask.sum() < 2:
        return {'r_squared': 0, 'rmse': np.nan, 'error': 'No overlapping data'}

    log_exp = np.log10(Id_exp[mask])
    log_model = np.log10(Id_interp[mask])
    ss_res = np.sum((log_exp - log_model) ** 2)
    ss_tot = np.sum((log_exp - log_exp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((log_exp - log_model) ** 2))

    if save_path and HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.semilogy(Vcb_exp, Id_exp, 'o', color='#4ecdc4', markersize=5,
                     label=f'Experiment (Vg1={Vg1}V)', alpha=0.7)
        ax.semilogy(Vcb_model, Id_model, '-', color='#FF6B6B', linewidth=2,
                     label=f'Model (R²={r2:.4f})')
        ax.set_xlabel('Vcb (V)', color='white', fontsize=11)
        ax.set_ylabel('Id (A)', color='white', fontsize=11)
        ax.set_title('NS-RAM Model vs Experiment', color='white', fontsize=13, fontweight='bold')
        ax.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor='#0d1117')
        plt.close()

    return {'r_squared': float(r2), 'rmse': float(rmse)}
