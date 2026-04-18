"""nsram.bsim4 — BSIM4-native 2T floating-body neuron

Built to match the modeling direction in Pazos et al. (2026) where the
avalanche-diode + parasitic-BJT subcircuit is replaced by BSIM4's
native impact ionization (Iii) + body-bias effects. No explicit
BVpar control parameter — the firing mechanism emerges from
Iii charging the floating body, Vbs lowering Vth, and the resulting
runaway drain current.

Equations follow the BSIM4.3.0 manual, UC Berkeley (2003). Section
references in docstrings point to chapters in that manual.

Four coupled current sources set the body charge balance:

    Cb · dVB/dt = Iii(Vds,Vgs)            # §6.1  impact ionization
                + IGIDL(Vds,Vgs) + IGISL  # §6.2  GIDL/GISL
                - Ibs(Vbs) - Ibd(Vbd)     # §10.1 junction diodes
                - VB/Rb                   # bulk leakage path

Vbs-dependent threshold (§2.2):

    Vth(Vbs) = VTH0 + K1·(√(Φs − Vbs) − √Φs) − K2·Vbs

BSIM4 has no native floating-body mode; that's BSIMSOI territory.
This module adds the body-charge integrator externally on top of
BSIM4's current equations, which is exactly what Sebastian is
doing in his LTSpice model with a complementary bipolar current.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

from nsram.physics import K_B, Q_E, EPS_SI, EPS_OX, thermal_voltage

Array = Union[float, np.ndarray]


# ═══════════════════════════════════════════════════════════════════════
# BSIM4 PARAMETERS (Appendix A of BSIM4.3.0 manual)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BSIM4Params:
    """BSIM4 compact-model parameters for a 2T NS-RAM floating-body cell.

    Defaults target a 180 nm bulk NMOS consistent with the original
    Pazos et al. Zenodo model card (PTM130bulk_lite.txt).  Override
    from a fit against Sebastian's measured I-V sweeps via
    nsram.fitting.fit_bsim4_impact().

    Groups:
        — Geometry (§A.1)
        — Threshold / body effect (§2.2, A.3)
        — Drain current BSIM-lite (§3, simplified)
        — Impact ionization (§6.1, A.5)
        — GIDL / GISL (§6.2, A.6)
        — Junction diodes (§10.1, A.12)
        — Body capacitance / bulk resistance (external)
    """

    # ── Geometry ─────────────────────────────────────────────────
    Leff: float = 180e-9        # m    effective channel length
    Weff: float = 1.0e-6        # m    effective channel width
    WeffCJ: float = 1.0e-6      # m    effective width for GIDL/GISL
    Nf: int = 1                 # —    number of fingers
    Aseff: float = 1.0e-13      # m²   source area  (~Weff·Ls)
    Adeff: float = 1.0e-13      # m²   drain  area
    Pseff: float = 3.0e-6       # m    source perimeter (excl. gate edge)
    Pdeff: float = 3.0e-6       # m    drain  perimeter

    # ── Oxide / substrate ────────────────────────────────────────
    Toxe: float = 3.3e-9        # m    electrical oxide thickness
    Nsub: float = 5.0e23        # m⁻³  substrate doping (5e17 cm⁻³ for 180 nm)

    # ── Threshold & body effect (§2.2, §A.3) ─────────────────────
    VTH0: float = 0.432         # V    long-channel Vth at Vbs=0
    K1: float = 0.55            # V^½  first-order body coefficient
    K2: float = 0.03            # —    second-order body coefficient
    PhiS: float = 0.80          # V    surface potential 2·φF

    # ── Drain current BSIM-lite (§3) ─────────────────────────────
    mu0: float = 0.045          # m²/Vs   channel mobility (450 cm²/Vs, NMOS)
    Vdsat0: float = 0.3         # V       saturation-region onset reference
    lambda_clm: float = 0.05    # 1/V     channel-length modulation

    # ── Impact ionization (§6.1, §A.5) ───────────────────────────
    ALPHA0: float = 6.0e-6      # A·m/V   first Iii coefficient (180 nm)
    ALPHA1: float = 0.0         # A/V     length-scaling Iii coefficient
    BETA0:  float = 22.0        # V       Chynoweth exponential coefficient

    # ── GIDL / GISL (§6.2, §A.6) ─────────────────────────────────
    AGIDL: float = 1.0e-10      # mho    pre-exponential GIDL coefficient
    BGIDL: float = 2.3e9        # V/m    exponential GIDL coefficient
    CGIDL: float = 0.5          # V³     body-bias dependence
    EGIDL: float = 0.8          # V      band-bending parameter
    # GISL shares the BSIM4.3.0 parameter set (later versions split)

    # ── Junction diodes (§10.1, §A.12) ───────────────────────────
    JSS: float = 1.0e-4         # A/m²   source bottom reverse sat current density
    JSD: float = 1.0e-4         # A/m²   drain  bottom
    JSWS: float = 0.0           # A/m    source sidewall
    JSWD: float = 0.0           # A/m    drain  sidewall
    NJS: float = 1.0            # —      source ideality factor
    NJD: float = 1.0            # —      drain  ideality factor
    IJTHSFWD: float = 0.1       # A      source forward-bias limiting current
    IJTHDFWD: float = 0.1       # A      drain  forward-bias limiting current

    # ── Body node (external — not in BSIM4) ──────────────────────
    Cb: float = 1.0e-12         # F      floating-body capacitance
    Rb: float = 1.0e6           # Ω      bulk leakage resistance
    Vbs_max: float = 1.2        # V      soft clamp on Vbs (avoid Gummel overflow)

    @property
    def body_tau(self) -> float:
        """Body RC time constant (s)."""
        return self.Rb * self.Cb


# Preset cards — extend as Sebastian sends measurement-calibrated values.
BSIM4_PRESETS = {
    "ns_ram_180nm": BSIM4Params(),
    "ns_ram_180nm_hot": BSIM4Params(
        VTH0=0.39, ALPHA0=8.0e-6, BETA0=20.0, AGIDL=3e-10,
    ),
    "generic_65nm": BSIM4Params(
        Leff=65e-9, Toxe=1.4e-9, Nsub=1.5e24,
        VTH0=0.42, K1=0.45, K2=0.01,
        ALPHA0=1.0e-5, BETA0=16.0,
        AGIDL=1.0e-9, JSS=5e-4,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# THRESHOLD & DRAIN CURRENT (§2.2, §3)
# ═══════════════════════════════════════════════════════════════════════

def vth_bsim4(Vbs: Array, p: BSIM4Params) -> Array:
    """Threshold voltage with BSIM4 body effect (§2.2, eq. 2.2.6).

        Vth(Vbs) = VTH0 + K1·(√(Φs − Vbs) − √Φs) − K2·Vbs

    Uses Vbseff-style soft clamp so the √ argument stays positive.
    """
    Vbs = np.asarray(Vbs, dtype=np.float64)
    # Soft clamp to prevent √(neg) when body goes deeply forward-biased
    Vbs_clip = np.minimum(Vbs, p.PhiS - 1e-3)
    delta_phi = np.sqrt(p.PhiS - Vbs_clip) - np.sqrt(p.PhiS)
    return p.VTH0 + p.K1 * delta_phi - p.K2 * Vbs_clip


def drain_current_bsim(Vgs: Array, Vds: Array, Vbs: Array,
                       p: BSIM4Params) -> Tuple[Array, Array]:
    """BSIM-lite Ids(Vgs, Vds, Vbs) with Vth(Vbs) body effect.

    Returns (Ids, Vdsat). This is NOT full BSIM4 — it is the
    compact drain-current skeleton required to drive the
    impact-ionization equation.  For fits to measurement we
    scale (mu0, Leff, Weff) — the shape matches BSIM4 in the
    regime where impact ionization dominates (saturation,
    moderate-to-strong inversion).
    """
    Vgs = np.asarray(Vgs, dtype=np.float64)
    Vds = np.asarray(Vds, dtype=np.float64)
    Vbs = np.asarray(Vbs, dtype=np.float64)
    Vth = vth_bsim4(Vbs, p)
    Cox = EPS_OX / p.Toxe
    beta = p.mu0 * Cox * p.Weff / p.Leff

    Vgt = np.maximum(Vgs - Vth, 0.0)
    Vdsat = np.maximum(Vgt, p.Vdsat0 * 1e-3)  # strictly positive
    Vdseff = np.minimum(Vds, Vdsat)

    # Triode  : Ids = β·[Vgt·Vds − Vds²/2]
    # Saturation: Ids = (β/2)·Vgt²·[1 + λ·(Vds − Vdsat)]
    I_triode = beta * (Vgt * Vdseff - 0.5 * Vdseff * Vdseff)
    I_sat = 0.5 * beta * Vgt * Vgt * (1.0 + p.lambda_clm * np.maximum(Vds - Vdsat, 0.0))
    Ids = np.where(Vds < Vdsat, I_triode, I_sat)
    Ids = np.where(Vgt <= 0.0, 0.0, Ids)
    return Ids, Vdsat


# ═══════════════════════════════════════════════════════════════════════
# IMPACT IONIZATION (§6.1, eq. 6.1.1)
# ═══════════════════════════════════════════════════════════════════════

def impact_ionization_bsim4(Vgs: Array, Vds: Array, Vbs: Array,
                             p: BSIM4Params) -> Array:
    """Iii(Vgs, Vds, Vbs) — BSIM4 impact-ionization current (§6.1).

        Iii = [(ALPHA0 + ALPHA1·Leff)/Leff] · (Vds − Vdseff)
              · exp(−BETA0 / (Vds − Vdseff)) · Ids

    This replaces the Zenodo Chynoweth/BVpar avalanche model.  The
    control knob is now (ALPHA0, BETA0) plus the device Vth(Vbs) —
    no breakdown voltage as a free parameter.
    """
    Ids, Vdsat = drain_current_bsim(Vgs, Vds, Vbs, p)
    dv = np.maximum(np.asarray(Vds, dtype=np.float64) - Vdsat, 1e-9)
    prefactor = (p.ALPHA0 + p.ALPHA1 * p.Leff) / p.Leff
    # Chynoweth exponential: exp(−BETA0/dv). BETA0 has units of V.
    exp_arg = np.clip(-p.BETA0 / dv, -80.0, 0.0)
    return prefactor * dv * np.exp(exp_arg) * Ids


# ═══════════════════════════════════════════════════════════════════════
# GIDL / GISL (§6.2, eqs. 6.2.1–6.2.2)
# ═══════════════════════════════════════════════════════════════════════

def gidl_current(Vds: Array, Vgs: Array, Vbs: Array,
                 p: BSIM4Params) -> Array:
    """IGIDL — gate-induced drain leakage (§6.2, eq. 6.2.1).

        IGIDL = AGIDL·WeffCJ·Nf · (Vds − Vgs − EGIDL)/(3·Toxe)
                · exp(−3·Toxe·BGIDL / (Vds − Vgs − EGIDL))
                · Vdb³ / (CGIDL + Vdb³)

    Secondary body-charging term, important at sub-65 nm but
    usually small at 180 nm unless you boost AGIDL.
    """
    Vds = np.asarray(Vds, dtype=np.float64)
    Vgs = np.asarray(Vgs, dtype=np.float64)
    Vbs = np.asarray(Vbs, dtype=np.float64)
    Vdg = Vds - Vgs - p.EGIDL
    Vdb = np.maximum(Vds - Vbs, 0.0)

    # Guard: GIDL only flows when Vdg > 0 (gate held low, drain high)
    Vdg = np.where(Vdg > 0.0, Vdg, 1e-12)
    exp_arg = np.clip(-3.0 * p.Toxe * p.BGIDL / Vdg, -80.0, 0.0)
    field_term = Vdg / (3.0 * p.Toxe)
    vb_term = (Vdb ** 3) / (p.CGIDL + Vdb ** 3 + 1e-18)
    igidl = p.AGIDL * p.WeffCJ * p.Nf * field_term * np.exp(exp_arg) * vb_term
    return np.where(Vds > Vgs + p.EGIDL, igidl, 0.0)


def gisl_current(Vds: Array, Vgs: Array, Vbs: Array,
                 p: BSIM4Params) -> Array:
    """IGISL — source-side counterpart of GIDL (§6.2, eq. 6.2.2)."""
    # For Vds > 0 in NMOS, GISL is negligible; kept for completeness.
    Vgd = -np.asarray(Vds, dtype=np.float64) - np.asarray(Vgs, dtype=np.float64) - p.EGIDL
    Vsb = -np.asarray(Vbs, dtype=np.float64)
    Vsb = np.maximum(Vsb, 0.0)
    Vgd = np.where(Vgd > 0.0, Vgd, 1e-12)
    exp_arg = np.clip(-3.0 * p.Toxe * p.BGIDL / Vgd, -80.0, 0.0)
    field_term = Vgd / (3.0 * p.Toxe)
    vb_term = (Vsb ** 3) / (p.CGIDL + Vsb ** 3 + 1e-18)
    return p.AGIDL * p.WeffCJ * p.Nf * field_term * np.exp(exp_arg) * vb_term


# ═══════════════════════════════════════════════════════════════════════
# JUNCTION DIODES (§10.1, eqs. 10.1.1 & 10.1.6)
# ═══════════════════════════════════════════════════════════════════════

def body_diode_current(Vbs: Array, p: BSIM4Params, T: float = 300.0,
                       side: str = "source") -> Array:
    """Ibs or Ibd — source/drain-body diode (§10.1).

        Ibx = Isbx · [exp(q·Vbx / (NJx·kT)) − 1]

    Linearised above IJTHxFWD so the Gummel-Poon shoulder does not
    overflow at heavy forward bias — matches BSIM4's own limiter.
    """
    Vbx = np.asarray(Vbs, dtype=np.float64)
    Vt = thermal_voltage(T)
    if side == "source":
        Js, Jsw, NJ, Ilim = p.JSS, p.JSWS, p.NJS, p.IJTHSFWD
        A, P = p.Aseff, p.Pseff
    else:
        Js, Jsw, NJ, Ilim = p.JSD, p.JSWD, p.NJD, p.IJTHDFWD
        A, P = p.Adeff, p.Pdeff
    Isat = A * Js + P * Jsw

    exp_arg = np.clip(Vbx / (NJ * Vt), -80.0, 80.0)
    Idio = Isat * (np.exp(exp_arg) - 1.0)

    # Soft-limit above IJTHFWD: linearise around the limit voltage.
    Vlim = NJ * Vt * np.log1p(Ilim / max(Isat, 1e-30))
    linear_branch = Ilim + (Isat / (NJ * Vt)) * np.exp(
        np.clip(Vlim / (NJ * Vt), -80.0, 80.0)
    ) * (Vbx - Vlim)
    return np.where(Vbx > Vlim, linear_branch, Idio)


# ═══════════════════════════════════════════════════════════════════════
# BODY-CHARGE ODE (BSIM4-native, replaces nsram.physics.body_charge_ode)
# ═══════════════════════════════════════════════════════════════════════

def body_charge_ode_bsim4(t: float, state, Vds, Vgs, p: BSIM4Params,
                          T: float = 300.0):
    """Floating-body charge balance driven by BSIM4 currents.

        Cb · dVB/dt = Iii + IGIDL − Ibs − Ibd − VB/Rb

    VB is the body potential measured from source (Vbs = VB).  There
    is no BVpar or avalanche voltage — all firing comes from Iii
    lowering Vth(Vbs) and driving Ids up until Ibs clamps it.

    State:
        [VB]           — body potential (V), single-state version
        [VB, Q_trap]   — optional SRH trap occupancy (0..1)

    Args:
        t         time (s), unused but required by scipy signature.
        state     [VB] or [VB, Q_trap].
        Vds       drain–source voltage (V) or callable(t)→V.
        Vgs       gate–source voltage (V) or callable(t)→V.
        p         BSIM4Params.
        T         temperature (K).

    Returns:
        Derivatives matching the input state shape.
    """
    VB = float(state[0])
    has_trap = len(state) > 1
    Q_trap = float(state[1]) if has_trap else 0.0

    _Vds = Vds(t) if callable(Vds) else float(Vds)
    _Vgs = Vgs(t) if callable(Vgs) else float(Vgs)

    Vbs = min(VB, p.Vbs_max)   # soft clamp

    Iii = float(impact_ionization_bsim4(_Vgs, _Vds, Vbs, p))
    Igidl = float(gidl_current(_Vds, _Vgs, Vbs, p))
    Ibs = float(body_diode_current(Vbs, p, T=T, side="source"))
    # Drain–body is reverse-biased (Vbd = Vbs − Vds ≈ negative for Vds>0)
    Ibd = float(body_diode_current(Vbs - _Vds, p, T=T, side="drain"))
    I_Rb = VB / p.Rb if p.Rb > 0 else 0.0

    dVB = (Iii + Igidl - Ibs - Ibd - I_Rb) / p.Cb

    if not has_trap:
        return [dVB]

    # Simple SRH trap bookkeeping for synaptic-mode retention.
    spike_indicator = 1.0 if VB > 0.6 else 0.0
    k_cap = 5e3
    k_em = 2e2
    dQ = k_cap * (1.0 - Q_trap) * spike_indicator - k_em * Q_trap
    return [dVB, dQ]


# ═══════════════════════════════════════════════════════════════════════
# 2T CELL TOPOLOGY (Sebastian's floating-body architecture)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TwoTransistorCell:
    """Pazos 2T NS-RAM cell in BSIM4 description.

    M1 (read/fire): floating body, drives Iii → charges VB.
    M2 (control) : sets effective Rb from body node to ground
                   (acts as the tunable bulk leakage path that
                   switches between neuron and synapse modes).

    Control knobs the user drives directly:
        Vg1  — gate of M1  (triggers impact ionization)
        Vg2  — gate of M2  (modulates Rb_eff, 10 kΩ … 1 MΩ)
        Vds  — drain–source of M1

    Outputs a spike whenever VB crosses Vth_spike from below.
    """

    bsim: BSIM4Params = field(default_factory=BSIM4Params)
    Vth_spike: float = 0.6         # V   body FB onset (snapback trigger)
    Vreset: float = 0.05           # V   post-spike body residual
    t_refrac: float = 1.6e-6       # s   refractory period

    # M2-controlled bulk leakage — maps Vg2 to Rb.
    Rb_neuron: float = 10e3
    Rb_synapse: float = 1e6
    Vg2_switch: float = 0.5        # V   midpoint of mode transition

    def rb_from_vg2(self, Vg2: float) -> float:
        """Sigmoid blend between neuron-mode Rb and synapse-mode Rb."""
        w = 1.0 / (1.0 + np.exp(-(Vg2 - self.Vg2_switch) / 0.05))
        return self.Rb_synapse * w + self.Rb_neuron * (1.0 - w)

    def simulate(self, Vg1: Callable[[float], float] | float,
                 Vg2: Callable[[float], float] | float,
                 Vds: Callable[[float], float] | float,
                 t_end: float, dt: float = 1e-8,
                 T: float = 300.0,
                 VB0: float = 0.0) -> dict:
        """Forward-Euler integrate the BSIM4 2T cell.

        Returns dict with arrays: t, VB, Iii, Ibs, Ids, spikes (indices).
        """
        n = int(np.ceil(t_end / dt))
        t = np.arange(n) * dt
        VB = np.empty(n, dtype=np.float64)
        Iii_log = np.empty(n, dtype=np.float64)
        Ibs_log = np.empty(n, dtype=np.float64)
        Ids_log = np.empty(n, dtype=np.float64)

        vb = VB0
        refrac_until = -1.0
        spikes = []
        for i in range(n):
            ti = t[i]
            _Vg1 = Vg1(ti) if callable(Vg1) else float(Vg1)
            _Vg2 = Vg2(ti) if callable(Vg2) else float(Vg2)
            _Vds = Vds(ti) if callable(Vds) else float(Vds)

            # Snap Rb each step — Vg2 can be time-varying.
            p = self.bsim
            p.Rb = self.rb_from_vg2(_Vg2)

            Vbs = min(vb, p.Vbs_max)
            Iii = float(impact_ionization_bsim4(_Vg1, _Vds, Vbs, p))
            Ibs = float(body_diode_current(Vbs, p, T=T, side="source"))
            Ibd = float(body_diode_current(Vbs - _Vds, p, T=T, side="drain"))
            Ids, _ = drain_current_bsim(_Vg1, _Vds, Vbs, p)
            Ids = float(Ids)
            I_Rb = vb / p.Rb

            dVB = (Iii - Ibs - Ibd - I_Rb) / p.Cb
            vb = vb + dt * dVB

            # Spike detect (rising edge across Vth_spike, outside refractory).
            fired = (vb >= self.Vth_spike) and (ti >= refrac_until)
            if fired:
                spikes.append(i)
                refrac_until = ti + self.t_refrac
                vb = self.Vreset   # partial reset — keep some charge

            VB[i] = vb
            Iii_log[i] = Iii
            Ibs_log[i] = Ibs
            Ids_log[i] = Ids

        return {
            "t": t, "VB": VB, "Iii": Iii_log, "Ibs": Ibs_log, "Ids": Ids_log,
            "spikes": np.asarray(spikes, dtype=np.int64),
        }


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE — map legacy DeviceParams → BSIM4Params for quick trials
# ═══════════════════════════════════════════════════════════════════════

def from_device_params(dp) -> BSIM4Params:
    """Best-effort bridge from nsram.physics.DeviceParams to BSIM4Params.

    Keeps geometry, Vth0, Cb, Rb from the legacy card.  Impact-ionization
    coefficients are left at BSIM4 defaults — calibrate against
    measurement with nsram.fitting.fit_bsim4_impact().
    """
    return BSIM4Params(
        Leff=dp.Ln, Weff=dp.Wn, WeffCJ=dp.Wn,
        Toxe=dp.Tox, VTH0=dp.Vth0,
        Cb=dp.Cb, Rb=dp.Rb_default,
    )
