"""BSIM4-native 2T NS-RAM floating-body neuron

Demonstrates the BSIM4 coupling Sebastian asked about in the
21-Apr NSRAM exchange: the firing mechanism is driven by BSIM4's
native impact ionization (ALPHA0, BETA0) + body-bias Vth(Vbs)
rather than the Zenodo avalanche/BVpar model.

Three sections:
  1. Iii(Vds) sweep at several Vgs, Vbs values — the raw BSIM4 physics
  2. 2T cell spiking in neuron mode (low Rb via Vg2) and synapse mode
  3. Fit ALPHA0 / BETA0 against a measured I-V curve (CSV-in ready)

Run:
    python examples/bsim4_2t_floating_body.py
"""

import numpy as np

from nsram import (
    BSIM4Params, BSIM4_PRESETS,
    impact_ionization_bsim4, vth_bsim4,
    TwoTransistorCell,
    fit_bsim4_impact, fit_bsim4_family,
    from_device_params, DeviceParams,
)


def demo_iii_sweep():
    print("── 1. Impact-ionization sweep (BSIM4 §6.1) ──")
    p = BSIM4_PRESETS["ns_ram_180nm"]
    Vds = np.linspace(1.0, 4.5, 15)
    for Vgs in (0.6, 0.9, 1.2):
        Iii = np.array([float(impact_ionization_bsim4(Vgs, v, 0.0, p))
                        for v in Vds])
        print(f"  Vgs={Vgs:.1f} V  |  Iii(1.5V)={Iii[2]:.2e}  "
              f"Iii(3.0V)={Iii[8]:.2e}  Iii(4.5V)={Iii[-1]:.2e}")

    print("\n── Body-bias lowers Vth (BSIM4 §2.2) ──")
    for Vbs in np.linspace(-0.3, 0.6, 7):
        print(f"  Vbs={Vbs:+.2f} V  →  Vth={float(vth_bsim4(Vbs, p)):.3f} V")


def demo_2t_cell():
    print("\n── 2. 2T floating-body cell ──")
    cell = TwoTransistorCell()

    # Neuron mode: Vg2 low → Rb ≈ 10 kΩ  (fast body drain, oscillates)
    res = cell.simulate(Vg1=0.9, Vg2=0.2, Vds=3.8, t_end=20e-6, dt=1e-8)
    rate_kHz = len(res["spikes"]) / (res["t"][-1] + 1e-15) / 1e3
    print(f"  neuron mode  (Vg2=0.2): rate={rate_kHz:7.1f} kHz, "
          f"VB_max={res['VB'].max():.3f} V, peak Iii={res['Iii'].max():.2e} A")

    # Synapse mode: Vg2 high → Rb ≈ 1 MΩ  (charge retention, non-spiking)
    res = cell.simulate(Vg1=0.9, Vg2=0.9, Vds=3.8, t_end=20e-6, dt=1e-8)
    rate_kHz = len(res["spikes"]) / (res["t"][-1] + 1e-15) / 1e3
    print(f"  synapse mode (Vg2=0.9): rate={rate_kHz:7.1f} kHz, "
          f"VB_max={res['VB'].max():.3f} V, peak Iii={res['Iii'].max():.2e} A")


def demo_fit():
    print("\n── 3. Fit ALPHA0 / BETA0 against (synthetic) measurement ──")
    # Pretend this came from Sebastian's LTSpice / silicon CSV at Vgs=1.0 V
    true = BSIM4Params(ALPHA0=4.0e-6, BETA0=24.0)
    Vds = np.linspace(1.0, 4.5, 25)
    Isub = np.array([float(impact_ionization_bsim4(1.0, v, 0.0, true))
                     for v in Vds])
    rng = np.random.default_rng(0)
    Isub *= 1.0 + 0.05 * rng.standard_normal(len(Vds))  # 5 % noise
    Isub = np.maximum(Isub, 1e-30)

    fit = fit_bsim4_impact(Vds, Isub, Vgs=1.0, Vbs=0.0,
                            base=BSIM4Params())
    print(f"  true  : ALPHA0=4.00e-6  BETA0=24.00")
    print(f"  fit   : ALPHA0={fit['ALPHA0']:.2e}  BETA0={fit['BETA0']:.2f}  "
          f"R²={fit['r_squared']:.4f}")
    print("  → fit['params'] is ready to drop into TwoTransistorCell or "
          "body_charge_ode_bsim4.")


def demo_legacy_bridge():
    print("\n── 4. Legacy DeviceParams → BSIM4 bridge ──")
    dp = DeviceParams()
    bp = from_device_params(dp)
    print(f"  Leff={bp.Leff*1e9:.0f} nm, VTH0={bp.VTH0:.3f} V, "
          f"Cb={bp.Cb*1e12:.2f} pF, Rb={bp.Rb:.0e} Ω")
    print("  (ALPHA0/BETA0 stay at BSIM4 defaults until you fit measurement.)")


if __name__ == "__main__":
    demo_iii_sweep()
    demo_2t_cell()
    demo_fit()
    demo_legacy_bridge()
