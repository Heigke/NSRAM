"""VG2 sweep — demonstrates the SRH↔TM bridge.

Sweeps the VG2 center voltage across the array and measures how
reservoir computing performance changes. This is the core scientific
finding: VG2 controls the STP type, which controls temporal processing.
"""

import sys; sys.path.insert(0, '.')
import numpy as np
from nsram.physics import DimensionlessParams, charge_capture_rate
from nsram.network import NSRAMNetwork
from nsram.benchmarks import xor_accuracy, memory_capacity, narma_prediction

rng = np.random.RandomState(42)
T = 4000; wo = 600
inputs = rng.uniform(-1, 1, T).astype(np.float64)

print("="*65)
print("  VG2 → STP Type → Reservoir Performance")
print("="*65)
print(f"\n  {'VG2_mid':>7s}  {'U_mean':>7s}  {'STP type':>12s}  {'XOR-1':>7s}  {'MC':>7s}  {'NAR10':>7s}")
print(f"  {'-'*58}")

for v_mid in np.arange(0.28, 0.52, 0.02):
    p = DimensionlessParams()
    net = NSRAMNetwork(N=128, params=p, seed=42, backend='auto')

    # Override VG2 distribution
    new_vg2 = (v_mid - 0.04 + 0.08 * np.random.RandomState(42).rand(128)).astype(np.float32)
    net._Vg2 = new_vg2
    k_cap_new = charge_capture_rate(new_vg2, 1000.0, 0.40, 0.05)
    net._U_base = np.clip(k_cap_new / 1000.0 * p.U * 2, 0.01, 0.95).astype(np.float32)
    net._tau_fac = ((1.0 - net._U_base) * p.tau_fac).astype(np.float32)
    if net.backend == 'torch':
        net._to_device()

    out = net.run(inputs, noise_sigma=0.01)
    S = out['states']

    u_mean = net._U_base.mean()
    stp_type = "STD" if u_mean > 0.5 else "mixed" if u_mean > 0.2 else "STF"
    xor1 = xor_accuracy(S, inputs, wo, 1)
    mc = memory_capacity(S, inputs, wo)
    nar = narma_prediction(S, inputs, wo, 10)

    print(f"  {v_mid:>7.2f}V  {u_mean:>7.3f}  {stp_type:>12s}  {xor1:>6.1%}  {mc:>7.3f}  {nar:>7.3f}")
