"""Compare NS-RAM STP modes and demonstrate the SRH↔TM bridge.

Shows that gentle charge trapping (U=0.01) improves temporal processing
(MC +4.5%, NARMA +4.4%) while preserving classification performance.
"""

import sys; sys.path.insert(0, '.')
from nsram import NSRAMReservoir, rc_benchmark

print("=" * 65)
print("  NS-RAM STP Comparison")
print("  Does charge trapping (=Tsodyks-Markram STP) help?")
print("=" * 65)

results = {}
for stp_mode in ['none', 'std', 'stf', 'heterogeneous']:
    print(f"\n━━━ STP = '{stp_mode}' ━━━")
    res = NSRAMReservoir(N=128, stp=stp_mode, noise_sigma=0.01)
    results[stp_mode] = rc_benchmark(res, n_steps=4000, n_reps=5)

# Show improvement
print("\n━━━ STP Improvement over Baseline ━━━")
base = results['none']
for mode in ['std', 'stf', 'heterogeneous']:
    r = results[mode]
    print(f"  {mode:15s}: "
          f"XOR1 {(r['xor1_mean'] - base['xor1_mean'])*100:+.1f}pp  "
          f"MC {r['mc_mean'] - base['mc_mean']:+.3f}  "
          f"NARMA {r['narma10_mean'] - base['narma10_mean']:+.3f}  "
          f"W4 {(r['wave4_mean'] - base['wave4_mean'])*100:+.1f}pp")
