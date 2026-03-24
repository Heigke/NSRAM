#!/usr/bin/env python3
"""Systematic STP parameter tuning for nsram.

Problem: Default STP hurts performance (95% baseline → 85% with het STP).
Root cause: U=0.5 is too aggressive — depletes synaptic resources too fast.

Strategy: Grid search over (U, tau_rec, tau_fac, alpha_theta, alpha_weight)
to find the sweet spot where STP HELPS without destroying the reservoir.

From the VG2 sweep (fig8): high VG2 (low U) is better → optimal U is LOW.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nsram.physics import DimensionlessParams
from nsram.network import NSRAMNetwork
from nsram.benchmarks import xor_accuracy, memory_capacity, narma_prediction, waveform_classification

rng = np.random.RandomState(42)
T = 4000; wo = 600
inputs = rng.uniform(-1, 1, T).astype(np.float64)

def evaluate(params, n_reps=3):
    metrics = []
    for rep in range(n_reps):
        net = NSRAMNetwork(N=128, params=params, seed=42 + rep * 100, backend='auto')
        out = net.run(inputs, noise_sigma=0.01)
        S = out['states']; spk = out['spikes']
        m = {
            'xor1': xor_accuracy(S, inputs, wo, 1),
            'xor2': xor_accuracy(S, inputs, wo, 2),
            'mc': memory_capacity(S, inputs, wo),
            'narma10': narma_prediction(S, inputs, wo, 10),
            'wave4': waveform_classification(S, inputs, wo),
            'active': int((spk.sum(1) > 0).sum()),
            'rate': spk.sum() / (128 * T),
        }
        metrics.append(m)
    avg = {k: np.mean([r[k] for r in metrics]) for k in metrics[0]}
    return avg

print("="*75)
print("  nsram STP Parameter Tuning")
print("="*75)

# ─── Phase 1: Baseline (no STP) ───
print("\n━━━ Phase 1: Baseline ━━━")
p_base = DimensionlessParams(U=0.0, alpha_theta=0.0, alpha_weight=0.0)
base = evaluate(p_base)
print(f"  NO STP: XOR1={base['xor1']:.1%} MC={base['mc']:.3f} "
      f"NAR10={base['narma10']:.3f} W4={base['wave4']:.1%} "
      f"rate={base['rate']:.3f}")

# ─── Phase 2: U sweep (key parameter) ───
print("\n━━━ Phase 2: U sweep (utilization) ━━━")
best_u = 0; best_score = 0
for U in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
    p = DimensionlessParams(U=U, tau_rec=10.0, tau_fac=5.0,
                             alpha_theta=0.15, alpha_weight=0.30)
    m = evaluate(p, n_reps=2)
    score = m['xor1'] + m['mc']/3 + m['narma10'] + m['wave4']
    flag = " ★" if score > best_score else ""
    if score > best_score:
        best_score = score; best_u = U
    print(f"  U={U:.2f}: XOR1={m['xor1']:.1%} MC={m['mc']:.3f} "
          f"NAR10={m['narma10']:.3f} W4={m['wave4']:.1%} "
          f"score={score:.3f}{flag}")

print(f"\n  Best U: {best_u}")

# ─── Phase 3: tau_rec sweep at best U ───
print(f"\n━━━ Phase 3: tau_rec sweep (U={best_u}) ━━━")
best_trec = 10; best_score2 = 0
for trec in [2, 5, 8, 10, 15, 20, 30, 50, 100]:
    p = DimensionlessParams(U=best_u, tau_rec=trec, tau_fac=5.0,
                             alpha_theta=0.15, alpha_weight=0.30)
    m = evaluate(p, n_reps=2)
    score = m['xor1'] + m['mc']/3 + m['narma10'] + m['wave4']
    flag = " ★" if score > best_score2 else ""
    if score > best_score2:
        best_score2 = score; best_trec = trec
    print(f"  τ_rec={trec:3d}: XOR1={m['xor1']:.1%} MC={m['mc']:.3f} "
          f"NAR10={m['narma10']:.3f} W4={m['wave4']:.1%} score={score:.3f}{flag}")

print(f"\n  Best τ_rec: {best_trec}")

# ─── Phase 4: alpha sweep at best U, tau_rec ───
print(f"\n━━━ Phase 4: alpha sweep (U={best_u}, τ_rec={best_trec}) ━━━")
best_ath = 0; best_aw = 0; best_score3 = 0
for ath in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    for aw in [0.0, 0.10, 0.20, 0.30, 0.50]:
        p = DimensionlessParams(U=best_u, tau_rec=best_trec, tau_fac=5.0,
                                 alpha_theta=ath, alpha_weight=aw)
        m = evaluate(p, n_reps=2)
        score = m['xor1'] + m['mc']/3 + m['narma10'] + m['wave4']
        if score > best_score3:
            best_score3 = score; best_ath = ath; best_aw = aw
            print(f"  α_θ={ath:.2f} α_w={aw:.2f}: XOR1={m['xor1']:.1%} MC={m['mc']:.3f} "
                  f"NAR10={m['narma10']:.3f} W4={m['wave4']:.1%} score={score:.3f} ★")

print(f"\n  Best alphas: α_θ={best_ath}, α_w={best_aw}")

# ─── Phase 5: tau_fac sweep ───
print(f"\n━━━ Phase 5: tau_fac sweep ━━━")
best_tfac = 5; best_score4 = 0
for tfac in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    p = DimensionlessParams(U=best_u, tau_rec=best_trec, tau_fac=tfac,
                             alpha_theta=best_ath, alpha_weight=best_aw)
    m = evaluate(p, n_reps=2)
    score = m['xor1'] + m['mc']/3 + m['narma10'] + m['wave4']
    flag = " ★" if score > best_score4 else ""
    if score > best_score4:
        best_score4 = score; best_tfac = tfac
    print(f"  τ_fac={tfac:5.1f}: XOR1={m['xor1']:.1%} MC={m['mc']:.3f} "
          f"NAR10={m['narma10']:.3f} W4={m['wave4']:.1%} score={score:.3f}{flag}")

print(f"\n  Best τ_fac: {best_tfac}")

# ─── Phase 6: bg_frac + spectral_radius joint sweep ───
print(f"\n━━━ Phase 6: bg_frac × spectral_radius sweep ━━━")
best_bg = 0.95; best_sr = 0.90; best_score5 = 0
for bg in [0.88, 0.90, 0.92, 0.95, 0.98]:
    for sr in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05]:
        p = DimensionlessParams(U=best_u, tau_rec=best_trec, tau_fac=best_tfac,
                                 alpha_theta=best_ath, alpha_weight=best_aw,
                                 bg_frac=bg, spectral_radius=sr)
        m = evaluate(p, n_reps=2)
        score = m['xor1'] + m['mc']/3 + m['narma10'] + m['wave4']
        if score > best_score5:
            best_score5 = score; best_bg = bg; best_sr = sr
            print(f"  bg={bg:.2f} sr={sr:.2f}: XOR1={m['xor1']:.1%} MC={m['mc']:.3f} "
                  f"NAR10={m['narma10']:.3f} W4={m['wave4']:.1%} score={score:.3f} ★")

print(f"\n  Best: bg_frac={best_bg}, sr={best_sr}")

# ─── Phase 7: Final validation with best params ───
print(f"\n━━━ Phase 7: Final Validation (5 reps) ━━━")
p_best = DimensionlessParams(
    U=best_u, tau_rec=best_trec, tau_fac=best_tfac,
    alpha_theta=best_ath, alpha_weight=best_aw,
    bg_frac=best_bg, spectral_radius=best_sr,
)
print(f"  Params: U={best_u}, τ_rec={best_trec}, τ_fac={best_tfac}, "
      f"α_θ={best_ath}, α_w={best_aw}, bg={best_bg}, sr={best_sr}")

final_stp = evaluate(p_best, n_reps=5)
final_base = evaluate(p_base, n_reps=5)

print(f"\n  {'Metric':<12s}  {'No STP':>8s}  {'Best STP':>8s}  {'Delta':>8s}")
print(f"  {'-'*42}")
for k in ['xor1', 'xor2', 'mc', 'narma10', 'wave4']:
    b = final_base[k]; s = final_stp[k]; d = s - b
    if k in ('xor1', 'xor2', 'wave4'):
        print(f"  {k:<12s}  {b:>7.1%}  {s:>7.1%}  {d*100:>+7.1f}pp")
    else:
        print(f"  {k:<12s}  {b:>8.3f}  {s:>8.3f}  {d:>+8.3f}")

# Save best params
import json
out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'best_stp_params.json')
with open(out_path, 'w') as f:
    json.dump({
        'U': best_u, 'tau_rec': best_trec, 'tau_fac': best_tfac,
        'alpha_theta': best_ath, 'alpha_weight': best_aw,
        'bg_frac': best_bg, 'spectral_radius': best_sr,
        'baseline': {k: float(v) for k, v in final_base.items()},
        'best_stp': {k: float(v) for k, v in final_stp.items()},
    }, f, indent=2)
print(f"\n  Saved: {out_path}")
