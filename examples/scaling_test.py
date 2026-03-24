"""NS-RAM scaling test — 16 to 1024 neurons.

Shows how reservoir performance scales with network size.
Uses GPU acceleration automatically when available.
"""

import sys, os, time
sys.path.insert(0, '.')
import numpy as np
from nsram import NSRAMReservoir
from nsram.benchmarks import xor_accuracy, memory_capacity, narma_prediction, waveform_classification

rng = np.random.RandomState(42)
T = 4000; wo = 600
inputs = rng.uniform(-1, 1, T).astype(np.float64)

print("="*65)
print("  NS-RAM Scaling Test")
print("="*65)

N_values = [16, 32, 64, 128, 256, 512, 1024]

print(f"\n  {'N':>6s}  {'XOR-1':>7s}  {'MC':>7s}  {'NAR10':>7s}  {'W4':>7s}  {'Time':>6s}")
print(f"  {'-'*48}")

for N in N_values:
    t0 = time.time()
    res = NSRAMReservoir(N=N, noise_sigma=0.01, seed=42)
    out = res.run(inputs)
    S = out['states']
    elapsed = time.time() - t0

    xor1 = xor_accuracy(S, inputs, wo, 1)
    mc = memory_capacity(S, inputs, wo)
    nar = narma_prediction(S, inputs, wo, 10)
    w4 = waveform_classification(S, inputs, wo)

    print(f"  {N:>6d}  {xor1:>6.1%}  {mc:>7.3f}  {nar:>7.3f}  {w4:>6.1%}  {elapsed:>5.1f}s")
