"""Find the optimal NS-RAM reservoir configuration."""

import sys; sys.path.insert(0, '.')
import numpy as np
from nsram.neuron import NSRAMParams
from nsram.network import NSRAMNetwork
from nsram.benchmarks import rc_benchmark, xor_accuracy, memory_capacity, narma_prediction, waveform_classification

rng = np.random.RandomState(42)
inputs = rng.uniform(-1, 1, 3000).astype(np.float64)
wo = 500

# Direct network run (bypass reservoir wrapper to debug)
params = NSRAMParams(N=128, seed=42, variability=0.10, bg_frac=0.95)
net = NSRAMNetwork(N=128, params=params, connectivity='sparse',
                    spectral_radius=0.90, backend='auto', seed=42)

print("Direct network test:")
out = net.run(inputs, noise_sigma=0.01, syn_scale=0.30)
S = out['states']; spk = out['spikes']
active = (spk.sum(1) > 0).sum()
total = spk.sum()
print(f"  Active: {active}/128, Spikes: {total:.0f}")
print(f"  XOR1: {xor_accuracy(S, inputs, wo, 1):.1%}")
print(f"  MC:   {memory_capacity(S, inputs, wo):.3f}")
print(f"  NAR10:{narma_prediction(S, inputs, wo, 10):.3f}")
print(f"  Wave4:{waveform_classification(S, inputs, wo):.1%}")

# Check state range
print(f"  State range: [{S.min():.2f}, {S.max():.2f}], mean={S.mean():.3f}, std={S.std():.3f}")
print(f"  Spike rate: {total / (128 * 3000):.3f} spk/neuron/step")
