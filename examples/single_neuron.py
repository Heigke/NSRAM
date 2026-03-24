"""Single NS-RAM neuron simulation — device-level physics.

Demonstrates I-V curves and time-domain spiking at different Vg1/Vg2.
"""

import sys; sys.path.insert(0, '.')
import numpy as np
from nsram import NSRAMNeuron, breakdown_voltage

print("="*60)
print("  NS-RAM Single Neuron Characterization")
print("="*60)

# I-V curves at different gate voltages
print("\n━━━ I-V Curves ━━━")
for vg1 in [0.2, 0.3, 0.4, 0.5]:
    n = NSRAMNeuron(Vg1=vg1)
    iv = n.iv_curve()
    peak_I = iv['Id'].max()
    bvpar = float(breakdown_voltage(vg1))
    print(f"  Vg1={vg1:.1f}V: BVpar={bvpar:.2f}V, peak Id={peak_I:.2e}A")

# Time-domain simulation
print("\n━━━ Spiking Behavior ━━━")
for vg1, vg2 in [(0.30, 0.45), (0.35, 0.40), (0.40, 0.35), (0.45, 0.30)]:
    n = NSRAMNeuron(Vg1=vg1, Vg2=vg2)
    trace = n.simulate(duration=500e-6, dt=50e-9, I_ext=5e-9, noise_sigma=1e-10)
    print(f"  Vg1={vg1:.2f}V Vg2={vg2:.2f}V: {trace['n_spikes']} spikes in 500μs "
          f"({trace['n_spikes']/500e-6:.0f} Hz), E={trace['energy_J']*1e15:.1f} fJ")

# Temperature sweep
print("\n━━━ Temperature Dependence (Tbv1 = -21.3 μ/K) ━━━")
for T in [250, 275, 300, 325, 350]:
    n = NSRAMNeuron(Vg1=0.40, temperature=float(T))
    trace = n.simulate(duration=200e-6, dt=50e-9, I_ext=5e-9)
    bv = float(breakdown_voltage(0.40, T=float(T)))
    print(f"  T={T}K: BVpar={bv:.3f}V, {trace['n_spikes']} spikes")
