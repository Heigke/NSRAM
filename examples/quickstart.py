"""NS-RAM Quickstart — reservoir computing in 5 lines."""

import sys; sys.path.insert(0, '.')
from nsram import NSRAMReservoir, rc_benchmark

res = NSRAMReservoir(N=128, stp='heterogeneous')
print(res)
results = rc_benchmark(res, n_steps=3000, n_reps=3)
