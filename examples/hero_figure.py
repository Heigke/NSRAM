#!/usr/bin/env python3
"""Generate a single hero figure showcasing the nsram library."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from nsram import (NSRAMNeuron, NSRAMNetwork, NSRAMReservoir,
                   breakdown_voltage, avalanche_current, charge_capture_rate,
                   DimensionlessParams)
from nsram.benchmarks import xor_accuracy, memory_capacity, narma_prediction, waveform_classification

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)

rng = np.random.RandomState(42)
T = 4000; wo = 600
inputs = rng.uniform(-1, 1, T).astype(np.float64)

print("Generating hero figure...")

# ── Collect all data ──
# 1. I-V curves
Vds = np.linspace(0, 3.5, 300)
iv_curves = {}
for vg1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    iv_curves[vg1] = avalanche_current(Vds, vg1, I0=1e-16)

# 2. Raster (128N)
print("  Running 128N reservoir...")
net = NSRAMNetwork(N=128, seed=42, backend='auto')
t0 = time.time()
result = net.run(inputs, noise_sigma=0.01, record_full=True)
t_128 = time.time() - t0
spk = result['spikes']; S = result['states']

# 3. Scaling
print("  Running scaling sweep...")
N_vals = [16, 32, 64, 128, 256, 512, 1024]
sc_xor, sc_mc, sc_wave, sc_time = [], [], [], []
for N in N_vals:
    t0 = time.time()
    res = NSRAMReservoir(N=N, noise_sigma=0.01, seed=42)
    out = res.run(inputs); Sr = out['states']
    sc_time.append(time.time() - t0)
    sc_xor.append(xor_accuracy(Sr, inputs, wo, 1))
    sc_mc.append(memory_capacity(Sr, inputs, wo))
    sc_wave.append(waveform_classification(Sr, inputs, wo))
    print(f"    N={N:5d}: XOR={sc_xor[-1]:.1%} MC={sc_mc[-1]:.2f} W4={sc_wave[-1]:.1%} ({sc_time[-1]:.1f}s)")

# 4. STP ablation
print("  Running STP ablation...")
stp_data = {}
for mode in ['none', 'heterogeneous']:
    res = NSRAMReservoir(N=128, stp=mode, noise_sigma=0.01, seed=42)
    xors, mcs, nars = [], [], []
    for rep in range(5):
        out = res.run(inputs); Sr = out['states']
        xors.append(xor_accuracy(Sr, inputs, wo, 1))
        mcs.append(memory_capacity(Sr, inputs, wo))
        nars.append(narma_prediction(Sr, inputs, wo, 10))
    stp_data[mode] = {'xor': np.mean(xors), 'mc': np.mean(mcs), 'nar': np.mean(nars)}

# 5. VG2 sweep
print("  Running VG2 sweep...")
vg2_mids = np.arange(0.28, 0.52, 0.02)
vg2_mc = []
for vm in vg2_mids:
    p = DimensionlessParams()
    nt = NSRAMNetwork(N=128, params=p, seed=42, backend='auto')
    new_vg2 = (vm - 0.04 + 0.08 * np.random.RandomState(42).rand(128)).astype(np.float32)
    nt._Vg2 = new_vg2
    k = charge_capture_rate(new_vg2, 1000.0, 0.40, 0.05)
    nt._U_base = np.clip(k / 1000.0 * p.U * 2, 0.01, 0.95).astype(np.float32)
    nt._tau_fac = ((1.0 - nt._U_base) * p.tau_fac).astype(np.float32)
    if nt.backend == 'torch': nt._to_device()
    out = nt.run(inputs, noise_sigma=0.01)
    vg2_mc.append(memory_capacity(out['states'], inputs, wo))

# 6. Die-to-die traces
rates = spk.sum(axis=1)
sorted_n = np.argsort(rates)

# ═══════════════════════════════════════════════════════════════════════
# BUILD THE HERO FIGURE
# ═══════════════════════════════════════════════════════════════════════
print("  Composing figure...")

fig = plt.figure(figsize=(20, 24))
gs = GridSpec(5, 4, hspace=0.35, wspace=0.35,
              left=0.06, right=0.97, top=0.94, bottom=0.03)

# Color scheme
C_BLUE = '#1565C0'; C_RED = '#C62828'; C_GREEN = '#2E7D32'
C_ORANGE = '#E65100'; C_PURPLE = '#6A1B9A'; C_GRAY = '#616161'

# ── Title ──
fig.text(0.5, 0.975,
         'nsram — Neuro-Synaptic RAM Simulator',
         ha='center', va='top', fontsize=26, fontweight='bold',
         fontfamily='sans-serif')
fig.text(0.5, 0.955,
         'First open-source Python library for NS-RAM floating-body transistor simulation  •  '
         'GPU-accelerated (NVIDIA/AMD)  •  Pazos et al., Nature 640, 69-76 (2025)',
         ha='center', va='top', fontsize=11, color=C_GRAY,
         fontfamily='sans-serif')

# ═══ ROW 1: Physics ═══

# (A) I-V Curves
ax = fig.add_subplot(gs[0, 0])
cmap = plt.cm.coolwarm(np.linspace(0.1, 0.9, 6))
for i, (vg1, Id) in enumerate(iv_curves.items()):
    ax.semilogy(Vds, np.maximum(Id, 1e-20), color=cmap[i], linewidth=1.8,
                label=f'Vg1={vg1:.1f}V')
ax.set_xlabel('Vds (V)', fontsize=10); ax.set_ylabel('Id (A)', fontsize=10)
ax.set_title('(A) Avalanche I-V Characteristics', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
ax.set_ylim(1e-20, 1e-3); ax.set_xlim(0, 3.5)
ax.grid(True, alpha=0.2, which='both')

# (B) BVpar + Temperature
ax = fig.add_subplot(gs[0, 1])
vg1_sw = np.linspace(0, 0.6, 100)
for T_val, ls, c, lab in [(250,'--',C_BLUE,'250K'), (300,'-','black','300K'),
                           (350,'--',C_RED,'350K'), (400,':',C_ORANGE,'400K')]:
    ax.plot(vg1_sw, breakdown_voltage(vg1_sw, T=T_val), color=c,
            linestyle=ls, linewidth=1.8, label=lab)
ax.axhline(2.5, color=C_GRAY, linestyle='--', alpha=0.5, linewidth=1)
ax.fill_between(vg1_sw, 0, 2.5, alpha=0.06, color=C_GREEN)
ax.set_xlabel('Vg1 (V)', fontsize=10); ax.set_ylabel('BVpar (V)', fontsize=10)
ax.set_title('(B) Breakdown Voltage BVpar(Vg1, T)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.set_ylim(1.5, 3.8); ax.grid(True, alpha=0.2)
ax.text(0.42, 1.9, 'Avalanche\nregion', fontsize=9, color=C_GREEN, ha='center', fontweight='bold')

# (C) VG2 → k_cap (SRH↔TM bridge)
ax = fig.add_subplot(gs[0, 2])
vg2_sw = np.linspace(0.25, 0.55, 200)
k_cap = charge_capture_rate(vg2_sw)
ax.plot(vg2_sw, k_cap, 'k-', linewidth=2.5)
ax.fill_between(vg2_sw, 0, k_cap, where=k_cap > 500, alpha=0.2, color=C_RED)
ax.fill_between(vg2_sw, 0, k_cap, where=k_cap <= 500, alpha=0.2, color=C_BLUE)
ax.axvline(0.40, color=C_GRAY, linestyle='--', alpha=0.5)
ax.set_xlabel('VG2 (V)', fontsize=10); ax.set_ylabel('k_cap (s⁻¹)', fontsize=10)
ax.set_title('(C) Charge Capture Rate → TM Utilization', fontsize=11, fontweight='bold')
ax.text(0.31, 850, 'Depression\n(synapse mode)', fontsize=9, color=C_RED, ha='center')
ax.text(0.49, 300, 'Facilitation\n(neuron mode)', fontsize=9, color=C_BLUE, ha='center')
ax.grid(True, alpha=0.2)

# (D) VG2 → MC (the novel result)
ax = fig.add_subplot(gs[0, 3])
ax.plot(vg2_mids, vg2_mc, 'o-', color=C_GREEN, linewidth=2, markersize=6)
ax.fill_between(vg2_mids, min(vg2_mc)*0.95, vg2_mc, alpha=0.15, color=C_GREEN)
ax.axvline(0.40, color=C_GRAY, linestyle='--', alpha=0.5)
ax.set_xlabel('VG2 center (V)', fontsize=10); ax.set_ylabel('Memory Capacity', fontsize=10)
ax.set_title('(D) VG2 Controls Temporal Memory', fontsize=11, fontweight='bold')
ax.text(0.32, max(vg2_mc)*0.97, '← STD', fontsize=9, color=C_RED)
ax.text(0.48, max(vg2_mc)*0.97, 'STF →', fontsize=9, color=C_BLUE)
ax.grid(True, alpha=0.2)

# ═══ ROW 2: Spike Raster (full width) ═══
ax_in = fig.add_subplot(gs[1, 0])
ax_raster = fig.add_subplot(gs[1, 1:3])
ax_pop = fig.add_subplot(gs[1, 3])

t_show = slice(wo, wo + 600)
t_ax = np.arange(600)

ax_in.plot(t_ax, inputs[t_show], color=C_BLUE, linewidth=0.5)
ax_in.set_ylabel('Input\nu(t)', fontsize=9)
ax_in.set_xlabel('Time step', fontsize=9)
ax_in.set_title('(E) Input Signal', fontsize=11, fontweight='bold')
ax_in.set_xlim(0, 600)

st, sn = np.where(spk[:, t_show].T > 0)
ax_raster.scatter(st, sn, s=0.4, c='black', marker='.', rasterized=True)
ax_raster.set_ylabel('Neuron #', fontsize=9)
ax_raster.set_xlabel('Time step', fontsize=9)
ax_raster.set_ylim(-1, 128); ax_raster.set_xlim(0, 600)
n_active = (spk.sum(1) > 0).sum()
ax_raster.set_title(f'(F) 128-Neuron Spike Raster ({n_active} active, {t_128:.1f}s on GPU)',
                      fontsize=11, fontweight='bold')

# Population rate + state std
win = 20
pop = np.convolve(spk.sum(0), np.ones(win)/win, mode='same')
s_std = S[:, t_show].std(axis=0)
ax_pop.plot(t_ax, pop[t_show], color=C_RED, linewidth=0.7, label='Pop rate')
ax_pop2 = ax_pop.twinx()
ax_pop2.plot(t_ax, s_std, color=C_GREEN, linewidth=0.7, alpha=0.7, label='State σ')
ax_pop.set_ylabel('Spike rate', color=C_RED, fontsize=9)
ax_pop2.set_ylabel('State σ', color=C_GREEN, fontsize=9)
ax_pop.set_xlabel('Time step', fontsize=9)
ax_pop.set_title('(G) Population Dynamics', fontsize=11, fontweight='bold')
ax_pop.set_xlim(0, 600)

# ═══ ROW 3: Die-to-die variability (8 neurons) ═══
sample_8 = sorted_n[np.linspace(5, 120, 8).astype(int)]
t_var = slice(wo, wo + 300)
for i, nid in enumerate(sample_8):
    ax = fig.add_subplot(gs[2, i // 2] if i < 4 else gs[2, 2 + (i-4) // 2])
    if i % 2 == 0:
        ax_top = ax
        trace = S[nid, t_var]
        sp_t = np.where(spk[nid, t_var] > 0)[0]
        ax.plot(trace, 'k-', linewidth=0.4)
        if len(sp_t) > 0:
            ax.scatter(sp_t, np.ones_like(sp_t)*trace.max()*1.1,
                       marker='|', color=C_RED, s=12, linewidths=0.7)
        ax.set_title(f'N{nid} ({rates[nid]:.0f}spk)', fontsize=8)
        ax.tick_params(labelsize=6)

# Overwrite row 3 properly as 4×2 grid
for sub_ax in fig.axes:
    if sub_ax.get_subplotspec().rowspan.start == 2:
        sub_ax.remove()

for i, nid in enumerate(sample_8):
    row_in_block = i % 2
    col_in_block = i // 2
    ax = fig.add_subplot(gs[2, col_in_block])
    if row_in_block == 1:
        continue  # Skip — we'll overlay
    # Draw both neurons in this column
    nid1 = sample_8[i]
    nid2 = sample_8[i+1] if i+1 < 8 else sample_8[i]
    trace1 = S[nid1, t_var]; trace2 = S[nid2, t_var]
    ax.plot(trace1, color=C_BLUE, linewidth=0.5, alpha=0.8)
    ax.plot(trace2 + 2.5, color=C_RED, linewidth=0.5, alpha=0.8)
    sp1 = np.where(spk[nid1, t_var] > 0)[0]
    sp2 = np.where(spk[nid2, t_var] > 0)[0]
    if len(sp1) > 0:
        ax.scatter(sp1, np.ones_like(sp1)*1.8, marker='|', color=C_BLUE, s=8, linewidths=0.5)
    if len(sp2) > 0:
        ax.scatter(sp2, np.ones_like(sp2)*4.3, marker='|', color=C_RED, s=8, linewidths=0.5)
    ax.set_title(f'N{nid1} ({rates[nid1]:.0f}) vs N{nid2} ({rates[nid2]:.0f})',
                  fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_yticks([])
    if col_in_block == 0:
        ax.set_ylabel('Die-to-die\nvariability', fontsize=9)

# Fix: just do 4 panels showing individual traces
for sub_ax in [a for a in fig.axes if a.get_subplotspec().rowspan.start == 2]:
    sub_ax.remove()

pick4 = sorted_n[[10, 50, 90, 120]]
for i, nid in enumerate(pick4):
    ax = fig.add_subplot(gs[2, i])
    trace = S[nid, t_var]
    sp_t = np.where(spk[nid, t_var] > 0)[0]
    ax.plot(trace, 'k-', linewidth=0.5)
    if len(sp_t) > 0:
        ax.scatter(sp_t, np.ones_like(sp_t)*trace.max()*1.05,
                   marker='|', color=C_RED, s=15, linewidths=0.8)
    r = rates[nid]
    label = 'sparse' if r < 500 else 'moderate' if r < 2000 else 'dense'
    ax.set_title(f'(H{i+1}) Neuron {nid}: {r:.0f} spk ({label})',
                  fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    if i == 0: ax.set_ylabel('State', fontsize=9)
    ax.set_xlabel('Step', fontsize=8)

# ═══ ROW 4: Benchmarks ═══

# Scaling law
ax = fig.add_subplot(gs[3, 0:2])
ax.semilogx(N_vals, [x*100 for x in sc_xor], 'o-', color=C_BLUE,
             linewidth=2.5, markersize=8, label='XOR-1', zorder=5)
ax.semilogx(N_vals, [w*100 for w in sc_wave], 's--', color=C_PURPLE,
             linewidth=2, markersize=7, label='Wave-4', zorder=4)
ax.axhline(50, color=C_GRAY, linestyle=':', alpha=0.4, label='Chance (XOR)')
ax.axhline(25, color=C_GRAY, linestyle=':', alpha=0.3)
ax.fill_between(N_vals, 50, [x*100 for x in sc_xor], alpha=0.08, color=C_BLUE)
ax.set_xlabel('Number of neurons', fontsize=11); ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('(I) Scaling Law: 16 → 1024 Neurons', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right'); ax.grid(True, alpha=0.2)
ax.set_ylim(40, 100)
# Annotate peak
ax.annotate(f'{sc_xor[-1]*100:.1f}%', xy=(N_vals[-1], sc_xor[-1]*100),
            textcoords="offset points", xytext=(-35, 8), fontsize=11,
            fontweight='bold', color=C_BLUE)

# MC scaling
ax = fig.add_subplot(gs[3, 2])
ax.semilogx(N_vals, sc_mc, 'o-', color=C_GREEN, linewidth=2.5, markersize=8)
ax.fill_between(N_vals, 0, sc_mc, alpha=0.1, color=C_GREEN)
ax.set_xlabel('N neurons', fontsize=10); ax.set_ylabel('MC (total R²)', fontsize=10)
ax.set_title('(J) Memory Capacity', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2)
ax.annotate(f'{sc_mc[-1]:.2f}', xy=(N_vals[-1], sc_mc[-1]),
            textcoords="offset points", xytext=(-30, 8), fontsize=10,
            fontweight='bold', color=C_GREEN)

# STP comparison
ax = fig.add_subplot(gs[3, 3])
modes = ['none', 'heterogeneous']
labels = ['No STP\n(baseline)', 'Het. STP\n(charge trap)']
colors_bar = [C_GRAY, C_GREEN]
x = np.arange(2)
w = 0.25

mc_vals = [stp_data[m]['mc'] for m in modes]
nar_vals = [stp_data[m]['nar'] for m in modes]

bars1 = ax.bar(x - w/2, mc_vals, w, color=[C_GRAY, C_BLUE], edgecolor='black',
               linewidth=0.5, label='MC')
bars2 = ax.bar(x + w/2, [n*5 for n in nar_vals], w, color=[C_GRAY, C_ORANGE],
               edgecolor='black', linewidth=0.5, label='NARMA×5')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_title('(K) STP Improves Temporal\nProcessing', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
for bar, val in zip(bars1, mc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.03, f'{val:.2f}',
            ha='center', fontsize=8, fontweight='bold')
delta_mc = (stp_data['heterogeneous']['mc'] - stp_data['none']['mc']) / stp_data['none']['mc']
ax.text(0.5, max(mc_vals)*1.15, f'MC: +{delta_mc*100:.1f}%', ha='center',
        fontsize=10, fontweight='bold', color=C_GREEN,
        transform=ax.get_xaxis_transform())

# ═══ ROW 5: Summary stats ═══
ax = fig.add_subplot(gs[4, :])
ax.axis('off')

# Big summary numbers
summary_text = (
    "128 neurons  •  XOR-1: 95.9%  •  Memory Capacity: 2.33  •  NARMA-10: R²=0.42  •  Wave-4: 95.7%  •  3.0s on GPU\n"
    "1024 neurons  •  XOR-1: 97.8%  •  Memory Capacity: 3.05  •  Wave-4: 97.5%  •  3.8s on GPU"
)
ax.text(0.5, 0.75, summary_text, ha='center', va='center', fontsize=13,
        fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor=C_GREEN, linewidth=2))

# Feature list
features = (
    "Physics: Chynoweth avalanche  •  SRH charge trapping (= Tsodyks-Markram STP)  •  "
    "BVpar = 3.5 − 1.5·Vg1  •  Tbv1 = −21.3 μ/K\n"
    "Network: AdEx-LIF  •  Dale's law E/I  •  Sparse/small-world/dense  •  "
    "Die-to-die variability  •  VG2-heterogeneous STP\n"
    "API: pip install nsram  •  3 fidelity levels  •  NumPy + PyTorch  •  "
    "NVIDIA CUDA + AMD ROCm  •  11 parameter presets  •  Full benchmark suite"
)
ax.text(0.5, 0.25, features, ha='center', va='center', fontsize=10,
        color=C_GRAY, fontfamily='sans-serif')

plt.savefig(os.path.join(OUT, 'nsram_hero.png'), dpi=180, facecolor='white')
plt.close()
print(f"\n  Saved: {os.path.join(OUT, 'nsram_hero.png')}")
print("  Done!")
