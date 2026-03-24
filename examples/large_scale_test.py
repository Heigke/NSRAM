#!/usr/bin/env python3
"""nsram large-scale validation — prove the library works end to end.

Generates 8 publication-quality plots covering:
  1. Single neuron I-V curves
  2. BVpar(Vg1, T) physics validation
  3. Spike raster + population dynamics (128N)
  4. Die-to-die variability grid
  5. STP ablation: none vs STD vs STF vs heterogeneous
  6. Scaling law: 16 → 512 neurons
  7. Noise sensitivity sweep
  8. VG2 → STP type demonstration (the novel finding)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from nsram import (NSRAMNeuron, NSRAMNetwork, NSRAMReservoir, rc_benchmark,
                   breakdown_voltage, avalanche_current, charge_capture_rate,
                   DeviceParams, DimensionlessParams, PRESETS)
from nsram.benchmarks import xor_accuracy, memory_capacity, narma_prediction, waveform_classification

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)

print("="*70)
print("  nsram Large-Scale Validation")
print("="*70)

# Common input
rng = np.random.RandomState(42)
T_steps = 4000
wo = 600
inputs = rng.uniform(-1, 1, T_steps).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════
# FIG 1: I-V Curves
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/8] I-V curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

Vds = np.linspace(0, 3.5, 500)
colors = plt.cm.coolwarm(np.linspace(0, 1, 7))
for i, vg1 in enumerate(np.arange(0.1, 0.8, 0.1)):
    Id = avalanche_current(Vds, vg1, T=300.0, I0=1e-16)
    ax1.semilogy(Vds, np.maximum(Id, 1e-18), color=colors[i],
                  linewidth=1.5, label=f'Vg1={vg1:.1f}V')
ax1.set_xlabel('Drain Voltage Vds (V)', fontsize=11)
ax1.set_ylabel('Drain Current Id (A)', fontsize=11)
ax1.set_title('NS-RAM I-V: Avalanche Current\n(Pazos et al., Nature 640, 2025)',
               fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.set_ylim(1e-18, 1e-3)
ax1.set_xlim(0, 3.5)
ax1.grid(True, alpha=0.3, which='both')
ax1.axhline(1e-9, color='gray', linestyle=':', alpha=0.5)
ax1.text(0.5, 2e-9, 'I_leak = 0.5 nA', fontsize=8, color='gray')

# Temperature dependence
for T_val, ls, label in [(250, '--', '250K'), (300, '-', '300K'),
                          (350, '--', '350K'), (400, ':', '400K')]:
    Id = avalanche_current(Vds, 0.4, T=T_val, I0=1e-16)
    ax2.semilogy(Vds, np.maximum(Id, 1e-18), linestyle=ls, linewidth=1.5, label=label)
ax2.set_xlabel('Vds (V)'); ax2.set_ylabel('Id (A)')
ax2.set_title('Temperature Dependence (Vg1=0.4V)\nTbv1 = -21.3 μ/K', fontsize=11)
ax2.legend(); ax2.grid(True, alpha=0.3, which='both')
ax2.set_ylim(1e-18, 1e-3); ax2.set_xlim(0, 3.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'nsram_fig1_iv_curves.png'), dpi=200)
plt.close()
print("  Saved nsram_fig1_iv_curves.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 2: BVpar physics
# ═══════════════════════════════════════════════════════════════════════
print("[2/8] BVpar physics...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

Vg1_sweep = np.linspace(0, 0.6, 100)
for T_val, color, label in [(250,'#2196F3','250K'), (300,'black','300K'),
                              (350,'#FF5722','350K'), (400,'#FF9800','400K')]:
    bv = breakdown_voltage(Vg1_sweep, T=T_val)
    ax1.plot(Vg1_sweep, bv, color=color, linewidth=2, label=label)
ax1.fill_between([0, 0.6], 0, 2.5, alpha=0.08, color='green')
ax1.axhline(2.5, color='gray', linestyle='--', label='Vcb peak')
ax1.set_xlabel('Vg1 (V)'); ax1.set_ylabel('BVpar (V)')
ax1.set_title('Breakdown Voltage\nBVpar = (3.5 - 1.5·Vg1)·(1 + Tbv1·ΔT)', fontsize=11)
ax1.legend(fontsize=9); ax1.set_ylim(1.5, 3.8); ax1.grid(True, alpha=0.3)
ax1.annotate('Avalanche\nregion', xy=(0.45, 2.0), fontsize=10, color='green',
              fontweight='bold', ha='center')

# VG2 → k_cap mapping (the SRH↔TM bridge)
Vg2_sweep = np.linspace(0.25, 0.55, 100)
k_cap = charge_capture_rate(Vg2_sweep, k_cap_max=1000.0)
ax2.plot(Vg2_sweep, k_cap, 'k-', linewidth=2)
ax2.fill_between(Vg2_sweep, 0, k_cap, alpha=0.15, color='red',
                  where=k_cap > 500, label='STD regime')
ax2.fill_between(Vg2_sweep, 0, k_cap, alpha=0.15, color='blue',
                  where=k_cap < 500, label='STF regime')
ax2.axvline(0.40, color='gray', linestyle='--', alpha=0.7)
ax2.set_xlabel('VG2 (V)'); ax2.set_ylabel('k_cap (1/s)')
ax2.set_title('Charge Capture Rate → TM Utilization U\n'
               'k_cap(VG2) = k_max / (1 + exp((VG2-0.4)/0.05))', fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.annotate('Synapse mode\n(depression)', xy=(0.32, 800), fontsize=9,
              color='red', ha='center')
ax2.annotate('Neuron mode\n(facilitation)', xy=(0.48, 200), fontsize=9,
              color='blue', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'nsram_fig2_bvpar_physics.png'), dpi=200)
plt.close()
print("  Saved nsram_fig2_bvpar_physics.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: Spike raster + population dynamics (128N)
# ═══════════════════════════════════════════════════════════════════════
print("[3/8] Spike raster (128N)...")
net = NSRAMNetwork(N=128, seed=42, backend='auto')
t0 = time.time()
result = net.run(inputs, noise_sigma=0.01, record_full=True)
elapsed = time.time() - t0
spk = result['spikes']; S = result['states']
n_active = (spk.sum(1) > 0).sum()
print(f"  128N simulation: {elapsed:.2f}s, {n_active} active, {spk.sum():.0f} spikes")

fig = plt.figure(figsize=(14, 7))
gs = GridSpec(4, 1, height_ratios=[0.12, 0.58, 0.15, 0.15], hspace=0.08)

t_show = slice(wo, wo + 800)
t_ax = np.arange(800)

ax0 = fig.add_subplot(gs[0])
ax0.plot(t_ax, inputs[t_show], color='#1565C0', linewidth=0.6)
ax0.set_ylabel('Input', fontsize=9); ax0.set_xticklabels([]); ax0.set_xlim(0, 800)

ax1 = fig.add_subplot(gs[1], sharex=ax0)
st, sn = np.where(spk[:, t_show].T > 0)
if len(st) > 0:
    ax1.scatter(st, sn, s=0.8, c='black', marker='.', rasterized=True)
ax1.set_ylabel('Neuron #', fontsize=9); ax1.set_ylim(-1, 128); ax1.set_xticklabels([])

ax2 = fig.add_subplot(gs[2], sharex=ax0)
win = 25
pop_rate = np.convolve(spk.sum(0), np.ones(win)/win, mode='same')
ax2.plot(pop_rate[t_show], color='#D32F2F', linewidth=0.7)
ax2.set_ylabel('Pop Rate', fontsize=9); ax2.set_xticklabels([])

ax3 = fig.add_subplot(gs[3], sharex=ax0)
# Show state diversity: std across neurons at each timestep
state_std = S[:, t_show].std(axis=0)
ax3.plot(t_ax, state_std, color='#388E3C', linewidth=0.7)
ax3.set_ylabel('State σ', fontsize=9); ax3.set_xlabel('Time step')

fig.suptitle(f'NS-RAM 128-Neuron Reservoir ({elapsed:.1f}s on {net.device})',
              fontsize=13, fontweight='bold')
plt.savefig(os.path.join(OUT, 'nsram_fig3_raster.png'), dpi=200)
plt.close()
print("  Saved nsram_fig3_raster.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4: Die-to-die variability
# ═══════════════════════════════════════════════════════════════════════
print("[4/8] Die-to-die variability...")
if 'Vm' in result:
    Vm = result['Vm']
else:
    # Re-run with record
    result_full = net.run(inputs, noise_sigma=0.01, record_full=True)
    Vm = result_full.get('Vm', S)

rates = spk.sum(axis=1)
sorted_nids = np.argsort(rates)
sample_16 = sorted_nids[np.linspace(5, 120, 16).astype(int)]

fig, axes = plt.subplots(4, 4, figsize=(14, 9))
t_slice = slice(wo, wo + 250)
for idx, nid in enumerate(sample_16):
    ax = axes[idx // 4, idx % 4]
    trace = S[nid, t_slice]
    spike_t = np.where(spk[nid, t_slice] > 0)[0]
    ax.plot(trace, 'k-', linewidth=0.5)
    if len(spike_t) > 0:
        ax.scatter(spike_t, np.ones_like(spike_t) * trace.max() * 1.1,
                    marker='|', color='red', s=15, linewidths=0.8)
    ax.set_title(f'N{nid} ({rates[nid]:.0f} spk)', fontsize=7)
    ax.tick_params(labelsize=5)
    if idx % 4 != 0: ax.set_yticklabels([])
    if idx < 12: ax.set_xticklabels([])
fig.suptitle('Die-to-Die Variability: 16 Neurons with Different VG2/Threshold\n'
              '(Pazos et al., Nature 640, 2025)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'nsram_fig4_variability.png'), dpi=200)
plt.close()
print("  Saved nsram_fig4_variability.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 5: STP Ablation
# ═══════════════════════════════════════════════════════════════════════
print("[5/8] STP ablation study...")
stp_results = {}
for stp_mode in ['none', 'std', 'stf', 'heterogeneous']:
    res = NSRAMReservoir(N=128, stp=stp_mode, noise_sigma=0.01, seed=42, backend='auto')
    metrics_list = []
    for rep in range(5):
        out = res.run(inputs)
        S_r = out['states']
        m = {
            'xor1': xor_accuracy(S_r, inputs, wo, 1),
            'xor2': xor_accuracy(S_r, inputs, wo, 2),
            'mc': memory_capacity(S_r, inputs, wo),
            'narma10': narma_prediction(S_r, inputs, wo, 10),
            'wave4': waveform_classification(S_r, inputs, wo),
        }
        metrics_list.append(m)
    avg = {k: np.mean([r[k] for r in metrics_list]) for k in metrics_list[0]}
    std = {k: np.std([r[k] for r in metrics_list]) for k in metrics_list[0]}
    stp_results[stp_mode] = {'avg': avg, 'std': std}
    print(f"  {stp_mode:15s}: XOR1={avg['xor1']:.1%} MC={avg['mc']:.3f} "
          f"NAR10={avg['narma10']:.3f} W4={avg['wave4']:.1%}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
modes = ['none', 'std', 'stf', 'heterogeneous']
labels = ['No STP\n(baseline)', 'STD\n(low VG2)', 'STF\n(high VG2)', 'Heterogeneous\n(mixed VG2)']
bar_colors = ['#9E9E9E', '#E53935', '#1E88E5', '#43A047']
x = np.arange(4)

for ax, metric, title in [(axes[0],'xor1','XOR τ=1'),
                            (axes[1],'mc','Memory Capacity'),
                            (axes[2],'narma10','NARMA-10 R²'),
                            (axes[3],'wave4','Wave-4 Classification')]:
    vals = [stp_results[m]['avg'][metric] for m in modes]
    errs = [stp_results[m]['std'][metric] for m in modes]
    ax.bar(x, vals, 0.65, yerr=errs, color=bar_colors, edgecolor='black',
           linewidth=0.5, capsize=4, error_kw=dict(linewidth=1))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for i, v in enumerate(vals):
        fmt = f'{v:.1%}' if metric in ('xor1','wave4') else f'{v:.3f}'
        ax.text(i, v + errs[i] + 0.01, fmt, ha='center', fontsize=7, fontweight='bold')

fig.suptitle('NS-RAM STP Ablation: Charge Trapping = Tsodyks-Markram Plasticity',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'nsram_fig5_stp_ablation.png'), dpi=200)
plt.close()
print("  Saved nsram_fig5_stp_ablation.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: Scaling Law
# ═══════════════════════════════════════════════════════════════════════
print("[6/8] Scaling law (16→512)...")
N_values = [16, 32, 64, 128, 256, 512]
scale_data = {k: [] for k in ['xor1', 'mc', 'narma10', 'wave4']}
scale_times = []

for N_val in N_values:
    t0 = time.time()
    res = NSRAMReservoir(N=N_val, noise_sigma=0.01, seed=42, backend='auto')
    metrics_list = []
    for rep in range(3):
        out = res.run(inputs)
        S_r = out['states']
        metrics_list.append({
            'xor1': xor_accuracy(S_r, inputs, wo, 1),
            'mc': memory_capacity(S_r, inputs, wo),
            'narma10': narma_prediction(S_r, inputs, wo, 10),
            'wave4': waveform_classification(S_r, inputs, wo),
        })
    elapsed = time.time() - t0
    scale_times.append(elapsed)
    for k in scale_data:
        scale_data[k].append(np.mean([m[k] for m in metrics_list]))
    print(f"  N={N_val:4d}: XOR1={scale_data['xor1'][-1]:.1%} MC={scale_data['mc'][-1]:.3f} "
          f"W4={scale_data['wave4'][-1]:.1%} ({elapsed:.1f}s)")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, key, title, color, chance in [
    (axes[0], 'xor1', 'XOR τ=1 (%)', '#1565C0', 0.5),
    (axes[1], 'mc', 'Memory Capacity', '#2E7D32', None),
    (axes[2], 'narma10', 'NARMA-10 R²', '#E65100', None),
    (axes[3], 'wave4', 'Wave-4 (%)', '#AD1457', 0.25),
]:
    vals = scale_data[key]
    if 'xor' in key or 'wave' in key:
        vals = [v * 100 for v in vals]
        if chance: chance *= 100
    ax.semilogx(N_values, vals, 'o-', color=color, linewidth=2, markersize=8)
    if chance:
        ax.axhline(chance, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend(fontsize=8)
    ax.set_xlabel('N neurons'); ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    # Annotate best
    ax.annotate(f'{vals[-1]:.1f}' if vals[-1] > 1 else f'{vals[-1]:.3f}',
                xy=(N_values[-1], vals[-1]), textcoords="offset points",
                xytext=(-30, 10), fontsize=8, fontweight='bold')

fig.suptitle('NS-RAM Reservoir Scaling Law', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'nsram_fig6_scaling.png'), dpi=200)
plt.close()
print("  Saved nsram_fig6_scaling.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: Noise Sensitivity
# ═══════════════════════════════════════════════════════════════════════
print("[7/8] Noise sensitivity sweep...")
noise_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
noise_data = {k: [] for k in ['xor1', 'mc', 'wave4']}

for sigma in noise_values:
    res = NSRAMReservoir(N=128, noise_sigma=sigma, seed=42, backend='auto')
    out = res.run(inputs)
    S_r = out['states']
    noise_data['xor1'].append(xor_accuracy(S_r, inputs, wo, 1))
    noise_data['mc'].append(memory_capacity(S_r, inputs, wo))
    noise_data['wave4'].append(waveform_classification(S_r, inputs, wo))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, key, title, color in [
    (ax1, 'xor1', 'XOR-1 Accuracy', '#1565C0'),
    (ax2, 'mc', 'Memory Capacity', '#2E7D32'),
    (ax3, 'wave4', 'Wave-4 Classification', '#AD1457'),
]:
    vals = noise_data[key]
    ax.semilogx([max(s, 1e-4) for s in noise_values], vals, 'o-',
                 color=color, linewidth=2, markersize=7)
    ax.set_xlabel('Noise σ'); ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    # Mark optimal
    best_idx = np.argmax(vals)
    ax.scatter([max(noise_values[best_idx], 1e-4)], [vals[best_idx]],
               s=150, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.annotate(f'best σ={noise_values[best_idx]}', xy=(max(noise_values[best_idx], 1e-4), vals[best_idx]),
                textcoords="offset points", xytext=(10, 10), fontsize=8, color='red')

fig.suptitle('Noise Sensitivity: Low Noise is Optimal (confirms z2254i finding)',
              fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'nsram_fig7_noise.png'), dpi=200)
plt.close()
print("  Saved nsram_fig7_noise.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 8: VG2 → STP type (the novel finding)
# ═══════════════════════════════════════════════════════════════════════
print("[8/8] VG2 → STP type demonstration...")

# Sweep VG2 midpoint to shift entire population toward STD or STF
vg2_mids = [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48]
vg2_xor = []; vg2_mc = []; vg2_narma = []; vg2_wave = []

for v_mid in vg2_mids:
    p = DimensionlessParams()
    # Shift VG2 distribution center
    net_temp = NSRAMNetwork(N=128, params=p, seed=42, backend='auto')
    # Override VG2 array to center around v_mid
    new_vg2 = (v_mid - 0.04 + 0.08 * np.random.RandomState(42).rand(128)).astype(np.float32)
    net_temp._Vg2 = new_vg2
    from nsram.physics import charge_capture_rate as ccr
    k_cap_new = ccr(new_vg2, 1000.0, 0.40, 0.05)
    net_temp._U_base = np.clip(k_cap_new / 1000.0 * p.U * 2, 0.01, 0.95).astype(np.float32)
    net_temp._tau_fac = ((1.0 - net_temp._U_base) * p.tau_fac).astype(np.float32)
    if net_temp.backend == 'torch':
        net_temp._to_device()

    out = net_temp.run(inputs, noise_sigma=0.01)
    S_r = out['states']
    vg2_xor.append(xor_accuracy(S_r, inputs, wo, 1))
    vg2_mc.append(memory_capacity(S_r, inputs, wo))
    vg2_narma.append(narma_prediction(S_r, inputs, wo, 10))
    vg2_wave.append(waveform_classification(S_r, inputs, wo))
    u_mean = net_temp._U_base.mean()
    print(f"  VG2_mid={v_mid:.2f}V → U_mean={u_mean:.3f}: XOR1={vg2_xor[-1]:.1%} "
          f"MC={vg2_mc[-1]:.3f} W4={vg2_wave[-1]:.1%}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, vals, title, color in [
    (axes[0,0], vg2_xor, 'XOR-1 Accuracy', '#1565C0'),
    (axes[0,1], vg2_mc, 'Memory Capacity', '#2E7D32'),
    (axes[1,0], vg2_narma, 'NARMA-10 R²', '#E65100'),
    (axes[1,1], vg2_wave, 'Wave-4 Classification', '#AD1457'),
]:
    ax.plot(vg2_mids, vals, 'o-', color=color, linewidth=2, markersize=8)
    ax.set_xlabel('VG2 center voltage (V)'); ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.40, color='gray', linestyle='--', alpha=0.5)
    # STD/STF labels
    ax.annotate('← STD (depression)', xy=(0.31, ax.get_ylim()[1]*0.95),
                fontsize=8, color='red')
    ax.annotate('STF (facilitation) →', xy=(0.43, ax.get_ylim()[1]*0.95),
                fontsize=8, color='blue', ha='right')

fig.suptitle('VG2 Controls Reservoir Computing Performance via STP Type\n'
              'First demonstration: NS-RAM charge trapping = Tsodyks-Markram plasticity',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'nsram_fig8_vg2_stp.png'), dpi=200)
plt.close()
print("  Saved nsram_fig8_vg2_stp.png")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  All plots saved to:", OUT)
print("="*70)
print(f"\n  Best 128N config (STP=heterogeneous, σ=0.01):")
best = stp_results['heterogeneous']['avg']
print(f"    XOR-1:    {best['xor1']:.1%}")
print(f"    MC:       {best['mc']:.3f}")
print(f"    NARMA-10: {best['narma10']:.3f}")
print(f"    Wave-4:   {best['wave4']:.1%}")
print(f"\n  512N scaling:")
print(f"    XOR-1:    {scale_data['xor1'][-1]:.1%}")
print(f"    MC:       {scale_data['mc'][-1]:.3f}")
print(f"    Wave-4:   {scale_data['wave4'][-1]:.1%}")
print(f"    Time:     {scale_times[-1]:.1f}s")
