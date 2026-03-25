#!/usr/bin/env python3
"""Reproduce key figures from Pazos et al. Nature 640 (2025) using nsram.

Generates model predictions for:
  Fig 2: I-V family at multiple Vg1 (avalanche onset shift)
  Fig 3: Firing frequency map f(Vg1, Vds) + energy per spike
  Fig 4: LTP/LTD potentiation-depression cycles
  Fig 5: Retention at multiple temperatures + endurance

Also runs novel experiments:
  - Crossbar array yield prediction (8×8, 32×32, 128×128)
  - Mode-switching reservoir (unique NS-RAM property)
  - Optimal operating point search in Vg1-Vg2-Rb space

These plots can be sent directly to Sebastian for validation.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from nsram.physics import (
    DeviceParams, breakdown_voltage, avalanche_current, thermal_voltage,
    charge_capture_rate, srh_trapping_ode,
)
from nsram.characterize import (
    simulate_pulse_response, simulate_ltp_ltd, simulate_retention,
    simulate_endurance, simulate_voltage_ramp, deep_nwell_iv,
    energy_per_spike,
)
from nsram.fitting import monte_carlo, technology_comparison
from nsram.neuron import NSRAMNeuron

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)


def fig2_iv_family():
    """Fig 2: I-V curves at multiple Vg1 showing BVpar shift."""
    print("  Fig 2: I-V family...")
    p = DeviceParams()
    Vds = np.linspace(0, 4.0, 500)
    Vg1_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(Vg1_values)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM I-V Family — Model Prediction (cf. Nature Fig 2)',
                 fontsize=13, fontweight='bold', color='white')

    # Linear scale
    ax1.set_facecolor('#0d1117')
    for i, vg1 in enumerate(Vg1_values):
        Id = avalanche_current(Vds, vg1, 300, p.Is)
        ax1.plot(Vds, Id * 1e6, color=colors[i], linewidth=1.5,
                 label=f'Vg1={vg1:.1f}V')
        bvpar = float(breakdown_voltage(vg1, 300))
        ax1.axvline(bvpar, color=colors[i], linestyle=':', alpha=0.3)
    ax1.set_xlabel('Vds (V)', color='white')
    ax1.set_ylabel('Id (µA)', color='white')
    ax1.set_title('(a) Linear scale', color='white')
    ax1.legend(fontsize=6, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.15)
    ax1.set_ylim(0, 50)

    # Log scale
    ax2.set_facecolor('#0d1117')
    for i, vg1 in enumerate(Vg1_values):
        Id = avalanche_current(Vds, vg1, 300, p.Is)
        ax2.semilogy(Vds, Id, color=colors[i], linewidth=1.5)
    ax2.set_xlabel('Vds (V)', color='white')
    ax2.set_ylabel('Id (A)', color='white')
    ax2.set_title('(b) Log scale — 8 decades', color='white')
    ax2.set_ylim(1e-16, 1e-4)
    ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.15)

    # BVpar vs Vg1 inset
    ax_inset = ax2.inset_axes([0.05, 0.55, 0.35, 0.4])
    ax_inset.set_facecolor('#1a1a2e')
    bvpars = [float(breakdown_voltage(vg1, 300)) for vg1 in Vg1_values]
    ax_inset.plot(Vg1_values, bvpars, 'o-', color='#4ecdc4', markersize=4)
    ax_inset.set_xlabel('Vg1 (V)', fontsize=7, color='gray')
    ax_inset.set_ylabel('BVpar (V)', fontsize=7, color='gray')
    ax_inset.set_title('BVpar = 3.5 - 1.5×Vg1', fontsize=7, color='#4ecdc4')
    ax_inset.tick_params(colors='gray', labelsize=5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_fig2_iv_family.png'), dpi=150, facecolor='#0d1117')
    plt.close()


def fig3_firing_map():
    """Fig 3: Firing frequency and energy maps."""
    print("  Fig 3: Firing frequency map...")

    # Simplified: compute BVpar threshold crossings
    Vg1_arr = np.linspace(0.05, 0.7, 30)
    Vds_arr = np.linspace(1.0, 4.0, 30)
    freq_map = np.zeros((30, 30))
    energy_map = np.zeros((30, 30))

    p = DeviceParams()
    for i, vg1 in enumerate(Vg1_arr):
        bvpar = float(breakdown_voltage(vg1, 300))
        for j, vds in enumerate(Vds_arr):
            if vds > bvpar:
                # Estimate frequency from excess voltage
                excess = vds - bvpar
                # f ~ excess / (Rb × Cb) — higher excess → faster charging → higher freq
                freq = excess / (p.Rb_neuron * p.Cb) * 0.1
                freq = np.clip(freq, 0, 500e3)
                freq_map[i, j] = freq
                if freq > 0:
                    energy_map[i, j] = p.E_spike

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Firing Frequency & Energy Maps (cf. Nature Fig 3)',
                 fontsize=13, fontweight='bold', color='white')

    ax1.set_facecolor('#0d1117')
    im1 = ax1.imshow(freq_map / 1e3, extent=[Vds_arr[0], Vds_arr[-1], Vg1_arr[-1], Vg1_arr[0]],
                      aspect='auto', cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('Vds (V)', color='white')
    ax1.set_ylabel('Vg1 (V)', color='white')
    ax1.set_title('(a) Firing Frequency (kHz)', color='white')
    cb1 = plt.colorbar(im1, ax=ax1); cb1.ax.tick_params(colors='gray')
    ax1.tick_params(colors='gray')
    # BVpar boundary line
    bv_line = [float(breakdown_voltage(vg1, 300)) for vg1 in Vg1_arr]
    ax1.plot(bv_line, Vg1_arr, '--', color='#4ecdc4', linewidth=2, label='BVpar boundary')
    ax1.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    ax2.set_facecolor('#0d1117')
    im2 = ax2.imshow(energy_map * 1e15, extent=[Vds_arr[0], Vds_arr[-1], Vg1_arr[-1], Vg1_arr[0]],
                      aspect='auto', cmap='viridis', interpolation='bilinear')
    ax2.set_xlabel('Vds (V)', color='white')
    ax2.set_ylabel('Vg1 (V)', color='white')
    ax2.set_title('(b) Energy per Spike (fJ)', color='white')
    cb2 = plt.colorbar(im2, ax=ax2); cb2.ax.tick_params(colors='gray')
    ax2.tick_params(colors='gray')
    ax2.plot(bv_line, Vg1_arr, '--', color='#FF6B6B', linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_fig3_firing_map.png'), dpi=150, facecolor='#0d1117')
    plt.close()


def fig4_ltp_ltd():
    """Fig 4: LTP/LTD potentiation-depression cycles."""
    print("  Fig 4: LTP/LTD cycles...")
    result = simulate_ltp_ltd(n_pulses_ltp=100, n_pulses_ltd=100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Synaptic Plasticity — LTP/LTD (cf. Nature Fig 4)',
                 fontsize=13, fontweight='bold', color='white')

    # Conductance evolution
    ax1.set_facecolor('#0d1117')
    ax1.plot(result['pulse_number_ltp'], result['conductance_ltp'],
             'o-', color='#4CAF50', markersize=2, linewidth=1.5, label='LTP (potentiation)')
    ax1.plot(result['pulse_number_ltd'], result['conductance_ltd'],
             's-', color='#F44336', markersize=2, linewidth=1.5, label='LTD (depression)')
    ax1.set_xlabel('Pulse Number', color='white')
    ax1.set_ylabel('Conductance (a.u.)', color='white')
    ax1.set_title(f'(a) Conductance Evolution — {result["n_levels"]} levels', color='white')
    ax1.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.15)

    # Trapped charge
    ax2.set_facecolor('#0d1117')
    all_pulses = np.concatenate([result['pulse_number_ltp'], result['pulse_number_ltd']])
    all_Q = np.concatenate([result['Q_ltp'], result['Q_ltd']])
    ax2.plot(all_pulses, all_Q, '-', color='#2196F3', linewidth=2)
    ax2.axhline(0, color='gray', alpha=0.3)
    ax2.set_xlabel('Pulse Number', color='white')
    ax2.set_ylabel('Trapped Charge Q', color='white')
    ax2.set_title(f'(b) Charge Trapping — lin_LTP={result["linearity_ltp"]:.2f}, lin_LTD={result["linearity_ltd"]:.2f}',
                   color='white')
    ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.15)
    ax2.annotate(f'LTP: low Vg2\nhigh capture', xy=(25, all_Q[25]),
                  fontsize=8, color='#4CAF50')
    ax2.annotate(f'LTD: high Vg2\nhigh emission', xy=(150, all_Q[150] if len(all_Q) > 150 else 0),
                  fontsize=8, color='#F44336')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_fig4_ltp_ltd.png'), dpi=150, facecolor='#0d1117')
    plt.close()


def fig5_retention_endurance():
    """Fig 5: Retention at temperatures + endurance."""
    print("  Fig 5: Retention & endurance...")

    ret = simulate_retention(Q0=0.8, duration=1e5, temperatures=[300, 358, 398])
    end = simulate_endurance(n_cycles=1000000, degradation_rate=1e-7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Retention & Endurance (cf. Nature Fig 5)',
                 fontsize=13, fontweight='bold', color='white')

    # Retention
    ax1.set_facecolor('#0d1117')
    colors_t = {'300': '#4CAF50', '358': '#FF9800', '398': '#F44336'}
    labels_t = {300: '27°C', 358: '85°C', 398: '125°C'}
    for T in [300, 358, 398]:
        Q = ret['Q_vs_T'][T]
        tau = ret['tau_vs_T'][T]
        ax1.semilogx(ret['time'], Q, linewidth=2, color=colors_t[str(T)],
                      label=f'{labels_t[T]} (τ={tau:.0f}s)')
    ax1.axvline(1e4, color='gray', linestyle='--', alpha=0.5, label='>10⁴s target')
    ax1.set_xlabel('Time (s)', color='white')
    ax1.set_ylabel('Normalized Charge Q/Q₀', color='white')
    ax1.set_title(f'(a) Retention — Ea={ret["Ea_extracted"]:.2f} eV', color='white')
    ax1.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)
    ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.15)
    ax1.set_ylim(0, 1.0)

    # Endurance
    ax2.set_facecolor('#0d1117')
    ax2.semilogx(end['cycles'], end['BVpar'], '-', color='#2196F3', linewidth=2,
                  label='BVpar')
    ax2.axhline(end['BVpar'][0] * 0.8, color='#F44336', linestyle='--', alpha=0.5,
                 label='80% failure threshold')
    ax2.set_xlabel('Write Cycles', color='white')
    ax2.set_ylabel('BVpar (V)', color='white')
    ax2.set_title(f'(b) Endurance — failure at {end["failure_cycle"]:.0e} cycles', color='white')
    ax2.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)
    ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.15)

    # On/off ratio on secondary axis
    ax2b = ax2.twinx()
    ax2b.semilogx(end['cycles'], end['on_off_ratio'], '--', color='#FF9800',
                    linewidth=1.5, alpha=0.7, label='On/Off ratio')
    ax2b.set_ylabel('On/Off Ratio', color='#FF9800')
    ax2b.tick_params(colors='#FF9800')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_fig5_retention_endurance.png'), dpi=150, facecolor='#0d1117')
    plt.close()


def novel_crossbar_yield():
    """Novel: Crossbar array yield prediction at different scales."""
    print("  Novel: Crossbar array yield...")
    p = DeviceParams()

    sizes = [8, 16, 32, 64, 128, 256]
    sigmas = [0.03, 0.05, 0.10, 0.15]
    yields = np.zeros((len(sigmas), len(sizes)))

    for i, sigma in enumerate(sigmas):
        mc = monte_carlo(p, n_samples=10000, sigma_dict={'BV0': sigma, 'Vth0': sigma})
        single_yield = mc['yield_percent'] / 100.0
        for j, sz in enumerate(sizes):
            # Array yield = (single_cell_yield) ^ (N_cells)
            n_cells = sz * sz
            yields[i, j] = single_yield ** n_cells * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Crossbar Array Yield Prediction (Novel)',
                 fontsize=13, fontweight='bold', color='white')

    ax1.set_facecolor('#0d1117')
    colors_s = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    for i, sigma in enumerate(sigmas):
        ax1.semilogy(sizes, yields[i], 'o-', color=colors_s[i], linewidth=2,
                      markersize=6, label=f'σ = {sigma*100:.0f}%')
    ax1.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90% yield target')
    ax1.set_xlabel('Array Size (N×N)', color='white')
    ax1.set_ylabel('Array Yield (%)', color='white')
    ax1.set_title('(a) Yield vs Array Size', color='white')
    ax1.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)
    ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.15)
    ax1.set_ylim(0.01, 110)

    # Max array size for 90% yield
    ax2.set_facecolor('#0d1117')
    max_sizes = []
    sigma_range = np.linspace(0.01, 0.20, 50)
    for sigma in sigma_range:
        mc = monte_carlo(p, n_samples=5000, sigma_dict={'BV0': sigma, 'Vth0': sigma})
        sy = mc['yield_percent'] / 100.0
        if sy > 0:
            # N_max: sy^(N²) = 0.9 → N² = log(0.9)/log(sy)
            if sy < 1.0:
                n2 = np.log(0.9) / np.log(sy)
                max_sizes.append(int(np.sqrt(max(n2, 1))))
            else:
                max_sizes.append(1000)
        else:
            max_sizes.append(0)

    ax2.plot(sigma_range * 100, max_sizes, '-', color='#4ecdc4', linewidth=2.5)
    ax2.fill_between(sigma_range * 100, max_sizes, alpha=0.15, color='#4ecdc4')
    ax2.set_xlabel('Process Variability σ (%)', color='white')
    ax2.set_ylabel('Max Array Size (N×N) for 90% Yield', color='white')
    ax2.set_title('(b) Variability Budget', color='white')
    ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.15)
    ax2.axvline(5, color='#FF9800', linestyle='--', alpha=0.5, label='Typical CMOS (5%)')
    ax2.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_novel_crossbar_yield.png'), dpi=150, facecolor='#0d1117')
    plt.close()


def novel_mode_switching_reservoir():
    """Novel: Reservoir that exploits NS-RAM's unique mode switching.

    No other neuromorphic device can do this: neurons dynamically switch
    between neuron mode (fast spiking) and synapse mode (charge storage)
    based on Vg2. This creates a reservoir with built-in short-term memory.
    """
    print("  Novel: Mode-switching reservoir...")
    from nsram.network import NSRAMNetwork
    from nsram.benchmarks import memory_capacity, xor_accuracy, narma_prediction

    configs = {
        'Neuron only (Vg2=0)': {'U': 0.0, 'tau_rec': 1, 'tau_fac': 1},
        'Weak STP (U=0.01)': {'U': 0.01, 'tau_rec': 15, 'tau_fac': 10},
        'Medium STP (U=0.1)': {'U': 0.1, 'tau_rec': 15, 'tau_fac': 10},
        'Strong STP (U=0.5)': {'U': 0.5, 'tau_rec': 15, 'tau_fac': 10},
        'Heterogeneous': None,  # Uses default heterogeneous
    }

    rng = np.random.RandomState(42)
    inputs = rng.uniform(-1, 1, 5000).astype(np.float32)
    results = {}

    for name, stp_params in configs.items():
        from nsram.physics import DimensionlessParams
        if stp_params is not None:
            p = DimensionlessParams(**stp_params)
        else:
            p = DimensionlessParams()  # Default heterogeneous
        net = NSRAMNetwork(512, params=p, seed=42)
        r = net.run(inputs)
        states = r['states']

        mc = memory_capacity(states, inputs, washout=500, max_delay=15)
        xor1 = xor_accuracy(states, inputs, washout=500, tau=1)
        narma = narma_prediction(states, inputs, washout=500, order=10)

        results[name] = {'mc': mc, 'xor1': xor1, 'narma': narma}
        print(f"    {name:30s}: MC={mc:.3f} XOR={xor1:.1%} NARMA={narma:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Mode-Switching Reservoir — STP Impact (Novel)',
                 fontsize=13, fontweight='bold', color='white')

    names = list(results.keys())
    colors_r = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for idx, (metric, label) in enumerate([('mc', 'Memory Capacity'),
                                             ('xor1', 'XOR-1 Accuracy'),
                                             ('narma', 'NARMA-10 R²')]):
        ax = axes[idx]; ax.set_facecolor('#0d1117')
        vals = [results[n][metric] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors_r, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split('(')[0].strip() for n in names],
                            color='white', fontsize=7, rotation=25, ha='right')
        ax.set_title(label, color='white', fontsize=11)
        ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15, axis='y')
        # Highlight best
        best_idx = np.argmax(vals)
        bars[best_idx].set_edgecolor('white')
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_novel_mode_switching.png'), dpi=150, facecolor='#0d1117')
    plt.close()


def novel_operating_point_search():
    """Novel: Optimal Vg1 × spectral_radius × variability parameter sweep."""
    print("  Novel: Operating point search...")
    from nsram.network import NSRAMNetwork
    from nsram.physics import DimensionlessParams
    from nsram.benchmarks import memory_capacity, xor_accuracy

    rng = np.random.RandomState(42)
    inputs = rng.uniform(-1, 1, 3000).astype(np.float32)

    sr_values = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05]
    bg_values = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]

    mc_map = np.zeros((len(bg_values), len(sr_values)))
    xor_map = np.zeros((len(bg_values), len(sr_values)))

    for i, bg in enumerate(bg_values):
        for j, sr in enumerate(sr_values):
            p = DimensionlessParams(bg_frac=bg, spectral_radius=sr)
            net = NSRAMNetwork(256, params=p, seed=42)
            r = net.run(inputs)
            states = r['states']
            mc_map[i, j] = memory_capacity(states, inputs, washout=500)
            xor_map[i, j] = xor_accuracy(states, inputs, washout=500, tau=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Optimal Operating Point (Novel)',
                 fontsize=13, fontweight='bold', color='white')

    ax1.set_facecolor('#0d1117')
    im1 = ax1.imshow(mc_map, extent=[sr_values[0], sr_values[-1], bg_values[-1], bg_values[0]],
                      aspect='auto', cmap='viridis', interpolation='bilinear')
    ax1.set_xlabel('Spectral Radius', color='white')
    ax1.set_ylabel('Background Current (bg_frac)', color='white')
    ax1.set_title('(a) Memory Capacity', color='white')
    cb1 = plt.colorbar(im1, ax=ax1); cb1.ax.tick_params(colors='gray')
    ax1.tick_params(colors='gray')
    # Mark optimum
    best = np.unravel_index(mc_map.argmax(), mc_map.shape)
    ax1.plot(sr_values[best[1]], bg_values[best[0]], '*', color='white', markersize=15)
    ax1.annotate(f'MC={mc_map.max():.2f}', xy=(sr_values[best[1]], bg_values[best[0]]),
                  xytext=(10, 10), textcoords='offset points', color='white', fontsize=9,
                  fontweight='bold')

    ax2.set_facecolor('#0d1117')
    im2 = ax2.imshow(xor_map * 100, extent=[sr_values[0], sr_values[-1], bg_values[-1], bg_values[0]],
                      aspect='auto', cmap='magma', interpolation='bilinear')
    ax2.set_xlabel('Spectral Radius', color='white')
    ax2.set_ylabel('Background Current (bg_frac)', color='white')
    ax2.set_title('(b) XOR-1 Accuracy (%)', color='white')
    cb2 = plt.colorbar(im2, ax=ax2); cb2.ax.tick_params(colors='gray')
    ax2.tick_params(colors='gray')
    best2 = np.unravel_index(xor_map.argmax(), xor_map.shape)
    ax2.plot(sr_values[best2[1]], bg_values[best2[0]], '*', color='white', markersize=15)
    ax2.annotate(f'XOR={xor_map.max():.0%}', xy=(sr_values[best2[1]], bg_values[best2[0]]),
                  xytext=(10, 10), textcoords='offset points', color='white', fontsize=9,
                  fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_novel_operating_point.png'), dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"    Best MC: bg={bg_values[best[0]]}, sr={sr_values[best[1]]}, MC={mc_map.max():.3f}")
    print(f"    Best XOR: bg={bg_values[best2[0]]}, sr={sr_values[best2[1]]}, XOR={xor_map.max():.1%}")


def main():
    print("=" * 65)
    print("  NS-RAM Paper Figure Reproduction + Novel Experiments")
    print("=" * 65)

    t0 = time.time()

    # Reproduce paper figures
    fig2_iv_family()
    fig3_firing_map()
    fig4_ltp_ltd()
    fig5_retention_endurance()

    # Novel experiments
    novel_crossbar_yield()
    novel_mode_switching_reservoir()
    novel_operating_point_search()

    # Technology comparison
    print("  Technology comparison...")
    technology_comparison(save_path=os.path.join(OUT, 'nsram_tech_comparison.png'))

    elapsed = time.time() - t0
    print(f"\n  All plots saved to {OUT}/")
    print(f"  Total time: {elapsed:.0f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
