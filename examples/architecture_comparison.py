#!/usr/bin/env python3
"""NS-RAM Architecture Comparison — 5 neuron models, 10 benchmarks, 4 scales.

Compares NS-RAM AdEx-LIF against Izhikevich, Parametric LIF, Hodgkin-Huxley,
and software Echo State Network across reservoir computing benchmarks.

Generates publication-quality comparison plots.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch

from nsram.network import NSRAMNetwork
from nsram.neurons import IzhikevichNetwork, PLIFNetwork, HHNetwork
from nsram.benchmarks import (
    xor_accuracy, memory_capacity, narma_prediction,
    waveform_classification, mackey_glass, kernel_rank,
    nonlinear_memory_capacity,
)
from nsram.analysis import (
    firing_rate, isi_statistics, fano_factor, avalanche_analysis,
    effective_dimension, raster_plot, isi_histogram,
)
from nsram.encoding import rate_encode, population_encode

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)


class SoftwareESN:
    """Plain Echo State Network (tanh) as software baseline."""
    def __init__(self, N=128, n_inputs=1, sparsity=0.1, spectral_radius=0.9, seed=42):
        self.N = N
        rng = np.random.RandomState(seed)
        mask = (rng.rand(N, N) < sparsity).astype(np.float32)
        np.fill_diagonal(mask, 0)
        W = rng.randn(N, N).astype(np.float32) * mask
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W *= spectral_radius / eigs.max()
        self.W = W
        self.W_in = rng.randn(N, n_inputs).astype(np.float32) * 0.3
        self.leak = 0.3

    def run(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[:, None]
        T = signal.shape[0]
        x = np.zeros(self.N, dtype=np.float32)
        states = np.zeros((self.N, T), dtype=np.float32)
        spikes = np.zeros((self.N, T), dtype=np.float32)
        for t in range(T):
            x = (1 - self.leak) * x + self.leak * np.tanh(
                self.W @ x + self.W_in @ signal[t])
            states[:, t] = x
            spikes[:, t] = (x > 0.5).astype(np.float32)
        return {'states': states, 'spikes': spikes}


def benchmark_network(net, name, n_steps=3000, washout=500, seed=42):
    """Run all benchmarks on a network."""
    rng = np.random.RandomState(seed)
    inputs = rng.uniform(-1, 1, n_steps).astype(np.float32)

    t0 = time.time()
    result = net.run(inputs)
    elapsed = time.time() - t0

    states = result['states']
    spikes = result['spikes']

    metrics = {'name': name, 'time': elapsed, 'N': net.N}

    # RC Benchmarks
    metrics['xor1'] = xor_accuracy(states, inputs, washout, tau=1)
    metrics['xor5'] = xor_accuracy(states, inputs, washout, tau=5)
    metrics['mc'] = memory_capacity(states, inputs, washout, max_delay=15)
    metrics['narma5'] = narma_prediction(states, inputs, washout, order=5)
    metrics['narma10'] = narma_prediction(states, inputs, washout, order=10)
    metrics['wave4'] = waveform_classification(states, inputs, washout, n_classes=4)
    metrics['mg17'] = mackey_glass(states, washout, tau=17)
    metrics['krank'] = kernel_rank(states, washout)

    # Nonlinear MC
    nmc = nonlinear_memory_capacity(states, inputs, washout)
    metrics['nmc_total'] = nmc['total_nmc']
    metrics['nmc_linear'] = nmc['linear_mc']
    metrics['nmc_nonlinear'] = nmc['nonlinear_mc']

    # Spike statistics
    rates = firing_rate(spikes)
    metrics['mean_rate'] = float(rates.mean())
    metrics['active_frac'] = float((rates > 0).mean())

    isi = isi_statistics(spikes)
    metrics['mean_cv'] = float(np.nanmean(isi['cv_isi']))

    ff = fano_factor(spikes)
    metrics['mean_fano'] = float(np.nanmean(ff))

    aval = avalanche_analysis(spikes)
    metrics['avalanche_exp'] = aval['size_exponent']
    metrics['is_critical'] = aval['is_critical']

    metrics['eff_dim'] = effective_dimension(states)

    return metrics, states, spikes


def main():
    print("=" * 70)
    print("  NS-RAM Architecture Comparison")
    print("  5 neuron models × 10 benchmarks × multiple scales")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.mem_get_info()[0]/1e9:.0f} GB free")

    all_results = []

    # ── Scale 1: 128 neurons (small, fast) ──
    for N in [128, 512, 2000]:
        print(f"\n{'━' * 60}")
        print(f"  Scale: N = {N}")
        print(f"{'━' * 60}")

        networks = {
            'NS-RAM': NSRAMNetwork(N, n_inputs=1, connectivity='sparse', seed=42),
            'Izhikevich': IzhikevichNetwork(N, n_inputs=1, preset='mixed', seed=42),
            'PLIF': PLIFNetwork(N, n_inputs=1, seed=42),
            'ESN (tanh)': SoftwareESN(N, n_inputs=1, seed=42),
        }
        # HH is very slow for large N, only run at small scale
        if N <= 512:
            networks['Hodgkin-Huxley'] = HHNetwork(N, n_inputs=1, seed=42)

        for name, net in networks.items():
            print(f"\n  {name} (N={N})...", end=' ', flush=True)
            try:
                m, states, spikes = benchmark_network(net, f'{name} (N={N})')
                m['scale'] = N
                all_results.append(m)
                print(f"XOR1={m['xor1']:.1%} MC={m['mc']:.2f} MG={m['mg17']:.3f} "
                      f"KR={m['krank']} ED={m['eff_dim']:.1f} "
                      f"rate={m['mean_rate']:.1f} ({m['time']:.1f}s)")

                # Save raster for 128-neuron scale
                if N == 128:
                    raster_plot(spikes[:50, :500], title=f'{name} (N={N}) — Spike Raster',
                                save_path=os.path.join(OUT, f'raster_{name.replace(" ", "_").replace("(", "").replace(")", "")}_N{N}.png'))
                    plt.close('all')

            except Exception as e:
                print(f"FAILED: {e}")

    # ── Summary Table ──
    print(f"\n\n{'=' * 90}")
    print(f"  {'Architecture':<25s} {'N':>5s} {'XOR-1':>7s} {'XOR-5':>7s} {'MC':>6s} "
          f"{'NARMA':>6s} {'MG-17':>6s} {'Wave4':>7s} {'KRank':>6s} {'EffDim':>7s}")
    print(f"{'=' * 90}")
    for m in sorted(all_results, key=lambda x: (x['scale'], x['name'])):
        print(f"  {m['name']:<25s} {m['scale']:>5d} {m['xor1']:>6.1%} {m['xor5']:>6.1%} "
              f"{m['mc']:>6.2f} {m['narma10']:>6.3f} {m['mg17']:>6.3f} "
              f"{m['wave4']:>6.1%} {m['krank']:>6d} {m['eff_dim']:>7.1f}")

    # ── Publication Plots ──
    print("\n  Generating comparison plots...")

    fig = plt.figure(figsize=(22, 14), facecolor='#0d1117')
    fig.suptitle('NS-RAM vs Standard Neuron Models — Reservoir Computing Benchmarks',
                 fontsize=16, fontweight='bold', color='white', y=0.97)

    gs = GridSpec(2, 4, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.97, top=0.92, bottom=0.06)

    colors = {
        'NS-RAM': '#4CAF50',
        'Izhikevich': '#2196F3',
        'PLIF': '#FF9800',
        'ESN (tanh)': '#9C27B0',
        'Hodgkin-Huxley': '#F44336',
    }

    # Group results by architecture (use N=2000 where available, else largest)
    best_results = {}
    for m in all_results:
        arch = m['name'].split(' (N=')[0]
        if arch not in best_results or m['scale'] > best_results[arch]['scale']:
            best_results[arch] = m

    archs = list(best_results.keys())
    arch_colors = [colors.get(a, '#888888') for a in archs]

    # Plot 1: RC benchmark radar
    metrics_radar = ['xor1', 'xor5', 'mc', 'narma10', 'mg17', 'wave4']
    labels_radar = ['XOR-1', 'XOR-5', 'MC', 'NARMA-10', 'MG-17', 'Wave-4']

    ax = fig.add_subplot(gs[0, 0], polar=True)
    ax.set_facecolor('#0d1117')
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]

    for arch in archs:
        m = best_results[arch]
        # Normalize metrics to [0, 1]
        vals = []
        for k in metrics_radar:
            v = m[k]
            if k == 'mc':
                v = min(v / 5.0, 1.0)  # MC can be > 1
            vals.append(v)
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', label=arch, color=colors.get(arch, '#888'),
                linewidth=1.5, markersize=4)
        ax.fill(angles, vals, alpha=0.1, color=colors.get(arch, '#888'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, color='white', fontsize=7)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='gray')
    ax.set_title('(A) RC Benchmark Radar', color='white', fontsize=10, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7,
              facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    # Plot 2: Scaling (XOR-1 vs N)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor('#0d1117')
    for arch in archs:
        xs, ys = [], []
        for m in all_results:
            if m['name'].startswith(arch):
                xs.append(m['scale'])
                ys.append(m['xor1'])
        if xs:
            ax.semilogx(xs, [y * 100 for y in ys], 'o-', label=arch,
                         color=colors.get(arch, '#888'), linewidth=2, markersize=6)
    ax.set_xlabel('Neurons', color='white', fontsize=10)
    ax.set_ylabel('XOR-1 Accuracy (%)', color='white', fontsize=10)
    ax.set_title('(B) XOR Scaling', color='white', fontsize=10)
    ax.grid(True, alpha=0.2); ax.tick_params(colors='gray')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    # Plot 3: Memory Capacity vs N
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor('#0d1117')
    for arch in archs:
        xs, ys = [], []
        for m in all_results:
            if m['name'].startswith(arch):
                xs.append(m['scale'])
                ys.append(m['mc'])
        if xs:
            ax.semilogx(xs, ys, 's-', label=arch,
                         color=colors.get(arch, '#888'), linewidth=2, markersize=6)
    ax.set_xlabel('Neurons', color='white', fontsize=10)
    ax.set_ylabel('Memory Capacity', color='white', fontsize=10)
    ax.set_title('(C) Memory Capacity Scaling', color='white', fontsize=10)
    ax.grid(True, alpha=0.2); ax.tick_params(colors='gray')

    # Plot 4: Kernel Rank vs N
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor('#0d1117')
    for arch in archs:
        xs, ys = [], []
        for m in all_results:
            if m['name'].startswith(arch):
                xs.append(m['scale'])
                ys.append(m['krank'])
        if xs:
            ax.semilogx(xs, ys, '^-', label=arch,
                         color=colors.get(arch, '#888'), linewidth=2, markersize=6)
    ax.set_xlabel('Neurons', color='white', fontsize=10)
    ax.set_ylabel('Kernel Rank', color='white', fontsize=10)
    ax.set_title('(D) Nonlinear Transformation Capacity', color='white', fontsize=10)
    ax.grid(True, alpha=0.2); ax.tick_params(colors='gray')

    # Plot 5: Bar chart — best results
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#0d1117')
    bar_metrics = ['xor1', 'mc', 'narma10', 'mg17', 'wave4']
    bar_labels = ['XOR-1', 'MC/5', 'NARMA-10', 'MG-17', 'Wave-4']
    x = np.arange(len(bar_metrics))
    width = 0.15
    for i, arch in enumerate(archs):
        m = best_results[arch]
        vals = [m['xor1'], m['mc'] / 5, m['narma10'], m['mg17'], m['wave4']]
        ax.bar(x + i * width, vals, width, label=arch, color=colors.get(arch, '#888'),
               alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x + width * (len(archs) - 1) / 2)
    ax.set_xticklabels(bar_labels, color='white', fontsize=10)
    ax.set_ylabel('Score', color='white', fontsize=10)
    ax.set_title('(E) Best Benchmark Results (largest N)', color='white', fontsize=10)
    ax.grid(True, alpha=0.2, axis='y'); ax.tick_params(colors='gray')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white',
              loc='upper left')

    # Plot 6: Criticality — avalanche exponents
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('#0d1117')
    for i, arch in enumerate(archs):
        m = best_results[arch]
        exp = m['avalanche_exp']
        if not np.isnan(exp):
            c = colors.get(arch, '#888')
            ax.barh(i, exp, color=c, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axvline(1.5, color='#FF6B6B', linestyle='--', alpha=0.5, label='Critical (α=1.5)')
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs, color='white', fontsize=8)
    ax.set_xlabel('Avalanche Size Exponent α', color='white', fontsize=10)
    ax.set_title('(F) Criticality', color='white', fontsize=10)
    ax.tick_params(colors='gray')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    # Plot 7: Effective Dimension vs N
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor('#0d1117')
    for arch in archs:
        xs, ys = [], []
        for m in all_results:
            if m['name'].startswith(arch):
                xs.append(m['scale'])
                ys.append(m['eff_dim'])
        if xs:
            ax.semilogx(xs, ys, 'D-', label=arch,
                         color=colors.get(arch, '#888'), linewidth=2, markersize=6)
    ax.set_xlabel('Neurons', color='white', fontsize=10)
    ax.set_ylabel('Effective Dimension', color='white', fontsize=10)
    ax.set_title('(G) State Space Dimensionality', color='white', fontsize=10)
    ax.grid(True, alpha=0.2); ax.tick_params(colors='gray')

    plt.savefig(os.path.join(OUT, 'nsram_architecture_comparison.png'),
                dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {os.path.join(OUT, 'nsram_architecture_comparison.png')}")

    # ── MNIST comparison ──
    print("\n  Running MNIST comparison (N=2000)...")
    from nsram.vision import NSRAMClassifier
    from torchvision import datasets, transforms

    train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    X_tr = train.data.float().reshape(-1, 784)[:5000] / 255.0
    y_tr = train.targets[:5000]
    X_te = test.data.float().reshape(-1, 784)[:1000] / 255.0
    y_te = test.targets[:1000]

    print("  NS-RAM Classifier (N=2000)...")
    clf = NSRAMClassifier(N=2000, n_steps=8, seed=42)
    clf.fit(X_tr, y_tr, verbose=False)
    nsram_acc = clf.score(X_te, y_te, verbose=False)
    print(f"    NS-RAM: {nsram_acc:.1%}")

    # Simple ESN classifier for comparison
    print("  ESN Classifier (N=2000)...")
    esn_states = []
    for i in range(0, len(X_tr), 200):
        batch = X_tr[i:i+200].numpy()
        for img in batch:
            esn = SoftwareESN(2000, n_inputs=784, sparsity=0.01, seed=42)
            # Single-step encoding
            x = np.tanh(esn.W_in @ img)
            esn_states.append(x)
    esn_train = np.array(esn_states)
    # Ridge readout
    Y = np.zeros((len(y_tr), 10))
    Y[np.arange(len(y_tr)), y_tr.numpy()] = 1
    XtX = esn_train.T @ esn_train + 1.0 * np.eye(2000)
    XtY = esn_train.T @ Y
    W_read = np.linalg.solve(XtX, XtY)
    # Test
    esn_test_states = []
    for img in X_te.numpy():
        esn = SoftwareESN(2000, n_inputs=784, sparsity=0.01, seed=42)
        x = np.tanh(esn.W_in @ img)
        esn_test_states.append(x)
    esn_test = np.array(esn_test_states)
    esn_pred = (esn_test @ W_read).argmax(axis=1)
    esn_acc = (esn_pred == y_te.numpy()).mean()
    print(f"    ESN:    {esn_acc:.1%}")

    print(f"\n{'=' * 70}")
    print(f"  NS-RAM v0.5.0 Architecture Comparison Complete")
    print(f"  {len(all_results)} configurations tested")
    print(f"  MNIST: NS-RAM {nsram_acc:.1%} vs ESN {esn_acc:.1%}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
