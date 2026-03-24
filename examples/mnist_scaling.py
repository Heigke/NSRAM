#!/usr/bin/env python3
"""NS-RAM MNIST scaling experiment.

Sweeps: neuron count (500→10K), timesteps (2→16), sparsity, spectral radius.
Generates publication plot showing scaling law.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nsram.vision import NSRAMClassifier

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

def load_mnist():
    from torchvision import datasets, transforms
    train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    return (train.data.float().reshape(-1, 784) / 255.0, train.targets,
            test.data.float().reshape(-1, 784) / 255.0, test.targets)

def main():
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"MNIST: {len(X_train)} train, {len(X_test)} test")

    results = {}

    # ── Sweep 1: Neuron count ──
    print("\n━━━ Sweep 1: Neuron Count (steps=8) ━━━")
    N_values = [500, 1000, 2000, 3000, 5000, 8000]
    for N in N_values:
        t0 = time.time()
        clf = NSRAMClassifier(N=N, n_steps=8, seed=42)
        clf.fit(X_train, y_train, verbose=False)
        acc = clf.score(X_test, y_test, verbose=False)
        elapsed = time.time() - t0
        results[f'N_{N}'] = acc
        print(f"  N={N:5d}: {acc:.1%} ({elapsed:.0f}s)")

    # ── Sweep 2: Timesteps ──
    print("\n━━━ Sweep 2: Timesteps (N=5000) ━━━")
    step_values = [2, 4, 6, 8, 12, 16]
    for steps in step_values:
        t0 = time.time()
        clf = NSRAMClassifier(N=5000, n_steps=steps, seed=42)
        clf.fit(X_train, y_train, verbose=False)
        acc = clf.score(X_test, y_test, verbose=False)
        elapsed = time.time() - t0
        results[f'steps_{steps}'] = acc
        print(f"  steps={steps:2d}: {acc:.1%} ({elapsed:.0f}s)")

    # ── Sweep 3: Spectral radius ──
    print("\n━━━ Sweep 3: Spectral Radius (N=5000, steps=8) ━━━")
    sr_values = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1]
    for sr in sr_values:
        clf = NSRAMClassifier(N=5000, n_steps=8, spectral_radius=sr, seed=42)
        clf.fit(X_train, y_train, verbose=False)
        acc = clf.score(X_test, y_test, verbose=False)
        results[f'sr_{sr}'] = acc
        print(f"  sr={sr:.2f}: {acc:.1%}")

    # ── Best config ──
    print("\n━━━ Best Config: N=8000, steps=12 ━━━")
    t0 = time.time()
    clf = NSRAMClassifier(N=8000, n_steps=12, spectral_radius=0.95, seed=42)
    clf.fit(X_train, y_train, verbose=True)
    acc_best = clf.score(X_test, y_test, verbose=True)
    elapsed = time.time() - t0
    results['best'] = acc_best
    print(f"  Best: {acc_best:.2%} ({elapsed:.0f}s)")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Reservoir MNIST Scaling Laws',
                  fontsize=14, fontweight='bold', color='white')

    # Neuron scaling
    ax = axes[0]; ax.set_facecolor('#0d1117')
    n_accs = [results[f'N_{N}'] * 100 for N in N_values]
    ax.semilogx(N_values, n_accs, 'o-', color='#4CAF50', linewidth=2.5, markersize=8)
    ax.set_xlabel('Neurons', color='white', fontsize=11)
    ax.set_ylabel('Accuracy (%)', color='white', fontsize=11)
    ax.set_title('(A) Neuron Count Scaling', fontsize=12, color='white')
    ax.grid(True, alpha=0.2); ax.tick_params(colors='gray')
    ax.annotate(f'{n_accs[-1]:.1f}%', xy=(N_values[-1], n_accs[-1]),
                textcoords="offset points", xytext=(-30, 10), fontsize=11,
                fontweight='bold', color='#4CAF50')

    # Timestep scaling
    ax = axes[1]; ax.set_facecolor('#0d1117')
    s_accs = [results[f'steps_{s}'] * 100 for s in step_values]
    ax.plot(step_values, s_accs, 's-', color='#2196F3', linewidth=2.5, markersize=8)
    ax.set_xlabel('Timesteps', color='white', fontsize=11)
    ax.set_ylabel('Accuracy (%)', color='white', fontsize=11)
    ax.set_title('(B) Temporal Integration', fontsize=12, color='white')
    ax.grid(True, alpha=0.2); ax.tick_params(colors='gray')

    # Spectral radius
    ax = axes[2]; ax.set_facecolor('#0d1117')
    sr_accs = [results[f'sr_{sr}'] * 100 for sr in sr_values]
    ax.plot(sr_values, sr_accs, '^-', color='#FF9800', linewidth=2.5, markersize=8)
    ax.set_xlabel('Spectral Radius', color='white', fontsize=11)
    ax.set_ylabel('Accuracy (%)', color='white', fontsize=11)
    ax.set_title('(C) Edge of Chaos', fontsize=12, color='white')
    ax.grid(True, alpha=0.2); ax.tick_params(colors='gray')
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'nsram_mnist_scaling.png')
    plt.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"\n  Saved: {path}")

    # Summary
    print(f"\n{'='*55}")
    print(f"  NS-RAM Reservoir MNIST Results")
    print(f"{'='*55}")
    print(f"  Best accuracy: {acc_best:.2%} (N=8000, steps=12, sr=0.95)")
    print(f"  5000 neurons:  {results['N_5000']:.1%}")
    print(f"  1000 neurons:  {results['N_1000']:.1%}")
    print(f"  Optimal SR:    {sr_values[np.argmax(sr_accs)]:.2f}")


if __name__ == '__main__':
    main()
