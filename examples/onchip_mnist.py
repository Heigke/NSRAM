#!/usr/bin/env python3
"""On-chip learning benchmark: FF vs EP vs e-prop on MNIST.

All three learning rules use ONLY local computation + 1 global signal.
No backpropagation. Tapeout-compatible.

Architecture for each:
  Input (784) → Hidden (1000) → Output (10)

GPU-accelerated, 60K training images.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nsram.onchip_learning import ForwardForward, EquilibriumPropagation

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


def load_mnist():
    """Load MNIST from local data or torchvision."""
    try:
        from torchvision import datasets, transforms
        train = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.ToTensor())
        test = datasets.MNIST('data', train=False, download=True,
                               transform=transforms.ToTensor())
        X_train = train.data.float().reshape(-1, 784) / 255.0
        y_train = train.targets
        X_test = test.data.float().reshape(-1, 784) / 255.0
        y_test = test.targets
        return X_train, y_train, X_test, y_test
    except:
        # Fallback: load from local MNIST files
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'MNIST', 'raw')
        import struct
        def read_idx(path):
            with open(path, 'rb') as f:
                magic = struct.unpack('>I', f.read(4))[0]
                dims = struct.unpack(f'>{magic % 256}I', f.read(4 * (magic % 256)))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
        X_train = torch.tensor(read_idx(os.path.join(data_dir, 'train-images-idx3-ubyte')).reshape(-1, 784) / 255.0, dtype=torch.float32)
        y_train = torch.tensor(read_idx(os.path.join(data_dir, 'train-labels-idx1-ubyte')), dtype=torch.long)
        X_test = torch.tensor(read_idx(os.path.join(data_dir, 't10k-images-idx3-ubyte')).reshape(-1, 784) / 255.0, dtype=torch.float32)
        y_test = torch.tensor(read_idx(os.path.join(data_dir, 't10k-labels-idx1-ubyte')), dtype=torch.long)
        return X_train, y_train, X_test, y_test


def train_forward_forward(X_train, y_train, X_test, y_test, n_epochs=5, hidden=1000):
    """Train with Forward-Forward algorithm."""
    print(f"\n  Training Forward-Forward (hidden={hidden})...")
    ff = ForwardForward(lr=0.03)
    ff.init([784, hidden, hidden], device=DEVICE)

    accs = []
    for epoch in range(n_epochs):
        t0 = time.time()
        n_correct = 0; n_total = 0

        # Shuffle
        perm = torch.randperm(len(X_train))

        for i in range(0, len(X_train), 100):  # Mini-batch of 100
            batch_idx = perm[i:i+100]
            for idx in batch_idx:
                x = X_train[idx].to(DEVICE)
                y = y_train[idx].item()

                # Positive: real image with correct label embedded
                x_pos = x.clone()
                x_pos[:10] = 0  # Clear first 10 pixels
                x_pos[y] = 1.0  # Set correct label pixel

                # Negative: real image with WRONG label
                wrong_y = (y + np.random.randint(1, 10)) % 10
                x_neg = x.clone()
                x_neg[:10] = 0
                x_neg[wrong_y] = 1.0

                ff.train_step(x_pos, x_neg)
                n_total += 1

        # Test accuracy
        correct = 0
        for j in range(min(len(X_test), 2000)):
            x = X_test[j].to(DEVICE)
            best_goodness = -1; best_label = 0
            for label in range(10):
                x_test = x.clone(); x_test[:10] = 0; x_test[label] = 1.0
                h = ff.predict(x_test)
                g = (h**2).sum().item()
                if g > best_goodness:
                    best_goodness = g; best_label = label
            if best_label == y_test[j].item():
                correct += 1
        acc = correct / min(len(X_test), 2000)
        accs.append(acc)
        elapsed = time.time() - t0
        print(f"    Epoch {epoch}: acc={acc:.1%} ({elapsed:.0f}s)")

    return accs


def train_equilibrium_prop(X_train, y_train, X_test, y_test, n_epochs=5, hidden=500):
    """Train with Equilibrium Propagation."""
    print(f"\n  Training Equilibrium Propagation (hidden={hidden})...")
    ep = EquilibriumPropagation(lr=0.1, beta=1.0, settle_steps=30)
    ep.init([784, hidden, 10], device=DEVICE)

    accs = []
    for epoch in range(n_epochs):
        t0 = time.time()
        perm = torch.randperm(len(X_train))

        for i in range(0, min(len(X_train), 10000)):
            idx = perm[i]
            x = X_train[idx].to(DEVICE)
            # Target: one-hot
            target = torch.zeros(10, device=DEVICE)
            target[y_train[idx].item()] = 1.0

            ep.train_step(x, target)

        # Test
        correct = 0
        for j in range(min(len(X_test), 2000)):
            x = X_test[j].to(DEVICE)
            states = ep.settle(x)
            pred = states[-1].argmax().item()
            if pred == y_test[j].item():
                correct += 1
        acc = correct / min(len(X_test), 2000)
        accs.append(acc)
        elapsed = time.time() - t0
        print(f"    Epoch {epoch}: acc={acc:.1%} ({elapsed:.0f}s)")

    return accs


def train_reward_readout(X_train, y_train, X_test, y_test, n_epochs=5, N=2000):
    """Baseline: fixed NS-RAM reservoir + reward-modulated readout."""
    print(f"\n  Training Reward-Modulated Readout (N={N})...")
    from nsram.network import NSRAMNetwork
    from nsram.physics import DimensionlessParams

    # Build reservoir
    params = DimensionlessParams()
    net = NSRAMNetwork(N=N, n_inputs=784, params=params, backend='auto', seed=42)

    # Readout weights
    W_out = torch.zeros(10, N, device=DEVICE)
    elig = torch.zeros(10, N, device=DEVICE)

    accs = []
    for epoch in range(n_epochs):
        t0 = time.time()
        perm = torch.randperm(len(X_train))

        for i in range(min(len(X_train), 5000)):
            idx = perm[i]
            x = X_train[idx].numpy().reshape(1, -1)

            # Run reservoir for 5 steps
            result = net.run(np.tile(x, (5, 1)), noise_sigma=0.01)
            state = torch.tensor(result['states'][:, -1], device=DEVICE)

            # Readout
            logits = W_out @ state
            pred = logits.argmax().item()
            target = y_train[idx].item()

            # Reward: +1 correct, -1 wrong
            reward = 1.0 if pred == target else -0.1

            # Update readout
            one_hot = torch.zeros(10, device=DEVICE); one_hot[pred] = 1
            elig = 0.9 * elig + torch.outer(one_hot, state)
            W_out += 0.005 * reward * elig
            W_out.clamp_(-1, 1)

        # Test
        correct = 0
        for j in range(min(len(X_test), 2000)):
            x = X_test[j].numpy().reshape(1, -1)
            result = net.run(np.tile(x, (3, 1)), noise_sigma=0.01)
            state = torch.tensor(result['states'][:, -1], device=DEVICE)
            pred = (W_out @ state).argmax().item()
            if pred == y_test[j].item():
                correct += 1
        acc = correct / min(len(X_test), 2000)
        accs.append(acc)
        elapsed = time.time() - t0
        print(f"    Epoch {epoch}: acc={acc:.1%} ({elapsed:.0f}s)")

    return accs


def main():
    print("="*65)
    print("  On-Chip Learning Benchmark — MNIST")
    print("  All rules: NO backprop, tapeout-compatible")
    print("="*65)

    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  MNIST: {len(X_train)} train, {len(X_test)} test")

    results = {}

    # 1. Forward-Forward
    results['FF'] = train_forward_forward(X_train, y_train, X_test, y_test,
                                           n_epochs=5, hidden=500)

    # 2. Equilibrium Propagation
    results['EP'] = train_equilibrium_prop(X_train, y_train, X_test, y_test,
                                            n_epochs=5, hidden=500)

    # 3. Reward-modulated NS-RAM reservoir
    results['R-NSRAM'] = train_reward_readout(X_train, y_train, X_test, y_test,
                                               n_epochs=3, N=1000)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('On-Chip Learning: MNIST (No Backpropagation)',
                  fontsize=14, fontweight='bold', color='white')

    colors = {'FF': '#4CAF50', 'EP': '#2196F3', 'R-NSRAM': '#FF9800'}
    labels = {'FF': 'Forward-Forward (1-bit global)',
              'EP': 'Equilibrium Propagation (1 scalar)',
              'R-NSRAM': 'NS-RAM Reservoir + Reward (1 scalar)'}

    for name, accs in results.items():
        ax1.plot(range(len(accs)), [a*100 for a in accs], 'o-',
                  color=colors[name], linewidth=2.5, markersize=8, label=labels[name])
    ax1.set_xlabel('Epoch', color='white', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', color='white', fontsize=11)
    ax1.set_title('Learning Curves', fontsize=12, color='white')
    ax1.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax1.grid(True, alpha=0.2); ax1.set_facecolor('#0d1117')
    ax1.tick_params(colors='gray'); ax1.axhline(10, color='gray', linestyle=':', alpha=0.3)

    # Final accuracy bars
    final_accs = {name: accs[-1]*100 for name, accs in results.items()}
    x = np.arange(len(final_accs))
    ax2.bar(x, list(final_accs.values()), color=list(colors.values()),
            edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels([labels[n].split('(')[0] for n in final_accs],
                                              fontsize=9, color='white')
    for i, (name, acc) in enumerate(final_accs.items()):
        ax2.text(i, acc + 1, f'{acc:.1f}%', ha='center', fontsize=12,
                  fontweight='bold', color=colors[name])
    ax2.set_ylabel('Final Accuracy (%)', color='white', fontsize=11)
    ax2.set_title('Final Results (All Tapeout-Compatible)', fontsize=12, color='white')
    ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='gray')
    ax2.axhline(10, color='gray', linestyle=':', alpha=0.3, label='Random')
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    path = os.path.join(OUT, 'nsram_onchip_mnist.png')
    plt.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"\n  Saved: {path}")

    print(f"\n{'='*65}")
    print(f"  Final Results (NO backpropagation, tapeout-compatible)")
    print(f"{'='*65}")
    for name, accs in results.items():
        gs = {'FF': '1-bit phase', 'EP': '1 scalar nudge', 'R-NSRAM': '1 scalar reward'}
        print(f"  {labels[name]:<45s}: {accs[-1]:.1%}  (global signal: {gs[name]})")


if __name__ == '__main__':
    main()
