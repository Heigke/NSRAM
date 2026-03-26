#!/usr/bin/env python3
"""NS-RAM Scalable Learner — Reservoir Features + On-Chip Readout Learning

Architecture that WORKS and scales to 100K+:
  Stage 1: NS-RAM reservoir (fixed, large) → rich temporal features
  Stage 2: Small learnable readout via charge-trapping e-prop

This exploits BOTH NS-RAM strengths:
  - Reservoir: avalanche nonlinearity → 97% XOR, 99.6% Mackey-Glass
  - Learning: SRH charge trapping → weight updates map to silicon

Progressively harder benchmarks:
  Level 1: MNIST (baseline, should match 96%+)
  Level 2: Fashion-MNIST (harder, target 80%+)
  Level 3: Temporal spoken digits (SHD-like, target 70%+)
  Level 4: VizDoom with temporal retina (target: consistent kills)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NSRAMReservoirEncoder:
    """Large NS-RAM reservoir as fixed feature extractor. GPU-accelerated."""

    def __init__(self, N, n_inputs, n_steps=8, spectral_radius=0.95,
                 sparsity=0.02, variability=0.10, seed=42):
        self.N = N
        self.n_steps = n_steps
        self.device = DEVICE
        rng = np.random.RandomState(seed)
        v = variability

        self.theta = torch.tensor(
            np.clip(1 + v*0.05*rng.randn(N), 0.5, 2).astype(np.float32), device=DEVICE)
        self.bg = 0.88 * self.theta
        self.dT = torch.tensor(
            np.clip(0.1 + v*0.015*rng.randn(N), 0.02, 0.5).astype(np.float32), device=DEVICE)
        self.tau_syn = torch.tensor(
            np.clip(0.5 + v*0.1*rng.randn(N), 0.1, 2).astype(np.float32), device=DEVICE)

        self.W_in = torch.tensor(
            rng.randn(N, n_inputs).astype(np.float32) * 0.1, device=DEVICE)

        mask = (rng.rand(N, N) < sparsity).astype(np.float32)
        np.fill_diagonal(mask, 0)
        W = rng.randn(N, N).astype(np.float32) * mask
        N_exc = int(N * 0.8)
        signs = np.ones(N, dtype=np.float32); signs[N_exc:] = -1
        W = np.abs(W) * signs[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W *= spectral_radius / eigs.max()
        self.W_rec = torch.tensor(W, device=DEVICE)

    @torch.no_grad()
    def encode_batch(self, X_batch):
        """Encode batch through reservoir. X_batch: (B, D). Returns (B, N)."""
        B = X_batch.shape[0]
        N = self.N
        Vm = torch.zeros(B, N, device=self.device)
        syn = torch.zeros(B, N, device=self.device)
        ft = torch.zeros(B, N, device=self.device)

        I_in = X_batch @ self.W_in.T

        for t in range(self.n_steps):
            I_syn = syn @ self.W_rec.T * 0.3
            leak = -Vm
            exp_t = self.dT.unsqueeze(0) * torch.exp(
                torch.clamp((Vm - self.theta.unsqueeze(0)) / self.dT.unsqueeze(0).clamp(min=1e-6), -10, 5))
            Vm = Vm + leak + self.bg.unsqueeze(0) + I_in + I_syn + exp_t
            Vm += 0.01 * torch.randn(B, N, device=self.device)
            Vm.clamp_(-2, 5)
            spiked = Vm >= self.theta.unsqueeze(0)
            Vm[spiked] = 0
            syn[spiked] += 1
            syn *= torch.exp(-1.0 / self.tau_syn.unsqueeze(0))
            ft = 0.8 * ft + 0.2 * Vm

        return Vm + 0.3 * ft


class ChargeTrapReadout:
    """Learnable readout using NS-RAM charge-trapping weight updates.

    Weights stored as trapped charge Q ∈ [0, 1] with 14 levels.
    Learning: supervised charge injection based on error signal.
    Maps to silicon: Vg2 controls write strength, charge level = weight.
    """

    def __init__(self, n_in, n_out, seed=42):
        rng = np.random.RandomState(seed)
        self.n_in = n_in
        self.n_out = n_out

        # Charge states (14 levels, like Pazos Fig 4)
        self.Q = torch.tensor(
            rng.uniform(0.3, 0.7, (n_out, n_in)).astype(np.float32), device=DEVICE)
        self.W_scale = 2.0 / np.sqrt(n_in)

        # Charge trapping dynamics
        self.k_cap = 0.5    # Capture rate (Vg2-dependent in hardware)
        self.k_em = 0.2     # Emission rate
        self.eta = 0.001    # Learning rate

    @property
    def W(self):
        return self.W_scale * (2 * self.Q - 1)

    @torch.no_grad()
    def forward(self, features):
        """features: (B, N_in). Returns (B, n_out) logits."""
        return features @ self.W.T

    @torch.no_grad()
    def learn(self, features, targets, predictions):
        """Update charge states based on error.

        This is the tapeout-realistic weight update:
        - Compute error: target_onehot - softmax(predictions)
        - dQ = eta × error^T × features (outer product)
        - Charge capture for positive dQ, emission for negative
        - Quantize to 14 levels (NS-RAM conductance states)
        """
        B = features.shape[0]
        target_oh = torch.zeros(B, self.n_out, device=DEVICE)
        target_oh.scatter_(1, targets.unsqueeze(1), 1.0)

        probs = torch.softmax(predictions, dim=1)
        error = target_oh - probs  # (B, n_out)

        # Outer product: gradient of W
        grad = error.T @ features / B  # (n_out, n_in)

        # Charge trapping update (SRH-like)
        # Positive gradient → capture (Q increases → W increases)
        # Negative gradient → emission (Q decreases → W decreases)
        dQ_cap = self.k_cap * (1 - self.Q) * grad.clamp(min=0)
        dQ_em = self.k_em * self.Q * (-grad).clamp(min=0)
        dQ = self.eta * (dQ_cap - dQ_em)

        self.Q = (self.Q + dQ).clamp(0, 1)

        # Quantize to 14 levels (like real NS-RAM)
        self.Q = (self.Q * 13).round() / 13

        return float(error.abs().mean())


def run_benchmark(name, X_train, y_train, X_test, y_test, n_classes,
                  reservoir_size, n_epochs=20, batch_size=200):
    """Run a complete benchmark with reservoir + charge-trap readout."""
    print(f"\n{'─' * 55}")
    print(f"  {name}: {reservoir_size:,} neurons, {n_classes} classes")
    print(f"{'─' * 55}")

    n_features = X_train.shape[1]
    reservoir = NSRAMReservoirEncoder(reservoir_size, n_features, seed=42)
    readout = ChargeTrapReadout(reservoir_size, n_classes, seed=42)

    print(f"  Encoding training data ({len(X_train)} samples)...")
    t0 = time.time()
    train_features = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size]
        feat = reservoir.encode_batch(batch)
        train_features.append(feat)
    train_features = torch.cat(train_features, dim=0)
    print(f"  Encoded in {time.time()-t0:.1f}s")

    print(f"  Encoding test data ({len(X_test)} samples)...")
    test_features = []
    for i in range(0, len(X_test), batch_size):
        feat = reservoir.encode_batch(X_test[i:i+batch_size])
        test_features.append(feat)
    test_features = torch.cat(test_features, dim=0)

    train_accs = []
    test_accs = []

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(X_train), device=DEVICE)
        total_err = 0
        correct = 0

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            feat = train_features[idx]
            targets = y_train[idx]

            logits = readout.forward(feat)
            err = readout.learn(feat, targets, logits)
            total_err += err

            correct += (logits.argmax(dim=1) == targets).sum().item()

        train_acc = correct / len(X_train)
        train_accs.append(train_acc)

        # Test
        logits = readout.forward(test_features)
        test_acc = (logits.argmax(dim=1) == y_test).float().mean().item()
        test_accs.append(test_acc)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch:2d}: train={train_acc:.1%} test={test_acc:.1%}")

    return train_accs, test_accs


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}, "
              f"{torch.cuda.mem_get_info()[0]/1e9:.0f} GB free")

    from torchvision import datasets, transforms

    results = {}

    # ═══ Level 1: MNIST ═══
    train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    for N in [2000, 10000, 50000]:
        X_tr = train.data[:10000].float().reshape(-1, 784).to(DEVICE) / 255.0
        y_tr = train.targets[:10000].to(DEVICE)
        X_te = test.data[:2000].float().reshape(-1, 784).to(DEVICE) / 255.0
        y_te = test.targets[:2000].to(DEVICE)

        tr_acc, te_acc = run_benchmark(
            f'MNIST (N={N:,})', X_tr, y_tr, X_te, y_te, 10,
            reservoir_size=N, n_epochs=20)
        results[f'MNIST_N{N}'] = te_acc[-1]

    # ═══ Level 2: Fashion-MNIST ═══
    train_f = datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_f = datasets.FashionMNIST('data', train=False, download=True, transform=transforms.ToTensor())

    for N in [5000, 20000, 50000]:
        X_tr = train_f.data[:10000].float().reshape(-1, 784).to(DEVICE) / 255.0
        y_tr = train_f.targets[:10000].to(DEVICE)
        X_te = test_f.data[:2000].float().reshape(-1, 784).to(DEVICE) / 255.0
        y_te = test_f.targets[:2000].to(DEVICE)

        tr_acc, te_acc = run_benchmark(
            f'Fashion-MNIST (N={N:,})', X_tr, y_tr, X_te, y_te, 10,
            reservoir_size=N, n_epochs=25)
        results[f'FMNIST_N{N}'] = te_acc[-1]

    # ═══ Level 3: Temporal Pattern Recognition ═══
    print(f"\n{'─' * 55}")
    print(f"  Level 3: Temporal Sequence Classification")
    print(f"{'─' * 55}")

    # Generate temporal spike patterns (SHD-like)
    rng = np.random.RandomState(42)
    n_classes = 10
    n_channels = 100
    n_steps = 50
    n_train_t = 2000
    n_test_t = 500

    def make_temporal_data(n_samples):
        X = np.zeros((n_samples, n_channels * n_steps), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            c = rng.randint(n_classes)
            y[i] = c
            # Each class has a unique temporal pattern
            for ch in range(n_channels):
                freq = 0.5 + c * 0.3 + ch * 0.01
                phase = c * 0.7 + ch * 0.1
                t = np.arange(n_steps) / n_steps
                rate = 0.3 * (1 + np.sin(2 * np.pi * freq * t + phase))
                X[i, ch*n_steps:(ch+1)*n_steps] = (rng.rand(n_steps) < rate).astype(np.float32)
        return X, y

    X_tr_t, y_tr_t = make_temporal_data(n_train_t)
    X_te_t, y_te_t = make_temporal_data(n_test_t)

    X_tr_t = torch.tensor(X_tr_t, device=DEVICE)
    y_tr_t = torch.tensor(y_tr_t, device=DEVICE)
    X_te_t = torch.tensor(X_te_t, device=DEVICE)
    y_te_t = torch.tensor(y_te_t, device=DEVICE)

    for N in [10000, 50000]:
        tr_acc, te_acc = run_benchmark(
            f'Temporal Patterns (N={N:,})', X_tr_t, y_tr_t, X_te_t, y_te_t, 10,
            reservoir_size=N, n_epochs=30)
        results[f'Temporal_N{N}'] = te_acc[-1]

    # ═══ Summary ═══
    print(f"\n{'=' * 60}")
    print(f"  NS-RAM Scalable Learner — All Results")
    print(f"  Reservoir (fixed) + Charge-Trap Readout (learns on-chip)")
    print(f"{'=' * 60}")
    for name, acc in results.items():
        print(f"  {name:30s}: {acc:.1%}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
    fig.suptitle('NS-RAM Scalable Learner — Reservoir + On-Chip Readout',
                 fontsize=14, fontweight='bold', color='white')

    # MNIST scaling
    ax = axes[0]; ax.set_facecolor('#0d1117')
    mnist_ns = [2000, 10000, 50000]
    mnist_accs = [results.get(f'MNIST_N{n}', 0) * 100 for n in mnist_ns]
    ax.semilogx(mnist_ns, mnist_accs, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Reservoir Neurons', color='white')
    ax.set_ylabel('Test Accuracy (%)', color='white')
    ax.set_title('MNIST', color='white', fontsize=12)
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2)
    for n, a in zip(mnist_ns, mnist_accs):
        ax.annotate(f'{a:.1f}%', (n, a), textcoords="offset points",
                     xytext=(0, 10), ha='center', color='#4CAF50', fontsize=9)

    # Fashion-MNIST scaling
    ax = axes[1]; ax.set_facecolor('#0d1117')
    fmnist_ns = [5000, 20000, 50000]
    fmnist_accs = [results.get(f'FMNIST_N{n}', 0) * 100 for n in fmnist_ns]
    ax.semilogx(fmnist_ns, fmnist_accs, 'c-s', linewidth=2, markersize=8)
    ax.set_xlabel('Reservoir Neurons', color='white')
    ax.set_ylabel('Test Accuracy (%)', color='white')
    ax.set_title('Fashion-MNIST', color='white', fontsize=12)
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2)
    for n, a in zip(fmnist_ns, fmnist_accs):
        ax.annotate(f'{a:.1f}%', (n, a), textcoords="offset points",
                     xytext=(0, 10), ha='center', color='#00BCD4', fontsize=9)

    # Temporal scaling
    ax = axes[2]; ax.set_facecolor('#0d1117')
    temp_ns = [10000, 50000]
    temp_accs = [results.get(f'Temporal_N{n}', 0) * 100 for n in temp_ns]
    ax.semilogx(temp_ns, temp_accs, 'r-^', linewidth=2, markersize=8)
    ax.set_xlabel('Reservoir Neurons', color='white')
    ax.set_ylabel('Test Accuracy (%)', color='white')
    ax.set_title('Temporal Patterns', color='white', fontsize=12)
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_scalable_learner.png'), dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"\n  Saved: {os.path.join(OUT, 'nsram_scalable_learner.png')}")


if __name__ == '__main__':
    main()
