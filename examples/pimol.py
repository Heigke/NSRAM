#!/usr/bin/env python3
"""PIMOL: Physics-Informed Modular Online Learner

Addresses the three root causes of catastrophic forgetting:
  1. Global credit → Local FF goodness per expert (no backprop between experts)
  2. Entangled repr → Modular experts with router (knowledge isolation)
  3. Stability-plasticity → Dual-timescale NS-RAM weights (body + oxide)

Every operation maps to NS-RAM silicon:
  W_fast = body potential (continuous, volatile, τ~50 steps)
  W_slow = oxide trapping (14 discrete levels, non-volatile, τ~10⁴s)
  Physics optimizer: body capacitance = momentum, VG2 = adaptive lr
  Consolidation: surprise-gated body→oxide transfer via SRH trapping ODE
"""

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════
# 1. DUAL-TIMESCALE WEIGHT LAYER
# ═══════════════════════════════════════════════════════════════════

class NSRAMDualWeight:
    """Synapse layer with two physical weight components.

    W_eff = W_slow + alpha * W_fast

    W_slow: 14-level quantized (oxide trapping, non-volatile)
    W_fast: continuous (body potential, volatile, decays each step)
    """

    def __init__(self, n_in, n_out, n_levels=14, alpha=0.5, tau_body=50, seed=42):
        rng = np.random.RandomState(seed)
        self.n_levels = n_levels
        self.alpha = alpha
        self.tau_body = tau_body

        # Slow weight: oxide-trapped charge, 14 discrete levels
        W0 = rng.randn(n_out, n_in).astype(np.float32) * 0.3 / np.sqrt(n_in)
        self.W_slow = torch.tensor(W0, device=DEVICE)
        self._quantize_slow()

        # Fast weight: body potential, continuous, volatile
        self.W_fast = torch.zeros(n_out, n_in, device=DEVICE)

        # Bias (small, continuous)
        self.b = torch.zeros(n_out, device=DEVICE)

    def _quantize_slow(self):
        wmin, wmax = self.W_slow.min(), self.W_slow.max()
        if wmax - wmin < 1e-10:
            return
        wn = (self.W_slow - wmin) / (wmax - wmin)
        self.W_slow = (torch.round(wn * (self.n_levels - 1)) / (self.n_levels - 1)) * (wmax - wmin) + wmin

    @property
    def W(self):
        return self.W_slow + self.alpha * self.W_fast

    def forward(self, x):
        return torch.relu(x @ self.W.T + self.b)

    def decay_body(self):
        """Body potential leaks each step (volatile)."""
        self.W_fast *= (1.0 - 1.0 / self.tau_body)


# ═══════════════════════════════════════════════════════════════════
# 2. PHYSICS-BASED OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

def physics_update(layer, pre, post, error_signal, lr_base=0.03):
    """Update W_fast using NS-RAM charge injection physics.

    Body capacitance = natural momentum (W_fast accumulates)
    VG2-controlled capture rate = adaptive learning rate

    Args:
        layer: NSRAMDualWeight
        pre: (B, n_in) pre-synaptic activations
        post: (B, n_out) post-synaptic activations
        error_signal: scalar, positive = reinforce, negative = anti-reinforce
        lr_base: base learning rate
    """
    B = pre.shape[0]

    # Adaptive learning rate: larger error → higher capture rate
    # Mimics VG2 control: |error| drives capture probability
    # charge_capture_rate is sigmoid: high error → high rate, low error → low rate
    adaptive_lr = lr_base * torch.sigmoid(torch.tensor(abs(error_signal) * 3.0))

    # Hebbian outer product (purely local: pre × post)
    pre_norm = pre / (pre.norm(dim=1, keepdim=True) + 1e-6)
    dW = (post.T @ pre_norm) / B  # (n_out, n_in)

    # Inject current into body: CB * dVB = I_inject
    # Direction: positive error → reinforce (increase weights that fired)
    #            negative error → anti-reinforce (decrease)
    sign = 1.0 if error_signal > 0 else -1.0
    layer.W_fast += float(adaptive_lr) * sign * dW

    # Clamp body potential (finite dynamic range)
    layer.W_fast.clamp_(-1.0, 1.0)

    # Bias update
    layer.b += float(adaptive_lr) * 0.1 * sign * post.mean(0)

    # Body leaks (volatile)
    layer.decay_body()


# ═══════════════════════════════════════════════════════════════════
# 3. EXPERT (local FF network)
# ═══════════════════════════════════════════════════════════════════

class Expert:
    """Small 2-layer FF network with dual-timescale weights.

    Each expert has its own local goodness objective — no gradient
    flows between experts.
    """

    def __init__(self, n_in, n_hidden=256, n_classes=10, n_levels=14, seed=42):
        self.layer1 = NSRAMDualWeight(n_in, n_hidden, n_levels, seed=seed)
        self.layer2 = NSRAMDualWeight(n_hidden, n_hidden, n_levels, seed=seed + 100)
        self.n_hidden = n_hidden
        self.theta = 1.0
        self.competence = torch.zeros(10, device=DEVICE)
        # Fixed random feedback matrices for DFA (hardwired in silicon)
        rng = np.random.RandomState(seed + 500)
        self.B_fb1 = torch.tensor(rng.randn(n_hidden, n_classes).astype(np.float32) * 0.05, device=DEVICE)
        self.B_fb2 = torch.tensor(rng.randn(n_hidden, n_classes).astype(np.float32) * 0.05, device=DEVICE)

    def forward(self, x):
        h1 = self.layer1.forward(x)
        h1_norm = h1 / (h1.norm(dim=1, keepdim=True) + 1e-6)
        h2 = self.layer2.forward(h1_norm)
        return h2

    def goodness(self, h):
        return (h ** 2).mean(dim=1)

    def learn_ff(self, x_pos, x_neg, lr=0.03):
        """Local FF learning step. Returns goodness scores."""
        h1_pos = self.layer1.forward(x_pos)
        h1_neg = self.layer1.forward(x_neg)
        err1 = float(self.goodness(h1_pos).mean() - self.goodness(h1_neg).mean())
        physics_update(self.layer1, x_pos, h1_pos, err1, lr)
        physics_update(self.layer1, x_neg, h1_neg, -err1 * 0.5, lr)

        h1p_n = h1_pos / (h1_pos.norm(dim=1, keepdim=True) + 1e-6)
        h1n_n = h1_neg / (h1_neg.norm(dim=1, keepdim=True) + 1e-6)
        h2_pos = self.layer2.forward(h1p_n.detach())
        h2_neg = self.layer2.forward(h1n_n.detach())
        err2 = float(self.goodness(h2_pos).mean() - self.goodness(h2_neg).mean())
        physics_update(self.layer2, h1p_n.detach(), h2_pos, err2, lr)
        physics_update(self.layer2, h1n_n.detach(), h2_neg, -err2 * 0.5, lr)
        return float(self.goodness(h1_pos).mean()), float(self.goodness(h1_neg).mean())

    def learn_dfa(self, x, error, lr=0.01):
        """DFA learning: direct error projection to each layer.
        Fixed random feedback (hardware-realistic). Much stronger than FF."""
        h1 = self.layer1.forward(x)
        h1_n = h1 / (h1.norm(dim=1, keepdim=True) + 1e-6)
        h2 = self.layer2.forward(h1_n)

        # DFA: project output error directly to each hidden layer
        # Layer 2: direct error
        le2 = error @ self.B_fb2.T * (h2 > 0).float()
        dW2 = (le2.T @ h1_n) / x.shape[0]
        self.layer2.W_fast += lr * dW2
        self.layer2.W_fast.clamp_(-1, 1)
        self.layer2.b += lr * 0.1 * le2.mean(0)
        self.layer2.decay_body()

        # Layer 1: direct error (not through layer 2)
        le1 = error @ self.B_fb1.T * (h1 > 0).float()
        dW1 = (le1.T @ x) / x.shape[0]
        self.layer1.W_fast += lr * dW1
        self.layer1.W_fast.clamp_(-1, 1)
        self.layer1.b += lr * 0.1 * le1.mean(0)
        self.layer1.decay_body()

    def consolidate(self, confidence):
        """Transfer body → oxide when confident. SRH trapping physics."""
        gate = max(0, confidence - 0.5)
        if gate < 0.01:
            return
        k_cap = gate * 5.0
        k_em = 0.005
        for layer in [self.layer1, self.layer2]:
            # SRH: dQ = k_cap * (1-Q) * signal - k_em * Q
            body_signal = layer.W_fast.clamp(-1, 1)
            dQ = k_cap * (1 - layer.W_slow) * body_signal.clamp(min=0) - k_em * layer.W_slow
            layer.W_slow = (layer.W_slow + 0.01 * dQ).clamp(-2, 2)
            layer._quantize_slow()
            layer.W_fast *= 0.3  # Partial body discharge after consolidation


# ═══════════════════════════════════════════════════════════════════
# 4. ROUTER
# ═══════════════════════════════════════════════════════════════════

class Router:
    """Lightweight input-to-expert gating. Reward-modulated updates."""

    def __init__(self, n_in, n_experts, seed=42):
        rng = np.random.RandomState(seed)
        self.W = torch.tensor(
            rng.randn(n_experts, n_in).astype(np.float32) * 0.01,
            device=DEVICE)
        self.n_experts = n_experts
        self.temperature = 1.0

    def route(self, x, k=2):
        """Select top-k experts for input x. Returns (weights, indices)."""
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-6)
        logits = x_norm @ self.W[:self.n_experts].T
        weights = F.softmax(logits / self.temperature, dim=1)
        topk_w, topk_i = weights.topk(min(k, self.n_experts), dim=1)
        return topk_w, topk_i

    def update(self, x, expert_indices, reward):
        """Bandit-style update: reinforce experts that got reward."""
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-6)
        for b in range(x.shape[0]):
            for idx in expert_indices[b]:
                self.W[idx] += 0.001 * reward[b] * x_norm[b]

    def expand(self, new_expert_idx, n_in):
        """Add routing weights for a new expert."""
        if new_expert_idx >= self.W.shape[0]:
            new_row = torch.randn(1, n_in, device=DEVICE) * 0.01
            self.W = torch.cat([self.W, new_row], dim=0)


# ═══════════════════════════════════════════════════════════════════
# 5. PIMOL — Full System
# ═══════════════════════════════════════════════════════════════════

class PIMOL:
    """Physics-Informed Modular Online Learner."""

    def __init__(self, input_dim=784, expert_dim=256, n_classes=10,
                 n_initial=2, n_max=8, n_levels=14, seed=42):
        self.n_classes = n_classes
        self.n_max = n_max
        self.expert_dim = expert_dim
        self.input_dim = input_dim
        self.seed = seed

        # Experts
        self.experts = [Expert(input_dim, expert_dim, n_classes, n_levels, seed=seed + i)
                        for i in range(n_initial)]
        self.n_active = n_initial

        # Router
        self.router = Router(input_dim, n_max, seed=seed + 50)

        # Combiner: simple linear readout from concatenated expert outputs
        rng = np.random.RandomState(seed + 200)
        max_concat = n_max * expert_dim
        self.W_out = torch.tensor(
            rng.randn(n_classes, max_concat).astype(np.float32) * 0.01,
            device=DEVICE)
        self.b_out = torch.zeros(n_classes, device=DEVICE)

        # Consolidation
        self.consol_threshold = 0.7
        self.step_count = 0
        self.recent_acc = []

    def forward(self, x, k=2):
        """Forward pass: route → experts → combine → classify."""
        B = x.shape[0]
        k_actual = min(k, self.n_active)
        gate_w, gate_i = self.router.route(x, k=k_actual)

        # Collect expert outputs
        combined = torch.zeros(B, self.n_max * self.expert_dim, device=DEVICE)
        for b in range(B):
            for j in range(k_actual):
                idx = gate_i[b, j].item()
                if idx < self.n_active:
                    h = self.experts[idx].forward(x[b:b+1])
                    start = idx * self.expert_dim
                    combined[b, start:start+self.expert_dim] = h[0] * gate_w[b, j]

        logits = combined @ self.W_out.T + self.b_out
        return logits, gate_i

    def train_step(self, x, y, lr=0.03):
        """One online learning step. Returns accuracy on this batch."""
        B = x.shape[0]
        nc = self.n_classes

        # Forward
        logits, gate_i = self.forward(x)
        pred = logits.argmax(dim=1)
        correct = (pred == y).float()
        acc = correct.mean().item()
        reward = correct * 2 - 1  # +1 correct, -1 wrong

        # Update combiner (simple delta rule — small, fine to be supervised)
        target = torch.zeros(B, nc, device=DEVICE)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        error = target - F.softmax(logits, dim=1)
        combined = torch.zeros(B, self.n_max * self.expert_dim, device=DEVICE)
        for b in range(B):
            for j in range(min(2, self.n_active)):
                idx = gate_i[b, j].item()
                if idx < self.n_active:
                    h = self.experts[idx].forward(x[b:b+1])
                    start = idx * self.expert_dim
                    combined[b, start:start+self.expert_dim] = h[0]
        self.W_out += 0.01 * (error.T @ combined) / B
        self.b_out += 0.01 * error.mean(0)

        # Compute error for DFA (target - softmax)
        dfa_error = target - F.softmax(logits, dim=1)  # (B, nc)

        # Train active experts with DFA (stronger than FF, still local)
        for b in range(B):
            for j in range(min(2, self.n_active)):
                idx = gate_i[b, j].item()
                if idx < self.n_active:
                    self.experts[idx].learn_dfa(
                        x[b:b+1], dfa_error[b:b+1], lr=lr)

        # Router update
        self.router.update(x, gate_i, reward)

        # Consolidation check
        confidence = F.softmax(logits, dim=1).max(dim=1).values.mean().item()
        if confidence > self.consol_threshold:
            for idx in range(self.n_active):
                self.experts[idx].consolidate(confidence)

        # Expert spawning
        self.step_count += 1
        self.recent_acc.append(acc)
        if len(self.recent_acc) > 100:
            self.recent_acc.pop(0)
        if (self.step_count % 500 == 0 and
                self.n_active < self.n_max and
                np.mean(self.recent_acc[-100:]) < 0.3):
            self._spawn_expert()

        return acc

    def _spawn_expert(self):
        """Add new expert when existing ones are struggling."""
        new_idx = self.n_active
        if new_idx >= self.n_max:
            return
        self.experts.append(Expert(self.input_dim, self.expert_dim, self.n_classes, 14,
                                    seed=self.seed + 1000 + new_idx))
        self.router.expand(new_idx, self.input_dim)
        self.n_active += 1
        print(f'    [Spawned expert {new_idx}]')

    def predict(self, x):
        with torch.no_grad():
            logits, _ = self.forward(x)
        return logits.argmax(dim=1)

    def deploy_mode(self):
        """Final consolidation — inference uses only quantized W_slow."""
        for exp in self.experts[:self.n_active]:
            exp.consolidate(1.0)
            for layer in [exp.layer1, exp.layer2]:
                layer.W_fast.zero_()
                layer.alpha = 0


# ═══════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate(model, X, y, batch=512):
    correct = 0
    for i in range(0, len(X), batch):
        pred = model.predict(X[i:i+batch])
        correct += (pred == y[i:i+batch]).sum().item()
    return correct / len(X)


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    from torchvision import datasets, transforms

    # Load data
    tr_m = datasets.MNIST('data', True, download=True, transform=transforms.ToTensor())
    te_m = datasets.MNIST('data', False, download=True, transform=transforms.ToTensor())
    tr_f = datasets.FashionMNIST('data', True, download=True, transform=transforms.ToTensor())
    te_f = datasets.FashionMNIST('data', False, download=True, transform=transforms.ToTensor())

    X_tr_m = tr_m.data[:50000].float().reshape(-1, 784).to(DEVICE) / 255.0
    y_tr_m = tr_m.targets[:50000].to(DEVICE)
    X_te_m = te_m.data[:5000].float().reshape(-1, 784).to(DEVICE) / 255.0
    y_te_m = te_m.targets[:5000].to(DEVICE)
    X_tr_f = tr_f.data[:50000].float().reshape(-1, 784).to(DEVICE) / 255.0
    y_tr_f = tr_f.targets[:50000].to(DEVICE)
    X_te_f = te_f.data[:5000].float().reshape(-1, 784).to(DEVICE) / 255.0
    y_te_f = te_f.targets[:5000].to(DEVICE)

    batch_size = 64
    results = {}

    # ── Run PIMOL ──
    print(f"\n{'='*60}")
    print("  PIMOL: Physics-Informed Modular Online Learner")
    print("  Sequential: MNIST → Fashion-MNIST")
    print(f"{'='*60}")

    model = PIMOL(input_dim=784, expert_dim=256, n_initial=3, n_max=6, seed=42)
    print(f"  {model.n_active} experts, {model.expert_dim}D each")

    # Phase 1: MNIST
    print("\n  Phase 1: MNIST (online, 1 epoch)...")
    t0 = time.time()
    accs_mnist = []
    perm = torch.randperm(len(X_tr_m), device=DEVICE)
    for i in range(0, len(X_tr_m), batch_size):
        idx = perm[i:i+batch_size]
        acc = model.train_step(X_tr_m[idx], y_tr_m[idx])
        accs_mnist.append(acc)
        if (i // batch_size) % 100 == 0:
            eval_acc = evaluate(model, X_te_m, y_te_m)
            print(f"    Step {i//batch_size}: batch={acc:.1%} eval={eval_acc:.1%} "
                  f"experts={model.n_active}")

    mnist_after_p1 = evaluate(model, X_te_m, y_te_m)
    print(f"  → MNIST after Phase 1: {mnist_after_p1:.1%} ({time.time()-t0:.0f}s)")

    # Phase 2: Fashion-MNIST
    print("\n  Phase 2: Fashion-MNIST (online, 1 epoch)...")
    t0 = time.time()
    accs_fmnist = []
    perm = torch.randperm(len(X_tr_f), device=DEVICE)
    for i in range(0, len(X_tr_f), batch_size):
        idx = perm[i:i+batch_size]
        acc = model.train_step(X_tr_f[idx], y_tr_f[idx])
        accs_fmnist.append(acc)
        if (i // batch_size) % 100 == 0:
            m_acc = evaluate(model, X_te_m, y_te_m)
            f_acc = evaluate(model, X_te_f, y_te_f)
            print(f"    Step {i//batch_size}: MNIST={m_acc:.1%} FMNIST={f_acc:.1%} "
                  f"experts={model.n_active}")

    mnist_after_p2 = evaluate(model, X_te_m, y_te_m)
    fmnist_after_p2 = evaluate(model, X_te_f, y_te_f)
    forgetting = (mnist_after_p1 - mnist_after_p2) * 100
    print(f"  → MNIST after Phase 2: {mnist_after_p2:.1%}")
    print(f"  → FMNIST after Phase 2: {fmnist_after_p2:.1%}")
    print(f"  → Forgetting: {forgetting:.1f}pp")

    # Deploy mode (only 14-level oxide weights)
    model.deploy_mode()
    mnist_deployed = evaluate(model, X_te_m, y_te_m)
    fmnist_deployed = evaluate(model, X_te_f, y_te_f)
    print(f"  → Deployed (14-level): MNIST={mnist_deployed:.1%} FMNIST={fmnist_deployed:.1%}")

    results['PIMOL'] = {
        'mnist_p1': mnist_after_p1, 'mnist_p2': mnist_after_p2,
        'fmnist_p2': fmnist_after_p2, 'forgetting': forgetting,
        'mnist_deployed': mnist_deployed, 'fmnist_deployed': fmnist_deployed,
        'n_experts': model.n_active,
    }

    # ── Baseline: DFA (from our earlier work) ──
    print(f"\n{'='*60}")
    print("  Baseline: DFA (no modularity, no dual-timescale)")
    print(f"{'='*60}")

    rng = np.random.RandomState(42)

    class SimpleDFA:
        def __init__(self):
            self.W1 = torch.tensor(rng.randn(500, 784).astype(np.float32) * 0.1 / np.sqrt(784), device=DEVICE)
            self.b1 = torch.zeros(500, device=DEVICE)
            self.W2 = torch.tensor(rng.randn(10, 500).astype(np.float32) * 0.1 / np.sqrt(500), device=DEVICE)
            self.b2 = torch.zeros(10, device=DEVICE)
            self.B_fb = torch.tensor(rng.randn(500, 10).astype(np.float32) * 0.05, device=DEVICE)
        def forward(self, x):
            h = torch.relu(x @ self.W1.T + self.b1)
            return h @ self.W2.T + self.b2, h
        def train_step(self, x, y, lr=0.01):
            logits, h = self.forward(x)
            tgt = torch.zeros(x.shape[0], 10, device=DEVICE)
            tgt.scatter_(1, y.unsqueeze(1), 1.0)
            err = F.softmax(logits, 1) - tgt
            self.W2 -= lr * (err.T @ h) / x.shape[0]
            self.b2 -= lr * err.mean(0)
            le = err @ self.B_fb.T * (h > 0).float()
            self.W1 -= lr * (le.T @ x) / x.shape[0]
            self.b1 -= lr * le.mean(0)
        def predict(self, x):
            logits, _ = self.forward(x)
            return logits.argmax(1)

    dfa = SimpleDFA()

    # Phase 1
    perm = torch.randperm(len(X_tr_m), device=DEVICE)
    for i in range(0, len(X_tr_m), 512):
        dfa.train_step(X_tr_m[perm[i:i+512]], y_tr_m[perm[i:i+512]])
    dfa_mnist_p1 = (dfa.predict(X_te_m) == y_te_m).float().mean().item()

    # Phase 2
    perm = torch.randperm(len(X_tr_f), device=DEVICE)
    for i in range(0, len(X_tr_f), 512):
        dfa.train_step(X_tr_f[perm[i:i+512]], y_tr_f[perm[i:i+512]])
    dfa_mnist_p2 = (dfa.predict(X_te_m) == y_te_m).float().mean().item()
    dfa_fmnist_p2 = (dfa.predict(X_te_f) == y_te_f).float().mean().item()
    dfa_forget = (dfa_mnist_p1 - dfa_mnist_p2) * 100

    print(f"  DFA: MNIST_p1={dfa_mnist_p1:.1%} → MNIST_p2={dfa_mnist_p2:.1%} "
          f"(forgetting: {dfa_forget:.1f}pp) FMNIST={dfa_fmnist_p2:.1%}")

    results['DFA'] = {
        'mnist_p1': dfa_mnist_p1, 'mnist_p2': dfa_mnist_p2,
        'fmnist_p2': dfa_fmnist_p2, 'forgetting': dfa_forget,
    }

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  PIMOL vs DFA — Continual Learning Results")
    print(f"{'='*60}")
    print(f"  {'Method':<15s} {'MNIST_p1':>10s} {'MNIST_p2':>10s} {'Forget':>10s} {'FMNIST':>10s}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        print(f"  {name:<15s} {r['mnist_p1']:>9.1%} {r['mnist_p2']:>9.1%} "
              f"{r['forgetting']:>9.1f}pp {r['fmnist_p2']:>9.1%}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
    fig.suptitle('PIMOL: Physics-Informed Modular Online Learner',
                 fontsize=14, fontweight='bold', color='white')

    # Training curves
    ax = axes[0]; ax.set_facecolor('#0d1117')
    window = 50
    if len(accs_mnist) > window:
        ma_m = [np.mean(accs_mnist[max(0,i-window):i+1]) for i in range(len(accs_mnist))]
        ax.plot(ma_m, color='#4CAF50', linewidth=1.5, label='MNIST')
    if len(accs_fmnist) > window:
        ma_f = [np.mean(accs_fmnist[max(0,i-window):i+1]) for i in range(len(accs_fmnist))]
        offset = len(accs_mnist)
        ax.plot(range(offset, offset + len(ma_f)), ma_f, color='#2196F3', linewidth=1.5, label='FMNIST')
    ax.axvline(len(accs_mnist), color='gray', linestyle='--', alpha=0.5, label='Task switch')
    ax.set_xlabel('Training step', color='white')
    ax.set_ylabel('Batch accuracy (50-step avg)', color='white')
    ax.set_title('Online Training Curve', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15)

    # Forgetting comparison
    ax = axes[1]; ax.set_facecolor('#0d1117')
    methods = list(results.keys())
    forgets = [results[m]['forgetting'] for m in methods]
    colors = ['#4ecdc4', '#F44336']
    ax.bar(range(len(methods)), forgets, color=colors[:len(methods)], alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, color='white')
    ax.set_ylabel('Forgetting (pp)', color='white')
    ax.set_title('Catastrophic Forgetting', color='white')
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15, axis='y')

    # Final accuracy comparison
    ax = axes[2]; ax.set_facecolor('#0d1117')
    x_pos = np.arange(len(methods))
    w = 0.35
    mnist_vals = [results[m]['mnist_p2'] * 100 for m in methods]
    fmnist_vals = [results[m]['fmnist_p2'] * 100 for m in methods]
    ax.bar(x_pos - w/2, mnist_vals, w, color='#4CAF50', alpha=0.85, label='MNIST')
    ax.bar(x_pos + w/2, fmnist_vals, w, color='#2196F3', alpha=0.85, label='FMNIST')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, color='white')
    ax.set_ylabel('Accuracy (%)', color='white')
    ax.set_title('After Both Tasks', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'pimol_results.png'), dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"\n  Saved: {os.path.join(OUT, 'pimol_results.png')}")

    # Save raw results
    with open(os.path.join(OUT, 'pimol_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
