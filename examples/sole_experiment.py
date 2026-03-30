#!/usr/bin/env python3
"""SOLE: Surprise-Orthogonal Learning with self-distillation and Experts

Combines 5 cutting-edge components that nobody has assembled together:
  1. Surprise gating (Titans-style): only learn from surprising inputs
  2. SVD orthogonal projection (Adaptive SVD): protect critical subspace
  3. MoE experts: task-specific modules with routing
  4. Self-distillation (SDFT): model teaches itself to not forget
  5. Fast/slow consolidation (Hope-style): multi-timescale weights

Tests on 3 sequential tasks: MNIST → Fashion-MNIST → KMNIST
"""

import sys, os, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════
# SHARED NETWORK
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, ni=784, nh=500, nc=10):
        super().__init__()
        self.fc1 = nn.Linear(ni, nh)
        self.fc2 = nn.Linear(nh, nh)
        self.fc3 = nn.Linear(nh, nc)
        self.ln1 = nn.LayerNorm(nh)
        self.ln2 = nn.LayerNorm(nh)

    def forward(self, x):
        h = self.ln1(F.relu(self.fc1(x)))
        h = self.ln2(F.relu(self.fc2(h)))
        return self.fc3(h)


# ═══════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════

def train_finetune(tasks, test_data, n_epochs=1, lr=0.001):
    """Baseline: plain fine-tuning (catastrophic forgetting expected)."""
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for task_idx, (X_tr, y_tr, name) in enumerate(tasks):
        for ep in range(n_epochs):
            perm = torch.randperm(len(X_tr), device=DEVICE)
            for i in range(0, len(X_tr), 128):
                logits = model(X_tr[perm[i:i+128]])
                loss = F.cross_entropy(logits, y_tr[perm[i:i+128]])
                opt.zero_grad(); loss.backward(); opt.step()

        # Evaluate on all tasks seen so far
        accs = {}
        for X_te, y_te, tname in test_data:
            with torch.no_grad():
                accs[tname] = (model(X_te).argmax(1) == y_te).float().mean().item()
        history.append(accs)
        print(f'    After {name}: {" | ".join(f"{k}={v:.1%}" for k,v in accs.items())}')

    return history


def train_ewc(tasks, test_data, n_epochs=1, lr=0.001, ewc_lambda=5000):
    """EWC baseline (our proven gold standard)."""
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fisher = {}
    old_params = {}
    history = []

    for task_idx, (X_tr, y_tr, name) in enumerate(tasks):
        for ep in range(n_epochs):
            perm = torch.randperm(len(X_tr), device=DEVICE)
            for i in range(0, len(X_tr), 128):
                logits = model(X_tr[perm[i:i+128]])
                loss = F.cross_entropy(logits, y_tr[perm[i:i+128]])
                # EWC penalty
                for n, p in model.named_parameters():
                    if n in fisher:
                        loss += ewc_lambda * (fisher[n] * (p - old_params[n]) ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()

        # Compute Fisher for this task
        fisher_new = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        idx = torch.randperm(len(X_tr))[:3000]
        for i in idx:
            model.zero_grad()
            loss = F.cross_entropy(model(X_tr[i:i+1]), y_tr[i:i+1])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher_new[n] += p.grad.data ** 2 / 3000
        # Accumulate Fisher across tasks
        for n in fisher_new:
            fisher[n] = fisher.get(n, torch.zeros_like(fisher_new[n])) + fisher_new[n]
        old_params = {n: p.clone().detach() for n, p in model.named_parameters()}

        accs = {}
        for X_te, y_te, tname in test_data:
            with torch.no_grad():
                accs[tname] = (model(X_te).argmax(1) == y_te).float().mean().item()
        history.append(accs)
        print(f'    After {name}: {" | ".join(f"{k}={v:.1%}" for k,v in accs.items())}')

    return history


# ═══════════════════════════════════════════════════════════════════
# SOLE: The Novel Combination
# ═══════════════════════════════════════════════════════════════════

class SOLE:
    """Surprise-Orthogonal Learning with self-distillation and Experts."""

    def __init__(self, ni=784, nh=500, nc=10, n_experts=3,
                 surprise_threshold=0.5, svd_protect_ratio=0.7,
                 distill_weight=1.0, ema_decay=0.999):
        self.nc = nc
        self.surprise_threshold = surprise_threshold
        self.svd_protect_ratio = svd_protect_ratio
        self.distill_weight = distill_weight
        self.ema_decay = ema_decay

        # Expert pool
        self.experts = nn.ModuleList([MLP(ni, nh, nc) for _ in range(n_experts)]).to(DEVICE)
        self.router = nn.Linear(ni, n_experts).to(DEVICE)
        self.n_active = 1  # Start with 1 expert

        # Slow model (EMA of fast model — for stability)
        self.slow_model = copy.deepcopy(self.experts[0])
        for p in self.slow_model.parameters():
            p.requires_grad = False

        # Optimizer for all experts + router
        self.opt = torch.optim.Adam(
            list(self.experts.parameters()) + list(self.router.parameters()),
            lr=0.001)

        # SVD projection state (computed after each task)
        self.critical_subspaces = {}  # name → U_critical for each layer

        # Surprise predictor: running mean of loss
        self.loss_ema = 2.0  # Initial high loss (everything is surprising)
        self.loss_alpha = 0.01

        self.step_count = 0
        self.task_count = 0

    def forward(self, x):
        """Route input to top-1 expert, return logits."""
        # Router
        gate = F.softmax(self.router(x), dim=1)[:, :self.n_active]
        expert_idx = gate.argmax(dim=1)

        # Gather predictions from selected expert (batched)
        logits = torch.zeros(x.shape[0], self.nc, device=DEVICE)
        for eidx in range(self.n_active):
            mask = expert_idx == eidx
            if mask.any():
                logits[mask] = self.experts[eidx](x[mask])
        return logits, expert_idx

    def compute_surprise(self, x, y):
        """Surprise = how much worse is the loss than expected?"""
        with torch.no_grad():
            logits, _ = self.forward(x)
            loss = F.cross_entropy(logits, y).item()
        surprise = max(0, loss - self.loss_ema)
        self.loss_ema = self.loss_alpha * loss + (1 - self.loss_alpha) * self.loss_ema
        return surprise

    def project_orthogonal(self):
        """Project gradients orthogonal to critical subspaces (SVD protection)."""
        if not self.critical_subspaces:
            return

        for name, param in self.experts.named_parameters():
            if param.grad is not None and name in self.critical_subspaces:
                U_crit = self.critical_subspaces[name]  # (d, k) critical directions
                grad = param.grad.data
                if grad.dim() == 2 and U_crit.shape[0] == grad.shape[0]:
                    # Project out the critical component
                    proj = U_crit @ (U_crit.T @ grad)
                    param.grad.data = grad - proj

    def self_distill(self, x, logits_before):
        """Minimize KL divergence from pre-update predictions."""
        logits_after, _ = self.forward(x)
        kl = F.kl_div(
            F.log_softmax(logits_after, dim=1),
            F.softmax(logits_before, dim=1),
            reduction='batchmean'
        )
        return self.distill_weight * kl

    def update_slow(self):
        """EMA update: slow model ← fast model."""
        for p_slow, p_fast in zip(self.slow_model.parameters(),
                                   self.experts[0].parameters()):
            p_slow.data.mul_(self.ema_decay).add_(p_fast.data, alpha=1-self.ema_decay)

    def train_step(self, x, y):
        """One SOLE training step combining all 5 components."""
        self.step_count += 1

        # 1. SURPRISE: Should we learn from this batch?
        surprise = self.compute_surprise(x, y)
        if surprise < self.surprise_threshold * 0.1:
            # Not surprising enough — skip (already know this)
            return 0.0, 'skip'

        # Scale learning rate by surprise (more surprise → learn harder)
        surprise_scale = min(2.0, 1.0 + surprise)

        # 2. Cache pre-update logits for self-distillation
        with torch.no_grad():
            logits_before = torch.zeros(x.shape[0], self.nc, device=DEVICE)
            for eidx in range(self.n_active):
                logits_before += self.experts[eidx](x) / self.n_active

        # 3. Forward + task loss
        logits, expert_idx = self.forward(x)
        task_loss = F.cross_entropy(logits, y)

        # 4. Self-distillation loss (don't forget yourself)
        distill_loss = self.self_distill(x, logits_before.detach())

        # Total loss
        loss = task_loss + distill_loss

        # Backward
        self.opt.zero_grad()
        loss.backward()

        # 5. SVD orthogonal projection (protect critical subspace)
        self.project_orthogonal()

        # Scale gradients by surprise
        for p in self.experts.parameters():
            if p.grad is not None:
                p.grad.data *= surprise_scale

        self.opt.step()

        # 6. Update slow model (EMA consolidation)
        self.update_slow()

        acc = (logits.argmax(1) == y).float().mean().item()
        return acc, f'surp={surprise:.2f}'

    def end_task(self, X_tr, y_tr, task_name):
        """After finishing a task: compute critical subspaces for protection."""
        print(f'    Computing SVD protection for {task_name}...')

        # Compute per-parameter Fisher (importance)
        fisher = {}
        model = self.experts[0]  # Use first expert for now
        idx = torch.randperm(len(X_tr))[:3000]
        for n, p in model.named_parameters():
            fisher[n] = torch.zeros_like(p)

        for i in idx:
            model.zero_grad()
            loss = F.cross_entropy(model(X_tr[i:i+1]), y_tr[i:i+1])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / 3000

        # SVD: find critical directions for weight matrices
        for name, param in model.named_parameters():
            full_name = f'0.{name}'  # Expert 0
            if param.dim() == 2:  # Weight matrices only
                F_mat = fisher[name]
                # Weight importance = element-wise Fisher
                # SVD of Fisher-weighted weight matrix
                W_imp = param.data * torch.sqrt(F_mat + 1e-10)
                try:
                    U, S, Vh = torch.linalg.svd(W_imp, full_matrices=False)
                    # Keep top-k directions as critical (protect these)
                    k = max(1, int(S.shape[0] * (1 - self.svd_protect_ratio)))
                    k = min(k, S.shape[0])
                    self.critical_subspaces[full_name] = U[:, :k].detach()
                except:
                    pass  # SVD can fail on degenerate matrices

        self.task_count += 1

        # Optionally activate a new expert for the next task
        if self.task_count < len(self.experts):
            self.n_active = min(self.task_count + 1, len(self.experts))
            print(f'    Activated expert {self.n_active-1} for next task')

    def predict(self, x):
        with torch.no_grad():
            # Ensemble all active experts
            logits = torch.zeros(x.shape[0], self.nc, device=DEVICE)
            for eidx in range(self.n_active):
                logits += self.experts[eidx](x)
            return logits.argmax(1)


def train_sole(tasks, test_data, n_epochs=1, **kwargs):
    """Train SOLE on sequential tasks."""
    model = SOLE(**kwargs)
    history = []

    for task_idx, (X_tr, y_tr, name) in enumerate(tasks):
        for ep in range(n_epochs):
            perm = torch.randperm(len(X_tr), device=DEVICE)
            n_skip = 0
            for i in range(0, len(X_tr), 128):
                acc, info = model.train_step(X_tr[perm[i:i+128]], y_tr[perm[i:i+128]])
                if 'skip' in str(info):
                    n_skip += 1

        # End of task: compute protection
        model.end_task(X_tr, y_tr, name)

        accs = {}
        for X_te, y_te, tname in test_data:
            accs[tname] = (model.predict(X_te) == y_te).float().mean().item()
        history.append(accs)
        skip_pct = n_skip / (len(X_tr) // 128) * 100
        print(f'    After {name}: {" | ".join(f"{k}={v:.1%}" for k,v in accs.items())} '
              f'(skipped {skip_pct:.0f}%)')

    return history


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    from torchvision import datasets, transforms

    # Load 3 tasks
    def load(cls, n_tr=30000, n_te=5000):
        tr = cls('data', True, download=True, transform=transforms.ToTensor())
        te = cls('data', False, download=True, transform=transforms.ToTensor())
        return (tr.data[:n_tr].float().reshape(-1, 784).to(DEVICE) / 255,
                tr.targets[:n_tr].to(DEVICE),
                te.data[:n_te].float().reshape(-1, 784).to(DEVICE) / 255,
                te.targets[:n_te].to(DEVICE))

    X_trm, y_trm, X_tem, y_tem = load(datasets.MNIST)
    X_trf, y_trf, X_tef, y_tef = load(datasets.FashionMNIST)
    # Task 3: Permuted MNIST (random pixel shuffle — different distribution, same labels)
    perm_idx = torch.randperm(784)
    X_trp = X_trm[:, perm_idx]
    y_trp = y_trm.clone()
    X_tep = X_tem[:, perm_idx]
    y_tep = y_tem.clone()

    tasks = [(X_trm, y_trm, 'MNIST'), (X_trf, y_trf, 'FMNIST'), (X_trp, y_trp, 'PermMNIST')]
    test_data = [(X_tem, y_tem, 'MNIST'), (X_tef, y_tef, 'FMNIST'), (X_tep, y_tep, 'PermMNIST')]

    results = {}

    # ── Baselines ──
    print(f"\n{'='*60}")
    print("  1. Fine-tune (catastrophic forgetting)")
    print(f"{'='*60}")
    results['Finetune'] = train_finetune(tasks, test_data)

    print(f"\n{'='*60}")
    print("  2. EWC (Fisher regularization)")
    print(f"{'='*60}")
    results['EWC'] = train_ewc(tasks, test_data, ewc_lambda=5000)

    # ── SOLE ──
    print(f"\n{'='*60}")
    print("  3. SOLE (Surprise + Orthogonal + self-Distill + Experts)")
    print(f"{'='*60}")
    results['SOLE'] = train_sole(tasks, test_data,
                                  n_experts=3, surprise_threshold=0.3,
                                  svd_protect_ratio=0.5, distill_weight=2.0)

    # ── SOLE without surprise gating (ablation) ──
    print(f"\n{'='*60}")
    print("  4. SOLE no-surprise (ablation: learn from everything)")
    print(f"{'='*60}")
    results['SOLE-noSurp'] = train_sole(tasks, test_data,
                                         n_experts=3, surprise_threshold=0.0,
                                         svd_protect_ratio=0.5, distill_weight=2.0)

    # ── Summary ──
    print(f"\n{'='*65}")
    print("  SEQUENTIAL CONTINUAL LEARNING: MNIST → FMNIST → KMNIST")
    print(f"{'='*65}")
    print(f"  {'Method':<18s} {'M_after_3':>8s} {'F_after_3':>8s} {'K_after_3':>8s} {'Joint':>8s} {'M_forget':>8s}")
    print(f"  {'-'*58}")

    for name, hist in results.items():
        final = hist[-1]  # After all 3 tasks
        m = final.get('MNIST', 0)
        f = final.get('FMNIST', 0)
        k = final.get('KMNIST', 0)
        joint = (m + f + k) / 3
        # Forgetting: MNIST accuracy after task 1 vs after task 3
        m_after_1 = hist[0].get('MNIST', 0)
        forget = (m_after_1 - m) * 100
        print(f'  {name:<18s} {m:>7.1%} {f:>7.1%} {k:>7.1%} {joint:>7.1%} {forget:>7.1f}pp')

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
    fig.suptitle('SOLE: Surprise-Orthogonal Learning — 3 Sequential Tasks',
                 fontsize=14, fontweight='bold', color='white')

    task_names = ['MNIST', 'FMNIST', 'PermMNIST']
    colors = {'Finetune': '#F44336', 'EWC': '#4CAF50', 'SOLE': '#2196F3', 'SOLE-noSurp': '#FF9800'}

    for i, tn in enumerate(task_names):
        ax = axes[i]; ax.set_facecolor('#0d1117')
        for method, hist in results.items():
            vals = [h.get(tn, 0) * 100 for h in hist]
            ax.plot(range(len(vals)), vals, 'o-', label=method, color=colors.get(method, '#888'),
                    linewidth=2, markersize=6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['After T1', 'After T2', 'After T3'], color='white', fontsize=8)
        ax.set_ylabel('Accuracy (%)', color='white')
        ax.set_title(f'{tn} Retention', color='white', fontsize=12)
        ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15)
        if i == 0:
            ax.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'sole_results.png'), dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"\n  Saved: {os.path.join(OUT, 'sole_results.png')}")


if __name__ == '__main__':
    main()
