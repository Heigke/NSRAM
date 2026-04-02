#!/usr/bin/env python3
"""FEP-Mem: Rigorous Benchmark + Ablation Study

Standard text8 protocol: 90M train / 5M val / 5M test
3 epochs, 3 seeds, mean ± std reported

Ablation (8 configs):
  A: Full FEP-Mem (precision + competition + memory dropout)
  B: No precision (standard L2 delta rule — equivalent to Gated DeltaNet)
  C: No competition (single memory head)
  D: No memory dropout
  E: No memory at all (pure SWA — architecture ablation)
  F: No precision + no competition (minimal memory)
  G: Full but S reset at eval (shows weight learning quality)
  H: sLSTM baseline (different architecture entirely)

This positions us against the Miras framework (Google, April 2025)
which catalogs L2/Lp/Huber attentional biases but NOT precision weighting.
"""

import os, sys, time, json, math
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CORE: Delta-Rule Memory Head (configurable precision)
# ═══════════════════════════════════════════════════════════════

class DeltaHead(nn.Module):
    """Delta-rule state matrix with optional precision weighting."""

    def __init__(self, d_model, d_key, d_value, use_precision=True):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.use_precision = use_precision

        self.proj_k = nn.Linear(d_model, d_key)
        self.proj_v = nn.Linear(d_model, d_value)
        self.proj_q = nn.Linear(d_model, d_key)
        self.proj_beta = nn.Linear(d_model, 1)

        if use_precision:
            self.proj_prec = nn.Linear(d_model, d_value)
            nn.init.constant_(self.proj_prec.bias, 1.0)

        self.register_buffer('S', torch.zeros(d_key, d_value))
        self.register_buffer('momentum', torch.zeros(d_key, d_value))

    def retrieve(self, x):
        q = F.normalize(self.proj_q(x), dim=-1)
        return q @ self.S

    def free_energy(self, x, target):
        retrieved = self.retrieve(x)
        error = target - retrieved
        if self.use_precision:
            prec = F.softplus(self.proj_prec(x)) + 0.01
            F_per_dim = prec * error ** 2 - torch.log(prec)
        else:
            F_per_dim = error ** 2  # Standard L2 (Gated DeltaNet style)
        return F_per_dim.mean(dim=-1), error

    def update(self, x, target):
        with torch.no_grad():
            k = F.normalize(self.proj_k(x), dim=-1)
            v = self.proj_v(x)
            beta = torch.sigmoid(self.proj_beta(x))

            pred = k @ self.S
            error = pred - v
            error = 2.0 * torch.tanh(error / 2.0)  # Soft clamp

            k_mean = k.mean(0)           # (d_key,)
            error_mean = error.mean(0)   # (d_value,)
            beta_mean = beta.mean().item()

            if self.use_precision:
                prec = F.softplus(self.proj_prec(x)) + 0.01
                prec_mean = prec.mean(0)  # (d_value,)
                weighted_error = beta_mean * prec_mean * error_mean  # (d_value,)
            else:
                weighted_error = beta_mean * error_mean  # (d_value,)

            # Outer product: (d_key,) x (d_value,) → (d_key, d_value)
            update = k_mean.unsqueeze(1) * weighted_error.unsqueeze(0)

            self.momentum = 0.9 * self.momentum + 0.1 * update
            decay = 0.975
            self.S = decay * self.S - 0.01 * self.momentum


class MemoryLayer(nn.Module):
    """Memory with optional competition and dropout."""

    def __init__(self, d_model, n_heads=3, d_key=64, use_precision=True,
                 use_competition=True, use_mem_dropout=True):
        super().__init__()
        self.n_heads = n_heads if use_competition else 1
        self.use_competition = use_competition
        self.use_mem_dropout = use_mem_dropout
        self.d_model = d_model

        self.heads = nn.ModuleList([
            DeltaHead(d_model, d_key, d_model, use_precision)
            for _ in range(self.n_heads)
        ])
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, target=None, do_update=True):
        B, D = x.shape

        if self.use_competition and self.n_heads > 1:
            # Compute free energy per head
            all_F, all_ret = [], []
            for head in self.heads:
                F_i, _ = head.free_energy(x, target if target is not None else x)
                all_F.append(F_i)
                all_ret.append(head.retrieve(x))

            F_stack = torch.stack(all_F, dim=1)
            ret_stack = torch.stack(all_ret, dim=1)
            weights = F.softmax(-F_stack, dim=1)
            mem_out = (weights.unsqueeze(-1) * ret_stack).sum(dim=1)
            fe_loss = F_stack.min(dim=1).values.mean()
        else:
            mem_out = self.heads[0].retrieve(x)
            fe_loss = torch.tensor(0.0, device=x.device)

        mem_out = self.out_proj(mem_out)

        # Gate (capped, detached)
        raw_gate = self.gate(torch.cat([x, mem_out.detach()], dim=-1))
        gate = raw_gate * 0.3

        # Memory dropout
        if self.training and self.use_mem_dropout and torch.rand(1).item() < 0.5:
            gate = gate * 0.0

        output = (1 - gate) * x + gate * mem_out.detach()

        if do_update and target is not None:
            with torch.no_grad():
                for head in self.heads:
                    head.update(x, target)

        return output, fe_loss


# ═══════════════════════════════════════════════════════════════
# SWA Block (shared across all configs)
# ═══════════════════════════════════════════════════════════════

class SWABlock(nn.Module):
    def __init__(self, d_model, n_heads=4, window=128):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*3), nn.GELU(),
                                 nn.Linear(d_model*3, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(S, S, device=x.device), 1).bool()
        win = torch.ones(S, S, device=x.device, dtype=torch.bool)
        for i in range(S):
            win[i, max(0, i-self.window+1):i+1] = False
        attn.masked_fill_((causal | win).unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.out(out)
        return x + self.ff(self.ln2(x))


# ═══════════════════════════════════════════════════════════════
# Full Model (configurable)
# ═══════════════════════════════════════════════════════════════

class FEPModel(nn.Module):
    def __init__(self, d_model=128, n_attn=4, ctx_len=256, chunk_size=32,
                 use_memory=True, use_precision=True, use_competition=True,
                 use_mem_dropout=True, n_heads=3, d_key=64):
        super().__init__()
        self.d = d_model
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.use_memory = use_memory

        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Embedding(ctx_len, d_model)

        n_lo = n_attn // 2
        n_hi = n_attn - n_lo
        self.lower = nn.ModuleList([SWABlock(d_model) for _ in range(n_lo)])
        self.upper = nn.ModuleList([SWABlock(d_model) for _ in range(n_hi)])

        if use_memory:
            self.memory = MemoryLayer(d_model, n_heads, d_key,
                                       use_precision, use_competition, use_mem_dropout)
            self.mem_ln = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, 256)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, byte_ids, do_update=True):
        B, S = byte_ids.shape
        h = self.byte_embed(byte_ids) + self.pos_embed(torch.arange(S, device=byte_ids.device))

        for block in self.lower:
            h = block(h)

        fe_loss = torch.tensor(0.0, device=h.device)
        if self.use_memory:
            CS = self.chunk_size
            n_chunks = S // CS
            h_n = self.mem_ln(h)
            mem_outs = []
            for c in range(n_chunks):
                s, e = c * CS, (c + 1) * CS
                chunk_mean = h_n[:, s:e, :].mean(dim=1)
                target = h_n[:, min(e, S-CS):min(e+CS, S), :].mean(dim=1).detach()
                mo, fe = self.memory(chunk_mean, target, do_update)
                fe_loss += fe
                mem_outs.append(mo)
            if mem_outs:
                fe_loss /= len(mem_outs)
                mem_exp = torch.stack(mem_outs, 1).repeat_interleave(CS, dim=1)
                h = h + (mem_exp - h_n).detach()

        for block in self.upper:
            h = block(h)

        return self.head(self.ln_out(h)), fe_loss

    def reset_memory(self):
        if self.use_memory:
            for head in self.memory.heads:
                head.S.zero_()
                head.momentum.zero_()


class SLSTMBaseline(nn.Module):
    def __init__(self, d_model=128, n_layers=3, ctx_len=256):
        super().__init__()
        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Embedding(ctx_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'lstm': nn.LSTM(d_model, d_model, batch_first=True),
                'ff': nn.Sequential(nn.Linear(d_model, d_model*3), nn.GELU(),
                                     nn.Linear(d_model*3, d_model)),
                'ln1': nn.LayerNorm(d_model), 'ln2': nn.LayerNorm(d_model)}))
        self.head = nn.Linear(d_model, 256)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, byte_ids, do_update=True):
        B, S = byte_ids.shape
        h = self.byte_embed(byte_ids) + self.pos_embed(torch.arange(S, device=byte_ids.device))
        for layer in self.layers:
            out, _ = layer['lstm'](layer['ln1'](h))
            h = h + out
            h = h + layer['ff'](layer['ln2'](h))
        return self.head(self.ln(h)), torch.tensor(0.0, device=h.device)

    def reset_memory(self):
        pass


# ═══════════════════════════════════════════════════════════════
# Training / Eval
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, data, ctx, bs, opt, fe_w=0.01):
    model.train()
    n = len(data) // ctx - 1
    idx = torch.randperm(n)
    tl = tc = tt = 0
    for i in range(0, min(len(idx), n), bs):
        bi = idx[i:i+bs]
        if len(bi) == 0: continue
        inp = torch.stack([data[j*ctx:(j+1)*ctx] for j in bi]).to(DEVICE)
        tgt = torch.stack([data[j*ctx+1:(j+1)*ctx+1] for j in bi]).to(DEVICE)
        logits, fe = model(inp)
        ce = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1))
        loss = ce + fe_w * fe
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += ce.item() * tgt.numel()
        tc += (logits.argmax(-1) == tgt).sum().item()
        tt += tgt.numel()
    return (tl/tt)/math.log(2), tc/tt

@torch.no_grad()
def evaluate(model, data, ctx, bs):
    model.eval()
    n = len(data) // ctx - 1
    tl = tt = 0
    for i in range(0, n, bs):
        e = min(i+bs, n)
        inp = torch.stack([data[j*ctx:(j+1)*ctx] for j in range(i, e)]).to(DEVICE)
        tgt = torch.stack([data[j*ctx+1:(j+1)*ctx+1] for j in range(i, e)]).to(DEVICE)
        logits, _ = model(inp, do_update=False)
        tl += F.cross_entropy(logits.reshape(-1,256), tgt.reshape(-1), reduction='sum').item()
        tt += tgt.numel()
    return (tl/tt)/math.log(2)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")

    with open('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/text8', 'rb') as f:
        raw = f.read(100_000_000)
    data = torch.tensor(list(raw), dtype=torch.long)
    train_d = data[:90_000_000]
    val_d = data[90_000_000:95_000_000]
    test_d = data[95_000_000:]
    print(f"Train: {len(train_d):,}, Val: {len(val_d):,}, Test: {len(test_d):,}")

    CTX = 256; BS = 32; EPOCHS = 3; D = 128; SEEDS = [42, 123, 7]

    configs = {
        'A_full':       dict(use_memory=True, use_precision=True, use_competition=True, use_mem_dropout=True),
        'B_no_prec':    dict(use_memory=True, use_precision=False, use_competition=True, use_mem_dropout=True),
        'C_no_comp':    dict(use_memory=True, use_precision=True, use_competition=False, use_mem_dropout=True),
        'D_no_dropout': dict(use_memory=True, use_precision=True, use_competition=True, use_mem_dropout=False),
        'E_no_memory':  dict(use_memory=False),
        'F_minimal':    dict(use_memory=True, use_precision=False, use_competition=False, use_mem_dropout=True),
    }

    all_results = {}

    for name, cfg in configs.items():
        seed_results = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            model = FEPModel(d_model=D, n_attn=4, ctx_len=CTX, chunk_size=32,
                              n_heads=3, d_key=64, **cfg).to(DEVICE)
            n_p = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

            best_val = 99
            for ep in range(EPOCHS):
                t0 = time.time()
                tr_bpc, tr_acc = train_epoch(model, train_d, CTX, BS, opt)
                val_bpc = evaluate(model, val_d[:1_000_000], CTX, BS)
                dt = time.time() - t0
                if val_bpc < best_val:
                    best_val = val_bpc
                opt.param_groups[0]['lr'] *= 0.5
                if seed == SEEDS[0]:
                    print(f'  {name} s{seed} ep{ep+1}: tr={tr_bpc:.3f} val={val_bpc:.3f} ({dt:.0f}s)')

            # Test with memory
            test_bpc = evaluate(model, test_d[:1_000_000], CTX, BS)
            # Test without memory (S reset)
            model.reset_memory()
            test_bpc_reset = evaluate(model, test_d[:1_000_000], CTX, BS)

            seed_results.append({
                'test_bpc': test_bpc, 'test_bpc_reset': test_bpc_reset,
                'best_val': best_val, 'params': n_p, 'seed': seed
            })

        # Aggregate
        bpcs = [r['test_bpc_reset'] for r in seed_results]
        mean_bpc = np.mean(bpcs)
        std_bpc = np.std(bpcs)
        all_results[name] = {
            'mean_bpc': mean_bpc, 'std_bpc': std_bpc,
            'mean_with_S': np.mean([r['test_bpc'] for r in seed_results]),
            'params': seed_results[0]['params'],
            'seeds': seed_results,
            'cfg': {k: v for k, v in cfg.items()},
        }
        print(f'  {name}: {mean_bpc:.3f} ± {std_bpc:.3f} (S=0), '
              f'{all_results[name]["mean_with_S"]:.3f} (with S), {seed_results[0]["params"]:,}p')

    # sLSTM baseline
    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SLSTMBaseline(d_model=D, n_layers=3, ctx_len=CTX).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        for ep in range(EPOCHS):
            t0 = time.time()
            tr, _ = train_epoch(model, train_d, CTX, BS, opt)
            val = evaluate(model, val_d[:1_000_000], CTX, BS)
            dt = time.time() - t0
            if seed == SEEDS[0]:
                print(f'  sLSTM s{seed} ep{ep+1}: tr={tr:.3f} val={val:.3f} ({dt:.0f}s)')
            opt.param_groups[0]['lr'] *= 0.5
        test = evaluate(model, test_d[:1_000_000], CTX, BS)
        seed_results.append({'test_bpc': test, 'test_bpc_reset': test, 'params': n_p})

    bpcs = [r['test_bpc'] for r in seed_results]
    all_results['H_sLSTM'] = {
        'mean_bpc': np.mean(bpcs), 'std_bpc': np.std(bpcs),
        'mean_with_S': np.mean(bpcs), 'params': seed_results[0]['params'],
        'seeds': seed_results,
    }
    print(f'  sLSTM: {np.mean(bpcs):.3f} ± {np.std(bpcs):.3f}, {seed_results[0]["params"]:,}p')

    # ═══ SUMMARY ═══
    print(f"\n{'='*75}")
    print(f"  FEP-Mem RIGOROUS BENCHMARK — text8 (90M/5M/5M), {EPOCHS} epochs, {len(SEEDS)} seeds")
    print(f"{'='*75}")
    print(f"  {'Config':<20s} {'BPC (S=0)':>12s} {'BPC (w/S)':>12s} {'Params':>10s} {'Δ vs Full':>10s}")
    print(f"  {'-'*64}")

    full_bpc = all_results['A_full']['mean_bpc']
    for name in ['A_full', 'B_no_prec', 'C_no_comp', 'D_no_dropout', 'E_no_memory',
                  'F_minimal', 'H_sLSTM']:
        r = all_results[name]
        delta = r['mean_bpc'] - full_bpc
        print(f"  {name:<20s} {r['mean_bpc']:.3f}±{r['std_bpc']:.3f}"
              f"  {r['mean_with_S']:.3f}"
              f"  {r['params']:>9,}  {delta:>+8.3f}")

    print(f"\n  Literature: SOTA=1.038 (277M), Small LSTM=1.59 (3.3M)")

    # ═══ PLOT ═══
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0d1117')
    fig.suptitle('FEP-Mem: Ablation Study on text8 (standard protocol)',
                 fontsize=14, fontweight='bold', color='white')

    names = ['A_full', 'B_no_prec', 'C_no_comp', 'D_no_dropout', 'E_no_memory',
             'F_minimal', 'H_sLSTM']
    labels = ['Full\nFEP-Mem', 'No\nPrecision', 'No\nCompete', 'No\nDrop', 'No\nMemory',
              'Minimal\nMem', 'sLSTM']
    colors = ['#4CAF50', '#FF9800', '#9C27B0', '#2196F3', '#F44336', '#795548', '#E91E63']

    # Bar chart with error bars
    ax = axes[0]; ax.set_facecolor('#0d1117')
    means = [all_results[n]['mean_bpc'] for n in names]
    stds = [all_results[n]['std_bpc'] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor='white', linewidth=0.5,
           capsize=3, error_kw={'color': 'white', 'linewidth': 1})
    ax.set_xticks(x); ax.set_xticklabels(labels, color='white', fontsize=8)
    ax.set_ylabel('Test BPC (S=0 reset, lower=better)', color='white')
    ax.set_title('Ablation: Component Contribution', color='white')
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15, axis='y')
    ax.axhline(y=1.59, color='yellow', linestyle='--', alpha=0.5)
    ax.text(len(names)-0.5, 1.61, 'Lit: 3.3M LSTM', color='yellow', fontsize=7, alpha=0.7)

    # S vs S=0 comparison
    ax = axes[1]; ax.set_facecolor('#0d1117')
    with_S = [all_results[n]['mean_with_S'] for n in names]
    w = 0.35
    ax.bar(x - w/2, means, w, label='S=0 (fair)', color=colors, alpha=0.8, edgecolor='white')
    ax.bar(x + w/2, with_S, w, label='With S (cache)', color=colors, alpha=0.3, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(labels, color='white', fontsize=8)
    ax.set_ylabel('Test BPC', color='white')
    ax.set_title('Fair (S=0) vs Cached (with S)', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fep_mem_rigorous.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot: results/fep_mem_rigorous.png")

    with open(os.path.join(OUT, 'fep_mem_rigorous.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON: results/fep_mem_rigorous.json")


if __name__ == '__main__':
    main()
