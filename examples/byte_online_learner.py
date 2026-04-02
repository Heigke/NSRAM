#!/usr/bin/env python3
"""BOLT: Byte-level Online Learning — Architecture Comparison

Truly streaming byte-level online learner. No tokenizer. No epochs. No task boundaries.
Processes one byte at a time from ANY source: text, images, audio, HW streams.

Compares 5 architectures × 3 weight regimes:

Architectures:
  1. SSM: Selective State Space (Mamba-style diagonal recurrence)
  2. sLSTM: Extended LSTM with exponential gating (Hochreiter 2024)
  3. TTT-Linear: Test-Time Training (hidden state = learnable weights)
  4. Reservoir+Linear: Fixed random reservoir + online linear readout
  5. Transformer: Standard causal attention (baseline)

Weight regimes:
  1. Standard: float32 + Adam
  2. NSRAM: 14-level quantized + SRH trapping physics
  3. DualTimescale: fast (body) + slow (oxide) consolidation

Input: bytes 0-255. Output: 256-way softmax (next byte prediction).
"""

import sys, os, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from nsram.physics import charge_capture_rate, srh_trapping_ode
    HAS_NSRAM = True
except ImportError:
    HAS_NSRAM = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE 1: SSM (Selective State Space — Mamba-style)
# ═══════════════════════════════════════════════════════════════════

class SSMLayer(nn.Module):
    """Simplified selective state-space layer (diagonal A, input-dependent B,C,Δ)."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A (log-space for stability)
        self.log_A = nn.Parameter(torch.log(torch.linspace(1, d_state, d_state).repeat(d_model, 1)))

        # Input-dependent projections
        self.proj_delta = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.proj_D = nn.Linear(d_model, d_model)  # Skip connection

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, S, D). Returns (B, S, D)."""
        B, S, D = x.shape
        h = self.norm(x)

        # Selective parameters
        delta = F.softplus(self.proj_delta(h))  # (B, S, D)
        B_inp = self.proj_B(h)  # (B, S, N)
        C_inp = self.proj_C(h)  # (B, S, N)

        # Discretize A
        A = -torch.exp(self.log_A)  # (D, N)

        # Scan (sequential for correctness, could be parallelized)
        state = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []

        for t in range(S):
            dt = delta[:, t, :].unsqueeze(-1)  # (B, D, 1)
            dA = torch.exp(dt * A.unsqueeze(0))  # (B, D, N)
            dB = dt * B_inp[:, t, :].unsqueeze(1)  # (B, 1, N) → (B, D, N) via broadcast
            # But we need input mixed in: state = dA * state + dB * x
            x_t = h[:, t, :].unsqueeze(-1)  # (B, D, 1)
            state = dA * state + dB * x_t  # (B, D, N)
            y_t = (state * C_inp[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, S, D)
        return x + y + self.proj_D(h)


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE 2: sLSTM (Exponential Gating)
# ═══════════════════════════════════════════════════════════════════

class sLSTMLayer(nn.Module):
    """Simplified sLSTM with exponential gating (Hochreiter 2024)."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Gates: input, forget, output, cell candidate
        self.W = nn.Linear(d_model, 4 * d_model)
        self.U = nn.Linear(d_model, 4 * d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, S, D)."""
        B, S, D = x.shape
        h_t = torch.zeros(B, D, device=x.device)
        c_t = torch.zeros(B, D, device=x.device)
        n_t = torch.ones(B, D, device=x.device)  # Normalizer state

        h_out = self.norm(x)
        outputs = []

        for t in range(S):
            gates = self.W(h_out[:, t, :]) + self.U(h_t)
            i, f, o, z = gates.chunk(4, dim=-1)

            # Exponential gating (the sLSTM innovation)
            i_t = torch.exp(i)  # Exponential input gate
            f_t = torch.sigmoid(f)  # Standard forget gate (or exp for full sLSTM)
            o_t = torch.sigmoid(o)

            # Update cell with normalizer
            c_t = f_t * c_t + i_t * torch.tanh(z)
            n_t = f_t * n_t + i_t  # Normalizer tracks scale
            h_t = o_t * (c_t / (n_t + 1e-6))  # Normalized output

            outputs.append(h_t)

        y = torch.stack(outputs, dim=1)
        return x + y


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE 3: TTT-Linear (Test-Time Training)
# ═══════════════════════════════════════════════════════════════════

class TTTLinearLayer(nn.Module):
    """Test-Time Training: hidden state IS a learnable linear model.

    At each step: (1) predict from current state weights, (2) compute loss,
    (3) update state weights via gradient step. The "state" is a matrix W.
    """

    def __init__(self, d_model, d_state=64, ttt_lr=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ttt_lr = ttt_lr

        # Project input to key/value for TTT
        self.proj_k = nn.Linear(d_model, d_state)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_q = nn.Linear(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, S, D)."""
        B, S, D = x.shape
        h = self.norm(x)

        # TTT state: a linear model W ∈ R^{D × d_state}
        W = torch.zeros(B, D, self.d_state, device=x.device)

        keys = self.proj_k(h)    # (B, S, d_state)
        values = self.proj_v(h)  # (B, S, D)
        queries = self.proj_q(h) # (B, S, d_state)

        outputs = []
        for t in range(S):
            q_t = queries[:, t, :]  # (B, d_state)
            k_t = keys[:, t, :]     # (B, d_state)
            v_t = values[:, t, :]   # (B, D)

            # Predict from current state
            y_t = torch.bmm(W, q_t.unsqueeze(-1)).squeeze(-1)  # (B, D)

            # TTT update: online gradient step
            # loss = ||W @ k_t - v_t||^2
            pred = torch.bmm(W, k_t.unsqueeze(-1)).squeeze(-1)  # (B, D)
            error = pred - v_t  # (B, D)

            # Gradient: d(loss)/d(W) = 2 * error ⊗ k_t
            grad = torch.bmm(error.unsqueeze(-1), k_t.unsqueeze(1))  # (B, D, d_state)
            W = W - self.ttt_lr * grad  # Online update

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return x + y


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE 4: Reservoir + Online Linear Readout
# ═══════════════════════════════════════════════════════════════════

class ReservoirLayer(nn.Module):
    """Fixed random reservoir + online-updated linear readout.

    The reservoir is a randomly initialized recurrent network (not trained).
    Only the linear readout is trained. This is the FPGA-compatible approach.
    """

    def __init__(self, d_model, d_reservoir=256, spectral_radius=0.95, leak=0.3):
        super().__init__()
        self.d_model = d_model
        self.d_reservoir = d_reservoir
        self.leak = leak

        # Fixed reservoir weights (not trained)
        W_res = torch.randn(d_reservoir, d_reservoir) * 0.1
        # Scale to desired spectral radius
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W_res).abs()
            W_res = W_res * (spectral_radius / eigenvalues.max())
        self.register_buffer('W_res', W_res)

        # Fixed input projection
        self.register_buffer('W_in', torch.randn(d_reservoir, d_model) * 0.1)

        # Trainable readout
        self.readout = nn.Linear(d_reservoir, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, S, D)."""
        B, S, D = x.shape
        h = self.norm(x)
        state = torch.zeros(B, self.d_reservoir, device=x.device)
        outputs = []

        for t in range(S):
            # Reservoir update: leaky integration
            inp = h[:, t, :] @ self.W_in.T  # (B, d_res)
            state = (1 - self.leak) * state + self.leak * torch.tanh(
                state @ self.W_res.T + inp)
            outputs.append(self.readout(state))

        y = torch.stack(outputs, dim=1)
        return x + y


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE 5: Causal Transformer (baseline)
# ═══════════════════════════════════════════════════════════════════

class TransformerLayer(nn.Module):
    """Standard causal self-attention block."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn.masked_fill_(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.out(out)
        x = x + self.ff(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════
# NS-RAM WEIGHT QUANTIZATION (applied on top of any architecture)
# ═══════════════════════════════════════════════════════════════════

def quantize_model_nsram(model, n_levels=14):
    """Post-step hook: quantize all trainable weights to 14 levels + SRH decay."""
    levels = torch.linspace(-1, 1, n_levels, device=DEVICE)

    def hook(grad):
        return grad  # STE: pass gradients through

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            with torch.no_grad():
                # Quantize to nearest level
                flat = param.data.reshape(-1, 1)
                dists = (flat - levels.reshape(1, -1)).abs()
                idx = dists.argmin(dim=1)
                param.data = levels[idx].reshape(param.shape)


class DualTimescaleWrapper:
    """Wraps a model to add fast/slow weight dynamics."""

    def __init__(self, model, tau_fast=500, consolidate_every=200, alpha=0.3):
        self.model = model
        self.tau_fast = tau_fast
        self.consolidate_every = consolidate_every
        self.alpha = alpha
        self.step = 0

        # Create fast weight buffers for all parameters
        self.fast_weights = {}
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                self.fast_weights[name] = torch.zeros_like(param.data)

    def post_step(self):
        self.step += 1
        decay = math.exp(-1.0 / self.tau_fast)

        for name, param in self.model.named_parameters():
            if name in self.fast_weights:
                # Decay fast weights
                self.fast_weights[name] *= decay

                # Accumulate gradient residual
                if param.grad is not None:
                    self.fast_weights[name] -= 0.01 * param.grad.data

                # Apply: effective = slow + alpha * fast
                param.data += self.alpha * self.fast_weights[name]

                # Consolidation
                if self.step % self.consolidate_every == 0:
                    mag = self.fast_weights[name].abs()
                    thresh = mag.mean() + mag.std()
                    mask = mag > thresh
                    param.data[mask] += self.fast_weights[name][mask] * 0.1
                    self.fast_weights[name][mask] *= 0.5


# ═══════════════════════════════════════════════════════════════════
# UNIFIED MODEL
# ═══════════════════════════════════════════════════════════════════

class ByteModel(nn.Module):
    """Universal byte-level model with configurable architecture."""

    def __init__(self, arch='ssm', d_model=128, n_layers=2, context_len=128):
        super().__init__()
        self.context_len = context_len
        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)

        layer_cls = {
            'ssm': lambda: SSMLayer(d_model, d_state=16),
            'slstm': lambda: sLSTMLayer(d_model),
            'ttt': lambda: TTTLinearLayer(d_model, d_state=64, ttt_lr=0.01),
            'reservoir': lambda: ReservoirLayer(d_model, d_reservoir=256),
            'transformer': lambda: TransformerLayer(d_model, n_heads=4),
        }[arch]

        self.layers = nn.ModuleList([layer_cls() for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 256)
        self.arch = arch

    def forward(self, byte_ids):
        """byte_ids: (B, S) long tensor 0-255. Returns (B, S, 256) logits."""
        B, S = byte_ids.shape
        emb = self.byte_embed(byte_ids) + self.pos_embed(torch.arange(S, device=byte_ids.device))
        h = emb
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ═══════════════════════════════════════════════════════════════════
# BYTE STREAMS
# ═══════════════════════════════════════════════════════════════════

def stream_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    for b in data:
        yield b


def stream_mnist():
    from torchvision import datasets, transforms
    ds = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    for img, label in ds:
        yield label
        for p in (img.squeeze() * 255).byte().flatten().tolist():
            yield p


def stream_audio():
    sr = 8000
    freqs = [261, 293, 329, 349, 392, 440, 493, 523]
    for _ in range(200):
        freq = freqs[np.random.randint(len(freqs))]
        dur = np.random.uniform(0.1, 0.5)
        t = np.arange(int(sr * dur)) / sr
        wave = (np.sin(2 * np.pi * freq * t) * 127 + 128).astype(np.uint8)
        for b in wave:
            yield int(b)


def stream_mixed(text_path, switch_every=4096):
    """Interleave modalities with no boundary signal."""
    makers = [
        ('text', lambda: stream_file(text_path)),
        ('image', stream_mnist),
        ('audio', stream_audio),
    ]
    current = 0
    gen = makers[0][1]()
    count = 0
    total = 0
    while total < 500000:
        try:
            b = next(gen)
            yield b, makers[current][0]
            count += 1
            total += 1
            if count >= switch_every:
                current = (current + 1) % len(makers)
                gen = makers[current][1]()
                count = 0
        except StopIteration:
            current = (current + 1) % len(makers)
            gen = makers[current][1]()
            count = 0


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_streaming(model, byte_stream, max_bytes=50000, window=128,
                    eval_every=5000, name='stream', regime='standard',
                    dt_wrapper=None):
    """Train on streaming bytes with sliding windows for GPU efficiency."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()

    metrics = {'byte': [], 'loss': [], 'bpc': [], 'acc': [], 'mod': []}
    buffer = []
    total = 0
    r_loss = r_acc = r_n = 0
    last_mod = name

    for item in byte_stream:
        if total >= max_bytes:
            break
        if isinstance(item, tuple):
            b, last_mod = item
        else:
            b = item
        buffer.append(b)
        total += 1

        if len(buffer) >= window + 1:
            inp = torch.tensor(buffer[:window], dtype=torch.long, device=DEVICE).unsqueeze(0)
            tgt = torch.tensor(buffer[1:window+1], dtype=torch.long, device=DEVICE).unsqueeze(0)

            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Weight regime post-processing
            if regime == 'nsram':
                quantize_model_nsram(model)
            elif regime == 'dual' and dt_wrapper:
                dt_wrapper.post_step()

            with torch.no_grad():
                acc = (logits.argmax(-1) == tgt).float().mean().item()
            r_loss += loss.item(); r_acc += acc; r_n += 1
            buffer = buffer[window // 2:]

        if total % eval_every == 0 and r_n > 0:
            al = r_loss / r_n; ab = al / math.log(2); aa = r_acc / r_n
            metrics['byte'].append(total)
            metrics['loss'].append(al); metrics['bpc'].append(ab)
            metrics['acc'].append(aa); metrics['mod'].append(last_mod)
            print(f'  [{name}] {total:6d}B: bpc={ab:.2f}, acc={aa:.3f} [{last_mod}]')
            r_loss = r_acc = r_n = 0

    return metrics


def eval_stream(model, byte_stream, max_bytes=5000, window=128):
    model.eval()
    buf = []; tl = ta = n = 0
    with torch.no_grad():
        for i, item in enumerate(byte_stream):
            if i >= max_bytes: break
            buf.append(item[0] if isinstance(item, tuple) else item)
            if len(buf) >= window + 1:
                inp = torch.tensor(buf[:window], dtype=torch.long, device=DEVICE).unsqueeze(0)
                tgt = torch.tensor(buf[1:window+1], dtype=torch.long, device=DEVICE).unsqueeze(0)
                logits = model(inp)
                tl += F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1)).item()
                ta += (logits.argmax(-1) == tgt).float().mean().item()
                n += 1; buf = buf[window // 2:]
    return (tl/n, tl/n/math.log(2), ta/n) if n > 0 else (8, 8/math.log(2), 0)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"NS-RAM physics: {'YES' if HAS_NSRAM else 'NO'}")

    # Find text
    text_path = None
    base = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [os.path.join(base, '..'), os.path.join(base, '..', '..')]
    for sd in search_dirs:
        for p in ['data/tinyshakespeare/tiny_shakespeare.txt', 'data/tinyshakespeare.txt',
                  'data/tiny_shakespeare.txt', 'data/input.txt']:
            full = os.path.join(sd, p)
            if os.path.exists(full):
                text_path = full; break
        if text_path: break
    print(f"Text: {text_path} ({os.path.getsize(text_path):,} bytes)")

    ARCHS = ['ssm', 'slstm', 'ttt', 'reservoir', 'transformer']
    REGIMES = ['standard', 'nsram', 'dual']
    N_BYTES = 40000
    cfg = dict(d_model=128, n_layers=2, context_len=128)

    all_results = {}

    # ══════════════════════════════════════════════════════
    # EXP 1: Architecture comparison (text, standard weights)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  EXP 1: Architecture comparison — text stream, standard weights")
    print(f"{'='*65}")

    arch_metrics = {}
    for arch in ARCHS:
        model = ByteModel(arch=arch, **cfg).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        print(f"\n  --- {arch} ({n_p:,} params) ---")
        arch_metrics[arch] = train_streaming(
            model, stream_file(text_path), max_bytes=N_BYTES,
            window=cfg['context_len'], eval_every=10000, name=arch)
    all_results['arch_text'] = arch_metrics

    # ══════════════════════════════════════════════════════
    # EXP 2: Weight regime comparison (best 2 archs + transformer)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  EXP 2: Weight regime comparison — text stream")
    print(f"{'='*65}")

    regime_metrics = {}
    test_archs = ['ssm', 'slstm', 'transformer']  # Compare regimes on these

    for arch in test_archs:
        for regime in REGIMES:
            key = f'{arch}_{regime}'
            model = ByteModel(arch=arch, **cfg).to(DEVICE)
            dt_wrapper = DualTimescaleWrapper(model) if regime == 'dual' else None
            print(f"\n  --- {key} ---")
            regime_metrics[key] = train_streaming(
                model, stream_file(text_path), max_bytes=N_BYTES,
                window=cfg['context_len'], eval_every=10000,
                name=key, regime=regime, dt_wrapper=dt_wrapper)
    all_results['regime_text'] = regime_metrics

    # ══════════════════════════════════════════════════════
    # EXP 3: Mixed modality stream (key test)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  EXP 3: Mixed stream (text↔image↔audio) — best architectures")
    print(f"{'='*65}")

    mix_metrics = {}
    for arch in ['ssm', 'slstm', 'ttt', 'reservoir']:
        model = ByteModel(arch=arch, **cfg).to(DEVICE)
        print(f"\n  --- {arch} ---")
        mix_metrics[arch] = train_streaming(
            model, stream_mixed(text_path), max_bytes=N_BYTES * 2,
            window=cfg['context_len'], eval_every=10000, name=f'mix_{arch}')
    all_results['mixed'] = mix_metrics

    # ══════════════════════════════════════════════════════
    # EXP 4: Forgetting (text→image→eval text)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  EXP 4: Forgetting — text (40KB) → image (40KB) → eval text")
    print(f"{'='*65}")

    forget_results = {}
    for arch in ['ssm', 'slstm', 'ttt', 'reservoir', 'transformer']:
        for regime in ['standard', 'dual']:
            key = f'{arch}_{regime}'
            model = ByteModel(arch=arch, **cfg).to(DEVICE)
            dt = DualTimescaleWrapper(model) if regime == 'dual' else None
            print(f"\n  --- {key} ---")

            # Learn text
            train_streaming(model, stream_file(text_path), max_bytes=N_BYTES,
                             window=cfg['context_len'], eval_every=50000,
                             name=f'{key}_text', regime=regime, dt_wrapper=dt)
            _, bpc_b, acc_b = eval_stream(model, stream_file(text_path))
            print(f"  BEFORE: bpc={bpc_b:.2f}, acc={acc_b:.3f}")

            # Learn images
            train_streaming(model, stream_mnist(), max_bytes=N_BYTES,
                             window=cfg['context_len'], eval_every=50000,
                             name=f'{key}_img', regime=regime, dt_wrapper=dt)
            _, bpc_a, acc_a = eval_stream(model, stream_file(text_path))
            print(f"  AFTER:  bpc={bpc_a:.2f}, acc={acc_a:.3f}")

            fpp = (acc_b - acc_a) * 100
            print(f"  FORGET: {fpp:+.1f}pp")
            forget_results[key] = {'before': acc_b, 'after': acc_a, 'forget_pp': fpp}
    all_results['forgetting'] = forget_results

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print("  BOLT: BYTE-LEVEL ONLINE LEARNING — FINAL SUMMARY")
    print(f"{'='*75}")

    # Arch comparison
    print(f"\n  Architecture comparison (text, standard weights):")
    print(f"  {'Arch':<15s} {'Final BPC':>10s} {'Final Acc':>10s}")
    print(f"  {'-'*35}")
    for arch in ARCHS:
        m = arch_metrics[arch]
        bpc = m['bpc'][-1] if m['bpc'] else float('nan')
        acc = m['acc'][-1] if m['acc'] else float('nan')
        print(f"  {arch:<15s} {bpc:>9.2f} {acc:>9.3f}")

    # Forgetting
    print(f"\n  Forgetting (text→image→text eval):")
    print(f"  {'Config':<25s} {'Before':>8s} {'After':>8s} {'Forget':>10s}")
    print(f"  {'-'*51}")
    for key in sorted(forget_results.keys()):
        fr = forget_results[key]
        print(f"  {key:<25s} {fr['before']:>7.3f} {fr['after']:>7.3f} {fr['forget_pp']:>+9.1f}pp")

    # ══════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='#0d1117')
    fig.suptitle('BOLT: Byte-level Online Learning — 5 Architectures × 3 Weight Regimes',
                 fontsize=14, fontweight='bold', color='white')

    arch_colors = {'ssm': '#4CAF50', 'slstm': '#2196F3', 'ttt': '#FF9800',
                   'reservoir': '#9C27B0', 'transformer': '#F44336'}

    # P1: Architecture BPC curves
    ax = axes[0, 0]; ax.set_facecolor('#0d1117')
    for arch in ARCHS:
        m = arch_metrics[arch]
        if m['byte']:
            ax.plot(m['byte'], m['bpc'], 'o-', label=arch,
                    color=arch_colors[arch], linewidth=2, markersize=4)
    ax.set_xlabel('Bytes', color='white'); ax.set_ylabel('BPC', color='white')
    ax.set_title('Architecture Comparison (Text)', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15)

    # P2: Mixed stream
    ax = axes[0, 1]; ax.set_facecolor('#0d1117')
    for arch in mix_metrics:
        m = mix_metrics[arch]
        if m['byte']:
            ax.plot(m['byte'], m['bpc'], 'o-', label=arch,
                    color=arch_colors.get(arch, '#888'), linewidth=2, markersize=4)
    ax.set_xlabel('Bytes', color='white'); ax.set_ylabel('BPC', color='white')
    ax.set_title('Mixed Stream (text↔image↔audio)', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', fontsize=8)
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15)

    # P3: Forgetting bars
    ax = axes[1, 0]; ax.set_facecolor('#0d1117')
    forget_keys = sorted(forget_results.keys())
    fvals = [forget_results[k]['forget_pp'] for k in forget_keys]
    colors_f = ['#4CAF50' if 'dual' in k else '#F44336' for k in forget_keys]
    bars = ax.barh(range(len(forget_keys)), fvals, color=colors_f, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(forget_keys)))
    ax.set_yticklabels(forget_keys, color='white', fontsize=7)
    ax.set_xlabel('Forgetting (pp)', color='white')
    ax.set_title('Forgetting: text → image → text eval', color='white')
    ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15, axis='x')
    ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.3)

    # P4: Final BPC comparison (arch × regime)
    ax = axes[1, 1]; ax.set_facecolor('#0d1117')
    if regime_metrics:
        rkeys = sorted(regime_metrics.keys())
        rbpc = [regime_metrics[k]['bpc'][-1] if regime_metrics[k]['bpc'] else 8 for k in rkeys]
        c2 = []
        for k in rkeys:
            if 'standard' in k: c2.append('#F44336')
            elif 'nsram' in k: c2.append('#4CAF50')
            else: c2.append('#2196F3')
        ax.bar(range(len(rkeys)), rbpc, color=c2, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(rkeys)))
        ax.set_xticklabels(rkeys, color='white', fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('BPC', color='white')
        ax.set_title('Weight Regime Comparison', color='white')
        ax.tick_params(colors='gray'); ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'bolt_results.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot: results/bolt_results.png")

    with open(os.path.join(OUT, 'bolt_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON: results/bolt_results.json")


if __name__ == '__main__':
    main()
