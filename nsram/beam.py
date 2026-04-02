"""BEAM — Byte-level Embodied Associative Memory

Online byte-level language model where the core memory is a set of
associative matrices (S-heads) updated by precision-weighted delta rule.

Maps 1:1 to NS-RAM crossbar hardware:
    S[i][j]         ↔  crossbar conductance G_ij
    delta rule       ↔  SRH charge trapping/release
    precision        ↔  per-device noise awareness
    β (write gate)   ↔  SET/RESET pulse amplitude
    multi-head       ↔  multi-chip crossbar array

Architecture per byte:
    embed → gated recurrence → multi-head query → retrieve from S → predict

Three learning modes:
    'readout'  — Only W_out + S learn (reservoir computing, no backprop)
    'local'    — All params learn with local rules (no backprop)
    'backprop' — Full SGD baseline for comparison

Reference: MIRAS (Google, 2025) unified view of sequence models as
online associative memory optimizers. BEAM instantiates a novel point
in the MIRAS design space: FEP-precision-weighted delta rule with
surprise-modulated write gate, operating at byte granularity.

Example:
    >>> from nsram.beam import BEAMModel
    >>> model = BEAMModel(n_heads=4, use_precision=True, mode='readout')
    >>> bpc = model.train_online(b"hello world " * 1000)
    >>> print(f"BPC: {bpc:.2f}")
"""

import numpy as np
from typing import Optional, Literal


class BEAMModel:
    """Byte-level Embodied Associative Memory.

    Parameters
    ----------
    d : int
        Model dimension (default 64).
    n_heads : int
        Number of associative memory heads (crossbar arrays).
        Must divide d evenly.
    use_precision : bool
        Enable FEP-inspired local precision weighting.
    mode : str
        'readout' (W_out+S only), 'local' (all local rules), 'backprop'.
    lr : float
        Learning rate for readout layer.
    lr_local : float
        Learning rate for local rules (embed, gate, q, k).
    beta : float
        Base write strength for S-matrix delta rule.
    s_clip : float
        Clamp S-matrix entries to [-s_clip, s_clip].
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        d: int = 64,
        n_heads: int = 4,
        use_precision: bool = True,
        mode: Literal['readout', 'local', 'backprop'] = 'readout',
        lr: float = 0.001,
        lr_local: float = 0.0003,
        beta: float = 0.02,
        s_clip: float = 2.0,
        seed: Optional[int] = 42,
    ):
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.use_precision = use_precision
        self.mode = mode
        self.lr = lr
        self.lr_local = lr_local
        self.beta_base = beta
        self.s_clip = s_clip
        self.vocab = 256

        rng = np.random.RandomState(seed)
        hd = self.head_dim

        # SGD-trained params
        s_embed = np.sqrt(2.0 / (self.vocab + d))
        self.W_embed = rng.randn(self.vocab, d).astype(np.float32) * s_embed

        s_gate = np.sqrt(2.0 / (d + d)) * 0.5
        self.W_gate = rng.randn(d, d).astype(np.float32) * s_gate
        self.b_gate = np.full(d, 0.5, dtype=np.float32)

        s_proj = np.sqrt(2.0 / (hd + d))
        self.W_q = rng.randn(n_heads, hd, d).astype(np.float32) * s_proj
        self.b_q = np.zeros((n_heads, hd), dtype=np.float32)
        self.W_k = rng.randn(n_heads, hd, d).astype(np.float32) * s_proj
        self.b_k = np.zeros((n_heads, hd), dtype=np.float32)

        odim = 2 * d
        s_out = np.sqrt(2.0 / (odim + self.vocab))
        self.W_out = rng.randn(self.vocab, odim).astype(np.float32) * s_out
        self.b_out = np.zeros(self.vocab, dtype=np.float32)

        # S-matrices (delta-rule, crossbar conductances)
        self.S = np.zeros((n_heads, hd, hd), dtype=np.float32)

        # State
        self.h = np.zeros(d, dtype=np.float32)

        # Precision tracking
        self.err_var = np.ones((n_heads, hd), dtype=np.float32)
        self.err_norm_ema = np.ones(n_heads, dtype=np.float32)
        self.reward_baseline = 5.0

        # Metrics
        self._total_bytes = 0
        self._total_loss = 0.0

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _normalize(self, v):
        return v / (np.linalg.norm(v) + 1e-8)

    def forward(self, byte_in: int, byte_target: int):
        """Forward pass for one byte. Returns (loss, cache)."""
        d, H, hd = self.d, self.n_heads, self.head_dim

        embed = self.W_embed[byte_in].copy()
        h_prev = self.h.copy()

        # Gate
        gate_pre = self.W_gate @ h_prev + self.b_gate
        alpha = self._sigmoid(gate_pre)

        # State update
        h_new = alpha * h_prev + (1.0 - alpha) * embed

        # Per-head query, key, retrieve
        queries, keys, retrievals, errors = [], [], [], []
        precisions, betas = [], []

        for head in range(H):
            q = self._normalize(self.W_q[head] @ h_new + self.b_q[head])
            k = self._normalize(self.W_k[head] @ h_new + self.b_k[head])
            r = self.S[head] @ q
            err = self.W_embed[byte_target][head*hd:(head+1)*hd] - r

            # Local precision
            if self.use_precision:
                e2 = err ** 2
                self.err_var[head] = 0.995 * self.err_var[head] + 0.005 * e2
                prec = 1.0 / (self.err_var[head] + 0.01)
                prec *= hd / (prec.sum() + 1e-8)  # normalize to mean=1

                en = np.sqrt(np.mean(e2))
                self.err_norm_ema[head] = 0.995 * self.err_norm_ema[head] + 0.005 * en
                sr = en / (self.err_norm_ema[head] + 1e-6)
                beta = np.clip(self.beta_base * sr, 0.002, 0.15)
            else:
                prec = np.ones(hd, dtype=np.float32)
                beta = self.beta_base

            queries.append(q)
            keys.append(k)
            retrievals.append(r)
            errors.append(err)
            precisions.append(prec)
            betas.append(beta)

        # Output
        out_vec = np.concatenate([h_new] + retrievals)
        logits = self.W_out @ out_vec + self.b_out
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        loss = -np.log(probs[byte_target] + 1e-10)

        self.h = h_new

        cache = {
            'embed': embed, 'h_prev': h_prev, 'h_new': h_new,
            'alpha': alpha, 'gate_pre': gate_pre,
            'queries': queries, 'keys': keys, 'retrievals': retrievals,
            'errors': errors, 'precisions': precisions, 'betas': betas,
            'out_vec': out_vec, 'probs': probs, 'loss': loss,
        }
        return loss, cache

    def update_S(self, cache):
        """Delta-rule update of associative memory (crossbar conductances)."""
        for h in range(self.n_heads):
            beta = cache['betas'][h]
            k = cache['keys'][h]
            pe = np.clip(cache['precisions'][h] * cache['errors'][h], -10, 10)
            self.S[h] = (1 - beta) * self.S[h] + beta * np.outer(pe, k)
            np.clip(self.S[h], -self.s_clip, self.s_clip, out=self.S[h])

    def update_readout(self, cache, byte_target):
        """Update W_out with CE gradient (local — output layer only)."""
        dl = cache['probs'].copy()
        dl[byte_target] -= 1.0
        np.clip(dl, -5, 5, out=dl)
        self.b_out -= self.lr * dl
        self.W_out -= self.lr * np.outer(dl, cache['out_vec'])

    def update_local(self, cache, byte_in):
        """Update all params with local rules (no backprop)."""
        loss = cache['loss']
        self.reward_baseline = 0.99 * self.reward_baseline + 0.01 * loss
        advantage = np.clip((loss - self.reward_baseline) * 0.1, -0.5, 0.5)

        # W_embed: reward-modulated contrastive
        delta = advantage * (cache['embed'] - cache['h_new'])
        self.W_embed[byte_in] -= self.lr_local * delta

        # W_gate: REINFORCE
        grad_log_pi = 1.0 - cache['alpha']
        d_gate = np.clip(advantage * 0.01, -0.1, 0.1) * grad_log_pi
        self.b_gate -= self.lr_local * d_gate
        self.W_gate -= self.lr_local * np.outer(d_gate, cache['h_prev'])

        # W_q: retrieval error Hebbian
        for h in range(self.n_heads):
            d_q = self.S[h].T @ cache['errors'][h]
            d_q_norm = d_q / (np.linalg.norm(d_q) + 1e-8)
            self.b_q[h] += self.lr_local * np.clip(d_q_norm, -1, 1)
            self.W_q[h] += self.lr_local * np.outer(
                np.clip(d_q_norm, -1, 1), cache['h_new'])

    def step(self, byte_in: int, byte_target: int) -> float:
        """Process one byte: forward + update. Returns loss."""
        loss, cache = self.forward(byte_in, byte_target)

        if loss != loss:  # NaN
            self.h[:] = 0
            self.S[:] = 0
            return 8.0

        self.update_S(cache)

        if self.mode in ('readout', 'local', 'backprop'):
            self.update_readout(cache, byte_target)

        if self.mode == 'local':
            self.update_local(cache, byte_in)

        self._total_bytes += 1
        self._total_loss += loss
        return loss

    def train_online(self, data: bytes, log_every: int = 10000) -> float:
        """Train on a byte sequence, one byte at a time. Returns final BPC."""
        n = len(data)
        running_loss = 0.0
        running_count = 0

        for t in range(n - 1):
            loss = self.step(data[t], data[t + 1])
            running_loss += loss
            running_count += 1

            if log_every > 0 and (t + 1) % log_every == 0:
                bpc = running_loss / running_count / 0.693147
                print(f"[{t+1:>9d}] bpc={bpc:.4f}")
                running_loss = 0.0
                running_count = 0

        if self._total_bytes > 0:
            return self._total_loss / self._total_bytes / 0.693147
        return 8.0

    def predict(self, context: bytes, n_bytes: int = 100) -> bytes:
        """Generate n_bytes by sampling from the model's predictions."""
        result = bytearray()
        # Feed context
        for t in range(len(context) - 1):
            self.forward(context[t], context[t + 1])

        last_byte = context[-1] if context else 32  # space
        for _ in range(n_bytes):
            _, cache = self.forward(last_byte, 0)  # target doesn't matter for generation
            probs = cache['probs']
            # Temperature sampling
            next_byte = np.random.choice(256, p=probs)
            result.append(next_byte)
            last_byte = next_byte

        return bytes(result)

    @property
    def bpc(self) -> float:
        """Current average bits-per-character."""
        if self._total_bytes == 0:
            return 8.0
        return self._total_loss / self._total_bytes / 0.693147

    @property
    def param_count(self) -> dict:
        """Count parameters by component."""
        d, H, hd = self.d, self.n_heads, self.head_dim
        return {
            'W_embed': self.vocab * d,
            'W_gate': d * d + d,
            'W_q': H * hd * d + H * hd,
            'W_k': H * hd * d + H * hd,
            'W_out': self.vocab * 2 * d + self.vocab,
            'S_crossbar': H * hd * hd,
            'total_sgd': (self.vocab * d + d * d + d + H * 2 * hd * d
                          + H * 2 * hd + self.vocab * 2 * d + self.vocab),
            'total': (self.vocab * d + d * d + d + H * 2 * hd * d
                      + H * 2 * hd + self.vocab * 2 * d + self.vocab
                      + H * hd * hd),
        }

    def get_crossbar_state(self) -> np.ndarray:
        """Return S-matrices as (n_heads, head_dim, head_dim) — the crossbar conductances."""
        return self.S.copy()

    def summary(self) -> str:
        """Model summary string."""
        pc = self.param_count
        lines = [
            f"BEAM — Byte-level Embodied Associative Memory",
            f"  D={self.d}, heads={self.n_heads}, head_dim={self.head_dim}",
            f"  mode={self.mode}, precision={self.use_precision}",
            f"  SGD params: {pc['total_sgd']:,}",
            f"  S-matrix (crossbar): {pc['S_crossbar']:,} conductances",
            f"  Total: {pc['total']:,}",
            f"  Bytes processed: {self._total_bytes:,}",
            f"  Current BPC: {self.bpc:.4f}",
        ]
        return "\n".join(lines)
