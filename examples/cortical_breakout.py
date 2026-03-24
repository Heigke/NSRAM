#!/usr/bin/env python3
"""Cortical NS-RAM Brain plays Breakout — 200K neurons, layered architecture.

A biologically-inspired cortical architecture:
  - V1: 50K neurons — visual processing (ball, paddle, bricks encoding)
  - V2: 50K neurons — feature integration
  - PFC: 50K neurons — decision making
  - M1: 50K neurons — motor output

Each layer has sparse internal recurrence + feedforward connections.
This is closer to what a real NS-RAM tapeout would look like.

The game: simplified Breakout (ball bounces off paddle, breaks bricks).
Harder than Pong — requires spatial awareness + planning.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════
# BREAKOUT ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class Breakout:
    def __init__(self, cols=8, rows=4):
        self.cols = cols; self.rows = rows
        self.reset()

    def reset(self):
        self.bx = 0.5; self.by = 0.3
        a = np.random.uniform(0.3, 0.7) * np.pi
        self.bvx = 0.02 * np.cos(a) * (1 if np.random.rand() > 0.5 else -1)
        self.bvy = 0.02 * np.sin(a)
        self.px = 0.5; self.pw = 0.15
        # Bricks: rows×cols grid at top
        self.bricks = np.ones((self.rows, self.cols), dtype=bool)
        self.score = 0; self.lives = 3; self.total_bricks = self.rows * self.cols
        return self.obs()

    def obs(self):
        """Returns observation vector: ball pos/vel + paddle + brick states."""
        brick_flat = self.bricks.flatten().astype(np.float64)
        return np.concatenate([
            [self.bx * 2 - 1, self.by * 2 - 1,
             self.bvx / 0.02, self.bvy / 0.02,
             self.px * 2 - 1],
            brick_flat * 2 - 1  # 32 brick states
        ])

    def step(self, action):
        """action: -1 (left), 0 (stay), +1 (right)"""
        self.px = np.clip(self.px + action * 0.04, self.pw/2, 1 - self.pw/2)
        self.bx += self.bvx; self.by += self.bvy
        reward = 0

        # Wall bounces
        if self.bx <= 0 or self.bx >= 1:
            self.bvx *= -1; self.bx = np.clip(self.bx, 0.01, 0.99)
        if self.by >= 1:
            self.bvy *= -1; self.by = 0.99

        # Paddle bounce
        if self.by <= 0.08 and self.bvy < 0:
            if abs(self.bx - self.px) < self.pw / 2:
                self.bvy = abs(self.bvy)
                offset = (self.bx - self.px) / (self.pw / 2)
                self.bvx += offset * 0.005
                self.by = 0.09
                reward = 0.1  # Small reward for hitting
            elif self.by <= 0:
                self.lives -= 1; reward = -1
                self.bx = 0.5; self.by = 0.3
                self.bvx = 0.02 * (1 if np.random.rand() > 0.5 else -1)
                self.bvy = 0.02

        # Brick collision
        if self.by > 0.6:
            brick_r = int((self.by - 0.65) / (0.35 / self.rows))
            brick_c = int(self.bx * self.cols)
            brick_r = np.clip(brick_r, 0, self.rows - 1)
            brick_c = np.clip(brick_c, 0, self.cols - 1)
            if self.bricks[brick_r, brick_c]:
                self.bricks[brick_r, brick_c] = False
                self.bvy *= -1
                self.score += 1
                reward = 1.0

        done = self.lives <= 0 or self.score >= self.total_bricks
        return self.obs(), reward, done


# ═══════════════════════════════════════════════════════════════════════
# CORTICAL NS-RAM BRAIN (layered architecture)
# ═══════════════════════════════════════════════════════════════════════

class CorticalBrain:
    """4-layer cortical NS-RAM architecture.

    V1 (50K) → V2 (50K) → PFC (50K) → M1 (50K)

    Each layer has:
      - Sparse internal recurrence (100 connections/neuron)
      - Feedforward connections to next layer (50 connections/neuron)
      - AdEx-LIF dynamics with STP
      - Dale's law (80% exc, 20% inh)
    """

    def __init__(self, layer_sizes=[50000, 50000, 50000, 50000],
                 n_inputs=37, k_internal=100, k_ff=50, seed=42):
        self.layers = layer_sizes
        self.N_total = sum(layer_sizes)
        self.n_layers = len(layer_sizes)
        self.device = DEVICE
        rng = np.random.RandomState(seed)

        print(f"  Building cortical brain: {self.N_total:,} neurons ({self.n_layers} layers)")
        var = 0.10
        N = self.N_total

        # Global neuron parameters
        self.tau = torch.tensor(np.clip(1+var*0.15*rng.randn(N),.3,3).astype(np.float32), device=DEVICE)
        self.theta = torch.tensor(np.clip(1+var*0.05*rng.randn(N),.5,2).astype(np.float32), device=DEVICE)
        self.t_ref = torch.tensor(np.clip(.05+var*.01*rng.randn(N),.01,.2).astype(np.float32), device=DEVICE)
        self.dT = torch.tensor(np.clip(.1+var*.015*rng.randn(N),.02,.5).astype(np.float32), device=DEVICE)
        bg = np.clip(.88+var*.088*rng.randn(N),.5,1.2).astype(np.float32)
        self.I_bg = torch.tensor(bg * self.theta.cpu().numpy(), device=DEVICE)
        self.U = torch.tensor(np.clip(.01+.003*rng.randn(N),.001,.05).astype(np.float32), device=DEVICE)
        self.tau_rec = torch.tensor(np.clip(15+5*rng.randn(N),3,50).astype(np.float32), device=DEVICE)
        self.tau_fac = torch.tensor(((1-self.U.cpu().numpy())*10).astype(np.float32), device=DEVICE)
        self.tau_syn = torch.tensor(np.clip(.5+.1*rng.randn(N),.1,2).astype(np.float32), device=DEVICE)

        # Layer boundaries
        boundaries = np.cumsum([0] + layer_sizes)
        self.layer_slices = [(boundaries[i], boundaries[i+1]) for i in range(self.n_layers)]

        # Input weights (to V1 only)
        self.W_in = torch.zeros(N, n_inputs, device=DEVICE)
        s0, e0 = self.layer_slices[0]
        self.W_in[s0:e0] = torch.tensor(rng.randn(layer_sizes[0], n_inputs).astype(np.float32) * 0.2, device=DEVICE)

        # Build sparse connectivity
        print(f"  Building sparse connectivity...")
        all_rows, all_cols, all_vals = [], [], []
        N_exc_frac = 0.8

        for layer_idx in range(self.n_layers):
            s, e = self.layer_slices[layer_idx]
            n = e - s
            N_exc = int(n * N_exc_frac)
            nsign = np.ones(n, np.float32); nsign[N_exc:] = -1

            # Internal recurrence
            for i in range(n):
                targets = rng.choice(n, min(k_internal, n-1), replace=False)
                targets = targets[targets != i] + s
                k_actual = len(targets)
                all_rows.extend([i + s] * k_actual)
                all_cols.extend(targets.tolist())
                w = np.abs(rng.randn(k_actual).astype(np.float32)) * nsign[i] * 0.3 / np.sqrt(k_actual)
                all_vals.extend(w.tolist())

            # Feedforward to next layer
            if layer_idx < self.n_layers - 1:
                s_next, e_next = self.layer_slices[layer_idx + 1]
                n_next = e_next - s_next
                for i in range(n):
                    targets = rng.choice(n_next, min(k_ff, n_next), replace=False) + s_next
                    k_actual = len(targets)
                    all_rows.extend([i + s] * k_actual)
                    all_cols.extend(targets.tolist())
                    w = np.abs(rng.randn(k_actual).astype(np.float32)) * nsign[i] * 0.4 / np.sqrt(k_actual)
                    all_vals.extend(w.tolist())

        nnz = len(all_rows)
        indices = torch.tensor([all_rows, all_cols], dtype=torch.long, device=DEVICE)
        values = torch.tensor(all_vals, dtype=torch.float32, device=DEVICE)
        self.W = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
        print(f"  Sparse W: {nnz/1e6:.1f}M connections ({nnz*12/1e9:.2f} GB)")

        # Readout from M1 (last layer) → 3 actions
        s_m1, e_m1 = self.layer_slices[-1]
        self.W_out = torch.zeros(3, N, device=DEVICE)
        self.elig = torch.zeros(3, N, device=DEVICE)

        self.reset_state()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Brain ready. GPU: {mem:.1f} GB")

    def reset_state(self):
        N = self.N_total
        self.Vm = torch.zeros(N, device=DEVICE)
        self.syn = torch.zeros(N, device=DEVICE)
        self.x_stp = torch.ones(N, device=DEVICE)
        self.u_stp = self.U.clone()
        self.refrac = torch.zeros(N, device=DEVICE)
        self.rate = torch.zeros(N, device=DEVICE)
        self.ft = torch.zeros(N, device=DEVICE)
        self.spike_hist = torch.zeros(N, device=DEVICE)

    @torch.no_grad()
    def step(self, obs):
        N = self.N_total
        u = torch.tensor(obs, dtype=torch.float32, device=self.device)
        I_in = self.W_in @ u
        I_syn = torch.sparse.mm(self.W, (self.syn * self.u_stp * self.x_stp).unsqueeze(1)).squeeze() * 0.3
        active = (self.refrac <= 0).float()
        leak = -self.Vm / self.tau
        exp_t = self.dT * torch.exp(torch.clamp((self.Vm - self.theta) / self.dT.clamp(min=1e-6), -10, 5))
        self.Vm += active * (leak + self.I_bg + I_in + I_syn + exp_t)
        self.Vm += active * 0.01 * torch.randn(N, device=self.device)
        self.Vm.clamp_(-2, 5)
        spiked = (self.Vm >= self.theta) & (self.refrac <= 0)
        if spiked.any():
            self.Vm[spiked] = 0; self.refrac[spiked] = self.t_ref[spiked]
            self.syn[spiked] += 1; self.rate[spiked] += 5
            self.u_stp[spiked] += self.U[spiked] * (1 - self.u_stp[spiked])
            self.x_stp[spiked] -= self.u_stp[spiked] * self.x_stp[spiked]
        self.syn *= torch.exp(-1 / self.tau_syn); self.rate *= 0.95
        self.x_stp += (1 - self.x_stp) / self.tau_rec.clamp(min=0.5)
        self.u_stp += (self.U - self.u_stp) / self.tau_fac.clamp(min=0.5)
        self.refrac = (self.refrac - 1).clamp(min=0)
        self.ft = 0.8 * self.ft + 0.2 * self.Vm
        self.spike_hist = 0.85 * self.spike_hist + 0.15 * spiked.float()

        state = self.Vm + 0.3 * self.ft + 0.1 * self.x_stp
        logits = self.W_out @ state
        probs = torch.softmax(logits, dim=0)
        act = torch.multinomial(probs, 1).item()
        one_hot = torch.zeros(3, device=self.device); one_hot[act] = 1
        self.elig = 0.95 * self.elig + torch.outer(one_hot, state)
        return act - 1, spiked.sum().item(), probs.cpu().numpy()

    @torch.no_grad()
    def reward(self, r):
        self.W_out += 0.003 * r * self.elig; self.W_out.clamp_(-1, 1)

    def layer_activity(self):
        """Get per-layer activity grids."""
        grids = []
        for s, e in self.layer_slices:
            n = e - s
            sz = int(np.ceil(np.sqrt(n)))
            data = torch.zeros(sz * sz, device=DEVICE)
            data[:n] = self.spike_hist[s:e]
            grids.append(data.cpu().numpy().reshape(sz, sz))
        return grids


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"GPU free: {free:.0f} GB")

    # Scale layers based on available memory
    if free > 20:
        sizes = [50000, 50000, 50000, 50000]  # 200K total
    elif free > 10:
        sizes = [25000, 25000, 25000, 25000]  # 100K total
    else:
        sizes = [10000, 10000, 10000, 10000]  # 40K total

    N_total = sum(sizes)
    brain = CorticalBrain(layer_sizes=sizes, k_internal=80, k_ff=40)
    env = Breakout(cols=8, rows=4)

    # ── Train ──
    print(f"\n{'='*65}")
    print(f"  Training {N_total:,}-neuron cortical brain on Breakout")
    print(f"  Architecture: V1({sizes[0]:,}) → V2({sizes[1]:,}) → PFC({sizes[2]:,}) → M1({sizes[3]:,})")
    print(f"{'='*65}\n")

    all_scores = []; all_lives = []
    record_episodes = []

    for ep in range(25):
        obs = env.reset()
        ep_spikes = 0
        frames = []
        t0 = time.time()

        for step in range(800):
            act, n_spk, probs = brain.step(obs)
            obs, reward, done = env.step(act)
            ep_spikes += n_spk
            if reward != 0:
                brain.reward(reward)
            # Record select episodes
            if ep in [0, 6, 12, 18, 24] and step % 4 == 0:
                grids = brain.layer_activity()
                frames.append({
                    'bx': env.bx, 'by': env.by, 'px': env.px,
                    'bricks': env.bricks.copy(),
                    'score': env.score, 'lives': env.lives,
                    'grids': grids, 'probs': probs.copy(),
                    'spikes': n_spk, 'step': step,
                })
            if done:
                break

        elapsed = time.time() - t0
        all_scores.append(env.score); all_lives.append(env.lives)
        if frames:
            record_episodes.append({'ep': ep, 'frames': frames})
        print(f"  Ep {ep:3d}: bricks={env.score:2d}/{env.total_bricks} lives={env.lives} "
              f"spikes={ep_spikes/1e6:.0f}M ({elapsed:.1f}s)")

    # ── Generate video ──
    if record_episodes:
        last_ep = record_episodes[-1]
        frames = last_ep['frames']
        print(f"\n  Rendering video ({len(frames)} frames)...")

        fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
        gs = GridSpec(2, 5, hspace=0.25, wspace=0.15,
                      left=0.03, right=0.97, top=0.90, bottom=0.05)

        fig.text(0.5, 0.96, f'Cortical NS-RAM Brain — {N_total:,} Neurons Playing Breakout',
                 ha='center', fontsize=16, fontweight='bold', color='white')
        fig.text(0.5, 0.925, f'V1({sizes[0]//1000}K) → V2({sizes[1]//1000}K) → PFC({sizes[2]//1000}K) → M1({sizes[3]//1000}K)',
                 ha='center', fontsize=11, color='#888')

        # 4 brain layers (top row)
        layer_names = ['V1 (Visual)', 'V2 (Features)', 'PFC (Decision)', 'M1 (Motor)']
        layer_cmaps = ['inferno', 'magma', 'plasma', 'viridis']
        brain_imgs = []
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            ax.set_facecolor('#0d1117')
            img = ax.imshow(frames[0]['grids'][i], cmap=layer_cmaps[i],
                            aspect='auto', interpolation='bilinear', vmin=0, vmax=0.3)
            ax.set_title(layer_names[i], fontsize=10, color='white', pad=3)
            ax.tick_params(colors='gray', labelsize=5)
            brain_imgs.append(img)

        # Game view (bottom left, 2 cols wide)
        ax_game = fig.add_subplot(gs[1, 0:2])
        ax_game.set_facecolor('#1a1a2e')
        ax_game.set_xlim(0, 1); ax_game.set_ylim(0, 1); ax_game.set_aspect('equal')
        ax_game.set_title('Breakout', fontsize=11, color='white')

        ball = Circle((0.5, 0.3), 0.015, color='#ff6b6b', zorder=5)
        ax_game.add_patch(ball)
        paddle = Rectangle((0.425, 0.03), 0.15, 0.025, color='#4ecdc4', zorder=5)
        ax_game.add_patch(paddle)

        brick_patches = []
        brick_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
        for r in range(env.rows):
            row_patches = []
            for c in range(env.cols):
                x = c / env.cols + 0.005
                y = 0.65 + r * (0.35 / env.rows) + 0.005
                w = 1 / env.cols - 0.01
                h = 0.35 / env.rows - 0.01
                p = Rectangle((x, y), w, h, color=brick_colors[r % 4], zorder=3)
                ax_game.add_patch(p)
                row_patches.append(p)
            brick_patches.append(row_patches)

        score_txt = ax_game.text(0.5, -0.05, '', transform=ax_game.transAxes,
                                  ha='center', fontsize=12, color='white', fontweight='bold')

        # Stats (bottom right)
        ax_stats = fig.add_subplot(gs[0, 4])
        ax_stats.set_facecolor('#0d1117'); ax_stats.axis('off')
        stats_txt = ax_stats.text(0.05, 0.5, '', transform=ax_stats.transAxes,
                                   fontsize=10, color='white', fontfamily='monospace', va='center')

        # Training curve (bottom middle)
        ax_curve = fig.add_subplot(gs[1, 2:4])
        ax_curve.set_facecolor('#0d1117')
        ax_curve.plot(all_scores, 'g-o', markersize=4, linewidth=1.5)
        ax_curve.set_xlabel('Episode', color='white', fontsize=9)
        ax_curve.set_ylabel('Bricks Broken', color='white', fontsize=9)
        ax_curve.set_title('Learning Progress', fontsize=10, color='white')
        ax_curve.tick_params(colors='gray')
        for spine in ax_curve.spines.values(): spine.set_color('#333')
        ax_curve.grid(True, alpha=0.2)

        # Action probs
        ax_prob = fig.add_subplot(gs[1, 4])
        ax_prob.set_facecolor('#0d1117')
        bars = ax_prob.barh(range(3), [.33]*3, color=['#e74c3c','#95a5a6','#2ecc71'], height=0.6)
        ax_prob.set_xlim(0, 1); ax_prob.set_yticks(range(3))
        ax_prob.set_yticklabels(['← Left', '· Stay', 'Right →'], fontsize=9, color='white')
        ax_prob.set_title('Action', fontsize=10, color='white')
        ax_prob.tick_params(colors='gray')
        for spine in ax_prob.spines.values(): spine.set_color('#333')

        def update(i):
            f = frames[i % len(frames)]
            for j, img in enumerate(brain_imgs):
                img.set_data(f['grids'][j])
            ball.set_center((f['bx'], f['by']))
            paddle.set_xy((f['px'] - 0.075, 0.03))
            for r in range(env.rows):
                for c in range(env.cols):
                    brick_patches[r][c].set_visible(f['bricks'][r, c])
            score_txt.set_text(f"Bricks: {f['score']}/{env.total_bricks}  Lives: {f['lives']}")
            for bar, w in zip(bars, f['probs']):
                bar.set_width(w)
            stats_txt.set_text(
                f"{'═'*20}\n"
                f"Neurons: {N_total:>9,}\n"
                f"Layers:  {4:>9}\n"
                f"Step:    {f['step']:>9}\n"
                f"Spikes:  {f['spikes']:>9,}\n"
                f"{'═'*20}")
            return [*brain_imgs, ball, paddle, score_txt, stats_txt, *bars]

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
        path = os.path.join(OUT, 'nsram_cortical_breakout.mp4')
        writer = animation.FFMpegWriter(fps=20, bitrate=4000, extra_args=['-pix_fmt', 'yuv420p'])
        ani.save(path, writer=writer, dpi=120)
        print(f"  Saved: {path}")
        plt.close()

    # Summary image
    print("  Generating summary...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
    fig.suptitle(f'NS-RAM Cortical Brain Breakout — {N_total:,} Neurons',
                  fontsize=14, fontweight='bold', color='white')
    axes[0].plot(all_scores, 'g-o', markersize=5); axes[0].set_title('Bricks Broken', color='white')
    axes[0].set_facecolor('#0d1117'); axes[0].tick_params(colors='gray'); axes[0].grid(True, alpha=0.2)
    axes[1].plot(all_lives, 'b-s', markersize=5); axes[1].set_title('Lives Remaining', color='white')
    axes[1].set_facecolor('#0d1117'); axes[1].tick_params(colors='gray'); axes[1].grid(True, alpha=0.2)
    win_rate = [s / env.total_bricks for s in all_scores]
    axes[2].plot(win_rate, 'r-^', markersize=5); axes[2].set_title('Completion %', color='white')
    axes[2].set_facecolor('#0d1117'); axes[2].tick_params(colors='gray'); axes[2].grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_cortical_breakout.png'), dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {os.path.join(OUT, 'nsram_cortical_breakout.png')}")

    print(f"\n{'='*65}")
    print(f"  Results: {N_total:,} neurons, 4-layer cortical architecture")
    print(f"  Best: {max(all_scores)} bricks / {env.total_bricks}")
    print(f"  First 5 avg: {np.mean(all_scores[:5]):.1f}, Last 5 avg: {np.mean(all_scores[-5:]):.1f}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
