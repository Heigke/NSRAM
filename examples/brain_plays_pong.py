#!/usr/bin/env python3
"""NS-RAM Brain Plays Pong — 50K spiking neurons learning to play.

A large-scale NS-RAM reservoir controls a Pong paddle.
The brain receives ball position and paddle position as input,
and the readout layer maps reservoir states to paddle movement.

Training: reward-modulated Hebbian learning on the readout weights.
When the brain scores → strengthen active connections.
When it misses → weaken them.

Generates an animation showing the brain activity alongside the game.

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python examples/brain_plays_pong.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.animation as animation

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU free: {free/1e9:.0f} GB")


# ═══════════════════════════════════════════════════════════════════════
# PONG ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class Pong:
    """Minimal Pong environment. Ball bounces, paddle moves up/down."""

    def __init__(self, width=1.0, height=1.0, ball_speed=0.03, paddle_h=0.2):
        self.w = width; self.h = height
        self.ball_speed = ball_speed; self.paddle_h = paddle_h
        self.reset()

    def reset(self):
        self.ball_x = 0.5; self.ball_y = 0.5
        angle = np.random.uniform(-0.5, 0.5)
        self.ball_vx = self.ball_speed * np.cos(angle)
        self.ball_vy = self.ball_speed * np.sin(angle)
        if np.random.rand() > 0.5: self.ball_vx *= -1
        self.paddle_y = 0.5
        self.score = 0; self.misses = 0
        return self.state()

    def state(self):
        """Returns [ball_x, ball_y, ball_vx, ball_vy, paddle_y] normalized to [-1, 1]."""
        return np.array([
            self.ball_x * 2 - 1,
            self.ball_y * 2 - 1,
            self.ball_vx / self.ball_speed,
            self.ball_vy / self.ball_speed,
            self.paddle_y * 2 - 1,
        ], dtype=np.float64)

    def step(self, action):
        """action: -1 (down), 0 (stay), +1 (up). Returns (state, reward, done)."""
        # Move paddle
        self.paddle_y += action * 0.04
        self.paddle_y = np.clip(self.paddle_y, self.paddle_h/2, 1 - self.paddle_h/2)

        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Bounce off top/bottom
        if self.ball_y <= 0 or self.ball_y >= self.h:
            self.ball_vy *= -1
            self.ball_y = np.clip(self.ball_y, 0.01, self.h - 0.01)

        # Bounce off left wall (opponent side — always returns)
        if self.ball_x <= 0:
            self.ball_vx = abs(self.ball_vx)
            self.ball_x = 0.01

        reward = 0
        done = False

        # Check right wall (our paddle side)
        if self.ball_x >= self.w - 0.05:
            if abs(self.ball_y - self.paddle_y) < self.paddle_h / 2:
                # HIT!
                self.ball_vx = -abs(self.ball_vx)
                self.ball_x = self.w - 0.06
                # Add spin based on where ball hit paddle
                offset = (self.ball_y - self.paddle_y) / (self.paddle_h / 2)
                self.ball_vy += offset * 0.01
                self.score += 1
                reward = 1.0
            elif self.ball_x >= self.w:
                # MISS
                self.misses += 1
                reward = -1.0
                # Reset ball
                self.ball_x = 0.3; self.ball_y = np.random.uniform(0.2, 0.8)
                self.ball_vx = self.ball_speed
                self.ball_vy = np.random.uniform(-0.5, 0.5) * self.ball_speed

        return self.state(), reward, done


# ═══════════════════════════════════════════════════════════════════════
# LARGE-SCALE NS-RAM BRAIN
# ═══════════════════════════════════════════════════════════════════════

class NSRAMBrain:
    """Large-scale NS-RAM spiking reservoir with reward-modulated readout.

    Architecture:
      Input (5) → W_in (N×5) → Reservoir (N neurons, sparse recurrent) → W_out (3×N) → Action

    The reservoir is a fixed random NS-RAM network.
    Only W_out is learned via reward-modulated Hebbian rule.
    """

    def __init__(self, N=50000, n_inputs=5, connectivity_k=100, seed=42):
        self.N = N
        self.seed = seed
        self.device = DEVICE
        rng = np.random.RandomState(seed)
        print(f"  Building {N:,}-neuron brain...")

        # Per-neuron parameters (die-to-die variability)
        var = 0.10
        self.tau = torch.tensor(np.clip(1.0 + var*0.15*rng.randn(N), 0.3, 3).astype(np.float32), device=DEVICE)
        self.theta = torch.tensor(np.clip(1.0 + var*0.05*rng.randn(N), 0.5, 2).astype(np.float32), device=DEVICE)
        self.t_ref = torch.tensor(np.clip(0.05 + var*0.01*rng.randn(N), 0.01, 0.2).astype(np.float32), device=DEVICE)
        self.dT = torch.tensor(np.clip(0.10 + var*0.015*rng.randn(N), 0.02, 0.5).astype(np.float32), device=DEVICE)
        bg = np.clip(0.88 + var*0.088*rng.randn(N), 0.5, 1.2).astype(np.float32)
        self.I_bg = torch.tensor(bg * self.theta.cpu().numpy(), device=DEVICE)

        # STP (tuned: U=0.01, tau_rec=15)
        U_base = np.clip(0.01 + 0.003*rng.randn(N), 0.001, 0.05).astype(np.float32)
        self.U = torch.tensor(U_base, device=DEVICE)
        self.tau_rec = torch.tensor(np.clip(15 + 5*rng.randn(N), 3, 50).astype(np.float32), device=DEVICE)
        self.tau_fac = torch.tensor(((1-U_base)*10).astype(np.float32), device=DEVICE)
        self.alpha_w = 0.30

        # Input weights (N × 5)
        self.W_in = torch.tensor(rng.randn(N, n_inputs).astype(np.float32) * 0.3, device=DEVICE)

        # Sparse recurrent weights (k connections per neuron)
        print(f"  Building sparse connectivity ({connectivity_k} conn/neuron)...")
        k = connectivity_k
        nnz = N * k
        row_idx = np.repeat(np.arange(N), k)
        col_idx = np.zeros(nnz, dtype=np.int64)
        vals = np.zeros(nnz, dtype=np.float32)
        # Dale's law
        N_exc = int(N * 0.8)
        neuron_sign = np.ones(N, dtype=np.float32)
        neuron_sign[N_exc:] = -1.0

        for i in range(N):
            targets = rng.choice(N, k, replace=False)
            col_idx[i*k:(i+1)*k] = targets
            w = np.abs(rng.randn(k).astype(np.float32)) * neuron_sign[i] * 0.3 / np.sqrt(k)
            vals[i*k:(i+1)*k] = w

        indices = torch.tensor(np.stack([row_idx, col_idx]), device=DEVICE)
        values = torch.tensor(vals, device=DEVICE)
        self.W = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
        print(f"  Sparse W: {nnz/1e6:.1f}M connections ({nnz*12/1e9:.2f} GB)")

        # Readout weights (3 actions × N) — THIS is what we learn
        self.W_out = torch.zeros(3, N, device=DEVICE)  # Start at zero
        self.eligibility = torch.zeros(3, N, device=DEVICE)  # Eligibility trace

        # Synaptic decay
        self.tau_syn = torch.tensor(np.clip(0.5 + 0.1*rng.randn(N), 0.1, 2).astype(np.float32), device=DEVICE)

        # State
        self.Vm = torch.zeros(N, device=DEVICE)
        self.syn = torch.zeros(N, device=DEVICE)
        self.x_stp = torch.ones(N, device=DEVICE)
        self.u_stp = self.U.clone()
        self.refrac = torch.zeros(N, device=DEVICE)
        self.rate_est = torch.zeros(N, device=DEVICE)
        self.ft = torch.zeros(N, device=DEVICE)

        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"  Brain ready. GPU memory: {mem_used:.1f} GB")

    @torch.no_grad()
    def step(self, obs, noise_sigma=0.01, lr=0.001):
        """One timestep: observe → spike → act. Returns action (-1, 0, +1)."""
        N = self.N
        u = torch.tensor(obs, dtype=torch.float32, device=self.device)

        I_in = self.W_in @ u
        stp_mod = self.u_stp * self.x_stp * (1 + self.alpha_w * 0)
        syn_input = self.syn * stp_mod
        I_syn = torch.sparse.mm(self.W, syn_input.unsqueeze(1)).squeeze()

        active = (self.refrac <= 0).float()
        leak = -self.Vm / self.tau
        exp_t = self.dT * torch.exp(torch.clamp((self.Vm - self.theta) / self.dT.clamp(min=1e-6), -10, 5))
        self.Vm += active * (leak + self.I_bg + I_in + I_syn * 0.3 + exp_t)
        self.Vm += active * noise_sigma * torch.randn(N, device=self.device)
        self.Vm.clamp_(-2, 5)

        spiked = (self.Vm >= self.theta) & (self.refrac <= 0)
        if spiked.any():
            self.Vm[spiked] = 0
            self.refrac[spiked] = self.t_ref[spiked]
            self.syn[spiked] += 1
            self.rate_est[spiked] += 5
            self.u_stp[spiked] += self.U[spiked] * (1 - self.u_stp[spiked])
            self.x_stp[spiked] -= self.u_stp[spiked] * self.x_stp[spiked]

        self.syn *= torch.exp(-1 / self.tau_syn)
        self.rate_est *= 0.95
        self.x_stp += (1 - self.x_stp) / self.tau_rec.clamp(min=0.5)
        self.u_stp += (self.U - self.u_stp) / self.tau_fac.clamp(min=0.5)
        self.refrac = (self.refrac - 1).clamp(min=0)
        self.ft = 0.8 * self.ft + 0.2 * self.Vm

        # Readout: state → action logits
        state_vec = self.Vm + 0.3 * self.ft + 0.1 * self.x_stp
        logits = self.W_out @ state_vec  # (3,)

        # Softmax → action
        probs = torch.softmax(logits, dim=0)
        action_idx = torch.multinomial(probs, 1).item()

        # Update eligibility trace (what was active when action was chosen)
        one_hot = torch.zeros(3, device=self.device)
        one_hot[action_idx] = 1.0
        self.eligibility = 0.95 * self.eligibility + torch.outer(one_hot, state_vec)

        spike_count = spiked.sum().item()
        return action_idx - 1, spike_count, probs.cpu().numpy()  # -1, 0, +1

    @torch.no_grad()
    def reward(self, r, lr=0.002):
        """Apply reward signal to update readout weights."""
        self.W_out += lr * r * self.eligibility
        self.W_out.clamp_(-1, 1)

    def get_activity_grid(self, grid_size=100):
        """Get neuron activity as 2D grid for visualization."""
        rates = self.rate_est[:grid_size*grid_size].cpu().numpy()
        return rates.reshape(grid_size, grid_size)

    def get_spike_count(self):
        return (self.rate_est > 0.1).sum().item()


# ═══════════════════════════════════════════════════════════════════════
# TRAINING + ANIMATION
# ═══════════════════════════════════════════════════════════════════════

def train_and_record(N=50000, n_episodes=20, steps_per_episode=500):
    """Train brain on Pong and record data for animation."""

    brain = NSRAMBrain(N=N, connectivity_k=100)
    env = Pong(ball_speed=0.025, paddle_h=0.25)

    all_scores = []
    all_miss = []
    episode_data = []  # For animation

    print(f"\n{'='*60}")
    print(f"  Training {N:,}-neuron NS-RAM brain on Pong")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0; ep_spikes = 0
        frames = []

        t0 = time.time()
        for step in range(steps_per_episode):
            action, n_spk, probs = brain.step(obs, noise_sigma=0.01)
            obs, reward, done = env.step(action)
            ep_spikes += n_spk

            if reward != 0:
                brain.reward(reward, lr=0.003)
                ep_reward += reward

            # Record frames for animation (every 5th step)
            if ep in [0, n_episodes//4, n_episodes//2, 3*n_episodes//4, n_episodes-1]:
                if step % 3 == 0:
                    frames.append({
                        'ball': (env.ball_x, env.ball_y),
                        'paddle': env.paddle_y,
                        'activity': brain.get_activity_grid(80),
                        'probs': probs.copy(),
                        'score': env.score,
                        'misses': env.misses,
                    })

        elapsed = time.time() - t0
        all_scores.append(env.score)
        all_miss.append(env.misses)

        if frames:
            episode_data.append({'ep': ep, 'frames': frames})

        hit_rate = env.score / max(env.score + env.misses, 1)
        print(f"  Ep {ep:3d}: score={env.score:3d} miss={env.misses:3d} "
              f"hit_rate={hit_rate:.0%} spikes={ep_spikes/1e6:.1f}M "
              f"({elapsed:.1f}s, {steps_per_episode/elapsed:.0f} steps/s)")

    return brain, all_scores, all_miss, episode_data


def make_summary_figure(scores, misses, episode_data, N):
    """Create summary figure showing training progress + brain snapshots."""

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.97, top=0.93, bottom=0.04)

    fig.text(0.5, 0.97, f'NS-RAM Brain Plays Pong — {N:,} Spiking Neurons on GPU',
             ha='center', fontsize=18, fontweight='bold')

    # Row 1: Training curves
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(scores, 'g-o', linewidth=2, markersize=5, label='Hits')
    ax1.plot(misses, 'r-s', linewidth=2, markersize=5, label='Misses')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Count')
    ax1.set_title('(A) Training Progress', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2:4])
    hit_rates = [s / max(s + m, 1) for s, m in zip(scores, misses)]
    ax2.plot(hit_rates, 'b-o', linewidth=2.5, markersize=6)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Hit Rate')
    ax2.set_title('(B) Learning Curve', fontsize=13, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    if len(hit_rates) > 1:
        ax2.annotate(f'{hit_rates[-1]:.0%}', xy=(len(hit_rates)-1, hit_rates[-1]),
                     textcoords="offset points", xytext=(-20, 10), fontsize=14,
                     fontweight='bold', color='blue')

    # Row 2-3: Brain snapshots from different episodes
    if episode_data:
        for idx, ep_data in enumerate(episode_data[:4]):
            ep = ep_data['ep']
            frames = ep_data['frames']
            if not frames:
                continue

            # Pick a mid-episode frame
            frame = frames[len(frames)//2]

            # Brain activity
            ax = fig.add_subplot(gs[1, idx])
            im = ax.imshow(frame['activity'], cmap='hot', aspect='auto',
                           interpolation='bilinear', vmin=0, vmax=5)
            ax.set_title(f'(C{idx+1}) Ep {ep}: Brain Activity\n'
                         f'Score={frame["score"]}, Miss={frame["misses"]}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Neuron column'); ax.set_ylabel('Neuron row')

            # Game state
            ax = fig.add_subplot(gs[2, idx])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            # Ball
            bx, by = frame['ball']
            ax.scatter([bx], [by], s=100, c='red', zorder=5)
            # Paddle
            py = frame['paddle']
            ax.plot([0.95, 0.95], [py - 0.125, py + 0.125], 'b-', linewidth=6)
            # Net
            ax.axvline(0.5, color='gray', linestyle=':', alpha=0.3)
            ax.set_title(f'(D{idx+1}) Game State', fontsize=10, fontweight='bold')
            ax.set_facecolor('#1a1a2e')
            # Action probs
            probs = frame['probs']
            ax.text(0.05, 0.95, f'↑{probs[2]:.0%} ·{probs[1]:.0%} ↓{probs[0]:.0%}',
                    transform=ax.transAxes, fontsize=8, color='white', va='top',
                    fontfamily='monospace')

    plt.savefig(os.path.join(OUT, 'nsram_brain_pong.png'), dpi=150, facecolor='white')
    plt.close()
    print(f"\n  Saved: {os.path.join(OUT, 'nsram_brain_pong.png')}")


def make_animation(episode_data, N):
    """Create MP4 animation of brain playing pong."""
    if not episode_data or not episode_data[-1]['frames']:
        print("  No frames to animate")
        return

    frames = episode_data[-1]['frames']  # Last episode
    ep = episode_data[-1]['ep']

    fig, (ax_brain, ax_game) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'NS-RAM Brain ({N:,} neurons) — Episode {ep}', fontsize=14, fontweight='bold')

    # Initial frame
    brain_img = ax_brain.imshow(frames[0]['activity'], cmap='hot', aspect='auto',
                                 interpolation='bilinear', vmin=0, vmax=5)
    ax_brain.set_title('Brain Activity'); ax_brain.set_xlabel('Column'); ax_brain.set_ylabel('Row')

    ax_game.set_xlim(0, 1); ax_game.set_ylim(0, 1); ax_game.set_aspect('equal')
    ax_game.set_facecolor('#1a1a2e')
    ball_dot, = ax_game.plot([], [], 'ro', markersize=10)
    paddle_line, = ax_game.plot([], [], 'b-', linewidth=6)
    score_text = ax_game.text(0.05, 0.95, '', transform=ax_game.transAxes,
                               color='white', fontsize=12, va='top')

    def update(i):
        f = frames[i % len(frames)]
        brain_img.set_data(f['activity'])
        bx, by = f['ball']
        ball_dot.set_data([bx], [by])
        py = f['paddle']
        paddle_line.set_data([0.95, 0.95], [py-0.125, py+0.125])
        score_text.set_text(f"Score: {f['score']}  Miss: {f['misses']}")
        return brain_img, ball_dot, paddle_line, score_text

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=50, blit=True)
    path = os.path.join(OUT, 'nsram_brain_pong.gif')
    ani.save(path, writer='pillow', fps=20)
    plt.close()
    print(f"  Saved animation: {path}")


def main():
    # Scale based on available GPU memory
    free_gb = 10  # Conservative default
    if DEVICE == 'cuda':
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9

    # Size the brain to use ~60% of free memory
    # Each neuron needs ~200 bytes state + sparse weights
    if free_gb > 20:
        N = 50000
    elif free_gb > 10:
        N = 20000
    else:
        N = 5000

    print(f"Free GPU: {free_gb:.0f} GB → using N={N:,} neurons")

    brain, scores, misses, ep_data = train_and_record(
        N=N, n_episodes=30, steps_per_episode=600)

    make_summary_figure(scores, misses, ep_data, N)
    make_animation(ep_data, N)

    # Final stats
    print(f"\n{'='*60}")
    print(f"  Final Results ({N:,} neurons)")
    print(f"{'='*60}")
    print(f"  First 5 episodes avg hit rate: {np.mean([s/(s+m+1e-9) for s,m in zip(scores[:5], misses[:5])]):.0%}")
    print(f"  Last 5 episodes avg hit rate:  {np.mean([s/(s+m+1e-9) for s,m in zip(scores[-5:], misses[-5:])]):.0%}")
    print(f"  Best episode: score={max(scores)}, hit_rate={max(scores)/(max(scores)+misses[scores.index(max(scores))]+1e-9):.0%}")


if __name__ == '__main__':
    main()
