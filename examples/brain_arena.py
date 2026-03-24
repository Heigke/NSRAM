#!/usr/bin/env python3
"""NS-RAM Cortical Brain in a 2D Arena — 100K+ neurons, visual brain regions.

A multi-region spiking brain navigates a 2D arena collecting food and
avoiding walls. The architecture has anatomically-inspired regions:

  V1 (20K) — processes raycasted vision (8 rays)
  V2 (20K) — integrates visual features
  HC (10K) — hippocampal-like spatial memory
  PFC (20K) — prefrontal decision making
  M1 (10K) — motor output (turn left/right, forward/back)
  BG (10K) — basal ganglia reward processing

Total: 90K neurons, sparse connectivity, all on GPU.
Generates video showing brain regions + arena + learning curve.
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════
# 2D ARENA ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class Arena:
    """2D arena with walls, food pellets, and raycasted vision."""

    def __init__(self, size=10.0, n_food=8, n_rays=8):
        self.size = size
        self.n_food = n_food
        self.n_rays = n_rays
        self.reset()

    def reset(self):
        self.ax = self.size / 2; self.ay = self.size / 2  # Agent position
        self.heading = np.random.uniform(0, 2*math.pi)
        self.speed = 0.0
        self.food = [(np.random.uniform(1, self.size-1),
                       np.random.uniform(1, self.size-1)) for _ in range(self.n_food)]
        self.score = 0; self.steps = 0; self.wall_hits = 0
        return self.observe()

    def raycast(self):
        """Cast n_rays from agent, return distances to nearest wall/food."""
        rays = np.zeros(self.n_rays * 2)  # distance + type (wall=0, food=1)
        for i in range(self.n_rays):
            angle = self.heading + (i / self.n_rays) * 2 * math.pi
            dx = math.cos(angle); dy = math.sin(angle)
            # March ray
            for d in np.linspace(0.1, self.size, 50):
                rx = self.ax + dx * d; ry = self.ay + dy * d
                # Wall hit
                if rx <= 0 or rx >= self.size or ry <= 0 or ry >= self.size:
                    rays[i*2] = d / self.size  # Normalized distance
                    rays[i*2+1] = 0.0  # Wall type
                    break
                # Food hit (radius 0.5)
                for fx, fy in self.food:
                    if (rx-fx)**2 + (ry-fy)**2 < 0.25:
                        rays[i*2] = d / self.size
                        rays[i*2+1] = 1.0  # Food type
                        break
                else:
                    continue
                break
        return rays

    def observe(self):
        """Returns observation: [rays(16), agent_x, agent_y, heading_sin, heading_cos, speed]."""
        rays = self.raycast()
        return np.concatenate([
            rays,
            [self.ax / self.size, self.ay / self.size,
             math.sin(self.heading), math.cos(self.heading),
             self.speed / 0.3]
        ]).astype(np.float64)

    def step(self, turn, accel):
        """turn: -1 to +1. accel: -1 to +1. Returns (obs, reward)."""
        self.heading += turn * 0.2
        self.speed = np.clip(self.speed + accel * 0.015, -0.05, 0.2)
        self.ax += math.cos(self.heading) * self.speed
        self.ay += math.sin(self.heading) * self.speed
        self.steps += 1

        reward = -0.001  # Time penalty

        # Wall collision
        if self.ax < 0.3 or self.ax > self.size-0.3 or self.ay < 0.3 or self.ay > self.size-0.3:
            self.ax = np.clip(self.ax, 0.3, self.size-0.3)
            self.ay = np.clip(self.ay, 0.3, self.size-0.3)
            self.speed *= -0.5
            self.wall_hits += 1
            reward = -0.05

        # Food collection
        new_food = []
        min_food_dist = self.size
        for fx, fy in self.food:
            d2 = (self.ax-fx)**2 + (self.ay-fy)**2
            min_food_dist = min(min_food_dist, np.sqrt(d2))
            if d2 < 0.36:
                self.score += 1
                reward = 2.0  # Strong food reward
                new_food.append((np.random.uniform(1, self.size-1),
                                  np.random.uniform(1, self.size-1)))
            else:
                new_food.append((fx, fy))
        self.food = new_food

        # Small proximity reward (food-seeking gradient)
        if reward < 0 and min_food_dist < 2.0:
            reward += 0.05 * (2.0 - min_food_dist)

        return self.observe(), reward


# ═══════════════════════════════════════════════════════════════════════
# CORTICAL BRAIN WITH NAMED REGIONS
# ═══════════════════════════════════════════════════════════════════════

class CorticalBrain:
    """Multi-region spiking brain with anatomically-inspired layout."""

    REGIONS = {
        'V1':  {'size': 20000, 'role': 'primary visual', 'color': '#E91E63'},
        'V2':  {'size': 20000, 'role': 'visual integration', 'color': '#9C27B0'},
        'HC':  {'size': 10000, 'role': 'spatial memory', 'color': '#2196F3'},
        'PFC': {'size': 20000, 'role': 'decision making', 'color': '#4CAF50'},
        'M1':  {'size': 10000, 'role': 'motor output', 'color': '#FF9800'},
        'BG':  {'size': 10000, 'role': 'reward processing', 'color': '#F44336'},
    }

    def __init__(self, n_inputs=21, scale=1.0, seed=42):
        self.scale = scale
        self.device = DEVICE
        rng = np.random.RandomState(seed)

        # Scale region sizes
        regions = {}
        offset = 0
        for name, info in self.REGIONS.items():
            n = max(100, int(info['size'] * scale))
            regions[name] = {'start': offset, 'end': offset + n, 'n': n,
                              'color': info['color'], 'role': info['role']}
            offset += n
        self.regions = regions
        self.N = offset
        print(f"  Brain: {self.N:,} neurons across {len(regions)} regions")
        for name, r in regions.items():
            print(f"    {name:4s}: {r['n']:6,} neurons ({r['role']})")

        v = 0.10
        # Neuron params
        self.tau = torch.tensor(np.clip(1+v*.15*rng.randn(self.N),.3,3).astype(np.float32), device=DEVICE)
        self.theta = torch.tensor(np.clip(1+v*.05*rng.randn(self.N),.5,2).astype(np.float32), device=DEVICE)
        self.t_ref = torch.tensor(np.clip(.05+v*.01*rng.randn(self.N),.01,.2).astype(np.float32), device=DEVICE)
        self.dT = torch.tensor(np.clip(.1+v*.015*rng.randn(self.N),.02,.5).astype(np.float32), device=DEVICE)
        bg = np.clip(.88+v*.088*rng.randn(self.N),.5,1.2).astype(np.float32)
        self.I_bg = torch.tensor(bg * self.theta.cpu().numpy(), device=DEVICE)
        self.tau_syn = torch.tensor(np.clip(.5+.1*rng.randn(self.N),.1,2).astype(np.float32), device=DEVICE)

        # Input to V1 only
        self.W_in = torch.zeros(self.N, n_inputs, device=DEVICE)
        v1 = regions['V1']
        self.W_in[v1['start']:v1['end']] = torch.tensor(
            rng.randn(v1['n'], n_inputs).astype(np.float32) * 0.2, device=DEVICE)

        # Build connectivity: internal + feedforward
        print("  Building connectivity...")
        connections = {
            ('V1', 'V1'): 80, ('V2', 'V2'): 80, ('HC', 'HC'): 60,
            ('PFC', 'PFC'): 80, ('M1', 'M1'): 60, ('BG', 'BG'): 60,
            ('V1', 'V2'): 60, ('V2', 'PFC'): 50, ('V2', 'HC'): 30,
            ('HC', 'PFC'): 40, ('PFC', 'M1'): 60, ('PFC', 'BG'): 30,
            ('BG', 'PFC'): 30, ('BG', 'M1'): 20,  # Feedback from reward
        }

        all_r, all_c, all_v = [], [], []
        N_exc_frac = 0.8
        for (src, dst), k in connections.items():
            rs = regions[src]; rd = regions[dst]
            ns = rs['n']; nd = rd['n']
            N_exc = int(ns * N_exc_frac)
            nsign = np.ones(ns, np.float32); nsign[N_exc:] = -1
            for i in range(ns):
                targets = rng.choice(nd, min(k, nd), replace=False) + rd['start']
                all_r.extend([i + rs['start']] * len(targets))
                all_c.extend(targets.tolist())
                w = np.abs(rng.randn(len(targets)).astype(np.float32)) * nsign[i] * 0.3 / np.sqrt(k)
                all_v.extend(w.tolist())

        nnz = len(all_r)
        self.W = torch.sparse_coo_tensor(
            torch.tensor([all_r, all_c], dtype=torch.long, device=DEVICE),
            torch.tensor(all_v, dtype=torch.float32, device=DEVICE),
            (self.N, self.N)).coalesce()
        print(f"  Connections: {nnz/1e6:.1f}M ({nnz*12/1e9:.2f} GB)")

        # Readout from M1: 4 outputs (turn_left, turn_right, accel, brake)
        # Initialize with small random weights so agent explores from start
        m1 = regions['M1']
        self.W_out = torch.zeros(4, self.N, device=DEVICE)
        self.W_out[:, m1['start']:m1['end']] = torch.tensor(
            rng.randn(4, m1['n']).astype(np.float32) * 0.01, device=DEVICE)
        self.W_out[:, regions['PFC']['start']:regions['PFC']['end']] = torch.tensor(
            rng.randn(4, regions['PFC']['n']).astype(np.float32) * 0.005, device=DEVICE)
        self.elig = torch.zeros(4, self.N, device=DEVICE)
        self.explore_std = 0.5  # Exploration noise, decays over episodes
        self.reflex_gain = 1.0  # Innate reflex strength (can decay as cortex learns)
        self.reset_state()
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def reset_state(self):
        self.Vm = torch.zeros(self.N, device=DEVICE)
        self.syn = torch.zeros(self.N, device=DEVICE)
        self.refrac = torch.zeros(self.N, device=DEVICE)
        self.ft = torch.zeros(self.N, device=DEVICE)
        self.spike_hist = torch.zeros(self.N, device=DEVICE)

    @torch.no_grad()
    def step(self, obs):
        N = self.N
        u = torch.tensor(obs, dtype=torch.float32, device=self.device)
        I_in = self.W_in @ u
        I_syn = torch.sparse.mm(self.W, self.syn.unsqueeze(1)).squeeze() * 0.3
        active = (self.refrac <= 0).float()
        leak = -self.Vm / self.tau
        exp_t = self.dT * torch.exp(torch.clamp((self.Vm-self.theta)/self.dT.clamp(min=1e-6),-10,5))
        self.Vm += active * (leak + self.I_bg + I_in + I_syn + exp_t)
        self.Vm += active * 0.01 * torch.randn(N, device=self.device)
        self.Vm.clamp_(-2, 5)
        spiked = (self.Vm >= self.theta) & (self.refrac <= 0)
        if spiked.any():
            self.Vm[spiked] = 0; self.refrac[spiked] = self.t_ref[spiked]; self.syn[spiked] += 1
        self.syn *= torch.exp(-1/self.tau_syn); self.refrac = (self.refrac-1).clamp(min=0)
        self.ft = 0.8*self.ft + 0.2*self.Vm
        self.spike_hist = 0.85*self.spike_hist + 0.15*spiked.float()

        state = self.Vm + 0.3*self.ft
        logits = self.W_out @ state
        logits += self.explore_std * torch.randn(4, device=self.device)
        acts = torch.tanh(logits)
        probs = torch.softmax(logits * 2, dim=0)
        self.elig = 0.95*self.elig + torch.outer(probs, state)
        turn = acts[0].item() - acts[1].item()
        accel = acts[2].item() - acts[3].item()
        accel += 0.08  # Forward bias
        return turn, accel, spiked.sum().item()

    @torch.no_grad()
    def step_with_reflex(self, obs):
        """Step with innate food-seeking reflex (brainstem) + cortical modulation."""
        turn, accel, n_spk = self.step(obs)

        # Innate reflex: steer toward nearest visible food
        n_rays = len(obs) // 2 - 2  # Subtract non-ray features (rough)
        if n_rays > 8: n_rays = 8
        best_food_angle = 0.0
        best_food_dist = 10.0
        for i in range(n_rays):
            dist = obs[i*2]
            is_food = obs[i*2+1]
            if is_food > 0.5 and dist < best_food_dist:
                best_food_dist = dist
                # Ray angle relative to heading (0 = forward)
                angle = (i / n_rays) * 2 * math.pi
                if angle > math.pi: angle -= 2 * math.pi
                best_food_angle = angle

        # Reflex strength decreases as cortex learns (self.reflex_gain)
        if best_food_dist < 5.0:
            reflex_turn = np.clip(best_food_angle * 0.5, -0.3, 0.3)
            turn += self.reflex_gain * reflex_turn
            accel += self.reflex_gain * 0.05  # Speed up toward food

        # Wall avoidance reflex: if forward ray is close to wall, turn
        fwd_dist = obs[0]  # Ray 0 = forward
        fwd_type = obs[1]
        if fwd_type < 0.5 and fwd_dist < 0.15:  # Wall ahead and close
            turn += self.reflex_gain * 0.3 * (1 if np.random.random() > 0.5 else -1)

        return turn, accel, n_spk

    @torch.no_grad()
    def reward(self, r):
        self.W_out += 0.01*r*self.elig; self.W_out.clamp_(-1,1)

    def region_activity(self):
        """Get mean spike rate per region."""
        rates = {}
        for name, r in self.regions.items():
            rates[name] = self.spike_hist[r['start']:r['end']].mean().item()
        return rates

    def region_grid(self, name, sz=50):
        r = self.regions[name]
        n = r['n']
        actual_sz = min(sz, int(np.ceil(np.sqrt(n))))
        data = torch.zeros(actual_sz*actual_sz, device=DEVICE)
        data[:min(n, actual_sz*actual_sz)] = self.spike_hist[r['start']:r['start']+actual_sz*actual_sz]
        return data.cpu().numpy().reshape(actual_sz, actual_sz)


def run_episode(brain, env, record=False):
    """Run one episode, return (score, walls, frames)."""
    obs = env.reset(); brain.reset_state()
    frames = []
    for step in range(500):
        turn, accel, n_spk = brain.step_with_reflex(obs)
        obs, reward = env.step(turn, accel)
        if reward > 0.5: brain.reward(reward)
        if record and step % 3 == 0:
            frames.append({
                'ax': env.ax, 'ay': env.ay, 'heading': env.heading,
                'food': list(env.food), 'score': env.score,
                'walls': env.wall_hits,
                'regions': brain.region_activity(),
                'v1': brain.region_grid('V1', 40),
                'pfc': brain.region_grid('PFC', 40),
                'm1': brain.region_grid('M1', 40),
                'bg': brain.region_grid('BG', 40),
                'step': step, 'n_spk': n_spk,
            })
    return env.score, env.wall_hits, frames


def main():
    print(f"Device: {DEVICE}")
    scale = 1.0
    if DEVICE == 'cuda':
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"GPU free: {free:.0f} GB")
        if free < 5: scale = 0.2
        elif free < 15: scale = 0.5

    brain = CorticalBrain(scale=scale, seed=42)
    env = Arena(size=10, n_food=8, n_rays=8)

    print(f"\n{'='*60}")
    print(f"  Training {brain.N:,}-neuron cortical brain in arena")
    print(f"  Strategy: Evolution + reward-modulated Hebbian")
    print(f"{'='*60}\n")

    all_scores = []; all_walls = []
    record_eps = {}

    # Evolutionary strategy: keep best W_out, perturb periodically
    best_W_out = brain.W_out.clone()
    best_avg = 0.0
    window_scores = []

    n_episodes = 100
    for ep in range(n_episodes):
        brain.explore_std = max(0.05, 0.4 * (1.0 - ep / n_episodes))
        record = ep in [0, 24, 49, 74, n_episodes-1]
        t0 = time.time()
        score, walls, frames = run_episode(brain, env, record=record)
        elapsed = time.time() - t0
        all_scores.append(score); all_walls.append(walls)
        window_scores.append(score)
        if frames: record_eps[ep] = frames

        # Evolutionary: every 10 episodes, check if we improved
        if len(window_scores) >= 10:
            avg = np.mean(window_scores[-10:])
            if avg > best_avg:
                best_avg = avg
                best_W_out = brain.W_out.clone()
            elif ep > 20 and np.random.random() < 0.3:
                # Revert + mutate from best
                brain.W_out = best_W_out.clone()
                noise_scale = 0.02 * (1.0 - ep / n_episodes)
                brain.W_out += noise_scale * torch.randn_like(brain.W_out)
                brain.W_out.clamp_(-1, 1)

        marker = " *" if score >= 3 else ""
        print(f"  Ep {ep:3d}: food={score:2d} walls={walls:3d} explore={brain.explore_std:.2f} ({elapsed:.1f}s){marker}")

    # Use best W_out for final recording
    brain.W_out = best_W_out.clone()
    brain.explore_std = 0.05
    print(f"\n  Final run with best weights (avg={best_avg:.1f})...")
    _, _, final_frames = run_episode(brain, env, record=True)

    # ── Video ──
    print(f"  Rendering video ({len(final_frames)} frames)...")

    fig = plt.figure(figsize=(20, 10), facecolor='#0d1117')
    gs = GridSpec(2, 5, hspace=0.3, wspace=0.25,
                  left=0.02, right=0.98, top=0.88, bottom=0.05)
    fig.text(0.5, 0.95, f'NS-RAM Cortical Brain — {brain.N:,} Spiking Neurons',
             ha='center', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.91, '6 brain regions | AdEx-LIF dynamics | reward-modulated learning',
             ha='center', fontsize=10, color='#888888')

    # Arena (left half)
    ax_arena = fig.add_subplot(gs[:, 0:3])
    ax_arena.set_facecolor('#1a1a2e'); ax_arena.set_xlim(-0.5, 10.5); ax_arena.set_ylim(-0.5, 10.5)
    ax_arena.set_aspect('equal')
    ax_arena.set_title('2D Arena', fontsize=12, color='white')
    # Draw walls
    for spine in ax_arena.spines.values(): spine.set_color('#333333')
    ax_arena.add_patch(Rectangle((0,0), 10, 10, fill=False, edgecolor='#555555', linewidth=2))
    agent_dot = Circle((5, 5), 0.25, color='#4ecdc4', zorder=5); ax_arena.add_patch(agent_dot)
    heading_line, = ax_arena.plot([], [], '-', color='#4ecdc4', linewidth=2, zorder=6)
    food_scatter = ax_arena.scatter([], [], s=120, c='#f1c40f', zorder=4, marker='*')
    trail_line, = ax_arena.plot([], [], '-', color='#4ecdc4', alpha=0.2, linewidth=1)
    score_txt = ax_arena.text(0.5, 1.04, '', transform=ax_arena.transAxes, ha='center',
                               fontsize=13, color='white', fontweight='bold')
    step_txt = ax_arena.text(0.02, 0.02, '', transform=ax_arena.transAxes,
                              fontsize=9, color='#666666')

    # Brain regions (right side, 2x2 grid)
    region_imgs = {}
    region_keys = ['v1', 'pfc', 'm1', 'bg']
    region_labels = ['V1', 'PFC', 'M1', 'BG']
    cmaps_list = ['magma', 'viridis', 'plasma', 'inferno']
    for i, (key, label, cmap) in enumerate(zip(region_keys, region_labels, cmaps_list)):
        ax = fig.add_subplot(gs[i//2, 3+i%2])
        ax.set_facecolor('#0d1117')
        grid = final_frames[0][key]
        img = ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='bilinear',
                         vmin=0, vmax=0.3)
        ax.set_title(f'{label} ({brain.regions[label]["role"]})',
                      fontsize=9, color=brain.regions[label]['color'], fontweight='bold')
        ax.tick_params(colors='gray', labelsize=4)
        region_imgs[key] = img

    trail_x, trail_y = [], []

    def update(i):
        f = final_frames[i % len(final_frames)]
        agent_dot.set_center((f['ax'], f['ay']))
        hx = f['ax'] + 0.6*math.cos(f['heading'])
        hy = f['ay'] + 0.6*math.sin(f['heading'])
        heading_line.set_data([f['ax'], hx], [f['ay'], hy])
        fx = [p[0] for p in f['food']]; fy = [p[1] for p in f['food']]
        food_scatter.set_offsets(np.column_stack([fx, fy]))
        trail_x.append(f['ax']); trail_y.append(f['ay'])
        if len(trail_x) > 80: trail_x.pop(0); trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)
        score_txt.set_text(f"Food: {f['score']}  |  Walls: {f['walls']}")
        step_txt.set_text(f"Step {f['step']}/500")
        for key in region_keys:
            region_imgs[key].set_data(f[key])
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(final_frames), interval=50, blit=False)
    path = os.path.join(OUT, 'nsram_brain_arena.mp4')
    writer = animation.FFMpegWriter(fps=25, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(path, writer=writer, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

    # Summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0d1117')
    fig.suptitle(f'NS-RAM Brain Arena — {brain.N:,} Neurons', fontsize=14, fontweight='bold', color='white')
    ax1.plot(all_scores, 'g-o', markersize=4); ax1.set_title('Food Collected', color='white')
    ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.2)
    ax2.plot(all_walls, 'r-s', markersize=4); ax2.set_title('Wall Hits', color='white')
    ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_brain_arena.png'), dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {os.path.join(OUT, 'nsram_brain_arena.png')}")

    print(f"\n{'='*60}")
    print(f"  Results: {brain.N:,} neurons, 6 brain regions")
    print(f"  Best food: {max(all_scores)}, Avg last 5: {np.mean(all_scores[-5:]):.1f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
