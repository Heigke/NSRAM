#!/usr/bin/env python3
"""NS-RAM Brain Plays DOOM — 100K spiking neurons vs demons.

Uses VizDoom's 'defend_the_center' scenario: the agent stands in the center
of a circular arena while demons approach from all sides. Must turn and shoot.

Architecture:
  V1  (25K) — visual cortex, processes 84×84 grayscale frames
  V2  (15K) — feature integration, motion detection
  PFC (25K) — prefrontal decision making
  M1  (15K) — motor cortex: turn_left, turn_right, shoot
  BG  (10K) — basal ganglia: reward processing
  SC  (10K) — superior colliculus: threat detection + orientation

Total: 100K neurons, AdEx-LIF dynamics, reward-modulated learning.
Generates video showing brain regions + Doom gameplay.
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import vizdoom as vzd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_doom_env(visible=False):
    """Create VizDoom defend_the_center environment."""
    game = vzd.DoomGame()
    game.load_config(os.path.join(os.path.dirname(vzd.__file__),
                                   'scenarios', 'defend_the_center.cfg'))
    game.set_window_visible(visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_episode_timeout(2100)  # ~60 seconds at 35fps
    game.init()
    return game


def preprocess_frame(frame):
    """Downsample 160×120 → 40×30 = 1200 features, normalize to [0,1]."""
    if frame is None:
        return np.zeros(1200, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[0] if frame.shape[0] == 1 else frame.mean(axis=2)
    # Simple block-average downsampling
    h, w = frame.shape
    bh, bw = h // 30, w // 40
    small = frame[:30*bh, :40*bw].reshape(30, bh, 40, bw).mean(axis=(1, 3))
    return (small.ravel() / 255.0).astype(np.float32)


class DoomBrain:
    """100K-neuron spiking brain for FPS gameplay."""

    REGIONS = {
        'V1':  {'size': 25000, 'color': '#E91E63', 'role': 'visual cortex'},
        'V2':  {'size': 15000, 'color': '#9C27B0', 'role': 'feature integration'},
        'PFC': {'size': 25000, 'color': '#4CAF50', 'role': 'decision making'},
        'M1':  {'size': 15000, 'color': '#FF9800', 'role': 'motor cortex'},
        'BG':  {'size': 10000, 'color': '#F44336', 'role': 'reward processing'},
        'SC':  {'size': 10000, 'color': '#2196F3', 'role': 'threat detection'},
    }

    def __init__(self, n_inputs=1200, n_actions=3, scale=1.0, seed=42):
        self.device = DEVICE
        self.n_actions = n_actions  # turn_left, turn_right, shoot
        rng = np.random.RandomState(seed)

        # Build regions
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
        # Neuron parameters (AdEx-LIF)
        self.theta = torch.tensor(
            np.clip(1 + v*0.05*rng.randn(self.N), 0.5, 2).astype(np.float32), device=DEVICE)
        self.bg = 0.88 * self.theta
        self.dT = torch.tensor(
            np.clip(0.1 + v*0.015*rng.randn(self.N), 0.02, 0.5).astype(np.float32), device=DEVICE)
        self.tau_syn = torch.tensor(
            np.clip(0.5 + v*0.1*rng.randn(self.N), 0.1, 2).astype(np.float32), device=DEVICE)

        # Input weights: visual input → V1 only
        v1 = regions['V1']
        self.W_in = torch.zeros(self.N, n_inputs, device=DEVICE)
        self.W_in[v1['start']:v1['end']] = torch.tensor(
            rng.randn(v1['n'], n_inputs).astype(np.float32) * 0.15, device=DEVICE)

        # Inter-region connectivity (sparse, feedforward + feedback)
        print("  Building connectivity...")
        connections = {
            ('V1', 'V1'): 60, ('V2', 'V2'): 60, ('PFC', 'PFC'): 80,
            ('M1', 'M1'): 40, ('BG', 'BG'): 40, ('SC', 'SC'): 40,
            ('V1', 'V2'): 50, ('V1', 'SC'): 30,     # Visual → integration + threat
            ('V2', 'PFC'): 40, ('SC', 'PFC'): 40,    # Features + threat → decision
            ('PFC', 'M1'): 50, ('PFC', 'BG'): 20,    # Decision → motor + reward
            ('BG', 'PFC'): 25, ('BG', 'M1'): 15,     # Reward modulates decision + motor
            ('SC', 'M1'): 20,                          # Fast threat → motor (reflex)
        }

        all_r, all_c, all_v = [], [], []
        for (src, dst), k in connections.items():
            rs, rd = regions[src], regions[dst]
            ns, nd = rs['n'], rd['n']
            N_exc = int(ns * 0.8)
            signs = np.ones(ns, np.float32)
            signs[N_exc:] = -1
            for i in range(ns):
                targets = rng.choice(nd, min(k, nd), replace=False) + rd['start']
                all_r.extend([i + rs['start']] * len(targets))
                all_c.extend(targets.tolist())
                w = np.abs(rng.randn(len(targets)).astype(np.float32)) * signs[i] * 0.3 / np.sqrt(k)
                all_v.extend(w.tolist())

        self.W = torch.sparse_coo_tensor(
            torch.tensor([all_r, all_c], dtype=torch.long, device=DEVICE),
            torch.tensor(all_v, dtype=torch.float32, device=DEVICE),
            (self.N, self.N)).coalesce()
        nnz = len(all_r)
        print(f"  Connections: {nnz/1e6:.1f}M")

        # Readout from M1 → 3 actions
        m1 = regions['M1']
        self.W_out = torch.zeros(n_actions, self.N, device=DEVICE)
        self.W_out[:, m1['start']:m1['end']] = torch.tensor(
            rng.randn(n_actions, m1['n']).astype(np.float32) * 0.01, device=DEVICE)
        # Also read from SC (fast reflexive path)
        sc = regions['SC']
        self.W_out[:, sc['start']:sc['end']] = torch.tensor(
            rng.randn(n_actions, sc['n']).astype(np.float32) * 0.005, device=DEVICE)

        self.elig = torch.zeros(n_actions, self.N, device=DEVICE)
        self.explore_std = 0.3
        self.reset_state()
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def reset_state(self):
        self.Vm = torch.zeros(self.N, device=DEVICE)
        self.syn = torch.zeros(self.N, device=DEVICE)
        self.ft = torch.zeros(self.N, device=DEVICE)
        self.spike_hist = torch.zeros(self.N, device=DEVICE)

    @torch.no_grad()
    def step(self, obs):
        """One brain step. obs: (1200,) float32. Returns action index + spike count."""
        u = torch.tensor(obs, dtype=torch.float32, device=self.device)
        I_in = self.W_in @ u
        I_syn = torch.sparse.mm(self.W, self.syn.unsqueeze(1)).squeeze() * 0.3

        leak = -self.Vm
        exp_t = self.dT * torch.exp(
            torch.clamp((self.Vm - self.theta) / self.dT.clamp(min=1e-6), -10, 5))
        self.Vm = self.Vm + leak + self.bg + I_in + I_syn + exp_t
        self.Vm += 0.01 * torch.randn(self.N, device=self.device)
        self.Vm.clamp_(-2, 5)

        spiked = self.Vm >= self.theta
        if spiked.any():
            self.Vm[spiked] = 0
            self.syn[spiked] += 1
        self.syn *= torch.exp(-1.0 / self.tau_syn)
        self.ft = 0.8 * self.ft + 0.2 * self.Vm
        self.spike_hist = 0.85 * self.spike_hist + 0.15 * spiked.float()

        state = self.Vm + 0.3 * self.ft
        logits = self.W_out @ state
        logits += self.explore_std * torch.randn(self.n_actions, device=self.device)

        probs = torch.softmax(logits * 3, dim=0)
        self.elig = 0.95 * self.elig + torch.outer(probs, state)

        action = logits.argmax().item()
        return action, spiked.sum().item()

    @torch.no_grad()
    def reward(self, r):
        self.W_out += 0.008 * r * self.elig
        self.W_out.clamp_(-1, 1)

    def region_activity(self):
        rates = {}
        for name, r in self.regions.items():
            rates[name] = self.spike_hist[r['start']:r['end']].mean().item()
        return rates

    def region_grid(self, name, sz=40):
        r = self.regions[name]
        n = r['n']
        actual_sz = min(sz, int(np.ceil(np.sqrt(n))))
        data = torch.zeros(actual_sz * actual_sz, device=DEVICE)
        data[:min(n, actual_sz*actual_sz)] = self.spike_hist[r['start']:r['start']+actual_sz*actual_sz]
        return data.cpu().numpy().reshape(actual_sz, actual_sz)


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"GPU free: {free:.0f} GB")

    scale = 1.0
    if DEVICE == 'cpu' or (DEVICE == 'cuda' and free < 10):
        scale = 0.3
        print(f"  Reduced scale: {scale}")

    brain = DoomBrain(scale=scale, seed=42)

    print(f"\n{'='*65}")
    print(f"  NS-RAM Brain vs DOOM — {brain.N:,} Spiking Neurons")
    print(f"  Scenario: Defend the Center (demons approaching from all sides)")
    print(f"{'='*65}\n")

    # Actions: [TURN_LEFT, TURN_RIGHT, ATTACK]
    actions = [[True, False, False], [False, True, False], [False, False, True]]
    action_names = ['TURN_LEFT', 'TURN_RIGHT', 'SHOOT']

    all_kills = []
    all_survived = []
    record_eps = {}
    best_W_out = brain.W_out.clone()
    best_avg_kills = 0.0

    n_episodes = 50
    for ep in range(n_episodes):
        game = create_doom_env(visible=False)
        game.new_episode()
        brain.reset_state()
        brain.explore_std = max(0.05, 0.3 * (1.0 - ep / n_episodes))

        kills = 0
        frames_data = []
        prev_vars = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        step_count = 0
        t0 = time.time()

        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break

            obs = preprocess_frame(state.screen_buffer)
            action_idx, n_spk = brain.step(obs)

            # Multi-step: repeat action for 4 frames (frame skip)
            reward = game.make_action(actions[action_idx], 4)

            # Check kills
            if not game.is_episode_finished():
                new_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                if new_kills > prev_vars:
                    brain.reward(2.0)  # Strong reward for kill
                    kills += int(new_kills - prev_vars)
                    prev_vars = new_kills

            # Survival reward (small positive for staying alive)
            if not game.is_episode_finished() and step_count % 10 == 0:
                brain.reward(0.05)

            # Record frames for video
            record = ep in [0, n_episodes//4, n_episodes//2, 3*n_episodes//4, n_episodes-1]
            if record and step_count % 2 == 0 and state is not None:
                frames_data.append({
                    'frame': state.screen_buffer.copy() if state.screen_buffer is not None else None,
                    'action': action_names[action_idx],
                    'kills': kills,
                    'step': step_count,
                    'regions': brain.region_activity(),
                    'v1': brain.region_grid('V1', 40),
                    'pfc': brain.region_grid('PFC', 40),
                    'sc': brain.region_grid('SC', 30),
                    'n_spk': n_spk,
                })

            step_count += 1

        survived = step_count * 4  # Frames survived (with frame skip)
        elapsed = time.time() - t0
        all_kills.append(kills)
        all_survived.append(survived)
        if frames_data:
            record_eps[ep] = frames_data
        game.close()

        # Evolutionary: track best
        if len(all_kills) >= 5:
            avg = np.mean(all_kills[-5:])
            if avg > best_avg_kills:
                best_avg_kills = avg
                best_W_out = brain.W_out.clone()
            elif ep > 10 and np.random.random() < 0.2:
                brain.W_out = best_W_out.clone()
                noise = 0.015 * (1.0 - ep / n_episodes)
                brain.W_out += noise * torch.randn_like(brain.W_out)
                brain.W_out.clamp_(-1, 1)

        marker = " ★" if kills >= 3 else ""
        print(f"  Ep {ep:3d}: kills={kills:2d} survived={survived:4d}f "
              f"explore={brain.explore_std:.2f} ({elapsed:.1f}s){marker}")

    # ── Final run with best weights ──
    brain.W_out = best_W_out.clone()
    brain.explore_std = 0.05
    print(f"\n  Final run with best weights (avg_kills={best_avg_kills:.1f})...")
    game = create_doom_env(visible=False)
    game.new_episode()
    brain.reset_state()
    final_frames = []
    final_kills = 0
    prev_vars = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    step_count = 0

    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break
        obs = preprocess_frame(state.screen_buffer)
        action_idx, n_spk = brain.step(obs)
        game.make_action(actions[action_idx], 4)

        if not game.is_episode_finished():
            new_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            if new_kills > prev_vars:
                brain.reward(2.0)
                final_kills += int(new_kills - prev_vars)
                prev_vars = new_kills

        if step_count % 2 == 0 and state is not None:
            final_frames.append({
                'frame': state.screen_buffer.copy() if state.screen_buffer is not None else None,
                'action': action_names[action_idx],
                'kills': final_kills,
                'step': step_count,
                'regions': brain.region_activity(),
                'v1': brain.region_grid('V1', 40),
                'pfc': brain.region_grid('PFC', 40),
                'sc': brain.region_grid('SC', 30),
            })
        step_count += 1

    print(f"  Final: {final_kills} kills, {step_count*4} frames survived")
    game.close()

    # ── Render Video ──
    if not final_frames:
        print("  No frames to render!")
        return

    print(f"\n  Rendering video ({len(final_frames)} frames)...")

    fig = plt.figure(figsize=(20, 10), facecolor='#0d1117')
    gs = GridSpec(2, 4, hspace=0.3, wspace=0.25,
                  left=0.02, right=0.98, top=0.88, bottom=0.05)
    fig.text(0.5, 0.95, f'NS-RAM Brain vs DOOM — {brain.N:,} Spiking Neurons',
             ha='center', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.91, 'Defend the Center | AdEx-LIF dynamics | 6 brain regions',
             ha='center', fontsize=10, color='#888888')

    # Game screen (left)
    ax_game = fig.add_subplot(gs[:, 0:2])
    ax_game.set_facecolor('#0d1117')
    ax_game.set_title('DOOM: Defend the Center', fontsize=12, color='white')
    f0 = final_frames[0]
    if f0['frame'] is not None and f0['frame'].ndim >= 2:
        frame_display = f0['frame'][0] if f0['frame'].ndim == 3 else f0['frame']
    else:
        frame_display = np.zeros((120, 160), dtype=np.uint8)
    game_img = ax_game.imshow(frame_display, cmap='gray', aspect='auto')
    ax_game.tick_params(colors='gray', labelsize=5)
    info_txt = ax_game.text(0.5, 1.04, '', transform=ax_game.transAxes, ha='center',
                             fontsize=13, color='white', fontweight='bold')
    action_txt = ax_game.text(0.02, 0.02, '', transform=ax_game.transAxes,
                               fontsize=10, color='#4ecdc4',
                               bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    # Brain regions (right, 2×2)
    region_imgs = {}
    rkeys = ['v1', 'pfc', 'sc']
    rlabels = ['V1', 'PFC', 'SC']
    cmaps = ['magma', 'viridis', 'inferno']
    for i, (key, label, cmap) in enumerate(zip(rkeys, rlabels, cmaps)):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, 2 + col])
        ax.set_facecolor('#0d1117')
        grid = final_frames[0][key]
        img = ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='bilinear',
                         vmin=0, vmax=0.3)
        ax.set_title(f'{label} ({brain.regions[label]["role"]})',
                      fontsize=9, color=brain.regions[label]['color'], fontweight='bold')
        ax.tick_params(colors='gray', labelsize=4)
        region_imgs[key] = img

    # Kill counter plot
    ax_kills = fig.add_subplot(gs[1, 3])
    ax_kills.set_facecolor('#0d1117')
    ax_kills.set_title('Kill Timeline', fontsize=9, color='#F44336', fontweight='bold')
    kill_line, = ax_kills.plot([], [], '-', color='#F44336', linewidth=2)
    ax_kills.set_xlim(0, len(final_frames))
    ax_kills.set_ylim(0, max(final_kills + 1, 5))
    ax_kills.set_xlabel('Step', color='gray', fontsize=8)
    ax_kills.set_ylabel('Kills', color='gray', fontsize=8)
    ax_kills.tick_params(colors='gray', labelsize=6)
    ax_kills.grid(True, alpha=0.2)

    kill_history = []

    def update(i):
        f = final_frames[i % len(final_frames)]
        if f['frame'] is not None:
            fd = f['frame'][0] if f['frame'].ndim == 3 else f['frame']
            game_img.set_data(fd)
        info_txt.set_text(f"Kills: {f['kills']}  |  Step: {f['step']}")
        action_txt.set_text(f"Action: {f['action']}")
        for key in rkeys:
            region_imgs[key].set_data(f[key])
        kill_history.append(f['kills'])
        kill_line.set_data(range(len(kill_history)), kill_history)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(final_frames),
                                   interval=80, blit=False)
    path = os.path.join(OUT, 'nsram_brain_doom.mp4')
    writer = animation.FFMpegWriter(fps=15, bitrate=5000,
                                     extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(path, writer=writer, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

    # Summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle(f'NS-RAM Brain vs DOOM — {brain.N:,} Neurons',
                 fontsize=14, fontweight='bold', color='white')
    ax1.plot(all_kills, 'r-o', markersize=4, linewidth=1.5)
    ax1.set_title('Kills per Episode', color='white', fontsize=11)
    ax1.set_xlabel('Episode', color='gray'); ax1.set_ylabel('Kills', color='gray')
    ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.2)

    # Moving average
    if len(all_kills) >= 5:
        ma = [np.mean(all_kills[max(0,i-4):i+1]) for i in range(len(all_kills))]
        ax1.plot(ma, 'y-', linewidth=2, alpha=0.7, label='5-ep avg')
        ax1.legend(facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    ax2.plot(all_survived, 'g-s', markersize=4, linewidth=1.5)
    ax2.set_title('Frames Survived', color='white', fontsize=11)
    ax2.set_xlabel('Episode', color='gray'); ax2.set_ylabel('Frames', color='gray')
    ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_brain_doom.png'), dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {os.path.join(OUT, 'nsram_brain_doom.png')}")

    print(f"\n{'='*65}")
    print(f"  NS-RAM Brain vs DOOM — Results")
    print(f"  {brain.N:,} neurons, 6 brain regions, {n_episodes} episodes")
    print(f"  Best kills: {max(all_kills)}")
    print(f"  Avg last 10: {np.mean(all_kills[-10:]):.1f} kills")
    print(f"  Max survived: {max(all_survived)} frames")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
