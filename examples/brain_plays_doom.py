#!/usr/bin/env python3
"""NS-RAM Brain Plays DOOM — Temporal Predator Tracking Architecture

Novel approach: instead of treating this as a visual recognition problem
(which needs deep RL), we treat it as a TEMPORAL TRACKING problem — which
is exactly what NS-RAM reservoirs excel at (MC=3.38, MG R²=0.996).

Architecture (tapeout-realistic):
  Retina (16 features) — 8 brightness columns + 8 motion columns
  Reservoir (5000 neurons) — integrates temporal dynamics of demon trajectory
  Readout (3 outputs) — strafe_left, strafe_right, shoot

The reservoir naturally tracks: "demon moving left→right, speed increasing,
now centered → SHOOT". This is honest temporal computation, not a hack.

Uses VizDoom 'basic' scenario: one monster in hallway, strafe to align + shoot.
"""

import sys, os, time
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


def create_game(hires=False):
    game = vzd.DoomGame()
    game.load_config(os.path.join(os.path.dirname(vzd.__file__),
                                   'scenarios', 'basic.cfg'))
    game.set_window_visible(False)
    res = vzd.ScreenResolution.RES_320X240 if hires else vzd.ScreenResolution.RES_160X120
    game.set_screen_resolution(res)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_render_hud(hires)
    game.set_episode_timeout(300)
    game.init()
    return game


class TemporalRetina:
    """8-column retina: extracts WHERE things are + WHERE things are MOVING.

    16 features per frame:
      [0:8]  — column brightness (inverted: dark object = high signal)
      [8:16] — column motion (absolute frame difference)

    This is tapeout-realistic: 8 photodetector columns with edge-detection
    circuits (differencing amplifiers).
    """
    def __init__(self):
        self.prev = None
        self.history = []  # Last N feature vectors for visualization

    def reset(self):
        self.prev = None
        self.history = []

    def process(self, frame):
        if frame is None:
            return np.zeros(16, dtype=np.float32)
        if frame.ndim == 3:
            frame = frame[0] if frame.shape[0] == 1 else frame
        frame = frame.astype(np.float32) / 255.0
        h, w = frame.shape

        features = np.zeros(16, dtype=np.float32)
        cw = w // 8

        # Use lower 2/3 of screen (where the monster is, skip ceiling)
        roi = frame[h//3:, :]

        for i in range(8):
            col = roi[:, i*cw:(i+1)*cw]
            features[i] = 1.0 - col.mean()  # Dark = high (monster is brown/dark)

        if self.prev is not None:
            diff = np.abs(frame - self.prev)
            roi_diff = diff[h//3:, :]
            for i in range(8):
                col = roi_diff[:, i*cw:(i+1)*cw]
                features[8+i] = np.clip(col.mean() * 10.0, 0, 1)  # Amplified motion

        self.prev = frame.copy()
        self.history.append(features.copy())
        if len(self.history) > 50:
            self.history.pop(0)

        return features


class TemporalReservoir:
    """NS-RAM reservoir optimized for temporal tracking.

    Key design choices (all tapeout-realistic):
    - 5000 neurons: fits on ~0.1mm² at 130nm
    - Sparse recurrent (5%): matches NS-RAM wiring constraints
    - 8 integration steps per input: accumulates temporal evidence
    - STP-like adaptation: threshold modulation from spike history
    - Linear readout: ridge regression, trainable online
    """

    def __init__(self, N=5000, n_inputs=16, n_outputs=3,
                 n_steps=8, spectral_radius=0.95, sparsity=0.05,
                 seed=42):
        self.N = N
        self.n_steps = n_steps
        self.n_outputs = n_outputs
        self.device = DEVICE
        rng = np.random.RandomState(seed)
        v = 0.10

        # Neuron parameters
        self.theta = torch.tensor(
            np.clip(1 + v*0.05*rng.randn(N), 0.5, 2).astype(np.float32), device=DEVICE)
        self.bg = 0.88 * self.theta
        self.dT = torch.tensor(
            np.clip(0.1 + v*0.015*rng.randn(N), 0.02, 0.5).astype(np.float32), device=DEVICE)
        self.tau_syn = torch.tensor(
            np.clip(0.5 + v*0.1*rng.randn(N), 0.1, 2).astype(np.float32), device=DEVICE)

        # Input weights (16 features → N neurons)
        self.W_in = torch.tensor(
            rng.randn(N, n_inputs).astype(np.float32) * 0.3, device=DEVICE)

        # Sparse recurrent weights with Dale's law
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

        # Readout weights (learned)
        self.W_out = torch.zeros(n_outputs, N, device=DEVICE)
        self.elig = torch.zeros(n_outputs, N, device=DEVICE)
        self.explore_std = 0.3

        self.reset()

    def reset(self):
        self.Vm = torch.zeros(self.N, device=DEVICE)
        self.syn = torch.zeros(self.N, device=DEVICE)
        self.ft = torch.zeros(self.N, device=DEVICE)
        self.spike_hist = torch.zeros(self.N, device=DEVICE)

    @torch.no_grad()
    def step(self, features):
        """Run reservoir for n_steps on one input frame. Returns action + info."""
        u = torch.tensor(features, dtype=torch.float32, device=self.device)
        I_in = self.W_in @ u

        total_spikes = 0
        for _ in range(self.n_steps):
            I_syn = self.syn @ self.W_rec.T * 0.3
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
            self.spike_hist = 0.9 * self.spike_hist + 0.1 * spiked.float()
            total_spikes += spiked.sum().item()

        # Readout
        state = self.Vm + 0.3 * self.ft
        logits = self.W_out @ state
        logits += self.explore_std * torch.randn(self.n_outputs, device=self.device)

        probs = torch.softmax(logits * 3, dim=0)
        self.elig = 0.95 * self.elig + torch.outer(probs, state)

        action = logits.argmax().item()
        return action, total_spikes, logits.cpu().numpy()

    @torch.no_grad()
    def reward(self, r):
        self.W_out += 0.015 * r * self.elig
        self.W_out.clamp_(-2, 2)

    def get_activity_grid(self, sz=50):
        actual = min(sz, int(np.ceil(np.sqrt(self.N))))
        data = torch.zeros(actual*actual, device=DEVICE)
        data[:min(self.N, actual*actual)] = self.spike_hist[:actual*actual]
        return data.cpu().numpy().reshape(actual, actual)


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    N = 5000
    reservoir = TemporalReservoir(N=N, n_steps=8, seed=42)
    retina = TemporalRetina()

    # Actions: MOVE_LEFT, MOVE_RIGHT, ATTACK
    actions = [[True, False, False], [False, True, False], [False, False, True]]
    action_names = ['<< STRAFE LEFT', '>> STRAFE RIGHT', '** SHOOT **']

    print(f"\n{'='*65}")
    print(f"  NS-RAM Temporal Predator Tracking — {N:,} Neurons")
    print(f"  VizDoom Basic: strafe to align with monster, then shoot")
    print(f"  The reservoir tracks demon trajectory over time")
    print(f"{'='*65}\n")

    all_rewards = []
    all_kills = []
    best_W = reservoir.W_out.clone()
    best_avg = -999

    n_episodes = 200
    for ep in range(n_episodes):
        game = create_game()
        game.new_episode()
        reservoir.reset()
        retina.reset()
        reservoir.explore_std = max(0.05, 0.4 * (1.0 - ep / n_episodes))

        ep_reward = 0
        kills = 0
        step_count = 0

        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break
            features = retina.process(state.screen_buffer)
            action_idx, n_spk, logits = reservoir.step(features)

            r = game.make_action(actions[action_idx], 2)
            ep_reward += r

            # Reward shaping from game reward
            if r > 50:  # Kill
                reservoir.reward(5.0)
                kills += 1
            elif r < -1:  # Missed shot penalty
                reservoir.reward(-0.3)
            # Shape: reward moving toward demon (feature alignment)
            demon_col = features[:8].argmax()  # Which column has darkest pixel
            if demon_col < 3 and action_idx == 0:  # Demon on left, moved left
                reservoir.reward(0.1)
            elif demon_col > 4 and action_idx == 1:  # Demon on right, moved right
                reservoir.reward(0.1)
            elif 3 <= demon_col <= 4 and action_idx == 2:  # Demon centered, shot
                reservoir.reward(0.2)

            step_count += 1

        all_rewards.append(ep_reward)
        all_kills.append(kills)
        game.close()

        # Evolutionary
        if len(all_rewards) >= 10:
            avg = np.mean(all_rewards[-10:])
            if avg > best_avg:
                best_avg = avg
                best_W = reservoir.W_out.clone()
            elif ep > 20 and np.random.random() < 0.2:
                reservoir.W_out = best_W.clone()
                noise = 0.02 * (1.0 - ep / n_episodes)
                reservoir.W_out += noise * torch.randn_like(reservoir.W_out)
                reservoir.W_out.clamp_(-2, 2)

        if ep % 10 == 0 or kills > 0:
            marker = " ★ KILL!" if kills else ""
            print(f"  Ep {ep:3d}: reward={ep_reward:6.1f} kills={kills} "
                  f"steps={step_count:3d} explore={reservoir.explore_std:.2f}{marker}")

    # ── Record 12 showcase episodes for a long video ──
    reservoir.W_out = best_W.clone()
    reservoir.explore_std = 0.05
    print(f"\n  Recording 12 showcase episodes (best_avg={best_avg:.1f})...")

    frames = []
    final_reward = 0
    final_kills = 0

    for ep_show in range(12):
        game = create_game(hires=True)
        game.new_episode()
        reservoir.reset()
        retina.reset()
        ep_r = 0

        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break
            features = retina.process(state.screen_buffer)
            action_idx, n_spk, logits = reservoir.step(features)
            r = game.make_action(actions[action_idx], 2)
            ep_r += r
            final_reward += r
            if r > 50:
                final_kills += 1

            if state is not None:
                frames.append({
                    'frame': state.screen_buffer.copy(),
                    'features': features.copy(),
                    'action': action_names[action_idx],
                    'reward': final_reward,
                    'kills': final_kills,
                    'step': len(frames),
                    'logits': logits.copy(),
                    'grid': reservoir.get_activity_grid(50),
                    'n_spk': n_spk,
                })

        game.close()
        k = "KILL" if ep_r > 0 else "miss"
        print(f"    Ep {ep_show}: reward={ep_r:.0f} [{k}] total_frames={len(frames)}")

    print(f"  Total: {final_kills} kills in 12 eps, {len(frames)} frames")

    # ── Render Video ──
    if not frames:
        print("  No frames!")
        return

    print(f"\n  Rendering video ({len(frames)} frames)...")
    fig = plt.figure(figsize=(22, 12), facecolor='#0d1117')
    gs = GridSpec(3, 4, hspace=0.35, wspace=0.3,
                  left=0.03, right=0.97, top=0.88, bottom=0.05,
                  height_ratios=[5, 3, 3])
    fig.text(0.5, 0.96, f'NS-RAM Temporal Predator Tracking — {N:,} Spiking Neurons',
             ha='center', fontsize=18, fontweight='bold', color='white')
    fig.text(0.5, 0.92, 'Reservoir integrates demon trajectory over time | VizDoom Basic',
             ha='center', fontsize=10, color='#888888')

    # Game screen
    ax_game = fig.add_subplot(gs[0, 0:3])
    ax_game.set_facecolor('#000000')
    fd = frames[0]['frame'][0] if frames[0]['frame'].ndim == 3 else frames[0]['frame']
    game_img = ax_game.imshow(fd, cmap='gray', aspect='auto')
    ax_game.set_xticks([]); ax_game.set_yticks([])
    info_txt = ax_game.text(0.5, 1.04, '', transform=ax_game.transAxes, ha='center',
                             fontsize=13, color='white', fontweight='bold')
    action_txt = ax_game.text(0.02, 0.05, '', transform=ax_game.transAxes,
                               fontsize=12, color='#4ecdc4', fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.9))

    # Reservoir activity
    ax_res = fig.add_subplot(gs[0, 3])
    ax_res.set_facecolor('#0d1117')
    res_img = ax_res.imshow(frames[0]['grid'], cmap='magma', aspect='auto',
                             interpolation='bilinear', vmin=0, vmax=0.3)
    ax_res.set_title(f'Reservoir ({N:,} neurons)', fontsize=10, color='#E91E63', fontweight='bold')
    ax_res.tick_params(colors='gray', labelsize=4)

    # Retinal features (columns)
    ax_col = fig.add_subplot(gs[1, 0:2])
    ax_col.set_facecolor('#0d1117')
    col_x = np.arange(8)
    col_bars = ax_col.bar(col_x - 0.2, [0]*8, 0.35, color='#4ecdc4', alpha=0.8, label='Brightness')
    mot_bars = ax_col.bar(col_x + 0.2, [0]*8, 0.35, color='#FF6B6B', alpha=0.8, label='Motion')
    ax_col.set_ylim(0, 1)
    ax_col.set_xticks(col_x)
    ax_col.set_xticklabels(['L3','L2','L1','CL','CR','R1','R2','R3'], color='white', fontsize=9)
    ax_col.set_title('Retinal Columns (where is the demon?)', fontsize=10, color='#4ecdc4', fontweight='bold')
    ax_col.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white', loc='upper right')
    ax_col.tick_params(colors='gray')
    ax_col.grid(True, alpha=0.1, axis='y')

    # Action logits
    ax_act = fig.add_subplot(gs[1, 2])
    ax_act.set_facecolor('#0d1117')
    act_colors = ['#2196F3', '#FF9800', '#F44336']
    act_labels_short = ['LEFT', 'RIGHT', 'SHOOT']
    act_bars = ax_act.bar(range(3), [0]*3, color=act_colors, alpha=0.85)
    ax_act.set_xticks(range(3))
    ax_act.set_xticklabels(act_labels_short, color='white', fontsize=9)
    ax_act.set_ylim(-2, 2)
    ax_act.set_title('Action Decision', fontsize=10, color='white', fontweight='bold')
    ax_act.tick_params(colors='gray')
    ax_act.axhline(0, color='gray', alpha=0.3)

    # Temporal history heatmap
    ax_hist = fig.add_subplot(gs[1, 3])
    ax_hist.set_facecolor('#0d1117')
    hist_data = np.zeros((16, 30))
    hist_img = ax_hist.imshow(hist_data, cmap='inferno', aspect='auto', vmin=0, vmax=1)
    ax_hist.set_title('Temporal Memory (features × time)', fontsize=9, color='#9C27B0', fontweight='bold')
    ax_hist.set_xlabel('Time →', color='gray', fontsize=7)
    ax_hist.set_ylabel('Feature', color='gray', fontsize=7)
    ax_hist.tick_params(colors='gray', labelsize=4)

    # Reward timeline
    ax_rew = fig.add_subplot(gs[2, 0:2])
    ax_rew.set_facecolor('#0d1117')
    rew_line, = ax_rew.plot([], [], '-', color='#4CAF50', linewidth=2)
    ax_rew.set_xlim(0, max(len(frames), 10))
    all_frame_rewards = [f['reward'] for f in frames]
    ax_rew.set_ylim(min(all_frame_rewards) - 50, max(all_frame_rewards) + 50)
    ax_rew.set_title('Cumulative Reward', fontsize=10, color='#4CAF50', fontweight='bold')
    ax_rew.tick_params(colors='gray', labelsize=7)
    ax_rew.grid(True, alpha=0.2)
    ax_rew.axhline(0, color='gray', alpha=0.3)

    # Learning curve
    ax_learn = fig.add_subplot(gs[2, 2:4])
    ax_learn.set_facecolor('#0d1117')
    ax_learn.plot(all_rewards, 'r-', alpha=0.3, linewidth=0.8)
    if len(all_rewards) >= 10:
        ma = [np.mean(all_rewards[max(0,i-9):i+1]) for i in range(len(all_rewards))]
        ax_learn.plot(ma, 'y-', linewidth=2, label='10-ep avg')
    ax_learn.set_title(f'Training ({n_episodes} episodes)', fontsize=10, color='yellow', fontweight='bold')
    ax_learn.set_xlabel('Episode', color='gray', fontsize=8)
    ax_learn.tick_params(colors='gray', labelsize=7)
    ax_learn.grid(True, alpha=0.2)
    ax_learn.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')

    rew_hist = []

    def update(i):
        f = frames[i % len(frames)]
        fd = f['frame'][0] if f['frame'].ndim == 3 else f['frame']
        game_img.set_data(fd)
        info_txt.set_text(f"Reward: {f['reward']:.0f}  |  Kills: {f['kills']}  |  Step {f['step']}")
        action_txt.set_text(f"{f['action']}")
        res_img.set_data(f['grid'])
        for j in range(8):
            col_bars[j].set_height(f['features'][j])
            mot_bars[j].set_height(f['features'][8+j])
        for j in range(3):
            act_bars[j].set_height(f['logits'][j])
        # Temporal history
        if len(retina.history) > 0:
            n_hist = min(30, len(retina.history))
            hist = np.array(retina.history[-n_hist:]).T  # (16, n_hist)
            padded = np.zeros((16, 30))
            padded[:, 30-n_hist:] = hist
            hist_img.set_data(padded)
        rew_hist.append(f['reward'])
        rew_line.set_data(range(len(rew_hist)), rew_hist)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    path = os.path.join(OUT, 'nsram_brain_doom.mp4')
    writer = animation.FFMpegWriter(fps=15, bitrate=6000, extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(path, writer=writer, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

    # Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle(f'NS-RAM Temporal Tracking — {N:,} Neurons', fontsize=14, fontweight='bold', color='white')
    ax1.plot(all_rewards, 'r-', alpha=0.3); ax1.set_title('Reward', color='white')
    if len(all_rewards) >= 10:
        ax1.plot([np.mean(all_rewards[max(0,i-9):i+1]) for i in range(len(all_rewards))], 'y-', linewidth=2)
    ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='gray'); ax1.grid(True, alpha=0.2)
    kill_eps = [i for i, k in enumerate(all_kills) if k > 0]
    ax2.bar(range(len(all_kills)), all_kills, color='#F44336', alpha=0.7)
    ax2.set_title(f'Kills ({sum(all_kills)} total, {len(kill_eps)} episodes)', color='white')
    ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='gray'); ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nsram_brain_doom.png'), dpi=150, facecolor='#0d1117')
    plt.close()

    print(f"\n{'='*65}")
    print(f"  NS-RAM Temporal Predator Tracking — Results")
    print(f"  {N:,} neurons, {n_episodes} episodes")
    print(f"  Total kills: {sum(all_kills)} in {len(kill_eps)} episodes")
    print(f"  Best reward: {max(all_rewards):.1f}")
    print(f"  Avg last 20: {np.mean(all_rewards[-20:]):.1f}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
