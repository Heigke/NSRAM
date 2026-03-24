#!/usr/bin/env python3
"""Generate high-quality video of NS-RAM brain playing Pong.

Records EVERY frame (not just every 3rd) for smooth playback.
Dual view: brain activity heatmap + game state side by side.
Shows spike raster strip, action probabilities, and score.
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Pong:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bx = 0.5; self.by = 0.5
        a = np.random.uniform(-0.4, 0.4)
        self.bvx = 0.025 * (1 if np.random.rand() > 0.5 else -1)
        self.bvy = 0.025 * np.sin(a)
        self.py = 0.5; self.score = 0; self.misses = 0
        return self.obs()

    def obs(self):
        return np.array([self.bx*2-1, self.by*2-1, self.bvx/0.025,
                          self.bvy/0.025, self.py*2-1], dtype=np.float64)

    def step(self, action):
        self.py = np.clip(self.py + action * 0.04, 0.12, 0.88)
        self.bx += self.bvx; self.by += self.bvy
        if self.by <= 0 or self.by >= 1:
            self.bvy *= -1; self.by = np.clip(self.by, 0.01, 0.99)
        if self.bx <= 0:
            self.bvx = abs(self.bvx); self.bx = 0.01
        reward = 0
        if self.bx >= 0.92:
            if abs(self.by - self.py) < 0.13:
                self.bvx = -abs(self.bvx); self.bx = 0.91
                self.bvy += (self.by - self.py) * 0.05
                self.score += 1; reward = 1
            elif self.bx >= 1:
                self.misses += 1; reward = -1
                self.bx = 0.3; self.by = np.random.uniform(0.2, 0.8)
                self.bvx = 0.025; self.bvy = np.random.uniform(-0.4, 0.4) * 0.025
        return self.obs(), reward


class Brain:
    def __init__(self, N=50000, k=100):
        self.N = N; rng = np.random.RandomState(42)
        v = 0.10
        self.tau = torch.tensor(np.clip(1+v*0.15*rng.randn(N),.3,3).astype(np.float32), device=DEVICE)
        self.theta = torch.tensor(np.clip(1+v*0.05*rng.randn(N),.5,2).astype(np.float32), device=DEVICE)
        self.t_ref = torch.tensor(np.clip(.05+v*.01*rng.randn(N),.01,.2).astype(np.float32), device=DEVICE)
        self.dT = torch.tensor(np.clip(.1+v*.015*rng.randn(N),.02,.5).astype(np.float32), device=DEVICE)
        bg = np.clip(.88+v*.088*rng.randn(N),.5,1.2).astype(np.float32)
        self.I_bg = torch.tensor(bg * self.theta.cpu().numpy(), device=DEVICE)
        self.U = torch.tensor(np.clip(.01+.003*rng.randn(N),.001,.05).astype(np.float32), device=DEVICE)
        self.tau_rec = torch.tensor(np.clip(15+5*rng.randn(N),3,50).astype(np.float32), device=DEVICE)
        self.tau_fac = torch.tensor(((1-self.U.cpu().numpy())*10).astype(np.float32), device=DEVICE)
        self.tau_syn = torch.tensor(np.clip(.5+.1*rng.randn(N),.1,2).astype(np.float32), device=DEVICE)
        self.W_in = torch.tensor(rng.randn(N,5).astype(np.float32)*0.3, device=DEVICE)

        # Sparse recurrent
        N_exc = int(N*0.8); nsign = np.ones(N, np.float32); nsign[N_exc:] = -1
        nnz = N*k; ri = np.repeat(np.arange(N),k); ci = np.zeros(nnz, np.int64)
        vals = np.zeros(nnz, np.float32)
        for i in range(N):
            t = rng.choice(N,k,replace=False); ci[i*k:(i+1)*k] = t
            vals[i*k:(i+1)*k] = np.abs(rng.randn(k).astype(np.float32)) * nsign[i] * 0.3/np.sqrt(k)
        self.W = torch.sparse_coo_tensor(torch.tensor(np.stack([ri,ci]),device=DEVICE),
                                          torch.tensor(vals,device=DEVICE),(N,N)).coalesce()
        self.W_out = torch.zeros(3,N,device=DEVICE)
        self.elig = torch.zeros(3,N,device=DEVICE)
        self.reset_state()

    def reset_state(self):
        N = self.N
        self.Vm = torch.zeros(N,device=DEVICE)
        self.syn = torch.zeros(N,device=DEVICE)
        self.x_stp = torch.ones(N,device=DEVICE)
        self.u_stp = self.U.clone()
        self.refrac = torch.zeros(N,device=DEVICE)
        self.rate = torch.zeros(N,device=DEVICE)
        self.ft = torch.zeros(N,device=DEVICE)
        self.spike_hist = torch.zeros(N,device=DEVICE)

    @torch.no_grad()
    def step(self, obs):
        N = self.N
        u = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        I_in = self.W_in @ u
        I_syn = torch.sparse.mm(self.W, (self.syn*self.u_stp*self.x_stp).unsqueeze(1)).squeeze() * 0.3
        active = (self.refrac<=0).float()
        leak = -self.Vm/self.tau
        exp_t = self.dT * torch.exp(torch.clamp((self.Vm-self.theta)/self.dT.clamp(min=1e-6),-10,5))
        self.Vm += active*(leak+self.I_bg+I_in+I_syn+exp_t) + active*0.01*torch.randn(N,device=DEVICE)
        self.Vm.clamp_(-2,5)
        spiked = (self.Vm>=self.theta)&(self.refrac<=0)
        if spiked.any():
            self.Vm[spiked]=0; self.refrac[spiked]=self.t_ref[spiked]
            self.syn[spiked]+=1; self.rate[spiked]+=5
            self.u_stp[spiked]+=self.U[spiked]*(1-self.u_stp[spiked])
            self.x_stp[spiked]-=self.u_stp[spiked]*self.x_stp[spiked]
        self.syn *= torch.exp(-1/self.tau_syn)
        self.rate *= 0.95
        self.x_stp += (1-self.x_stp)/self.tau_rec.clamp(min=0.5)
        self.u_stp += (self.U-self.u_stp)/self.tau_fac.clamp(min=0.5)
        self.refrac = (self.refrac-1).clamp(min=0)
        self.ft = 0.8*self.ft + 0.2*self.Vm
        # Exponential moving avg of spikes for viz
        self.spike_hist = 0.9*self.spike_hist + 0.1*spiked.float()

        state = self.Vm + 0.3*self.ft + 0.1*self.x_stp
        logits = self.W_out @ state
        probs = torch.softmax(logits, dim=0)
        act = torch.multinomial(probs,1).item()
        one_hot = torch.zeros(3,device=DEVICE); one_hot[act]=1
        self.elig = 0.95*self.elig + torch.outer(one_hot, state)
        n_spk = spiked.sum().item()
        return act-1, n_spk, probs.cpu().numpy()

    @torch.no_grad()
    def reward(self, r):
        self.W_out += 0.003*r*self.elig
        self.W_out.clamp_(-1,1)

    def activity_grid(self):
        # Show ALL neurons as close to square as possible
        sz = int(np.ceil(np.sqrt(self.N)))  # 224 for 50K
        data = torch.zeros(sz*sz, device=DEVICE)
        data[:self.N] = self.spike_hist
        return data.cpu().numpy().reshape(sz, sz)

    def raster_strip(self, n=200):
        """Top-N most active neurons spike state."""
        return self.spike_hist[:n].cpu().numpy()


def main():
    print(f"Device: {DEVICE}")
    N = 50000

    brain = Brain(N=N)
    env = Pong()

    # Pre-train for 15 episodes so the brain has learned something
    print("Pre-training 15 episodes...")
    for ep in range(15):
        obs = env.reset()
        for _ in range(600):
            act, _, _ = brain.step(obs)
            obs, r = env.step(act)
            if r != 0: brain.reward(r)
        print(f"  Ep {ep}: score={env.score} miss={env.misses}")

    # Now record a full episode for video
    print("\nRecording video episode...")
    obs = env.reset()
    brain.reset_state()

    frames = []
    n_frames = 800

    for step in range(n_frames):
        act, n_spk, probs = brain.step(obs)
        obs, r = env.step(act)
        if r != 0: brain.reward(r)

        frames.append({
            'bx': env.bx, 'by': env.by,
            'py': env.py, 'score': env.score, 'miss': env.misses,
            'brain': brain.activity_grid().copy(),
            'raster': brain.raster_strip(300).copy(),
            'probs': probs.copy(),
            'spikes': n_spk, 'step': step,
        })

    print(f"  Recorded {len(frames)} frames. Score={env.score}, Miss={env.misses}")
    print("  Rendering video...")

    # ═══ Build animation ═══
    fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
    gs = GridSpec(3, 2, height_ratios=[0.08, 0.72, 0.20],
                  width_ratios=[0.55, 0.45],
                  hspace=0.08, wspace=0.08,
                  left=0.04, right=0.96, top=0.92, bottom=0.04)

    fig.text(0.5, 0.97, f'NS-RAM Spiking Brain — {N:,} Neurons Playing Pong',
             ha='center', fontsize=16, fontweight='bold', color='white')

    # Brain activity (main view)
    ax_brain = fig.add_subplot(gs[1, 0])
    ax_brain.set_facecolor('#0d1117')
    brain_img = ax_brain.imshow(frames[0]['brain'], cmap='inferno', aspect='auto',
                                 interpolation='bilinear', vmin=0, vmax=0.3)
    sz = int(np.ceil(np.sqrt(N)))
    ax_brain.set_title(f'Neural Activity ({sz}×{sz} grid — all {N:,} neurons)',
                        fontsize=10, color='white', pad=5)
    ax_brain.tick_params(colors='gray', labelsize=7)
    for spine in ax_brain.spines.values(): spine.set_color('#333')

    # Game view
    ax_game = fig.add_subplot(gs[1, 1])
    ax_game.set_facecolor('#16213e')
    ax_game.set_xlim(-0.02, 1.02); ax_game.set_ylim(-0.02, 1.02)
    ax_game.set_aspect('equal')
    ax_game.set_title('Pong', fontsize=10, color='white', pad=5)
    for spine in ax_game.spines.values(): spine.set_color('#333')
    ax_game.tick_params(colors='gray', labelsize=7)
    # Court lines
    ax_game.axvline(0.5, color='#1a3a5c', linestyle='--', linewidth=1, alpha=0.5)
    for y in np.linspace(0, 1, 20):
        ax_game.plot([0.5], [y], 's', color='#1a3a5c', markersize=2)

    ball = Circle((0.5, 0.5), 0.02, color='#ff6b6b', zorder=5)
    ax_game.add_patch(ball)
    paddle = Rectangle((0.94, 0.37), 0.03, 0.26, color='#4ecdc4', zorder=5)
    ax_game.add_patch(paddle)
    # Ball trail
    trail_dots, = ax_game.plot([], [], 'o', color='#ff6b6b', alpha=0.2, markersize=3)

    score_txt = ax_game.text(0.5, 1.05, '', transform=ax_game.transAxes,
                              ha='center', fontsize=14, color='white', fontweight='bold')
    action_txt = ax_game.text(0.02, 0.02, '', transform=ax_game.transAxes,
                               fontsize=9, color='#aaa', fontfamily='monospace')

    # Raster strip (top)
    ax_raster = fig.add_subplot(gs[0, :])
    ax_raster.set_facecolor('#0d1117')
    raster_data = np.zeros((30, len(frames)))  # 30 neurons × time
    raster_img = ax_raster.imshow(raster_data, cmap='Greens', aspect='auto',
                                   vmin=0, vmax=0.3, interpolation='none')
    ax_raster.set_ylabel('N#', fontsize=8, color='white')
    ax_raster.set_title('Spike Raster (30 neurons)', fontsize=9, color='white', pad=2)
    ax_raster.tick_params(colors='gray', labelsize=6)
    time_line = ax_raster.axvline(0, color='yellow', linewidth=1, alpha=0.7)

    # Action probability bars (bottom)
    ax_prob = fig.add_subplot(gs[2, 0])
    ax_prob.set_facecolor('#0d1117')
    bar_labels = ['↓ Down', '· Stay', '↑ Up']
    bar_colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    bars = ax_prob.barh(range(3), [0.33]*3, color=bar_colors, height=0.6)
    ax_prob.set_xlim(0, 1); ax_prob.set_yticks(range(3))
    ax_prob.set_yticklabels(bar_labels, fontsize=10, color='white')
    ax_prob.set_title('Action Probabilities', fontsize=10, color='white', pad=5)
    ax_prob.tick_params(colors='gray')
    for spine in ax_prob.spines.values(): spine.set_color('#333')

    # Stats (bottom right)
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.set_facecolor('#0d1117')
    ax_stats.axis('off')
    stats_txt = ax_stats.text(0.1, 0.5, '', transform=ax_stats.transAxes,
                               fontsize=12, color='white', fontfamily='monospace',
                               va='center')

    # Pre-fill raster data
    for i, f in enumerate(frames):
        raster_data[:, i] = f['raster'][:30]
    raster_img.set_data(raster_data)

    ball_trail_x, ball_trail_y = [], []

    def update(i):
        f = frames[i]

        # Brain
        brain_img.set_data(f['brain'])

        # Ball
        ball.set_center((f['bx'], f['by']))
        ball_trail_x.append(f['bx']); ball_trail_y.append(f['by'])
        if len(ball_trail_x) > 15:
            ball_trail_x.pop(0); ball_trail_y.pop(0)
        trail_dots.set_data(ball_trail_x, ball_trail_y)

        # Paddle
        paddle.set_y(f['py'] - 0.13)

        # Score
        score_txt.set_text(f"Score: {f['score']}  |  Miss: {f['miss']}")

        # Action probs
        for bar, w in zip(bars, f['probs']):
            bar.set_width(w)

        # Action text
        acts = ['↓', '·', '↑']
        best = np.argmax(f['probs'])
        action_txt.set_text(f"Action: {acts[best]}  Spikes: {f['spikes']//1000}K")

        # Raster timeline
        time_line.set_xdata([i, i])

        # Stats
        hr = f['score']/(f['score']+f['miss']) if (f['score']+f['miss']) > 0 else 0
        stats_txt.set_text(
            f"NS-RAM Brain Stats\n"
            f"{'─'*25}\n"
            f"Neurons:    {N:>10,}\n"
            f"Connections:{N*100:>10,}\n"
            f"Step:       {f['step']:>10}\n"
            f"Spikes:     {f['spikes']:>10,}\n"
            f"Hit Rate:   {hr:>10.0%}\n"
        )

        return brain_img, ball, paddle, trail_dots, score_txt, action_txt, time_line, stats_txt, *bars

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=40, blit=False)

    # Save MP4
    mp4_path = os.path.join(OUT, 'nsram_brain_pong.mp4')
    writer = animation.FFMpegWriter(fps=25, bitrate=3000,
                                     extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(mp4_path, writer=writer, dpi=120)
    print(f"  Saved: {mp4_path}")

    # Also save GIF (lower res)
    gif_path = os.path.join(OUT, 'nsram_brain_pong_hq.gif')
    ani.save(gif_path, writer='pillow', fps=15, dpi=80)
    print(f"  Saved: {gif_path}")

    plt.close()

    # Save a still frame montage
    fig2, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor='#0d1117')
    fig2.suptitle(f'NS-RAM Brain Playing Pong — {N:,} Neurons, Key Moments',
                   fontsize=14, fontweight='bold', color='white')
    for idx, frame_i in enumerate(np.linspace(50, len(frames)-50, 10).astype(int)):
        f = frames[frame_i]
        ax = axes[idx//5, idx%5]
        ax.set_facecolor('#16213e')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.scatter([f['bx']], [f['by']], s=80, c='#ff6b6b', zorder=5)
        ax.plot([0.95, 0.95], [f['py']-0.13, f['py']+0.13], color='#4ecdc4', linewidth=4)
        ax.set_title(f"t={f['step']} | {f['score']}-{f['miss']}",
                      fontsize=9, color='white')
        ax.tick_params(colors='gray', labelsize=5)
    plt.tight_layout()
    still_path = os.path.join(OUT, 'nsram_brain_pong_moments.png')
    plt.savefig(still_path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {still_path}")


if __name__ == '__main__':
    main()
