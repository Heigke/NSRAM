#!/usr/bin/env python3
"""Generate showcase videos: learning rules side-by-side + large brain.

Video 1: Split-screen — V-STDP brain vs random brain playing Pong simultaneously
Video 2: 200K cortical brain with all 4 layers visible, playing Pong
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
from matplotlib.patches import Circle, Rectangle

from nsram.learning import VoltageSTDP, HomeostaticPlasticity

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Pong:
    def __init__(self):
        self.reset()
    def reset(self):
        self.bx=.5;self.by=.5
        self.bvx=.025*(1 if np.random.rand()>.5 else -1)
        self.bvy=.025*np.sin(np.random.uniform(-.4,.4))
        self.py=.5;self.score=0;self.miss=0
        return self.obs()
    def obs(self):
        return np.array([self.bx*2-1,self.by*2-1,self.bvx/.025,self.bvy/.025,self.py*2-1])
    def step(self,a):
        self.py=np.clip(self.py+a*.04,.12,.88)
        self.bx+=self.bvx;self.by+=self.bvy
        if self.by<=0 or self.by>=1: self.bvy*=-1;self.by=np.clip(self.by,.01,.99)
        if self.bx<=0: self.bvx=abs(self.bvx);self.bx=.01
        r=0
        if self.bx>=.92:
            if abs(self.by-self.py)<.13:
                self.bvx=-abs(self.bvx);self.bx=.91
                self.bvy+=(self.by-self.py)*.05;self.score+=1;r=1
            elif self.bx>=1:
                self.miss+=1;r=-1;self.bx=.3;self.by=np.random.uniform(.2,.8)
                self.bvx=.025;self.bvy=np.random.uniform(-.4,.4)*.025
        return self.obs(),r


class Brain:
    def __init__(self, N=50000, k=100, use_learning=True, seed=42):
        self.N=N; self.use_learning=use_learning
        rng=np.random.RandomState(seed); v=.1
        self.tau=torch.tensor(np.clip(1+v*.15*rng.randn(N),.3,3).astype(np.float32),device=DEVICE)
        self.theta_base=torch.tensor(np.clip(1+v*.05*rng.randn(N),.5,2).astype(np.float32),device=DEVICE)
        self.t_ref=torch.tensor(np.clip(.05+v*.01*rng.randn(N),.01,.2).astype(np.float32),device=DEVICE)
        self.dT=torch.tensor(np.clip(.1+v*.015*rng.randn(N),.02,.5).astype(np.float32),device=DEVICE)
        bg=np.clip(.88+v*.088*rng.randn(N),.5,1.2).astype(np.float32)
        self.I_bg=torch.tensor(bg*self.theta_base.cpu().numpy(),device=DEVICE)
        self.tau_syn=torch.tensor(np.clip(.5+.1*rng.randn(N),.1,2).astype(np.float32),device=DEVICE)
        self.W_in=torch.tensor(rng.randn(N,5).astype(np.float32)*.3,device=DEVICE)

        N_exc=int(N*.8);nsign=np.ones(N,np.float32);nsign[N_exc:]=-1
        nnz=N*k;ri=np.repeat(np.arange(N),k);ci=np.zeros(nnz,np.int64)
        vals=np.zeros(nnz,np.float32)
        for i in range(N):
            t=rng.choice(N,k,replace=False);ci[i*k:(i+1)*k]=t
            vals[i*k:(i+1)*k]=np.abs(rng.randn(k).astype(np.float32))*nsign[i]*.3/np.sqrt(k)
        self.pre_idx=torch.tensor(ri,device=DEVICE,dtype=torch.long)
        self.post_idx=torch.tensor(ci,device=DEVICE,dtype=torch.long)
        self.W_vals=torch.tensor(vals,device=DEVICE)
        self.W=torch.sparse_coo_tensor(torch.stack([self.pre_idx,self.post_idx]),self.W_vals,(N,N)).coalesce()
        self.W_out=torch.zeros(3,N,device=DEVICE)
        self.elig_out=torch.zeros(3,N,device=DEVICE)

        if use_learning:
            self.vstdp=VoltageSTDP(A_plus=.002,A_minus=.001)
            self.vstdp.init_traces(N,device=DEVICE)
            self.homeo=HomeostaticPlasticity(target_rate=.05,adaptation_rate=.0005)
            self.homeo.init(N,device=DEVICE)
        else:
            self.vstdp=None;self.homeo=None
        self.reset()

    def reset(self):
        N=self.N
        self.Vm=torch.zeros(N,device=DEVICE);self.syn=torch.zeros(N,device=DEVICE)
        self.refrac=torch.zeros(N,device=DEVICE);self.ft=torch.zeros(N,device=DEVICE)
        self.spike_hist=torch.zeros(N,device=DEVICE)

    @torch.no_grad()
    def step(self,obs):
        N=self.N;u=torch.tensor(obs,dtype=torch.float32,device=DEVICE)
        I_in=self.W_in@u
        I_syn=torch.sparse.mm(self.W,self.syn.unsqueeze(1)).squeeze()*.3
        theta=self.theta_base.clone()
        if self.homeo: theta=theta+self.homeo.get_threshold_shift()
        active=(self.refrac<=0).float()
        leak=-self.Vm/self.tau
        exp_t=self.dT*torch.exp(torch.clamp((self.Vm-theta)/self.dT.clamp(min=1e-6),-10,5))
        self.Vm+=active*(leak+self.I_bg+I_in+I_syn+exp_t)
        self.Vm+=active*.01*torch.randn(N,device=DEVICE)
        self.Vm.clamp_(-2,5)
        spiked=(self.Vm>=theta)&(self.refrac<=0)
        if spiked.any():
            self.Vm[spiked]=0;self.refrac[spiked]=self.t_ref[spiked];self.syn[spiked]+=1
        self.syn*=torch.exp(-1/self.tau_syn);self.refrac=(self.refrac-1).clamp(min=0)
        self.ft=.8*self.ft+.2*self.Vm;self.spike_hist=.85*self.spike_hist+.15*spiked.float()

        if self.vstdp:
            self.vstdp.update(self.W_vals,self.pre_idx,self.post_idx,spiked,self.Vm)
            self.W=torch.sparse_coo_tensor(torch.stack([self.pre_idx,self.post_idx]),self.W_vals,(N,N)).coalesce()
        if self.homeo: self.homeo.update(spiked)

        state=self.Vm+.3*self.ft
        logits=self.W_out@state;probs=torch.softmax(logits,dim=0)
        act=torch.multinomial(probs,1).item()
        one_hot=torch.zeros(3,device=DEVICE);one_hot[act]=1
        self.elig_out=.95*self.elig_out+torch.outer(one_hot,state)
        return act-1,spiked.sum().item(),probs.cpu().numpy()

    @torch.no_grad()
    def reward(self,r):
        self.W_out+=.003*r*self.elig_out;self.W_out.clamp_(-1,1)

    def grid(self):
        sz=int(np.ceil(np.sqrt(self.N)))
        d=torch.zeros(sz*sz,device=DEVICE);d[:self.N]=self.spike_hist
        return d.cpu().numpy().reshape(sz,sz)


def record_episode(brain, env, steps=800):
    obs=env.reset(); brain.reset(); frames=[]
    for step in range(steps):
        act,nspk,probs=brain.step(obs)
        obs,r=env.step(act)
        if r!=0: brain.reward(r)
        if step%2==0:
            frames.append({'bx':env.bx,'by':env.by,'py':env.py,
                           'score':env.score,'miss':env.miss,
                           'brain':brain.grid().copy(),'probs':probs.copy(),
                           'spikes':nspk,'step':step})
    return frames


def make_split_screen_video(N=50000):
    """Video 1: Learning brain vs random brain side by side."""
    print(f"\n  Building two {N:,}-neuron brains...")
    brain_learn = Brain(N=N, use_learning=True, seed=42)
    brain_random = Brain(N=N, use_learning=False, seed=42)

    # Pre-train the learning brain
    print("  Pre-training learning brain (20 episodes)...")
    env = Pong()
    for ep in range(20):
        obs = env.reset()
        for _ in range(600):
            act, _, _ = brain_learn.step(obs)
            obs, r = env.step(act)
            if r != 0: brain_learn.reward(r)
        if ep % 5 == 0:
            print(f"    Ep {ep}: score={env.score} miss={env.miss}")

    # Record both playing same sequence
    print("  Recording episodes...")
    np.random.seed(99)
    env_l = Pong(); env_r = Pong()
    # Sync random states
    env_r.bx=env_l.bx;env_r.by=env_l.by;env_r.bvx=env_l.bvx;env_r.bvy=env_l.bvy

    frames_l = record_episode(brain_learn, env_l, 800)
    frames_r = record_episode(brain_random, env_r, 800)
    n_frames = min(len(frames_l), len(frames_r))

    print(f"  Learned: {env_l.score}-{env_l.miss} | Random: {env_r.score}-{env_r.miss}")
    print(f"  Rendering split-screen ({n_frames} frames)...")

    # Build figure
    fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
    gs = GridSpec(2, 2, hspace=0.15, wspace=0.08,
                  left=0.03, right=0.97, top=0.90, bottom=0.04)

    fig.text(0.25, 0.95, 'V-STDP + Homeostatic Brain (learned)',
             ha='center', fontsize=13, fontweight='bold', color='#4ecdc4')
    fig.text(0.75, 0.95, 'Random Brain (no learning)',
             ha='center', fontsize=13, fontweight='bold', color='#e74c3c')
    fig.text(0.5, 0.98, f'NS-RAM {N:,} Neurons — Learning vs Random',
             ha='center', fontsize=16, fontweight='bold', color='white')

    # Left brain
    ax_bl = fig.add_subplot(gs[0, 0]); ax_bl.set_facecolor('#0d1117')
    img_l = ax_bl.imshow(frames_l[0]['brain'], cmap='inferno', aspect='auto',
                          interpolation='bilinear', vmin=0, vmax=0.3)
    ax_bl.set_title('Brain Activity', fontsize=10, color='#4ecdc4')
    ax_bl.tick_params(colors='gray', labelsize=5)

    # Right brain
    ax_br = fig.add_subplot(gs[0, 1]); ax_br.set_facecolor('#0d1117')
    img_r = ax_br.imshow(frames_r[0]['brain'], cmap='inferno', aspect='auto',
                          interpolation='bilinear', vmin=0, vmax=0.3)
    ax_br.set_title('Brain Activity', fontsize=10, color='#e74c3c')
    ax_br.tick_params(colors='gray', labelsize=5)

    # Left game
    ax_gl = fig.add_subplot(gs[1, 0]); ax_gl.set_facecolor('#16213e')
    ax_gl.set_xlim(0, 1); ax_gl.set_ylim(0, 1); ax_gl.set_aspect('equal')
    ball_l = Circle((.5,.5), .02, color='#ff6b6b', zorder=5); ax_gl.add_patch(ball_l)
    pad_l = Rectangle((.94,.37), .03, .26, color='#4ecdc4', zorder=5); ax_gl.add_patch(pad_l)
    score_l = ax_gl.text(.5, 1.04, '', transform=ax_gl.transAxes, ha='center',
                          fontsize=14, color='#4ecdc4', fontweight='bold')
    trail_l, = ax_gl.plot([], [], 'o', color='#ff6b6b', alpha=0.15, markersize=3)

    # Right game
    ax_gr = fig.add_subplot(gs[1, 1]); ax_gr.set_facecolor('#16213e')
    ax_gr.set_xlim(0, 1); ax_gr.set_ylim(0, 1); ax_gr.set_aspect('equal')
    ball_r = Circle((.5,.5), .02, color='#ff6b6b', zorder=5); ax_gr.add_patch(ball_r)
    pad_r = Rectangle((.94,.37), .03, .26, color='#e74c3c', zorder=5); ax_gr.add_patch(pad_r)
    score_r = ax_gr.text(.5, 1.04, '', transform=ax_gr.transAxes, ha='center',
                          fontsize=14, color='#e74c3c', fontweight='bold')
    trail_r, = ax_gr.plot([], [], 'o', color='#ff6b6b', alpha=0.15, markersize=3)

    for ax in [ax_gl, ax_gr]:
        ax.axvline(.5, color='#1a3a5c', linestyle='--', linewidth=1, alpha=.3)
        ax.tick_params(colors='gray', labelsize=5)

    trail_lx, trail_ly, trail_rx, trail_ry = [], [], [], []

    def update(i):
        fl = frames_l[i % n_frames]; fr = frames_r[i % n_frames]
        img_l.set_data(fl['brain']); img_r.set_data(fr['brain'])
        ball_l.set_center((fl['bx'], fl['by'])); ball_r.set_center((fr['bx'], fr['by']))
        pad_l.set_y(fl['py']-.13); pad_r.set_y(fr['py']-.13)
        score_l.set_text(f"Score: {fl['score']}  Miss: {fl['miss']}")
        score_r.set_text(f"Score: {fr['score']}  Miss: {fr['miss']}")
        trail_lx.append(fl['bx']);trail_ly.append(fl['by'])
        trail_rx.append(fr['bx']);trail_ry.append(fr['by'])
        for lst in [trail_lx,trail_ly,trail_rx,trail_ry]:
            while len(lst)>12: lst.pop(0)
        trail_l.set_data(trail_lx,trail_ly);trail_r.set_data(trail_rx,trail_ry)
        return img_l,img_r,ball_l,ball_r,pad_l,pad_r,score_l,score_r,trail_l,trail_r

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
    path = os.path.join(OUT, 'nsram_learn_vs_random.mp4')
    writer = animation.FFMpegWriter(fps=25, bitrate=4000, extra_args=['-pix_fmt','yuv420p'])
    ani.save(path, writer=writer, dpi=100)
    plt.close()
    print(f"  Saved: {path}")
    return env_l.score, env_l.miss, env_r.score, env_r.miss


def make_learning_progress_video(N=30000):
    """Video 2: Watch the brain LEARN over 5 episodes at different stages."""
    print(f"\n  Building {N:,}-neuron brain for learning progress video...")
    brain = Brain(N=N, use_learning=True, seed=42)

    # Record at: episode 0 (untrained), 10, 20, 30, 40
    episode_frames = {}
    env = Pong()

    for ep in range(41):
        obs = env.reset()
        do_record = ep in [0, 10, 20, 30, 40]
        frames = []
        for step in range(600):
            act, nspk, probs = brain.step(obs)
            obs, r = env.step(act)
            if r != 0: brain.reward(r)
            if do_record and step % 3 == 0:
                frames.append({'bx':env.bx,'by':env.by,'py':env.py,
                               'score':env.score,'miss':env.miss,
                               'brain':brain.grid().copy(),'step':step})
        if do_record:
            episode_frames[ep] = frames
            hr = env.score / max(env.score+env.miss, 1)
            print(f"    Ep {ep}: score={env.score} miss={env.miss} hit={hr:.0%} ({len(frames)} frames)")

    # Build 5-panel video
    print("  Rendering learning progress video...")
    eps_to_show = [0, 10, 20, 30, 40]
    max_frames = min(len(episode_frames[e]) for e in eps_to_show)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor='#0d1117')
    fig.text(0.5, 0.97, f'NS-RAM Brain Learning Over Time — {N:,} Neurons',
             ha='center', fontsize=16, fontweight='bold', color='white')

    brain_imgs = []; balls = []; pads = []; score_txts = []
    for i, ep in enumerate(eps_to_show):
        f = episode_frames[ep][0]
        # Brain
        ax = axes[0, i]; ax.set_facecolor('#0d1117')
        img = ax.imshow(f['brain'], cmap='inferno', aspect='auto',
                         interpolation='bilinear', vmin=0, vmax=0.3)
        ax.set_title(f'Episode {ep}', fontsize=11, color='white', fontweight='bold')
        ax.tick_params(colors='gray', labelsize=4)
        brain_imgs.append(img)

        # Game
        ax = axes[1, i]; ax.set_facecolor('#16213e')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        b = Circle((.5,.5), .018, color='#ff6b6b', zorder=5); ax.add_patch(b); balls.append(b)
        p = Rectangle((.94,.37), .03, .26, color='#4ecdc4', zorder=5); ax.add_patch(p); pads.append(p)
        t = ax.text(.5, -.06, '', transform=ax.transAxes, ha='center',
                     fontsize=10, color='white', fontweight='bold')
        score_txts.append(t)
        ax.tick_params(colors='gray', labelsize=4)

    def update(i):
        for j, ep in enumerate(eps_to_show):
            f = episode_frames[ep][i % len(episode_frames[ep])]
            brain_imgs[j].set_data(f['brain'])
            balls[j].set_center((f['bx'], f['by']))
            pads[j].set_y(f['py'] - .13)
            hr = f['score'] / max(f['score']+f['miss'], 1)
            score_txts[j].set_text(f"{f['score']}-{f['miss']} ({hr:.0%})")
        return brain_imgs + balls + pads + score_txts

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50, blit=False)
    path = os.path.join(OUT, 'nsram_learning_progress.mp4')
    writer = animation.FFMpegWriter(fps=20, bitrate=4000, extra_args=['-pix_fmt','yuv420p'])
    ani.save(path, writer=writer, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print(f"Device: {DEVICE}")
    if DEVICE=='cuda':
        free=torch.cuda.mem_get_info(0)[0]/1e9
        print(f"GPU free: {free:.0f} GB")

    # Video 1: Learning vs Random split-screen
    print("\n" + "="*60)
    print("  VIDEO 1: V-STDP Brain vs Random Brain")
    print("="*60)
    sl, ml, sr, mr = make_split_screen_video(N=50000)

    # Video 2: Learning progress over episodes
    print("\n" + "="*60)
    print("  VIDEO 2: Learning Progress (Ep 0→40)")
    print("="*60)
    make_learning_progress_video(N=30000)

    print(f"\n{'='*60}")
    print(f"  All videos saved to: {OUT}/")
    print(f"  - nsram_learn_vs_random.mp4")
    print(f"  - nsram_learning_progress.mp4")
    print(f"  Learned brain: {sl}-{ml} | Random: {sr}-{mr}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
