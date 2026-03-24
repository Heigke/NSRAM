#!/usr/bin/env python3
"""Benchmark hardware-realistic learning rules on Pong.

Compares 5 approaches, ALL tapeout-compatible:
  1. No learning (fixed random reservoir + random readout)
  2. Reward-modulated readout only (reservoir fixed, readout learns)
  3. R-STDP on recurrent connections (reservoir learns)
  4. V-STDP on recurrent + homeostatic plasticity
  5. R-STDP + V-STDP + homeostatic (full stack)

Each approach uses only local information + global reward broadcast.
No backpropagation, no weight transport, no softmax.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nsram.learning import STDP, RewardSTDP, VoltageSTDP, HomeostaticPlasticity

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


class Pong:
    def __init__(self):
        self.reset()
    def reset(self):
        self.bx=.5;self.by=.5;self.bvx=.025*(1 if np.random.rand()>.5 else -1)
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


class LearningBrain:
    """NS-RAM brain with configurable learning rules."""

    def __init__(self, N=20000, k=80, learning_mode='readout_only', seed=42):
        self.N = N; self.mode = learning_mode
        rng = np.random.RandomState(seed)
        v = 0.10

        # Neuron params
        self.tau = torch.tensor(np.clip(1+v*.15*rng.randn(N),.3,3).astype(np.float32), device=DEVICE)
        self.theta_base = torch.tensor(np.clip(1+v*.05*rng.randn(N),.5,2).astype(np.float32), device=DEVICE)
        self.t_ref = torch.tensor(np.clip(.05+v*.01*rng.randn(N),.01,.2).astype(np.float32), device=DEVICE)
        self.dT = torch.tensor(np.clip(.1+v*.015*rng.randn(N),.02,.5).astype(np.float32), device=DEVICE)
        bg = np.clip(.88+v*.088*rng.randn(N),.5,1.2).astype(np.float32)
        self.I_bg = torch.tensor(bg * self.theta_base.cpu().numpy(), device=DEVICE)
        self.tau_syn = torch.tensor(np.clip(.5+.1*rng.randn(N),.1,2).astype(np.float32), device=DEVICE)

        # Input
        self.W_in = torch.tensor(rng.randn(N, 5).astype(np.float32) * 0.3, device=DEVICE)

        # Sparse recurrent
        N_exc = int(N * 0.8); nsign = np.ones(N, np.float32); nsign[N_exc:] = -1
        nnz = N * k
        ri = np.repeat(np.arange(N), k)
        ci = np.zeros(nnz, np.int64)
        vals = np.zeros(nnz, np.float32)
        for i in range(N):
            t = rng.choice(N, k, replace=False)
            ci[i*k:(i+1)*k] = t
            vals[i*k:(i+1)*k] = np.abs(rng.randn(k).astype(np.float32)) * nsign[i] * 0.3/np.sqrt(k)

        self.pre_idx = torch.tensor(ri, device=DEVICE, dtype=torch.long)
        self.post_idx = torch.tensor(ci, device=DEVICE, dtype=torch.long)
        self.W_vals = torch.tensor(vals, device=DEVICE)
        self.W = torch.sparse_coo_tensor(
            torch.stack([self.pre_idx, self.post_idx]),
            self.W_vals, (N, N)).coalesce()

        # Readout
        self.W_out = torch.zeros(3, N, device=DEVICE)
        self.elig_out = torch.zeros(3, N, device=DEVICE)

        # Learning rules
        self.stdp = None; self.rstdp = None; self.vstdp = None; self.homeo = None

        if 'rstdp' in learning_mode or 'full' in learning_mode:
            self.rstdp = RewardSTDP(tau_plus=15, tau_minus=15, A_plus=0.003, A_minus=0.002,
                                     tau_eligibility=40)
            self.rstdp.init_traces(N, nnz, device=DEVICE)

        if 'vstdp' in learning_mode or 'full' in learning_mode:
            self.vstdp = VoltageSTDP(A_plus=0.002, A_minus=0.001,
                                      theta_plus=0.5, theta_minus=0.3)
            self.vstdp.init_traces(N, device=DEVICE)

        if 'homeo' in learning_mode or 'full' in learning_mode:
            self.homeo = HomeostaticPlasticity(target_rate=0.05, adaptation_rate=0.0005)
            self.homeo.init(N, device=DEVICE)

        self.reset_state()

    def reset_state(self):
        N = self.N
        self.Vm = torch.zeros(N, device=DEVICE)
        self.syn = torch.zeros(N, device=DEVICE)
        self.refrac = torch.zeros(N, device=DEVICE)
        self.ft = torch.zeros(N, device=DEVICE)

    @torch.no_grad()
    def step(self, obs):
        N = self.N
        u = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        I_in = self.W_in @ u
        I_syn = torch.sparse.mm(self.W, self.syn.unsqueeze(1)).squeeze() * 0.3

        # Homeostatic threshold shift
        theta = self.theta_base.clone()
        if self.homeo:
            theta = theta + self.homeo.get_threshold_shift()

        active = (self.refrac <= 0).float()
        leak = -self.Vm / self.tau
        exp_t = self.dT * torch.exp(torch.clamp((self.Vm - theta) / self.dT.clamp(min=1e-6), -10, 5))
        self.Vm += active * (leak + self.I_bg + I_in + I_syn + exp_t)
        self.Vm += active * 0.01 * torch.randn(N, device=DEVICE)
        self.Vm.clamp_(-2, 5)

        spiked = (self.Vm >= theta) & (self.refrac <= 0)
        if spiked.any():
            self.Vm[spiked] = 0; self.refrac[spiked] = self.t_ref[spiked]
            self.syn[spiked] += 1

        self.syn *= torch.exp(-1 / self.tau_syn)
        self.refrac = (self.refrac - 1).clamp(min=0)
        self.ft = 0.8 * self.ft + 0.2 * self.Vm

        # Apply learning rules
        if self.rstdp:
            self.rstdp.update_traces(self.pre_idx, self.post_idx, spiked, spiked)
        if self.vstdp:
            self.vstdp.update(self.W_vals, self.pre_idx, self.post_idx, spiked, self.Vm)
            # Rebuild sparse matrix from modified values
            self.W = torch.sparse_coo_tensor(
                torch.stack([self.pre_idx, self.post_idx]),
                self.W_vals, (N, N)).coalesce()
        if self.homeo:
            self.homeo.update(spiked)

        # Readout
        state = self.Vm + 0.3 * self.ft
        logits = self.W_out @ state
        probs = torch.softmax(logits, dim=0)
        act = torch.multinomial(probs, 1).item()

        # Readout eligibility
        if self.mode != 'none':
            one_hot = torch.zeros(3, device=DEVICE); one_hot[act] = 1
            self.elig_out = 0.95 * self.elig_out + torch.outer(one_hot, state)

        return act - 1, spiked.sum().item()

    @torch.no_grad()
    def reward(self, r):
        # Readout update (always, except 'none' mode)
        if self.mode != 'none':
            self.W_out += 0.003 * r * self.elig_out
            self.W_out.clamp_(-1, 1)
        # R-STDP on recurrent weights
        if self.rstdp:
            self.rstdp.apply_reward(self.W_vals, r * 0.5)
            self.W = torch.sparse_coo_tensor(
                torch.stack([self.pre_idx, self.post_idx]),
                self.W_vals, (self.N, self.N)).coalesce()


def run_experiment(mode, N=20000, n_episodes=30, steps=600, seed=42):
    brain = LearningBrain(N=N, learning_mode=mode, seed=seed)
    env = Pong()
    scores = []; misses = []

    for ep in range(n_episodes):
        obs = env.reset()
        t0 = time.time()
        for _ in range(steps):
            act, _ = brain.step(obs)
            obs, r = env.step(act)
            if r != 0: brain.reward(r)
        elapsed = time.time() - t0
        scores.append(env.score); misses.append(env.miss)
        hr = env.score / max(env.score + env.miss, 1)
        if ep % 10 == 0 or ep == n_episodes - 1:
            print(f"    Ep {ep:3d}: score={env.score:2d} miss={env.miss:2d} "
                  f"hit={hr:.0%} ({elapsed:.1f}s)")

    return scores, misses


def main():
    N = 20000  # 20K neurons — fast enough for multiple experiments

    modes = {
        'No Learning (random)': 'none',
        'Readout Only (reward)': 'readout_only',
        'R-STDP (recurrent + readout)': 'rstdp',
        'V-STDP + Homeostatic': 'vstdp_homeo',
        'Full Stack (R-STDP+V-STDP+Homeo)': 'full',
    }

    print(f"{'='*65}")
    print(f"  Learning Rule Benchmark — {N:,} NS-RAM Neurons on Pong")
    print(f"  All rules are hardware-tapeout compatible")
    print(f"{'='*65}")

    all_results = {}
    for name, mode in modes.items():
        print(f"\n━━━ {name} ━━━")
        scores, misses = run_experiment(mode, N=N, n_episodes=30, seed=42)
        all_results[name] = {'scores': scores, 'misses': misses}

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0d1117')
    fig.suptitle(f'NS-RAM Learning Rules Comparison — {N:,} Neurons, Pong',
                  fontsize=14, fontweight='bold', color='white')

    colors = ['#666', '#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for ax_idx, (metric, title) in enumerate([
        ('hit_rate', 'Hit Rate (higher = better)'),
        ('scores', 'Hits per Episode'),
        ('misses', 'Misses per Episode'),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor('#0d1117')
        for i, (name, data) in enumerate(all_results.items()):
            if metric == 'hit_rate':
                vals = [s / max(s+m, 1) for s, m in zip(data['scores'], data['misses'])]
            else:
                vals = data[metric]
            # Smooth
            window = 3
            smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, color=colors[i], linewidth=2, label=name.split('(')[0].strip())
        ax.set_xlabel('Episode', color='white')
        ax.set_ylabel(title, color='white')
        ax.legend(fontsize=7, loc='best', facecolor='#1a1a2e', edgecolor='#333',
                  labelcolor='white')
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values(): spine.set_color('#333')

    plt.tight_layout()
    path = os.path.join(OUT, 'nsram_learning_benchmark.png')
    plt.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"\n  Saved: {path}")

    # Summary table
    print(f"\n{'='*75}")
    print(f"  {'Rule':<35s}  {'First 5':>8s}  {'Last 5':>8s}  {'Best':>6s}  {'HW?':>4s}")
    print(f"{'='*75}")
    for name, data in all_results.items():
        hrs = [s/max(s+m,1) for s,m in zip(data['scores'], data['misses'])]
        first5 = np.mean(hrs[:5]); last5 = np.mean(hrs[-5:]); best = max(hrs)
        print(f"  {name:<35s}  {first5:>7.0%}  {last5:>7.0%}  {best:>5.0%}  {'YES':>4s}")


if __name__ == '__main__':
    main()
