"""nsram.onchip_learning — On-Chip Learning Rules for NS-RAM Tapeout

Learning rules that can be implemented ENTIRELY in NS-RAM silicon.
No backpropagation, no floating-point gradients, no weight transport.

Each rule uses only:
  - Local pre/post spike times or rates (available at each synapse)
  - Local charge state Q (trapped charge in floating body)
  - At most ONE global scalar signal (reward, phase toggle, or nudge)

Rules:

1. ForwardForward — Hinton 2022, adapted for spiking NS-RAM
   Goodness = sum of squared spike rates per layer.
   Positive phase: real data → maximize goodness.
   Negative phase: corrupted data → minimize goodness.
   Weight update: purely local Hebbian, modulated by goodness sign.
   Global signal: 1 bit (positive/negative phase label).
   Published: 99.58% MNIST, 75.64% CIFAR-10 (FFGAF-SNN, 2025).

2. EquilibriumPropagation — Scellier & Bengio 2017
   Free phase: let network settle to energy minimum.
   Nudged phase: inject small signal at output, let settle again.
   Weight update: difference in pre×post correlations between phases.
   Global signal: 1 scalar (nudge strength β).
   NS-RAM native: charge settling IS the free phase.
   Published: 90% MNIST (Ising hardware), 88.3% CIFAR-10 (simulation).

3. Eprop — Eligibility propagation (Bellec et al. 2020)
   Eligibility trace per synapse = temporal record of contribution.
   Global reward signal modulates all eligible synapses.
   The trace maps directly to NS-RAM's multi-timescale charge retention.
   Published: 91.12% Speech Commands (SpiNNaker 2 hardware).

All three need AT MOST one global signal. All weight updates are local.
"""

import torch
import numpy as np
from typing import Optional


class ForwardForward:
    """Forward-Forward learning for spiking NS-RAM networks.

    Each layer computes "goodness" = mean squared spike rate.
    Positive data → increase goodness (strengthen active synapses).
    Negative data → decrease goodness (weaken active synapses).

    Hardware mapping:
      - Spike counter per neuron (digital, trivial) → goodness
      - Positive/negative phase: 1-bit global signal
      - Weight update: ΔW = lr × phase_sign × pre_rate × post_rate
      - In NS-RAM: phase_sign modulates VG2 (high = potentiate, low = depress)

    The "negative data" can be generated on-chip by shuffling input
    channels or adding noise — no external data pipeline needed.
    """

    def __init__(self, lr=0.01, threshold_goodness=1.0, n_layers=1):
        self.lr = lr
        self.theta_g = threshold_goodness
        self.n_layers = n_layers

    def init(self, layer_sizes, device='cpu'):
        """Initialize per-layer weight matrices."""
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            W = torch.randn(n_out, n_in, device=device) * 0.1 / np.sqrt(n_in)
            self.layers.append({
                'W': W,
                'size_in': n_in,
                'size_out': n_out,
            })
        self.device = device

    @torch.no_grad()
    def forward_pass(self, x, layer_idx):
        """Forward through one layer. Returns activations (spike rates)."""
        W = self.layers[layer_idx]['W']
        # Normalize input (layer norm — implementable as divisive inhibition)
        x_norm = x / (x.norm() + 1e-6) * np.sqrt(len(x))
        h = torch.relu(W @ x_norm)  # ReLU ≈ spike rate thresholding
        return h

    @torch.no_grad()
    def compute_goodness(self, h):
        """Goodness = sum of squared activations.

        Hardware: each neuron's spike count squared, summed.
        This is a scalar — computable with a simple accumulator.
        """
        return (h ** 2).sum()

    @torch.no_grad()
    def update_weights(self, x, h, is_positive, layer_idx):
        """Update weights based on goodness.

        Δw_ij = lr × sign × h_i × x_j
        where sign = +1 if positive phase, -1 if negative phase

        Hardware: VG2 pulse direction encodes sign.
        Magnitude = pre_rate × post_rate (charge trapping product).
        """
        W = self.layers[layer_idx]['W']
        goodness = self.compute_goodness(h)

        if is_positive:
            # Want goodness > threshold → strengthen if below
            sign = 1.0
        else:
            # Want goodness < threshold → weaken if above
            sign = -1.0

        # Outer product learning rule (purely local)
        x_norm = x / (x.norm() + 1e-6) * np.sqrt(len(x))
        dW = sign * self.lr * torch.outer(h, x_norm)
        W += dW
        # Weight clamp (physical limit of charge trapping range)
        W.clamp_(-1, 1)

        return goodness.item()

    @torch.no_grad()
    def train_step(self, x_pos, x_neg):
        """One training step: positive and negative passes through all layers.

        Returns: list of (goodness_pos, goodness_neg) per layer.
        """
        results = []
        h_pos = x_pos
        h_neg = x_neg

        for i in range(len(self.layers)):
            # Positive pass
            h_pos_new = self.forward_pass(h_pos, i)
            g_pos = self.update_weights(h_pos, h_pos_new, True, i)

            # Negative pass
            h_neg_new = self.forward_pass(h_neg, i)
            g_neg = self.update_weights(h_neg, h_neg_new, False, i)

            results.append((g_pos, g_neg))
            h_pos = h_pos_new.detach()
            h_neg = h_neg_new.detach()

        return results

    @torch.no_grad()
    def predict(self, x):
        """Inference: forward pass, return final layer goodness."""
        h = x
        for i in range(len(self.layers)):
            h = self.forward_pass(h, i)
        return h


class EquilibriumPropagation:
    """Equilibrium Propagation for NS-RAM.

    The most natural fit for analog charge-settling dynamics.

    Free phase: let network settle to energy minimum (= charge equilibrium)
    Nudged phase: inject β at output (= small current), let settle again
    ΔW = (1/β) × (corr_nudged - corr_free)

    Hardware mapping:
      - Free phase: NS-RAM array runs until membrane voltages stabilize
      - Nudged phase: inject small current at output neurons
      - Correlation: pre_v × post_v (body voltage product at each synapse)
      - Stored in charge trapping: free→trap Q_free, nudged→trap Q_nudged
      - Weight update: Q_nudged - Q_free (difference in trapped charge)

    This requires TWO settle phases but NO backward pass.
    """

    def __init__(self, lr=0.01, beta=0.5, settle_steps=20):
        self.lr = lr
        self.beta = beta
        self.settle_steps = settle_steps

    def init(self, layer_sizes, device='cpu'):
        self.n_layers = len(layer_sizes) - 1
        self.layers = []
        for i in range(self.n_layers):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            W = torch.randn(n_out, n_in, device=device) * 0.3 / np.sqrt(n_in)
            b = torch.zeros(n_out, device=device)
            self.layers.append({'W': W, 'b': b})
        self.device = device

    @torch.no_grad()
    def energy(self, states, target=None):
        """Compute network energy (Hopfield-like).

        E = -Σ_l (s_l^T W_l s_{l-1} + b_l^T s_l) + Σ_l ρ(s_l)
        where ρ(s) = s²/2 (quadratic cost of activation)
        """
        E = 0
        for i, layer in enumerate(self.layers):
            s_pre = states[i]
            s_post = states[i + 1]
            E -= (s_post @ layer['W'] @ s_pre + layer['b'] @ s_post)
            E += 0.5 * (s_post ** 2).sum()
        # Output cost (if target provided)
        if target is not None:
            E += 0.5 * self.beta * ((states[-1] - target) ** 2).sum()
        return E

    @torch.no_grad()
    def settle(self, x, target=None):
        """Let network settle to equilibrium.

        Each neuron updates: s_i ← σ(Σ_j W_ij s_j + b_i - β*(s_i - target_i))
        Iterate until convergence.

        Hardware: this IS what NS-RAM does naturally — charge settles.
        """
        n = len(self.layers) + 1
        states = [x.clone()]
        # Initialize hidden states
        for i in range(self.n_layers):
            s = torch.zeros(self.layers[i]['W'].shape[0], device=self.device)
            states.append(s)

        # Settle iterations
        for _ in range(self.settle_steps):
            for i in range(self.n_layers):
                drive = self.layers[i]['W'] @ states[i] + self.layers[i]['b']
                # Add nudge at output layer
                if target is not None and i == self.n_layers - 1:
                    drive += self.beta * (target - states[i + 1])
                # Activation: hard sigmoid (implementable as threshold comparison)
                states[i + 1] = torch.clamp(drive, 0, 1)

        return states

    @torch.no_grad()
    def train_step(self, x, target):
        """One EP training step.

        1. Free phase: settle without target
        2. Nudged phase: settle with target
        3. Weight update: correlations difference
        """
        # Free phase
        states_free = self.settle(x, target=None)

        # Nudged phase
        states_nudged = self.settle(x, target=target)

        # Weight update: ΔW = (1/β) × (corr_nudged - corr_free)
        for i in range(self.n_layers):
            corr_free = torch.outer(states_free[i+1], states_free[i])
            corr_nudged = torch.outer(states_nudged[i+1], states_nudged[i])
            dW = (self.lr / self.beta) * (corr_nudged - corr_free)
            self.layers[i]['W'] += dW
            self.layers[i]['W'].clamp_(-2, 2)

            db = (self.lr / self.beta) * (states_nudged[i+1] - states_free[i+1])
            self.layers[i]['b'] += db

        return states_free, states_nudged


class Eprop:
    """Eligibility Propagation for spiking NS-RAM networks.

    Each synapse maintains an eligibility trace e_ij(t) that records
    its recent contribution to post-synaptic spiking.

    When a global reward/error signal R arrives:
      ΔW_ij = lr × R × e_ij

    The trace decays exponentially (= charge detrapping in NS-RAM):
      de_ij/dt = -e_ij/τ_e + pre_spike_i × post_factor_j

    Hardware mapping:
      - Eligibility trace = trapped charge in oxide (τ = 1-100ms)
      - Reward signal = global VG2 pulse (1 wire to all synapses)
      - Weight update = charge modulation when reward arrives
      - Multi-timescale traces from NS-RAM's 3 τ values (z2213)

    Published: 91.12% Speech Commands on SpiNNaker 2.
    """

    def __init__(self, lr=0.005, tau_elig=30.0, tau_slow=100.0):
        self.lr = lr
        self.tau_elig = tau_elig
        self.tau_slow = tau_slow

    def init(self, N, n_connections, device='cpu'):
        self.elig_fast = torch.zeros(n_connections, device=device)
        self.elig_slow = torch.zeros(n_connections, device=device)
        self.pre_trace = torch.zeros(N, device=device)
        self.device = device

    @torch.no_grad()
    def update_traces(self, pre_idx, post_idx, pre_spikes, post_spikes, post_Vm):
        """Update eligibility traces each timestep.

        Eligibility = pre_trace × (post_spike - baseline)
        This encodes "this synapse contributed to this spike".
        """
        # Pre-synaptic trace (exponential decay)
        self.pre_trace *= (1 - 1/self.tau_elig)
        self.pre_trace[pre_spikes] += 1.0

        # Post-synaptic factor: spike minus baseline expectation
        # In hardware: comparator against running average
        post_factor = post_spikes.float() - 0.05  # baseline rate

        # Eligibility = pre × post_factor (per connection)
        de = self.pre_trace[pre_idx] * post_factor[post_idx]

        # Fast and slow traces (multi-timescale, like NS-RAM's 3 τ values)
        self.elig_fast = self.elig_fast * (1 - 1/self.tau_elig) + de
        self.elig_slow = self.elig_slow * (1 - 1/self.tau_slow) + de * 0.1

    @torch.no_grad()
    def apply_reward(self, W_values, reward):
        """Apply global reward to update weights via eligibility.

        ΔW = lr × reward × (elig_fast + 0.5 × elig_slow)

        Hardware: reward pulse on shared VG2 rail.
        Trapped charge (eligibility) × VG2 pulse = weight change.
        """
        combined_elig = self.elig_fast + 0.5 * self.elig_slow
        W_values += self.lr * reward * combined_elig
        W_values.clamp_(-1, 1)
