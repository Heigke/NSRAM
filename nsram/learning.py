"""nsram.learning — Hardware-Realistic Learning Rules for NS-RAM

Only learning rules that map to NS-RAM silicon are included here.
Each rule uses ONLY information available locally at the synapse:
  - Pre-synaptic spike time
  - Post-synaptic spike time
  - Local charge trapping state (Q)
  - Global reward/neuromodulator signal (single scalar broadcast)

NO backpropagation, NO weight transport, NO global normalization.

Rules implemented:
  1. STDP — Spike-Timing Dependent Plasticity
     Maps to: charge trapping rate depends on relative spike timing.
     In NS-RAM: pre-before-post → potentiation (more trapping),
                post-before-pre → depression (less trapping).
     VG2 pulse timing controls the sign and magnitude.

  2. R-STDP — Reward-Modulated STDP (three-factor rule)
     Maps to: STDP trace × global reward signal.
     In NS-RAM: eligibility stored as trapped charge,
                reward signal broadcast via shared VG2 rail.

  3. Voltage-STDP (V-STDP) — uses membrane voltage, not just spikes
     Maps to: charge trapping depends on Vm at time of pre spike.
     In NS-RAM: body potential at pre-spike time modulates trapping.
     More biologically realistic, better learning (Clopath et al. 2010).

  4. Homeostatic Intrinsic Plasticity
     Maps to: VG2 adaptation based on neuron's own firing rate.
     In NS-RAM: charge accumulation shifts threshold over long timescales.
     Keeps neurons in their dynamic range without external control.

All rules operate on sparse weight matrices (GPU-compatible).
"""

import torch
import numpy as np
from typing import Optional


class STDP:
    """Spike-Timing Dependent Plasticity.

    Δw = A+ × exp(-Δt/τ+) if pre before post (LTP)
    Δw = A- × exp(+Δt/τ-) if post before pre (LTD)

    Hardware mapping: charge trapping in pre-synaptic NS-RAM cell.
    When post fires shortly after pre → trapping increases (VG2 pulse).
    When pre fires shortly after post → detrapping occurs.

    The τ+/τ- time constants map to SRH capture/emission rates.
    """

    def __init__(self, tau_plus=20.0, tau_minus=20.0,
                 A_plus=0.005, A_minus=0.005,
                 w_max=1.0, w_min=-1.0):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_max = w_max
        self.w_min = w_min

    def init_traces(self, N, device='cpu'):
        """Initialize pre/post spike traces."""
        self.pre_trace = torch.zeros(N, device=device)
        self.post_trace = torch.zeros(N, device=device)

    @torch.no_grad()
    def update(self, W_values, pre_idx, post_idx, pre_spikes, post_spikes):
        """Update weights based on spike timing.

        Args:
            W_values: sparse weight values tensor (modify in-place)
            pre_idx: presynaptic neuron indices for each connection
            post_idx: postsynaptic neuron indices for each connection
            pre_spikes: (N,) bool tensor of which neurons spiked (pre)
            post_spikes: (N,) bool tensor (post, same as pre for recurrent)

        This modifies W_values in-place (the sparse tensor's values).
        """
        # Decay traces
        self.pre_trace *= (1 - 1/self.tau_plus)
        self.post_trace *= (1 - 1/self.tau_minus)

        # Update traces where spikes occurred
        self.pre_trace[pre_spikes] += 1.0
        self.post_trace[post_spikes] += 1.0

        # LTP: pre trace at time of post spike
        # For each connection (i→j): if j spiked, Δw = A+ × pre_trace[i]
        post_spiked_mask = post_spikes[post_idx]
        if post_spiked_mask.any():
            dw_ltp = self.A_plus * self.pre_trace[pre_idx[post_spiked_mask]]
            W_values[post_spiked_mask] += dw_ltp

        # LTD: post trace at time of pre spike
        # For each connection (i→j): if i spiked, Δw = -A- × post_trace[j]
        pre_spiked_mask = pre_spikes[pre_idx]
        if pre_spiked_mask.any():
            dw_ltd = self.A_minus * self.post_trace[post_idx[pre_spiked_mask]]
            W_values[pre_spiked_mask] -= dw_ltd

        # Clamp weights
        W_values.clamp_(self.w_min, self.w_max)


class RewardSTDP:
    """Reward-Modulated STDP (R-STDP / three-factor rule).

    Δw = reward × eligibility_trace
    eligibility = STDP_trace (decaying record of what STDP would have done)

    Hardware mapping:
      - Eligibility = trapped charge in oxide (SRH slow decay ~1-100ms)
      - Reward = global VG2 pulse broadcast to all synapses
      - When reward arrives, trapped charge modulates weight permanently

    This is the most tapeout-realistic learning rule because:
      1. Eligibility is local (charge in each synapse's oxide)
      2. Reward is global (single wire/rail)
      3. Weight update is multiplicative (trapped charge × VG2 pulse)
    """

    def __init__(self, tau_plus=20.0, tau_minus=20.0,
                 A_plus=0.01, A_minus=0.008,
                 tau_eligibility=50.0,
                 w_max=1.0, w_min=-1.0):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_elig = tau_eligibility
        self.w_max = w_max
        self.w_min = w_min

    def init_traces(self, N, n_connections, device='cpu'):
        self.pre_trace = torch.zeros(N, device=device)
        self.post_trace = torch.zeros(N, device=device)
        self.eligibility = torch.zeros(n_connections, device=device)

    @torch.no_grad()
    def update_traces(self, pre_idx, post_idx, pre_spikes, post_spikes):
        """Update traces each timestep (NO weight change yet)."""
        self.pre_trace *= (1 - 1/self.tau_plus)
        self.post_trace *= (1 - 1/self.tau_minus)
        self.pre_trace[pre_spikes] += 1.0
        self.post_trace[post_spikes] += 1.0

        # Compute instantaneous STDP signal
        dw = torch.zeros_like(self.eligibility)
        post_mask = post_spikes[post_idx]
        if post_mask.any():
            dw[post_mask] += self.A_plus * self.pre_trace[pre_idx[post_mask]]
        pre_mask = pre_spikes[pre_idx]
        if pre_mask.any():
            dw[pre_mask] -= self.A_minus * self.post_trace[post_idx[pre_mask]]

        # Accumulate into eligibility trace (slow decay = oxide charge retention)
        self.eligibility *= (1 - 1/self.tau_elig)
        self.eligibility += dw

    @torch.no_grad()
    def apply_reward(self, W_values, reward):
        """Apply reward signal → weight update.

        In hardware: reward pulse on VG2 rail ×  trapped charge → permanent change.
        """
        W_values += reward * self.eligibility
        W_values.clamp_(self.w_min, self.w_max)


class VoltageSTDP:
    """Voltage-based STDP (Clopath et al. 2010).

    Δw+ = A+ × [Vm_post - θ+]+ × x_pre  (LTP when post depolarized + pre active)
    Δw- = A- × [Vm_pre - θ-]+            (LTD when pre fires and post was recent)

    Hardware mapping:
      - Vm_post is the body potential of the post-synaptic NS-RAM cell
      - θ+, θ- are threshold voltages (set by VG2)
      - x_pre is the pre-synaptic trace (residual charge from pre spike)
      - The key advantage: uses VOLTAGE, not spike timing
        → more robust, captures subthreshold integration
        → body potential is directly available in NS-RAM (no extra circuit)

    This is arguably the most natural NS-RAM learning rule because
    the body potential IS the computational variable.
    """

    def __init__(self, A_plus=0.005, A_minus=0.003,
                 theta_plus=0.5, theta_minus=0.3,
                 tau_x=15.0, tau_vm=10.0,
                 w_max=1.0, w_min=-1.0):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.theta_plus = theta_plus
        self.theta_minus = theta_minus
        self.tau_x = tau_x
        self.tau_vm = tau_vm
        self.w_max = w_max
        self.w_min = w_min

    def init_traces(self, N, device='cpu'):
        self.x_pre = torch.zeros(N, device=device)       # Pre-synaptic trace
        self.vm_slow = torch.zeros(N, device=device)      # Slow Vm average

    @torch.no_grad()
    def update(self, W_values, pre_idx, post_idx, pre_spikes, Vm):
        """Update weights using voltage + pre spike traces.

        Args:
            W_values: weight values to modify in-place
            pre_idx, post_idx: connection indices
            pre_spikes: (N,) bool, which neurons spiked
            Vm: (N,) current membrane voltage
        """
        # Update traces
        self.x_pre *= (1 - 1/self.tau_x)
        self.x_pre[pre_spikes] += 1.0
        self.vm_slow = self.vm_slow * (1 - 1/self.tau_vm) + Vm * (1/self.tau_vm)

        # LTP: post Vm above θ+ AND pre trace active
        vm_post = Vm[post_idx]
        ltp_gate = torch.clamp(vm_post - self.theta_plus, min=0)
        x_at_post = self.x_pre[pre_idx]
        dw_ltp = self.A_plus * ltp_gate * x_at_post

        # LTD: pre spikes AND post was recently depolarized
        pre_mask = pre_spikes[pre_idx]
        vm_post_slow = self.vm_slow[post_idx]
        ltd_gate = torch.clamp(vm_post_slow - self.theta_minus, min=0)
        dw_ltd = torch.zeros_like(W_values)
        dw_ltd[pre_mask] = self.A_minus * ltd_gate[pre_mask]

        W_values += dw_ltp - dw_ltd
        W_values.clamp_(self.w_min, self.w_max)


class HomeostaticPlasticity:
    """Homeostatic Intrinsic Plasticity — keeps neurons in dynamic range.

    If neuron fires too much → raise threshold (less excitable)
    If neuron fires too little → lower threshold (more excitable)

    Hardware mapping: This IS the NS-RAM charge trapping mechanism.
    Accumulated trapped charge raises effective Vth.
    Charge emission lowers it.
    No additional circuit needed — it's built into the device physics.

    Target rate is set by the balance between k_cap and k_em,
    which is controlled by VG2.
    """

    def __init__(self, target_rate=0.05, adaptation_rate=0.001,
                 max_shift=0.3):
        self.target_rate = target_rate
        self.eta = adaptation_rate
        self.max_shift = max_shift

    def init(self, N, device='cpu'):
        self.rate_estimate = torch.zeros(N, device=device)
        self.threshold_shift = torch.zeros(N, device=device)

    @torch.no_grad()
    def update(self, spikes):
        """Update threshold based on running rate estimate.

        Args:
            spikes: (N,) bool tensor
        """
        self.rate_estimate = 0.99 * self.rate_estimate + 0.01 * spikes.float()
        error = self.rate_estimate - self.target_rate
        self.threshold_shift += self.eta * error
        self.threshold_shift.clamp_(-self.max_shift, self.max_shift)

    def get_threshold_shift(self):
        """Returns threshold adjustment to add to base threshold."""
        return self.threshold_shift
