"""nsram.encoding — Spike Encoding and Decoding

Converts between analog signals and spike trains using biologically
motivated encoding schemes. All functions are vectorized and GPU-compatible.

Encoders:
    rate_encode       — Poisson rate coding
    latency_encode    — Time-to-first-spike
    delta_encode      — Event-driven change detection (like DVS cameras)
    population_encode — Gaussian receptive field population coding
    phase_encode      — Phase-of-firing encoding

Decoders:
    rate_decode       — Average spike count → value
    ttfs_decode       — Time-to-first-spike → value
    population_decode — Population vector decoding
"""

import numpy as np
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def rate_encode(signal, n_steps: int = 100, gain: float = 1.0,
                seed: Optional[int] = None):
    """Poisson rate encoding — the most common spike encoding.

    Each input value is treated as a firing rate. At each timestep,
    a spike is emitted with probability proportional to the input.

    Args:
        signal: (N,) or (B, N) analog values in [0, 1]
        n_steps: Number of timesteps to generate
        gain: Scaling factor for firing probability
        seed: Random seed

    Returns:
        spikes: (n_steps, *signal.shape) binary spike tensor
    """
    rng = np.random.RandomState(seed)
    signal = np.asarray(signal, dtype=np.float32)
    signal = np.clip(signal * gain, 0, 1)

    shape = (n_steps,) + signal.shape
    spikes = (rng.rand(*shape) < signal[None, ...]).astype(np.float32)
    return spikes


def latency_encode(signal, n_steps: int = 100, tau: float = 5.0,
                   normalize: bool = True):
    """Latency (time-to-first-spike) encoding.

    Higher input values spike earlier. The spike time is:
        t_spike = tau * log(1 / (value + eps))

    Args:
        signal: (N,) analog values in [0, 1]
        n_steps: Total timesteps
        tau: Time constant controlling spread
        normalize: If True, normalize spike times to [0, n_steps]

    Returns:
        spikes: (n_steps, N) binary — exactly one spike per neuron
    """
    signal = np.asarray(signal, dtype=np.float32).ravel()
    N = len(signal)

    # Higher value → earlier spike
    spike_times = tau * np.log(1.0 / (np.clip(signal, 1e-6, 1.0)))
    if normalize:
        st_min, st_max = spike_times.min(), spike_times.max()
        if st_max > st_min:
            spike_times = (spike_times - st_min) / (st_max - st_min) * (n_steps - 1)
        else:
            spike_times = np.zeros(N)

    spike_times = np.clip(spike_times, 0, n_steps - 1).astype(int)
    spikes = np.zeros((n_steps, N), dtype=np.float32)
    spikes[spike_times, np.arange(N)] = 1.0
    return spikes


def delta_encode(signal, threshold: float = 0.1):
    """Delta modulation encoding — event-driven, like DVS cameras.

    Emits a positive spike when signal increases by > threshold,
    negative spike when it decreases by > threshold.

    Args:
        signal: (T,) or (T, N) time series
        threshold: Change threshold for spike emission

    Returns:
        spikes: same shape as signal, values in {-1, 0, +1}
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]

    T, N = signal.shape
    spikes = np.zeros((T, N), dtype=np.float32)
    ref = signal[0].copy()

    for t in range(1, T):
        diff = signal[t] - ref
        pos = diff >= threshold
        neg = diff <= -threshold
        spikes[t, pos] = 1.0
        spikes[t, neg] = -1.0
        ref[pos] = signal[t, pos]
        ref[neg] = signal[t, neg]

    if spikes.shape[1] == 1:
        return spikes.ravel()
    return spikes


def population_encode(signal, n_neurons: int = 20, v_min: float = 0.0,
                      v_max: float = 1.0, sigma: Optional[float] = None):
    """Gaussian receptive field population encoding.

    Each input dimension is encoded by n_neurons with Gaussian tuning curves
    spanning [v_min, v_max]. Response = exp(-(x - center)² / (2σ²)).

    Args:
        signal: (T,) or (T, D) analog values
        n_neurons: Neurons per input dimension
        v_min, v_max: Input range
        sigma: Tuning curve width (default: auto from spacing)

    Returns:
        encoded: (T, D * n_neurons) firing rates in [0, 1]
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    T, D = signal.shape

    centers = np.linspace(v_min, v_max, n_neurons)
    if sigma is None:
        sigma = (v_max - v_min) / (n_neurons - 1) * 0.5

    # (T, D, 1) - (n_neurons,) → (T, D, n_neurons)
    encoded = np.exp(-0.5 * ((signal[:, :, None] - centers[None, None, :]) / sigma) ** 2)
    return encoded.reshape(T, D * n_neurons).astype(np.float32)


def phase_encode(signal, n_steps: int = 100, freq: float = 1.0):
    """Phase-of-firing encoding.

    A background oscillation runs at frequency freq. Each neuron fires
    when the oscillation phase matches its preferred phase (determined
    by input value). Higher values → earlier phase.

    Args:
        signal: (N,) values in [0, 1]
        n_steps: Total timesteps
        freq: Oscillation frequency (cycles per n_steps)

    Returns:
        spikes: (n_steps, N) binary spikes
    """
    signal = np.asarray(signal, dtype=np.float32).ravel()
    N = len(signal)

    # Preferred phase: high value → phase 0, low value → phase 2π
    preferred = (1.0 - np.clip(signal, 0, 1)) * 2 * np.pi

    t = np.arange(n_steps)
    phase = 2 * np.pi * freq * t / n_steps  # (T,)

    # Spike when current phase is within ±0.1 of preferred phase
    phase_diff = np.abs(np.sin(0.5 * (phase[:, None] - preferred[None, :])))
    spikes = (phase_diff < 0.05).astype(np.float32)
    return spikes


# ── Decoders ──

def rate_decode(spikes, axis=0):
    """Decode by averaging spike count.

    Args:
        spikes: (T, N) spike train
        axis: Time axis to average over

    Returns:
        rates: (N,) average firing rates
    """
    return np.mean(np.asarray(spikes), axis=axis)


def ttfs_decode(spikes):
    """Time-to-first-spike decoding — earlier spike = higher value.

    Args:
        spikes: (T, N) binary spike train

    Returns:
        values: (N,) decoded values in [0, 1], higher = earlier spike
    """
    spikes = np.asarray(spikes)
    T, N = spikes.shape
    first_spike = np.full(N, T, dtype=np.float32)
    for n in range(N):
        idx = np.nonzero(spikes[:, n])[0]
        if len(idx) > 0:
            first_spike[n] = idx[0]

    # Normalize: earlier → higher value
    values = 1.0 - first_spike / T
    return values


def population_decode(responses, n_neurons: int = 20,
                      v_min: float = 0.0, v_max: float = 1.0):
    """Population vector decoding.

    Weighted average of tuning curve centers by response amplitude.

    Args:
        responses: (T, D*n_neurons) or (D*n_neurons,) population responses
        n_neurons: Neurons per input dimension
        v_min, v_max: Value range

    Returns:
        decoded: (T, D) or (D,) decoded values
    """
    responses = np.asarray(responses, dtype=np.float32)
    squeeze = responses.ndim == 1
    if squeeze:
        responses = responses[None, :]
    T, total = responses.shape
    D = total // n_neurons

    centers = np.linspace(v_min, v_max, n_neurons)
    reshaped = responses.reshape(T, D, n_neurons)

    # Weighted average
    weights = reshaped / (reshaped.sum(axis=-1, keepdims=True) + 1e-8)
    decoded = (weights * centers[None, None, :]).sum(axis=-1)

    if squeeze:
        return decoded[0]
    return decoded
