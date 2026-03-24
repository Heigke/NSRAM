"""nsram.analysis — Spike Train Analysis and Visualization

Tools for analyzing spiking network dynamics:

Statistics:
    firing_rate          — Per-neuron mean firing rate
    isi_statistics       — Inter-spike interval: mean, CV, burst detection
    fano_factor          — Spike count variability
    correlation_matrix   — Pairwise spike correlation
    synchrony_index      — Population synchrony measure

Information Theory:
    spike_entropy        — Entropy of spike train
    mutual_information   — MI between spike trains
    transfer_entropy     — Directed information flow

Dynamics:
    avalanche_analysis   — Power-law avalanche size distribution
    effective_dimension  — Dimensionality of neural state space
    lyapunov_estimate    — Largest Lyapunov exponent estimate

Visualization:
    raster_plot          — Spike raster with optional rate overlay
    isi_histogram        — ISI distribution per neuron/population
    rate_plot            — Population firing rate over time
    state_portrait       — 2D/3D phase portrait of network state
"""

import numpy as np
from typing import Optional, Tuple, List

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════
# SPIKE STATISTICS
# ═══════════════════════════════════════════════════════════════════

def firing_rate(spikes, dt: float = 1.0, window: Optional[int] = None):
    """Compute firing rate per neuron.

    Args:
        spikes: (N, T) binary spike matrix
        dt: Timestep duration (for converting to Hz)
        window: If given, compute sliding window rate (returns (N, T))

    Returns:
        rates: (N,) mean rates in Hz, or (N, T) instantaneous rates
    """
    spikes = np.asarray(spikes)
    if window is None:
        return spikes.sum(axis=1) / (spikes.shape[1] * dt)
    else:
        kernel = np.ones(window) / (window * dt)
        rates = np.array([np.convolve(s, kernel, mode='same') for s in spikes])
        return rates


def isi_statistics(spikes, dt: float = 1.0):
    """Inter-spike interval statistics.

    Returns per-neuron: mean ISI, CV (coefficient of variation),
    burst ratio (fraction of ISI < 5*dt).

    Args:
        spikes: (N, T) binary spike matrix
        dt: Timestep duration

    Returns:
        dict with keys 'mean_isi', 'cv_isi', 'burst_ratio', 'all_isis'
    """
    spikes = np.asarray(spikes)
    N, T = spikes.shape

    mean_isis = np.full(N, np.nan)
    cv_isis = np.full(N, np.nan)
    burst_ratios = np.full(N, 0.0)
    all_isis = []

    for n in range(N):
        spike_times = np.nonzero(spikes[n])[0] * dt
        if len(spike_times) < 2:
            all_isis.append(np.array([]))
            continue
        isis = np.diff(spike_times)
        all_isis.append(isis)
        mean_isis[n] = isis.mean()
        if isis.mean() > 0:
            cv_isis[n] = isis.std() / isis.mean()
        burst_ratios[n] = (isis < 5 * dt).mean()

    return {
        'mean_isi': mean_isis,
        'cv_isi': cv_isis,
        'burst_ratio': burst_ratios,
        'all_isis': all_isis,
    }


def fano_factor(spikes, window: int = 100):
    """Fano factor: var(count) / mean(count) in sliding windows.

    FF = 1 for Poisson, < 1 for regular, > 1 for bursty.

    Args:
        spikes: (N, T) binary spike matrix
        window: Window size in timesteps

    Returns:
        ff: (N,) Fano factor per neuron
    """
    spikes = np.asarray(spikes)
    N, T = spikes.shape
    n_windows = T // window
    if n_windows < 2:
        return np.full(N, np.nan)

    counts = spikes[:, :n_windows * window].reshape(N, n_windows, window).sum(axis=2)
    means = counts.mean(axis=1)
    variances = counts.var(axis=1)
    ff = np.where(means > 0, variances / means, np.nan)
    return ff


def correlation_matrix(spikes, bin_size: int = 10):
    """Pairwise spike count correlation.

    Args:
        spikes: (N, T) binary spike matrix
        bin_size: Bin width for counting spikes

    Returns:
        corr: (N, N) Pearson correlation matrix
    """
    spikes = np.asarray(spikes)
    N, T = spikes.shape
    n_bins = T // bin_size
    binned = spikes[:, :n_bins * bin_size].reshape(N, n_bins, bin_size).sum(axis=2)
    return np.corrcoef(binned)


def synchrony_index(spikes, bin_size: int = 5):
    """Population synchrony: ratio of population variance to mean single-neuron variance.

    SI = 1 for perfectly synchronous, 0 for independent.

    Args:
        spikes: (N, T) binary spike matrix
        bin_size: Bin width

    Returns:
        float: synchrony index
    """
    spikes = np.asarray(spikes)
    N, T = spikes.shape
    n_bins = T // bin_size
    binned = spikes[:, :n_bins * bin_size].reshape(N, n_bins, bin_size).sum(axis=2)

    pop_rate = binned.mean(axis=0)
    var_pop = pop_rate.var()
    var_single = binned.var(axis=1).mean()
    if var_single < 1e-10:
        return 0.0
    return float(var_pop / var_single)


# ═══════════════════════════════════════════════════════════════════
# INFORMATION THEORY
# ═══════════════════════════════════════════════════════════════════

def spike_entropy(spikes, bin_size: int = 10):
    """Shannon entropy of binned spike counts.

    Args:
        spikes: (N, T) binary spike matrix
        bin_size: Bin width

    Returns:
        H: (N,) entropy in bits per neuron
    """
    spikes = np.asarray(spikes)
    N, T = spikes.shape
    n_bins = T // bin_size
    binned = spikes[:, :n_bins * bin_size].reshape(N, n_bins, bin_size).sum(axis=2)

    H = np.zeros(N)
    for n in range(N):
        counts = binned[n]
        values, freqs = np.unique(counts, return_counts=True)
        probs = freqs / freqs.sum()
        H[n] = -np.sum(probs * np.log2(probs + 1e-12))
    return H


def transfer_entropy(source, target, k: int = 1, bin_size: int = 5):
    """Transfer entropy from source to target spike train.

    TE(S→T) = H(T_future | T_past) - H(T_future | T_past, S_past)

    Args:
        source: (T,) source spike train
        target: (T,) target spike train
        k: History length
        bin_size: Binning for discretization

    Returns:
        float: TE in bits
    """
    source = np.asarray(source)
    target = np.asarray(target)
    T = min(len(source), len(target))
    n_bins = T // bin_size
    s = source[:n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)
    t = target[:n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)

    # Discretize to 0/1 (any spike in bin)
    s = (s > 0).astype(int)
    t = (t > 0).astype(int)

    if len(t) < k + 2:
        return 0.0

    # Build joint distributions
    counts = {}
    for i in range(k, len(t) - 1):
        t_past = tuple(t[i - k:i])
        s_past = tuple(s[i - k:i])
        t_future = t[i]
        key = (t_future, t_past, s_past)
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    # Compute conditional entropies
    p_joint = {k: v / total for k, v in counts.items()}

    # Marginals
    p_tp = {}
    p_tp_sp = {}
    for (tf, tp, sp), p in p_joint.items():
        p_tp[tp] = p_tp.get(tp, 0) + p
        p_tp_sp[(tp, sp)] = p_tp_sp.get((tp, sp), 0) + p

    te = 0.0
    for (tf, tp, sp), p in p_joint.items():
        p_tf_tp = sum(v for (tf2, tp2, sp2), v in p_joint.items()
                      if tp2 == tp and tf2 == tf)
        p_tf_tp_sp = p
        if p_tf_tp > 0 and p_tf_tp_sp > 0 and p_tp_sp.get((tp, sp), 0) > 0:
            te += p * np.log2(
                (p_tf_tp_sp / p_tp_sp[(tp, sp)]) /
                (p_tf_tp / p_tp[tp]) + 1e-12
            )
    return float(te)


# ═══════════════════════════════════════════════════════════════════
# DYNAMICS ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def avalanche_analysis(spikes, bin_size: int = 1):
    """Analyze neuronal avalanches — hallmark of criticality.

    An avalanche is a contiguous sequence of time bins with >= 1 spike.
    At criticality, avalanche sizes follow a power law with exponent ≈ -1.5.

    Args:
        spikes: (N, T) binary spike matrix
        bin_size: Temporal bin width

    Returns:
        dict: 'sizes', 'durations', 'size_exponent', 'is_critical'
    """
    spikes = np.asarray(spikes)
    N, T = spikes.shape
    n_bins = T // bin_size
    pop_activity = spikes[:, :n_bins * bin_size].reshape(N, n_bins, bin_size).sum(axis=(0, 2))

    sizes = []
    durations = []
    current_size = 0
    current_dur = 0

    for t in range(len(pop_activity)):
        if pop_activity[t] > 0:
            current_size += pop_activity[t]
            current_dur += 1
        elif current_size > 0:
            sizes.append(current_size)
            durations.append(current_dur)
            current_size = 0
            current_dur = 0

    if current_size > 0:
        sizes.append(current_size)
        durations.append(current_dur)

    sizes = np.array(sizes, dtype=np.float64)
    durations = np.array(durations, dtype=np.float64)

    # Estimate power-law exponent via MLE (Clauset et al. 2009)
    if len(sizes) > 10 and sizes.min() >= 1:
        x_min = 1.0
        alpha = 1 + len(sizes) / np.sum(np.log(sizes / x_min))
        is_critical = 1.3 < alpha < 1.7  # Expected: ~1.5
    else:
        alpha = np.nan
        is_critical = False

    return {
        'sizes': sizes,
        'durations': durations,
        'size_exponent': float(alpha),
        'is_critical': bool(is_critical),
        'n_avalanches': len(sizes),
    }


def effective_dimension(states, n_components: int = 20):
    """Effective dimensionality of neural state space (participation ratio).

    ED = (Σλ_i)² / Σλ_i² where λ_i are eigenvalues of the covariance matrix.
    ED = 1 for rank-1, ED = N for full rank.

    Args:
        states: (N, T) state matrix
        n_components: Max eigenvalues to compute

    Returns:
        float: effective dimensionality
    """
    states = np.asarray(states)
    N, T = states.shape
    states = states - states.mean(axis=1, keepdims=True)

    # Use covariance matrix (smaller dimension)
    if N <= T:
        C = states @ states.T / T
    else:
        C = states.T @ states / T

    eigs = np.real(np.linalg.eigvalsh(C))
    eigs = eigs[eigs > 0]
    if len(eigs) == 0:
        return 0.0

    ed = (eigs.sum() ** 2) / (eigs ** 2).sum()
    return float(ed)


def lyapunov_estimate(states, dt: float = 1.0, n_neighbors: int = 5):
    """Estimate largest Lyapunov exponent from state time series.

    Uses the Rosenstein (1993) method: track divergence of initially
    nearby trajectories in state space.

    Args:
        states: (N, T) state matrix
        dt: Timestep
        n_neighbors: Number of nearest neighbors

    Returns:
        float: estimated largest Lyapunov exponent (positive = chaotic)
    """
    states = np.asarray(states)
    N, T = states.shape
    if T < 100:
        return np.nan

    # Use PCA to reduce to manageable dimension
    states_c = states - states.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(states_c, full_matrices=False)
    d = min(10, N)
    proj = (U[:, :d].T @ states_c).T  # (T, d)

    # Find nearest neighbors (excluding temporal neighbors)
    from scipy.spatial import cKDTree
    tree = cKDTree(proj)
    min_sep = max(10, T // 50)  # Minimum temporal separation

    divergences = []
    sample_idx = np.random.choice(T - min_sep, min(200, T - min_sep), replace=False)

    for i in sample_idx:
        dists, idxs = tree.query(proj[i], k=n_neighbors + 1)
        for j in idxs[1:]:
            if abs(i - j) < min_sep:
                continue
            max_t = min(50, T - max(i, j))
            if max_t < 5:
                continue
            d0 = np.linalg.norm(proj[i] - proj[j])
            if d0 < 1e-10:
                continue
            for dt_step in range(1, max_t):
                d_t = np.linalg.norm(proj[i + dt_step] - proj[j + dt_step])
                if d_t > 0:
                    divergences.append((dt_step, np.log(d_t / d0)))
            break

    if len(divergences) < 10:
        return np.nan

    divergences = np.array(divergences)
    # Linear fit of log(divergence) vs time → Lyapunov exponent
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(divergences[:, 0], divergences[:, 1], deg=1)
    return float(coeffs[1]) / dt


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def raster_plot(spikes, dt: float = 1.0, title: str = 'Spike Raster',
                show_rate: bool = True, rate_window: int = 50,
                colors=None, figsize=(12, 6), save_path=None,
                dark: bool = True):
    """Publication-quality spike raster plot.

    Args:
        spikes: (N, T) binary spike matrix
        dt: Timestep (for x-axis in real time)
        title: Plot title
        show_rate: Overlay population firing rate
        rate_window: Window for rate smoothing
        colors: Per-neuron colors (N,) or None
        figsize: Figure size
        save_path: If given, save to file
        dark: Dark theme

    Returns:
        fig, ax
    """
    assert HAS_MPL, "matplotlib required for visualization"
    spikes = np.asarray(spikes)
    N, T = spikes.shape
    t_axis = np.arange(T) * dt

    bg = '#0d1117' if dark else 'white'
    fg = 'white' if dark else 'black'
    fig, axes = plt.subplots(2 if show_rate else 1, 1, figsize=figsize,
                              facecolor=bg, sharex=True,
                              height_ratios=[3, 1] if show_rate else [1])
    if not show_rate:
        axes = [axes]

    ax = axes[0]
    ax.set_facecolor(bg)

    for n in range(N):
        spike_times = np.nonzero(spikes[n])[0] * dt
        c = colors[n] if colors is not None else ('#4ecdc4' if dark else '#1f77b4')
        ax.scatter(spike_times, np.full_like(spike_times, n),
                   s=0.5, c=c, marker='|', linewidths=0.5)

    ax.set_ylabel('Neuron', color=fg, fontsize=11)
    ax.set_title(title, color=fg, fontsize=13, fontweight='bold')
    ax.set_ylim(-0.5, N - 0.5)
    ax.tick_params(colors='gray')

    if show_rate:
        ax2 = axes[1]
        ax2.set_facecolor(bg)
        kernel = np.ones(rate_window) / (rate_window * dt)
        pop_rate = np.convolve(spikes.sum(axis=0), kernel, mode='same')
        ax2.fill_between(t_axis, pop_rate, alpha=0.5, color='#FF6B6B')
        ax2.plot(t_axis, pop_rate, color='#FF6B6B', linewidth=0.8)
        ax2.set_ylabel('Pop. Rate', color=fg, fontsize=10)
        ax2.set_xlabel('Time' + (f' ({dt} ms)' if dt != 1.0 else ''), color=fg, fontsize=11)
        ax2.tick_params(colors='gray')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=bg)
    return fig, axes


def isi_histogram(spikes, dt: float = 1.0, n_bins: int = 50,
                  log_scale: bool = True, title: str = 'ISI Distribution',
                  save_path=None, dark: bool = True):
    """ISI histogram for the entire population.

    Args:
        spikes: (N, T) binary spike matrix
        dt: Timestep
        n_bins: Number of histogram bins
        log_scale: Log-scale x-axis
        save_path: Save path

    Returns:
        fig, ax
    """
    assert HAS_MPL, "matplotlib required"
    stats = isi_statistics(spikes, dt)
    all_isis = np.concatenate([x for x in stats['all_isis'] if len(x) > 0])

    if len(all_isis) == 0:
        return None, None

    bg = '#0d1117' if dark else 'white'
    fg = 'white' if dark else 'black'
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor=bg)
    ax.set_facecolor(bg)

    if log_scale and all_isis.min() > 0:
        bins = np.logspace(np.log10(all_isis.min()), np.log10(all_isis.max()), n_bins)
        ax.set_xscale('log')
    else:
        bins = n_bins

    ax.hist(all_isis, bins=bins, color='#4ecdc4', alpha=0.7, edgecolor='none')
    ax.set_xlabel('ISI', color=fg, fontsize=11)
    ax.set_ylabel('Count', color=fg, fontsize=11)
    ax.set_title(title, color=fg, fontsize=13, fontweight='bold')

    # Annotate CV
    mean_cv = np.nanmean(stats['cv_isi'])
    ax.text(0.95, 0.95, f'Mean CV = {mean_cv:.2f}',
            transform=ax.transAxes, ha='right', va='top',
            color=fg, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3))
    ax.tick_params(colors='gray')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=bg)
    return fig, ax
