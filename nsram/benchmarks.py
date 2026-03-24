"""nsram.benchmarks — Standard Reservoir Computing Benchmarks

Provides XOR, Memory Capacity, NARMA, and waveform classification
benchmarks with a single function call.

Example:
    >>> from nsram import NSRAMReservoir, rc_benchmark
    >>> res = NSRAMReservoir(128)
    >>> results = rc_benchmark(res, n_reps=5)
    >>> print(results)
"""

import numpy as np
from typing import Optional


def _ridge(X, y, alpha=1.0):
    """Ridge regression: w = (X'X + αI)⁻¹ X'y."""
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def xor_accuracy(states, inputs, washout=500, tau=1):
    """Temporal XOR accuracy at delay τ.

    Target: y(t) = sign(u(t)) XOR sign(u(t-τ))
    """
    T = states.shape[1]
    sp = washout + (T - washout) // 2
    X = states[:, washout + tau:].T
    y = ((inputs[washout + tau:] > 0) != (inputs[washout:T - tau] > 0)).astype(float)
    s = sp - washout - tau
    if s < 20 or len(y) - s < 20:
        return 0.5
    w = _ridge(X[:s], y[:s])
    acc = ((X[s:] @ w > 0.5) == (y[s:] > 0.5)).mean()
    return max(acc, 1 - acc)


def memory_capacity(states, inputs, washout=500, max_delay=15):
    """Total memory capacity: MC = Σ_d r²(d).

    Measures how well the reservoir remembers past inputs.
    """
    T = states.shape[1]
    sp = washout + (T - washout) // 2
    mc = 0.0
    for d in range(1, max_delay + 1):
        X = states[:, washout + d:].T
        y = inputs[washout:T - d]
        s = sp - washout - d
        if s < 20 or len(y) - s < 20:
            continue
        w = _ridge(X[:s], y[:s])
        pred = X[s:] @ w
        yt = y[s:]
        if np.std(yt) < 1e-10 or np.std(pred) < 1e-10:
            continue
        mc += np.corrcoef(pred, yt)[0, 1] ** 2
    return mc


def narma_prediction(states, inputs, washout=500, order=10):
    """NARMA-N prediction R².

    NARMA-10 is the standard nonlinear temporal benchmark.
    """
    T = min(states.shape[1], len(inputs))
    y = np.zeros(T)
    u = (inputs[:T] + 1) / 2 * 0.5
    for t in range(order, T):
        y[t] = (0.3 * y[t-1]
                + 0.05 * y[t-1] * np.sum(y[t-order:t])
                + 1.5 * u[t-1] * u[t-order]
                + 0.1)
        y[t] = np.tanh(y[t])
    sp = washout + (T - washout) // 2
    X = states[:, washout:T].T
    yt = y[washout:T]
    s = sp - washout
    if s < 20 or len(yt) - s < 20:
        return 0.0
    w = _ridge(X[:s], yt[:s])
    pred = X[s:] @ w
    y2 = yt[s:]
    ss_res = np.sum((y2 - pred) ** 2)
    ss_tot = np.sum((y2 - y2.mean()) ** 2)
    return max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0


def waveform_classification(states, inputs, washout=500, n_classes=4):
    """N-class waveform classification accuracy."""
    T = states.shape[1]
    sp = washout + (T - washout) // 2
    bounds = np.linspace(-1, 1, n_classes + 1)
    labels = np.digitize(inputs[:T], bounds[1:-1])
    X = states[:, washout:T].T
    yl = labels[washout:T]
    s = sp - washout
    preds = np.zeros((T - washout - s, n_classes))
    for c in range(n_classes):
        w = _ridge(X[:s], (yl[:s] == c).astype(float))
        preds[:, c] = X[s:] @ w
    return (np.argmax(preds, axis=1) == yl[s:]).mean()


def mackey_glass(states, washout=500, tau=17, n_ahead=1):
    """Mackey-Glass chaotic time series prediction.

    Generates the MG series: dx/dt = β*x(t-τ)/(1+x(t-τ)^n) - γ*x(t)
    and measures prediction R² at n_ahead steps.

    This is THE standard chaotic prediction benchmark for reservoir computing.

    Args:
        states: (N, T) reservoir state matrix
        washout: Washout period
        tau: Delay parameter (17 = chaotic)
        n_ahead: Prediction horizon

    Returns:
        float: R² of prediction
    """
    T = states.shape[1]

    # Generate Mackey-Glass series
    mg = np.zeros(T + tau + 100)
    mg[:tau + 1] = 0.9  # Initial condition
    beta, gamma, n_exp = 0.2, 0.1, 10
    for t in range(tau, len(mg) - 1):
        x_tau = mg[t - tau]
        mg[t + 1] = mg[t] + (beta * x_tau / (1 + x_tau**n_exp) - gamma * mg[t]) * 1.0
    mg = mg[100:100 + T]  # Discard transient

    sp = washout + (T - washout) // 2
    X = states[:, washout:T - n_ahead].T
    y = mg[washout + n_ahead:T]
    s = sp - washout
    if s < 20 or len(y) - s < 20:
        return 0.0
    w = _ridge(X[:s], y[:s])
    pred = X[s:] @ w
    y2 = y[s:]
    ss_res = np.sum((y2 - pred) ** 2)
    ss_tot = np.sum((y2 - y2.mean()) ** 2)
    return max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0


def kernel_rank(states, washout=500, threshold=0.01):
    """Kernel quality / rank — measures nonlinear transformation capacity.

    Counts the number of significant singular values in the state matrix.
    Higher rank = richer nonlinear transformation = better reservoir.

    Args:
        states: (N, T) reservoir state matrix
        washout: Washout period
        threshold: Relative threshold for significant singular values

    Returns:
        int: effective rank (number of significant singular values)
    """
    X = states[:, washout:].T  # (T-washout, N)
    X = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if S[0] < 1e-10:
        return 0
    relative = S / S[0]
    return int((relative > threshold).sum())


def generalization_rank(states, inputs, washout=500, n_patterns=50):
    """Generalization rank — measures generalization vs memorization.

    Trains on n_patterns different target functions and measures
    how many the reservoir can simultaneously approximate.

    Args:
        states: (N, T) reservoir state matrix
        inputs: (T,) input signal
        washout: Washout period
        n_patterns: Number of random target patterns

    Returns:
        int: number of patterns with R² > 0.5
    """
    T = states.shape[1]
    sp = washout + (T - washout) // 2
    rng = np.random.RandomState(42)
    n_good = 0

    for p in range(n_patterns):
        # Random nonlinear target: polynomial of delayed inputs
        d = rng.randint(1, 10)
        power = rng.choice([1, 2, 3])
        y = np.roll(inputs[:T], d) ** power
        X = states[:, washout:T].T
        yt = y[washout:T]
        s = sp - washout
        if s < 20 or len(yt) - s < 20:
            continue
        w = _ridge(X[:s], yt[:s])
        pred = X[s:] @ w
        y2 = yt[s:]
        ss_res = np.sum((y2 - pred) ** 2)
        ss_tot = np.sum((y2 - y2.mean()) ** 2)
        r2 = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0
        if r2 > 0.5:
            n_good += 1

    return n_good


def nonlinear_memory_capacity(states, inputs, washout=500, max_delay=10, max_degree=3):
    """Nonlinear memory capacity — memory × nonlinearity tradeoff.

    Measures capacity for delayed nonlinear functions:
    y(t) = u(t-d)^k for d=1..max_delay, k=1..max_degree.

    Total NMC = Σ_d,k r²(d,k). Subsumes linear MC.

    Args:
        states: (N, T) state matrix
        inputs: (T,) input signal
        washout: Washout period

    Returns:
        dict: 'total_nmc', 'linear_mc', 'nonlinear_mc', 'per_delay_degree'
    """
    T = states.shape[1]
    sp = washout + (T - washout) // 2
    nmc_grid = np.zeros((max_delay, max_degree))

    for d in range(1, max_delay + 1):
        for k in range(1, max_degree + 1):
            X = states[:, washout + d:].T
            y = inputs[washout:T - d] ** k
            s = sp - washout - d
            if s < 20 or len(y) - s < 20:
                continue
            w = _ridge(X[:s], y[:s])
            pred = X[s:] @ w
            y2 = y[s:]
            if np.std(y2) < 1e-10 or np.std(pred) < 1e-10:
                continue
            r2 = np.corrcoef(pred, y2)[0, 1] ** 2
            nmc_grid[d - 1, k - 1] = r2

    return {
        'total_nmc': float(nmc_grid.sum()),
        'linear_mc': float(nmc_grid[:, 0].sum()),
        'nonlinear_mc': float(nmc_grid[:, 1:].sum()),
        'per_delay_degree': nmc_grid,
    }


def rc_benchmark(reservoir, n_steps=3000, washout=500, n_reps=5,
                  seed=42, verbose=True):
    """Run full reservoir computing benchmark suite.

    Args:
        reservoir: NSRAMReservoir instance (or any object with .transform())
        n_steps: Number of timesteps
        washout: Initial washout period
        n_reps: Number of repetitions
        seed: Random seed for input generation
        verbose: Print results

    Returns:
        dict with mean and std of all metrics

    Example:
        >>> from nsram import NSRAMReservoir, rc_benchmark
        >>> res = NSRAMReservoir(128, stp='heterogeneous')
        >>> results = rc_benchmark(res)
        >>> print(f"XOR-1: {results['xor1_mean']:.1%}")
    """
    rng = np.random.RandomState(seed)
    inputs = rng.uniform(-1, 1, n_steps).astype(np.float64)

    all_metrics = {k: [] for k in ['xor1', 'xor2', 'xor5', 'mc',
                                     'narma5', 'narma10', 'wave4']}

    for rep in range(n_reps):
        states = reservoir.transform(inputs)

        all_metrics['xor1'].append(xor_accuracy(states, inputs, washout, 1))
        all_metrics['xor2'].append(xor_accuracy(states, inputs, washout, 2))
        all_metrics['xor5'].append(xor_accuracy(states, inputs, washout, 5))
        all_metrics['mc'].append(memory_capacity(states, inputs, washout))
        all_metrics['narma5'].append(narma_prediction(states, inputs, washout, 5))
        all_metrics['narma10'].append(narma_prediction(states, inputs, washout, 10))
        all_metrics['wave4'].append(waveform_classification(states, inputs, washout))

        if verbose:
            m = {k: v[-1] for k, v in all_metrics.items()}
            print(f"  rep {rep}: XOR1={m['xor1']:.1%} MC={m['mc']:.3f} "
                  f"NARMA10={m['narma10']:.3f} Wave4={m['wave4']:.1%}")

    results = {}
    for k, v in all_metrics.items():
        results[f'{k}_mean'] = np.mean(v)
        results[f'{k}_std'] = np.std(v)
    results['n_reps'] = n_reps
    results['n_steps'] = n_steps

    if verbose:
        print(f"\n  Summary ({n_reps} reps):")
        for k in ['xor1', 'mc', 'narma10', 'wave4']:
            m, s = results[f'{k}_mean'], results[f'{k}_std']
            fmt = f'{m:.1%}±{s:.1%}' if k in ('xor1', 'wave4') else f'{m:.3f}±{s:.3f}'
            print(f"    {k}: {fmt}")

    return results
