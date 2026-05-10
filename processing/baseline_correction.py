# processing/baseline_correction.py
"""
Baseline correction algorithms for NMR spectroscopy.

Implements:
  - airPLS (adaptive iteratively reweighted Penalized Least Squares)
  - SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
  - Rolling ball minimum
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def airPLS(y, lam=1e5, p=0.01, niter=100, return_weights=False):
    """
    airPLS - adaptive iteratively reweighted Penalized Least Squares.

    Baseline estimation algorithm that adaptively adjusts weights to
    converge to the true baseline underneath peaks.

    Parameters
    ----------
    y : ndarray (n,)
        Input signal (can have negative values)
    lam : float
        Smoothing parameter (larger = smoother baseline)
        Typical range: 1e4 to 1e7 for NMR
    p : float
        Asymmetry parameter (0 < p < 1)
        Controls how aggressively peaks are excluded
        Typical: 0.001-0.01 for NMR (small = only positive deviations penalized)
    niter : int
        Maximum number of iterations
    return_weights : bool
        If True, also return final weights array

    Returns
    -------
    baseline : ndarray (n,)
        Estimated baseline
    weights : ndarray (n,) (optional)
        Final iteration weights (only if return_weights=True)

    References
    ----------
    Zhang, Z.-M., et al. (2010). Anal. Chem., 82, 832-842.
    """
    n = len(y)
    if n < 10:
        return np.zeros(n) if return_weights else np.zeros(n)

    # Initialize weights to ones
    w = np.ones(n)
    W = sparse.diags(w)

    # Second difference matrix (for curvature penalty)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n-2))

    # Precompute (W + lam * D * D^T)^-1 structure
    # We'll update W each iteration
    Z = lam * D.dot(D.T)

    for iteration in range(niter):
        # Solve for baseline: (W + Z) * z = W * y
        Q = W + Z

        # Use sparse solver
        try:
            z = spsolve(Q, w * y)
        except Exception:
            # Fallback to dense solver for small systems
            z = np.linalg.solve(Q.toarray(), w * y)

        # Calculate residuals (difference between data and baseline)
        d = y - z

        # Update weights: downweight positive residuals (peaks)
        # Only points below baseline (negative d) keep high weight
        # Points above baseline (positive d, peaks) get exponentially decreasing weight
        w_negative = d < 0
        w[w_negative] = 1.0
        w[~w_negative] = p * np.exp(-d[~w_negative] / (np.std(d[d < 0]) + 1e-10))

        # Check convergence
        if iteration > 0:
            # Check if weights have stabilized
            w_change = np.sum(np.abs(w - w_prev)) / n
            if w_change < 1e-4:
                break

        w_prev = w.copy()
        W = sparse.diags(w)

    if return_weights:
        return z, w
    return z


def snip(y, window=20, niter=100, direction='both'):
    """
    SNIP - Statistics-sensitive Non-linear Iterative Peak-clipping.

    Aggressive peak-clipping algorithm that iteratively removes peaks
    to estimate the baseline. Works well for complex, sloping baselines.

    Parameters
    ----------
    y : ndarray (n,)
        Input signal
    window : int
        Half-window size for clipping operation
        Larger = more aggressive baseline estimation
    niter : int
        Number of iterations
    direction : str
        'both', 'increasing', or 'decreasing'
        Controls which directions are considered for clipping

    Returns
    -------
    baseline : ndarray (n,)
        Estimated baseline

    References
    ----------
    Ryan, C. G., et al. (1988). Nucl. Instr. Meth. B, 34, 396-402.
    """
    n = len(y)
    if n < 5:
        return np.zeros(n)

    # Work with absolute values for stability
    y_abs = np.abs(y.astype(float))
    L = y_abs.copy()

    # Clip parameter (typically 0.2-0.5)
    clip = 0.2

    for iteration in range(niter):
        L_new = L.copy()

        for i in range(window, n - window):
            # Check left and right neighborhoods
            left_min = np.min(L[i-window:i])
            right_min = np.min(L[i+1:i+window+1])

            # SNIP clipping condition
            if direction == 'both':
                threshold = min(left_min, right_min)
            elif direction == 'increasing':
                threshold = left_min
            else:  # decreasing
                threshold = right_min

            # Apply clipping
            if L[i] > threshold:
                L_new[i] = threshold * (1 - clip) + L[i] * clip

        L = L_new

    # The clipped signal approximates the baseline
    # Smooth it slightly for NMR applications
    baseline = np.convolve(L, np.ones(5)/5, mode='same')

    return baseline


def rolling_ball_minimum(y, ball_radius=50):
    """
    Rolling ball minimum baseline estimation.

    Simple morphological operation: roll a ball of given radius
    underneath the signal and track the ball's center.

    Parameters
    ----------
    y : ndarray (n,)
        Input signal
    ball_radius : int
        Radius of the rolling ball in data points

    Returns
    -------
    baseline : ndarray (n,)
        Estimated baseline
    """
    n = len(y)
    if n < ball_radius:
        return np.zeros(n)

    # Use morphological erosion (minimum filter)
    from scipy.ndimage import minimum_filter1d

    # Erode with ball radius
    eroded = minimum_filter1d(y, size=ball_radius*2+1)

    # Smooth the result
    from scipy.ndimage import gaussian_filter1d
    baseline = gaussian_filter1d(eroded, sigma=ball_radius//4)

    return baseline


def estimate_baseline_level(y, method='median', edge_fraction=0.1):
    """
    Estimate a constant baseline level (for simple cases).

    Parameters
    ----------
    y : ndarray (n,)
        Input signal
    method : str
        'median', 'mean', 'mode', or 'edge'
    edge_fraction : float
        Fraction of data to use for 'edge' method (from both ends)

    Returns
    -------
    baseline_level : float
        Estimated constant baseline level
    """
    if method == 'median':
        return float(np.median(y))
    elif method == 'mean':
        return float(np.mean(y))
    elif method == 'mode':
        # Use histogram mode (robust to peaks)
        hist, bin_edges = np.histogram(y, bins=100)
        mode_idx = np.argmax(hist)
        return float((bin_edges[mode_idx] + bin_edges[mode_idx+1]) / 2)
    elif method == 'edge':
        # Use signal edges (assumes peaks are in the middle)
        n_edge = max(5, int(len(y) * edge_fraction))
        edge_values = np.concatenate([y[:n_edge], y[-n_edge:]])
        return float(np.median(edge_values))
    else:
        return 0.0


def correct_baseline(ppm, intensity, method='airPLS', **kwargs):
    """
    Apply baseline correction to NMR spectrum.

    Parameters
    ----------
    ppm : ndarray (n,)
        Chemical shift axis
    intensity : ndarray (n,)
        Raw intensity values
    method : str
        'airPLS', 'snip', 'rolling', or 'constant'
    **kwargs : dict
        Additional parameters passed to the specific method

    Returns
    -------
    corrected : ndarray (n,)
        Baseline-corrected intensity
    baseline : ndarray (n,)
        Estimated baseline

    Examples
    --------
    >>> corrected, baseline = correct_baseline(ppm, intensity,
    ...                                        method='airPLS', lam=1e6)
    """
    if len(intensity) != len(ppm):
        raise ValueError("ppm and intensity must have same length")

    # Ensure numpy arrays
    intensity = np.asarray(intensity, dtype=float)
    ppm = np.asarray(ppm, dtype=float)

    if method == 'airPLS':
        lam = kwargs.get('lam', 1e5)
        p = kwargs.get('p', 0.01)
        niter = kwargs.get('niter', 100)
        baseline = airPLS(intensity, lam=lam, p=p, niter=niter)
    elif method == 'snip':
        window = kwargs.get('window', 20)
        niter = kwargs.get('niter', 100)
        direction = kwargs.get('direction', 'both')
        baseline = snip(intensity, window=window, niter=niter, direction=direction)
    elif method == 'rolling':
        ball_radius = kwargs.get('ball_radius', 50)
        baseline = rolling_ball_minimum(intensity, ball_radius=ball_radius)
    elif method == 'constant':
        est_method = kwargs.get('est_method', 'median')
        edge_frac = kwargs.get('edge_fraction', 0.1)
        baseline = np.full_like(intensity,
                                estimate_baseline_level(intensity,
                                                        method=est_method,
                                                        edge_fraction=edge_frac))
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    corrected = intensity - baseline

    return corrected, baseline
