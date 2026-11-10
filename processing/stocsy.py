# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/stocsy.py
"""
STOCSY -- Statistical Total Correlation Spectroscopy.

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic
    1H NMR data sets. Analytical Chemistry 77(5):1282-1289.

Signals of one molecule rise and fall together across samples because they
share one concentration. A seed (a driver peak) is correlated with every
other point of the spectrum; the covariance is the height of the trace and
the correlation its colour.

    SPA-STOCSY (2023). Bioinformatics 39:btad593. (clusters of co-varying
    points linked to a library; states the limit kept here: a molecule whose
    concentration does not vary across the samples has nothing to correlate.)
    Fisher, R.A. (1915). Biometrika 10:507-521. (z transform of r, used for
    the interval.)
    Fieller, E.C., Hartley, H.O., Pearson, E.S. (1957). Biometrika 44:
    470-481. (variance of the Spearman z.)

Two entry points, kept independent:

  compute_peak_correlation_matrix  -- Graph STOCSY (peak x peak Pearson r)
                                      used for annotation scoring
  stocsy_trace / compute_stocsy_spectrum
                                   -- Seed STOCSY (seed -> every spectral
                                      point), shown in the STOCSY window

What r is measured on is a ``Basis`` and nothing else:

  * raw areas (what the annotation scoring reads);
  * PQN -- every sample divided by its dilution factor (Dieterle et al. 2006,
    Anal Chem 78:4281). Measured on the synthetic bench, a 30% dilution CV
    moves the share of DIFFERENT molecules reading r >= 0.80 from 0.3% to
    6.9%, and PQN brings it back to 0.6%. The factor is the median quotient
    over PEAKS, so a molecule with many peaks dominates it and PQN also
    divides away part of that molecule's own variation: with no dilution it
    costs recall (39% -> 22% of same-molecule pairs at r >= 0.80);
  * within-group centering -- the group means are removed, so a pair that
    only moves together because two molecules are higher in one group does
    not count. The p-value then has N - groups - 1 degrees of freedom.

Neither function imports from GUI layers.
"""
import numpy as np
from scipy import stats

# Fewer samples than this and a correlation of 0.80 is not evidence: r >= 0.80
# is reached by chance in about 10% of random pairs at n = 5, 40% at n = 3.
STOCSY_MIN_SAMPLES = 5

# The correlation the annotation scoring asks of a partner
# (annotation_scoring.STOCSY_MIN_R): where significance needs more than this,
# the analyst is told.
ANNOTATION_BAR = 0.80

METHODS = ('pearson', 'spearman')
NORMALIZATIONS = ('none', 'pqn')

# Why a peak was left out of the matrix. The analyst is told; a peak that
# silently has no correlation reads as "not corroborated".
SKIP_REASONS = {
    'no_areas': 'no per-sample areas',
    'too_few_samples': 'too few samples',
    'non_finite': 'a sample area is NaN/inf',
    'length_mismatch': 'area vector length differs from the other peaks',
    'flat': 'identical area in every sample (nothing to correlate)',
}


# ── inference on r ───────────────────────────────────────────────────────────

def critical_r(dof, alpha=0.05) -> float:
    """Smallest |r| that is significant (two-sided) at ``alpha`` with ``dof``
    residual degrees of freedom; 1.0 when there are none."""
    if dof is None or dof < 1:
        return 1.0
    t = float(stats.t.isf(alpha / 2.0, dof))
    return float(t / np.sqrt(dof + t * t))


def r_pvalue(r, dof):
    """Two-sided p-value of r with ``dof`` residual degrees of freedom."""
    r = np.asarray(r, dtype=float)
    if dof is None or dof < 1:
        return np.full(r.shape, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = r * np.sqrt(dof / (1.0 - r * r))
    return 2.0 * stats.t.sf(np.abs(t), dof)


def r_interval(r, dof, method='pearson', level=0.95):
    """Confidence interval of r by Fisher's z. NaN below 2 degrees of freedom,
    where the standard error does not exist."""
    r = np.asarray(r, dtype=float)
    if dof is None or dof < 2:
        nan = np.full(r.shape, np.nan)
        return nan, nan.copy()
    se = (np.sqrt(1.06) if method == 'spearman' else 1.0) / np.sqrt(dof - 1.0)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    h = float(stats.norm.isf((1.0 - level) / 2.0)) * se
    return np.tanh(z - h), np.tanh(z + h)


# ── what r is measured on ────────────────────────────────────────────────────

def pqn_factors(M):
    """Dilution factor of every sample (column of ``M``, signals x samples):
    the median quotient against the median profile, over the signals that are
    positive in both. 1.0 where fewer than three can be compared."""
    from processing.quant_concentration import apply_pqn
    M = np.asarray(M, dtype=float)
    by_signal = {i: {s: float(v) for s, v in enumerate(row) if np.isfinite(v)}
                 for i, row in enumerate(M)}
    _, f = apply_pqn(by_signal)
    return np.array([float(f.get(s, 1.0)) for s in range(M.shape[1])])


def dilution_spread(M) -> float:
    """sd/mean of the PQN factors: how much of the sample-to-sample variation
    is the whole sample being more or less concentrated."""
    f = pqn_factors(M)
    return float(np.std(f) / np.mean(f)) if f.size and np.mean(f) > 0 else 0.0


class Basis:
    """What the correlation is taken over, applied identically to every
    vector -- peaks, the seed, each point of the spectrum.

    ``groups`` (one label per sample) is needed only for ``within_groups``.
    Groups of one sample have no within-group variation and are dropped, and
    that is said in ``notes``; without two usable groups the option is
    refused (``within_groups`` False) and said, never silently ignored.
    """

    def __init__(self, n_samples, *, groups=None, factors=None,
                 within_groups=False, method='pearson'):
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}, got {method!r}')
        n = int(n_samples)
        self.method = method
        self.notes: list[str] = []
        self.factors = None if factors is None else np.asarray(factors, float)
        if self.factors is not None and self.factors.size != n:
            raise ValueError('one normalisation factor per sample is needed')
        keep = np.arange(n)
        self._gid = None
        self.n_groups = 0
        self.within_groups = False
        if within_groups:
            if groups is None or len(groups) != n:
                self.notes.append('Within-group correlation needs the sample '
                                  'groups; all samples used together.')
            else:
                lab = np.asarray([str(g) for g in groups])
                names, counts = np.unique(lab, return_counts=True)
                usable = names[counts >= 2]
                if len(usable) < 2:
                    self.notes.append(
                        'Within-group correlation needs two groups with at '
                        'least two samples each; all samples used together.')
                else:
                    in_use = np.isin(lab, usable)
                    if not in_use.all():
                        self.notes.append(
                            f'{int((~in_use).sum())} sample(s) in groups of '
                            'one left out of the within-group correlation.')
                    keep = np.flatnonzero(in_use)
                    self._gid = np.searchsorted(usable, lab[keep])
                    self.n_groups = len(usable)
                    self.within_groups = True
        self.keep = keep
        self.n = int(keep.size)
        self.dof = (self.n - self.n_groups - 1 if self.within_groups
                    else self.n - 2)
        if self.n < STOCSY_MIN_SAMPLES or critical_r(self.dof) > ANNOTATION_BAR:
            self.notes.append(
                f'Only {self.n} samples: |r| must reach '
                f'{critical_r(self.dof):.2f} to be significant (p < 0.05). '
                'Read r as a hint, not as evidence.')

    def scale(self, V):
        """``V`` (variables x all samples) restricted to this basis' samples
        and divided by their normalisation factors: the values as the
        correlation sees them before any ranking or centring."""
        V = np.atleast_2d(np.asarray(V, dtype=float))[:, self.keep]
        return V if self.factors is None else V / self.factors[self.keep]

    def apply(self, V, *, ranked=True):
        """``V`` on this basis: scaled, then ranked (Spearman) and centred
        within group."""
        V = self.scale(V)
        if ranked and self.method == 'spearman':
            V = stats.rankdata(V, axis=1)
        if self.within_groups:
            V = V.copy()
            for g in range(self.n_groups):
                idx = np.flatnonzero(self._gid == g)
                V[:, idx] -= V[:, idx].mean(axis=1, keepdims=True)
        return V

    def correlate(self, A, B):
        """r of every row of ``A`` against every row of ``B`` (a x b)."""
        return _pearson_cross(self.apply(A), self.apply(B))

    def stats(self, r):
        """``(p, ci_low, ci_high)`` of r on this basis' degrees of freedom."""
        lo, hi = r_interval(r, self.dof, self.method)
        return r_pvalue(r, self.dof), lo, hi

    @property
    def r_critical(self) -> float:
        return critical_r(self.dof)


def _pearson_cross(A, B):
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    na = np.sqrt((Ac * Ac).sum(axis=1))
    nb = np.sqrt((Bc * Bc).sum(axis=1))
    flat_a = ~(na > 1e-9 * np.abs(A).max(axis=1))
    flat_b = ~(nb > 1e-9 * np.abs(B).max(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        R = (Ac @ Bc.T) / (na[:, None] * nb[None, :])
    R[flat_a, :] = np.nan
    R[:, flat_b] = np.nan
    return np.clip(R, -1.0, 1.0)


def build_basis(M_all, *, groups=None, normalize='none', within_groups=False,
                method='pearson') -> Basis:
    """A ``Basis`` for a study whose peak areas are ``M_all`` (peaks x
    samples). The PQN factors are taken from ALL the peaks, whatever subset is
    later correlated: a dilution is read over the whole sample."""
    if normalize not in NORMALIZATIONS:
        raise ValueError(f'normalize must be one of {NORMALIZATIONS}, '
                         f'got {normalize!r}')
    M_all = np.atleast_2d(np.asarray(M_all, dtype=float))
    factors = pqn_factors(M_all) if normalize == 'pqn' else None
    return Basis(M_all.shape[1], groups=groups, factors=factors,
                 within_groups=within_groups, method=method)


# ── Graph STOCSY: peak x peak ────────────────────────────────────────────────

def peak_vectors(peaks, *, included=None, min_samples=STOCSY_MIN_SAMPLES):
    """``(ranks, M, skipped)``: the per-sample area vectors that can be
    correlated (M is peaks x samples) and ``{rank: reason}`` for those that
    cannot, the reasons being keys of ``SKIP_REASONS``."""
    ranks, rows, skipped = [], [], {}
    width = None
    for p in peaks:
        if p.get('rank') is None:
            continue
        rank = int(p['rank'])
        aps = p.get('area_per_sample')
        arr = None if aps is None else np.asarray(aps, dtype=float)
        if arr is None or arr.ndim != 1:
            skipped[rank] = 'no_areas'
            continue
        if included is not None:
            arr = arr[[i for i in included if 0 <= i < len(arr)]]
        if len(arr) < min_samples:
            skipped[rank] = 'too_few_samples'
        elif not np.all(np.isfinite(arr)):
            skipped[rank] = 'non_finite'
        elif width is not None and len(arr) != width:
            skipped[rank] = 'length_mismatch'
        else:
            width = len(arr)
            ranks.append(rank)
            rows.append(arr)
    return ranks, (np.vstack(rows) if rows else np.empty((0, 0))), skipped


def peak_correlation_report(peaks, min_samples=STOCSY_MIN_SAMPLES,
                            included=None):
    """Pairwise Pearson r between every pair of detected peaks, and what was
    left out.

    Returns ``{'corr': {(rank_i, rank_j): r}, 'skipped': {rank: reason},
    'n_samples': int}`` with rank_i < rank_j. A peak is in ``skipped`` when it
    has no usable area vector (see ``SKIP_REASONS``); a pair is absent only
    because one of its peaks is.

    ``included`` must name the analytical samples. Blanks belong outside it,
    and leaving them in does not merely add noise -- it manufactures
    correlation. Every peak falls to nothing in a blank, so every pair shares
    that descent and reads as co-varying whether or not the molecules have
    anything to do with each other. Measured on 32 samples at a 20% CV with
    five blanks appended: two INDEPENDENT analytes read r = +0.74 to +0.80,
    against |r| < 0.25 over the samples alone.
    """
    ranks, M, skipped = peak_vectors(peaks, included=included,
                                     min_samples=min_samples)
    n_samples = int(M.shape[1]) if M.size else 0
    if len(ranks) < 2:
        return {'corr': {}, 'skipped': skipped, 'n_samples': n_samples}
    flat = np.std(M, axis=1) <= 0
    for r, f in zip(ranks, flat):
        if f:
            skipped[r] = 'flat'
    if (~flat).sum() < 2:
        return {'corr': {}, 'skipped': skipped, 'n_samples': n_samples}
    kept = [r for r, f in zip(ranks, flat) if not f]
    # One call over the whole matrix: a 1047-peak study is 548k pairs, which
    # took ~25 s one np.corrcoef at a time.
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.corrcoef(M[~flat])
    corr = {}
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            r = float(R[i, j])
            if np.isfinite(r):
                lo, hi = ((kept[i], kept[j]) if kept[i] < kept[j]
                          else (kept[j], kept[i]))
                corr[(lo, hi)] = r
    return {'corr': corr, 'skipped': skipped, 'n_samples': n_samples}


def compute_peak_correlation_matrix(peaks, min_samples=STOCSY_MIN_SAMPLES,
                                    included=None):
    """``{(rank_i, rank_j): r}`` of ``peak_correlation_report``, for the
    annotation scoring. ``peaks`` carry ``'rank'`` and ``'area_per_sample'``."""
    return peak_correlation_report(peaks, min_samples, included)['corr']


def rank_partners(seed_vec, ranks, M, basis, *, seed_rank=None):
    """The peaks ``M`` (rows named by ``ranks``) ordered by how strongly they
    co-vary with ``seed_vec`` on ``basis``, strongest first.

    Each entry: ``rank, r, p, ci_low, ci_high, n, dof``. NaN r (a flat
    vector) goes last and is kept, so the list names every peak asked about.
    """
    if len(ranks) == 0:
        return []
    r = basis.correlate(np.asarray(seed_vec, float)[None, :], M)[0]
    p, lo, hi = basis.stats(r)
    out = [{'rank': int(k), 'r': float(r[i]), 'p': float(p[i]),
            'ci_low': float(lo[i]), 'ci_high': float(hi[i]),
            'n': basis.n, 'dof': basis.dof}
           for i, k in enumerate(ranks) if k != seed_rank]
    out.sort(key=lambda d: (not np.isfinite(d['r']), -d['r']
                            if np.isfinite(d['r']) else 0.0))
    return out


# ── why two signals correlate: all samples, within groups, between groups ────

def covariation_parts(seed_vec, M, groups, *, factors=None, method='pearson'):
    """r of ``seed_vec`` with every row of ``M`` over all samples, within the
    groups (the group means taken out) and between the group means.

    Two signals can correlate because the molecule is the same, because the
    two molecules both differ between groups, or because both follow a
    sample-level factor (the amount of tissue) that moves inside the groups
    too. The split says which: a high r that is all ``between`` is a group
    effect (STORM, Posma et al. 2012: correlations driven by population
    differences), one that survives ``within`` is a shared sample factor or
    one molecule. ``groups`` is one label per sample; the between r is NaN
    with fewer than three groups. ``factors`` are the normalisation factors
    of ``build_basis`` (PQN) or None.
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    seed = np.asarray(seed_vec, dtype=float)[None, :]
    n = M.shape[1]
    base = Basis(n, factors=factors, method=method)
    r_all = base.correlate(seed, M)[0]
    within = Basis(n, groups=groups, factors=factors, within_groups=True,
                   method=method)
    r_within = (within.correlate(seed, M)[0] if within.within_groups
                else np.full(M.shape[0], np.nan))
    lab = np.asarray([str(g) for g in groups])
    names = sorted(set(lab))
    if len(names) >= 3:
        S = base.scale(seed)[0]
        V = base.scale(M)
        keep = base.keep
        lk = lab[keep]
        Sg = np.array([S[lk == u].mean() for u in names])[None, :]
        Vg = np.column_stack([V[:, lk == u].mean(axis=1) for u in names])
        r_between = _pearson_cross(Sg, Vg)[0]
    else:
        r_between = np.full(M.shape[0], np.nan)
    return {'all': r_all, 'within': r_within, 'between': r_between}


# ── fixed ratio between signals of one molecule ──────────────────────────────

# Area noise of an integrated window against what white noise of the spectrum's
# own level gives (sigma_point * dx * sqrt(points)). Measured on the blank tubes
# of the Mice skin study (429 peaks): 2.2 (IQR 1.4-3.2) times it -- the foot line
# and the baseline add to what the sum of independent points gives.
AREA_NOISE_KAPPA = 2.2


def area_noise(sigma_point, dx, n_points, kappa=AREA_NOISE_KAPPA):
    """sd of the area of a window of ``n_points`` points: ``kappa`` times what
    white noise of ``sigma_point`` per point gives."""
    return float(kappa) * float(sigma_point) * abs(float(dx)) * np.sqrt(
        np.asarray(n_points, dtype=float))


def fixed_ratio(seed_vec, M, sigma_seed, sigma_rows, *, min_valid=8,
                min_area_sigma=3.0):
    """How constant the ratio of every row of ``M`` to ``seed_vec`` is across
    the samples, against what measurement noise alone leaves.

    The signals of one molecule stand in a fixed ratio (the protons they hold,
    a multiplet's line fractions), whatever the concentration of the molecule
    or the dilution of the sample; both cancel in a ratio. Two molecules do
    not keep a ratio unless their amounts move in lockstep.

        Wei, S. et al. (2011). Ratio analysis nuclear magnetic resonance
        spectroscopy for selective metabolite identification in complex
        samples. Analytical Chemistry 83(20):7616-7623.

    Only the samples where both areas stand ``min_area_sigma`` noises over
    zero are read (a log of a noisy zero is nothing). ``sigma_seed`` and
    ``sigma_rows`` are the area noises (``area_noise``). Returns arrays over the
    rows of ``M``:

      sd         sd of log(row / seed) over the samples read
      noise_sd   what the noise alone gives (delta method, mean over samples)
      excess_sd  sqrt(max(sd^2 - noise_sd^2, 0)): the variation of the ratio
                 that noise does not explain (0.1 = 10%)
      q, p       reduced chi-square of the ratio against a constant and its
                 upper-tail p: small p = the ratio varies more than noise
      ratio      median of row / seed
      n          samples read; NaN everywhere below ``min_valid``
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    seed = np.asarray(seed_vec, dtype=float)
    sig = np.asarray(sigma_rows, dtype=float)[:, None]
    ok = (M > min_area_sigma * sig) & (seed[None, :] > min_area_sigma * float(sigma_seed))
    n = ok.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        lr = np.where(ok, np.log(M / seed[None, :]), np.nan)
        v = np.where(ok, (sig / M) ** 2 + (float(sigma_seed) / seed[None, :]) ** 2, np.nan)
        w = np.where(ok, 1.0 / v, 0.0)
        mu = np.nansum(w * np.nan_to_num(lr), axis=1) / np.maximum(w.sum(axis=1), 1e-300)
        Q = np.nansum(w * np.nan_to_num(lr - mu[:, None]) ** 2, axis=1)
        mean_lr = np.nanmean(lr, axis=1)
        sd = np.sqrt(np.nansum(np.nan_to_num(lr - mean_lr[:, None]) ** 2, axis=1)
                     / np.maximum(n - 1, 1))
        noise_sd = np.sqrt(np.nanmean(v, axis=1))
        ratio = np.exp(np.nanmedian(lr, axis=1))
    dof = np.maximum(n - 1, 1)
    q = Q / dof
    p = stats.chi2.sf(Q, dof)
    excess = np.sqrt(np.maximum(sd ** 2 - noise_sd ** 2, 0.0))
    bad = n < int(min_valid)
    out = {'sd': sd, 'noise_sd': noise_sd, 'excess_sd': excess, 'q': q, 'p': p,
           'ratio': ratio, 'n': n}
    for k, a in out.items():
        if k != 'n':
            a[bad] = np.nan
    return out


# ── Seed STOCSY: seed -> spectrum ────────────────────────────────────────────

def stocsy_trace(ppm, X, seed_vec, basis, *, masked_regions=(),
                 min_std_factor=None):
    """Correlate ``seed_vec`` (one value per sample) with every spectral point.

    ``X`` is points x samples. Returns ``{'r', 'cov', 'std'}``, each over the
    points: r on ``basis``; ``cov`` the covariance with the seed (Cloarec's
    trace height) and ``std`` the spread across samples, both on the
    normalised, un-ranked values. NaN where a point does not vary at all.

    ``masked_regions`` are (high, low) ppm pairs -- the residual water -- set
    to NaN. ``min_std_factor`` sets to NaN the points whose spread across
    samples is under that many times the median spread: the median over the
    whole spectrum is the noise floor, and below it r is the correlation of
    noise with the seed.
    """
    ppm = np.asarray(ppm, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    seed = np.asarray(seed_vec, dtype=float)[None, :]
    r = _pearson_cross(basis.apply(seed), basis.apply(X))[0]
    Xt, st = basis.apply(X, ranked=False), basis.apply(seed, ranked=False)
    n = Xt.shape[1]
    cov = ((Xt - Xt.mean(axis=1, keepdims=True))
           @ (st - st.mean()).T)[:, 0] / max(n - 1, 1)
    std = Xt.std(axis=1, ddof=1) if n > 1 else np.zeros(len(ppm))
    if min_std_factor is not None and np.any(std > 0):
        low = std < float(min_std_factor) * float(np.median(std))
        r, cov = np.where(low, np.nan, r), np.where(low, np.nan, cov)
    for hi, lo in masked_regions or ():
        m = (ppm <= max(hi, lo)) & (ppm >= min(hi, lo))
        r[m] = np.nan
        cov[m] = np.nan
    return {'r': r, 'cov': cov, 'std': std}


def compute_stocsy_spectrum(ppm, all_int, seed_ppm, seed_window_ppm=0.02, *,
                            included=None, basis=None):
    """Seed-based STOCSY: correlate a seed region against every spectral point.

    The seed vector is the mean intensity across the seed window, per sample.
    ``included`` are the sample columns to use (blanks stay out, as in Graph
    STOCSY); ``basis`` is a ``Basis`` over those samples (raw Pearson when
    None). Returns Pearson r at each point, NaN where either side does not
    vary.
    """
    ppm = np.asarray(ppm, dtype=float)
    X = np.asarray(all_int, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if included is not None:
        X = X[:, list(included)]
    mask = np.abs(ppm - seed_ppm) <= seed_window_ppm
    if not mask.any():
        return np.full(len(ppm), np.nan)
    seed_vec = X[mask, :].mean(axis=0)
    if basis is None:
        basis = Basis(X.shape[1])
    return stocsy_trace(ppm, X, seed_vec, basis)['r']
