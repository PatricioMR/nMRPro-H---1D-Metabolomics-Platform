# processing/quant_joint_fit.py
"""Measure overlapping analytes together, with room for what is not known.

A window integrates everything inside it. Where an analyte's signals sit
alone that is the right measurement -- an inter-laboratory study on a
synthetic urine of known composition found integration trueness better than
5% for about nine metabolites in ten when the signals were isolated -- and
where they overlap it is the wrong one: the same study found the
deconvolution workflows the more accurate on overlapping signals, and on
the reference bench filled to 100 analytes the window-integrated amounts
read 28% high, rising to +73% below 0.1 mM, because the neighbours' area is
inside the window.

So the analytes whose shapes overlap are fitted together, sample by
sample: the observed trace over their union is a non-negative sum of their
shapes (each column the one the cascade builds -- every listed signal,
area per proton, at the listed positions plus the study's drift) and of
FREE components, one Lorentzian at every detected peak no quantified
analyte has a line under. The free components are what the published
methods all carry and this engine did not: a template for what is not in
the library (BATMAN models it with wavelets, COLMAR1d keeps unassigned
peaks, ASICS a residual). Without them an unknown analyte's area has
nowhere to go but into a known one's amount. The group shares one shift
and one width per sample, searched over a fraction of a linewidth and a
few width factors, because what moves or broadens a sample moves or
broadens every resonance in it.

Analytes that overlap nothing keep their integrated amount: the fit is
only asked where integration is known to fail.

References
----------
Hao J, Astle W, De Iorio M, Ebbels TMD. BATMAN -- an R package for the
automated quantification of metabolites from NMR spectra using a Bayesian
model. Bioinformatics 2012;28(15):2088-2090. And: Hao J et al. Bayesian
deconvolution and quantification of metabolites in complex 1D NMR spectra
using BATMAN. Nat Protoc 2014;9:1416-1427.

Tardivel PJC, Canlet C, Lefort G, Tremblay-Franco M, Debrauwer L,
Concordet D, Servien R. ASICS: an automatic method for identification and
quantification of metabolites in complex 1D 1H NMR spectra. Metabolomics
2017;13:109.

Canlet C, Deborde C, Cahoreau E, et al. NMR metabolite quantification of a
synthetic urine sample: an inter-laboratory comparison of processing
workflows. Metabolomics 2023;19(7):65. doi:10.1007/s11306-023-02028-4.

COLMAR1d: a web server for automated, quantitative one-dimensional NMR-
based metabolomics at arbitrary magnetic fields (2024), PMC11525900.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

# A shape's support: where it stands above this share of its own tallest
# point. Lorentzian tails never reach zero; below a fiftieth they are not
# worth a place in anyone's fit.
SUPPORT_FRACTION = 0.02
# Which detected peaks are known: every line of every quantified signal
# claims the NEAREST detected apex within this many linewidths, one each;
# an apex no line claims is not in the library. Nearest, not "under the
# shape": judged by the height of a known shape at the apex, lactate's
# weak outer quartet lines read as nobody's and took 13% of lactate, and an
# unknown 1.2 linewidths beside a known singlet read as the singlet's --
# the Lorentzian is still 16% of its top there. One claim per line lets a
# known line displaced by the reference's own pH keep its peak, and leaves
# a second peak beside it free.
CLAIM_REACH_LW = 1.0
# A detected peak must stand this many noise sigmas on the mean trace to be
# given a free component; below it the fit would be modelling noise.
FREE_MIN_SNR = 5.0
# The group's common shift per sample, searched within this many
# linewidths, in this many steps.
SHIFT_REACH_LW = 0.5
SHIFT_STEPS = 7
# The group's common width per sample, as a factor on the study's measured
# linewidth. The linewidth is measured on the mean trace, and a mean of
# samples each a little shifted is broader than any of them: 3.35 Hz on the
# reference study, whose samples are 2.6 Hz. Shapes that much too wide put
# a triplet's area into the singlet it sits on. BATMAN lets the width vary
# for the same reason.
WIDTH_SCALES = (0.7, 0.8, 0.9, 1.0, 1.15)
# Each SIGNAL's own offset from where the reference puts it, found once on
# the mean trace and then held: within this many linewidths, on a grid of
# this many steps, in this many passes over the group. The reference was
# measured at its own pH and temperature, and even its own two
# descriptions of one signal need not agree -- taurine's doublet at 3.406
# sits 2.6 Hz apart between its measured profile and its measured line
# pattern, a whole linewidth. A shape placed a linewidth off loses its area
# to whatever sits where it should have been: taurine read -36% and the
# betaine singlet under it +26%. BATMAN lets every multiplet move on its
# own; ASICS aligns locally.
PART_SHIFT_REACH_LW = 1.0
PART_SHIFT_STEPS = 9
PART_SHIFT_PASSES = 2


def _lorentz_unit(ppm, centre, fwhm):
    """A unit-area Lorentzian."""
    g = float(fwhm) / 2.0
    return (g / np.pi) / ((np.asarray(ppm, float) - float(centre)) ** 2 + g * g)


def _support(val, idx):
    top = float(np.max(val)) if val.size else 0.0
    if top <= 0:
        return np.array([], dtype=int)
    return idx[val >= SUPPORT_FRACTION * top]


def free_components(columns, peaks, ppm, mean, *, sigma, linewidth_ppm,
                    weight=None):
    """``[(centre_ppm, fwhm_ppm)]`` for the detected peaks no quantified
    signal's line claims, standing FREE_MIN_SNR above the noise."""
    ppm = np.asarray(ppm, float)
    w = np.ones(ppm.size) if weight is None else np.asarray(weight, float)
    apices = []
    for p in peaks or ():
        try:
            c = float(p.get('ppm'))
        except (TypeError, ValueError):
            continue
        i = int(np.argmin(np.abs(ppm - c)))
        if w[i] <= 0 or not sigma or float(mean[i]) < FREE_MIN_SNR * float(sigma):
            continue
        apices.append(c)
    if not apices:
        return []
    apices = np.asarray(sorted(apices))
    claimed = np.zeros(apices.size, dtype=bool)
    reach = CLAIM_REACH_LW * float(linewidth_ppm)
    for col in columns:
        for comp, _nh in col.parts:
            for line in (comp.lines or ()):
                at = float(line) + float(comp.shift_ppm)
                d = np.abs(apices - at)
                k = int(np.argmin(d))
                if d[k] <= reach:
                    claimed[k] = True
    return [(float(c), float(linewidth_ppm))
            for c, taken in zip(apices, claimed) if not taken]


def _dense(col, union, shift=0.0, width=1.0, *, ppm):
    """A column's curve over the group's union (sorted indices), dense."""
    v = np.zeros(union.size)
    idx, val = col.curve(ppm, shift, width)
    if idx is None:
        return v
    k = np.searchsorted(union, idx)
    ok = (k < union.size)
    ok[ok] &= union[k[ok]] == idx[ok]
    np.add.at(v, k[ok], val[ok])
    return v


def _solve(A, t):
    """Non-negative least squares: the amplitudes and the squared residual.
    The active-set solver -- tens of columns over thousands of points is
    exactly its case, and it is an order of magnitude faster than a
    general bounded solver there."""
    x, rnorm = nnls(A, t, maxiter=50 * max(A.shape[1], 1))
    return x, float(rnorm) ** 2


def refine_part_shifts(cols, free_block, ppm, t, union, linewidth_ppm):
    """Slide every signal of every column on its own over the mean trace,
    holding the best offset. Coordinate descent: one signal at a time, the
    rest of the group fitted with it. Returns ``{(mid, i): offset_ppm}``.

    The offsets are written on the components (``shift_ppm``), so every
    later curve of the column carries them."""
    reach = PART_SHIFT_REACH_LW * float(linewidth_ppm)
    grid = np.linspace(-reach, reach, PART_SHIFT_STEPS)
    vecs = [_dense(c, union, ppm=ppm) for c in cols]
    base = {id(comp): float(comp.shift_ppm) for c in cols for comp, _nh in c.parts}
    out = {}
    for _pass in range(PART_SHIFT_PASSES):
        for k, col in enumerate(cols):
            for i, (comp, _nh) in enumerate(col.parts):
                best = None
                for d in grid:
                    comp.shift_ppm = base[id(comp)] + d
                    vk = _dense(col, union, ppm=ppm)
                    A = np.column_stack(vecs[:k] + [vk] + vecs[k + 1:]
                                        + ([free_block] if free_block is not None
                                           and free_block.size else []))
                    _x, r2 = _solve(A, t)
                    if best is None or r2 < best[0]:
                        best = (r2, d, vk)
                comp.shift_ppm = base[id(comp)] + best[1]
                vecs[k] = best[2]
                out[(col.mid, i)] = best[1]
    return out


def _groups(nodes):
    """Connected components of nodes whose supports share a point."""
    n = len(nodes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sets = [set(s.tolist()) for s in nodes]
    for a in range(n):
        for b in range(a + 1, n):
            if sets[a] and sets[b] and not sets[a].isdisjoint(sets[b]):
                parent[find(a)] = find(b)
    out: dict = {}
    for a in range(n):
        out.setdefault(find(a), []).append(a)
    return list(out.values())


def joint_measure(columns, peaks, ppm, X, included, *, sigma, linewidth_ppm,
                  weight=None):
    """Fit the overlapping analytes together, per sample.

    ``columns`` is the cascade's ``Column`` for every quantified analyte.
    Returns ``{mid: {'amplitudes': {sample: area per proton}, 'group':
    [mid, ...], 'free': n, 'residual': fraction}}`` for the analytes that
    share their support with another analyte or with a free component;
    the others are left to their integration and do not appear.

    ``residual`` is the median over the samples of the share of the
    group's trace the fit leaves unexplained -- how much of the region
    neither the library nor the free components account for.
    """
    ppm = np.asarray(ppm, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    included = [int(s) for s in included]
    if not columns or not included:
        return {}
    w = np.ones(ppm.size) if weight is None else np.asarray(weight, float)
    mean = X[:, included].mean(axis=1)

    def supports():
        out_ = []
        for col in columns:
            idx, val = col.curve(ppm, 0.0)
            out_.append(_support(val, idx) if idx is not None
                        else np.array([], dtype=int))
        return out_

    # 1. Every signal to its own place first, on the mean trace, among the
    #    analytes it overlaps. Who is unknown is decided after: judged
    #    against the listed positions, glutamate's own lines at 2.346 and
    #    myo-inositol's triplet at 3.614 -- each placed more than a
    #    linewidth off by the reference -- went unclaimed, became free
    #    components, and took their analytes' area.
    offsets: dict = {}
    col_sup = supports()
    for grp in _groups(col_sup):
        union = np.unique(np.concatenate([col_sup[k] for k in grp]))
        union = union[w[union] > 0]
        if union.size < 3:
            continue
        offsets.update(refine_part_shifts([columns[k] for k in grp], None,
                                          ppm, mean[union], union,
                                          linewidth_ppm))

    # 2. What no signal's line now claims is not in the library.
    frees = free_components(columns, peaks, ppm, mean, sigma=sigma,
                            linewidth_ppm=linewidth_ppm, weight=w)

    # 3. Who is fitted with whom, on the supports as they now stand.
    col_sup = supports()
    free_sup = []
    for c, fw in frees:
        sel = np.flatnonzero(np.abs(ppm - c) <= 4.0 * fw)
        v = _lorentz_unit(ppm[sel], c, fw)
        free_sup.append(_support(v, sel))
    nodes = col_sup + free_sup
    n_col = len(columns)

    step = float(linewidth_ppm) * SHIFT_REACH_LW
    shifts = np.linspace(-step, step, SHIFT_STEPS) if SHIFT_STEPS > 1 else [0.0]

    out: dict = {}
    for grp in _groups(nodes):
        cols_in = [k for k in grp if k < n_col]
        frees_in = [k - n_col for k in grp if k >= n_col]
        if not cols_in or len(grp) < 2:
            continue                 # alone: integration measures it
        union = np.unique(np.concatenate([nodes[k] for k in grp]))
        union = union[w[union] > 0]
        if union.size < 3:
            continue
        designs = []
        for wsc in WIDTH_SCALES:
            for sh in shifts:
                cols_m = [_dense(columns[k], union, sh, wsc, ppm=ppm)
                          for k in cols_in]
                frees_m = [_lorentz_unit(ppm[union], frees[f][0] + sh,
                                         frees[f][1] * wsc) for f in frees_in]
                designs.append(np.column_stack(cols_m + frees_m))
        amps = {columns[k].mid: {} for k in cols_in}
        resid = []
        for s in included:
            t = X[union, s]
            best = None
            for A in designs:
                x_, r2 = _solve(A, t)
                if best is None or r2 < best[0]:
                    best = (r2, x_)
            r2, x = best
            for j, k in enumerate(cols_in):
                amps[columns[k].mid][s] = float(x[j])
            tt = float(np.sum(t ** 2))
            resid.append(r2 / tt if tt > 0 else float('nan'))
        members = [columns[k].mid for k in cols_in]
        for mid in members:
            out[mid] = {'amplitudes': amps[mid], 'group': members,
                        'free': len(frees_in),
                        'residual': float(np.nanmedian(resid))
                        if resid else float('nan'),
                        'signal_offsets_ppm': [v for (m, _i), v in
                                               offsets.items() if m == mid]}
    return out
