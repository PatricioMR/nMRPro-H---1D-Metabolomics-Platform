# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
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
from scipy.optimize import lsq_linear, nnls

from processing.quant_cascade import amplitude_cap


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
# The width searched in every sample rather than once per group: p90 16% ->
# 12% on the 18-analyte tube, nothing on the 50, and four times the time.
WIDTH_PER_SAMPLE = False
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
#
# Kept at 1.0 with the samples aligned (2026-09-30), although the aligned
# mean trace reads its linewidth 25% narrower (4.5 mppm against 6.0 on the
# 100-analyte GISSMO tube at 600 MHz) and 1.3 would restore the reach in
# ppm: three dots 33 -> 37 real there, but an unknown peak twice as tall
# 0.8 linewidth beside a known singlet then takes 25% of it (+3% at 1.0).
# The bench holds no compound outside the library; a real sample does.
#
# 2.0 since 2026-10-11 (ADR-076): in a real extract every signal stands off
# listing + drift by its own amount -- the Mice ISS skin study a median 5.4
# mppm, a linewidth 4.3 -- and at 1.0 the fit called a real shape misplaced.
# GISSMO tube of 100, three seeds, each environment moved 5 mppm: found 213
# -> 215, strangers 114 -> 107, two dots or more 153/49 -> 166/37, three
# dots 14/3 -> 32/5; without the scatter 241/73 -> 242/80, two dots or more
# 192/13 -> 194/14. 3.0 found more in the scatter (227) but strangers 94 in
# the plain tube. Steps keep the 0.25-linewidth grid.
PART_SHIFT_REACH_LW = 2.0
PART_SHIFT_STEPS = 17
PART_SHIFT_PASSES = 2
# How far a neighbour's amount may move from the group's while one signal
# is being placed: the cascade's own tolerance for an analyte it did not
# measure (quant_cascade.FIXED_TOLERANCE).
NEIGHBOUR_TOLERANCE = 0.25
# The share of the fit's time that placing the signals takes, for the
# progress the fit reports: 193 of 309 s on the Mice ISS study.
PLACING_SHARE = 0.6
# No analyte's amount may draw any of its signals taller than the trace
# stands there (``quant_cascade.amplitude_cap``, the cascade's bound since
# ADR-028). Without it the fit is free to buy one tall peak with an analyte
# whose other lines stand on nothing: on the Mice ISS skin extract the
# water's foot below the excluded band (4.65-4.70 ppm, 82 000 sigma) went to
# eight compounds with a line there, saligenin's aromatic lines drawn at
# 9 600 sigma over a trace of 3, and the overshoot of those phantom lines
# zeroed the analytes underneath -- 175 of the 221 columns (92 with it).
# GISSMO tube of 100, 600 MHz, three seeds: found 249 -> 254, strangers 57
# -> 48, two dots 149/20 -> 156/8, three dots 41/0 -> 45/0, error median
# 19 -> 17%, p90 68 -> 65%; scattered 5 mppm: found 211 -> 204, strangers
# 111 -> 105, three dots 32/4 -> 36/4, p90 100 -> 81% (ADR-094).
AMPLITUDE_CAP = True


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


def _solve(A, t, hi=None):
    """Non-negative least squares: the amplitudes and the squared residual.
    The active-set solver -- tens of columns over thousands of points is
    exactly its case, and it is an order of magnitude faster than a
    general bounded solver there.

    Solved on the triangular factor of ``A`` (``A = QR``): ``|Ax - t|^2 =
    |Rx - Q't|^2 + |t - QQ't|^2``, the second term independent of ``x`` --
    the same problem, minimum and residual, over n rows instead of the
    group's thousands of points. The active set refits its passive columns
    at every step; over the full trace that was 52 ms a solve and half the
    joint fit's time.

    The factor is taken of ``[A | t]`` with R alone: its last column is
    ``Q't`` and its corner is the part of ``t`` outside ``A``'s span, so Q
    -- thousands of points by every column -- is never formed. Same
    problem, same answer to rounding (1e-13), half the time (0.17 -> 0.09 s
    on 20 000 x 200).

    ``hi`` bounds each amplitude from above (AMPLITUDE_CAP); the bounded
    active set then solves the same reduced problem."""
    m, n = A.shape
    if n == 0:
        return np.zeros(0), float(np.sum(t ** 2))
    bounded = hi is not None and bool(np.any(np.isfinite(hi)))
    if m <= n:
        R, b, corner = A, t, 0.0
    else:
        Ra = np.linalg.qr(np.column_stack([A, t]), mode='r')
        R, b, corner = Ra[:n, :n], Ra[:n, n], float(Ra[n, n]) ** 2
    if not bounded:
        x, rnorm = nnls(R, b, maxiter=50 * n)
        return x, float(rnorm) ** 2 + corner
    x = lsq_linear(R, b, bounds=(np.zeros(n), np.asarray(hi, float)),
                   method='bvls').x
    # the bounded solver leaves rounding (1e-17) where the active set's
    # zero is exact, and a reference's support is what is above zero
    x[np.abs(R).max(axis=0) * x <= 1e-12 * float(np.max(np.abs(b)))] = 0.0
    return x, float(np.sum((R @ x - b) ** 2)) + corner


# A sample fits only the columns its group's reference holds. Each sample
# solved alone, over ~140 candidate columns, gives a peak to analyte A in one
# replicate and to analyte B in its neighbour: non-negative least squares is
# sparse, and which of two alternatives wins moves with the noise. On the
# Murilo triplicates (ADR-080) that was 197 cells where an analyte read zero
# in one replicate and positive in its mates, in windows 26 noises tall.
# The reference is the mean trace of the replicates, solved once; the columns
# it leaves at zero are out of every one of those replicates' fits, so they
# answer the same question. The group's mean and not the study's: a
# study-wide mean dilutes an analyte only some groups hold and erased five
# of RawMG's 21 (guanidinoacetate, positive in 17 of 32 samples, among them).
# A pull toward the reference's amplitudes (ridge) was tried and left 169 of
# the 197 cells. An analyte a group's mean does not hold is absent from every
# replicate of that group, consistently.
REFERENCE_SUPPORT = True


def _solve_on_support(A, t, keep, hi=None):
    """``_solve`` over the columns ``keep`` (indices); the others are zero."""
    if keep.size == 0 or keep.size == A.shape[1]:
        return _solve(A, t, hi)
    x_sub, r2 = _solve(A[:, keep], t, None if hi is None else hi[keep])
    x = np.zeros(A.shape[1])
    x[keep] = x_sub
    return x, r2


def _column_caps(part_sets, trace, weight, sigma, n_free=0):
    """Each column's AMPLITUDE_CAP, from its signals' sparse curves
    (``part_sets``, one list per column) over ``trace``; the free
    components, and every column with the cap off, are unbounded."""
    hi = np.full(len(part_sets) + n_free, np.inf)
    if AMPLITUDE_CAP and sigma:
        for j, parts in enumerate(part_sets):
            hi[j] = amplitude_cap(parts, trace, weight, sigma)
    return hi


def _part_vec(comp, nh, ppm, union):
    """One signal's curve over the group's union, area = its protons, at the
    component's current offset. The same shape ``Column.part_curves``
    draws, one signal at a time."""
    v = np.zeros(union.size)
    x_all = ppm[union]
    sel = np.flatnonzero((x_all >= comp.lo - abs(comp.shift_ppm))
                         & (x_all <= comp.hi + abs(comp.shift_ppm)))
    if sel.size < 3:
        return v
    x = x_all[sel]
    y = np.asarray(comp.model(x, comp.shift_ppm, comp.width_scale), dtype=float)
    top = float(y.max())
    if top <= 0:
        return v
    y = y / top
    order = np.argsort(x)
    area = float(np.trapezoid(y[order], x=x[order]))
    if area <= 0:
        return v
    v[sel] = y * (float(nh) / area)
    return v


def refine_part_shifts(cols, free_block, ppm, t, union, linewidth_ppm, *,
                       sigma=None, progress=None):
    """Slide every signal of every column on its own over the mean trace,
    holding the best offset. Coordinate descent: one signal at a time, the
    rest of the group fitted with it. Returns ``{(mid, i): offset_ppm}``.

    Each candidate is judged where it differs from the others -- the
    signal's own reach -- against the rest of the group held at the
    amounts one fit of the WHOLE group gives, with only the analyte's own
    amount free (in closed form). Refitting the whole group for every
    signal and every candidate took 4.6 hours on one 50-analyte tube; a
    small free fit of the neighbourhood alone was fast and wrong, because
    a neighbour free there is not held by its own signals elsewhere (p90
    48% -> 55% on that tube).

    The offsets are written on the components (``shift_ppm``), so every
    later curve of the column carries them. ``sigma`` bounds the group's
    amounts (AMPLITUDE_CAP); None leaves them unbounded. ``progress(done,
    total, unit, note)`` is told each analyte placed, and may raise to stop
    the run."""
    reach = PART_SHIFT_REACH_LW * float(linewidth_ppm)
    # nearest zero first, and only a strict improvement moves a signal: a
    # signal whose analyte is not under it fits equally at every offset,
    # and the first of the grid -- a whole reach away -- used to win,
    # which read as 'at the edge' and got the analyte refused
    grid = sorted(np.linspace(-reach, reach, PART_SHIFT_STEPS), key=abs)
    x_u = ppm[union]
    parts = [[_part_vec(comp, nh, ppm, union) for comp, nh in c.parts]
             for c in cols]
    V = np.column_stack([np.sum(ps, axis=0) if ps else np.zeros(union.size)
                         for ps in parts]) if cols else np.zeros((union.size, 0))
    F = (free_block if free_block is not None and np.size(free_block)
         else np.zeros((union.size, 0)))
    n_c = V.shape[1]
    base = {id(comp): float(comp.shift_ppm) for c in cols for comp, _nh in c.parts}
    ones = np.ones(t.size)

    def part_cap(pv):
        # one signal's AMPLITUDE_CAP; a column's is the least of its signals'
        if not (AMPLITUDE_CAP and sigma):
            return np.inf
        idx = np.flatnonzero(pv)
        if not idx.size:
            return np.inf           # a signal outside the union draws nothing
        return amplitude_cap([(idx, pv[idx])], t, ones, sigma)

    # kept per signal and redone only for the signal that moved: rebuilt for
    # every column at every step it was half the joint fit's time
    caps = [[part_cap(pv) for pv in ps] for ps in parts]
    out = {}
    # The group's amounts change only when a signal moves. Most do not --
    # an analyte that is not there stays where it is listed -- and solving
    # the whole group again for nothing was 60% of this function's time.
    amp, moved = None, True
    for _pass in range(PART_SHIFT_PASSES):
        for k, col in enumerate(cols):
            if progress is not None:
                progress(_pass * len(cols) + k, PART_SHIFT_PASSES * len(cols),
                         'analytes', 'placing each signal')
            # the group's amounts, with this column as it now stands
            if moved:
                hi = np.array([min(c, default=np.inf) for c in caps]
                              + [np.inf] * F.shape[1])
                amp, _r2 = _solve(np.column_stack([V, F]), t, hi)
                moved = False
            a_col, a_free = amp[:n_c], amp[n_c:]
            for i, (comp, nh) in enumerate(col.parts):
                lo = comp.lo - abs(base[id(comp)]) - 2 * reach
                hi = comp.hi + abs(base[id(comp)]) + 2 * reach
                L = np.flatnonzero((x_u >= lo) & (x_u <= hi))
                if L.size < 3:
                    continue
                # the neighbours with lines here may give or take up to
                # NEIGHBOUR_TOLERANCE of their group amount; the rest are
                # held; this analyte's own amount is free
                near = [j for j in range(n_c) if j != k and np.any(V[L, j])]
                near_f = [j for j in range(F.shape[1]) if np.any(F[L, j])]
                far = [j for j in range(n_c) if j != k and j not in near]
                held = V[L][:, far] @ a_col[far] if far else 0.0
                if F.shape[1]:
                    far_f = [j for j in range(F.shape[1]) if j not in near_f]
                    if far_f:
                        held = held + F[L][:, far_f] @ a_free[far_f]
                target = t[L] - held
                rest = V[L, k] - parts[k][i][L]
                nb = [a_col[j] for j in near] + [a_free[j] for j in near_f]
                lo_b = np.array([0.0] + [max(a * (1 - NEIGHBOUR_TOLERANCE), 0.0)
                                         for a in nb])
                hi_b = np.array([np.inf] + [max(a * (1 + NEIGHBOUR_TOLERANCE), 1e-12)
                                            for a in nb])
                Nb = [V[L, j] for j in near] + [F[L, j] for j in near_f]
                was = comp.shift_ppm
                best = None
                for d in grid:
                    comp.shift_ppm = base[id(comp)] + d
                    pv = _part_vec(comp, nh, ppm, union)
                    A = np.column_stack([rest + pv[L]] + Nb)
                    # the bounded active set: exact, and a few columns
                    # is its case (the iterative default spent ~1 ms a
                    # candidate on overhead, a third of the joint fit)
                    res = lsq_linear(A, target, bounds=(lo_b, hi_b),
                                     method='bvls')
                    r2 = float(np.sum((A @ res.x - target) ** 2))
                    if best is None or r2 < best[0] * (1.0 - 1e-9):
                        best = (r2, d, pv)
                comp.shift_ppm = base[id(comp)] + best[1]
                out[(col.mid, i)] = best[1]
                if comp.shift_ppm == was:
                    continue            # the same curve: nothing changes
                V[:, k] += best[2] - parts[k][i]
                parts[k][i] = best[2]
                caps[k][i] = part_cap(best[2])
                moved = True
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


def _reference_groups(groups, included):
    """The sample sets that each follow one reference: a group of replicates
    (two or more of them included), and the samples outside any such group
    together. With no groups, the whole study."""
    inc = [int(s) for s in included]
    if not groups:
        return [inc]
    out, seen = [], set()
    for idxs in groups.values():
        members = [int(i) for i in idxs if int(i) in inc]
        if len(members) >= 2:
            out.append(members)
            seen.update(members)
    rest = [s for s in inc if s not in seen]
    if len(rest) >= 2:
        out.append(rest)
    return out


def joint_measure(columns, peaks, ppm, X, included, *, sigma, linewidth_ppm,
                  weight=None, baseline=None, groups=None, progress=None):
    """Fit the overlapping analytes together, per sample.

    ``columns`` is the cascade's ``Column`` for every quantified analyte.
    Returns ``{mid: {'amplitudes': {sample: area per proton}, 'group':
    [mid, ...], 'free': n, 'residual': fraction, 'signal_offsets':
    [(listed centre ppm, fitted offset ppm), ...]}}`` for the analytes that
    share their support with another analyte or with a free component;
    the others are left to their integration and do not appear.

    ``residual`` is the median over the samples of the share of the
    group's trace the fit leaves unexplained -- how much of the region
    neither the library nor the free components account for.

    ``groups`` is ``{label: [sample, ...]}``, the replicates. Each group
    follows its own reference (REFERENCE_SUPPORT), so an analyte only some
    groups hold is not erased by the others that dilute it in a study-wide
    mean.

    ``baseline`` (the shape of ``X``, ``quant_integration.cluster_baseline``)
    is subtracted before anything is fitted: fixed, not fitted, because a
    free constant or slope in the fit takes the pedestal the overlapping
    tails build under a crowded region, which is area, and hands it to
    nobody. None fits the trace as it stands.

    ``progress(done, total, unit, note, frac=)`` is told how far the fit
    is -- the count of the step it is in, and ``frac`` of the whole fit --
    and may raise to stop the run.
    """
    ppm = np.asarray(ppm, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    included = [int(s) for s in included]
    if not columns or not included:
        return {}
    if baseline is not None:
        X = X - np.asarray(baseline, float).reshape(X.shape)
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
    col_groups = _groups(col_sup)
    n_place = max(sum(len(g) for g in col_groups), 1)
    placed = 0
    for grp in col_groups:
        union = np.unique(np.concatenate([col_sup[k] for k in grp]))
        union = union[w[union] > 0]
        if union.size < 3:
            placed += len(grp)
            continue

        def placing(done, total, unit='', note='', _at=placed, _n=len(grp)):
            progress(done, total, unit, note, frac=PLACING_SHARE * (
                _at + _n * done / max(total, 1)) / n_place)
        offsets.update(refine_part_shifts(
            [columns[k] for k in grp], None, ppm, mean[union], union,
            linewidth_ppm, sigma=sigma,
            progress=placing if progress is not None else None))
        placed += len(grp)

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
    fit_groups = _groups(nodes)
    for i_g, grp in enumerate(fit_groups):
        cols_in = [k for k in grp if k < n_col]
        frees_in = [k - n_col for k in grp if k >= n_col]
        if not cols_in or len(grp) < 2:
            continue                 # alone: integration measures it
        union = np.unique(np.concatenate([nodes[k] for k in grp]))
        union = union[w[union] > 0]
        if union.size < 3:
            continue
        def design(sh, wsc):
            cols_m = [_dense(columns[k], union, sh, wsc, ppm=ppm)
                      for k in cols_in]
            frees_m = [_lorentz_unit(ppm[union], frees[f][0] + sh,
                                     frees[f][1] * wsc) for f in frees_in]
            return np.column_stack(cols_m + frees_m)

        # the group's width once, on the mean trace: what broadens it
        # broadens every sample alike, and five widths in every sample
        # multiplied the fit by five for nothing the samples disagree on
        def caps(sh, wsc):
            parts = [columns[k].part_curves(ppm, sh, wsc) for k in cols_in]
            return lambda trace: _column_caps(parts, trace, w, sigma,
                                              len(frees_in))

        best_w = None
        for wsc in WIDTH_SCALES:
            _x, r2 = _solve(design(0.0, wsc), mean[union],
                            caps(0.0, wsc)(mean))
            if best_w is None or r2 < best_w[0]:
                best_w = (r2, wsc)
        grid = ([(sh, wsc) for wsc in WIDTH_SCALES for sh in shifts]
                if WIDTH_PER_SAMPLE else [(sh, best_w[1]) for sh in shifts])
        designs = [design(sh, wsc) for sh, wsc in grid]
        cap_of = [caps(sh, wsc) for sh, wsc in grid]
        amps = {columns[k].mid: {} for k in cols_in}
        resid = []
        chosen = {}
        support_of = {}
        if REFERENCE_SUPPORT and len(included) > 1:
            for members in _reference_groups(groups, included):
                ref = X[:, members].mean(axis=1)
                fits = [_solve(A, ref[union], cap(ref))
                        for A, cap in zip(designs, cap_of)]
                keep = np.flatnonzero(
                    fits[int(np.argmin([f[1] for f in fits]))][0] > 0)
                support_of.update({s: keep for s in members})
        for i_s, s in enumerate(included):
            if progress is not None:
                progress(i_s, len(included), 'samples', 'fitting each sample',
                         frac=PLACING_SHARE + (1 - PLACING_SHARE) * (
                             i_g + i_s / len(included)) / len(fit_groups))
            t = X[union, s]
            best = None
            for i_d, A in enumerate(designs):
                hi = cap_of[i_d](X[:, s])
                x_, r2 = (_solve(A, t, hi) if s not in support_of else
                          _solve_on_support(A, t, support_of[s], hi))
                if best is None or r2 < best[0]:
                    best = (r2, x_, i_d)
            r2, x, i_d = best
            chosen[s] = float(shifts[i_d % len(shifts)])
            for j, k in enumerate(cols_in):
                amps[columns[k].mid][s] = float(x[j])
            tt = float(np.sum(t ** 2))
            resid.append(r2 / tt if tt > 0 else float('nan'))
        members = [columns[k].mid for k in cols_in]
        for mid in members:
            out[mid] = {'amplitudes': amps[mid], 'group': members,
                        # where each sample's shapes stood and how wide,
                        # so a report draws the curve that was fitted
                        'shift_by_sample': dict(chosen),
                        'width_scale': float(best_w[1]),
                        'offset_limit': float(linewidth_ppm)
                        * PART_SHIFT_REACH_LW,
                        'free': len(frees_in),
                        'residual': float(np.nanmedian(resid))
                        if resid else float('nan'),
                        'signal_offsets': [
                            ((comp.lo + comp.hi) / 2.0, offsets[(mid, i)])
                            for c in columns if c.mid == mid
                            for i, (comp, _nh) in enumerate(c.parts)
                            if (mid, i) in offsets]}
    return out
