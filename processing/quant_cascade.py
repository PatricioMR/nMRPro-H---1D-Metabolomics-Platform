"""The cascade: analytes the election missed, found jointly on what the
quantified ones leave of the spectrum (ADR-028).

The election quantifies one peak at a time and an analyte enters only by
winning a peak. An analyte spread thin over several signals -- valine's
two doublets, leucine's, glutamine's three multiplets -- can lose every
one of them to a neighbour listed with a stronger single signal and
still be the only thing that explains the residual as a whole. The
cascade looks at that whole.

One column per analyte: the sum of the shapes of every signal the
reference lists for it (its measured profile, its measured pattern or a
first-order comb from the listed coupling; a singlet is one line), each
scaled to an area of its proton count and laid at the listed position
plus the study's drift. One amplitude per analyte therefore couples all
its signals -- the reference's relative intensities as a constraint,
not a count. Forward selection with a non-negative refit (orthogonal
matching pursuit): at each step every analyte not yet in is projected on
the current residual, allowed to slide within SHIFT_REACH_LW linewidths,
and the one removing the most squared residual enters if its shape then
stands at least CASCADE_MIN_SNR sigma at its tallest point; the active
set is refitted jointly and the residual updated. It stops when nothing
left clears the bar.

Every signal caps the amplitude. The trace is a sum of non-negative
shapes, so at each listed signal the analyte's height can be at most what
the trace holds there, plus noise. The amplitude is bounded above by the
tightest of these caps -- a physical bound, not a tolerance -- in the
selection and in every refit. An analyte whose projection asks for more
than CAP_RATIO times its cap is claiming a residual its own listing says
it cannot own (ADR-025's rule applied to the column) and does not enter
at that step.

The application runs it as a second opinion (``cascade_missed``): every
analyte the election quantified is subtracted, sample by sample, at the
amount it was measured at, and the cascade selects on the rest. It can
only add; nothing the election audited moves.

Selecting and measuring are not the same question, and are not asked the
same way. Selection runs on the residual with the quantified analytes'
own regions taken out of the weight, so that the remainder a subtraction
leaves cannot pass for an analyte. Measurement then runs on the spectrum
as it is (``measure_jointly``): the analytes taken in and the quantified
ones whose signals reach into their regions are fitted together, the
quantified ones free only within the precision their own measurement has
(``FIXED_TOLERANCE``), so an overlap is divided between shapes instead of
being cut out of one of them. The amount comes with the uncertainty that
fit gives it -- the standard error of the amplitude, which grows exactly
as an analyte's shape becomes a combination of its neighbours' -- and an
amount whose uncertainty is a large part of itself is reported as an
identification without a number.

Non-negative least squares: Lawson & Hanson, Solving Least Squares
Problems, Prentice-Hall 1974. Orthogonal matching pursuit: Pati,
Rezaiifar & Krishnaprasad, Proc. 27th Asilomar Conf. Signals, Systems and
Computers, 1993. A spectrum as a sum of reference shapes (targeted
profiling): Weljie, Newton, Mercier, Carlson & Slupsky, Anal. Chem. 78,
4430-4442, 2006.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from processing.quant_shared import component_from_evidence, listed_signal_evidence

CASCADE_MIN_SNR = 5.0       # the analyte's tallest point, at its amplitude
CASCADE_MAX_STEPS = 60
SHIFT_STEPS = 7             # within +- SHIFT_REACH_LW linewidths
SHIFT_REACH_LW = 0.5
CAP_RATIO = 3.0             # projection over cap beyond this: contradicted
CAP_NOISE_SIGMAS = 3.0      # what a region may hold above its trace
MIN_FREE_SHARE = 0.25       # of a column's weight, outside what is spoken for
# Below this share the analyte may still be identified, but its amount is
# not a measurement: measured leave-one-out on the bench, an analyte with
# 60% or more of its weight free came out within 5% (worst 17%), and one
# below that within 26% (worst 65%) -- and its ratio between two groups
# moved by as much, which would put a difference in the statistics that
# the spectrum does not hold.
MIN_FREE_TO_QUANTIFY = 0.6
FIXED_TOLERANCE = 0.25      # a quantified analyte's amplitude, in the joint fit
# The widest disagreement between an analyte's own signals that still
# leaves an amount worth printing. The standard error from the noise says
# nothing here -- it is under 1% for every analyte on the bench, including
# those out by a third -- because what is uncertain is not the noise but
# the division of an overlap.
#
# Measured two ways, which do not agree and are both kept in view. Leave
# one analyte out of the study and ask the cascade for it alone, and the
# statistic separates at a tenth: those inside come out to a median 7%
# and at worst 14%, those outside to a median 21%. Run the whole study
# and the analytes admitted between a tenth and three tenths come out
# fine -- at three tenths the bench finds sixteen of its eighteen instead
# of fourteen, with the same false positives and the same error at the
# ninetieth percentile. An analyte measured with everything else in place
# is not the analyte measured alone, and it is the first that a study
# reports, so it is the first that sets this.
MAX_SIGNAL_DISAGREEMENT = 0.30
OWNED_SHARE = 0.7           # of a point's height explained: it is theirs
SIGNAL_AGREEMENT_SNR = 5.0  # a signal quiet than this does not testify


def signal_protons(dp) -> float | None:
    """A listed signal's proton count: what the reference measured under
    its profile where it did, its listing otherwise; None when neither."""
    for key in ('profile_nh', 'protons'):
        v = dp.get(key)
        try:
            v = float(v) if v not in (None, '') else None
        except (TypeError, ValueError):
            v = None
        if v and v > 0:
            return v
    return None


class Column:
    """An analyte's shape over the spectrum, sparse, in area-per-proton
    units: one unit of amplitude is one unit of area under every proton."""

    def __init__(self, mid, name):
        self.mid, self.name = mid, name
        self.parts: list = []          # (component, nh)

    @property
    def nh_total(self) -> float:
        return float(sum(nh for _c, nh in self.parts))

    def part_curves(self, ppm, shift=0.0) -> list:
        """Each signal's sparse curve, area = its protons: [(idx, val)]."""
        out = []
        for comp, nh in self.parts:
            sel = np.flatnonzero((ppm >= comp.lo - abs(shift)) & (ppm <= comp.hi + abs(shift)))
            if sel.size < 3:
                continue
            x = ppm[sel]
            y = np.asarray(comp.model(x, comp.shift_ppm + shift, comp.width_scale), dtype=float)
            top = float(y.max())
            if top <= 0:
                continue
            y = y / top
            order = np.argsort(x)
            area = float(np.trapezoid(y[order], x=x[order]))
            if area <= 0:
                continue
            out.append((sel, y * (float(nh) / area)))
        return out

    def curve(self, ppm, shift=0.0):
        parts = self.part_curves(ppm, shift)
        if not parts:
            return None, None
        idx = np.concatenate([p[0] for p in parts])
        val = np.concatenate([p[1] for p in parts])
        # signals overlapping in ppm add up
        uniq, inv = np.unique(idx, return_inverse=True)
        out = np.zeros(uniq.size)
        np.add.at(out, inv, val)
        return uniq, out


def build_columns(metabolites, *, field_mhz, linewidth_ppm, drift_ppm,
                  signals_of=None) -> list:
    """One column per annotated metabolite from the signals the reference
    lists for it (``met['db_peaks']``, or what ``signals_of(met)`` returns).
    A signal with no shape (an unresolved 'm' the reference did not
    measure) or no proton count is left out; an analyte with no signal
    left has no column."""
    cols = []
    for met in metabolites or ():
        mid = met.get('id')
        col = Column(mid, met.get('name', mid))
        # the study's common drift, not the analyte's own from annotation:
        # a wrong match there would misplace every signal of the column;
        # the cascade slides instead
        rows = signals_of(met) if signals_of else (met.get('db_peaks') or ())
        for dp in rows:
            c0 = dp.get('ppm_center', dp.get('ppm'))
            if c0 is None:
                continue
            ev = listed_signal_evidence(dp, centre_ppm=float(c0) + float(drift_ppm),
                                        field_mhz=float(field_mhz))
            if ev is None:
                continue
            nh = signal_protons(dp)
            if not nh:
                continue
            comp = component_from_evidence(
                mid, ev, label=str(dp.get('type') or dp.get('multiplicity') or ''),
                linewidth_ppm=float(linewidth_ppm), nh=nh)
            if comp is not None:
                col.parts.append((comp, nh))
        if col.parts:
            cols.append(col)
    return cols


def exclusion_weight(ppm, excluded_regions=()) -> np.ndarray:
    """1 where the spectrum counts, 0 inside the regions detection was told
    to ignore (the water)."""
    w = np.ones(np.asarray(ppm).size)
    for a, b in excluded_regions or ():
        w[(ppm >= min(a, b)) & (ppm <= max(a, b))] = 0.0
    return w


def amplitude_cap(parts, trace, weight, sigma) -> float:
    """The largest amplitude every listed signal's region can hold: at each
    signal, the trace's height over the points where the shape stands at
    half its top or more, plus CAP_NOISE_SIGMAS sigma, over the shape's
    top. A signal in an excluded region says nothing."""
    cap = np.inf
    for idx, val in parts:
        w = weight[idx]
        if not np.any(w > 0):
            continue
        top = float(val.max())
        core = idx[(val >= 0.5 * top) & (w > 0)]
        if core.size == 0:
            continue
        held = max(float(trace[core].max()), 0.0) + CAP_NOISE_SIGMAS * float(sigma)
        cap = min(cap, held / top)
    return cap


def subtract_fixed(cols, ppm, X, fixed, *, sigma=None, included=None,
                   min_snr=CASCADE_MIN_SNR, owned_share=OWNED_SHARE):
    """``X`` (points x samples) minus every fixed analyte at its per-sample
    amplitude ``fixed[mid][sample]``; the columns left free; and the points
    the fixed analytes own.

    A point is theirs where, at the amounts they were measured at, they
    already account for ``owned_share`` of the height the spectrum has
    there. What a quantified analyte's window leaves after an imperfect
    subtraction -- lactate's doublet at 800 sigma leaves a percent of
    itself, wings included -- is that analyte's, not a residual to
    explain, and an analyte offered it would be fitting the error of a
    lineshape. The share matters: owning every point one of them merely
    reaches would hand a 14-sigma betaine tail the 178-sigma singlet of
    a creatine standing under it."""
    X = np.asarray(X, dtype=float)
    Y = X.copy()
    cover = np.zeros(np.asarray(ppm).size)
    free = []
    cols_i = [int(i) for i in (included if included is not None
                               else range(X.shape[1]))
              if 0 <= int(i) < X.shape[1]]
    for col in cols:
        amps = fixed.get(col.mid)
        if not amps:
            free.append(col)
            continue
        idx, val = col.curve(ppm, 0.0)
        if idx is None:
            continue
        good = []
        for s_i, a in amps.items():
            s_i = int(s_i)
            if a is None or not np.isfinite(a) or a <= 0 or not 0 <= s_i < Y.shape[1]:
                continue
            Y[idx, s_i] -= float(a) * val
            if s_i in cols_i:
                good.append(float(a))
        if good:
            cover[idx] += float(np.mean(good)) * val
    owned = np.zeros(np.asarray(ppm).size, dtype=bool)
    if sigma and cols_i:
        held = X[:, cols_i].mean(axis=1)
        owned = (cover >= float(owned_share) * held) & (
            cover >= float(min_snr) * float(sigma))
    return Y, free, owned


@dataclass
class Entry:
    col: Column
    shift_ppm: float
    idx: np.ndarray
    val: np.ndarray
    cap: float
    free_share: float = 1.0
    amplitude: float = 0.0
    apex_snr: float = 0.0
    step: int = 0


@dataclass
class CascadeResult:
    active: list = field(default_factory=list)          # [Entry]
    contradicted: dict = field(default_factory=dict)    # mid -> (projection, cap)
    stopped_at: tuple | None = None                     # (name, apex_snr) of the best left


def cascade(cols, ppm, trace, *, sigma, linewidth_ppm, weight=None,
            min_snr=CASCADE_MIN_SNR, max_steps=CASCADE_MAX_STEPS,
            cap_ratio=CAP_RATIO) -> CascadeResult:
    """Forward selection over ``cols`` on one trace (see the module)."""
    ppm = np.asarray(ppm, dtype=float)
    t_full = np.asarray(trace, dtype=float)
    weight = np.ones(ppm.size) if weight is None else np.asarray(weight, dtype=float)
    r = t_full * weight
    reach = SHIFT_REACH_LW * float(linewidth_ppm)
    shifts = np.linspace(-reach, reach, SHIFT_STEPS)
    cache = {}
    for col in cols:
        variants = []
        for s in shifts:
            parts = col.part_curves(ppm, s)
            if not parts:
                continue
            idx, val = col.curve(ppm, s)
            free_share = (float(np.dot(val * weight[idx], val * weight[idx]))
                          / float(np.dot(val, val)))
            val = val * weight[idx]
            nn = float(np.dot(val, val))
            if nn <= 0:
                continue
            cap = amplitude_cap(parts, t_full, weight, sigma)
            # An analyte whose signals lie where another analyte's are
            # already accounted for has no evidence of its own: what is
            # left there is the subtraction's remainder, and fitting the
            # wings of somebody else's peak is how a cysteamine appears
            # under a taurine. No cap means every signal was spoken for.
            if not np.isfinite(cap) or free_share < MIN_FREE_SHARE:
                continue
            variants.append((s, idx, val, nn, float(val.max()), cap, free_share))
        if variants:
            cache[col.mid] = variants
    out = CascadeResult()
    active = out.active
    for step in range(max_steps):
        best = None
        taken = {e.col.mid for e in active}
        for col in cols:
            if col.mid in taken or col.mid not in cache:
                continue
            for s, idx, val, nn, top, cap, free in cache[col.mid]:
                proj = float(np.dot(r[idx], val)) / nn
                if proj <= 0:
                    continue
                if proj > cap_ratio * cap:
                    out.contradicted[col.mid] = (proj, cap)
                    continue
                a = min(proj, cap)
                gain = 2.0 * a * proj * nn - a * a * nn   # residual removed at a
                if best is None or gain > best[0]:
                    best = (gain, col, s, idx, val, a, top, cap, free)
        if best is None:
            break
        gain, col, s, idx, val, a, top, cap, free = best
        apex_snr = a * top / float(sigma) if sigma else float('inf')
        if apex_snr < min_snr:
            out.stopped_at = (col.name, apex_snr)
            break
        active.append(Entry(col, float(s), idx, val, float(cap),
                            free_share=float(free), step=step + 1))
        # joint non-negative refit of the active set over the union of
        # their supports, every amplitude under its cap
        union = np.unique(np.concatenate([e.idx for e in active]))
        pos = {v: i for i, v in enumerate(union)}
        A = np.zeros((union.size, len(active)))
        for k, e in enumerate(active):
            A[[pos[v] for v in e.idx], k] = e.val
        ub = np.array([e.cap for e in active])
        res = lsq_linear(A, t_full[union] * weight[union], bounds=(0, ub), lsmr_tol='auto')
        amps = np.asarray(res.x, dtype=float)
        for k, e in enumerate(active):
            e.amplitude = float(amps[k])
            e.apex_snr = float(amps[k] * e.val.max() / sigma) if sigma else float('inf')
        r = t_full * weight
        r[union] -= A @ amps
    for mid in {e.col.mid for e in active}:
        out.contradicted.pop(mid, None)
    return out


def _design(cols_shifts, ppm, union_pos, n):
    """The design matrix of the given (column, shift) pairs over ``union``."""
    A = np.zeros((n, len(cols_shifts)))
    for k, (col, sh) in enumerate(cols_shifts):
        idx, val = col.curve(ppm, sh)
        if idx is None:
            continue
        keep = [(union_pos[v], w) for v, w in zip(idx, val) if v in union_pos]
        for row, w in keep:
            A[row, k] += w
    return A


def measure_jointly(active, owners, ppm, X, included, *, sigma, weight=None,
                    tolerance=FIXED_TOLERANCE):
    """The amounts of the analytes the cascade took in, fitted together
    with the quantified analytes that reach into their regions.

    ``owners`` is ``[(column, {sample: amplitude})]`` for the analytes the
    election measured. Each is free between ``1 -/+ tolerance`` of what it
    measured: its number stands, its precision does not pretend to be
    perfect, and an overlap is then divided between the two shapes rather
    than taken out of the newcomer. The newcomers are free from zero to
    the ceiling their own regions set.

    Returns ``({mid: {sample: amplitude}}, {mid: disagreement between the
    analyte's own signals}, {mid: share of samples the ceiling held down})``.

    The disagreement is the measurement's own error bar, and it is the
    one a chemist would ask for: each signal of the analyte is measured
    again on its own, with everything else in the fit taken off, and the
    widest departure from their median is reported as a fraction of it.
    An analyte whose doublet and triplet give the same amount has been
    measured; one whose signals differ by a third has been assigned, not
    measured. The standard error from the noise cannot say this -- at
    these heights it is a fraction of a percent while the amount is out
    by a third -- because the error here is not noise but the division of
    an overlap. An analyte with one signal, or one whose others are too
    faint to test, has no check and says so (NaN).
    """
    if not active:
        return {}, {}, {}
    ppm = np.asarray(ppm, dtype=float)
    w = np.ones(ppm.size) if weight is None else np.asarray(weight, dtype=float)
    pairs = [(e.col, e.shift_ppm) for e in active] + [(c, 0.0) for c, _a in owners]
    supports = []
    for col, sh in pairs:
        idx, _val = col.curve(ppm, sh)
        if idx is not None:
            supports.append(idx)
    union = np.unique(np.concatenate(supports))
    union = union[w[union] > 0]
    pos = {v: i for i, v in enumerate(union)}
    A = _design(pairs, ppm, pos, union.size)
    n_act = len(active)
    out, rel, hit = {}, {}, {}
    errs: dict = {}
    for s_i in included:
        t = np.asarray(X[:, s_i], dtype=float)
        lo = np.zeros(len(pairs))
        hi = np.empty(len(pairs))
        for k, e in enumerate(active):
            hi[k] = amplitude_cap(e.col.part_curves(ppm, e.shift_ppm), t, w, sigma)
        for k, (_c, amps) in enumerate(owners):
            a = float(amps.get(int(s_i), 0.0) or 0.0)
            lo[n_act + k] = max(a * (1.0 - tolerance), 0.0)
            hi[n_act + k] = max(a * (1.0 + tolerance), 1e-12)
        res = lsq_linear(A, t[union], bounds=(lo, hi), lsmr_tol='auto')
        x = np.asarray(res.x, dtype=float)
        for k, e in enumerate(active):
            out.setdefault(e.col.mid, {})[int(s_i)] = float(x[k])
            hit[e.col.mid] = hit.get(e.col.mid, 0) + int(
                np.isfinite(hi[k]) and x[k] >= 0.999 * hi[k])
        # What the analyte's own signals say on their own, once everything
        # else in the fit is taken off: if they are its signals, at one
        # amount, they agree. Where they do not, the amount rests on an
        # assignment the spectrum does not support, and no standard error
        # from the noise would show it -- at these heights that error is a
        # fraction of a percent while the amount is out by a third.
        fitted = A @ x
        for k, e in enumerate(active):
            if x[k] <= 0:
                continue
            per_signal = []
            for pidx, pval in e.col.part_curves(ppm, e.shift_ppm):
                rows_ = [pos[v] for v in pidx if v in pos]
                cols_ = [j for j, v in enumerate(pidx) if v in pos]
                if len(rows_) < 3:
                    continue
                v = pval[cols_]
                nn2 = float(np.dot(v, v))
                if nn2 <= 0 or float(v.max()) * x[k] < SIGNAL_AGREEMENT_SNR * sigma:
                    continue          # too faint to testify
                other = fitted[rows_] - x[k] * v
                a_i = float(np.dot(t[union][rows_] - other, v)) / nn2
                # weighted by what the signal is worth as a witness: its
                # squared norm, which grows with its protons and its
                # height, so a tall clean doublet outvotes a faint one
                per_signal.append((max(a_i, 0.0), nn2))
            if len(per_signal) >= 2:
                a = np.array([v for v, _w in per_signal])
                wt = np.array([w for _v, w in per_signal])
                med = float(np.median(a))
                if med > 0:
                    errs.setdefault(e.col.mid, []).append(
                        float(np.dot(wt, np.abs(a - med)) / wt.sum()) / med)
            elif per_signal:
                errs.setdefault(e.col.mid, []).append(None)   # nothing to check
    n = max(len(list(included)), 1)
    for e in active:
        got = [v for v in (errs.get(e.col.mid) or ()) if v is not None]
        rel[e.col.mid] = float(np.median(got)) if got else float('nan')
    return out, rel, {m: v / n for m, v in hit.items()}


def cascade_missed(metabolites, ppm, intensities, included, *, fixed_per_proton,
                   noise_sigma, linewidth_ppm, drift_ppm, field_mhz,
                   excluded_regions=(), min_snr=CASCADE_MIN_SNR,
                   diagnostics=None) -> dict:
    """What the cascade adds to a quantified study.

    ``fixed_per_proton`` is ``{mid: {sample: area per proton}}`` for every
    analyte the election quantified, in the spectrum's own units (before
    any normalisation): each is subtracted from every sample at that
    amount and its column is not offered. The cascade then runs on the
    mean of the included samples, and each analyte it takes in gets its
    amplitude per sample by a bounded non-negative refit on that sample.

    Returns ``{mid: {'name', 'per_proton': {sample: a}, 'nh', 'apex_snr',
    'shift_ppm', 'at_cap', 'signals': [(ppm, label, nh, predicted_snr)],
    'window': (lo, hi) of the tallest signal, 'step'}}`` in area-per-proton
    units, the same the election's windows give. ``diagnostics``, when a
    dict, gets the order taken, the analytes contradicted by their own
    listing and where the cascade stopped.
    """
    ppm = np.asarray(ppm, dtype=float)
    X = np.asarray(intensities, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    incl = [int(i) for i in included if 0 <= int(i) < X.shape[1]]
    if not incl or not noise_sigma or not field_mhz or not linewidth_ppm:
        return {}
    cols = build_columns(metabolites, field_mhz=field_mhz,
                         linewidth_ppm=linewidth_ppm, drift_ppm=drift_ppm)
    Y, free, owned = subtract_fixed(cols, ppm, X, fixed_per_proton or {},
                                    sigma=float(noise_sigma), included=incl,
                                    min_snr=min_snr)
    weight = exclusion_weight(ppm, excluded_regions)
    weight[owned] = 0.0
    trace = Y[:, incl].mean(axis=1)
    res = cascade(free, ppm, trace, sigma=float(noise_sigma),
                  linewidth_ppm=float(linewidth_ppm), weight=weight, min_snr=min_snr)
    # Measured on the spectrum as it is, beside the quantified analytes
    # that reach into these regions -- not on what the subtraction left.
    taken = {e.col.mid for e in res.active}
    reach = [(c, fixed_per_proton[c.mid]) for c in cols
             if c.mid in (fixed_per_proton or {})
             and any(np.intersect1d(e.idx, (c.curve(ppm, 0.0)[0]
                                            if c.curve(ppm, 0.0)[0] is not None
                                            else np.empty(0, int))).size
                     for e in res.active)]
    per, rel_err, capped = measure_jointly(
        res.active, reach, ppm, X, incl, sigma=float(noise_sigma), weight=weight)
    out = {}
    for e in res.active:
        amps = per.get(e.col.mid) or {}
        if not any(a > 0 for a in amps.values()):
            continue
        signals, tallest = [], None
        for (comp, nh), (idx, val) in zip(e.col.parts, e.col.part_curves(ppm, e.shift_ppm)):
            top = float(val.max())
            centre = float(ppm[idx[int(np.argmax(val))]])
            snr = e.amplitude * top / float(noise_sigma)
            signals.append((centre, comp.label, float(nh), float(snr)))
            if tallest is None or snr > tallest[0]:
                tallest = (snr, float(comp.lo + e.shift_ppm), float(comp.hi + e.shift_ppm))
        out[e.col.mid] = {
            'name': e.col.name,
            'per_proton': amps,
            'nh': e.col.nh_total,
            'apex_snr': float(e.apex_snr),
            'shift_ppm': float(e.shift_ppm),
            'at_cap': bool(e.amplitude >= 0.999 * e.cap),
            'capped_share': float(capped.get(e.col.mid, 0.0)),
            'free_share': float(e.free_share),
            'disagreement': float(rel_err.get(e.col.mid, float('nan'))),
            'neighbours': [c.name for c, _a in reach],
            'signals': signals,
            'window': (tallest[1], tallest[2]) if tallest else None,
            'step': int(e.step),
        }
    if diagnostics is not None:
        names = {c.mid: c.name for c in cols}
        diagnostics['cascade'] = {
            'columns': len(cols), 'free': len(free),
            'order': [(e.step, e.col.name, e.apex_snr, e.shift_ppm) for e in res.active],
            'contradicted': {names.get(m, m): (p, c) for m, (p, c) in res.contradicted.items()},
            'stopped_at': res.stopped_at,
        }
    return out
