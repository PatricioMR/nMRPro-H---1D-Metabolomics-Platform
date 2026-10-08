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
CASCADE_MAX_STEPS = 150
# Selection is by gain, so the late steps are the weak ones. The first
# CASCADE_FREE_STEPS take any analyte standing CASCADE_MIN_SNR; after them
# the bar is CASCADE_LATE_MIN_SNR. GISSMO bench, three seeds (2026-10-01):
# tube of 100, steps 1-30 were 98% real at two dots, 31-60 83%, 61-80 65%,
# 81-100 43%; at 50 compounds no step past 60 found a real one. Past 60,
# an apex of 25 sigma kept 22 real against 14 strangers (61%), a bar of 5
# kept 36 against 88. The cap of 60 steps this replaces held the tubes of
# 50 and 100 at 60 on every seed.
CASCADE_FREE_STEPS = 60
CASCADE_LATE_MIN_SNR = 25.0
SHIFT_STEPS = 7             # within +- SHIFT_REACH_LW linewidths
SHIFT_REACH_LW = 0.5
# Past the fine reach, a coarser grid out to this many linewidths: in a real
# extract every compound stands some way from where the reference lists it
# (NMR_Murilo: a median 4.5 linewidths, up to 14). A column that slid there
# is shown at one dot (quant_engine). On the GISSMO bench with each compound
# 20 mppm off its listing, tube of 18, three seeds: found 34 -> 45 of 54, real
# at two dots or more 9 -> 26; unshifted, found 51 -> 53, three dots 13/0.
CASCADE_WIDE_REACH_LW = 8.0
CASCADE_WIDE_STEP_LW = 0.5
CASCADE_APEX_HALF_LW = 0.25   # a wide-slid column must stand on an apex this wide
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
#
# Raised to 0.45 on the crowded tubes (2026-09-29, seed 0, with the joint
# fit refusing misplaced shapes): the analytes the cascade identified and
# then refused were in the tube at 30-48% disagreement (putrescine, UDP-
# glucuronate, IMP), the first stranger at 47% (homoserine). 18 analytes:
# nothing moves. 50: 29 -> 32 found, false positives 7 -> 7, p90 26 -> 31%.
# 100: 54 -> 58 found, false positives 5 -> 5, p90 67 -> 66%. At 0.50 the
# first stranger comes in. What enters is a cascade row, one dot.
MAX_SIGNAL_DISAGREEMENT = 0.45
OWNED_SHARE = 0.7           # of a point's height explained: it is theirs
# Where a quantified analyte's signals are subtracted: each may settle
# within this many linewidths of its listed place, at the amount measured,
# where it leaves the least of the mean trace -- the joint fit's own reach
# (quant_joint_fit.PART_SHIFT_REACH_LW). Subtracted at the listed place, a
# line standing a few mppm off leaves a remainder shaped like a derivative,
# tens of sigma tall beside a large signal, and a compound listing a line
# there takes it. GISSMO tubes at 600 MHz, three seeds (2026-09-30): 18
# analytes, rows of compounds not in the tube 32 -> 25, at two dots 10 ->
# 7; 100 analytes 21 -> 18 and 9 -> 5, real found 189 -> 194. Settling the
# width too changed nothing; settling the cascade's own entries signal by
# signal helped the strangers as much as the real ones (18: 7 -> 8 false
# at two dots).
SETTLE_REACH_LW = 1.0
SETTLE_STEPS = 17
SIGNAL_AGREEMENT_SNR = 5.0  # a signal quiet than this does not testify


def signal_protons(dp) -> float | None:
    """A listed signal's proton count, as the reference lists it -- its
    stoichiometry; None when it lists none.

    Not the protons measured under its profile on the reference spectrum
    (``profile_nh``): that share carries the reference's own artefacts into
    every study. Lactate's quartet holds 0.65 of its proton on the BMRB
    spectrum, attenuated by the water suppression, so its doublet's share
    came out 3.35 and every lactate was divided by it: 10% low, in any
    sample (2026-09-30, ADR-060)."""
    v = dp.get('protons')
    try:
        v = float(v) if v not in (None, '') else None
    except (TypeError, ValueError):
        v = None
    return v if v and v > 0 else None


class Column:
    """An analyte's shape over the spectrum, sparse, in area-per-proton
    units: one unit of amplitude is one unit of area under every proton."""

    def __init__(self, mid, name):
        self.mid, self.name = mid, name
        self.parts: list = []          # (component, nh)

    @property
    def nh_total(self) -> float:
        return float(sum(nh for _c, nh in self.parts))

    @staticmethod
    def _part_curve(comp, nh, ppm, shift, width):
        sel = np.flatnonzero((ppm >= comp.lo - abs(shift)) & (ppm <= comp.hi + abs(shift)))
        if sel.size < 3:
            return None
        x = ppm[sel]
        y = np.asarray(comp.model(x, comp.shift_ppm + shift,
                                  comp.width_scale * float(width)), dtype=float)
        top = float(y.max())
        if top <= 0:
            return None
        y = y / top
        order = np.argsort(x)
        area = float(np.trapezoid(y[order], x=x[order]))
        if area <= 0:
            return None
        return sel, y * (float(nh) / area)

    def part_curves(self, ppm, shift=0.0, width=1.0) -> list:
        """Each signal's sparse curve, area = its protons: [(idx, val)].
        ``width`` scales every line's width, as a sample broader or
        narrower than the study's measured linewidth needs."""
        out = [self._part_curve(comp, nh, ppm, shift, width)
               for comp, nh in self.parts]
        return [c for c in out if c is not None]

    def part_curve(self, k, ppm, shift=0.0, width=1.0):
        """The sparse curve of signal ``k`` alone, ``(idx, val)`` or
        ``(None, None)``."""
        comp, nh = self.parts[k]
        c = self._part_curve(comp, nh, ppm, shift, width)
        return c if c is not None else (None, None)

    def curve(self, ppm, shift=0.0, width=1.0):
        parts = self.part_curves(ppm, shift, width)
        if not parts:
            return None, None
        idx = np.concatenate([p[0] for p in parts])
        val = np.concatenate([p[1] for p in parts])
        # signals overlapping in ppm add up
        uniq, inv = np.unique(idx, return_inverse=True)
        out = np.zeros(uniq.size)
        np.add.at(out, inv, val)
        return uniq, out


def slide_grid(linewidth_ppm):
    """Where the cascade may slide a column: SHIFT_STEPS positions within
    SHIFT_REACH_LW linewidths either side of the study's drift, and, past
    them, a coarser grid out to CASCADE_WIDE_REACH_LW linewidths (a
    compound's own offset from the reference: ``0`` leaves it off)."""
    lw = float(linewidth_ppm)
    fine = np.linspace(-SHIFT_REACH_LW * lw, SHIFT_REACH_LW * lw, SHIFT_STEPS)
    if CASCADE_WIDE_REACH_LW <= SHIFT_REACH_LW:
        return fine
    k = np.arange(1, int(CASCADE_WIDE_REACH_LW / CASCADE_WIDE_STEP_LW) + 1)
    coarse = k * CASCADE_WIDE_STEP_LW * lw
    coarse = coarse[coarse > SHIFT_REACH_LW * lw * 1.2]
    return np.sort(np.concatenate([-coarse, fine, coarse]))


# Each signal of a column finds its own place (ADR-065). The cascade slides an
# analyte's column as one block; in an extract every chemical environment
# moves on its own -- each proton has its own titration curve (Tredwell et al.,
# Metabolomics 2016;12:152) and the matrix moves each differently (Takis et
# al., Nat Commun 2017;8:1662) -- and on the NMR_Murilo cell extract the
# cascade's signals stood a median 13 mppm (3.7 linewidths) off the peaks they
# were meant to explain. BATMAN gives every multiplet its own shift with a
# prior (Hao et al., Nat Protoc 2014;9:1416), ASICS distorts every peak
# locally after a global shift (Tardivel et al., Metabolomics 2017;13:109).
# Here, once the rigid cascade has chosen, each signal of an analyte it took
# in moves on its own within CASCADE_PART_REACH_LW linewidths to where the
# spectrum, less every other analyte taken in, best matches it -- only if
# that is CASCADE_PART_MIN_GAIN better than where the column put it -- and
# the amount is measured there. Only the amount: who is taken in, and the
# evidence its identity score reads, stay the rigid column's. Letting the
# signals move while choosing (GISSMO bench, 600 MHz, tubes 18/50, 3 seeds)
# found more at two dots with signals scattered 5 mppm, but lost real ones at
# two dots on the standard bench (18: 30 -> 22): free signals agree on an
# amount whether or not they are one molecule, and the score loses its best
# witness. Measured this way, identity is unchanged on both benches; the
# median error with scattered signals falls (tube 50: 26-31% -> 20-25%) and
# the extract's peaks explained rise from 41% to 48%. 1 linewidth: 2 and 4
# explained fewer on the extract.
CASCADE_PART_SHIFTS = True
CASCADE_PART_REACH_LW = 1.0
CASCADE_PART_STEPS = 9
CASCADE_PART_MIN_GAIN = 0.05


def part_offsets(col, ppm, trace, shift, linewidth_ppm, weight=None) -> list:
    """Each signal's own offset (ppm) from the column's ``shift``: the one in
    +-CASCADE_PART_REACH_LW linewidths where its curve best matches
    ``trace`` (normalised projection), kept at 0 unless it does
    CASCADE_PART_MIN_GAIN better. Nearest zero first, so a tie stays put."""
    ppm = np.asarray(ppm, dtype=float)
    t = np.asarray(trace, dtype=float)
    w = np.ones(ppm.size) if weight is None else np.asarray(weight, dtype=float)
    reach = CASCADE_PART_REACH_LW * float(linewidth_ppm)
    grid = sorted(np.linspace(-reach, reach, CASCADE_PART_STEPS), key=abs)
    out = []
    for comp, nh in col.parts:
        best_d, best_v = 0.0, None
        for d in grid:
            c = Column._part_curve(comp, nh, ppm, shift + d, 1.0)
            if c is None:
                continue
            idx, val = c
            v = val * w[idx]
            nn = float(np.dot(v, v))
            if nn <= 0:
                continue
            score = float(np.dot(t[idx] * w[idx], v)) / np.sqrt(nn)
            if best_v is None:
                best_d, best_v, zero = d, score, score
            elif score > best_v:
                best_d, best_v = d, score
        if best_v is None or not (best_d != 0.0
                                  and best_v > zero + CASCADE_PART_MIN_GAIN * abs(zero)):
            best_d = 0.0
        out.append(float(best_d))
    return out


def with_offsets(col, offsets):
    """A copy of ``col`` whose signals carry their own ``offsets`` (ppm), so
    every curve drawn from it later -- the per-sample fit, the subtraction,
    the report -- is the one the cascade chose."""
    import copy
    if not offsets or not any(offsets):
        return col
    new = Column(col.mid, col.name)
    for (comp, nh), d in zip(col.parts, offsets):
        c2 = copy.copy(comp)
        c2.shift_ppm = float(comp.shift_ppm) + float(d)
        new.parts.append((c2, nh))
    return new


def slide_step(linewidth_ppm) -> float:
    """One step of ``slide_grid``: how finely the cascade places a column."""
    reach = SHIFT_REACH_LW * float(linewidth_ppm or 0.0)
    return 2.0 * reach / max(SHIFT_STEPS - 1, 1)


def fit_curve_record(col, ppm, window, amplitudes, groups, *, label,
                     shift=0.0, shift_by_sample=None, width=1.0,
                     baseline=None, offsets=(), offset_limit=None,
                     pad_ppm=0.1, residual_of=None) -> dict | None:
    """What a fitted analyte's row carries so a report draws the number the
    table shows.

    The signal in focus is the column's signal that overlaps ``window`` (the
    elected lines), or the nearest one -- drawn whole, every line of it,
    at the shift and width the fit chose, one unit per proton. Beside it:
    each group's fitted amplitude by the table's own statistic (mean over
    the samples where it is positive) with how many samples that is, each
    group's mean of the baseline the fit stood on (None when it stood on
    zero), whether the fitted signal is the one the election found
    (``focus_matches_window``), and which of the column's signals the fit
    left at the edge of how far it may move them (``offsets`` against
    ``offset_limit``): a shape held at that edge is not where the trace's
    peak is, and its amount is the amount of a misplaced shape.

    ``residual_of(sel)``, when given, returns the spectrum on the points
    ``sel`` (points x samples) minus every other analyte at its fitted
    amount: what the cascade fitted this analyte to. Its mean per group is
    ``residual_by_group``; a figure that drew only the full trace would
    show an analyte standing under a neighbour a thousand times taller as
    a flat line.

    ``unit_by_group`` is every signal of the column inside the crop (the
    whole fit); ``focus_unit_by_group`` the focus signal alone, whose area
    (amplitude x ``focus_nh``) is the area the row's table shows.

    ``amplitudes`` are ``{sample: area per proton}`` in the spectrum's own
    units, before any normalisation, so the curve scales onto the trace.
    """
    ppm = np.asarray(ppm, dtype=float)
    shift_by_sample = dict(shift_by_sample or {})
    w_lo, w_hi = min(window), max(window)
    parts = []
    for comp, nh in col.parts:
        lo_p = comp.lo + comp.shift_ppm + shift
        hi_p = comp.hi + comp.shift_ppm + shift
        # a part's support is padded around its lines; the lines are the
        # signal, the pad is room for its tails
        lines = [float(v) + comp.shift_ppm + shift for v in (comp.lines or ())]
        a, b = (min(lines), max(lines)) if lines else (lo_p, hi_p)
        parts.append((a, b, float(nh)))
    if not parts:
        return None
    # a singlet's lines span nothing: meeting the window is what counts
    meets = [a <= w_hi and b >= w_lo for a, b, _ in parts]
    overlap = [min(b, w_hi) - max(a, w_lo) if m else -np.inf
               for (a, b, _), m in zip(parts, meets)]
    k = (int(np.argmax(overlap)) if any(meets) else
         int(np.argmin([abs((a + b) / 2 - (w_lo + w_hi) / 2)
                        for a, b, _ in parts])))
    f_lo, f_hi, f_nh = parts[k]
    lo, hi = min(f_lo, w_lo) - pad_ppm, max(f_hi, w_hi) + pad_ppm
    sel = np.flatnonzero((ppm >= lo) & (ppm <= hi))
    if sel.size < 2:
        return None
    amp_by_group, n_by_group, base_by_group, unit_by_group = {}, {}, {}, {}
    focus_unit_by_group, residual_by_group = {}, {}
    R = residual_of(sel) if residual_of is not None else None
    for g, members in (groups or {}).items():
        members = list(members)
        vals = [float(amplitudes[s]) for s in members
                if s in amplitudes and np.isfinite(amplitudes[s])]
        pos = [v for v in vals if v > 0]
        n_by_group[str(g)] = [len(pos), len(vals)]
        if pos:
            amp_by_group[str(g)] = float(np.mean(pos))
        sh = [shift_by_sample[s] for s in members if s in shift_by_sample]
        idx, val = col.curve(ppm, shift + (float(np.median(sh)) if sh else 0.0),
                             width)
        unit = np.zeros(sel.size)
        if idx is not None:
            j = np.searchsorted(sel, idx)
            ok = j < sel.size
            ok[ok] &= sel[j[ok]] == idx[ok]
            unit[j[ok]] = val[ok]
        unit_by_group[str(g)] = unit.tolist()
        f_idx, f_val = col.part_curve(
            k, ppm, shift + (float(np.median(sh)) if sh else 0.0), width)
        f_unit = np.zeros(sel.size)
        if f_idx is not None:
            j = np.searchsorted(sel, f_idx)
            ok = j < sel.size
            ok[ok] &= sel[j[ok]] == f_idx[ok]
            f_unit[j[ok]] = f_val[ok]
        focus_unit_by_group[str(g)] = f_unit.tolist()
        if R is not None and members:
            residual_by_group[str(g)] = R[:, members].mean(axis=1).tolist()
        if baseline is not None and members:
            B = np.asarray(baseline, dtype=float)
            base_by_group[str(g)] = B[sel][:, members].mean(axis=1).tolist()
    at_limit = []
    if offset_limit:
        at_limit = [float(c) for c, off in offsets or ()
                    if abs(float(off)) >= 0.999 * float(offset_limit)]
    return {'ppm': ppm[sel].tolist(), 'unit_by_group': unit_by_group,
            'focus_unit_by_group': focus_unit_by_group,
            'residual_by_group': residual_by_group or None,
            'amp_by_group': amp_by_group, 'n_by_group': n_by_group,
            'baseline_by_group': base_by_group or None, 'label': label,
            'focus': [float(f_lo), float(f_hi)], 'focus_nh': f_nh,
            'focus_matches_window': bool(any(meets)),
            'at_limit': at_limit, 'n_signals': len(parts)}


def build_columns(metabolites, *, field_mhz, linewidth_ppm, drift_ppm,
                  signals_of=None, drift_by_id=None) -> list:
    """One column per annotated metabolite from the signals the reference
    lists for it (``met['db_peaks']``, or what ``signals_of(met)`` returns).
    A signal with no shape (an unresolved 'm' the reference did not
    measure) or no proton count is left out; an analyte with no signal
    left has no column.

    ``drift_by_id`` gives a compound its own drift in place of ``drift_ppm``
    (``{mid: ppm}``): what pH and matrix did to it, known from the peak the
    election matched to one of its signals."""
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
            _drift = (drift_by_id or {}).get(mid, drift_ppm)
            ev = listed_signal_evidence(dp, centre_ppm=float(c0) + float(_drift),
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


def _settle(parts0, grid_parts, r, a_mean):
    """For each signal of one subtracted analyte, the candidate place that
    leaves the least of ``r`` (the mean residual with the analyte already
    taken out at ``parts0``). Returns the chosen ``(idx, val)`` per signal,
    nearest to the listed place among equals."""
    chosen = []
    for i, (i0, v0) in enumerate(parts0):
        best = (0.0, (i0, v0))
        for cand in grid_parts:
            i_s, v_s = cand[i]
            u = np.union1d(i0, i_s)
            d = np.zeros(u.size)
            d[np.searchsorted(u, i0)] += a_mean * v0
            d[np.searchsorted(u, i_s)] -= a_mean * v_s
            gain = float(np.sum(r[u] ** 2) - np.sum((r[u] + d) ** 2))
            if gain > best[0]:
                best = (gain, (i_s, v_s))
        chosen.append(best[1])
    return chosen


def subtract_fixed(cols, ppm, X, fixed, *, sigma=None, included=None,
                   min_snr=CASCADE_MIN_SNR, owned_share=OWNED_SHARE,
                   linewidth_ppm=None):
    """``X`` (points x samples) minus every fixed analyte at its per-sample
    amplitude ``fixed[mid][sample]``; the columns left free; and the points
    the fixed analytes own.

    With ``linewidth_ppm`` each fixed signal is subtracted where it settles
    (SETTLE_REACH_LW), largest analyte first; without it, at its listed
    place.

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
    taken = []
    for col in cols:
        amps = fixed.get(col.mid)
        if not amps:
            free.append(col)
            continue
        parts = col.part_curves(ppm, 0.0)
        if not parts:
            continue
        per = {}
        for s_i, a in amps.items():
            s_i = int(s_i)
            if a is None or not np.isfinite(a) or a <= 0 or not 0 <= s_i < Y.shape[1]:
                continue
            per[s_i] = float(a)
            for idx, val in parts:
                Y[idx, s_i] -= float(a) * val
        good = [a for s_i, a in per.items() if s_i in cols_i]
        taken.append((col, parts, per, float(np.mean(good)) if good else 0.0))
    if linewidth_ppm and cols_i:
        reach = SETTLE_REACH_LW * float(linewidth_ppm)
        grid = sorted(np.linspace(-reach, reach, SETTLE_STEPS), key=abs)
        r = Y[:, cols_i].mean(axis=1)
        for k in sorted(range(len(taken)), key=lambda k: -taken[k][3]):
            col, parts, per, a_mean = taken[k]
            if a_mean <= 0:
                continue
            grid_parts = [col.part_curves(ppm, sh) for sh in grid]
            if any(len(g) != len(parts) for g in grid_parts):
                continue
            chosen = _settle(parts, grid_parts, r, a_mean)
            for (i0, v0), (i_s, v_s) in zip(parts, chosen):
                if i_s is i0:
                    continue
                for s_i, a in per.items():
                    Y[i0, s_i] += a * v0
                    Y[i_s, s_i] -= a * v_s
                r[i0] += a_mean * v0
                r[i_s] -= a_mean * v_s
            taken[k] = (col, chosen, per, a_mean)
    for col, parts, per, a_mean in taken:
        if a_mean > 0:
            for idx, val in parts:
                cover[idx] += a_mean * val
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


def _is_apex(r, i, half) -> bool:
    """Whether point ``i`` is the top of a peak in ``r``: no point within
    ``half`` points either side stands higher. A column slid away from where
    the reference lists it must land on a peak of the residual; on the
    slope of a large one it would only be fitting that peak's tail."""
    lo, hi = max(i - half, 0), min(i + half + 1, r.size)
    return bool(r[i] >= np.max(r[lo:hi]))


def cascade(cols, ppm, trace, *, sigma, linewidth_ppm, weight=None,
            min_snr=CASCADE_MIN_SNR, max_steps=CASCADE_MAX_STEPS,
            cap_ratio=CAP_RATIO) -> CascadeResult:
    """Forward selection over ``cols`` on one trace (see the module)."""
    ppm = np.asarray(ppm, dtype=float)
    t_full = np.asarray(trace, dtype=float)
    weight = np.ones(ppm.size) if weight is None else np.asarray(weight, dtype=float)
    r = t_full * weight
    shifts = slide_grid(linewidth_ppm)
    step_ppm = abs(float(ppm[1] - ppm[0])) if ppm.size > 1 else 1.0
    apex_half = max(int(round(CASCADE_APEX_HALF_LW * float(linewidth_ppm) / step_ppm)), 1)
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
            variants.append((s, idx, val, nn, float(val.max()), cap, free_share,
                             int(idx[int(np.argmax(val))])))
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
            for s, idx, val, nn, top, cap, free, apex in cache[col.mid]:
                if (abs(s) > 1.2 * SHIFT_REACH_LW * float(linewidth_ppm)
                        and not _is_apex(r, apex, apex_half)):
                    continue
                proj = float(np.dot(r[idx], val)) / nn
                if proj <= 0:
                    continue
                if proj > cap_ratio * cap:
                    out.contradicted[col.mid] = (proj, cap)
                    continue
                a = min(proj, cap)
                if (step >= CASCADE_FREE_STEPS and sigma
                        and a * top / float(sigma) < CASCADE_LATE_MIN_SNR):
                    continue          # the late bar is for the candidate, not the run
                gain = 2.0 * a * proj * nn - a * a * nn   # residual removed at a
                if best is None or gain > best[0]:
                    best = (gain, col, s, idx, val, a, top, cap, free)
        if best is None:
            break
        gain, col, s, idx, val, a, top, cap, free = best
        apex_snr = a * top / float(sigma) if sigma else float('inf')
        bar = min_snr if step < CASCADE_FREE_STEPS else max(
            min_snr, CASCADE_LATE_MIN_SNR)
        if apex_snr < bar:
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
        # bvls, as quant_shared.resolve_window: a cap near zero sent the
        # iterative default into a loop it did not leave
        res = lsq_linear(A, t_full[union] * weight[union], bounds=(0, ub),
                         method='bvls')
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
        res = lsq_linear(A, t[union], bounds=(lo, hi), method='bvls')
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


# The cascade's identity score: a logistic model over cascade_features, fitted
# on the 1515 analytes the cascade took in over the GISSMO bench at 600 MHz --
# tubes of 18, 50 and 100 compounds, three seeds each, with every compound
# 0, 10 or 20 mppm off where the reference lists it (585 really in the tube).
# Tested by leaving out one tube, shift and seed at a time: AUC 0.92 with no
# offset, 0.87 at 10 mppm, 0.75 at 20 -- against 0.87 / 0.79 for the best
# single features (apex SNR, disagreement between signals) with no offset. It
# loses power with the offset because the features themselves do: a column
# laid where the compound is not has no good evidence to give. The bench is
# the one the weights came from and a real extract is not it: the score is
# shown on the row.
CASCADE_SCORE_MEAN = (3.74508, 4.02431, 1.09975, 0.92879, 0.25193, 0.42046, 1.53515, 3.31691, 0.09109, 0.66007, 2.02588, 1.43083)
CASCADE_SCORE_SCALE = (1.01854, 0.96012, 1.06514, 0.12676, 0.3885, 0.49363, 0.42069, 0.84987, 0.28774, 0.47369, 0.42437, 0.98445)
CASCADE_SCORE_COEF = (-1.33071, 2.36708, -0.61654, 0.29815, 0.02969, -0.51235, 0.05465, 0.19615, 0.058, -0.13228, -0.29081, -0.3863)
CASCADE_SCORE_INTERCEPT = -0.78118
# Below the first an analyte the cascade took in is a candidate, not an
# identification; from the second it is worth two dots. Swept offline on the
# rows of five bench regimes (found of 54, 150, 300; real / strangers at two
# dots or more, tube 100 without offset): no minimum 223, 201/21 at 0.5 and
# 175/8 at 0.7; a minimum of 0.15 found 220 with 58 strangers in all, 0.5 only
# 202 with 24. Two dots at 0.7: no stranger at two dots or more in the tubes
# of 18 and 50.
# 0.6 instead of 0.7 (ADR-061 session of 2026-10-03; the scores run to 0.70, so
# 0.7 raised nothing): tubes of 18 and 50, 600 MHz, three seeds, real / strangers
# at two dots, 18: 22/0 -> 30/0, 50: 67/8 -> 73/13; with a common drift of 13
# and 4 of scatter, 18: 19/5 -> 27/6, 50: 44/15 -> 58/18. Three dots untouched
# (cascade rows are never three) and the number found does not change.
CASCADE_SCORE_MIN = 0.15
CASCADE_SCORE_TWO = 0.6
CASCADE_BY_SCORE = True

CASCADE_FEATURES = ('log apex snr', 'log matched snr', 'disagreement',
                    'free share', 'capped share', 'at cap', 'log signals',
                    'log step', 'placed', 'wide', 'log nh', 'log shift')


def cascade_features(res, linewidth_ppm) -> list:
    """What the cascade knows about an analyte it took in, as the features
    of ``cascade_score``: how tall its tallest signal and all its signals
    together stand over the noise, how well its own signals agree, how much
    of its listing the residual left it, whether a peak's amplitude held it
    down, how far it slid, how late it was taken."""
    sig = np.array([v[3] for v in (res.get('signals') or ())], dtype=float)
    dis = res.get('disagreement')
    dis = 3.0 if dis is None or not np.isfinite(dis) else min(float(dis), 3.0)
    shift = abs(float(res.get('shift_ppm') or 0.0))
    lw = float(linewidth_ppm)
    return [np.log1p(float(res.get('apex_snr') or 0.0)),
            np.log1p(float(np.sqrt(np.sum(sig ** 2))) if sig.size else 0.0),
            dis, float(res.get('free_share') or 0.0),
            float(res.get('capped_share') or 0.0),
            float(bool(res.get('at_cap'))), np.log1p(sig.size),
            np.log1p(float(res.get('step') or 0)),
            float(shift <= 1.001 * slide_step(lw)),
            float(shift > 1.001 * SHIFT_REACH_LW * lw),
            np.log1p(float(res.get('nh') or 0.0)),
            np.log1p(1000.0 * shift)]


# The signals of one analyte move together across samples: they are one
# molecule, so they carry one concentration per sample (STOCSY: Cloarec et al.,
# Anal Chem 2005;77:1282; SPA-STOCSY, Bioinformatics 2023;39:btad593). The
# cascade fits ONE amplitude to all of an analyte's signals, so it cannot see
# whether the spectrum agrees across samples; this reads that agreement from
# the spectrum (what the other analytes leave) at each signal, sample by
# sample. Evidence that raises, never one that punishes: an analyte whose
# concentration does not vary across the samples has nothing to correlate and
# is indistinguishable, not wrong. It uses no group label.
COVAR_MIN_SNR = 8.0          # a signal is read when its fit stands this far over the noise
COVAR_MAX_SIGNALS = 3        # the tallest ones
COVAR_MIN_R = 0.8            # every pair of the read signals correlates at least this much
COVAR_MIN_VARIATION = 3.0    # and each varies across samples this many noises: the r is not noise
COVAR_MIN_SAMPLES = 8
COVAR_HALF_WIDTH_LW = 0.6    # where a signal's height is read around its centre


def signal_covariation(heights, *, sigma) -> dict | None:
    """How well the signals of one analyte agree across samples.

    ``heights`` is one array per signal, its height in every included sample.
    Returns ``{'r_min', 'r_median', 'variation', 'n_signals', 'n_samples'}``
    (``variation``: the smallest standard deviation across samples among the
    signals, in noise units), or None with fewer than two signals or
    ``COVAR_MIN_SAMPLES`` samples."""
    H = [np.asarray(h, dtype=float) for h in heights]
    if len(H) < 2 or not sigma or float(sigma) <= 0:
        return None
    n = min(len(h) for h in H)
    if n < COVAR_MIN_SAMPLES:
        return None
    H = [h[:n] for h in H]
    if any(not np.all(np.isfinite(h)) for h in H):
        return None
    sd = [float(np.std(h, ddof=1)) for h in H]
    if min(sd) <= 0:
        return {'r_min': float('nan'), 'r_median': float('nan'),
                'variation': 0.0, 'n_signals': len(H), 'n_samples': n}
    rs = [float(np.corrcoef(H[i], H[j])[0, 1])
          for i in range(len(H)) for j in range(i + 1, len(H))]
    return {'r_min': float(min(rs)), 'r_median': float(np.median(rs)),
            'variation': float(min(sd) / float(sigma)),
            'n_signals': len(H), 'n_samples': n}


def covariation_confirms(cov) -> bool:
    """Whether the signals of an analyte move together across the samples
    firmly enough to count as evidence of one molecule."""
    return bool(cov and np.isfinite(cov.get('r_min', float('nan')))
                and cov['r_min'] >= COVAR_MIN_R
                and cov['variation'] >= COVAR_MIN_VARIATION)


def cascade_score(res, linewidth_ppm) -> float:
    """The probability the cascade's model gives that an analyte it took in
    is really in the sample (see ``CASCADE_SCORE_COEF``)."""
    f = np.asarray(cascade_features(res, linewidth_ppm), dtype=float)
    z = (f - np.asarray(CASCADE_SCORE_MEAN)) / np.asarray(CASCADE_SCORE_SCALE)
    t = float(np.dot(z, np.asarray(CASCADE_SCORE_COEF)) + CASCADE_SCORE_INTERCEPT)
    return float(1.0 / (1.0 + np.exp(-t)))


def others_removed(Y, ppm, sel, found, cols, mid):
    """``Y`` (the spectrum minus the quantified analytes) on the points
    ``sel`` (sorted indices), with every other analyte the cascade took in
    also subtracted at its fitted amount and place: what ``mid`` was
    fitted to, sample by sample (points x samples)."""
    R = np.asarray(Y, dtype=float)[sel].copy()
    for other, res in found.items():
        col = cols.get(other)
        if other == mid or col is None:
            continue
        for idx, val in col.part_curves(ppm, float(res.get('shift_ppm') or 0.0)):
            j = np.searchsorted(sel, idx)
            ok = j < sel.size
            ok[ok] &= sel[j[ok]] == idx[ok]
            for s_i, a in (res.get('per_proton') or {}).items():
                if 0 <= int(s_i) < R.shape[1]:
                    R[j[ok], int(s_i)] -= float(a) * val[ok]
    return R


def cascade_missed(metabolites, ppm, intensities, included, *, fixed_per_proton,
                   noise_sigma, linewidth_ppm, drift_ppm, field_mhz,
                   excluded_regions=(), min_snr=CASCADE_MIN_SNR,
                   diagnostics=None, residual_out=None) -> dict:
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
    units, the same the election's windows give. ``residual_out``, when a
    dict, gets ``'Y'``: the spectrum with every quantified analyte
    subtracted, which ``others_removed`` turns into the residual one
    analyte was fitted to. ``diagnostics``, when a
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
                                    min_snr=min_snr,
                                    linewidth_ppm=float(linewidth_ppm))
    weight = exclusion_weight(ppm, excluded_regions)
    weight[owned] = 0.0
    if residual_out is not None:
        residual_out['Y'] = Y
    trace = Y[:, incl].mean(axis=1)
    res = cascade(free, ppm, trace, sigma=float(noise_sigma),
                  linewidth_ppm=float(linewidth_ppm), weight=weight,
                  min_snr=min_snr)
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
    if CASCADE_PART_SHIFTS:
        # who was taken in, and what testifies to its identity, stay the
        # rigid column's; the amount is measured with each signal where the
        # spectrum, less every other analyte taken in, puts it
        fitted = np.zeros(ppm.size)
        for e in res.active:
            fitted[e.idx] += e.amplitude * e.val
        for e in res.active:
            if len(e.col.parts) < 2:
                continue
            own = np.zeros(ppm.size)
            own[e.idx] = e.amplitude * e.val
            placed = with_offsets(e.col, part_offsets(
                e.col, ppm, trace * weight - fitted + own, e.shift_ppm,
                float(linewidth_ppm), weight))
            if placed is not e.col:
                e.col = placed
        per, _rel, _cap = measure_jointly(
            res.active, reach, ppm, X, incl, sigma=float(noise_sigma), weight=weight)
    if residual_out is not None:
        # what the run accounts for on the mean of the included samples: the
        # quantified analytes where they settled, and what the cascade took in
        model = (X - Y)[:, incl].mean(axis=1)
        for e in res.active:
            amps = per.get(e.col.mid) or {}
            a = float(np.mean([float(amps.get(s, 0.0)) for s in incl]))
            if a > 0:
                for idx, val in e.col.part_curves(ppm, e.shift_ppm):
                    model[idx] += a * val
        residual_out['model'] = model
        residual_out['trace'] = X[:, incl].mean(axis=1)
    out = {}
    listed = {c.mid: c for c in cols}
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
            # the column as the cascade placed it, every signal's own offset
            # included: what the per-sample fit, the report and the review
            # must draw
            'column': e.col,
            # how far each signal stands from where the column put it (ppm)
            'part_offsets': [float(c.shift_ppm) - float(c0.shift_ppm) for (c, _n), (c0, _n0)
                             in zip(e.col.parts, listed[e.col.mid].parts)],
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
