# processing/quant_engine.py
"""Elect, integrate, concentrate — the whole of quantification, in that order.

Three decisions, each made once, by a function that can be read on its own:

  elect        which signal quantifies which metabolite, and why the others
               got nothing (processing.quantification)
  integrate    the area of that signal's multiplet in every sample, foot to
               foot (processing.quant_integration)
  concentrate  that area per proton, against the standard, in mM
               (processing.quant_concentration)

Deliberately absent: JG-Deconv. The old engine folded its fractional areas,
its NNLS rejections and its per-line scaling into the same expression that
computed the concentration, so a number could not be read without knowing
which of them had fired. This engine measures what is in the window and says
so; sharing an overlapped window between analytes is a separate question and
gets a separate answer, later.

Also absent: the four modes that were alternatives to multiplet integration,
and the three paths that put a metabolite back after it lost. A metabolite
that lost has no number here, and the reason it has none is recorded.
"""
from __future__ import annotations

import numpy as np

from processing.quant_concentration import (
    apply_pqn, concentrations, signal_nh, standard_per_proton)
from processing.quant_deconvolution import (
    MIN_CLAIM_FRACTION, claimants_in_window, fraction_in_window,
    solve_windows, subtract_neighbours)
from processing.quant_detection_limit import (
    BELOW_LOD, BELOW_LOQ, EXCLUDED_REGION, K_LOD, STRONG_ABOVE_LOQ,
    UNRESOLVED, censor, censor_text, detection_limits, drift_offsets,
    integration_noise, local_noise_sigma, predicted_window, strength_band,
    window_is_excluded)
from processing.quant_integration import (
    MIN_TRUSTED_CAPTURE,
    integrate_across_samples, integrate_sample, integrate_signal,
    lorentzian_captured_fraction, multiplet_window, pascal_weights,
    pattern_lines, refine_apex_ppm)
from processing.quant_shape import measure_shape
from processing.quant_shared import (
    REMAINDER_NOTE, RESIDUAL_ACCEPT, RESIDUAL_EXPLAINED, fingerprint_holds,
    resolve_free_apices, resolve_listed_intruders, resolve_residuals,
    resolve_shared_windows)
from processing.quant_snr import (
    best_group, group_evidence, noise_level, representative_snr, window_snr)
from processing.quantification import (
    AMOUNT_CONTRADICTED, LOST_COMPETITION, NO_MEASURABLE_WINDOW,
    PARTIAL_MULTIPLET, REJECTION_TEXT, SHAPE_UNIDENTIFIED, SHAPES_OFF,
    SINGLET_UNSUPPORTED, WINDOW_TAKEN, elect, is_competitor)
from processing.annotation_scoring import (
    PATTERN_MULT, is_singlet_like, normalize_mult)

# The status dots the results table shows. Confidence is about how much of the
# expected evidence this signal actually carried, not about how large the
# number is: a big concentration measured on one unverified line is not a
# confident one.
STATUS_STRONG = '●●●'
STATUS_FAIR = '●●○'
STATUS_WEAK = '●○○'
STATUS_NONE = 'n.q.'

# Every censored row reads as weak confidence in the Conf column, and its
# reason says which kind of weakness. The column answers one question -- how
# much is this worth -- and putting a second vocabulary in it ("< LOQ",
# "overlap?") made the reader translate between two scales to compare two rows.
# The bound itself still travels, in the concentration column and the reason.
STATUS_CENSORED = {
    BELOW_LOD: STATUS_WEAK,
    BELOW_LOQ: STATUS_WEAK,
    UNRESOLVED: STATUS_WEAK,
    EXCLUDED_REGION: STATUS_WEAK,
}


def _cv_percent(values):
    v = [float(x) for x in (values or ()) if np.isfinite(x) and x > 0]
    if len(v) < 2:
        return float('nan')
    m = float(np.mean(v))
    if m <= 0:
        return float('nan')
    return float(np.std(v, ddof=1) / m * 100.0)


_DOTS = {3: STATUS_STRONG, 2: STATUS_FAIR, 1: STATUS_WEAK}


def _structural_evidence(signal) -> int:
    """How much of the expected evidence this signal's SHAPE carried.

    Three: the multiplet was complete and its coupling agreed. Two: complete,
    coupling off but inside the gate. One: everything else that still produced
    a number — a singlet, which has no coupling to agree with, among them.
    """
    ev = signal.evidence or {}
    n_f, n_e = ev.get('n_found'), ev.get('n_expected')
    complete = (n_f is not None and n_e is not None and n_f >= n_e)
    dj = ev.get('delta_j_hz')
    if complete and dj is not None and abs(float(dj)) <= 0.20:
        return 3
    if complete and dj is not None:
        return 2
    return 1


def _confidence(signal, strength=None, shared=False) -> tuple[int, str]:
    """What the result is worth: the worse of its shape and its size.

    Two independent things can spoil a number and neither can make up for the
    other. A textbook multiplet measured at the quantification limit is a
    confident identity and an imprecise quantity; a towering singlet with a
    rival on its peak is the reverse. Averaging them would let either hide the
    other, so the lower one stands.

    ``shared`` says the window is not this analyte's alone -- either the area
    was resolved out of one held with others, or it is a window that could not
    be split and so still holds somebody else's signal. Two dots either way,
    whatever the multiplet's shape: in the first case the number rests on
    another analyte's concentration being right, and in the second it is an
    upper bound wearing a concentration's clothes.

    ``strength`` is the band from ``strength_band`` — 3 when the area clears
    the quantification limit by the agreed factor, 2 otherwise. It never
    reaches 1: an analyte whose peak was identified and whose contest it won is
    not made doubtful by being scarce, and being under the limit says the
    quantity is small, not that the assignment was wrong.
    """
    level = _structural_evidence(signal)
    if strength is not None:
        level = min(level, int(strength))
    if shared:
        level = min(level, SHARED_MAX_CONFIDENCE)
    return level, _DOTS[level]


# A share resolved by subtraction is worth less than an area measured on a
# window nobody else is in, however clean the multiplet's shape. It rests on
# another analyte's concentration, and on the assumption that the claimants
# listed are all of them. Two dots, never three.
SHARED_MAX_CONFIDENCE = 2

# Below this, the window holds less of an analyte's signal than it leaves
# outside, and dividing the share by it multiplies the error faster than it
# corrects the area. The claimant still counts in the balance -- its
# contribution to the window is real and gets subtracted -- it simply does not
# get a concentration out of a window that is mostly not looking at it.
MIN_OWN_SHARE = 0.5


def _signal_windows(candidates, peaks_by_rank, ppm, intensities, *,
                    included, local_baseline, align, blank_indices,
                    linewidth_ppm, noise_sigma=None, drift_ppm=0.0,
                    excluded_regions=()):
    """One window per candidate signal, and everybody whose lines reach it.

    Built over the candidates and not over the election's winners, because a
    winner has exactly one window here and a claimant of that window could then
    never be anchored anywhere -- every contest would end in two unknowns. The
    metabolite's other matched signals, clean and merely unused by the
    election, are what measure it.

    One window per signal, and not one per connected group of overlapping
    signals. Overlap is transitive and the groups are not: with A on lines
    1-2, B on 2-3 and C on 3-4, the group is one window with three claimants
    and two unknowns, although A and C do not touch. As three windows it is
    three equations, each with the claimants that actually reach it. On the
    reference study the grouped form left 49 of 70 contested windows with more
    unknowns than equations.

    Each claimant is weighed by the fraction of ITS OWN lines the window holds.
    The balance is

        A(W) = sum over claimants of  concentration * protons * fraction

    and the fraction is a term of it, never a test the window has to pass. An
    earlier version refused any window holding less than 75% of its lines --
    the threshold the capture correction uses to decide whether a correction is
    worth applying -- and that threw away 69 of 113 windows, among them the
    uncontested ones that would have anchored the rest. Foot-to-foot windows in
    a crowded region are narrow by construction: the feet are the valleys, and
    the valleys are close when the neighbours are.

    A window containing a detected peak that no candidate claims is refused.
    That peak is a term the balance does not have, and the analyte credited
    with the remainder would be credited with it.

    Where a metabolite's signal was predicted and NOT detected, that is a
    measurement too, and the one the balance was missing. On the reference
    study half the columns were exact multiples of another: an analyte with
    one signal, in one window, shared with an analyte in the same position.
    Nothing in those windows tells them apart. But if the first should also
    show a doublet at 7.5 ppm and there is nothing there, its concentration
    is bounded by the noise in that region and the second's is not -- and
    that is a row in which their columns differ.

    So an undetected signal becomes a window at its predicted position, its
    area whatever is there, and it enters the system as any other equation.
    Two conditions, because a database shift is not where a real sample puts
    the line. The metabolite must have at least one DETECTED signal in the
    study, so the analyte is known to be present and this is a question of
    how much rather than whether. And the window slides over the drift the
    study measured, keeping the position with the MOST area -- so if the
    line moved, the bound follows it, and it can only ever be too loose. A
    bound that is too loose costs nothing; one that is too tight puts an
    analyte at zero that is there.

    A detected peak nobody claims inside the slid window refuses it, as it
    does any window: that peak may be this very signal, drifted further than
    allowed for.
    """
    sigs, by_key, undetected = [], {}, []
    counts = {'candidates': 0, 'no_lines': 0, 'not_competitor': 0,
              'no_protons': 0, 'components': 0, 'contested': 0,
              'no_window': 0, 'foreign_peak': 0, 'thin_share': 0,
              'largest': 0, 'largest_lines': 0,
              'bounds': 0, 'bounds_excluded': 0, 'bounds_foreign': 0,
              'bounds_no_anchor': 0}
    for c in candidates or ():
        counts['candidates'] += 1
        if (c.evidence or {}).get('shape_fit'):
            # a shape holds no peak and shares no window; its region is
            # measured on its own terms (processing.quant_shape)
            counts['shape'] = counts.get('shape', 0) + 1
            continue
        ranks = sorted(int(r) for r in (c.line_ranks or ()))
        if not ranks and c.peak_rank is not None:
            ranks = [int(c.peak_rank)]
        if not ranks:
            counts['no_lines'] += 1
            pred = [float(v) for v in
                    ((c.evidence or {}).get('predicted_ppms') or ())
                    if v is not None]
            nh = signal_nh((c.evidence or {}).get('protons'))
            if pred and nh and np.isfinite(nh) and nh > 0:
                undetected.append({'met_id': str(c.met_id), 'nh': float(nh),
                                   'ppms': pred,
                                   'weights': pascal_weights(len(pred))})
            continue
        if not is_competitor(c.evidence):
            counts['not_competitor'] += 1
            continue
        nh = signal_nh((c.evidence or {}).get('protons'))
        if not nh or not np.isfinite(nh) or nh <= 0:
            counts['no_protons'] += 1
            continue
        key = (str(c.met_id), frozenset(ranks))
        if key in by_key:
            continue
        by_key[key] = c
        own = [peaks_by_rank[r] for r in ranks if r in peaks_by_rank]
        if len(own) != len(ranks):
            counts['no_window'] += 1
            continue
        sigs.append({'met_id': str(c.met_id), 'nh': float(nh),
                     'ranks': tuple(ranks), 'peaks': own,
                     'ppms': [float(pk.get('ppm', 0.0)) for pk in own],
                     'weights': pascal_weights(len(own))})

    hwhm = float(linewidth_ppm) / 2.0
    # Where the lines really are, to better than the grid. Every fraction below
    # is evaluated at the window's edges, and the trapezium the local baseline
    # removes is read off the lineshape there -- half a grid point of error in
    # an apex moves it more than anything else in the model.
    _arr = np.asarray(intensities, dtype=float)
    if _arr.ndim == 1:
        _arr = _arr.reshape(-1, 1)
    _cols = [c for c in (included or range(_arr.shape[1]))
             if 0 <= c < _arr.shape[1]]
    _ref = _arr[:, _cols].mean(axis=1) if _cols else _arr.mean(axis=1)
    _apex = {}
    for sig in sigs:
        sig['ppms'] = [_apex.setdefault(r, refine_apex_ppm(ppm, _ref, pk_ppm))
                       for r, pk_ppm in zip(sig['ranks'], sig['ppms'])]
    claimed_ranks = {r for sig in sigs for r in sig['ranks']}
    windows, reasons = [], {}
    for sig in sigs:
        win = multiplet_window(sig['peaks'])
        if win is None:
            counts['no_window'] += 1
            continue
        lo, hi = win
        # The bounds the integration will actually use. A window whose ends
        # fall between grid points is integrated between the points inside it,
        # and the local baseline's line is drawn through THOSE -- so a model
        # evaluated at the requested ends is describing a window half a point
        # wider at each side. Where the trapezium is large that half point is
        # the difference between agreeing with the integral to 0.06% and being
        # 7% under it.
        _in = np.asarray(ppm, dtype=float)
        _in = _in[(_in >= lo) & (_in <= hi)]
        if _in.size >= 2:
            lo, hi = float(_in.min()), float(_in.max())
        key = '%s@%s' % (sig['met_id'],
                         '_'.join(str(r) for r in sig['ranks']))
        counts['components'] += 1

        unclaimed = [r for r, pk in peaks_by_rank.items()
                     if int(r) not in claimed_ranks
                     and lo <= float(pk.get('ppm', -1e9)) <= hi]
        if unclaimed:
            counts['foreign_peak'] += 1
            reasons[key] = {
                'claimants': [(sig['met_id'], sig['nh'])],
                'reason': ('%d detected peak(s) inside the window are matched '
                           'to no analyte; the balance would be missing a term'
                           % len(unclaimed))}
            continue

        # Only the signals near enough to put anything here. The fraction says
        # how much, and below MIN_CLAIM_FRACTION it says almost nothing.
        contributions, fractions, geometry, own_ranks = [], {}, {}, {}
        for other in sigs:
            if min(other['ppms']) > hi + 40.0 * hwhm:
                continue
            if max(other['ppms']) < lo - 40.0 * hwhm:
                continue
            f = fraction_in_window(other['ppms'], other['weights'], lo, hi,
                                   hwhm, local_baseline=local_baseline)
            if not f or f <= 0:
                continue
            contributions.append((other['met_id'], other['nh'], f))
            mid = other['met_id']
            # Two fractions, and they answer different questions. The one with
            # the baseline in it is the term of the balance -- what integrating
            # this window actually returns for this analyte. The bare
            # geometric one is whether the window is looking at the analyte at
            # all, which is what the floor on reporting asks: a straight line
            # drawn through a signal's own tails is a property of how it was
            # measured, not evidence that the signal is elsewhere.
            g = (fraction_in_window(other['ppms'], other['weights'], lo, hi,
                                    hwhm)
                 if local_baseline else f)
            if f > fractions.get(mid, 0.0):
                fractions[mid] = float(f)
                geometry[mid] = float(g) if g and g > 0 else float(f)
                own_ranks[mid] = other['ranks']
        claimants = claimants_in_window(contributions)
        if not claimants:
            # Not even the signal the window was built on reaches the floor:
            # the window is so much narrower than the line that what the local
            # baseline removes is most of what it holds. Said, not skipped --
            # a window that vanishes without a word is how a step comes to look
            # like it is not there.
            counts['thin_share'] += 1
            reasons[key] = {
                'claimants': [(sig['met_id'], sig['nh'])],
                'reason': ('the window is too narrow for the line: after the '
                           'baseline it holds under %.0f%% of any claimant'
                           % (100.0 * MIN_CLAIM_FRACTION))}
            continue

        n_cl = len(claimants)
        if n_cl > 1:
            counts['contested'] += 1
        if n_cl > counts['largest']:
            counts['largest'] = n_cl
            counts['largest_lines'] = len(sig['ranks'])

        out = integrate_signal(sig['peaks'], ppm, intensities,
                               included=included,
                               local_baseline=local_baseline, align=align,
                               blank_indices=blank_indices)
        if not out or not out['areas']:
            counts['no_window'] += 1
            continue
        # One over the noise on this window's area, so a wide noisy window
        # does not outvote a narrow clean one in the joint solution.
        _p = np.asarray(ppm, dtype=float)
        n_pts = int(((_p >= lo) & (_p <= hi)).sum())
        dppm = float(abs(_p[1] - _p[0])) if _p.size > 1 else 0.0
        noise_area = (integration_noise(noise_sigma, dppm, n_pts)
                      if noise_sigma else None)
        windows.append({
            'key': key, 'ranks': sig['ranks'], 'owner': sig['met_id'],
            'weight': (1.0 / float(noise_area)
                       if noise_area and np.isfinite(noise_area)
                       and noise_area > 0 else 1.0),
            'claimants': claimants, 'areas': dict(out['areas']),
            'window': (lo, hi), 'fractions': fractions, 'geometry': geometry,
            'lines': own_ranks,
            'protons': {m: (nh_eff / fractions[m]) for m, nh_eff in claimants
                        if fractions.get(m)},
            'peaks': sig['peaks'],
            'shifts': out.get('shifts'),
            'blank_area': out.get('blank_area', float('nan'))})
    # ── The signals that should be there and are not ─────────────────────────
    present = {sig['met_id'] for sig in sigs}
    _in_all = np.asarray(ppm, dtype=float)
    for u in undetected:
        if u['met_id'] not in present:
            counts['bounds_no_anchor'] += 1
            continue
        win = predicted_window(u['ppms'], linewidth_ppm=linewidth_ppm)
        if win is None:
            continue
        lo0, hi0 = win
        # Slide over the drift and keep the position holding the most: a bound
        # may be loose, never tight.
        best, best_off, best_med = None, 0.0, float('-inf')
        for off in drift_offsets(drift_ppm, linewidth_ppm):
            lo, hi = lo0 + off, hi0 + off
            if window_is_excluded(lo, hi, excluded_regions):
                continue
            areas = integrate_across_samples(
                ppm, intensities, lo, hi, included=included,
                local_baseline=local_baseline)
            if not areas:
                continue
            med = float(np.median(list(areas.values())))
            if med > best_med:
                best, best_off, best_med = areas, off, med
        if best is None:
            counts['bounds_excluded'] += 1
            continue
        lo, hi = lo0 + best_off, hi0 + best_off
        _in = _in_all[(_in_all >= lo) & (_in_all <= hi)]
        if _in.size >= 2:
            lo, hi = float(_in.min()), float(_in.max())
        unclaimed = [r for r, pk in peaks_by_rank.items()
                     if int(r) not in claimed_ranks
                     and lo <= float(pk.get('ppm', -1e9)) <= hi]
        if unclaimed:
            counts['bounds_foreign'] += 1
            continue
        ppms = [float(v) + best_off for v in u['ppms']]
        contributions = [(u['met_id'], u['nh'],
                          fraction_in_window(ppms, u['weights'], lo, hi, hwhm,
                                             local_baseline=local_baseline))]
        for other in sigs:
            if min(other['ppms']) > hi + 40.0 * hwhm:
                continue
            if max(other['ppms']) < lo - 40.0 * hwhm:
                continue
            f = fraction_in_window(other['ppms'], other['weights'], lo, hi,
                                   hwhm, local_baseline=local_baseline)
            if f and f > 0:
                contributions.append((other['met_id'], other['nh'], f))
        claimants = claimants_in_window(contributions)
        if not any(m == u['met_id'] for m, _ in claimants):
            continue
        n_pts = int(_in.size)
        dppm = (float(abs(_in_all[1] - _in_all[0])) if _in_all.size > 1
                else 0.0)
        noise_area = (integration_noise(noise_sigma, dppm, n_pts)
                      if noise_sigma else None)
        counts['bounds'] += 1
        windows.append({
            'key': '%s@predicted_%.3f' % (u['met_id'], float(np.mean(ppms))),
            'ranks': (), 'owner': None, 'bound': True,
            'weight': (1.0 / float(noise_area)
                       if noise_area and np.isfinite(noise_area)
                       and noise_area > 0 else 1.0),
            'claimants': claimants, 'areas': dict(best),
            'window': (lo, hi), 'fractions': {}, 'geometry': {},
            'lines': {}, 'protons': {}, 'peaks': [], 'shifts': None,
            'blank_area': float('nan')})
    return windows, by_key, reasons, counts


def _window_limits(ppm, intensities, lo, hi, *, included, noise_sigma,
                   linewidth_ppm, segments=None):
    """The SNR and the detection limits of one window, from one noise.

    One noise for the whole of quantification. The SNR column used to
    reconstruct its own from the peak's recorded ratio, which disagreed with
    the limits beside it by a factor of two and, over a baseline offset,
    produced an SNR of 3149 for an analyte at 0.008 mM.
    """
    sigma_here = local_noise_sigma(ppm, intensities, lo, hi,
                                   linewidth_ppm=linewidth_ppm,
                                   global_sigma=noise_sigma) or noise_sigma
    snr = window_snr(ppm, intensities, lo, hi, sigma_here, included=included)
    _p = np.asarray(ppm, dtype=float)
    # The points the integration summed: the segments', where the signal was
    # measured on pieces -- the noise on an area grows with the points in it.
    _in = np.zeros(_p.shape, dtype=bool)
    for a, b in (segments or [(lo, hi)]):
        _in |= (_p >= min(a, b)) & (_p <= max(a, b))
    n_pts = int(_in.sum())
    dppm = float(abs(_p[1] - _p[0])) if _p.size > 1 else 0.0
    lod_a, loq_a = detection_limits(sigma_here, dppm, n_pts)
    return snr, lod_a, loq_a


def _resolve_shared(candidates, peaks_by_rank, ppm, intensities, *,
                    included, local_baseline, align, blank_indices,
                    linewidth_ppm, noise_sigma, drift_ppm=0.0,
                    excluded_regions=(), diagnostics=None):
    """What each analyte's own share of the shared peaks is.

    Every window is one equation and they are solved together, because they are
    not independent problems: a metabolite appears in as many windows as it has
    signals and each holds a different fraction of it. One at a time, a window
    with two claimants and no independent measurement of either has two
    unknowns and yields nothing -- 215 of the reference study's 242 contested
    windows were of that kind. Together they are 291 equations in about 130
    unknowns.

    Returns ``({met_id: info}, reasons)`` with the same shape the integration
    loop produces, so a share and an uncontested area travel through
    concentration, confidence and the row builder by one path.
    """
    windows, by_key, reasons, counts = _signal_windows(
        candidates, peaks_by_rank, ppm, intensities, included=included,
        local_baseline=local_baseline, align=align,
        blank_indices=blank_indices, linewidth_ppm=linewidth_ppm,
        noise_sigma=noise_sigma, drift_ppm=drift_ppm,
        excluded_regions=excluded_regions)
    per_proton, why, solver_stats = solve_windows(windows)

    # Which window speaks for each metabolite. The one built on its own signal
    # measures it; failing that, the one holding most of it. It decides the
    # region, the limits and the proton count the row reports -- the
    # concentration itself came from all of them at once.
    home: dict = {}
    for w in windows:
        if w.get('bound'):
            continue
        for mid, _nh_eff in w['claimants']:
            better = (mid == w.get('owner'),
                      float((w.get('geometry') or {}).get(mid, 0.0)))
            if better > home.get(mid, ((False, -1.0), None))[0]:
                home[mid] = (better, w)

    out = {}
    for mid, by_sample in per_proton.items():
        entry = home.get(mid)
        if entry is None:
            continue
        w = entry[1]
        cand = by_key.get((mid, frozenset(w['lines'].get(mid) or ())))
        if cand is None:
            cand = next((c for (m, r), c in by_key.items() if m == mid), None)
        if cand is None:
            continue
        g = float((w.get('geometry') or {}).get(mid, 1.0))
        if g < MIN_OWN_SHARE:
            counts['thin_share'] += 1
            why.setdefault(mid, 'no window holds as much as half of it; the '
                                'closest holds %.0f%%' % (100.0 * g))
            continue
        nh = float((w.get('protons') or {}).get(mid, 0.0))
        if nh <= 0:
            continue
        lo, hi = w['window']
        snr, lod_a, loq_a = _window_limits(
            ppm, intensities, lo, hi, included=included,
            noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm)
        areas = {int(s_i): float(v) * nh for s_i, v in by_sample.items()}
        med = (float(np.median(list(areas.values()))) if areas
               else float('nan'))
        own = [peaks_by_rank[r] for r in (w['lines'].get(mid) or ())
               if r in peaks_by_rank]
        out[mid] = {
            'signal': cand, 'nh': nh, 'snr': snr,
            'captured': float((w.get('fractions') or {}).get(mid, 1.0)),
            'detected': {s_i: bool(np.isfinite(v) and v >= K_LOD)
                         for s_i, v in snr.items()},
            'lines': [float(pk.get('ppm', 0.0)) for pk in own],
            'weights': pascal_weights(len(own)),
            'lod_area': lod_a, 'loq_area': loq_a, 'area_median': med,
            'strength': strength_band(med, loq_a),
            'window': (lo, hi), 'areas': areas,
            'width_ppm': abs(hi - lo), 'shifts': w.get('shifts'),
            'blank_area': w.get('blank_area', float('nan')),
            'shared_with': [m for m, _ in w['claimants'] if m != mid],
        }

    # A window's own refusal belongs to the signal it was built on, and to no
    # one else. An analyte reaching 5% into somebody else's window is not
    # answerable for that window failing, and marking it so penalised analytes
    # that had a clean window of their own.
    for entry in reasons.values():
        mid = entry['claimants'][0][0] if entry['claimants'] else None
        if mid is not None and mid not in out:
            why.setdefault(mid, entry['reason'])

    # What an analyte is answerable for. A refusal costs it confidence only
    # when the window that measures it is one it shares -- there the number
    # really is an upper bound holding somebody else's signal. A column the
    # joint system could not pin down, on a window nobody else is in, is a
    # statement about the system and not about that measurement: its own
    # window measured it, and the direct measurement stands.
    #
    # Read off which lines more than one metabolite is matched to, rather than
    # off the windows: a window refused before it was ever built has no
    # claimant list, and the analyte sharing that peak is exactly the one that
    # needs saying so.
    owners: dict = {}
    for (mid, ranks) in by_key:
        for r in ranks:
            owners.setdefault(int(r), set()).add(mid)
    sharing = {mid for (mid, ranks) in by_key
               for r in ranks if len(owners.get(int(r), ())) > 1}
    penalised = {mid: t for mid, t in why.items() if mid in sharing}

    if diagnostics is not None:
        # The widest contests, claimant by claimant, with what annotation said
        # about each. Twelve analytes on three lines is not a quantification
        # problem, and which gate let the twelve through can be read here.
        n_signals: dict = {}
        for (mid, _r) in by_key:
            n_signals[mid] = n_signals.get(mid, 0) + 1
        contests = []
        for w in sorted(windows, key=lambda w: -len(w['claimants'])):
            if w.get('bound') or len(w['claimants']) < 3:
                continue
            rows = []
            for mid, _nh_eff in w['claimants']:
                cand = by_key.get((mid, frozenset(w['lines'].get(mid) or ())))
                if cand is None:
                    cand = next((c for (m, r), c in by_key.items()
                                 if m == mid), None)
                ev = (cand.evidence or {}) if cand is not None else {}
                name = ((cand.payload or {}).get('name') if cand is not None
                        and isinstance(cand.payload, dict) else None)
                rows.append({
                    'met': mid, 'name': name or mid,
                    'mult': ev.get('multiplicity'),
                    'dppm': ev.get('ppm_diff'), 'dj': ev.get('delta_j_hz'),
                    'found': ev.get('n_found'), 'expected': ev.get('n_expected'),
                    'quality': ev.get('quality'),
                    'fraction': float((w.get('geometry') or {}).get(mid, 0.0)),
                    'signals': n_signals.get(mid, 0),
                })
            contests.append({'window': w['window'], 'owner': w.get('owner'),
                             'claimants': rows})
            if len(contests) >= 6:
                break
        tally: dict = {}
        for text in why.values():
            key = text.split(' -- ')[0].split(';')[0]
            tally[key] = tally.get(key, 0) + 1
        counts['resolved'] = len(out)
        counts['unsolved'] = len(why)
        diagnostics['shared'] = {**counts, **(solver_stats or {}),
                                 'reasons': tally,
                                 'metabolites': sorted(out),
                                 'penalised': len(penalised),
                                 'contests': contests}
    return out, penalised



# A singlet is quantified on its own only above this signal-to-noise:
# position is its whole evidence, and a peak of four sigma at a position
# is noise at a position. Ten, set with the mixture bench in view (wrong
# singlets at 1-17 sigma, right ones at 34-202), took creatinine, glycine,
# formate and sarcosine out of an organoid study whose singlets mostly
# stand at 5-10; five is the bar a signal had to clear to be checkable
# at all (quant_shared.FINGERPRINT_SUPPORT_SNR), and under detection's
# own floor.
SINGLET_MIN_SNR = 5.0
# The election is repeated without a winner whose amount its fingerprint
# contradicts, at most this many times: the peak goes to the next.
AMOUNT_ROUNDS = 3
# A measured amount can be twice the truth through overlap alone --
# glutamine's triplet read at 2x with glutamate's line inside it -- so a
# winner is withdrawn only when another of its signals falls short by
# this factor: twice is overlap, three times is absence. (Lysine on the
# same peak fell short eightfold.)
AMOUNT_RATIO = 3.0


def quantify(*, candidates, peaks_by_rank, ppm, intensities, samples_info,
             included, groups, dj_gold_hz, ambiguous_ranks=(),
             standard_areas=None, standard_nh=1.0, standard_conc=0.0,
             standard_name='Internal standard', standard_ppm=0.0,
             standard_snr=None,
             external_standard_samples=None, dilution=None, correction=None,
             use_pqn=True, local_baseline=True, proton_correction=True,
             align=False, blank_indices=None,
             noise_sigma=None, linewidth_ppm=0.008,
             drift_ppm=0.0, excluded_regions=(), deconvolve=True,
             quantify_shapes=True, resolve_shared=False,
             shared_overrides=None, diagnostics=None,
             residual_pool=None, singlet_min_snr=None) -> list[dict]:
    """Quantify a study and return one row per metabolite.

    ``singlet_min_snr`` is the signal-to-noise a singlet on its own must
    reach to be quantified (SINGLET_MIN_SNR when None): the analyst's
    choice per study, since no one number separates a faint real singlet
    from a peak of noise at a position.

    Runs the election and the measurement (``_quantify_once``), then asks
    of every winner whether the amount its signal implies is one the rest
    of its fingerprint can carry (``fingerprint_holds``, ADR-012 at the
    measured amount): a winner it contradicts is withdrawn, its row says
    so, and the election is run again without it so the peak goes to the
    next candidate -- lysine at 1.9 mM, on glutamine's triplet, would have
    CH2 multiplets the spectrum does not hold. A singlet is held to more:
    SINGLET_MIN_SNR, and at least one other listed signal seen at the
    amount it implies (an analyte listed with one signal is asked only
    the first). Up to AMOUNT_ROUNDS rounds. See ``_quantify_once`` for
    the rest.
    """
    kw = dict(
        peaks_by_rank=peaks_by_rank, ppm=ppm, intensities=intensities,
        samples_info=samples_info, included=included, groups=groups,
        dj_gold_hz=dj_gold_hz, ambiguous_ranks=ambiguous_ranks,
        standard_areas=standard_areas, standard_nh=standard_nh,
        standard_conc=standard_conc, standard_name=standard_name,
        standard_ppm=standard_ppm, standard_snr=standard_snr,
        external_standard_samples=external_standard_samples,
        dilution=dilution, correction=correction, use_pqn=use_pqn,
        local_baseline=local_baseline, proton_correction=proton_correction,
        align=align, blank_indices=blank_indices, noise_sigma=noise_sigma,
        linewidth_ppm=linewidth_ppm, drift_ppm=drift_ppm,
        excluded_regions=excluded_regions, deconvolve=deconvolve,
        quantify_shapes=quantify_shapes, resolve_shared=resolve_shared,
        shared_overrides=shared_overrides, diagnostics=diagnostics,
        residual_pool=residual_pool, singlet_min_snr=singlet_min_snr)
    cands = list(candidates or ())
    withdrawn: dict = {}
    rows: list = []
    for _round in range(AMOUNT_ROUNDS):
        rows, bad = _quantify_once(candidates=cands, withdrawn=withdrawn, **kw)
        if not bad:
            break
        withdrawn.update(bad)
        cands = [c for c in cands if c.met_id not in withdrawn]
        pool = kw.get('residual_pool')
        if pool:
            kw['residual_pool'] = [c for c in pool if c.met_id not in withdrawn]
    diag = kw.get('diagnostics')
    if diag is not None:
        diag['amount_withdrawn'] = {m: r for m, (r, _d) in withdrawn.items()}
    return rows


def _quantify_once(*, candidates, peaks_by_rank, ppm, intensities,
                   samples_info, included, groups, dj_gold_hz,
                   ambiguous_ranks=(),
                   standard_areas=None, standard_nh=1.0, standard_conc=0.0,
                   standard_name='Internal standard', standard_ppm=0.0,
                   standard_snr=None,
                   external_standard_samples=None, dilution=None,
                   correction=None,
                   use_pqn=True, local_baseline=True, proton_correction=True,
                   align=False, blank_indices=None,
                   noise_sigma=None, linewidth_ppm=0.008,
                   drift_ppm=0.0, excluded_regions=(), deconvolve=True,
                   quantify_shapes=True, resolve_shared=False,
                   shared_overrides=None, diagnostics=None,
                   residual_pool=None, withdrawn=None, singlet_min_snr=None):
    """One election and measurement. Returns ``(rows, withdrawn)``: the
    rows, and the winners whose measured amount the fingerprint
    contradicts (``{met_id: (reason, detail)}``) -- empty when the table
    stands. ``withdrawn`` (in) are the analytes earlier rounds took out;
    they get their rows here.

    Every metabolite that was a candidate gets a row, quantified or not: an
    analyte that vanishes from the table cannot be questioned, and the reason
    it has no number is the part an analyst can act on.

    ``candidates`` are ``processing.quantification.Candidate`` objects whose
    ``payload`` carries whatever the caller needs to build display fields —
    this engine only reads ``met_id``, the evidence, and the line peaks.

    ``deconvolve`` off leaves every winner credited with the whole of its
    window, shared or not, and reports nothing for the analytes that lost one.
    That is what the engine did before the balance existed, and it is the
    setting to compare against when a number moves.

    ``quantify_shapes`` off leaves every complex multiplet (``m*``, ADR-013
    -- fitted to the trace or matched on the peak list) out of the election
    and out of the table's numbers, each reported as not quantified with
    the switch named. The rest of the run is then the letter-multiplet and
    singlet engine. It used to leave out only the trace-fitted ones; a
    pattern matched on two apices still competed, and on the mixture bench
    nine such "2/2" patterns won peaks that belonged to the panel's real
    triplets (raffinose took glutamine's) with the switch off.

    ``resolve_shared`` on resolves, by their shapes, the winners' windows
    that hold another accepted analyte's signal (processing.quant_shared):
    the owner is credited with what its shape accounts for, the intruder
    held to its own clean measurement, the rest a remainder on the row.
    Off, a window is its owner's whole, as before. ``shared_overrides`` is
    what the analyst decided per owner in the review window (see
    ``quant_shared.resolve_shared_windows``).

    ``residual_pool`` (with ``resolve_shared``) is the signals the election
    did not accept. A winner's window whose accepted shapes leave a
    remainder is offered them, and one is taken in on three measured
    conditions (processing.quant_shared.resolve_residuals): its analyte is
    then quantified from that window, marked as identified by the
    remainder, at the lowest confidence.

    ``diagnostics``, when a dict is passed, is filled with what the run did
    that no row can show — how many windows were shared, and why the ones that
    resolved nothing resolved nothing. A step that silently does nothing is
    indistinguishable from a step that is not there.
    """
    candidates = list(candidates or ())
    residual_pool = list(residual_pool or ())
    shapes_off: dict = {}
    if not quantify_shapes:
        kept = [c for c in candidates if not _is_complex(c)]
        with_peaks = {c.met_id for c in kept}
        for c in candidates:
            if _is_complex(c) and c.met_id not in with_peaks:
                shapes_off[c.met_id] = SHAPES_OFF
        candidates = kept
        # off means off: not offered to the remainders either
        residual_pool = [c for c in residual_pool if not _is_complex(c)]
        if diagnostics is not None:
            diagnostics['shapes'] = {'off': True, 'left_out': len(shapes_off)}
    else:
        # On, a complex multiplet still never competes for a peak: it is
        # offered to the remainders (resolve_residuals, resolve_free_apices)
        # and identified only where a free apex, its own amount and its
        # whole shape agree. Fitted alone and elected unopposed it was
        # worth 41 false positives on the mixture bench.
        _complex = [c for c in candidates if _is_complex(c)]
        candidates = [c for c in candidates if not _is_complex(c)]
        _pooled = {id(c) for c in residual_pool}
        residual_pool += [c for c in _complex if id(c) not in _pooled]
        # an analyte whose every signal is a shape has a row either way:
        # one that vanishes from the table cannot be questioned
        _with_peaks = {c.met_id for c in candidates}
        for c in _complex:
            if c.met_id not in _with_peaks:
                shapes_off[c.met_id] = SHAPE_UNIDENTIFIED
    election = elect(candidates, dj_gold_hz=dj_gold_hz,
                    ambiguous_ranks=ambiguous_ranks)
    residual_ids: set = set()
    taken_over: dict = {}

    # ── Integrate every winner, per sample ───────────────────────────────────
    # A winner whose lines bound no window is recorded as such rather than
    # dropped. An analyte that simply vanishes between the election and the
    # table cannot be questioned, and this is the one place where a metabolite
    # can win and still have nothing to measure.
    measured, per_proton_areas, unmeasurable = {}, {}, []
    partial: dict = {}
    for mid, cand in election.winners.items():
        if (cand.evidence or {}).get('shape_fit'):
            # A multiplet judged as a shape holds no peak; its amount is
            # the copy of its shape that fits under each sample's trace
            # (processing.quant_shape). The shape's area is exact, so no
            # capture correction and no neighbour subtraction apply.
            nh = signal_nh((cand.evidence or {}).get('protons'))
            out = measure_shape(cand.evidence, ppm, intensities,
                                included=included, noise_sigma=noise_sigma,
                                blank_indices=blank_indices)
            if not out or not out['areas'] or not nh or nh <= 0:
                unmeasurable.append(mid)
                continue
            measured[mid] = {'signal': cand, 'nh': nh,
                             'noise_sigma': noise_sigma, **out}
            per_proton_areas[mid] = {s: a / nh
                                     for s, a in out['areas'].items()}
            continue
        lines = [peaks_by_rank[r] for r in sorted(cand.line_ranks)
                 if r in peaks_by_rank]
        if not lines and cand.peak_rank in peaks_by_rank:
            lines = [peaks_by_rank[cand.peak_rank]]
        out = integrate_signal(lines, ppm, intensities, included=included,
                               local_baseline=local_baseline, align=align,
                               blank_indices=blank_indices,
                               linewidth_ppm=linewidth_ppm,
                               other_peaks=list(peaks_by_rank.values()))
        if not out or not out['areas']:
            unmeasurable.append(mid)
            continue
        nh = signal_nh((cand.evidence or {}).get('protons'))
        lo, hi = out['window']
        # What the window left outside. A Lorentzian's tails never end, and how
        # much a window holds depends on its width in half-widths and on how
        # many lines share it -- so two signals integrated foot to foot lose
        # different fractions, and the ratio between them carries the
        # difference. On a study whose answer was known every analyte came back
        # about 7% high against a standard whose own window held 94% of its
        # line while a doublet's held 83%.
        _lw = (cand.evidence or {}).get('line_weights')
        # The whole pattern, not only the lines detection placed: a triplet
        # with one line found is a third of a pattern, and integrating that
        # line as if it were the signal, then dividing by the triplet's
        # protons, reports half of what is there. The missing lines stand
        # where the annotation predicted them, and the fraction the window
        # holds is computed over all of them -- so the area comes back as
        # the multiplet's. Below MIN_TRUSTED_CAPTURE of the pattern the
        # scale is not believed (one line of a triplet: x2 on a line that
        # may be somebody else's) and the analyte is not quantified.
        _ppms, _weights, _found_mask, _share = pattern_lines(
            (cand.evidence or {}).get('predicted_ppms') or (),
            [float(p.get('ppm', 0.0)) for p in lines],
            weights=(list(_lw) if _lw else None))
        _missing = [p for p, m in zip(_ppms, _found_mask) if not m]
        if _share < MIN_TRUSTED_CAPTURE:
            partial[mid] = (len(lines), len(_ppms), _share, _missing)
            continue
        # A window built from the lines and the neighbours is known: the
        # correction is arithmetic on it. Only a window from detection's
        # feet alone is refused when it holds too little.
        captured = lorentzian_captured_fraction(
            _ppms, _weights, lo, hi, linewidth_ppm / 2.0,
            min_trusted=(0.0 if out.get('geometry') == 'lines'
                         else MIN_TRUSTED_CAPTURE),
            segments=out.get('segments'))
        if captured:
            out['areas'] = {s_i: a / captured
                            for s_i, a in out['areas'].items()}
        snr, lod_a, loq_a = _window_limits(
            ppm, intensities, lo, hi, included=included,
            noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm,
            segments=out.get('segments'))
        med_area = (float(np.median(list(out['areas'].values())))
                    if out['areas'] else float('nan'))
        # Which samples actually show this signal, at the same ratio the
        # detection limit is defined by. The CSV reports the others as 'n.d.'
        # rather than as a concentration: a window over noise integrates to a
        # perfectly plausible number, and writing it would put a measurement
        # where the analyte was not seen.
        detected = {s_i: bool(np.isfinite(v) and v >= K_LOD)
                    for s_i, v in snr.items()}
        measured[mid] = {'signal': cand, 'nh': nh, 'snr': snr,
                         'captured': captured, 'detected': detected,
                         'lines': [float(p.get('ppm', 0.0)) for p in lines],
                         'pattern_lines': list(_ppms),
                         'missing_lines': _missing,
                         'pattern_share': _share,
                         'weights': _weights,
                         'lod_area': lod_a, 'loq_area': loq_a,
                         'area_median': med_area,
                         'strength': strength_band(med_area, loq_a), **out}
        per_proton_areas[mid] = {s: a / nh for s, a in out['areas'].items()}

    # ── The signal on each group's mean ─────────────────────────────────
    # The figure the report draws is one group's mean, and the SNR and the
    # shape agreement beside it must be read on THAT trace, for THIS
    # signal -- not the anchor peak's SNR over the study's mean while the
    # picture shows a group. Measured once here, so the table, the plot and
    # the PDF cannot disagree about which group they are looking at.
    if groups and linewidth_ppm:
        for mid, info in measured.items():
            lo, hi = info['window']
            ev = group_evidence(
                ppm, intensities, groups, lo=lo, hi=hi,
                noise=noise_sigma, lines=info.get('lines') or (),
                weights=info.get('weights') or (), hwhm=linewidth_ppm / 2.0)
            info['group_evidence'] = ev
            info['best_group'] = best_group(ev)

    # ── What each window holds of its neighbours ─────────────────────────────
    # A window is not a container for one signal. Lorentzian tails never reach
    # zero, so a strong neighbour a short way off puts a real amount of itself
    # inside — on a study whose answer was known, alanine read 6.4% high and
    # that 6.4% was lactate, 0.148 ppm away and eight times larger.
    #
    # Only for signals whose own capture correction was accepted. Where it was
    # refused the window is not describing the signal, so its area is the raw
    # one while a leak computed over the same width is the full one — taking
    # the second from the first subtracts a share the area was never credited
    # with. Measured: applying it there pushed every analyte negative and the
    # mean error from 2.8% to 4.1%.
    _trusted = {k: v for k, v in measured.items() if v.get('captured')}
    if len(_trusted) > 1 and linewidth_ppm:
        _corrected, _removed = subtract_neighbours(
            {k: {'window': v['window'], 'lines': v['lines'],
                 'weights': v['weights'], 'areas': v['areas'],
                 'segments': v.get('segments')}
             for k, v in _trusted.items()},
            linewidth_ppm / 2.0)
        for k, v in _trusted.items():
            v['areas'] = _corrected.get(k, v['areas'])
            v['neighbour_share'] = _removed.get(k, 0.0)
            per_proton_areas[k] = {s_i: a / v['nh']
                                   for s_i, a in v['areas'].items()}

    # ── A window holding another accepted analyte's signal ───────────────
    # Resolved by the shapes of the accepted analytes only, after each has
    # been measured on its own; an intruder is held to what that
    # measurement predicts (processing.quant_shared, ADR-015).
    if resolve_shared:
        _shared_names = {c.met_id: ((c.payload or {}).get('name')
                                    if isinstance(c.payload, dict) else None)
                         or c.met_id for c in candidates}
        _resolved = resolve_shared_windows(
            measured, candidates, ppm, intensities,
            per_proton_areas=per_proton_areas, included=included,
            linewidth_ppm=linewidth_ppm, names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            overrides=shared_overrides, noise_sigma=noise_sigma)
        if diagnostics is not None:
            diagnostics['shape_deconvolution'] = {
                'windows': len(_resolved),
                'remainder_median': {m: float(np.median(list(
                    r['remainder'].values()))) for m, r in _resolved.items()}}
        # ── The remainder as evidence ───────────────────────────────────
        _claimed_ranks = set()
        for _w in election.winners.values():
            _claimed_ranks |= set(_w.line_ranks)
            if _w.peak_rank is not None:
                _claimed_ranks.add(_w.peak_rank)
        _taken = resolve_residuals(
            measured, list(residual_pool or ()), ppm, intensities,
            included=included, linewidth_ppm=linewidth_ppm,
            names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            per_proton_areas=per_proton_areas, noise_sigma=noise_sigma,
            fingerprint_of=lambda c: ((c.payload or {}).get('met') or {}).get(
                'db_peaks') if isinstance(c.payload, dict) else None,
            free_apex_of=lambda c: [
                float(peaks_by_rank[r]['ppm']) for r in _found_ranks(c)
                if r in peaks_by_rank and r not in _claimed_ranks],
            offer_if=lambda rec: _structural_evidence(rec['signal']) >= 2,
            drift_ppm=float((residual_pool or [None])[0].evidence.get(
                'study_drift_ppm') or 0.0) if residual_pool
            and getattr(residual_pool[0], 'evidence', None) else 0.0)
        # ── A free apex nobody holds: the spectrum's own remainder ──────
        _claimed_ranks |= {r for _t in _taken.values()
                           for r in _found_ranks(_t['signal'])}
        _free_peaks = {r: p for r, p in peaks_by_rank.items()
                       if r not in _claimed_ranks}
        _alone = resolve_free_apices(
            measured, residual_pool, _free_peaks, ppm, intensities,
            included=included, linewidth_ppm=linewidth_ppm,
            names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            ranks_of=_found_ranks,
            noise_sigma=noise_sigma,
            fingerprint_of=lambda c: ((c.payload or {}).get('met') or {}).get(
                'db_peaks') if isinstance(c.payload, dict) else None,
            drift_ppm=float((residual_pool or [None])[0].evidence.get(
                'study_drift_ppm') or 0.0) if residual_pool
            and getattr(residual_pool[0], 'evidence', None) else 0.0,
            skip=set(_taken),
            trials_out=(diagnostics.setdefault('free_apex_trials', {})
                        if diagnostics is not None else None))
        _taken.update(_alone)
        for _cm, _t in _taken.items():
            _nh = float(_t['nh'])
            measured[_cm] = {
                'signal': _t['signal'], 'nh': _nh, 'areas': dict(_t['areas']),
                'window': _t['window'], 'segments': [],
                'captured': None, 'lines': [], 'weights': [],
                'shared_with': [_t['owner']],
                'by_residual': _t['owner'],
                'residual_before': _t['remainder_before'],
                'residual_after': _t['remainder_after'],
                'deconv': _t['deconv'],
                'noise_sigma': noise_sigma,
                'area_median': (float(np.median(list(_t['areas'].values())))
                                if _t['areas'] else float('nan')),
                # no limit of its own: the window is the owner's, and the
                # identification is the weakest the table gives
                'lod_area': float('nan'), 'loq_area': float('nan'),
                'snr': {},
                'strength': 1,
            }
            per_proton_areas[_cm] = {s_i: a / _nh
                                     for s_i, a in _t['areas'].items()}
            residual_ids.add(_cm)
        # ── What the quantified analytes list inside other windows ─────
        _listed = resolve_listed_intruders(
            measured, ppm, intensities, included=included,
            linewidth_ppm=linewidth_ppm, names=_shared_names,
            lines_of=lambda c: _found_lines(c, peaks_by_rank),
            per_proton_areas=per_proton_areas, noise_sigma=noise_sigma)
        if diagnostics is not None:
            diagnostics['listed_intruders'] = _listed
        for _m in [m for m, r in measured.items() if r.get('taken_over')]:
            _labels = measured[_m]['taken_over']
            taken_over[_m] = (WINDOW_TAKEN, ' ' + ', '.join(_labels) + '.')
            measured.pop(_m, None)
            per_proton_areas.pop(_m, None)
            residual_ids.discard(_m)
        if diagnostics is not None:
            diagnostics['residual'] = {
                'taken': {m: t['owner'] for m, t in _taken.items()},
                'trials': {m: r.get('residual_trials') for m, r in
                           measured.items() if r.get('residual_trials')}}

    # ── Two analytes on one peak ───────────────────────────────────────
    # The election gives a contested peak to one owner, which is right -- two
    # analytes integrating the same area is the defect that started this work.
    # But the area belongs to both, and until here the winner was credited with
    # all of it while the other was credited with nothing. Where every claimant
    # but one is measured elsewhere, subtracting what those must be
    # contributing leaves the rest, and the rest is arithmetic on measured
    # quantities rather than an estimate.
    if deconvolve:
        shared, shared_reasons = _resolve_shared(
            candidates, peaks_by_rank, ppm, intensities, included=included,
            local_baseline=local_baseline, align=align,
            blank_indices=blank_indices, linewidth_ppm=linewidth_ppm,
            noise_sigma=noise_sigma, drift_ppm=drift_ppm,
            excluded_regions=excluded_regions, diagnostics=diagnostics)
    else:
        shared, shared_reasons = {}, {}
        if diagnostics is not None:
            diagnostics['shared'] = {'off': True}
    recovered = set()
    # A window that resolved nothing leaves its winner credited with the whole
    # of it, including the part that is somebody else's. That has to be on the
    # row: a number silently left too large is worse than one refused.
    for mid, why in shared_reasons.items():
        if mid in measured and mid not in shared:
            measured[mid]['unshared_reason'] = why
    for mid, info in shared.items():
        if mid in measured:
            # Its number came from a window it does not hold alone; what it
            # holds of that window is this.
            measured[mid] = {**measured[mid], 'areas': info['areas'],
                             'shared_with': info['shared_with'],
                             'strength': info['strength'],
                             'area_median': info['area_median']}
        elif election.rejected.get(mid) == LOST_COMPETITION:
            # Only this one. Losing a contested peak is a verdict about which
            # analyte owns an area, and the balance answers exactly that. The
            # other rejections are verdicts about identity -- an ambiguous rank
            # says the search could not tell two candidates apart, an
            # unverified multiplet that the lines were never found -- and no
            # amount of arithmetic on areas settles who a signal belongs to.
            measured[mid] = info
            recovered.add(mid)
        else:
            continue
        per_proton_areas[mid] = {s_i: a / measured[mid]['nh']
                                 for s_i, a in measured[mid]['areas'].items()}

    # ── Dilution correction across samples ───────────────────────────────────
    # PQN before the standard: it corrects how concentrated the sample is, and
    # the standard then says what the corrected signal is worth. Running it
    # after would divide the answer by a factor derived from the answer.
    pqn_factors = {}
    if use_pqn and len(per_proton_areas) >= 3:
        per_proton_areas, pqn_factors = apply_pqn(per_proton_areas)

    standard = None
    if standard_areas:
        standard = standard_per_proton(
            standard_areas, standard_nh,
            external_samples=external_standard_samples)

    # ── The amount a winner implies, against the rest of its fingerprint ──
    _mean = None
    _arr_all = np.asarray(intensities, dtype=float)
    if _arr_all.ndim == 1:
        _arr_all = _arr_all.reshape(-1, 1)
    _cols = [i for i in (included if included is not None
                         else range(_arr_all.shape[1]))
             if 0 <= i < _arr_all.shape[1]]
    if _cols:
        _mean = _arr_all[:, _cols].mean(axis=1)
    bad: dict = {}
    if _mean is not None and noise_sigma and linewidth_ppm:
        for mid, info in list(measured.items()):
            if info.get('by_residual'):
                continue                  # checked when it was taken in
            sig = info['signal']
            ev = sig.evidence or {}
            met = (sig.payload or {}).get('met') if isinstance(sig.payload, dict) else None
            db_peaks = (met or {}).get('db_peaks') or ()
            pp = per_proton_areas.get(mid) or {}
            vals = [v for v in pp.values() if v is not None and np.isfinite(v) and v > 0]
            if not vals:
                continue
            held, verdicts = fingerprint_holds(
                db_peaks, float(np.median(vals)), ppm, _mean,
                noise=noise_sigma, linewidth_ppm=linewidth_ppm,
                drift_ppm=float(ev.get('study_drift_ppm') or 0.0),
                skip_ppm=ev.get('db_ppm'), excluded_regions=excluded_regions,
                ratio=AMOUNT_RATIO)
            info['fingerprint_at_amount'] = verdicts
            if not held:
                bad[mid] = (AMOUNT_CONTRADICTED, _fingerprint_detail(verdicts))
                continue
            # a singlet on its own: position is all it has
            if _structural_evidence(sig) <= 1 and is_singlet_like(
                    ev.get('multiplicity') or ev.get('type') or 's'):
                if ev.get('manual_quality') == 'good':
                    continue
                # the SNR the table shows: the signal's on the group the
                # figure draws, not the per-sample window estimate
                snr_rep = (info['group_evidence'][info['best_group']]['snr']
                           if info.get('best_group') else
                           representative_snr(info.get('snr')))
                # only what could be checked counts: a signal predicted
                # too faint to look for is not a signal not seen, and an
                # analyte whose every other signal is faint at this
                # amount is asked the signal-to-noise alone
                checkable = [v for v in verdicts
                             if v[3] == 'present' or v[3].startswith('contradicted')]
                seen = any(v[3] == 'present' for v in checkable)
                _min_snr = float(SINGLET_MIN_SNR if singlet_min_snr is None
                                 else singlet_min_snr)
                if (np.isfinite(snr_rep) and snr_rep < _min_snr) or (
                        checkable and not seen):
                    bad[mid] = (SINGLET_UNSUPPORTED,
                                _fingerprint_detail(verdicts, snr=snr_rep))
    for mid in bad:
        measured.pop(mid, None)
        per_proton_areas.pop(mid, None)
        residual_ids.discard(mid)

    rows = []
    for mid, info in measured.items():
        nh = info['nh']
        areas = {s: v * nh for s, v in per_proton_areas[mid].items()}
        conc = concentrations(areas, nh, standard=standard,
                              standard_conc=standard_conc,
                              external=bool(external_standard_samples),
                              dilution=dilution, correction=correction,
                              proton_correction=proton_correction)
        by_group = {g: [conc[s] for s in idxs
                        if s in conc and np.isfinite(conc[s]) and conc[s] > 0]
                    for g, idxs in (groups or {}).items()}
        good = [v for v in conc.values() if np.isfinite(v) and v > 0]
        level, dots = _confidence(
            info['signal'], info.get('strength'),
            shared=bool(info.get('shared_with')
                        or info.get('unshared_reason')))
        rows.append(_row(mid, info, conc, by_group, good, level, dots,
                         pqn_factors, standard, standard_conc,
                         bool(external_standard_samples), dilution,
                         correction))

    # What the integration subtracted, so a report can draw the very region
    # the number came from: with a local baseline the area is between the
    # trace and the line joining the window's ends, not down to zero.
    for r in rows:
        r['_local_baseline'] = bool(local_baseline)

    names = {c.met_id: ((c.payload or {}).get('name') if isinstance(
        c.payload, dict) else None) or c.met_id for c in (candidates or ())}
    for mid, reason in shapes_off.items():
        if mid in residual_ids or mid in measured:
            continue
        rows.append(_rejected_row(mid, reason))
    for mid, (n_found, n_all, share, missing) in partial.items():
        rows.append(_rejected_row(
            mid, PARTIAL_MULTIPLET,
            detail=(f' Here {n_found} of {n_all} lines were placed, holding '
                    f'{100 * share:.0f}% of the pattern; the missing lines '
                    f'were expected at '
                    f'{", ".join("%.4f" % v for v in missing)} ppm.')))
    for mid, reason in election.rejected.items():
        if mid in recovered or mid in residual_ids:
            continue
        rows.append(_rejected_row(mid, reason, shared_reasons.get(mid),
                                  beaten=election.beaten.get(mid),
                                  names=names))
    # A winner with no detected line is not a failure to measure — it is a
    # measurement of an upper bound, and reporting it as one is the whole of
    # processing.quant_detection_limit.
    for mid in unmeasurable:
        rows.append(_censored_row(
            election.winners[mid], ppm, intensities, included=included,
            standard=standard, standard_conc=standard_conc,
            noise_sigma=noise_sigma, linewidth_ppm=linewidth_ppm,
            drift_ppm=drift_ppm, local_baseline=local_baseline,
            external=bool(external_standard_samples),
            dilution=dilution, correction=correction,
            excluded_regions=excluded_regions))

    for mid, (reason, detail) in (withdrawn or {}).items():
        rows.append(_rejected_row(mid, reason, detail=detail))
    for mid, (reason, detail) in taken_over.items():
        rows.append(_rejected_row(mid, reason, detail=detail))
    for mid, (reason, detail) in bad.items():
        rows.append(_rejected_row(mid, reason, detail=detail))

    if standard_areas:
        rows.append(_standard_row(standard_areas, standard_nh, standard_conc,
                                  standard_name, standard_ppm, standard_snr,
                                  groups, standard,
                                  external_samples=external_standard_samples))
    return rows, bad


def _fingerprint_detail(verdicts, snr=None) -> str:
    """What the fingerprint check saw, for the row."""
    bits = []
    if snr is not None and np.isfinite(snr):
        bits.append(f'SNR {snr:.0f}')
    for c, mu, e, v in verdicts or ():
        if v.startswith('contradicted'):
            bits.append(f'{c:.2f} ppm {mu} expected SNR {e:.0f}, {v}')
        elif v == 'present':
            bits.append(f'{c:.2f} ppm {mu} present')
    return (' ' + '; '.join(bits) + '.') if bits else ''


def _censored_row(cand, ppm, intensities, *, included, standard,
                  standard_conc, noise_sigma, linewidth_ppm, drift_ppm,
                  local_baseline, external=False, dilution=None,
                  correction=None, excluded_regions=()) -> dict:
    """An analyte the experiment could not see, reported as the bound it is.

    The limits travel through the same chain as any concentration -- divided by
    nH, referenced to the standard -- because a limit in area units cannot be
    compared with the numbers in the table, and a limit nobody can compare is
    not a limit.

    The window is always integrated against a local baseline here, whatever the
    run's integration setting says. Currie's limits describe the NET signal --
    what stands above the background -- so comparing an integral that still
    contains the background against a limit built from noise alone compares two
    different quantities. Measured on a 197-point window: pure noise sitting on
    an offset of +2 sigma, with no peak in it at all, integrates to three times
    the quantification limit and is reported as an unresolved signal. Every one
    of the reference study's 28 censored analytes came back that way.

    The run's own ``local_baseline`` flag is a different decision: it governs
    how a quantified area is measured, where it costs a known ~5% on the ratio
    between two analytes. Here there is no choice to make.
    """
    ev = cand.evidence or {}
    mid = cand.met_id
    win = predicted_window(ev.get('predicted_ppms'),
                           linewidth_ppm=linewidth_ppm, drift_ppm=drift_ppm)
    if win is None or noise_sigma is None or not np.isfinite(noise_sigma):
        return _rejected_row(mid, NO_MEASURABLE_WINDOW)
    lo, hi = win
    ppm = np.asarray(ppm, dtype=float)
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_pts = int(((ppm >= lo) & (ppm <= hi)).sum())
    if n_pts < 2:
        return _rejected_row(mid, NO_MEASURABLE_WINDOW)
    dppm = float(abs(ppm[1] - ppm[0])) if ppm.size > 1 else 0.0

    # Where the analyte may have drifted to, tried one position at a time and
    # the best kept. Widening the window to cover the range instead would pay
    # for the whole range in noise, while a signal only ever occupies one
    # window's worth of it — on the reference study that made every limit
    # twice what the spectrum supports.
    idx = list(included if included is not None else range(arr.shape[1]))
    best_area, areas, best_off = float('-inf'), {}, 0.0
    for off in drift_offsets(drift_ppm, linewidth_ppm):
        here = {}
        for s_i in idx:
            if 0 <= s_i < arr.shape[1]:
                # Signed: the limit is built from noise that goes both
                # ways, and the quantified area's clip at the baseline
                # would put every noise window above it.
                a = integrate_sample(ppm, arr[:, s_i], lo + off, hi + off,
                                     local_baseline=True, clip=False)
                if a is not None:
                    here[int(s_i)] = float(a)
        if not here:
            continue
        med = float(np.median(list(here.values())))
        if med > best_area:
            best_area, areas, best_off = med, here, off
    if not areas:
        return _rejected_row(mid, NO_MEASURABLE_WINDOW)
    lo, hi = lo + best_off, hi + best_off
    area = best_area

    # An excluded region holds no analyte by construction: what is measured
    # there is the suppressed solvent. It gets neither a limit nor a verdict
    # about signal — a limit would report the solvent as the analyte's bound,
    # and an "unresolved" would report it as a buried analyte.
    if window_is_excluded(lo, hi, excluded_regions):
        row = _rejected_row(mid, EXCLUDED_REGION)
        row.update({
            'ppm': float((lo + hi) / 2.0),
            'status': STATUS_CENSORED[EXCLUDED_REGION],
            '_status_reasons': [censor_text(EXCLUDED_REGION, float('nan'),
                                            float('nan'))],
            '_censored': EXCLUDED_REGION,
            '_multiplet_window': [float(lo), float(hi)],
            '_hits': [cand.payload] if cand.payload is not None else [],
        })
        return row

    # The noise of THIS region, not of the quiet one the global estimate came
    # from. On the reference study six analytes contesting the histidine
    # imidazole region were handed identical limits because the global sigma
    # knew nothing about how crowded it is.
    sigma_here = local_noise_sigma(ppm, arr, lo, hi,
                                   linewidth_ppm=linewidth_ppm,
                                   global_sigma=noise_sigma) or noise_sigma
    lod_a, loq_a = detection_limits(sigma_here, dppm, n_pts)
    verdict = censor(area, lod_a, loq_a)

    nh = signal_nh(ev.get('protons'))
    lod_mm, loq_mm = (_as_concentration(v, nh, standard, standard_conc,
                                        external=external,
                                        dilution=dilution,
                                        correction=correction)
                      for v in (lod_a, loq_a))

    row = _rejected_row(mid, verdict)
    row.update({
        'ppm': float((lo + hi) / 2.0),
        'raw_area': area,
        'corr_area': area,
        'snr': (area / (lod_a / K_LOD)
                if np.isfinite(area) and lod_a and np.isfinite(lod_a)
                else float('nan')),
        'status': STATUS_CENSORED[verdict],
        '_status_reasons': [censor_text(verdict, lod_mm, loq_mm)],
        '_censored': verdict,
        '_lod_mm': lod_mm,
        '_loq_mm': loq_mm,
        '_noise_sigma': float(sigma_here),
        '_multiplet_window': [float(lo), float(hi)],
        '_nh': float(nh),
        '_areas_per_sample': areas,
        '_ann_score': float(ev.get('score') or 0.0),
        '_hits': [cand.payload] if cand.payload is not None else [],
        '_is_singlet': ev.get('multiplicity') == 's',
    })
    return row


def _as_concentration(area, nh, standard, standard_conc, *, external=False,
                      dilution=None, correction=None) -> float:
    """One area in the same units the table's concentrations are in.

    Through ``concentrations``, the same function every quantified row uses,
    rather than a second expression that says the same thing. It said almost
    the same thing: it left out the dilution and the correction factor, so a
    study that applied either had its limits on a different scale from the
    numbers they were meant to be compared with -- and a limit that cannot be
    compared is not a limit.

    The study's typical factor is the median across samples: a bound describes
    the run, not one injection of it.
    """
    if not np.isfinite(area):
        return float('nan')
    per_sample = concentrations({0: float(area)}, nh, standard=(
        {0: float(np.median([v for v in standard.values()
                             if v and np.isfinite(v) and v > 0]))}
        if standard else None),
        standard_conc=standard_conc, external=external,
        dilution={0: _median_factor(dilution)},
        correction={0: _median_factor(correction)})
    return float(per_sample.get(0, float('nan')))


def _median_factor(factors) -> float:
    v = [float(x) for x in (factors or {}).values()
         if x is not None and np.isfinite(x) and x > 0]
    return float(np.median(v)) if v else 1.0


def _standard_row(areas, nh, conc_nominal, name, ppm_pos, snr, groups,
                  standard, external_samples=None) -> dict:
    """The internal standard, measured like everything it standardises.

    Its recovery is the check on the whole run: the standard was added at a
    known concentration, so reading it back against itself must return that
    concentration. When it does not, the areas it divided are not trustworthy
    either, and the analyst needs to see that before reading any other row.

    Read over the injections that actually contain it. An externally injected
    standard is absent from the analytical samples by design, so integrating
    its window there measures the baseline underneath it; averaging that in
    drags the recovery down in proportion to how many analytical samples the
    study has and reports a preparation failure that did not happen. On a
    21-sample study with 5 external injections it read 57.5%.
    """
    if external_samples:
        keep = {int(s) for s in external_samples}
        areas = {s: a for s, a in (areas or {}).items() if int(s) in keep}
        groups = {g: [i for i in idxs if int(i) in keep]
                  for g, idxs in (groups or {}).items()}
    conc = concentrations(areas, nh, standard=standard,
                          standard_conc=conc_nominal)
    by_group = {g: [conc[s] for s in idxs
                    if s in conc and np.isfinite(conc[s]) and conc[s] > 0]
                for g, idxs in (groups or {}).items()}
    good = [v for v in conc.values() if np.isfinite(v) and v > 0]
    row = _rejected_row(name, 'standard')
    row.update({
        'metabolite': name,
        'hmdb_id': name,
        'ppm': float(ppm_pos or 0.0),
        'n_peaks': 1,
        'raw_area': float(np.mean(list(areas.values()))) if areas else float('nan'),
        'corr_area': float(np.mean(list(areas.values()))) if areas else float('nan'),
        'conc_mm': float(np.mean(good)) if good else float('nan'),
        'cv_by_group': {g: _cv_percent(v) for g, v in by_group.items()},
        'snr': float(snr) if snr is not None else 0.0,
        'status': STATUS_STRONG,
        '_confidence': 3,
        '_not_quantified': not good,
        '_status_reasons': [],
        '_nh': float(nh),
        '_areas_per_sample': dict(areas),
        '_concs_per_sample': conc,
        '_concs_by_group': by_group,
        '_is_is': True,
    })
    row.pop('_rejection', None)
    return row


def _row(mid, info, conc, by_group, good, level, dots, pqn_factors,
         standard=None, standard_conc=0.0, external=False, dilution=None,
         correction=None) -> dict:
    lo, hi = info['window']
    sig = info['signal']
    nh_ = info['nh']
    lod_mm, loq_mm = (_as_concentration(info.get(k), nh_, standard,
                                        standard_conc, external=external,
                                        dilution=dilution,
                                        correction=correction)
                      for k in ('lod_area', 'loq_area'))
    return {
        'hmdb_id': mid,
        'metabolite': mid,
        'ppm': float((lo + hi) / 2.0),
        'n_peaks': len(sig.line_ranks) or 1,
        'raw_area': float(np.mean(list(info['areas'].values()))),
        'corr_area': float(np.mean(list(info['areas'].values()))),
        'conc_mm': float(np.mean(good)) if good else float('nan'),
        'cv_by_group': {g: _cv_percent(v) for g, v in by_group.items()},
        # The SNR shown is the signal's on the group the figure draws
        # (best cosine, then highest SNR); the per-sample window SNR the
        # limits are built on stays in _snr_per_sample.
        'snr': (info['group_evidence'][info['best_group']]['snr']
                if info.get('best_group') else
                representative_snr(info.get('snr'))),
        '_snr_group': info.get('best_group'),
        '_snr_by_group': {g: e['snr'] for g, e in
                          (info.get('group_evidence') or {}).items()},
        '_cosine_by_group': {g: e['cosine'] for g, e in
                             (info.get('group_evidence') or {}).items()},
        'status': dots,
        'vip_score': float('nan'),
        '_confidence': level,
        '_integ_mode': 'Shape' if info.get('shape') else 'Multiplet',
        # What the shape was, for a report to draw the fit it quantified
        '_shape_fit': ((sig.evidence or {}).get('shape_fit')
                       if info.get('shape') else None),
        '_shape_explained': info.get('shape_explained'),
        '_deconv': info.get('deconv'),
        '_by_residual': info.get('by_residual'),
        '_residual_trials': info.get('residual_trials'),
        '_noise_sigma': info.get('noise_sigma'),
        '_not_quantified': not good,
        '_peak': None,
        '_peak_rank': sig.peak_rank,
        '_display_peak_rank': sig.peak_rank,
        '_multiplet_window': [float(lo), float(hi)],
        # The pieces of that window the area was summed over, and the
        # foreign peaks between the lines that kept them apart: a triplet
        # measured on its three lines around somebody's doublet says so.
        '_integ_segments': [[float(a), float(b)]
                            for a, b in (info.get('segments') or ())],
        '_foreign_inside': [float(v) for v in (info.get('foreign_inside')
                                               or ())],
        # How much of the window the trace ran under the baseline: what the
        # integration counted as nothing (quant_integration.integrate_sample).
        '_below_baseline_share': info.get('below_baseline_share'),
        # Where the signal's own lines are, so a plot can mark who is who --
        # and where the pattern's missing lines were expected, so a signal
        # scaled from part of its multiplet shows which part.
        '_line_ppms': [float(v) for v in (info.get('lines') or ())],
        '_missing_lines': [float(v) for v in (info.get('missing_lines') or ())],
        '_pattern_share': info.get('pattern_share'),
        # Every line the reference lists for this signal, at the centre the
        # annotation found -- the pattern the letter promises, drawn beside
        # the lines actually integrated.
        '_reference_lines': [float(v) for v in
                             ((sig.evidence or {}).get('reference_ppms') or ())],
        '_nh': float(info['nh']),
        '_areas_per_sample': info['areas'],
        '_concs_per_sample': conc,
        '_concs_by_group': by_group,
        '_pqn_factors': pqn_factors,
        '_snr_per_sample': info.get('snr', {}),
        '_detected_per_sample': info.get('detected', {}),
        '_peak_align_shifts': info.get('shifts') or None,
        '_blk_mean_area': info.get('blank_area', float('nan')),
        '_is_singlet': (sig.evidence or {}).get('multiplicity') == 's',
        '_multiplicity': (sig.evidence or {}).get('multiplicity'),
        '_ann_score': float((sig.evidence or {}).get('score') or 0.0),
        # Every quantified analyte carries its own limits now. Without them
        # nobody could tell a number ten times the limit from one at 1.1x, and
        # the confidence had no way to say which it was.
        '_lod_mm': lod_mm,
        '_loq_mm': loq_mm,
        '_strength': info.get('strength'),
        '_captured_fraction': info.get('captured'),
        '_neighbour_share': info.get('neighbour_share', 0.0),
        '_shared_with': list(info.get('shared_with') or ()),
        '_unshared_reason': info.get('unshared_reason'),
        '_status_reasons': (_residual_reason(info)
                            + (_deconv_reason(info) if info.get('deconv')
                               else _shared_reason(info))
                            + _pattern_reason(info)
                            + _segments_reason(info)
                            + _baseline_reason(info)
                            + _shape_reason(info)
                            + _weak_reason(info, loq_mm)),
        '_hits': [sig.payload] if sig.payload is not None else [],
        # Whether this number came out of a window held with other analytes,
        # and how much of its own signal that window held. The overlap
        # correction the old stage promised, from the engine that now does it.
        '_used_j_resolve': bool(info.get('shared_with')),
        '_j_resolve_frac': (float(info.get('captured') or 1.0)
                            if info.get('shared_with') else 1.0),
        '_used_mcr': False,
        '_mcr': None,
        '_lorentz': None,
        '_is_is': False,
    }


def _found_ranks(cand) -> list:
    return sorted(cand.line_ranks or ()) or (
        [cand.peak_rank] if cand.peak_rank is not None else [])


def _found_lines(cand, peaks_by_rank) -> list:
    """The peaks detection found for a signal, in ppm; empty when it found
    none."""
    return [float(peaks_by_rank[r].get('ppm', 0.0))
            for r in _found_ranks(cand) if r in peaks_by_rank]


def _residual_reason(info) -> list:
    """Said of an analyte identified by what another analyte's window left
    unexplained, and of a window that took such a signal in."""
    owner = info.get('by_residual')
    if owner == 'spectrum':
        return [f"Identified on a free apex: a detected peak no accepted "
                f"analyte holds, at which this analyte's reference shape, "
                f"scaled to the apex, fits under the trace over its whole "
                f"width and accounts for "
                f"{100 * (1 - float(info.get('residual_after') or 0)):.0f}% "
                f"of its region, with the rest of its fingerprint able to "
                f"carry the amount. Its lines were not confirmed one by "
                f"one: a weak identification, to be checked."]
    if owner:
        return [f"Identified by the remainder of {owner}'s window: the "
                f"accepted shapes left {100 * float(info.get('residual_before') or 0):.0f}% "
                f"of it unexplained, this analyte's reference shape accounts "
                f"for it (remainder {100 * float(info.get('residual_after') or 0):.0f}% "
                f"with it), and its shape can be told from theirs. Its "
                f"lines were not confirmed one by one: a weak identification, "
                f"to be checked."]
    d = info.get('deconv') or {}
    if d.get('residual'):
        return [f"{', '.join(d['residual'])} was taken into this window on "
                f"the remainder the accepted shapes left; its share is what "
                f"its reference shape accounts for."]
    trials = info.get('residual_trials') or []
    refused = [t['label'] for t in trials if t.get('verdict') != 'taken']
    if refused and not d.get('residual'):
        return [f"The remainder was offered to {', '.join(refused)}; none "
                f"met the three conditions (remainder under "
                f"{100 * RESIDUAL_ACCEPT:.0f}%, at least "
                f"{100 * RESIDUAL_EXPLAINED:.0f}% of it explained, shape "
                f"separable)."]
    return []


def _deconv_reason(info) -> list:
    """Said of a window resolved by shapes: with whom, and what the shapes
    left unexplained."""
    d = info.get('deconv')
    if not d:
        return []
    mid = getattr(info.get('signal'), 'met_id', None)
    others = [c['label'] for c in d.get('components', ())
              if c.get('met_id') != mid]
    rem = d.get('remainder') or {}
    med = float(np.median(list(rem.values()))) if rem else 0.0
    out = [f"Shared window resolved by shape with {', '.join(others)}: this "
           f"area is what this analyte's own shape accounts for."]
    if d.get('listed'):
        out.append(f"{', '.join(d['listed'])}: the reference lists it "
                   f"inside this window; its analyte is quantified elsewhere, "
                   f"and its share here is what that measurement predicts, "
                   f"subtracted (kept only because it fits under the trace).")
    if [f for f in (d.get('fixed') or ()) if f not in (d.get('listed') or ())]:
        out.append(f"{', '.join(f for f in d['fixed'] if f not in (d.get('listed') or ()))}: "
                   f"a shape this signal's own shape makes, so its share "
                   f"was not fitted but fixed at what its own clean signal "
                   f"predicts, and subtracted.")
    placed = [c['label'] for c in d.get('components', ())
              if c.get('placed_at_expected')]
    if placed:
        out.append(f"{', '.join(placed)} was placed where annotation "
                   f"expects it: detection had given it this signal's own "
                   f"line.")
    if d.get('skipped'):
        out.append(f"{', '.join(d['skipped'])} is listed here too but "
                   f"took no share: not placed by detection, a shape this "
                   f"signal's own shape makes, or left out by the analyst.")
    if med >= REMAINDER_NOTE:
        out.append(f'The accepted shapes leave {100 * med:.0f}% of the '
                   f'window unexplained: something else is here.')
    return out


def _pattern_reason(info) -> list:
    """Said of a multiplet measured on part of its lines: which were
    missing, and that the area was scaled by the pattern to the whole."""
    missing = [float(v) for v in (info.get('missing_lines') or ())]
    if not missing:
        return []
    share = float(info.get('pattern_share') or 0.0)
    n_all = len(info.get('pattern_lines') or ()) or (
        len(info.get('lines') or ()) + len(missing))
    return [f'Partial multiplet: {n_all - len(missing)} of {n_all} lines were '
            f'placed by detection, holding {100 * share:.0f}% of the pattern; '
            f'the missing lines were expected at '
            f'{", ".join("%.4f" % v for v in missing)} ppm. The area is the '
            f'placed lines scaled by the pattern to the whole multiplet, '
            f'so it assumes the missing lines are there, under something.']


# Above this share of the window under the baseline, the row says so: a
# window on a signal has a few noise points under its baseline, not a
# quarter of its length.
BASELINE_SILENCED_SHARE = 0.25


def _baseline_reason(info) -> list:
    """Said of a signal whose window ran under its local baseline over a
    good part of its length: that part counted nothing, and a window
    standing that much under its own baseline is a window on a flank or a
    dip, not on the signal."""
    share = info.get('below_baseline_share')
    if share is None or float(share) < BASELINE_SILENCED_SHARE:
        return []
    return ['the trace runs under the local baseline on %d%% of the window; '
            'that part counted nothing (only area above the baseline is '
            'integrated)' % round(100 * float(share))]


def _is_complex(cand) -> bool:
    """A complex multiplet: judged as a measured pattern (``m*``), with or
    without a shape fitted to the trace."""
    ev = cand.evidence or {}
    return bool(ev.get('shape_fit')) or normalize_mult(
        ev.get('multiplicity')) == PATTERN_MULT


def _segments_reason(info) -> list:
    """Said of a signal whose window holds a foreign peak between its lines:
    which peaks, and on how many pieces the lines were then integrated."""
    foreign = [float(v) for v in (info.get('foreign_inside') or ())]
    if not foreign or info.get('deconv'):
        return []
    segs = info.get('segments') or ()
    where = ', '.join('%.4f' % v for v in foreign)
    what = ('%d foreign peak%s at %s ppm %s between this signal\'s lines'
            % (len(foreign), 's' if len(foreign) > 1 else '', where,
               'lie' if len(foreign) > 1 else 'lies'))
    if len(segs) > 1:
        return ['%s; integrated on the lines alone, in %d pieces, the '
                'gap left out' % (what, len(segs))]
    return ['%s, too close to its lines to cut around; integrated whole'
            % what]


def _shape_reason(info) -> list:
    """Said of a shape quantified in a region it does not hold alone: the
    copy that fits under the trace is bounded by the least overlapped apex,
    so the number is an upper bound there."""
    if not info.get('shape'):
        return []
    ex = info.get('shape_explained')
    if ex is None:
        return ['Quantified from its fitted shape; no peak claimed.']
    if float(ex) >= 0.9:
        return [f'Quantified from its fitted shape, which explains '
                f'{100 * float(ex):.0f}% of its region.']
    return [f'Upper bound: quantified from its fitted shape in a region it '
            f'shares -- the shape explains {100 * float(ex):.0f}% of it.']


def _weak_reason(info, loq_mm) -> list:
    """Said when the area does not clear the quantification limit by the
    agreed factor. The identity is not in question here — only the precision.
    """
    if info.get('strength', 3) >= 3:
        return []
    return [f'Weak signal: the measured area does not reach '
            f'{STRONG_ABOVE_LOQ:.0f}x the limit of quantification '
            f'({loq_mm:.4g} mM). The analyte was identified and won its peak; '
            f'it is the quantity that is small, and the number less precise.']


def _shared_reason(info) -> list:
    """What the reader needs to know about an area that was shared."""
    with_ = list(info.get('shared_with') or ())
    if not with_:
        why = info.get('unshared_reason')
        if not why:
            return []
        return [f'Shared window, not split: this area is the whole of a '
                f'window other analytes also sit in, so it is an upper bound '
                f'on what belongs to this one. It could not be split because '
                f'{why}.']
    return [f'Shared window: the signal is held with {", ".join(with_)}. '
            f'This area is what remains once their own measurements account '
            f'for what they contribute here, so it depends on those being '
            f'right and on there being no other analyte in the window.']


def _rejected_row(mid, reason, shared_reason=None, *, beaten=None,
                  names=None, detail='') -> dict:
    """A metabolite with no number still gets a row, and says why.

    An analyte that disappears from the table cannot be questioned, and the
    reason is the actionable half: a withdrawn multiplet comes back when
    detection resolves its lines, an ambiguous rank when the analyst marks one
    candidate, a beaten one only if the winner is wrong.
    """
    return {
        'hmdb_id': mid,
        'metabolite': mid,
        'ppm': 0.0,
        'n_peaks': 0,
        'raw_area': float('nan'),
        'corr_area': float('nan'),
        'conc_mm': float('nan'),
        'cv_by_group': {},
        'snr': 0.0,
        'status': STATUS_NONE,
        'vip_score': float('nan'),
        '_confidence': 0,
        '_integ_mode': 'Multiplet',
        '_not_quantified': True,
        '_status_reasons': ([REJECTION_TEXT.get(reason, str(reason)) + detail]
                            + ([f'Its share of the shared window could not be '
                                f'resolved: {shared_reason}']
                               if shared_reason else [])
                            + [f'Lost peak #{r} to '
                               f'{(names or {}).get(w, w)} on {how}.'
                               for r, w, how in (beaten or ())]),
        '_rejection': reason,
        '_peak': None,
        '_peak_rank': None,
        '_display_peak_rank': None,
        '_multiplet_window': None,
        '_nh': float('nan'),
        '_areas_per_sample': {},
        '_concs_per_sample': {},
        '_concs_by_group': {},
        '_pqn_factors': {},
        '_snr_per_sample': {},
        '_detected_per_sample': {},
        '_peak_align_shifts': None,
        '_blk_mean_area': float('nan'),
        '_is_singlet': False,
        '_ann_score': 0.0,
        '_hits': [],
        '_used_j_resolve': False,
        '_j_resolve_frac': 1.0,
        '_used_mcr': False,
        '_mcr': None,
        '_lorentz': None,
        '_is_is': False,
    }
