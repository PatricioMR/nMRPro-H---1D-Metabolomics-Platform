# processing/annotation_detectability.py
"""What can be expected to show up, and what competes with what.

Two questions the annotation asks of the spectrum itself, before any candidate
is judged:

  * **Detectability** — given how strong a metabolite's best signal is, how
    strong should its other signals be? A resonance predicted to sit under the
    noise floor is absent for a physical reason, not because the assignment is
    wrong.

    This module MEASURES that expectation and stops there. It used to also act
    on it, flagging signals "below detection limit" and hiding them from the
    tree, from the overlap map and from quantification. That decision belongs
    with the singlet-anchoring work — a signal predicted to be invisible is
    exactly the case an anchor from a companion multiplet should settle — and
    until that exists, hiding signals on a proton-ratio estimate removes
    evidence the analyst never gets to see.
  * **Overlap** — which metabolites lay claim to each detected peak, counting
    every line of a multiplet and not only its anchor.

Both were methods on the results window. Neither touches a widget.
"""
from __future__ import annotations

import numpy as np

# ppm tolerance for tying a predicted line to a detected peak when building the
# overlap map. Tighter than the annotation tolerance on purpose: this asks
# "is this the same peak", not "could this be the same signal".
OVERLAP_TOL_PPM = 0.015

# Where the noise is estimated from when the spectrum reaches that far.
NOISE_TAIL_PPM = 9.5


def proton_count(entry, default: float = 1.0) -> float:
    """Proton count of a DB signal, fractional included.

    BMRB splits a group's protons across the signals it is observed in, so
    values like 0.5 and 0.75 are normal and meaningful. Reading them with
    ``int()`` turns 0.5 into 0, and an expected SNR computed from a zero proton
    ratio is zero — which marked the signal as below the detection limit and hid
    it from the analyst. Measured on a real project: 86 of 114 fractional-nH
    signals were hidden that way, with an expected SNR of exactly 0.00.
    """
    try:
        value = float(entry.get('protons', default) or default)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def estimate_noise(ppm_arr, intensity_arr) -> float:
    """Noise level from the far-upfield tail, or a low percentile as fallback."""
    tail_mask = ppm_arr > NOISE_TAIL_PPM
    if tail_mask.sum() < 20:
        noise = float(np.percentile(np.abs(intensity_arr), 5))
    else:
        tail = intensity_arr[tail_mask]
        noise = float(np.median(np.abs(tail - np.median(tail)))) * 1.4826
    return noise if noise > 0 else 1.0


def mark_detectability(metabolites, ppm_arr, intensity_arr) -> float:
    """Annotate every matched peak and DB peak with its expected SNR.

    The anchor is the metabolite's strongest signal weighted by proton count;
    every other signal's expected SNR follows from the proton ratio. Returns the
    noise level used.

    Nothing is hidden or excluded on the strength of this estimate: it is
    reported, and the analyst decides.
    """
    noise = estimate_noise(ppm_arr, intensity_arr)

    for met in metabolites:
        matched = met.get('matched_peaks', [])
        for mp in matched:
            ppk = mp.get('peak_ppm', mp.get('db_ppm', 0))
            idx = int(np.argmin(np.abs(ppm_arr - ppk)))
            mp['obs_intensity'] = float(intensity_arr[idx])
            mp['obs_snr'] = mp['obs_intensity'] / noise

        if not matched:
            continue

        anchor = max(matched,
                     key=lambda p: p.get('obs_snr', 0) * proton_count(p))
        anchor_snr = anchor.get('obs_snr', 0)
        anchor_protons = proton_count(anchor)

        for mp in matched:
            expected = (anchor_snr * (proton_count(mp) / anchor_protons)
                        if anchor_protons > 0 else anchor_snr)
            mp['expected_snr'] = expected

        # DB peaks too, matched or not — the estimate is informative for an
        # unmatched signal as well.
        for dp in met.get('db_peaks', []):
            dp['expected_snr'] = (
                anchor_snr * (proton_count(dp) / anchor_protons)
                if anchor_protons > 0 else anchor_snr)

    return noise


def build_overlap_map(metabolites, peaks, *, tol: float = OVERLAP_TOL_PPM):
    """Return ``{peak_rank: [metabolite names]}`` — who claims each peak.

    Every line of a multiplet is registered, not just its anchor, so a doublet
    belonging to X still counts as a competitor for Y when Y overlaps only one
    of its legs.
    """
    rank_by_ppm = {float(pk.get('ppm', 0)): pk.get('rank')
                   for pk in (peaks or []) if pk.get('rank') is not None}
    ppms = sorted(rank_by_ppm)

    def _rank_near(value):
        best_d, best_r = tol, None
        for p in ppms:
            d = abs(p - value)
            if d < best_d:
                best_d, best_r = d, rank_by_ppm[p]
            elif p > value + tol:
                break
        return best_r

    claims: dict = {}

    def _claim(rank, name):
        if rank is None:
            return
        holders = claims.setdefault(rank, [])
        if name not in holders:
            holders.append(name)

    for met in metabolites:
        name = met['name']
        for mp in met.get('matched_peaks', []):
            rank = mp.get('peak_rank')
            _claim(rank, name)
            lines = mp.get('predicted_lines') or []
            if len(lines) > 1:
                for line in lines:
                    other = _rank_near(float(line.get('ppm', 0)))
                    if other != rank:
                        _claim(other, name)
    return claims


# ── When absence is evidence ─────────────────────────────────────────────────
# Every gate above judges a signal on its own: does this doublet sit where the
# database says, with the coupling the database gives? On the reference study
# eight para-substituted phenols passed it on the same doublet at 6.9 ppm --
# 2 of 2 lines, delta-J under 0.5 Hz, delta-ppm under 0.05, every one Good --
# and every one is right about the signal. The ortho protons of any of them
# give that doublet. What tells them apart is not in it.
#
# It is in the signals they do NOT share. 4-hydroxybenzaldehyde's aldehyde
# proton sits alone at 9.7 ppm; if the doublet at 6.9 is its 2H, the singlet
# at 9.7 is half that area, and if 9.7 holds noise the doublet is somebody
# else's. That is the analyst's first question about any candidate -- "if it
# were here, would I see the big one?" -- and it was carried in the score as a
# 5% coverage penalty that no confirmed signal could fall through.
#
# Absence is evidence only when absence was measurable (ADR-007), so the test
# fires under four conditions and none of them is negotiable:
#
#   the signal would be clearly visible   expected SNR at or above the floor
#   its region is not excluded            nothing in a suppressed region counts
#   no detected peak sits in the window   a peak there is overlap, or the
#                                         signal drifted; either way, not absent
#   what IS there is far below expected   by the ratio, not by a little
#
# What fires is a contradiction of the fingerprint, and it marks every one of
# the analyte's signals Bad with the reason on the card. Manual marks are never
# overwritten, and the analyst can lift it.

# A missing signal only counts when it should have been unmistakable. Three
# times detection's own SNR floor of 6: below that, "absent" and "buried" are
# the same observation.
CONTRADICTION_MIN_EXPECTED_SNR = 20.0

# And what is there must fall short by this factor. Pure noise measured peak to
# peak reads about 4 sigma, so a signal expected at 20 is contradicted only by
# a region holding under 4 -- the noise itself -- and not by a weak line.
CONTRADICTION_RATIO = 5.0


def _line_share(mult) -> float:
    """The tallest line's share of a first-order multiplet's height.

    A singlet puts its whole proton count into one line; a doublet halves it, a
    triplet's centre line carries half, a quartet's inner lines three eighths.
    Comparing a singlet's apex with a triplet's without this would expect the
    triplet twice as tall as it can be.
    """
    from processing.annotation_scoring import normalize_mult
    m = normalize_mult(mult or 's')
    n = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'quint': 5, 'p': 5, 'sext': 6,
         'sept': 7, 'h': 6}.get(m, 1)
    if n <= 1:
        return 1.0
    row = [1.0]
    for _ in range(n - 1):
        row = [a + b for a, b in zip([0.0] + row, row + [0.0])]
    return max(row) / sum(row)


# How far a resonance may sit from where the database puts it, as a multiple
# of the drift the study itself showed (the median |delta-ppm| of everything the
# search matched). Three medians covers the bulk of a drift distribution; a
# peak further away than that is not this signal wearing a different shift.
DRIFT_MULTIPLE = 3.0


def search_half_width(*, tolerance_ppm, drift_ppm, linewidth_ppm) -> float:
    """Where a missing signal could be, if it were there at all.

    Not the annotation tolerance. That number answers "could this be the same
    signal" and is generous by design -- 0.05 ppm, thirty hertz at 600 MHz --
    and used as the width of a window it is wide enough to contain a detected
    peak almost anywhere in a crowded spectrum: on the reference study peaks
    sit 0.026 ppm apart on average, so a window 0.1 ppm wide holds one of
    somebody's nearly every time, and the gate fired on nothing that
    mattered.

    The honest answer is the drift the study showed, with margin, and never
    less than the line itself. Capped at the tolerance, because the search
    would not have matched anything further out either.
    """
    tol = abs(float(tolerance_ppm or 0.0))
    d = abs(float(drift_ppm or 0.0))
    lw = abs(float(linewidth_ppm or 0.0))
    half = max(DRIFT_MULTIPLE * d, 2.0 * lw)
    if half <= 0:
        return tol
    return min(tol, half) if tol > 0 else half


def fingerprint_contradictions(metabolites, ppm_arr, intensity_arr, peaks, *,
                               tolerance_ppm, excluded_regions=(),
                               search_ppm=None,
                               min_expected_snr=CONTRADICTION_MIN_EXPECTED_SNR,
                               ratio=CONTRADICTION_RATIO, manual_marks=None,
                               stats=None):
    """Mark Bad every analyte whose fingerprint the spectrum contradicts.

    Returns the number of analytes marked. Each one carries
    ``_fingerprint_contradictions``: the missing signals, what each should have
    read and what its region held, so the card can show the arithmetic.

    ``search_ppm`` is the half-width of the window a missing signal is looked
    for in -- see ``search_half_width``; the tolerance when not given.
    ``stats``, when a dict is passed, is filled with how many predicted
    signals were considered and on which condition each one stopped, because
    a gate that fires on nothing looks exactly like one that is not there.
    """
    from processing.spectral_regions import in_excluded_region

    manual_marks = manual_marks or {}
    ppm = np.asarray(ppm_arr, dtype=float)
    y = np.asarray(intensity_arr, dtype=float)
    if ppm.size < 10 or y.size != ppm.size:
        return 0
    noise = estimate_noise(ppm, y)
    tol = abs(float(search_ppm if search_ppm is not None else tolerance_ppm))
    peak_ppms = np.array([float(p.get('ppm', 0.0)) for p in (peaks or ())])
    counts = {'metabolites': 0, 'predicted': 0, 'matched': 0,
              'unresolved': 0, 'faint': 0, 'excluded': 0,
              'peak_in_window': 0, 'signal_present': 0, 'contradicted': 0,
              'analytes_marked': 0, 'noise': float(noise),
              'search_half_width_ppm': float(tol)}

    def _apex_snr(mp):
        ppk = float(mp.get('peak_ppm') or mp.get('db_ppm') or 0.0)
        i = int(np.argmin(np.abs(ppm - ppk)))
        return max(float(y[i]), 0.0) / noise

    n_marked = 0
    for met in metabolites or ():
        matched = [mp for mp in (met.get('matched_peaks') or ())
                   if mp.get('peak_rank') is not None
                   and mp.get('annotation_quality') != 'bad']
        if not matched:
            continue
        # The signal that sets the scale: the tallest per proton, per line.
        anchor = max(matched, key=_apex_snr)
        a_snr = _apex_snr(anchor)
        a_per = (proton_count(anchor)
                 * _line_share(anchor.get('type') or anchor.get('multiplicity')))
        if a_snr <= 0 or a_per <= 0:
            continue

        counts['metabolites'] += 1
        matched_at = {round(float(mp.get('db_ppm', 0.0)), 3) for mp in matched}
        found = []
        for dp in met.get('db_peaks') or ():
            counts['predicted'] += 1
            mult = dp.get('type') or ''
            if str(mult).lower().strip() in ('m', 'multiplet', 'peak', ''):
                counts['unresolved'] += 1
                continue
            centre = float(dp.get('ppm_center', dp.get('ppm', 0.0)))
            if round(centre, 3) in matched_at:
                counts['matched'] += 1
                continue
            expected = a_snr * (proton_count(dp) * _line_share(mult)) / a_per
            if expected < float(min_expected_snr):
                counts['faint'] += 1
                continue
            lo, hi = centre - tol, centre + tol
            if (in_excluded_region(lo, excluded_regions)
                    or in_excluded_region(hi, excluded_regions)
                    or in_excluded_region(centre, excluded_regions)):
                counts['excluded'] += 1
                continue
            if peak_ppms.size and np.any((peak_ppms >= lo) & (peak_ppms <= hi)):
                counts['peak_in_window'] += 1
                continue
            m = (ppm >= lo) & (ppm <= hi)
            if m.sum() < 3:
                continue
            seg = y[m]
            measured = (float(seg.max()) - float(seg.min())) / noise
            if measured * float(ratio) < expected:
                counts['contradicted'] += 1
                found.append({'ppm': centre, 'multiplicity': mult,
                              'protons': proton_count(dp),
                              'expected_snr': float(expected),
                              'measured_snr': float(measured)})
            else:
                counts['signal_present'] += 1
        if not found:
            met.pop('_fingerprint_contradictions', None)
            continue

        met['_fingerprint_contradictions'] = found
        text = '; '.join(f'{c["ppm"]:.2f} ppm {c["multiplicity"]} '
                         f'{c["protons"]:g}H expected SNR '
                         f'{c["expected_snr"]:.0f}, region holds '
                         f'{c["measured_snr"]:.0f}'
                         for c in found)
        marked_any = False
        for mp in met.get('matched_peaks') or ():
            if (met.get('id'), mp.get('peak_rank')) in manual_marks:
                continue
            if mp.get('annotation_quality') == 'bad':
                continue
            mp['annotation_quality'] = 'bad'
            mp['auto_anchor'] = False
            mp.setdefault('_status_reasons', []).append(
                f'Fingerprint contradicted: if this were '
                f'{met.get("name", met.get("id"))}, its other signal(s) would '
                f'be visible and the spectrum holds noise there ({text}).')
            marked_any = True
        if marked_any:
            n_marked += 1
    counts['analytes_marked'] = n_marked
    if stats is not None:
        stats.update(counts)
    return n_marked
