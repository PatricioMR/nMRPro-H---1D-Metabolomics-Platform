# gui/notation/multiplet_scorer.py
"""
Multiplet Pattern Fitting for NMR Annotation.

Core principle
--------------
J-coupling (Hz) is field-independent — it is a molecular property.
The ppm separation between lines of a multiplet depends on the spectrometer:

    Δppm = J_Hz / spectrometer_MHz

Algorithm
---------
For each DB candidate (with known multiplicity and J):

1.  PREDICT expected line pattern around the DB centre (in Hz-space).

2.  SEARCH experimental peaks:
    - For a doublet  (2 lines, J known):
        Find all pairs of detected peaks where:
          |distance_ppm × mhz - J_hz| < J_tolerance_hz  (default 10% of J)
          AND relative intensities ≈ 1:1
        The best pair is the doublet. Centre = midpoint.

    - For a triplet (3 lines):
        Find all triplets where consecutive spacings ≈ J,
        relative intensities ≈ 1:2:1.

    - For dd, ddd, etc.:
        Enumerate candidate centres, score each.

3.  SCORE = position_score × 0.5 + intensity_score × 0.3 + coverage × 0.2

4.  Return centre, predicted_lines (at experimental positions), score.
"""

import numpy as np
from processing.signal_quality import SIGNAL_AVERAGE
import re as _re
from itertools import combinations
from scipy.optimize import linear_sum_assignment


# ── Multiplet patterns ────────────────────────────────────────────────────────

_SIMPLE = {
    's':    ([1],             0),
    'd':    ([1, 1],          1),
    't':    ([1, 2, 1],       1),
    'q':    ([1, 3, 3, 1],    1),
    'quin': ([1, 4, 6, 4, 1], 1),
    'sex':  ([1, 5, 10, 10, 5, 1], 1),
    'sep':  ([1, 6, 15, 20, 15, 6, 1], 1),
}
_COMPOUND = {
    'dd': 2, 'ddd': 3, 'dt': 2, 'td': 2, 'dq': 2,
}
_ALIAS = {
    'singlet': 's', 'doublet': 'd', 'triplet': 't',
    'quartet': 'q', 'quintet': 'quin',
    'doublet of doublets': 'dd',
    'doublet of triplets': 'dt',
    'triplet of doublets': 'td',
    'doublet of quartets': 'dq',
    'multiplet': 'm', 'broad': 's', 'broad singlet': 's',
}


def _norm(mult):
    m = str(mult or 'm').strip().lower()
    return _ALIAS.get(m, m)


def _parse_j(j_str):
    nums = _re.findall(r'\d+\.?\d*', str(j_str or ''))
    return [float(x) for x in nums if float(x) > 0]


def _offsets_from_pattern(norm, j_values_hz, mhz):
    """
    Returns list of (offset_ppm_from_centre, rel_intensity).
    Centre is intensity-weighted mean = 0 by definition.
    J values are in Hz; conversion: offset_ppm = J_hz / mhz.
    """
    if norm in _SIMPLE:
        ints, n_j = _SIMPLE[norm]
        n = len(ints)
        if n == 1:
            return [(0.0, 1.0)]
        if not j_values_hz:
            return []
        j  = j_values_hz[0]
        jp = j / mhz
        positions = [(k - (n - 1) / 2) * jp for k in range(n)]
        max_i = max(ints)
        return [(pos, i / max_i) for pos, i in zip(positions, ints)]

    # Compound multiplets
    n_j   = _COMPOUND.get(norm, 1)
    if not j_values_hz or len(j_values_hz) < n_j:
        return []
    j_list = j_values_hz[:n_j]
    lines  = [(0.0, 1.0)]
    for jv in j_list:
        jp = jv / mhz
        new = []
        for pos, rel in lines:
            new.append((pos + jp / 2, rel))
            new.append((pos - jp / 2, rel))
        # Merge nearly identical positions
        new.sort(key=lambda x: x[0])
        merged = []
        for p, r in new:
            if merged and abs(p - merged[-1][0]) < 1e-6:
                merged[-1] = (merged[-1][0], merged[-1][1] + r)
            else:
                merged.append([p, r])
        lines = merged

    # Recentre: weighted mean → 0
    tot   = sum(r for _, r in lines)
    ctr   = sum(p * r for p, r in lines) / tot if tot else 0
    lines = [(p - ctr, r) for p, r in lines]
    max_r = max(r for _, r in lines)
    return [(p, r / max_r) for p, r in lines]


# ── Peak-pair J matching ──────────────────────────────────────────────────────

def _get_corrected_height(ppm_val, ppm_arr, int_arr, half_win):
    """
    Local linear baseline-corrected height for a peak at ppm_val.

    Anchors the baseline at the two edges of [ppm_val - half_win,
    ppm_val + half_win], interpolates linearly, and returns:
        h_corrected = spectrum_at_ppm_val - baseline_at_ppm_val

    The half_win should be set so the anchor points fall outside the
    signal cluster.  Peaks sitting on the Lorentzian tail of a large
    neighbour will have a near-zero or negative corrected height (clipped
    to 0), allowing the SNR filter to reject them.
    """
    lo = ppm_val - half_win
    hi = ppm_val + half_win
    i_lo = int(np.argmin(np.abs(ppm_arr - lo)))
    i_hi = int(np.argmin(np.abs(ppm_arr - hi)))
    i_pk = int(np.argmin(np.abs(ppm_arr - ppm_val)))
    if i_lo == i_hi:
        return None
    x_lo, y_lo = float(ppm_arr[i_lo]), float(int_arr[i_lo])
    x_hi, y_hi = float(ppm_arr[i_hi]), float(int_arr[i_hi])
    frac     = (ppm_val - x_lo) / (x_hi - x_lo) if x_hi != x_lo else 0.5
    baseline = y_lo + frac * (y_hi - y_lo)
    return max(0.0, float(int_arr[i_pk]) - baseline)


def _find_doublet_pair(peaks, j_hz, mhz, ppm_exp, tolerance,
                       j_tolerance_frac=0.30,
                       ppm_array=None, intensity_array=None):
    """
    Find the best doublet pair near ppm_exp using a bottom-up flow:

      1. Collect all peaks within ±(20 Hz / mhz + tolerance) of ppm_exp.
         This window always captures both lines of any doublet with J ≤ 20 Hz,
         regardless of whether ppm_exp is one of the lines or the centre.

      2. Minimum-height filter — "bom SNR":
         Both peaks must have height ≥ MIN_H_FRAC × (tallest candidate).
         When spectrum arrays are provided, uses local linear baseline-corrected
         heights instead of raw heights, eliminating inflated tails from
         adjacent large peaks.

      3. Evaluate every remaining pair:
         - j_hz > 0  → hard J filter: |J_obs - J_db| ≤ j_tolerance_frac × J_db
         - j_hz == 0 → soft J score: prefer 5–12 Hz (typical NMR doublets);
                       J outside 1–22 Hz is rejected outright.

      4. Score = J_match × 0.50 + intensity_ratio × 0.35 + position × 0.15

    Returns (centre_ppm, peak1, peak2, score) or None.
    """
    if not peaks:
        return None

    # Fixed wide window: reaches any doublet partner (J ≤ 20 Hz) from this anchor.
    search_win = 20.0 / mhz + tolerance
    candidates = [p for p in peaks if abs(p['ppm'] - ppm_exp) <= search_win]
    if len(candidates) < 2:
        return None

    # ── Baseline-corrected heights ─────────────────────────────────────────────
    # When spectrum arrays are available, replace raw peak heights with local
    # linear baseline-corrected values.  This prevents the Lorentzian tail of a
    # neighbouring large peak from inflating the apparent height of a noise/
    # baseline point and causing it to be selected as a doublet partner.
    baseline_win = search_win * 1.2   # half-window for baseline estimation
    corr_heights = {}
    if ppm_array is not None and intensity_array is not None:
        ppm_arr = np.asarray(ppm_array)
        int_arr = np.asarray(intensity_array)
        for p in candidates:
            ch = _get_corrected_height(p['ppm'], ppm_arr, int_arr, baseline_win)
            corr_heights[id(p)] = ch if ch is not None else p.get('height', 0)
    else:
        for p in candidates:
            corr_heights[id(p)] = p.get('height', 0)

    # ── Minimum-height filter ("bom SNR") ─────────────────────────────────────
    max_h_cands = max(corr_heights[id(p)] for p in candidates)
    min_h       = max_h_cands * 0.20        # 20 % of the tallest nearby peak
    candidates  = [p for p in candidates if corr_heights[id(p)] >= min_h]
    if len(candidates) < 2:
        return None

    best       = None
    best_score = -1.0

    for p1, p2 in combinations(candidates, 2):
        sep   = abs(p1['ppm'] - p2['ppm'])
        j_obs = sep * mhz

        if j_obs < 0.4 or j_obs > 30.0:
            continue

        if j_hz > 0:
            expected_sep = j_hz / mhz
            j_tol        = max(expected_sep * j_tolerance_frac, 0.001)
            if abs(sep - expected_sep) > j_tol:
                continue
            j_sc = max(0.0, 1.0 - abs(sep - expected_sep) / j_tol)
        else:
            # DB J unknown → soft score favouring 5–12 Hz range
            if 5.0 <= j_obs <= 12.0:
                j_sc = 1.0
            elif j_obs < 5.0:
                j_sc = j_obs / 5.0
            else:
                j_sc = max(0.0, 1.0 - (j_obs - 12.0) / 10.0)

        # Intensity ratio (1:1 expected for doublet) using corrected heights
        h1 = corr_heights[id(p1)]
        h2 = corr_heights[id(p2)]
        max_h = max(h1, h2)
        if max_h <= 0:
            continue
        ratio  = min(h1, h2) / max_h
        int_sc = max(0.0, ratio - 0.40) / 0.60 if ratio < 1.0 else 1.0

        # Position: doublet centre near the anchor (tie-breaker)
        centre = (p1['ppm'] + p2['ppm']) / 2
        pos_sc = max(0.0, 1.0 - abs(centre - ppm_exp) / (tolerance + search_win))

        score = j_sc * 0.50 + int_sc * 0.35 + pos_sc * 0.15

        if score > best_score:
            best_score = score
            best       = (centre, p1, p2, score)

    return best


def _find_triplet(peaks, j_hz, mhz, ppm_exp, tolerance,
                  j_tolerance_frac=0.25):
    """
    For a triplet: find 3 peaks with spacing ≈ J and intensities ≈ 1:2:1.
    Returns (centre_ppm, [p1,p2,p3], score) or None.
    """
    if not peaks or j_hz <= 0:
        return None

    expected_sep = j_hz / mhz
    j_tol        = max(expected_sep * j_tolerance_frac, 0.001)
    search_win   = expected_sep * 2 + tolerance * 2

    raw_cands = [p for p in peaks if abs(p['ppm'] - ppm_exp) <= search_win * 1.5]
    # SNR filter: require ≥ 15% of tallest nearby peak (triplet outer lines can be
    # weaker due to the 1:2:1 ratio, so threshold is slightly lower than for doublets)
    if raw_cands:
        max_h_t = max(p.get('height', 0) for p in raw_cands)
        raw_cands = [p for p in raw_cands if p.get('height', 0) >= max_h_t * 0.15]
    candidates = sorted(raw_cands, key=lambda p: p['ppm'])

    best = None
    best_score = -1.0

    for p1, p2, p3 in combinations(candidates, 3):
        ppms = sorted([p1['ppm'], p2['ppm'], p3['ppm']])
        sep1 = ppms[1] - ppms[0]
        sep2 = ppms[2] - ppms[1]
        if abs(sep1 - expected_sep) > j_tol:
            continue
        if abs(sep2 - expected_sep) > j_tol:
            continue
        if abs(sep1 - sep2) > j_tol:
            continue

        # Intensities 1:2:1
        pks = sorted([p1, p2, p3], key=lambda p: p['ppm'])
        h   = [pk['height'] for pk in pks]
        max_h = max(h)
        if max_h <= 0: continue
        r   = [hi / max_h for hi in h]
        # Expected: [0.5, 1.0, 0.5]
        int_sc = max(0.0, 1.0 - (
            abs(r[0] - 0.5) + abs(r[1] - 1.0) + abs(r[2] - 0.5)) / 3)

        j_sc    = max(0.0, 1.0 - (abs(sep1 - expected_sep) +
                                    abs(sep2 - expected_sep)) / (2 * j_tol))
        centre  = ppms[1]  # central line
        pos_sc  = max(0.0, 1.0 - abs(centre - ppm_exp) / (tolerance + search_win))
        score   = j_sc * 0.5 + int_sc * 0.35 + pos_sc * 0.15

        if score > best_score:
            best_score = score
            best = (centre, pks, score)

    return best


# ── General candidate-scan scorer ─────────────────────────────────────────────

def _scan_candidates(ppm_exp, offsets, peaks, tolerance, mhz):
    """
    For compound multiplets (dd, ddd…): scan all candidate centres
    (individual peaks + midpoints between pairs) and score each.
    Returns (best_centre, matched_list, best_score).
    """
    if not offsets or not peaks:
        return ppm_exp, [], 0.0

    max_off  = max(abs(o) for o, _ in offsets)
    window   = max_off * 2 + tolerance * 2

    nearby = [p for p in peaks if abs(p['ppm'] - ppm_exp) <= window * 1.5]
    if not nearby:
        return ppm_exp, [], 0.0

    centres = {round(p['ppm'], 5) for p in nearby}
    for p1, p2 in combinations(nearby, 2):
        centres.add(round((p1['ppm'] + p2['ppm']) / 2, 5))
    centres.add(round(ppm_exp, 5))

    best_centre, best_score, best_matched = ppm_exp, -1.0, []
    n_lines = len(offsets)

    for ctr in centres:
        if abs(ctr - ppm_exp) > window * 1.5:
            continue

        predicted = [(ctr + off, rel) for off, rel in offsets]
        ppms_obs  = [p['ppm']    for p in nearby]
        heights   = [p['height'] for p in nearby]
        max_h     = max(heights) if heights else 1.0

        spacing  = (2 * max_off / (n_lines - 1)) if n_lines > 1 else max_off
        eff_tol  = max(0.0005, min(tolerance, spacing * 0.55))

        matched = _hungarian_line_matches(
            predicted, nearby, max_distance=eff_tol, max_height=max_h)

        if not matched:
            continue

        tot_w    = sum(m['pred_rel'] for m in matched)
        coverage = len(matched) / n_lines
        pos_s    = sum(max(0.0, 1.0 - m['distance'] / max(m['eff_tol'], 1e-9))
                       * m['pred_rel'] for m in matched) / tot_w
        int_s    = sum(max(0.0, 1.0 - abs(m['pred_rel'] - m['obs_rel']))
                       * m['pred_rel'] for m in matched) / tot_w
        score    = pos_s * 0.5 + int_s * 0.3 + coverage * 0.2

        if score > best_score:
            best_score, best_centre, best_matched = score, ctr, matched

    return best_centre, best_matched, max(0.0, best_score)


# ── Public API ────────────────────────────────────────────────────────────────

def _score_from_transitions(db_trans, result, peaks, ppm_c, ppm_exp,
                             norm, tolerance, mhz):
    """
    Score using actual transition line positions from the DB.

    The DB transitions are at reference conditions (pH, temp, field).
    We shift them to match the experimental spectrum by computing the
    offset between the DB centre and the detected experimental peak.

    Steps:
    1. Compute DB centre = intensity-weighted mean of transition ppms
    2. Find best offset: scan candidate centres from nearby experimental peaks
    3. Apply offset to all transitions → shifted_lines
    4. Score: for each line, find nearest experimental peak
    """
    t_ppms  = np.array([t['ppm']       for t in db_trans])
    t_ints  = np.array([t['intensity'] for t in db_trans])

    # Normalise intensities
    max_i = t_ints.max() if t_ints.max() > 0 else 1.0
    t_rels = t_ints / max_i

    # DB centre (intensity-weighted)
    db_centre = float(np.average(t_ppms, weights=t_ints))

    # Span of the multiplet pattern
    span = float(t_ppms.max() - t_ppms.min())
    search_win = max(span * 0.8, tolerance) * 2

    # Candidate centres from experimental peaks nearby
    nearby = [p for p in peaks
              if abs(p['ppm'] - ppm_exp) <= search_win + span]
    if not nearby:
        # No experimental peaks found → return low score, DB-position lines
        lines = [{'ppm': ppm_exp + (p - db_centre), 'rel_intensity': float(r)}
                 for p, r in zip(t_ppms, t_rels)]
        return 0.0, f'{norm.upper()} no nearby peaks', lines

    # Build candidate centres: individual peaks + midpoints
    cand_centres = {round(p['ppm'], 5) for p in nearby}
    for p1, p2 in combinations(nearby, 2):
        cand_centres.add(round((p1['ppm'] + p2['ppm']) / 2, 5))

    best_score   = -1.0
    best_centre  = ppm_exp
    best_lines   = []
    best_matched = []

    n_lines  = len(t_ppms)
    eff_tol  = max(span / n_lines * 0.6, 0.003) if n_lines > 1 else tolerance

    for ctr in cand_centres:
        if abs(ctr - ppm_exp) > search_win * 1.5:
            continue
        offset     = ctr - db_centre
        pred_ppms  = t_ppms + offset
        ppms_obs   = [p['ppm']    for p in nearby]
        heights    = [p['height'] for p in nearby]
        max_h      = max(heights) if heights else 1.0

        raw_matches = _hungarian_line_matches(
            list(zip(pred_ppms, t_rels)), nearby,
            max_distance=eff_tol * 2, max_height=max_h)
        matched = []
        for m in raw_matches:
            matched.append({
                'pos_sc': max(0.0, 1.0 - m['distance'] / max(eff_tol * 2, 1e-9)),
                'int_sc': max(0.0, 1.0 - abs(m['pred_rel'] - m['obs_rel'])),
                'weight': m['pred_rel'],
                'obs_ppm': m['obs_ppm'],
                'pred_ppm': m['pred_ppm'],
            })

        if not matched:
            continue
        tot_w    = sum(m['weight'] for m in matched)
        coverage = len(matched) / n_lines
        pos_s    = sum(m['pos_sc'] * m['weight'] for m in matched) / tot_w
        int_s    = sum(m['int_sc'] * m['weight'] for m in matched) / tot_w
        score    = pos_s * 0.5 + int_s * 0.3 + coverage * 0.2

        if score > best_score:
            best_score   = score
            best_centre  = ctr
            best_matched = matched

    # Build best_lines from experimentally matched positions only
    if best_matched:
        best_lines = [{'ppm': m['obs_ppm'], 'rel_intensity': m['weight']}
                      for m in sorted(best_matched, key=lambda x: x['obs_ppm'])]
    else:
        best_lines = [{'ppm': ppm_exp + (p - db_centre), 'rel_intensity': float(r)}
                      for p, r in zip(t_ppms, t_rels)]

    # Compute observed J from matched lines
    if len(best_lines) >= 2:
        sorted_l   = sorted(best_lines, key=lambda x: x['ppm'])
        seps       = [sorted_l[i+1]['ppm'] - sorted_l[i]['ppm']
                      for i in range(len(sorted_l)-1)]
        j_obs_hz   = sum(seps) / len(seps) * mhz
        j_db_hz    = span / max(n_lines - 1, 1) * mhz
    else:
        j_obs_hz = j_db_hz = 0.0

    n_matched = len(best_matched)
    delta  = best_centre - ppm_c
    detail = (f'{norm.upper()} ({n_matched}/{n_lines} lines from DB transitions)'
              f'  J_obs={j_obs_hz:.2f}Hz  J_db={j_db_hz:.2f}Hz'
              f'  centre={best_centre:.4f}  Δ={delta:+.4f}'
              f'  score={best_score:.2f}')
    return float(max(0.0, best_score)), detail, best_lines


def _hungarian_line_matches(predicted, observed, max_distance, max_height=None):
    """Match predicted and observed lines one-to-one with Hungarian assignment."""
    if not predicted or not observed or max_distance <= 0:
        return []
    obs_ppm = np.asarray([float(p["ppm"]) for p in observed], dtype=float)
    heights = np.asarray([float(p.get("height", 0.0)) for p in observed], dtype=float)
    scale = float(max_height if max_height is not None else (heights.max() or 1.0))
    if scale <= 0:
        scale = 1.0
    pred_ppm = np.asarray([float(p[0]) for p in predicted], dtype=float)
    pred_rel = np.asarray([float(p[1]) for p in predicted], dtype=float)
    dist = np.abs(pred_ppm[:, None] - obs_ppm[None, :])
    obs_rel = heights / scale
    cost = dist / max_distance + 0.35 * np.abs(pred_rel[:, None] - obs_rel[None, :])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(rows, cols):
        d = float(dist[i, j])
        if d >= max_distance:
            continue
        matches.append({
            "obs_ppm": float(obs_ppm[j]),
            "obs_rel": float(obs_rel[j]),
            "pred_ppm": float(pred_ppm[i]),
            "pred_rel": float(pred_rel[i]),
            "distance": d,
            "eff_tol": max_distance,
            "peak": observed[j],
        })
    return matches


def score_multiplet_match(result, peaks, tolerance=0.005,
                          spectrometer_mhz=None,
                          ppm_array=None, intensity_array=None):
    """
    Score an annotation result against detected peaks.

    Parameters
    ----------
    result          : dict from annotation stage
    peaks           : list of detected peak dicts
    tolerance       : ppm tolerance for DB search window
    spectrometer_mhz: field strength (Hz→ppm conversion)
    ppm_array       : full spectrum ppm axis (ndarray), for baseline correction
    intensity_array : full spectrum intensity axis (ndarray), for baseline correction

    Returns (score 0-1, detail_str, predicted_lines_list)
    """
    mhz     = float(spectrometer_mhz or 600.0)
    ppm_c   = float(result.get('ppm_center') or result.get('ppm') or 0)
    ppm_exp = float(result.get('peak_ppm') or ppm_c)
    mult    = result.get('multiplicity') or result.get('type') or 'm'
    j_str   = result.get('j_coupling', '')

    norm   = _norm(mult)
    j_vals = _parse_j(j_str)

    # ── Use DB transitions when available (no j_coupling needed) ────────────
    db_trans = result.get('db_transitions', [])
    # 'm*' is an unresolved multiplet whose pattern was measured; it arrives
    # here only with that pattern as its transitions, and nothing else about
    # it can be scored.
    if db_trans and len(db_trans) >= 2 and norm not in ('s', 'm', ''):
        return _score_from_transitions(db_trans, result, peaks, ppm_c, ppm_exp,
                                        norm, tolerance, mhz)

    # ── Singlet (only true singlets and broad multiplets) ────────────────────
    if norm in ('s', 'm', ''):
        nearby = [p for p in peaks if abs(p['ppm'] - ppm_exp) <= tolerance]
        if not nearby:
            return 0.0, f'{norm.upper()} @ {ppm_exp:.4f} — no peak', []

        # Choose the closest detected peak as the match
        best_pk = min(nearby, key=lambda p: abs(p['ppm'] - ppm_exp))
        exp_ppm  = best_pk['ppm']

        # PPM distance score (Gaussian)
        ps = max(0.0, 1.0 - abs(exp_ppm - ppm_c) / max(tolerance, 1e-9))

        # Singlet purity: penalise if multiple peaks crowd the window.
        # A true singlet should be alone in its PPM neighbourhood.
        wider = [p for p in peaks if abs(p['ppm'] - exp_ppm) <= tolerance * 2
                 and p is not best_pk]
        if wider:
            max_ratio = max(p.get('height', 0) for p in wider) \
                        / max(best_pk.get('height', 1e-9), 1e-9)
            purity = max(0.40, 1.0 - 0.60 * min(max_ratio, 1.0))
        else:
            purity = 1.0

        score = ps * purity
        detail = (f'{norm.upper()} @ {exp_ppm:.4f}'
                  f'{"" if purity == 1.0 else f"  purity={purity:.2f}"}')
        lines = [{'ppm': exp_ppm, 'rel_intensity': 1.0}]
        return float(score), detail, lines

    # ── Doublet ───────────────────────────────────────────────────────────────
    if norm == 'd':
        # j_db = 0 means J is unknown; _find_doublet_pair uses soft scoring then.
        if not j_vals:
            return 0.0, 'D - J ausente; confirmacao manual necessaria', []
        j_db = j_vals[0]

        # Search from both anchors:
        #   ppm_exp = detected peak closest to DB ppm (typically ONE doublet line)
        #   ppm_c   = DB reference centre (true target doublet centre)
        # The wide window inside _find_doublet_pair reaches both lines from either anchor.
        best_pair_result = None
        best_pair_score  = -1.0
        for anchor in dict.fromkeys([ppm_exp, ppm_c]):   # unique, order preserved
            rp = _find_doublet_pair(peaks, j_db, mhz, anchor, tolerance,
                                    ppm_array=ppm_array,
                                    intensity_array=intensity_array)
            if rp is not None and rp[3] > best_pair_score:
                best_pair_score  = rp[3]
                best_pair_result = rp

        if best_pair_result is not None:
            centre, p1, p2, score = best_pair_result
            ppm_lo = min(p1['ppm'], p2['ppm'])
            ppm_hi = max(p1['ppm'], p2['ppm'])
            lines  = [
                {'ppm': ppm_lo, 'rel_intensity': 1.0},
                {'ppm': ppm_hi, 'rel_intensity': 1.0},
            ]
            sep_hz   = (ppm_hi - ppm_lo) * mhz
            delta    = centre - ppm_c
            j_db_str = f'  J_db={j_vals[0]:.2f} Hz' if j_vals else ''
            detail   = (f'D  J_obs={sep_hz:.2f} Hz{j_db_str}'
                        f'  centre={centre:.4f}  Δ={delta:+.4f}'
                        f'  score={score:.2f}')
            return float(score), detail, lines

        # Pair not found — model centred on ppm_c (DB centre), not ppm_exp.
        j_model  = j_vals[0]
        half_sep = j_model / (2.0 * mhz)
        offsets_d = [(-half_sep, 1.0), (half_sep, 1.0)]
        centre_d, _, score_d = _scan_candidates(
            ppm_c, offsets_d, peaks, tolerance, mhz)
        delta_d  = centre_d - ppm_c
        lines_d  = [{'ppm': centre_d + off, 'rel_intensity': rel}
                    for off, rel in offsets_d]
        j_db_str = f'  J_db={j_vals[0]:.2f} Hz' if j_vals else ''
        detail_d = (f'D (model){j_db_str}  J_used={j_model:.1f} Hz'
                    f'  centre={centre_d:.4f}  Δ={delta_d:+.4f}'
                    f'  score={score_d:.2f}')
        return float(score_d), detail_d, lines_d

    # ── Triplet ───────────────────────────────────────────────────────────────
    if norm == 't':
        if not j_vals:
            return 0.0, 'T - J ausente; confirmacao manual necessaria', []
        j_t = j_vals[0]
        best_t       = None
        best_t_score = -1.0
        for anchor in dict.fromkeys([ppm_exp, ppm_c]):
            rt = _find_triplet(peaks, j_t, mhz, anchor, tolerance)
            if rt is not None and rt[2] > best_t_score:
                best_t_score = rt[2]
                best_t       = rt
        if best_t is not None:
            centre, pks, score = best_t
            sorted_pks = sorted(pks, key=lambda p: p['ppm'])
            max_h = max(p['height'] for p in sorted_pks) or 1.0
            lines = [{'ppm': p['ppm'],
                      'rel_intensity': p['height'] / max_h}
                     for p in sorted_pks]
            sep_hz = abs(sorted_pks[1]['ppm'] - sorted_pks[0]['ppm']) * mhz
            delta  = centre - ppm_c
            j_db_str = f'  J_db={j_vals[0]:.2f} Hz' if j_vals else ''
            detail = (f'T  J_obs={sep_hz:.2f} Hz{j_db_str}'
                      f'  centre={centre:.4f}  Δ={delta:+.4f}  score={score:.2f}')
            return float(score), detail, lines
        # Triplet not found — model centred on ppm_c
        j_model  = j_vals[0]
        jp       = j_model / mhz
        offsets_t = [(-jp, 0.5), (0.0, 1.0), (jp, 0.5)]
        centre_t, _, score_t = _scan_candidates(
            ppm_c, offsets_t, peaks, tolerance, mhz)
        lines_t = [{'ppm': centre_t + off, 'rel_intensity': rel}
                   for off, rel in offsets_t]
        delta_t  = centre_t - ppm_c
        j_db_str = f'  J_db={j_vals[0]:.2f} Hz' if j_vals else ''
        detail_t = (f'T (model){j_db_str}  J_used={j_model:.1f} Hz'
                    f'  centre={centre_t:.4f}  Δ={delta_t:+.4f}'
                    f'  score={score_t:.2f}')
        return float(score_t), detail_t, lines_t

    # ── General: candidate scan (dd, ddd, q, etc.) ────────────────────────────
    # Use ppm_c (DB centre) as the primary anchor — more accurate than ppm_exp
    # which is a single detected line that may not be the multiplet centre.
    offsets = _offsets_from_pattern(norm, j_vals, mhz)
    centre, matched, score = _scan_candidates(
        ppm_c, offsets, peaks, tolerance, mhz)

    # Only include experimentally matched lines so the fingerprint visualization
    # accurately reflects which lines were actually detected.
    if matched:
        lines = [{'ppm': m['obs_ppm'], 'rel_intensity': m['pred_rel']}
                 for m in sorted(matched, key=lambda x: x['obs_ppm'])]
    else:
        lines = [{'ppm': centre + off, 'rel_intensity': rel}
                 for off, rel in offsets]

    j_desc  = ', '.join(f'{j:.1f}' for j in j_vals)
    n_lines = len(offsets)
    delta   = centre - ppm_c
    detail  = (f'{norm.upper()} ({n_lines} lines)'
               f'  J={j_desc} Hz'
               f'  centre={centre:.4f}  Δ={delta:+.4f}'
               f'  {len(matched)}/{n_lines} matched'
               f'  score={score:.2f}')
    return float(score), detail, lines



# ── Observed multiplicity classification ─────────────────────────────────────

# Compatibility table: (expected_norm, observed_norm) → score 0–1
# Read as "we expected X, but we saw Y → how compatible are they?"
_COMPAT: dict[tuple[str, str], float] = {
    # Perfect identity
    ('s',    's'):   1.00,
    ('d',    'd'):   1.00,
    ('t',    't'):   1.00,
    ('q',    'q'):   1.00,
    ('quin', 'quin'):1.00,
    ('dd',   'dd'):  1.00,
    ('ddd',  'ddd'): 1.00,
    ('dt',   'dt'):  1.00,
    ('td',   'td'):  1.00,

    # dd / dt / td can look like d if one J is very small / unresolved
    ('dd',  'd'):   0.50,
    ('ddd', 'd'):   0.35,
    ('ddd', 'dd'):  0.55,
    ('dt',  'd'):   0.40,
    ('dt',  't'):   0.55,
    ('td',  'd'):   0.40,
    ('td',  't'):   0.55,

    # Hard mismatches — strong but not absolute penalty
    ('s',  'd'):   0.12,
    ('s',  't'):   0.05,
    ('s',  'q'):   0.04,
    ('s',  'dd'):  0.08,
    ('s',  'm'):   0.50,   # "broad" singlet in DB may look multiplet
    ('d',  's'):   0.18,   # doublet partially unresolved can look singlet
    ('d',  't'):   0.10,
    ('d',  'q'):   0.06,
    ('d',  'm'):   0.60,
    ('t',  's'):   0.08,
    ('t',  'd'):   0.12,
    ('t',  'q'):   0.15,
    ('t',  'm'):   0.65,
    ('q',  's'):   0.05,
    ('q',  'd'):   0.10,
    ('q',  't'):   0.15,
    ('q',  'm'):   0.65,

    # expected multiplet (m) → very flexible
    ('m',  's'):   0.65,
    ('m',  'd'):   0.70,
    ('m',  't'):   0.75,
    ('m',  'q'):   0.75,
    ('m',  'dd'):  0.80,
    ('m',  'm'):   0.85,
}


def classify_observed_multiplicity(peaks_in_window: list,
                                   j_hz: float = 0.0,
                                   mhz: float = 600.0
                                   ) -> tuple[str, float]:
    """
    Given detected peaks that fall within the search window of a DB candidate,
    classify what multiplicity pattern is actually observed.

    Parameters
    ----------
    peaks_in_window : list of peak dicts with 'ppm' and 'height'
    j_hz            : expected J coupling in Hz (0 = unknown)
    mhz             : spectrometer frequency (for Hz→ppm conversion)

    Returns
    -------
    (observed_mult: str, confidence: float 0–1)
    observed_mult is a canonical multiplicity string ('s', 'd', 't', 'q', 'dd', 'm', '?')
    confidence reflects how certain we are about the classification.
    '?' means not enough information.
    """
    n = len(peaks_in_window)
    if n == 0:
        return '?', 0.0
    if n == 1:
        return 's', 1.0

    pks = sorted(peaks_in_window, key=lambda p: p['ppm'])
    heights = [p.get('height', 1.0) for p in pks]
    max_h = max(heights) if max(heights) > 0 else 1.0
    r = [h / max_h for h in heights]          # normalised intensities

    if n == 2:
        sep_ppm = pks[1]['ppm'] - pks[0]['ppm']
        # Intensity ratio for a doublet should be ~1:1
        int_conf = max(0.0, 1.0 - abs(r[0] - r[1]))
        if j_hz > 0:
            expected_sep = j_hz / mhz
            j_match = max(0.0, 1.0 - abs(sep_ppm - expected_sep)
                          / max(expected_sep * 0.30, 0.001))
            conf = j_match * 0.60 + int_conf * 0.40
        else:
            # No J known: 2 peaks of similar height → likely doublet
            conf = 0.45 + int_conf * 0.40
        return ('d', min(conf, 1.0)) if conf > 0.30 else ('m', 0.40)

    if n == 3:
        ppms = [p['ppm'] for p in pks]
        sep1 = ppms[1] - ppms[0]
        sep2 = ppms[2] - ppms[1]
        # Equal spacing → triplet
        spacing_sym = max(0.0, 1.0 - abs(sep1 - sep2) / max(sep1, sep2, 1e-6))
        # Intensity 1:2:1 → r ≈ [0.5, 1.0, 0.5]
        int_sc = max(0.0, 1.0 - (abs(r[0] - 0.5) + abs(r[1] - 1.0)
                                  + abs(r[2] - 0.5)) / 3.0)
        if j_hz > 0:
            expected_sep = j_hz / mhz
            j_match = max(0.0, 1.0 - abs((sep1 + sep2) / 2 - expected_sep)
                          / max(expected_sep * 0.30, 0.001))
            conf = j_match * 0.50 + int_sc * 0.30 + spacing_sym * 0.20
        else:
            conf = spacing_sym * 0.50 + int_sc * 0.50
        return ('t', min(conf, 1.0)) if conf > 0.35 else ('m', 0.50)

    if n == 4:
        # Quartet (1:3:3:1) or dd (1:1:1:1) or other
        # Quartet: outer ≈ 0.33, inner ≈ 1.0
        q_sc = max(0.0, 1.0 - (abs(r[0] - 0.33) + abs(r[1] - 1.00)
                                 + abs(r[2] - 1.00) + abs(r[3] - 0.33)) / 4.0)
        # dd: all lines roughly equal
        dd_sc = max(0.0, 1.0 - sum(abs(ri - 1.0) for ri in r) / 4.0)
        if q_sc > dd_sc and q_sc > 0.35:
            return 'q', q_sc
        if dd_sc > 0.40:
            return 'dd', dd_sc
        return 'm', 0.55

    # 5+ peaks → multiplet
    return 'm', 0.80


def multiplicity_compat_score(expected_mult: str,
                              observed_mult: str,
                              observed_confidence: float = 1.0) -> float:
    """
    Return a 0–1 compatibility score between the DB-expected multiplicity
    and the multiplicity actually observed in the experimental spectrum.

    A score of 1.0 means "perfectly compatible" (no penalty).
    A score near 0 means "strong mismatch — very unlikely to be this metabolite".

    observed_confidence blends the penalty towards neutral (1.0) when the
    observation is uncertain, avoiding false rejects on noisy/overlapping signals.
    """
    if observed_mult in ('?', None) or observed_confidence < 0.20:
        return 1.0     # not enough information — do not penalise

    exp_n = _norm(expected_mult)
    obs_n = _norm(observed_mult)

    if obs_n == 'm':
        # We observed a generic multiplet — be lenient
        base = _COMPAT.get(('m', 'm'), 0.75)
    elif exp_n == 'm':
        # DB entry says "multiplet" — it can look like anything
        base = _COMPAT.get(('m', obs_n), 0.75)
    else:
        base = _COMPAT.get((exp_n, obs_n), 1.0 if exp_n == obs_n else 0.35)

    # Blend towards neutral proportionally to (1 - observed_confidence)
    # Low confidence → closer to 1.0 → don't trust the observation
    return float(np.clip(base * observed_confidence
                         + 1.0 * (1.0 - observed_confidence), 0.0, 1.0))

