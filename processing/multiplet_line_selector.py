"""
Intra-multiplet line selection for NMR quantification.

Given an annotation hit with predicted_lines (expected line positions and
relative intensities), this module identifies which individual lines are
"clean" (free from overlap by other metabolites) and which are contaminated.

Two modes:
  - NNLS mode  : uses own_frac from J-Resolve to classify each line.
  - Ratio mode : compares observed heights against expected Pascal ratios
                 using a robust median-anchored deviation ratio (DR).

The output _line_selection dict is attached to each hit before the integration
loop, allowing selective integration of clean lines scaled to full-multiplet
equivalent area.
"""
from __future__ import annotations

import numpy as np
from typing import Any

# ── Thresholds ────────────────────────────────────────────────────────────────
NNLS_CLEAN_THRESHOLD    = 0.70   # own_frac ≥ this → line is clean
NNLS_MARGINAL_THRESHOLD = 0.40   # own_frac in [0.40, 0.70) → marginal (accepted)
RATIO_MAX_DR_UP         = 0.35   # DR > this → positive contamination
RATIO_MAX_DR_DOWN       = 0.30   # DR < -this → suppressed / contaminated
POOR_QUALITY_DR_UP      = 0.20   # tighter DR threshold for gaussian_quality='poor'
MIN_CLEAN_FRACTION      = 0.25   # clean_intensity_fraction below this → skip
EDGE_PPM_MARGIN         = 0.03   # ppm from spectrum edge → warn but don't reject
_SINGLET_RESULT = {
    'mode': 'singlet',
    'clean_ranks': [],
    'contaminated_ranks': [],
    'scale_factor': 1.0,
    'clean_intensity_fraction': 1.0,
    'contamination_report': 'singlet or unresolved — no line selection applied',
}


# ── Public API ────────────────────────────────────────────────────────────────

def select_clean_lines(
    hit: dict,
    pk_by_rank: dict[int, dict],
    j_resolve_contribs: dict[int, dict[str, float]],
    mid_str: str,
    use_nnls: bool,
    ppm_limits: tuple[float, float],
) -> dict:
    """Return a _line_selection dict for *hit*.

    Parameters
    ----------
    hit              : annotation result dict; must have 'predicted_lines',
                       'matched_peaks' (or 'peak_rank'), 'multiplicity'.
    pk_by_rank       : {rank: peak_dict} from the detection stage.
    j_resolve_contribs: {rank: {mid_str: own_frac}} from J-Resolve NNLS.
    mid_str          : string metabolite ID matching keys in j_resolve_contribs.
    use_nnls         : whether J-Resolve has been run.
    ppm_limits       : (ppm_min, ppm_max) of the acquired spectrum.
    """
    mult = (hit.get('multiplicity') or hit.get('type') or 's').strip().lower()
    predicted = hit.get('predicted_lines') or []

    # Singlets, 'm', and hits with ≤1 predicted line need no line selection.
    if mult in ('s', 'm') or len(predicted) <= 1:
        return dict(_SINGLET_RESULT)

    matched_ranks = _collect_matched_ranks(hit, pk_by_rank)
    # If no lines could be matched to detected peaks, skip line selection
    # entirely — the integration will fall back to the full multiplet region.
    if not matched_ranks or all(r is None for r in matched_ranks):
        return dict(_SINGLET_RESULT)

    if use_nnls:
        return _nnls_mode(predicted, matched_ranks, pk_by_rank,
                          j_resolve_contribs, mid_str)
    return _ratio_mode(predicted, matched_ranks, pk_by_rank, ppm_limits)


# ── Mode implementations ──────────────────────────────────────────────────────

def _nnls_mode(
    predicted: list[dict],
    matched_ranks: list[int | None],
    pk_by_rank: dict,
    j_resolve_contribs: dict,
    mid_str: str,
) -> dict:
    clean_ranks: list[int] = []
    contaminated_ranks: list[int] = []
    clean_rel = 0.0
    total_rel = 0.0
    notes: list[str] = []

    for line, rank in zip(predicted, matched_ranks):
        rel = float(line.get('rel_intensity', 1.0))
        total_rel += rel

        if rank is None:
            contaminated_ranks.append(-1)
            notes.append(f"line@{line.get('ppm', '?'):.3f}: no peak — contaminated")
            continue

        pk = pk_by_rank.get(rank, {})
        fracs = j_resolve_contribs.get(rank, {})
        own_frac = fracs.get(mid_str, None)

        if own_frac is None:
            # No NNLS entry: trust gaussian_quality
            gq = pk.get('gaussian_quality', 'good')
            own_frac = 1.0 if gq == 'good' else 0.0

        if own_frac >= NNLS_CLEAN_THRESHOLD:
            clean_ranks.append(rank)
            clean_rel += rel
            notes.append(f"rank{rank}: own_frac={own_frac:.2f} → clean")
        elif own_frac >= NNLS_MARGINAL_THRESHOLD:
            # Marginal: accept but note
            clean_ranks.append(rank)
            clean_rel += rel
            notes.append(f"rank{rank}: own_frac={own_frac:.2f} → marginal (accepted)")
        else:
            contaminated_ranks.append(rank)
            notes.append(f"rank{rank}: own_frac={own_frac:.2f} → contaminated")

    return _build_result('nnls', clean_ranks, contaminated_ranks,
                         total_rel, clean_rel, '; '.join(notes))


def _ratio_mode(
    predicted: list[dict],
    matched_ranks: list[int | None],
    pk_by_rank: dict,
    ppm_limits: tuple[float, float],
) -> dict:
    # Build observed heights and expected rel_intensities for matched lines.
    obs: list[float] = []
    rels: list[float] = []
    ranks_ordered: list[int | None] = []

    for line, rank in zip(predicted, matched_ranks):
        rel = float(line.get('rel_intensity', 1.0))
        rels.append(rel)
        ranks_ordered.append(rank)
        if rank is None:
            obs.append(0.0)
        else:
            pk = pk_by_rank.get(rank, {})
            obs.append(float(pk.get('height', 0.0)))

    obs_arr  = np.array(obs,  dtype=float)
    rels_arr = np.array(rels, dtype=float)

    # Reference height: minimum of (obs/rel) among matched lines.
    # Minimum is the best robust estimator for detecting additive contamination
    # (which always increases obs): contaminated lines have obs/rel >> H_ref,
    # while clean lines cluster near H_ref.  For multiplets with ≥3 lines,
    # median would also work, but minimum handles doublets correctly where
    # median = average and deviations cancel out.
    valid_mask = (matched_ranks_mask(matched_ranks)) & (rels_arr > 0) & (obs_arr > 0)
    if valid_mask.sum() == 0:
        return dict(_SINGLET_RESULT)

    ratios = obs_arr[valid_mask] / rels_arr[valid_mask]
    H_ref  = float(np.min(ratios))
    if H_ref <= 0:
        return dict(_SINGLET_RESULT)

    expected_arr = H_ref * rels_arr

    clean_ranks: list[int] = []
    contaminated_ranks: list[int] = []
    clean_rel = 0.0
    total_rel = float(rels_arr.sum())
    notes: list[str] = []

    for i, (rank, rel, h_obs, h_exp) in enumerate(
            zip(ranks_ordered, rels_arr, obs_arr, expected_arr)):
        if rank is None:
            contaminated_ranks.append(-1)
            notes.append(f"line{i}: no peak — contaminated")
            continue

        pk = pk_by_rank.get(rank, {})
        gq = pk.get('gaussian_quality', 'good')
        pk_ppm = float(pk.get('ppm', 0.0))

        # gaussian_quality == 'overlap' → always contaminated
        if gq == 'overlap':
            contaminated_ranks.append(rank)
            notes.append(f"rank{rank}: gaussian_quality=overlap → contaminated")
            continue

        # Deviation ratio
        if h_exp > 0:
            dr = (h_obs - h_exp) / h_exp
        else:
            dr = 0.0

        # Tighter threshold for poor-quality peaks
        dr_up = POOR_QUALITY_DR_UP if gq == 'poor' else RATIO_MAX_DR_UP

        edge_warn = ''
        if _is_edge_peak(pk_ppm, ppm_limits):
            edge_warn = ' [edge]'

        if dr > dr_up:
            contaminated_ranks.append(rank)
            notes.append(f"rank{rank}: DR={dr:+.2f}>{dr_up:.2f} → contaminated{edge_warn}")
        elif dr < -RATIO_MAX_DR_DOWN:
            contaminated_ranks.append(rank)
            notes.append(f"rank{rank}: DR={dr:+.2f}<-{RATIO_MAX_DR_DOWN:.2f} → contaminated{edge_warn}")
        else:
            clean_ranks.append(rank)
            clean_rel += rel
            notes.append(f"rank{rank}: DR={dr:+.2f} → clean{edge_warn}")

    return _build_result('ratio', clean_ranks, contaminated_ranks,
                         total_rel, clean_rel, '; '.join(notes))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_matched_ranks(hit: dict, pk_by_rank: dict) -> list[int | None]:
    """Map each predicted_line to the nearest detected peak rank.

    Uses greedy matching from highest rel_intensity downward to prevent
    double-assignment when two predicted lines are close together.
    Returns a list parallel to predicted_lines; None where no match found.
    """
    predicted = hit.get('predicted_lines') or []
    if not predicted:
        return []

    # Build candidate set: ranks from matched_peaks if available, else all ranks.
    matched_peaks = hit.get('matched_peaks') or []
    if matched_peaks:
        candidate_ranks = [p.get('rank') for p in matched_peaks
                           if p.get('rank') is not None]
    else:
        # Fallback: use the single peak_rank the hit is anchored on
        pr = hit.get('peak_rank')
        candidate_ranks = [pr] if pr is not None else list(pk_by_rank.keys())

    # Sort predicted lines from largest rel_intensity to smallest (greedy).
    indexed = sorted(enumerate(predicted),
                     key=lambda x: -float(x[1].get('rel_intensity', 1.0)))

    result: list[int | None] = [None] * len(predicted)
    used: set[int] = set()

    for orig_i, line in indexed:
        pred_ppm = float(line.get('ppm', 0.0))
        rel      = float(line.get('rel_intensity', 1.0))
        # Tolerance: half a typical J coupling (≈3.5 Hz at 400 MHz → 0.009 ppm);
        # minimum 0.003 ppm to handle singlet-like lines.
        tol = max(0.003, rel * 0.015)

        best_rank = None
        best_dist = float('inf')
        for r in candidate_ranks:
            if r in used:
                continue
            pk = pk_by_rank.get(r)
            if pk is None:
                continue
            dist = abs(float(pk.get('ppm', 0.0)) - pred_ppm)
            if dist < tol and dist < best_dist:
                best_dist = dist
                best_rank = r

        result[orig_i] = best_rank
        if best_rank is not None:
            used.add(best_rank)

    return result


def matched_ranks_mask(matched_ranks: list[int | None]) -> np.ndarray:
    return np.array([r is not None for r in matched_ranks], dtype=bool)


def _build_result(
    mode: str,
    clean_ranks: list[int],
    contaminated_ranks: list[int],
    total_rel: float,
    clean_rel: float,
    report: str,
) -> dict:
    if total_rel <= 0 or clean_rel <= 0:
        return {
            'mode': 'skip',
            'clean_ranks': [],
            'contaminated_ranks': contaminated_ranks,
            'scale_factor': 1.0,
            'clean_intensity_fraction': 0.0,
            'contamination_report': report + ' | skip: no clean lines',
        }

    cif = clean_rel / total_rel
    if cif < MIN_CLEAN_FRACTION:
        return {
            'mode': 'skip',
            'clean_ranks': clean_ranks,
            'contaminated_ranks': contaminated_ranks,
            'scale_factor': 1.0,
            'clean_intensity_fraction': cif,
            'contamination_report': report + f' | skip: cif={cif:.2f}<{MIN_CLEAN_FRACTION}',
        }

    return {
        'mode': mode,
        'clean_ranks': clean_ranks,
        'contaminated_ranks': contaminated_ranks,
        'scale_factor': total_rel / clean_rel,
        'clean_intensity_fraction': cif,
        'contamination_report': report,
    }


def _is_edge_peak(ppm: float, ppm_limits: tuple[float, float]) -> bool:
    lo, hi = min(ppm_limits), max(ppm_limits)
    return ppm < lo + EDGE_PPM_MARGIN or ppm > hi - EDGE_PPM_MARGIN
