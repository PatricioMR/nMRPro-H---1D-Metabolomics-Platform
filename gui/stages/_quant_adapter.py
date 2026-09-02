# gui/stages/_quant_adapter.py
"""Feeding the new quantification engine from what the stages already hold.

The engine in ``processing.quant_engine`` takes plain data and returns plain
rows. This is the only place that knows about stage objects, Tk variables and
the shapes annotation left behind — so the engine stays testable without a GUI
and this file stays small enough to read.

Two functions, deliberately apart: one gathers the inputs, the other builds the
candidates. The gathering is where a wrong matrix or a missing selection does
its damage, and it should be possible to look at it without reading anything
else.
"""
from __future__ import annotations

import numpy as np

from processing.annotation_scoring import DJ_GOLD_HZ_DEFAULT, SCORING_DEFAULTS
from processing.multiplet_line_selector import multiplet_line_peaks
from processing.quant_engine import quantify
from processing.quant_integration import integrate_sample
from processing.quantification import Candidate
from processing.sample_selection import included_sample_indices


def build_candidates(metabolites, peaks_by_rank, *, quality_marks=None):
    """One Candidate per annotated signal that is not marked Bad.

    The evidence handed over is exactly what ``candidate_key`` reads, so the
    ranking here and the ranking the Annotation Results window shows are the
    same computation on the same numbers. Nothing is recomputed: a signal's ΔJ,
    its line counts and its intensity agreement were settled by the annotation
    search, and asking again would be asking a second time and risking a second
    answer.
    """
    qm = quality_marks or {}
    out = []
    for met in metabolites or ():
        mid = str(met.get('id', met.get('name', '')))
        if not mid:
            continue
        for mp in met.get('matched_peaks', []) or ():
            if mp.get('annotation_quality') == 'bad':
                continue
            rank = mp.get('peak_rank')
            lines = multiplet_line_peaks(mp, peaks_by_rank)
            out.append(Candidate(
                met_id=mid,
                peak_rank=rank,
                line_ranks=frozenset(p['rank'] for p in lines
                                     if p.get('rank') is not None),
                evidence={
                    'multiplicity': mp.get('type') or mp.get('multiplicity'),
                    'delta_j_hz': mp.get('delta_j_hz'),
                    'ppm_diff': mp.get('ppm_diff'),
                    'n_found': mp.get('_n_found_lines'),
                    'n_expected': mp.get('_n_exp_lines'),
                    'intensity_agreement': mp.get('intensity_agreement'),
                    'cosine': mp.get('_s_cosine'),
                    'quality': mp.get('annotation_quality'),
                    'manual_quality': (qm.get((mid, rank))
                                       or qm.get((met.get('id'), rank))),
                    'score': mp.get('composite_score',
                                    mp.get('multiplet_score', 0.0)),
                    'protons': mp.get('protons'),
                },
                payload={'name': met.get('name', mid), 'met': met, 'mp': mp},
            ))
    return out


def standard_areas_from_peak(ref_peak, ppm, intensities, *,
                             local_baseline=True):
    """The reference's area in every sample, integrated exactly as an analyte.

    The same window rule and the same baseline: a standard measured differently
    from what it standardises is not a standard, and the ratio would carry the
    difference into every concentration in the study.
    """
    if ref_peak is None:
        return {}
    lo = min(float(ref_peak.get('start_ppm', 0)), float(ref_peak.get('end_ppm', 0)))
    hi = max(float(ref_peak.get('start_ppm', 0)), float(ref_peak.get('end_ppm', 0)))
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = {}
    for s in range(arr.shape[1]):
        a = integrate_sample(ppm, arr[:, s], lo, hi,
                             local_baseline=local_baseline)
        if a is not None:
            out[s] = abs(float(a))
    return out


def sample_groups(samples_info):
    """``{group: [sample indices]}`` as the analyst labelled them."""
    groups = {}
    for i, info in enumerate(samples_info or ()):
        g = (info or {}).get('label') or (info or {}).get('group') or 'Unknown'
        groups.setdefault(str(g), []).append(i)
    return groups


def quantify_project(*, metabolites, peaks, ppm, intensities, samples_info,
                     ambiguous_ranks=(), quality_marks=None, ref_peak=None,
                     ref_nh=1.0, ref_conc=0.0, external_is_samples=None,
                     dilution=None, correction=None, dj_gold_hz=None,
                     annotation_params=None, use_pqn=True,
                     local_baseline=True):
    """Quantify one project and return the rows the results window reads.

    ``intensities`` must be the project's corrected matrix — what
    ``app.corrected_intensities()`` returns. Handing the raw one over would
    integrate whatever broad component the baseline model was there to remove,
    and would do it silently, since a raw spectrum integrates to a perfectly
    plausible number.

    Blanks and excluded samples are dropped by the same rule detection uses, so
    a study's three engines cannot disagree about which samples it contains.
    """
    peaks_by_rank = {p.get('rank'): p for p in (peaks or ())
                     if p.get('rank') is not None}
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    included = included_sample_indices(samples_info, arr.shape[1])

    gold = dj_gold_hz
    if gold is None:
        gold = float((annotation_params or {}).get(
            'dj_gold_hz', SCORING_DEFAULTS.get('dj_gold_hz',
                                               DJ_GOLD_HZ_DEFAULT)))

    return quantify(
        candidates=build_candidates(metabolites, peaks_by_rank,
                                    quality_marks=quality_marks),
        peaks_by_rank=peaks_by_rank,
        ppm=np.asarray(ppm, dtype=float),
        intensities=arr,
        samples_info=samples_info,
        included=included,
        groups=sample_groups(samples_info),
        dj_gold_hz=gold,
        ambiguous_ranks=ambiguous_ranks,
        standard_areas=standard_areas_from_peak(
            ref_peak, ppm, arr, local_baseline=local_baseline),
        standard_nh=ref_nh,
        standard_conc=ref_conc,
        external_standard_samples=external_is_samples,
        dilution=dilution,
        correction=correction,
        use_pqn=use_pqn,
        local_baseline=local_baseline,
    )
