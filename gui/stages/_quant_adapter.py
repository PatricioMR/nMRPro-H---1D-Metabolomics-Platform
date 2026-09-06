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
from processing.quant_concentration import standard_per_proton
from processing.quant_detection_limit import (
    project_drift, project_noise)
from processing.quant_snr import noise_level, representative_snr, window_snr
from processing.spectral_regions import measure_linewidth_ppm
from processing.quantification import Candidate, eligible_metabolite_ids
from processing.sample_selection import included_sample_indices


def build_candidates(metabolites, peaks_by_rank, *, quality_marks=None,
                     eligible_ids=None):
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
        if eligible_ids is not None and mid not in eligible_ids:
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
                    # The corroboration criterion. Annotation measured it; the
                    # election has to see the same number, or the winner here
                    # and the winner the Annotation window shows would be
                    # decided on different evidence.
                    'stocsy_partners': mp.get('stocsy_partners'),
                    'cosine': mp.get('_s_cosine'),
                    'quality': mp.get('annotation_quality'),
                    'manual_quality': (qm.get((mid, rank))
                                       or qm.get((met.get('id'), rank))),
                    'score': mp.get('composite_score',
                                    mp.get('multiplet_score', 0.0)),
                    'protons': mp.get('protons'),
                    # Where the database says the lines are. Only read when
                    # detection found none of them, to bound what could have
                    # been there — never to place a peak.
                    'predicted_ppms': [l.get('ppm') for l
                                       in (mp.get('predicted_lines') or ())
                                       if l.get('ppm') is not None]
                                      or ([mp.get('ppm')]
                                          if mp.get('ppm') is not None else []),
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


def standard_vector(ref_peak, ppm, intensities, ref_nh, *,
                    external_is_samples=None, local_baseline=True):
    """The standard's area per proton as one value per sample.

    The manual region editor recomputes a single analyte's concentration when
    the analyst drags its window, and it has to divide by exactly what the run
    divided by. Computing its own standard would let one row be measured
    against a different reference than the rest of the table.
    """
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    per = standard_per_proton(
        standard_areas_from_peak(ref_peak, ppm, arr,
                                 local_baseline=local_baseline),
        ref_nh, external_samples=external_is_samples)
    if not per:
        return None
    return [float(per.get(s, 0.0)) for s in range(arr.shape[1])]


def sample_groups(samples_info, n_samples=None):
    """``{group: [sample indices]}`` as the analyst labelled them.

    Blanks are not a group. They are the measurement of what the window
    collects without the analyte, and averaging them in with the samples would
    pull every group mean towards the buffer.
    """
    groups = {}
    for i, info in enumerate(samples_info or ()):
        if n_samples is not None and i >= n_samples:
            break
        info = info or {}
        g = info.get('label') or info.get('group') or 'Unknown'
        if str(info.get('type', g)).upper() == 'BLK':
            continue
        groups.setdefault(str(g), []).append(i)
    return groups


def blank_indices(samples_info, n_samples=None):
    """The samples labelled BLK, by the same rule the groups exclude them."""
    out = []
    for i, info in enumerate(samples_info or ()):
        if n_samples is not None and i >= n_samples:
            break
        info = info or {}
        lbl = info.get('label', 'Unknown')
        if str(info.get('type', lbl)).upper() == 'BLK':
            out.append(i)
    return out


def dilution_and_correction(samples_info, group_dilution, n_samples):
    """``(dilution, correction)`` per sample, as two separate mappings.

    They are separate because they enter the calculation at different points:
    the correction factor is always applied, while the dilution cancels in the
    ratio when the standard sits in the sample and does not when it was
    injected on its own. Folding them into one number made that distinction
    impossible to honour.
    """
    dil, corr = {}, {}
    gd = group_dilution or {}
    for i, info in enumerate(samples_info or ()):
        if i >= n_samples:
            break
        info = info or {}
        lbl = info.get('label', 'Unknown')
        c = float(info.get('correction_factor', 1.0) or 1.0)
        dil[i] = float(gd.get(lbl, 1.0)) * c
        corr[i] = c
    return dil, corr


def anchor_metabolite_ids(anchor_lines):
    """``_anchor_lines`` is keyed ``(met_id, peak_rank)``; this is the ids."""
    return {str(k[0]) for k in (anchor_lines or {})}


def manual_good_ids(quality_marks):
    return {str(k[0]) for k, v in (quality_marks or {}).items() if v == 'good'}


def quantify_project(*, metabolites, peaks, ppm, intensities, samples_info,
                     ambiguous_ranks=(), quality_marks=None, anchor_lines=None,
                     ref_peak=None, ref_nh=1.0, ref_conc=0.0,
                     ref_name='Internal standard',
                     external_is_samples=None, group_dilution=None,
                     dj_gold_hz=None, annotation_params=None, use_pqn=True,
                     local_baseline=True, proton_correction=True, align=False,
                     subtract_blanks=False, noise_sigma=None,
                     excluded_regions=()):
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
    n_s = arr.shape[1]
    ppm = np.asarray(ppm, dtype=float)
    included = included_sample_indices(samples_info, n_s)
    blanks = blank_indices(samples_info, n_s)
    dilution, correction = dilution_and_correction(
        samples_info, group_dilution, n_s)

    gold = dj_gold_hz
    if gold is None:
        gold = float((annotation_params or {}).get(
            'dj_gold_hz', SCORING_DEFAULTS.get('dj_gold_hz',
                                               DJ_GOLD_HZ_DEFAULT)))

    eligible = eligible_metabolite_ids(
        metabolites,
        anchor_met_ids=anchor_metabolite_ids(anchor_lines),
        manual_good_ids=manual_good_ids(quality_marks))

    std_areas = standard_areas_from_peak(ref_peak, ppm, arr,
                                         local_baseline=local_baseline)
    std_snr = None
    if ref_peak is not None and std_areas:
        lo = min(float(ref_peak.get('start_ppm', 0)),
                 float(ref_peak.get('end_ppm', 0)))
        hi = max(float(ref_peak.get('start_ppm', 0)),
                 float(ref_peak.get('end_ppm', 0)))
        std_snr = representative_snr(window_snr(
            ppm, arr, lo, hi, noise_level([ref_peak], ppm, arr),
            included=included))

    return quantify(
        candidates=build_candidates(metabolites, peaks_by_rank,
                                    quality_marks=quality_marks,
                                    eligible_ids=eligible),
        peaks_by_rank=peaks_by_rank,
        ppm=ppm,
        intensities=arr,
        samples_info=samples_info,
        included=included,
        groups=sample_groups(samples_info, n_s),
        dj_gold_hz=gold,
        ambiguous_ranks=ambiguous_ranks,
        standard_areas=std_areas,
        standard_nh=ref_nh,
        standard_conc=ref_conc,
        standard_name=ref_name,
        standard_ppm=float((ref_peak or {}).get('ppm', 0.0)),
        standard_snr=std_snr,
        external_standard_samples=external_is_samples,
        dilution=dilution,
        correction=correction,
        use_pqn=use_pqn,
        local_baseline=local_baseline,
        proton_correction=proton_correction,
        align=align,
        blank_indices=blanks if subtract_blanks else None,
        # For the analytes detection found nothing for: the study's own noise,
        # linewidth and drift, so the bound reported is this experiment's and
        # not a constant.
        #
        # The caller's sigma comes first because a detected peak carries no
        # 'noise' key -- detection keeps one noise_std for the spectrum, on the
        # stage. Relying on the per-peak fallback made project_noise return NaN
        # and every censored analyte fell back to "no measurable window",
        # which turned the whole limit-of-detection path into a no-op.
        noise_sigma=(float(noise_sigma)
                     if noise_sigma not in (None, 0)
                     else project_noise(peaks)),
        # Measured on the trace, not from the peaks' recorded widths. Those
        # are valley-to-valley and move with min_distance -- 4.40 to 7.81 Hz
        # across one sweep on one dataset -- so a window padded by them would
        # be wide or narrow according to a detection setting rather than to
        # the spectrum. It is the same measurement the detection panel's
        # resolution warning uses, so the project has one linewidth.
        linewidth_ppm=(measure_linewidth_ppm(ppm, arr, peaks) or 0.008),
        drift_ppm=project_drift(metabolites),
        # Detection was told to ignore these; nothing measured there is the
        # analyte's, so no limit can be read from them either.
        excluded_regions=excluded_regions,
    )
