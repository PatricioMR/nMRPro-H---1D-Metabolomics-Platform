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

import json

import numpy as np

from processing.annotation_scoring import (
    DJ_GOLD_HZ_DEFAULT, SCORING_DEFAULTS, annotate_drift_spread, study_drift)
from processing.multiplet_line_selector import multiplet_line_peaks
from processing.quant_engine import quantify
from processing.quant_shared import annotate_fingerprint_counts
from processing.quant_integration import (
    baseline_anchors, integrate_sample, lorentzian_captured_fraction)
from processing.quant_concentration import standard_per_proton
from processing.quant_detection_limit import (
    project_drift, project_noise)
from processing.quant_snr import noise_level, representative_snr, window_snr
from processing.spectral_regions import measure_linewidth_ppm
from processing.quantification import Candidate, eligible_metabolite_ids
from processing.sample_selection import included_sample_indices


def build_candidates(metabolites, peaks_by_rank, *, quality_marks=None,
                     eligible_ids=None, pool=None):
    """One Candidate per annotated signal that is not marked Bad.

    The evidence handed over is exactly what ``candidate_key`` reads, so the
    ranking here and the ranking the Annotation Results window shows are the
    same computation on the same numbers. Nothing is recomputed: a signal's ΔJ,
    its line counts and its intensity agreement were settled by the annotation
    search, and asking again would be asking a second time and risking a second
    answer.

    ``pool``, a list when given, receives the signals that do NOT become
    candidates -- marked Bad by the annotation, or of an analyte outside
    ``eligible_ids`` -- except those the analyst marked Bad and those of an
    analyte the fingerprint contradicted. They are what the remainder of a
    resolved window is offered (quant_shared.resolve_residuals): a shape the
    election could not verify line by line may still be the one thing that
    explains what the accepted shapes leave.
    """
    qm = quality_marks or {}
    annotate_drift_spread(metabolites)
    # where a signal is expected in this study: the database position at
    # the study's common drift (quant_shared places an intruder there when
    # detection gave it the owner's line instead of one of its own)
    _common = study_drift(metabolites)
    out = []
    for met in metabolites or ():
        mid = str(met.get('id', met.get('name', '')))
        if not mid:
            continue
        _eligible = eligible_ids is None or mid in eligible_ids
        _contradicted = bool(met.get('_fingerprint_contradictions'))
        for mp in met.get('matched_peaks', []) or ():
            rank = mp.get('peak_rank')
            _manual_bad = ((qm.get((mid, rank)) or qm.get((met.get('id'), rank)))
                           == 'bad')
            _is_pattern = (str(mp.get('type') or mp.get('multiplicity') or '')
                           .lower().strip() == 'm*')
            # A complex multiplet never competes for a peak (it holds none
            # it found line by line); with the shapes switch on it is
            # offered to the spectrum's remainders like any refused signal
            # (quant_shared.resolve_residuals), and identified only where a
            # free apex, its own amount and its whole shape agree.
            _to_pool = ((not _eligible) or mp.get('annotation_quality') == 'bad'
                        or _is_pattern)
            if _to_pool and (pool is None or _manual_bad or _contradicted):
                continue
            _target = pool if _to_pool else out
            lines = multiplet_line_peaks(mp, peaks_by_rank)
            # A multiplet judged as a shape (ADR-013) claims no peak: its
            # lines were never found one by one. It carries the fit instead,
            # and the engine measures it from its shape.
            _pf = mp.get('pattern_fit') or {}
            _shape = None
            if (_pf.get('snr') is not None and mp.get('peak_ppm') is not None
                    and mp.get('study_linewidth_hz')
                    and mp.get('pattern_field_mhz')):
                _shape = {
                    'origin_ppm': float(mp['peak_ppm']) - float(
                        mp.get('pattern_centre_offset_ppm') or 0.0),
                    'field_mhz': mp.get('pattern_field_mhz'),
                    'study_linewidth_hz': mp.get('study_linewidth_hz'),
                    'pattern_profile': mp.get('pattern_profile'),
                    'profile_fwhm_hz': mp.get('profile_fwhm_hz'),
                    'pattern_lines': (json.loads(mp['pattern_lines'])
                                      if isinstance(mp.get('pattern_lines'), str)
                                      else mp.get('pattern_lines')),
                    'snr': _pf.get('snr'), 'explained': _pf.get('explained'),
                    'apex_ppms': [l.get('ppm') for l
                                  in (mp.get('predicted_lines') or ())],
                    'apex_rel': [l.get('rel_intensity', 1.0) for l
                                 in (mp.get('predicted_lines') or ())],
                }
            if _shape is not None and _target is out:
                rank, lines = None, []
            _target.append(Candidate(
                met_id=mid,
                peak_rank=rank,
                line_ranks=frozenset(p['rank'] for p in lines
                                     if p.get('rank') is not None),
                evidence={
                    'multiplicity': mp.get('type') or mp.get('multiplicity'),
                    'delta_j_hz': mp.get('delta_j_hz'),
                    # How well the reference coupling itself is known, where
                    # the DB measured it. The ranking judges delta-J no finer
                    # than this.
                    'j_sigma_hz': mp.get('j_sigma'),
                    'ppm_diff': mp.get('ppm_diff'),
                    'n_found': mp.get('_n_found_lines'),
                    'n_expected': mp.get('_n_exp_lines'),
                    'intensity_agreement': mp.get('intensity_agreement'),
                    'lines_explained': mp.get('lines_explained'),
                    'lines_covary': mp.get('lines_covary'),
                    'drift_spread_ppm': mp.get('drift_spread_ppm'),
                    'ppm_off_drift': mp.get('ppm_off_drift'),
                    # The corroboration criterion. Annotation measured it; the
                    # election has to see the same number, or the winner here
                    # and the winner the Annotation window shows would be
                    # decided on different evidence.
                    'stocsy_partners': mp.get('stocsy_partners'),
                    'cosine': mp.get('_s_cosine'),
                    'fp_contradicted': mp.get('fp_contradicted'),
                    'fp_consistent': mp.get('fp_consistent'),
                    'fp_checked': mp.get('fp_checked'),
                    'quality': mp.get('annotation_quality'),
                    # whether the analyte may be quantified at all: a signal
                    # in the pool of one that may is owed a row, measured
                    # or not
                    'eligible': _eligible,
                    'manual_quality': (qm.get((mid, rank))
                                       or qm.get((met.get('id'), rank))),
                    'score': mp.get('composite_score',
                                    mp.get('multiplet_score', 0.0)),
                    # Divided by the listed protons, the stoichiometry --
                    # not by those measured under the profile on the
                    # reference spectrum, which carry its artefacts (see
                    # quant_cascade.signal_protons).
                    'protons': mp.get('protons'),
                    # A measured pattern's relative line heights, for the
                    # capture correction; a first-order multiplet carries none
                    # and the engine falls back to the binomial pattern.
                    'line_weights': ([float(l.get('rel_intensity') or 1.0)
                                      for l in (mp.get('predicted_lines') or ())]
                                     if (mp.get('type') or mp.get('multiplicity'))
                                     == 'm*' else None),
                    # Where the database says the lines are. Only read when
                    # detection found none of them, to bound what could have
                    # been there — never to place a peak.
                    'predicted_ppms': [l.get('ppm') for l
                                       in (mp.get('predicted_lines') or ())
                                       if l.get('ppm') is not None]
                                      or ([mp.get('ppm')]
                                          if mp.get('ppm') is not None else []),
                    'shape_fit': _shape,
                    # The shape the reference measured for this signal,
                    # letter or singlet, for a shared window's fit
                    # (quant_shared): the pure compound's profile placed on
                    # the found lines.
                    'pattern_profile': mp.get('pattern_profile'),
                    'profile_fwhm_hz': mp.get('profile_fwhm_hz'),
                    'pattern_field_mhz': mp.get('pattern_field_mhz'),
                    'db_ppm': mp.get('db_ppm'),
                    'study_drift_ppm': _common,
                    'expected_ppm': (float(mp['db_ppm']) + float(_common)
                                     if _common is not None
                                     and mp.get('db_ppm') is not None
                                     else None),
                    # The reference's lines at the found centre, for the
                    # plots: never integrated, never placed by them.
                    'reference_ppms': [float(l['ppm']) for l
                                       in (mp.get('reference_lines') or ())
                                       if l.get('ppm') is not None],
                },
                payload={'name': met.get('name', mid), 'met': met, 'mp': mp},
            ))
    return out


def standard_window(ref_peak, linewidth_ppm=None, other_peaks=()):
    """A window centred on the standard, not one bounded by its neighbours.

    Detection records a peak's extent valley to valley, which for a peak with
    no close neighbour on one side is wildly asymmetric: measured on a
    synthetic study, TSP at 0.0 ppm got ``[-0.0117, +0.7206]`` — its apex at
    1.6% of a window 440 Hz wide, because nothing stood between it and the next
    signal 0.72 ppm away. At one drift setting the apex landed exactly on the
    boundary and half the peak was integrated away, and since every
    concentration is divided by this area, every concentration in that study
    more than doubled.

    Nothing could have shown it from the results: the standard's recovery is
    computed against itself and reads 1.00 whatever the window does.

    So the window is symmetric about the apex and sized by the study's
    linewidth. Five half-widths holds about 87% of a Lorentzian, and the same
    fraction for every signal measured this way, which is what a ratio needs.

    It is shortened only by a real neighbour — half the distance to the nearest
    other detected peak — never by the recorded extent, which is the
    unreliable quantity being corrected. Clamping to that extent instead made
    the window swing between 4.6 and 31.8 Hz across drift settings on the
    synthetic study, since the boundary it clamped to was itself an artefact.
    """
    apex = float(ref_peak.get('ppm', 0.0))
    lw = abs(float(linewidth_ppm or 0.0))
    if lw <= 0:
        lw = abs(float(ref_peak.get('width_ppm', 0.0) or 0.0)) or 0.006
    half = 5.0 * lw
    rank = ref_peak.get('rank')
    gaps = [abs(float(p.get('ppm', apex)) - apex)
            for p in (other_peaks or ())
            if p.get('rank') != rank and p.get('ppm') is not None]
    gaps = [g for g in gaps if g > 0]
    if gaps:
        half = min(half, min(gaps) / 2.0)
    return apex - half, apex + half


def standard_areas_from_peak(ref_peak, ppm, intensities, *,
                             local_baseline=True, linewidth_ppm=None,
                             other_peaks=()):
    """The reference's area in every sample, integrated exactly as an analyte.

    The same baseline, and a window centred on the signal — see
    ``standard_window`` for why the recorded extent will not do. A standard
    measured differently from what it standardises is not a standard, and the
    difference goes into every concentration in the study.
    """
    if ref_peak is None:
        return {}
    lo, hi = standard_window(ref_peak, linewidth_ppm, other_peaks)
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    anchors = (baseline_anchors(
        lo, hi, [ref_peak.get('ppm')] + [q.get('ppm') for q in other_peaks
                                         or ()],
        linewidth_ppm, trace=(ppm, arr.mean(axis=1)))
        if local_baseline and linewidth_ppm else None)
    # Corrected for what the window leaves outside, the same way an analyte's
    # area is. Uncorrected, the standard held 94% of its line while a doublet
    # held 83%, and every concentration in the study carried the difference.
    captured = lorentzian_captured_fraction(
        [float(ref_peak.get('ppm', 0.0))], [1.0], lo, hi,
        (linewidth_ppm or 0.0) / 2.0,
        anchors=anchors[:2] if anchors else None) or 1.0
    out = {}
    for s in range(arr.shape[1]):
        a = integrate_sample(ppm, arr[:, s], lo, hi,
                             local_baseline=local_baseline, anchors=anchors)
        if a is not None:
            out[s] = float(a) / captured
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


def project_field_mhz(metabolites):
    """The spectrometer frequency annotation ran at, as it wrote on every
    matched peak; None when nothing was matched."""
    for m in metabolites or ():
        for mp in m.get('matched_peaks') or ():
            f = mp.get('pattern_field_mhz')
            if f:
                return float(f)
    return None


def anchor_metabolite_ids(anchor_lines):
    """``_anchor_lines`` is keyed ``(met_id, peak_rank)``; this is the ids."""
    return {str(k[0]) for k in (anchor_lines or {})}


def manual_good_ids(quality_marks):
    return {str(k[0]) for k, v in (quality_marks or {}).items() if v == 'good'}


# The per-sample fields the engine returns are {sample_index: value}, because
# a sample it could not measure is absent rather than present with a zero.
# Everything downstream — the statistics matrix, the CSV export, the manual
# region editor, the per-sample peak view, the lipidomics path — indexes them
# by position instead, and has since before the engine was rewritten.
#
# Handing a dict to those readers is not a crash so much as a silent lie:
# ``list(a_dict)`` yields its KEYS, so the export would have written 0, 1, 2 in
# the concentration column, and the statistics stage would have raised on a
# slice. One boundary converts, here, because this module is the only one that
# knows both shapes.
_PER_SAMPLE_FIELDS = ('_concs_per_sample', '_areas_per_sample',
                      '_snr_per_sample', '_detected_per_sample')


def _as_sample_list(mapping, n_samples):
    """``{index: value}`` as a list of ``n_samples``, NaN where unmeasured.

    NaN and not zero: a sample the analyte was not measured in has no number,
    and a zero there would enter a group mean as a real measurement of nothing.
    """
    if isinstance(mapping, list):
        return mapping
    out = [float('nan')] * int(n_samples)
    for k, v in (mapping or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(out):
            out[i] = bool(v) if isinstance(v, bool) else float(v)
    return out


def _rows_indexed_by_sample(rows, n_samples):
    for r in rows:
        for key in _PER_SAMPLE_FIELDS:
            if key in r:
                r[key] = _as_sample_list(r.get(key), n_samples)
    return rows


def _for_display(rows, metabolites, peaks_by_rank):
    """Fill in what the table and everything after it read.

    The engine works in metabolite ids, because two analytes can share a
    display name and an id cannot. Names, the annotation hit and the detected
    peak are display concerns, and they belong on this boundary rather than in
    the stage: a caller that gets rows from here should get complete ones.
    """
    name_by_id = {str(m.get('id', '')): m.get('name', m.get('id', ''))
                  for m in (metabolites or ())}
    for r in rows:
        if r.get('_is_is'):
            continue
        mid = str(r.get('hmdb_id', ''))
        r['metabolite'] = name_by_id.get(mid, mid)
        payload = (r.get('_hits') or [None])[0]
        if isinstance(payload, dict) and 'mp' in payload:
            r['_hits'] = [payload['mp']]
        pr = r.get('_peak_rank')
        if pr is not None and pr in peaks_by_rank:
            pk = peaks_by_rank[pr]
            r['_peak'] = pk
            r['start_idx'] = pk.get('start_idx')
            r['end_idx'] = pk.get('end_idx')
    return rows


def quantify_project(*, metabolites, peaks, ppm, intensities, samples_info,
                     ambiguous_ranks=(), quality_marks=None, anchor_lines=None,
                     ref_peak=None, ref_nh=1.0, ref_conc=0.0,
                     ref_name='Internal standard',
                     external_is_samples=None, group_dilution=None,
                     dj_gold_hz=None, annotation_params=None, use_pqn=True,
                     local_baseline=True, proton_correction=True, align=False,
                     subtract_blanks=False, noise_sigma=None,
                     excluded_regions=(), deconvolve=True,
                     quantify_shapes=True, resolve_shared=False,
                     shared_overrides=None, diagnostics=None,
                     singlet_min_snr=None, cascade=False, ppm_band_ppm=None,
                     claim_signals=True, max_disagreement=None,
                     raw_intensities=None, joint_fit=False):
    """Quantify one project and return the rows the results window reads.

    ``intensities`` must be the project's corrected matrix — what
    ``app.corrected_intensities()`` returns. Handing the raw one over would
    integrate whatever broad component the baseline model was there to remove,
    and would do it silently, since a raw spectrum integrates to a perfectly
    plausible number.

    Blanks and excluded samples are dropped by the same rule detection uses, so
    a study's three engines cannot disagree about which samples it contains.

    ``diagnostics``, when a dict is passed, comes back holding what the run did
    that the rows cannot show -- see ``quant_engine.quantify``.

    ``cascade`` adds, after the election, the analytes the cascade finds on
    what the quantified ones leave (quant_engine.quantify).
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

    _lw = measure_linewidth_ppm(ppm, arr, peaks) or 0.008
    std_areas = standard_areas_from_peak(ref_peak, ppm, arr,
                                         local_baseline=local_baseline,
                                         linewidth_ppm=_lw,
                                         other_peaks=peaks or ())
    std_snr = None
    if ref_peak is not None and std_areas:
        lo, hi = standard_window(ref_peak, _lw, peaks or ())
        std_snr = representative_snr(window_snr(
            ppm, arr, lo, hi, noise_level([ref_peak], ppm, arr),
            included=included))

    # What the rest of each analyte's fingerprint says about each of its
    # candidate peaks, at the amount that peak implies -- the election's
    # second key after the manual mark (annotation_scoring.candidate_key).
    # Read on the mean of the measured samples with the study's drift.
    if noise_sigma and included:
        _mean = arr[:, [i for i in included if 0 <= i < arr.shape[1]]].mean(axis=1)
        annotate_fingerprint_counts(
            metabolites, ppm, _mean, noise=noise_sigma, linewidth_ppm=_lw,
            drift_ppm=float(study_drift(metabolites) or 0.0),
            excluded_regions=excluded_regions)
    _pool: list = []
    _cands = build_candidates(metabolites, peaks_by_rank,
                              quality_marks=quality_marks,
                              eligible_ids=eligible,
                              pool=_pool if resolve_shared else None)
    return _for_display(_rows_indexed_by_sample(quantify(
        candidates=_cands,
        residual_pool=_pool,
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
        linewidth_ppm=_lw,
        drift_ppm=project_drift(metabolites),
        # Detection was told to ignore these; nothing measured there is the
        # analyte's, so no limit can be read from them either.
        excluded_regions=excluded_regions,
        deconvolve=deconvolve,
        quantify_shapes=quantify_shapes,
        resolve_shared=resolve_shared,
        shared_overrides=shared_overrides,
        diagnostics=diagnostics,
        singlet_min_snr=singlet_min_snr,
        cascade=cascade,
        ppm_band_ppm=ppm_band_ppm,
        claim_signals=claim_signals,
        max_disagreement=max_disagreement,
        raw_intensities=raw_intensities,
        joint_fit=joint_fit,
        metabolites=metabolites,
        field_mhz=project_field_mhz(metabolites),
    ), n_s), metabolites, peaks_by_rank)
