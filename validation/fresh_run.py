"""An analysis the way a new user runs it: open, detect, annotate, quantify,
touching nothing.

Every setting comes from ``controllers.analysis_defaults``, the same place
the interface builds its controls from, and every step calls the engine the
interface calls, in the interface's order and with its arguments: the
project-level baseline model before detection; the noise from the default
region, or the detector's own estimate where the spectrum has no such
region; detection with its internal alignment; the STOCSY matrix over the
detected peaks; the search on the raw intensities, as the annotation stage
hands it; quantification on the corrected ones.

What a new user DOES have to say is what the software cannot know: the
spectrometer field when the data does not carry it, and how much internal
standard went into the tube. Those are arguments, not defaults.

    python -m validation.fresh_run            # the 18-analyte reference study

``overrides`` replaces any default by name, which is how a proposed change
to a default is measured before it is made.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controllers.analysis_defaults import ANNOTATION, DETECTION, QUANTIFICATION
from gui.stages._annotation_helpers import ambiguous_peak_ranks
from gui.stages._quant_adapter import quantify_project
from processing.baseline_model import BaselineCache, BaselineModel
from processing.noise_estimation import estimate_noise_from_region
from processing.peak_alignment import apply_rigid_shifts
from processing.peak_detection import as_app_peak, detect_peaks
from processing.stocsy import compute_peak_correlation_matrix
from validation.app_annotation import annotate as app_annotate


# The Align dialog's own starting values (visualization_stage).
ALIGN_SEGMENTS = 50
ALIGN_MAX_SHIFT_PPM = 0.030


def defaults(overrides=None) -> dict:
    """Every default in one flat dict, with ``overrides`` applied."""
    d = {**DETECTION, **ANNOTATION,
         **{k: v for k, v in QUANTIFICATION.items() if k != 'baseline_model'},
         'quant_on_model': QUANTIFICATION['baseline_model']}
    unknown = set(overrides or {}) - set(d)
    if unknown:
        raise KeyError(f'not a default: {sorted(unknown)}')
    d.update(overrides or {})
    return d


def region_noise(ppm, raw, region):
    """The detection stage's own rule: a region measured high -> low, and
    ``None`` -- so the detector estimates it -- when there is no usable one.
    """
    hi, lo = float(region[0]), float(region[1])
    if hi <= lo:
        return None
    try:
        sigma = estimate_noise_from_region(ppm, raw, hi, lo)
    except (ValueError, IndexError):
        return None
    return float(sigma) if np.isfinite(sigma) and sigma > 0 else None


def prepare(ppm, raw, samples_info, *, db_path, field_mhz,
            aligned_first=False, keep_detection_alignment=True,
            overrides=None, log=print):
    """Detection and annotation, as the interface runs them: everything
    before an amount is measured.

    ``aligned_first`` presses the Align button before detecting, with the
    dialog's own settings (icoshift, 50 segments, 0.030 ppm): it replaces
    the intensities every later step reads and tells detection not to
    align again, as the application does. Without it detection aligns
    rigidly and, with ``keep_detection_alignment`` (the default, as the
    detection stage does), the same shifts are carried to the matrix
    annotation and quantification read. Turned off, they read the
    unaligned data -- the behaviour before 2026-09-30.
    """
    d = defaults(overrides)
    ppm = np.asarray(ppm, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if aligned_first:
        from processing.peak_alignment import icoshift_align
        with contextlib.redirect_stdout(io.StringIO()):
            raw, _shifts, _segs = icoshift_align(
                ppm, raw, ref_idx=0, n_segments=ALIGN_SEGMENTS,
                max_shift_ppm=ALIGN_MAX_SHIFT_PPM)
        raw = np.asarray(raw, dtype=float)

    model = BaselineModel(enabled=d['baseline_model'],
                          method=d['baseline_method'],
                          params={'niter': d['baseline_niter']}
                          if d['baseline_method'] == 'snip'
                          else {'lam': d['baseline_lam']})
    corr = np.asarray(BaselineCache().corrected(ppm, raw, model,
                                                already_corrected=False),
                      dtype=float)
    bl2d = (raw - corr) if not np.array_equal(raw, corr) else None

    noise = region_noise(ppm, raw, d['noise_region'])
    excluded = list(d['excluded_regions'])
    with contextlib.redirect_stdout(io.StringIO()):
        details = detect_peaks(
            ppm, corr, noise_std=noise, min_snr=d['min_snr'],
            min_distance=d['min_distance_ppm'], baseline_correct=False,
            baseline_2d=bl2d, baseline_method=d['baseline_method'],
            baseline_lam=d['baseline_lam'], baseline_niter=d['baseline_niter'],
            global_align=not aligned_first, samples_info=samples_info,
            return_details=True, per_sample_detection=d['per_sample'],
            detect_shoulders=d['detect_shoulders'],
            shoulder_curvature_k=d['shoulder_k'],
            min_group_presence=d['min_group_presence'],
            excluded_regions=excluded)
    peaks = [as_app_peak(p) for p in details.get('peaks') or ()]
    if (keep_detection_alignment and not aligned_first
            and details.get('alignment_applied')):
        # Detection placed every peak window on the rigidly aligned copy it
        # made; carrying on with the unaligned matrix integrates those
        # windows where the peaks are not. The interface does the same: it
        # replaces the project's intensities and corrects them again.
        raw = apply_rigid_shifts(raw, details.get('shift_points'))
        corr = np.asarray(BaselineCache().corrected(ppm, raw, model,
                                                    already_corrected=False),
                          dtype=float)
    noise_std = float(details.get('noise_std') or noise or 0.0)
    if not peaks:
        raise RuntimeError('detection found nothing')

    included = [int(s['index']) for s in samples_info
                if str(s.get('type', 'Sample')).upper() not in ('BLK', 'BLANK')
                and not s.get('excluded')]
    peak_corr = (compute_peak_correlation_matrix(peaks, included=included)
                 if len(included) >= 3 else {})
    log(f'  detection: {len(peaks)} peaks, noise {noise_std:,.0f} '
        f'({"region" if noise else "estimated"}), baseline model '
        f'{"on" if d["baseline_model"] else "off"}')

    with contextlib.redirect_stdout(io.StringIO()):
        mets, params = app_annotate(
            peaks, ppm, raw, samples_info, db_path=db_path,
            field_mhz=float(field_mhz), tolerance_ppm=d['tolerance_ppm'],
            noise_std=noise_std, min_snr=d['min_snr'],
            min_distance_ppm=d['min_distance_ppm'],
            excluded_regions=excluded, peak_correlation=peak_corr,
            n_samples_correlated=raw.shape[1] if raw.ndim == 2 else 1)
    return {'ppm': ppm, 'raw': raw, 'corr': corr, 'peaks': peaks,
            'mets': mets, 'params': params, 'noise_std': noise_std,
            'excluded': excluded, 'settings': d}


def quantify_prepared(prep, samples_info, *, standard_ppm, standard_nh,
                      standard_conc, standard_name='IS', overrides=None):
    """The quantification step alone, on what ``prepare`` detected and
    annotated. Only the quantification's own settings are read from
    ``overrides``: changing how an amount is measured does not change what
    was detected or annotated, so a test of the one need not redo the
    other."""
    d = defaults(overrides)
    ppm, raw, corr = prep['ppm'], prep['raw'], prep['corr']
    peaks, mets, params = prep['peaks'], prep['mets'], prep['params']
    noise_std, excluded = prep['noise_std'], prep['excluded']
    with contextlib.redirect_stdout(io.StringIO()):
        ref = min(peaks, key=lambda p: abs(float(p['ppm']) - standard_ppm))
        rows = quantify_project(
            metabolites=mets, peaks=peaks, ppm=ppm,
            intensities=corr if d['quant_on_model'] else raw,
            samples_info=samples_info, ref_peak=ref, ref_nh=standard_nh,
            ref_conc=standard_conc, ref_name=standard_name,
            noise_sigma=noise_std,
            ambiguous_ranks=ambiguous_peak_ranks(mets, manual_quality={}),
            annotation_params=params,
            use_pqn=(d['normalisation'] == 'IS+PQN'),
            local_baseline=d['local_baseline'],
            proton_correction=d['proton_correction'],
            align=d['align'], subtract_blanks=d['subtract_blanks'],
            excluded_regions=excluded, deconvolve=False,
            quantify_shapes=d['quantify_shapes'],
            resolve_shared=d['resolve_shared'],
            singlet_min_snr=d['singlet_min_snr'],
            cascade=d['cascade'],
            max_disagreement=d['max_disagreement_pct'] / 100.0,
            raw_intensities=raw, joint_fit=d['joint_fit'])
    return rows


def fresh_run(ppm, raw, samples_info, *, db_path, field_mhz, standard_ppm,
              standard_nh, standard_conc, standard_name='IS',
              aligned_first=False, keep_detection_alignment=True,
              overrides=None, log=print):
    """``(peaks, metabolites, rows, noise_std, settings)`` for one study:
    ``prepare`` then ``quantify_prepared``."""
    prep = prepare(ppm, raw, samples_info, db_path=db_path,
                   field_mhz=field_mhz, aligned_first=aligned_first,
                   keep_detection_alignment=keep_detection_alignment,
                   overrides=overrides, log=log)
    rows = quantify_prepared(prep, samples_info, standard_ppm=standard_ppm,
                             standard_nh=standard_nh,
                             standard_conc=standard_conc,
                             standard_name=standard_name, overrides=overrides)
    return (prep['peaks'], prep['mets'], rows, prep['noise_std'],
            prep['settings'])


def score(rows, truth) -> dict:
    """The three tables' numbers for one run."""
    q = [r for r in rows if not r.get('_not_quantified') and not r.get('_is_is')]
    real = [r for r in q if r['hmdb_id'] in truth]
    fake = [r for r in q if r['hmdb_id'] not in truth]
    errs = []
    for r in real:
        for g, vals in (r.get('_concs_by_group') or {}).items():
            want = truth[r['hmdb_id']].get(g)
            if want and vals:
                errs.append(float(np.median(vals)) / float(want) - 1.0)
    e = np.array(errs) if errs else np.array([np.nan])
    dots = {c: [0, 0] for c in (3, 2, 1)}
    for r in q:
        c = int(r.get('_confidence') or 0)
        if c in dots:
            dots[c][0 if r['hmdb_id'] in truth else 1] += 1
    tp = dots[3][0] + dots[2][0]
    fp = dots[3][1] + dots[2][1]
    fn = dots[1][0] + (len(truth) - len(real))
    tn = dots[1][1]
    p = tp / (tp + fp) if tp + fp else float('nan')
    rc = tp / (tp + fn) if tp + fn else float('nan')
    return {
        'found': len(real), 'truth': len(truth), 'fp': len(fake),
        'median': float(np.nanmedian(np.abs(e))),
        'p90': float(np.nanpercentile(np.abs(e), 90)),
        'bias': float(np.nanmedian(e)),
        'within10': float(np.mean(np.abs(e) <= 0.10)),
        'within20': float(np.mean(np.abs(e) <= 0.20)),
        'precision_names': len(real) / max(1, len(real) + len(fake)),
        'dots': dots, 'tp': tp, 'fp2': fp, 'fn': fn, 'tn': tn,
        'precision': p, 'recall': rc,
        'f1': 2 * p * rc / (p + rc) if (p + rc) else float('nan'),
        'missed': sorted(truth[m]['name'] for m in truth
                         if m not in {r['hmdb_id'] for r in real}),
    }


def tables(results, log=print):
    """The three tables, one line per labelled run."""
    log(f"{'':30s} {'achados':>8s} {'FP':>4s} {'med':>5s} {'p90':>5s} "
        f"{'vies':>5s} {'prec.nomes':>10s}")
    for lab, s in results:
        log(f"{lab[:30]:30s} {s['found']:5d}/{s['truth']:<2d} {s['fp']:4d} "
            f"{100*s['median']:4.0f}% {100*s['p90']:4.0f}% "
            f"{100*s['bias']:+4.0f}% {100*s['precision_names']:9.0f}%")
    log('')
    log(f"{'':30s} " + ''.join(f"{'*' * l:>18s}" for l in (3, 2, 1)))
    for lab, s in results:
        cells = []
        for l in (3, 2, 1):
            r_, f_ = s['dots'][l]
            n = r_ + f_
            cells.append(f"{r_}R {f_}F ={100*r_/n:4.0f}%" if n else '-')
        log(f"{lab[:30]:30s} " + ''.join(f'{c:>18s}' for c in cells))
    log('')
    log(f"{'':30s} {'VP':>4s} {'FP':>4s} {'FN':>4s} {'VN':>4s} "
        f"{'precisao':>9s} {'recall':>7s} {'F1':>6s}")
    for lab, s in results:
        log(f"{lab[:30]:30s} {s['tp']:4d} {s['fp2']:4d} {s['fn']:4d} "
            f"{s['tn']:4d} {100*s['precision']:8.0f}% "
            f"{100*s['recall']:6.0f}% {100*s['f1']:5.0f}%")


def main():
    from validation import mixture_bench as mb
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        X, _sigma, truth, _f = mb.build_study(mb.DB_DEFAULT, seed=a.seed)
    _p, _m, rows, _n, _d = fresh_run(
        mb.PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT, field_mhz=mb.MHZ,
        standard_ppm=mb.STANDARD_PPM, standard_nh=mb.STANDARD_NH,
        standard_conc=mb.STANDARD_MM, standard_name='DSS')
    tables([('utilizador novo, defaults', score(rows, truth))])


if __name__ == '__main__':
    main()
