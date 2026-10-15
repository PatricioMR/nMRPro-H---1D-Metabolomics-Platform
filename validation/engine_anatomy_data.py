"""The numbers behind the engine-anatomy report: every table a figure of the
report is drawn from, written as CSV from the bench runs already on disk.

    python -m validation.engine_anatomy_data <out dir>

Runs read (bench_runs labels, 600 MHz GISSMO tubes):
  standard tubes 18/50/100, seeds 0-2      sb (score off) / si (score on)
  extract-like r50/r75, seeds 103-105      o2b / o2i  (compound 6 + signal 4 mppm)
  extract + biology r50/r75, seeds 103-105 o3b / o3i  (+ amount CV 50%, dilution 30%)
  clean r50/r75 before/after the mode drift  rec0_funnel / d1_funnel (seeds 100-102)
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.bench_runs import OUT_DEFAULT, _path

REGIMES = {
    # name: (tubes, signal shift, compound shift, seeds, before, after, biology)
    'Standard': (('18', '50', '100'), 0.0, 0.0, (0, 1, 2), 'sb', 'si', None),
    'Extract': (('r50', 'r75'), 4.0, 6.0, (103, 104, 105), 'o2b', 'o2i', None),
    'Extract + biology': (('r50', 'r75'), 4.0, 6.0, (103, 104, 105), 'o3b', 'o3i', (0.5, 0.3)),
}
VERSION = {'before': 'Engine v0.73', 'after': 'Engine with identity score'}


def _rows(label, tube, sh, rs, seed):
    p = _path(OUT_DEFAULT, label, tube, sh, seed, 600, 0.0, rs)
    if not os.path.exists(p):
        return None, None
    d = pickle.load(open(p, 'rb'))
    return d['rows'], d['truth']


def _quantified(rows):
    return [r for r in rows if not r.get('_not_quantified') and not r.get('_is_is')]


def write(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'{os.path.basename(path)}: {len(rows)} rows')


def levels_and_summary(out):
    lv, sm, sc, rc, rj = [], [], [], [], []
    for reg, (tubes, sh, rs, seeds, before, after, _bio) in REGIMES.items():
        for tube in tubes:
            for ver, label in (('before', before), ('after', after)):
                dots = {3: [0, 0], 2: [0, 0], 1: [0, 0]}
                errs, truth_n, found, fake = [], 0, 0, 0
                bands = Counter()
                rej = Counter()
                for s in seeds:
                    rows, truth = _rows(label, tube, sh, rs, s)
                    if rows is None:
                        continue
                    truth_n += len(truth)
                    q = _quantified(rows)
                    got = {r['hmdb_id'] for r in q}
                    for r in q:
                        real = r['hmdb_id'] in truth
                        c = int(r.get('_confidence') or 0)
                        if c in dots:
                            dots[c][0 if real else 1] += 1
                        found += real
                        fake += not real
                        if real:
                            for g, vals in (r.get('_concs_by_group') or {}).items():
                                want = truth[r['hmdb_id']].get(g)
                                if want and vals:
                                    m = float(np.median(vals))
                                    errs.append(m / float(want) - 1.0)
                                    if ver == 'after' and g == 'A':
                                        sc.append((reg, tube, s, r.get('metabolite'), round(float(want), 4),
                                                   round(m, 4), c))
                    for mid, t in truth.items():
                        a = float(t['A'])
                        band = ('0.05-0.1' if a < 0.1 else '0.1-0.2' if a < 0.2 else
                                '0.2-0.4' if a < 0.4 else '0.4-1' if a <= 1.0 else '>1')
                        bands[(band, 'n')] += 1
                        bands[(band, 'found')] += mid in got
                    if ver == 'after':
                        for r in rows:
                            if r.get('_is_is') or not r.get('_not_quantified'):
                                continue
                            why = str(r.get('_rejection') or 'no number')
                            rej[(why, r['hmdb_id'] in truth)] += 1
                for lvl in (3, 2, 1):
                    lv.append((reg, tube, VERSION[ver], lvl, dots[lvl][0], dots[lvl][1]))
                e = np.array(errs) if errs else np.array([np.nan])
                sm.append((reg, tube, VERSION[ver], truth_n, found, fake,
                           round(100 * found / truth_n, 1) if truth_n else '',
                           round(100 * float(np.nanmedian(np.abs(e))), 1),
                           round(100 * float(np.nanpercentile(np.abs(e), 90)), 1),
                           round(100 * float(np.nanmedian(e)), 1),
                           round(100 * float(np.mean(np.abs(e) <= 0.2)), 1)))
                for band in ('0.05-0.1', '0.1-0.2', '0.2-0.4', '0.4-1'):
                    if bands[(band, 'n')]:
                        rc.append((reg, tube, VERSION[ver], band, bands[(band, 'n')],
                                   bands[(band, 'found')]))
                if ver == 'after':
                    for (why, real), n in sorted(rej.items()):
                        rj.append((reg, tube, why, 'real' if real else 'stranger', n))
    write(os.path.join(out, 'confidence_levels.csv'),
          ('regime', 'tube', 'version', 'dots', 'real', 'stranger'), lv)
    write(os.path.join(out, 'recovery_summary.csv'),
          ('regime', 'tube', 'version', 'in_tube', 'found', 'strangers', 'recovery_pct',
           'error_median_pct', 'error_p90_pct', 'bias_pct', 'within20_pct'), sm)
    write(os.path.join(out, 'measured_vs_true.csv'),
          ('regime', 'tube', 'seed', 'analyte', 'true_mM', 'measured_mM', 'dots'), sc)
    write(os.path.join(out, 'recovery_by_concentration.csv'),
          ('regime', 'tube', 'version', 'band_mM', 'analytes', 'found'), rc)
    write(os.path.join(out, 'not_reported_reasons.csv'),
          ('regime', 'tube', 'reason', 'kind', 'n'), rj)


def stage_funnel(out):
    """Real analytes and strangers through the stages, current engine."""
    from validation import gissmo_bench as gb
    from validation.recovery_funnel import analyte_records
    rows_out = []
    for reg, tube in (('Standard', '100'), ('Extract + biology', 'r75')):
        tubes, sh, rs, seeds, _b, after, bio = REGIMES[reg]
        gb.SIGNAL_SHIFT = sh / 1000.0
        gb.AMOUNT_SD, gb.DILUTION_SD = bio or (0.12, 0.0)
        c = Counter()
        for s in seeds:
            recs, _str = analyte_records(tube, s, 600, after, shift=sh, ref_shift=rs)
            prep, truth = gb.prepared(gb.FIELDS[600], s, tube, ref_shift=rs / 1000.0)
            rows, _t = _rows(after, tube, sh, rs, s)
            c['In the tube', 1] += len(recs)
            c['Line above 5 sigma', 1] += sum(1 for r in recs if r['n_lines5'] > 0)
            c['Detected', 1] += sum(1 for r in recs if r['detected'])
            for met in prep['mets']:
                mps = [m for m in met.get('matched_peaks') or () if not m.get('is_sub_peak')]
                if not mps:
                    continue
                real = int(str(met.get('id')) in truth)
                c['Proposed by annotation', real] += 1
                if any(m.get('annotation_quality') != 'bad' for m in mps):
                    c['A signal not rejected', real] += 1
            for r in rows:
                if r.get('_is_is'):
                    continue
                real = int(r['hmdb_id'] in truth)
                if not r.get('_not_quantified'):
                    c['Reported', real] += 1
                    if int(r.get('_confidence') or 0) >= 2:
                        c['Reported at two dots or more', real] += 1
                    if int(r.get('_confidence') or 0) >= 3:
                        c['Reported at three dots', real] += 1
        for i, st in enumerate(('In the tube', 'Line above 5 sigma', 'Detected',
                                'Proposed by annotation', 'A signal not rejected', 'Reported',
                                'Reported at two dots or more', 'Reported at three dots')):
            rows_out.append((reg, tube, i + 1, st, c[st, 1],
                             c[st, 0] if st not in ('In the tube', 'Line above 5 sigma', 'Detected') else ''))
        gb.SIGNAL_SHIFT, gb.AMOUNT_SD, gb.DILUTION_SD = 0.0, 0.12, 0.0
    write(os.path.join(out, 'stage_funnel.csv'),
          ('regime', 'tube', 'step', 'stage', 'real', 'stranger'), rows_out)


def drift_effect(out):
    """Recovery by concentration before and after the study drift by the
    mode (ADR-085), clean random tubes of 50 and 75, seeds 100-102."""
    rows = []
    for ver, lab in (('Drift from the panel', 'rec0'), ('Drift by the mode', 'd1')):
        p = os.path.join(OUT_DEFAULT, f'{lab}_funnel.pkl')
        if not os.path.exists(p):
            continue
        recs = pickle.load(open(p, 'rb'))['recs']
        for lo, hi, band in ((0.05, 0.1, '0.05-0.1'), (0.1, 0.2, '0.1-0.2'),
                             (0.2, 0.4, '0.2-0.4'), (0.4, 1.01, '0.4-1')):
            sel = [r for r in recs if lo <= r['conc'] < hi]
            rows.append((ver, band, len(sel), sum(r['stage'] == 'number' for r in sel)))
    write(os.path.join(out, 'drift_mode_effect.csv'), ('version', 'band_mM', 'analytes', 'found'), rows)


def peaks_vs_analytes(out):
    src = os.path.join(os.path.dirname(OUT_DEFAULT), 'composition', 'comp_runs.csv')
    rows = []
    with open(src, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['analytes']), int(r['peaks'])))
    write(os.path.join(out, 'peaks_vs_analytes.csv'), ('analytes', 'detected_peaks'), rows)


def score_weights(out):
    from processing import identity_score as ids
    names = ['row: log SNR', 'row: annotation score', 'row: singlet', 'row: from the cascade',
             'row: identity level', 'row: fingerprint signals present',
             'row: fingerprint signals contradicted', 'row: fingerprint signals faint',
             'row: shape axis', 'row: shared window', 'row: shape explained',
             'row: share of window kept', 'row: neighbour share', 'row: captured fraction',
             'row: refused by the joint fit', 'row: cascade crowded', 'row: singlet off the drift',
             'cascade: apex SNR', 'cascade: signals SNR', 'cascade: disagreement',
             'cascade: free share', 'cascade: covariation measured', 'cascade: covariation r',
             'row: log peaks',
             'annotation: signals listed', 'annotation: signals matched', 'annotation: share matched',
             'annotation: Good signals', 'annotation: share Good', 'annotation: incomplete multiplets',
             'annotation: lines not co-varying', 'annotation: other Bad', 'annotation: singlets only',
             'annotation: best SNR', 'annotation: protons found', 'annotation: best delta-J',
             'annotation: delta-J measured', 'annotation: nearest to drift',
             'annotation: median off drift', 'annotation: covariation median r',
             'annotation: covariation min r', 'annotation: covariation measured',
             'annotation: fingerprint contradicted', 'annotation: peaks shared with rivals']
    rows = [(n, round(float(w), 4)) for n, w in zip(names, ids.SCORE_COEF)]
    write(os.path.join(out, 'identity_score_weights.csv'), ('feature', 'weight'), rows)


def measured_fixed(out):
    """Measurements made during the 13-14 Oct 2026 sessions by
    validation scripts (extract_regime, drift estimators); kept as they
    were printed, so the report can cite them."""
    write(os.path.join(out, 'drift_estimators.csv'),
          ('study', 'kind', 'panel_mppm', 'median_mppm', 'mode_mppm', 'truth_mppm'),
          [('r50 seed 100', 'bench', 0.9, 7.3, 7.4, 8), ('r50 seed 101', 'bench', 8.2, 8.2, 8.0, 8),
           ('r50 seed 102', 'bench', -20.0, 7.3, 8.2, 8), ('r75 seed 100', 'bench', -23.6, 7.0, 7.8, 8),
           ('r75 seed 101', 'bench', -16.7, 8.1, 8.0, 8), ('r75 seed 102', 'bench', 1.7, 8.9, 8.6, 8),
           ('Mice ISS skin', 'real', 13.2, 9.7, 11.2, ''), ('Murilo cells', 'real', 14.0, 2.8, 12.4, ''),
           ('RawMG', 'real', 8.6, 5.5, 12.6, ''), ('Liver', 'real', 14.2, 8.3, 15.8, ''),
           ('Organoids', 'real', 11.2, 11.9, 12.2, '')])
    write(os.path.join(out, 'study_regimes.csv'),
          ('study', 'within_group_cv_pct', 'study_cv_pct', 'linewidth_hz_median'),
          [('Mice ISS skin', 67, 72, 2.56), ('Liver', 46, 53, 1.56), ('Murilo cells', 14, 42, 0.95),
           ('RawMG', '', '', 1.56), ('Bench (old)', 12, 12, 2.6), ('Bench (extract + biology)', 63, 63, 2.6)])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DEFAULT, 'engine_anatomy')
    os.makedirs(out, exist_ok=True)
    levels_and_summary(out)
    drift_effect(out)
    peaks_vs_analytes(out)
    score_weights(out)
    measured_fixed(out)
    stage_funnel(out)


if __name__ == '__main__':
    main()
