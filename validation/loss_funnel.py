"""Where the bench's true analytes are lost, and what tells a true one from
a stranger.

For every analyte really in a GISSMO tube, the stage it reached:

    annotated      a listed signal matched to a detected peak
    in study       a matched signal inside the study's drift band
    candidate      a signal handed to the election (not Bad, eligible)
    row            the quantification gave it a row
    number         the row has a concentration
    dots           its confidence

and, for every quantified row, true or stranger, the evidence the engine
recorded about it, so the two can be compared level by level.

    python -m validation.loss_funnel --tubes 18 50 --seeds 0 1 2
    python -m validation.loss_funnel --rows-dir <dir of pickled rows>
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from processing.annotation_scoring import in_study
from processing.quant_engine import stands_in_study


def _reason(row) -> str:
    if row is None:
        return 'no row'
    if not row.get('_not_quantified'):
        return 'number'
    first = (row.get('_status_reasons') or ['?'])[0]
    keys = (('complex multiplet', 'm* never competes'),
            ('joint fit', 'joint fit refused'),
            ('Identified by the cascade but not quantified', 'cascade: crowded'),
            ('lost the competition', 'lost the election'),
            ('contradicted', 'amount contradicted'),
            ('singlet on its own', 'lone singlet'),
            ('window', 'window taken'))
    for k, lab in keys:
        if k in first:
            return lab
    return first[:50]


def annotation_evidence(met, band_ppm=None) -> dict:
    mps = [mp for mp in (met or {}).get('matched_peaks') or ()
           if not mp.get('is_sub_peak')]
    sig = {}
    for mp in mps:
        k = round(float(mp.get('db_ppm') or 0.0), 4)
        q = mp.get('annotation_quality')
        best = sig.get(k)
        rank = {'good': 3, 'average': 2, 'poor': 1, 'bad': 0}.get(q, 1)
        if best is None or rank > best[0]:
            sig[k] = (rank, in_study(mp, band_ppm), mp.get('ppm_off_drift'),
                      mp.get('type'))
    listed = len((met or {}).get('db_peaks') or ())
    return {
        'listed': listed,
        'matched': len(sig),
        'in_study': sum(1 for v in sig.values() if v[1]),
        'good': sum(1 for v in sig.values() if v[0] == 3),
        'good_in_study': sum(1 for v in sig.values() if v[0] == 3 and v[1]),
        'bad_only': bool(sig) and all(v[0] == 0 for v in sig.values()),
        'stands': stands_in_study(met, band_ppm) if met else False,
        'off_drift_mppm': (1000 * float(np.median([abs(v[2]) for v in sig.values()
                                                   if v[2] is not None]))
                           if any(v[2] is not None for v in sig.values()) else None),
    }


def row_evidence(row) -> dict:
    ax = row.get('_confidence_axes') or {}
    fp = row.get('_fingerprint_at_amount') or ()
    verdicts = Counter(str(v[3]).split(' ')[0] for v in fp if len(v) > 3)
    return {
        'dots': int(row.get('_confidence') or 0),
        'identity': ax.get('identity'), 'shape': ax.get('shape'),
        'shared': bool(ax.get('shared')),
        'cascade': bool(row.get('_by_cascade')),
        'residual': bool(row.get('_by_residual')),
        'joint': bool(row.get('_joint_fit')),
        'snr': float(row.get('snr') or float('nan')),
        'present': verdicts.get('present', 0),
        'contradicted': verdicts.get('contradicted', 0),
        'singlet': bool(row.get('_is_singlet')),
        'score': (row.get('_cascade') or {}).get('score'),
        'disagreement': (row.get('_cascade') or {}).get('disagreement'),
    }


def tube_runs(tubes, seeds, shifts, rows_dir=None, prefix='desligado'):
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    from validation.fresh_run import quantify_prepared
    for sh in shifts:
        gb.SIGNAL_SHIFT = sh / 1000.0
        for t in tubes:
            for s in seeds:
                prep, truth = gb.prepared(gb.FIELDS[600], s, t)
                path = rows_dir and os.path.join(rows_dir, f'{prefix}_{t}_{sh:g}_{s}.pkl')
                if path and os.path.exists(path):
                    rows = pickle.load(open(path, 'rb'))['rows']
                else:
                    mb.MHZ = gb.FIELDS[600]
                    with contextlib.redirect_stdout(io.StringIO()):
                        rows = quantify_prepared(
                            prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM,
                            standard_nh=mb.STANDARD_NH, standard_conc=mb.STANDARD_MM,
                            standard_name='DSS')
                yield (t, sh, s), prep, truth, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--tubes', nargs='+', default=['18', '50'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--shifts', type=float, nargs='+', default=[0.0, 5.0])
    ap.add_argument('--rows-dir', default=None)
    ap.add_argument('--prefix', default='desligado', help='rows file prefix in --rows-dir')
    ap.add_argument('--dump', default=None, help='pickle every record here')
    a = ap.parse_args()
    lost, kept = [], []
    for key, prep, truth, rows in tube_runs(a.tubes, a.seeds, a.shifts, a.rows_dir, a.prefix):
        mets = {str(m.get('id')): m for m in prep['mets']}
        by_id = {r['hmdb_id']: r for r in rows}
        for mid, t in truth.items():
            ev = annotation_evidence(mets.get(mid))
            r = by_id.get(mid)
            rec = {'run': key, 'mid': mid, 'name': t['name'], 'real': True,
                   'stage': _reason(r), **ev}
            if r is not None and not r.get('_not_quantified'):
                rec.update(row_evidence(r))
            (kept if rec['stage'] == 'number' else lost).append(rec)
        for mid, r in by_id.items():
            if mid in truth or r.get('_not_quantified') or r.get('_is_is'):
                continue
            kept.append({'run': key, 'mid': mid, 'name': r.get('metabolite'),
                         'real': False, 'stage': 'number',
                         **annotation_evidence(mets.get(mid)), **row_evidence(r)})
    if a.dump:
        pickle.dump({'lost': lost, 'kept': kept}, open(a.dump, 'wb'))
    print(f'{len(lost)} true analytes without a number; '
          f'{sum(1 for k in kept if k["real"])} with; '
          f'{sum(1 for k in kept if not k["real"])} strangers with one.\n')
    print('Lost, by stage:')
    for stage, n in Counter(r['stage'] for r in lost).most_common():
        print(f'  {n:4d}  {stage}')
    print('\nLost with no row, by annotation evidence:')
    nr = [r for r in lost if r['stage'] == 'no row']
    for lab, f in (('nothing matched', lambda r: r['matched'] == 0),
                   ('matched, none in study band', lambda r: r['matched'] and not r['in_study']),
                   ('only Bad matches', lambda r: r['bad_only']),
                   ('in study, not standing (cascade gate)', lambda r: r['in_study'] and not r['stands']),
                   ('standing in study', lambda r: r['stands'])):
        sel = [r for r in nr if f(r)]
        print(f'  {len(sel):4d}  {lab}  {sorted(Counter(x["name"] for x in sel).items())[:6]}')


if __name__ == '__main__':
    main()
