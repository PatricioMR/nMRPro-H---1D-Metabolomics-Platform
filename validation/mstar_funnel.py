# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Where analytes with complex multiplets (m*) win or are lost, and what
tells a real one from a stranger.

One record per annotated analyte with at least one m* signal on a GISSMO
tube: its signals by kind (confirmed letters, singlets, m* and whether the
annotation judged each m* Good), the m* pattern fit the annotation measured
(``pattern_fit``: SNR, share explained, R2), its distance to the study's
drift, the fingerprint verdicts -- and what the engine reported for it (the
rows of ``validation/out/bench_runs/<label>_...``).

    python -m validation.mstar_funnel --tubes 18 100 --seeds 0 1 2 --label base2
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'bench_runs')
LETTERS = {'d', 't', 'q', 'dd', 'dt', 'td', 'ddd', 'quint', 'sext', 'sept',
           'dq', 'qd', 'tt'}


def _kind(mp):
    m = str(mp.get('type') or mp.get('multiplicity') or '').lower().strip()
    return 'm*' if m == 'm*' else 's' if m == 's' else 'letter' if m in LETTERS else m


def records(mets, rows, truth):
    from processing.annotation_scoring import annotate_drift_spread
    annotate_drift_spread(mets)
    by = {r['hmdb_id']: r for r in rows if not r.get('_is_is')}
    out = []
    for met in mets:
        mps = [mp for mp in met.get('matched_peaks') or ()
               if mp.get('peak_rank') is not None]
        ms = [mp for mp in mps if _kind(mp) == 'm*']
        if not ms:
            continue
        pf = [mp.get('pattern_fit') or {} for mp in ms]
        good_ms = [mp for mp in ms if mp.get('annotation_quality') == 'good']
        lets = [mp for mp in mps if _kind(mp) == 'letter']
        r = by.get(met.get('id')) or {}
        q = bool(r) and not r.get('_not_quantified')
        off = [1000 * mp['ppm_off_drift'] for mp in ms
               if mp.get('ppm_off_drift') is not None]
        fpv = met.get('_fingerprint_verdicts') or []
        out.append({
            'id': met.get('id'), 'name': met.get('name'),
            'real': met.get('id') in truth,
            'n_listed': len(met.get('db_peaks') or ()),
            'n_mstar': len(ms), 'n_mstar_good': len(good_ms),
            'frac_mstar_good': len(good_ms) / max(len(ms), 1),
            'n_letter': len(lets),
            'n_letter_good': sum(1 for mp in lets if mp.get('annotation_quality') == 'good'),
            'n_singlet': sum(1 for mp in mps if _kind(mp) == 's'),
            'pf_snr_max': max((p.get('snr') or 0) for p in pf),
            'pf_snr_med': float(np.median([(p.get('snr') or 0) for p in pf])),
            'pf_expl_max': max((p.get('explained') or 0) for p in pf),
            'pf_expl_med': float(np.median([(p.get('explained') or 0) for p in pf])),
            'pf_r2_max': max((p.get('r2') or -1) for p in pf),
            'pf_r2_med': float(np.median([(p.get('r2') if p.get('r2') is not None else -1) for p in pf])),
            'off_min': min(off) if off else np.nan,
            'off_med': float(np.median(off)) if off else np.nan,
            'fp_contra': bool(met.get('_fingerprint_contradictions')),
            'fp_present': sum(1 for v in fpv if str(v[-1] if isinstance(v, (list, tuple)) else v).startswith('present')),
            'quantified': q, 'conf': int(r.get('_confidence') or 0) if q else 0,
            'mode': str(r.get('_integ_mode') or ''),
            'reason': str((r.get('_status_reasons') or [''])[0])[:80],
        })
    return out


def auc(v, lab):
    v = np.asarray(v, float); lab = np.asarray(lab, bool)
    ok = np.isfinite(v); v, lab = v[ok], lab[ok]
    p, n = v[lab], v[~lab]
    if not len(p) or not len(n):
        return float('nan')
    return float(np.mean([(pi > n).mean() + 0.5 * (pi == n).mean() for pi in p]))


def main():
    from validation import gissmo_bench as gb
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--tubes', nargs='+', default=['18', '100'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--label', default='base2')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    allrec = []
    for t in a.tubes:
        for sd in a.seeds:
            prep, truth = gb.prepared(gb.FIELDS[600], sd, t)
            rows = pickle.load(open(os.path.join(OUT, f'{a.label}_{t}_600_0_{sd}.pkl'), 'rb'))['rows']
            for r in records(prep['mets'], rows, truth):
                r.update(tube=t, seed=sd)
                allrec.append(r)
    if a.out:
        pickle.dump(allrec, open(a.out, 'wb'))
    for t in a.tubes:
        R = [r for r in allrec if r['tube'] == t]
        real = [r for r in R if r['real']]
        print(f'\n=== tube {t}: analytes with m* {len(R)} (real {len(real)}, strangers {len(R) - len(real)})')
        dep = [r for r in real if r['n_letter_good'] == 0]
        print(f'  real ones with NO confirmed letter (m*-dependent or singlet): {len(dep)}; '
              f'quantified {sum(r["quantified"] for r in dep)} '
              f'(>=2 dots {sum(r["conf"] >= 2 for r in dep)})')
        lab = [r['real'] for r in R]
        for f in ('n_mstar_good', 'frac_mstar_good', 'pf_snr_max', 'pf_snr_med',
                  'pf_expl_max', 'pf_expl_med', 'pf_r2_max', 'pf_r2_med',
                  'off_min', 'off_med', 'n_letter_good', 'n_singlet', 'fp_present'):
            sign = -1 if f.startswith('off') else 1
            print(f'  {f:16s} AUC {auc([sign * r[f] for r in R], lab):.2f}')


if __name__ == '__main__':
    main()
