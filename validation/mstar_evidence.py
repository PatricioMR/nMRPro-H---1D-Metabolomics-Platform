"""Evidence a complex multiplet (m*) carries that a stranger's shape does
not: candidates for judging one, measured against GISSMO tubes.

For every m* signal the annotation matched to a peak:

* ``d2corr`` -- correlation of the second derivative of the reference
  pattern with that of the mean trace over the pattern's reach, at the best
  offset within half a linewidth of where the annotation put it. The second
  derivative keeps lines and valleys and drops the broad sum of
  neighbours, which is why it is used on overlapped multiplets
  (e.g. Savitzky A, Golay MJE. Smoothing and differentiation of data by
  simplified least squares procedures. Anal Chem 1964;36:1627);
* ``halves_r`` -- across the samples, the correlation of the amounts the
  pattern's left and right halves read (least squares, each half alone).
  One compound's lines rise and fall together; a stranger's halves stand
  on different compounds;
* ``corr`` -- the plain correlation, for comparison.

    python -m validation.mstar_evidence --tubes 18 100 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import savgol_filter

from validation.global_fit_probe import library, render
from validation.mstar_funnel import auc, _kind


def signal_evidence(ppm, Y, y, sig, mp, *, field, fwhm_hz, lw_ppm):
    """``(d2corr, halves_r, corr)`` for one m* signal, or None."""
    c0 = float(mp['db_ppm']); pk = float(mp['peak_ppm'])
    base_shift = pk - c0
    step = abs(float(ppm[1] - ppm[0]))
    win = max(5, int(round(0.5 * fwhm_hz / field / step)) | 1)
    d2y = savgol_filter(y, win, 2, deriv=2)
    best = None
    for d in np.arange(-0.5 * lw_ppm, 0.5 * lw_ppm + 1e-12, step):
        t = render(ppm, [sig], base_shift + d - (sig[0] - c0), field, fwhm_hz)
        if t.max() <= 0:
            return None
        sup = t > 0.05 * t.max()
        if sup.sum() < 8:
            return None
        d2t = savgol_filter(t, win, 2, deriv=2)
        r2 = float(np.corrcoef(d2t[sup], d2y[sup])[0, 1])
        if best is None or r2 > best[0]:
            best = (r2, t, sup)
    r2, t, sup = best
    corr = float(np.corrcoef(t[sup], y[sup])[0, 1])
    idx = np.flatnonzero(sup)
    # halves by the template's area, not by its width
    cum = np.cumsum(t[idx]); cut = idx[np.searchsorted(cum, cum[-1] / 2)]
    left = sup & (np.arange(ppm.size) <= cut)
    right = sup & (np.arange(ppm.size) > cut)
    if left.sum() < 3 or right.sum() < 3:
        return r2, np.nan, corr
    tl, tr = t[left], t[right]
    al = Y[left].T @ tl / float(tl @ tl)
    ar = Y[right].T @ tr / float(tr @ tr)
    hr = float(np.corrcoef(al, ar)[0, 1]) if al.std() > 0 and ar.std() > 0 else np.nan
    return r2, hr, corr


def collect(tubes, seeds):
    from validation import gissmo_bench as gb, mixture_bench as mb
    lib = library()
    out = []
    for tube in tubes:
        for sd in seeds:
            prep, truth = gb.prepared(gb.FIELDS[600], sd, tube)
            ppm = prep['ppm']; Y = prep['corr']; y = Y.mean(axis=1)
            field = gb.FIELDS[600]; lw = mb.LINEWIDTH_HZ / field
            for met in prep['mets']:
                ent = lib.get(str(met.get('id')))
                if ent is None:
                    continue
                for mp in met.get('matched_peaks') or ():
                    if (_kind(mp) != 'm*' or mp.get('peak_ppm') is None
                            or mp.get('db_ppm') is None):
                        continue
                    sig = min(ent['signals'], key=lambda s: abs(s[0] - float(mp['db_ppm'])))
                    if abs(sig[0] - float(mp['db_ppm'])) > 0.01:
                        continue
                    ev = signal_evidence(ppm, Y, y, sig, mp, field=field,
                                         fwhm_hz=mb.LINEWIDTH_HZ, lw_ppm=lw)
                    if ev is None:
                        continue
                    out.append({'tube': tube, 'seed': sd, 'id': met.get('id'),
                                'name': met.get('name'), 'real': met.get('id') in truth,
                                'db_ppm': float(mp['db_ppm']),
                                'good': mp.get('annotation_quality') == 'good',
                                'd2corr': ev[0], 'halves_r': ev[1], 'corr': ev[2]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--tubes', nargs='+', default=['18', '100'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out')
    a = ap.parse_args()
    rec = collect(a.tubes, a.seeds)
    if a.out:
        pickle.dump(rec, open(a.out, 'wb'))
    for t in a.tubes:
        R = [r for r in rec if r['tube'] == t]
        lab = [r['real'] for r in R]
        print(f"tube {t}: m* signals {len(R)} (real {sum(lab)})")
        for f in ('corr', 'd2corr', 'halves_r'):
            print(f"   {f:9s} AUC {auc([r[f] for r in R], lab):.2f}")


if __name__ == '__main__':
    main()
