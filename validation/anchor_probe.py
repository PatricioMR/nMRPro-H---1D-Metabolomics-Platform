# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Probe: a confirmed multiplet fixes the amount; the analyte's other
signals are looked for at that amount, wherever they stand nearby.

Proposed by the user (2026-10-11): inside a ppm window, which signals
represent the same quantity as a reference signal whose J is right? On
the Mice ISS skin extract the confirmed signals of plausible analytes
stood a median 5.4 mppm (p90 35) from listing + the study's drift, with a
linewidth of 4.3: placed at the listing, a shape test misses most of
them. The amount, unlike the position, does not move with the matrix.

Rules, fixed before the results were looked at:

* reference: a matched signal the annotation confirmed (Good) with a
  letter multiplicity and every expected line found; of several, the one
  with the lowest ceiling (overlap only adds; ADR-069). Its amount per
  proton is the least-squares copy of its template on the mean trace,
  held under the trace;
* every other listed signal of the analyte, templated from its measured
  pattern (validation.global_fit_probe.library) at that amount, is slid
  within +-``--window`` of listing + the analyte's own drift: *found*
  where the copy fits under the trace (>= 90% of its area below trace +
  3 sigma) and correlates with it (>= ``--corr``); *contradicted* where
  the copy would stand >= 10 sigma and fits nowhere; *small* under 5
  sigma; *unclear* otherwise (or with its top in an excluded region);
* identified: a reference, >= 1 found, 0 contradicted.

    python -m validation.anchor_probe tube --size 100 --seeds 0 1 2
    python -m validation.anchor_probe prep PREP.pkl --project PATH

Measured 2026-10-11 (GISSMO 600, 3 seeds): alone, tube of 100 68 real /
33 false (tube of 18 13/5). Against the engine's rows of the tube of 100:
confirms 63 real / 3 false; of what the engine does not report it finds
5 real / 30 false; "contradicted" falls on 22 real of 29 -- the amount the
reference fixes does not predict the other signals' heights through these
templates and their overlaps, on the bench as on the Mice ISS extract
(lactate's quartet contradicted by its own doublet). A confirmation, not a
finder (vault: Diagnostico-Mice-v06 section 10).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import project_file
from validation.global_fit_probe import library, quiet_sigma, render

LETTERS = {'d', 't', 'q', 'dd', 'dt', 'td', 'ddd', 'quint', 'sext', 'sept',
           'dq', 'qd', 'tt'}


def _reference(met):
    """The analyte's confirmed letter multiplets: [(db_ppm, peak_ppm)]."""
    out = []
    for mp in met.get('matched_peaks') or ():
        mult = str(mp.get('type') or mp.get('multiplicity') or '').lower().strip()
        nf, ne = mp.get('_n_found_lines'), mp.get('_n_exp_lines')
        if (mult in LETTERS and mp.get('annotation_quality') == 'good'
                and mp.get('peak_ppm') is not None and mp.get('db_ppm') is not None
                and nf is not None and ne is not None and int(ne) >= 2
                and int(nf) >= int(ne)):
            out.append((float(mp['db_ppm']), float(mp['peak_ppm'])))
    return out


def judge(ppm, y, sigma, met, ent, *, field, fwhm_hz, mask, window=0.035,
          corr_min=0.7):
    """``None`` without a reference, else a dict of the verdicts."""
    refs = _reference(met)
    if not refs:
        return None
    sigs = ent['signals']
    step = abs(float(ppm[1] - ppm[0]))

    def sig_near(c):
        return min(sigs, key=lambda s: abs(s[0] - c))

    cand = []
    for db_c, pk_c in refs:
        s = sig_near(db_c)
        if abs(s[0] - db_c) > 0.01:
            continue
        sh = pk_c - s[0]
        # its own position, refined within half a linewidth
        best = None
        for d in np.arange(-0.002, 0.0021, step):
            t = render(ppm, [s], sh + d, field, fwhm_hz)
            sup = (t > 0.05 * t.max()) & mask
            if sup.sum() < 4:
                continue
            a = float(t[sup] @ y[sup]) / float(t[sup] @ t[sup])
            ceil = float(np.min(np.maximum(y[sup & (t > 0.5 * t.max())], 0)
                                / t[sup & (t > 0.5 * t.max())]))
            a = max(min(a, ceil), 0.0)
            r = float(np.corrcoef(t[sup], y[sup])[0, 1])
            if best is None or r > best[0]:
                best = (r, a, sh + d, s, ceil)
        if best is not None:
            cand.append(best)
    if not cand:
        return None
    # the lowest ceiling among the clean ones speaks for the amount
    r_ref, amp, drift, s_ref, _c = min(cand, key=lambda b: b[4])
    verdicts = []
    for s in sigs:
        if s is s_ref:
            continue
        base = render(ppm, [s], drift, field, fwhm_hz) * amp
        top = base > 0.5 * base.max() if base.max() > 0 else None
        if top is None or not mask[top].all():
            verdicts.append((s[0], 'unclear', None))
            continue
        h = float(base.max()) / sigma
        if h < 5:
            verdicts.append((s[0], 'small', h))
            continue
        fits_any, best_r = False, -1.0
        n_w = int(round(window / step))
        for n in range(-n_w, n_w + 1, 2):
            t = np.roll(base, n)
            sup = (t > 0.05 * t.max()) & mask
            if sup.sum() < 4:
                continue
            under = float(np.clip(np.minimum(t[sup], y[sup] + 3 * sigma), 0, None).sum()
                          / max(t[sup].sum(), 1e-30))
            if under < 0.9:
                continue
            fits_any = True
            r = float(np.corrcoef(t[sup], y[sup])[0, 1])
            best_r = max(best_r, r)
        if fits_any and best_r >= corr_min:
            verdicts.append((s[0], 'found', h))
        elif not fits_any and h >= 10:
            verdicts.append((s[0], 'contradicted', h))
        else:
            verdicts.append((s[0], 'unclear', h))
    n_f = sum(1 for v in verdicts if v[1] == 'found')
    n_c = sum(1 for v in verdicts if v[1] == 'contradicted')
    return {'ref_ppm': s_ref[0], 'ref_r': r_ref, 'amp': amp, 'found': n_f,
            'contradicted': n_c, 'verdicts': verdicts,
            'identified': n_f >= 1 and n_c == 0}


def run_mets(ppm, y, sigma, mets, *, field, fwhm_hz, mask, **kw):
    lib = library()
    out = {}
    for met in mets:
        mid = str(met.get('id'))
        ent = lib.get(mid)
        if ent is None:
            continue
        j = judge(ppm, y, sigma, met, ent, field=field, fwhm_hz=fwhm_hz,
                  mask=mask, **kw)
        if j is not None:
            out[mid] = dict(j, name=met.get('name'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('what', choices=('tube', 'prep'))
    ap.add_argument('prep_path', nargs='?')
    ap.add_argument('--project')
    ap.add_argument('--size', default='100')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--window', type=float, default=0.035)
    ap.add_argument('--corr', type=float, default=0.7)
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.what == 'tube':
        from validation import gissmo_bench as gb, mixture_bench as mb
        tot = {'tp': 0, 'fp': 0, 'n': 0}
        res = []
        for sd in a.seeds:
            prep, truth = gb.prepared(gb.FIELDS[600], sd, a.size)
            ppm = prep['ppm']; y = prep['corr'].mean(axis=1)
            sig = quiet_sigma(ppm, y, 9.6, 9.95)
            mask = (ppm > 0.5) & (ppm < 9.5) & ~((ppm > 4.5) & (ppm < 5.0))
            got = run_mets(ppm, y, sig, prep['mets'], field=gb.FIELDS[600],
                           fwhm_hz=mb.LINEWIDTH_HZ, mask=mask,
                           window=a.window, corr_min=a.corr)
            ids = {m for m, j in got.items() if j['identified']}
            tp, fp = len(ids & set(truth)), len(ids - set(truth))
            print(f'tube {a.size} s{sd}: judged {len(got)}  identified {len(ids)}'
                  f'  real {tp}/{len(truth)}  false {fp}')
            tot['tp'] += tp; tot['fp'] += fp; tot['n'] += len(truth)
            res.append((sd, got, truth))
        print(f"TOTAL real {tot['tp']}/{tot['n']}  false {tot['fp']}  "
              f"precision {100 * tot['tp'] / max(tot['tp'] + tot['fp'], 1):.0f}%")
        if a.out:
            pickle.dump(res, open(a.out, 'wb'))
    else:
        from processing.baseline_model import BaselineCache, BaselineModel
        d = pickle.load(open(a.prep_path, 'rb'))['prep']
        st = project_file.read(a.project)
        ppm = d['ppm']; X = d['raw']
        C = np.asarray(BaselineCache().corrected(
            ppm, X, BaselineModel.from_dict(st['baseline_model']),
            already_corrected=False), float)
        S = [i for i, s in enumerate(st['samples_info']) if s['type'] != 'BLK']
        y = C[:, S].mean(axis=1)
        sig = quiet_sigma(ppm, y, 10.0, 12.0)
        mask = (ppm > 0.5) & (ppm < 9.5) & ~((ppm > 4.5) & (ppm < 5.0))
        got = run_mets(ppm, y, sig, d['mets'], field=600.71, fwhm_hz=2.6,
                       mask=mask, window=a.window, corr_min=a.corr)
        if a.out:
            pickle.dump(got, open(a.out, 'wb'))
        for mid, j in sorted(got.items(), key=lambda kv: kv[1]['name']):
            print(f"{'ID ' if j['identified'] else '   '}{j['name'][:30]:30s} ref {j['ref_ppm']:.3f} "
                  f"found {j['found']} contradicted {j['contradicted']}  "
                  + ' '.join(f"{p:.2f}:{v[0]}" for p, v, _h in j['verdicts']))


if __name__ == '__main__':
    main()
