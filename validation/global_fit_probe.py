"""Probe: identify by fitting every reference template at once.

The engine identifies peak first -- detect, match each peak to candidates,
elect one per peak, then measure. The tools that work in production fit
the whole library to the spectrum together and let identity come out of
the fit (Tardivel PJC, Canlet C, Lefort G, et al. ASICS: an automatic
method for identification and quantification of metabolites in complex 1D
1H NMR spectra. Metabolomics 2017;13:109; Lefort G, Liaubet L, Canlet C,
et al. ASICS: an R package for a whole analysis workflow of 1D 1H NMR
spectra. Bioinformatics 2019;35:4356). This probe asks whether doing so
would identify better, before any of it goes near the engine.

Templates come from the reference database alone: every listed signal as
its measured pattern (``pattern_lines``, Hz offsets and relative heights
read on the BMRB spectrum), Lorentzian lines of the study's linewidth,
scaled to the signal's protons. Steps, as in ASICS:

1. each compound is moved by the shift (one per compound, within
   ``--max-shift`` of the study drift) that best correlates its template
   with the spectrum;
2. a compound one of whose major signals would stand where the spectrum
   holds nothing at the amount its own best signal implies is removed
   (it cannot be there);
3. the remaining templates are fitted together by non-negative least
   squares to the mean spectrum;
4. identified: a compound whose fitted template stands at least ``--k``
   sigma tall and whose removal worsens the fit by at least ``--z``
   standard deviations of the residual over its support.

    python -m validation.global_fit_probe tube --size 100 --seeds 0 1 2
    python -m validation.global_fit_probe mice --project PATH

Measured 2026-10-11 (GISSMO 600, 3 seeds; ``fit.last_info`` holds the
features): the ceiling (each signal bounds the amount from above, the
compound is no more than its lowest) brings 265 of 300 real analytes of the
tube of 100 into the fit, but the best separating feature, how well the
spectrum left by the others correlates with the template (AUC 0.79 on the
tube of 18), gives at >= 0.6 only 98 real / 17 false out of sample, where
the engine reports 185 / 21 at two dots or more. On the Mice ISS skin
extract: 37 names, 18 plausible. Templates read at 500 MHz on the BMRB
spectra do not reproduce a 600 MHz extract closely enough to decide by
shape alone; not adopted (vault: Diagnostico-Mice-v06).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import nnls

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'databases', 'BMRB_Metabolomics_Mammalian_v05.db')

LINE = {'s': [(0.0, 1.0)], 'd': [(-.5, 1.0), (.5, 1.0)],
        't': [(-1, 1.0), (0, 2.0), (1, 1.0)],
        'q': [(-1.5, 1.0), (-.5, 3.0), (.5, 3.0), (1.5, 1.0)]}


def library(db=DB):
    """``{id: {'name', 'signals': [(ppm, protons, [(hz, rel), ...])]}}``."""
    con = sqlite3.connect(db)
    names = dict(con.execute('SELECT bmrb_id, name FROM metabolites'))
    lib = {}
    for bid, ppm, mult, j, nh, pl in con.execute(
            'SELECT bmrb_id, ppm, multiplicity, j_use, protons, pattern_lines '
            'FROM shifts_1h'):
        if ppm is None:
            continue
        nh = float(nh or 1.0)
        lines = None
        if pl:
            try:
                lines = [(float(a), float(b)) for a, b in json.loads(pl)]
            except (ValueError, TypeError):
                lines = None
        if not lines:
            m = str(mult or 's').lower().strip()[:1]
            base = LINE.get(m, LINE['s'])
            jj = float(j or 7.0)
            lines = [(o * jj, h) for o, h in base]
        lib.setdefault(bid, {'name': names.get(bid, bid), 'signals': []})
        lib[bid]['signals'].append((float(ppm), nh, lines))
    return lib


def render(ppm, signals, shift, field, fwhm_hz):
    """Template on ``ppm``: each signal's lines, area = protons."""
    w = 0.5 * fwhm_hz / field                    # half width, ppm
    out = np.zeros_like(ppm)
    for c, nh, lines in signals:
        tot = sum(h for _o, h in lines) or 1.0
        for o, h in lines:
            x0 = c + shift + o / field
            out += (nh * h / tot) * (w / np.pi) / ((ppm - x0) ** 2 + w ** 2)
    return out


def fit(ppm, y, sigma, lib, *, field, fwhm_hz, drift, max_shift, mask,
        k=5.0, z=5.0, absent_frac=0.25, log=print):
    """Identification by a global fit; returns ``[(id, amp, height_sigma, z)]``."""
    step = abs(float(ppm[1] - ppm[0]))
    shifts = np.arange(drift - max_shift, drift + max_shift + 1e-12, step * 2)
    yy = np.where(mask, y, 0.0)
    cols, ids, info = [], [], {}
    for mid, ent in lib.items():
        base = render(ppm, ent['signals'], 0.0, field, fwhm_hz)
        if base.max() <= 0:
            continue
        # 1. its own shift: best correlation of template with the spectrum
        best, bsh = -np.inf, 0.0
        sup0 = base > 0.02 * base.max()
        for sh in shifts:
            n = int(round(sh / (ppm[1] - ppm[0])))
            t = np.roll(base, n)
            sup = np.roll(sup0, n) & mask
            if sup.sum() < 5:
                continue
            tt, yv = t[sup], yy[sup]
            sc = float(tt @ yv) / (np.linalg.norm(tt) * np.linalg.norm(yv) + 1e-30)
            if sc > best:
                best, bsh = sc, sh
        t = render(ppm, ent['signals'], bsh, field, fwhm_hz)
        t = np.where(mask, t, 0.0)
        if t.max() <= 0:
            continue
        # 2. a ceiling: overlap only adds, so every signal of the compound
        # bounds its amount from above by what the spectrum holds under the
        # signal's top; the compound can be no more than the lowest of them.
        # Signals whose top falls in an excluded region say nothing.
        ceil = np.inf
        for c, nh, lines in ent['signals']:
            ts = render(ppm, [(c, nh, lines)], bsh, field, fwhm_hz)
            top = ts > 0.5 * ts.max()
            if ts.max() <= 0 or not mask[top].all():
                continue
            ceil = min(ceil, float(np.min(np.maximum(yy[top], 0) / ts[top])))
        if not np.isfinite(ceil) or ceil * t.max() < k * sigma:
            continue            # at the most it could be, it would not be seen
        cols.append(t)
        ids.append(mid)
        info[mid] = {'shift': bsh, 'corr': best, 'ceil': ceil}
    A = np.column_stack(cols)[mask]
    b = y[mask]
    from scipy.optimize import lsq_linear
    ub = np.array([1.05 * info[m]['ceil'] for m in ids])
    amp = lsq_linear(A, b, bounds=(np.zeros(len(ids)), ub), method='bvls',
                     max_iter=50 * len(ids)).x
    resid = b - A @ amp
    out = []
    for j, mid in enumerate(ids):
        if amp[j] <= 0:
            continue
        col = A[:, j] * amp[j]
        h = float(col.max()) / sigma
        sup = col > 0.05 * col.max()
        # fit without it, the others held: what its absence adds to the misfit
        r_wo = resid[sup] + col[sup]
        dz = (float(r_wo @ r_wo) - float(resid[sup] @ resid[sup])) / (
            sigma ** 2 * np.sqrt(2 * max(sup.sum(), 1)))
        # the spectrum left for it once the others are taken: does it hold
        # this compound's shape (correlation), and how much of it is this
        rest = b[sup] - (A[sup] @ amp - col[sup])
        cc = float(np.corrcoef(col[sup], rest)[0, 1]) if sup.sum() > 3 else 0.0
        own = float(col[sup].sum() / max(np.maximum(rest, 0).sum(), 1e-30))
        info[mid].update(height=h, dz=dz, rest_corr=cc, own=own,
                         at_ceiling=float(amp[j] / info[mid]['ceil']),
                         n_signals=len(lib[mid]['signals']))
        if h >= k and dz >= z:
            out.append((mid, float(amp[j]), h, dz))
    fit.last_info = {m: info[m] for m, *_ in out}
    log(f'  templates {len(lib)} kept after absence {len(ids)} '
        f'fitted >0 {int((amp > 0).sum())} identified {len(out)}')
    return out


def quiet_sigma(ppm, y, lo, hi):
    k = (ppm >= lo) & (ppm <= hi)
    v = y[k]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def run_tube(a):
    from validation import gissmo_bench as gb
    lib = library()
    res = []
    for seed in a.seeds:
        X, truth = gb.tube(gb.FIELDS[600], seed, a.size)
        ppm = gb.PPM
        y = X.mean(axis=1)
        sig = quiet_sigma(ppm, y, 9.6, 9.95)
        mask = (ppm > 0.5) & (ppm < 9.5) & ~((ppm > 4.5) & (ppm < 5.0))
        from validation import mixture_bench as mb
        got = fit(ppm, y, sig, lib, field=600.0, fwhm_hz=mb.LINEWIDTH_HZ,
                  drift=a.drift, max_shift=a.max_shift, mask=mask, k=a.k, z=a.z)
        ids = {g[0] for g in got}
        tp = len(ids & set(truth)); fp = len(ids - set(truth))
        print(f'tube {a.size} s{seed}: found {tp}/{len(truth)}  false {fp}')
        res.append((tp, fp, len(truth)))
    tp, fp, n = (sum(r[i] for r in res) for i in range(3))
    print(f'TOTAL found {tp}/{n} ({100 * tp / n:.0f}%)  false {fp}  '
          f'precision {100 * tp / max(tp + fp, 1):.0f}%')


def run_mice(a):
    import pickle
    from processing.baseline_model import BaselineCache, BaselineModel
    st = pickle.load(open(a.project, 'rb'))
    ppm = np.asarray(st['ppm'], float)
    X = np.load(a.npy) if a.npy else np.asarray(st['all_intensities'], float)
    C = np.asarray(BaselineCache().corrected(
        ppm, X, BaselineModel.from_dict(st['baseline_model']),
        already_corrected=False), float)
    si = st['samples_info']
    S = [i for i, s in enumerate(si) if s['type'] != 'BLK']
    y = C[:, S].mean(axis=1)
    sig = quiet_sigma(ppm, y, 10.0, 12.0)
    mask = (ppm > 0.5) & (ppm < 9.5) & ~((ppm > 4.5) & (ppm < 5.0))
    lib = library()
    got = fit(ppm, y, sig, lib, field=600.71, fwhm_hz=a.fwhm, drift=a.drift,
              max_shift=a.max_shift, mask=mask, k=a.k, z=a.z)
    got.sort(key=lambda g: -g[3])
    names = [(lib[g[0]]['name'], round(g[2]), round(g[3])) for g in got]
    if a.out:
        json.dump(names, open(a.out, 'w'), indent=0)
    for n in names:
        print(f'  {n[0][:40]:40s} height {n[1]:6d} sigma   z {n[2]:6d}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('what', choices=('tube', 'mice'))
    ap.add_argument('--size', default='100')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--project')
    ap.add_argument('--npy', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--drift', type=float, default=0.0)
    ap.add_argument('--max-shift', type=float, default=0.02)
    ap.add_argument('--fwhm', type=float, default=2.6)
    ap.add_argument('--k', type=float, default=5.0)
    ap.add_argument('--z', type=float, default=5.0)
    a = ap.parse_args()
    {'tube': run_tube, 'mice': run_mice}[a.what](a)


if __name__ == '__main__':
    main()
