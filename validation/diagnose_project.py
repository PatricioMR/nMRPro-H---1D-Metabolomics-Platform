"""Annotate and quantify a saved project the way the application does,
and, when the study's true concentrations are known, say how it did.

The bench's own script builds a spectrum and runs the engine on it,
leaving out everything the analyst does by hand -- the alignment, the
baseline, the exclusions. This takes a project that has been through
those hands and carries on from where it was saved, using the settings
the project itself holds: its peaks, its noise, its excluded regions.

    python -m validation.diagnose_project [path.nmrproj] [--noise N]

With ``--noise`` the run uses that noise level instead of the project's,
which is how to see what the estimate is worth.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gui.stages._annotation_helpers import ambiguous_peak_ranks
from gui.stages._quant_adapter import quantify_project
from processing.spectral_regions import parse_regions
from validation import mixture_bench as mb
from validation.app_annotation import annotate as app_annotate

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out',
                       'MixtureBench_v01.nmrproj')


def measured_noise(ppm, X) -> float:
    """The noise where the spectrum has no signal, as a check on what the
    project was told."""
    m = (ppm >= 9.0) & (ppm <= 10.0)
    return float(np.asarray(X)[m].std()) if m.sum() > 100 else float('nan')


def run(path, *, noise=None, field_mhz=None, tolerance=None, log=print):
    st = pickle.load(open(path, 'rb'))
    ppm = np.asarray(st['ppm'], dtype=float)
    X = np.asarray(st['all_intensities'], dtype=float)
    si = st['samples_info']
    pd = st.get('pd') or {}
    peaks = list(pd.get('peaks') or ())
    if not peaks:
        raise SystemExit('the project has no peaks: run detection first')
    excl = parse_regions(pd.get('excluded_regions_applied')
                         or pd.get('excluded_regions') or '') or ()
    noise_std = float(noise if noise is not None else (pd.get('noise_std') or 0))
    min_snr = float(pd.get('min_snr_var') or 6.0)
    min_dist = float(pd.get('min_width_var') or 0.003)
    mhz = float(field_mhz or mb.MHZ)
    tol = float(tolerance or mb.TOLERANCE)

    log(f'{len(peaks)} peaks, noise {noise_std:,.0f} '
        f'(the spectrum measures {measured_noise(ppm, X):,.0f}), '
        f'aligned={st.get("alignment_corrected")}, excluded {excl}')

    with contextlib.redirect_stdout(io.StringIO()):
        mets, params = app_annotate(
            peaks, ppm, X, si, db_path=mb.DB_DEFAULT, field_mhz=mhz,
            tolerance_ppm=tol, noise_std=noise_std, min_snr=min_snr,
            min_distance_ppm=min_dist, excluded_regions=excl)
        ref = min(peaks, key=lambda p: abs(float(p['ppm']) - mb.STANDARD_PPM))
        rows = quantify_project(
            metabolites=mets, peaks=peaks, ppm=ppm, intensities=X,
            samples_info=si, ref_peak=ref, ref_nh=mb.STANDARD_NH,
            ref_conc=mb.STANDARD_MM, ref_name='DSS', noise_sigma=noise_std,
            ambiguous_ranks=ambiguous_peak_ranks(mets, manual_quality={}),
            use_pqn=False, local_baseline=False, quantify_shapes=True,
            deconvolve=False, resolve_shared=True, cascade=True,
            excluded_regions=excl, annotation_params=params)
    return st, mets, rows


def diagnose(rows, truth, log=print):
    by_name = {v['name']: k for k, v in truth.items()}
    got = {r['hmdb_id']: r for r in rows
           if not r.get('_not_quantified') and not r.get('_is_is')}
    found, missed, errs = [], [], []
    for mid, t in sorted(truth.items(), key=lambda kv: -kv[1]['A']):
        r = got.get(mid)
        if r is None:
            missed.append(t['name'])
            continue
        groups = r.get('_concs_by_group') or {}
        a = float(np.median(groups.get('A') or [np.nan]))
        b = float(np.median(groups.get('B') or [np.nan]))
        ea = (a / t['A'] - 1) * 100 if t['A'] else float('nan')
        eb = (b / t['B'] - 1) * 100 if t['B'] else float('nan')
        found.append((t['name'], t['A'], a, ea, t['B'], b, eb,
                      r.get('_identity_level'), r.get('_confidence'),
                      r.get('_integ_mode')))
        errs += [abs(ea), abs(eb)]
    false = sorted((r['metabolite'], float(r.get('conc_mm') or 0))
                   for mid, r in got.items() if mid not in truth)
    log(f'\nfound {len(found)}/{len(truth)}   false positives {len(false)}')
    if errs:
        log(f'error: median {np.median(errs):.0f}%  '
            f'p90 {np.percentile(errs, 90):.0f}%  '
            f'bias {np.mean([e for e in errs]):+.0f}%')
    log(f'\n{"analyte":26s} {"A true":>7s} {"A got":>7s} {"err":>6s} '
        f'{"B true":>7s} {"B got":>7s} {"err":>6s}  id/conf  how')
    for n, ta, a, ea, tb, b, eb, idl, cf, mode in found:
        log(f'{n[:26]:26s} {ta:7.3f} {a:7.3f} {ea:+5.0f}% '
            f'{tb:7.3f} {b:7.3f} {eb:+5.0f}%    {idl}/{cf}  {mode}')
    if missed:
        log(f'\nmissed: {", ".join(missed)}')
    if false:
        log('\nfalse positives, and what the software says about each:')
        _by = {r['metabolite']: r for r in rows}
        _weak = 0
        for n, c in sorted(false, key=lambda v: -v[1]):
            r = _by.get(n) or {}
            loq = r.get('_loq_mm')
            under = (loq is not None and np.isfinite(loq) and c < float(loq))
            _weak += int(bool(r.get('_confidence') == 1 or under))
            log(f'  {n[:30]:30s} {c:7.3f} mM  id/conf '
                f'{r.get("_identity_level")}/{r.get("_confidence")}  '
                f'{"under its own LOQ" if under else "":17s} '
                f'{str(r.get("_identity_why") or "")[:44]}')
        log(f'  -> {_weak} of {len(false)} are at one dot or under their own '
            f'limit of quantification')
    return found, missed, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default=DEFAULT)
    ap.add_argument('--noise', type=float, default=None)
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        _X, _s, truth, _f = mb.build_study(mb.DB_DEFAULT, seed=0)
    _st, _mets, rows = run(a.path, noise=a.noise)
    diagnose(rows, truth)


if __name__ == '__main__':
    main()
