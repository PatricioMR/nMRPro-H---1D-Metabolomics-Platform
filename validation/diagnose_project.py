# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import project_file
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


def run(path, *, noise=None, field_mhz=None, tolerance=None,
        local_baseline=None, baseline=None, db_path=None, params=None,
        singlet_min_snr=None, log=print):
    """Run the project through the application's own engines.

    ``params`` overrides the scoring parameters the analyst would set in
    the Annotation Parameters dialog, so a sweep can ask what a different
    gate would have done to this study.

    ``db_path`` is the reference the ANNOTATION searches, which need not be
    the one the spectrum was built from. Pointing them at different
    databases is the honest experiment: a real spectrum is not made out of
    the listing used to annotate it, so a bench that searches the same
    table it drew the peaks from is measuring the engine on the easy case.
    The truth stays with whoever built the study.
    """
    st = project_file.read(path)
    ppm = np.asarray(st['ppm'], dtype=float)
    # The project's baseline model, applied as the application applies it.
    # quantify_project wants the corrected matrix: handing it the raw one
    # integrates the broad component the model was there to remove, and
    # does it silently.
    from processing.baseline_model import BaselineModel, BaselineCache
    raw = np.asarray(st['all_intensities'], dtype=float)
    _bl = st.get('baseline_model')
    model = BaselineModel.from_dict(_bl) if _bl else None
    if baseline is not None and model is not None:
        model.enabled = bool(baseline)
    X = BaselineCache().corrected(
        ppm, raw, model,
        already_corrected=bool(st.get('baseline_corrected', False)))
    X = np.asarray(X, dtype=float)
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
    qnt = st.get('qnt') or {}
    local_bl = (bool(qnt.get('local_baseline'))
                if local_baseline is None else bool(local_baseline))

    _moved = float(np.abs(X - raw).max())
    log(f'{len(peaks)} peaks, noise {noise_std:,.0f} '
        f'(the spectrum measures {measured_noise(ppm, X):,.0f}), '
        f'aligned={st.get("alignment_corrected")}, excluded {excl}')
    log(f'  baseline model {"applied" if _moved > 0 else "does nothing"}'
        f' (largest change {_moved:,.0f}), local baseline {local_bl}')

    with contextlib.redirect_stdout(io.StringIO()):
        mets, params = app_annotate(
            peaks, ppm, X, si, db_path=db_path or mb.DB_DEFAULT, field_mhz=mhz,
            tolerance_ppm=tol, noise_std=noise_std, min_snr=min_snr,
            min_distance_ppm=min_dist, excluded_regions=excl,
            params=params)
        ref = min(peaks, key=lambda p: abs(float(p['ppm']) - mb.STANDARD_PPM))
        rows = quantify_project(
            metabolites=mets, peaks=peaks, ppm=ppm, intensities=X,
            samples_info=si, ref_peak=ref, ref_nh=mb.STANDARD_NH,
            ref_conc=mb.STANDARD_MM, ref_name='DSS', noise_sigma=noise_std,
            ambiguous_ranks=ambiguous_peak_ranks(mets, manual_quality={}),
            singlet_min_snr=singlet_min_snr,
            use_pqn=False, local_baseline=local_bl, quantify_shapes=True,
            deconvolve=False, resolve_shared=True, cascade=True,
            excluded_regions=excl, annotation_params=params,
            raw_intensities=raw)
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
    broad = {r['metabolite']: float(r['_broad_removed'])
             for r in rows if r.get('_broad_removed')}
    for n, ta, a, ea, tb, b, eb, idl, cf, mode in found:
        log(f'{n[:26]:26s} {ta:7.3f} {a:7.3f} {ea:+5.0f}% '
            f'{tb:7.3f} {b:7.3f} {eb:+5.0f}%    {idl}/{cf}  {mode}'
            + (f'  baseline took {100 * broad[n]:.0f}%'
               if broad.get(n, 0) >= 0.05 else ''))
    if missed:
        log(f'\nmissed: {", ".join(missed)}')
    if false:
        log('\nfalse positives, and what the software says about each:')
        _by = {r['metabolite']: r for r in rows}
        _weak = 0
        for n, c in sorted(false, key=lambda v: -v[1]):
            r = _by.get(n) or {}
            _weak += int(r.get('_confidence') == 1)
            log(f'  {n[:30]:30s} {c:7.3f} mM  id/conf '
                f'{r.get("_identity_level")}/{r.get("_confidence")}  '
                f'{str(r.get("_identity_why") or "")[:44]}')
        log(f'  -> {_weak} of {len(false)} are at one dot')
    return found, missed, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default=DEFAULT)
    ap.add_argument('--noise', type=float, default=None)
    ap.add_argument('--baseline', dest='baseline', action='store_true',
                    default=None,
                    help="apply the project's baseline model even if it is "
                         'off; --no-baseline to skip it even if it is on')
    ap.add_argument('--no-baseline', dest='baseline', action='store_false')
    ap.add_argument('--local-baseline', dest='local_baseline',
                    action='store_true', default=None,
                    help="subtract a line under each window, whatever the "
                         'project says')
    ap.add_argument('--db', default=None,
                    help='the reference the annotation searches; the study '
                         'was built from mixture_bench.DB_DEFAULT and the '
                         'truth always comes from there')
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        _X, _s, truth, _f = mb.build_study(mb.DB_DEFAULT, seed=0)
    _st, _mets, rows = run(a.path, noise=a.noise,
                           local_baseline=a.local_baseline,
                           baseline=a.baseline, db_path=a.db)
    diagnose(rows, truth)


if __name__ == '__main__':
    main()
