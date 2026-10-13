"""How much of each study the analysis explains, beside what it reports.

Quantifies, the way a new user's run does (``fresh_run`` defaults), saved
projects -- on the annotation they hold -- and GISSMO tubes, and prints per
study: analytes quantified and their dots, the share of detected peaks and
of signal area the final model explains (processing.spectrum_coverage), and
the tallest peaks it leaves unexplained. On a tube the bench's own scores
come too, since there the answer is known.

    python -m validation.coverage_report --projects murilo mice rawmg
    python -m validation.coverage_report --tubes 18 --seeds 0
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import project_file
from gui.stages._annotation_helpers import ambiguous_peak_ranks
from gui.stages._quant_adapter import quantify_project
from processing.baseline_model import BaselineCache, BaselineModel
from processing.spectral_regions import parse_regions
from validation.fresh_run import defaults, quantify_prepared, score

PROJECTS = {
    'murilo': r'G:\My Drive\Mestrado\NMR\NMR_Murilo0826\NMR_Murilo_Cell_v02.nmrproj',
    'mice': r'G:\My Drive\Mestrado\BEPE\NMR 052226 - Mice\MRP_MiceISS_NEW_v2.nmrproj',
    'rawmg': r'G:\My Drive\Mestrado\BEPE\NMR 0826 - RawMG\RawMG_analise_v1.nmrproj',
}


def _f(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def run_project(path, diagnostics):
    """The project's own detection and annotation, quantified with the
    defaults a new user gets."""
    st = project_file.read(path)
    ppm = np.asarray(st['ppm'], dtype=float)
    raw = np.asarray(st['all_intensities'], dtype=float)
    _bl = st.get('baseline_model')
    model = BaselineModel.from_dict(_bl) if _bl else None
    corr = np.asarray(BaselineCache().corrected(
        ppm, raw, model, already_corrected=bool(st.get('baseline_corrected'))),
        dtype=float)
    pd = st.get('pd') or {}
    peaks = list(pd.get('peaks') or ())
    excl = parse_regions(pd.get('excluded_regions_applied')
                         or pd.get('excluded_regions') or '') or ()
    ann = st.get('ann') or {}
    mets = list(ann.get('metabolites') or ())
    qnt = st.get('qnt') or {}
    ref_ppm = _f(qnt.get('ref_ppm'), 0.0)
    ref = min(peaks, key=lambda p: abs(float(p['ppm']) - ref_ppm))
    d = defaults()
    with contextlib.redirect_stdout(io.StringIO()):
        rows = quantify_project(
            metabolites=mets, peaks=peaks, ppm=ppm,
            intensities=corr if d['quant_on_model'] else raw,
            samples_info=st['samples_info'], ref_peak=ref,
            ref_nh=_f(qnt.get('ref_nh'), 9.0),
            ref_conc=_f(qnt.get('ref_conc'), 0.0), ref_name='IS',
            noise_sigma=float(pd.get('noise_std') or 0.0),
            ambiguous_ranks=ambiguous_peak_ranks(mets, manual_quality={}),
            annotation_params=ann.get('annotation_params'),
            use_pqn=(d['normalisation'] == 'IS+PQN'),
            local_baseline=d['local_baseline'],
            proton_correction=d['proton_correction'],
            align=d['align'], subtract_blanks=d['subtract_blanks'],
            excluded_regions=excl, deconvolve=False,
            quantify_shapes=d['quantify_shapes'],
            resolve_shared=d['resolve_shared'],
            singlet_min_snr=d['singlet_min_snr'], cascade=d['cascade'],
            max_disagreement=d['max_disagreement_pct'] / 100.0,
            raw_intensities=raw, joint_fit=d['joint_fit'],
            diagnostics=diagnostics)
    return rows


def run_tube(size, field, seed, diagnostics):
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    prep, truth = gb.prepared(gb.FIELDS[field], seed, size)
    mb.MHZ = gb.FIELDS[field]
    with contextlib.redirect_stdout(io.StringIO()):
        rows = quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM,
                                 standard_nh=mb.STANDARD_NH,
                                 standard_conc=mb.STANDARD_MM,
                                 standard_name='DSS', diagnostics=diagnostics)
    return rows, truth


def summary(rows) -> dict:
    q = [r for r in rows if not r.get('_not_quantified') and not r.get('_is_is')]
    dots = {c: sum(1 for r in q if int(r.get('_confidence') or 0) == c)
            for c in (3, 2, 1)}
    return {'quantified': len(q), 'dots': dots}


def report(label, rows, diag, truth=None, secs=0.0, orphans=8):
    s = summary(rows)
    cov = diag.get('coverage') or {}
    line = (f"{label:<22s} quantif. {s['quantified']:3d}  "
            f"●●● {s['dots'][3]:2d} ●● {s['dots'][2]:2d} ● {s['dots'][1]:2d}  "
            f"picos explicados {cov.get('explained', 0):3d}/{cov.get('peaks', 0):<3d} "
            f"({100 * cov.get('peak_fraction', float('nan')):4.0f}%)  "
            f"área {100 * cov.get('area_fraction', float('nan')):4.0f}%  "
            f"[{secs:.0f} s]")
    print(line, flush=True)
    if truth is not None:
        sc = score(rows, truth)
        d = sc['dots']
        print(f"{'':22s} achados {sc['found']}/{sc['truth']}  FP {sc['fp']}  "
              f"med {100 * sc['median']:.0f}%  p90 {100 * sc['p90']:.0f}%  "
              f"●●● {d[3][0]}R/{d[3][1]}F  ●● {d[2][0]}R/{d[2][1]}F  "
              f"● {d[1][0]}R/{d[1][1]}F", flush=True)
    for p, snr, share in (cov.get('orphans') or [])[:orphans]:
        print(f"{'':26s}órfão {p:7.4f} ppm  SNR {snr:7.1f}  modelo {100 * share:3.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--projects', nargs='*', default=[], choices=sorted(PROJECTS))
    ap.add_argument('--tubes', nargs='*', default=[])
    ap.add_argument('--fields', type=int, nargs='+', default=[600])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--signal-shift-mppm', type=float, default=0.0)
    ap.add_argument('--orphans', type=int, default=8)
    a = ap.parse_args()
    for name in a.projects:
        diag: dict = {}
        t0 = time.time()
        rows = run_project(PROJECTS[name], diag)
        report(name, rows, diag, secs=time.time() - t0, orphans=a.orphans)
    if a.tubes:
        from validation import gissmo_bench as gb
        gb.SIGNAL_SHIFT = a.signal_shift_mppm / 1000.0
    for size in a.tubes:
        for fk in a.fields:
            for sd in a.seeds:
                diag = {}
                t0 = time.time()
                rows, truth = run_tube(size, fk, sd, diag)
                lab = f'tubo {size} @{fk} s{sd}' + (
                    f' disp{a.signal_shift_mppm:g}' if a.signal_shift_mppm else '')
                report(lab, rows, diag, truth, secs=time.time() - t0,
                       orphans=a.orphans)


if __name__ == '__main__':
    main()
