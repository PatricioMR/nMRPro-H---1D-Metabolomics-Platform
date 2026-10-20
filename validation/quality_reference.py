# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""What each confidence level has been measured to mean, written for the
report.

Runs the crowded reference bench (``validation.crowded_bench``: the 18
analytes of the reference study, then filled to 50 and 100 with analytes
drawn from the reference database at 0.05-1 mM) on three seeds, with the
application's defaults, and records per tube and per confidence level how
many rows were real analytes, how many were not, and how far the real
ones' amounts were from the truth. The report prints these beside every
row's dots: a level is a measured precision, not a fraction of three.

    python -m validation.quality_reference                 # GISSMO, 500 and 600
    python -m validation.quality_reference --generator profiles

Writes ``data/quality_reference.json``, keyed by spectrometer field: the
field the analyst sets in the annotation stage picks the entry the report
prints (``data.quality_reference.for_field``). By default the tubes are
simulated from GISSMO spin systems at each field (``validation.
gissmo_bench``), so the spectra are not the profiles the annotation
searches for; ``--generator profiles`` measures the older bench instead.
A field enters the file only once it has been measured.
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

from app_paths import APP_VERSION
from validation import mixture_bench as mb
from validation.crowded_bench import CONC_RANGE_MM, prepared
from validation.fresh_run import quantify_prepared

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'data', 'quality_reference.json')
TUBES = (18, 50, 100)
SEEDS = (0, 1, 2)


def _row_error(r, truth) -> float | None:
    """Relative error of a real analyte's amount: the median over the
    groups of (median measured / true - 1)."""
    t = truth.get(r['hmdb_id'])
    if not t:
        return None
    e = [float(np.median(v)) / t[g] - 1.0
         for g, v in (r.get('_concs_by_group') or {}).items()
         if t.get(g) and v]
    return float(np.median(e)) if e else None


def _bench_runs(n):
    """``[(rows, truth)]`` of the reference bench's tube of ``n``."""
    out = []
    for seed in SEEDS:
        prep, truth = prepared(n, seed)
        rows = quantify_prepared(prep, mb.SAMPLES,
                                 standard_ppm=mb.STANDARD_PPM,
                                 standard_nh=mb.STANDARD_NH,
                                 standard_conc=mb.STANDARD_MM,
                                 standard_name='DSS')
        out.append((rows, truth, len(prep['peaks'])))
    return out


def _warm(job):
    """One GISSMO run in a worker: gissmo_bench.run keeps its rows, so the
    tables below read them back instead of computing them one by one."""
    field, n, seed = job
    from validation.gissmo_bench import run
    run(field, seed, str(n))
    return job


def warm_gissmo(fields, jobs):
    """Every (field, tube, seed) the tables need, in parallel processes --
    a Python process uses one core and the runs are independent."""
    from concurrent.futures import ProcessPoolExecutor
    threads = str(max(1, (os.cpu_count() or 1) // max(jobs, 1)))
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = threads
    todo = [(f, n, s) for f in fields for n in TUBES for s in SEEDS]
    from validation.bench_runs import _below_normal
    with ProcessPoolExecutor(max_workers=jobs, initializer=_below_normal) as ex:
        for f, n, s in ex.map(_warm, todo):
            print(f'  {f:.0f} MHz tube {n} seed {s}', flush=True)


def _gissmo_runs(field):
    def runs(n):
        from validation.gissmo_bench import run
        return [run(field, seed, str(n))[1:4] for seed in SEEDS]
    return runs


def tube_stats(n, runs=_bench_runs) -> dict:
    """Pooled over SEEDS: per level the real and false rows and the real
    rows' errors; overall recall and precision at two dots or more."""
    levels = {lv: {'real': 0, 'false': 0, 'errors': []} for lv in (1, 2, 3)}
    found = truth_n = 0
    all_err = []
    peaks = []
    for rows, truth, n_peaks in runs(n):
        peaks.append(n_peaks)
        truth_n += len(truth)
        for r in rows:
            if r.get('_not_quantified') or r.get('_is_is'):
                continue
            lv = min(max(int(r.get('_confidence') or 1), 1), 3)
            if r['hmdb_id'] in truth:
                levels[lv]['real'] += 1
                found += 1
                e = _row_error(r, truth)
                if e is not None:
                    levels[lv]['errors'].append(e)
                    all_err.append(e)
            else:
                levels[lv]['false'] += 1
    # how crowded the mixture's spectrum is, in the one unit a real sample
    # can be measured in: the report matches a project to it
    out = {'analytes': n, 'runs': len(SEEDS), 'levels': {},
           'peaks': float(np.mean(peaks))}
    for lv, d in levels.items():
        tot = d['real'] + d['false']
        err = np.abs(d['errors'])
        out['levels'][str(lv)] = {
            'real': d['real'], 'false': d['false'],
            'precision': d['real'] / tot if tot else None,
            'error_median': float(np.median(err)) if err.size else None,
            'error_p90': float(np.percentile(err, 90)) if err.size else None,
        }
    tp = levels[2]['real'] + levels[3]['real']
    fp = levels[2]['false'] + levels[3]['false']
    all_abs = np.abs(all_err)
    out.update({
        'found': found / truth_n,
        'recall_2plus': tp / truth_n,
        'precision_2plus': tp / (tp + fp) if tp + fp else None,
        'false_per_run': (sum(d['false'] for d in levels.values())
                          / len(SEEDS)),
        'error_median': float(np.median(all_abs)) if all_abs.size else None,
        'error_p90': (float(np.percentile(all_abs, 90))
                      if all_abs.size else None),
    })
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--generator', choices=('gissmo', 'profiles'),
                    default='gissmo',
                    help='gissmo: the tubes simulated from GISSMO spin systems '
                         '(validation.gissmo_bench), at every --fields; '
                         'profiles: the reference bench, drawn from the '
                         "database's own profiles, at mixture_bench.MHZ")
    ap.add_argument('--fields', type=int, nargs='+', default=[500, 600])
    ap.add_argument('--jobs', type=int, default=8,
                    help='GISSMO runs computed in parallel before the tables')
    a = ap.parse_args()
    doc = {}
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            doc = json.load(f)
    if a.generator == 'profiles':
        plan = [(int(mb.MHZ), mb.MHZ, _bench_runs)]
    else:
        from validation.gissmo_bench import FIELDS
        plan = [(fk, FIELDS[fk], _gissmo_runs(FIELDS[fk])) for fk in a.fields]
        if a.jobs > 1:
            warm_gissmo([FIELDS[fk] for fk in a.fields], a.jobs)
    for key, field, runs in plan:
        doc[str(key)] = {
            'software': APP_VERSION,
            'date': datetime.date.today().isoformat(),
            'database': os.path.basename(mb.DB_DEFAULT),
            'generator': a.generator,
            'design': {
                'field_mhz': field, 'linewidth_hz': mb.LINEWIDTH_HZ,
                'samples': len(mb.SAMPLES), 'groups': 2,
                'noise': f'lactate doublet at SNR {mb.NOISE_LACTATE_SNR:.0f}',
                'concentration_range_mm': list(CONC_RANGE_MM),
                'reference_analytes': len(mb.PANEL), 'seeds': len(SEEDS),
                'internal_standard': f'DSS {mb.STANDARD_MM} mM',
                'spectra': ('GISSMO spin systems solved at the field (Dashti '
                            'et al., Anal Chem 2017)' if a.generator == 'gissmo'
                            else "the reference database's own profiles"),
            },
            'tubes': {str(n): tube_stats(n, runs) for n in TUBES},
        }
        print(f'{key} MHz ({a.generator}) ->', OUT, flush=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)


if __name__ == '__main__':
    main()
