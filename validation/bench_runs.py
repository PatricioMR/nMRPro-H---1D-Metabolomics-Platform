"""Run the GISSMO bench in parallel and print the tables pooled over seeds.

Every (tube, scatter, seed) is an independent run: the spectrum, detection
and annotation come from gissmo_bench's cache, and only quantification is
computed. Runs go to separate processes -- one Python process uses one core,
and linear algebra this size gains little from more threads -- and each
run's rows are pickled as soon as it ends, so an interrupted batch resumes
where it stopped.

    python -m validation.bench_runs run --label hoje --tubes 100 --jobs 4
    python -m validation.bench_runs report --labels antes hoje --tubes 100

The table: found / theoretical, strangers, real/false and precision per
confidence level, the error pooled over every real analyte and group
(median, p90, bias, share within 20%), and the peaks the final model
explains (processing.spectrum_coverage).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'bench_runs')


def _path(out, label, tube, shift, seed, field, sample_shift=0.0):
    tag = f'_sm{sample_shift:g}' if sample_shift else ''
    return os.path.join(out, f'{label}_{tube}_{field}_{shift:g}{tag}_{seed}.pkl')


def _legacy(gb):
    """The engine as it was before ADR-069: the fingerprint read at the
    tallest matched peak, and a singlet released from the position cap by
    the joint fit. Its annotation is cached apart (a tag on the key)."""
    from processing import annotation_detectability as ad
    from processing import quant_engine as qe
    ad.FINGERPRINT_ANCHOR = 'tallest'
    qe.SINGLET_RELEASED_BY_FIT = True
    tag = gb._shift_tag
    gb._shift_tag = lambda *args, **kw: tag(*args, **kw) + '_adr069off'


def _apply_sets(sets):
    """``module.NAME=literal`` assignments, for a variant measured
    before it becomes a default."""
    import ast
    import importlib
    for item in sets or ():
        target, value = item.split('=', 1)
        mod, name = target.rsplit('.', 1)
        setattr(importlib.import_module(mod), name, ast.literal_eval(value))


def _one(job):
    """One run in a worker process."""
    out, label, tube, shift, seed, field, legacy, sample_shift, align, mstar, sets = job
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    from validation.fresh_run import quantify_prepared
    if legacy:
        _legacy(gb)
    gb.SIGNAL_SHIFT = shift / 1000.0
    gb.SAMPLE_SHIFT = sample_shift / 1000.0
    gb.ALIGN_FIRST = align
    from processing import quant_engine as qe
    qe.M_STAR_COMPETE_MIN_GOOD = mstar
    _apply_sets(sets)
    t0 = time.time()
    prep, truth = gb.prepared(gb.FIELDS[field], seed, tube)
    mb.MHZ = gb.FIELDS[field]
    diag: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        rows = quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM,
                                 standard_nh=mb.STANDARD_NH,
                                 standard_conc=mb.STANDARD_MM,
                                 standard_name='DSS', diagnostics=diag)
    path = _path(out, label, tube, shift, seed, field, sample_shift)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump({'rows': rows, 'truth': truth,
                     'coverage': diag.get('coverage')}, f)
    os.replace(path + '.tmp', path)
    return job, time.time() - t0


def run(a):
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.out, a.label, t, sh, s, a.field, a.legacy, a.sample_shift,
             a.align_first, a.mstar_min_good, tuple(a.set or ()))
            for sh in a.shifts for t in a.tubes for s in a.seeds
            if not os.path.exists(_path(a.out, a.label, t, sh, s, a.field,
                                        a.sample_shift))]
    if not jobs:
        print('nothing to run')
        return
    # one process per core is the gain; leave each a share of the threads
    threads = max(1, (os.cpu_count() or 1) // max(a.jobs, 1))
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = str(threads)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for f in as_completed(futs):
            (_o, _l, t, sh, s, _f, _g, _m, _a, _ms, _st), secs = f.result()
            print(f'done tube {t} scatter {sh:g} seed {s} in {secs:.0f} s '
                  f'({time.time() - t0:.0f} s elapsed)', flush=True)


def _errors(rows, truth):
    e = []
    for r in rows:
        if r.get('_not_quantified') or r.get('_is_is') or r['hmdb_id'] not in truth:
            continue
        for g, vals in (r.get('_concs_by_group') or {}).items():
            want = truth[r['hmdb_id']].get(g)
            if want and vals:
                e.append(float(np.median(vals)) / float(want) - 1.0)
    return e


def report(a):
    regimes = {0.0: 'padrão', 5.0: 'disperso 5 mppm'}
    print(f'Somatório das sementes {a.seeds} (GISSMO {a.field} MHz). '
          'R = real, F = falso (não está no tubo).')
    for t in a.tubes:
        for sh in a.shifts:
            print(f'\n### Tubo {t} — {regimes.get(sh, f"disperso {sh:g} mppm")}')
            print('| variante | encontrados / teórico | falsos | ●●● R/F (precisão) '
                  '| ●● R/F (precisão) | ● R/F (precisão) | erro mediano | p90 '
                  '| viés | ≤ 20% | picos explicados |')
            print('|---|---|---|---|---|---|---|---|---|---|---|')
            for label in a.labels:
                found = theo = fp = ex = pk = n = 0
                dots = {3: [0, 0], 2: [0, 0], 1: [0, 0]}
                errs = []
                for s in a.seeds:
                    p = _path(a.out, label, t, sh, s, a.field, a.sample_shift)
                    if not os.path.exists(p):
                        continue
                    n += 1
                    d = pickle.load(open(p, 'rb'))
                    rows, truth = d['rows'], d['truth']
                    q = [x for x in rows if not x.get('_not_quantified')
                         and not x.get('_is_is')]
                    found += sum(1 for x in q if x['hmdb_id'] in truth)
                    fp += sum(1 for x in q if x['hmdb_id'] not in truth)
                    theo += len(truth)
                    for x in q:
                        c = int(x.get('_confidence') or 0)
                        if c in dots:
                            dots[c][0 if x['hmdb_id'] in truth else 1] += 1
                    errs += _errors(rows, truth)
                    cov = d.get('coverage') or {}
                    ex += cov.get('explained', 0)
                    pk += cov.get('peaks', 0)
                if not n:
                    continue
                e = np.abs(np.array(errs)) if errs else np.array([np.nan])
                cells = [f'{r}/{f} ({100 * r / (r + f):.0f}%)' if r + f else '0/0'
                         for r, f in (dots[3], dots[2], dots[1])]
                tag = '' if n == len(a.seeds) else f' ({n} sem.)'
                print(f'| {label}{tag} | {found}/{theo} ({100 * found / theo:.0f}%) | {fp} | '
                      + ' | '.join(cells)
                      + f' | {100 * np.nanmedian(e):.0f}% | {100 * np.nanpercentile(e, 90):.0f}% '
                        f'| {100 * np.nanmedian(errs):+.0f}% | {100 * np.mean(e <= 0.20):.0f}% '
                        f'| {ex}/{pk} ({100 * ex / pk if pk else float("nan"):.0f}%) |')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('what', choices=('run', 'report'))
    ap.add_argument('--label', default='atual')
    ap.add_argument('--labels', nargs='+', default=['atual'])
    ap.add_argument('--tubes', nargs='+', default=['18', '50'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--shifts', type=float, nargs='+', default=[0.0, 5.0])
    ap.add_argument('--field', type=int, default=600)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--sample-shift', type=float, default=0.0,
                    help='mppm each environment moves from sample to sample '
                         '(gissmo_bench.SAMPLE_SHIFT)')
    ap.add_argument('--align-first', choices=('valley', 'icoshift'),
                    default=None, help='align the tube before detection')
    ap.add_argument('--set', action='append', default=[],
                    help='module.NAME=value (Python literal) set in every worker, '
                         'e.g. processing.quant_shared.LISTING_FROM_STUDY=True')
    ap.add_argument('--mstar-min-good', type=int, default=None,
                    help='quant_engine.M_STAR_COMPETE_MIN_GOOD for the run')
    ap.add_argument('--legacy', action='store_true',
                    help='the engine before ADR-069, for comparison')
    a = ap.parse_args()
    {'run': run, 'report': report}[a.what](a)


if __name__ == '__main__':
    main()
