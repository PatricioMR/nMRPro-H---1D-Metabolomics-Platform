"""The reference study, filled up: 50 and 100 analytes instead of 18.

A real extract holds a hundred or two hundred metabolites, most of them
minor. The 18-analyte study says how the software does on a spectrum with
room between its signals; this says what happens as the room goes. The
experiment is otherwise identical -- the same 12 samples, the same noise
(anchored to lactate's doublet), the same linewidth and drift, the same
18 analytes at the same amounts -- and every run goes through
``validation.fresh_run`` with the application's own defaults, so the only
thing that changes is how full the tube is.

The analytes added are drawn at random, with a fixed seed, from the
mammalian reference the annotation searches (so every one of them is
findable in principle), at amounts drawn log-uniformly between 0.05 and
1 mM -- the range of the reference study's minor analytes -- and the same
in both groups. The draws nest: the 100-analyte tube is the 50-analyte tube
with fifty more. Excluded from the draw: anything with a signal within
0.1 ppm of the internal standard, which would sit on the reference peak.

    python -m validation.crowded_bench              # 18, 50 and 100
    python -m validation.crowded_bench --sizes 18 50 100 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation import mixture_bench as mb
from validation.fresh_run import fresh_run, score, tables

CONC_RANGE_MM = (0.05, 1.0)
STANDARD_CLEARANCE_PPM = 0.1


def panel(db_path, n, *, seed=0):
    """``[(name, conc_mM, group_factor)]``: the 18 of the reference study
    and ``n - 18`` more drawn from the reference."""
    base = list(mb.PANEL)
    if n <= len(base):
        return base[:n]
    con = sqlite3.connect(db_path)
    taken = {name for name, _c, _f in base}
    pool = []
    for bid, name in con.execute('select bmrb_id, name from metabolites '
                                 'order by bmrb_id'):
        if name in taken:
            continue
        ppms = [float(r[0]) for r in con.execute(
            'select coalesce(ppm_meas, ppm) from shifts_1h where bmrb_id=?',
            (bid,)) if r[0] is not None]
        if not ppms:
            continue
        if any(abs(p - mb.STANDARD_PPM) < STANDARD_CLEARANCE_PPM for p in ppms):
            continue
        pool.append(name)
    # One shuffle and one draw of amounts for the whole pool, then the first
    # n - 18 of it: the 100-analyte tube is the 50-analyte tube with fifty
    # more added, each at the same amount it had there.
    rng = np.random.default_rng(10_000 + seed)
    order = rng.permutation(len(pool))
    lo, hi = np.log(CONC_RANGE_MM[0]), np.log(CONC_RANGE_MM[1])
    concs = np.exp(rng.uniform(lo, hi, size=len(pool)))
    if n - len(base) > len(pool):
        raise ValueError(f'the reference holds only {len(pool)} more')
    return base + [(pool[i], float(round(concs[k], 3)), 1.0)
                   for k, i in enumerate(order[:n - len(base)])]


def run(n, seed=0, log=print):
    """``(score, n_peaks, seconds)`` for one study of ``n`` analytes."""
    pan = panel(mb.DB_DEFAULT, n, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        X, _sigma, truth, _f = mb.build_study(mb.DB_DEFAULT, seed=seed,
                                              panel=pan)
    t0 = time.monotonic()
    peaks, _mets, rows, _noise, _d = fresh_run(
        mb.PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT, field_mhz=mb.MHZ,
        standard_ppm=mb.STANDARD_PPM, standard_nh=mb.STANDARD_NH,
        standard_conc=mb.STANDARD_MM, standard_name='DSS',
        log=lambda *a: None)
    s = score(rows, truth)
    # the 18 of the reference study, scored on their own inside the crowd
    core = {mid: t for mid, t in truth.items()
            if t['name'] in {nm for nm, _c, _f in mb.PANEL}}
    s['core'] = score([r for r in rows if r['hmdb_id'] in core
                       or r['hmdb_id'] not in truth], core)
    return s, len(peaks), time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--sizes', type=int, nargs='+', default=[18, 50, 100])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    a = ap.parse_args()
    results, core = [], []
    for n in a.sizes:
        for seed in a.seeds:
            s, n_peaks, secs = run(n, seed)
            lab = f'{n} analitos' + (f' (seed {seed})' if len(a.seeds) > 1 else '')
            print(f'{lab}: {n_peaks} picos, {secs:.0f} s, falhou '
                  f'{len(s["missed"])}', flush=True)
            results.append((lab, s))
            core.append((lab, s['core']))
    print('\n=== TODOS OS ANALITOS DO TUBO ===\n')
    tables(results)
    print('\n=== SO OS 18 DO ESTUDO DE REFERENCIA, DENTRO DE CADA TUBO ===\n')
    tables(core)


if __name__ == '__main__':
    main()
