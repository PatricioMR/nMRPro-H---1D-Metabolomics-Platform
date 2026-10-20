# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Where the true analytes of the random tubes are lost, against what the
spectrum lets anyone find.

For every analyte really in a tube (``gissmo_bench.random_tube``), its own
noiseless trace is rendered as the tube was built, and from it:

    snr        the tallest line of the analyte over the noise sigma of the
               mean trace (the one detection measures against)
    clean      the tallest line where the analyte is at least half of
               everything there (the others' noiseless traces summed)
    detected   a detected peak within half a linewidth of a line >= 5 sigma
    annotated  the annotation matched one of its listed signals to a peak
    stage      what the quantification did with it (loss_funnel._reason)
    dots       the confidence of its number

so that a loss can be put at the stage that caused it, and the analytes no
method could find (nothing above the noise, or never on top anywhere) are
told apart from the ones the engine dropped.

    python -m validation.recovery_funnel --label rec0 --tubes r50 r75 --seeds 100 101 102
    python -m validation.recovery_funnel --label d1 --set processing.annotation_scoring.DRIFT_BY_MODE=True
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.bench_runs import OUT_DEFAULT, _path
from validation.loss_funnel import _reason, annotation_evidence

SNR_BANDS = (0, 3, 5, 10, 20, 50, np.inf)


def _band(v, edges=SNR_BANDS):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= v < hi:
            return f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'>={lo:g}'
    return '?'


def analyte_traces(composition, field, seed=None, ref_shift=0.0):
    """``{name: noiseless trace at its amount}``, rendered as
    ``gissmo_bench.build`` renders the tube (without the per-sample spread
    of the amount, which averages out). With ``gissmo_bench.SIGNAL_SHIFT``
    each signal is moved by the same draw the tube was built with."""
    from processing.spin_system import load_systems, render, stick_spectrum
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    systems = load_systems()
    signal_rng = np.random.default_rng((seed or 0) + 9_000_011)
    shift_rng = np.random.default_rng((seed or 0) + 7_000_003)
    out = {}
    for name, conc, _f in composition:
        sys_ = (gb._spread_signals(systems[name], gb.SIGNAL_SHIFT, signal_rng)
                if gb.SIGNAL_SHIFT else systems[name])
        sp, si = stick_spectrum(sys_, field)
        own = float(shift_rng.normal(0.0, ref_shift)) if ref_shift else 0.0
        out[name] = conc * render(gb.PPM, sp + mb.DRIFT_PPM + own, si * mb.AREA_UNIT,
                                  mb.LINEWIDTH_HZ, field)
    return out


def noise_sigma(field):
    """The noise of the study's mean trace, which is what detection
    measures against (``prep['noise_std']`` reads 3% above it)."""
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    import sqlite3
    mb.MHZ = field
    con = sqlite3.connect(mb.DB_DEFAULT)
    _m, lac = mb._db_signals(con, 'L-lactic acid')
    per_sample = (float(mb.compound_trace(gb.PPM, lac, 4.0, shift=0.0).max())
                  / mb.NOISE_LACTATE_SNR)
    return per_sample / np.sqrt(len(mb.SAMPLES))


def _local_maxima(y, floor):
    i = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > floor)) + 1
    return i


def analyte_records(tube, seed, field, label, out=OUT_DEFAULT, shift=0.0, ref_shift=0.0):
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    f = gb.FIELDS[field]
    comp = gb._composition(tube, seed)
    prep, truth = gb.prepared(f, seed, tube, ref_shift=ref_shift / 1000.0)
    with open(_path(out, label, tube, shift, seed, field, 0.0, ref_shift), 'rb') as fh:
        rows = pickle.load(fh)['rows']
    sigma = noise_sigma(f)
    traces = analyte_traces(comp, f, seed, ref_shift / 1000.0)
    total = sum(traces.values())
    half_lw = 0.5 * mb.LINEWIDTH_HZ / f
    peaks = np.sort(np.array([float(p['ppm']) for p in prep['peaks']]))
    mets = {str(m.get('id')): m for m in prep['mets']}
    by_id = {r['hmdb_id']: r for r in rows}
    name_to_mid = {t['name']: mid for mid, t in truth.items()}
    recs = []
    for name, conc, _fac in comp:
        mid = name_to_mid[name]
        y = traces[name]
        idx = _local_maxima(y, 3 * sigma)
        snr = float(y.max() / sigma)
        share = y[idx] / np.maximum(total[idx], 1e-30) if idx.size else np.array([])
        clean = float(max((y[i] / sigma for i, s in zip(idx, share) if s >= 0.5),
                          default=0.0))
        lines5 = [gb.PPM[i] for i in idx if y[i] >= 5 * sigma]
        detected = bool(peaks.size and any(
            np.min(np.abs(peaks - p)) <= half_lw for p in lines5))
        ev = annotation_evidence(mets.get(mid))
        r = by_id.get(mid)
        stage = _reason(r)
        recs.append({
            'run': (tube, seed), 'mid': mid, 'name': name, 'conc': conc,
            'snr': snr, 'clean': clean, 'n_lines5': len(lines5),
            'detected': detected, 'annotated': bool(ev['matched']),
            'stage': stage,
            'dots': int(r.get('_confidence') or 0) if stage == 'number' else 0,
            **{k: ev[k] for k in ('in_study', 'bad_only', 'stands') if k in ev},
        })
    strangers = [r for mid, r in by_id.items()
                 if mid not in truth and not r.get('_not_quantified')
                 and not r.get('_is_is')]
    return recs, strangers


def report(recs, strangers_by_run, log=print):
    n = len(recs)
    found = [r for r in recs if r['stage'] == 'number']
    log(f'{n} true analytes, {len(found)} with a number '
        f'({100 * len(found) / n:.0f}%), '
        f'{sum(len(s) for s in strangers_by_run.values())} strangers with one '
        f'({len(strangers_by_run)} runs)\n')

    log('| clean-line SNR | analytes | found | % | not detected | not annotated | engine lost |')
    log('|---|---|---|---|---|---|---|')
    for b in [_band(lo) for lo in SNR_BANDS[:-1]]:
        sel = [r for r in recs if _band(r['clean']) == b]
        if not sel:
            continue
        f = sum(r['stage'] == 'number' for r in sel)
        nd = sum(not r['detected'] for r in sel)
        na = sum(r['detected'] and not r['annotated'] for r in sel)
        el = sum(r['detected'] and r['annotated'] and r['stage'] != 'number' for r in sel)
        log(f'| {b} | {len(sel)} | {f} | {100 * f / len(sel):.0f}% | {nd} | {na} | {el} |')

    log('\n| concentration mM | analytes | found | % |')
    log('|---|---|---|---|')
    for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)):
        sel = [r for r in recs if lo <= r['conc'] < hi]
        if sel:
            f = sum(r['stage'] == 'number' for r in sel)
            log(f'| {lo}-{hi:.1f} | {len(sel)} | {f} | {100 * f / len(sel):.0f}% |')

    log('\nLost, by stage (detected and annotated, clean SNR >= 10):')
    reach = [r for r in recs if r['stage'] != 'number' and r['detected']
             and r['annotated'] and r['clean'] >= 10]
    for st, k in Counter(r['stage'] for r in reach).most_common():
        log(f'  {k:4d}  {st}')
    log('\nLost, by stage (all):')
    for st, k in Counter(r['stage'] for r in recs if r['stage'] != 'number').most_common():
        log(f'  {k:4d}  {st}')
    log('\nFound, by dots: ' + ', '.join(
        f'{d}: {sum(r["dots"] == d for r in found)}' for d in (3, 2, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--label', default='rec0')
    ap.add_argument('--tubes', nargs='+', default=['r50', 'r75'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102])
    ap.add_argument('--field', type=int, default=600)
    ap.add_argument('--dump', default=None)
    ap.add_argument('--ref-shift', type=float, default=0.0,
                    help='mppm each compound stands off its listing (bench_runs --ref-shift)')
    ap.add_argument('--shift', type=float, default=0.0,
                    help='scatter of each signal, mppm (bench_runs --shifts)')
    ap.add_argument('--set', action='append', default=[],
                    help='module.NAME=literal, as the variant was run with')
    a = ap.parse_args()
    from validation import gissmo_bench as gb
    from validation.bench_runs import _apply_sets, _prep_sets
    _apply_sets(a.set)
    gb.PREP_SETTINGS = _prep_sets(a.set)
    gb.SIGNAL_SHIFT = a.shift / 1000.0
    allr, strangers = [], {}
    for t in a.tubes:
        recs_t = []
        for s in a.seeds:
            recs, st = analyte_records(t, s, a.field, a.label, shift=a.shift, ref_shift=a.ref_shift)
            recs_t += recs
            strangers[(t, s)] = st
        print(f'## {t}')
        report(recs_t, {k: v for k, v in strangers.items() if k[0] == t})
        print()
        allr += recs_t
    if a.dump:
        with open(a.dump, 'wb') as f:
            pickle.dump({'recs': allr, 'strangers': strangers}, f)


if __name__ == '__main__':
    main()
