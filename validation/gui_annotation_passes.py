"""The application's own annotation passes, run over the bench, without a window.

The bench builds its study and annotates it with its own orchestration
(``mixture_bench.annotate``), while the application annotates in
``gui.stages._annotation_search`` and then runs twelve further passes on
the results window before anything is quantified. The scoring primitives
are the same on both sides; the order and the passes are not, and three
of the window's passes set ``annotation_quality`` -- which is what the
confidence bands and every ΔJ gate downstream read.

So the bench's numbers were the engine's answer to the bench's own
annotation, not to the application's. This closes most of that gap
without a refactor and without a display: every pass the window runs
after the search is a plain method over ``metabolites``, ``peaks``,
``ppm``, ``mean_intensity`` and ``_params`` -- no widget is touched --
so they are bound to a stub and run in the same order the window runs
them. What is still the bench's own is the *search*: which database rows
are matched to which peak, and with what tolerance.

    python -m validation.gui_annotation_passes

It prints what the application's passes change about the bench's
annotation -- signal by signal -- and then quantifies both and compares.
"""
from __future__ import annotations

import contextlib
import copy
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gui.stages._annotation_helpers import ambiguous_peak_ranks
from gui.stages._quant_adapter import quantify_project
from validation import mixture_bench as mb
from validation.app_annotation import Window


def _marks(mets):
    out = {}
    for m in mets:
        for mp in m.get('matched_peaks') or ():
            if mp.get('db_ppm') is None:
                continue
            out[(m['id'], round(float(mp['db_ppm']), 3))] = mp.get('annotation_quality')
    return out


def _quantify(mets, peaks, ppm, X, params, noise):
    amb = ambiguous_peak_ranks(mets, manual_quality={})
    ref = min(peaks, key=lambda p: abs(float(p['ppm']) - mb.STANDARD_PPM))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rows = quantify_project(
            metabolites=mets, peaks=peaks, ppm=ppm, intensities=X,
            samples_info=mb.SAMPLES, ref_peak=ref, ref_nh=mb.STANDARD_NH,
            ref_conc=mb.STANDARD_MM, ref_name='DSS', noise_sigma=noise,
            ambiguous_ranks=amb, use_pqn=False, local_baseline=False,
            quantify_shapes=True, deconvolve=False, resolve_shared=True,
            cascade=True, annotation_params=params)
    return rows


def _report(rows, truth, label):
    q = {r['hmdb_id'] for r in rows
         if not r.get('_not_quantified') and not r.get('_is_is')}
    found = sorted(truth[m]['name'] for m in q if m in truth)
    fp = len(q) - len(found)
    conf = {}
    for r in rows:
        if r.get('_not_quantified') or r.get('_is_is'):
            continue
        conf[r['metabolite']] = (r.get('_identity_level'), r.get('_confidence'))
    print(f'{label}: {len(found)}/{len(truth)} found, {fp} false positives')
    return q, found, fp, conf


def main():
    print('building the study and annotating it the bench way…')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        X, sigma, truth, _foreign = mb.build_study(mb.DB_DEFAULT, seed=0)
        peaks = mb.detect_peaks(
            mb.PPM, X, noise_std=sigma, min_snr=6.0, min_distance=0.003,
            baseline_correct=False, global_align=False, samples_info=mb.SAMPLES,
            per_sample_detection=True, detect_shoulders=False,
            min_group_presence=3, excluded_regions=mb.WATER)
        mets, params, _ = mb.annotate(mb.PPM, X, peaks, sigma, mb.DB_DEFAULT)
    before = copy.deepcopy(mets)

    print("\nrunning the application's own passes over the same annotation:")
    win = Window(mets, peaks, mb.PPM, X, mhz=mb.MHZ, params=params,
                 samples_info=mb.SAMPLES, noise=sigma, excluded=mb.WATER)
    seen = {'marks': _marks(before)}
    steps = []

    def _watch(name, mets_now):
        now = _marks(mets_now)
        diff = {k: (seen['marks'].get(k), now.get(k))
                for k in set(now) | set(seen['marks'])
                if seen['marks'].get(k) != now.get(k)}
        seen['marks'] = now
        if diff:
            steps.append((name, diff))

    after = win.run(mb.DB_DEFAULT, watch=_watch)

    print('\nwhich pass changed what:')
    _name0 = {m['id']: m['name'] for m in before}
    for name, diff in steps:
        mine = [(_name0.get(k[0], k[0])[:22], round(k[1], 3), v[0], '->', v[1])
                for k, v in sorted(diff.items()) if k[0] in truth]
        print(f'   {name:26s} {len(diff):4d} marks')
        for row in mine:
            print(f'      of the study: {row}')

    m0, m1 = _marks(before), _marks(after)
    changed = {k: (m0.get(k), m1.get(k)) for k in set(m0) | set(m1)
               if m0.get(k) != m1.get(k)}
    name_of = {m['id']: m['name'] for m in before}
    print(f'\nquality marks: {len(m0)} signals, {len(changed)} changed by the '
          f"application's passes")
    tally: dict = {}
    for (mid, _ppm), (a, b) in changed.items():
        tally[(a, b)] = tally.get((a, b), 0) + 1
    for (a, b), n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f'   {str(a):>9s} -> {str(b):<9s} {n:4d}')
    print('\n  of the analytes the study contains:')
    for (mid, ppm_), (a, b) in sorted(changed.items()):
        if mid in truth:
            print(f'   {name_of.get(mid, mid)[:26]:26s} {ppm_:7.3f}  {a} -> {b}')

    print()
    r0 = _quantify(before, peaks, mb.PPM, X, params, sigma)
    q0, f0, fp0, c0 = _report(r0, truth, 'bench annotation      ')
    r1 = _quantify(after, peaks, mb.PPM, X, params, sigma)
    q1, f1, fp1, c1 = _report(r1, truth, "application's passes  ")

    gained, lost = sorted(q1 - q0), sorted(q0 - q1)
    print(f'\n  gained: {[name_of.get(m, m) for m in gained]}')
    print(f'  lost:   {[name_of.get(m, m) for m in lost]}')
    moved = {k: (c0[k], c1[k]) for k in set(c0) & set(c1) if c0[k] != c1[k]}
    print(f'  confidence moved for {len(moved)} analytes: '
          f'{ {k: v for k, v in list(moved.items())[:8]} }')


if __name__ == '__main__':
    main()
