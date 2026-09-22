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
from gui.widgets.annotation_results_window import AnnotationResultsWindow
from validation import mixture_bench as mb

# The order the results window runs them in, after the search
# (annotation_results_window, the block under 'Computing scores…').
PASSES = ('_fetch_fingerprints', '_compute_detectability', '_compute_overlap',
          '_build_child_candidates', '_compute_scores', '_enrich_scores',
          '_auto_mark_good', '_promote_singlets', '_apply_fingerprint_gate',
          '_nnls_spectral_validation', '_build_peak_assignments')


class _Stage:
    """The annotation stage as the window reads it: the field, and the
    manual marks (none here -- nothing was marked by hand)."""

    def __init__(self, mhz):
        self.spectrometer_var = type('_V', (), {'get': lambda _s: mhz})()
        self._quality_marks = {}
        self._child_quality_marks = {}
        self._anchor_lines = {}


class _Detection:
    """The detection stage as the passes read it: the noise the run
    measured and the regions it was told to ignore. Without them the
    fingerprint gate falls back to a tail estimate of the noise and puts
    the water in as a signal, and every "would I have seen it" verdict is
    answered against the wrong spectrum."""

    def __init__(self, noise, excluded):
        self.noise_std = float(noise)
        self._excluded_regions = list(excluded or ())


class _App:
    def __init__(self, ppm, X, mhz, samples_info, *, noise, excluded):
        self.ppm = ppm
        self.all_intensities = X
        self.samples_info = samples_info
        self.stages = {'annotation': _Stage(mhz),
                       'peak_detection': _Detection(noise, excluded)}
        self.root = None


class Window:
    """Enough of the results window for its passes to run: the data they
    read and the bookkeeping they write, and nothing of the interface."""

    def __init__(self, metabolites, peaks, ppm, X, *, mhz, params, samples_info,
                 noise, excluded=()):
        self.metabolites = metabolites
        self.peaks = peaks
        self.ppm = np.asarray(ppm, dtype=float)
        self.all_intensities = np.asarray(X, dtype=float)
        incl = [s['index'] for s in samples_info]
        self.mean_intensity = self.all_intensities[:, incl].mean(axis=1)
        self.app = _App(self.ppm, self.all_intensities, mhz, samples_info,
                        noise=noise, excluded=excluded)
        self._params = dict(params)
        self._noise_level = float(noise)
        self._lines_dirty = set()
        self._fingerprint_stats = {}
        self._peak_overlap = {}
        self._child_candidates = {}
        self._peak_assignments = {}
        self._peak_assignment_status = {}
        self._peak_candidates = {}
        self._ambiguous_peak_ranks = set()
        self._ppm_calibration_info = None

    def _sync_anchors_to_peaks(self):
        """The window keeps the anchor lines it drew in step with the peak
        list; there are no drawn lines here."""

    def __getattr__(self, name):
        """Any helper a pass reaches for comes from the window's own class,
        bound to this stub. Only the data and the bookkeeping are ours; the
        code is the application's, or the comparison would be with a copy
        of it again."""
        fn = getattr(AnnotationResultsWindow, name, None)
        if fn is None or not callable(fn):
            raise AttributeError(name)
        return fn.__get__(self, type(self))

    def run(self, db_path, log=print, watch=None):
        """Run the passes in order. ``watch`` is called after each one, so a
        caller can attribute a change to the pass that made it rather than
        to the block as a whole."""
        for name in PASSES:
            fn = getattr(AnnotationResultsWindow, name)
            try:
                fn(self, db_path) if name == '_fetch_fingerprints' else fn(self)
            except Exception as exc:            # a pass that needs the window
                log(f'  {name}: skipped ({type(exc).__name__}: {exc})')
            else:
                log(f'  {name}: ok')
            if watch is not None:
                watch(name, self.metabolites)
        return self.metabolites


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
