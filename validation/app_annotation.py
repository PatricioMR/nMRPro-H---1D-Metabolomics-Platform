# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Annotating a study the way the application does, without a window.

The application searches the database (controllers.annotation_controller)
and then runs eleven further passes on the results window before anything
is quantified -- three of which decide ``annotation_quality``, which every
gate downstream reads. Those passes are plain methods over the
metabolites, the peaks, the trace and the scoring parameters: no widget is
touched. Here they are bound to a stub and run in the application's own
order, so a study can be annotated exactly as the analyst's would be,
from a script.

``annotate()`` is the whole path -- search, grouping, passes -- and
returns what the quantification stage expects. Nothing in this module
knows about the bench, so the bench can use it.
"""
from __future__ import annotations

import numpy as np

from controllers.annotation_controller import search_database
from gui.widgets.annotation_results_window import AnnotationResultsWindow
from processing.annotation_scoring import effective_annotation_params
from processing.peak_detection import canonical_peak_order
from processing.signal_quality import is_annotatable


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


def annotate(peaks, ppm, intensities, samples_info, *, db_path, field_mhz,
             tolerance_ppm, noise_std, min_snr, min_distance_ppm,
             excluded_regions=(), tiers=('Good', 'Average'), params=None,
             peak_correlation=None, n_samples_correlated=0,
             log=lambda *a: None):
    """Annotate a study exactly as the application does: its search, its
    grouping, its passes, in its order -- and none of its window.

    Returns ``(metabolites, params)``, the same pair the bench's own
    ``annotate`` returns, so a caller can put one where the other was and
    compare what the engine then makes of it.

    ``excluded_regions`` are what detection was told to ignore, and they
    are not optional in practice: the fingerprint gate asks whether a
    signal that should be visible is there, and a signal under the
    suppressed water is absent for a reason that says nothing about the
    analyte. Without them hydroxyproline was contradicted by its own
    triplet at 4.66 ppm, which the water covers.

    ``peak_correlation`` is the STOCSY matrix detection computes over the
    peaks' per-sample areas, which the application hands the search and
    the ranking reads as corroboration. Left out, the bench ranked every
    contest without its sixth criterion.
    """
    sc = effective_annotation_params(params, tolerance_ppm)
    # the order the application's results window reads them in
    peaks = canonical_peak_order(peaks)
    annotatable = [p for p in peaks if is_annotatable(p, set(tiers))]
    found = search_database(
        db_path=db_path, db_type='hmdb', peaks_to_search=peaks,
        annotatable_peaks=annotatable, tiers=set(tiers), ppm=ppm,
        intensities=intensities, samples_info=samples_info,
        field_mhz=field_mhz, tolerance_ppm=tolerance_ppm, params=sc,
        noise_std=noise_std, min_snr=min_snr,
        min_distance_ppm=min_distance_ppm,
        peak_correlation=peak_correlation or {},
        n_samples_correlated=int(n_samples_correlated or 0))
    log(f'  search: {len(found.results)} matches on '
        f'{found.peaks_matched}/{found.peaks_annotatable} peaks '
        f'({found.elapsed_s:.1f}s)')
    win = Window([], peaks, ppm, intensities, mhz=field_mhz, params=sc,
                 samples_info=samples_info, noise=noise_std,
                 excluded=excluded_regions)
    # the window's own grouping of matches into metabolites, which touches
    # nothing of itself
    win.metabolites = AnnotationResultsWindow._group_by_metabolite(
        win, found.results)
    log(f'  grouped into {len(win.metabolites)} metabolites')
    win.run(db_path, log=log)
    return win.metabolites, sc


# Fields of a peak that place it on the axis, moved together when the peak is
# read at its apex in one sample instead of on the study's reference.
_PPM_FIELDS = ('ppm', 'start_ppm', 'end_ppm', 'cluster_lo_ppm', 'cluster_hi_ppm')
_IDX_FIELDS = ('index', 'start_idx', 'end_idx', 'cluster_lo', 'cluster_hi')


def peaks_of_sample(peaks, s, ppm):
    """The study's peaks as sample ``s`` alone shows them: only those
    detected in it, each moved to its own apex there, with that sample's
    share of the height and area. Ranks are the study's, so a signal
    matched here is the same peak the quantification measures."""
    ppm = np.asarray(ppm, dtype=float)
    step = float(ppm[1] - ppm[0])
    out = []
    for p in peaks:
        pres = p.get('aligned_per_sample')
        if pres is not None and s < len(pres) and not pres[s]:
            continue
        q = dict(p)
        per = p.get('ppm_per_sample')
        delta = (float(per[s]) - float(p['ppm'])
                 if per is not None and s < len(per) else 0.0)
        for k in _PPM_FIELDS:
            if q.get(k) is not None:
                q[k] = float(q[k]) + delta
        dpts = int(round(delta / step)) if step else 0
        for k in _IDX_FIELDS:
            if q.get(k) is not None:
                q[k] = int(np.clip(int(q[k]) + dpts, 0, ppm.size - 1))
        areas = p.get('area_per_sample')
        areas = np.asarray(areas if areas is not None else (), dtype=float)
        if areas.size > s and np.mean(areas) > 0:
            share = max(float(areas[s]) / float(np.mean(areas)), 0.0)
            q['area'] = float(areas[s])
            q['height'] = float(p.get('height', 0.0)) * share
            q['snr'] = float(p.get('snr', 0.0)) * share
        out.append(q)
    return out


def _signal_key(met, mp):
    return (str(met.get('id')), round(float(mp.get('db_ppm') or 0.0), 3))


def annotate_per_sample(peaks, ppm, intensities, samples_info, *,
                        min_group_presence=3, log=lambda *a: None, **kw):
    """Annotate every sample on its own trace, then keep a signal where the
    samples agree.

    Each analytical sample is annotated alone, exactly as ``annotate`` does
    a study, on the peaks it holds at its own apices
    (``peaks_of_sample``). A signal of the study's annotation is then
    accepted where at least ``min_group_presence`` samples of one group
    (all of a smaller group) accepted it -- detection's own presence rule
    -- and Good where that many marked it Good; otherwise it is Bad. A
    sample with an artefact then speaks for itself only, instead of
    entering the trace every verdict is read on.

    The study's annotation is still made: it is the structure the
    quantification reads (lines, windows, anchors), and only the verdicts
    are taken from the samples. A metabolite no study-level match holds
    is taken from the first sample, in index order, that accepted it.

    Returns ``(metabolites, params, votes)``; ``votes`` maps each signal
    to the samples that accepted it and those that marked it Good.
    """
    from processing.sample_selection import included_sample_indices

    X = np.asarray(intensities, dtype=float)
    base, sc = annotate(peaks, ppm, X, samples_info, log=log, **kw)
    incl = included_sample_indices(samples_info, X.shape[1])
    group = {s: samples_info[s].get('label', samples_info[s].get('group', '?'))
             for s in incl}
    sizes = {}
    for g in group.values():
        sizes[g] = sizes.get(g, 0) + 1
    votes, first = {}, {}
    for s in incl:
        info = dict(samples_info[s], index=0)
        mets_s, _ = annotate(peaks_of_sample(peaks, s, ppm), ppm, X[:, [s]],
                             [info], **kw)
        for met in mets_s:
            for mp in met.get('matched_peaks') or ():
                if mp.get('peak_rank') is None:
                    continue
                q = mp.get('annotation_quality')
                if q == 'bad':
                    continue
                v = votes.setdefault(_signal_key(met, mp),
                                     {'accepted': [], 'good': []})
                v['accepted'].append(s)
                if q == 'good':
                    v['good'].append(s)
                first.setdefault(str(met.get('id')), met)
        log(f'  sample {s} ({group[s]}): {len(mets_s)} metabolites')

    def _enough(samples):
        cnt = {}
        for s in samples:
            cnt[group[s]] = cnt.get(group[s], 0) + 1
        return any(n >= min(int(min_group_presence), sizes[g])
                   for g, n in cnt.items())

    def _apply(met):
        for mp in met.get('matched_peaks') or ():
            v = votes.get(_signal_key(met, mp), {'accepted': [], 'good': []})
            mp['_per_sample_accepted'] = list(v['accepted'])
            if _enough(v['good']):
                mp['annotation_quality'] = 'good'
            elif _enough(v['accepted']):
                mp['annotation_quality'] = None
            else:
                mp['annotation_quality'] = 'bad'
                mp.setdefault('_status_reasons', []).append(
                    f'accepted in {len(v["accepted"])} sample(s), fewer than '
                    f'{min_group_presence} of any group')

    in_base = {str(m.get('id')) for m in base}
    for mid in sorted(set(first) - in_base):
        base.append(first[mid])
    for met in base:
        _apply(met)
    return base, sc, votes
