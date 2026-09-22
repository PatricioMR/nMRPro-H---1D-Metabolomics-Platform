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
from processing.annotation_scoring import SCORING_DEFAULTS
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
             tiers=('Good', 'Average'), params=None, log=lambda *a: None):
    """Annotate a study exactly as the application does: its search, its
    grouping, its passes, in its order -- and none of its window.

    Returns ``(metabolites, params)``, the same pair the bench's own
    ``annotate`` returns, so a caller can put one where the other was and
    compare what the engine then makes of it.
    """
    sc = dict(SCORING_DEFAULTS)
    sc['dppm_gold'] = float(tolerance_ppm)
    sc.update(params or {})
    annotatable = [p for p in peaks if is_annotatable(p, set(tiers))]
    found = search_database(
        db_path=db_path, db_type='hmdb', peaks_to_search=peaks,
        annotatable_peaks=annotatable, tiers=set(tiers), ppm=ppm,
        intensities=intensities, samples_info=samples_info,
        field_mhz=field_mhz, tolerance_ppm=tolerance_ppm, params=sc,
        noise_std=noise_std, min_snr=min_snr,
        min_distance_ppm=min_distance_ppm)
    log(f'  search: {len(found.results)} matches on '
        f'{found.peaks_matched}/{found.peaks_annotatable} peaks '
        f'({found.elapsed_s:.1f}s)')
    win = Window([], peaks, ppm, intensities, mhz=field_mhz, params=sc,
                 samples_info=samples_info, noise=noise_std,
                 excluded=())
    # the window's own grouping of matches into metabolites, which touches
    # nothing of itself
    win.metabolites = AnnotationResultsWindow._group_by_metabolite(
        win, found.results)
    log(f'  grouped into {len(win.metabolites)} metabolites')
    win.run(db_path, log=log)
    return win.metabolites, sc
