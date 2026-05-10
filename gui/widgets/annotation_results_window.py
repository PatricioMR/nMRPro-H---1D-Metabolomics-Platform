# gui/widgets/annotation_results_window.py
"""
Metabolite Annotation Results Window.

Layout:
  Left  (A): Hierarchical Treeview
             ├─ Metabolite name  (coverage %, n matched / n total peaks)
             │    ├─ Peak #1  3.2341 ppm  Δ 0.002  ✓ matched
             │    └─ Peak #2  1.4100 ppm  Δ 0.005  ✓ matched
             └─ ...

  Right (B): Fingerprint comparison plot
             • Grey filled area  = experimental spectrum (mean of all samples)
             • Coloured sticks   = DB fingerprint (multiplet centres + heights)
             • Matched sticks    = green;  unmatched = red
             • Info card with coverage %, cosine similarity, n matched
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gui.notation.evidence_weighting import (
    dynamic_composite,
    _BASE_WEIGHTS_ENRICHMENT_NS,
    _BASE_WEIGHTS_ENRICHMENT_S,
)
from gui.notation.confidence_classifier import (
    classify_annotation,
    CLASS_COLOURS,
)


# ── Local baseline helper ─────────────────────────────────────────────────────

def _local_linear_baseline(y, edge_frac=0.15, percentile=10):
    """
    Estimate a linear baseline from the low-percentile of the edge regions.
    Returns an array the same shape as y, clipped to not exceed y.
    Used to remove broad underlying signals before visualising narrow multiplets.

    Uses a larger edge fraction (15 %) and lower percentile (10th) so that
    broad baseline humps are properly removed even when the window edges
    are partially on the hump shoulders.
    """
    n      = len(y)
    n_edge = max(3, int(n * edge_frac))
    bl_l   = float(np.percentile(y[:n_edge],  percentile))
    bl_r   = float(np.percentile(y[-n_edge:], percentile))
    bl     = np.linspace(bl_l, bl_r, n)
    return np.minimum(bl, y)


# ── Multiplet intensity helpers ───────────────────────────────────────────────

def _binomial_coeffs(n):
    """Pascal's triangle row n (0-indexed). Row 1 = [1,1], row 2 = [1,2,1]..."""
    from math import comb
    return [comb(n - 1, k) for k in range(n)]


def _relative_intensities_from_type(mult_type, protons):
    """
    Return normalised relative intensities for a multiplet.
    E.g. triplet → [1, 2, 1] / 4  (sum=1)

    Multiplicity mapping (standard NMR):
      s  = singlet  → [1]
      d  = doublet  → [1, 1]
      t  = triplet  → [1, 2, 1]
      q  = quartet  → [1, 3, 3, 1]
      p  = pentet   → [1, 4, 6, 4, 1]
      dd = dd       → [1, 1, 1, 1]   (approximation)
      m  = multiplet → flat (protons equal height)
    """
    mt = (mult_type or 'm').lower().strip()
    mapping = {
        's':  [1],
        'd':  [1, 1],
        't':  [1, 2, 1],
        'q':  [1, 3, 3, 1],
        'p':  [1, 4, 6, 4, 1],
        'h':  [1, 5, 10, 10, 5, 1],
        'dd': [1, 1, 1, 1],
        'dt': [1, 2, 1, 1, 2, 1],
        'td': [1, 1, 2, 2, 1, 1],
    }
    if mt in mapping:
        coeffs = mapping[mt]
    elif mt == 'm' and protons and protons > 1:
        coeffs = [1] * protons
    else:
        coeffs = [1]
    total = sum(coeffs)
    return [c / total for c in coeffs]  # normalised, sum = 1


def _build_stick_positions(ppm_center, mult_type, protons, j_coupling_hz=None,
                           field_mhz=600.0, ppm_span=None):
    """
    Generate individual stick (ppm, relative_intensity) pairs for a multiplet.
    If J-coupling and field strength are known, space sticks by J/field (in ppm).
    If j_coupling_hz is absent but ppm_span (ppm_end - ppm_start from DB) is
    provided, derive spacing from the observed span.
    Otherwise fall back to a default spacing (~7 Hz at the given field).
    """
    intensities = _relative_intensities_from_type(mult_type, protons)
    n = len(intensities)
    if n == 1:
        return [(ppm_center, intensities[0])]

    # Try to parse J coupling
    j_hz = None
    if j_coupling_hz:
        try:
            j_hz = float(str(j_coupling_hz).split(',')[0].strip())
        except Exception:
            pass

    if j_hz:
        spacing_ppm = j_hz / field_mhz
    elif ppm_span and ppm_span > 1e-4 and n > 1:
        # Derive spacing from the DB-recorded ppm span (ppm_end - ppm_start)
        spacing_ppm = ppm_span / (n - 1)
    else:
        # Default ~7 Hz (typical 3J coupling) at the given field
        spacing_ppm = 7.0 / field_mhz

    # Centre the sticks symmetrically around ppm_center
    total_span = spacing_ppm * (n - 1)
    start_ppm  = ppm_center - total_span / 2
    sticks = [(start_ppm + i * spacing_ppm, intensities[i]) for i in range(n)]
    return sticks


def _rri_score(exp_peaks_in_region, db_sticks, tolerance=0.02):
    """
    Relative Intensity Ratio (RRI) score — how well the experimental
    relative intensities match the DB.

    Returns a score 0–1 (1 = perfect intensity match).
    """
    if len(db_sticks) < 2:
        return None  # singlet: no ratio to compare
    if not exp_peaks_in_region:
        return 0.0

    # For each DB stick, find the closest experimental signal
    db_ints  = []
    exp_ints = []
    for ppm_db, int_db in db_sticks:
        closest = min(exp_peaks_in_region, key=lambda p: abs(p[0] - ppm_db))
        if abs(closest[0] - ppm_db) <= tolerance:
            db_ints.append(int_db)
            exp_ints.append(closest[1])

    if len(db_ints) < 2:
        return None
    db_arr  = np.array(db_ints,  dtype=float)
    exp_arr = np.array(exp_ints, dtype=float)
    # Normalise both to max=1
    db_arr  /= db_arr.max()
    exp_arr /= exp_arr.max() if exp_arr.max() > 0 else 1
    # Mean absolute deviation from ideal ratios (1 = perfect, 0 = terrible)
    mad = float(np.mean(np.abs(db_arr - exp_arr)))
    return max(0.0, 1.0 - mad)


def _peak_shape_score(ppm_arr, intensity_arr, shifted_sticks,
                      center_ppm, db_entry, hwhm_default=0.004):
    """
    Compute how well the experimental spectrum shape overlaps the
    simulated DB multiplet (sum of Lorentzianas at shifted_sticks).

    Returns a score 0–1:
      1.0 = perfect overlap
      The score is cosine similarity between the experimental slice
      and the simulated DB shape over a ±6×HWHM window.
    """
    if not shifted_sticks:
        return None

    db_width = max(
        abs(db_entry.get('ppm_end', center_ppm) -
            db_entry.get('ppm_start', center_ppm)),
        0.002)
    hwhm = max(db_width / 2, hwhm_default)
    # Window: wide enough to contain the full multiplet + ~4 hwhm of shoulder,
    # but not so wide that neighbouring peaks or broad baseline contaminate the
    # cosine.  hwhm * 8 gives ≈ 4 × linewidth padding on each side.
    win  = hwhm * 8

    lo = center_ppm - win
    hi = center_ppm + win
    mask = (ppm_arr >= lo) & (ppm_arr <= hi)
    if mask.sum() < 3:
        return None

    x   = ppm_arr[mask]
    y_e = intensity_arr[mask].astype(float)

    # Remove linear baseline from experimental slice so that the broad baseline
    # hump doesn't dominate and drag the cosine similarity down.
    # Estimate from the 15% edge samples on each side.
    n = len(y_e)
    edge_n = max(2, int(n * 0.15))
    xl = float(np.mean(x[:edge_n]));   yl = float(np.mean(y_e[:edge_n]))
    xr = float(np.mean(x[-edge_n:]));  yr = float(np.mean(y_e[-edge_n:]))
    if abs(xr - xl) > 1e-9:
        slope = (yr - yl) / (xr - xl)
        y_e   = y_e - (yl + slope * (x - xl))
    y_e = np.clip(y_e, 0, None)   # only positive signal contributes

    # Simulated DB shape at same x points
    y_db = np.zeros_like(x)
    for st_ppm, st_rel in shifted_sticks:
        y_db += st_rel * hwhm**2 / ((x - st_ppm)**2 + hwhm**2)

    # Normalise both to unit vectors
    ne = np.linalg.norm(y_e)
    nd = np.linalg.norm(y_db)
    if ne == 0 or nd == 0:
        return 0.0
    return float(np.dot(y_e / ne, y_db / nd))


def _cosine_similarity(exp_ppm, exp_int, db_peaks, tolerance=0.05):
    """
    Compute cosine similarity between experimental spectrum (as a sparse
    vector at detected peak positions) and the DB fingerprint.
    Only considers peaks within the experimental ppm range.
    """
    if not db_peaks:
        return 0.0
    # Build vectors aligned on the DB peak positions
    v_exp, v_db = [], []
    for dp in db_peaks:
        dp_ppm = dp.get('ppm_center', dp.get('ppm', 0))
        # Find closest experimental peak
        diffs = [abs(ep - dp_ppm) for ep in exp_ppm]
        if not diffs:
            continue
        min_diff = min(diffs)
        idx      = diffs.index(min_diff)
        v_db.append(float(dp.get('height', dp.get('intensity', 1.0)) or 1.0))
        v_exp.append(exp_int[idx] if min_diff <= tolerance else 0.0)
    if not v_exp or not any(v_exp) or not any(v_db):
        return 0.0
    v_exp = np.array(v_exp, dtype=float)
    v_db  = np.array(v_db,  dtype=float)
    norm  = np.linalg.norm(v_exp) * np.linalg.norm(v_db)
    return float(np.dot(v_exp, v_db) / norm) if norm > 0 else 0.0


class AnnotationResultsWindow:
    def __init__(self, parent, results, peaks, ppm, all_intensities,
                 db_path, db_conn=None, elapsed=0, app=None,
                 saved_quality=None, progress_cb=None,
                 saved_metabolites=None):
        """
        Parameters
        ----------
        results           : flat list of match dicts from search_task
        peaks             : list of detected peak dicts (from PeakDetectionStage)
        ppm               : np.ndarray  full ppm axis
        all_intensities   : np.ndarray  (n_points × n_samples)
        db_path           : path to the sqlite DB (to fetch full fingerprints)
        db_conn           : open sqlite3.Connection  (may be None — reopen if needed)
        elapsed           : float  search time in seconds
        app               : App instance (for group colours)
        saved_quality     : dict {(met_id, peak_rank): 'good'|'bad'} to restore
                            user annotations when the window is reopened.
        progress_cb       : callable(step: int, total: int, msg: str) or None.
                            Called from the background thread via root.after so
                            the main thread can update the progress bar safely.
        saved_metabolites : pre-built metabolites list from a previous session.
                            When provided, skips _group_by_metabolite and uses
                            this directly — preserving all enrichment (delta_j_hz,
                            scores, anchor_status, etc.).
        """
        self.results          = results
        self.ppm              = ppm
        self.all_intensities  = all_intensities
        self.mean_intensity   = np.mean(all_intensities, axis=1)
        self.db_path          = db_path
        self.app              = app
        self._current_met     = None
        self._saved_metabolites = saved_metabolites  # restored from project

        # ── Annotation parameters (user-editable via ⚙ Parameters dialog) ────
        self._params = {
            # Score thresholds
            'good_score_thresh':  0.80,   # Rules 3/4/5: min score → Good
            'bad_score_thresh':   0.50,   # Rule 1: max score → Bad (singlets)
            # ΔJ criteria (Hz)
            'dj_gold_hz':         0.20,   # Rule 2 gold: ΔJ < this → Good candidate
            'dj_bad_hz':          0.50,   # Rule 0: ΔJ ≥ this → Bad (non-singlet); [0.20,0.50) stays yellow for J-Resolve
            'dj_soft_lo_hz':      0.50,   # s_dj ramp: ΔJ < this → s_dj=1.0
            'dj_soft_hi_hz':      3.00,   # s_dj ramp: ΔJ > this → s_dj=0.0
            'dj_plausible_hz':    2.00,   # multiplet confirmation criterion A
            'dj_neutral':         0.00,   # s_dj when ΔJ unmeasurable (excluded via j_conf=0)
            # Δppm criteria (ppm)
            'dppm_gold':          0.05,   # Rule 2: max Δppm for gold standard
            'dppm_anchor_high':   0.02,   # anchor pos high confidence threshold
            'dppm_anchor_med':    0.05,   # anchor pos medium confidence threshold
            # Fingerprint gate
            'fp_coverage_pct':    50.0,   # min % fingerprint coverage (≥2 DB peaks)
            # NNLS validation
            'nnls_min_frac':      0.03,   # min fraction of max NNLS coefficient
            # RANSY anchor thresholds
            'anchor_clean_lo':    0.35,   # area ratio lower bound → clean
            'anchor_clean_hi':    2.50,   # area ratio upper bound → clean
            'anchor_snr_bdl':     3.0,    # SNR below this → BDL
            # Score weights — non-singlet
            # sum = 0.57+0.14+0.10+0.05+0.09+0.05 = 1.00
            'w_dj':               0.57,
            'w_cosine_ns':        0.14,
            'w_ppm_ns':           0.10,
            'w_found_ns':         0.05,
            'w_rri_ns':           0.09,
            'w_reverse_ns':       0.05,   # reverse score (nH-weighted DB coverage)
            # Score weights — singlet
            # sum = 0.32+0.27+0.13+0.18+0.10 = 1.00
            'w_cosine_s':         0.32,
            'w_ppm_s':            0.27,
            'w_found_s':          0.13,
            'w_rri_s':            0.18,
            'w_reverse_s':        0.10,   # reverse score (nH-weighted DB coverage)
            # Score bonuses/penalties (independent NMR evidence, literature model)
            'bonus_gold_j':        0.20,   # Rule 2: gold standard J (all lines + ΔJ < 0.2 Hz + Δppm < 0.05)
            'gold_j_score_floor':  0.82,   # guaranteed score floor when gold J conditions are met
            'bonus_ransy_high':    0.15,   # Rule 6b: RANSY clean anchor + high position confidence
            'bonus_ransy':         0.10,   # Rule 6: RANSY clean anchor
            'bonus_mcr':           0.10,   # Rule MCR: MCR spectral deconvolution confirmed
            'penalty_singlet_only': 0.10,  # Rule 8: singlet-only without J confirmation
            'penalty_overlap_per': 0.05,   # Rules 3/4: −0.05 per competitor (capped at penalty_overlap_max)
            'penalty_overlap_max': 0.15,   # maximum overlap penalty cap
        }

        # ── Resolve the tk root for thread-safe after() calls ─────────────────
        _root = None
        if app is not None:
            _root = getattr(app, 'root', None)
        if _root is None:
            try:
                _root = parent.winfo_toplevel()
            except Exception:
                pass

        _TOTAL_STEPS = 11   # pure-data steps below

        def _step(n, msg):
            """Post a progress update to the main thread from any thread."""
            if progress_cb is None:
                return
            if _root is not None:
                try:
                    _root.after(0, progress_cb, n, _TOTAL_STEPS, msg)
                except Exception:
                    pass
            else:
                try:
                    progress_cb(n, _TOTAL_STEPS, msg)
                except Exception:
                    pass

        # ── Phase 1 (pure data — runs in caller's thread or a bg thread) ──────
        import threading as _threading

        _error   = []   # written by thread, read by main thread
        _ready   = _threading.Event()

        def _data_phase():
            try:
                _step(1, 'Expanding peaks…')
                self.peaks = self._expand_peaks_with_subpeaks(peaks)

                _step(2, 'Grouping metabolites…')
                _using_saved = self._saved_metabolites is not None
                if _using_saved:
                    self.metabolites = self._saved_metabolites
                else:
                    self.metabolites = self._group_by_metabolite(results)

                # Skip DB fingerprint fetch when using saved metabolites that
                # already have db_peaks — db_peaks, sticks, cosine, and rri
                # were all computed when the project was originally annotated.
                _has_saved_fingerprints = (
                    _using_saved
                    and bool(self.metabolites)
                    and all(met.get('db_peaks') for met in self.metabolites)
                )
                if _has_saved_fingerprints:
                    _step(3, f'Using saved fingerprints ({len(self.metabolites)} metabolites)…')
                else:
                    _step(3, f'Fetching DB fingerprints ({len(self.metabolites)} metabolites)…')
                    self._fetch_fingerprints(db_path)

                _step(4, 'Computing detectability…')
                self._compute_detectability()

                if not _using_saved:
                    _step(5, 'Restoring saved quality marks…')
                    if saved_quality:
                        self._apply_saved_quality(saved_quality)

                _step(6, 'Computing peak overlap…')
                self._compute_overlap()

                _step(7, 'Building candidate assignments…')
                self._build_child_candidates()

                if not _using_saved:
                    _step(8, 'Enriching ΔJ from spectrum…')
                    self._enrich_delta_j_from_spectrum()

                    _step(9, 'Computing scores…')
                    self._compute_scores()
                    self._propagate_multiplet_confirmation()
                    self._filter_incompatible_candidates()
                    self._propagate_j_anchor_quality()
                    self._propagate_anchor_position()
                    self._enrich_scores()
                    self._adjust_scores_for_competition()
                    self._classify_confidence_classes()
                    self._auto_mark_good()
                    self._require_multi_peak_confirmation()
                    self._nnls_spectral_validation()
                else:
                    _step(8, 'Restoring saved state…')
                    # Metabolites already have all quality marks embedded — do NOT
                    # re-run auto-marking or NNLS validation, which would alter
                    # auto-assigned marks and produce a different state than saved.
                    _step(9, '')

                _step(10, 'Building peak assignments…')
                self._build_peak_assignments()

                _step(11, 'Calibrating PPM axis…')
                self._calibrate_ppm_from_anchors()

            except Exception as _exc:
                import traceback
                _error.append(_exc)
                traceback.print_exc()
            finally:
                _ready.set()

        if progress_cb is not None and _root is not None:
            # Run data phase in background thread; block via Event.wait() would
            # freeze the main thread again.  Instead we chain via after().
            _t = _threading.Thread(target=_data_phase, daemon=True)
            _t.start()

            # Block main thread with a local event-loop pump until thread done.
            # This keeps Tk responsive (progress bar animates) while we wait.
            while not _ready.wait(timeout=0.05):
                try:
                    _root.update()
                except Exception:
                    break
        else:
            # No progress UI — just run synchronously (fast path / tests).
            _data_phase()

        if _error:
            raise _error[0]

        # ── Phase 2: UI construction (must be on main thread) ─────────────────
        if progress_cb is not None:
            try:
                progress_cb(_TOTAL_STEPS, _TOTAL_STEPS, 'Building interface…')
                if _root:
                    _root.update_idletasks()
            except Exception:
                pass

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"Annotation Results  —  {len(self.metabolites)} metabolites  "
            f"({elapsed:.1f}s)")
        self.win.geometry("1300x650")
        self.win.minsize(900, 500)

        self._iid_to_row      = {}
        self._iid_to_child    = {}
        self._iid_to_subchild = {}

        self._build()
        self._populate()
        self._center()
        if getattr(self, '_ppm_calibration_info', None):
            self.win.title(
                f"Annotation Results  [PPM calibrated: {self._ppm_calibration_info}]")

    def _apply_saved_quality(self, saved_quality):
        """Re-apply persisted quality marks to the freshly built metabolites."""
        for met in self.metabolites:
            for mp in met['matched_peaks']:
                key = (met['id'], mp.get('peak_rank'))
                if key in saved_quality:
                    mp['annotation_quality'] = saved_quality[key]

    # ── Data preparation ──────────────────────────────────────────────────────

    def _expand_peaks_with_subpeaks(self, peaks):
        """
        Mirror the expansion annotation_stage does before matching:
        replace deconvolved parent peaks with their individual sub-peaks so
        that orange markers and line counts refer to the same peaks that
        were actually matched against the DB.
        """
        try:
            _pk_stage = (getattr(self.app, 'stages', {}) or {}).get('peak_detection') \
                        or (getattr(self.app, 'stages', {}) or {}).get('peaks')
            _ann_st   = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            if _pk_stage is None and _ann_st is not None:
                _pk_stage = getattr(_ann_st, 'peak_stage', None)
            _ctrl = getattr(_pk_stage, '_deconv_controller', None) if _pk_stage else None
            if _ctrl is None or not _ctrl.results:
                return list(peaks)

            expanded = []
            for _pk in peaks:
                _rk = int(_pk.get('rank', -1))
                _found_sub = False
                for _ci, _cp in enumerate(_ctrl.peaks):
                    if int(_cp.get('rank', -1)) == _rk and _cp.get('_deconvolved'):
                        _res  = _ctrl.results.get(_ci, {})
                        if not _res.get('_saved'):
                            break
                        _subs = [sp for sp in _res.get('peaks', []) if sp.get('ppm')]
                        if _subs:
                            for _sp in _subs:
                                expanded.append({
                                    'ppm':         float(_sp['ppm']),
                                    'rank':        _rk,
                                    'status':      'Good',
                                    'height':      float(_sp.get('height', 0)),
                                    'width_ppm':   float(_sp.get('width',
                                                         _sp.get('width_ppm', 0.004))),
                                    'start_ppm':   float(_sp.get('start_ppm',
                                                         _sp['ppm'] - 0.005)),
                                    'end_ppm':     float(_sp.get('end_ppm',
                                                         _sp['ppm'] + 0.005)),
                                    'is_sub_peak': True,
                                    'snr':         float(_sp.get('snr', 5.0)),
                                    # carry region data for deconvolved display
                                    'region_ppm':       _sp.get('region_ppm'),
                                    'region_sub_peaks': _sp.get('region_sub_peaks', []),
                                })
                            _found_sub = True
                            break
                if not _found_sub:
                    expanded.append(_pk)
            return expanded
        except Exception:
            return list(peaks)

    def _group_by_metabolite(self, results):
        """
        Group flat match rows by metabolite ID.
        Returns list of dicts: {id, name, matched_peaks, coverage, ...}
        """
        groups = {}
        for r in results:
            mid  = (r.get('hmdb_id') or r.get('bmrb_id') or r.get('id', '?'))
            name = r.get('name', '?')
            if mid not in groups:
                groups[mid] = {
                    'id':            mid,
                    'name':          name,
                    'matched_peaks': [],   # (peak_ppm, db_ppm, ppm_diff, peak_rank)
                    'db_peaks':      [],   # full DB fingerprint (fetched later)
                    'coverage':      0.0,
                    'cosine':        0.0,
                    'source':        r.get('database', r.get('source', '?')),
                }
            dp_ppm = r.get('ppm_center', r.get('ppm', 0))
            pl     = r.get('predicted_lines', [])
            # Multiplet centre = intensity-weighted mean of matched lines.
            # This correctly handles doublet (1:1), triplet (1:2:1), etc.
            if len(pl) > 1:
                tot_rel = sum(l.get('rel_intensity', 1.0) for l in pl)
                if tot_rel > 0:
                    obs_centre = sum(l['ppm'] * l.get('rel_intensity', 1.0)
                                     for l in pl) / tot_rel
                else:
                    obs_centre = (pl[0]['ppm'] + pl[-1]['ppm']) / 2
                ppm_diff   = abs(obs_centre - dp_ppm)
                peak_ppm   = obs_centre
            else:
                peak_ppm  = r.get('peak_ppm', 0)
                ppm_diff  = r.get('ppm_diff', abs(peak_ppm - dp_ppm))
            groups[mid]['matched_peaks'].append({
                'peak_ppm':        peak_ppm,
                'peak_rank':       r.get('peak_rank', '?'),
                'db_ppm':          dp_ppm,
                'db_start_ppm':    r.get('start_ppm', dp_ppm),
                'db_end_ppm':      r.get('end_ppm', dp_ppm),
                'is_lipidomics':   bool(r.get('is_lipidomics', False)),
                'ppm_diff':        ppm_diff,
                'protons':         r.get('protons', ''),
                'type':            r.get('type', r.get('match_type', '')),
                'predicted_lines': pl,
                'multiplet_score': r.get('multiplet_score', 0.0),
                'composite_score': r.get('composite_score', 0.0),
                'multiplet_detail':r.get('multiplet_detail', ''),
                'j_coupling':      r.get('j_coupling', ''),
                'is_sub_peak':     bool(r.get('is_sub_peak', False)),
                'coherence_score': r.get('coherence_score', 1.0),
                'peak_coverage':   r.get('peak_coverage',   1.0),
                'ratio_match':     r.get('ratio_match',      1.0),
                'mcr_r2':          r.get('mcr_r2'),
                'mcr_pearson_r':   r.get('mcr_pearson_r'),
                'mcr_confirmed':   r.get('mcr_confirmed'),
                'mcr_cross_corr':  r.get('mcr_cross_corr'),
                'stocsy_coherence':r.get('stocsy_coherence'),
            })
            # Track best composite score for metabolite sorting
            cs = r.get('composite_score', 0)
            if cs > groups[mid].get('best_score', 0):
                groups[mid]['best_score'] = cs

        # Keep a single representative matched peak per (peak_rank, db_ppm) key,
        # preferring the highest score to avoid duplicated rows for the same peak.
        for _mid, g in groups.items():
            best_by_key = {}
            for mp in g.get('matched_peaks', []):
                key = (
                    mp.get('peak_rank'),
                    round(float(mp.get('db_ppm', 0) or 0), 4),
                )
                prev = best_by_key.get(key)
                if prev is None:
                    best_by_key[key] = mp
                    continue
                cur_score = float(mp.get('composite_score', mp.get('multiplet_score', 0)) or 0)
                prv_score = float(prev.get('composite_score', prev.get('multiplet_score', 0)) or 0)
                if cur_score > prv_score:
                    best_by_key[key] = mp
                elif cur_score == prv_score:
                    if float(mp.get('ppm_diff', 999)) < float(prev.get('ppm_diff', 999)):
                        best_by_key[key] = mp
            g['matched_peaks'] = list(best_by_key.values())
        return list(groups.values())

    def _fetch_fingerprints(self, db_path):
        """Fetch fingerprint peaks. Supports BMRB (shifts_1h) and HMDB (multiplets) schemas."""
        import sqlite3
        if not db_path or not str(db_path).endswith('.db'):
            # No SQLite database available (e.g., built-in CSV library) —
            # populate db_peaks from the matched peak ranges already in results.
            for met in self.metabolites:
                if not met['db_peaks']:
                    met['db_peaks'] = [
                        {'ppm_center': ((float(mp.get('db_start_ppm', mp['db_ppm'])) +
                                         float(mp.get('db_end_ppm',   mp['db_ppm']))) / 2.0)
                                       if mp.get('is_lipidomics') else float(mp['db_ppm']),
                         'ppm_start':  min(float(mp.get('db_start_ppm', mp['db_ppm'])),
                                           float(mp.get('db_end_ppm',   mp['db_ppm'])))
                                       if mp.get('is_lipidomics') else float(mp['db_ppm']) - 0.005,
                         'ppm_end':    max(float(mp.get('db_start_ppm', mp['db_ppm'])),
                                           float(mp.get('db_end_ppm',   mp['db_ppm'])))
                                       if mp.get('is_lipidomics') else float(mp['db_ppm']) + 0.005,
                         'type':       mp.get('type', ''),
                         'protons':    1}
                        for mp in met['matched_peaks']
                    ]
                n_db = len(met['db_peaks'])
                n_matched = len(set(round(mp['db_ppm'], 3)
                                    for mp in met['matched_peaks']))
                met['coverage'] = (n_matched / n_db * 100) if n_db else 0.0
                # Cosine falls back to coverage fraction when no spectral DB
                met['cosine'] = met['coverage'] / 100.0
            return
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur  = conn.cursor()
            tables = {r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            is_bmrb = 'shifts_1h' in tables
            for met in self.metabolites:
                mid = met['id']
                if is_bmrb:
                    cur.execute(
                        "SELECT ppm AS ppm_center, ppm AS ppm_start,"
                        " ppm AS ppm_end, '' AS protons,"
                        " multiplicity AS type, j_coupling, atom_name"
                        " FROM shifts_1h WHERE bmrb_id = ?"
                        " ORDER BY ppm DESC", (mid,))
                    rows = [dict(r) for r in cur.fetchall()]
                    if 'transitions' in tables:
                        for dp in rows:
                            cur.execute(
                                "SELECT ppm, intensity FROM transitions"
                                " WHERE bmrb_id = ? AND ABS(ppm - ?) < 0.15"
                                " ORDER BY ppm DESC", (mid, dp['ppm_center']))
                            tr = cur.fetchall()
                            if tr:
                                dp['sticks'] = [(r['ppm'], r['intensity'] or 1.0)
                                                for r in tr]
                    met['db_peaks'] = rows
                else:
                    cur.execute(
                        "SELECT ppm_center, ppm_start, ppm_end, protons, type, j_coupling "
                        "FROM multiplets WHERE hmdb_id = ? "
                        "ORDER BY ppm_center DESC", (met['id'],))
                    rows = [dict(r) for r in cur.fetchall()]
                    if not rows:
                        cur.execute(
                            "SELECT ppm AS ppm_center, ppm AS ppm_start,"
                            " ppm AS ppm_end, '' AS protons, 'peak' AS type"
                            " FROM peaks WHERE hmdb_id = ? ORDER BY ppm DESC",
                            (met['id'],))
                        rows = [dict(r) for r in cur.fetchall()]
                    met['db_peaks'] = rows

                # Remove unresolved multiplets (M) from db_peaks — they are
                # excluded from annotation matching and cannot be fingerprinted.
                met['db_peaks'] = [
                    p for p in met['db_peaks']
                    if (p.get('type') or '').lower().strip() not in ('m', 'multiplet')
                ]

                # Compute coverage: what fraction of DB peaks are matched
                n_db      = len(met['db_peaks'])
                n_matched = len(set(
                    round(mp['db_ppm'], 3)
                    for mp in met['matched_peaks']))
                met['coverage'] = (n_matched / n_db * 100) if n_db else 0.0

                # Build stick positions + attach scorer predicted_lines
                _mp_by_dbppm = {round(mp['db_ppm'], 3): mp
                                for mp in met['matched_peaks']}
                for dp in met['db_peaks']:
                    dp_key = round(dp.get('ppm_center', dp.get('ppm', 0)), 3)
                    sticks = _build_stick_positions(
                        dp.get('ppm_center', dp.get('ppm', 0)),
                        dp.get('type', 'm'),
                        dp.get('protons', 1),
                        dp.get('j_coupling'))
                    dp['sticks'] = sticks
                    mp_ref = _mp_by_dbppm.get(dp_key)
                    dp['predicted_lines'] = (
                        mp_ref.get('predicted_lines', []) if mp_ref else [])

                # Per-peak shape score: for each matched peak, compute how
                # well the experimental shape overlaps the simulated DB multiplet.
                # Score = max(individual peak scores) * 0.7
                #       + mean(other peak scores)  * 0.3
                # → one perfect match already gives a high score.
                peak_scores = []
                ppm_arr = self.ppm
                int_arr = self.mean_intensity
                has_spec = (ppm_arr is not None and int_arr is not None
                            and len(ppm_arr) > 10)
                for mp in met['matched_peaks']:
                    db_ppm   = mp['db_ppm']
                    # Always find closest db_peak (no hard threshold)
                    db_entry = (min(met['db_peaks'],
                                    key=lambda p: abs(p.get('ppm_center',
                                        p.get('ppm', 0)) - db_ppm))
                                if met['db_peaks'] else None) or \
                               {'ppm_center': db_ppm,
                                'ppm_start': db_ppm, 'ppm_end': db_ppm}
                    # Build sticks — prefer real transitions, then reconstruct
                    pred_lines = mp.get('predicted_lines') or []
                    if pred_lines:
                        shifted = [(l['ppm'], l['rel_intensity'])
                                   for l in pred_lines]
                    else:
                        sticks = db_entry.get('sticks') or []
                        if not sticks:
                            mult = db_entry.get('type', 's') or 's'
                            try:    protons = int(db_entry.get('protons') or 1)
                            except: protons = 1
                            sticks = _build_stick_positions(
                                db_ppm, mult, protons, db_entry.get('j_coupling'))
                        if not sticks:
                            sticks = [(db_ppm, 1.0)]
                        offset  = mp['peak_ppm'] - db_ppm
                        shifted = [(sp + offset, si) for sp, si in sticks]
                    if not has_spec:
                        peak_scores.append(
                            max(0.0, 1.0 - mp.get('ppm_diff', 0) / 0.05))
                        continue
                    pk = next((p for p in self.peaks
                               if abs(p['ppm'] - mp['peak_ppm']) < 0.01), None)
                    hwhm = (max(pk.get('width_ppm',
                                pk.get('width_at_half', 0.004)), 0.002) / 2
                            if pk else 0.004)
                    ps = _peak_shape_score(
                        ppm_arr, int_arr, shifted,
                        mp['peak_ppm'], db_entry, hwhm_default=hwhm)
                    if ps is None:
                        ps = max(0.0, 1.0 - mp.get('ppm_diff', 0) / 0.05)
                    if ps is not None:
                        peak_scores.append(ps)

                if peak_scores:
                    best  = max(peak_scores)
                    rest  = sorted(peak_scores)[:-1]  # all but best
                    mean_rest = float(np.mean(rest)) if rest else best
                    # Best peak dominates (70%), others contribute (30%)
                    met['cosine'] = best * 0.70 + mean_rest * 0.30
                else:
                    met['cosine'] = met['coverage'] / 100.0

                # RRI across matched multiplets
                exp_peaks_region = []
                if has_spec and self.peaks:
                    for p in self.peaks:
                        idx = int(np.argmin(np.abs(ppm_arr - p['ppm'])))
                        exp_peaks_region.append(
                            (p['ppm'], float(int_arr[idx])))
                rri_scores = []
                for dp in met['db_peaks']:
                    rri = _rri_score(exp_peaks_region, dp.get('sticks', []))
                    if rri is not None:
                        rri_scores.append(rri)
                met['rri'] = float(np.mean(rri_scores)) if rri_scores else None

            conn.close()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Fingerprint fetch error: {e}")

        # Safety net: cosine must not stay at 0 when coverage > 0
        for met in self.metabolites:
            if met.get('cosine', 0.0) == 0.0 and met.get('coverage', 0.0) > 0:
                met['cosine'] = met['coverage'] / 100.0

    # ── PPM axis calibration ──────────────────────────────────────────────────

    def _calibrate_ppm_from_anchors(self):
        """
        Use J-confirmed anchor metabolites (ΔJ < 0.5 Hz) as internal chemical-shift
        references to shift the DB reference values toward the experimental spectrum.

        The experimental ppm axis and all_intensities are NEVER modified — they are
        the ground truth. Instead, db_ppm values stored in each matched peak (mp)
        are shifted so that ppm_diff and ppm_score reflect the corrected alignment.

        The computed shift is saved to app._ppm_calibration so the annotation stage
        can apply the same offset on the next search run.

        With 1 anchor  → constant offset: db_ppm_corrected = db_ppm + Δ
        With 2+ anchors → per-region shift via linear fit or piecewise interp
        """
        self._ppm_calibration_info = None

        if self.ppm is None:
            return

        # Gather control points from J-confirmed assignments
        # obs_ppm = where the peak actually is in the spectrum
        # db_ppm  = where the DB says it should be
        # Δ = obs_ppm - db_ppm → how much to add to db_ppm to match the spectrum
        control = []   # (obs_ppm, db_ppm, met_name)
        seen_ranks = set()
        for rank, (met, mp) in self._peak_assignments.items():
            if rank in seen_ranks:
                continue
            dj = mp.get('delta_j_hz')
            if dj is None or dj >= 0.5:
                continue
            if mp.get('annotation_quality') == 'bad':
                continue
            obs_ppm = float(mp.get('peak_ppm') or 0)
            db_ppm  = float(mp.get('db_ppm') or mp.get('ppm_center') or 0)
            if obs_ppm == 0 or db_ppm == 0:
                continue
            if abs(obs_ppm - db_ppm) < 1e-5:
                continue   # already aligned
            control.append((obs_ppm, db_ppm, met.get('name', '?')))
            seen_ranks.add(rank)

        if not control:
            return

        obs_arr = np.array([c[0] for c in control])
        db_arr  = np.array([c[1] for c in control])
        names   = [c[2] for c in control]
        # Δ per anchor: how much to add to db_ppm to reach obs_ppm
        delta_arr = obs_arr - db_arr   # positive = spectrum is downfield of DB

        n = len(control)

        if n == 1:
            delta_mean = float(delta_arr[0])
            def _db_shift(db_ppm_val):
                return db_ppm_val + delta_mean
            info = f"DB shifted {delta_mean:+.4f} ppm via {names[0]}"

        elif n == 2:
            # Linear interpolation between the two anchors
            coeffs = np.polyfit(db_arr, obs_arr, 1)
            def _db_shift(db_ppm_val):
                return float(np.polyval(coeffs, db_ppm_val))
            delta_mean = float(np.mean(delta_arr))
            info = (f"DB linear shift (mean {delta_mean:+.4f} ppm) via "
                    f"{', '.join(names)}")

        else:
            # Piecewise: for each db_ppm, interpolate the expected obs_ppm
            idx   = np.argsort(db_arr)
            db_s  = db_arr[idx]
            obs_s = obs_arr[idx]
            def _db_shift(db_ppm_val):
                arr = np.asarray(db_ppm_val, dtype=float)
                lo_ext = obs_s[0]  + (arr - db_s[0])
                hi_ext = obs_s[-1] + (arr - db_s[-1])
                out = np.interp(arr, db_s, obs_s)
                out = np.where(arr < db_s[0],  lo_ext, out)
                out = np.where(arr > db_s[-1], hi_ext, out)
                return float(out) if out.ndim == 0 else out
            delta_mean = float(np.mean(delta_arr))
            info = (f"DB piecewise shift ({n} anchors, mean {delta_mean:+.4f} ppm) via "
                    f"{', '.join(names)}")

        # Get tolerance for ppm_score recomputation
        try:
            ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            tolerance = float(ann_stage.tolerance_var.get()) if ann_stage else 0.05
        except Exception:
            tolerance = 0.05

        def _gauss_ppm_score(exp, db):
            sigma = tolerance / 2.0
            return float(np.exp(-0.5 * ((exp - db) / sigma) ** 2))

        # Apply: shift db_ppm in every matched peak, recompute ppm_diff / ppm_score
        # The experimental peak_ppm is NOT touched.
        for met in self.metabolites:
            for mp in met['matched_peaks']:
                obs_ppm = float(mp.get('peak_ppm') or 0)
                if obs_ppm == 0:
                    continue
                old_db = float(mp.get('db_ppm') or mp.get('ppm_center') or 0)
                if old_db == 0:
                    continue
                new_db = float(_db_shift(old_db))
                mp['db_ppm_calibrated'] = new_db   # store separately, preserve original
                mp['ppm_diff']  = float(abs(obs_ppm - new_db))

        # Recompute scores after ppm_diff is updated, then rebuild assignments
        self._compute_scores()
        self._propagate_multiplet_confirmation()   # depends on _s_dj set by scores
        self._filter_incompatible_candidates()
        self._propagate_j_anchor_quality()
        self._propagate_anchor_position()
        self._enrich_scores()
        self._adjust_scores_for_competition()
        self._classify_confidence_classes()
        self._auto_mark_good()
        self._require_multi_peak_confirmation()
        self._nnls_spectral_validation()
        self._build_peak_assignments()

        self._ppm_calibration_info = info

        # Persist to app so next annotation search applies the same shift
        if self.app is not None:
            try:
                self.app._ppm_calibration = {
                    'info':       info,
                    'delta_mean': delta_mean,
                    'anchors':    [(float(o), float(d), nm)
                                   for o, d, nm in zip(obs_arr, db_arr, names)],
                    'n':          n,
                }
            except Exception:
                pass


    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        paned = tk.PanedWindow(
            self.win, orient=tk.HORIZONTAL,
            sashwidth=6, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Panel A ───────────────────────────────────────────────────────────
        frame_a = ttk.Frame(paned)
        paned.add(frame_a, minsize=420)

        n_pk  = len(set(mp['peak_rank']
                        for m in self.metabolites
                        for mp in m['matched_peaks']))
        n_met = len(self.metabolites)
        self._summary_lbl = ttk.Label(
            frame_a,
            text=f"{n_pk} peaks assigned  ·  {n_met} metabolites",
            font=("Arial", 9, "italic"), foreground="gray")
        self._summary_lbl.pack(anchor=tk.W, padx=4, pady=(4, 2))
        ttk.Label(frame_a,
                  text="Ctrl+1 = Good  ·  Ctrl+2 = Undef (→ MCR)  ·  Ctrl+3 = Bad",
                  font=("Arial", 7), foreground="#888").pack(
            anchor=tk.W, padx=4, pady=(0, 2))

        # Name search
        sf = ttk.Frame(frame_a)
        sf.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Label(sf, text="🔍").pack(side=tk.LEFT)
        self._flt_name = tk.StringVar()
        ttk.Entry(sf, textvariable=self._flt_name,
                  width=22).pack(side=tk.LEFT, padx=2)
        self._flt_name.trace_add('write', lambda *_: self._populate())
        ttk.Button(sf, text='✕', width=2,
                   command=lambda: self._flt_name.set('')
                   ).pack(side=tk.LEFT)

        # Quality filter checkboxes
        qf = ttk.Frame(frame_a)
        qf.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Label(qf, text='Show:', font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 4))
        self._ann_filt_good  = tk.BooleanVar(value=True)
        self._ann_filt_bad   = tk.BooleanVar(value=True)
        self._ann_filt_undef = tk.BooleanVar(value=True)
        self._ann_filt_bdl   = tk.BooleanVar(value=False)
        # 'm' multiplets are permanently excluded (no J values available)
        self._ann_filt_mult_m = False
        for text, var, bg, fg in [
            ('✓ Good',  self._ann_filt_good,  '#c8f7c5', '#1a7a1a'),
            ('✗ Bad',   self._ann_filt_bad,   '#ffd6d6', '#9a0000'),
            ('— Undef', self._ann_filt_undef, '#f0f0f0', '#444444'),
        ]:
            tk.Checkbutton(qf, text=text, variable=var,
                           bg=bg, fg=fg, selectcolor=bg,
                           activebackground=bg, font=('Arial', 8),
                           command=self._populate).pack(side=tk.LEFT, padx=2)

        # Treeview
        tf  = ttk.Frame(frame_a)
        tf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        vsb = ttk.Scrollbar(tf, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tf, orient=tk.HORIZONTAL)

        self.tree = ttk.Treeview(
            tf,
            columns=('name', 'ppm', 'start', 'end', 'score', 'quality', 'overlap'),
            show='tree headings',
            selectmode='browse',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)

        self.tree.heading('#0',      text='Peak #',
                          command=lambda: self._sort_by('ppm'))
        self.tree.heading('name',    text='Metabolite',
                          command=lambda: self._sort_by('name'))
        self.tree.heading('ppm',     text='PPM',
                          command=lambda: self._sort_by('ppm'))
        self.tree.heading('start',   text='Start')
        self.tree.heading('end',     text='End')
        self.tree.heading('score',   text='Score',
                          command=lambda: self._sort_by('score'))
        self.tree.heading('quality', text='Quality / Match')
        self.tree.heading('overlap', text='Overlap')
        self.tree.column('#0',      width=90,  minwidth=70)
        self.tree.column('name',    width=190, minwidth=130)
        self.tree.column('ppm',     width=80,  minwidth=60, anchor=tk.CENTER)
        self.tree.column('start',   width=78,  minwidth=60, anchor=tk.CENTER)
        self.tree.column('end',     width=78,  minwidth=60, anchor=tk.CENTER)
        self.tree.column('score',   width=65,  minwidth=50, anchor=tk.CENTER)
        self.tree.column('quality', width=70,  minwidth=50, anchor=tk.CENTER)
        self.tree.column('overlap', width=65,  minwidth=50, anchor=tk.CENTER)

        self.tree.tag_configure('met_high',   foreground='#1a7a1a',
                                background='#f0fff0')
        self.tree.tag_configure('met_mid',    foreground='#7a6000',
                                background='#fffff0')
        self.tree.tag_configure('met_low',    foreground='#9a0000',
                                background='#fff0f0')
        self.tree.tag_configure('peak_match', foreground='#1a1a2e')
        self.tree.tag_configure('mult_info',  foreground='#1a5c2e',
                                background='#e8f5e9', font=('Courier', 8))
        self.tree.tag_configure('qual_good',  background='#d6f5d6')
        self.tree.tag_configure('qual_bad',   background='#f5d6d6')
        self.tree.tag_configure('qual_mid',   background='#fff3cc')  # overlap warning
        self.tree.tag_configure('qual_bdl',   foreground='#999999', background='#f0f0f0')
        self.tree.tag_configure('sub_peak',          foreground='#444444', background='#f8f8ff')
        self.tree.tag_configure('sub_peak_matched',   foreground='#115511', background='#d6f5d6')
        self.tree.tag_configure('sub_peak_unmatched', foreground='#771111', background='#f5d6d6')
        self.tree.tag_configure('anchor_line', foreground='#0044cc', background='#ddeeff',
                                font=('Courier', 8, 'bold'))
        self.tree.tag_configure('singlet_anchored', foreground='#004466',
                                background='#d4f0ff')
        self.tree.tag_configure('qual_poor', foreground='#7a3d00',
                                background='#fff0d0')
        # Confidence class tags (A–X)
        for _cls, (_bg, _fg) in CLASS_COLOURS.items():
            self.tree.tag_configure(
                f'class_{_cls}', foreground=_fg, background=_bg,
                font=('TkDefaultFont', 9, 'bold'))
        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        self.tree.bind('<Button-3>', self._on_right_click)
        self.tree.bind('<<TreeviewClose>>', self._on_tree_close)
        self.win.bind('<Control-Key-1>', lambda e: self._classify('good'))
        self.win.bind('<Control-Key-2>', lambda e: self._classify(None))  # → Undef → MCR
        self.win.bind('<Control-Key-3>', lambda e: self._classify('bad'))

        # Export / action buttons
        bf = ttk.Frame(frame_a)
        bf.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(bf, text="Export CSV",
                   command=self._export).pack(side=tk.LEFT)
        ttk.Button(bf, text="Close",
                   command=self.win.destroy).pack(side=tk.RIGHT)

        # ── Panel B ───────────────────────────────────────────────────────────
        frame_b = ttk.Frame(paned)
        paned.add(frame_b, minsize=500)

        ttk.Label(frame_b, text="Fingerprint Comparison",
                  font=("Arial", 10, "bold")).pack(
            anchor=tk.W, padx=4, pady=(4, 0))
        self.match_info = ttk.Label(
            frame_b, text="Select a metabolite to see fingerprint",
            font=("Arial", 8), foreground="gray")
        self.match_info.pack(anchor=tk.W, padx=4)
        ttk.Label(frame_b,
                  text="Tip: right-drag on zoom view to select peak region",
                  font=("Arial", 7), foreground="#999").pack(anchor=tk.W, padx=4)

        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('#f8f8f8')

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=frame_b)
        self.plot_canvas.draw()
        
        # Toolbar: BOTTOM inside frame_b, before canvas TOP
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self._toolbar = NavigationToolbar2Tk(self.plot_canvas, frame_b, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Scroll/pan on the fingerprint plot
        self._pan_active = False
        self._pan_start  = None
        self.plot_canvas.mpl_connect('scroll_event',         self._zoom)
        self.plot_canvas.mpl_connect('button_press_event',   self._pan_press)
        self.plot_canvas.mpl_connect('button_release_event', self._pan_release)
        self.plot_canvas.mpl_connect('motion_notify_event',  self._pan_move)

        # Quality card — scrollable so it never gets cut off
        qf_outer = ttk.LabelFrame(frame_b, text="Match Quality", padding=4)
        qf_outer.pack(fill=tk.X, padx=4, pady=(0, 2))

        _qf_canvas = tk.Canvas(qf_outer, height=110, bd=0, highlightthickness=0)
        _qf_vsb    = ttk.Scrollbar(qf_outer, orient=tk.VERTICAL,
                                   command=_qf_canvas.yview)
        _qf_canvas.configure(yscrollcommand=_qf_vsb.set)
        _qf_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        _qf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        qf = ttk.Frame(_qf_canvas, padding=2)
        _qf_win = _qf_canvas.create_window((0, 0), window=qf, anchor='nw')

        def _qf_configure(e):
            _qf_canvas.configure(scrollregion=_qf_canvas.bbox('all'))
        def _qf_canvas_resize(e):
            _qf_canvas.itemconfig(_qf_win, width=e.width)
        qf.bind('<Configure>', _qf_configure)
        _qf_canvas.bind('<Configure>', _qf_canvas_resize)

        def _qf_wheel(e, c=_qf_canvas):
            if e.num == 4:   c.yview_scroll(-1, 'units')
            elif e.num == 5: c.yview_scroll(1, 'units')
            else:            c.yview_scroll(int(-e.delta / 60), 'units')
        _qf_canvas.bind('<MouseWheel>', _qf_wheel)
        _qf_canvas.bind('<Button-4>',   _qf_wheel)
        _qf_canvas.bind('<Button-5>',   _qf_wheel)

        self._q_labels = {}
        for i, (key, label, hint) in enumerate([
            ('coverage',  'Coverage',    '% DB peaks matched'),
            ('cosine',    'Similarity',  'Cosine similarity (0–1)'),
            ('n_matched', 'Matched/RRI', 'peaks found / total · RRI'),
            ('j_obs',     'J / ΔJ',      'J obs vs DB (Hz) · right-drag to reselect'),
            ('mcr',       'MCR',         'R²  Pearson r  cross-sample corr (overlap regions)'),
            ('stocsy',    'STC-Like',    'Mean pairwise Pearson r between matched peaks across samples'),
            ('reverse',   'Reverse',     'nH-weighted DB coverage (MS/MS reverse score, 0–1)'),
        ]):
            ttk.Label(qf, text=f"{label}:",
                      font=("Arial", 8, "bold")).grid(
                row=i, column=0, sticky=tk.W, padx=4, pady=1)
            vl = ttk.Label(qf, text="—", font=("Arial", 8))
            vl.grid(row=i, column=1, sticky=tk.W, padx=4)
            ind = tk.Label(qf, text="●", font=("Arial", 10), fg='#cccccc')
            ind.grid(row=i, column=2, padx=2)
            ttk.Label(qf, text=hint,
                      font=("Arial", 7), foreground='#666').grid(
                row=i, column=3, sticky=tk.W, padx=2)
            self._q_labels[key] = (vl, ind)

        # Reset button row inside quality card
        reset_row = ttk.Frame(qf)
        reset_row.grid(row=6, column=0, columnspan=4, sticky=tk.EW, pady=(4, 2))
        self._reset_btn = tk.Button(
            reset_row, text='↺ Reset peaks to original',
            font=('Arial', 8), fg='#cc5500', bg='#fff8f0',
            relief='groove', cursor='hand2', command=self._rc_reset)
        self._reset_btn.pack(side=tk.LEFT, padx=4)
        self._reset_modified_lbl = ttk.Label(
            reset_row, text='', font=('Arial', 7, 'italic'), foreground='#cc5500')
        self._reset_modified_lbl.pack(side=tk.LEFT)

        self._current_det_pk_mcr = None  # last det_pk with MCR

        # ── Right-click drag state (region selection in zoom view) ────────────
        self._rc_drag_start   = None   # xdata where right-drag started
        self._rc_span_patch   = None   # axvspan shown during drag
        self._current_dp      = None   # db_peak currently shown in zoom
        self._current_met_dp  = None   # met currently shown in zoom
        self._current_mp_ref  = None   # matched_peak ref currently shown
        self._orig_pred_lines = None   # backup taken on FIRST display of dp

        # ── Right-click drag canvas events ────────────────────────────────────
        # Use tkinter native bindings for right-drag so the NavigationToolbar
        # (which intercepts button=3 in pan/zoom mode) cannot block the events.
        _cvw = self.plot_canvas.get_tk_widget()
        _cvw.bind('<ButtonPress-3>',   self._tk_rc_press)
        _cvw.bind('<B3-Motion>',       self._tk_rc_move)
        _cvw.bind('<ButtonRelease-3>', self._tk_rc_release)

    # ── Detectability & Overlap ───────────────────────────────────────────────

    def _compute_detectability(self):
        """
        For each matched peak, estimate its SNR from the observed spectrum and
        the relative proton count within the metabolite.  Peaks with expected
        SNR < 3.0 are marked  below_detection_limit = True  so they are
        excluded from overlap detection and hidden from the tree.

        Also marks db_peaks with  bdl = True  when their expected SNR (propagated
        from the anchor matched peak via nH ratio) is below the noise threshold.
        BDL db_peaks are hidden from the child-row list entirely — they are not
        shown as ✓ or ✗ because the signal cannot be expected to appear in the
        experimental spectrum.
        """
        ppm_arr  = self.ppm
        int_arr  = self.mean_intensity
        if ppm_arr is None or int_arr is None or len(ppm_arr) < 10:
            return

        # Noise estimate: MAD of the spectrum tail (far upfield, > 9.5 ppm)
        tail_mask = ppm_arr > 9.5
        if tail_mask.sum() < 20:
            # fall back: lowest 5th percentile of the whole spectrum
            noise = float(np.percentile(np.abs(int_arr), 5)) or 1.0
        else:
            tail = int_arr[tail_mask]
            noise = float(np.median(np.abs(tail - np.median(tail)))) * 1.4826 or 1.0
        if noise <= 0:
            noise = 1.0
        self._noise_level = noise   # stored for _propagate_j_anchor_quality

        for met in self.metabolites:
            matched = met.get('matched_peaks', [])

            # Compute observed SNR for each matched peak from the spectrum
            for mp in matched:
                ppk = mp.get('peak_ppm', mp.get('db_ppm', 0))
                idx = int(np.argmin(np.abs(ppm_arr - ppk)))
                mp['obs_intensity'] = float(int_arr[idx])
                mp['obs_snr']       = mp['obs_intensity'] / noise

            if not matched:
                continue

            # Anchor = matched peak with highest observed SNR × nH
            # (nH × SNR gives the best proxy for concentration contribution)
            def _nh(mp):
                try:
                    return int(mp.get('protons', 1) or 1)
                except Exception:
                    return 1

            anchor        = max(matched, key=lambda p: p.get('obs_snr', 0) * _nh(p))
            anchor_snr    = anchor.get('obs_snr', 0)
            anchor_protons = _nh(anchor)

            # Mark matched_peaks
            for mp in matched:
                p_protons = _nh(mp)
                expected_snr = (anchor_snr * (p_protons / anchor_protons)
                                if anchor_protons > 0 else anchor_snr)
                mp['expected_snr']          = expected_snr
                mp['below_detection_limit'] = expected_snr < 3.0

            # Mark db_peaks — covers both matched and unmatched DB entries.
            # An unmatched DB peak that is BDL is simply absent because it is
            # below the noise floor, not because the metabolite is wrong.
            for dp in met.get('db_peaks', []):
                try:
                    dp_protons = int(dp.get('protons', 1) or 1)
                except Exception:
                    dp_protons = 1
                expected_snr = (anchor_snr * (dp_protons / anchor_protons)
                                if anchor_protons > 0 else anchor_snr)
                dp['bdl']          = expected_snr < 3.0
                dp['expected_snr'] = expected_snr

    def _compute_overlap(self):
        """Build peak_rank → [metabolite names] map for the overlap column.
        BDL peaks are excluded — they are not considered real competitors.

        For multiplets, ALL predicted lines' closest experimental peaks are
        registered so that a doublet in metabolite X blocks anchor status for
        metabolite Y even when Y only overlaps one line of the doublet.
        """
        tol = 0.015  # ppm tolerance for matching predicted lines to exp peaks

        # Fast lookup: ppm value → rank for experimental peaks
        pk_ppm_rank = {}
        for pk in (self.peaks or []):
            rk = pk.get('rank')
            if rk is not None:
                pk_ppm_rank[float(pk.get('ppm', 0))] = rk
        pk_ppms = sorted(pk_ppm_rank.keys())

        def _find_rank(ppm_val):
            best_d, best_r = tol, None
            for p in pk_ppms:
                d = abs(p - ppm_val)
                if d < best_d:
                    best_d, best_r = d, pk_ppm_rank[p]
                elif p > ppm_val + tol:
                    break
            return best_r

        peak_mets = {}
        for met in self.metabolites:
            name = met['name']
            for mp in met['matched_peaks']:
                if mp.get('below_detection_limit'):
                    continue
                rank = mp.get('peak_rank')
                if rank is not None:
                    lst = peak_mets.setdefault(rank, [])
                    if name not in lst:
                        lst.append(name)

                # For multiplets: also register sibling predicted-line ranks
                pred_lines = mp.get('predicted_lines') or []
                if len(pred_lines) > 1:
                    for line in pred_lines:
                        r = _find_rank(float(line.get('ppm', 0)))
                        if r is not None and r != rank:
                            lst2 = peak_mets.setdefault(r, [])
                            if name not in lst2:
                                lst2.append(name)

        self._peak_overlap = peak_mets

    def _build_child_candidates(self, tol=0.015):
        """For each DB-peak PPM, collect ALL metabolites that have a DB peak
        within *tol* ppm — regardless of experimental peak rank assignment.
        Result: _child_candidates = {round(dp_ppm,3): [(score, met, db_pk, mp_ref)]}
        where mp_ref is the matched_peak entry (or None) for that metabolite."""
        from collections import defaultdict
        by_ppm = defaultdict(dict)   # round(ppm,3) → {met_id: (score, met, db_pk, mp_ref)}
        for met in self.metabolites:
            met_score  = float(met.get('composite_score', 0) or 0)
            mid        = met.get('id', met.get('name', id(met)))
            mp_by_key  = {round(float(mp.get('db_ppm', 0)), 3): mp
                          for mp in met.get('matched_peaks', [])}
            for db_pk in met.get('db_peaks', []):
                dp_key = round(float(db_pk.get('ppm_center', db_pk.get('ppm', 0))), 3)
                mp_ref = mp_by_key.get(dp_key)
                entry  = (met_score, met, db_pk, mp_ref)
                if mid not in by_ppm[dp_key] or met_score > by_ppm[dp_key][mid][0]:
                    by_ppm[dp_key][mid] = entry
        all_ppms = sorted(by_ppm.keys())
        self._child_candidates = {}
        for dp_ppm in all_ppms:
            combined = {}
            for p2 in all_ppms:
                if abs(p2 - dp_ppm) > tol:
                    continue
                for mid, entry in by_ppm[p2].items():
                    if mid not in combined or entry[0] > combined[mid][0]:
                        combined[mid] = entry
            self._child_candidates[dp_ppm] = list(combined.values())

    def _apply_resolve(self, peak_rank, chosen_met_id, quality_map):
        """
        Called after the user assigns a peak in PeakResolveWindow.
        Updates annotation_quality and assignment, then repopulates.
        """
        # Apply quality_map to all matching mp dicts
        for met in self.metabolites:
            mid = met['id']
            if mid not in quality_map:
                continue
            for mp2 in met['matched_peaks']:
                if mp2.get('peak_rank') == peak_rank:
                    mp2['annotation_quality'] = quality_map[mid]

        # Update assignment to the chosen metabolite
        chosen = next((m for m in self.metabolites if m['id'] == chosen_met_id), None)
        if chosen:
            mp = next((m for m in chosen['matched_peaks']
                       if m.get('peak_rank') == peak_rank), None)
            if mp:
                self._peak_assignments[peak_rank] = (chosen, mp)

        self._sync_anchors_to_peaks()
        self._populate()

    def _apply_resolve_child(self, peak_rank, chosen_met_id, quality_map):
        """
        Called when the user resolves candidates via a CHILD (DB peak) row.
        - Marks the chosen metabolite's matched_peak as 'good' for this rank.
        - Marks others as 'bad' per quality_map.
        - Auto-promotes _peak_assignments to the Good metabolite.
        - Persists to ann._quality_marks.
        """
        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')

        for met in self.metabolites:
            mid = met['id']
            q   = quality_map.get(mid)
            if q is None:
                continue
            for mp in met['matched_peaks']:
                if mp.get('peak_rank') == peak_rank:
                    mp['annotation_quality'] = q
                    if ann is not None:
                        key = (mid, peak_rank)
                        if q:
                            ann._quality_marks[key] = q
                        else:
                            ann._quality_marks.pop(key, None)

        # Auto-promote the chosen metabolite as canonical assignment
        chosen = next((m for m in self.metabolites if m['id'] == chosen_met_id), None)
        if chosen:
            mp = next((m for m in chosen['matched_peaks']
                       if m.get('peak_rank') == peak_rank), None)
            if mp:
                self._peak_assignments[peak_rank] = (chosen, mp)

        self._populate()

    def _set_anchor_line(self, met, mp_ref, line_idx):
        """Toggle anchor on an individual predicted line of a multiplet.

        Stores `(met_id, peak_rank) → line_idx` in annotation_stage._anchor_lines
        and in mp_ref['anchor_line_idx'].  Calling again with the same line_idx
        removes the anchor (toggle behaviour).
        """
        ann          = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        anchor_store = getattr(ann, '_anchor_lines', None) if ann else None
        if anchor_store is None:
            return  # annotation stage not available

        met_id     = met['id']
        peak_rank  = mp_ref.get('peak_rank') if mp_ref else None
        anchor_key = (met_id, peak_rank)

        current = anchor_store.get(anchor_key)
        if current == line_idx:
            # Toggle off
            anchor_store.pop(anchor_key, None)
            if mp_ref is not None:
                mp_ref.pop('anchor_line_idx', None)
        else:
            # Set anchor
            anchor_store[anchor_key] = line_idx
            if mp_ref is not None:
                mp_ref['anchor_line_idx'] = line_idx

        self._populate()

    @staticmethod
    def _is_mult_m(mp):
        t = (mp.get('type', '') or mp.get('multiplicity', '') or '').strip().lower()
        return t in ('m', 'multiplet')

    def _measure_j_from_spectrum(self, x, y, plot_sticks, field_mhz):
        """
        Measure observed J (Hz) from local maxima in the experimental spectrum
        slice (x, y already baseline-corrected and sliced to the multiplet window).
        Returns median line separation in Hz, or None if < 2 lines detected.
        """
        if x is None or y is None or len(x) < 6:
            return None
        y_n = np.array(y, dtype=float)
        y_n = np.clip(y_n - y_n.min(), 0, None)
        if y_n.max() < 1e-12:
            return None
        y_n /= y_n.max()
        try:
            from scipy.signal import find_peaks as _fp
            min_dist = max(2, int(len(x) * 0.06))
            pks, _ = _fp(y_n, height=0.15, prominence=0.10, distance=min_dist)
        except Exception:
            return None
        if len(pks) < 2:
            return None
        ppms_obs = np.array(x)[pks]
        seps_hz  = [abs(ppms_obs[k+1] - ppms_obs[k]) * field_mhz
                    for k in range(len(ppms_obs) - 1)]
        return float(np.median(seps_hz)) if seps_hz else None

    def _get_field_mhz(self):
        """Return spectrometer frequency in MHz from the annotation stage."""
        try:
            ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            if ann_stage and hasattr(ann_stage, 'spectrometer_var'):
                return float(ann_stage.spectrometer_var.get())
        except Exception:
            pass
        return 600.0

    @staticmethod
    def _delta_j_hz(mp, field_mhz=600.0):
        """Return |J_obs - J_db| in Hz, or None if not applicable."""
        pl = mp.get('predicted_lines', [])
        if len(pl) < 2:
            return None
        ppms = sorted(p['ppm'] for p in pl)
        seps = [ppms[i+1] - ppms[i] for i in range(len(ppms)-1)]
        j_obs = sum(seps) / len(seps) * field_mhz
        j_raw = str(mp.get('j_coupling') or '').strip()
        if not j_raw:
            return None
        try:
            j_db = float(j_raw.split(',')[0].strip())
        except ValueError:
            return None
        return abs(j_obs - j_db)

    def _enrich_delta_j_from_spectrum(self):
        """
        For every non-singlet matched peak, measure J by snapping detected peaks
        to the predicted_lines positions (same method used in the fingerprint plot).
        This guarantees that delta_j_hz in mp_ref matches what is displayed in
        the Match Quality card.
        """
        if self.ppm is None:
            return

        field_mhz = self._get_field_mhz()

        import re as _re3
        for met in self.metabolites:
            for mp in met['matched_peaks']:
                mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                if mult in ('s', 'singlet', '', 'm', 'multiplet'):
                    continue

                pred = mp.get('predicted_lines') or []
                if len(pred) < 2:
                    continue

                db_j_raw = str(mp.get('j_coupling') or '').strip()
                _j_nums  = ([float(x) for x in _re3.findall(r'\d+\.?\d*', db_j_raw)
                             if float(x) > 0] if db_j_raw else [])
                db_j_hz  = _j_nums[0] if _j_nums else None

                # Snap tolerance: half the minimum predicted-line spacing.
                # If the peak was re-aligned after annotation, shift predicted
                # lines to the current peak ppm before snapping.
                ppms_pred = sorted(float(p['ppm']) for p in pred)
                spacings  = [abs(ppms_pred[k+1] - ppms_pred[k])
                             for k in range(len(ppms_pred) - 1)]
                snap_r    = max(min(spacings) * 0.45, 0.003) if spacings else 0.005

                # Alignment correction: only shift predicted lines if the detected
                # peak is OUTSIDE the multiplet's own footprint — i.e. the peak has
                # been manually re-anchored far from where annotation placed it.
                # For triplets/quartets peak_rank may refer to any line (including
                # outer lines), which would give |shift| ≈ J_ppm; the previous
                # 0.5×min_spacing threshold incorrectly shifted those cases.
                _pred_cen   = sum(ppms_pred) / len(ppms_pred)
                _half_span  = max(ppms_pred) - _pred_cen  # distance center→outer
                _cur_pk     = next((pk for pk in (self.peaks or [])
                                    if pk.get('rank') == mp.get('peak_rank')), None)
                if _cur_pk is not None:
                    _cur_ppm = float(_cur_pk.get('ppm', _pred_cen))
                    _shift   = _cur_ppm - _pred_cen
                    if abs(_shift) > _half_span + min(spacings) * 0.5:
                        ppms_pred = [p + _shift for p in ppms_pred]

                # Snap detected peaks (self.peaks) to each predicted line,
                # tracking which line INDEX was matched so we can compute J
                # correctly regardless of which lines were detected.
                used_ranks     = set()
                line_to_det    = {}   # predicted-line index → detected ppm
                for li, line_ppm in enumerate(ppms_pred):
                    best_pk, best_d = None, snap_r
                    for pk in (self.peaks or []):
                        if pk.get('rank') in used_ranks:
                            continue
                        d = abs(float(pk.get('ppm', 0)) - line_ppm)
                        if d <= best_d:
                            best_d, best_pk = d, pk
                    if best_pk is not None:
                        used_ranks.add(best_pk.get('rank'))
                        line_to_det[li] = float(best_pk['ppm'])

                snapped_ppm = list(line_to_det.values())
                if len(snapped_ppm) < 2:
                    continue

                # J = ppm_span / (line_index_span).  Handles all detection patterns:
                #   adjacent lines (idx 1,2):  span=J,  idx_span=1  → J   ✓
                #   outer lines   (idx 0,2):   span=2J, idx_span=2  → J   ✓
                #   all lines     (idx 0,1,2): span=2J, idx_span=2  → J   ✓
                _snapped_indices = list(line_to_det.keys())
                _idx_span = max(_snapped_indices) - min(_snapped_indices)
                _ppm_span = max(snapped_ppm) - min(snapped_ppm)
                j_obs = _ppm_span * field_mhz / _idx_span if _idx_span > 0 else _ppm_span * field_mhz
                mp['observed_j_hz'] = j_obs
                if db_j_hz is not None:
                    mp['delta_j_hz'] = float(abs(j_obs - db_j_hz))

    def _compute_scores(self):
        """
        Compute composite_score for each matched_peak using two formulae:

        Non-singlet (doublet, triplet, quartet, …):
            score = 0.60 * s_dj  +  0.20 * s_cosine  +  0.10 * s_ppm  +  0.10 * s_found

        Singlet / 'm' / unresolvable multiplet:
            score = 0.45 * s_cosine  +  0.35 * s_ppm  +  0.20 * s_found
            (s_dj has no contribution — singlets have no J-coupling to validate)

        Components:
          s_dj     : J quality for this DB multiplet (non-singlets only)
                       n_found < n_expected → proportional penalty × 0.5
                       n_found == n_expected, ΔJ < 0.5 Hz → 1.0
                       n_found == n_expected, ΔJ in [0.5, 3] → linear 1→0
                       n_found == n_expected, ΔJ > 3 → 0.0
                       cannot measure → 0.8 (partial credit)
          s_cosine : existing 'cosine' value stored on the metabolite
          s_ppm    : Gaussian decay on ppm_diff (σ = tolerance/2, clamped 0–1)
          s_found  : fraction of metabolite's DB peaks that have a matched peak

        Thresholds → annotation_quality (if not manually set):
          score > 0.8 → Good (green)
          0.5–0.8     → Undef (yellow)
          < 0.5       → Poor / Bad (red)
        """
        import re as _re_sc
        _n_map_sc = {'d': 2, 'doublet': 2, 't': 3, 'triplet': 3,
                     'q': 4, 'quartet': 4, 'dd': 4, 'dt': 6,
                     'td': 6, 'tt': 9, 'ddd': 8}

        try:
            ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            _tol = float(getattr(ann_stage, 'tolerance', 0.05) or 0.05)
        except Exception:
            _tol = 0.05
        _sigma_ppm = _tol / 2.0
        _field_sc  = self._get_field_mhz()

        for met in self.metabolites:
            # ── s_found: fraction of DB peaks with a matched detected peak ──
            # Only count peaks whose ΔJ is within tolerance (< 0.5 Hz).
            # Peaks with ΔJ ≥ 0.5 Hz (bad multiplet match) do NOT contribute.
            # BDL peaks (expected_snr < 3σ from anchor via nH ratio) are excluded
            # from BOTH numerator and denominator: their absence is expected and
            # their presence is irrelevant to the metabolite identity score.
            # Non-BDL unmatched peaks DO lower s_found — they should have appeared.
            db_peaks_all   = [p for p in met.get('db_peaks', [])
                              if (p.get('type') or '').lower().strip()
                              not in ('m', 'multiplet')
                              and not p.get('bdl', False)]
            n_db_total     = len(db_peaks_all) or 1

            def _mp_dj_ok(mp2):
                """Return True if this matched peak passes the ΔJ criterion."""
                mult2 = str(mp2.get('type') or mp2.get('multiplicity') or 's').lower().strip()
                if mult2 in ('s', 'singlet', '', 'm', 'multiplet'):
                    return True   # singlets have no J — always OK
                dj2 = mp2.get('delta_j_hz')
                return dj2 is None or float(dj2) < 0.5  # None = not measured yet → allow

            matched_db_set = {round(float(mp2.get('db_ppm', 0)), 3)
                              for mp2 in met['matched_peaks']
                              if _mp_dj_ok(mp2)}
            n_db_found     = sum(1 for p in db_peaks_all
                                 if round(float(p.get('ppm_center',
                                                      p.get('ppm', 0))), 3)
                                 in matched_db_set)
            s_found = float(n_db_found) / float(n_db_total)

            # ── s_reverse: nH-weighted DB coverage (reverse score, MS/MS style) ──
            # Fraction of total DB fingerprint intensity found in the experimental spectrum.
            # Penalises missing intense peaks (high nH) more than missing weak ones (nH=1).
            # Degrades gracefully to s_found when proton counts are unavailable in DB.
            _nh_total   = sum(max(int(p.get('protons') or 1), 1)
                              for p in db_peaks_all)
            _nh_matched = sum(max(int(p.get('protons') or 1), 1)
                              for p in db_peaks_all
                              if round(float(p.get('ppm_center',
                                                    p.get('ppm', 0))), 3)
                              in matched_db_set)
            s_reverse = float(_nh_matched) / float(_nh_total or 1)

            # ── cosine at metabolite level ───────────────────────────────────
            s_cosine = float(met.get('cosine', 0) or 0)

            for mp in met['matched_peaks']:
                # ── s_ppm ────────────────────────────────────────────────────
                ppm_diff = float(mp.get('ppm_diff', 0) or 0)
                s_ppm    = float(np.exp(-0.5 * (ppm_diff / _sigma_ppm) ** 2))

                # ── s_dj ─────────────────────────────────────────────────────
                mult_sc  = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                s_dj     = 1.0   # default for singlets
                _dj_conf = 1.0   # 0.0 = J unmeasurable → component excluded from score

                if mult_sc not in ('s', 'singlet', '', 'm', 'multiplet'):
                    # Count lines via predicted_lines (same source as tree).
                    # For singlets with is_sub_peak=True, predicted_lines contains
                    # ALL siblings — use DB multiplicity map instead.
                    pred_sc  = mp.get('predicted_lines', [])
                    n_exp_sc = _n_map_sc.get(mult_sc, len(pred_sc) if pred_sc else 2)

                    # If predicted_lines is empty or has fewer entries than expected
                    # (sub-peak annotation may store only 1 entry, or none at all),
                    # build synthetic snap targets from peak_ppm + DB J spacing —
                    # same fallback as the fingerprint display uses via plot_sticks.
                    if len(pred_sc) < n_exp_sc and _field_sc > 0:
                        _obs_ctr_fb = (float(mp.get('peak_ppm') or 0)
                                       or float(mp.get('db_ppm') or 0))
                        _j_raw_fb   = str(mp.get('j_coupling') or '').strip()
                        _j_fb_list  = [float(x) for x in
                                       _re_sc.findall(r'\d+\.?\d*', _j_raw_fb)
                                       if float(x) > 0]
                        _j_fb = _j_fb_list[0] if _j_fb_list else None
                        if _obs_ctr_fb and _j_fb:
                            _j_ppm_fb = _j_fb / _field_sc
                            _half_fb  = (n_exp_sc - 1) / 2.0
                            pred_sc = [{'ppm': _obs_ctr_fb + (_k - _half_fb) * _j_ppm_fb,
                                        'rel_intensity': 1.0}
                                       for _k in range(n_exp_sc)]

                    if pred_sc and n_exp_sc >= 2:
                        # Use the user-configured tolerance as snap radius so any
                        # peak within the annotation search window can be matched.
                        # The _used_sc set prevents the same peak from matching two lines.
                        _tol_sc = _tol

                        _used_sc  = set()
                        _found_sc = []
                        for _line_sc in pred_sc:
                            _lp = float(_line_sc.get('ppm', 0))
                            _best_pk_sc = None
                            _best_d_sc  = _tol_sc
                            for _pk_sc in self.peaks:
                                if _pk_sc.get('rank') in _used_sc:
                                    continue
                                _d_sc = abs(float(_pk_sc.get('ppm', 0)) - _lp)
                                if _d_sc <= _best_d_sc:
                                    _best_d_sc  = _d_sc
                                    _best_pk_sc = _pk_sc
                            if _best_pk_sc is not None:
                                _used_sc.add(_best_pk_sc.get('rank'))
                                _found_sc.append(float(_best_pk_sc['ppm']))

                        # Deduplicate by position: a parent peak and its sub-peak
                        # share the same physical signal but have different ranks.
                        # Use a tight threshold (1/4 linewidth ≈ 0.003 ppm) so that
                        # genuine multiplet lines (J ≈ 6 Hz → 0.01 ppm apart) are
                        # NOT collapsed — only true coincident parent/sub pairs are.
                        _dedup_thr = 0.003
                        _dedup_sc = []
                        for _fp in _found_sc:
                            if not any(abs(_fp - _ep) < _dedup_thr for _ep in _dedup_sc):
                                _dedup_sc.append(_fp)
                        _found_sc = _dedup_sc

                        # ── J-consistency validation ─────────────────────────
                        # Every consecutive spacing between found lines must be
                        # individually close to DB J (±1 Hz). A spurious nearby
                        # peak makes one spacing wildly wrong while others are OK.
                        # If any individual spacing fails, drop the peak that
                        # causes the worst spacing and accept n_found = n_exp - 1.
                        if len(_found_sc) >= 2 and _field_sc > 0:
                            _j_raw_val = str(mp.get('j_coupling') or '').strip()
                            _j_db_nums_v = [float(x) for x in
                                            _re_sc.findall(r'\d+\.?\d*', _j_raw_val)
                                            if float(x) > 0]
                            _j_db_v = _j_db_nums_v[0] if _j_db_nums_v else None
                            if _j_db_v is not None and len(_found_sc) == n_exp_sc:
                                _srt_v = sorted(_found_sc)
                                _seps_v = [abs(_srt_v[k+1] - _srt_v[k]) * _field_sc
                                           for k in range(len(_srt_v) - 1)]
                                # Check each individual spacing against DB J
                                _bad_seps = [abs(s - _j_db_v) > 2.0 for s in _seps_v]
                                if any(_bad_seps):
                                    # Find which peak to drop: try every candidate,
                                    # keep the removal that minimises max individual
                                    # spacing deviation (worst-case spacing).
                                    _best_drop, _best_worst = None, max(
                                        abs(s - _j_db_v) for s in _seps_v)
                                    for _drop_i in range(len(_found_sc)):
                                        _cand = [p for j, p in enumerate(_found_sc)
                                                 if j != _drop_i]
                                        if len(_cand) < 2:
                                            continue
                                        _s2 = sorted(_cand)
                                        _sep2 = [abs(_s2[k+1] - _s2[k]) * _field_sc
                                                 for k in range(len(_s2) - 1)]
                                        _worst2 = max(abs(s - _j_db_v) for s in _sep2)
                                        if _worst2 < _best_worst:
                                            _best_worst = _worst2
                                            _best_drop  = _drop_i
                                    if _best_drop is not None:
                                        _found_sc.pop(_best_drop)

                        n_found_sc = len(_found_sc)
                        mp['_n_found_lines'] = n_found_sc
                        mp['_n_exp_lines']   = n_exp_sc

                        # Sync predicted_lines with the snap-confirmed peaks so
                        # the fingerprint orange arrows show exactly the same
                        # peaks counted in _n_found_lines (single source of truth).
                        if _found_sc:
                            _orig_pl  = mp.get('predicted_lines') or []
                            _snap_pl  = []
                            _used_fp  = set()
                            for _l in sorted(_orig_pl, key=lambda l: l['ppm']):
                                _lp   = float(_l['ppm'])
                                _best = min(
                                    (fp for fp in _found_sc if fp not in _used_fp),
                                    key=lambda fp: abs(fp - _lp), default=None)
                                if _best is not None and abs(_best - _lp) <= _tol_sc:
                                    _used_fp.add(_best)
                                    _snap_pl.append({'ppm': _best,
                                                     'rel_intensity': _l.get('rel_intensity', 1.0)})
                            # Any confirmed peak that had no original predicted_lines
                            # counterpart (e.g. fallback to synthetic positions).
                            for _fp in sorted(_found_sc):
                                if _fp not in _used_fp:
                                    _snap_pl.append({'ppm': _fp, 'rel_intensity': 1.0})
                            if _snap_pl:
                                _max_ri_snap = max(l['rel_intensity'] for l in _snap_pl) or 1.0
                                for _sl in _snap_pl:
                                    _sl['rel_intensity'] /= _max_ri_snap
                                mp['predicted_lines'] = sorted(_snap_pl,
                                                                key=lambda l: l['ppm'])

                        if n_found_sc < n_exp_sc:
                            # Missing lines — ΔJ unmeasurable; clear stale value
                            # so Rule 0 / Rule 2 in _auto_mark_good don't use
                            # an annotation-search ΔJ that no longer applies.
                            mp['delta_j_hz']    = None
                            mp['observed_j_hz'] = None
                            s_dj = float(n_found_sc) / float(n_exp_sc) * 0.5
                        else:
                            # All lines found — always recompute ΔJ from snap
                            # positions (more current than annotation-search value).
                            dj_sc = None
                            if len(_found_sc) >= 2 and _field_sc > 0:
                                _sorted_sc = sorted(_found_sc)
                                _seps_sc   = [abs(_sorted_sc[k+1] - _sorted_sc[k]) * _field_sc
                                              for k in range(len(_sorted_sc) - 1)]
                                _j_obs_sc  = float(np.median(_seps_sc))
                                _j_raw_sc  = str(mp.get('j_coupling') or '').strip()
                                _j_db_nums = [float(x) for x in
                                              _re_sc.findall(r'\d+\.?\d*', _j_raw_sc)
                                              if float(x) > 0]
                                _j_db_sc   = _j_db_nums[0] if _j_db_nums else None
                                dj_sc = abs(_j_obs_sc - _j_db_sc) if _j_db_sc else None
                                # Write fresh measurements back so _auto_mark_good
                                # and _sync_anchors_to_peaks use consistent data.
                                mp['observed_j_hz'] = float(_j_obs_sc)
                                if dj_sc is not None:
                                    mp['delta_j_hz'] = float(dj_sc)
                                # Update ppm_diff from snap-observed centre vs DB ppm
                                _obs_c_sc = float(np.mean(_found_sc))
                                _db_p_sc  = float(mp.get('db_ppm') or 0)
                                if _db_p_sc != 0:
                                    mp['ppm_diff']  = float(abs(_obs_c_sc - _db_p_sc))
                                    mp['peak_ppm']  = float(_obs_c_sc)

                            if dj_sc is None:
                                s_dj     = self._params['dj_neutral']  # = 0.0
                                _dj_conf = 0.0  # all lines found but J not computable
                            elif dj_sc < self._params['dj_soft_lo_hz']:
                                s_dj = 1.0
                            elif dj_sc < self._params['dj_soft_hi_hz']:
                                _lo = self._params['dj_soft_lo_hz']
                                _hi = self._params['dj_soft_hi_hz']
                                s_dj = max(0.0, 1.0 - (dj_sc - _lo) / (_hi - _lo))
                            else:
                                s_dj = 0.0
                    else:
                        # No predicted_lines — fall back to delta_j_hz stored on mp
                        dj_sc = mp.get('delta_j_hz')
                        if dj_sc is None:
                            s_dj     = self._params['dj_neutral']  # = 0.0
                            _dj_conf = 0.0  # no predicted lines and no stored ΔJ
                        elif dj_sc < self._params['dj_soft_lo_hz']:
                            s_dj = 1.0
                        elif dj_sc < self._params['dj_soft_hi_hz']:
                            _lo = self._params['dj_soft_lo_hz']
                            _hi = self._params['dj_soft_hi_hz']
                            s_dj = max(0.0, 1.0 - (dj_sc - _lo) / (_hi - _lo))
                        else:
                            s_dj = 0.0

                # Singlets have no J-coupling to validate — use a formula
                # that only weighs real spectral evidence.  The old formula
                # gave s_dj=1.0 for free (60% of score), inflating singlet
                # scores to ~0.82 even when no peaks were found.
                _is_singlet_sc = mult_sc in ('s', 'singlet', '', 'm', 'multiplet',
                                             'br s', 'brs', 'br. s')
                # RRI (Relative Intensity Ratio): already computed by _fetch_fingerprints()
                # as met['rri'] = mean _rri_score across DB peaks.  Defaults to 0 when
                # DB has no stick intensity data (no penalty in that case).
                s_rri = float(met.get('rri') or 0)

                if _is_singlet_sc:
                    score = dynamic_composite(
                        scores={'cosine': s_cosine, 'ppm': s_ppm,
                                'found': s_found, 'rri': s_rri, 'rev': s_reverse},
                        confidences={'cosine': 1.0, 'ppm': 1.0,
                                     'found': 1.0, 'rri': 1.0, 'rev': 1.0},
                        base_weights=_BASE_WEIGHTS_ENRICHMENT_S,
                    )
                else:
                    # _dj_conf=0.0 when J unmeasurable → dj excluded, rest renormalised
                    score = dynamic_composite(
                        scores={'dj': s_dj, 'cosine': s_cosine, 'ppm': s_ppm,
                                'found': s_found, 'rri': s_rri, 'rev': s_reverse},
                        confidences={'dj': _dj_conf, 'cosine': 1.0, 'ppm': 1.0,
                                     'found': 1.0, 'rri': 1.0, 'rev': 1.0},
                        base_weights=_BASE_WEIGHTS_ENRICHMENT_NS,
                    )
                mp['_s_dj']      = s_dj
                mp['_s_ppm']     = s_ppm
                mp['_s_found']   = s_found
                mp['_s_cosine']  = s_cosine
                mp['_s_rri']     = s_rri
                mp['_s_reverse'] = s_reverse

                # ── Gate Rule 0b: nenhum peak encontrado → score = 0 ──────────
                # s_dj default = 0.8 (neutral) inflaciona score mesmo sem peaks.
                _gate_sc = 0.0 if s_found == 0.0 else 1.0

                # ── Bonus Rule 2: gold standard J ─────────────────────────────
                # All multiplet lines found + ΔJ < dj_gold_hz + Δppm < dppm_gold
                # → strongest primary evidence in NMR.
                _bonus_sc  = 0.0
                _dj_gold_sc = False
                _nf_gs    = mp.get('_n_found_lines')
                _ne_gs    = mp.get('_n_exp_lines')
                _all_lines_gs = (_nf_gs is not None and _ne_gs is not None
                                 and int(_nf_gs) >= int(_ne_gs) > 0)
                _dj_gs    = float(mp.get('delta_j_hz') or 999)
                _dppm_gs  = float(mp.get('ppm_diff') or 0)
                if (not _is_singlet_sc and _all_lines_gs
                        and mp.get('delta_j_hz') is not None
                        and _dj_gs < self._params['dj_gold_hz']
                        and _dppm_gs < self._params['dppm_gold']):
                    _bonus_sc  += self._params['bonus_gold_j']
                    _dj_gold_sc = True

                # Bonus MCR: spectral deconvolution confirmation (Rule MCR)
                if mp.get('mcr_confirmed') is True:
                    _bonus_sc += self._params['bonus_mcr']

                # Rule 0 (ΔJ gate for non-singlets) and RANSY/singlet_only bonuses
                # are applied in _enrich_scores() — they depend on data computed by
                # _propagate_multiplet_confirmation() and _propagate_j_anchor_quality().
                _enriched_base = float(np.clip(score * _gate_sc + _bonus_sc, 0.0, 1.0))

                # ── Gold J floor: guarantee 'Good' when J+ppm are within threshold ──
                # If gold standard J is met, the score must never fall below the 'Good'
                # threshold — J coupling is the definitive NMR identification evidence.
                if _dj_gold_sc and _gate_sc == 1.0:
                    _enriched_base = max(_enriched_base, self._params['gold_j_score_floor'])

                mp['composite_score'] = _enriched_base
                mp['_score_gate']     = _gate_sc
                mp['_score_bonus']    = _bonus_sc
                mp['_score_penalty']  = 0.0
                mp['_dj_gold']        = _dj_gold_sc

            # Propagate best score to metabolite level for sorting
            if met['matched_peaks']:
                met['composite_score'] = max(
                    float(mp2.get('composite_score', 0)) for mp2 in met['matched_peaks'])

    def _enrich_scores(self):
        """
        Segunda fase de scoring: aplica bonuses/gates que dependem de dados
        calculados após _compute_scores() (anchor_status, j_met_confirmed).

        Gates:
          Rule 0: não-singlet + ΔJ ≥ dj_bad_hz + não confirmado → score = 0

        Bonuses:
          Rule 6b: RANSY anchor limpo + posição alta confiança → +bonus_ransy_high
          Rule 6:  RANSY anchor limpo                          → +bonus_ransy

        Penalties:
          Rule 8: singlet-only sem confirmação J → −penalty_singlet_only
        """
        _sing_types = {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs', 'br. s'}

        for met in self.metabolites:
            _mps = met['matched_peaks']

            # Compute metabolite-level singlet-only unconfirmed flag
            _all_sing = all(
                str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                in _sing_types for mp in _mps)
            _has_j_anchor = any(mp.get('j_met_confirmed') for mp in _mps)
            _singlet_only_unconf = _all_sing and not _has_j_anchor

            for mp in _mps:
                score = float(mp.get('composite_score', 0) or 0)
                _gate = float(mp.get('_score_gate', 1.0))
                _bonus = float(mp.get('_score_bonus', 0.0))
                _penalty = 0.0

                mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                _is_sing = mult in _sing_types

                # Gate Rule 0: ΔJ ≥ dj_bad_hz (0.50 Hz) for non-singlets
                if not _is_sing and _gate != 0.0:
                    _dj_enr = mp.get('delta_j_hz')
                    if (_dj_enr is not None
                            and float(_dj_enr) >= self._params['dj_bad_hz']
                            and not mp.get('j_met_confirmed', False)):
                        _gate = 0.0
                        # Conservative zeroing: base + bonus were already merged into
                        # composite_score, so we zero the total rather than re-separating.
                        score = 0.0
                        mp['composite_score'] = 0.0
                        mp['_score_gate'] = 0.0
                        mp['_score_penalty'] = 0.0
                        continue

                # Bonuses RANSY (Rules 6 / 6b)
                _anchor = mp.get('anchor_status')
                _pos_conf = mp.get('anchor_pos_confidence')
                if _anchor == 'clean':
                    _bonus += (self._params['bonus_ransy_high']
                               if _pos_conf == 'high'
                               else self._params['bonus_ransy'])

                # Penalty: singlet-only without J confirmation (Rule 8)
                if _singlet_only_unconf:
                    _penalty += self._params['penalty_singlet_only']

                enriched = float(np.clip(score + _bonus - _penalty, 0.0, 1.0))
                mp['composite_score'] = enriched
                mp['_score_bonus']    = _bonus
                mp['_score_penalty']  = _penalty

        # Update metabolite-level score after enrichment
        for met in self.metabolites:
            if met['matched_peaks']:
                met['composite_score'] = max(
                    float(mp.get('composite_score', 0)) for mp in met['matched_peaks'])

    def _classify_confidence_classes(self):
        """
        Assign confidence_class ('A'–'X') to each metabolite based on
        the evidence available at this point in the scoring pipeline.
        Called after _adjust_scores_for_competition so all scores are final.
        """
        for met in self.metabolites:
            met['confidence_class'] = classify_annotation(met)

    def _adjust_scores_for_competition(self):
        """
        Third scoring phase: cross-metabolite adjustments.

        1. Overlap penalty: −penalty_overlap_per per competitor at the same rank
           (capped at penalty_overlap_max). Skipped for peaks with gate = 0.

        2. Gold J tiebreak (Rule 2): if two candidates both reach gold standard J
           at the same rank, only the best ΔJ keeps the bonus. The loser loses
           bonus_gold_j (drops below 'Good' for user inspection via J-Resolve).
        """
        rank_peaks = {}  # {peak_rank: [mp, ...]}
        for met in self.metabolites:
            for mp in met['matched_peaks']:
                r = mp.get('peak_rank')
                if r is not None:
                    rank_peaks.setdefault(r, []).append(mp)

        for rank, mps in rank_peaks.items():
            if len(mps) <= 1:
                continue

            # Overlap penalty
            penalty = min(self._params['penalty_overlap_per'] * (len(mps) - 1),
                          self._params['penalty_overlap_max'])
            for mp in mps:
                if float(mp.get('_score_gate', 1.0)) == 0.0:
                    continue  # already zeroed by gate — do not double-penalise
                new_s = max(0.0, float(mp.get('composite_score', 0)) - penalty)
                mp['composite_score']  = new_s
                mp['_score_penalty']   = float(mp.get('_score_penalty', 0)) + penalty

            # Tiebreak gold standard J (Rule 2)
            gold_mps = [mp for mp in mps
                        if float(mp.get('_score_bonus', 0)) >= self._params['bonus_gold_j']
                        and mp.get('delta_j_hz') is not None]
            if len(gold_mps) >= 2:
                best_dj = min(float(mp['delta_j_hz']) for mp in gold_mps)
                for mp in gold_mps:
                    if float(mp['delta_j_hz']) > best_dj:
                        # Remove gold J bonus → drops to yellow (user arbitrates via J-Resolve)
                        removed = self._params['bonus_gold_j']
                        mp['composite_score'] = max(0.0,
                                                    float(mp['composite_score']) - removed)
                        mp['_score_bonus']    = max(0.0,
                                                    float(mp.get('_score_bonus', 0)) - removed)

        # Update metabolite-level score after competition adjustments
        for met in self.metabolites:
            if met['matched_peaks']:
                met['composite_score'] = max(
                    float(mp.get('composite_score', 0)) for mp in met['matched_peaks'])

    def _propagate_multiplet_confirmation(self):
        """
        If a metabolite has ≥1 non-singlet matched_peak that plausibly matches
        the DB multiplet, mark ALL its matched_peaks with j_met_confirmed=True.

        Called AFTER _compute_scores() so that _s_dj is available.

        Two confirmation criteria (either suffices):
          A. delta_j_hz < 2.0 Hz  — J coupling measured and close to DB value
          B. ALL expected lines found (_n_found_lines >= _n_exp_lines >= 2)
             AND s_dj >= 0.5 (decent J quality or field unknown).
             Partial line matches (1 of 2 doublet lines) do NOT qualify —
             they cannot confirm the multiplet in a crowded spectrum where
             a single coincidental peak may lie near one expected line.

        Peaks that reach this threshold are kept as yellow candidates — the strict
        0.2 Hz quality gate in _auto_mark_good() Rule 2 still determines whether
        the peak is 'good' or merely a candidate for J-Resolve to arbitrate.
        """
        DJ_TOL = self._params['dj_plausible_hz']  # plausible J match; gold standard is dj_gold_hz
        # 'm'/'multiplet' are also treated as singlet-like: they have no
        # well-defined J-coupling in the DB and s_dj defaults to 1.0 in
        # _compute_scores(), which would falsely satisfy criterion B below.
        SINGLET_TYPES = {'s', 'singlet', '', 'br s', 'br. s', 'm', 'multiplet'}
        for met in self.metabolites:
            confirmed = False
            for mp in met['matched_peaks']:
                mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                if mult in SINGLET_TYPES:
                    continue
                dj   = mp.get('delta_j_hz')
                s_dj = float(mp.get('_s_dj', -1))
                # Criterion A: J measured and within 2.0 Hz
                if dj is not None and float(dj) < DJ_TOL:
                    confirmed = True
                    break
                # Criterion B: ALL expected lines found with sufficient J quality.
                # Requires snapping was attempted (_n_found_lines/_n_exp_lines set)
                # and that ALL lines were found (not just any partial snap), with
                # s_dj >= 0.5 (ΔJ plausible or unmeasurable with full line count).
                # This prevents a single coincidental line snap from propagating
                # j_met_confirmed to singlet peaks of the same metabolite.
                _nf_b = mp.get('_n_found_lines')
                _ne_b = mp.get('_n_exp_lines')
                if (_nf_b is not None and _ne_b is not None
                        and int(_nf_b) >= int(_ne_b) and int(_ne_b) >= 2
                        and s_dj >= 0.5):
                    confirmed = True
                    break
            if confirmed:
                for mp in met['matched_peaks']:
                    mp['j_met_confirmed'] = True

    def _filter_incompatible_candidates(self):
        """
        Mark candidates 'bad' only when there is a clear physical mismatch.
        Competition between candidates is left to J-Resolve.

        Rule 0 — ΔJ deviation:
            Non-singlet candidates whose measured coupling constant deviates ≥ 0.5 Hz
            from the DB value are marked bad.  Skipped for metabolites whose identity
            is confirmed by any peak with ΔJ < 2.0 Hz (j_met_confirmed=True) —
            those peaks survive as yellow candidates for J-Resolve to arbitrate.

        Line-count filtering has been intentionally removed: an unresolved doublet
        legitimately appears as a single detected peak, so matching predicted-line
        count against detected-peak count produces false negatives.

        Manual quality marks are never overwritten.
        """
        def _is_singlet(mp):
            mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
            return mult in ('s', 'singlet', '')

        # Group non-manually-marked entries by rank
        by_rank = {}
        for met in self.metabolites:
            for mp in met['matched_peaks']:
                if mp.get('annotation_quality') is not None:
                    continue
                rank = mp.get('peak_rank')
                if rank is None:
                    continue
                by_rank.setdefault(rank, []).append((met, mp))

        for rank, entries in by_rank.items():
            # ── Rule 0: ΔJ ≥ 0.5 Hz → Bad notation ──────────────────────────
            # Non-singlet candidates whose coupling deviates ≥ 0.5 Hz are bad.
            # Exception: if the metabolite is confirmed by another peak's J
            # (j_met_confirmed=True), the peak is kept as a yellow candidate for
            # J-Resolve — the bad ΔJ may simply reflect overlap, not wrong ID.
            for met, mp in entries:
                if _is_singlet(mp):
                    continue
                if mp.get('j_met_confirmed'):
                    continue
                dj = mp.get('delta_j_hz')
                if dj is not None and float(dj) >= 0.5:
                    mp['annotation_quality'] = 'bad'

        # ── Rule 1: J-anchor singlet exclusion ───────────────────────────────
        # If a J-confirmed metabolite has a singlet that is positionally and
        # quantitatively consistent with its companion multiplet ('clean' anchor
        # status, 'high'/'medium' position confidence), competing metabolites
        # that lack any J-confirmation at any peak are excluded at this rank.
        # This formalises "the doublet found its singlet — it wins the dispute."
        for rank, entries in by_rank.items():
            j_anchored = None
            for met, mp in entries:
                if not _is_singlet(mp):
                    continue
                if (mp.get('j_met_confirmed')
                        and mp.get('anchor_status') == 'clean'
                        and mp.get('anchor_pos_confidence') in ('high', 'medium')):
                    cur_conf = mp.get('anchor_pos_confidence')
                    prev_conf = j_anchored[1].get('anchor_pos_confidence') if j_anchored else None
                    if (j_anchored is None
                            or (cur_conf == 'high' and prev_conf != 'high')):
                        j_anchored = (met, mp)
            if j_anchored is None:
                continue
            j_met, j_mp = j_anchored
            for met, mp in entries:
                if met is j_met:
                    continue
                if mp.get('annotation_quality') is not None:
                    continue  # never overwrite explicit marks
                if not mp.get('j_met_confirmed'):
                    mp['annotation_quality'] = 'bad'
                    mp.setdefault('_status_reasons', []).append(
                        f'Excluded by J-anchor singlet of {j_met["name"]} '
                        f'(pos={j_mp.get("anchor_pos_confidence")}, '
                        f'ratio={j_mp.get("anchor_ratio")})')

    # ── RANSY-like J-anchor intensity propagation ─────────────────────────────

    def _peak_area(self, mp):
        """Integrate the observed mean spectrum over this matched-peak's ppm bounds.

        Uses (in priority order):
          1. db_start_ppm / db_end_ppm stored on the matched_peak
          2. Span of predicted_lines ± half a typical linewidth
          3. db_ppm ± 0.020 ppm (singlet fallback)

        Returns the absolute trapz integral (intensity × ppm).  This is the
        correct qNMR measure (area ∝ n_H × concentration) and works equally
        well for singlets and multiplets.
        """
        ppm_arr = self.ppm
        int_arr = self.mean_intensity
        if ppm_arr is None or int_arr is None or len(ppm_arr) < 4:
            return 0.0

        db_ppm = float(mp.get('db_ppm', mp.get('peak_ppm', 0)) or 0)

        lo = mp.get('db_start_ppm')
        hi = mp.get('db_end_ppm')
        # Only trust them if they actually bracket the centre
        if lo is not None and hi is not None:
            lo, hi = float(lo), float(hi)
            if abs(lo - hi) < 1e-4:          # degenerate bounds — discard
                lo = hi = None

        if lo is None or hi is None:
            pred = mp.get('predicted_lines') or []
            if pred:
                ppms = [float(l['ppm']) for l in pred if 'ppm' in l]
                if ppms:
                    lo = min(ppms) - 0.006
                    hi = max(ppms) + 0.006

        if lo is None or hi is None:
            lo, hi = db_ppm - 0.020, db_ppm + 0.020

        lo, hi = min(lo, hi), max(lo, hi)
        if hi - lo < 0.008:          # ensure a minimum integration window
            mid = (lo + hi) / 2.0
            lo, hi = mid - 0.012, mid + 0.012

        mask = (ppm_arr >= lo) & (ppm_arr <= hi)
        if mask.sum() < 2:
            return 0.0
        return float(abs(np.trapezoid(int_arr[mask], ppm_arr[mask])))

    def _propagate_j_anchor_quality(self):
        """RANSY-like anchor propagation (Anal. Chem. 2011, doi:10.1021/ac201625f).

        For each metabolite that has at least one J-confirmed non-singlet peak
        (ΔJ < 0.5 Hz), use that peak as a concentration reference.  The area of
        every other matched peak of the same metabolite should be proportional to
        its proton count (qNMR fundamental: area ∝ n_H × concentration).

        Three outcomes are written to mp['anchor_status']:
          'clean'   — area ratio 0.35–2.50 → consistent, no overlap
          'overlap' — ratio > 2.50 → excess signal, likely overlapping compound
          'bdl'     — obs_snr < 3 → below detection limit (SNR already on mp)
          'absent'  — ratio < 0.35 and SNR ≥ 3 → present but weaker than expected

        The thresholds are generous to absorb the ≤ 30 % intra-molecular T1
        differential saturation typical of standard metabolomics protocols
        (1D NOESY-presat, d1 = 4 s, 600 MHz).
        """
        CLEAN_LO = self._params['anchor_clean_lo']
        CLEAN_HI = self._params['anchor_clean_hi']

        if self.ppm is None or self.mean_intensity is None:
            return

        for met in self.metabolites:
            mps = met['matched_peaks']

            # ── Find best J-confirmed non-singlet anchor ──────────────────────
            anchor    = None
            best_sdj  = -1.0
            for mp in mps:
                mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                if mult in ('s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs'):
                    continue
                dj   = mp.get('delta_j_hz')
                s_dj = float(mp.get('_s_dj', 0) or 0)
                if dj is not None and float(dj) < self._params['dj_soft_lo_hz'] and s_dj > best_sdj:
                    anchor  = mp
                    best_sdj = s_dj

            if anchor is None:
                continue   # no J-anchor for this metabolite → skip

            # ── Per-proton area reference from anchor ─────────────────────────
            anchor_area = self._peak_area(anchor)
            try:
                anchor_protons = max(1, int(anchor.get('protons', 1) or 1))
            except (TypeError, ValueError):
                anchor_protons = 1

            if anchor_area <= 0:
                continue

            ref_per_H = anchor_area / anchor_protons

            # Mark the anchor multiplet itself as the reference
            anchor['anchor_status']     = 'self'
            anchor['anchor_ratio']      = 1.0
            anchor['anchor_expected_I'] = round(anchor_area, 6)

            # ── Classify every other matched peak ─────────────────────────────
            for mp in mps:
                if mp is anchor:
                    continue

                try:
                    p_protons = max(1, int(mp.get('protons', 1) or 1))
                except (TypeError, ValueError):
                    p_protons = 1

                expected_area = ref_per_H * p_protons
                obs_area      = self._peak_area(mp)
                ratio         = obs_area / expected_area if expected_area > 0 else 0.0
                obs_snr       = float(mp.get('obs_snr', 0) or 0)

                if obs_snr < self._params['anchor_snr_bdl']:
                    status = 'bdl'
                elif ratio > CLEAN_HI:
                    status = 'overlap'
                elif ratio >= CLEAN_LO:
                    status = 'clean'
                else:
                    status = 'absent'

                mp['anchor_status']     = status
                mp['anchor_ratio']      = round(ratio, 3)
                mp['anchor_expected_I'] = round(expected_area, 6)

    def _propagate_anchor_position(self):
        """
        For each metabolite with a J-confirmed anchor, predict every other peak's
        expected ppm using the DB Δδ (anchor_obs_ppm + (db_ppm_other - db_ppm_anchor)).
        Compares the prediction to the observed peak_ppm and assigns a confidence tier:
          'high'    — |Δδ| ≤ 0.02 ppm  (Golotvin high-confidence zone)
          'medium'  — |Δδ| ≤ 0.05 ppm
          'low'     — |Δδ| ≤ 0.10 ppm
          'mismatch'— |Δδ| > 0.10 ppm  (peak at unexpected relative position)
        """
        for met in self.metabolites:
            # Best J-anchor: non-singlet, delta_j_hz < 0.5 Hz, highest _s_dj
            anchor, best_sdj = None, -1.0
            for mp in met['matched_peaks']:
                mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                if mult in ('s', 'singlet', '', 'm', 'multiplet'):
                    continue
                dj = mp.get('delta_j_hz')
                if dj is None or dj >= self._params['dj_soft_lo_hz']:
                    continue
                sdj = float(mp.get('_s_dj') or 0.0)
                if sdj > best_sdj:
                    best_sdj, anchor = sdj, mp

            if anchor is None:
                continue

            anchor_obs = float(anchor.get('peak_ppm') or anchor.get('db_ppm') or 0)
            anchor_db  = float(anchor.get('db_ppm') or 0)

            for mp in met['matched_peaks']:
                if mp is anchor:
                    continue
                obs_ppm = mp.get('peak_ppm')
                if obs_ppm is None:
                    continue
                obs_ppm   = float(obs_ppm)
                pred_ppm  = anchor_obs + (float(mp.get('db_ppm') or 0) - anchor_db)
                delta_ppm = obs_ppm - pred_ppm

                abs_d = abs(delta_ppm)
                if abs_d <= self._params['dppm_anchor_high']:
                    conf = 'high'
                elif abs_d <= self._params['dppm_anchor_med']:
                    conf = 'medium'
                elif abs_d <= 0.10:
                    conf = 'low'
                else:
                    conf = 'mismatch'

                mp['anchor_pos_confidence'] = conf
                mp['anchor_pos_delta_ppm']  = round(delta_ppm, 5)
                mp['anchor_pos_pred_ppm']   = round(pred_ppm, 5)
                mp.setdefault('_status_reasons', []).append(
                    f'Pos-anchor {conf}: pred {pred_ppm:.4f} ppm | '
                    f'Δδ={delta_ppm:+.3f} ppm')

    def _auto_mark_good(self):
        """
        Deriva annotation_quality directamente do composite_score enriquecido.

        O composite_score já incorpora todos os gates e bonuses:
          - Gate Rule 0b (s_found=0) e Rule 0 (ΔJ ≥ 0.5 Hz não-singlet) → score = 0
          - Bonus Rule 2 (gold standard J), Rules 6/6b (RANSY), Rule MCR
          - Penalty Rule 8 (singlet-only sem J), overlap (Rules 3/4/5)
          aplicados em _compute_scores() + _enrich_scores() + _adjust_scores_for_competition().

        Thresholds:
          score ≥ good_score_thresh (0.80) → 'good'   (verde)
          score < bad_score_thresh  (0.50) → 'bad'    (vermelho)
          else                             → None      (amarelo)
          score ≥ 0.80 e singlet-only sem J sem fp_gate → 'poor' (laranja)

        BDL (below detection limit) é tratado como flag independente.
        Marcações manuais (Ctrl+1/2/3) nunca são sobrescritas.
        """
        _sing_types_am = {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs', 'br. s'}

        for met in self.metabolites:
            _mps_all = met['matched_peaks']

            # Singlet-only unconfirmed flag (para threshold 'poor')
            _all_sing_am = all(
                str(mp2.get('type') or mp2.get('multiplicity') or 's').lower().strip()
                in _sing_types_am for mp2 in _mps_all)
            _has_j_am = any(mp2.get('j_met_confirmed') for mp2 in _mps_all)
            _singlet_only_am = _all_sing_am and not _has_j_am

            _n_db_fps = len(met.get('db_peaks', []))
            _fp_cov   = float(met.get('coverage', 100.0))
            _fp_gate  = (_n_db_fps <= 1) or (_fp_cov >= self._params['fp_coverage_pct'])

            ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            _qm_am    = getattr(ann_stage, '_quality_marks', {}) if ann_stage else {}

            for mp in _mps_all:
                # Protect manual quality marks
                if (met.get('id'), mp.get('peak_rank')) in _qm_am:
                    continue
                if mp.get('annotation_quality') is not None:
                    continue

                score = float(mp.get('composite_score', 0) or 0)

                # BDL: independent flag (Rule 7), not a quality mark
                if mp.get('anchor_status') in ('bdl', 'absent'):
                    mp['below_detection_limit'] = True
                    continue

                # Threshold principal baseado no score enriquecido
                if score >= self._params['good_score_thresh']:        # ≥ 0.80
                    if _singlet_only_am and not _fp_gate:
                        mp['annotation_quality'] = 'poor'
                    else:
                        mp['annotation_quality'] = 'good'
                elif score < self._params['bad_score_thresh']:        # < 0.50
                    mp['annotation_quality'] = 'bad'
                else:                                                 # 0.50–0.80
                    if _singlet_only_am:
                        mp['annotation_quality'] = 'poor'
                    # else: None (yellow) — utilizador inspecciona

    def _require_multi_peak_confirmation(self):
        """
        Post-processing filter: metabolites with ≥2 DB peaks must have ≥2
        DISTINCT DB peak positions independently confirmed to remain 'good'.

        Rationale: peak-centric scoring in crowded biological NMR spectra leads
        to many coincidental matches.  A false positive metabolite with 5 DB
        peaks may accidentally win at 1-2 peak ranks, giving it 1-2 'good' marks
        at the SAME DB position (e.g., two detected peaks both near the same DB
        singlet).  Counting distinct confirmed DB positions is a much stronger
        discriminator: genuine analytes must have multiple *different* peaks
        confirmed, not multiple detected peaks all attributable to the same spot.

        Manual quality marks are always preserved.
        Single-peak DB entries are exempt (can't require 2 when only 1 exists).
        """
        _ann_s = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        _qm    = getattr(_ann_s, '_quality_marks', {}) if _ann_s else {}

        for met in self.metabolites:
            n_db = len(met.get('db_peaks', []))
            if n_db < 2:
                continue  # single-peak DB entry — nothing to require

            good_mps = [mp for mp in met['matched_peaks']
                        if mp.get('annotation_quality') == 'good']
            if not good_mps:
                continue  # nothing to downgrade

            # Count distinct DB peak positions among confirmed peaks.
            # Multiple detected peaks all matching the same DB ppm count as 1.
            confirmed_db_ppms = {round(float(mp.get('db_ppm', 0)), 3)
                                 for mp in good_mps}
            if len(confirmed_db_ppms) >= 2:
                continue  # ≥2 distinct DB positions confirmed — sufficient evidence

            # Only 1 (or 0) distinct DB position confirmed — downgrade unless manually marked
            for mp in good_mps:
                key = (met.get('id'), mp.get('peak_rank'))
                if key in _qm:
                    continue  # manual mark — never override
                mp['annotation_quality'] = None  # yellow

    def _nnls_spectral_validation(self):
        """
        Global NNLS spectral reconstruction filter.

        Builds a synthetic spectrum for every 'good' candidate metabolite
        (Lorentzian-broadened DB sticks), stacks them into matrix A, then
        solves the non-negative least-squares problem A @ x ≈ experimental
        spectrum.  Metabolites whose coefficient x[i] is < 3 % of the
        maximum contributor are downgraded to yellow — they do not meaningfully
        contribute to explaining the observed spectrum and are likely false
        positives.

        Conservative threshold (3 %) protects low-concentration analytes.
        Manual quality marks are always preserved.
        No-op when DB has no stick data or < 2 good candidates.
        """
        try:
            from scipy.optimize import nnls as _nnls
            from gui.utils.mcr_utils import simulate_lorentzian_profile as _slp
        except ImportError:
            return

        ppm_arr = getattr(self, 'ppm', None)
        int_arr = getattr(self, 'mean_intensity', None)
        if ppm_arr is None or int_arr is None or len(ppm_arr) < 10:
            return

        _ann_s = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        _qm    = getattr(_ann_s, '_quality_marks', {}) if _ann_s else {}

        # Collect metabolites with at least one 'good' peak
        good_mets = [met for met in self.metabolites
                     if any(mp.get('annotation_quality') == 'good'
                            for mp in met['matched_peaks'])]
        if len(good_mets) < 2:
            return  # NNLS trivial with 0-1 component

        # Estimate HWHM from detected peak widths
        _pw = [float(p.get('width_ppm', 0)) for p in self.peaks if p.get('width_ppm')]
        hwhm = (float(np.median(_pw)) / 2.0) if _pw else 0.003
        hwhm = max(0.001, min(hwhm, 0.020))

        b = np.asarray(int_arr, dtype=float).copy()
        b = np.clip(b, 0.0, None)

        profiles   = []
        valid_mets = []
        for met in good_mets:
            profile = np.zeros(len(ppm_arr), dtype=float)
            for dp in met.get('db_peaks', []):
                sticks = dp.get('sticks')
                if not sticks:
                    pc = dp.get('ppm_center', dp.get('ppm', 0))
                    sticks = [(pc, 1.0)] if pc else []
                if sticks:
                    profile += _slp(sticks, ppm_arr, hwhm=hwhm)
            norm = float(np.linalg.norm(profile))
            if norm < 1e-12:
                continue  # no spectral footprint — skip (no-op for this met)
            profiles.append(profile / norm)
            valid_mets.append(met)

        if len(valid_mets) < 2:
            return  # not enough profiles to constrain NNLS

        A = np.column_stack(profiles)
        try:
            x, _ = _nnls(A, b)
        except Exception:
            return

        x_max = float(x.max()) if x.max() > 0 else 1.0
        NNLS_MIN_FRACTION = self._params['nnls_min_frac']

        for i, met in enumerate(valid_mets):
            amp = float(x[i])
            x_norm = amp / x_max
            # Store amplitude on all matched peaks regardless of quality
            for mp in met['matched_peaks']:
                mp['nnls_amplitude'] = amp
            if x_norm >= NNLS_MIN_FRACTION:
                continue
            # Downgrade 'good' marks (unless manual)
            for mp in met['matched_peaks']:
                if mp.get('annotation_quality') != 'good':
                    continue
                key = (met.get('id'), mp.get('peak_rank'))
                if key in _qm:
                    continue
                mp['annotation_quality'] = None  # yellow

    def _build_peak_assignments(self):
        """
        Build peak_rank → (met, mp) assignment enforcing 1-peak-1-metabolite.

        For each detected peak rank, all (met, mp) pairs that matched it become
        candidates. The default assignment is the highest composite_score.
        If a match is already manually marked 'good', it takes priority.
        """
        all_cands = {}  # rank → [(score, met, mp)]
        for met in self.metabolites:
            for mp in met['matched_peaks']:
                rank = mp.get('peak_rank')
                if rank is None:
                    continue
                if mp.get('annotation_quality') == 'bad':
                    continue   # bad ΔJ/Δppm — not a real competitor
                score = float(
                    mp.get('composite_score',
                           mp.get('multiplet_score', 0.0)) or 0.0)
                all_cands.setdefault(rank, []).append((score, met, mp))

                # For multiplets: also register as candidate for every detected
                # peak that falls within the predicted_lines span.  This ensures
                # L-Alanine (doublet) appears in the Candidates window of both
                # its doublet lines, not only the primary matched rank.
                pred = mp.get('predicted_lines') or []
                if len(pred) >= 2 and self.peaks:
                    _ppms_pred = [float(l['ppm']) for l in pred]
                    _span_lo   = min(_ppms_pred)
                    _span_hi   = max(_ppms_pred)
                    _min_sep   = min(abs(_ppms_pred[k+1] - _ppms_pred[k])
                                    for k in range(len(_ppms_pred) - 1))
                    _margin    = max(_min_sep * 0.6, 0.003)
                    for pk in self.peaks:
                        other_rank = pk.get('rank')
                        if other_rank is None or other_rank == rank:
                            continue
                        pk_ppm = float(pk.get('ppm', 0))
                        if _span_lo - _margin <= pk_ppm <= _span_hi + _margin:
                            # Avoid duplicate entries
                            existing = all_cands.get(other_rank, [])
                            if not any(m2 is met and mp2 is mp
                                       for _, m2, mp2 in existing):
                                all_cands.setdefault(other_rank, []).append(
                                    (score, met, mp))

        self._peak_candidates = {}   # rank → [(score, met, mp)] sorted desc
        self._peak_assignments = {}  # rank → (met, mp)

        for rank, cands in all_cands.items():
            # Sort: auto-anchored (all lines found + ΔJ < 0.2 + Δppm < 0.05) first, then by composite score.
            # This ensures a J-confirmed match wins over a higher-scoring singlet
            # when both compete for the same peak.
            def _sort_key(x):
                _sc, _m, _mp = x
                dj = _mp.get('delta_j_hz')
                j_anchor_bonus = 0.20 if (dj is not None and dj < 0.5) else 0.0
                # J-anchored clean singlet: companion multiplet proves identity
                _mt = str(_mp.get('type') or _mp.get('multiplicity') or 's').lower().strip()
                singlet_j_bonus = 0.25 if (
                    _mt in ('s', 'singlet', '', 'm', 'multiplet')
                    and _mp.get('j_met_confirmed')
                    and _mp.get('anchor_status') == 'clean'
                    and _mp.get('anchor_pos_confidence') in ('high', 'medium')
                ) else 0.0
                return _sc + j_anchor_bonus + singlet_j_bonus
            cands.sort(key=_sort_key, reverse=True)
            self._peak_candidates[rank] = cands
            # Prefer a manually-assigned 'good' match, otherwise take best score
            manual = next(
                ((m, mp2) for _, m, mp2 in cands
                 if mp2.get('annotation_quality') == 'good'),
                None)
            self._peak_assignments[rank] = manual if manual else (
                cands[0][1], cands[0][2])

        # Propagate assignments → detected peaks so Peak List stays in sync
        self._sync_anchors_to_peaks()

    def _sync_anchors_to_peaks(self):
        """
        Propagate annotation assignments → detected peaks (self.peaks).

        For every peak rank that has a 'good' assignment:
        - Set  detected_peak['assigned_metabolite'] = met['name']
        - Set  detected_peak['is_anchor_line']      = True  (so Peak List shows ⚓)
        - Set  detected_peak['needs_j_resolve']     = True  if a valid competitor exists

        Anchor rules:
        - Manual anchor (user override) → always anchor
        - J-anchored singlet (j_met_confirmed + anchor_status clean/overlap) → always anchor;
          clean: no J-Resolve needed; overlap: needs_j_resolve=True + anchor_expected_I
          propagated to detected peak so J-Resolve can subtract the known contribution.
        - Multiplet: must be J-confirmed (ΔJ < 0.5 Hz) AND no valid competitor at rank
        - Plain singlet: qual == 'good' AND no valid competitor at rank
        - Valid competitor = another metabolite with annotation_quality == 'good' at
          the same rank that is also J-confirmed (if multiplet) or a good singlet.

        If the peak is J-confirmed but has a valid competitor → needs_j_resolve = True.
        """
        if not self.peaks:
            return

        # Build rank → detected peak lookup
        pk_by_rank = {p.get('rank'): p for p in self.peaks if p.get('rank') is not None}

        # Clear all annotation fields first, then re-apply from current assignments
        for pk in self.peaks:
            pk.pop('assigned_metabolite', None)
            pk.pop('is_anchor_line', None)
            pk.pop('needs_j_resolve', None)
            pk.pop('anchor_expected_I', None)
            pk.pop('anchor_metabolite', None)

        ann_stage    = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        anchor_store = getattr(ann_stage, '_anchor_lines', {}) if ann_stage else {}

        def _is_singlet_mp(m):
            t = str(m.get('type') or m.get('multiplicity') or 's').lower().strip()
            return t in ('s', 'singlet', '', 'm', 'multiplet')

        def _mp_j_ok(m):
            """True if this matched-peak qualifies as anchor:
            singlet (no J needed) OR multiplet where ALL predicted lines are
            found AND ΔJ < 0.2 Hz AND Δppm < 0.05 ppm."""
            if _is_singlet_mp(m):
                return True
            dj      = m.get('delta_j_hz')
            dppm    = m.get('ppm_diff')
            n_found = m.get('_n_found_lines')
            n_exp   = m.get('_n_exp_lines')
            all_found = (n_found is not None and n_exp is not None
                         and int(n_found) >= int(n_exp) and int(n_exp) > 0)
            dj_ok   = dj is not None and float(dj) < 0.2
            dppm_ok = dppm is None or float(dppm) < 0.05
            return all_found and dj_ok and dppm_ok

        for rank, (met, mp) in self._peak_assignments.items():
            qual = mp.get('annotation_quality')
            if qual == 'bad':
                continue
            pk = pk_by_rank.get(rank)
            if pk is None:
                continue
            pk['assigned_metabolite'] = met.get('name', '')

            has_manual_anchor = any(k[0] == met.get('id') and k[1] == rank
                                    for k in anchor_store)

            # Determine if this assignment is J-qualified
            this_j_ok = qual == 'good' and _mp_j_ok(mp)

            # Valid competitors: other 'good' candidates at the same rank
            # whose own J-quality also passes (singlet or ΔJ < 0.5 Hz)
            valid_competitors = [
                (m2, mp2)
                for _, m2, mp2 in self._peak_candidates.get(rank, [])
                if m2 is not met
                and mp2.get('annotation_quality') == 'good'
                and _mp_j_ok(mp2)
            ]

            # Helper: store anchor_line_idx for quantification
            def _store_anchor_line_idx():
                if ann_stage is None:
                    return
                met_id    = met.get('id')
                anchor_key = (met_id, rank)
                if anchor_key in anchor_store:
                    return
                pred   = mp.get('predicted_lines', [])
                pk_ppm = float(pk.get('ppm', 0))
                best_li = 0
                if pred:
                    dists   = [abs(float(pl.get('ppm', 0)) - pk_ppm) for pl in pred]
                    best_li = int(dists.index(min(dists)))
                if not hasattr(ann_stage, '_anchor_lines'):
                    ann_stage._anchor_lines = {}
                ann_stage._anchor_lines[anchor_key] = best_li
                anchor_store[anchor_key]             = best_li

            # ── J-anchored singlet: companion multiplet proves identity ──────
            _mult_t  = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
            _is_sing = _mult_t in ('s', 'singlet', '', 'm', 'multiplet')
            is_j_anchored_singlet = (
                _is_sing
                and qual == 'good'
                and mp.get('j_met_confirmed')
                and mp.get('anchor_status') in ('clean', 'overlap')
            )

            if has_manual_anchor:
                pk['is_anchor_line'] = True
            elif is_j_anchored_singlet:
                # Wins regardless of competitors: the companion multiplet confirmed
                # the metabolite identity, so this singlet is definitively assigned.
                pk['is_anchor_line'] = True
                _store_anchor_line_idx()
                # Pass expected intensity to detected peak for J-Resolve Phase 1
                if mp.get('anchor_expected_I') is not None:
                    pk['anchor_expected_I'] = mp['anchor_expected_I']
                    pk['anchor_metabolite'] = met.get('name', '')
                if mp.get('anchor_status') == 'overlap':
                    # Excess signal present → J-Resolve resolves the competitor
                    pk['needs_j_resolve'] = True
            elif this_j_ok:
                if not valid_competitors:
                    # J-confirmed and exclusive → auto-anchor
                    pk['is_anchor_line'] = True
                    _store_anchor_line_idx()
                else:
                    # J-confirmed but disputed → flag for J-Resolve
                    pk['needs_j_resolve'] = True

    def _classify(self, quality):
        """Mark the selected row as Good / Bad / Undefined (Ctrl+1/2/3)."""
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid in self._iid_to_row:
            self._classify_one(iid, quality)
        elif iid in self._iid_to_child:
            self._classify_child(iid, quality)
        elif iid in self._iid_to_subchild:
            self._classify_subchild(iid, quality)

    def _classify_one(self, iid, quality):
        """Classify a parent peak-assignment row (Good/Bad/None=Undefined)."""
        rank, met, mp, det_pk = self._iid_to_row[iid]
        mp['annotation_quality'] = quality

        # Persist mark so it survives window close/reopen
        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        if ann is not None:
            if not hasattr(ann, '_quality_marks') or ann._quality_marks is None:
                ann._quality_marks = {}
            key = (met['id'], mp.get('peak_rank'))
            if quality:
                ann._quality_marks[key] = quality
            else:
                ann._quality_marks.pop(key, None)
            # Mark project dirty
            pm = getattr(self.app, 'project_manager', None)
            if pm is not None:
                pm.mark_dirty()

        # If marked Good, promote this metabolite as the assignment for this peak
        if quality == 'good':
            self._peak_assignments[rank] = (met, mp)
        self._sync_anchors_to_peaks()

        if quality == 'good':
            qual_mark = '✓ Good'
        elif quality == 'bad':
            qual_mark = '✗ Bad'
        else:
            qual_mark = '—'
        score = float(mp.get('composite_score', mp.get('multiplet_score', 0.0)) or 0.0)
        tag = 'met_high' if score >= 0.8 else 'met_mid' if score >= 0.5 else 'met_low'
        if quality == 'good':
            tags = (tag, 'qual_good')
        elif quality == 'bad':
            tags = (tag, 'qual_bad')
        else:
            tags = (tag,)
        cur = list(self.tree.item(iid, 'values'))
        cur[5] = qual_mark
        self.tree.item(iid, values=cur, tags=tags)

    def _classify_child(self, ciid, quality):
        """Classify a child DB-peak row (Good/Bad/None=Undefined)."""
        met, dp, mp_ref = self._iid_to_child[ciid]
        dp_ppm  = float(dp.get('ppm_center', dp.get('ppm', 0)))
        key     = (met['id'], round(dp_ppm, 3))

        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        if ann is not None:
            if not hasattr(ann, '_child_quality_marks') or ann._child_quality_marks is None:
                ann._child_quality_marks = {}
            if quality:
                ann._child_quality_marks[key] = quality
            else:
                ann._child_quality_marks.pop(key, None)
            pm = getattr(self.app, 'project_manager', None)
            if pm is not None:
                pm.mark_dirty()

        if quality == 'good':
            tags = ('peak_match', 'qual_good')
        elif quality == 'bad':
            tags = ('peak_match', 'qual_bad')
        else:
            # Revert to automatic colouring based on match/overlap
            mp_ref  = next(
                (mp2 for mp2 in met['matched_peaks']
                 if round(float(mp2.get('db_ppm', 0)), 3) == round(dp_ppm, 3)),
                None)
            is_matched  = mp_ref is not None
            child_pk    = mp_ref.get('peak_rank') if mp_ref else None
            n_cands     = len(self._peak_candidates.get(child_pk, [])) if child_pk else 0
            if is_matched and n_cands > 1:
                tags = ('peak_match', 'qual_mid')
            elif is_matched:
                tags = ('peak_match', 'qual_good')
            else:
                tags = ('peak_match', 'qual_bad')

        if quality == 'good':
            match_mark = '✓ Good'
        elif quality == 'bad':
            match_mark = '✗ Bad'
        else:
            match_mark = '✓' if tags[1] != 'qual_bad' else '✗'

        cur = list(self.tree.item(ciid, 'values'))
        cur[5] = match_mark
        self.tree.item(ciid, values=cur, tags=tags)

    def _classify_subchild(self, sciid, quality):
        """Classify an individual predicted-line row (sub-child) as Good/Bad/Undef.

        When Good: annotates the linked detected peak in self.peaks with the
        metabolite name (sets 'assigned_metabolite').  Bad/Undef removes it.
        """
        met, dp, mp_ref, line_dict, line_idx, nearest_pk = self._iid_to_subchild[sciid]
        child_rank = mp_ref.get('peak_rank') if mp_ref else None

        # Persist to annotation stage
        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        if ann is not None:
            if not hasattr(ann, '_subline_quality_marks') or ann._subline_quality_marks is None:
                ann._subline_quality_marks = {}
            key = (met['id'], child_rank, line_idx)
            if quality:
                ann._subline_quality_marks[key] = quality
            else:
                ann._subline_quality_marks.pop(key, None)
            pm = getattr(self.app, 'project_manager', None)
            if pm is not None:
                pm.mark_dirty()

        # Propagate to the linked detected peak in self.peaks
        # Also check if this line is the anchor
        ann2         = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        anchor_store2 = getattr(ann2, '_anchor_lines', {}) if ann2 else {}
        anchor_key2  = (met['id'], child_rank)
        this_is_anchor = (anchor_store2.get(anchor_key2) == line_idx)

        if nearest_pk is not None:
            if quality == 'good':
                nearest_pk['assigned_metabolite'] = met.get('name', '')
                if this_is_anchor:
                    nearest_pk['is_anchor_line'] = True
            else:
                if nearest_pk.get('assigned_metabolite') == met.get('name', ''):
                    nearest_pk.pop('assigned_metabolite', None)
                if this_is_anchor:
                    nearest_pk.pop('is_anchor_line', None)

        # Update tree row appearance
        score_col = '⚓' if this_is_anchor else ''
        if quality == 'good':
            qual_lbl  = '✓ Good'
            new_tags = ('sub_peak', 'sub_peak_matched', 'qual_good')
        elif quality == 'bad':
            qual_lbl  = '✗ Bad'
            new_tags = ('sub_peak',
                        'sub_peak_matched' if nearest_pk is not None else 'sub_peak_unmatched',
                        'qual_bad')
        else:
            qual_lbl  = f'Peak #{nearest_pk["rank"]}' if nearest_pk else ''
            new_tags = ('sub_peak',
                        'sub_peak_matched' if nearest_pk is not None else 'sub_peak_unmatched')
        if this_is_anchor:
            new_tags = new_tags + ('anchor_line',)

        cur = list(self.tree.item(sciid, 'values'))
        cur[4] = score_col
        cur[5] = qual_lbl
        self.tree.item(sciid, values=cur, tags=new_tags)

    def _on_tree_close(self, event):
        """When a parent row is collapsed, refresh its score in the treeview."""
        iid = self.tree.focus()
        if iid not in self._iid_to_row:
            return
        rank, met, mp, det_pk = self._iid_to_row[iid]
        # Recalculate composite score from only Good child peaks
        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        cqm = getattr(ann, '_child_quality_marks', {}) if ann else {}
        good_children = [
            dp for dp in met.get('db_peaks', [])
            if cqm.get((met['id'], round(float(dp.get('ppm_center', dp.get('ppm', 0))), 3)))
               == 'good'
        ]
        if good_children:
            n_good    = len(good_children)
            n_db      = max(len(met.get('db_peaks', [])), 1)
            coverage  = n_good / n_db
            new_score = float(mp.get('composite_score',
                                     mp.get('multiplet_score', 0.0)) or 0.0)
            # Blend original score with coverage of Good peaks
            new_score = new_score * 0.6 + coverage * 0.4
            mp['composite_score'] = new_score
            tag = 'met_high' if new_score >= 0.7 else 'met_mid' if new_score >= 0.4 else 'met_low'
            qual = mp.get('annotation_quality')
            tags = (tag, 'qual_good') if qual == 'good' else (tag, 'qual_bad') if qual == 'bad' else (tag,)
            cur = list(self.tree.item(iid, 'values'))
            cur[4] = f'{new_score:.3f}'
            self.tree.item(iid, values=cur, tags=tags)

    # ── Table ─────────────────────────────────────────────────────────────────

    def _sort_by(self, key):
        """Toggle sort on the peak assignments list and repopulate."""
        cur = getattr(self, '_sort_key', None)
        rev = getattr(self, '_sort_rev', True)
        self._sort_rev = not rev if cur == key else True
        self._sort_key = key
        self._populate()

    def _on_right_click(self, event):
        """Right-click on a peak-assignment row → candidates / Peak List / Deconv."""
        iid = self.tree.identify_row(event.y)
        if not iid:
            return

        # ── Parent row (peak assignment) ──────────────────────────────────────
        if iid in self._iid_to_row:
            rank, met, mp, det_pk = self._iid_to_row[iid]
            peak_ppm = float(mp.get('peak_ppm', 0))
            cands_raw = self._peak_candidates.get(rank, [])
            candidates = [(m, mp2) for _, m, mp2 in cands_raw]

            def _open_peak_list():
                app = getattr(self, 'app', None)
                if not app: return
                stage = getattr(app, 'stages', {}).get('peak_detection')
                if not stage: return
                try:
                    from gui.widgets.peak_list_window import PeakListWindow
                    win = PeakListWindow(
                        app.root, stage.peaks,
                        app.ppm, app.all_intensities, app=app)
                    if hasattr(win, 'scroll_to_rank'):
                        win.scroll_to_rank(rank)
                except Exception as e:
                    print(f'Peak List nav: {e}')

            def _open_deconv():
                app = getattr(self, 'app', None)
                if not app: return
                stage = getattr(app, 'stages', {}).get('peak_detection')
                if not stage: return
                try:
                    peak = next(
                        (p for p in stage.peaks if p.get('rank') == rank), None)
                    if peak is None:
                        peak = min(stage.peaks,
                                   key=lambda p: abs(p['ppm'] - peak_ppm),
                                   default=None)
                    if peak is None:
                        return
                    from gui.notation.deconvolution import (
                        DeconvController, DeconvWindow)
                    if (stage._deconv_controller is None
                            or stage._deconv_controller.processor
                            is not stage.processor):
                        stage._deconv_controller = DeconvController(
                            stage.processor)
                        if stage._overlaps:
                            stage._deconv_controller.set_regions(
                                stage._overlaps)
                    ctrl = stage._deconv_controller
                    region_idx, child_idx = ctrl.add_peak(
                        peak, target_ppm=peak['ppm'])
                    app.switch_stage('peak_detection')
                    stage.btn_open_deconv.config(state='normal')
                    dw = DeconvWindow(app.root, ctrl, stage)
                    dw.select_region(region_idx, child_idx)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f'Deconv nav: {e}')

            def _open_resolve():
                from gui.widgets.peak_resolve_window import PeakResolveWindow
                PeakResolveWindow(
                    self.win,
                    candidates=candidates,
                    peak_rank=rank,
                    peak_ppm=peak_ppm,
                    ppm=self.ppm,
                    mean_intensity=self.mean_intensity,
                    on_assign=lambda chosen_id, qmap: self._apply_resolve(
                        rank, chosen_id, qmap),
                    peaks_list=self.peaks,
                    eligible_overrides=getattr(self.app, 'eligible_overrides', {}))

            def _anchor_singlet():
                """Force-assign the current singlet candidate and mark all
                other candidates at this rank as Bad."""
                db_type = str(mp.get('type') or mp.get('multiplicity') or '').lower().strip()
                if db_type not in ('s', 'singlet', ''):
                    import tkinter.messagebox as _mb
                    _mb.showwarning(
                        'Not a singlet',
                        f'{met["name"]} at this ppm is not a singlet (DB type: {db_type}).\n'
                        'Anchor is only meaningful for singlet candidates.',
                        parent=self.win)
                    return
                quality_map = {}
                for _, m2, mp2 in cands_raw:
                    if m2['id'] == met['id']:
                        mp['annotation_quality'] = 'good'
                        quality_map[m2['id']] = 'good'
                    else:
                        mp2['annotation_quality'] = 'bad'
                        quality_map[m2['id']] = 'bad'
                self._apply_resolve(rank, met['id'], quality_map)

            # Determine if current assignment is a singlet (anchor eligible)
            cur_db_type = str(mp.get('type') or mp.get('multiplicity') or '').lower().strip()
            is_singlet_match = cur_db_type in ('s', 'singlet', '')
            obs_is_singlet   = str(mp.get('observed_mult', '')).lower() in ('s', 'singlet', '')
            has_multiplet_competitors = any(
                str(mp2.get('type') or mp2.get('multiplicity') or '').lower().strip()
                not in ('s', 'singlet', '')
                for _, m2, mp2 in cands_raw if m2['id'] != met['id']
            )

            menu = tk.Menu(self.win, tearoff=0)
            menu.add_command(
                label=f'📋 Open Peak #{rank} in Peak List',
                command=_open_peak_list)
            menu.add_command(
                label=f'🧅 Send Peak #{rank} to Deconvolution Tool',
                command=_open_deconv)
            menu.add_separator()
            menu.add_command(
                label=f'🔀 View Candidates  ({len(candidates)} metabolites)',
                command=_open_resolve)
            if is_singlet_match and (obs_is_singlet or has_multiplet_competitors):
                menu.add_command(
                    label=f'⚓ Anchor as Singlet — {met["name"]} wins (others → Bad)',
                    command=_anchor_singlet)
            menu.add_separator()
            menu.add_command(
                label=f'PPM {peak_ppm:.4f}  ·  {met["name"]}',
                state='disabled')
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        # ── Child row (DB peak) ───────────────────────────────────────────────
        elif iid in self._iid_to_child:
            met, dp, mp_ref = self._iid_to_child[iid]
            dp_ppm     = float(dp.get('ppm_center', dp.get('ppm', 0)))
            child_rank = mp_ref.get('peak_rank') if mp_ref else None
            dp_key_rc  = round(dp_ppm, 3)
            cands_raw  = self._child_candidates.get(dp_key_rc, [])
            candidates = [(m2, mp_r) for _, m2, _db_pk, mp_r in cands_raw if mp_r is not None]

            def _open_resolve_child(_candidates=candidates, _rank=child_rank,
                                     _ppm=dp_ppm):
                if not _candidates:
                    return
                from gui.widgets.peak_resolve_window import PeakResolveWindow
                PeakResolveWindow(
                    self.win,
                    candidates=_candidates,
                    peak_rank=_rank,
                    peak_ppm=_ppm,
                    ppm=self.ppm,
                    mean_intensity=self.mean_intensity,
                    on_assign=lambda chosen_id, qmap: self._apply_resolve_child(
                        _rank, chosen_id, qmap),
                    peaks_list=self.peaks,
                    eligible_overrides=getattr(self.app, 'eligible_overrides', {}))

            def _open_mcr_child(_met=met, _mp_ref=mp_ref, _dp=dp,
                                 _rank=child_rank, _ppm=dp_ppm):
                app       = getattr(self, 'app', None)
                pd_stage  = (getattr(app, 'stages', {}).get('peak_detection')
                             if app else None)
                ann_stage = (getattr(app, 'stages', {}).get('annotation')
                             if app else None)
                peak = None
                if pd_stage and _rank is not None:
                    peak = next((p for p in pd_stage.peaks
                                 if p.get('rank') == _rank), None)
                if peak is None:
                    peak = {'rank': _rank or 0, 'ppm': _ppm,
                            'start_ppm': _ppm - 0.015,
                            'end_ppm':   _ppm + 0.015}
                # mp_ref can be None when no matched peak was recorded yet —
                # build a minimal stub so AnnotationMCRWindow can proceed
                mp = _mp_ref if _mp_ref is not None else {
                    'db_ppm':    _ppm,
                    'peak_ppm':  _ppm,
                    'peak_rank': _rank,
                    'type':      _dp.get('type', 's'),
                    'protons':   _dp.get('protons', 1),
                    'composite_score': 0.0,
                }
                from gui.widgets.annotation_mcr_window import AnnotationMCRWindow
                AnnotationMCRWindow(
                    self.win,
                    [(_met, mp, peak)],
                    app,
                    ann_stage=ann_stage,
                    ann_results=self)

            menu = tk.Menu(self.win, tearoff=0)
            menu.add_command(
                label=f'DB peak @ {dp_ppm:.4f} ppm  [{dp.get("type","?")}]  —  {met["name"]}',
                state='disabled')
            if cands_raw:
                menu.add_separator()
                menu.add_command(
                    label=f'\u21c6 View Candidates  ({len(candidates)} metabolites)',
                    command=_open_resolve_child)
            menu.add_separator()
            menu.add_command(
                label='\U0001f52c MCR Deconvolution',
                command=_open_mcr_child)
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        # ── Sub-child row (individual predicted line) ─────────────────────────
        elif iid in self._iid_to_subchild:
            met, dp, mp_ref, line_dict, line_idx, nearest_pk = self._iid_to_subchild[iid]
            line_ppm   = float(line_dict.get('ppm', 0))
            rel_int    = float(line_dict.get('rel_intensity', 0))
            child_rank = mp_ref.get('peak_rank') if mp_ref else None

            # Current anchor state
            ann          = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            anchor_store = getattr(ann, '_anchor_lines', None) if ann else None
            anchor_key   = (met['id'], child_rank)
            current_anc  = anchor_store.get(anchor_key) if anchor_store is not None else None
            is_anchor    = (current_anc == line_idx)

            # Candidates for this predicted line: all metabolites with a DB peak near line_ppm
            line_key_rc = round(line_ppm, 3)
            cands_raw   = self._child_candidates.get(line_key_rc, [])
            candidates  = [(m2, mp_r) for _, m2, _db_pk, mp_r in cands_raw if mp_r is not None]

            def _toggle_anchor(_met=met, _mp_ref=mp_ref, _line_idx=line_idx,
                                _line_ppm=line_ppm):
                self._set_anchor_line(_met, _mp_ref, _line_idx)

            def _open_resolve_line(_candidates=candidates, _rank=child_rank,
                                    _ppm=line_ppm):
                if not _candidates:
                    return
                from gui.widgets.peak_resolve_window import PeakResolveWindow
                PeakResolveWindow(
                    self.win,
                    candidates=_candidates,
                    peak_rank=_rank,
                    peak_ppm=_ppm,
                    ppm=self.ppm,
                    mean_intensity=self.mean_intensity,
                    on_assign=lambda chosen_id, qmap: self._apply_resolve_child(
                        _rank, chosen_id, qmap),
                    peaks_list=self.peaks,
                    eligible_overrides=getattr(self.app, 'eligible_overrides', {}))

            menu = tk.Menu(self.win, tearoff=0)
            menu.add_command(
                label=f'Linha @ {line_ppm:.4f} ppm  —  int. rel. {rel_int:.2f}  —  {met["name"]}',
                state='disabled')
            menu.add_separator()
            if is_anchor:
                menu.add_command(label='⚓ Remove Anchor', command=_toggle_anchor)
            else:
                menu.add_command(label='⚓ Set as Anchor', command=_toggle_anchor)
            if cands_raw:
                menu.add_separator()
                menu.add_command(
                    label=f'\u21c6 Ver Candidatos  ({len(candidates)} metabolitos)',
                    command=_open_resolve_line)
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

    def _populate(self):
        self.tree.delete(*self.tree.get_children(''))

        # Name filter
        name_q = getattr(self, '_flt_name', None)
        q = name_q.get().lower() if name_q else ''

        # Quality filter
        show_good   = getattr(self, '_ann_filt_good',   None)
        show_bad    = getattr(self, '_ann_filt_bad',    None)
        show_undef  = getattr(self, '_ann_filt_undef',  None)
        show_bdl    = getattr(self, '_ann_filt_bdl',    None)
        show_good   = show_good.get()   if show_good   else True
        show_bad    = show_bad.get()    if show_bad    else True
        show_undef  = show_undef.get()  if show_undef  else True
        show_bdl    = show_bdl.get()    if show_bdl    else True
        show_mult_m = False  # 'm' multiplets permanently excluded

        def _is_mult_m(mp):
            t = (mp.get('type', '') or mp.get('multiplicity', '') or '').strip().lower()
            return t in ('m', 'multiplet')

        def _qual_visible(mp):
            # Filter out 'm' (multiplet without J) unless explicitly shown
            if _is_mult_m(mp) and not show_mult_m:
                return False
            if mp.get('below_detection_limit'):
                return show_bdl
            qual = mp.get('annotation_quality')
            if qual == 'good':  return show_good
            if qual == 'bad':   return show_bad
            if qual == 'poor':  return show_undef  # 'poor' follows the undef filter
            return show_undef   # None / undefined

        def _met_visible(met):
            """Hide metabolite entirely if ALL its matched peaks are 'm' and filter is off."""
            if show_mult_m:
                return True
            non_m = [mp2 for mp2 in met['matched_peaks'] if not _is_mult_m(mp2)]
            return len(non_m) > 0

        # ── Collect visible metabolites (one row per metabolite) ─────────────
        # A metabolite is visible if at least one of its matched peaks passes
        # the quality filter and the name filter.
        ann_stage    = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        anchor_store = getattr(ann_stage, '_anchor_lines', {}) if ann_stage else {}
        _user_tol    = float(getattr(ann_stage, 'tolerance', 0.05) or 0.05)

        visible_mets = []
        for met in self.metabolites:
            if q and q not in met['name'].lower():
                continue
            if not _met_visible(met):
                continue
            # At least one matched peak must pass the quality filter
            visible_mps = [mp2 for mp2 in met['matched_peaks']
                           if _qual_visible(mp2)]
            if not visible_mps:
                continue
            visible_mets.append(met)

        # Sort metabolites
        sk  = getattr(self, '_sort_key', 'ppm')
        rev = getattr(self, '_sort_rev', True)
        if sk == 'name':
            visible_mets.sort(key=lambda m: m['name'].lower(), reverse=not rev)
        elif sk == 'score':
            visible_mets.sort(
                key=lambda m: float(m.get('composite_score', 0) or 0),
                reverse=rev)
        else:  # ppm / default
            visible_mets.sort(
                key=lambda m: float(
                    min((float(mp2.get('peak_ppm', 0)) for mp2 in m['matched_peaks']),
                        default=0)),
                reverse=rev)

        # Update summary label
        n_total_peaks = len(self.peaks) if self.peaks else len(self._peak_assignments)
        n_matched     = len(self._peak_assignments)
        lbl = getattr(self, '_summary_lbl', None)
        if lbl:
            lbl.config(
                text=f"{n_matched}/{n_total_peaks} peaks annotated  ·  "
                     f"{n_matched} matched in DB  ·  "
                     f"{len(visible_mets)} metabolites visible")

        self._iid_to_row      = {}
        self._iid_to_child    = {}
        self._iid_to_subchild = {}

        for ri, met in enumerate(visible_mets):
            met_id = met.get('id')

            # ── Aggregate quality / score across all matched peaks ─────────────
            score = float(met.get('composite_score', 0) or 0)

            has_anchor = any(k[0] == met_id for k in anchor_store)

            # Overall quality: best quality among matched peaks
            quals = [mp2.get('annotation_quality') for mp2 in met['matched_peaks']]
            is_bdl = all(mp2.get('below_detection_limit', False)
                         for mp2 in met['matched_peaks'])
            if has_anchor and not is_bdl:
                qual = 'good'
                qual_mark = '⚓ Good'
            elif 'good' in quals:
                qual = 'good'
                qual_mark = '✓ Good'
            elif all(q2 == 'bad' for q2 in quals if q2 is not None):
                qual = 'bad'
                qual_mark = '✗ Bad'
            elif is_bdl:
                qual = None
                qual_mark = '⬇ BDL'
            elif all(q2 in ('poor', 'bad', None) for q2 in quals) and 'poor' in quals:
                qual = 'poor'
                qual_mark = '~ Poor'
            else:
                qual = None
                qual_mark = '—'

            # Overlap: any matched peak with competitors?
            n_ovlp = sum(1 for mp2 in met['matched_peaks']
                         if len(self._peak_candidates.get(
                             mp2.get('peak_rank'), [])) > 1)
            ovlp_str = f"⚠ {n_ovlp}" if n_ovlp > 0 else '—'

            # Representative ppm: ppm of best-scoring matched peak
            best_mp = max(met['matched_peaks'],
                          key=lambda mp2: float(mp2.get('composite_score', 0) or 0),
                          default=None)
            peak_ppm  = float(best_mp.get('peak_ppm', 0)) if best_mp else 0.0
            best_rank = best_mp.get('peak_rank') if best_mp else None
            det_pk    = next((p for p in self.peaks if p.get('rank') == best_rank),
                             None) if best_rank else None
            if best_mp and ('db_start_ppm' in best_mp or 'db_end_ppm' in best_mp):
                b_lo = float(best_mp.get('db_start_ppm', peak_ppm))
                b_hi = float(best_mp.get('db_end_ppm', peak_ppm))
                start_ppm = max(b_lo, b_hi)
                end_ppm = min(b_lo, b_hi)
            elif det_pk:
                start_ppm = float(det_pk.get('start_ppm', peak_ppm + 0.01))
                end_ppm   = float(det_pk.get('end_ppm',   peak_ppm - 0.01))
            else:
                start_ppm = peak_ppm + 0.01
                end_ppm   = peak_ppm - 0.01

            tag  = ('met_high' if score >= 0.8
                    else 'met_mid' if score >= 0.5 else 'met_low')
            if is_bdl:
                tags = (tag, 'qual_bdl')
            else:
                tags = (tag, 'qual_good') if qual == 'good' else \
                       (tag, 'qual_bad')  if qual == 'bad'  else \
                       (tag, 'qual_poor') if qual == 'poor' else (tag, 'qual_mid')

            _conf_cls  = met.get('confidence_class', '')
            _name_disp = f"[{_conf_cls}] {met['name']}" if _conf_cls else met['name']

            piid = f"met{ri}"
            self._iid_to_row[piid] = (best_rank, met, best_mp, det_pk)

            try:
                self.tree.insert(
                    '', tk.END, iid=piid,
                    text='',
                    values=(
                        _name_disp,
                        f"{peak_ppm:.4f}",
                        f"{start_ppm:.4f}",
                        f"{end_ppm:.4f}",
                        f"{score:.3f}",
                        qual_mark,
                        ovlp_str),
                    tags=tags,
                    open=False)
            except Exception as e:
                print(f"Tree insert error for {met['name']}: {e}")
                continue

            # Build lookups for child rows
            matched_mp_by_dbppm = {round(float(mp2.get('db_ppm', 0)), 3): mp2
                                   for mp2 in met['matched_peaks']}
            matched_db_ppms = set(matched_mp_by_dbppm.keys())
            # ri is metabolite index; child rows use ri for unique iid prefix
            rank = best_rank  # used only for cqm key below

            # Load persisted child quality marks for this metabolite
            ann     = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            cqm     = getattr(ann, '_child_quality_marks', {}) if ann else {}

            # Children: DB peaks of the assigned metabolite.
            # Excluded: (1) unresolved multiplets ('m') whose J is unknown;
            #           (2) BDL peaks that were NOT detected — expected SNR < 3σ
            #               and no experimental match, so showing ✗ would be
            #               misleading (absence is expected, not informative).
            # BDL peaks that WERE detected are kept and shown gray (⬇).
            _dp_list = [
                p for p in met['db_peaks']
                if (p.get('type') or '').lower().strip() not in ('m', 'multiplet')
                and (not p.get('bdl', False)
                     or round(float(p.get('ppm_center', p.get('ppm', 0))), 3)
                        in matched_db_ppms)
            ]
            for ci, dp in enumerate(sorted(
                    _dp_list,
                    key=lambda p: p.get('ppm_center', p.get('ppm', 0)),
                    reverse=True)):
                ciid    = f"dp{ri}_{ci}"
                dp_ppm  = float(dp.get('ppm_center', dp.get('ppm', 0)))
                dp_type = dp.get('type', '?') or '?'
                try:
                    protons_str = f"  {int(dp.get('protons') or 0)}H"
                except (ValueError, TypeError):
                    protons_str = f"  {dp.get('protons','')}" if dp.get('protons') else ''
                dp_start = float(dp.get('ppm_start', dp_ppm))
                dp_end   = float(dp.get('ppm_end',   dp_ppm))

                # ── Match status ───────────────────────────────────────────────
                dp_key     = round(dp_ppm, 3)
                is_matched = dp_key in matched_db_ppms
                match_mark = '✓' if is_matched else '✗'

                # ── Overlap: how many other metabolites have a DB peak near this PPM ─
                mp_ref_child  = matched_mp_by_dbppm.get(dp_key)
                child_pk_rank = mp_ref_child.get('peak_rank') if mp_ref_child else None
                n_db_cands    = len(self._child_candidates.get(dp_key, []))
                child_ovlp    = f"⚠ {n_db_cands}" if n_db_cands > 1 else '⚓'

                # ── Count detected lines for this DB multiplet ────────────────
                # Use the SAME source as the sub-child rows: predicted_lines from
                # mp_ref_child with the same _sc_tol tolerance. This guarantees the
                # count matches exactly what is shown in the tree.
                _dp_mult       = str(dp.get('type') or dp.get('multiplicity') or 's').lower().strip()
                _pred_lines_c  = mp_ref_child.get('predicted_lines', []) if mp_ref_child else []
                _is_singlet_dp = _dp_mult in ('s', 'singlet', '')
                # For singlets: always 1 expected line regardless of predicted_lines
                # (predicted_lines may contain sub-peak siblings when is_sub_peak=True)
                _n_map_dp = {'d':2,'doublet':2,'t':3,'triplet':3,'q':4,'quartet':4,
                             'dd':4,'dt':6,'td':6,'tt':9,'ddd':8}
                if _is_singlet_dp:
                    _n_exp_c = 1
                else:
                    _n_exp_c = len(_pred_lines_c) if _pred_lines_c else _n_map_dp.get(_dp_mult, 2)
                _n_found_c     = 1   # default for singlets / no predicted_lines
                _lines_complete = True

                if is_matched and _pred_lines_c and not _is_singlet_dp:
                    # Prefer values already written by the fingerprint snap (single
                    # source of truth). Fall back to local snap only when not yet set.
                    _cached_nf = mp_ref_child.get('_n_found_lines') if mp_ref_child else None
                    _cached_ne = mp_ref_child.get('_n_exp_lines')   if mp_ref_child else None
                    if (_cached_nf is not None and _cached_ne is not None
                            and int(_cached_ne) == _n_exp_c):
                        _n_found_c     = int(_cached_nf)
                        _lines_complete = (_n_found_c >= _n_exp_c)
                    else:
                        try:
                            _used_ranks_c = set()
                            _n_found_c    = 0
                            for _line in _pred_lines_c:
                                _lp = float(_line.get('ppm', 0))
                                _best_pk = None
                                _best_d  = _user_tol
                                for _pk in self.peaks:
                                    if _pk.get('rank') in _used_ranks_c:
                                        continue
                                    _d = abs(float(_pk.get('ppm', 0)) - _lp)
                                    if _d <= _best_d:
                                        _best_d  = _d
                                        _best_pk = _pk
                                if _best_pk is not None:
                                    _used_ranks_c.add(_best_pk.get('rank'))
                                    _n_found_c += 1
                            _lines_complete = (_n_found_c >= _n_exp_c)
                        except Exception:
                            _lines_complete = True  # can't determine — don't penalise

                # ── Apply persisted child quality mark (overrides auto-colour) ─
                child_saved_quality = cqm.get((met['id'], dp_key))
                # Check ΔJ criterion for auto-colour
                _child_dj = mp_ref_child.get('delta_j_hz') if mp_ref_child else None
                _child_mult = str(mp_ref_child.get('type') or mp_ref_child.get('multiplicity') or 's').lower().strip() if mp_ref_child else 's'
                _child_is_singlet = _child_mult in ('s', 'singlet', '', 'm', 'multiplet')
                _child_dj_bad = (not _child_is_singlet
                                 and _child_dj is not None
                                 and float(_child_dj) >= 0.5)
                if child_saved_quality == 'good':
                    child_tags = ('peak_match', 'qual_good')
                    match_mark = '✓ Good'
                elif child_saved_quality == 'bad':
                    child_tags = ('peak_match', 'qual_bad')
                    match_mark = '✗ Bad'
                else:
                    # Auto-colour: red if not matched, lines missing, or ΔJ ≥ 0.5 Hz
                    if not is_matched:
                        child_tags = ('peak_match', 'qual_bad')
                        match_mark = '✗'
                    elif not _lines_complete:
                        child_tags = ('peak_match', 'qual_bad')
                        match_mark = f'✗ {_n_found_c}/{_n_exp_c}'
                    elif _child_dj_bad:
                        child_tags = ('peak_match', 'qual_bad')
                        match_mark = f'✗ ΔJ {_child_dj:.2f}Hz'
                    elif (mp_ref_child is not None
                          and mp_ref_child.get('annotation_quality') == 'poor'):
                        # Singlet-only unconfirmed metabolite — no J-anchor proof
                        child_tags = ('peak_match', 'qual_poor')
                        match_mark = '~ Poor'
                    elif n_db_cands > 1:
                        child_tags = ('peak_match', 'qual_mid')
                        match_mark = f'✓ {_n_found_c}/{_n_exp_c}' if _n_exp_c > 1 else '✓'
                    else:
                        child_tags = ('peak_match', 'qual_good')
                        match_mark = f'✓ {_n_found_c}/{_n_exp_c}' if _n_exp_c > 1 else '✓'

                # ── J-anchor badge for singlets confirmed by companion multiplet ─
                if (_child_is_singlet
                        and child_saved_quality != 'bad'
                        and mp_ref_child is not None
                        and mp_ref_child.get('j_met_confirmed')
                        and mp_ref_child.get('anchor_status') == 'clean'):
                    child_tags = child_tags + ('singlet_anchored',)
                    _a_conf    = mp_ref_child.get('anchor_pos_confidence', '')
                    _conf_badge = {'high': '●', 'medium': '◑', 'low': '○'}.get(_a_conf, '')
                    match_mark  = f'⚓{_conf_badge} anchored'
                elif (_child_is_singlet
                        and child_saved_quality != 'bad'
                        and mp_ref_child is not None
                        and mp_ref_child.get('anchor_status') == 'overlap'):
                    if not match_mark.startswith('✗'):
                        match_mark = match_mark + '  ⚓⚠'

                # ── BDL override: detected but below expected SNR threshold ──────
                # Show in gray to indicate the signal is marginal. Only applied
                # when the user has not explicitly saved a quality mark.
                if dp.get('bdl') and is_matched and not child_saved_quality:
                    child_tags = ('peak_match', 'qual_bdl')
                    if not match_mark.startswith('✗'):
                        match_mark = '⬇ ' + match_mark

                self._iid_to_child[ciid] = (met, dp, mp_ref_child)
                try:
                    self.tree.insert(
                        piid, tk.END, iid=ciid,
                        text=f"  ↳ {dp_ppm:.4f}",
                        values=(
                            f"{dp_type}{protons_str}",
                            f"{dp_ppm:.4f}",
                            f"{dp_start:.4f}",
                            f"{dp_end:.4f}",
                            '',
                            match_mark,
                            child_ovlp),
                        tags=child_tags)
                except Exception as e:
                    print(f"  child insert error: {e}")

                # ── Sub-child rows: individual predicted lines (for multiplets) ──
                pred_lines = mp_ref_child.get('predicted_lines', []) if mp_ref_child else []
                if len(pred_lines) >= 2:
                    ann           = (getattr(self.app, 'stages', {}) or {}).get('annotation')
                    anchor_store  = getattr(ann, '_anchor_lines', {}) if ann else {}
                    anchor_key    = (met['id'], child_pk_rank)
                    anchor_idx    = anchor_store.get(anchor_key)  # int or None

                    # ── Auto-anchor: if this DB peak is unique in the database
                    # (no other metabolite has a peak here), auto-anchor the
                    # most-intense predicted line (highest rel_intensity).
                    if (anchor_idx is None
                            and n_db_cands <= 1
                            and ann is not None
                            and anchor_store is not None):
                        best_li   = max(range(len(pred_lines)),
                                        key=lambda k: float(pred_lines[k].get('rel_intensity', 0)))
                        anchor_store[anchor_key]  = best_li
                        anchor_idx                = best_li
                        if mp_ref_child is not None:
                            mp_ref_child['anchor_line_idx'] = best_li

                    # Tolerance for matching a predicted line to a detected peak.
                    # Use the user-configured PPM tolerance so any peak within the
                    # annotation search window can be snapped. The _used_sc_ranks
                    # set prevents the same detected peak from matching two lines.
                    _sc_tol = _user_tol

                    # Sort lines by ppm descending to match NMR convention
                    sorted_lines = sorted(enumerate(pred_lines),
                                          key=lambda t: t[1].get('ppm', 0),
                                          reverse=True)
                    slqm = getattr(ann, '_subline_quality_marks', {}) if ann else {}
                    _used_sc_ranks = set()   # prevent same peak from matching two lines
                    for li_sorted, (orig_li, line_dict) in enumerate(sorted_lines):
                        sciid    = f"sc{ri}_{ci}_{orig_li}"
                        line_ppm = float(line_dict.get('ppm', 0))
                        rel_int  = float(line_dict.get('rel_intensity', 0))
                        is_anchor = (anchor_idx is not None and orig_li == anchor_idx)
                        anchor_lbl = '  ⚓' if is_anchor else ''

                        # ── Look up nearest peak in the peak list ─────────────
                        nearest_pk = None
                        best_dist  = _sc_tol
                        for _pk in self.peaks:
                            if _pk.get('rank') in _used_sc_ranks:
                                continue
                            _d = abs(float(_pk.get('ppm', 0)) - line_ppm)
                            if _d <= best_dist:
                                best_dist  = _d
                                nearest_pk = _pk
                        if nearest_pk is not None:
                            _used_sc_ranks.add(nearest_pk.get('rank'))

                        # ── Apply persisted sub-line quality mark ──────────────
                        saved_sl_q = slqm.get((met['id'], child_pk_rank, orig_li))
                        if saved_sl_q == 'good' and nearest_pk is not None:
                            nearest_pk['assigned_metabolite'] = met.get('name', '')
                        # Set is_anchor_line on the detect peak for peak list display
                        if is_anchor and nearest_pk is not None:
                            nearest_pk['is_anchor_line'] = True

                        # Only show lines that have a matching detected peak
                        if nearest_pk is None:
                            continue

                        pk_rank   = nearest_pk.get('rank', '?')
                        pk_ppm    = float(nearest_pk.get('ppm', line_ppm))
                        pk_start  = float(nearest_pk.get('start_ppm', pk_ppm))
                        pk_end    = float(nearest_pk.get('end_ppm',   pk_ppm))
                        if saved_sl_q == 'good':
                            sci_color = ('sub_peak', 'sub_peak_matched', 'qual_good')
                            qual_lbl  = '✓ Good'
                        elif saved_sl_q == 'bad':
                            sci_color = ('sub_peak', 'sub_peak_matched', 'qual_bad')
                            qual_lbl  = '✗ Bad'
                        else:
                            sci_color = ('sub_peak', 'sub_peak_matched')
                            qual_lbl  = f'Peak #{pk_rank}'
                        if is_anchor:
                            sci_color = sci_color + ('anchor_line',)

                        score_col   = '⚓' if is_anchor else ''
                        self._iid_to_subchild[sciid] = (met, dp, mp_ref_child,
                                                         line_dict, orig_li,
                                                         nearest_pk)
                        try:
                            self.tree.insert(
                                ciid, tk.END, iid=sciid,
                                text=f"    ↳ {pk_ppm:.4f}{anchor_lbl}",
                                values=(
                                    f"  rel. {rel_int:.2f}",
                                    f'{pk_ppm:.4f}',
                                    f'{pk_start:.4f}',
                                    f'{pk_end:.4f}',
                                    score_col,
                                    qual_lbl,
                                    ''),
                                tags=sci_color)
                        except Exception as _se:
                            print(f"  sub-child insert error: {_se}")


        # Select first row
        children = self.tree.get_children('')
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self._on_select()

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]

        if iid in self._iid_to_row:
            # Parent row: full metabolite fingerprint
            rank, met, mp, det_pk = self._iid_to_row[iid]
            self._current_met = met
            self._plot_fingerprint(met)

        elif iid in self._iid_to_child:
            # Child row: zoom into this specific DB peak inside the fingerprint panel
            met, dp, mp_ref = self._iid_to_child[iid]
            self._current_met = met
            self._plot_db_peak_zoom(met, dp)

        elif iid in self._iid_to_subchild:
            # Sub-child row: show the zoom view for the parent DB peak
            met, dp, mp_ref, line_dict, line_idx, nearest_pk = self._iid_to_subchild[iid]
            self._current_met = met
            self._plot_db_peak_zoom(met, dp)

    # ── DB-peak zoom ──────────────────────────────────────────────────────────

    def _plot_db_peak_zoom(self, met, dp):
        """
        Zoom the fingerprint comparison panel to a single DB peak (multiplet).

        Shows:
          • Grey fill + dark line  = experimental spectrum in the region
          • Green sticks + shape   = DB peak that was MATCHED in the experiment
          • Red   sticks + shape   = DB peak NOT FOUND in the experiment
          • Orange Δppm arrow      = shift between DB centre and observed centre
          • Yellow/green highlight = region of the multiplet
        """
        # Track current context for right-click drag region selection.
        # Backup predicted_lines ONCE when switching to a new db_peak,
        # so reset always returns to the state at first display.
        if dp is not self._current_dp:
            # Save original for the peak we're leaving (if any)
            self._orig_pred_lines = None
            self._show_reset_btn(False)
        self._current_dp     = dp
        self._current_met_dp = met

        self.ax.clear()

        dp_ppm  = float(dp.get('ppm_center', dp.get('ppm', 0)))
        dp_type = (dp.get('type', 's') or 's').strip()

        # ── Is this DB peak matched? ───────────────────────────────────────────
        matched_db_ppms = {round(float(mp.get('db_ppm', 0)), 3)
                           for mp in met['matched_peaks']}
        is_matched = round(dp_ppm, 3) in matched_db_ppms
        mp_ref = next((mp for mp in met['matched_peaks']
                       if round(float(mp.get('db_ppm', 0)), 3) == round(dp_ppm, 3)),
                      None)
        self._current_mp_ref = mp_ref
        # Take the backup exactly once — on the first call for this dp.
        # _orig_pred_lines is cleared when dp changes (above).
        if self._orig_pred_lines is None and mp_ref is not None:
            self._orig_pred_lines = list(mp_ref.get('predicted_lines') or [])

        # ── Overlap: how many metabolites claim the same exp. peak ─────────────
        child_pk_rank = mp_ref.get('peak_rank') if mp_ref else None
        overlap_cands = self._peak_candidates.get(child_pk_rank, []) \
                        if child_pk_rank is not None else []
        n_overlap  = max(0, len(overlap_cands) - 1)   # exclude self
        ovlp_names = [m.get('name', '?')
                      for _, m, _ in overlap_cands
                      if m.get('name') != met['name']]

        # ── Centre on experimental position if matched, else on DB ppm ──────────
        if mp_ref:
            centre = float(mp_ref.get('peak_ppm', dp_ppm))
            offset = centre - dp_ppm
        else:
            centre = dp_ppm
            offset = 0.0

        # ── Build DB sticks from multiplicity pattern (ground truth from DB) ────
        # Uses successive-coupling convolution so compound multiplets (dd, dt, ddd…)
        # are built correctly: each coupling letter splits each existing stick.
        import re as _re_db
        _db_mult  = str(dp.get('type') or dp.get('multiplicity') or 's').lower().strip()
        is_singlet_dp = _db_mult in ('s', 'singlet', '')
        _db_j_str = str(dp.get('j_coupling') or '').strip()
        _db_j_hz_list = [float(x) for x in _re_db.findall(r'\d+\.?\d*', _db_j_str)
                         if float(x) > 0]
        _fmhz = self._get_field_mhz() or 600.0

        def _split_sticks(sticks, j_ppm, pattern):
            """Convolve every stick with a coupling pattern."""
            n = len(pattern)
            half_span = j_ppm * (n - 1) / 2.0
            out = []
            for ppm_s, int_s in sticks:
                for k, rel in enumerate(pattern):
                    out.append((ppm_s - half_span + k * j_ppm, int_s * rel))
            return out

        # Map each letter in mult string to its coupling pattern
        _letter_pattern = {'d': [1, 1], 't': [1, 2, 1], 'q': [1, 3, 3, 1],
                           'p': [1, 4, 6, 4, 1]}
        # Decompose mult string into successive couplings
        # e.g. 'dd'→['d','d'], 'dt'→['d','t'], 'ddd'→['d','d','d'], 's'→[]
        _simple = {'s': [], 'singlet': [], 'd': ['d'], 'doublet': ['d'],
                   't': ['t'], 'triplet': ['t'], 'q': ['q'], 'quartet': ['q']}
        if _db_mult in _simple:
            _coupling_letters = _simple[_db_mult]
        else:
            # Parse letter by letter for compound multiplets
            _coupling_letters = [c for c in _db_mult if c in _letter_pattern]

        _db_sticks_raw = [(centre, 1.0)]
        for _li, _letter in enumerate(_coupling_letters):
            if _li < len(_db_j_hz_list):
                _j_hz = _db_j_hz_list[_li]
            elif _db_j_hz_list:
                # Only one J known: use it for first coupling, use a smaller
                # typical secondary coupling (J1/4, min 1.5 Hz) for the rest
                # so compound multiplet lines don't collapse onto each other.
                _j_hz = max(_db_j_hz_list[0] / 4.0, 1.5)
            else:
                # No J at all: use descending typical couplings (8, 2, 1 Hz…)
                _typical = [8.0, 2.0, 1.0]
                _j_hz = _typical[_li] if _li < len(_typical) else 1.0
            _j_ppm = _j_hz / _fmhz
            _pat = _letter_pattern[_letter]
            _db_sticks_raw = _split_sticks(_db_sticks_raw, _j_ppm, _pat)

        # Normalise intensities to max=1
        if _db_sticks_raw:
            _max_int = max(i for _, i in _db_sticks_raw)
            if _max_int > 0:
                _db_sticks_raw = [(p, i / _max_int) for p, i in _db_sticks_raw]
        if not _db_sticks_raw:
            _db_sticks_raw = [(centre, 1.0)]

        # Prefer predicted_lines (already shifted to observed ppm during annotation
        # search) over raw DB sticks centred on peak_ppm (which may be one outer
        # line of the multiplet, not the multiplet centre).
        if mp_ref and not is_singlet_dp:
            _pl_raw = mp_ref.get('predicted_lines') or []
            if len(_pl_raw) >= 2:
                _max_ri_pl = max(float(l.get('rel_intensity', 1.0)) for l in _pl_raw) or 1.0
                plot_sticks = [(float(l['ppm']),
                                float(l.get('rel_intensity', 1.0)) / _max_ri_pl)
                               for l in _pl_raw]
                # Recompute centre as intensity-weighted observed multiplet centre
                _tot_w = sum(s[1] for s in plot_sticks)
                centre = (sum(s[0] * s[1] for s in plot_sticks) / _tot_w
                          if _tot_w > 0 else centre)
            else:
                plot_sticks = _db_sticks_raw
        else:
            plot_sticks = _db_sticks_raw

        db_width = max(
            abs(dp.get('ppm_end', dp_ppm) - dp.get('ppm_start', dp_ppm)), 0.002)

        # Window is derived from the ACTUAL matched stick span, not DB metadata.
        # This prevents an overly wide window that includes unrelated signals.
        stick_ppms_plot = [s[0] for s in plot_sticks]
        stick_span_plot = (max(stick_ppms_plot) - min(stick_ppms_plot)
                           ) if len(plot_sticks) > 1 else 0.0
        half_win = max(stick_span_plot * 2.5 + 0.012, 0.025)
        hwhm     = max(stick_span_plot / 6 if stick_span_plot > 0
                       else db_width / 2, 0.001)

        lo = centre - half_win
        hi = centre + half_win

        # ── Experimental spectrum ──────────────────────────────────────────────
        mask  = (self.ppm >= lo) & (self.ppm <= hi)
        x_exp = self.ppm[mask]

        # Use best-group mean (most intense group wins the match display)
        groups, gcols = self._build_group_info()
        _best_lbl, y_best = self._best_group_mean(mask, groups)
        y_exp = y_best.copy()

        # ── Check if this DB match was to a deconvolved sub-peak ─────────────────
        # Use self.peaks (already expanded with sub-peaks by _expand_peaks_with_subpeaks).
        # If the matched peak is a sub-peak, find all sibling sub-peaks sharing the
        # same rank and build Lorentzian components from their stored parameters.
        _deconv_components = []   # list of (x_comp, y_comp) for individual sub-peaks
        _use_deconv_display = False
        _saved_sub_pks = []       # sub-peak dicts for marker matching

        _is_sub = bool(mp_ref.get('is_sub_peak', False)) if mp_ref else False
        if _is_sub and mp_ref is not None:
            _parent_rank = mp_ref.get('peak_rank')
            # All siblings: sub-peaks with the same parent rank
            _siblings = [p for p in (self.peaks or [])
                         if p.get('is_sub_peak') and int(p.get('rank', -1)) == _parent_rank]
            if _siblings:
                # region_ppm is stored on each sub-peak during expansion
                _reg_ppm_raw = _siblings[0].get('region_ppm')
                if _reg_ppm_raw is not None and len(_reg_ppm_raw) >= 4:
                    _reg_ppm = np.asarray(_reg_ppm_raw, dtype=float)
                    _reg_mask = (_reg_ppm >= lo) & (_reg_ppm <= hi)
                    _x_reg   = _reg_ppm[_reg_mask]
                    if len(_x_reg) >= 4:
                        _y_sum = np.zeros(len(_x_reg))
                        for _sp in _siblings:
                            _cen = float(_sp.get('ppm', 0))
                            _hw  = float(_sp.get('width_ppm',
                                         _sp.get('width_at_half', 0.004))) / 2.0
                            _hw  = max(_hw, 1e-5)
                            _ht  = float(_sp.get('height', 1.0))
                            _yc  = _ht * _hw**2 / ((_x_reg - _cen)**2 + _hw**2)
                            _deconv_components.append((_x_reg, _yc))
                            _y_sum += _yc
                            _saved_sub_pks.append({'ppm': _cen, 'height': _ht})
                        if _y_sum.max() > 0:
                            _use_deconv_display = True
                            x_exp = _x_reg
                            y_exp = _y_sum

        y_norm = np.zeros_like(y_exp)          # always defined
        if len(x_exp) >= 4 and y_exp.max() > 0:

            s_lo = min(stick_ppms_plot) - hwhm * 4
            s_hi = max(stick_ppms_plot) + hwhm * 4
            doublet_mask = (x_exp >= s_lo) & (x_exp <= s_hi)
            ref_max = (float(y_exp[doublet_mask].max())
                       if doublet_mask.any() and y_exp[doublet_mask].max() > 0
                       else float(y_exp.max()))
            ref_max = max(ref_max, 1e-10)
            y_norm  = y_exp / ref_max

            if _use_deconv_display:
                # ── Deconvolved region display ───────────────────────────────
                # Show faded raw experimental spectrum behind the components
                _raw_mask = (self.ppm >= lo) & (self.ppm <= hi)
                _x_raw    = self.ppm[_raw_mask]
                _, _y_raw_best = self._best_group_mean(_raw_mask, groups)
                if _y_raw_best.max() > 0:
                    _y_raw_n = _y_raw_best / ref_max
                    self.ax.fill_between(_x_raw, 0, _y_raw_n,
                                         color='#aaaaaa', alpha=0.12, zorder=1)
                    self.ax.plot(_x_raw, _y_raw_n,
                                 color='#aaaaaa', linewidth=0.8,
                                 linestyle=':', zorder=1, label='Experimental (raw)')

                # Individual deconvolved components — build index by peak ppm
                _comp_colors  = ['#1a6eb5', '#e07000', '#8a2be2', '#009966', '#cc2200']
                _comp_peak_y  = {}   # peak_ppm → normalised component peak height
                for _ki, (_xc, _yc) in enumerate(_deconv_components):
                    _yc_n = _yc / ref_max
                    _cc   = _comp_colors[_ki % len(_comp_colors)]
                    self.ax.fill_between(_xc, 0, _yc_n,
                                         color=_cc, alpha=0.18, zorder=2)
                    self.ax.plot(_xc, _yc_n,
                                 color=_cc, linewidth=0.9,
                                 linestyle='-', alpha=0.70, zorder=3)
                    # record peak position → height for marker placement
                    if len(_yc_n) > 0:
                        _pk_idx = int(np.argmax(_yc_n))
                        _comp_peak_y[float(_xc[_pk_idx])] = float(_yc_n[_pk_idx])

                # Sum fill (no labelled line — keeps legend clean)
                self.ax.fill_between(x_exp, 0, y_norm,
                                     color='#1a6eb5', alpha=0.15, zorder=4)

                # Override y_ref_markers: use component peak heights for markers
                # so the orange arrow sits on the matched component's own peak,
                # not the global sum.
                _y_ref_deconv = y_norm.copy()
                y_ref_markers = _y_ref_deconv   # updated below after _det_matched
            else:
                # ── Raw experimental spectrum ────────────────────────────────
                if groups and self.all_intensities is not None:
                    n_s = self.all_intensities.shape[1]
                    for g, idxs in groups.items():
                        if g == _best_lbl:
                            continue
                        _cols = [i for i in idxs if i < n_s]
                        if not _cols:
                            continue
                        _raw_mask2 = (self.ppm >= lo) & (self.ppm <= hi)
                        _gy   = np.mean(self.all_intensities[_raw_mask2, :][:, _cols], axis=1)
                        if ref_max > 0:
                            _gy_n = _gy / ref_max
                        else:
                            continue
                        col = gcols.get(g, '#888888')
                        self.ax.plot(x_exp, _gy_n, color=col,
                                     lw=0.9, alpha=0.50, zorder=1)

                self.ax.fill_between(x_exp, 0, y_norm,
                                     color='#aaaaaa', alpha=0.30, zorder=1)
                col_best = gcols.get(_best_lbl, '#1a1a6e') if _best_lbl else '#1a1a6e'
                self.ax.plot(x_exp, y_norm,
                             color=col_best, linewidth=1.2, zorder=2,
                             label=_best_lbl if _best_lbl else 'Experimental')

            y_ref_markers = y_norm

        # ── Snap detected peaks to predicted lines BEFORE drawing DB curve ─────
        # This lets us recentre plot_sticks on the actual observed peak positions
        # so the DB matched curve aligns with the experimental spectrum.
        _ann_st_plot = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        _plot_tol    = float(getattr(_ann_st_plot, 'tolerance', 0.05) or 0.05)

        if _use_deconv_display and _saved_sub_pks:
            _all_det = [{'ppm': float(sp.get('ppm', 0)),
                         'rank': f'_sub_{_si}',
                         '_is_deconv_sub': True}
                        for _si, sp in enumerate(_saved_sub_pks)]
        else:
            _all_det = list(self.peaks or [])

        _pred_lines_plot = (mp_ref.get('predicted_lines') or []) if mp_ref else []
        if _pred_lines_plot and len(_pred_lines_plot) >= 2:
            _pl_ppms_raw = sorted(float(l['ppm']) for l in _pred_lines_plot)
            _pred_centre = sum(_pl_ppms_raw) / len(_pl_ppms_raw)
            _cur_peak_ppm = float(mp_ref.get('peak_ppm', dp_ppm)) if mp_ref else dp_ppm
            _alignment_shift = _cur_peak_ppm - _pred_centre
            _min_sep = min(abs(_pl_ppms_raw[k+1] - _pl_ppms_raw[k])
                           for k in range(len(_pl_ppms_raw) - 1))
            _half_span_p = (max(_pl_ppms_raw) - _pred_centre)
            if abs(_alignment_shift) > _half_span_p + _min_sep * 0.5:
                _snap_ppms = [(float(l['ppm']) + _alignment_shift, 1.0)
                              for l in _pred_lines_plot]
            else:
                _snap_ppms = [(float(l['ppm']), 1.0) for l in _pred_lines_plot]
        else:
            _snap_ppms = plot_sticks

        _used_ranks = set()
        _det_matched = []
        for _st_ppm_s, _ in _snap_ppms:
            _candidates_s = [p for p in _all_det
                             if p.get('rank') not in _used_ranks
                             and abs(float(p.get('ppm', 0)) - _st_ppm_s) <= _plot_tol]
            if _candidates_s:
                _best_s = min(_candidates_s, key=lambda p: abs(float(p['ppm']) - _st_ppm_s))
                _used_ranks.add(_best_s.get('rank'))
                _det_matched.append(_best_s)
            else:
                _det_matched.append(None)

        _det_found = [p for p in _det_matched if p is not None]
        n_found    = len(_det_found)
        n_expected_snap = len(plot_sticks)

        # Rebuild plot_sticks centred on observed peaks so DB curve aligns correctly
        if _det_found and not is_singlet_dp:
            _obs_ctr_pre = float(np.mean([float(p['ppm']) for p in _det_found]))
            # Shift _db_sticks_raw (centred on 'centre') to observed centre
            _sticks_shift = _obs_ctr_pre - centre
            plot_sticks = [(p + _sticks_shift, r) for p, r in _db_sticks_raw]
            # Recompute window params from corrected sticks
            stick_ppms_plot = [s[0] for s in plot_sticks]
            stick_span_plot = (max(stick_ppms_plot) - min(stick_ppms_plot)
                               ) if len(plot_sticks) > 1 else 0.0
            hwhm = max(stick_span_plot / 6 if stick_span_plot > 0
                       else db_width / 2, 0.001)

        # ── DB multiplet simulated shape ───────────────────────────────────────
        x_db = np.linspace(lo, hi, 600)
        y_db = np.zeros_like(x_db)
        for st_ppm, st_rel in plot_sticks:
            y_db += st_rel * hwhm**2 / ((x_db - st_ppm)**2 + hwhm**2)
        if y_db.max() > 0:
            y_db /= y_db.max()
            # When deconvolved display is active, scale DB curve to the matched
            # component's actual normalised height so the fit overlays correctly.
            _cpy = locals().get('_comp_peak_y', {})
            if _use_deconv_display and _cpy:
                _mcp = min(_cpy.keys(), key=lambda cp: abs(cp - centre))
                y_db *= _cpy[_mcp]

        db_color = '#1a7a1a' if is_matched else '#cc2200'
        db_fill  = '#44bb44' if is_matched else '#ee4444'
        lbl      = 'DB matched' if is_matched else 'DB — not found'
        self.ax.fill_between(x_db, 0, y_db,
                             color=db_fill, alpha=0.22, zorder=3)
        self.ax.plot(x_db, y_db,
                     color=db_color, linewidth=1.5, linestyle='--',
                     zorder=4, label=lbl)

        # Individual sticks (highlight the multiplet lines)
        _db_scale = float(y_db.max()) if len(y_db) > 0 and y_db.max() > 0 else 1.0
        for st_ppm, st_rel in plot_sticks:
            _sh = st_rel * _db_scale * 0.95
            self.ax.plot([st_ppm, st_ppm], [0, _sh],
                         color=db_color, linewidth=2.2, alpha=0.85, zorder=5)
            self.ax.scatter([st_ppm], [_sh],
                            color=db_color, s=45, zorder=6)

        # ── Highlight span around the multiplet ───────────────────────────────
        span_lo = min(s[0] for s in plot_sticks) - hwhm * 2.5
        span_hi = max(s[0] for s in plot_sticks) + hwhm * 2.5
        span_color = '#00cc44' if is_matched else '#ffaa00'
        self.ax.axvspan(span_hi, span_lo,   # NMR inverted axis
                        alpha=0.07, color=span_color, zorder=0)

        # ── Mark detected experimental peaks near DB sticks + J from them ───────
        # For each DB stick, find the closest detected peak within a tight radius
        # (hwhm * 3 or tolerance). This avoids picking up unrelated peaks in the
        # broad display window. Each detected peak is used at most once.
        dp_type_j     = (dp.get('type') or '').lower().strip()
        is_singlet_dp = dp_type_j in ('s', 'singlet', '')
        _n_lines_map_plot = {'s':1,'singlet':1,'d':2,'doublet':2,'t':3,'triplet':3,
                             'q':4,'quartet':4,'dd':4,'dt':6,'td':6,'tt':9,'ddd':8}
        n_expected = _n_lines_map_plot.get(dp_type_j, len(plot_sticks))

        import re as _re2
        db_j_raw   = str(dp.get('j_coupling') or '').strip()
        _j_db_nums = ([float(x) for x in _re2.findall(r'\d+\.?\d*', db_j_raw)
                       if float(x) > 0] if db_j_raw else [])
        db_j_val   = _j_db_nums[0] if _j_db_nums else None

        try:
            ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            mhz_plot  = float(ann_stage.spectrometer_var.get()) \
                        if (ann_stage and hasattr(ann_stage, 'spectrometer_var')) \
                        else 600.0
        except Exception:
            mhz_plot = 600.0

        _y_ref        = locals().get('y_ref_markers', y_norm)
        _comp_peak_y  = locals().get('_comp_peak_y', {})   # from deconv display block

        if len(x_exp) > 0:
            # Light-gray down-arrows for every detected peak inside the plot
            # window — so the user can see all candidate peaks in the region
            # and judge which one is the "best match" for this analyte.
            _matched_ranks = {p.get('rank') for p in _det_found}
            for _pk in (self.peaks or []):
                try:
                    _pk_ppm = float(_pk.get('ppm', 0))
                except (TypeError, ValueError):
                    continue
                if not (lo <= _pk_ppm <= hi):
                    continue
                if _pk.get('rank') in _matched_ranks:
                    continue  # will be drawn in bold orange below
                _idx_r = int(np.argmin(np.abs(x_exp - _pk_ppm))) if len(x_exp) else 0
                _h_r   = float(_y_ref[_idx_r]) if len(_y_ref) > _idx_r else 0.0
                self.ax.plot([_pk_ppm], [_h_r + 0.04], 'v',
                             color='#b0b0b0', ms=6, zorder=7,
                             alpha=0.70, markeredgewidth=0, clip_on=True)
                self.ax.text(_pk_ppm, _h_r + 0.085,
                             f"#{_pk.get('rank', '')}",
                             fontsize=6, color='#888888',
                             ha='center', va='bottom', zorder=7, clip_on=True)

        if is_matched and len(x_exp) > 0:
            # Place bold orange markers on detected peaks matched to DB sticks.
            # When deconvolved display is active, use the matched component's own
            # peak height instead of the sum, so the arrow points to the correct peak.
            for dp_pk in _det_found:
                pk_ppm = float(dp_pk['ppm'])
                if _comp_peak_y:
                    # Find nearest component peak to this detected ppm
                    _best_cp = min(_comp_peak_y.keys(),
                                   key=lambda cp: abs(cp - pk_ppm),
                                   default=None)
                    h = _comp_peak_y[_best_cp] if _best_cp is not None else 0.0
                else:
                    idx = int(np.argmin(np.abs(x_exp - pk_ppm)))
                    h   = float(_y_ref[idx]) if len(_y_ref) > idx else 0.0
                self.ax.plot([pk_ppm], [h + 0.04], 'v',
                             color='#cc5500', ms=9, zorder=9,
                             alpha=0.95, markeredgewidth=0, clip_on=True)
                self.ax.text(pk_ppm, h + 0.095,
                             f"#{dp_pk.get('rank', '')}",
                             fontsize=7, color='#cc5500', fontweight='bold',
                             ha='center', va='bottom', zorder=9, clip_on=True)

            if not is_singlet_dp:
                mult_ok   = False
                j_obs_det = None
                dj_mean   = None

                # Use values computed by the scorer at annotation time.
                # Recomputation here is forbidden — single source of truth.
                # Exception: drag-select updates mp_ref in-place at line ~1592.
                _stored_j_obs = (float(mp_ref['observed_j_hz'])
                                 if mp_ref and mp_ref.get('observed_j_hz') is not None
                                 else None)
                _stored_dj    = (float(mp_ref['delta_j_hz'])
                                 if mp_ref and mp_ref.get('delta_j_hz') is not None
                                 else None)

                if _stored_j_obs is not None:
                    j_obs_det = _stored_j_obs
                    dj_mean   = _stored_dj
                    mult_ok   = (n_found == n_expected)
                elif n_found >= 2:
                    # Fallback for results annotated before this fix (no stored J).
                    _matched_idx  = [i for i, p in enumerate(_det_matched) if p is not None]
                    _ppms_det     = [float(_det_matched[i]['ppm']) for i in _matched_idx]
                    _idx_span     = max(_matched_idx) - min(_matched_idx)
                    _ppm_span_det = max(_ppms_det) - min(_ppms_det)
                    j_obs_det     = (_ppm_span_det * mhz_plot / _idx_span
                                     if _idx_span > 0 else _ppm_span_det * mhz_plot)
                    mult_ok       = (n_found == n_expected)
                    dj_mean       = abs(j_obs_det - db_j_val) if db_j_val else None

                # ── J card ────────────────────────────────────────────────────
                # J mismatch (ΔJ > 3 Hz): lines were found within ppm tolerance
                # but their actual spacing ≠ DB J.  Snap uses 30 Hz window so
                # geometrically "found" ≠ physically matched.  Flag clearly.
                _dj_hard    = 3.0   # Hz — same threshold as s_dj → 0
                _dj_mismatch = (dj_mean is not None and dj_mean >= _dj_hard)

                if j_obs_det is not None:
                    _mult_tag = '' if mult_ok else f'  ⚠ {n_found}/{n_expected} lines'
                    if dj_mean is not None:
                        j_card  = (f'obs {j_obs_det:.2f} / {db_j_val:.2f} Hz'
                                   f'  —  ΔJ {dj_mean:.2f} Hz{_mult_tag}')
                        j_color = ('#9a0000' if (not mult_ok or _dj_mismatch) else
                                   '#1a7a1a' if dj_mean < 1.0 else
                                   '#7a6000' if dj_mean < 3.0 else '#9a0000')
                    else:
                        j_card  = f'obs {j_obs_det:.2f} Hz{_mult_tag}'
                        j_color = '#7a6000' if mult_ok else '#9a0000'

                    if _dj_mismatch:
                        # ΔJ too large: peaks found within ppm window but wrong J.
                        # Show explicit mismatch label instead of misleading n/n count.
                        _ann_color  = '#9a0000'
                        _ann_text   = (f'J = {j_obs_det:.2f} Hz'
                                       f'  ⚠ ΔJ {dj_mean:.2f} Hz  (J mismatch)')
                        _ann_fc     = '#fff0f0'
                    else:
                        _ann_color = ('#1a7a1a' if (mult_ok and dj_mean is not None and dj_mean < 1.0)
                                      else '#7a6000' if (mult_ok and dj_mean is not None and dj_mean < 3.0)
                                      else '#9a0000')
                        _ann_text  = f'J = {j_obs_det:.2f} Hz  ({n_found}/{n_expected} lines)'
                        _ann_fc    = 'white'
                    self.ax.text(0.02, 0.04, _ann_text,
                                 transform=self.ax.transAxes,
                                 fontsize=7, color=_ann_color, va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor=_ann_fc,
                                           edgecolor=_ann_color, alpha=0.85))
                else:
                    j_card  = f'⚠ only {n_found}/{n_expected} lines detected'
                    j_color = '#9a0000'
                    self.ax.text(0.02, 0.04, j_card,
                                 transform=self.ax.transAxes,
                                 fontsize=7, color='#9a0000', va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           edgecolor='#9a0000', alpha=0.70))
                self._set_q('j_obs', j_card, j_color)

            else:
                # Singlet — show J-anchor validation if available
                _a_status = mp_ref.get('anchor_status')         if mp_ref else None
                _a_ratio  = mp_ref.get('anchor_ratio')          if mp_ref else None
                _a_conf   = mp_ref.get('anchor_pos_confidence') if mp_ref else None
                _a_dppm   = mp_ref.get('anchor_pos_delta_ppm')  if mp_ref else None
                _j_conf   = mp_ref.get('j_met_confirmed')       if mp_ref else False

                if _a_status == 'clean' and _j_conf:
                    _dppm_str = f'  Δδ {_a_dppm:+.4f} ppm' if _a_dppm is not None else ''
                    _conf_str = f'  [{_a_conf}]'             if _a_conf              else ''
                    j_card    = f'⚓ J-anchor  ratio {_a_ratio:.2f}{_dppm_str}{_conf_str}'
                    j_color   = ('#1a7a1a' if _a_conf == 'high'
                                 else '#7a6000' if _a_conf in ('medium', 'low')
                                 else '#004466')
                    self.ax.text(0.02, 0.04,
                                 f'⚓ J-anchor  ratio {_a_ratio:.2f}'
                                 + (f'  Δδ {_a_dppm:+.4f} ppm'
                                    if _a_dppm is not None else ''),
                                 transform=self.ax.transAxes, fontsize=7,
                                 color=j_color, va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='#d4f0ff',
                                           edgecolor=j_color, alpha=0.80))
                elif _a_status == 'overlap':
                    j_card  = (f'⚓ overlap  ratio {_a_ratio:.2f}'
                               if _a_ratio is not None else '⚓ anchor: overlap')
                    j_color = '#cc7700'
                elif _a_status in ('absent', 'bdl'):
                    j_card  = f'⚓ anchor: {_a_status}'
                    j_color = '#9a0000'
                else:
                    j_card  = '— singlet'
                    j_color = '#888888'
                self._set_q('j_obs', j_card, j_color)

        # ── Δppm arrow if matched and shift is visible ─────────────────────────
        if mp_ref and abs(offset) > 0.0002:
            ppm_diff = float(mp_ref.get('ppm_diff', abs(offset)))
            # Set xlim first so coordinate conversion is valid
            self.ax.set_xlim(hi + half_win * 0.05, lo - half_win * 0.05)
            x_lo_lim, x_hi_lim = self.ax.get_xlim()
            def _ppm_to_ax(ppm):
                return (ppm - x_lo_lim) / (x_hi_lim - x_lo_lim)
            x_exp_ax = _ppm_to_ax(centre)
            x_db_ax  = _ppm_to_ax(dp_ppm)
            # Place arrow below the legend (upper-right area occupies ~0.85–1.0)
            y_arr_ax  = 0.78
            y_text_ax = 0.84

            # Both endpoints must be within the visible axes [0,1].
            # If the DB position is outside the window, show text-only Δppm label.
            _ax_margin = 0.02
            _both_visible = (_ax_margin <= x_exp_ax <= 1 - _ax_margin and
                             _ax_margin <= x_db_ax  <= 1 - _ax_margin)
            if _both_visible:
                self.ax.annotate(
                    '', xy=(x_exp_ax, y_arr_ax), xytext=(x_db_ax, y_arr_ax),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='<->', color='#cc5500', lw=1.2))
                self.ax.text((x_exp_ax + x_db_ax) / 2, y_text_ax,
                             f'Δ {ppm_diff:.4f} ppm',
                             transform=self.ax.transAxes,
                             ha='center', fontsize=8, color='#cc5500')
            else:
                # DB position outside window — show Δppm as text with directional arrow
                _dir = '→' if dp_ppm < centre else '←'
                self.ax.text(0.98, y_text_ax,
                             f'{_dir} Δ {ppm_diff:.4f} ppm',
                             transform=self.ax.transAxes,
                             ha='right', va='center', fontsize=8, color='#cc5500',
                             bbox=dict(boxstyle='round,pad=0.15',
                                       facecolor='white', edgecolor='#cc5500',
                                       alpha=0.85))

        # ── Axes ──────────────────────────────────────────────────────────────
        # xlim may already be set by the arrow block — set it unconditionally.
        self.ax.set_xlim(hi + half_win * 0.05, lo - half_win * 0.05)
        # Y-axis: use 95th-percentile of all plotted y-values so that a single
        # tall peak at the window edge (e.g. an unrelated metabolite) does not
        # compress the region of interest to a flat line.
        try:
            _yd_all = np.concatenate([
                np.asarray(l.get_ydata(), dtype=float)
                for l in self.ax.lines if len(l.get_ydata()) > 0
            ])
            _yd_pos = _yd_all[_yd_all > 0]
            y_top = max(1.15,
                        float(np.percentile(_yd_pos, 95)) if len(_yd_pos) > 0
                        else 1.30)
        except Exception:
            y_top = 1.30
        self.ax.set_ylim(-0.06, y_top * 1.12)
        self.ax.set_xlabel('δ (ppm)', fontsize=9)
        self.ax.set_ylabel('Intensity (normalised)', fontsize=8)
        if is_matched:
            match_str = '✓ Matched'
            title_col = '#1a7a1a'
        else:
            match_str = '✗ Not found in spectrum'
            title_col = '#cc2200'
        self.ax.set_title(
            f"{met['name']}  ·  {dp_ppm:.4f} ppm  [{dp_type.upper()}]  "
            f"{match_str}",
            fontsize=9, fontweight='bold', color=title_col)
        self.ax.legend(loc='upper right', fontsize=7)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, alpha=0.2, linestyle=':')
        self.ax.set_facecolor('#fafafa')
        self.fig.tight_layout()
        self.plot_canvas.draw_idle()

        # ── Quality card ──────────────────────────────────────────────────────
        self._set_q('coverage',
                    f"DB: {dp_ppm:.4f} ppm  [{dp_type.upper()}]",
                    '#1a7a1a' if is_matched else '#9a0000')
        if mp_ref:
            ppm_diff  = float(mp_ref.get('ppm_diff', abs(offset)))
            s_cosine  = float(mp_ref.get('_s_cosine', 0) or 0)
            cos_text  = (f"R² = {s_cosine:.3f}  ·  Δ = {ppm_diff:.4f} ppm"
                         if s_cosine > 0 else f"Δ = {ppm_diff:.4f} ppm")
            cos_color = ('#1a7a1a' if s_cosine >= 0.7 else
                         '#7a6000' if s_cosine >= 0.4 else '#9a0000')
            if s_cosine == 0:
                cos_color = ('#1a7a1a' if ppm_diff < 0.005 else
                             '#7a6000' if ppm_diff < 0.02 else '#9a0000')
            self._set_q('cosine', cos_text, cos_color)
        else:
            self._set_q('cosine', 'No experimental match', '#9a0000')
        n_sticks = len(plot_sticks)
        ovlp_card = f"  ⚠ {n_overlap} competing metabolite(s)" if n_overlap > 0 else ''
        # Show detected/expected lines so the quality card matches the table child row
        if n_sticks > 1 and is_matched:
            _lines_txt = f"{n_found}/{n_sticks} lines detected"
            _lines_col = ('#8B6914' if n_overlap > 0
                          else '#1a7a1a' if n_found >= n_sticks
                          else '#7a6000' if n_found > 0 else '#9a0000')
        else:
            _lines_txt = f"{'✓ matched' if is_matched else '✗ missing'}"
            _lines_col = ('#8B6914' if (is_matched and n_overlap > 0)
                          else '#1a7a1a' if is_matched else '#9a0000')
        self._set_q('n_matched', f"{_lines_txt}{ovlp_card}", _lines_col)
        # Cache counts on mp_ref so the table can use them without re-scanning
        if mp_ref is not None and n_sticks > 1:
            mp_ref['_n_found_lines'] = n_found
            mp_ref['_n_exp_lines']   = n_sticks
        # J quality card: populated by the J block above when n_sticks≥2,
        # cleared here when the peak is a singlet or unmatched.
        if n_sticks < 2 or not is_matched:
            self._set_q('j_obs',
                        '— singlet' if n_sticks < 2 else '— not matched',
                        '#888888')

        # MCR quality at peak level
        if mp_ref is not None and mp_ref.get('mcr_r2') is not None:
            _r2_pk  = float(mp_ref.get('mcr_r2') or 0)
            _pr_pk  = mp_ref.get('mcr_pearson_r')
            _cc_pk  = mp_ref.get('mcr_cross_corr')
            _conf_pk = mp_ref.get('mcr_confirmed')
            _parts_pk = [f"R²={_r2_pk:.3f}"]
            if _pr_pk is not None:
                _parts_pk.append(f"r={_pr_pk:.2f}")
            if _cc_pk is not None:
                _parts_pk.append(f"cc={_cc_pk:.2f}")
            _sym_pk   = '✓' if _conf_pk is True else ('✗' if _conf_pk is False else '?')
            _color_pk = '#1a7a1a' if _conf_pk is True else ('#9a0000' if _conf_pk is False else '#7a6000')
            self._set_q('mcr', f"{_sym_pk}  {' · '.join(_parts_pk)}", _color_pk)
        else:
            self._set_q('mcr', '— not run', '#888888')

        # STC-Like: graph STOCSY inter-peak correlation
        _stc = mp_ref.get('stocsy_coherence') if mp_ref else None
        if _stc is None:
            # Fallback: compute on-the-fly from current peak correlation matrix
            # (stored value is None when peak detection ran with <3 samples at
            # annotation time; re-running peak detection refreshes _peak_corr)
            _stc = self._compute_stocsy_for_met(met)
        if _stc is None:
            _pk_stage  = getattr(getattr(self, 'app', None), 'stages', {})
            _pk_stage  = _pk_stage.get('peak_detection') if isinstance(_pk_stage, dict) else None
            _n_samp    = getattr(_pk_stage, '_n_samples_pk', 0)
            if _n_samp < 3:
                self._set_q('stocsy', f'— n/a (re-run peak detection with ≥3 samples)', '#888888')
            elif len(met.get('matched_peaks', [])) < 2:
                self._set_q('stocsy', '— n/a (only 1 peak matched)', '#888888')
            else:
                self._set_q('stocsy', '— n/a (no peak pair in matrix)', '#888888')
        else:
            _stc_txt   = f"r = {_stc:.3f}"
            _stc_color = ('#1a7a1a' if _stc >= 0.90
                          else '#7a6000' if _stc >= 0.70
                          else '#8B6914' if _stc >= 0.50
                          else '#9a0000')
            self._set_q('stocsy', _stc_txt, _stc_color)

        # Reverse score: nH-weighted DB coverage
        _rev_pk = float((mp_ref.get('_s_reverse') if mp_ref else None) or 0)
        _rev_col = ('#1a7a1a' if _rev_pk >= 0.80
                    else '#7a6000' if _rev_pk >= 0.50
                    else '#9a0000')
        self._set_q('reverse', f"{_rev_pk:.3f}", _rev_col)

        # Modified indicator
        is_modified = (self._orig_pred_lines is not None and
                       mp_ref is not None and
                       mp_ref.get('predicted_lines') != self._orig_pred_lines)
        self._show_reset_btn(is_modified)

        info_ovlp = f"  ⚠ overlap: {', '.join(ovlp_names[:2])}" if ovlp_names else ''
        self.match_info.config(
            text=f"{met['name']}  ·  {dp_ppm:.4f} ppm  [{dp_type.upper()}]  "
                 f"{'— MATCHED' if is_matched else '— NOT FOUND'}{info_ovlp}")

    # ── Plot helpers ─────────────────────────────────────────────────────────

    def _build_group_info(self):
        """
        Returns (groups, gcols) where:
          groups : dict  label → [sample col indices]   (BLK excluded)
          gcols  : dict  label → matplotlib colour
        """
        import matplotlib.cm as _cm
        si = getattr(self.app, 'samples_info', []) if self.app else []
        gc = getattr(self.app, 'group_colors', {}) if self.app else {}
        groups: dict = {}
        for _i, _s in enumerate(si):
            lbl = _s.get('label', 'All')
            if str(_s.get('type', lbl)).upper() != 'BLK':
                groups.setdefault(lbl, []).append(_i)
        _cmap = _cm.get_cmap('tab10', max(len(groups), 1))
        gcols = {g: gc.get(g) or _cmap(i)
                 for i, g in enumerate(groups)}
        return groups, gcols

    def _best_group_mean(self, mask, groups):
        """
        Return (best_label, best_y_array) for the group with the highest
        peak amplitude in the masked region.  Falls back to overall mean.
        """
        raw = self.all_intensities
        if raw is None or raw.ndim < 2 or not groups:
            return None, self.mean_intensity[mask].copy()
        n_s = raw.shape[1]
        best_lbl, best_y, best_val = None, None, -1.0
        for lbl, idxs in groups.items():
            cols = [i for i in idxs if i < n_s]
            if not cols:
                continue
            gm  = np.mean(raw[mask, :][:, cols], axis=1)
            if gm.size == 0:
                continue
            val = float(np.max(gm))
            if val > best_val:
                best_val, best_lbl, best_y = val, lbl, gm
        if best_y is None:
            mi = self.mean_intensity[mask]
            best_y = mi.copy() if mi.size > 0 else np.zeros(1)
        return best_lbl, best_y

    def _deconv_sub_peaks_for(self, peak_rank):
        """
        Return list of sub-peak dicts for a deconvolved parent peak by rank,
        or None if that peak was not deconvolved by the user.
        """
        pk_stage = (getattr(self.app, 'stages', {}) or {}).get('peak_detection')
        if pk_stage is None:
            return None
        ctrl = getattr(pk_stage, '_deconv_controller', None)
        if ctrl is None:
            return None
        for i, p in enumerate(ctrl.peaks):
            if int(p.get('rank', -1)) == peak_rank and p.get('_deconvolved'):
                res = ctrl.results.get(i, {})
                subs = [sp for sp in res.get('peaks', [])
                        if sp.get('status') != 'Poor']
                return subs if subs else None
        return None

    def _deconv_sub_peaks_near(self, target_ppm, tolerance=0.05):
        """
        Search ALL deconvolved regions for sub-peaks within `tolerance` ppm of
        `target_ppm`. Returns (parent_rank, [sub_peak, ...]) for the region
        whose parent peak is closest to target_ppm, or (None, None) if none found.

        This allows matching DB peaks to deconvolved sub-peaks by PPM position,
        regardless of which parent peak rank was originally matched.
        """
        pk_stage = (getattr(self.app, 'stages', {}) or {}).get('peak_detection')
        if pk_stage is None:
            return None, None
        ctrl = getattr(pk_stage, '_deconv_controller', None)
        if ctrl is None:
            return None, None

        best_rank  = None
        best_subs  = None
        best_dist  = float('inf')

        for i, p in enumerate(ctrl.peaks):
            if not p.get('_deconvolved'):
                continue
            res  = ctrl.results.get(i, {})
            subs = [sp for sp in res.get('peaks', [])
                    if sp.get('status') != 'Poor']
            if not subs:
                continue
            # Check if any sub-peak is within tolerance of target_ppm
            for sp in subs:
                dist = abs(float(sp['ppm']) - target_ppm)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_rank = int(p.get('rank', -1))
                    best_subs = subs   # return all sub-peaks of that region

        return best_rank, best_subs

    # ── End helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_fill_colors(score: float) -> tuple:
        """Mapeamento score (0-1) → (fill_color, line_color) para fingerprint plot."""
        if score >= 0.80:
            return '#44bb44', '#1a7a1a'   # green: good annotation
        if score >= 0.50:
            return '#ddaa00', '#8b6000'   # amber: intermediate annotation
        return '#ee6644', '#aa2200'       # red-orange: poor annotation

    def _plot_fingerprint(self, met):
        # Switching to fingerprint view — clear per-peak zoom state
        self._current_dp      = None
        self._current_met_dp  = None
        self._current_mp_ref  = None
        self._orig_pred_lines = None
        self._show_reset_btn(False)
        self._set_q('j_obs',    '— select a DB peak row', '#888888')
        self._set_q('reverse',  '—', '#888888')

        self.ax.clear()

        db_peaks = met['db_peaks']
        if not db_peaks:
            self.ax.text(0.5, 0.5, "No DB fingerprint available",
                         transform=self.ax.transAxes,
                         ha='center', va='center', color='gray')
            self.plot_canvas.draw_idle()
            return

        # ── Determine ppm window ──────────────────────────────────────────────
        db_ppms = [p.get('ppm_center', p.get('ppm', 0)) for p in db_peaks]
        # Include predicted_lines positions in span
        all_ppms = list(db_ppms)
        for _mp in met['matched_peaks']:
            for _l in _mp.get('predicted_lines', []):
                all_ppms.append(_l['ppm'])
        span_lo = min(all_ppms) - 0.08
        span_hi = max(all_ppms) + 0.08

        # ── Experimental signal — only plot segments around matched peaks ──────
        exp_peak_ppms = [mp['peak_ppm'] for mp in met['matched_peaks']]
        y_global_max  = 1.0
        y_segments    = []   # (x_seg, y_mean_seg, y_all_seg)

        groups, gcols = self._build_group_info()

        for mp in met['matched_peaks']:
            pred_l = mp.get('predicted_lines', [])
            if pred_l:
                ppms_l = [l['ppm'] for l in pred_l]
                ep_lo  = min(ppms_l)
                ep_hi  = max(ppms_l)
            else:
                ep_lo = ep_hi = mp['peak_ppm']
            det_pk = next((p for p in self.peaks
                           if p.get('rank') == mp.get('peak_rank')), None)
            if det_pk is None and self.peaks:
                det_pk = min(self.peaks,
                             key=lambda p: abs(p['ppm'] - mp['peak_ppm']))
                if abs(det_pk['ppm'] - mp['peak_ppm']) > 0.05:
                    det_pk = None
            fwhm   = (det_pk.get('width_at_half', det_pk.get('width_ppm', 0.004))
                      if det_pk else 0.004)
            if fwhm is None or fwhm <= 0: fwhm = 0.004
            win    = max(fwhm * 6, (ep_hi - ep_lo) + 0.02)
            centre = (ep_lo + ep_hi) / 2
            mask   = (self.ppm >= centre - win) & (self.ppm <= centre + win)
            x_seg  = self.ppm[mask]
            if len(x_seg) >= 2:
                y_mean_seg = self.mean_intensity[mask].copy()
                bl_mean    = _local_linear_baseline(y_mean_seg)
                y_mean_seg = np.maximum(y_mean_seg - bl_mean, 0.0)
                if (self.all_intensities is not None
                        and self.all_intensities.ndim == 2):
                    y_all_seg = self.all_intensities[mask, :].copy()
                    for _k in range(y_all_seg.shape[1]):
                        bl_k = _local_linear_baseline(y_all_seg[:, _k])
                        y_all_seg[:, _k] = np.maximum(y_all_seg[:, _k] - bl_k, 0.0)
                    y_segments.append((x_seg, y_mean_seg, y_all_seg, mp))
                else:
                    y_segments.append((x_seg, y_mean_seg, None, mp))

        if y_segments:
            # Determine best (most intense) group across all segments
            _best_lbl = None
            if groups and self.all_intensities is not None:
                _best_val = -1.0
                n_s = self.all_intensities.shape[1]
                for _g, _idxs in groups.items():
                    _cols = [i for i in _idxs if i < n_s]
                    if not _cols:
                        continue
                    for _, _, _y_all, _ in y_segments:
                        if _y_all is None:
                            continue
                        _gm = np.mean(_y_all[:, _cols], axis=1)
                        _v  = float(np.max(_gm))
                        if _v > _best_val:
                            _best_val, _best_lbl = _v, _g

            # Global max from best group (for normalisation)
            _norm_segs = []
            for x_seg, y_mean_seg, y_all_seg, mp in y_segments:
                if _best_lbl and y_all_seg is not None:
                    n_s   = y_all_seg.shape[1]
                    _cols = [i for i in groups[_best_lbl] if i < n_s]
                    _ref  = np.mean(y_all_seg[:, _cols], axis=1) if _cols else y_mean_seg
                else:
                    _ref  = y_mean_seg
                _norm_segs.append(float(np.max(_ref)))
            y_global_max = max(_norm_segs) if _norm_segs else 1.0
            y_global_max = y_global_max if y_global_max > 0 else 1.0

            _legend_done = set()
            n_s = (self.all_intensities.shape[1]
                   if self.all_intensities is not None else 0)

            for x_seg, y_mean_seg, y_all_seg, mp in y_segments:
                gmax = y_global_max
                if groups and y_all_seg is not None:
                    for g, idxs in groups.items():
                        col   = gcols.get(g, '#888')
                        is_best = (g == _best_lbl)
                        _cols = [i for i in idxs if i < n_s]
                        ys    = [y_all_seg[:, i] for i in _cols]
                        # Individual sample lines (thin, low alpha)
                        for y_s in ys:
                            self.ax.plot(x_seg, y_s / gmax, color=col,
                                         lw=0.5,
                                         alpha=0.35 if is_best else 0.18,
                                         zorder=1)
                        if ys and g not in _legend_done:
                            ymean_g = np.mean(ys, axis=0)
                            self.ax.plot(x_seg, ymean_g / gmax, color=col,
                                         lw=1.8 if is_best else 1.0,
                                         alpha=0.95 if is_best else 0.50,
                                         zorder=2,
                                         label=g if is_best else f'{g} (dim)')
                            _legend_done.add(g)
                        elif ys:
                            ymean_g = np.mean(ys, axis=0)
                            self.ax.plot(x_seg, ymean_g / gmax, color=col,
                                         lw=1.8 if is_best else 1.0,
                                         alpha=0.95 if is_best else 0.50,
                                         zorder=2)
                else:
                    y_n = y_mean_seg / gmax
                    self.ax.fill_between(x_seg, 0, y_n, color='#aaaaaa',
                                         alpha=0.35, zorder=1)
                    self.ax.plot(x_seg, y_n, color='#444444',
                                 lw=0.8, zorder=2, label='Experimental')

                # ── Deconvolved sub-peak overlay (if region was deconvolved) ──
                sub_pks = self._deconv_sub_peaks_for(mp.get('peak_rank'))
                if sub_pks:
                    x_dn = np.linspace(float(x_seg[0]), float(x_seg[-1]), 400)
                    y_dn = np.zeros_like(x_dn)
                    for sp in sub_pks:
                        amp  = float(sp.get('height', 0))
                        cen  = float(sp['ppm'])
                        wid  = float(sp.get('width', sp.get('width_ppm', 0.003)))
                        display_wid = min(wid, 0.0025)  # cap so individual peaks stay visually distinct
                        y_dn += amp * display_wid**2 / ((x_dn - cen)**2 + display_wid**2)
                    if y_dn.max() > 0:
                        # Normalise sub-peak envelope to gmax scale
                        bl_ref = float(np.max(y_mean_seg)) if y_mean_seg.max() > 0 else 1.0
                        y_dn_n = y_dn / y_dn.max() * (bl_ref / gmax)
                        self.ax.plot(x_dn, y_dn_n,
                                     color='#0055bb', lw=1.4,
                                     linestyle=':', alpha=0.80, zorder=3,
                                     label='Deconv.' if 'Deconv.' not in _legend_done else '')
                        if 'Deconv.' not in _legend_done:
                            _legend_done.add('Deconv.')

        # ── DB fingerprint sticks ─────────────────────────────────────────────
        matched_db_ppms = {round(mp['db_ppm'], 3)
                           for mp in met['matched_peaks']}

        db_height_max = max((p.get('height', 1.0) for p in db_peaks), default=1.0)

        # Build lookup: db_ppm → matched experimental ppm
        db_to_exp = {round(mp['db_ppm'], 3): mp['peak_ppm']
                     for mp in met['matched_peaks']}

        # ── Simulated DB spectrum ─────────────────────────────────────────────
        # For each DB multiplet, build a sum of Lorentzianas using the
        # relative intensities from the multiplet type.
        # Matched multiplets are positioned at the experimental peak (aligned);
        # unmatched ones stay at their DB position.
        x_sim  = np.linspace(span_lo - 0.05, span_hi + 0.05, 1000)
        y_sim  = np.zeros_like(x_sim)

        # Build peak-width lookup from experimental peaks
        _pk_width = {}
        for _pk in self.peaks:
            w = _pk.get('width_ppm') or _pk.get('width_at_half', 0.002)
            _pk_width[round(_pk['ppm'], 4)] = float(w) / 2  # HWHM

        import re as _re
        for p in db_peaks:
            dp_ppm     = p.get('ppm_center', p.get('ppm', 0))
            is_matched = round(dp_ppm, 3) in matched_db_ppms
            pred_lines = p.get('predicted_lines') if is_matched else None

            if pred_lines:
                # HWHM from the actual experimental peaks matched to each line
                for line in pred_lines:
                    xp   = line['ppm']
                    ri   = line['rel_intensity']
                    # Find nearest experimental peak for its linewidth
                    near = min(self.peaks, key=lambda pk: abs(pk['ppm'] - xp),
                               default=None) if self.peaks else None
                    if near:
                        w = (near.get('width_ppm') or
                             near.get('width_at_half') or 0.004)
                        hwhm = max(float(w) / 2, 0.0005)
                    else:
                        hwhm = 0.002
                    y_sim += ri * hwhm**2 / ((x_sim - xp)**2 + hwhm**2)
            else:
                # Unmatched: use DB window / 6 for linewidth estimate
                db_w  = max(abs(p.get('ppm_end', dp_ppm) -
                                p.get('ppm_start', dp_ppm)), 0.001)
                _jn   = [float(x) for x in _re.findall(r'\d+\.?\d*',
                         str(p.get('j_coupling', '') or '')) if float(x) > 0]
                hwhm  = max(db_w / 6, 0.001)
                anchor = db_to_exp.get(round(dp_ppm, 3), dp_ppm)
                offset = anchor - dp_ppm
                for st_ppm, st_rel in (p.get('sticks') or [(dp_ppm, 1.0)]):
                    xp = st_ppm + offset
                    y_sim += st_rel * hwhm**2 / ((x_sim - xp)**2 + hwhm**2)

        # Normalise simulated spectrum to 1
        if y_sim.max() > 0:
            y_sim /= y_sim.max()

        # ── DB fingerprint — matched regions coloured by score, unmatched grey ──
        # Unmatched regions: grey dashed (no score — absence only)
        y_unmatched = np.zeros_like(y_sim)
        db_to_mp_fp = {round(mp['db_ppm'], 3): mp for mp in met['matched_peaks']}
        for p in db_peaks:
            dp_ppm = p.get('ppm_center', p.get('ppm', 0))
            if round(dp_ppm, 3) in matched_db_ppms:
                continue
            db_w = max(abs(p.get('ppm_end', dp_ppm) - p.get('ppm_start', dp_ppm)), 0.002)
            anchor   = db_to_exp.get(round(dp_ppm, 3), dp_ppm)
            region   = (x_sim >= anchor - db_w * 3) & (x_sim <= anchor + db_w * 3)
            y_unmatched[region] = np.maximum(y_unmatched[region], y_sim[region])
        if y_unmatched.max() > 0:
            self.ax.fill_between(x_sim, 0, y_unmatched,
                                  color='#aaaaaa', alpha=0.20,
                                  label='DB unmatched', zorder=3)
            self.ax.plot(x_sim, y_unmatched,
                         color='#888888', linewidth=1.0, linestyle='--', zorder=4)

        # Matched regions: colour by composite_score of the corresponding matched_peak
        _legend_score_done = set()
        for p in db_peaks:
            dp_ppm = p.get('ppm_center', p.get('ppm', 0))
            if round(dp_ppm, 3) not in matched_db_ppms:
                continue
            mp_fp  = db_to_mp_fp.get(round(dp_ppm, 3))
            score  = float((mp_fp or {}).get('composite_score', 0) or 0)
            fc, ec = self._score_to_fill_colors(score)
            db_w   = max(abs(p.get('ppm_end', dp_ppm) - p.get('ppm_start', dp_ppm)), 0.002)
            pred_lines = p.get('predicted_lines') if mp_fp else None
            if pred_lines:
                ppms   = [l['ppm'] for l in pred_lines]
                cx     = (max(ppms) + min(ppms)) / 2
                win    = (max(ppms) - min(ppms)) / 2 + db_w
            else:
                cx  = db_to_exp.get(round(dp_ppm, 3), dp_ppm)
                win = db_w * 3
            region = (x_sim >= cx - win) & (x_sim <= cx + win)
            y_p    = np.where(region, y_sim, 0.0)
            if y_p.max() > 0:
                _sc_lbl = f"score={score:.2f}" if score not in _legend_score_done else ''
                _legend_score_done.add(score)
                self.ax.fill_between(x_sim, 0, y_p, color=fc, alpha=0.30,
                                      label=_sc_lbl, zorder=3)
                self.ax.plot(x_sim, y_p, color=ec, linewidth=1.2, zorder=4)

        # ── Δppm labels on matched peaks ──────────────────────────────────────
        for mp in met['matched_peaks']:
            _sc_mp = float(mp.get('composite_score', 0) or 0)
            _, _ec_mp = self._score_to_fill_colors(_sc_mp)
            self.ax.annotate(
                f"Δ{mp['ppm_diff']:.3f}",
                xy=(mp['peak_ppm'], 0.92),
                xytext=(2, 3), textcoords='offset points',
                fontsize=6, color=_ec_mp, alpha=0.9)

        # ── J-anchor BDL / absent peak markers ───────────────────────────────
        # Peaks where the J-confirmed anchor indicates the signal is below the
        # detection threshold (obs_snr < 3) or weaker than expected (ratio < 0.35).
        # The metabolite IS confirmed present via the anchor, so the DB position
        # is correct — the signal is theoretically present but undetectable at
        # this concentration.  Draw as gray dashed vertical lines (RANSY principle).
        _bdl_label_done = False
        for mp in met['matched_peaks']:
            if mp.get('anchor_status') not in ('bdl', 'absent'):
                continue
            pos = float(mp.get('peak_ppm') or mp.get('db_ppm') or 0)
            if pos <= 0:
                continue
            lbl = '< 3σ' if not _bdl_label_done else ''
            self.ax.axvline(pos, color='#aaaaaa', linewidth=1.3,
                            linestyle='--', alpha=0.75, zorder=5, label=lbl)
            self.ax.text(pos, 1.06, '< 3σ',
                         fontsize=6, ha='center', color='#999999',
                         style='italic', zorder=6)
            _bdl_label_done = True

        # ── Legend / axes ─────────────────────────────────────────────────────
        self.ax.set_xlim(span_hi + 0.05, span_lo - 0.05)   # NMR inverted
        self.ax.set_ylim(bottom=-0.20, top=1.15)
        self.ax.set_xlabel('δ (ppm)', fontsize=9)
        self.ax.set_ylabel('Intensity (normalised to peak max)', fontsize=8)
        _mean_score_fp = (sum(mp.get('composite_score', 0) or 0
                             for mp in met['matched_peaks'])
                          / max(len(met['matched_peaks']), 1))
        _fc_title, _ = self._score_to_fill_colors(_mean_score_fp)
        self.ax.set_title(f"{met['name']}  [{_mean_score_fp:.2f}]",
                          fontsize=10, fontweight='bold', color=_fc_title)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, alpha=0.2, linestyle=':')
        # ── MCR deconvolution overlay ─────────────────────────────────────────
        # Find the detected peak for this metabolite (any matched peak rank)
        _det_pk_mcr = None
        for _mp2 in met['matched_peaks']:
            _rk2 = _mp2.get('peak_rank')
            if _rk2 is not None:
                _det_pk_mcr = next(
                    (p for p in self.peaks if p.get('rank') == _rk2), None)
                if _det_pk_mcr and _det_pk_mcr.get('mcr_annotation'):
                    break

        if _det_pk_mcr is not None:
            _mcr = _det_pk_mcr.get('mcr_annotation')
            if _mcr and _mcr.get('K', 1) >= 2:
                _r2 = _mcr.get('metrics', {}).get('r2', float('nan'))
                _K  = _mcr['K']
                self.ax.text(
                    0.02, 0.97,
                    f'MCR  K={_K}  R²={_r2:.3f}',
                    transform=self.ax.transAxes,
                    fontsize=7, va='top', color='#c62828',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#c62828', alpha=0.75))

        # Legend — collect all labeled artists automatically
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        _base_handles = [
            Patch(facecolor='#aaaaaa', alpha=0.35, label='Experimental'),
            Line2D([0], [0], color='#1a7a1a', lw=2, label='DB matched'),
            Line2D([0], [0], color='#cc2200', lw=1.2,
                   alpha=0.5, label='DB unmatched'),
        ]
        _auto_h, _auto_l = self.ax.get_legend_handles_labels()
        _bdl_lbl_set = {'Experimental', 'DB matched', 'DB unmatched'}
        if any(mp.get('anchor_status') in ('bdl', 'absent')
               for mp in met['matched_peaks']):
            _base_handles.append(
                Line2D([0], [0], color='#aaaaaa', lw=1.3,
                       linestyle='--', alpha=0.75, label='< 3σ (anchor BDL)'))
            _bdl_lbl_set.add('< 3σ')
        _extra = [(h, l) for h, l in zip(_auto_h, _auto_l)
                  if l not in _bdl_lbl_set]
        _all_h = _base_handles + [h for h, _ in _extra]
        self.ax.legend(handles=_all_h, loc='upper right', fontsize=7)

        self.ax.set_facecolor('#fafafa')
        self.fig.tight_layout()
        self.plot_canvas.draw_idle()

        # ── Quality card ──────────────────────────────────────────────────────
        n_db  = len(db_peaks)
        n_mat = len(matched_db_ppms)
        cov   = met['coverage']
        cos   = met['cosine']

        def _col(v, good, ok):
            if v >= good: return '#1a7a1a'
            if v >= ok:   return '#7a6000'
            return '#9a0000'

        self._set_q('coverage',  f"{cov:.1f}%",
                    _col(cov, 70, 40))
        self._set_q('cosine',    f"{cos:.4f}",
                    _col(cos, 0.80, 0.50))
        rri = met.get('rri')
        rri_txt   = f"{rri:.3f}" if rri is not None else 'N/A (singlets)'
        rri_color = (_col(rri, 0.85, 0.60) if rri is not None
                     else '#888888')
        self._set_q('n_matched',
                    f"{n_mat}/{n_db}  ·  RRI {rri_txt}",
                    rri_color)

        # MCR quality at metabolite level: summarise across all matched peaks
        _mcr_mps = [mp2 for mp2 in met['matched_peaks'] if mp2.get('mcr_r2') is not None]
        if not _mcr_mps:
            self._set_q('mcr', '— not run', '#888888')
        else:
            _best = max(_mcr_mps, key=lambda m: float(m.get('mcr_r2') or 0))
            _r2   = float(_best.get('mcr_r2') or 0)
            _pr   = _best.get('mcr_pearson_r')
            _cc   = _best.get('mcr_cross_corr')
            _conf = _best.get('mcr_confirmed')
            _parts = [f"R²={_r2:.3f}"]
            if _pr is not None:
                _parts.append(f"r={_pr:.2f}")
            if _cc is not None:
                _parts.append(f"cc={_cc:.2f}")
            _sym   = '✓' if _conf is True else ('✗' if _conf is False else '?')
            _color = '#1a7a1a' if _conf is True else ('#9a0000' if _conf is False else '#7a6000')
            self._set_q('mcr', f"{_sym}  {' · '.join(_parts)}", _color)

        # Reverse score: nH-weighted DB coverage (mean across matched peaks)
        _rev_vals = [float(mp2.get('_s_reverse', 0) or 0) for mp2 in met['matched_peaks']]
        _rev = sum(_rev_vals) / len(_rev_vals) if _rev_vals else 0.0
        self._set_q('reverse', f"{_rev:.3f}", _col(_rev, 0.80, 0.50))

        self.match_info.config(
            text=(f"{met['name']}  ·  {n_mat}/{n_db} DB peaks matched  "
                  f"·  {cov:.0f}% coverage  ·  cosine {cos:.3f}"))

    def _plot_peak_zoom(self, met, mp):
        """
        Show a single matched peak: experimental shape vs DB Lorentzian.

        The DB provides ppm_center + ppm_start/end (= width at base).
        We construct a normalised Lorentzian from those and overlay it
        on the normalised experimental slice around the peak.
        """
        self.ax.clear()

        exp_ppm  = float(mp['peak_ppm'])
        db_ppm   = float(mp['db_ppm'])
        ppm_diff = float(mp['ppm_diff'])

        # Find the DB peak entry to get width
        db_entry = next(
            (p for p in met['db_peaks']
             if abs(p.get('ppm_center', p.get('ppm', 0)) - db_ppm) < 0.01),
            None)
        if db_entry:
            db_width = max(
                abs(db_entry.get('ppm_end', db_ppm) -
                    db_entry.get('ppm_start', db_ppm)),
                0.002)
        else:
            db_width = 0.005   # fallback FWHM

        # Zoom window: 4× the larger of db_width or fwhm of experimental peak
        exp_peak = next(
            (p for p in self.peaks
             if p.get('rank') == mp.get('peak_rank')),
            None)
        exp_fwhm = (exp_peak.get('width_at_half', db_width)
                    if exp_peak else db_width)
        half_win = max(db_width * 4, exp_fwhm * 4, 0.015)

        lo = min(exp_ppm, db_ppm) - half_win
        hi = max(exp_ppm, db_ppm) + half_win

        # ── Experimental shape ────────────────────────────────────────────────
        mask  = (self.ppm >= lo) & (self.ppm <= hi)
        x_exp = self.ppm[mask]
        y_exp = self.mean_intensity[mask]

        if len(x_exp) >= 2:
            # Subtract linear baseline estimated from window edges before
            # normalising, so the doublet/triplet lines fill the full y range
            # instead of appearing as tiny bumps on top of a broad baseline hump.
            _n = len(y_exp)
            _en = max(2, int(_n * 0.12))
            _xl = float(np.mean(x_exp[:_en]));  _yl = float(np.mean(y_exp[:_en]))
            _xr = float(np.mean(x_exp[-_en:])); _yr = float(np.mean(y_exp[-_en:]))
            if abs(_xr - _xl) > 1e-9:
                _slope = (_yr - _yl) / (_xr - _xl)
                y_exp_bc = y_exp - (_yl + _slope * (x_exp - _xl))
            else:
                y_exp_bc = y_exp - _yl
            y_exp_bc = np.clip(y_exp_bc, 0, None)
            y_max = float(np.max(y_exp_bc)) if np.max(y_exp_bc) > 0 else 1.0
            y_norm = y_exp_bc / y_max
            self.ax.fill_between(x_exp, 0, y_norm,
                                  color='#4477aa', alpha=0.30,
                                  label='Experimental', zorder=1)
            self.ax.plot(x_exp, y_norm,
                         color='#1a1a6e', linewidth=1.2, zorder=2)

        # ── DB multiplet shape: sum of Lorentzians at each stick position ──────
        # The sticks are anchored at exp_ppm (already shifted by offset above)
        db_entry = next(
            (p for p in met['db_peaks']
             if abs(p.get('ppm_center', p.get('ppm', 0)) - db_ppm) < 0.01),
            None)
        sticks = (db_entry.get('sticks') if db_entry else None) or [(db_ppm, 1.0)]
        # Shift sticks to align with experimental peak
        offset  = exp_ppm - db_ppm
        sticks  = [(sp + offset, si) for sp, si in sticks]
        hwhm    = max(db_width / 2, 0.001)

        x_db    = np.linspace(lo, hi, 600)
        y_db    = np.zeros_like(x_db)
        for st_ppm, st_rel in sticks:
            y_db += st_rel * hwhm**2 / ((x_db - st_ppm)**2 + hwhm**2)
        if y_db.max() > 0:
            y_db /= y_db.max()

        self.ax.fill_between(x_db, 0, y_db,
                              color='#44aa44', alpha=0.25,
                              label='DB fingerprint', zorder=1)
        self.ax.plot(x_db, y_db,
                     color='#1a7a1a', linewidth=1.5,
                     linestyle='--', zorder=3)

        # Individual sticks showing relative intensities
        for st_ppm, st_rel in sticks:
            self.ax.plot([st_ppm, st_ppm], [0, st_rel],
                         color='#1a7a1a', linewidth=1.5,
                         alpha=0.7, zorder=4)
            self.ax.scatter([st_ppm], [st_rel],
                            color='#1a7a1a', s=25, zorder=5)

        # ── Centre lines ──────────────────────────────────────────────────────
        self.ax.axvline(exp_ppm, color='#1a1a6e', linewidth=1.0,
                        linestyle=':', alpha=0.8, zorder=4,
                        label=f'Exp {exp_ppm:.4f}')
        # Only draw DB reference line if it falls within the visible window
        if lo <= db_ppm <= hi:
            self.ax.axvline(db_ppm, color='#1a7a1a', linewidth=1.0,
                            linestyle=':', alpha=0.8, zorder=4,
                            label=f'DB {db_ppm:.4f}')

        # ── Shift annotation ─────────────────────────────────────────────────
        # Set xlim first so coordinate conversion is valid.
        self.ax.set_xlim(hi + half_win * 0.1, lo - half_win * 0.1)

        # Use axes-fraction coordinates so the arrow stays inside the plot
        # even when db_ppm falls outside the visible window.
        x_lo_lim, x_hi_lim = self.ax.get_xlim()
        def _to_ax(ppm):
            return (ppm - x_lo_lim) / (x_hi_lim - x_lo_lim)

        x_exp_ax = _to_ax(exp_ppm)
        x_db_ax  = _to_ax(db_ppm)
        # Clamp db end of arrow to [0.01, 0.99] so it stays inside axes
        x_db_clamped = max(0.01, min(0.99, x_db_ax))
        y_ann_ax  = 0.78
        y_text_ax = 0.84

        if abs(x_exp_ax - x_db_ax) > 0.005:
            self.ax.annotate(
                '', xy=(x_exp_ax, y_ann_ax),
                xytext=(x_db_clamped, y_ann_ax),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='#cc5500', lw=1.2))
        # Label: always at a safe position (centre of visible window)
        x_label_ax = max(0.05, min(0.95, (x_exp_ax + x_db_clamped) / 2))
        self.ax.text(x_label_ax, y_text_ax,
                     f'Δ {ppm_diff:.4f} ppm',
                     transform=self.ax.transAxes,
                     ha='center', fontsize=8, color='#cc5500')

        # ── Axes ──────────────────────────────────────────────────────────────
        # xlim already set above; re-apply to be explicit.
        self.ax.set_ylim(-0.05, 1.25)
        self.ax.set_xlabel('δ (ppm)', fontsize=9)
        self.ax.set_ylabel('Intensity (normalised to peak max)', fontsize=8)
        self.ax.set_title(
            f"{met['name']}  ·  Peak #{mp['peak_rank']}",
            fontsize=9, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=7)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, alpha=0.2, linestyle=':')
        self.ax.set_facecolor('#fafafa')
        self.fig.tight_layout()
        self.plot_canvas.draw_idle()

        # Update quality card for this peak
        self._set_q('coverage',
                    f"Δ = {ppm_diff:.4f} ppm",
                    '#1a7a1a' if ppm_diff < 0.005 else
                    '#7a6000' if ppm_diff < 0.02 else '#9a0000')
        self._set_q('cosine',
                    f"width {db_width:.4f} ppm",
                    '#1a7a1a' if db_width < 0.02 else '#7a6000')
        # RRI for this specific multiplet
        exp_region = [(p['ppm'],
                        float(self.mean_intensity[
                            int(np.argmin(np.abs(self.ppm - p['ppm'])))]))
                      for p in self.peaks
                      if lo <= p['ppm'] <= hi]
        rri = _rri_score(exp_region, sticks) if len(sticks) > 1 else None
        rri_txt = f"{rri:.3f}" if rri is not None else 'N/A'
        self._set_q('n_matched',
                    f"#{mp['peak_rank']}  {mp.get('type','')}  RRI {rri_txt}",
                    ('#1a7a1a' if rri is not None and rri >= 0.85
                     else '#7a6000' if rri is not None and rri >= 0.60
                     else '#9a0000') if rri is not None else '#888888')
        _rev_pz = float(mp.get('_s_reverse', 0) or 0)
        self._set_q('reverse', f"{_rev_pz:.3f}",
                    '#1a7a1a' if _rev_pz >= 0.80 else '#7a6000' if _rev_pz >= 0.50 else '#9a0000')
        _pl = mp.get('predicted_lines', [])
        if len(_pl) > 1:
            _obs_c  = (_pl[0]['ppm'] + _pl[-1]['ppm']) / 2
            _j_obs  = abs(_pl[-1]['ppm'] - _pl[0]['ppm']) * 600.0
            _jdb    = mp.get('j_coupling', '')
            self.match_info.config(
                text=(f"{met['name']}  ·  centre {_obs_c:.4f}  →  "
                      f"db {db_ppm:.4f}  Δ{ppm_diff:.4f} ppm  ·  "
                      f"J_obs={_j_obs:.2f}Hz"))
        else:
            self.match_info.config(
                text=(f"{met['name']}  ·  exp {exp_ppm:.4f}  →  "
                      f"db {db_ppm:.4f}  Δ{ppm_diff:.4f} ppm  ·  "
                      f"width {db_width:.4f} ppm"))

    def _compute_stocsy_for_met(self, met) -> 'float | None':
        """Compute mean pairwise Pearson r for a metabolite's matched peaks.

        Uses the live _peak_corr matrix from peak_detection stage.  Returns None
        when the matrix is unavailable or fewer than 2 matched peaks have ranks.
        """
        _pk_stage = getattr(getattr(self, 'app', None), 'stages', {})
        if isinstance(_pk_stage, dict):
            _pk_stage = _pk_stage.get('peak_detection')
        _pk_corr = getattr(_pk_stage, '_peak_corr', {})
        if not _pk_corr:
            return None
        ranks = [mp.get('peak_rank') for mp in met.get('matched_peaks', [])
                 if mp.get('peak_rank') is not None]
        if len(ranks) < 2:
            return None
        pairs = [(min(a, b), max(a, b))
                 for i, a in enumerate(ranks)
                 for b in ranks[i + 1:]]
        rs = [_pk_corr[p] for p in pairs if p in _pk_corr]
        return float(sum(rs) / len(rs)) if rs else None

    def _set_q(self, key, text, color):
        if key in self._q_labels:
            vl, ind = self._q_labels[key]
            vl.config(text=text)
            ind.config(fg=color)

    # ── Right-click drag region selection ────────────────────────────────────

    def _tk_ppm_from_event(self, e):
        """Convert tkinter widget pixel event to ppm (matplotlib data x-coord)."""
        try:
            # ax.transData.inverted() maps display (figure-level px, y-up) to data.
            # Tkinter y is measured top-down; flip to get matplotlib display y.
            h = self.plot_canvas.get_tk_widget().winfo_height()
            x_data, _ = self.ax.transData.inverted().transform((e.x, h - e.y))
            return float(x_data)
        except Exception:
            return None

    def _tk_rc_press(self, e):
        xdata = self._tk_ppm_from_event(e)
        if xdata is None:
            return
        self._rc_drag_start = xdata
        if self._rc_span_patch is not None:
            try:
                self._rc_span_patch.remove()
            except Exception:
                pass
            self._rc_span_patch = None
        self.plot_canvas.draw_idle()

    def _tk_rc_move(self, e):
        if self._rc_drag_start is None:
            return
        xdata = self._tk_ppm_from_event(e)
        if xdata is None:
            return
        x0, x1 = sorted([self._rc_drag_start, xdata])
        if self._rc_span_patch is not None:
            try:
                self._rc_span_patch.remove()
            except Exception:
                pass
        self._rc_span_patch = self.ax.axvspan(
            x0, x1, color='#ff9900', alpha=0.18, zorder=10)
        self.plot_canvas.draw_idle()

    def _tk_rc_release(self, e):
        if self._rc_drag_start is None:
            return
        xdata = self._tk_ppm_from_event(e)
        start = self._rc_drag_start
        self._rc_drag_start = None
        if self._rc_span_patch is not None:
            try:
                self._rc_span_patch.remove()
            except Exception:
                pass
            self._rc_span_patch = None
        if xdata is None:
            self.plot_canvas.draw_idle()
            return
        if self._current_dp is None:
            self.plot_canvas.draw_idle()
            try:
                self.match_info.config(
                    text='Select a peak in the list first, then right-drag.',
                    foreground='orange')
            except Exception:
                pass
            return
        lo_ppm, hi_ppm = min(start, xdata), max(start, xdata)
        if abs(hi_ppm - lo_ppm) < 0.0005:
            self.plot_canvas.draw_idle()
            return
        self._rc_select_region(lo_ppm, hi_ppm)

    def _show_reset_btn(self, modified):
        """Update the 'modified' indicator next to the reset button."""
        try:
            if modified:
                self._reset_modified_lbl.config(text='● peaks modified by selection')
            else:
                self._reset_modified_lbl.config(text='')
        except Exception:
            pass

    def _rc_press(self, event):
        if event.button != 3 or event.inaxes != self.ax:
            return
        if event.xdata is None:
            return
        self._rc_drag_start = event.xdata
        # Remove old span if any
        if self._rc_span_patch is not None:
            try:
                self._rc_span_patch.remove()
            except Exception:
                pass
            self._rc_span_patch = None
        self.plot_canvas.draw_idle()

    def _rc_move(self, event):
        if self._rc_drag_start is None or event.inaxes != self.ax:
            return
        if event.xdata is None:
            return
        x0, x1 = sorted([self._rc_drag_start, event.xdata])
        if self._rc_span_patch is not None:
            try:
                self._rc_span_patch.remove()
            except Exception:
                pass
        self._rc_span_patch = self.ax.axvspan(
            x0, x1, color='#ff9900', alpha=0.18, zorder=10)
        self.plot_canvas.draw_idle()

    def _rc_release(self, event):
        if event.button != 3 or self._rc_drag_start is None:
            return
        x_end = event.xdata
        start  = self._rc_drag_start
        self._rc_drag_start = None

        # Remove selection span
        if self._rc_span_patch is not None:
            try:
                self._rc_span_patch.remove()
            except Exception:
                pass
            self._rc_span_patch = None

        if x_end is None:
            self.plot_canvas.draw_idle()
            return
        if self._current_dp is None:
            self.plot_canvas.draw_idle()
            self.match_info.config(
                text='Select a peak in the list first, then right-drag.',
                foreground='orange')
            return

        lo_ppm, hi_ppm = min(start, x_end), max(start, x_end)
        if abs(hi_ppm - lo_ppm) < 0.0005:
            self.plot_canvas.draw_idle()
            return

        self._rc_select_region(lo_ppm, hi_ppm)

    # Multiplicity → number of expected peaks
    _MULT_N = {
        's': 1, 'd': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6,
        'dd': 4, 'dt': 6, 'td': 6, 'ddd': 8,
    }

    def _recompute_and_refresh(self):
        """Full scoring pipeline + UI refresh. Called after parameter changes."""
        try:
            self._compute_scores()
            self._propagate_multiplet_confirmation()
            self._filter_incompatible_candidates()
            self._propagate_j_anchor_quality()
            self._propagate_anchor_position()
            self._enrich_scores()
            self._adjust_scores_for_competition()
            self._classify_confidence_classes()
            self._auto_mark_good()
            self._require_multi_peak_confirmation()
            self._nnls_spectral_validation()
            self._build_peak_assignments()
            self._populate()
        except Exception:
            pass

    def _rescore_after_edit(self, met):
        """Re-run the full scoring pipeline after a manual predicted_lines edit.
        Updates mp_ref fields, annotation_quality, anchors, and refreshes tree rows.
        Restores the treeview selection to the same DB peak so the view doesn't reset."""
        # Save current context before repopulation wipes the tree
        _saved_met_id  = met.get('id') if met else None
        _saved_dp      = self._current_dp
        _saved_dp_ppm  = (float(_saved_dp.get('ppm_center', _saved_dp.get('ppm', 0)))
                          if _saved_dp is not None else None)
        try:
            self._compute_scores()
            self._enrich_scores()
            self._adjust_scores_for_competition()
            self._auto_mark_good()
            self._require_multi_peak_confirmation()
            self._nnls_spectral_validation()
            self._build_peak_assignments()
            self._populate()
        except Exception:
            return

        # Restore selection to the same child row (DB peak) without triggering
        # _on_select (which would re-render the plot and clear _orig_pred_lines).
        if _saved_met_id is None or _saved_dp_ppm is None:
            return
        target_ciid = None
        for ciid, (cmet, cdp, _cmp) in self._iid_to_child.items():
            if cmet.get('id') == _saved_met_id:
                cdp_ppm = float(cdp.get('ppm_center', cdp.get('ppm', 0)))
                if abs(cdp_ppm - _saved_dp_ppm) < 0.001:
                    target_ciid = ciid
                    break
        if target_ciid is None:
            return
        # Temporarily unbind to avoid _on_select re-rendering the plot.
        # _populate() already fired _on_select → _plot_fingerprint which set
        # _current_dp = None; restore all state so right-drag keeps working.
        self.tree.unbind('<<TreeviewSelect>>')
        try:
            self.tree.selection_set(target_ciid)
            self.tree.focus(target_ciid)
            self.tree.see(target_ciid)
            _restored = self._iid_to_child.get(target_ciid)
            if _restored:
                _cmet, _cdp, _cmp = _restored
                self._current_dp     = _cdp
                self._current_met_dp = _cmet
                self._current_mp_ref = _cmp
        finally:
            self.tree.bind('<<TreeviewSelect>>', self._on_select)

    def _rc_select_region(self, lo_ppm, hi_ppm):
        """Find N highest peaks in [lo_ppm, hi_ppm] and update predicted_lines."""
        dp     = self._current_dp
        mp_ref = self._current_mp_ref
        met    = self._current_met_dp
        if dp is None or met is None:
            return

        dp_type = (dp.get('type', 's') or 's').lower().strip()
        n_peaks = self._MULT_N.get(dp_type, 1)

        # Slice experimental spectrum
        mask = (self.ppm >= lo_ppm) & (self.ppm <= hi_ppm)
        if mask.sum() < 2:
            return
        x_reg = self.ppm[mask]
        y_reg = self.mean_intensity[mask].copy()

        # Try scipy find_peaks; fall back to top-N by intensity
        try:
            from scipy.signal import find_peaks as _fp
            # distance: at least 3 points between peaks
            peaks_idx, _ = _fp(y_reg, distance=max(1, len(y_reg) // (n_peaks * 4 + 1)))
        except ImportError:
            peaks_idx = np.array([])

        if len(peaks_idx) == 0:
            # Use top-N point indices by intensity
            peaks_idx = np.argsort(y_reg)[::-1][:n_peaks * 3]

        # Take N highest by intensity
        if len(peaks_idx) > n_peaks:
            top = sorted(peaks_idx, key=lambda i: y_reg[i], reverse=True)[:n_peaks]
            peaks_idx = np.array(top)

        if len(peaks_idx) == 0:
            return

        # Sort by ppm (ascending = high-field first as in NMR convention)
        peaks_idx = sorted(peaks_idx, key=lambda i: x_reg[i])

        # Build relative intensities from type
        intensities = _relative_intensities_from_type(dp_type, len(peaks_idx))

        new_pred_lines = []
        for k, idx in enumerate(peaks_idx):
            new_pred_lines.append({
                'ppm':           float(x_reg[idx]),
                'rel_intensity': intensities[k] if k < len(intensities) else 1.0,
            })

        # _orig_pred_lines is taken in _plot_db_peak_zoom on first display;
        # no separate backup needed here.

        # Update the matched_peak dict.
        # If this was an unmatched DB peak (mp_ref is None), create a new entry
        # so that the right-drag effectively turns it into a matched peak.
        if mp_ref is None and new_pred_lines and self._current_dp is not None:
            dp_ref = self._current_dp
            dp_ppm_ref = float(dp_ref.get('ppm_center', dp_ref.get('ppm', 0)))
            tot = sum(l['rel_intensity'] for l in new_pred_lines)
            obs_c = (sum(l['ppm'] * l['rel_intensity'] for l in new_pred_lines) / tot
                     if tot > 0
                     else sum(l['ppm'] for l in new_pred_lines) / len(new_pred_lines))
            mp_ref = {
                'db_ppm':          dp_ppm_ref,
                'peak_ppm':        obs_c,
                'ppm_diff':        abs(obs_c - dp_ppm_ref),
                'peak_rank':       None,
                'predicted_lines': new_pred_lines,
                'annotation_quality': None,
            }
            met.setdefault('matched_peaks', []).append(mp_ref)
            self._current_mp_ref = mp_ref
            self._orig_pred_lines = list(new_pred_lines)

        if mp_ref is not None:
            mp_ref['predicted_lines'] = new_pred_lines
            # Recompute ppm_diff using intensity-weighted centre
            if new_pred_lines:
                tot = sum(l['rel_intensity'] for l in new_pred_lines)
                if tot > 0:
                    obs_c = sum(l['ppm'] * l['rel_intensity']
                                for l in new_pred_lines) / tot
                else:
                    obs_c = (new_pred_lines[0]['ppm'] + new_pred_lines[-1]['ppm']) / 2
                mp_ref['ppm_diff'] = abs(obs_c - mp_ref.get('db_ppm', obs_c))
                mp_ref['peak_ppm'] = obs_c

            # Recompute J observed (spacing in Hz)
            ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            mhz = 600.0
            try:
                if ann_stage and hasattr(ann_stage, 'spectrometer_var'):
                    mhz = float(ann_stage.spectrometer_var.get())
            except Exception:
                pass
            if len(new_pred_lines) >= 2:
                sorted_pl = sorted(new_pred_lines, key=lambda l: l['ppm'])
                seps_ppm  = [abs(sorted_pl[i+1]['ppm'] - sorted_pl[i]['ppm'])
                             for i in range(len(sorted_pl) - 1)]
                j_obs_hz  = [s * mhz for s in seps_ppm]
                mp_ref['j_obs_hz']  = j_obs_hz
                mp_ref['j_obs_ppm'] = seps_ppm

        self._show_reset_btn(True)
        # Redraw the zoom view, then re-run scoring so quality/anchors update
        self._plot_db_peak_zoom(met, dp)
        self._rescore_after_edit(met)

    def _rc_reset(self):
        """Restore predicted_lines to the initial state (taken on first display)."""
        if self._current_dp is None:
            return
        if self._current_mp_ref is None or self._orig_pred_lines is None:
            return
        # Restore to initial backup
        restored = list(self._orig_pred_lines)  # keep backup intact
        self._current_mp_ref['predicted_lines'] = restored
        # Restore ppm_diff / peak_ppm
        if restored:
            tot = sum(l['rel_intensity'] for l in restored)
            if tot > 0:
                obs_c = sum(l['ppm'] * l['rel_intensity'] for l in restored) / tot
            else:
                obs_c = (restored[0]['ppm'] + restored[-1]['ppm']) / 2
            db_ppm = self._current_mp_ref.get('db_ppm', obs_c)
            self._current_mp_ref['ppm_diff'] = abs(obs_c - db_ppm)
            self._current_mp_ref['peak_ppm'] = obs_c
        # Keep _orig_pred_lines intact so repeated resets always go to the same state.
        self._plot_db_peak_zoom(self._current_met_dp, self._current_dp)
        self._rescore_after_edit(self._current_met_dp)

    # ── Zoom / pan ────────────────────────────────────────────────────────────

    def _zoom(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.85 if event.button == 'up' else 1 / 0.85
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xd   = event.xdata or (xlim[0]+xlim[1])/2
        yd   = event.ydata or (ylim[0]+ylim[1])/2
        self.ax.set_xlim(xd+(xlim[0]-xd)*factor, xd+(xlim[1]-xd)*factor)
        self.ax.set_ylim(yd+(ylim[0]-yd)*factor, yd+(ylim[1]-yd)*factor)
        self.plot_canvas.draw_idle()

    def _pan_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._pan_active = True
        self._pan_start  = (event.xdata, event.ydata)
        self.plot_canvas.get_tk_widget().config(cursor='fleur')

    def _pan_release(self, event):
        self._pan_active = False
        self._pan_start  = None
        self.plot_canvas.get_tk_widget().config(cursor='')

    def _pan_move(self, event):
        if not self._pan_active or event.inaxes != self.ax:
            return
        if event.xdata is None:
            return
        dx = self._pan_start[0] - event.xdata
        dy = self._pan_start[1] - event.ydata
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        self.ax.set_xlim(xl[0]+dx, xl[1]+dx)
        self.ax.set_ylim(yl[0]+dy, yl[1]+dy)
        self._pan_start = (event.xdata, event.ydata)
        self.plot_canvas.draw_idle()

    # ── Annotation Parameters dialog ─────────────────────────────────────────

    def _open_param_dialog(self):
        """Open a tabbed dialog for editing all annotation parameters."""
        dlg = tk.Toplevel(self.win)
        dlg.title("Annotation Parameters")
        dlg.resizable(False, False)
        dlg.transient(self.win)
        dlg.grab_set()

        nb = ttk.Notebook(dlg)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Storage for StringVar widgets: key → StringVar
        _vars = {}

        def _tab(title, rows):
            """Create a tab with label/entry pairs. rows = [(label, key, tooltip), ...]"""
            frm = ttk.Frame(nb, padding=10)
            nb.add(frm, text=title)
            for r, (lbl, key, tip) in enumerate(rows):
                ttk.Label(frm, text=lbl, anchor=tk.W).grid(
                    row=r, column=0, sticky=tk.W, padx=(0, 10), pady=3)
                var = tk.StringVar(value=str(self._params[key]))
                _vars[key] = var
                ent = ttk.Entry(frm, textvariable=var, width=10)
                ent.grid(row=r, column=1, sticky=tk.W, pady=3)
                if tip:
                    ttk.Label(frm, text=tip, foreground='gray',
                              font=('Arial', 8)).grid(
                        row=r, column=2, sticky=tk.W, padx=(6, 0))

        # ── Tab 1: Score & Quality ─────────────────────────────────────────
        _tab("Score & Quality", [
            ("Good score threshold",  'good_score_thresh', "Rules 3/4/5 — min score → Good"),
            ("Bad score threshold",   'bad_score_thresh',  "Rule 1 — singlets only"),
            ("Fingerprint coverage %", 'fp_coverage_pct',  "≥2 DB peaks: min coverage %"),
            ("NNLS min fraction",     'nnls_min_frac',     "< this → downgrade to yellow"),
        ])

        # ── Tab 2: ΔJ Criteria ────────────────────────────────────────────
        _tab("ΔJ Criteria (Hz)", [
            ("Gold standard ΔJ",    'dj_gold_hz',      "Rule 2: ΔJ < this → Good candidate"),
            ("Bad ΔJ threshold",    'dj_bad_hz',       "Rule 0: ΔJ ≥ this → Bad"),
            ("Plausible ΔJ (conf.)", 'dj_plausible_hz', "Multiplet confirmation criterion A"),
            ("s_dj ramp low",       'dj_soft_lo_hz',   "ΔJ < this → s_dj = 1.0"),
            ("s_dj ramp high",      'dj_soft_hi_hz',   "ΔJ > this → s_dj = 0.0"),
            ("s_dj neutral",        'dj_neutral',      "s_dj when ΔJ unmeasurable"),
        ])

        # ── Tab 3: Δppm Criteria ──────────────────────────────────────────
        _tab("Δppm Criteria (ppm)", [
            ("Gold standard Δppm",    'dppm_gold',         "Rule 2: max Δppm"),
            ("Anchor pos. high",      'dppm_anchor_high',  "High confidence threshold"),
            ("Anchor pos. medium",    'dppm_anchor_med',   "Medium confidence threshold"),
        ])

        # ── Tab 4: RANSY Anchor ───────────────────────────────────────────
        _tab("RANSY Anchor", [
            ("Clean ratio low",  'anchor_clean_lo',  "Min area ratio → clean"),
            ("Clean ratio high", 'anchor_clean_hi',  "Max area ratio → clean (above = overlap)"),
            ("SNR BDL threshold", 'anchor_snr_bdl',  "SNR < this → below detection limit"),
        ])

        # ── Tab 5: Score Weights ──────────────────────────────────────────
        _tab("Score Weights", [
            ("w_dj (non-singlet)",     'w_dj',        "ΔJ score weight"),
            ("w_cosine (non-singlet)", 'w_cosine_ns', ""),
            ("w_ppm (non-singlet)",    'w_ppm_ns',    ""),
            ("w_found (non-singlet)",  'w_found_ns',  ""),
            ("w_rri (non-singlet)",    'w_rri_ns',    ""),
            ("w_cosine (singlet)",     'w_cosine_s',  ""),
            ("w_ppm (singlet)",        'w_ppm_s',     ""),
            ("w_found (singlet)",      'w_found_s',   ""),
            ("w_rri (singlet)",        'w_rri_s',     ""),
        ])

        # ── Bottom buttons ─────────────────────────────────────────────────
        bf2 = ttk.Frame(dlg)
        bf2.pack(fill=tk.X, padx=8, pady=(0, 8))

        def _apply():
            errs = []
            new_params = {}
            for key, var in _vars.items():
                try:
                    new_params[key] = float(var.get())
                except ValueError:
                    errs.append(key)
            if errs:
                import tkinter.messagebox as mb
                mb.showerror("Invalid value",
                             f"Not a number: {', '.join(errs)}", parent=dlg)
                return
            self._params.update(new_params)
            self._recompute_and_refresh()
            dlg.destroy()

        def _reset():
            defaults = {
                'good_score_thresh': 0.80, 'bad_score_thresh': 0.50,
                'dj_gold_hz': 0.20, 'dj_bad_hz': 0.20,
                'dj_soft_lo_hz': 0.50, 'dj_soft_hi_hz': 3.00,
                'dj_plausible_hz': 2.00, 'dj_neutral': 0.50,
                'dppm_gold': 0.05, 'dppm_anchor_high': 0.02,
                'dppm_anchor_med': 0.05, 'fp_coverage_pct': 50.0,
                'nnls_min_frac': 0.03, 'anchor_clean_lo': 0.35,
                'anchor_clean_hi': 2.50, 'anchor_snr_bdl': 3.0,
                'w_dj': 0.60, 'w_cosine_ns': 0.15, 'w_ppm_ns': 0.10,
                'w_found_ns': 0.05, 'w_rri_ns': 0.10,
                'w_cosine_s': 0.35, 'w_ppm_s': 0.30,
                'w_found_s': 0.15, 'w_rri_s': 0.20,
            }
            for key, var in _vars.items():
                var.set(str(defaults.get(key, self._params[key])))

        ttk.Button(bf2, text="Reset Defaults", command=_reset).pack(side=tk.LEFT)
        ttk.Button(bf2, text="Apply & Recompute", command=_apply).pack(side=tk.RIGHT)
        ttk.Button(bf2, text="Cancel",
                   command=dlg.destroy).pack(side=tk.RIGHT, padx=6)

    # ── MCR Deconvolution for Undef peaks ────────────────────────────────────

    def _open_mcr_deconv(self):
        """Collect all Undef peaks and open AnnotationMCRWindow."""
        # Gather (met_dict, mp_dict, peak_dict) for every Undef matched peak
        app        = getattr(self, 'app', None)
        pd_stage   = getattr(app, 'stages', {}).get('peak_detection') if app else None
        ann_stage  = (getattr(app, 'stages', {}).get('annotation') if app else None)

        undef_items = []
        for met in self.metabolites:
            for mp in met['matched_peaks']:
                if mp.get('annotation_quality') is not None:
                    continue   # skip Good and Bad — only Undef
                rank = mp.get('peak_rank')
                peak = None
                if pd_stage and rank is not None:
                    peak = next((p for p in pd_stage.peaks
                                 if p.get('rank') == rank), None)
                if peak is None:
                    # Build a minimal stub so the window can still open
                    ppm = float(mp.get('peak_ppm', mp.get('db_ppm', 0)))
                    peak = {'rank': rank or 0, 'ppm': ppm,
                            'start_ppm': ppm - 0.015,
                            'end_ppm':   ppm + 0.015}
                undef_items.append((met, mp, peak))

        if not undef_items:
            from tkinter import messagebox
            messagebox.showinfo(
                'No Undef Peaks',
                'No Undefined peaks found.\n\n'
                'Use Ctrl+2 on a selected peak to mark it as Undef.',
                parent=self.win)
            return

        from gui.widgets.annotation_mcr_window import AnnotationMCRWindow
        AnnotationMCRWindow(
            self.win,
            undef_items,
            app,
            ann_stage=ann_stage,
            ann_results=self)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        from tkinter import filedialog
        import csv
        fn = filedialog.asksaveasfilename(
            title="Export Annotation Results",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile="annotation_results.csv",
            parent=self.win)
        if not fn:
            return
        try:
            with open(fn, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow([
                    'Peak #', 'Metabolite', 'Met ID',
                    'PPM', 'Start PPM', 'End PPM',
                    'Score', 'Quality', 'Candidates',
                    'DB PPM', 'Δ PPM', 'Multiplicity',
                ])
                for rank in sorted(self._peak_assignments):
                    met, mp = self._peak_assignments[rank]
                    det_pk = next((p for p in self.peaks
                                   if p.get('rank') == rank), None)
                    peak_ppm  = float(mp.get('peak_ppm', 0))
                    start_ppm = float(det_pk.get('start_ppm', peak_ppm)
                                      if det_pk else peak_ppm)
                    end_ppm   = float(det_pk.get('end_ppm', peak_ppm)
                                      if det_pk else peak_ppm)
                    score     = float(
                        mp.get('composite_score',
                               mp.get('multiplet_score', 0.0)) or 0.0)
                    n_cands   = len(self._peak_candidates.get(rank, []))
                    w.writerow([
                        rank,
                        met['name'],
                        met['id'],
                        f"{peak_ppm:.5f}",
                        f"{start_ppm:.5f}",
                        f"{end_ppm:.5f}",
                        f"{score:.4f}",
                        mp.get('annotation_quality', ''),
                        n_cands,
                        f"{mp.get('db_ppm', 0):.5f}",
                        f"{mp.get('ppm_diff', 0):.5f}",
                        mp.get('type', ''),
                    ])
            tk.messagebox.showinfo("Exported", f"Saved to:\n{fn}",
                                   parent=self.win)
        except Exception as e:
            tk.messagebox.showerror("Error", str(e), parent=self.win)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _center(self):
        self.win.update_idletasks()
        w  = self.win.winfo_width()
        h  = self.win.winfo_height()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")