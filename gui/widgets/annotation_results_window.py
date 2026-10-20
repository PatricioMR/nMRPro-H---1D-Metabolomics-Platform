# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
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
from gui.utils.mcr_utils import MCR_MIN_OVERLAP_K, MCR_MIN_SHAPE_R
from gui.utils.quality_palette import (
    QUALITY_BG, QUALITY_FG, QUALITY_LABEL,
    SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR,
    build_tier_choice, build_tier_filter_row, read_tier_choice,
    tier_from_annotation_quality, update_tier_counts)
from processing.annotation_detectability import (
    build_overlap_map, mark_detectability)
from app_paths import DECONVOLUTION_ENABLED
from gui.stages._annotation_search import _measured_columns_sql
from processing.annotation_detectability import fingerprint_contradictions


def _write_annotation_log(metabolites, stats):
    """What the fingerprint gate did, written where it can be read after the
    window closes. A gate that fires on nothing is indistinguishable from one
    that is not wired in, and the console scrolls away."""
    import os
    try:
        from app_paths import user_data_dir
        path = os.path.join(user_data_dir(), 'logs', 'annotation_debug.log')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('ANNOTATION DEBUG LOG\n')
            f.write('=' * 70 + '\n')
            f.write('FINGERPRINT GATE -- "if it were here, would I see the '
                    'big one?"\n')
            f.write(f'  drift (median |dppm| of matched signals): '
                    f'{stats.get("drift_ppm", 0):.4f} ppm; linewidth '
                    f'{stats.get("linewidth_ppm", 0):.4f} ppm; a missing '
                    f'signal is looked for within +/-'
                    f'{stats.get("search_half_width_ppm", 0):.4f} ppm.\n')
            f.write(f'  noise {stats.get("noise", 0):.4g} '
                    f'({stats.get("noise_source", "?")}); excluded regions '
                    f'{stats.get("excluded_regions", [])}\n')
            f.write(f'  {stats.get("metabolites", 0)} metabolites with a '
                    f'detected signal, {stats.get("predicted", 0)} database '
                    f'signals between them:\n')
            for k, label in (('matched', 'matched to a peak'),
                             ('unresolved', 'unresolved multiplet (m), '
                                           'absence not measurable'),
                             ('faint', 'would be faint (expected SNR under '
                                       'the floor)'),
                             ('excluded', 'inside an excluded region'),
                             ('peak_in_window', 'a detected peak sits in the '
                                                'window: overlap or drift, '
                                                'not absence'),
                             ('signal_present', 'something is there, if '
                                                'less than expected'),
                             ('contradicted', 'CONTRADICTED: nothing where '
                                              'an unmistakable signal should '
                                              'be')):
                f.write(f'    {stats.get(k, 0):5d}  {label}\n')
            f.write(f'  -> {stats.get("analytes_marked", 0)} analytes marked '
                    f'Bad.\n\n')
            f.write('MARKED BAD\n')
            for met in metabolites or ():
                c = met.get('_fingerprint_contradictions')
                if not c:
                    continue
                v = met.get('_fingerprint_verdicts') or {}
                f.write(f'  {met.get("name", met.get("id"))}  (its matched '
                        f'signals sit {v.get("own_shift_ppm", 0):+.3f} ppm '
                        f'from the database)\n')
                for x in c:
                    f.write(f'      {x["ppm"]:.3f} ppm {x["multiplicity"]} '
                            f'{x["protons"]:g}H  looked at '
                            f'{x.get("looked_at", x["ppm"]):.3f}: expected '
                            f'SNR {x["expected_snr"]:.0f}, region holds '
                            f'{x["measured_snr"]:.0f}\n')
            f.write('\nEVERY MATCHED SIGNAL, and what annotation made of it\n')
            f.write('  db ppm | mult | dppm | dJ or pattern residual (Hz) | '
                    'lines found/expected | quality | reasons\n')
            for met in sorted(metabolites or (),
                              key=lambda m: str(m.get('name', ''))):
                mps = [mp for mp in (met.get('matched_peaks') or ())
                       if mp.get('peak_rank') is not None]
                if not mps:
                    continue
                f.write(f'  {met.get("name", met.get("id"))}\n')
                for mp in sorted(mps, key=lambda m: float(m.get('db_ppm') or 0)):
                    dj = (mp.get('pattern_residual_hz')
                          if mp.get('pattern_residual_hz') is not None
                          else mp.get('delta_j_hz'))
                    dppm = mp.get('ppm_diff')
                    reasons = '; '.join(str(r)[:70] for r in
                                        (mp.get('_status_reasons') or [])[-2:])
                    _pf = mp.get('pattern_fit') or {}
                    if _pf.get('snr') is not None:
                        reasons = (f'fit SNR {float(_pf["snr"]):.0f}, explains '
                                   f'{100 * float(_pf.get("explained") or 0):.0f}%'
                                   f'; {reasons}')
                    elif mp.get('lines_explained') is not None:
                        reasons = (f'lines explain '
                                   f'{100 * float(mp["lines_explained"]):.0f}%'
                                   f'; {reasons}')
                    dppm_s = f'{dppm:+.3f}' if dppm is not None else '   -  '
                    dj_s = f'{dj:5.2f}' if dj is not None else '  -  '
                    f.write(f'      {float(mp.get("db_ppm") or 0):6.3f} '
                            f'{str(mp.get("type") or mp.get("multiplicity") or "-"):>3} '
                            f'{dppm_s} {dj_s} '
                            f'{str(mp.get("_n_found_lines", "-"))}/'
                            f'{str(mp.get("_n_exp_lines", "-")):<3} '
                            f'{str(mp.get("annotation_quality") or "undecided"):<9} '
                            f'{reasons}\n')
            f.write('\nNOT MARKED, and what stopped each missing signal\n')
            f.write('  (only metabolites with at least one missing signal '
                    'that would have been visible)\n')
            for met in metabolites or ():
                if met.get('_fingerprint_contradictions'):
                    continue
                v = met.get('_fingerprint_verdicts') or {}
                sig = [x for x in (v.get('signals') or ())
                       if x['verdict'] != 'faint']
                if not sig:
                    continue
                f.write(f'  {met.get("name", met.get("id"))}  anchor SNR '
                        f'{v.get("anchor_snr", 0):.0f}, shift '
                        f'{v.get("own_shift_ppm", 0):+.3f}\n')
                for x in sig:
                    held = (f', holds {x["measured_snr"]:.0f}'
                            if x.get('measured_snr') is not None else '')
                    f.write(f'      {x["ppm"]:.3f} ppm {x["multiplicity"]} '
                            f'expected {x["expected_snr"]:.0f}: '
                            f'{x["verdict"]}{held}\n')
    except Exception:
        pass
from processing.annotation_quality import promote_singlets, quality_verdict
from processing.annotation_quality import (
    metabolite_has_anchor, metabolite_tier)
from processing.annotation_scoring import (
    MANUAL_REGION_KEY, analyte_score, apply_mcr_dj, apply_per_sample_consensus,
    coverage_fraction, manual_region,
    resolve_dj, score_signal,
    SCORING_DEFAULTS as _SCORING_DEFAULTS,
    effective_annotation_params,
    peak_shape_score as _peak_shape_score,
    dj_score,
    expected_line_count,
    is_singlet_like,
    annotate_drift_spread,
    measure_multiplet,
    measured_letter,
    pattern_governs,
    pattern_offsets_hz,
    ppm_score,
    PATTERN_BAD_HZ,
    PATTERN_GOLD_HZ,
    PATTERN_MULT,
)


# Shown under the grade checkboxes in both Annotation Parameters dialogs.
_TIER_NOTE = (
    "Only signals with a ticked grade can carry an annotation. Every detected "
    "signal still takes part in measuring a multiplet's lines and its J — a "
    "Poor line of a doublet is still a line of that doublet."
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
    # Two vocabularies reach here: BMRB writes quint/sext/sept, HMDB writes
    # p/h/hept. Both must be understood — a BMRB quintet used to fall through
    # to the flat 'm' branch and be drawn as ONE line, i.e. modelled as a
    # singlet (35 quintets, 9 sextets and 7 septets in BMRB_Metabolomics_v01).
    mt = (mult_type or 'm').lower().strip()
    mapping = {
        's':     [1],
        'd':     [1, 1],
        't':     [1, 2, 1],
        'q':     [1, 3, 3, 1],
        'p':     [1, 4, 6, 4, 1],
        'quint': [1, 4, 6, 4, 1],
        'quin':  [1, 4, 6, 4, 1],
        'h':     [1, 5, 10, 10, 5, 1],
        'sext':  [1, 5, 10, 10, 5, 1],
        'sept':  [1, 6, 15, 20, 15, 6, 1],
        'hept':  [1, 6, 15, 20, 15, 6, 1],
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


def lines_from_transitions(conn, met_id, lo_ppm, hi_ppm, field_mhz=None):
    """Exact multiplet line positions from the DB ``transitions`` table.

    Returns ``[(ppm, rel_intensity)]`` for the metabolite's transitions inside
    ``[lo_ppm, hi_ppm]`` (preferring rows whose ``freq_mhz`` matches ``field_mhz``
    when that column exists), with intensities normalised to the strongest line.
    Empty list when no usable transitions — the caller then falls back to the
    J-modelled ``_build_stick_positions``.

    The DB is read-only at runtime; the query is parameterised.
    """
    if conn is None or met_id is None:
        return []
    lo, hi = (lo_ppm, hi_ppm) if lo_ppm <= hi_ppm else (hi_ppm, lo_ppm)
    try:
        cur = conn.cursor()
        cols = {r[1] for r in cur.execute('PRAGMA table_info(transitions)')}
        if not cols:
            return []
        id_col = 'hmdb_id' if 'hmdb_id' in cols else (
            'bmrb_id' if 'bmrb_id' in cols else None)
        if id_col is None or 'ppm' not in cols:
            return []
        has_int = 'intensity' in cols
        has_freq = 'freq_mhz' in cols
        sel = f"SELECT ppm, {'intensity' if has_int else '1.0'}" \
              f"{', freq_mhz' if has_freq else ''} FROM transitions " \
              f"WHERE {id_col} = ? AND ppm >= ? AND ppm <= ?"
        rows = cur.execute(sel, (str(met_id), lo, hi)).fetchall()
    except Exception:
        return []
    if not rows:
        return []
    # Prefer the frequency closest to the spectrometer when available.
    if has_freq and field_mhz:
        freqs = {r[2] for r in rows if r[2] is not None}
        if freqs:
            best = min(freqs, key=lambda f: abs(float(f) - float(field_mhz)))
            rows = [r for r in rows if r[2] == best]
    lines = [(float(r[0]), float(r[1]) if r[1] is not None else 1.0)
             for r in rows]
    if len(lines) < 1:
        return []
    max_i = max((i for _, i in lines), default=0.0)
    if max_i <= 0:
        max_i = 1.0
    lines = [(p, i / max_i) for p, i in lines]
    lines.sort(key=lambda t: t[0])
    return lines


def _build_stick_positions(ppm_center, mult_type, protons, j_coupling_hz=None,
                           field_mhz=600.0, ppm_span=None):
    """
    Generate individual stick (ppm, relative_intensity) pairs for a multiplet.

    Composite multiplets (dd, ddd, dt, td, tt, dq, tq) use recursive line
    splitting: each coupling successively splits all existing lines.  This
    produces the correct non-uniform spacing when J1 ≠ J2 (e.g. a dd with
    7 Hz and 3.5 Hz gives lines at -5.25, -1.75, +1.75, +5.25 Hz, not the
    wrong uniform -10.5, -3.5, +3.5, +10.5 Hz of the old code).

    Multiple J values are parsed from the j_coupling string (comma-separated).
    When fewer J values are stored than couplings required (e.g. only one J
    for a dd), the last known J is repeated for the remaining splittings.
    """
    mt = (mult_type or 's').lower().strip()

    # Parse ALL J couplings from the string
    j_list_hz = []
    if j_coupling_hz:
        raw = str(j_coupling_hz)
        for tok in raw.replace(';', ',').split(','):
            tok = tok.strip()
            try:
                v = float(tok)
                if v > 0:
                    j_list_hz.append(v)
            except (ValueError, TypeError):
                pass

    # Composite-multiplet patterns: (part_type, …)
    # Each part is a simple coupling ('d', 't', 'q', 'p') applied recursively.
    _COMPOSITE = {
        'dd':  ('d', 'd'),
        'ddd': ('d', 'd', 'd'),
        'dt':  ('d', 't'),
        'td':  ('t', 'd'),
        'tt':  ('t', 't'),
        'dq':  ('d', 'q'),
        'tq':  ('t', 'q'),
    }
    _N_PER = {'d': 2, 't': 3, 'q': 4, 'p': 5}
    _PASCAL = {
        2: [1, 1],
        3: [1, 2, 1],
        4: [1, 3, 3, 1],
        5: [1, 4, 6, 4, 1],
    }

    if mt in _COMPOSITE:
        parts = _COMPOSITE[mt]
        if not j_list_hz:
            return [(ppm_center, 1.0)]

        # Require every coupling needed by the declared composite pattern;
        # never repeat one known J as a fabricated second/third J.
        if len(j_list_hz) < len(parts):
            return [(ppm_center, 1.0)]
        j_padded = list(j_list_hz)

        # Recursive splitting: start with a single line at offset 0
        offsets = [0.0]
        weights = [1.0]
        for part, j_hz_part in zip(parts, j_padded):
            np_ = _N_PER.get(part, 2)
            j_ppm = j_hz_part / max(field_mhz, 1.0)
            coeffs = _PASCAL.get(np_, [1] * np_)
            half = (np_ - 1) / 2.0
            split = [(-half + i) * j_ppm for i in range(np_)]
            new_off, new_w = [], []
            for off, w in zip(offsets, weights):
                for sp, c in zip(split, coeffs):
                    new_off.append(off + sp)
                    new_w.append(w * c)
            offsets, weights = new_off, new_w

        # Sort by offset, normalise weights to sum=1
        pairs = sorted(zip(offsets, weights), key=lambda x: x[0])
        total_w = sum(w for _, w in pairs) or 1.0
        return [(ppm_center + off, w / total_w) for off, w in pairs]

    # ── Simple multiplets (s, d, t, q, p, h, m) ──────────────────────────────
    intensities = _relative_intensities_from_type(mult_type, protons)
    n = len(intensities)
    if n == 1:
        return [(ppm_center, intensities[0])]

    if j_list_hz:
        spacing_ppm = j_list_hz[0] / max(field_mhz, 1.0)
    elif ppm_span and ppm_span > 1e-4 and n > 1:
        spacing_ppm = ppm_span / (n - 1)
    else:
        # No J and no explicit transition span: keep the centre only.
        return [(ppm_center, intensities[0])]

    total_span = spacing_ppm * (n - 1)
    start_ppm  = ppm_center - total_span / 2
    return [(start_ppm + i * spacing_ppm, intensities[i]) for i in range(n)]


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




# NNLS spectral validation is not in use. It reconstructs the whole spectrum
# from every good candidate and solves a non-negative least-squares problem over
# a (spectrum points x metabolites) matrix — on a 65536-point spectrum with a
# hundred candidates that runs on the Tk thread while the window is being built.
# It was commented out on the build path and left live on three others; the flag
# keeps that decision in one place.
NNLS_VALIDATION_ENABLED = False

DPPM_GOLD_DEFAULT = _SCORING_DEFAULTS['dppm_gold']


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
                            scores, confidence class, etc.).
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
        # Defaults come from the scoring core, so the dialog, the search
        # worker and the scoring functions cannot disagree about them.
        # One rule for the window and for quantification
        # (annotation_scoring.effective_annotation_params): the analyst's
        # parameters, then the panel's 'Tolerance (ppm)' -- the only place
        # that number is entered, owning both how far to look for a candidate
        # and how far a signal may sit from the DB and still be accepted.
        _ann_stage = (app.stages.get('annotation')
                      if app and hasattr(app, 'stages') else None)
        _tol_hud = None
        if _ann_stage is not None:
            try:
                _tol_hud = float(_ann_stage.tolerance_var.get())
            except Exception:
                _tol_hud = None
        self._params = effective_annotation_params(
            getattr(_ann_stage, '_annotation_params', None), _tol_hud)

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
                # Every detected peak stays a snap target: line counting and
                # ΔJ measure the spectrum, they do not annotate it. Removing a
                # Poor line here would pair its partner with an unrelated
                # neighbour and report a coupling constant that was never
                # measured. Poor signals are excluded at the anchor instead.
                from processing.peak_detection import canonical_peak_order
                self.peaks = self._expand_peaks_with_subpeaks(
                    canonical_peak_order(peaks))

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

                    _step(9, 'Computing scores…')
                    self._compute_scores()
                    self._enrich_scores()
                    self._auto_mark_good()
                    self._promote_singlets()
                    self._apply_fingerprint_gate()
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
        self._tooltip_win     = None

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
                'ppm_score':       r.get('ppm_score', 0.0),
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
                'mcr_delta_j_hz':  r.get('mcr_delta_j_hz'),
                'mcr_shift_ppm':   r.get('mcr_shift_ppm'),
                'mcr_informative_only': r.get('mcr_informative_only', False),
                'mcr_info_reason': r.get('mcr_info_reason'),
                'mcr_shift_disagree_ppm': r.get('mcr_shift_disagree_ppm'),
                'stocsy_coherence':r.get('stocsy_coherence'),
                # Read by candidate_key as the corroboration criterion. Missing
                # from this list, it would reach the ranking as None and the
                # term would be a silent no-op -- the same failure that left
                # per_sample_consensus wired in and never applied.
                'stocsy_partners': r.get('stocsy_partners'),
                # Per-sample reading, written by the annotation search. The
                # rows travel with the signal: the across-samples view renders
                # them and measures nothing of its own, so what it shows and
                # what the pipeline decided cannot drift apart.
                'per_sample':      r.get('per_sample'),
                # Without this the scoring path's apply_per_sample_consensus
                # reads None and returns None — the samples' own ΔJ written,
                # wired in, and silently never applied.
                'per_sample_consensus': r.get('per_sample_consensus'),
                'intensity_agreement': r.get('intensity_agreement'),
                # What the search measured about the multiplet, and what the
                # reference DB measured about the signal. resolve_dj reads
                # these on the matched peak; missing here, a pattern reached
                # it with nothing to match against and every one came back
                # unmeasured -- after the search had matched them whole.
                '_n_found_lines':  r.get('_n_found_lines'),
                '_n_exp_lines':    r.get('_n_exp_lines'),
                'delta_j_hz':      r.get('delta_j_hz'),
                'observed_j_hz':   r.get('observed_j_hz'),
                'pattern_lines':   r.get('pattern_lines'),
                'pattern_residual_hz': r.get('pattern_residual_hz'),
                'pattern_merge_hz': r.get('pattern_merge_hz'),
                'pattern_floor': r.get('pattern_floor'),
                'pattern_sigma': r.get('pattern_sigma'),
                'pattern_field_mhz': r.get('pattern_field_mhz'),
                'pattern_tolerance_ppm': r.get('pattern_tolerance_ppm'),
                'pattern_fit': r.get('pattern_fit'),
                'lines_explained': r.get('lines_explained'),
                'lines_covary': r.get('lines_covary'),
                'pattern_profile': r.get('pattern_profile'),
                'profile_fwhm_hz': r.get('profile_fwhm_hz'),
                'profile_nh': r.get('profile_nh'),
                'study_linewidth_hz': r.get('study_linewidth_hz'),
                'pattern_centre_offset_ppm': r.get('pattern_centre_offset_ppm'),
                'expected_j_hz': r.get('expected_j_hz'),
                'j_sigma':         r.get('j_sigma'),
                'ppm_center':      r.get('ppm_center', dp_ppm),
                'intensity_n':     r.get('intensity_n'),
                'n_present':       r.get('n_present'),
                'n_included':      r.get('n_included'),
                'ppm_spread':      r.get('ppm_spread'),
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
                    # The same position the search keyed the signal on. With a
                    # measured DB (v02) the search reads COALESCE(ppm_meas,
                    # ppm); every join between a matched signal and its
                    # fingerprint row is round(ppm, 3) == round(ppm_center, 3),
                    # and a fingerprint still listing the curated position
                    # matched none of the 1401 measured signals.
                    _fp_cols = {r[1] for r in cur.execute(
                        'PRAGMA table_info(shifts_1h)')}
                    _fp_meas = _measured_columns_sql(_fp_cols)
                    _fp_pat = ("pattern_lines, pattern_n,"
                               if 'pattern_lines' in _fp_cols else '')
                    _fp_pat += (" pattern_profile, profile_fwhm_hz,"
                                if 'pattern_profile' in _fp_cols else '')
                    _fp_pat += (" profile_nh," if 'profile_nh' in _fp_cols
                                else '')
                    _fp_protons = ("s.protons" if 'protons' in _fp_cols
                                   else "''")
                    cur.execute(
                        f"SELECT {_fp_meas['ppm']} AS ppm_center,"
                        f" {_fp_meas['ppm']} AS ppm_start,"
                        f" {_fp_meas['ppm']} AS ppm_end, {_fp_protons} AS protons,"
                        f" multiplicity AS type, {_fp_meas['j']} AS j_coupling,"
                        f" atom_name, {_fp_pat}"
                        " s.ppm AS ppm_listed"
                        " FROM shifts_1h s WHERE bmrb_id = ?"
                        f" ORDER BY {_fp_meas['ppm']} DESC", (mid,))
                    rows = [dict(r) for r in cur.fetchall()]
                    from processing import spin_patterns as _spin
                    _spin.apply(rows, self._get_field_mhz(), name=met.get('name'))
                    # The search relabels an 'm' whose pattern the DB measured
                    # as a pattern multiplet; the fingerprint must agree, or
                    # the unresolved-multiplet sweep below drops the row and
                    # the found signal has no line to sit on.
                    for dp in rows:
                        _letter = (measured_letter(pattern_offsets_hz(dp))
                                   if dp.get('pattern_lines') else None)
                        if _letter is not None:
                            dp['type'] = _letter[0]
                            dp['j_coupling'] = f'{float(_letter[1]):.2f}'
                        elif pattern_governs(dp.get('type'),
                                             dp.get('pattern_lines')):
                            dp['type'] = PATTERN_MULT
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
                    # ── Local shape score, kept per signal ────────────────
                    # _peak_shape_score is measured in THIS signal's own window,
                    # so it answers "does this multiplet's predicted shape match
                    # what the spectrum shows here". Storing it per matched peak
                    # is what makes it usable to separate two metabolites
                    # competing for the same peak; the metabolite-level value
                    # below is the aggregate of these parts and answers a
                    # different question (is the analyte present at all).
                    #
                    # _s_cosine_local is None whenever the shape could not be
                    # measured. The Δppm/coverage substitutes used for the
                    # aggregate must NOT be stored as if they were shape: they
                    # are Δppm under another name, and a tie-break that silently
                    # falls back to Δppm is no longer orthogonal to it.
                    if not has_spec:
                        mp['_s_cosine_local'] = None
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
                    mp['_s_cosine_local'] = None if ps is None else float(ps)
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
            # Gold-standard NMR metabolomics anchor filter (MSI Level 1):
            # only confirmed-good peaks with Δppm < 0.02 ppm and all multiplet
            # lines found qualify as calibration references.
            if mp.get('annotation_quality') != 'good':
                continue
            obs_ppm = float(mp.get('peak_ppm') or 0)
            db_ppm  = float(mp.get('db_ppm') or mp.get('ppm_center') or 0)
            if obs_ppm == 0 or db_ppm == 0:
                continue
            raw_diff = abs(obs_ppm - db_ppm)
            if raw_diff < 1e-5:
                continue   # already aligned
            if raw_diff >= 0.02:
                continue   # Golotvin high-confidence threshold
            _nf = mp.get('_n_found_lines')
            _ne = mp.get('_n_exp_lines')
            if (_nf is not None and _ne is not None
                    and int(_ne) >= 2 and int(_nf) < int(_ne)):
                continue   # incomplete multiplet — all lines must be present
            # Require the parent metabolite to have sufficient overall quality.
            # A single coincidental "good" peak on a poor overall match is not a
            # reliable calibration reference — it's likely a false positive.
            _met_score = float(met.get('score', 0) or 0)
            if _met_score < 0.65:
                continue
            _n_db = len(met.get('db_peaks', []))
            if _n_db >= 2:
                # Multi-peak metabolites must have ≥2 distinct DB positions confirmed.
                # Prevents anchors where one singlet coincidentally overlaps 1 of 10+ DB peaks.
                _good_db_ppms = {
                    round(float(_mpi.get('db_ppm') or _mpi.get('ppm_center') or 0), 3)
                    for _mpi in met.get('matched_peaks', [])
                    if _mpi.get('annotation_quality') == 'good'
                    and float(_mpi.get('db_ppm') or _mpi.get('ppm_center') or 0) != 0
                }
                if len(_good_db_ppms) < 2:
                    continue
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

        # Recompute scores after ppm_diff is updated, then rebuild assignments.
        # Calibration moved the ppm axis, so the lines the search snapped no
        # longer describe these peaks — they must be measured again.
        self._lines_dirty = True
        self._compute_scores()
        self._enrich_scores()
        self._auto_mark_good()
        self._promote_singlets()
        self._apply_fingerprint_gate()
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
            text=f"{n_pk} signals assigned  ·  {n_met} metabolites",
            font=("Arial", 9, "italic"), foreground="gray")
        self._summary_lbl.pack(anchor=tk.W, padx=4, pady=(4, 2))
        ttk.Label(frame_a,
                  text=("Ctrl+1 = Good  ·  Ctrl+2 = Average"
                        + (" (→ MCR)" if DECONVOLUTION_ENABLED else "")
                        + "  ·  Ctrl+3 = Poor"),
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

        # Confidence filter — the same control the Peak List uses for signal
        # quality, so one scale is read the same way in both windows.
        qf = ttk.Frame(frame_a)
        qf.pack(fill=tk.X, padx=4, pady=(0, 4))
        self._ann_filt_good  = tk.BooleanVar(value=True)
        self._ann_filt_bad   = tk.BooleanVar(value=True)
        self._ann_filt_undef = tk.BooleanVar(value=True)
        # 'm' multiplets are permanently excluded (no J values available)
        self._ann_filt_mult_m = False
        self._conf_filt_cbs = build_tier_filter_row(
            qf,
            {SIGNAL_GOOD:    self._ann_filt_good,
             SIGNAL_AVERAGE: self._ann_filt_undef,
             SIGNAL_POOR:    self._ann_filt_bad},
            self._populate)

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

        self.tree.heading('#0',      text='Signal #',
                          command=lambda: self._sort_by('ppm'))
        self.tree.heading('name',    text='Metabolite',
                          command=lambda: self._sort_by('name'))
        self.tree.heading('ppm',     text='δ (ppm)',
                          command=lambda: self._sort_by('ppm'))
        self.tree.heading('start',   text='Start')
        self.tree.heading('end',     text='End')
        self.tree.heading('score',   text='Score',
                          command=lambda: self._sort_by('score'))
        self.tree.heading('quality', text='Confidence')
        self.tree.heading('overlap', text='Overlap')
        self.tree.column('#0',      width=90,  minwidth=70)
        self.tree.column('name',    width=190, minwidth=130)
        self.tree.column('ppm',     width=80,  minwidth=60, anchor=tk.CENTER)
        self.tree.column('start',   width=78,  minwidth=60, anchor=tk.CENTER)
        self.tree.column('end',     width=78,  minwidth=60, anchor=tk.CENTER)
        self.tree.column('score',   width=65,  minwidth=50, anchor=tk.CENTER)
        self.tree.column('quality', width=70,  minwidth=50, anchor=tk.CENTER)
        self.tree.column('overlap', width=65,  minwidth=50, anchor=tk.CENTER)

        # Metabolite rows are coloured by their confidence tier, not by the raw
        # score: the colour must say the same thing as the Confidence column.
        for _t, _tag in ((SIGNAL_GOOD,    'met_high'),
                         (SIGNAL_AVERAGE, 'met_mid'),
                         (SIGNAL_POOR,    'met_low')):
            self.tree.tag_configure(_tag, background=QUALITY_BG[_t],
                                    foreground=QUALITY_FG[_t])
        # The analyte row carries the analyte score and owns the signals below
        # it; bold separates the two levels at a glance. Only the font is set
        # here so it composes with the tier colour tags above.
        self.tree.tag_configure('met_parent',
                                font=('TkDefaultFont', 9, 'bold'))
        self.tree.tag_configure('peak_match', foreground='#1a1a2e')
        self.tree.tag_configure('mult_info',  foreground='#1a5c2e',
                                background='#e8f5e9', font=('Courier', 8))
        # qual_mid is the Average tier: a plausible match the analyst must
        # still confirm (contested peak, no verdict reached).
        self.tree.tag_configure('qual_good', background=QUALITY_BG[SIGNAL_GOOD],
                                foreground=QUALITY_FG[SIGNAL_GOOD])
        self.tree.tag_configure('qual_mid',  background=QUALITY_BG[SIGNAL_AVERAGE],
                                foreground=QUALITY_FG[SIGNAL_AVERAGE])
        self.tree.tag_configure('qual_bad',  background=QUALITY_BG[SIGNAL_POOR],
                                foreground=QUALITY_FG[SIGNAL_POOR])
        self.tree.tag_configure('sub_peak',          foreground='#444444', background='#f8f8ff')
        self.tree.tag_configure('sub_peak_matched',   foreground='#115511', background='#d6f5d6')
        self.tree.tag_configure('sub_peak_unmatched', foreground='#771111', background='#f5d6d6')
        self.tree.tag_configure('anchor_line', foreground='#0044cc', background='#ddeeff',
                                font=('Courier', 8, 'bold'))
        self._sep_iids: set = set()

        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        self.tree.bind('<Button-3>', self._on_right_click)
        self.tree.bind('<Motion>',   self._on_tree_motion)
        self.tree.bind('<Leave>',    lambda _e: self._hide_tooltip())
        self.win.bind('<Control-Key-1>', lambda e: self._classify('good'))
        self.win.bind('<Control-Key-2>', lambda e: self._classify(None))  # → Average → MCR
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
                  text="Tip: right-drag on zoom view to tell the software where "
                       "this signal is; it re-measures the lines inside",
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
            ('confidence', 'Score', 'Composite score 0–1 (≥0.75 Good · ≤0.50 Poor) — informative'),
            ('j_obs',     'J / ΔJ',      'J obs vs DB (Hz) · right-drag to reselect'),
            ('mcr',       'MCR',         'R²  Pearson r  cross-sample corr — plot shows the resolved component when K≥2 and r≥0.70'),
            ('n_matched', 'Matched',     'peaks found / total — informative'),
            ('coverage',  'Coverage',    '% DB peaks matched — informative'),
            ('cosine',    'Similarity',  'Cosine similarity (0–1) — informative'),
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
        reset_row.grid(row=8, column=0, columnspan=4, sticky=tk.EW, pady=(4, 2))
        self._reset_btn = tk.Button(
            reset_row, text='↺ Reset signals to original',
            font=('Arial', 8), fg='#cc5500', bg='#fff8f0',
            relief='groove', cursor='hand2', command=self._rc_reset)
        self._reset_btn.pack(side=tk.LEFT, padx=4)
        self._reset_modified_lbl = ttk.Label(
            reset_row, text='', font=('Arial', 7, 'italic'), foreground='#cc5500')
        self._reset_modified_lbl.pack(side=tk.LEFT)

        self._current_det_pk_mcr = None  # last det_pk with MCR

        # ── Right-click drag state (region selection in zoom view) ────────────
        # The drag records the analyst's region on the matched peak
        # (``manual_region_ppm``, processing.annotation_scoring); the
        # measurement is redone inside it by the scoring pass.
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
        """Expected SNR per signal — see processing.annotation_detectability."""
        if (self.ppm is None or self.mean_intensity is None
                or len(self.ppm) < 10):
            return
        self._noise_level = mark_detectability(
            self.metabolites, self.ppm, self.mean_intensity)

    def _compute_overlap(self):
        """Which metabolites claim each peak — see annotation_detectability."""
        self._peak_overlap = build_overlap_map(self.metabolites, self.peaks)

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
            if ann is not None:
                getattr(ann, '_manual_anchor_keys', set()).discard(anchor_key)
        else:
            # Set anchor (manual — persists across sync cycles)
            anchor_store[anchor_key] = line_idx
            if mp_ref is not None:
                mp_ref['anchor_line_idx'] = line_idx
            # Track as manual so _sync_anchors_to_peaks won't evict it
            if ann is not None:
                if not hasattr(ann, '_manual_anchor_keys'):
                    ann._manual_anchor_keys = set()
                ann._manual_anchor_keys.add(anchor_key)

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
    def _compute_scores(self, mets=None):
        """
        Compute composite_score for each matched_peak with a hierarchical model
        that mirrors manual spectrum reading (Δppm gate → multiplicity/ΔJ):

        Delegates to processing.annotation_scoring.signal_score — the same
        function the search worker and the rescue path use, so one signal has
        one score. This pass re-runs it with better inputs than the search had:
        the lines have been snapped to detected peaks and J measured from their
        real spacings.

        Non-singlet: ΔJ (w_dj) + Δppm (w_ppm_ns) + local shape (w_cosine_ns),
        renormalised over whichever terms could be measured.
        Singlet: Δppm + shape, capped below the Good band — one line is not an
        identification.

        cosine, coverage (s_found), reverse and STOCSY are informative only and
        do NOT enter the score; MCR is the tiebreaker for overlaps.

        Components:
          s_dj     : J quality for this DB multiplet (non-singlets only)
                       n_found < n_expected → proportional penalty × 0.5
                       n_found == n_expected, ΔJ < dj_gold_hz (0.20) → 1.0
                       n_found == n_expected, ΔJ in [0.20, 0.50) → linear decay 1→0
                       n_found == n_expected, ΔJ ≥ dj_bad_hz (0.50) → 0.0 (Rule 0 also fires)
                       cannot measure → 0.0, excluded from score via _dj_conf=0
          s_cosine : this signal's own shape agreement (_s_cosine_local),
                     falling back to the analyte-level aggregate
          s_ppm    : Gaussian decay on ppm_diff (σ = tolerance/2, clamped 0–1)
          s_found  : fraction of metabolite's DB peaks that have a matched peak

        Thresholds → annotation_quality (if not manually set):
          score > 0.8 → Good (green)
          0.5–0.8     → Average (yellow)
          < 0.5       → Poor / Bad (red)

        ``mets`` restricts the pass to those analytes. A pass over all of
        them re-fits every measured pattern's shape (seconds on a study
        with hundreds); an edit to one signal changes one analyte.
        """
        try:
            ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            _tol = float(getattr(ann_stage, 'tolerance', 0.05) or 0.05)
        except Exception:
            _tol = 0.05
        _sigma_ppm = _tol / 2.0
        _field_sc  = self._get_field_mhz()

        for met in (self.metabolites if mets is None else mets):
            s_found = coverage_fraction(
                met, dj_bad_hz=self._params['dj_bad_hz'])

            # ── cosine ───────────────────────────────────────────────────────
            # Aggregate over the analyte's signals; each matched peak overrides
            # it below with its own local value when the shape was measurable.
            s_cosine_met = float(met.get('cosine', 0) or 0)

            for mp in met['matched_peaks']:
                # ── s_dj ─────────────────────────────────────────────────────
                mult_sc = str(mp.get('type') or mp.get('multiplicity')
                              or 's').lower().strip()
                s_dj, _dj_conf = resolve_dj(
                    mp, met, self.peaks, mult=mult_sc, params=self._params,
                    field_mhz=_field_sc, tolerance=_tol,
                    remeasure=bool(getattr(self, '_lines_dirty', False)),
                    trace=self._mean_trace(),
                    sigma=getattr(self, '_noise_level', None))

                # What the samples measured outranks what the mean spectrum
                # did; MCR still has the last word inside an overlap, where the
                # lines belong to more than one analyte.
                _con = apply_per_sample_consensus(mp, params=self._params)
                if _con is not None:
                    s_dj, _dj_conf = _con

                _mcr = apply_mcr_dj(mp, params=self._params)
                if _mcr is not None:
                    s_dj, _dj_conf = _mcr

                # ── s_ppm ────────────────────────────────────────────────────
                # After the consensus, which may have replaced ppm_diff.
                ppm_diff = float(mp.get('ppm_diff', 0) or 0)
                s_ppm    = ppm_score(ppm_diff, _sigma_ppm)

                score_signal(mp, s_ppm=s_ppm, s_dj=s_dj, dj_conf=_dj_conf,
                             coverage=s_found, params=self._params)

                # This signal's own shape agreement when it could be measured;
                # the analyte-level aggregate only as a fallback.
                _cos_local = mp.get('_s_cosine_local')
                mp['_s_cosine'] = (float(_cos_local) if _cos_local is not None
                                   else s_cosine_met)

            # Analyte level: built from the signal scores, then softened by how
            # much of the expected fingerprint was found. Missing signals are
            # normal (overlap, water suppression, below detection), so the
            # penalty is gentle — a single confirmed multiplet keeps the analyte
            # well inside the Good band.
            if met['matched_peaks']:
                self._set_analyte_score(met)

    def _enrich_scores(self, mets=None):
        """
        Segunda fase de scoring: aplica gate ΔJ para não-singlets.

        Gate Rule 0: não-singlet + ΔJ ≥ dj_bad_hz → score = 0 (Bad)

        ``mets`` restricts the pass as in ``_compute_scores``.
        """
        _scope = self.metabolites if mets is None else mets
        for met in _scope:
            for mp in met['matched_peaks']:
                if is_singlet_like(mp.get('type') or mp.get('multiplicity') or 's'):
                    continue
                _gate = float(mp.get('_score_gate', 1.0))
                if _gate == 0.0:
                    continue
                # ΔJ rejection gate — one place. The score and the quality
                # verdict are two faces of one decision; they used to be applied
                # in two methods, one zeroing the score here and one marking the
                # peak 'bad' in _filter_incompatible_candidates.
                _dj_enr = mp.get('delta_j_hz')
                if (_dj_enr is not None
                        and float(_dj_enr) >= self._params['dj_bad_hz']):
                    mp['composite_score'] = 0.0
                    mp['_score_gate']     = 0.0
                    if mp.get('annotation_quality') is None:   # never a manual mark
                        mp['annotation_quality'] = 'bad'

        # Update metabolite-level score after enrichment
        for met in _scope:
            self._set_analyte_score(met)
        # The stale-line flag has been consumed by a full pass.
        if mets is None:
            self._lines_dirty = False

    def _set_analyte_score(self, met):
        """Recompute the analyte-level score of *met* from its signals.

        Every pass that touches signal scores must call this — taking
        ``max(signal scores)`` instead (as two later passes did) silently drops
        the coverage term and the corroboration lift, leaving the analyte score
        equal to its best signal and the two levels indistinguishable again.
        """
        mps = met.get('matched_peaks') or []
        if not mps:
            return
        sig_vals = [float(mp.get('composite_score', 0) or 0) for mp in mps]
        n_db_sig = len(met.get('db_peaks') or []) or len(sig_vals)
        coverage = min(1.0, len(sig_vals) / max(n_db_sig, 1))
        confirmed = any(mp.get('_confirmed') for mp in mps)
        bd = analyte_score(
            sig_vals, coverage=coverage, params=self._params,
            floor=(self._params['good_score_thresh'] if confirmed else None))
        met['composite_score']  = bd.value
        met['_score_breakdown'] = bd.equation()
    @staticmethod
    def _is_valid_candidate(mp, ppm_tol: float = DPPM_GOLD_DEFAULT) -> bool:
        """Mirrors the validity criteria used in peak_resolve_window.py.

        Valid = not bad  AND  (singlet OR ΔJ < 0.5 Hz)  AND  Δppm < ppm_tol

        ``ppm_tol`` defaults to the dppm_gold parameter, not a literal: it is
        the same gate _promote_singlets and _auto_mark_good apply, and three
        copies of one threshold drift apart.
        Used to filter the overlap / candidate counters so they match what the
        user sees as non-grey rows in the Resolve Peak window.
        """
        _sing = {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs', 'br. s'}
        if mp.get('annotation_quality') == 'bad':
            return False
        mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
        dj   = mp.get('delta_j_hz')
        dj_ok = mult in _sing or dj is None or float(dj) < 0.5
        ppm_diff = float(mp.get('ppm_diff', 999) or 999)
        return dj_ok and ppm_diff < ppm_tol

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

    def _auto_mark_good(self):
        """
        Deriva annotation_quality directamente do composite_score enriquecido.

        O composite_score já incorpora todos os gates e bonuses:
          - Gate Rule 0b (s_found=0) e Rule 0 (ΔJ ≥ 0.5 Hz não-singlet) → score = 0
          - Bonus Rule 2 (gold standard J), Rules 6/6b (RANSY), Rule MCR
          - Penalty Rule 8 (singlet-only sem J), overlap (Rules 3/4/5)
          aplicados em _compute_scores() + _enrich_scores().

        Thresholds:
          score ≥ good_score_thresh → 'good'   (verde)
          score < bad_score_thresh  → 'bad'    (vermelho)
          else                             → None      (amarelo)

        BDL (below detection limit) é tratado como flag independente.
        Marcações manuais (Ctrl+1/2/3) nunca são sobrescritas.
        """
        ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        _qm_am = getattr(ann_stage, '_quality_marks', {}) if ann_stage else {}

        for met in self.metabolites:
            for mp in met['matched_peaks']:
                if (met.get('id'), mp.get('peak_rank')) in _qm_am:
                    continue          # manual mark — never overwritten
                quality, reason = quality_verdict(mp, params=self._params)
                # Reset before assigning: a calibration that moved ppm_diff must
                # be able to clear a 'good' left over from a previous run.
                mp['annotation_quality'] = quality
                if reason:
                    mp.setdefault('_status_reasons', []).append(reason)

    def _promote_singlets(self):
        """Singlet quality — see processing.annotation_quality."""
        ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        promote_singlets(
            self.metabolites, self.peaks, params=self._params,
            manual_marks=(getattr(ann_stage, '_quality_marks', {})
                          if ann_stage else {}))

    def _apply_fingerprint_gate(self):
        """If it were here, would I see the big one? — see
        processing.annotation_detectability.fingerprint_contradictions."""
        if self.ppm is None or self.mean_intensity is None:
            return
        from processing.annotation_detectability import search_half_width
        from processing.quant_predicted_window import project_drift
        from processing.spectral_regions import measure_linewidth_ppm

        from processing.spectral_regions import parse_regions

        stages = getattr(self.app, 'stages', {}) or {}
        ann_stage = stages.get('annotation')
        pd_stage = stages.get('peak_detection')
        # The regions detection was told to ignore. Recorded on the stage by a
        # detection run in this session; otherwise the field, which is what a
        # loaded project's detection was given. Silently using none put the
        # water at 4.65 ppm in as "signal present" for saligenin.
        excluded = getattr(pd_stage, '_excluded_regions', None)
        if excluded is None:
            try:
                excluded = parse_regions(pd_stage.excluded_regions_var.get())
            except (ValueError, AttributeError):
                excluded = []
        noise_std = getattr(pd_stage, 'noise_std', None)
        tol = float(self._params.get('dppm_gold', 0.05))
        drift = project_drift(self.metabolites)
        lw = measure_linewidth_ppm(self.ppm, self.mean_intensity,
                                   self.peaks) or 0.008
        half = search_half_width(tolerance_ppm=tol, drift_ppm=drift,
                                 linewidth_ppm=lw)
        stats: dict = {}
        n = fingerprint_contradictions(
            self.metabolites, self.ppm, self.mean_intensity, self.peaks,
            tolerance_ppm=tol, search_ppm=half,
            excluded_regions=excluded or (),
            manual_marks=(getattr(ann_stage, '_quality_marks', {})
                          if ann_stage else {}),
            stats=stats, noise=noise_std, linewidth_ppm=lw)
        stats['excluded_regions'] = list(excluded or ())
        stats['noise_source'] = ('detection' if noise_std else
                                 'tail estimate (no detection noise)')
        self._fingerprint_stats = {**stats, 'drift_ppm': drift,
                                   'linewidth_ppm': lw}
        _write_annotation_log(self.metabolites, self._fingerprint_stats)
        if n:
            print(f'[annotation] fingerprint contradicted for {n} '
                  f'metabolite(s): a signal that would have been visible is '
                  f'not there; every signal of each marked Bad, reason on '
                  f'the card.')

    def _nnls_spectral_validation(self):
        """
        Global NNLS spectral reconstruction filter. DISABLED — see
        NNLS_VALIDATION_ENABLED.

        Off by default and gated in one place rather than commented out at some
        call sites and live at others, which is how it came to run on three
        paths and not on the one that builds the window.

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
        if not NNLS_VALIDATION_ENABLED:
            return
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

        _sing_types_nnls = {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs', 'br. s'}
        _dj_bad_nnls = float(self._params.get('dj_bad_hz', 0.5))

        for i, met in enumerate(valid_mets):
            amp = float(x[i])
            x_norm = amp / x_max
            # Store amplitude on all matched peaks regardless of quality
            for mp in met['matched_peaks']:
                mp['nnls_amplitude'] = amp
            if x_norm >= NNLS_MIN_FRACTION:
                continue
            # Downgrade 'good' marks (unless manual or J-confirmed multiplet).
            # J-confirmed multiplets (all expected lines found, ΔJ within tolerance)
            # have direct experimental evidence stronger than NNLS reconstruction —
            # the NNLS fit penalises metabolites whose unmatched DB peaks overlap
            # with other components, which is irrelevant for the confirmed signal.
            for mp in met['matched_peaks']:
                if mp.get('annotation_quality') != 'good':
                    continue
                key = (met.get('id'), mp.get('peak_rank'))
                if key in _qm:
                    continue
                mult_nnls = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                if mult_nnls not in _sing_types_nnls:
                    nf_n = mp.get('_n_found_lines')
                    ne_n = mp.get('_n_exp_lines')
                    dj_n = mp.get('delta_j_hz')
                    dj_ok_n = dj_n is None or float(dj_n) < _dj_bad_nnls
                    if (nf_n is not None and ne_n is not None
                            and int(nf_n) >= 2 and int(nf_n) == int(ne_n) and dj_ok_n):
                        continue  # J-confirmed multiplet — NNLS cannot override
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

        self._peak_candidates = {}   # rank -> [(score, met, mp)] sorted desc
        self._peak_assignments = {}  # rank -> (met, mp)
        self._peak_assignment_status = {}
        _ann_for_q = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        _qm_for_q = getattr(_ann_for_q, '_quality_marks', {}) if _ann_for_q else {}
        _manual_for_q = {(str(k[0]), k[1]): v
                         for k, v in (_qm_for_q or {}).items()}
        from gui.stages._annotation_helpers import ambiguous_peak_ranks
        # how far each metabolite's own signals sit from their common
        # drift, for the ranking to read
        annotate_drift_spread(self.metabolites)
        self._ambiguous_peak_ranks = ambiguous_peak_ranks(
            self.metabolites, manual_quality=_manual_for_q)

        for rank, cands in all_cands.items():
            # Sort: J-confirmed match wins over a higher-scoring singlet when both
            # compete for the same peak. Shared key keeps the Peak List in sync.
            from gui.stages._annotation_helpers import _assignment_sort_key
            cands.sort(key=lambda x: _assignment_sort_key(x[0], x[2]),
                       reverse=True)
            self._peak_candidates[rank] = cands
            _has_manual = any(
                _manual_for_q.get((str(m.get('id', m.get('name', ''))), rank))
                == 'good'
                for _, m, _ in cands)
            manual = next(
                ((m, mp2) for _, m, mp2 in cands
                 if _manual_for_q.get(
                     (str(m.get('id', m.get('name', ''))), rank)) == 'good'),
                None)
            self._peak_assignment_status[rank] = (
                'resolved' if (_has_manual or rank not in self._ambiguous_peak_ranks)
                else 'ambiguous')
            self._peak_assignments[rank] = manual if manual else (
                cands[0][1], cands[0][2])

        # Sync annotation_quality from matched_peaks back to the original
        # result dicts in _last_results_args['results'].  This ensures that
        # quantification can read h.get('annotation_quality') directly from
        # the result dict regardless of how auto_quality_marks was built.
        _ann_sync = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        _results_sync = (getattr(_ann_sync, '_last_results_args', None) or {}).get('results', [])
        if _results_sync:
            _aq_map = {}
            for _ms in self.metabolites:
                _mid_s = str(_ms.get('id', ''))
                for _mp_s in _ms.get('matched_peaks', []):
                    _pr_s = _mp_s.get('peak_rank')
                    if _pr_s is not None:
                        _aq_map[(_mid_s, _pr_s)] = _mp_s.get('annotation_quality')
            for _r_s in _results_sync:
                _r_mid = str(_r_s.get('hmdb_id') or _r_s.get('bmrb_id') or _r_s.get('id', ''))
                _r_pr  = _r_s.get('peak_rank')
                _qv    = _aq_map.get((_r_mid, _r_pr))
                if _qv is not None:
                    _r_s['annotation_quality'] = _qv
                elif 'annotation_quality' in _r_s and _r_s['annotation_quality'] != 'bad':
                    # Clear stale Good if quality pipeline downgraded this peak
                    _r_s.pop('annotation_quality', None)

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
            pk.pop('anchor_metabolite', None)

        ann_stage    = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        anchor_store = getattr(ann_stage, '_anchor_lines', {}) if ann_stage else {}

        # Evict stale auto-anchor entries: only manual anchors (right-click) persist
        # across sync cycles. Auto-anchors are rebuilt from scratch each time so that
        # a peak degraded by ΔJ / score changes doesn't remain "⚓ Good" incorrectly.
        if ann_stage is not None:
            _manual_keys = getattr(ann_stage, '_manual_anchor_keys', set())
            for _k in [k for k in list(anchor_store.keys()) if k not in _manual_keys]:
                anchor_store.pop(_k, None)

        def _is_singlet_mp(m):
            t = str(m.get('type') or m.get('multiplicity') or 's').lower().strip()
            return t in ('s', 'singlet', '', 'm', 'multiplet')

        def _mp_j_ok(m):
            """True if this matched-peak qualifies as anchor:
            singlet (no J needed) OR multiplet where ALL predicted lines are
            found AND ΔJ < dj_gold_hz AND Δppm < dppm_gold."""
            if _is_singlet_mp(m):
                return True
            dj      = m.get('delta_j_hz')
            dppm    = m.get('ppm_diff')
            n_found = m.get('_n_found_lines')
            n_exp   = m.get('_n_exp_lines')
            all_found = (n_found is not None and n_exp is not None
                         and int(n_found) >= int(n_exp) and int(n_exp) > 0)
            dj_ok   = dj is not None and float(dj) < self._params['dj_gold_hz']
            dppm_ok = dppm is None or float(dppm) < self._params['dppm_gold']
            return all_found and dj_ok and dppm_ok

        for rank, (met, mp) in self._peak_assignments.items():
            qual = mp.get('annotation_quality')
            if qual == 'bad':
                continue
            pk = pk_by_rank.get(rank)
            if pk is None:
                continue
            if getattr(self, '_peak_assignment_status', {}).get(rank) == 'ambiguous':
                # Candidates remain visible, but no false ownership is published.
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

            if has_manual_anchor:
                pk['is_anchor_line'] = True
            elif this_j_ok:
                if not valid_competitors:
                    # J-confirmed and exclusive → auto-anchor
                    pk['is_anchor_line'] = True
                    _store_anchor_line_idx()
                else:
                    # J-confirmed but disputed → flag for J-Resolve
                    pk['needs_j_resolve'] = True

    def _classify(self, quality):
        """Mark the selected row as Good / Average / Poor (Ctrl+1/2/3).

        Stored internally as 'good' / None / 'bad' — the displayed tiers share
        the Peak List vocabulary, the persisted keys do not change.
        """
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
        """Classify a parent peak-assignment row ('good'/'bad'/None=Average)."""
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

        _tier     = tier_from_annotation_quality(quality)
        qual_mark = QUALITY_LABEL[_tier]
        # Row colour follows the confidence tier, same as the column.
        tags = ({SIGNAL_GOOD: 'met_high', SIGNAL_AVERAGE: 'met_mid',
                 SIGNAL_POOR: 'met_low'}[_tier],)
        cur = list(self.tree.item(iid, 'values'))
        cur[5] = qual_mark
        self.tree.item(iid, values=cur, tags=tags)

    def _classify_child(self, ciid, quality):
        """Classify a child DB-peak row ('good'/'bad'/None=Average)."""
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
            n_cands     = sum(1 for c in self._peak_candidates.get(child_pk, [])
                              if self._is_valid_candidate(c[2])) if child_pk else 0
            if is_matched and n_cands > 1:
                tags = ('peak_match', 'qual_mid')
            elif is_matched:
                tags = ('peak_match', 'qual_good')
            else:
                tags = ('peak_match', 'qual_bad')

        if quality == 'good':
            match_mark = QUALITY_LABEL[SIGNAL_GOOD]
        elif quality == 'bad':
            match_mark = QUALITY_LABEL[SIGNAL_POOR]
        else:
            match_mark = '✓' if tags[1] != 'qual_bad' else '✗'

        cur = list(self.tree.item(ciid, 'values'))
        cur[5] = match_mark
        self.tree.item(ciid, values=cur, tags=tags)

    def _classify_subchild(self, sciid, quality):
        """Classify an individual predicted-line row (sub-child).

        When Good: annotates the linked detected peak in self.peaks with the
        metabolite name (sets 'assigned_metabolite').  Poor/Average removes it.
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
            qual_lbl  = QUALITY_LABEL[SIGNAL_GOOD]
            new_tags = ('sub_peak', 'sub_peak_matched', 'qual_good')
        elif quality == 'bad':
            qual_lbl  = QUALITY_LABEL[SIGNAL_POOR]
            new_tags = ('sub_peak',
                        'sub_peak_matched' if nearest_pk is not None else 'sub_peak_unmatched',
                        'qual_bad')
        else:
            qual_lbl  = f'Signal #{nearest_pk["rank"]}' if nearest_pk else ''
            new_tags = ('sub_peak',
                        'sub_peak_matched' if nearest_pk is not None else 'sub_peak_unmatched')
        if this_is_anchor:
            new_tags = new_tags + ('anchor_line',)

        cur = list(self.tree.item(sciid, 'values'))
        cur[4] = score_col
        cur[5] = qual_lbl
        self.tree.item(sciid, values=cur, tags=new_tags)

    def _refresh_child_match_mark(self, mp_ref):
        """Patch the Quality/Match cell for the child row linked to mp_ref.

        Called after the fingerprint snap writes authoritative _n_found_lines /
        _n_exp_lines back to mp_ref, so the table stays in sync without a full
        tree rebuild.  Skips rows with manually-saved quality marks.
        """
        nf   = int(mp_ref.get('_n_found_lines', 1))
        ne   = int(mp_ref.get('_n_exp_lines',   1))
        dj   = mp_ref.get('delta_j_hz')
        mult = str(mp_ref.get('type') or mp_ref.get('multiplicity') or 's').lower().strip()
        _is_sing  = mult in ('s', 'singlet', '', 'm', 'multiplet')
        _dj_bad   = not _is_sing and dj is not None and float(dj) >= 0.5
        _complete = nf >= ne
        for ciid, (_, _, c_mp) in self._iid_to_child.items():
            if c_mp is not mp_ref:
                continue
            cur = list(self.tree.item(ciid, 'values'))
            mm  = cur[5]
            if mm.startswith((QUALITY_LABEL[SIGNAL_GOOD],
                              QUALITY_LABEL[SIGNAL_POOR])):
                break   # manually saved — never auto-override
            if not _complete:
                cur[5] = f'✗ {nf}/{ne}'
                cur[6] = '—'
                self.tree.item(ciid, values=cur,
                               tags=('peak_match', 'qual_bad'))
            elif _dj_bad:
                cur[5] = f'✗ ΔJ {dj:.2f}Hz'
                cur[6] = '—'
                self.tree.item(ciid, values=cur,
                               tags=('peak_match', 'qual_bad'))
            elif ne > 1:
                cur[5] = f'✓ {nf}/{ne}'
                self.tree.item(ciid, values=cur)
            break

    def _hide_tooltip(self):
        if self._tooltip_win:
            try:
                self._tooltip_win.destroy()
            except Exception:
                pass
            self._tooltip_win = None

    def _show_score_tooltip(self, event, mp):
        """Tooltip for the Score column: the equation behind the value."""
        eq = mp.get('_score_breakdown')
        lines = ['Signal score' if eq else 'Score']
        if eq:
            lines.append(f'  {eq}')
            lines.append('')
            lines.append('  ΔJ    coupling agreement (dj_gold_hz → dj_bad_hz)')
            lines.append('  Δppm  chemical-shift agreement')
            lines.append("  shape cosine of this signal's own window")
        else:
            sc = mp.get('composite_score')
            lines.append(f'  {float(sc):.3f}' if sc is not None else '  n/a')
            lines.append('  (breakdown unavailable — rescore to populate it)')
        self._render_tooltip(event, '\n'.join(lines))

    def _render_tooltip(self, event, text):
        """Place a tooltip window near the cursor with *text*."""
        rx = self.win.winfo_rootx() + event.x + 16
        ry = self.win.winfo_rooty() + event.y + 10
        self._hide_tooltip()
        tw = tk.Toplevel(self.win)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{rx}+{ry}")
        tk.Label(tw, text=text, justify=tk.LEFT,
                 background='#fffde7', relief='solid', borderwidth=1,
                 font=('Courier', 8), padx=4, pady=3).pack()
        self._tooltip_win = tw

    def _on_tree_motion(self, event):
        """Show tooltip with _status_reasons when hovering over the Quality column."""
        iid = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        # #1=name, #2=ppm, #3=start, #4=end, #5=score, #6=quality
        if not iid or col not in ('#5', '#6'):
            self._hide_tooltip()
            return

        # Resolve mp from any row type
        mp = None
        if iid in self._iid_to_child:
            _, _, mp = self._iid_to_child[iid]
        elif iid in self._iid_to_row:
            _, _, mp, _ = self._iid_to_row[iid]

        if mp is None:
            self._hide_tooltip()
            return

        if col == '#5':
            # Score column: show the equation that produced the number, so the
            # value is auditable instead of opaque. Two different scores once
            # coexisted under the same key precisely because nobody could see
            # how either was built.
            self._show_score_tooltip(event, mp)
            return

        reasons = mp.get('_status_reasons') or []
        quality = mp.get('annotation_quality')
        score   = mp.get('composite_score')
        ppm_d   = mp.get('ppm_diff')
        nf      = mp.get('_n_found_lines')
        ne      = mp.get('_n_exp_lines')
        dj      = mp.get('delta_j_hz')

        lines = []
        if quality is not None:
            lines.append(f"Quality: {quality}")
        if score is not None:
            lines.append(f"Score: {float(score):.3f}")
        if ppm_d is not None:
            lines.append(f"Δppm: {float(ppm_d):.4f}")
        if dj is not None:
            lines.append(f"ΔJ: {float(dj):.2f} Hz")
        if nf is not None and ne is not None:
            lines.append(f"Lines: {nf}/{ne}")
        if reasons:
            lines.append('')
            lines.extend(reasons)

        if not lines:
            self._hide_tooltip()
            return

        self._render_tooltip(event, '\n'.join(lines))
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
        if not iid or iid in self._sep_iids:
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
                label=f'📋 Open Signal #{rank} in Signal List',
                command=_open_peak_list)
            if DECONVOLUTION_ENABLED:
                menu.add_command(
                    label=f'🧅 Send Signal #{rank} to Deconvolution Tool',
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
                # build a minimal stub so MCRWindow can proceed
                mp = _mp_ref if _mp_ref is not None else {
                    'db_ppm':    _ppm,
                    'peak_ppm':  _ppm,
                    'peak_rank': _rank,
                    'type':      _dp.get('type', 's'),
                    'protons':   _dp.get('protons', 1),
                    'composite_score': 0.0,
                }
                from gui.widgets.mcr_window import MCRWindow
                MCRWindow(
                    self.win,
                    [{'kind': 'undef', 'met': _met, 'mp': mp, 'peak': peak}],
                    app,
                    ann_stage=ann_stage,
                    ann_results=self)

            menu = tk.Menu(self.win, tearoff=0)
            menu.add_command(
                label=f'DB signal @ {dp_ppm:.4f} ppm  [{dp.get("type","?")}]  —  {met["name"]}',
                state='disabled')
            if cands_raw:
                menu.add_separator()
                menu.add_command(
                    label=f'\u21c6 View Candidates  ({len(candidates)} metabolites)',
                    command=_open_resolve_child)
            def _open_multiplet_samples(_met=met, _dp=dp, _mp_ref=mp_ref,
                                        _rank=child_rank):
                app = getattr(self, 'app', None)
                pd_stage = (getattr(app, 'stages', {}).get('peak_detection')
                            if app else None)
                if pd_stage and _rank is not None:
                    next((p for p in pd_stage.peaks
                          if p.get('rank') == _rank), None)
                mp = _mp_ref if _mp_ref is not None else {
                    'peak_rank': _rank, 'type': _dp.get('type', 's')}
                from gui.widgets.multiplet_sample_view import MultipletSampleView
                MultipletSampleView(self.win, _met, _dp, mp, self.ppm,
                                    getattr(app, 'all_intensities', None),
                                    getattr(app, 'samples_info', []), app)

            menu.add_separator()
            menu.add_command(
                label='\U0001f4ca View multiplet across all samples',
                command=_open_multiplet_samples)
            if DECONVOLUTION_ENABLED:
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
        self._sep_iids = set()

        # Name filter
        name_q = getattr(self, '_flt_name', None)
        q = name_q.get().lower() if name_q else ''

        # Quality filter
        show_good   = getattr(self, '_ann_filt_good',   None)
        show_bad    = getattr(self, '_ann_filt_bad',    None)
        show_undef  = getattr(self, '_ann_filt_undef',  None)
        show_good   = show_good.get()   if show_good   else True
        show_bad    = show_bad.get()    if show_bad    else True
        show_undef  = show_undef.get()  if show_undef  else True
        show_mult_m = False  # 'm' multiplets permanently excluded

        def _is_mult_m(mp):
            t = (mp.get('type', '') or mp.get('multiplicity', '') or '').strip().lower()
            return t in ('m', 'multiplet')


        def _met_visible(met):
            """Hide metabolite entirely if ALL its matched peaks are 'm' and filter is off."""
            if show_mult_m:
                return True
            non_m = [mp2 for mp2 in met['matched_peaks'] if not _is_mult_m(mp2)]
            return len(non_m) > 0

        # ── Collect visible metabolites (one row per metabolite) ─────────────
        ann_stage    = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        anchor_store = getattr(ann_stage, '_anchor_lines', {}) if ann_stage else {}
        _user_tol    = float(getattr(ann_stage, 'tolerance', 0.05) or 0.05)

        # One definition of the parent tier, shared by the row, the filter and
        # the count. They used to be three separate expressions that disagreed.
        _ambig = getattr(self, '_ambiguous_peak_ranks', set())

        def _met_parent_tier(met):
            return metabolite_tier(met, anchor_store=anchor_store,
                                   ambiguous_ranks=_ambig)

        def _met_has_anchor(met):
            return metabolite_has_anchor(met, anchor_store)

        # Everything the non-tier filters allow. The per-tier counts are read
        # from here, never from visible_mets: they label the tier checkboxes,
        # so making them follow those checkboxes meant unticking Average set
        # its own count to (0), with no way to see what re-ticking would bring
        # back.
        _pre_tier_mets = [
            met for met in self.metabolites
            if not (q and q not in met['name'].lower()) and _met_visible(met)
        ]

        _tier_counts = {SIGNAL_GOOD: 0, SIGNAL_AVERAGE: 0, SIGNAL_POOR: 0}
        for met in _pre_tier_mets:
            _tier_counts[_met_parent_tier(met)] += 1

        _tier_shown = {SIGNAL_GOOD: show_good, SIGNAL_POOR: show_bad,
                       SIGNAL_AVERAGE: show_undef}
        visible_mets = [met for met in _pre_tier_mets
                        if _tier_shown.get(_met_parent_tier(met), True)]

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
                text=f"{n_matched}/{n_total_peaks} signals annotated  ·  "
                     f"{n_matched} matched in DB  ·  "
                     f"{len(visible_mets)} metabolites visible")

        self._iid_to_row      = {}
        self._iid_to_child    = {}
        self._iid_to_subchild = {}

        for ri, met in enumerate(visible_mets):
            # ── Aggregate quality / score across all matched peaks ─────────────
            score = float(met.get('composite_score', 0) or 0)

            met_tier  = _met_parent_tier(met)
            qual_mark = ('⚓ Good'
                         if met_tier == SIGNAL_GOOD and _met_has_anchor(met)
                         else QUALITY_LABEL[met_tier])

            # Overlap: matched peaks with ≥1 valid competitor
            n_ovlp = sum(
                1 for mp2 in met['matched_peaks']
                if sum(1 for c in self._peak_candidates.get(mp2.get('peak_rank'), [])
                       if self._is_valid_candidate(c[2])) > 1
            )
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

            # Row colour follows the confidence tier shown in the column.
            tag = {SIGNAL_GOOD: 'met_high', SIGNAL_AVERAGE: 'met_mid',
                   SIGNAL_POOR: 'met_low'}[met_tier]
            tags = (tag, 'met_parent')

            _name_disp = met['name']

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

            if ri % 25 == 0 and ri > 0:
                try:
                    self.win.update_idletasks()
                except Exception:
                    pass

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
            # BDL peaks that WERE detected are kept and shown grey.
            _dp_list = [
                p for p in met['db_peaks']
                if (p.get('type') or '').lower().strip() not in ('m', 'multiplet')
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
                _self_id      = met.get('id')
                n_db_cands    = sum(
                    1 for _, _cm, _cdb, _cmp in self._child_candidates.get(dp_key, [])
                    if _cm.get('id') != _self_id
                    and _cmp is not None
                    and _cmp.get('annotation_quality') != 'bad'
                )
                # child_ovlp computed after match_mark — see below

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
                    # Multiplicity map takes priority — predicted_lines may have fewer
                    # entries than the true line count (e.g. only 2 stored for a quartet).
                    _n_exp_c = _n_map_dp.get(_dp_mult, len(_pred_lines_c) if _pred_lines_c else 2)
                _n_found_c     = 1   # default for singlets / no predicted_lines
                _lines_complete = True

                if is_matched and _pred_lines_c and not _is_singlet_dp:
                    # Prefer values already written by the fingerprint snap (single
                    # source of truth — it sees all predicted lines, not just stored ones).
                    # Always use cached values regardless of whether _cached_ne matches
                    # _n_exp_c: the fingerprint may predict more lines than stored here
                    # (e.g. 36 for a complex quintet vs 2 stored), and that full count
                    # is the authoritative denominator.
                    _cached_nf = mp_ref_child.get('_n_found_lines') if mp_ref_child else None
                    _cached_ne = mp_ref_child.get('_n_exp_lines')   if mp_ref_child else None
                    if _cached_nf is not None and _cached_ne is not None:
                        _n_exp_c       = int(_cached_ne)   # override with authoritative count
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
                _child_fit = ((mp_ref_child or {}).get('pattern_fit') or {}
                              if is_matched else {})
                if child_saved_quality == 'good':
                    child_tags = ('peak_match', 'qual_good')
                    match_mark = QUALITY_LABEL[SIGNAL_GOOD]
                elif child_saved_quality == 'bad':
                    child_tags = ('peak_match', 'qual_bad')
                    match_mark = QUALITY_LABEL[SIGNAL_POOR]
                elif _child_fit.get('snr') is not None:
                    # A multiplet judged as a shape: the fraction of its
                    # region the fitted shape explains, against the
                    # thresholds in the Annotation Parameters.
                    _ex = float(_child_fit.get('explained') or 0.0)
                    _p_good = float(self._params.get('shape_good', 0.7))
                    _p_min = float(self._params.get('shape_min', 0.4))
                    if not _lines_complete or _ex < _p_min:
                        child_tags = ('peak_match', 'qual_bad')
                        match_mark = (f'✗ {100 * _ex:.0f}%'
                                      if _lines_complete else '✗ not found')
                    elif _ex >= _p_good and n_db_cands <= 1:
                        child_tags = ('peak_match', 'qual_good')
                        match_mark = f'✓ {100 * _ex:.0f}%'
                    else:
                        child_tags = ('peak_match', 'qual_mid')
                        match_mark = f'~ {100 * _ex:.0f}%'
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
                    elif n_db_cands > 1:
                        child_tags = ('peak_match', 'qual_mid')
                        match_mark = f'✓ {_n_found_c}/{_n_exp_c}' if _n_exp_c > 1 else '✓'
                    else:
                        child_tags = ('peak_match', 'qual_good')
                        match_mark = f'✓ {_n_found_c}/{_n_exp_c}' if _n_exp_c > 1 else '✓'

                # ── Overlap column — computed after match_mark is finalised ────
                # ⚓ = exclusive (no competitors) — only shown when the peak is
                # actually good. A bad-quality row (✗) shows '—' so the ⚓ symbol
                # is not confused with a J-anchor badge.
                _row_bad = match_mark.startswith('✗')
                child_ovlp = ('⚓'             if n_db_cands == 0 and is_matched
                                                  and not _row_bad
                               else f"⚠ {n_db_cands}" if n_db_cands > 0
                               else '—')

                self._iid_to_child[ciid] = (met, dp, mp_ref_child)
                # Signal score of THIS resonance. The parent shows the analyte
                # score (bold); the children show what each signal contributed
                # to it, so the aggregate is readable as the sum of its parts.
                # Empty for a DB signal that was never matched — it has no score.
                _child_sc = (mp_ref_child or {}).get('composite_score')
                _child_sc_str = ('' if _child_sc is None
                                 else f"{float(_child_sc):.3f}")
                try:
                    self.tree.insert(
                        piid, tk.END, iid=ciid,
                        text=f"  ↳ {dp_ppm:.4f}",
                        values=(
                            f"{dp_type}{protons_str}",
                            f"{dp_ppm:.4f}",
                            f"{dp_start:.4f}",
                            f"{dp_end:.4f}",
                            _child_sc_str,
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
                            and n_db_cands == 0
                            and ann is not None
                            and anchor_store is not None
                            and mp_ref_child is not None
                            and mp_ref_child.get('annotation_quality') != 'bad'):
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
                            qual_lbl  = QUALITY_LABEL[SIGNAL_GOOD]
                        elif saved_sl_q == 'bad':
                            sci_color = ('sub_peak', 'sub_peak_matched', 'qual_bad')
                            qual_lbl  = QUALITY_LABEL[SIGNAL_POOR]
                        else:
                            sci_color = ('sub_peak', 'sub_peak_matched')
                            qual_lbl  = f'Signal #{pk_rank}'
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

        update_tier_counts(getattr(self, '_conf_filt_cbs', {}), _tier_counts)

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
        if iid in self._sep_iids:
            return   # separator row — ignore

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
        # Store the algorithm-original predicted_lines in the mp_ref dict itself
        # (once, on first display) so it survives navigation away and back.
        if mp_ref is not None and '_original_predicted_lines' not in mp_ref:
            mp_ref['_original_predicted_lines'] = list(mp_ref.get('predicted_lines') or [])
        # _orig_pred_lines mirrors the persistent store for the currently shown peak.
        self._orig_pred_lines = (list(mp_ref['_original_predicted_lines'])
                                 if mp_ref is not None else None)

        # ── Overlap: how many metabolites claim the same exp. peak ─────────────
        child_pk_rank = mp_ref.get('peak_rank') if mp_ref else None
        overlap_cands = [c for c in (self._peak_candidates.get(child_pk_rank, [])
                                     if child_pk_rank is not None else [])
                         if self._is_valid_candidate(c[2])]
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

        # A measured pattern is drawn as measured: its lines at their offsets
        # from the reference position, at their relative heights. Letters
        # and couplings built nothing for 'm*' and the pattern showed as one
        # line at the database position while its lines were found 0.02 ppm
        # away.
        _pattern_sticks = pattern_offsets_hz(dp) or (
            pattern_offsets_hz(mp_ref) if mp_ref else [])
        # The shape is the measured one whenever the DB measured it; but
        # the signal is a pattern -- judged on its residual, counted as
        # the study resolves it -- only when the row is one. A triplet
        # whose pattern agrees with its letter is judged on its J.
        _is_pattern_dp = (bool(_pattern_sticks)
                          and _db_mult == PATTERN_MULT)
        if _pattern_sticks:
            # peak_ppm is the centre of the found lines; the reference
            # position sits pattern_centre_offset_ppm below it.
            _origin = centre - float((mp_ref or {}).get(
                'pattern_centre_offset_ppm') or 0.0)
            _db_sticks_raw = [(_origin + off / _fmhz, rel)
                              for off, rel in _pattern_sticks]
            _coupling_letters = []
        else:
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
        if _pattern_sticks:
            plot_sticks = _db_sticks_raw
        elif mp_ref and not is_singlet_dp:
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
        _lw_hz_plot = (mp_ref or {}).get('study_linewidth_hz')
        if _pattern_sticks and _lw_hz_plot:
            # drawn at the study's own linewidth, so lines this study
            # cannot resolve merge on the curve as they do on the spectrum
            hwhm = max(float(_lw_hz_plot) / _fmhz / 2.0, 0.0005)

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

        # ── MCR-resolved component ───────────────────────────────────────────
        # A peak whose score came from MCR was measured on the deconvolved
        # component, not on the merged signal. Show the same thing the decision
        # was made on: raw fades to context, the resolved component is drawn.
        # Sub-peaks saved from the Deconvolution Tool take precedence — those
        # are the analyst's own explicit decision.
        _mcr_component = None
        if not _use_deconv_display and mp_ref is not None:
            _mcr_c = self._mcr_component_for(mp_ref, met)
            if _mcr_c is not None:
                _mx, _my = _mcr_c
                _mk = (_mx >= lo) & (_mx <= hi)
                if _mk.sum() >= 4 and float(np.max(_my[_mk])) > 0:
                    _mx, _my = _mx[_mk], _my[_mk].copy()
                    # A component cannot be taller than the merged signal it was
                    # separated out of. If the concentration scaling landed in
                    # different units from the plotted trace, fall back to the
                    # raw envelope so the two stay comparable on screen.
                    _raw_pk = float(np.max(y_exp)) if y_exp.size else 0.0
                    _cmp_pk = float(np.max(_my))
                    if _raw_pk > 0 and not (_raw_pk * 0.01 <= _cmp_pk <= _raw_pk * 1.5):
                        _my = _my / _cmp_pk * _raw_pk
                    _mcr_component = (_mx, _my)
                    _deconv_components.append(_mcr_component)
                    _use_deconv_display = True
                    x_exp = _mx
                    y_exp = _my.copy()

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
                    # The MCR component is this metabolite's own resolved signal
                    # and the thing its score was measured on — draw it solid and
                    # name it, rather than as one anonymous component of a set.
                    _is_mcr = (_mcr_component is not None
                               and _yc is _mcr_component[1])
                    if _is_mcr:
                        _cc = '#7b1fa2'
                    self.ax.fill_between(_xc, 0, _yc_n,
                                         color=_cc, alpha=0.18, zorder=2)
                    self.ax.plot(_xc, _yc_n,
                                 color=_cc,
                                 linewidth=1.6 if _is_mcr else 0.9,
                                 linestyle='-',
                                 alpha=0.95 if _is_mcr else 0.70, zorder=3,
                                 label='MCR component' if _is_mcr else None)
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
            _min_sep = (min(abs(_pl_ppms_raw[k+1] - _pl_ppms_raw[k])
                            for k in range(len(_pl_ppms_raw) - 1))
                        if len(_pl_ppms_raw) >= 2 else 0.0)
            _half_span_p = (max(_pl_ppms_raw) - _pred_centre)
            if len(_pl_ppms_raw) >= 2 and abs(_alignment_shift) > _half_span_p + _min_sep * 0.5:
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
        if _det_found and not is_singlet_dp and not _pattern_sticks:
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
        _shape_drawn = False
        if _is_pattern_dp and mp_ref and _lw_hz_plot:
            # the shape as it was fitted: the reference's profile where the
            # database holds one, its lines where not, broadened to the
            # study's width and placed where the reference position fell
            from processing.pattern_shape import (
                lines_model, parse_profile, profile_model)
            _origin_p = centre - float(mp_ref.get('pattern_centre_offset_ppm')
                                       or 0.0)
            _prof = parse_profile(mp_ref.get('pattern_profile'))
            if _prof is not None:
                _extra = max(0.0, float(_lw_hz_plot)
                             - float(mp_ref.get('profile_fwhm_hz') or 0.0))
                y_db = profile_model(x_db, _prof, _origin_p, _extra, _fmhz)
                _shape_drawn = True
            elif _pattern_sticks:
                y_db = lines_model(x_db, _pattern_sticks, _origin_p,
                                   float(_lw_hz_plot), _fmhz)
                _shape_drawn = True
        if not _shape_drawn:
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
            # A pattern fitted as a shape is drawn at the height it fitted:
            # the largest copy that sits under the trace. Where it reaches
            # the trace the pattern is alone; where the trace stands over it
            # something else shares the region.
            _pf = (mp_ref or {}).get('pattern_fit') or {}
            _tr = self._mean_trace() if _is_pattern_dp else None
            if _pf.get('amplitude') and _tr is not None and len(x_exp):
                # the fit is in the study mean's units; the panel shows the
                # best group's mean scaled to 1 -- the pattern is drawn at
                # the fraction of the mean's height it accounted for
                _tm = (_tr[0] >= min(stick_ppms_plot) - hwhm * 4) & (
                    _tr[0] <= max(stick_ppms_plot) + hwhm * 4)
                _mean_top = float(_tr[1][_tm].max()) if _tm.any() else 0.0
                if _mean_top > 0:
                    y_db *= min(1.5, float(_pf['amplitude']) / _mean_top)

        # ── What the Pearson r compares ─────────────────────────────────
        # MCR aligns the DB profile onto the signal before correlating, and this
        # panel already draws the DB curve at the observed position rather than
        # at the nominal DB ppm. The aligned profile and the drawn DB curve are
        # therefore the same curve — the component and its reference are already
        # superimposed, and adding a shifted copy only put a third curve an
        # extra offset away from both. The shift is reported as a number, on the
        # DB curve's own label and in the Match Quality card.
        _mcr_shift = (mp_ref.get('mcr_shift_ppm') if mp_ref is not None else None)
        _mcr_lbl_extra = ''
        if _use_deconv_display and _mcr_component is not None:
            _r_val = mp_ref.get('mcr_pearson_r') if mp_ref is not None else None
            _bits = []
            if _mcr_shift is not None and abs(float(_mcr_shift)) > 1e-4:
                # In ppm only. Hz is reserved for couplings here: a Hz figure
                # sitting beside the "J = 5.25 Hz" box reads as a coupling
                # constant of the resolved component, and an alignment distance
                # is not one.
                _sh = float(_mcr_shift)
                _bits.append(f'aligned {_sh:+.4f} ppm')
            if _r_val is not None:
                _bits.append(f'shape r={float(_r_val):.2f}')
            if _bits:
                _mcr_lbl_extra = '  — MCR: ' + ' · '.join(_bits)

        db_color = '#1a7a1a' if is_matched else '#cc2200'
        db_fill  = '#44bb44' if is_matched else '#ee4444'
        lbl      = ('DB matched' if is_matched else 'DB — not found')
        lbl      = lbl + _mcr_lbl_extra
        self.ax.fill_between(x_db, 0, y_db,
                             color=db_fill, alpha=0.22, zorder=3)
        self.ax.plot(x_db, y_db,
                     color=db_color, linewidth=1.5, linestyle='--',
                     zorder=4, label=lbl)

        # Individual sticks (highlight the multiplet lines). A shape has no
        # lines to point at: the curve is the whole of it.
        _db_scale = float(y_db.max()) if len(y_db) > 0 and y_db.max() > 0 else 1.0
        # A line the measurement placed but found no apex under (the comb
        # laid on an incomplete region) is drawn hollow and dotted: the
        # fingerprint says where it should be, the spectrum has not said
        # it is there.
        _unmeasured = {round(float(l['ppm']), 5)
                       for l in ((mp_ref or {}).get('predicted_lines') or [])
                       if l.get('measured') is False}
        for st_ppm, st_rel in ([] if _is_pattern_dp else plot_sticks):
            _sh = st_rel * _db_scale * 0.95
            _hollow = round(float(st_ppm), 5) in _unmeasured
            self.ax.plot([st_ppm, st_ppm], [0, _sh],
                         color=db_color, linewidth=2.2,
                         alpha=0.45 if _hollow else 0.85,
                         linestyle=':' if _hollow else '-', zorder=5)
            self.ax.scatter([st_ppm], [_sh],
                            facecolors='none' if _hollow else db_color,
                            edgecolors=db_color, s=45, zorder=6)

        # ── Highlight span around the multiplet ───────────────────────────────
        span_lo = min(s[0] for s in plot_sticks) - hwhm * 2.5
        span_hi = max(s[0] for s in plot_sticks) + hwhm * 2.5
        span_color = '#00cc44' if is_matched else '#ffaa00'
        self.ax.axvspan(span_hi, span_lo,   # NMR inverted axis
                        alpha=0.07, color=span_color, zorder=0)
        # The analyst's region, when one confines the measurement.
        _region = manual_region(mp_ref) if mp_ref is not None else None
        if _region is not None:
            self.ax.axvspan(_region[1], _region[0], alpha=0.12,
                            color='#ff9900', zorder=0,
                            label='analyst region')
            for _edge in _region:
                self.ax.axvline(_edge, color='#ff9900', lw=0.8, ls=':',
                                zorder=1)

        # ── Mark detected experimental peaks near DB sticks + J from them ───────
        # For each DB stick, find the closest detected peak within a tight radius
        # (hwhm * 3 or tolerance). This avoids picking up unrelated peaks in the
        # broad display window. Each detected peak is used at most once.
        dp_type_j     = (dp.get('type') or '').lower().strip()
        is_singlet_dp = dp_type_j in ('s', 'singlet', '')
        _n_lines_map_plot = {'s':1,'singlet':1,'d':2,'doublet':2,'t':3,'triplet':3,
                             'q':4,'quartet':4,'dd':4,'dt':6,'td':6,'tt':9,'ddd':8}
        n_expected = _n_lines_map_plot.get(dp_type_j, len(plot_sticks))
        if mp_ref and mp_ref.get('_n_exp_lines') is not None:
            # the scorer's count, the one the election and quantification read
            n_expected = int(mp_ref['_n_exp_lines'])

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
            for _pk in ([] if _is_pattern_dp else (self.peaks or [])):
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
            # A shape has no lines to point at.
            for dp_pk in ([] if _is_pattern_dp else _det_found):
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

            if _is_pattern_dp and mp_ref:
                # A multiplet judged as a shape: here or not (the fit's
                # SNR against detection's floor), alone or shared (the
                # fraction of the trace over its apices it explains).
                _pf = mp_ref.get('pattern_fit') or {}
                _res = mp_ref.get('pattern_residual_hz')
                _here = bool(mp_ref.get('_n_found_lines'))
                if _pf.get('snr') is not None:
                    _ex = float(_pf.get('explained') or 0.0)
                    if _here:
                        j_card = (f'shape fits  ·  SNR {float(_pf["snr"]):.0f}'
                                  f'  ·  explains {100 * _ex:.0f}% of the region')
                        j_color = ('#1a7a1a' if _res is not None
                                   and float(_res) <= PATTERN_GOLD_HZ
                                   else '#7a6000')
                    else:
                        j_card = (f'shape not found  ·  SNR '
                                  f'{float(_pf["snr"]):.1f} under the floor')
                        j_color = '#9a0000'
                    _ann_text = j_card
                else:
                    _nf = mp_ref.get('_n_found_lines')
                    _ne = mp_ref.get('_n_exp_lines')
                    j_card = (f'pattern on the peak list: {_nf}/{_ne} lines'
                              + (f', residual {float(_res):.2f} Hz'
                                 if _res is not None else ''))
                    j_color = '#7a6000' if _here else '#9a0000'
                    _ann_text = j_card
                self.ax.text(0.02, 0.04, _ann_text,
                             transform=self.ax.transAxes,
                             fontsize=7, color=j_color, va='bottom',
                             bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='white',
                                       edgecolor=j_color, alpha=0.85))
                self._set_q('j_obs', j_card, j_color)

            elif not is_singlet_dp:
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
                elif n_found >= 2 and not (mp_ref and mp_ref.get('_n_exp_lines') is not None):
                    # Fallback for results annotated before this fix (no stored J).
                    # Never where the scorer measured: J unset there means
                    # the lines were not all found, and a J of the view's
                    # own is not the scorer's.
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
                # A singlet carries no coupling constant, so there is
                # nothing to validate here.
                j_card, j_color = '', '#666666'
                j_card, j_color = '— singlet', '#888888'
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

            # Colour: green within annotation tolerance, red outside
            try:
                _ann_s_fp  = (getattr(self.app, 'stages', {}) or {}).get('annotation')
                _tvar_fp   = getattr(_ann_s_fp, 'tolerance_var', None) if _ann_s_fp else None
                _tol_fp    = float(_tvar_fp.get()) if _tvar_fp is not None else 0.05
            except Exception:
                _tol_fp = 0.05
            _dppm_color  = '#1a7a1a' if ppm_diff <= _tol_fp else '#cc1100'
            _dppm_edgecolor = '#0d4d0d' if ppm_diff <= _tol_fp else '#880000'

            # Both endpoints must be within the visible axes [0,1].
            # If the DB position is outside the window, show text-only Δppm label.
            _ax_margin = 0.02
            _both_visible = (_ax_margin <= x_exp_ax <= 1 - _ax_margin and
                             _ax_margin <= x_db_ax  <= 1 - _ax_margin)
            if _both_visible:
                self.ax.annotate(
                    '', xy=(x_exp_ax, y_arr_ax), xytext=(x_db_ax, y_arr_ax),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='<->', color=_dppm_color, lw=1.2))
                self.ax.text((x_exp_ax + x_db_ax) / 2, y_text_ax,
                             f'Δ {ppm_diff:.4f} ppm',
                             transform=self.ax.transAxes,
                             ha='center', fontsize=8, color=_dppm_color)
            else:
                # DB position outside window — show Δppm as text with directional arrow
                _dir = '→' if dp_ppm < centre else '←'
                self.ax.text(0.98, y_text_ax,
                             f'{_dir} Δ {ppm_diff:.4f} ppm',
                             transform=self.ax.transAxes,
                             ha='right', va='center', fontsize=8, color=_dppm_color,
                             bbox=dict(boxstyle='round,pad=0.15',
                                       facecolor='#fff0f0' if ppm_diff > _tol_fp else '#f0fff0',
                                       edgecolor=_dppm_edgecolor,
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
        _sc_z = float((mp_ref.get('composite_score') if mp_ref else None)
                      or met.get('composite_score', 0) or 0)
        self._set_q('confidence', f"{_sc_z:.3f}",
                    '#1a7a1a' if _sc_z >= self._params['good_score_thresh']
                    else '#9a0000' if _sc_z <= self._params['bad_score_thresh']
                    else '#7a6000')
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
            _lines_txt = f"{n_found}/{n_expected} lines detected"
            _lines_col = ('#8B6914' if n_overlap > 0
                          else '#1a7a1a' if n_found >= n_expected
                          else '#7a6000' if n_found > 0 else '#9a0000')
        else:
            _lines_txt = f"{'✓ matched' if is_matched else '✗ missing'}"
            _lines_col = ('#8B6914' if (is_matched and n_overlap > 0)
                          else '#1a7a1a' if is_matched else '#9a0000')
        self._set_q('n_matched', f"{_lines_txt}{ovlp_card}", _lines_col)
        # A view never writes into the annotation. This used to cache its own
        # snap here as _n_found_lines / _n_exp_lines -- the view's count over
        # the scorer's measured one -- and quantification changed with the
        # click.
        if mp_ref is not None and n_sticks > 1 and mp_ref.get('_n_exp_lines') is not None:
            self._refresh_child_match_mark(mp_ref)
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
            if mp_ref.get('mcr_informative_only'):
                # The numbers describe the fit, not the identity: either the
                # profile is a singlet (shape carries nothing after alignment)
                # or MCR resolved a different signal from the one matched here.
                _sym_pk, _color_pk = '—', '#555577'
                _parts_pk.append(mp_ref.get('mcr_info_reason')
                                 or 'informative only')
            else:
                _sym_pk   = '✓' if _conf_pk is True else ('✗' if _conf_pk is False else '?')
                _color_pk = ('#1a7a1a' if _conf_pk is True
                             else '#9a0000' if _conf_pk is False else '#7a6000')
            _shift_pk = mp_ref.get('mcr_shift_ppm')
            if _shift_pk is not None and abs(float(_shift_pk)) > 1e-4:
                _parts_pk.append(f'align {float(_shift_pk):+.4f} ppm')
            self._set_q('mcr', f"{_sym_pk}  {' · '.join(_parts_pk)}", _color_pk)
        else:
            self._set_q('mcr', '— not run', '#888888')

        # Modified indicator — the analyst's region confines this signal.
        self._show_reset_btn(mp_ref is not None
                             and manual_region(mp_ref) is not None)

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

    def _mean_trace(self):
        """``(ppm, mean intensity)`` of the study, the trace the search
        measured on; None without a spectrum."""
        ppm = getattr(self, 'ppm', None)
        raw = getattr(self, 'all_intensities', None)
        if ppm is None or raw is None:
            return None
        cached = getattr(self, '_mean_trace_cache', None)
        if cached is not None and cached[0] is raw:
            return cached[1]
        arr = np.asarray(raw, dtype=float)
        y = arr.mean(axis=1) if arr.ndim == 2 else arr
        out = (np.asarray(ppm, dtype=float), y)
        self._mean_trace_cache = (raw, out)
        return out

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


    # ── End helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_fill_colors(score: float) -> tuple:
        """Mapeamento score (0-1) → (fill_color, line_color) para fingerprint plot."""
        if score >= 0.80:
            return '#44bb44', '#1a7a1a'   # green: good annotation
        if score >= 0.50:
            return '#ddaa00', '#8b6000'   # amber: intermediate annotation
        return '#ee6644', '#aa2200'       # red-orange: poor annotation

    def _mcr_component_for(self, mp, met=None):
        """This candidate's MCR component, when MCR genuinely resolved one.

        Returns ``(ppm_region, contribution)`` or None. The contribution is in
        data units, not the unit-normalised S column, so it can be drawn against
        the experimental trace on the same scale.

        ``met`` supplies the compound identity. A matched_peak carries only its
        positions and scores — ``_group_by_metabolite`` keeps the HMDB/BMRB
        accession on the metabolite, not on each of its peaks — so without it
        the candidate lookup can never match and no component is ever found.

        The gate is the same one the scoring uses to trust MCR's J over the raw
        measurement (MCR_MIN_OVERLAP_K / MCR_MIN_SHAPE_R): there must have been a
        real overlap to separate, and this component's shape must match this
        candidate's profile. Deliberately NOT gated on ``mcr_confirmed`` — that
        is a region-level verdict on the global R², and a component can be well
        resolved for this candidate while the region fit falls short of it.
        Showing it then is the point: the analyst sees what the score was
        measured on, and the Match Quality card still reports the verdict.
        """
        rank = mp.get('peak_rank')
        if rank is None:
            return None
        pk = next((p for p in (self.peaks or []) if p.get('rank') == rank), None)
        ann = (pk or {}).get('mcr_annotation') or {}
        S       = ann.get('S')
        ppm_reg = ann.get('ppm_region')
        assigns = ann.get('assignments') or []
        cands   = ann.get('candidates') or []
        if S is None or ppm_reg is None or not len(assigns) or not cands:
            return None
        try:
            if int(ann.get('K') or 0) < MCR_MIN_OVERLAP_K:
                return None      # single component — nothing was deconvolved
        except (TypeError, ValueError):
            return None

        def _ident(d):
            return str(d.get('hmdb_id') or d.get('bmrb_id')
                       or d.get('id') or d.get('name') or '')

        want_id = _ident(met or {}) or _ident(mp)
        if not want_id:
            return None
        want_ppm = round(float(mp.get('db_ppm') or 0), 3)
        same_met = [i for i, c in enumerate(cands) if _ident(c) == want_id]
        if not same_met:
            return None
        # A metabolite can contribute more than one DB peak to the same region,
        # so prefer the candidate for this exact peak and fall back to the
        # compound's only entry.
        cand_idx = next(
            (i for i in same_met
             if round(float(cands[i].get('db_ppm') or 0), 3) == want_ppm),
            same_met[0] if len(same_met) == 1 else None)
        if cand_idx is None:
            return None
        # assignments is per component: entry k describes component k
        comp = next((k for k, a in enumerate(assigns)
                     if a is not None and len(a) and a[0] == cand_idx), None)
        if comp is None:
            return None

        _a = assigns[comp]
        shape_r = (_a[1] if len(_a) > 1 and _a[1] is not None
                   else mp.get('mcr_pearson_r'))
        if shape_r is None or float(shape_r) < MCR_MIN_SHAPE_R:
            return None          # component does not look like this compound

        try:
            S = np.asarray(S, dtype=float)
            if comp >= S.shape[1]:
                return None
            profile = S[:, comp]
            # S columns are unit-normalised, so their height is arbitrary. The
            # concentration matrix restores data units: contribution of
            # component k to a sample is C[sample, k] * S[:, k].
            C = ann.get('C')
            scale = 1.0
            if C is not None:
                C = np.asarray(C, dtype=float)
                if C.ndim == 2 and comp < C.shape[1]:
                    scale = float(np.mean(C[:, comp]))
                elif C.ndim == 1 and comp < C.size:
                    scale = float(C[comp])
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            return np.asarray(ppm_reg, dtype=float), profile * scale
        except Exception:
            return None

    def _plot_fingerprint(self, met):
        # Switching to fingerprint view — clear per-peak zoom state
        self._current_dp      = None
        self._current_met_dp  = None
        self._current_mp_ref  = None
        self._orig_pred_lines = None
        self._show_reset_btn(False)
        self._set_q('j_obs',    '— select a DB peak row', '#888888')

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
                # Where MCR resolved the overlap, the raw trace is the merged
                # signal — kept for context but dimmed, with the resolved
                # component drawn on top.
                _mcr_comp = self._mcr_component_for(mp, met)
                _exp_a = 0.30 if _mcr_comp is not None else 1.0
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
                                         alpha=(0.35 if is_best else 0.18) * _exp_a,
                                         zorder=1)
                        if ys and g not in _legend_done:
                            ymean_g = np.mean(ys, axis=0)
                            self.ax.plot(x_seg, ymean_g / gmax, color=col,
                                         lw=1.8 if is_best else 1.0,
                                         alpha=(0.95 if is_best else 0.50) * _exp_a,
                                         zorder=2,
                                         label=g if is_best else f'{g} (dim)')
                            _legend_done.add(g)
                        elif ys:
                            ymean_g = np.mean(ys, axis=0)
                            self.ax.plot(x_seg, ymean_g / gmax, color=col,
                                         lw=1.8 if is_best else 1.0,
                                         alpha=(0.95 if is_best else 0.50) * _exp_a,
                                         zorder=2)
                else:
                    y_n = y_mean_seg / gmax
                    self.ax.fill_between(x_seg, 0, y_n, color='#aaaaaa',
                                         alpha=0.35 * _exp_a, zorder=1)
                    self.ax.plot(x_seg, y_n, color='#444444',
                                 lw=0.8, alpha=_exp_a, zorder=2,
                                 label='Experimental')

                # ── MCR-resolved component ────────────────────────────────
                if _mcr_comp is not None:
                    _mx, _my = _mcr_comp
                    _in = (_mx >= float(min(x_seg))) & (_mx <= float(max(x_seg)))
                    if _in.sum() >= 2 and float(np.max(_my[_in])) > 0:
                        # Scale to the experimental trace so the shapes are
                        # comparable; MCR components carry arbitrary units.
                        _ref = (float(np.max(y_mean_seg))
                                if float(np.max(y_mean_seg)) > 0 else 1.0)
                        _my_n = (_my[_in] / float(np.max(_my[_in]))
                                 * (_ref / gmax))
                        _lbl = ('MCR component'
                                if 'MCR component' not in _legend_done else '')
                        self.ax.fill_between(_mx[_in], 0, _my_n,
                                             color='#7b1fa2', alpha=0.18,
                                             zorder=3)
                        self.ax.plot(_mx[_in], _my_n, color='#7b1fa2',
                                     lw=1.6, zorder=4, label=_lbl)
                        if _lbl:
                            _legend_done.add('MCR component')

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

        # ── Legend — collect all labeled artists automatically ────────────
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        _base_handles = [
            Patch(facecolor='#aaaaaa', alpha=0.35, label='Experimental'),
            Line2D([0], [0], color='#1a7a1a', lw=2, label='DB matched'),
            Line2D([0], [0], color='#cc2200', lw=1.2,
                   alpha=0.5, label='DB unmatched'),
        ]
        _auto_h, _auto_l = self.ax.get_legend_handles_labels()
        _base_lbls = {'Experimental', 'DB matched', 'DB unmatched'}
        _extra = [(h, l) for h, l in zip(_auto_h, _auto_l)
                  if l not in _base_lbls]
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

        _sc = float(met.get('composite_score', 0) or 0)
        self._set_q('confidence', f"{_sc:.3f}",
                    '#1a7a1a' if _sc >= self._params['good_score_thresh']
                    else '#9a0000' if _sc <= self._params['bad_score_thresh']
                    else '#7a6000')
        self._set_q('coverage',  f"{cov:.1f}%",
                    _col(cov, 70, 40))
        self._set_q('cosine',    f"{cos:.4f}",
                    _col(cos, 0.80, 0.50))
        self._set_q('n_matched', f"{n_mat}/{n_db} lines",
                    _col(n_mat / float(n_db or 1), 0.99, 0.50))

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

        self.match_info.config(
            text=(f"{met['name']}  ·  {n_mat}/{n_db} DB signals matched  "
                  f"·  {cov:.0f}% coverage  ·  cosine {cos:.3f}"))


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
                    text='Select a signal in the list first, then right-drag.',
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
                self._reset_modified_lbl.config(
                    text='● measured inside the analyst\'s region')
            else:
                self._reset_modified_lbl.config(text='')
        except Exception:
            pass




    def _recompute_and_refresh(self):
        """Full scoring pipeline + UI refresh. Called after parameter changes."""
        try:
            # A parameter change can move the line-matching tolerance, so the
            # measurement has to be redone rather than read back.
            self._lines_dirty = True
            self._compute_scores()
            self._enrich_scores()
            self._auto_mark_good()
            self._promote_singlets()
            self._nnls_spectral_validation()
            self._build_peak_assignments()
            self._populate()
        except Exception:
            pass

    def _rescore_after_edit(self, met):
        """Re-score *met* after the analyst edited one of its signals.
        Updates mp_ref fields, annotation_quality, anchors, and refreshes tree rows.
        Restores the treeview selection to the same DB peak so the view doesn't reset.

        Only *met* is re-scored: the analyst's region changes one signal of
        one analyte, and a pass over every analyte re-fits every pattern's
        shape -- six seconds on a study of 260 analytes, per drag."""
        # Save current context before repopulation wipes the tree
        _saved_met_id  = met.get('id') if met else None
        _saved_dp      = self._current_dp
        _saved_dp_ppm  = (float(_saved_dp.get('ppm_center', _saved_dp.get('ppm', 0)))
                          if _saved_dp is not None else None)
        try:
            self._compute_scores([met])
            self._enrich_scores([met])
            self._auto_mark_good()
            self._promote_singlets()
            self._nnls_spectral_validation()
            self._build_peak_assignments()
            self._populate()
        except Exception as _e:
            # The window's other passes swallow errors while building rows;
            # a broken measurement here must reach the analyst, not vanish.
            import traceback
            traceback.print_exc()
            try:
                self.match_info.config(
                    text=f'Re-scoring failed: {_e}', foreground='red')
            except Exception:
                pass
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
        """Record the analyst's region on the signal and re-measure it there.

        The window stores nothing but the interval: which peaks are the
        lines, their spacing, the coupling, the position and the score all
        come from the scoring pass (``resolve_dj`` → ``measure_in_region``),
        the same path every other signal is judged by. A DB signal the search
        never matched gains a matched peak here, so the analyst can point the
        software at a signal it missed.
        """
        dp     = self._current_dp
        met    = self._current_met_dp
        mp_ref = self._current_mp_ref
        if dp is None or met is None:
            return

        if mp_ref is None:
            dp_ppm = float(dp.get('ppm_center', dp.get('ppm', 0)))
            mp_ref = {
                'db_ppm':          dp_ppm,
                'peak_ppm':        dp_ppm,
                'ppm_diff':        0.0,
                'peak_rank':       None,
                'type':            dp.get('type', 's'),
                'protons':         dp.get('protons', ''),
                'j_coupling':      dp.get('j_coupling', ''),
                'predicted_lines': [],
                'annotation_quality': None,
                # What the reference measured about a pattern travels with
                # the signal; resolve_dj matches the shape from these.
                **{k: dp.get(k) for k in (
                    'pattern_lines', 'pattern_profile', 'profile_fwhm_hz',
                    'profile_nh') if dp.get(k) is not None},
            }
            met.setdefault('matched_peaks', []).append(mp_ref)
            self._current_mp_ref = mp_ref
        if '_original_predicted_lines' not in mp_ref:
            mp_ref['_original_predicted_lines'] = list(
                mp_ref.get('predicted_lines') or [])
        if '_original_position' not in mp_ref:
            mp_ref['_original_position'] = [mp_ref.get('peak_ppm'),
                                            mp_ref.get('ppm_diff')]

        mp_ref[MANUAL_REGION_KEY] = [float(lo_ppm), float(hi_ppm)]
        self._show_reset_btn(True)
        self._rescore_after_edit(met)
        self._plot_db_peak_zoom(met, dp)

    def _rc_reset(self):
        """Lift the analyst's region: the signal goes back to the algorithm's
        own lines and is measured again from them."""
        mp_ref = self._current_mp_ref
        if self._current_dp is None or mp_ref is None:
            return
        if manual_region(mp_ref) is None:
            return
        del mp_ref[MANUAL_REGION_KEY]
        original = mp_ref.get('_original_predicted_lines')
        if original:
            mp_ref['predicted_lines'] = list(original)
            _pos = mp_ref.get('_original_position') or [None, None]
            if _pos[0] is not None:
                mp_ref['peak_ppm'], mp_ref['ppm_diff'] = _pos
        elif mp_ref.get('peak_rank') is None:
            # Born of a region on a signal the search never matched: without
            # the region there is no evidence for it at all.
            self._current_met_dp['matched_peaks'].remove(mp_ref)
            self._current_mp_ref = None
        # The cached measurement describes the region's lines, not these.
        for _k in ('_n_found_lines', '_n_exp_lines'):
            mp_ref.pop(_k, None)
        self._show_reset_btn(False)
        self._rescore_after_edit(self._current_met_dp)
        self._plot_db_peak_zoom(self._current_met_dp, self._current_dp)

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
        """Edit the annotation parameters and rescore with them.

        Opens the shared dialog — this used to be a second implementation of the
        same list of parameters, kept in step with the menu one by hand.
        """
        from gui.widgets.annotation_params_dialog import open_annotation_params

        def _apply(new_params):
            self._params.update(new_params)
            _ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
            if _ann is not None:
                _ann._annotation_params = dict(self._params)
            self._recompute_and_refresh()

        open_annotation_params(
            self.win, self._params, _apply,
            tier_builder=build_tier_choice,
            tier_reader=read_tier_choice,
            tier_note=_TIER_NOTE)


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
                    'Signal #', 'Metabolite', 'Met ID',
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
                    n_cands   = sum(1 for c in self._peak_candidates.get(rank, [])
                                    if self._is_valid_candidate(c[2]))
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