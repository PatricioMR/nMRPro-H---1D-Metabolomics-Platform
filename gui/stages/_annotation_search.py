# gui/stages/_annotation_search.py
"""
_AnnotationSearchMixin — all search and rescue methods for AnnotationStage.

Extracted from annotation_stage.py to isolate:
  • _rescue_unassigned_with_mcr — rescue pass with relaxed DB search + MCR
  • search_internal_db          — main DB fingerprint search (threaded)
  • _rerun_mcr_for_peak         — re-run MCR on a single peak
  • _update_progress / _handle_search_error / _open_results_window — helpers
  • search_builtin_library      — built-in CSV library search
  • search_hmdb_online          — placeholder for online search
  • _export                     — CSV export of results
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import os
import time
import threading
import sqlite3
import csv

from ._annotation_helpers import (
    _align_peaks_to_reference, _ppm_score, _multiplet_score,
    _baseline_correct_linear, _fit_r2, _observe_j_hz,
    _mult_compat_check, _adaptive_mcr_mask,
    _ask_search_mode, _pick_database,
    _multi_peak_coherence,
    _stocsy_coherence,
)
from gui.notation.evidence_weighting import dynamic_composite


def is_unquantifiable_multiplicity(mult):
    """True for DB multiplicities that must not be used for annotation/quant.

    Covers generic multiplets ('m') — no J, no predictable pattern — and any
    BROAD signal ('br s', 'br. s.', 'br d', …): broad lines are exchangeable
    protons (OH/NH), whose integral and position drift with pH/temperature/
    solvent suppression, so they are unreliable quantification targets.
    """
    m = str(mult or '').lower().strip()
    return m in ('m', 'multiplet') or m.startswith('br')


class _AnnotationSearchMixin:
    """Mixin — mixed into AnnotationStage. Uses self.* freely."""

    # ── Rescue ────────────────────────────────────────────────────────────────

    def _rescue_unassigned_with_mcr(self):
        """Rescue Good peaks that remained unassigned after annotation."""
        if not self.peaks:
            messagebox.showinfo('Info', 'No peaks detected.')
            return
        if not self._last_results_args:
            messagebox.showinfo('Info', 'Run annotation first, then rescue unassigned peaks.')
            return

        db_path = (self._last_results_args.get('db_path')
                   or getattr(self.app, '_last_db_path', None))
        if not db_path or not str(db_path).lower().endswith('.db') or not os.path.isfile(db_path):
            messagebox.showinfo('Info', 'Rescue requires a valid SQLite DB (.db) from annotation search.')
            return

        try:
            tolerance = float(self.tolerance_var.get())
        except Exception:
            tolerance = 0.05
        relax_tol = tolerance * 1.10   # hard cap: never search beyond 10 % of user tolerance
        delta_max = tolerance * 1.10
        self._status(
            f'Rescue pass: tolerance {tolerance:.3f} ppm (cap {relax_tol:.4f} ppm ± 10 %).'
        )
        try:
            mhz = float(self.spectrometer_var.get())
        except Exception:
            mhz = 600.0

        existing_results = list(self._last_results_args.get('results', []))
        already_matched = {int(r.get('peak_rank')) for r in existing_results
                           if r.get('peak_rank') is not None}
        good_peaks = [p for p in self.peaks if p.get('status', 'Undefined') == 'Good']
        unassigned = [p for p in good_peaks if int(p.get('rank', -1)) not in already_matched]
        if not unassigned:
            messagebox.showinfo('Rescue', 'No unassigned Good peaks to rescue.')
            return

        shift_ppm, _align_ok = _align_peaks_to_reference(
            self.peaks,
            getattr(self.app, 'ppm', None),
            getattr(self.app, 'all_intensities', None))
        if not _align_ok:
            self._status(
                '⚠ Reference (DSS/TSP) not found — PPM axis may be '
                'misaligned. Check spectrum calibration.'
            )
        aligned_good = []
        for p in good_peaks:
            ap = dict(p)
            ap['ppm'] = p['ppm'] + shift_ppm
            ap['start_ppm'] = p.get('start_ppm', p['ppm']) + shift_ppm
            ap['end_ppm'] = p.get('end_ppm', p['ppm']) + shift_ppm
            aligned_good.append(ap)
        aligned_unassigned = [p for p in aligned_good if int(p.get('rank', -1)) in
                              {int(u.get('rank', -1)) for u in unassigned}]

        self._status(f'Rescue pass: scanning {len(aligned_unassigned)} unassigned peaks…')
        t0 = time.monotonic()
        conn = None
        rescued = []
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            tables = {r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            is_bmrb = 'shifts_1h' in tables
            is_hmdb = 'multiplets' in tables
            if not is_bmrb and not is_hmdb:
                messagebox.showerror('Incompatible DB', 'Database must contain shifts_1h or multiplets.')
                return

            _rescue_protons_col = ''
            if is_bmrb:
                _rescue_shift_cols = {r[1] for r in cur.execute(
                    "PRAGMA table_info(shifts_1h)").fetchall()}
                _rescue_protons_col = (", s.protons"
                                       if 'protons' in _rescue_shift_cols
                                       else ", 1 as protons")

            _ppm_arr = getattr(self.app, 'ppm', None)
            _raw_int = getattr(self.app, 'all_intensities', None)
            _int_arr = np.mean(_raw_int, axis=1) if _ppm_arr is not None and _raw_int is not None else None

            for peak in aligned_unassigned:
                ppm = float(peak['ppm'])
                p_lo = min(float(peak.get('start_ppm', ppm - relax_tol)),
                           float(peak.get('end_ppm', ppm + relax_tol))) - 0.03
                p_hi = max(float(peak.get('start_ppm', ppm - relax_tol)),
                           float(peak.get('end_ppm', ppm + relax_tol))) + 0.03

                rows = []
                if is_bmrb:
                    cur.execute(
                        "SELECT s.bmrb_id as hmdb_id, met.name,"
                        " s.ppm as ppm_center, s.ppm as ppm,"
                        " s.atom_name, s.multiplicity as type,"
                        f" s.j_coupling, s.multiplicity{_rescue_protons_col},"
                        " ? as peak_ppm, ? as peak_rank,"
                        " ABS(s.ppm - ?) as ppm_diff"
                        " FROM shifts_1h s"
                        " JOIN metabolites met ON s.bmrb_id = met.bmrb_id"
                        " WHERE s.ppm BETWEEN ? AND ?"
                        " ORDER BY ABS(s.ppm - ?) LIMIT 25",
                        (ppm, peak['rank'], ppm, p_lo, p_hi, ppm))
                    rows = [dict(r) for r in cur.fetchall()]
                    for r in rows:
                        r['match_type'] = 'shift'
                        r['database'] = 'BMRB'
                        r['shift_ppm'] = shift_ppm
                elif is_hmdb:
                    cur.execute("""
                        SELECT m.*, m.j_coupling, met.name,
                               ? as peak_ppm, ? as peak_rank,
                               ABS(m.ppm_center - ?) as ppm_diff
                        FROM multiplets m
                        JOIN metabolites met ON m.hmdb_id = met.hmdb_id
                        WHERE m.ppm_center BETWEEN ? AND ?
                        ORDER BY ABS(m.ppm_center - ?) LIMIT 25
                    """, (ppm, peak['rank'], ppm, p_lo, p_hi, ppm))
                    rows = [dict(r) for r in cur.fetchall()]
                    for r in rows:
                        r['match_type'] = 'multiplet'
                        r['database'] = 'HMDB'
                        r['shift_ppm'] = shift_ppm

                for r in rows:
                    ps = _ppm_score(r.get('peak_ppm', 0), r.get('ppm_center', r.get('ppm', 0)), relax_tol)
                    ms, md, pl = _multiplet_score(r, aligned_good, relax_tol, mhz,
                                                  ppm_array=_ppm_arr, intensity_array=_int_arr)
                    mult_type = (r.get('multiplicity') or r.get('type') or 's').lower().strip()
                    is_singlet = mult_type in ('s', 'singlet', '')
                    composite = ps if is_singlet else (0.4 * ps + 0.6 * ms)
                    compat, obs_mult = _mult_compat_check(r, aligned_good, relax_tol, mhz)
                    composite *= compat
                    r['observed_mult'] = obs_mult
                    r['mult_compat'] = round(compat, 3)
                    r['ppm_score'] = ps
                    r['multiplet_score'] = ms
                    r['composite_score'] = composite
                    r['multiplet_detail'] = md + (f'  |  obs={obs_mult} compat={compat:.2f}' if compat < 0.95 else '')
                    r['predicted_lines'] = pl
                    rescued.append(r)

            # Deduplicate per rescued peak and entry
            seen = {}
            for r in rescued:
                key = ((int(r['peak_rank']) if r.get('peak_rank') is not None else -1),
                       r.get('hmdb_id') or r.get('bmrb_id', ''),
                       round(float(r.get('ppm_center') or r.get('ppm') or 0), 3))
                if key not in seen or r.get('composite_score', 0) > seen[key].get('composite_score', 0):
                    seen[key] = r
            rescued = [r for r in seen.values() if r.get('ppm_diff', 0) <= delta_max]
            rescued.sort(key=lambda r: r.get('composite_score', 0), reverse=True)

            # Apply MCR on rescued ranks with >=2 candidates
            _by_rank = {}
            for r in rescued:
                _by_rank.setdefault((int(r['peak_rank']) if r.get('peak_rank') is not None else -1), []).append(r)
            _ppm_full = getattr(self.app, 'ppm', None)
            _all_int = getattr(self.app, 'all_intensities', None)
            if _ppm_full is not None and _all_int is not None:
                from gui.utils.mcr_utils import mcr_annotation_region, simulate_lorentzian_profile
                from gui.widgets.annotation_results_window import _build_stick_positions
                for peak in aligned_unassigned:
                    rk = int(peak['rank'])
                    cands = _by_rank.get(rk, [])
                    if len(cands) < 2:
                        continue
                    mask = _adaptive_mcr_mask(_ppm_full, _all_int, peak, cands, field_mhz=mhz)
                    if np.sum(mask) < 4:
                        continue
                    ppm_reg = _ppm_full[mask]
                    X_reg = (_all_int[mask, :] if _all_int.ndim == 2 else _all_int[mask, np.newaxis])
                    db_profiles = []
                    for c in cands:
                        try:
                            dp_ppm = float(c.get('ppm_center') or c.get('ppm') or 0)
                            mult = (c.get('type') or c.get('multiplicity') or 's')
                            try:
                                nh = int(c.get('protons') or 1)
                            except Exception:
                                nh = 1
                            sticks = _build_stick_positions(dp_ppm, mult, nh, c.get('j_coupling'))
                            if not sticks:
                                sticks = [(dp_ppm, 1.0)]
                            prof = simulate_lorentzian_profile(sticks, ppm_reg)
                        except Exception:
                            prof = np.zeros(len(ppm_reg))
                        db_profiles.append(prof)
                    if not db_profiles:
                        continue
                    S, C, mcr_m, K, assignments = mcr_annotation_region(
                        X_reg, db_profiles, r2_threshold=self.mcr_r2_threshold)
                    orig_peak = next((p for p in self.peak_stage.peaks if int(p.get('rank', -1)) == rk), None)
                    if orig_peak is not None:
                        orig_peak['mcr_annotation'] = {
                            'S': S, 'C': C, 'metrics': mcr_m, 'K': K,
                            'assignments': assignments, 'ppm_region': ppm_reg,
                            'candidates': cands, 'r2_threshold': self.mcr_r2_threshold,
                        }

            if not rescued:
                self._status_ok('Rescue completed: no robust new assignments found.')
                messagebox.showinfo('Rescue', 'No robust rescued candidates found.')
                return

            combined = existing_results + rescued
            combined.sort(key=lambda r: r.get('composite_score', 0), reverse=True)
            elapsed = time.monotonic() - t0
            self._status_ok(f'Rescue completed: +{len(rescued)} candidate match(es) in {elapsed:.2f}s.')
            self._open_results_window(combined, None, len(good_peaks), db_path, elapsed)
        except Exception as e:
            self._status_err(f'Rescue failed: {e}')
            messagebox.showerror('Rescue Error', str(e))
        finally:
            if conn:
                conn.close()

    # ── MCR ───────────────────────────────────────────────────────────────────

    def _run_mcr_on_results(self, results, mhz):
        """Run MCR deconvolution on all annotated peaks and enrich results in place.

        Runs for every peak_rank (not just overlapping ones). MCR extracts the
        spectral component for each candidate and computes Pearson r between the
        component and the DB profile — this is the shape match score. A low
        Pearson r flags a likely false positive regardless of ppm proximity.
        shift_tol_ppm=0.05 (30 Hz at 600 MHz) absorbs pH/temperature variation
        without risking alignment to neighboring peaks.
        Returns number of regions processed.
        """
        _ppm_full = getattr(self.app, 'ppm', None)
        _all_int  = getattr(self.app, 'all_intensities', None)
        if _ppm_full is None or _all_int is None:
            return 0

        by_rank = {}
        for r in results:
            rk = (int(r['peak_rank']) if r.get('peak_rank') is not None else -1)
            by_rank.setdefault(rk, []).append(r)

        all_ranks = by_rank  # process every rank, not just overlaps
        overlap_ranks = {rk: cands for rk, cands in by_rank.items() if len(cands) >= 2}

        from gui.utils.mcr_utils import mcr_annotation_region, simulate_lorentzian_profile
        from gui.widgets.annotation_results_window import _build_stick_positions

        n_validated = 0
        for rk, cands in all_ranks.items():
            orig_peak = next(
                (p for p in self.peak_stage.peaks if int(p.get('rank', -1)) == rk), None)
            if orig_peak is None:
                continue

            mask = _adaptive_mcr_mask(_ppm_full, _all_int, orig_peak, cands, field_mhz=mhz)
            if np.sum(mask) < 4:
                continue

            ppm_reg = _ppm_full[mask]
            X_reg   = (_all_int[mask, :] if _all_int.ndim == 2
                       else _all_int[mask, np.newaxis])

            db_profiles = []
            db_entries  = []
            for c in cands:
                try:
                    dp_ppm = float(c.get('ppm_center') or c.get('ppm') or 0)
                    mult   = (c.get('type') or c.get('multiplicity') or 's')
                    try:
                        nh = int(c.get('protons') or 1)
                    except Exception:
                        nh = 1
                    sticks = _build_stick_positions(dp_ppm, mult, nh, c.get('j_coupling'))
                    if not sticks:
                        sticks = [(dp_ppm, 1.0)]
                    prof = simulate_lorentzian_profile(sticks, ppm_reg)
                except Exception:
                    prof = np.zeros(len(ppm_reg))
                db_profiles.append(prof)
                db_entries.append(c)

            if not db_profiles:
                continue

            try:
                S, C, mcr_m, K, assignments, *_ = mcr_annotation_region(
                    X_reg, db_profiles,
                    r2_threshold=self.mcr_r2_threshold,
                    ppm_arr=ppm_reg,
                    db_entries=db_entries,
                    field_mhz=mhz,
                    shift_tol_ppm=0.05)
            except Exception:
                continue

            orig_peak['mcr_annotation'] = {
                'S': S, 'C': C, 'metrics': mcr_m, 'K': K,
                'assignments': assignments, 'ppm_region': ppm_reg,
                'candidates': cands, 'r2_threshold': self.mcr_r2_threshold,
                'validated': True,
            }

            mcr_r2 = mcr_m.get('r2', None)
            if mcr_r2 is None or (isinstance(mcr_r2, float) and np.isnan(mcr_r2)):
                mcr_confirmed = None
            else:
                top_idx = max(range(len(cands)),
                              key=lambda i: cands[i].get('composite_score', 0))
                pearson_r = next(
                    (a[1] for a in assignments if a[0] == top_idx), 0.0)
                # True positive: R² good AND shape matches (Pearson r ≥ 0.70)
                # False positive: shape mismatch (Pearson r < 0.40) regardless of R²
                if pearson_r >= 0.70 and mcr_r2 >= self.mcr_r2_threshold:
                    mcr_confirmed = True
                elif pearson_r < 0.40:
                    mcr_confirmed = False   # shape mismatch → likely false positive
                else:
                    mcr_confirmed = None    # inconclusive

            for c in cands:
                c['mcr_r2']        = mcr_r2
                c['mcr_confirmed'] = mcr_confirmed
                c_idx = cands.index(c)
                c['mcr_pearson_r'] = next(
                    (a[1] for a in assignments if a[0] == c_idx), None)

            n_validated += 1

        # C-matrix cross-sample correlation — confirms identity without relying on
        # absolute ppm position (robust to pH/temperature shifts)
        by_met = {}
        for r in results:
            mid = r.get('hmdb_id') or r.get('bmrb_id') or r.get('name')
            if mid:
                by_met.setdefault(mid, []).append(r)

        for rk, cands in all_ranks.items():
            orig_peak = next(
                (p for p in self.peak_stage.peaks if int(p.get('rank', -1)) == rk), None)
            if orig_peak is None or 'mcr_annotation' not in orig_peak:
                continue
            C_mat = orig_peak['mcr_annotation'].get('C')
            if C_mat is None or C_mat.shape[0] < 3:
                continue

            for k_idx, c in enumerate(cands):
                mid = c.get('hmdb_id') or c.get('bmrb_id') or c.get('name')
                if not mid or k_idx >= C_mat.shape[1]:
                    continue
                peers = [r for r in by_met.get(mid, [])
                         if (int(r['peak_rank']) if r.get('peak_rank') is not None else -1) != rk]
                if not peers:
                    continue
                c_vec = C_mat[:, k_idx]
                if np.std(c_vec) < 1e-12:
                    continue
                peer_corrs = []
                for peer in peers:
                    peer_rk  = (int(peer['peak_rank']) if peer.get('peak_rank') is not None else -1)
                    peer_peak = next(
                        (p for p in self.peak_stage.peaks
                         if int(p.get('rank', -1)) == peer_rk), None)
                    if peer_peak is None:
                        continue
                    p_ppm = float(peer_peak.get('ppm', 0))
                    win   = np.abs(_ppm_full - p_ppm) <= 0.04
                    if np.sum(win) < 2:
                        continue
                    peer_int = _all_int[win, :].mean(axis=0)
                    if np.std(peer_int) < 1e-12:
                        continue
                    peer_corrs.append(float(np.corrcoef(c_vec, peer_int)[0, 1]))
                if peer_corrs:
                    c['mcr_cross_corr'] = float(np.mean(peer_corrs))

        return n_validated

    # ── Search ────────────────────────────────────────────────────────────────

    def search_internal_db(self):
        if not self.peaks:
            messagebox.showinfo('Info',
                'No peaks detected.\nRun Peak Detection first.')
            return

        has_prev = (bool(self._quality_marks)
                    or bool(self._metabolites)
                    or bool(self._last_results_args))
        incremental = False
        if has_prev:
            choice = _ask_search_mode(self.app.root)
            if choice is None:
                return
            if choice == 'reset':
                self._quality_marks         = {}
                self._child_quality_marks   = {}
                self._anchor_lines          = {}
                self._subline_quality_marks = {}
                self._metabolites           = None
                self._last_results_args     = None
                self.annotations            = {}
                self._j_resolve_contributions = {}
                incremental = False
                # Close existing results window immediately so user sees clean state
                _old_win = getattr(self.app, '_annot_win', None)
                if _old_win is not None:
                    try:
                        if _old_win.win.winfo_exists():
                            _old_win.win.destroy()
                    except Exception:
                        pass
                    self.app._annot_win = None
                # ── Reset all downstream stages ───────────────────────────
                _stages = getattr(self.app, 'stages', {}) or {}

                # Quantification
                _q_stage = _stages.get('quantification')
                if _q_stage is not None:
                    _q_stage._quant_results       = []
                    _q_stage._all_peak_quant       = {}
                    _q_stage._quant_peak_overrides = {}
                    try:
                        _q_stage._populate_tree()
                    except Exception:
                        pass
                    try:
                        _q_stage.run_status.config(
                            text='Run annotation first',
                            foreground='red')
                    except Exception:
                        pass
                    for _btn_attr in ('btn_csv', 'btn_metaboanalyst'):
                        try:
                            getattr(_q_stage, _btn_attr).config(state='disabled')
                        except Exception:
                            pass

                # J-Resolve
                _jr_stage = _stages.get('j_resolve')
                if _jr_stage is not None:
                    _jr_stage._peak_overlap          = {}
                    _jr_stage._contributions         = {}
                    _jr_stage._status_map            = {}
                    _jr_stage._anchor_peaks          = {}
                    _jr_stage._nnls_rejected         = set()
                    _jr_stage._nnls_residual_norm    = None
                    _jr_stage._nnls_coverage         = {}
                    _jr_stage._nnls_cond             = None
                    try:
                        _jr_stage._tree.delete(*_jr_stage._tree.get_children())
                    except Exception:
                        pass
                    try:
                        _jr_stage._btn_run.config(state='disabled')
                    except Exception:
                        pass

                # Full-Spectrum MCR
                _fs_stage = _stages.get('fullspectrum_mcr')
                if _fs_stage is not None:
                    _fs_stage._mcr_regions  = []
                    _fs_stage._norm_factors = None
                    try:
                        _fs_stage._status_label.config(text='', foreground='black')
                    except Exception:
                        pass
                    try:
                        _fs_stage._result_label.config(text='No results yet.')
                    except Exception:
                        pass

                # Statistics — clear computed results and exclusions
                _st_stage = _stages.get('statistics')
                if _st_stage is not None:
                    _st_stage._current_method      = None
                    _st_stage._excluded_samples    = set()
                    _st_stage._excluded_metabolites = set()
                    try:
                        if _st_stage._fig is not None:
                            import matplotlib.pyplot as _plt
                            _plt.close(_st_stage._fig)
                            _st_stage._fig = None
                    except Exception:
                        pass
            else:
                incremental = True

        db_path = _pick_database(self.app.root)
        if not db_path:
            return

        self.app._last_db_path = db_path

        ext     = os.path.splitext(db_path)[1].lower()
        db_type = 'hmdb' if ext == '.db' else ('csv' if ext == '.csv' else None)
        if not db_type:
            messagebox.showerror('Error', 'Unsupported file type.')
            return

        self.internal_status.config(
            text=f'Selected: {os.path.basename(db_path)}',
            foreground='blue')

        tolerance = float(self.tolerance_var.get())
        try:
            mhz = float(self.spectrometer_var.get())
        except ValueError:
            mhz = 600.0

        all_peaks = [p for p in self.peaks if p.get('rank') is not None]

        if not all_peaks:
            messagebox.showinfo(
                'No Peaks',
                'No detected peaks found.\n\n'
                'Run Peak Detection first.')
            return

        if incremental and self._last_results_args:
            prev_results       = self._last_results_args.get('results', [])
            already_good_ranks = set()
            for r in prev_results:
                rk  = r.get('peak_rank')
                mid = r.get('hmdb_id') or r.get('bmrb_id', '')
                key = (mid, rk)
                if self._quality_marks.get(key) == 'good':
                    if rk is not None:
                        already_good_ranks.add(int(rk))
            peaks_to_search = [p for p in all_peaks
                               if int(p.get('rank', -1))
                               not in already_good_ranks]
            if not peaks_to_search:
                messagebox.showinfo(
                    'Nothing to update',
                    'All peaks already have confirmed annotations.\n'
                    'Use "Reset & start over" to redo everything.')
                return
            carry_results = [r for r in prev_results
                             if (int(r['peak_rank']) if r.get('peak_rank') is not None else -1) in already_good_ranks
                             and r.get('ppm_diff', float('inf')) <= tolerance]
        else:
            peaks_to_search = all_peaks
            carry_results   = []

        # Expand with deconvolved sub-peaks if available
        _pk_stage = getattr(self, 'peak_stage', None)
        _ctrl     = getattr(_pk_stage, '_deconv_controller', None) if _pk_stage else None
        if _ctrl is not None and _ctrl.results:
            _deconv_ranks_pre = set()
            for _ci2, _cp in enumerate(_ctrl.peaks):
                if (_cp.get('_deconvolved') and _cp.get('rank') is not None
                        and _ctrl.results.get(_ci2, {}).get('_saved')):
                    _deconv_ranks_pre.add(int(_cp['rank']))
            if _deconv_ranks_pre:
                _stripped = [r for r in carry_results
                             if (int(r['peak_rank']) if r.get('peak_rank') is not None else -1) not in _deconv_ranks_pre]
                if len(_stripped) != len(carry_results):
                    _missing_rks = _deconv_ranks_pre - {(int(r['peak_rank']) if r.get('peak_rank') is not None else -1)
                                                        for r in peaks_to_search}
                    _extra = [p for p in all_peaks
                              if int(p.get('rank', -1)) in _missing_rks]
                    peaks_to_search = peaks_to_search + _extra
                carry_results = _stripped
        if _ctrl is not None and _ctrl.results:
            _expanded = []
            _deconvolved_ranks = set()
            for _pk in peaks_to_search:
                _rk = int(_pk.get('rank', -1))
                _found_sub = False
                for _ci, _cp in enumerate(_ctrl.peaks):
                    if int(_cp.get('rank', -1)) == _rk and _cp.get('_deconvolved'):
                        _res = _ctrl.results.get(_ci, {})
                        if not _res.get('_saved'):
                            break
                        _subs = [sp for sp in _res.get('peaks', [])
                                 if sp.get('status') != 'Poor']
                        if _subs:
                            for _sp in _subs:
                                _pseudo = {
                                    'ppm':         float(_sp['ppm']),
                                    'rank':        _rk,
                                    'status':      'Good',
                                    'height':      float(_sp.get('height', 0)),
                                    'width_ppm':   float(_sp.get('width', _sp.get('width_ppm', 0.004))),
                                    'width_at_half': float(_sp.get('width', _sp.get('width_ppm', 0.004))),
                                    'is_sub_peak': True,
                                    'snr':         float(_sp.get('snr', 5.0)),
                                }
                                _expanded.append(_pseudo)
                            _deconvolved_ranks.add(_rk)
                            _found_sub = True
                            break
                if not _found_sub:
                    _expanded.append(_pk)
            peaks_to_search = _expanded

            from collections import defaultdict
            _sub_by_rank = defaultdict(list)
            for _pk in peaks_to_search:
                if _pk.get('is_sub_peak'):
                    _sub_by_rank[int(_pk.get('rank', -1))].append(_pk)
            _mult_map_rev = {1: 's', 2: 'd', 3: 't', 4: 'q', 5: 'quintet', 6: 'sextet'}
            for _rk, _sps in _sub_by_rank.items():
                if len(_sps) < 2:
                    for _sp in _sps:
                        _sp.setdefault('_obs_j_hz', None)
                        _sp.setdefault('_obs_mult', 's')
                    continue
                _ppms_sorted = sorted(float(_sp['ppm']) for _sp in _sps)
                _seps_hz = [abs(_ppms_sorted[k+1] - _ppms_sorted[k]) * mhz
                            for k in range(len(_ppms_sorted) - 1)]
                _obs_j = float(np.median(_seps_hz)) if _seps_hz else None
                _obs_m = _mult_map_rev.get(len(_sps), 'm')
                for _sp in _sps:
                    _sp['_obs_j_hz'] = _obs_j
                    _sp['_obs_mult'] = _obs_m

        # Unified progress window — covers all phases (search + build)
        pw = tk.Toplevel(self.app.root)
        pw.title('nMRPipe — Annotating…')
        pw.resizable(False, False)
        pw.transient(self.app.root)
        pw.grab_set()
        self.app.root.update_idletasks()
        _px = (self.app.root.winfo_rootx()
               + self.app.root.winfo_width() // 2 - 230)
        _py = (self.app.root.winfo_rooty()
               + self.app.root.winfo_height() // 2 - 65)
        pw.geometry(f'460x130+{_px}+{_py}')

        ttk.Label(pw, text=f'Database: {os.path.basename(db_path)}',
                  font=('Arial', 9)).pack(pady=(10, 2))
        self.progress_bar = ttk.Progressbar(
            pw, length=420, mode='determinate', maximum=100)
        self.progress_bar.pack(pady=(4, 2), padx=20)
        self.progress_op_label = ttk.Label(
            pw, text='Initialising…', font=('Arial', 8), foreground='gray')
        self.progress_op_label.pack()
        self.progress_time_label = ttk.Label(
            pw, text='', font=('Arial', 8), foreground='gray')
        self.progress_time_label.pack(pady=(2, 8))
        pw.update()

        def search_task():
            t0 = time.monotonic()
            all_results, peaks_matched = [], set()
            total_entries = 1
            conn = None
            _n_lines_map = {
                's': 1, 'd': 2, 't': 3, 'q': 4, 'quin': 5,
                'dd': 4, 'ddd': 8, 'dt': 6, 'td': 6, 'tt': 9,
                'dq': 8, 'tq': 12,
            }
            _ppm_arr = getattr(self.app, 'ppm', None)
            _raw_int = getattr(self.app, 'all_intensities', None)
            has_spec = (
                _ppm_arr is not None
                and _raw_int is not None
                and len(_ppm_arr) > 10)
            if has_spec:
                _mean_int = (np.mean(_raw_int, axis=1)
                             if getattr(_raw_int, 'ndim', 1) == 2
                             else np.asarray(_raw_int, dtype=float))

            def _observe_peak_mult(pk, ppm_arr, int_arr, field_mhz):
                p_lo = min(float(pk.get('start_ppm', pk['ppm'])),
                           float(pk.get('end_ppm',   pk['ppm'])))
                p_hi = max(float(pk.get('start_ppm', pk['ppm'])),
                           float(pk.get('end_ppm',   pk['ppm'])))
                span = p_hi - p_lo
                if span < 0.002 or ppm_arr is None or int_arr is None:
                    return 's', None
                mask = (ppm_arr >= p_lo) & (ppm_arr <= p_hi)
                if np.sum(mask) < 5:
                    return 's', None
                x = ppm_arr[mask]
                y = int_arr[mask].copy()
                y -= y.min()
                if y.max() < 1e-12:
                    return 's', None
                y /= y.max()
                from scipy.signal import find_peaks as _fp
                min_dist = max(2, int(len(x) * 0.08))
                pks, _ = _fp(y, height=0.20, prominence=0.12, distance=min_dist)
                n = len(pks)
                if n <= 1:
                    return 's', None
                ppms_pk = x[pks]
                seps_hz = [abs(ppms_pk[k+1] - ppms_pk[k]) * field_mhz
                           for k in range(len(ppms_pk) - 1)]
                j_est = float(np.median(seps_hz))
                mult_map = {2: 'd', 3: 't', 4: 'q'}
                return mult_map.get(n, 'm'), j_est

            if has_spec:
                for _pk in peaks_to_search:
                    if _pk.get('is_sub_peak'):
                        continue
                    _om, _oj = _observe_peak_mult(
                        _pk, _ppm_arr, _mean_int, mhz)
                    _pk['_obs_mult'] = _om
                    _pk['_obs_j_hz'] = _oj

            try:
                from gui.widgets.annotation_results_window import (
                    _build_stick_positions, lines_from_transitions)
                from gui.utils.mcr_utils import simulate_lorentzian_profile

                _cal = getattr(self.app, '_ppm_calibration', None)
                _cal_delta = float(_cal['delta_mean']) if _cal else 0.0
                # Guard against stale/corrupt calibration from old project versions.
                # The in-window calibration logic limits each anchor to Δ < 0.02 ppm,
                # so a valid delta_mean must be well below that.  Values beyond ±0.05
                # ppm indicate a broken or accumulated calibration — discard them.
                if abs(_cal_delta) > 0.05:
                    _cal_delta = 0.0

                ppm_vals   = [p['ppm'] for p in peaks_to_search]
                g_lo       = min(ppm_vals) - _cal_delta - max(tolerance * 2, 0.10)
                g_hi       = max(ppm_vals) - _cal_delta + max(tolerance * 2, 0.10)

                db_entries = []
                is_bmrb    = False

                if db_type == 'hmdb':
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    cur   = conn.cursor()
                    tables = {r[0] for r in cur.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table'").fetchall()}
                    has_shifts   = 'shifts_1h' in tables
                    has_multiplets = 'multiplets' in tables

                    # Detectar ID primário: bmrb_id (BMRB) vs hmdb_id (HMDB nativo)
                    # Prefer bmrb_id only when the column exists AND has non-NULL values;
                    # a column present but fully NULL means the DB uses hmdb_id as PK.
                    if has_shifts:
                        shift_cols = {r[1] for r in cur.execute(
                            "PRAGMA table_info(shifts_1h)").fetchall()}
                        if 'bmrb_id' in shift_cols:
                            _bmrb_count = cur.execute(
                                "SELECT COUNT(bmrb_id) FROM shifts_1h").fetchone()[0]
                            id_col = 'bmrb_id' if _bmrb_count > 0 else 'hmdb_id'
                        else:
                            id_col = 'hmdb_id'
                        met_id_col = id_col
                    else:
                        id_col = met_id_col = 'hmdb_id'

                    if not has_shifts and not has_multiplets:
                        self.app.root.after(0, lambda: messagebox.showerror(
                            'Incompatible DB',
                            'Database must contain shifts_1h or multiplets.'))
                        self.app.root.after(0, pw.destroy)
                        return

                    if has_shifts:
                        _protons_col = (", s.protons" if 'protons' in shift_cols
                                        else ", 1 as protons")
                        cur.execute(
                            f"SELECT s.{id_col} as hmdb_id, met.name,"
                            " s.ppm as ppm_center, s.ppm as ppm,"
                            " s.atom_name, s.multiplicity as type,"
                            f" s.j_coupling, s.multiplicity{_protons_col}"
                            " FROM shifts_1h s"
                            " JOIN metabolites met"
                            f"   ON s.{id_col} = met.{met_id_col}"
                            " WHERE s.ppm BETWEEN ? AND ?"
                            " ORDER BY s.ppm",
                            (g_lo, g_hi))
                        db_entries = [dict(r) for r in cur.fetchall()]
                        db_label = 'BMRB' if id_col == 'bmrb_id' else 'HMDB'
                        for e in db_entries:
                            e['match_type'] = 'shift'
                            e['database']   = db_label
                    else:
                        cur.execute(
                            "SELECT m.*, met.name"
                            " FROM multiplets m"
                            " JOIN metabolites met"
                            "   ON m.hmdb_id = met.hmdb_id"
                            " WHERE m.ppm_center BETWEEN ? AND ?"
                            " ORDER BY m.ppm_center",
                            (g_lo, g_hi))
                        db_entries = [dict(r) for r in cur.fetchall()]
                        for e in db_entries:
                            e['match_type'] = 'multiplet'
                            e['database']   = 'HMDB'

                elif db_type == 'csv':
                    import pandas as pd
                    df = pd.read_csv(db_path)
                    for _, row in df.iterrows():
                        rdict = dict(row)
                        ppm_v = float(rdict.get('ppm', 0))
                        if g_lo <= ppm_v <= g_hi:
                            rdict['ppm_center']  = ppm_v
                            rdict['database']    = 'CSV'
                            rdict['match_type']  = 'csv_match'
                            rdict.setdefault('type', 's')
                            db_entries.append(rdict)

                if _cal_delta != 0.0:
                    for e in db_entries:
                        for key in ('ppm_center', 'ppm'):
                            if key in e and e[key] is not None:
                                try:
                                    e[key] = float(e[key]) + _cal_delta
                                except (ValueError, TypeError):
                                    pass

                total_entries = max(1, len(db_entries))
                self.app.root.after(
                    0, lambda: self._status(
                        f'Fitting fingerprints for {total_entries} '
                        f'DB entries…'))

                # Bulk-fetch all sibling peaks for multi-peak coherence scoring.
                # One query for all candidate metabolites — avoids N+1 overhead.
                all_peaks_by_id = {}
                if conn is not None and db_type == 'hmdb':
                    try:
                        _cand_ids = list({
                            e.get('hmdb_id') or e.get('bmrb_id')
                            for e in db_entries
                            if (e.get('hmdb_id') or e.get('bmrb_id'))
                        })
                        if _cand_ids:
                            _id_col_bulk = id_col if has_shifts else 'hmdb_id'
                            _table_bulk  = 'shifts_1h' if has_shifts else 'multiplets'
                            _ppm_col     = 'ppm' if has_shifts else 'ppm_center'
                            _ph          = ','.join('?' * len(_cand_ids))
                            _bulk_rows   = cur.execute(
                                f"SELECT {_id_col_bulk} as mid, {_ppm_col} as ppm,"
                                f" COALESCE(intensity, 1.0) as intensity"
                                f" FROM {_table_bulk}"
                                f" WHERE {_id_col_bulk} IN ({_ph})",
                                _cand_ids,
                            ).fetchall()
                            for _br in _bulk_rows:
                                all_peaks_by_id.setdefault(_br[0], []).append(
                                    (_br[1], _br[2]))
                    except Exception:
                        pass  # coherence will default to 1.0 for all entries

                # Source reliability weights — penalise less-curated databases
                # so well-annotated sources rank above statistically-inferred entries.
                _SOURCE_WEIGHTS = {
                    'internal': 1.00,
                    'COLMAR':   0.90,
                    'BMRB':     0.82,
                    'HMDB':     0.68,
                    'CSV':      0.72,
                }

                # Graph STOCSY — retrieve precomputed peak correlation matrix
                # and sample count from peak_detection_stage.  Both default to
                # safe no-op values when detection hasn't run or n_samples < 3.
                _pk_stage  = getattr(self, 'peak_stage', None)
                _pk_corr   = getattr(_pk_stage, '_peak_corr',    {})
                _n_samp_st = getattr(_pk_stage, '_n_samples_pk', 0)

                for i, entry in enumerate(db_entries):
                    db_ppm   = float(
                        entry.get('ppm_center') or entry.get('ppm') or 0)
                    mult_raw = str(
                        entry.get('type') or
                        entry.get('multiplicity') or 's').lower().strip()
                    j_raw    = str(entry.get('j_coupling') or '').strip()
                    try:
                        j_hz = float(
                            j_raw.split(',')[0].strip()) if j_raw else 0.0
                    except (ValueError, AttributeError):
                        j_hz = 0.0

                    is_singlet  = mult_raw in ('s', 'singlet', '')

                    # Skip unquantifiable multiplicities: generic 'm' (no J, no
                    # predictable pattern) and BROAD/exchangeable signals (br s,
                    # br. s., br d…). Broad lines are exchangeable protons whose
                    # area/position drift → unreliable for quantification and a
                    # source of false matches (previously a broad singlet was
                    # mis-built as a 2-line doublet).
                    if is_unquantifiable_multiplicity(mult_raw):
                        continue

                    n_lns       = _n_lines_map.get(mult_raw, 2)
                    j_ppm       = (j_hz / mhz
                                   if j_hz > 0 and mhz > 0 else 0.0)
                    half_width  = max(
                        j_ppm * (n_lns / 2) * 1.4, tolerance, 0.010)


                    try:
                        protons = int(entry.get('protons') or 1)
                    except (ValueError, TypeError):
                        protons = 1
                    # Prefer EXACT line positions from the DB transitions table
                    # (independent of J, which is missing for ~half the lines);
                    # fall back to the J-modelled stick pattern when unavailable.
                    _met_id_tr = entry.get('hmdb_id') or entry.get('bmrb_id')
                    _tr_lines = lines_from_transitions(
                        conn, _met_id_tr,
                        db_ppm - half_width, db_ppm + half_width, mhz)
                    if len(_tr_lines) >= 2:
                        sticks = _tr_lines
                        _sp = [p for p, _ in _tr_lines]
                        half_width = max(half_width,
                                         (max(_sp) - min(_sp)) / 2.0 + tolerance)
                    else:
                        sticks = _build_stick_positions(
                            db_ppm, mult_raw, protons, j_raw)
                    if not sticks:
                        sticks = [(db_ppm, 1.0)]

                    # Local window: 10% of the DB ppm value, floored at user tolerance.
                    # Defines the candidate pool for sibling/J search.
                    # The anchor peak must still be within tolerance to be accepted.
                    search_window = max(tolerance, abs(db_ppm) * 0.10)
                    local_peaks = [p for p in peaks_to_search
                                   if abs(p['ppm'] - db_ppm) <= search_window]
                    if not local_peaks:
                        continue
                    # Any detected peak within tolerance can serve as anchor —
                    # quality filtering is left to the composite score.
                    # Siblings (J-coupling partners) can be any quality.
                    good_in_tol = [p for p in local_peaks
                                   if abs(p['ppm'] - db_ppm) <= tolerance]
                    if not good_in_tol:
                        # Spectrum-based fallback: no detected peak within
                        # tolerance, but the signal may still be present in the
                        # raw spectrum (peak missed by detection — sub-threshold,
                        # merged, or too close to a neighbour).
                        # Mirror what the manual right-drag selection does.
                        if not has_spec:
                            continue
                        _syn_lo = db_ppm - tolerance
                        _syn_hi = db_ppm + tolerance
                        _syn_mask = (_ppm_arr >= _syn_lo) & (_ppm_arr <= _syn_hi)
                        if _syn_mask.sum() < 3:
                            continue
                        _syn_y = _mean_int[_syn_mask]
                        _syn_x = _ppm_arr[_syn_mask]
                        _syn_max_i = int(np.argmax(_syn_y))
                        _syn_h = float(_syn_y[_syn_max_i])
                        # Require ≥ 3× the local noise floor to avoid false matches
                        _syn_noise = float(np.percentile(np.abs(_syn_y), 25)) + 1e-12
                        if _syn_h < _syn_noise * 3.0:
                            continue
                        _syn_peak = {
                            'ppm':           float(_syn_x[_syn_max_i]),
                            'rank':          None,
                            'status':        'Good',
                            'height':        _syn_h,
                            'width_ppm':     0.010,
                            'width_at_half': 0.010,
                            '_synthetic':    True,
                            '_obs_j_hz':     0,
                            '_obs_mult':     mult_raw,
                        }
                        good_in_tol = [_syn_peak]
                        # Augment local_peaks with spectrum-derived sibling
                        # positions so the sibling loop and _multiplet_score
                        # can find them even if peak detection missed them.
                        _syn_aug = list(local_peaks)
                        if not is_singlet and j_ppm > 0 and n_lns >= 2:
                            for _k in range(1, n_lns):
                                for _sgn in (+1, -1):
                                    _sib_c = _syn_peak['ppm'] + _sgn * _k * j_ppm
                                    _sm = ((_ppm_arr >= _sib_c - tolerance * 0.7) &
                                           (_ppm_arr <= _sib_c + tolerance * 0.7))
                                    if _sm.sum() < 2:
                                        continue
                                    _sy = _mean_int[_sm]
                                    _sx = _ppm_arr[_sm]
                                    _si = int(np.argmax(_sy))
                                    if float(_sy[_si]) < _syn_noise * 2.0:
                                        continue
                                    _sp = float(_sx[_si])
                                    if any(abs(p['ppm'] - _sp) < 0.005
                                           for p in local_peaks):
                                        continue
                                    _syn_aug.append({
                                        'ppm':           _sp,
                                        'rank':          None,
                                        'status':        'Good',
                                        'height':        float(_sy[_si]),
                                        'width_ppm':     0.010,
                                        'width_at_half': 0.010,
                                        '_synthetic':    True,
                                        '_obs_j_hz':     0,
                                        '_obs_mult':     mult_raw,
                                    })
                        local_peaks = _syn_aug
                    nearest   = min(good_in_tol, key=lambda p: abs(p['ppm'] - db_ppm))
                    delta_ppm = abs(nearest['ppm'] - db_ppm)
                    peak_rank = nearest.get('rank')   # None for spectrum-derived peaks
                    peak_ppm  = float(nearest['ppm'])

                    _eff_j_ppm = j_ppm
                    if _eff_j_ppm <= 0 and not is_singlet:
                        _obs_j_fb = nearest.get('_obs_j_hz', 0) or 0
                        _eff_j_ppm = float(_obs_j_fb) / mhz if (_obs_j_fb > 0 and mhz > 0) else 0.0

                    multiplet_centre = peak_ppm
                    if not is_singlet and _eff_j_ppm > 0 and n_lns >= 2:
                        found_line_ppms = [peak_ppm]
                        for k in range(1, n_lns):
                            for sign in (+1, -1):
                                expected = peak_ppm + sign * k * _eff_j_ppm
                                best_sib = min(
                                    (p for p in local_peaks if p is not nearest),
                                    key=lambda p: abs(p['ppm'] - expected),
                                    default=None)
                                if best_sib is not None:
                                    dist = abs(best_sib['ppm'] - expected)
                                    if dist <= tolerance * 1.5:
                                        if best_sib['ppm'] not in found_line_ppms:
                                            found_line_ppms.append(float(best_sib['ppm']))
                                        break
                        if len(found_line_ppms) >= 2:
                            multiplet_centre = float(np.mean(found_line_ppms))

                    _shift        = multiplet_centre - db_ppm
                    shifted_sticks = [(s + _shift, ri) for s, ri in sticks]

                    win_lo = multiplet_centre - half_width
                    win_hi = multiplet_centre + half_width

                    fitting_r2  = 0.0
                    r2_computed = False
                    if has_spec:
                        _exp_hwhm = max(
                            float(nearest.get('width_ppm',
                                  nearest.get('width_at_half', 0.008))) / 2,
                            0.003)
                        spec_mask = (_ppm_arr >= win_lo) & (_ppm_arr <= win_hi)
                        if np.sum(spec_mask) >= 4:
                            x_win  = _ppm_arr[spec_mask]
                            y_win  = _mean_int[spec_mask].copy()
                            y_corr = _baseline_correct_linear(x_win, y_win)
                            sim    = simulate_lorentzian_profile(
                                shifted_sticks, x_win, hwhm=_exp_hwhm)
                            fitting_r2, _ = _fit_r2(sim, y_corr)
                            r2_computed = True

                    peaks_matched.add(peak_rank)

                    if not is_singlet and shifted_sticks:
                        _nearest_stick = min(shifted_sticks,
                                             key=lambda sv: abs(sv[0] - peak_ppm))
                        ppm_score_val = _ppm_score(peak_ppm, _nearest_stick[0], tolerance)
                    else:
                        ppm_score_val = _ppm_score(multiplet_centre, db_ppm, tolerance)
                    mult_compat, _ = _mult_compat_check(
                        entry, peaks_to_search, tolerance, mhz)
                    mult_conf   = 1.0

                    obs_mult_pk = nearest.get('_obs_mult', 's')
                    obs_j_pk    = nearest.get('_obs_j_hz')
                    if is_singlet:
                        j_compat      = 0.3 if obs_mult_pk not in ('s', 'unknown') else 1.0
                        j_conf        = 1.0  # mult-consistency check — always valid for singlets
                    elif j_hz > 0 and obs_j_pk is not None and obs_j_pk > 0:
                        j_ratio       = abs(j_hz - obs_j_pk) / max(j_hz, obs_j_pk, 0.1)
                        j_compat      = float(max(0.0, 1.0 - 2.0 * j_ratio))
                        j_conf        = 1.0  # J was measurable
                    elif obs_mult_pk in ('s',) and not is_singlet:
                        j_compat      = 0.4   # wrong mult type — valid penalty
                        j_conf        = 1.0
                    else:
                        j_compat      = 0.0   # J not measurable — excluded by j_conf
                        j_conf        = 0.0   # exclude component; do not default to 0.7

                    composite = dynamic_composite(
                        scores={'ppm':  ppm_score_val,
                                'r2':   fitting_r2,
                                'mult': float(mult_compat),
                                'j':    j_compat},
                        confidences={'ppm':  1.0,
                                     'r2':   1.0 if r2_computed else 0.0,
                                     'mult': mult_conf,
                                     'j':    j_conf},
                    )

                    # Multi-peak coherence: penalise candidates whose sibling
                    # peaks are absent or have inconsistent relative intensities.
                    # MIN_COH=0.05 retains a small floor for genuinely congested
                    # spectra while avoiding the 35% inflation of the old value.
                    _met_id_coh = entry.get('hmdb_id') or entry.get('bmrb_id', '')
                    _coh, _cov, _rm = _multi_peak_coherence(
                        _met_id_coh, all_peaks_by_id, peaks_to_search, tolerance)
                    _MIN_COH = 0.05
                    composite *= _MIN_COH + (1.0 - _MIN_COH) * _coh

                    # Graph STOCSY boost: metabolites whose matched peaks
                    # co-vary across samples score higher.
                    # Weight w uses exponential saturation so n=3 gives a small
                    # but non-zero boost while n≥10 approaches the full effect.
                    _stocsy_coh = _stocsy_coherence(
                        _met_id_coh, all_peaks_by_id, peaks_to_search,
                        tolerance, _pk_corr)
                    if _stocsy_coh is not None:
                        _w_st = float(
                            1.0 - np.exp(-0.25 * max(0, _n_samp_st - 2)))
                        composite *= np.exp(
                            0.10 * max(0.0, _stocsy_coh) * _w_st)

                    _scorer_centre = None
                    _j_obs_scorer  = None
                    _dj_scorer     = None

                    if nearest.get('is_sub_peak'):
                        siblings = sorted(
                            [p for p in peaks_to_search
                             if p.get('rank') == peak_rank
                             and p.get('is_sub_peak')],
                            key=lambda p: p['ppm'], reverse=True)
                        if siblings:
                            max_h = max(p.get('height', 1) for p in siblings) or 1
                            predicted_lines = [
                                {'ppm': float(p['ppm']),
                                 'rel_intensity': float(p.get('height', 1)) / max_h}
                                for p in siblings]
                        else:
                            predicted_lines = [
                                {'ppm': float(s), 'rel_intensity': float(ri)}
                                for s, ri in shifted_sticks]
                    else:
                        # Use the scorer to find only confirmed experimental lines.
                        # Single source of truth: drives fingerprint, quality card,
                        # peak_ppm, ppm_diff, observed_j_hz, delta_j_hz.
                        _score_r = dict(entry)
                        _score_r['peak_ppm'] = multiplet_centre
                        _ms_main, _md_main, _pl_main = _multiplet_score(
                            _score_r, local_peaks, tolerance, mhz,
                            ppm_array=(_ppm_arr if has_spec else None),
                            intensity_array=(_mean_int if has_spec else None))
                        predicted_lines = (_pl_main if _pl_main else
                                           [{'ppm': float(s), 'rel_intensity': float(ri)}
                                            for s, ri in shifted_sticks])
                        if _pl_main:
                            _srt = sorted(_pl_main, key=lambda l: l['ppm'])
                            _tot_ri = (sum(l.get('rel_intensity', 1.0) for l in _srt)
                                       or 1.0)
                            _scorer_centre = (
                                sum(l['ppm'] * l.get('rel_intensity', 1.0)
                                    for l in _srt) / _tot_ri)
                            if len(_srt) >= 2:
                                _seps_hz = [
                                    (_srt[i + 1]['ppm'] - _srt[i]['ppm']) * mhz
                                    for i in range(len(_srt) - 1)]
                                _j_obs_scorer = sum(_seps_hz) / len(_seps_hz)
                                _dj_scorer = (float(abs(_j_obs_scorer - j_hz))
                                              if j_hz > 0 else None)
                        else:
                            _scorer_centre = multiplet_centre

                    obs_j_hz   = None
                    delta_j_hz = None
                    if not is_singlet and j_hz > 0:
                        peaks_in_win = [
                            p for p in peaks_to_search
                            if win_lo <= p['ppm'] <= win_hi]
                        obs_j_hz, delta_j_hz = _observe_j_hz(
                            peaks_in_win, j_hz, mhz, n_lns)

                    r = dict(entry)
                    r['fitting_r2']     = float(fitting_r2)
                    r['composite_score']= float(composite)
                    r['multiplet_score']= float(fitting_r2)
                    r['ppm_score']      = float(ppm_score_val)
                    r['mult_compat']    = float(mult_compat)
                    r['j_compat']       = float(j_compat)
                    r['db_ppm']         = float(db_ppm)
                    _eff_centre = (_scorer_centre if _scorer_centre is not None
                                   else multiplet_centre)
                    r['ppm_diff']       = float(abs(_eff_centre - db_ppm))
                    r['peak_ppm']       = _eff_centre
                    r['peak_rank']      = peak_rank
                    r['is_sub_peak']    = bool(nearest.get('is_sub_peak', False))
                    r['predicted_lines']= predicted_lines
                    r['observed_j_hz']  = (float(_j_obs_scorer)
                                           if _j_obs_scorer is not None
                                           else (float(obs_j_hz)
                                                 if obs_j_hz is not None else None))
                    r['delta_j_hz']     = (float(_dj_scorer)
                                           if _dj_scorer is not None
                                           else (float(delta_j_hz)
                                                 if delta_j_hz is not None else None))
                    r['j_coupling']     = j_raw
                    r['baseline_corrected'] = has_spec
                    r['multiplet_detail']   = (
                        _md_main if not nearest.get('is_sub_peak') else '')
                    r['observed_mult']      = obs_mult_pk
                    r['coherence_score']    = round(_coh, 3)
                    r['peak_coverage']      = round(_cov, 3)
                    r['ratio_match']        = round(_rm,  3)
                    r['stocsy_coherence']   = (round(_stocsy_coh, 3)
                                               if _stocsy_coh is not None
                                               else None)
                    _db_src      = r.get('database', 'HMDB')
                    _src_wt      = _SOURCE_WEIGHTS.get(_db_src, 0.70)
                    r['composite_score']  = round(
                        float(r['composite_score']) * _src_wt, 4)
                    r['source_weight']    = _src_wt
                    # Map detected peak to which predicted line it corresponds to.
                    # Stored as line_idx so _effective_nh can use it directly
                    # without positional estimation at quantification time.
                    _pl_li = r.get('predicted_lines') or []
                    if len(_pl_li) > 1:
                        _det_ppm_li = float(r.get('peak_ppm') or 0)
                        r['line_idx'] = min(
                            range(len(_pl_li)),
                            key=lambda _i: abs(float(_pl_li[_i].get('ppm', 0)) - _det_ppm_li))
                    else:
                        r['line_idx'] = 0
                    if r.get('ppm_diff', float('inf')) > tolerance:
                        continue
                    all_results.append(r)

                    if (i + 1) % 100 == 0 or i == total_entries - 1:
                        _el = time.monotonic() - t0
                        self.app.root.after(
                            0,
                            lambda _i=i + 1, _t=total_entries,
                                   _m=len(all_results),
                                   _pm=len(peaks_matched), _e=_el:
                            self._update_progress(_i, _t, _m, _pm, _e))

                seen = {}
                for r in all_results:
                    key = (
                        r.get('hmdb_id') or r.get('bmrb_id', ''),
                        round(float(
                            r.get('ppm_center') or r.get('ppm') or 0),
                            3))
                    if (key not in seen
                            or r['fitting_r2']
                            > seen[key]['fitting_r2']):
                        seen[key] = r
                all_results = list(seen.values())

                all_results.sort(
                    key=lambda r: r.get('fitting_r2', 0), reverse=True)

                if carry_results:
                    all_results = carry_results + all_results

                total_peaks = len(all_peaks)
                elapsed = time.monotonic() - t0
                nm  = len(all_results)
                np_ = len(peaks_matched)
                self.app.root.after(
                    0, lambda: self._status_ok(
                        f'Search complete — {nm} matches '
                        f'for {np_}/{total_peaks} peaks  '
                        f'({elapsed:.2f}s)'))
                self.app.root.after(0, lambda: self._open_results_window(
                    all_results, pw, total_peaks, db_path, elapsed))

            except Exception as e:
                self.app.root.after(
                    0, lambda: self._status_err(f'Search failed: {e}'))
                self.app.root.after(
                    0, lambda: self._handle_search_error(pw, e))
                import traceback; traceback.print_exc()
            finally:
                if conn:
                    conn.close()

        threading.Thread(target=search_task, daemon=True).start()

    # ── MCR recompute ─────────────────────────────────────────────────────────

    def _rerun_mcr_for_peak(self, peak, r2_threshold):
        """Re-run MCR-ALS on a single peak with a new R² threshold."""
        mcr_ann = peak.get('mcr_annotation')
        if not mcr_ann:
            return
        try:
            from gui.utils.mcr_utils import (
                mcr_annotation_region, simulate_lorentzian_profile)
            ppm_reg  = mcr_ann['ppm_region']
            cands    = mcr_ann['candidates']
            _all_int = getattr(self.app, 'all_intensities', None)
            _ppm_full = getattr(self.app, 'ppm', None)
            if _ppm_full is None or _all_int is None:
                return
            mask = _adaptive_mcr_mask(
                _ppm_full, _all_int, peak, cands, field_mhz=600.0)
            if mask.sum() < 4:
                return
            ppm_reg_new = _ppm_full[mask]
            X_reg = (_all_int[mask, :]
                     if _all_int.ndim == 2
                     else _all_int[mask, np.newaxis])
            from gui.widgets.annotation_results_window import _build_stick_positions
            db_profiles = []
            for c in cands:
                try:
                    dp_ppm = float(c.get('ppm_center') or c.get('ppm') or 0)
                    mult   = c.get('type') or c.get('multiplicity') or 's'
                    try:    nh = int(c.get('protons') or 1)
                    except: nh = 1
                    sticks = _build_stick_positions(dp_ppm, mult, nh,
                                                    c.get('j_coupling'))
                    if not sticks:
                        sticks = [(dp_ppm, 1.0)]
                    prof = simulate_lorentzian_profile(sticks, ppm_reg_new)
                except Exception:
                    prof = np.zeros(len(ppm_reg_new))
                db_profiles.append(prof)
            S, C, mcr_m, K, assignments = mcr_annotation_region(
                X_reg, db_profiles, r2_threshold=r2_threshold)
            peak['mcr_annotation'].update({
                'S': S, 'C': C, 'metrics': mcr_m, 'K': K,
                'assignments': assignments,
                'ppm_region': ppm_reg_new,
                'r2_threshold': r2_threshold,
            })
        except Exception as e:
            print(f'_rerun_mcr_for_peak error: {e}')

    # ── Progress update ───────────────────────────────────────────────────────

    def _update_progress(self, processed, total, matches, peaks_matched, elapsed):
        try:
            # Phase 1 occupies 0→55% of the unified bar
            self.progress_bar['value'] = int(processed / max(total, 1) * 55)
            self.progress_op_label.config(
                text=f'Searching — {processed}/{total} entries  ·  {matches} match(es)')
            self.progress_time_label.config(text=f'{elapsed:.1f} s')
        except Exception:
            pass

    def _handle_search_error(self, pw, error):
        try:
            pw.destroy()
        except Exception:
            pass
        messagebox.showerror('Search Error', str(error))

    # ── Open results window ───────────────────────────────────────────────────

    def _open_results_window(self, results, pw, total_peaks, db_path, elapsed):
        if not results:
            try:
                pw.destroy()
            except Exception:
                pass
            messagebox.showinfo('No Results',
                'No matches found.\n'
                'Try increasing the tolerance or checking the DB file.')
            return

        self._last_results_args = {
            'results':    results,
            'db_path':    db_path,
            'total_peaks': total_peaks,
            'elapsed':    elapsed,
        }

        try:
            self.app.btn_annot_shortcut.config(state='normal')
        except Exception:
            pass
        try:
            self.app.btn_deconv_shortcut.config(state='normal')
        except Exception:
            pass

        # Destroy any previous results window — this is a fresh search run.
        _existing = getattr(self.app, '_annot_win', None)
        if _existing is not None:
            try:
                if _existing.win.winfo_exists():
                    _existing.win.destroy()
            except Exception:
                pass
            self.app._annot_win = None

        # ── Phase 2: reuse pw for build progress (55→100%) ───────────────────
        _build_start = 55
        try:
            pw.title('nMRPipe — Building Results…')
            self.progress_bar['value'] = _build_start
            self.progress_op_label.config(text='Preparing results…')
            self.progress_time_label.config(text='')
            pw.update_idletasks()
        except Exception:
            pass

        run_mcr = getattr(self, 'mcr_run_var', None)
        if run_mcr is not None and run_mcr.get():
            try:
                mhz = float(self.spectrometer_var.get())
            except Exception:
                mhz = 600.0
            try:
                self.progress_op_label.config(text='Running MCR deconvolution…')
                self.progress_bar['value'] = 57
                pw.update_idletasks()
            except Exception:
                pass
            n_mcr = self._run_mcr_on_results(results, mhz)
            _build_start = 62
            try:
                self.progress_bar['value'] = _build_start
                self.progress_op_label.config(
                    text=f'MCR: {n_mcr} region(s) — building results…')
                pw.update_idletasks()
            except Exception:
                pass

        # progress_cb: steps 1-11 map from _build_start→95%; UI build → 100%
        def _progress_cb(step, total, msg):
            try:
                pct = _build_start + int(step / max(total, 1) * (95 - _build_start))
                self.progress_bar['value'] = pct
                self.progress_op_label.config(text=msg)
                pw.update_idletasks()
            except Exception:
                pass

        from gui.widgets.annotation_results_window import AnnotationResultsWindow
        _win = AnnotationResultsWindow(
            self.app.root,
            results,
            self.peak_stage.peaks,
            getattr(self.app, 'ppm', None),
            getattr(self.app, 'all_intensities', None),
            db_path=db_path,
            app=self.app,
            saved_quality=dict(self._quality_marks),
            progress_cb=_progress_cb)

        try:
            self.progress_bar['value'] = 100
            pw.update_idletasks()
        except Exception:
            pass
        try:
            pw.destroy()
        except Exception:
            pass

        self.app._annot_win = _win
        # Hide (withdraw) on close — keeps computed state so re-open is instant.
        self.app._annot_win.win.protocol(
            'WM_DELETE_WINDOW',
            lambda: _win.win.withdraw())
        self._metabolites = _win.metabolites

    # ── Built-in library search ───────────────────────────────────────────────

    def search_builtin_library(self):
        """Search the built-in NMR metabolite library without a file picker."""
        import pandas as pd

        mode = 'Metabolomics (default)'
        try:
            mode = str(getattr(self, 'internal_library_mode_var').get())
        except Exception:
            pass
        is_lipidomics = (mode == 'Lipidomics')

        if not self.peaks and not is_lipidomics:
            messagebox.showinfo('Info',
                'No peaks detected.\nRun Peak Detection first.')
            return

        _root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if mode == 'Lipidomics':
            lib_name = 'lipidomics_simple_library.csv'
            lib_label = 'Internal Lipidomics DB'
            prog_title = 'Searching Lipidomics Library'
            prog_text = 'Searching built-in lipidomics library…'
        else:
            lib_name = 'nmr_metabolite_library.csv'
            lib_label = 'Internal Library'
            prog_title = 'Searching Internal Library'
            prog_text = 'Searching built-in NMR metabolite library…'

        lib_path = os.path.join(_root, 'data', lib_name)
        if not os.path.isfile(lib_path):
            messagebox.showerror('Error',
                f'Built-in library not found:\n{lib_path}')
            return

        tolerance = float(self.tolerance_var.get())
        good_peaks = [p for p in self.peaks if p.get('rank') is not None]
        if (not is_lipidomics) and (not good_peaks):
            messagebox.showinfo('No Peaks',
                'No detected peaks found.\n\n'
                'Run Peak Detection first.')
            return

        self.hmdb_status.config(text='Internal DB: searching…', foreground='blue')
        self.app.root.update()

        pw = tk.Toplevel(self.app.root)
        pw.title(prog_title)
        pw.geometry('420x120')
        pw.transient(self.app.root)
        pw.grab_set()
        ttk.Label(pw, text=prog_text,
                  font=('Arial', 10)).pack(pady=10)
        pb = ttk.Progressbar(pw, length=360, mode='indeterminate')
        pb.pack(pady=10); pb.start(); pw.update()

        def _search():
            try:
                df = pd.read_csv(lib_path)
                all_results, peaks_matched = [], set()
                t0 = time.monotonic()
                if is_lipidomics:
                    # Lipidomics mode: match by full region overlap and keep one best
                    # detected peak per diagnostic region (not per single line).
                    for _, row in df.iterrows():
                        r_start = float(row['start_ppm'])
                        r_end = float(row['end_ppm'])
                        r_lo = min(r_start, r_end)
                        r_hi = max(r_start, r_end)
                        center = (r_start + r_end) / 2.0

                        overlapped = []
                        for peak in good_peaks:
                            p_lo = min(peak.get('start_ppm', peak['ppm']),
                                       peak.get('end_ppm',   peak['ppm'])) - tolerance
                            p_hi = max(peak.get('start_ppm', peak['ppm']),
                                       peak.get('end_ppm',   peak['ppm'])) + tolerance
                            if r_lo <= p_hi and r_hi >= p_lo:
                                overlapped.append(peak)
                        if not overlapped:
                            continue

                        peak = min(overlapped, key=lambda p: abs(float(p.get('ppm', 0)) - center))
                        mult = str(row.get('multiplicity', '') or '')
                        signal = str(row.get('signal', '') or '').strip()
                        note = str(row.get('comment', '') or '').strip()
                        raw_id = row.get('id', '')
                        try:
                            row_id = str(int(raw_id))
                        except Exception:
                            row_id = str(raw_id)

                        all_results.append({
                            'name':          str(row.get('name', 'Unknown')),
                            'id':            row_id,
                            'ppm':           center,
                            'ppm_center':    center,
                            'start_ppm':     r_start,
                            'end_ppm':       r_end,
                            'multiplicity':  mult or 'm',
                            'match_type':    signal or 'lipid_region',
                            'type':          signal or 'lipid_region',
                            'database':      lib_label,
                            'source':        lib_label,
                            'protons':       1,
                            'peak_ppm':      peak['ppm'],
                            'peak_rank':     peak['rank'],
                            'composite_score': 0.0,
                            'multiplet_score': 0.0,
                            'multiplet_detail': note,
                            'is_lipidomics': True,
                            'predicted_lines': [
                                {'ppm': r_start, 'rel_intensity': 0.5},
                                {'ppm': r_end, 'rel_intensity': 0.5},
                            ],
                        })
                        peaks_matched.add(peak['rank'])
                    if not good_peaks:
                        # If no detected peaks exist, still keep regions for simple integration.
                        all_results = []
                        for idx, row in df.iterrows():
                            r_start = float(row['start_ppm'])
                            r_end = float(row['end_ppm'])
                            center = (r_start + r_end) / 2.0
                            mult = str(row.get('multiplicity', '') or '')
                            signal = str(row.get('signal', '') or '').strip()
                            note = str(row.get('comment', '') or '').strip()
                            raw_id = row.get('id', '')
                            try:
                                row_id = str(int(raw_id))
                            except Exception:
                                row_id = str(raw_id)
                            all_results.append({
                                'name':          str(row.get('name', 'Unknown')),
                                'id':            row_id,
                                'ppm':           center,
                                'ppm_center':    center,
                                'start_ppm':     r_start,
                                'end_ppm':       r_end,
                                'multiplicity':  mult or 'm',
                                'match_type':    signal or 'lipid_region',
                                'type':          signal or 'lipid_region',
                                'database':      lib_label,
                                'source':        lib_label,
                                'protons':       1,
                                'peak_ppm':      center,
                                'peak_rank':     -(idx + 1),
                                'composite_score': 0.0,
                                'multiplet_score': 0.0,
                                'multiplet_detail': note,
                                'is_lipidomics': True,
                                'predicted_lines': [
                                    {'ppm': r_start, 'rel_intensity': 0.5},
                                    {'ppm': r_end, 'rel_intensity': 0.5},
                                ],
                            })
                else:
                    for peak in good_peaks:
                        p_lo = min(peak.get('start_ppm', peak['ppm']),
                                   peak.get('end_ppm',   peak['ppm'])) - tolerance
                        p_hi = max(peak.get('start_ppm', peak['ppm']),
                                   peak.get('end_ppm',   peak['ppm'])) + tolerance
                        mask = (df['start_ppm'] <= p_hi) & (df['end_ppm'] >= p_lo)
                        for _, row in df[mask].iterrows():
                            mult = str(row.get('multiplicity', '') or '')
                            signal = str(row.get('signal', '') or '').strip()
                            note = str(row.get('comment', '') or '').strip()
                            raw_id = row.get('id', '')
                            try:
                                row_id = str(int(raw_id))
                            except Exception:
                                row_id = str(raw_id)
                            all_results.append({
                                'name':          str(row.get('name', 'Unknown')),
                                'id':            row_id,
                                'ppm':           (row['start_ppm'] + row['end_ppm']) / 2.0,
                                'ppm_center':    (row['start_ppm'] + row['end_ppm']) / 2.0,
                                'start_ppm':     float(row['start_ppm']),
                                'end_ppm':       float(row['end_ppm']),
                                'multiplicity':  mult,
                                'match_type':    signal or mult or 'internal_library',
                                'type':          signal or mult or 'internal_library',
                                'database':      lib_label,
                                'source':        lib_label,
                                'protons':       1,
                                'peak_ppm':      peak['ppm'],
                                'peak_rank':     peak['rank'],
                                'composite_score': 0.0,
                                'multiplet_score': 0.0,
                                'multiplet_detail': note,
                                'is_lipidomics': False,
                            })
                            peaks_matched.add(peak['rank'])
                elapsed = time.monotonic() - t0
                n_m = len(all_results)
                n_p = len(peaks_matched)

                def _show():
                    if is_lipidomics:
                        self.hmdb_status.config(
                            text=f'Internal DB: {n_m} lipid regions loaded',
                            foreground='green')
                    else:
                        self.hmdb_status.config(
                            text=f'Internal DB: {n_m} matches ({n_p}/{len(good_peaks)} peaks)',
                            foreground='green')
                    self._open_results_window(all_results, pw, len(good_peaks), None, elapsed)
                    if self._last_results_args is not None:
                        self._last_results_args['internal_library_mode'] = mode

                self.app.root.after(0, _show)
            except Exception as exc:
                self.app.root.after(0, pw.destroy)
                self.app.root.after(
                    0, lambda: messagebox.showerror('Error',
                        f'Library search failed:\n{exc}'))
                import traceback; traceback.print_exc()

        threading.Thread(target=_search, daemon=True).start()

    # ── Online search ─────────────────────────────────────────────────────────

    def search_hmdb_online(self):
        if not self.peaks:
            messagebox.showinfo('Info',
                'No peaks detected.\nRun Peak Detection first.')
            return
        messagebox.showinfo('BMRB Online',
            'Online search not implemented in this version.\n'
            'Use Search Internal DB with a BMRB .db file.')

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, results):
        fn = filedialog.asksaveasfilename(
            title='Save Results',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')],
            initialfile='annotation_results.csv')
        if not fn:
            return
        try:
            with open(fn, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['ID', 'Name', 'Match Type', 'DB PPM',
                            'Peak PPM', 'Peak Rank', 'PPM Score',
                            'Multiplet Score', 'Composite Score',
                            'Multiplet Detail'])
                for r in results:
                    w.writerow([
                        r.get('hmdb_id', r.get('bmrb_id', '')),
                        r.get('name', ''),
                        r.get('match_type', ''),
                        f"{r.get('ppm_center', r.get('ppm', 0)):.4f}",
                        f"{r.get('peak_ppm', 0):.4f}",
                        r.get('peak_rank', ''),
                        f"{r.get('ppm_score', 0):.3f}",
                        f"{r.get('multiplet_score', 0):.3f}",
                        f"{r.get('composite_score', 0):.3f}",
                        r.get('multiplet_detail', ''),
                    ])
            messagebox.showinfo('Export', f'Saved:\n{fn}')
        except Exception as e:
            messagebox.showerror('Export Error', str(e))
