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
    _ppm_score, _multiplet_score,
    _baseline_correct_linear, _fit_r2,
    _mult_compat_check, _adaptive_mcr_mask,
    _ask_search_mode, _pick_database,
    _multi_peak_coherence,
    _stocsy_coherence,
)
from processing.annotation_scoring import (
    signal_score, dj_score, peak_shape_score, SCORING_DEFAULTS,
    measure_multiplet, expected_line_count, search_multiplet_in_window)
from processing.signal_quality import (
    SIGNAL_GOOD, SIGNAL_POOR, effective_tier, filter_annotatable,
    is_annotatable)

# Everything an MCR pass writes onto a result row. Listed once so a run with
# MCR switched off cannot leave a single field of the previous one behind.
MCR_RESULT_KEYS = (
    'mcr_r2', 'mcr_r2_pooled', 'mcr_r2_per_sample', 'mcr_confirmed',
    'mcr_pearson_r', 'mcr_delta_j_hz', 'mcr_shift_ppm', 'mcr_cross_corr',
    'mcr_informative_only', 'mcr_info_reason', 'mcr_shift_disagree_ppm',
)


def clear_stale_mcr(peaks, results=None):
    """Drop MCR state left by a previous search.

    A new search invalidates any earlier MCR result: the candidate set that
    produced it no longer exists. Nothing else clears it and the peak dicts
    outlive the search, so with MCR switched off the results window went on
    showing components and verdicts from the run before.

    Peaks created BY Full-Spectrum MCR are exempt: for them the annotation is
    derived from ``mcr_annotation``, not the other way round, so removing it
    would destroy the peak's own reason for existing.

    Returns (n_peaks_cleared, n_results_cleared).
    """
    n_peaks = 0
    for p in (peaks or []):
        if p.get('mcr_annotation') is not None and not p.get('_from_mcr'):
            p.pop('mcr_annotation', None)
            n_peaks += 1

    n_results = 0
    for r in (results or []):
        if any(k in r for k in MCR_RESULT_KEYS):
            for k in MCR_RESULT_KEYS:
                r.pop(k, None)
            n_results += 1
    return n_peaks, n_results


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
            messagebox.showinfo('Info', 'No signals detected.')
            return
        if not self._last_results_args:
            messagebox.showinfo('Info', 'Run annotation first, then rescue unassigned signals.')
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
        good_peaks = self._exclude_water_peaks(
            [p for p in self.peaks
             if p.get('status') == SIGNAL_GOOD
             and is_annotatable(p, self.annotatable_tiers())])
        unassigned = [p for p in good_peaks if int(p.get('rank', -1)) not in already_matched]
        if not unassigned:
            messagebox.showinfo('Rescue', 'No unassigned Good signals to rescue.')
            return

        # Mirror the main search alignment: shift DB entries by the in-app
        # DSS/TSP calibration delta and keep detected peaks in their raw frame.
        # Previously this used _align_peaks_to_reference (peak-shifting), which
        # diverged from the main pass and annotated the same DB at different ppm.
        _cal = getattr(self.app, '_ppm_calibration', None)
        _cal_delta = float(_cal['delta_mean']) if _cal else 0.0
        # Discard stale/corrupt calibration (the in-window logic caps each
        # anchor at Δ < 0.02 ppm; > ±0.05 indicates accumulated/broken delta).
        if abs(_cal_delta) > 0.05:
            _cal_delta = 0.0
        if _cal is None:
            self._status(
                '⚠ No DSS/TSP calibration stored — PPM axis may be '
                'misaligned. Check spectrum calibration.'
            )
        aligned_good = [dict(p) for p in good_peaks]
        aligned_unassigned = [p for p in aligned_good if int(p.get('rank', -1)) in
                              {int(u.get('rank', -1)) for u in unassigned}]

        self._status(f'Rescue pass: scanning {len(aligned_unassigned)} unassigned signals…')
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
                # Detected peaks stay raw; query in the DB frame (= peak frame
                # minus the calibration delta) and shift returned rows by
                # +_cal_delta so scoring compares both in the peak frame.
                _dbf = ppm - _cal_delta
                q_lo = p_lo - _cal_delta
                q_hi = p_hi - _cal_delta

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
                        " ORDER BY ABS(s.ppm - ?)",
                        (ppm, peak['rank'], _dbf, q_lo, q_hi, _dbf))
                    rows = [dict(r) for r in cur.fetchall()]
                    for r in rows:
                        r['match_type'] = 'shift'
                        r['database'] = 'BMRB'
                        r['shift_ppm'] = _cal_delta
                elif is_hmdb:
                    cur.execute("""
                        SELECT m.*, m.j_coupling, met.name,
                               ? as peak_ppm, ? as peak_rank,
                               ABS(m.ppm_center - ?) as ppm_diff
                        FROM multiplets m
                        JOIN metabolites met ON m.hmdb_id = met.hmdb_id
                        WHERE m.ppm_center BETWEEN ? AND ?
                        ORDER BY ABS(m.ppm_center - ?)
                    """, (ppm, peak['rank'], _dbf, q_lo, q_hi, _dbf))
                    rows = [dict(r) for r in cur.fetchall()]
                    for r in rows:
                        r['match_type'] = 'multiplet'
                        r['database'] = 'HMDB'
                        r['shift_ppm'] = _cal_delta
                # Bring DB ppm into the peak frame for downstream scoring.
                if _cal_delta:
                    for r in rows:
                        for _k in ('ppm_center', 'ppm'):
                            if r.get(_k) is not None:
                                r[_k] = float(r[_k]) + _cal_delta

                _resc_params = dict(SCORING_DEFAULTS)
                _resc_params['dppm_gold'] = relax_tol
                _resc_params.update(
                    getattr(self, '_annotation_params', None) or {})
                for r in rows:
                    ps = _ppm_score(r.get('peak_ppm', 0), r.get('ppm_center', r.get('ppm', 0)), relax_tol)
                    ms, md, pl = _multiplet_score(r, aligned_good, relax_tol, mhz,
                                                  ppm_array=_ppm_arr, intensity_array=_int_arr)
                    mult_type = (r.get('multiplicity') or r.get('type') or 's').lower().strip()
                    is_singlet = mult_type in ('s', 'singlet', '')
                    compat, obs_mult = _mult_compat_check(r, aligned_good, relax_tol, mhz)
                    # Same signal_score as the main search and the results
                    # window — this path used to carry its own formula
                    # (0.4·Δppm + 0.6·multiplet_score), a fourth way of
                    # producing a number under the key `composite_score`.
                    # Rescue never measures J, so the ΔJ term is excluded and
                    # the remaining weights renormalise; the multiplet score
                    # plays the shape role. Multiplicity disagreement stays a
                    # multiplicative veto: the model has no term for claiming a
                    # triplet where the spectrum shows a singlet.
                    _resc_bd = signal_score(
                        s_ppm=ps, s_dj=None, s_cosine=ms,
                        is_singlet=is_singlet, params=_resc_params)
                    composite = _resc_bd.value * compat
                    r['observed_mult'] = obs_mult
                    r['mult_compat'] = round(compat, 3)
                    r['ppm_score'] = ps
                    r['multiplet_score'] = ms
                    r['composite_score'] = composite
                    r['_score_breakdown'] = (
                        f'{_resc_bd.equation()}  x mult_compat({compat:.2f})')
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
                    S, C, mcr_m, K, assignments, *_ = mcr_annotation_region(
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

        from gui.utils.mcr_utils import (
            MCR_MIN_OVERLAP_K, MCR_MIN_SHAPE_R, MCR_SHIFT_AGREEMENT_PPM,
            _is_singlet_multiplicity, mcr_annotation_region,
            simulate_lorentzian_profile)
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

            # r2_median: per-sample R² (each sample scored against its own
            # mean), not the pooled 'r2' which uses one grand mean across all
            # samples and can hide a compound that only fits a few of them.
            mcr_r2 = mcr_m.get('r2_median', None)
            if mcr_r2 is None or (isinstance(mcr_r2, float) and np.isnan(mcr_r2)):
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

            for c_idx, c in enumerate(cands):
                # A singlet cannot be identified by component shape. After the
                # +-shift_tol_ppm alignment any isolated line correlates with any
                # other isolated line (r -> 1.0 by construction), so a high r
                # carries no information about which compound it is — and the
                # component may not even be a single line. MCR metrics stay
                # visible for such candidates, but they do not confirm anything.
                _c_mult = (c.get('type') or c.get('multiplicity') or 's')
                _c_is_singlet = _is_singlet_multiplicity(_c_mult)

                _c_assign_pre = next((a for a in assignments if a[0] == c_idx),
                                     None)
                _c_shift = (_c_assign_pre[2]
                            if _c_assign_pre is not None and len(_c_assign_pre) > 2
                            else None)
                # Did MCR work on the signal this match is about? The
                # annotation placed this DB peak at peak_ppm, an offset of
                # (peak_ppm - db_ppm). If MCR's own alignment reproduces that
                # offset it resolved the same signal; if it does not, its
                # metrics describe a different one.
                try:
                    _c_own_delta = (float(c.get('peak_ppm'))
                                    - float(c.get('db_ppm')))
                except (TypeError, ValueError):
                    _c_own_delta = None
                _c_disagree = (abs(float(_c_shift) - _c_own_delta)
                               if (_c_shift is not None
                                   and _c_own_delta is not None) else None)
                _c_far = (_c_disagree is not None
                          and _c_disagree > MCR_SHIFT_AGREEMENT_PPM)
                c['mcr_shift_disagree_ppm'] = _c_disagree

                c['mcr_informative_only'] = bool(_c_is_singlet or _c_far)
                if _c_is_singlet:
                    c['mcr_info_reason'] = 'singlet — shape carries no identity'
                elif _c_far:
                    c['mcr_info_reason'] = (
                        f'resolved a signal {_c_disagree:.4f} ppm '
                        f'away from this match')
                else:
                    c['mcr_info_reason'] = None

                c['mcr_r2']             = mcr_r2
                c['mcr_r2_pooled']      = mcr_m.get('r2', None)
                c['mcr_r2_per_sample']  = mcr_m.get('r2_per_sample', None)
                c['mcr_confirmed']      = (None if c['mcr_informative_only']
                                           else mcr_confirmed)
                c_assign = next((a for a in assignments if a[0] == c_idx), None)
                c['mcr_pearson_r'] = c_assign[1] if c_assign else None

                # Post-deconvolution ΔJ: measured on this candidate's own MCR
                # component (S column), not on the raw/merged detected peak.
                # Only meaningful in genuine overlaps (K>=2) with a decent
                # shape match — otherwise MCR isolated nothing worth trusting
                # over the raw measurement.
                c['mcr_shift_ppm'] = (c_assign[2] if c_assign is not None
                                      and len(c_assign) > 2 else None)

                c['mcr_delta_j_hz'] = None
                if (c_assign is not None and not c['mcr_informative_only']
                        and K >= MCR_MIN_OVERLAP_K
                        and c_assign[1] is not None
                        and c_assign[1] >= MCR_MIN_SHAPE_R):
                    meas_j = c_assign[4] if len(c_assign) > 4 else None
                    exp_j  = c_assign[5] if len(c_assign) > 5 else None
                    if meas_j is not None and exp_j is not None:
                        c['mcr_delta_j_hz'] = float(abs(meas_j - exp_j))

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

    def _exclude_water_peaks(self, peaks):
        """Drop peaks inside the water-suppression exclusion zone.

        Standard 1H-NMR metabolomics practice: the residual water region
        (~4.70–4.90 ppm) is dominated by solvent-suppression artefacts and is
        excluded from analysis. Controlled by the annotation-stage checkbox and
        the editable lo/hi range.
        """
        try:
            if not bool(self.exclude_water_var.get()):
                return peaks
            lo = float(self.water_lo_var.get())
            hi = float(self.water_hi_var.get())
        except Exception:
            return peaks
        if lo > hi:
            lo, hi = hi, lo
        kept = [p for p in peaks
                if not (lo <= float(p.get('ppm', 1e9)) <= hi)]
        _n = len(peaks) - len(kept)
        if _n:
            print(f'[annotation] water exclusion {lo:.2f}-{hi:.2f} ppm: '
                  f'skipped {_n} peak(s)')
        return kept

    def search_internal_db(self):
        if not self.peaks:
            messagebox.showinfo('Info',
                'No signals detected.\nRun Signal Detection first.')
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

        # Poor signals never receive an annotation, but they are still real
        # lines of the spectrum: a Poor line of a doublet must stay available as
        # a J-coupling partner. Dropping it here would leave the multiplet one
        # line short and let the snap pair the survivor with an unrelated
        # neighbour, fabricating a coupling constant. So the pool keeps
        # everything and the exclusion is applied to the anchor peak — the one
        # the annotation is actually attached to.
        all_peaks = self._exclude_water_peaks(
            [p for p in self.peaks if p.get('rank') is not None])
        # A new search invalidates any MCR result from a previous one — the
        # candidate set that produced it no longer exists. Nothing else clears
        # it and the peak dicts outlive the search, so with MCR switched off the
        # results window kept showing components from the run before.
        # Peaks created BY Full-Spectrum MCR are their own MCR result: for them
        # the annotation is the thing derived from mcr_annotation, not the other
        # way round, so those are left untouched.
        _stale_mcr, _ = clear_stale_mcr(self.peaks)
        if _stale_mcr:
            print(f'[annotation] cleared MCR results from {_stale_mcr} peak(s) '
                  f'left by a previous search')

        _tiers = self.annotatable_tiers()
        _annotatable, _rejected = filter_annotatable(all_peaks, _tiers)
        self._poor_skipped = len(_rejected)
        self._annotatable_tiers_used = _tiers
        if _rejected:
            print(f'[annotation] signal quality: skipped {len(_rejected)} '
                  f'signal(s); annotating '
                  f'{"/".join(sorted(_tiers))} only')

        if not _annotatable:
            messagebox.showinfo(
                'No Signals',
                ('No detected signal has an allowed quality grade.\n\n'
                 f'Annotation is set to use {"/".join(sorted(_tiers))} only. '
                 'Widen the selection in Annotation Parameters, review the '
                 'signal quality in the Peak List, or re-run Signal Detection.')
                if all_peaks else
                'No detected signals found.\n\n'
                'Run Signal Detection first.')
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
                    'All signals already have confirmed annotations.\n'
                    'Use "Reset & start over" to redo everything.')
                return
            carry_results = [r for r in prev_results
                             if (int(r['peak_rank']) if r.get('peak_rank') is not None else -1) in already_good_ranks
                             and r.get('ppm_diff', float('inf')) <= tolerance]
            _mcr_on = getattr(self, 'mcr_run_var', None)
            if not (_mcr_on is not None and _mcr_on.get()):
                # Carried rows keep their scores, not a verdict from an MCR pass
                # that is not part of this run.
                clear_stale_mcr(None, carry_results)
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
                        # A sub-peak inherits the grade of the signal it was
                        # deconvolved from unless deconvolution graded it
                        # itself. Stamping every sub-peak 'Good' let a Poor
                        # parent re-enter annotation through its own sub-peaks.
                        _parent_tier = effective_tier(_pk)
                        _subs = [sp for sp in _res.get('peaks', [])
                                 if sp.get('status') != SIGNAL_POOR]
                        if _subs:
                            for _sp in _subs:
                                _pseudo = {
                                    'ppm':         float(_sp['ppm']),
                                    'rank':        _rk,
                                    'status':      (_sp.get('status')
                                                    or _parent_tier),
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
        pw.title('NMRPro-H — Annotating…')
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
            # Line counts come from the scoring core's LINE_COUNTS — one table.
            # This local copy had 'quin' where BMRB writes 'quint', so every
            # quintet, sextet and septet fell to the default of 2 lines.
            _n_lines_map = {'s': 1}
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
                # Conservative thresholds: a noisy shoulder must not split a true
                # singlet into a 'd' (which would wrongly apply the j_compat=0.3
                # singlet penalty downstream). Require a prominent secondary line.
                pks, _ = _fp(y, height=0.25, prominence=0.18, distance=min_dist)
                n = len(pks)
                if n <= 1:
                    return 's', None
                ppms_pk = x[pks]
                seps_hz = [abs(ppms_pk[k+1] - ppms_pk[k]) * field_mhz
                           for k in range(len(ppms_pk) - 1)]
                j_est = float(np.median(seps_hz))
                # J floor: line separations below ~1.5 Hz are not resolvable
                # couplings at metabolomics linewidths — treat as a single line.
                if j_est < 1.5:
                    return 's', None
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
                # Scoring parameters: the user's Annotation Parameters when
                # they exist, the scoring core's defaults otherwise. One owner.
                _sc_params = dict(SCORING_DEFAULTS)
                # The Δppm gate follows the analyst's 'Tolerance (ppm)' — the
                # same number that defined the search window. An explicit
                # dppm_gold in the Parameters dialog still wins.
                _sc_params['dppm_gold'] = tolerance
                _sc_params.update(
                    getattr(self, '_annotation_params', None) or {})

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

                    n_lns       = _n_lines_map.get(
                        mult_raw, expected_line_count(mult_raw, 2))
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
                        # No database J: do not invent line positions.
                        sticks = (_build_stick_positions(
                            db_ppm, mult_raw, protons, j_raw)
                                  if j_hz > 0 else [])
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
                    # Siblings (J-coupling partners) can be any quality; the
                    # anchor cannot — a Poor signal is not annotated.
                    good_in_tol = [p for p in local_peaks
                                   if abs(p['ppm'] - db_ppm) <= tolerance
                                   and is_annotatable(p, _tiers)]
                    if not good_in_tol:
                        # The fallback below covers a signal peak detection
                        # missed entirely. It must not cover one that WAS
                        # detected and then excluded by its quality grade —
                        # rebuilding it from the raw spectrum would annotate
                        # exactly the signal the analyst ruled out.
                        if any(abs(p['ppm'] - db_ppm) <= tolerance
                               for p in local_peaks):
                            continue
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
                        # A spectrum-derived peak was never graded, so it
                        # carries no status and reads as Average. Claiming
                        # 'Good' would let it bypass the grade selection
                        # entirely — the one thing this filter exists to stop.
                        _syn_peak = {
                            'ppm':           float(_syn_x[_syn_max_i]),
                            'rank':          None,
                            'height':        _syn_h,
                            'width_ppm':     0.010,
                            'width_at_half': 0.010,
                            '_synthetic':    True,
                            '_obs_j_hz':     0,
                            '_obs_mult':     mult_raw,
                        }
                        if not is_annotatable(_syn_peak, _tiers):
                            continue
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

                    # A missing database J is intentionally not inferred from
                    # the observed spectrum. The analyst must confirm it.
                    _eff_j_ppm = j_ppm if j_hz > 0 else 0.0

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
                    shape_score_val = None
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
                        # Shape term of the signal score, measured in this
                        # signal's own window — the same function the results
                        # window uses, so both see one number.
                        shape_score_val = peak_shape_score(
                            _ppm_arr, _mean_int, shifted_sticks,
                            multiplet_centre, entry, hwhm_default=_exp_hwhm)

                    # peak_rank is None for spectrum-derived/synthetic peaks;
                    # excluding them keeps the "X/Y peaks matched" count honest
                    # (a single None would otherwise collapse all of them to 1).
                    if peak_rank is not None:
                        peaks_matched.add(peak_rank)

                    if not is_singlet and shifted_sticks:
                        _nearest_stick = min(shifted_sticks,
                                             key=lambda sv: abs(sv[0] - peak_ppm))
                        ppm_score_val = _ppm_score(peak_ppm, _nearest_stick[0], tolerance)
                    else:
                        ppm_score_val = _ppm_score(multiplet_centre, db_ppm, tolerance)
                    obs_mult_pk = nearest.get('_obs_mult', 's')

                    _scorer_centre = None

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
                        # Candidate line positions for the measurement below.
                        # J and the line count are NOT taken from here — a mean
                        # of whatever separations the scorer returned is not a
                        # coupling constant. measure_multiplet owns both.
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
                        else:
                            _scorer_centre = multiplet_centre

                    # ── Line measurement ─────────────────────────────
                    # Snap the DB multiplet's predicted lines onto detected
                    # peaks, count how many were actually found, and measure J
                    # from their real spacings. This decides whether the signal
                    # was identified at all, so it belongs here — before the
                    # score and before the candidate can compete — not in the
                    # results window, where it used to run.
                    #
                    # It replaces _observe_j_hz, which searched the peak
                    # separations for whichever one best matched the DB J. That
                    # is circular: ΔJ came out small by construction, because
                    # the separation minimising it was the one selected. It also
                    # never counted lines, so a doublet with one leg missing
                    # reported a confident J measured from unrelated peaks.
                    _n_exp_sr = max(
                        expected_line_count(mult_raw,
                                            fallback=len(predicted_lines) or 2),
                        n_lns)
                    _meas_sr = measure_multiplet(
                        predicted_lines, peaks_to_search,
                        mult=mult_raw, j_coupling=j_raw,
                        db_ppm=db_ppm, centre_hint=multiplet_centre,
                        n_expected=_n_exp_sr, tolerance=tolerance,
                        field_mhz=mhz)
                    if _meas_sr.measured and not _meas_sr.complete and j_hz > 0:
                        # Lines missing: before giving up, look in the window for
                        # the multiplet's comb — n_exp peaks spaced by J, every
                        # spacing agreeing with the DB J AND with each other.
                        # Line-matching works outward from stored positions and
                        # can miss a multiplet that sits off them; a template
                        # match does not depend on where it was expected.
                        _hw_sr = (j_hz / mhz) * (_n_exp_sr - 1) / 2 + tolerance
                        _comb_sr = search_multiplet_in_window(
                            [p_.get('ppm') for p_ in peaks_to_search],
                            centre_ppm=multiplet_centre, j_hz=j_hz,
                            n_expected=_n_exp_sr, field_mhz=mhz,
                            half_width_ppm=_hw_sr)
                        if _comb_sr:
                            _retry_sr = measure_multiplet(
                                [{'ppm': v, 'rel_intensity': 1.0} for v in _comb_sr],
                                peaks_to_search,
                                mult=mult_raw, j_coupling=j_raw,
                                db_ppm=db_ppm,
                                centre_hint=sum(_comb_sr) / len(_comb_sr),
                                n_expected=_n_exp_sr, tolerance=tolerance,
                                field_mhz=mhz)
                            if _retry_sr.complete:
                                _meas_sr = _retry_sr

                    n_found_sr = _meas_sr.n_found
                    n_exp_sr   = _meas_sr.n_expected
                    obs_j_hz   = _meas_sr.observed_j_hz
                    delta_j_hz = _meas_sr.delta_j_hz
                    if _meas_sr.measured and _meas_sr.predicted_lines:
                        predicted_lines = _meas_sr.predicted_lines
                    if _meas_sr.centre_ppm is not None:
                        _scorer_centre = float(_meas_sr.centre_ppm)

                    # ── Signal score ────────────────────────────────────────
                    # Same function the Annotation Results window uses, so the
                    # number the search produces and the number the user sees
                    # are the same quantity. They used to be two different
                    # formulas under one key: a Δppm-weighted blend here (45%
                    # Δppm, 10% J) and a ΔJ-led one there — the same signal
                    # scored 0.317 in one and 0.972 in the other.
                    #
                    # It is computed ONCE, here, on measured evidence: the
                    # lines have been snapped and J measured just above. The
                    # results window does not recompute it — a view shows
                    # results, it does not produce them.
                    _dj_sig = (dj_score(delta_j_hz,
                                        _sc_params['dj_gold_hz'],
                                        _sc_params['dj_bad_hz'],
                                        _sc_params['dj_neutral'])
                               if (not is_singlet and delta_j_hz is not None)
                               else None)
                    _shape_sig = (float(shape_score_val)
                                  if shape_score_val is not None else None)
                    _sig_bd = signal_score(s_ppm=ppm_score_val,
                                           s_dj=_dj_sig,
                                           s_cosine=_shape_sig,
                                           is_singlet=is_singlet,
                                           params=_sc_params)
                    composite = _sig_bd.value

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


                    r = dict(entry)
                    r['fitting_r2']     = float(fitting_r2)
                    r['composite_score']= float(composite)
                    r['_score_breakdown'] = _sig_bd.equation()
                    r['multiplet_score']= float(fitting_r2)
                    r['ppm_score']      = float(ppm_score_val)
                    r['db_ppm']         = float(db_ppm)
                    _eff_centre = (_scorer_centre if _scorer_centre is not None
                                   else multiplet_centre)
                    r['ppm_diff']       = float(abs(_eff_centre - db_ppm))
                    r['peak_ppm']       = _eff_centre
                    r['peak_rank']      = peak_rank
                    r['is_sub_peak']    = bool(nearest.get('is_sub_peak', False))
                    r['predicted_lines']= predicted_lines
                    # Single source: measure_multiplet above. It reports None
                    # when lines are missing, and None must stay None — a ΔJ
                    # measured on a different set of peaks is not this signal's.
                    r['observed_j_hz']  = (float(obs_j_hz)
                                           if obs_j_hz is not None else None)
                    r['delta_j_hz']     = (float(delta_j_hz)
                                           if delta_j_hz is not None else None)
                    r['_n_found_lines'] = n_found_sr
                    r['_n_exp_lines']   = n_exp_sr
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
                    # Database provenance is recorded, not multiplied into the
                    # score. Scaling by 0.82 (BMRB) meant a signal needed a raw
                    # 0.915 to reach the 0.75 Good threshold, and the Annotation
                    # Results window never applied the same factor — so the same
                    # signal read 0.317 here and 0.972 there. Provenance is a
                    # property of the reference, not evidence about this peak.
                    _db_src      = r.get('database', 'HMDB')
                    r['source_weight']    = _SOURCE_WEIGHTS.get(_db_src, 0.70)
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
                    # Dedup/rank by composite_score (single source of truth),
                    # not fitting_r2 — otherwise the prefilter could drop the
                    # candidate the window would score best (filter-by-A/sort-by-B).
                    if (key not in seen
                            or r.get('composite_score', 0)
                            > seen[key].get('composite_score', 0)):
                        seen[key] = r
                all_results = list(seen.values())

                all_results.sort(
                    key=lambda r: r.get('composite_score', 0), reverse=True)

                if carry_results:
                    all_results = carry_results + all_results

                total_peaks = len(_annotatable)
                elapsed = time.monotonic() - t0
                nm  = len(all_results)
                np_ = len(peaks_matched)
                # Report the Poor signals left out so the denominator above is
                # never mistaken for the full peak list.
                _skip = getattr(self, '_poor_skipped', 0)
                _tier_txt = '/'.join(sorted(_tiers))
                _skip_txt = (f'  ·  {_skip} signal(s) excluded by quality'
                             if _skip else '')
                self.app.root.after(
                    0, lambda: self._status_ok(
                        f'Search complete — {nm} matches '
                        f'for {np_}/{total_peaks} {_tier_txt} peaks  '
                        f'({elapsed:.2f}s){_skip_txt}'))
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
            # Use the Field Strength entered in the annotation panel, not a
            # hardcoded 600 MHz — the adaptive mask width is field-dependent.
            try:
                mhz = float(self.spectrometer_var.get())
            except Exception:
                mhz = 600.0
            mask = _adaptive_mcr_mask(
                _ppm_full, _all_int, peak, cands, field_mhz=mhz)
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
            S, C, mcr_m, K, assignments, *_ = mcr_annotation_region(
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
            pw.title('NMRPro-H — Building Results…')
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
                'No signals detected.\nRun Signal Detection first.')
            return

        from app_paths import asset_path
        solvent = None
        if mode == 'Lipidomics':
            from processing import lipidomics as _lp
            solvent = self.lipid_solvent() if hasattr(self, 'lipid_solvent') \
                else _lp.DEFAULT_SOLVENT
            lib_name = _lp.library_filename(solvent)
            lib_label = f'Internal Lipidomics DB ({solvent})'
            prog_title = 'Searching Lipidomics Library'
            prog_text = f'Integrating built-in lipidomics regions ({solvent})…'
        else:
            lib_name = 'nmr_metabolite_library.csv'
            lib_label = 'Internal Library'
            prog_title = 'Searching Internal Library'
            prog_text = 'Searching built-in NMR metabolite library…'

        lib_path = asset_path('data', lib_name)
        if not os.path.isfile(lib_path):
            messagebox.showerror('Error',
                f'Built-in library not found:\n{lib_path}')
            return

        tolerance = float(self.tolerance_var.get())
        good_peaks = self._exclude_water_peaks(
            [p for p in self.peaks if p.get('rank') is not None])
        if (not is_lipidomics) and (not good_peaks):
            messagebox.showinfo('No Signals',
                'No detected signals found.\n\n'
                'Run Signal Detection first.')
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
                    # Lipidomics is pure region integration: integrate EVERY library
                    # region, independent of detected peaks (no peak picking, no
                    # selection). Regions and library are owned by processing/lipidomics.
                    from processing import lipidomics as _lp
                    for ridx, reg in enumerate(_lp.build_regions(df)):
                        all_results.append({
                            'name':          reg['name'],
                            'id':            reg['id'],
                            'ppm':           reg['ppm'],
                            'ppm_center':    reg['ppm'],
                            'start_ppm':     reg['start_ppm'],
                            'end_ppm':       reg['end_ppm'],
                            'multiplicity':  'm',
                            'match_type':    reg['signal'] or 'lipid_region',
                            'type':          reg['signal'] or 'lipid_region',
                            'database':      lib_label,
                            'source':        lib_label,
                            'protons':       1,
                            'peak_ppm':      reg['ppm'],
                            'peak_rank':     -(ridx + 1),
                            'composite_score': 0.0,
                            'multiplet_score': 0.0,
                            'multiplet_detail': reg['comment'],
                            'is_lipidomics': True,
                            'lipid_solvent': solvent,
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
                            text=f'Internal DB: {n_m} matches ({n_p}/{len(good_peaks)} signals)',
                            foreground='green')
                    self._open_results_window(all_results, pw, len(good_peaks), None, elapsed)
                    if self._last_results_args is not None:
                        self._last_results_args['internal_library_mode'] = mode
                        self._last_results_args['lipid_solvent'] = solvent

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
                'No signals detected.\nRun Signal Detection first.')
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
                            'Signal PPM', 'Signal Rank', 'PPM Score',
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
