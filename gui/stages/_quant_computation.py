# gui/stages/_quant_computation.py
"""
_QuantCompMixin — computation methods for QuantificationStage.

Extracted from quantification_stage.py to isolate:
  • _run          — full quantification pipeline
  • _compute_vip_scores — PLS-DA VIP
  • update_sample_area  — per-sample manual region update
  • _refresh_tree_row   — single tree row refresh
  • _export             — CSV / MetaboAnalyst export
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import csv
import os

# ── Debug logging ─────────────────────────────────────────────────────────────
# Set to True to write a detailed quant_debug.log in the project root.
_QUANT_DEBUG = True
_QUANT_DEBUG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'quant_debug.log')

def _dbg_open():
    if not _QUANT_DEBUG:
        return None
    return open(_QUANT_DEBUG_PATH, 'w', encoding='utf-8')

def _dbg(f, *args):
    if f is None:
        return
    print(*args, file=f)
# ─────────────────────────────────────────────────────────────────────────────


def _per_sample_snr(p, all_int, n_s):
    """Per-sample SNR at a detected peak's integration window.

    Integration in the quant pipeline uses ONE fixed window per analyte for
    every sample.  In a sample where the analyte is weak/absent the window
    still integrates noise, so a reported area/concentration on its own cannot
    distinguish "present but small" from "not there".  This returns the SNR of
    the peak window in EACH sample (length ``n_s``) so callers can flag samples
    below the detection limit instead of silently reporting noise as signal.

    Returns
    -------
    (snr_list, noise_lvl)
        ``snr_list`` has one float per sample (NaN when noise is unknown);
        ``noise_lvl`` is the absolute noise level used (or None).
    """
    if p is None or all_int is None or n_s <= 0:
        return [float('nan')] * max(n_s, 0), None
    lo_i = min(p.get('start_idx', 0), p.get('end_idx', 0))
    hi_i = max(p.get('start_idx', 0), p.get('end_idx', 0))
    noise_lvl = p.get('noise', None)
    if (not noise_lvl or noise_lvl <= 0) and p.get('snr', 0) > 0:
        peak_max_all = float(np.max(np.mean(all_int[lo_i:hi_i + 1, :], axis=1)))
        noise_lvl = peak_max_all / p['snr'] if p['snr'] else None
    if not noise_lvl or noise_lvl <= 0:
        return [float('nan')] * n_s, noise_lvl
    snr_list = [float(np.max(all_int[lo_i:hi_i + 1, s])) / noise_lvl
                for s in range(n_s)]
    return snr_list, noise_lvl
# ─────────────────────────────────────────────────────────────────────────────

from ._quant_helpers import (
    _trapz, _pls_vip, _cv, _pqn, _effective_nh, _effective_nh_source,
    _mcr_als, _fit_lorentz_mean, _fit_lorentz_mean_with_bl,
    _fit_lorentz_sample, _fit_multi_lorentz,
    _align_sample_center, _lorentz_amplitude, _lorentz_shape,
    bootstrap_fraction_ci as _bootstrap_fraction_ci,
    jr_pointwise_multiplet_area,
)
from processing.multiplet_line_selector import select_clean_lines


class _QuantCompMixin:
    """Mixin — mixed into QuantificationStage.  Uses self.* freely."""

    def _detection_min_snr(self):
        """SNR threshold used by peak detection — reused to flag, per sample,
        whether an analyte is actually above the detection limit. Falls back to
        the pipeline default (3.0) when the peak-detection stage is unavailable."""
        try:
            return float(self.app.stages['peak_detection'].min_snr_var.get())
        except Exception:
            return 3.0

    # ── Core quantification ───────────────────────────────────────────────────

    def _run(self):
        print('[Quant] _run() v2 — good_met_ids filter ACTIVE')
        # ── Full reset before every run ───────────────────────────────────────
        self._quant_results          = []
        self._all_peak_quant         = {}
        self._quant_peak_overrides   = {}
        self._peak_competitor_count  = {}
        self._pqn_factor             = 1.0
        self._last_groups            = {}
        self._last_blk_indices       = []
        self._warn_no_sdj            = False
        # UI setup (run_status + placeholder) is done by _safe_run() before
        # this thread starts — nothing to do here.

        # ── Detect which pipeline produced results ────────────────────────────
        _fs_stage   = (getattr(self.app, 'stages', {}) or {}).get('fullspectrum_mcr')
        _fs_regions = getattr(_fs_stage, '_mcr_regions', []) if _fs_stage else []
        _fs_mode    = bool(_fs_regions)

        if not _fs_mode:
            if not hasattr(self.annotation_stage, '_last_results_args') or \
                    not self.annotation_stage._last_results_args:
                messagebox.showinfo(
                    'Info',
                    'No results found.\n\n'
                    'Run "Full Spectrum MCR" (step 2) or\n'
                    'use the advanced Annotation stage first.')
                return
            args    = self.annotation_stage._last_results_args or {}
            results = args.get('results', [])
            is_lipidomics_mode = (str(args.get('internal_library_mode', '')) == 'Lipidomics')
            if not results:
                messagebox.showinfo('Info', 'Annotation results are empty.')
                return
        else:
            results = []
            is_lipidomics_mode = False

        _integ        = self.integ_var.get()
        # 'Area' (single-line trapz) is temporarily disabled (under development).
        # The UI radiobutton is greyed out, but a loaded project could still carry
        # 'Area' — block it here and fall back to Multiplet so no run uses it.
        if _integ == 'Area':
            messagebox.showinfo(
                'Single-Line Area em desenvolvimento',
                'O método "Single-Line Area" está em desenvolvimento e foi '
                'temporariamente desativado.\n\n'
                'A quantificação vai usar "Multiplet (whole)".')
            self.integ_var.set('Multiplet')
            _integ = 'Multiplet'
        use_area      = (_integ == 'Area')
        use_lorentz   = (_integ == 'Lorentz')
        use_multiplet = (_integ == 'Multiplet')
        proton_corr   = self.proton_corr_var.get()
        use_mcr       = self.mcr_var.get()
        if use_mcr and use_multiplet:
            messagebox.showwarning(
                'MCR + Multiplet incompatíveis',
                'MCR Deconvolution está activado mas o modo de integração é Multiplet.\n'
                'MCR nunca será executado — Multiplet tem prioridade.\n\n'
                'Desactive MCR ou mude o modo para Area/Lorentz.')
            use_mcr = False
        local_baseline = bool(getattr(self, 'local_baseline_var', None) and
                              self.local_baseline_var.get())
        _R2_MULTI_TRIGGER  = 0.85   # gaussian_r2 < this → try multi-Lorentz
        _R2_MULTI_MIN_GAIN = 0.05   # multi-Lorentz must beat single by at least this
        use_blk_sub = self.blk_sub_var.get()
        # Per-peak alignment: shift integration window per sample to track peak centre.
        # Reuses the icoshift checkbox — replaces the old global icoshift behaviour.
        use_per_peak_align = (getattr(self, 'icoshift_var', None)
                              and self.icoshift_var.get())
        use_j_resolve = getattr(self, 'j_resolve_var', None)
        use_j_resolve = use_j_resolve.get() if use_j_resolve else False
        _j_resolve_contribs = {}
        _j_resolve_contribs_ps = {}
        if use_j_resolve:
            _ann_st = getattr(self, 'annotation_stage', None)
            _j_resolve_contribs = getattr(_ann_st, '_j_resolve_contributions', {}) or {}
            _j_resolve_contribs_ps = getattr(
                _ann_st, '_j_resolve_contributions_per_sample', {}) or {}
        norm_mode   = self.norm_var.get()

        if is_lipidomics_mode:
            # Lipidomics path is intentionally simple: integrate full ppm regions only.
            use_area = True
            use_lorentz = False
            use_mcr = False

        if self.ext_is_var.get() and self._ext_is_indices:
            # When called from _safe_run() (thread), askyesno already ran on
            # main thread and set _ext_is_confirmed=True. If called directly
            # (e.g. tests), fall back to showing the dialog inline.
            if not getattr(self, '_ext_is_confirmed', False):
                if not messagebox.askyesno(
                        'External IS — Confirm',
                        'External IS mode is active.\n\n'
                        'Please confirm that the IS concentration and nH are the same\n'
                        'for both the External IS samples and the analytical samples.\n\n'
                        'Proceed with quantification?'):
                    return

        ref_peak = None
        try:
            rank = int(self.ref_rank_var.get())
            ref_peak = next((p for p in (self.peak_stage.peaks or [])
                             if p.get('rank') == rank), None)
        except (ValueError, TypeError):
            pass
        if ref_peak is None:
            try:
                ref_peak = self._find_peak(
                    float(self.ref_ppm_var.get()), tol=0.05)
            except ValueError:
                pass
        self.ref_peak = ref_peak

        try:   ref_conc = float(self.ref_conc_var.get())
        except ValueError: ref_conc = 1.0
        try:   ref_nh = max(1, int(self.ref_nh_var.get()))
        except ValueError: ref_nh = 9

        try:
            _spec_mhz = float(self.annotation_stage.spectrometer_var.get())
        except Exception:
            _spec_mhz = 600.0

        ppm_arr = self.app.ppm
        all_int = self.app.all_intensities
        si      = self.app.samples_info or []
        if ppm_arr is None or all_int is None or all_int.ndim < 2:
            messagebox.showinfo('Info', 'No spectral data loaded.')
            return

        # Build a fallback protons lookup from the DB so that annotation results
        # from before the protons column was added still get the correct nH.
        # Key: (hmdb_id_or_bmrb_id, ppm_rounded_3dp) → protons (int)
        _db_protons: dict = {}
        try:
            import sqlite3 as _sql3
            _db_p = getattr(self.app, '_last_db_path', None)
            if _db_p and os.path.isfile(_db_p):
                _conn = _sql3.connect(_db_p)
                _cur  = _conn.cursor()
                # Detect ID column and whether protons column exists
                _sc = {r[1] for r in _cur.execute(
                    'PRAGMA table_info(shifts_1h)').fetchall()}
                if 'protons' in _sc:
                    _id_c = 'bmrb_id' if 'bmrb_id' in _sc else 'hmdb_id'
                    for _row in _cur.execute(
                            f'SELECT {_id_c}, ppm, protons FROM shifts_1h'
                            ' WHERE protons IS NOT NULL').fetchall():
                        _db_protons[(str(_row[0]), round(float(_row[1]), 3))] = int(_row[2])
                _conn.close()
        except Exception:
            pass

        _dbg_f = _dbg_open()
        _dbg(_dbg_f, '=' * 70)
        _dbg(_dbg_f, 'QUANT DEBUG LOG')
        _dbg(_dbg_f, '=' * 70)
        _dbg(_dbg_f, f'norm_mode    : {norm_mode}')
        _dbg(_dbg_f, f'integ_mode   : {_integ}')
        _dbg(_dbg_f, f'use_multiplet: {use_multiplet}')
        _dbg(_dbg_f, f'local_baseline:{local_baseline}')
        _dbg(_dbg_f, f'proton_corr  : {proton_corr}')
        _dbg(_dbg_f, f'use_blk_sub  : {use_blk_sub}')
        _dbg(_dbg_f, f'use_j_resolve: {use_j_resolve}')
        _dbg(_dbg_f, f'spectrometer : {_spec_mhz} MHz')
        _dbg(_dbg_f, f'n_samples    : {all_int.shape[1]}')
        _dbg(_dbg_f, f'ppm_range    : {float(ppm_arr.min()):.3f} – {float(ppm_arr.max()):.3f}')
        _dbg(_dbg_f, f'samples_info : {[(s.get("label"), s.get("type")) for s in si]}')

        # ── Optional icoshift pre-alignment (copy only — raw data unchanged) ──
        # Per-peak alignment is applied per integration window in the main loop
        # (use_per_peak_align).  Global icoshift is no longer run here.
        if use_per_peak_align:
            self._status('Pre-align: per-peak window alignment active', 'blue')

        n_s = all_int.shape[1]

        groups = {}
        blk_indices = []
        for i, s in enumerate(si):
            lbl = s.get('label', 'Unknown')
            if i >= n_s:
                continue
            if str(s.get('type', lbl)).upper() == 'BLK':
                blk_indices.append(i)
            else:
                groups.setdefault(lbl, []).append(i)
        if not groups:
            groups = {'All': list(range(n_s))}
        non_blk_indices = [i for g_idxs in groups.values() for i in g_idxs]

        _app_dil = getattr(self.app, 'group_dilution_factors', {})
        _sample_dil  = np.ones(n_s)   # dilution × correction — used in non-IS / external IS
        _sample_corr = np.ones(n_s)   # correction only — applied unconditionally in internal IS
        for i, s in enumerate(si):
            if i < n_s:
                lbl  = s.get('label', 'Unknown')
                dil  = _app_dil.get(lbl, 1.0)
                corr = float(s.get('correction_factor', 1.0) or 1.0)
                _sample_dil[i]  = dil * corr
                _sample_corr[i] = corr

        use_ext_is  = self.ext_is_var.get() and bool(self._ext_is_indices)
        ref_areas   = None
        if ref_peak is not None:
            lo_r = min(ref_peak.get('start_idx', 0), ref_peak.get('end_idx', 0))
            hi_r = max(ref_peak.get('start_idx', 0), ref_peak.get('end_idx', 0))
            # Use the same integration method as metabolites for consistency.
            _is_x     = ppm_arr[lo_r:hi_r + 1]
            _is_ymean = np.mean(all_int[lo_r:hi_r + 1, :], axis=1)
            _is_lorentz_params = None
            if use_lorentz and len(_is_x) >= 4:
                _is_A, _is_cen, _is_wid = _fit_lorentz_mean(_is_x, _is_ymean)
                _is_max_sh = max(abs(float(_is_x[-1] - _is_x[0])) * 0.2, 0.003)
                _is_lorentz_params = (_is_A, _is_cen, _is_wid, _is_max_sh)

            def _is_area(s):
                if _is_lorentz_params is not None:
                    _A0, _c0, _w0, _msh = _is_lorentz_params
                    _res = _fit_lorentz_sample(
                        _is_x, all_int[lo_r:hi_r + 1, s], _A0, _c0, _w0,
                        max_shift_ppm=_msh)
                    if _res is not None:
                        return _res[0]
                return abs(float(_trapz(all_int[lo_r:hi_r + 1, s], _is_x)))

            if use_ext_is:
                ext_areas = []
                for s in self._ext_is_indices:
                    if s < n_s:
                        ext_areas.append(_is_area(s) / ref_nh)
                if ext_areas:
                    _fixed_ref = float(np.mean(ext_areas))
                    ref_areas = [_fixed_ref] * n_s
                else:
                    use_ext_is = False
            if not use_ext_is:
                ref_areas = [_is_area(s) / ref_nh for s in range(n_s)]

        _dbg(_dbg_f, '')
        _dbg(_dbg_f, '── IS / Reference ──────────────────────────────────────')
        if ref_peak is not None:
            _dbg(_dbg_f, f'ref_peak     : rank={ref_peak.get("rank")} ppm={ref_peak.get("ppm"):.4f}')
            _dbg(_dbg_f, f'ref_conc     : {ref_conc} mM')
            _dbg(_dbg_f, f'ref_nh       : {ref_nh}')
            _dbg(_dbg_f, f'use_ext_is   : {use_ext_is}')
            if ref_areas is not None:
                _dbg(_dbg_f, f'ref_areas/nH : {[f"{v:.4e}" for v in ref_areas]}')
                _dbg(_dbg_f, f'  mean IS area/nH: {np.mean(ref_areas):.4e}')
        else:
            _dbg(_dbg_f, 'ref_peak     : NOT FOUND — no IS quantification')
        _dbg(_dbg_f, f'groups       : { {g: len(idxs) for g, idxs in groups.items()} }')
        _dbg(_dbg_f, f'blk_indices  : {blk_indices}')
        _dbg(_dbg_f, f'sample_dil   : {_sample_dil.tolist()}')
        _dbg(_dbg_f, f'sample_corr  : {_sample_corr.tolist()}')

        quality_marks = getattr(self.annotation_stage, '_quality_marks', {}) or {}

        auto_quality_marks = {}
        try:
            mets = list(getattr(self.annotation_stage, '_metabolites', []) or [])
            for met in mets:
                mid = str(met.get('id', ''))
                for mp in met.get('matched_peaks', []):
                    pr = mp.get('peak_rank')
                    qv = mp.get('annotation_quality')
                    if pr is None:
                        continue
                    if qv is not None:
                        auto_quality_marks[(mid, pr)] = qv
                    elif (mp.get('anchor_status') == 'clean'
                          and mp.get('j_met_confirmed')):
                        # This specific peak is J-anchor-confirmed (clean + j_met_confirmed).
                        # Treat it as good even if annotation_quality is None, mirroring
                        # the has_anchor condition in annotation_results_window._met_parent_qual.
                        auto_quality_marks[(mid, pr)] = 'good'
        except Exception as _aqm_exc:
            import traceback
            print(f'[Quant] auto_quality_marks build failed: {_aqm_exc}')
            traceback.print_exc()
            auto_quality_marks = {}

        # Build set of Good metabolites — mirrors _met_parent_qual in annotation_results_window.
        # Ensures only peaks belonging to Good metabolites are quantified.
        good_met_ids: set = set()
        try:
            _all_mets_gmi = list(getattr(self.annotation_stage, '_metabolites', []) or [])
            # _anchor_lines keys are (met_id, peak_rank) — same check as anchor_store in _met_parent_qual
            _anchor_lines = getattr(self.annotation_stage, '_anchor_lines', {}) or {}
            _anchor_met_ids = {str(k[0]) for k in _anchor_lines}
            for _met_gmi in _all_mets_gmi:
                _mid_gmi = str(_met_gmi.get('id', ''))
                _mps_gmi = _met_gmi.get('matched_peaks', [])
                _is_bdl_gmi = bool(_mps_gmi) and all(
                    mp.get('below_detection_limit', False) for mp in _mps_gmi)
                _has_anchor_gmi = (
                    _mid_gmi in _anchor_met_ids
                    or any(
                        (mp.get('anchor_status') == 'clean' and mp.get('j_met_confirmed'))
                        or mp.get('auto_anchor')
                        for mp in _mps_gmi)
                )
                if _has_anchor_gmi and not _is_bdl_gmi:
                    good_met_ids.add(_mid_gmi)
                    continue
                for _mp_gmi in _mps_gmi:
                    if _mp_gmi.get('annotation_quality') == 'good':
                        good_met_ids.add(_mid_gmi)
                        break
            # A manual Good mark makes the metabolite eligible even if it has no
            # auto-good / anchored peak (e.g. a user-confirmed multiplet with no
            # detected apex → DB gap-fill quantification).
            for (_mid_q, _pr_q), _qv in quality_marks.items():
                if _qv == 'good':
                    good_met_ids.add(str(_mid_q))
            print(f'[Quant] good_met_ids built: {len(good_met_ids)} Good metabolites '
                  f'from {len(_all_mets_gmi)} total')
        except Exception as _gmi_exc:
            import traceback
            print(f'[Quant] good_met_ids build failed: {_gmi_exc}')
            traceback.print_exc()
            good_met_ids = set()

        # ── Exclude NNLS-rejected analytes ────────────────────────────────────
        # J-Resolve's solver rejects analytes that fail its quality filters
        # (coverage / concentration floor). That rejection is a hard gate, not a
        # confidence note: a rejected analyte's area is unreliable and must NOT be
        # quantified. _nnls_rejected is only populated while J-Resolve is active
        # (cleared otherwise), so gating on use_j_resolve avoids acting on a stale
        # set. A manual Good mark is explicit user intent and overrides the auto
        # rejection — every exclusion (and override) is logged so it stays visible.
        if use_j_resolve:
            _jr_st = (getattr(getattr(self, 'app', None), 'stages', {}) or {}).get(
                'j_resolve')
            _nnls_rej = {str(x) for x in (getattr(_jr_st, '_nnls_rejected', set())
                                          or set())}
            if _nnls_rej:
                _manual_good = {str(k[0]) for (k, v) in quality_marks.items()
                                if v == 'good'}
                _drop = {m for m in good_met_ids
                         if str(m) in _nnls_rej and str(m) not in _manual_good}
                _kept = {m for m in good_met_ids
                         if str(m) in _nnls_rej and str(m) in _manual_good}
                if _drop:
                    good_met_ids -= _drop
                    print(f'[Quant] excluded {len(_drop)} NNLS-rejected analytes '
                          f'from quantification: {sorted(_drop)}')
                if _kept:
                    print(f'[Quant] {len(_kept)} NNLS-rejected analytes kept by '
                          f'manual Good mark: {sorted(_kept)}')

        # ── Propagate s_dj (J quality) from post-processed _metabolites ───────
        # s_dj is computed in annotation_results_window and stored on the
        # matched_peak as '_s_dj'. The flat `results` hits don't carry it, so
        # the competition key below maps it back by (mid, peak_rank). Singlets
        # default to s_dj=1.0 in the scorer, so the multiplet/singlet tier split
        # must additionally check multiplicity — never s_dj alone.
        _s_dj_map: dict = {}
        try:
            for _met_sdj in (getattr(self.annotation_stage, '_metabolites', []) or []):
                _mid_sdj = str(_met_sdj.get('id', ''))
                for _mp_sdj in _met_sdj.get('matched_peaks', []):
                    _pr_sdj = _mp_sdj.get('peak_rank')
                    if _pr_sdj is None or '_s_dj' not in _mp_sdj:
                        continue
                    _s_dj_map[(_mid_sdj, _pr_sdj)] = float(_mp_sdj.get('_s_dj', 0) or 0)
        except Exception:
            _s_dj_map = {}

        by_peak = {}
        for r in results:
            mid = r.get('hmdb_id') or r.get('bmrb_id') or r.get('id', r.get('name', 'unknown'))
            r['_mid'] = mid
            pr = r.get('peak_rank')
            by_peak.setdefault(pr, []).append(r)

        def _effective_quality(h):
            """Return the quality for a result hit.
            Manual mark (quality_marks) takes priority over auto_quality_marks
            built from post-processed _metabolites. Raw h['annotation_quality']
            is NOT used — it is stale (set before post-processing overwrites the
            matched_peak dicts via _group_by_metabolite)."""
            manual = quality_marks.get((h['_mid'], h.get('peak_rank')))
            if manual is not None:
                return manual
            return auto_quality_marks.get((str(h['_mid']), h.get('peak_rank')))

        _SINGLET_MULTS = {'s', 'br s', 'brs', 'bs', ''}

        def _quant_sort_key(h):
            base = float(h.get('composite_score', h.get('multiplet_score', 0)) or 0)
            # Mirror the +0.20 J-bonus used in display sort (annotation_results_window.py)
            # so that J-confirmed multiplets always beat singlets in quantification too.
            is_singlet = (h.get('multiplicity') or '').strip().lower() in _SINGLET_MULTS
            if not is_singlet:
                delta_j = h.get('delta_j_hz')
                s_dj = float(h.get('s_dj', 0) or 0)
                if (delta_j is not None and delta_j < 0.5) or s_dj >= 1.0:
                    base += 0.20
            return base

        # Peak assignments from AnnotationResultsWindow resolve conflicts by
        # deltaJ and user selection — use them to break ties when multiple
        # metabolites are marked 'good' for the same peak rank.
        _annot_win = getattr(getattr(self, 'app', None), '_annot_win', None)
        _peak_assignments = getattr(_annot_win, '_peak_assignments', {}) or {}

        # Fallback: reconstruct _peak_assignments from ann._metabolites when
        # the Annotation Results window is closed (e.g. first run after project load).
        # Uses quality_marks (already loaded from project) so manual decisions survive.
        if not _peak_assignments:
            _ann_mets = getattr(self.annotation_stage, '_metabolites', None) or []
            for _met in _ann_mets:
                _mid_fb = str(_met.get('id', ''))
                for _mp in _met.get('matched_peaks', []):
                    _pr = _mp.get('peak_rank')
                    if _pr is None or _pr in _peak_assignments:
                        continue
                    _manual_q = quality_marks.get((_mid_fb, _pr))
                    _eff_q = _manual_q if _manual_q else _mp.get('annotation_quality')
                    if _eff_q is None and (
                            _mp.get('anchor_status') == 'clean'
                            and _mp.get('j_met_confirmed')):
                        _eff_q = 'good'
                    if _eff_q == 'good':
                        _peak_assignments[_pr] = (_met, _mp)

        winning_hits = []
        if is_lipidomics_mode:
            # For lipidomics, keep all region definitions and quantify by ppm region.
            winning_hits = list(results)
        else:
            # Include all good-quality hits from Good metabolites.
            # met_map (built below) selects the single best hit per metabolite
            # via _hit_sort_key (anchor priority, NNLS purity, contamination, score).
            # This works identically for trapz and multiplicity modes.
            if not good_met_ids:
                print('[Quant] WARNING: good_met_ids is empty — no Good metabolites found. '
                      'Quantification aborted to avoid processing all analytes.')
                messagebox.showwarning(
                    'No Good Analytes',
                    'No metabolites are marked as Good in Annotation Results.\n\n'
                    'Open Annotation Results and verify that analytes are correctly '
                    'annotated before running Quantification.')
                return
            winning_hits = [
                r for r in results
                if _effective_quality(r) == 'good'
                and str(r['_mid']) in good_met_ids
            ]
            print(f'[Quant] good_met_ids={len(good_met_ids)} '
                  f'results={len(results)} winning_hits={len(winning_hits)}')

        # Authoritative annotation score per metabolite — the exact value shown in
        # Annotation Results (met composite_score = max composite_score over its
        # matched peaks).  Every _ann_score assignment below resolves through this
        # map so Quantification's peak info reports the SAME score as Annotation,
        # regardless of the integration branch taken (main / MCR / lipid / APQ).
        _mid_ann_score = {}
        for _r in results:
            _rm = str(_r.get('_mid', ''))
            if not _rm:
                continue
            _rs = float(_r.get('composite_score', _r.get('multiplet_score', 0)) or 0)
            if _rs > _mid_ann_score.get(_rm, 0.0):
                _mid_ann_score[_rm] = _rs

        if (not winning_hits) and (not _fs_mode) and (not is_lipidomics_mode):
            messagebox.showwarning(
                'No Good Assignments',
                'No peaks are marked as Good in Annotation Results.\n\n'
                'Open Annotation Results, select the correct metabolite '
                'for each peak and press Ctrl+1 (Good) before quantifying.')
            return

        peak_competitor_count = {}
        for r in results:
            pr = r.get('peak_rank')
            if pr is not None:
                peak_competitor_count[pr] = peak_competitor_count.get(pr, 0) + 1
        self._peak_competitor_count = peak_competitor_count

        # Build rank → detected peak for anchor and NNLS lookups below.
        _pk_by_rank = {p['rank']: p
                       for p in (self.peak_stage.peaks or [])
                       if p.get('rank') is not None}
        _ann_stage_q = getattr(self, 'annotation_stage', None)

        # ── Populate _line_selection — ONLY when J-Resolve is active ─────────
        # _line_selection is consumed only under use_j_resolve (integration at
        # ~1344/1740); the sort key falls back to safe defaults without it. It is
        # also only needed for hits that can actually win — good-quality hits of
        # Good metabolites — not every candidate in `results`. Computing it for
        # all results on every run was the main quantification slowdown.
        if use_j_resolve:
            _ppm_min = float(self.app.ppm.min()) if self.app.ppm is not None else 0.0
            _ppm_max = float(self.app.ppm.max()) if self.app.ppm is not None else 10.0
            for _sh in results:
                if (str(_sh.get('_mid', '')) not in good_met_ids
                        or _effective_quality(_sh) == 'bad'):
                    continue
                _sh['_line_selection'] = select_clean_lines(
                    hit=_sh,
                    pk_by_rank=_pk_by_rank,
                    j_resolve_contribs=_j_resolve_contribs,
                    mid_str=str(_sh.get('_mid', '')),
                    use_nnls=True,
                    ppm_limits=(_ppm_min, _ppm_max),
                )

        def _hit_sort_key(h, mid_str):
            pr   = h.get('peak_rank')
            pk   = _pk_by_rank.get(pr, {})
            score = float(h.get('composite_score',
                                 h.get('multiplet_score', 0)) or 0)

            # 1. Anchor peak: designated by _sync_anchors_to_peaks (J-confirmed,
            #    exclusive, or manually set).  Anchors are the most reliable
            #    quantification references — always prefer them.
            is_anchor = bool(pk.get('is_anchor_line', False))

            # 2. NNLS / J-Resolve purity: fraction of this peak's area that
            #    belongs to this metabolite (1.0 = pure, <1 = shared).
            #    Prefer the peak where the analyte contributes the most.
            jr_fracs   = _j_resolve_contribs.get(pr, {}) if use_j_resolve else {}
            own_frac   = float(jr_fracs.get(mid_str, 1.0))
            n_jr_cands = len(jr_fracs) if jr_fracs else 1  # competitors in NNLS

            # 3. Total annotation competitors at this rank (fewer = cleaner peak).
            n_cands = peak_competitor_count.get(pr, 1)

            # 4. Intra-multiplet line cleanliness: fraction of expected intensity
            #    that comes from uncontaminated lines.  Hits where the selector
            #    flagged mode='skip' are always sorted last.
            _ls = h.get('_line_selection', {})
            _contamination_penalty = 1.0 - float(_ls.get('clean_intensity_fraction', 1.0))

            # 5. Peak width (FWHM proxy): narrower peak = less spectral overlap.
            #    width_ppm ∝ Area/Height for isolated peaks; higher value signals
            #    overlap with neighbouring resonances.  Prefer narrower peaks.
            _pk_width = float(pk.get('width_ppm') or pk.get('width_at_half') or 0)

            # Sort ascending: lower tuple = better hit
            return (
                not is_anchor,              # anchors first
                _ls.get('mode') == 'skip',  # skip-flagged hits last
                -own_frac,                  # higher NNLS purity first
                _contamination_penalty,     # lower intra-multiplet contamination first
                n_jr_cands,                 # fewer NNLS competitors first
                n_cands,                    # fewer annotation competitors first
                _pk_width,                  # narrower peak (less overlap) first
                -score,                     # higher composite score first
            )

        met_map = {}
        _assigned_peaks = set()   # shared by greedy + rescue
        if is_lipidomics_mode:
            for h in winning_hits:
                mid = h['_mid']
                met_map.setdefault(mid, {'name': h.get('name', mid), 'hits': []})
                met_map[mid]['hits'].append(h)
        else:
            # ── Greedy peak assignment: one winner per multiplicity ──────────
            # Candidates: all non-bad (good + yellow/undef) hits for Good mets.
            # Each hit is one metabolite's multiplicity at a peak_rank, carrying
            # delta_j_hz (|J_obs − J_db|) and ppm_diff (|centre − db_ppm|).
            #
            # Winner per contested region = lowest delta_j_hz within the ppm
            # threshold (already guaranteed: search drops ppm_diff > tolerance);
            # ties break on lowest ppm_diff. A valid multiplet (non-singlet with
            # confirmed J, s_dj >= 1.0) always outranks singlets / weak-J hits.
            # Anchor / NNLS purity / contamination only break ties when both
            # delta_j_hz and ppm_diff are equal.
            #
            # Greedy: first unassigned (metabolite, multiplicity-peaks) pair wins
            # and claims every peak_rank inside its predicted-line window, so
            # those peaks cannot be reused by another metabolite. A metabolite
            # that loses a multiplicity falls to its next candidate automatically;
            # one that wins nothing is discarded (never enters met_map).
            def _compete_key(h, mid_str):
                _mult = (h.get('multiplicity') or '').strip().lower()
                _is_singlet = _mult in _SINGLET_MULTS
                _s_dj_v = _s_dj_map.get(
                    (mid_str, h.get('peak_rank')), h.get('s_dj', 0))
                _s_dj_v = float(_s_dj_v or 0)
                # Tier 0: confirmed multiplet. Singlets default to s_dj=1.0 in the
                # scorer, so the multiplicity check is what keeps them in tier 1.
                _is_valid_mult = (not _is_singlet) and _s_dj_v >= 1.0
                _tier = 0 if _is_valid_mult else 1
                _dj = h.get('delta_j_hz')
                _dj_key = float(_dj) if (_tier == 0 and _dj is not None) else 0.0
                _ppm_diff = float(h.get('ppm_diff', h.get('ppm_error', 999)) or 999)
                _h_q = _effective_quality(h)
                _q_rank = 0 if _h_q == 'good' else 1
                return (_tier, _dj_key, _ppm_diff, _q_rank,
                        _hit_sort_key(h, mid_str))

            # s_dj drives the multiplet/singlet tier in _compete_key. When it is
            # missing (e.g. a project loaded before scores were computed, or the
            # Annotation Results window never post-processed _metabolites), every
            # hit collapses to tier 1 and the multiplet-over-singlet priority is
            # silently lost. Surface this instead of hiding it (CLAUDE.md: never
            # silence data warnings) — the run still proceeds with ppm ordering.
            if not _s_dj_map and any(
                    str(_h.get('_mid', '')) in good_met_ids
                    and _effective_quality(_h) != 'bad'
                    and (_h.get('multiplicity') or '').strip().lower()
                        not in _SINGLET_MULTS
                    for _h in results):
                self._warn_no_sdj = True
                print('[Quant] WARNING: s_dj (J-quality) unavailable — '
                      'multiplet/singlet tier priority disabled; competition '
                      'ordered by ppm only.')

            _all_non_bad = []
            for _h in results:
                _h_mid = str(_h.get('_mid', ''))
                if _h_mid not in good_met_ids:
                    continue
                _h_q = _effective_quality(_h)
                if _h_q == 'bad':
                    continue
                _all_non_bad.append((_compete_key(_h, _h_mid), _h_mid, _h))
            _all_non_bad.sort(key=lambda x: x[0])

            _assigned_mets = set()
            for _sk, _h_mid, _h in _all_non_bad:
                if _h_mid in _assigned_mets:
                    continue
                _pr = _h.get('peak_rank')
                # peak_rank None = no detected peak (DB gap-fill candidate). Such
                # hits still compete (claim their predicted-lines region below) but
                # must NOT occupy the single None slot — otherwise the first one
                # would block every other gap-fill metabolite.
                if _pr is not None and _pr in _assigned_peaks:
                    continue
                _assigned_mets.add(_h_mid)
                if _pr is not None:
                    _assigned_peaks.add(_pr)
                met_map[_h_mid] = {'name': _h.get('name', _h_mid), 'hits': [_h]}

                # Non-singlet: claim every peak_rank whose ppm falls within the
                # multiplet window so competing metabolites cannot re-use those lines.
                _pl_gr = _h.get('predicted_lines') or []
                _pl_ppms_gr = [float(l['ppm']) for l in _pl_gr if 'ppm' in l]
                if len(_pl_ppms_gr) >= 2:
                    _pk_gr = _pk_by_rank.get(_pr, {})
                    _pkw_gr = float(_pk_gr.get('width_ppm',
                                               _pk_gr.get('width_at_half', 0.008)) or 0.008)
                    _mar_gr = max(_pkw_gr * 0.7, 0.010)
                    _gr_lo  = min(_pl_ppms_gr) - _mar_gr
                    _gr_hi  = max(_pl_ppms_gr) + _mar_gr
                    for _or, _op in _pk_by_rank.items():
                        if _or not in _assigned_peaks:
                            if _gr_lo <= float(_op.get('ppm', -999)) <= _gr_hi:
                                _assigned_peaks.add(_or)

            print(f'[Quant] greedy: {len(met_map)}/{len(good_met_ids)} mets placed '
                  f'using {len(_assigned_peaks)} claimed peak slots')

        # ── User peak override (from Quant Peaks window) ──────────────────────
        # _quant_peak_overrides maps hmdb_id → peak_rank chosen by the user.
        # Swap the selected hit in met_map so that the override peak is used.
        _overrides = getattr(self, '_quant_peak_overrides', {})
        if _overrides and not is_lipidomics_mode:
            _all_hits_by_mid = {}
            for h in results:
                _all_hits_by_mid.setdefault(str(h.get('_mid', '')), []).append(h)
            # Build string→original-key map to handle int/str mismatches
            _met_key_by_str = {str(k): k for k in met_map}
            for _ov_mid, _ov_rank in _overrides.items():
                _met_key = _ov_mid if _ov_mid in met_map else \
                           _met_key_by_str.get(str(_ov_mid))
                if _met_key is None:
                    continue
                _candidates = _all_hits_by_mid.get(str(_ov_mid), [])
                _forced = next((h for h in _candidates
                                if h.get('peak_rank') == _ov_rank), None)
                if _forced is None:
                    # Override peak not in results (e.g. multiplet line from
                    # _iid_to_subchild). Build synthetic hit from existing hit
                    # replacing peak_ppm with the overridden peak's position.
                    _ov_pk = next(
                        (p for p in (self.peak_stage.peaks or [])
                         if p.get('rank') == _ov_rank), None)
                    _base_hit = (met_map[_met_key]['hits'] or [None])[0]
                    if _ov_pk is not None and _base_hit is not None:
                        _forced = dict(_base_hit)
                        _forced['peak_rank'] = _ov_rank
                        _forced['peak_ppm']  = float(_ov_pk.get('ppm', 0))
                        _forced['ppm']       = float(_ov_pk.get('ppm', 0))
                if _forced is not None:
                    met_map[_met_key]['hits'] = [_forced]

        # ── J-Resolve: inject metabolites resolved from overlapping peaks ─────
        # The standard quantification pipeline keeps only ONE metabolite per
        # peak (the "Good" annotation winner).  For peaks resolved by J-Resolve,
        # additional metabolites whose fractions are stored in
        # ann._j_resolve_contributions should also be quantified — their areas
        # are the fraction of the shared peak area.
        if use_j_resolve and _j_resolve_contribs:
            _ann_mets_by_id = {
                str(m.get('id', '')): m
                for m in (getattr(self.annotation_stage, '_metabolites', []) or [])
            }
            _rank_to_peak = {p['rank']: p for p in (self.peak_stage.peaks or [])
                             if 'rank' in p}
            for _rank, _fracs in _j_resolve_contribs.items():
                _pk = _rank_to_peak.get(_rank)
                if _pk is None:
                    continue
                for _mid_str, _frac in _fracs.items():
                    _mid = str(_mid_str)
                    if _mid in met_map:
                        continue          # already in quantification
                    if _mid not in good_met_ids:
                        continue          # only inject Good analytes
                    if not (0.0 < _frac <= 1.0):
                        continue
                    _ann_met = _ann_mets_by_id.get(_mid)
                    if _ann_met is None:
                        continue
                    # Find the matched_peak entry for this rank
                    _mp = next(
                        (m for m in _ann_met.get('matched_peaks', [])
                         if m.get('peak_rank') == _rank), None)
                    if _mp is None:
                        continue
                    # Build a minimal hit compatible with the quant loop
                    _hit = dict(_mp)
                    _hit.update({
                        '_mid':               _mid,
                        'peak_rank':          _rank,
                        'peak_ppm':           float(_pk.get('ppm', _mp.get('peak_ppm',
                                                                             _mp.get('ppm', 0)))),
                        'annotation_quality': 'good',   # treat as good for quant
                        '_j_resolve_frac':    _frac,
                        '_j_resolve_injected': True,
                    })
                    met_map[_mid] = {'name': _ann_met['name'], 'hits': [_hit]}

        # ── Rescue: Good analytes that lost all peak-rank competitions ────────
        # Only valid with J-Resolve: spectral deconvolution assigns fractional
        # areas per analyte for shared peaks.  Without J-Resolve, each peak has
        # exactly one valid winner — rescued analytes would double-count the same
        # spectral region, producing scientifically invalid concentrations.
        if not is_lipidomics_mode and use_j_resolve:
            _rescued_mids = set()
            for _r in results:
                _rmid = str(_r.get('_mid', ''))
                if _rmid in met_map or _rmid in _rescued_mids:
                    continue
                if _rmid not in good_met_ids:
                    continue
                if _effective_quality(_r) != 'good':
                    continue
                _rescued_mids.add(_rmid)

            if _rescued_mids:
                _ann_mets_rescue = {
                    str(m.get('id', '')): m
                    for m in (getattr(self.annotation_stage, '_metabolites', []) or [])
                }
                for _rmid in _rescued_mids:
                    _rhits = [_r for _r in results
                              if str(_r.get('_mid', '')) == _rmid
                              and _effective_quality(_r) == 'good'
                              and _r.get('peak_rank') not in _assigned_peaks]
                    if not _rhits:
                        continue
                    _best_r = min(_rhits,
                                  key=lambda h, _m=_rmid: _hit_sort_key(h, _m))
                    _rpr = _best_r.get('peak_rank')
                    _rescued_hit = dict(_best_r)
                    _rescued_hit['_rescued_from_competition'] = True
                    _has_jr_frac = (
                        use_j_resolve
                        and (bool(_j_resolve_contribs.get(_rpr, {}).get(_rmid))
                             or _j_resolve_contribs_ps.get(_rpr, {}).get(_rmid)
                                is not None)
                    )
                    if not _has_jr_frac:
                        _rescued_hit['_overlapping_peak'] = True
                    _rann_met = _ann_mets_rescue.get(_rmid)
                    _rname = (_rann_met['name'] if _rann_met
                              else _best_r.get('name', _rmid))
                    met_map[_rmid] = {'name': _rname, 'hits': [_rescued_hit]}
                    _dbg(_dbg_f,
                         f'  [RESCUE] {_rname}  mid={_rmid}'
                         f'  peak_rank={_rpr}  has_jr={_has_jr_frac}')
            print(f'[Quant] rescued={len(_rescued_mids)}  '
                  f'met_map total={len(met_map)}')

        raw_areas = {}
        meta_info = {}

        _mcr_overridden_peaks = set()
        _all_mcr_peaks = list(self.peak_stage.peaks or []) + list(_fs_regions)
        for pk in _all_mcr_peaks:
            mcr_ann   = pk.get('mcr_annotation')
            if not mcr_ann:
                continue
            confirmed = mcr_ann.get('confirmed_assignments', {})
            if not confirmed:
                continue
            accepted_ks = [k for k, v in confirmed.items() if v and v.get('accepted')]
            if not accepted_ks:
                continue

            K_ann    = mcr_ann['K']
            S_mat    = mcr_ann['S']
            C_mat    = mcr_ann['C']
            ppm_reg  = mcr_ann['ppm_region']
            cands    = mcr_ann.get('candidates', [])
            assigns  = mcr_ann.get('assignments', [])
            K_actual = min(K_ann, S_mat.shape[1])

            for k in accepted_ks:
                if k >= K_actual:
                    continue
                cv        = confirmed[k]
                ci        = assigns[k][0] if k < len(assigns) else 0
                cand      = cands[ci] if ci < len(cands) else {}
                cid       = cv.get('hmdb_id') or cand.get('hmdb_id') or \
                            cand.get('bmrb_id') or f'mcr_{pk["rank"]}_{k}'
                cname     = cv.get('metabolite') or cand.get('name') or f'MCR C{k+1}'

                s_area_scale = abs(float(_trapz(S_mat[:, k], ppm_reg)))
                if C_mat.shape[0] == n_s:
                    a_vec = C_mat[:, k] * (s_area_scale if s_area_scale > 1e-15 else 1.0)
                else:
                    c_mean_k = float(np.mean(C_mat[:, k]))
                    a_vec    = np.full(n_s, c_mean_k * (s_area_scale if s_area_scale > 1e-15 else 1.0))
                a_vec = np.abs(a_vec)

                _nh_k = 1
                if cid in met_map and met_map[cid].get('hits'):
                    h0      = met_map[cid]['hits'][0]
                    det_ppm = float(h0.get('peak_ppm', h0.get('ppm', pk['ppm'])))
                    try:
                        _spec_mhz_l = float(self.annotation_stage.spectrometer_var.get())
                    except Exception:
                        _spec_mhz_l = 600.0
                    _nh_k = max(1e-6, _effective_nh(h0, det_ppm, _spec_mhz_l))
                if norm_mode in ('IS', 'IS+PQN') or proton_corr:
                    a_vec = a_vec / _nh_k

                raw_areas[cid] = a_vec
                meta_info[cid] = {
                    'name':          cname,
                    'hits':          met_map.get(cid, {}).get('hits', []),
                    '_peak':         pk,
                    '_used_mcr':     True,
                    '_mcr':          {
                        'S':       S_mat[:, k],
                        'lo_i':    int(np.argmin(np.abs(ppm_arr - ppm_reg.max()))),
                        'hi_i':    int(np.argmin(np.abs(ppm_arr - ppm_reg.min()))),
                        'metrics': mcr_ann.get('metrics', {}),
                    },
                    '_lorentz':      None,
                    '_ann_score':    _mid_ann_score.get(str(cid), 0.0),
                    '_nh':           _nh_k,
                    '_blk_mean_area': 0.0,
                    '_mcr_component': True,
                }
                _mcr_overridden_peaks.add(pk['rank'])
                met_map.pop(cid, None)

        met_map = {mid: info for mid, info in met_map.items()
                   if info.get('hits') and
                   info['hits'][0].get('peak_rank') not in _mcr_overridden_peaks}

        _lorentz_params  = {}
        _mcr_info        = {}
        _align_shifts    = {}   # mid → list[float] per-sample ppm shift

        # ── Multiplet window pre-pass ─────────────────────────────────────────
        # In Multiplet mode, each metabolite's integration window is [min(pred_ppms)
        # - margin, max(pred_ppms) + margin].  When two metabolites have overlapping
        # windows (different peak ranks, nearby ppm), they double-count the shared
        # spectral region.  This pre-pass resolves overlaps by clipping the
        # lower-priority metabolite's window at the overlap boundary.
        # Priority = composite_score of the selected hit (higher = keeps full window).
        _mult_windows_precomp = {}   # mid → [lo_ppm, hi_ppm]  (mutable)
        if use_multiplet and not is_lipidomics_mode:
            for _mid_pw, _info_pw in met_map.items():
                if not _info_pw.get('hits'):
                    continue
                _hit_pw = _info_pw['hits'][0]
                _p_pw = _pk_by_rank.get(_hit_pw.get('peak_rank'))
                if _p_pw is None:
                    continue
                _pl_pw   = _hit_pw.get('predicted_lines') or []
                _ppms_pw = [float(l['ppm']) for l in _pl_pw if 'ppm' in l]
                _pkw_pw  = float(_p_pw.get('width_ppm', _p_pw.get('width_at_half', 0.008)) or 0.008)
                _mar_pw  = max(_pkw_pw * 0.7, 0.010)
                if _ppms_pw:
                    _lo_pw = min(_ppms_pw) - _mar_pw
                    _hi_pw = max(_ppms_pw) + _mar_pw
                else:
                    _li_pw   = min(_p_pw.get('start_idx', 0), _p_pw.get('end_idx', 0))
                    _hii_pw  = max(_p_pw.get('start_idx', 0), _p_pw.get('end_idx', 0))
                    _lo_pw   = float(min(ppm_arr[_li_pw], ppm_arr[_hii_pw]))
                    _hi_pw   = float(max(ppm_arr[_li_pw], ppm_arr[_hii_pw]))
                _mult_windows_precomp[_mid_pw] = [_lo_pw, _hi_pw]

            # Sort descending by composite_score; higher score keeps full window.
            _sorted_pw = sorted(
                _mult_windows_precomp,
                key=lambda m: -float(
                    (met_map[m]['hits'][0] if met_map[m].get('hits') else {})
                    .get('composite_score', 0) or 0))

            for _i_pw, _ma in enumerate(_sorted_pw):
                _wa = _mult_windows_precomp[_ma]
                for _mb in _sorted_pw[_i_pw + 1:]:
                    _wb = _mult_windows_precomp.get(_mb)
                    if _wb is None:
                        continue
                    _ovlp_lo = max(_wa[0], _wb[0])
                    _ovlp_hi = min(_wa[1], _wb[1])
                    if _ovlp_lo >= _ovlp_hi:
                        continue  # no overlap
                    # _ma has higher priority — clip _mb
                    if _wb[0] < _wa[0] and _wb[1] > _wa[1]:
                        # _mb spans both sides of _ma — keep left portion
                        _mult_windows_precomp[_mb] = [_wb[0], _wa[0]]
                    elif _wb[1] > _wa[1]:
                        # overlap on left side of _mb
                        _mult_windows_precomp[_mb] = [_wa[1], _wb[1]]
                    else:
                        # overlap on right side of _mb (or _mb fully inside _ma)
                        _mult_windows_precomp[_mb] = [_wb[0], _wa[0]]

        print(f'[Quant] FINAL met_map={len(met_map)} analytes going to integration')
        _p_total = len(met_map)
        _p_done  = 0
        self._prog_update(2, f'Integrating {_p_total} metabolites…')
        for _p_i, (mid, info) in enumerate(met_map.items()):
            _met_name = info.get('name', str(mid))
            _pct = 2.0 + (_p_i / max(_p_total, 1)) * 85.0
            self._prog_update(_pct, f'({_p_i + 1}/{_p_total}) {_met_name}')
            if is_lipidomics_mode:
                s_areas = np.zeros(n_s)
                found = False
                lipid_regions = []
                for hit in info['hits']:
                    lo_ppm = float(hit.get('start_ppm', hit.get('ppm', 0) - 0.01))
                    hi_ppm = float(hit.get('end_ppm', hit.get('ppm', 0) + 0.01))
                    lo_ppm, hi_ppm = min(lo_ppm, hi_ppm), max(lo_ppm, hi_ppm)
                    sm = (ppm_arr >= lo_ppm) & (ppm_arr <= hi_ppm)
                    if not sm.any():
                        continue
                    found = True
                    lipid_regions.append((lo_ppm, hi_ppm))
                    a_vec = np.array([
                        abs(float(_trapz(all_int[sm, s], ppm_arr[sm])))
                        for s in range(n_s)
                    ], dtype=float)

                    _nh = 1.0
                    if proton_corr and _nh > 0:
                        a_vec = a_vec / _nh

                    _blk_mean_area = 0.0
                    if use_blk_sub and blk_indices:
                        _blk_mean_area = float(np.mean([a_vec[i] for i in blk_indices
                                                        if i < len(a_vec)]))
                        a_vec = a_vec - _blk_mean_area
                        a_vec = np.maximum(a_vec, 0.0)

                    s_areas += a_vec

                if not found:
                    continue
                if lipid_regions:
                    lo_all = min(rg[0] for rg in lipid_regions)
                    hi_all = max(rg[1] for rg in lipid_regions)
                else:
                    lo_all, hi_all = None, None

                meta_info[mid] = {
                    'name': info['name'],
                    'hits': info['hits'],
                    '_peak': None,
                    '_used_mcr': False,
                    '_lorentz': None,
                    '_mcr': None,
                    '_ann_score': _mid_ann_score.get(str(mid), 1.0),
                    '_nh': 1.0,
                    '_blk_mean_area': 0.0,
                    '_lipid_regions': lipid_regions,
                    '_lipid_window': (lo_all, hi_all),
                }
                raw_areas[mid] = s_areas
                continue

            s_areas = np.zeros(n_s)
            found   = False
            _hit_has_anchor = False
            _met_synthetic_db = False   # quantified from DB window (no detected peak)
            _j_frac_this_met = None
            _jr_multiplet_applied = False   # per-line JG frac already folded into a_vec
            _jr_line_fracs_store  = None    # [(line_ppm, frac)] for PDF per-line paint
            for hit in info['hits']:
                # Prefer rank-based lookup: annotation peak_rank maps directly to
                # the detected peak used during annotation, bypassing ppm proximity
                # search which can land on the wrong peak when lines are close.
                p = _pk_by_rank.get(hit.get('peak_rank')) or \
                    self._find_peak(hit.get('peak_ppm', hit.get('ppm', 0)))
                _synthetic_db = False
                if p is None:
                    # No detected peak. In Multiplet mode, gap-fill: integrate the
                    # DB-predicted window (exact line positions from transitions)
                    # by trapz, so a user-confirmed multiplet with visible signal
                    # but no detected apex is still quantified. (Area/Lorentz need
                    # a detected line → skip as before.)
                    _pl0 = hit.get('predicted_lines') or []
                    _pl0_ppms = [float(l['ppm']) for l in _pl0 if 'ppm' in l]
                    if use_multiplet and _pl0_ppms:
                        _cen = float(hit.get('peak_ppm') or hit.get('ppm')
                                     or float(np.mean(_pl0_ppms)))
                        _si = int(np.argmin(np.abs(ppm_arr - max(_pl0_ppms))))
                        _ei = int(np.argmin(np.abs(ppm_arr - min(_pl0_ppms))))
                        p = {'ppm': _cen,
                             'start_ppm': float(min(_pl0_ppms)),
                             'end_ppm':   float(max(_pl0_ppms)),
                             'start_idx': min(_si, _ei),
                             'end_idx':   max(_si, _ei),
                             'width_ppm': 0.008,
                             'rank':      hit.get('peak_rank')}
                        _synthetic_db = True
                        _met_synthetic_db = True
                    else:
                        continue
                found = True
                lo_i = min(p.get('start_idx', 0), p.get('end_idx', 0))
                hi_i = max(p.get('start_idx', 0), p.get('end_idx', 0))
                detected_ppm = float(hit.get('peak_ppm', hit.get('ppm', p.get('ppm', 0))))
                _ann_stage   = getattr(self, 'annotation_stage', None)
                _anchor_store = getattr(_ann_stage, '_anchor_lines', {}) if _ann_stage else {}
                _anchor_key  = (hit.get('_mid'), hit.get('peak_rank'))
                _anchor_idx  = _anchor_store.get(_anchor_key)
                if _anchor_idx is not None:
                    hit['anchor_line_idx']  = _anchor_idx
                    hit['predicted_lines']  = hit.get('predicted_lines',
                                                       hit.get('_predicted_lines', []))
                    _hit_has_anchor = True
                # Fill protons from DB fallback when annotation results predate
                # the protons column addition (hit.get('protons') is None or 0).
                if not hit.get('protons') and _db_protons:
                    _h_id  = str(hit.get('hmdb_id') or hit.get('bmrb_id') or '')
                    _h_ppm = round(float(
                        hit.get('ppm_center') or hit.get('db_ppm') or
                        hit.get('ppm') or 0), 3)
                    _p_fb = _db_protons.get((_h_id, _h_ppm))
                    if _p_fb:
                        hit['protons'] = _p_fb
                # use_multiplet integrates the whole multiplet window → full nH.
                # Otherwise, use fractional nH whenever predicted_lines > 1:
                # each annotation hit corresponds to one specific DB line, and
                # _effective_nh matches detected ppm to the closest predicted line.
                if use_multiplet:
                    nh = max(1.0, float(hit.get('protons', 1) or 1))
                elif len(hit.get('predicted_lines', [])) > 1:
                    nh = _effective_nh(hit, detected_ppm, _spec_mhz)
                else:
                    nh = max(1.0, float(hit.get('protons', 1) or 1))
                nh = max(0.01, nh)

                if use_multiplet:
                    # predicted_lines holds the observed ppm of every line of
                    # the multiplet (already shifted to the spectrum reference
                    # frame). Use min/max directly — no peak lookup needed.
                    _pl = hit.get('predicted_lines') or []
                    _pl_ppms = [float(l['ppm']) for l in _pl if 'ppm' in l]

                    _dbg(_dbg_f, f'  [MULTIPLET] {hit.get("metabolite", mid)}')
                    _dbg(_dbg_f, f'    multiplicity  : {hit.get("multiplicity") or hit.get("type")}')
                    _dbg(_dbg_f, f'    j_coupling    : {hit.get("j_coupling")}')
                    _dbg(_dbg_f, f'    predicted_lines ({len(_pl)}): {[round(v,4) for v in _pl_ppms]}')
                    _dbg(_dbg_f, f'    detected_ppm  : {detected_ppm:.4f}')

                    # Half-linewidth margin: use detected peak width if available
                    _pk_width = float(p.get('width_ppm', p.get('width_at_half', 0.008)) or 0.008)
                    _margin   = max(_pk_width * 0.7, 0.010)

                    if mid in _mult_windows_precomp:
                        # Use pre-computed (possibly clipped) window to avoid
                        # double-counting spectral regions shared with other analytes.
                        _m_lo, _m_hi = _mult_windows_precomp[mid]
                        # Overlap clipping can produce a degenerate window (lo >= hi).
                        # Two distinct situations require different handling:
                        #   Case A — detected peak IS within the predicted_lines range:
                        #     this sub-peak lost the region to a higher-scoring metabolite.
                        #     Keep the degenerate window → _sm_mult.sum()<2 fallback below
                        #     will integrate only this metabolite's own detected sub-peak.
                        #   Case B — detected peak is OUTSIDE the predicted_lines range:
                        #     the peak_rank points to an unrelated peak (e.g. after PPM
                        #     calibration shifted db_ppm away from the detected peak).
                        #     Use the full predicted-lines window so the correct spectral
                        #     region is integrated rather than the mismatched single peak.
                        if _m_hi <= _m_lo + 1e-5 and _pl_ppms:
                            _actual_det_ppm = float(p.get('ppm', detected_ppm))
                            _pl_lo = min(_pl_ppms) - _margin * 2
                            _pl_hi = max(_pl_ppms) + _margin * 2
                            _det_outside_pl = not (_pl_lo <= _actual_det_ppm <= _pl_hi)
                            if _det_outside_pl:
                                # Case B: restore full multiplet window
                                _m_lo = min(_pl_ppms) - _margin
                                _m_hi = max(_pl_ppms) + _margin
                            # else Case A: keep degenerate → single-sub-peak fallback
                    elif _pl_ppms:
                        _m_lo = min(_pl_ppms) - _margin
                        _m_hi = max(_pl_ppms) + _margin
                    else:
                        _m_lo = float(min(ppm_arr[lo_i], ppm_arr[hi_i]))
                        _m_hi = float(max(ppm_arr[lo_i], ppm_arr[hi_i]))
                    _dbg(_dbg_f, f'    integ window  : [{_m_lo:.4f}, {_m_hi:.4f}] ppm  (margin={_margin:.4f})')
                    # Store multiplet window for visualization/export
                    hit['_multiplet_window'] = (_m_lo, _m_hi)
                    _sm_mult = (ppm_arr >= _m_lo) & (ppm_arr <= _m_hi)
                    if _sm_mult.sum() < 2:
                        # Degenerate window (clipped by overlap or too narrow):
                        # fall back to the detected peak's index range.
                        _sm_mult = np.zeros(len(ppm_arr), dtype=bool)
                        _sm_mult[lo_i:hi_i + 1] = True
                        # Update stored window so table/visualization reflect the
                        # actual integration range, not the zero-width overlap result.
                        _fb_lo = float(min(ppm_arr[lo_i], ppm_arr[hi_i]))
                        _fb_hi = float(max(ppm_arr[lo_i], ppm_arr[hi_i]))
                        hit['_multiplet_window'] = (_fb_lo, _fb_hi)
                    _x_mult = ppm_arr[_sm_mult]
                    _seg_m  = all_int[_sm_mult, :].astype(float).copy()   # (m, n_s)
                    if local_baseline and _seg_m.shape[0] >= 2:
                        _bl_m = np.linspace(np.maximum(_seg_m[0, :], 0.0),
                                            np.maximum(_seg_m[-1, :], 0.0),
                                            _seg_m.shape[0])
                        _seg_m = _seg_m - _bl_m

                    # Per-line J-Resolve fractions: a multiplet's legs can be
                    # resolved differently by NNLS. Weight each window point by the
                    # fraction of its nearest predicted line so the integrated area
                    # reflects that per-leg reality (matches the JG visualizer).
                    # With a single/uniform fraction this is identical to the plain
                    # whole-window integral, so non-resolved multiplets are unchanged.
                    if use_j_resolve and len(_pl_ppms) >= 2:
                        _line_fr = []
                        for _lp in _pl_ppms:
                            _lr_q, _lr_d = None, _margin * 1.5
                            for _cpr, _cpk in _pk_by_rank.items():
                                _dd = abs(float(_cpk.get('ppm', 1e9)) - _lp)
                                if _dd < _lr_d:
                                    _lr_q, _lr_d = _cpr, _dd
                            _fval = 1.0   # line not NNLS-resolved → fully this analyte
                            if _lr_q is not None:
                                _fa_ps = _j_resolve_contribs_ps.get(_lr_q, {}).get(mid)
                                if _fa_ps is not None and len(_fa_ps) == n_s:
                                    _fval = np.clip(np.asarray(_fa_ps, float), 0.0, 1.0)
                                else:
                                    _fs = _j_resolve_contribs.get(_lr_q, {}).get(mid)
                                    if _fs is not None and 0.0 < float(_fs) <= 1.0:
                                        _fval = float(_fs)
                            _line_fr.append(_fval)
                        a_vec = jr_pointwise_multiplet_area(
                            _x_mult, _seg_m, _pl_ppms, _line_fr)
                        _jr_multiplet_applied = True
                        # Reported multiplet fraction = the one actually applied to
                        # the area: area-weighted blend of the leg fractions, i.e.
                        # weighted_area / plain_area. (A simple mean of the leg
                        # fractions would not match when the legs differ in area.)
                        _plain_area = np.array(
                            [abs(float(_trapz(_seg_m[:, s], _x_mult)))
                             for s in range(n_s)])
                        _ratio = np.divide(
                            a_vec, _plain_area,
                            out=np.ones_like(a_vec), where=_plain_area > 0)
                        _j_frac_this_met = float(np.mean(_ratio))
                        _jr_line_fracs_store = [
                            (float(_lp), float(np.mean(np.asarray(f))))
                            for _lp, f in zip(_pl_ppms, _line_fr)]
                    else:
                        a_vec = np.array([abs(float(_trapz(_seg_m[:, s], _x_mult)))
                                          for s in range(n_s)])
                elif use_lorentz and (hi_i - lo_i) >= 1:
                    x_r      = ppm_arr[lo_i:hi_i + 1]
                    y_mean_r = np.mean(all_int[lo_i:hi_i+1, :], axis=1)
                    A_l, cen_l, wid_l, _bl_y0, _bl_slope = \
                        _fit_lorentz_mean_with_bl(x_r, y_mean_r,
                                                  with_baseline=local_baseline)
                    # Post-fit R² on signal (baseline-corrected when local_baseline)
                    _bl_arr  = _bl_y0 + _bl_slope * (x_r - float(x_r[0]))
                    _fit_arr = _bl_arr + A_l * _lorentz_shape(x_r, cen_l, wid_l)
                    _ss_res  = float(np.sum((y_mean_r - _fit_arr) ** 2))
                    _ss_tot  = float(np.sum((y_mean_r - np.mean(y_mean_r)) ** 2))
                    _lor_r2  = (1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float('nan')
                    # Fit validity: R² >= 0.70 and width sane (< half window span)
                    _win_span = abs(float(x_r[-1] - x_r[0])) or 1e-6
                    _fit_ok   = (np.isfinite(_lor_r2) and _lor_r2 >= 0.70
                                 and wid_l < _win_span * 0.5)
                    if _fit_ok:
                        max_shift = max(_win_span * 0.15, 0.005)
                        shifts = []
                        a_list = []
                        for s in range(n_s):
                            y_s  = all_int[lo_i:hi_i+1, s]
                            res  = _fit_lorentz_sample(x_r, y_s, A_l, cen_l, wid_l,
                                                       max_shift_ppm=max_shift,
                                                       with_baseline=local_baseline)
                            if res is not None:
                                area_s, _, cen_s, _ = res
                                shifts.append(cen_s - cen_l)
                            else:
                                area_s = abs(float(_trapz(y_s, x_r)))
                                shifts.append(0.0)
                            a_list.append(area_s)
                        a_vec = np.array(a_list)
                        _lorentz_params[mid] = {
                            'cen':      cen_l,   'wid':      wid_l,
                            'shifts':   shifts,
                            'bl_y0':    _bl_y0,  'bl_slope': _bl_slope,
                            'bl_x0':    float(x_r[0]),
                            'r2':       _lor_r2,
                        }
                    else:
                        # Lorentz fit invalid (multi-peak region or poor baseline):
                        # fall back to trapz; multi-Lorentz trigger below may recover.
                        a_list = []
                        for s in range(n_s):
                            seg = all_int[lo_i:hi_i+1, s].copy()
                            if local_baseline and len(seg) >= 2:
                                bl = np.linspace(max(float(seg[0]), 0.0),
                                                 max(float(seg[-1]), 0.0), len(seg))
                                seg = seg - bl
                            a_list.append(abs(float(_trapz(seg, x_r))))
                        a_vec = np.array(a_list)
                elif use_mcr and (hi_i - lo_i) >= 3:
                    X = all_int[lo_i:hi_i+1, :]
                    try:
                        C, S_mcr, mcr_metrics = _mcr_als(X, n_components=1)
                        s_area_scale = abs(float(
                            _trapz(S_mcr[:, 0], ppm_arr[lo_i:hi_i+1])))
                        a_vec = C[:, 0] * (s_area_scale if s_area_scale > 0 else 1.0)
                        _mcr_info[mid] = {
                            'S':       S_mcr[:, 0],
                            'lo_i':    lo_i,
                            'hi_i':    hi_i,
                            'metrics': mcr_metrics,
                        }
                    except Exception:
                        a_vec = np.array([
                            abs(float(_trapz(all_int[lo_i:hi_i+1, s],
                                             ppm_arr[lo_i:hi_i+1])))
                            for s in range(n_s)])
                elif use_per_peak_align and (hi_i - lo_i) >= 3:
                    # Per-peak alignment: shift integration window per sample
                    # to follow actual peak centre, then integrate.
                    x_r      = ppm_arr[lo_i:hi_i + 1]
                    y_mean_r = np.mean(all_int[lo_i:hi_i + 1, :], axis=1)
                    _cen_ref = float(x_r[int(np.argmax(y_mean_r))])
                    _max_sh  = max(abs(float(x_r[-1] - x_r[0])) * 0.35, 0.003)
                    _half_w_ppm = abs(float(x_r[-1] - x_r[0])) / 2.0
                    # Reuse the per-sample apex centres computed at detection
                    # (consistent with detection/overlays); fall back to on-the-fly
                    # apex search when they are absent (older projects).
                    _stored_cen = p.get('ppm_per_sample')
                    _stored_ok  = p.get('aligned_per_sample')
                    _pa_shifts = []
                    a_list = []
                    for s in range(n_s):
                        if (_stored_cen is not None and s < len(_stored_cen)):
                            if _stored_ok is not None and s < len(_stored_ok) \
                                    and not _stored_ok[s]:
                                delta = 0.0          # not aligned here → keep window
                            else:
                                delta = float(_stored_cen[s]) - _cen_ref
                                delta = max(-_max_sh, min(_max_sh, delta))
                        else:
                            y_s   = all_int[lo_i:hi_i + 1, s]
                            cen_s = _align_sample_center(x_r, y_s, _cen_ref, _max_sh)
                            delta = cen_s - _cen_ref
                        _pa_shifts.append(delta)
                        # Shift window bounds by delta
                        lo_s_ppm = float(ppm_arr[lo_i]) + delta
                        hi_s_ppm = float(ppm_arr[hi_i]) + delta
                        lo_s_ppm, hi_s_ppm = min(lo_s_ppm, hi_s_ppm), max(lo_s_ppm, hi_s_ppm)
                        lo_s = int(np.argmin(np.abs(ppm_arr - lo_s_ppm)))
                        hi_s = int(np.argmin(np.abs(ppm_arr - hi_s_ppm)))
                        lo_s = max(0, min(lo_s, hi_s))
                        hi_s = min(len(ppm_arr) - 1, max(lo_s, hi_s))
                        seg = all_int[lo_s:hi_s + 1, s]
                        a_list.append(abs(float(_trapz(seg, ppm_arr[lo_s:hi_s + 1]))))
                    a_vec = np.array(a_list)
                    _align_shifts[mid] = _pa_shifts
                else:  # use_area (trapz) — explicit fallback, also handles lo_i == hi_i
                    _a_list_tb = []
                    for s in range(n_s):
                        seg = all_int[lo_i:hi_i + 1, s].copy()
                        if local_baseline and len(seg) >= 2:
                            bl = np.linspace(max(float(seg[0]), 0.0),
                                             max(float(seg[-1]), 0.0), len(seg))
                            seg = seg - bl
                        _a_list_tb.append(abs(float(_trapz(seg, ppm_arr[lo_i:hi_i + 1]))))
                    a_vec = np.array(_a_list_tb)
                    _dbg(_dbg_f, f'    integ(trapz): window=[{lo_i}:{hi_i}]'
                                 f'  n_pts={hi_i-lo_i+1}'
                                 f'  areas={[f"{v:.3e}" for v in a_vec.tolist()]}')

                # ── Auto multi-Lorentz when gaussian_r2 is low ────────────────
                # If the detected peak has poor shape quality (overlap / shoulder),
                # try deconvolving the full cluster with a sum-of-N-Lorentzians.
                # Only replace a_vec if the multi-Lorentz fit improves R² enough.
                _gr2  = float(p.get('gaussian_r2', 1.0) or 1.0)
                _n_cl = int(p.get('n_cluster_peaks', 1) or 1)

                _lorentz_rejected = use_lorentz and mid not in _lorentz_params
                # Skip in Multiplet mode: there we integrate the WHOLE multiplet
                # window (trapz) by design, so per-cluster multi-Lorentz curve_fit
                # would both contradict that and dominate runtime.
                if ((not use_multiplet)
                        and (_gr2 < _R2_MULTI_TRIGGER or _lorentz_rejected)
                        and _n_cl >= 2):
                    # cluster_lo / cluster_hi are spectrum indices (from peak
                    # detection) — use them directly; avoids searchsorted on the
                    # descending ppm_arr (NMR convention).
                    _cl_lo_idx = p.get('cluster_lo')   # int index or None
                    _cl_hi_idx = p.get('cluster_hi')   # int index or None

                    _cl_peers  = [
                        op for op in _pk_by_rank.values()
                        if op.get('cluster_lo') == _cl_lo_idx
                        and op.get('rank') is not None
                    ] if _cl_lo_idx is not None else []

                    _peer_ppms = sorted(
                        [float(op['ppm']) for op in _cl_peers
                         if op.get('ppm') is not None],
                        reverse=True)

                    if len(_peer_ppms) >= 2 and _cl_lo_idx is not None and _cl_hi_idx is not None:
                        _ml_lo = min(int(_cl_lo_idx), int(_cl_hi_idx))
                        _ml_hi = min(max(int(_cl_lo_idx), int(_cl_hi_idx)) + 1,
                                     len(ppm_arr))
                        _x_ml  = ppm_arr[_ml_lo:_ml_hi]

                        if len(_x_ml) >= 4:
                            _ym_ml = np.mean(all_int[_ml_lo:_ml_hi, :], axis=1)
                            _multi = _fit_multi_lorentz(
                                _x_ml, _ym_ml, all_int[_ml_lo:_ml_hi, :],
                                _peer_ppms)

                            if (_multi is not None
                                    and _multi['r2'] > _gr2 + _R2_MULTI_MIN_GAIN):
                                _tgt  = float(detected_ppm)
                                _best = min(_multi['components'],
                                            key=lambda c: abs(c[1] - _tgt))
                                a_vec = _best[2]

                sample_regions = p.get('sample_regions', {})
                if sample_regions:
                    mcr_s   = _mcr_info.get(mid)
                    s_vec   = mcr_s['S']       if mcr_s else None
                    o_lo_i  = mcr_s['lo_i']    if mcr_s else lo_i
                    o_hi_i  = mcr_s['hi_i']    if mcr_s else hi_i
                    s_scale = (abs(float(_trapz(s_vec, ppm_arr[o_lo_i:o_hi_i+1])))
                               if s_vec is not None else 1.0)

                    for s_idx, adj in sample_regions.items():
                        if not isinstance(s_idx, int) or s_idx >= n_s:
                            continue
                        lo_s = adj.get('start_ppm', ppm_arr[lo_i])
                        hi_s = adj.get('end_ppm',   ppm_arr[hi_i])
                        lo_s, hi_s = min(lo_s, hi_s), max(lo_s, hi_s)
                        sm = (ppm_arr >= lo_s) & (ppm_arr <= hi_s)
                        if not sm.any():
                            continue
                        col_s  = all_int[sm, s_idx]
                        ppm_s  = ppm_arr[sm]

                        if use_mcr and s_vec is not None:
                            s_lo = max(0, np.where(sm)[0].min() - o_lo_i)
                            s_hi = min(len(s_vec) - 1,
                                       np.where(sm)[0].max() - o_lo_i)
                            if s_hi > s_lo:
                                sv   = s_vec[s_lo:s_hi+1]
                                cv   = col_s[:len(sv)]
                                den  = float(sv @ sv)
                                C_s  = max(0.0, float(sv @ cv) / den) if den > 1e-12 else 0.0
                                a_vec[s_idx] = C_s * (s_scale if s_scale > 1e-12 else 1.0)
                            else:
                                a_vec[s_idx] = abs(float(_trapz(col_s, ppm_s))) if col_s.size > 1 else 0.0
                        elif use_lorentz:
                            try:
                                _, cen_l2, wid_l2 = _fit_lorentz_mean(ppm_s, col_s)
                                A_s2 = _lorentz_amplitude(ppm_s, col_s, cen_l2, wid_l2)
                                a_vec[s_idx] = float(A_s2) * (np.pi * float(wid_l2))
                            except Exception:
                                a_vec[s_idx] = abs(float(_trapz(col_s, ppm_s))) if col_s.size > 1 else 0.0
                        else:
                            a_vec[s_idx] = abs(float(_trapz(col_s, ppm_s))) if col_s.size > 1 else 0.0

                # ── Intra-multiplet line selection scaling ────────────────────
                # Only activates when some lines were excluded (scale_factor > 1).
                # If all lines are clean (scale_factor == 1.0) the original a_vec
                # is kept — re-integrating sub-regions would produce a different
                # result due to per-line baseline correction and is unnecessary.
                # Only applied when J-Resolve is active — stale _line_selection
                # from a prior J-Resolve run must not zero out areas.
                _ls = hit.get('_line_selection', {}) if use_j_resolve else {}
                _ls_mode = _ls.get('mode')
                _ls_sf   = float(_ls.get('scale_factor', 1.0))
                _line_sel_applied = False
                if use_j_resolve and _ls_mode == 'skip':
                    a_vec = np.zeros(n_s)
                    hit['_line_sel_skip'] = True
                elif (_ls_mode in ('nnls', 'ratio') and _ls_sf > 1.0
                      and _ls.get('clean_ranks') and not use_multiplet):
                    _clean_pks = [_pk_by_rank[r] for r in _ls['clean_ranks']
                                  if r in _pk_by_rank]
                    if _clean_pks:
                        _a_clean = np.zeros(n_s)
                        for _cp in _clean_pks:
                            _cl_lo = _cp.get('start_idx', lo_i)
                            _cl_hi = _cp.get('end_idx',   hi_i)
                            for s_idx in range(n_s):
                                seg = all_int[_cl_lo:_cl_hi + 1, s_idx].copy()
                                if local_baseline and len(seg) >= 2:
                                    bl = np.linspace(max(float(seg[0]), 0.0),
                                                     max(float(seg[-1]), 0.0),
                                                     len(seg))
                                    seg = seg - bl
                                _a_clean[s_idx] += abs(float(
                                    _trapz(seg, ppm_arr[_cl_lo:_cl_hi + 1])))
                        a_vec = _a_clean * _ls_sf
                        _line_sel_applied = True
                # ─────────────────────────────────────────────────────────────

                # For IS mode: always divide by nH so the IS ratio is proton-corrected.
                # For non-IS modes: only divide if user enabled proton_corr.
                if norm_mode in ('IS', 'IS+PQN') or proton_corr:
                    a_vec = a_vec / nh
                # J-Resolve fractional area correction.
                # Skip when the multiplet path already folded per-line fractions
                # into a_vec (avoids double-applying the fraction).
                if use_j_resolve and p is not None and not _jr_multiplet_applied:
                    _peak_rank = p.get('rank')
                    # Prefer per-sample fraction vector (achado A): preserves
                    # group differences in overlapped regions.  Falls back to the
                    # mean scalar fraction when no per-sample vector is present.
                    _frac_arr = _j_resolve_contribs_ps.get(_peak_rank, {}).get(mid)
                    if _frac_arr is not None and len(_frac_arr) == len(a_vec):
                        _fa = np.asarray(_frac_arr, dtype=float)
                        _fa = np.clip(_fa, 0.0, 1.0)
                        a_vec = a_vec * _fa
                        _j_frac_this_met = float(np.mean(_fa))
                    else:
                        _jrc = _j_resolve_contribs.get(_peak_rank, {})
                        _frac = _jrc.get(mid)
                        if _frac is not None and 0.0 < _frac <= 1.0:
                            a_vec = a_vec * _frac
                            _j_frac_this_met = _frac

                _blk_mean_area = 0.0
                if use_blk_sub and blk_indices:
                    _blk_mean_area = float(np.mean([a_vec[i] for i in blk_indices
                                                    if i < len(a_vec)]))
                    a_vec = a_vec - _blk_mean_area
                    a_vec = np.maximum(a_vec, 0.0)
                # ── Per-metabolite diagnostic log ─────────────────────────────
                if _dbg_f is not None:
                    _dbg(_dbg_f, '')
                    _dbg(_dbg_f, f'  [{info["name"]}]  mid={mid}')
                    # Peak
                    _dbg(_dbg_f, f'    peak       : rank={p.get("rank")}  ppm={p.get("ppm"):.4f}'
                                 f'  snr={p.get("snr", 0):.1f}  width={p.get("width_ppm", p.get("width_at_half", 0)):.5f} ppm'
                                 f'  is_sub_peak={p.get("is_sub_peak", False)}')
                    # Integration window
                    _ppm_lo = float(ppm_arr[lo_i]) if lo_i < len(ppm_arr) else 0.0
                    _ppm_hi = float(ppm_arr[hi_i]) if hi_i < len(ppm_arr) else 0.0
                    _dbg(_dbg_f, f'    window     : idx=[{lo_i}:{hi_i}]  ppm=[{min(_ppm_lo,_ppm_hi):.4f}:{max(_ppm_lo,_ppm_hi):.4f}]'
                                 f'  Δppm={abs(_ppm_hi-_ppm_lo):.5f}')
                    # Annotation details
                    _mult_str = (hit.get('multiplicity') or hit.get('pattern') or '?')
                    _j_str    = str(hit.get('j_coupling') or '—')
                    _db_ppm   = float(hit.get('ppm_center', hit.get('db_ppm', hit.get('ppm', 0))) or 0)
                    _score    = float(hit.get('composite_score', hit.get('multiplet_score', 0)) or 0)
                    _dbg(_dbg_f, f'    annotation : mult={_mult_str}  J={_j_str} Hz  protons={hit.get("protons")}  '
                                 f'db_ppm={_db_ppm:.4f}  score={_score:.4f}')
                    # nH with source
                    _, _nh_src = _effective_nh_source(hit, detected_ppm, _spec_mhz)
                    _dbg(_dbg_f, f'    nH         : {nh:.6f}  ← {_nh_src}')
                    # Integration mode + Lorentz params
                    _im = ("line_sel+trapz" if _line_sel_applied
                           else "lorentz" if use_lorentz
                           else "mcr" if use_mcr else "trapz")
                    _lp = _lorentz_params.get(mid)
                    if _lp and use_lorentz and not _line_sel_applied:
                        _r2_lp = float('nan')
                        try:
                            _xl = ppm_arr[lo_i:hi_i+1]
                            _yl = np.mean(all_int[lo_i:hi_i+1, :], axis=1)
                            _yfit = _lp['wid']**2 / ((_xl - _lp['cen'])**2 + _lp['wid']**2)
                            _yfit = _yfit / max(_yfit.max(), 1e-30) * _yl.max()
                            _ss_res = float(np.sum((_yl - _yfit)**2))
                            _ss_tot = float(np.sum((_yl - _yl.mean())**2))
                            _r2_lp  = 1.0 - _ss_res / _ss_tot if _ss_tot > 1e-30 else float('nan')
                        except Exception:
                            pass
                        _r2_warn = '  ⚠ BAD FIT' if np.isfinite(_r2_lp) and _r2_lp < 0.5 else ''
                        _dbg(_dbg_f, f'    integ      : {_im}  cen={_lp["cen"]:.4f}  wid={_lp["wid"]:.5f} ppm  R²={_r2_lp:.4f}{_r2_warn}')
                        if _lp.get('shifts'):
                            _sh = _lp['shifts']
                            _dbg(_dbg_f, f'    lz_shifts  : {[f"{v:+.4f}" for v in _sh]} ppm')
                    else:
                        _dbg(_dbg_f, f'    integ      : {_im}')
                    _dbg(_dbg_f, f'    area/nH    : {[f"{v:.4e}" for v in a_vec.tolist()]}')
                    if _line_sel_applied:
                        _dbg(_dbg_f,
                             f'    line_sel   : mode={_ls_mode}'
                             f'  clean={_ls["clean_ranks"]}'
                             f'  scale={_ls_sf:.3f}'
                             f'  cif={_ls["clean_intensity_fraction"]:.3f}')
                    if use_blk_sub and blk_indices:
                        _dbg(_dbg_f, f'    blk_mean   : {_blk_mean_area:.4e}')
                    if use_j_resolve and _j_frac_this_met is not None:
                        _dbg(_dbg_f, f'    j_frac     : {_j_frac_this_met:.4f}')
                # ──────────────────────────────────────────────────────────────
                s_areas += a_vec
                _ann_score = _mid_ann_score.get(str(mid))
                if _ann_score is None:
                    _ann_score = max(
                        (float(h.get('composite_score', h.get('multiplet_score', 0)) or 0)
                         for h in info['hits']), default=0.0)
                _used_jr = use_j_resolve and _j_frac_this_met is not None
                _jr_frac = _j_frac_this_met
                # Collect multiplet window and peak-ppm from the chosen hit.
                _mult_win = None
                _mult_ppm = None
                for _h in info['hits']:
                    if '_multiplet_window' in _h:
                        _mult_win = _h['_multiplet_window']
                        # peak_ppm = intensity-weighted centre of predicted_lines
                        # (set in _group_by_metabolite); more accurate than window
                        # centre especially when the window is clipped for overlap.
                        _mult_ppm = _h.get('peak_ppm') or _h.get('ppm_center') or _h.get('ppm')
                        break
                meta_info[mid] = {'name': info['name'], 'hits': info['hits'],
                                  '_peak': p, '_used_mcr': use_mcr and found,
                                  '_lorentz': _lorentz_params.get(mid),
                                  '_mcr': _mcr_info.get(mid),
                                  '_ann_score': _ann_score,
                                  '_nh': nh,
                                  '_blk_mean_area': _blk_mean_area,
                                  '_has_anchor': _hit_has_anchor,
                                  '_used_j_resolve': _used_jr,
                                  '_j_resolve_frac': _jr_frac,
                                  '_align_shifts': _align_shifts.get(mid),
                                  '_multiplet_window': _mult_win,
                                  '_multiplet_ppm':   _mult_ppm,
                                  '_jr_line_fracs':   _jr_line_fracs_store,
                                  '_synthetic_db':    _met_synthetic_db}

            if not found:
                continue
            raw_areas[mid] = s_areas

        if not raw_areas:
            if _fs_mode:
                messagebox.showinfo(
                    'Info',
                    'No MCR component was accepted.\n\n'
                    'Open the MCR Deconvolution window, select the regions,\n'
                    'and click "✓ Accept" on the desired components.')
            else:
                messagebox.showinfo('Info', 'No peaks matched annotation results.')
            return

        if norm_mode == 'PQN':
            # Per-sample PQN: divide each sample by median(area_i_s / mean_area_i).
            _pqn_mids  = list(raw_areas.keys())
            _pqn_mat   = np.array([raw_areas[m] for m in _pqn_mids], dtype=float)
            _pqn_means = np.nanmean(_pqn_mat, axis=1)   # (n_metabolites,)
            _pqn_factors = np.ones(n_s, dtype=float)
            for _s in range(n_s):
                _qs = [_pqn_mat[_i, _s] / _pqn_means[_i]
                       for _i in range(len(_pqn_mids))
                       if _pqn_means[_i] > 0
                       and np.isfinite(_pqn_mat[_i, _s])
                       and _pqn_mat[_i, _s] > 0]
                if len(_qs) >= 3:
                    _pqn_factors[_s] = float(np.median(_qs))
            _pqn_factors[_pqn_factors <= 0] = 1.0
            self._pqn_factor = float(np.median(_pqn_factors))
            for _m in _pqn_mids:
                raw_areas[_m] = raw_areas[_m] / _pqn_factors
        elif norm_mode == 'Total':
            totals = np.sum(np.stack(list(raw_areas.values())), axis=0)
            totals[totals == 0] = 1
            for mid in raw_areas:
                raw_areas[mid] = raw_areas[mid] / totals * 100

        self._last_norm_mode      = norm_mode
        self._last_ref_areas      = list(ref_areas) if ref_areas is not None else None
        self._last_ref_conc       = ref_conc
        self._last_sample_dil     = np.array(_sample_dil, dtype=float)
        self._last_sample_corr    = np.array(_sample_corr, dtype=float)
        self._last_groups         = dict(groups)
        self._last_blk_indices    = list(blk_indices) if blk_indices else []
        self._last_non_blk_indices = list(non_blk_indices) if non_blk_indices else []

        # Precompute Good-only competition count per peak_rank.
        # Used for confidence level — only Good metabolites are considered valid
        # competitors after filtering; Undef/Bad don't affect quantification.
        _good_peak_comp: dict = {}
        for _r_gpc in results:
            _gpc_mid = str(_r_gpc.get('_mid', ''))
            if _gpc_mid not in good_met_ids:
                continue
            if _effective_quality(_r_gpc) == 'bad':
                continue
            _gpc_pr = _r_gpc.get('peak_rank')
            if _gpc_pr is not None:
                _good_peak_comp[_gpc_pr] = _good_peak_comp.get(_gpc_pr, 0) + 1

        self._prog_update(88, 'Normalizing and building results…')
        self._quant_results = []
        for mid, s_areas in raw_areas.items():
            m    = meta_info.get(mid, {})
            p    = m.get('_peak')
            lw   = m.get('_lipid_window')
            _mw_val = m.get('_multiplet_window')
            _mp_ppm = m.get('_multiplet_ppm')
            if _mp_ppm is not None:
                # Multiplet mode: use the intensity-weighted centre of predicted_lines
                # (stored during integration).  More accurate than window centre,
                # and unaffected by overlap-clipping of the integration window.
                ppm_val = float(_mp_ppm)
            elif _mw_val and _mw_val[0] is not None and _mw_val[1] is not None:
                ppm_val = (float(_mw_val[0]) + float(_mw_val[1])) / 2.0
            elif p is not None:
                ppm_val = float(p.get('ppm', 0.0))
            elif lw and lw[0] is not None and lw[1] is not None:
                ppm_val = (float(lw[0]) + float(lw[1])) / 2.0
            else:
                ppm_val = 0.0

            if norm_mode in ('IS', 'IS+PQN') and ref_areas is not None:
                # IS quantification:
                #   conc_analyte = (area_A / nH_A) / (area_IS / nH_IS) × conc_IS
                # Both areas are already divided by their respective nH above.
                # Internal IS: physical dilution cancels in ratio — do NOT apply
                #   dilution_factor. But correction_factor (sample mass, post-
                #   processing normalisation) does NOT cancel → apply separately.
                # External IS: ref_areas is a fixed mean, so dilution does NOT
                #   cancel → apply full _sample_dil (dil × corr).
                if use_ext_is:
                    s_concs = np.array([
                        (s_areas[s] / ref_areas[s] * ref_conc * _sample_dil[s]
                         if ref_areas[s] > 0 else float('nan'))
                        for s in range(n_s)])
                else:
                    s_concs = np.array([
                        (s_areas[s] / ref_areas[s] * ref_conc * _sample_corr[s]
                         if ref_areas[s] > 0 else float('nan'))
                        for s in range(n_s)])
            else:
                s_concs = np.array(s_areas, dtype=float) * _sample_dil

            cv_by_group    = {}
            concs_by_group = {}
            ci_by_group    = {}   # group → (lo, hi) non-parametric bootstrap CI95
            for g, idxs in groups.items():
                vals = [float(s_concs[i]) for i in idxs
                        if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
                cv_by_group[g]    = _cv(vals)
                concs_by_group[g] = vals
                # Bootstrap-2: distribution-free CI of the group mean concentration
                # (robust for non-parametric metabolomics; CI=nan when n<6).
                _pt, _lo, _hi = _bootstrap_fraction_ci(vals)
                ci_by_group[g] = (_lo, _hi)

            if blk_indices:
                blk_vals = [float(s_concs[i]) for i in blk_indices
                            if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
                if blk_vals:
                    concs_by_group['BLK'] = blk_vals
                    cv_by_group['BLK']    = _cv(blk_vals)

            all_vals = [v for g, vv in concs_by_group.items()
                        if g != 'BLK' for v in vv
                        if np.isfinite(v) and v > 0]
            mean_conc = float(np.mean(all_vals)) if all_vals else float('nan')

            if _dbg_f is not None:
                _m_name = meta_info.get(mid, {}).get('name', mid)
                _dbg(_dbg_f, '')
                _dbg(_dbg_f, f'  [{_m_name}]  CONCENTRATIONS  mid={mid}')
                _dbg(_dbg_f, f'    areas/samp : {[f"{v:.4e}" for v in s_areas.tolist()]}')
                _dbg(_dbg_f, f'    conc/samp  : {[f"{v:.4f}" for v in s_concs.tolist()]} mM')
                for _g, _gv in concs_by_group.items():
                    _gcv = cv_by_group.get(_g, float('nan'))
                    _gmn = float(np.mean(_gv)) if _gv else float('nan')
                    _cv_flag = '  ⚠ HIGH-CV' if np.isfinite(_gcv) and _gcv > 50 else \
                               '  ! WARN-CV' if np.isfinite(_gcv) and _gcv > 30 else ''
                    _dbg(_dbg_f, f'    {_g:<8s}   : n={len(_gv)}  '
                                 f'mean={_gmn:.4f} mM  CV={_gcv:.1f}%{_cv_flag}')
                # Group ratio warning (largest/smallest group mean)
                _gmeans = {g: float(np.mean(v)) for g, v in concs_by_group.items()
                           if g != 'BLK' and v and np.isfinite(np.mean(v))}
                if len(_gmeans) >= 2:
                    _gvals = list(_gmeans.values())
                    _gratio = max(_gvals) / min(_gvals) if min(_gvals) > 0 else float('nan')
                    if np.isfinite(_gratio) and _gratio > 2.0:
                        _dbg(_dbg_f, f'    ⚠ GROUP-RATIO: {_gratio:.1f}×  '
                                     f'({" vs ".join(f"{g}={v:.2f}" for g,v in _gmeans.items())})')
                _dbg(_dbg_f, f'    mean_conc  : {mean_conc:.4f} mM')

            _mcr_met  = (m.get('_mcr') or {}).get('metrics', {})
            _lof      = _mcr_met.get('lof', float('nan'))
            # R²: prefer MCR fit quality; fall back to peak lineshape fit (gaussian_r2)
            _r2       = _mcr_met.get('r2', float('nan'))
            if not np.isfinite(_r2) and p:
                _r2 = float(p.get('gaussian_r2', float('nan')))
            cvs_finite = [v for g, v in cv_by_group.items()
                          if g != 'BLK' and np.isfinite(v)]
            worst_cv   = max(cvs_finite) if cvs_finite else float('nan')
            _snr_ps, _noise_lvl = _per_sample_snr(p, all_int, n_s)
            _nb_snrs = [_snr_ps[i] for i in non_blk_indices
                        if i < n_s and np.isfinite(_snr_ps[i])]
            _mean_snr = (float(np.mean(_nb_snrs)) if _nb_snrs
                         else (p.get('snr', 0) if p else 0))
            _det_min_snr = self._detection_min_snr()
            # Only flag a sample as NOT detected when we positively know its SNR
            # is below threshold; unknown SNR (NaN) stays "detected" so we never
            # hide a value without evidence of absence.
            _detected_ps = [bool((not np.isfinite(v)) or v >= _det_min_snr)
                            for v in _snr_ps]
            # Per-sample detection presence (from per-sample detection mode) is
            # authoritative for absence: a sample where the apex was not detected
            # is n.d., never a noise-derived value.
            _apk = (p or {}).get('aligned_per_sample')
            if _apk is not None and len(_apk) == len(_detected_ps):
                _detected_ps = [bool(d and a) for d, a in zip(_detected_ps, _apk)]

            _criteria = []
            _reasons  = []
            if np.isfinite(worst_cv):
                _s = 0 if worst_cv < 30 else 1 if worst_cv < 50 else 2
                _criteria.append(_s)
                _lbl = 'Good' if _s == 0 else 'Warn' if _s == 1 else 'Poor'
                _thr = '< 30%' if _s == 0 else '30–50%' if _s == 1 else '> 50%'
                _reasons.append(f'CV%: {worst_cv:.1f}%  →  {_lbl} ({_thr})')
            if np.isfinite(_lof):
                _s = 0 if _lof < 10 else 1 if _lof < 20 else 2
                _criteria.append(_s)
                _lbl = 'Good' if _s == 0 else 'Warn' if _s == 1 else 'Poor'
                _thr = '< 10%' if _s == 0 else '10–20%' if _s == 1 else '> 20%'
                _reasons.append(f'LOF: {_lof:.1f}%  →  {_lbl} ({_thr})')
            if np.isfinite(_r2):
                _s = 0 if _r2 > 0.95 else 1 if _r2 > 0.90 else 2
                _criteria.append(_s)
                _lbl = 'Good' if _s == 0 else 'Warn' if _s == 1 else 'Poor'
                _thr = '> 0.95' if _s == 0 else '0.90–0.95' if _s == 1 else '< 0.90'
                _reasons.append(f'R²: {_r2:.4f}  →  {_lbl} ({_thr})')
            _worst = max(_criteria) if _criteria else 1
            if not _reasons:
                _reasons = ['All parameters within good range']

            # ── Confidence level (1–3) ────────────────────────────────────────
            # Use the rank of the actual peak object used for quantification/display.
            # Falls back to the annotation hit's peak_rank only when no peak was found.
            _peak_rank_val = (p.get('rank') if p is not None
                              else (m.get('hits', [{}])[0].get('peak_rank')
                                    if m.get('hits') else None))
            _has_anchor   = m.get('_has_anchor', False)
            _used_jr      = m.get('_used_j_resolve', False)

            # Count Good-metabolite competitors for this peak_rank.
            # Undef/Bad metabolites excluded — only Good-level competitors matter.
            _jr_stage      = getattr(getattr(self, 'app', None),
                                     'stages', {}).get('j_resolve')
            _peak_overlap  = getattr(_jr_stage, '_peak_overlap', {})
            _nnls_rejected = getattr(_jr_stage, '_nnls_rejected', set())
            _n_elig = _good_peak_comp.get(_peak_rank_val, 1) if _peak_rank_val is not None else 1

            # Level 3: anchor exclusive OR NNLS resolved successfully
            _is_nnls_ok = _used_jr and (mid not in _nnls_rejected)
            if _has_anchor or _is_nnls_ok:
                _conf = 3
                _conf_reason = ('Conf 3: NNLS resolved' if _is_nnls_ok
                                else 'Conf 3: anchor peak')
            elif _n_elig <= 1 and (_r2 > 0.90 or _mean_snr > 10):
                _conf = 2
                _conf_reason = f'Conf 2: R²={_r2:.3f}, SNR={_mean_snr:.1f}, ≤1 eligible competitor'
            else:
                _conf = 1
                _conf_reason = (f'Conf 1: {_n_elig} eligible competitors'
                                if _n_elig > 1 else 'Conf 1: low R²/SNR')

            # Append NNLS reject note if applicable
            if mid in _nnls_rejected:
                _reasons.append('NNLS ✗: rejected by quality filter')

            # Warn when selected hit has contaminated/skipped line selection.
            # Only meaningful when J-Resolve is active — stale _line_selection
            # from a prior run must not trigger false warnings.
            _sel_hit = (m.get('hits') or [{}])[0]
            _sel_ls  = _sel_hit.get('_line_selection', {}) if use_j_resolve else {}
            if _sel_ls.get('mode') == 'skip':
                _reasons.append(
                    f'⚠ All multiplet lines contaminated '
                    f'(cif={_sel_ls.get("clean_intensity_fraction", 0):.2f}) '
                    f'— quantification unreliable')
            elif _sel_ls.get('contaminated_ranks'):
                _n_cont = len(_sel_ls['contaminated_ranks'])
                _reasons.append(
                    f'⚠ {_n_cont} line(s) excluded (overlap) — '
                    f'area scaled ×{_sel_ls.get("scale_factor", 1):.2f}')

            _conf_bar = {3: '●●●', 2: '●●○', 1: '●○○'}[_conf]
            _reasons.insert(0, _conf_reason)
            status = _conf_bar

            self._quant_results.append({
                'metabolite':        m.get('name', mid),
                'hmdb_id':           mid,
                'ppm':               ppm_val,
                'n_peaks':           len(m.get('hits', [])),
                'raw_area':          float(np.mean(s_areas)),
                'corr_area':         float(np.mean(s_areas)),
                'conc_mm':           mean_conc,
                'cv_by_group':       cv_by_group,
                'snr':               _mean_snr,
                'status':            status,
                '_confidence':       _conf,
                '_status_reasons':   _reasons,
                '_peak':             p,
                '_areas_per_sample': s_areas.tolist(),
                '_concs_per_sample': s_concs.tolist(),
                '_snr_per_sample':       _snr_ps,
                '_detected_per_sample':  _detected_ps,
                '_concs_by_group':   concs_by_group,
                '_conc_ci_by_group': ci_by_group,
                '_used_mcr':         m.get('_used_mcr', False),
                '_lorentz':          m.get('_lorentz'),
                '_mcr':              m.get('_mcr'),
                '_lipid_regions':    m.get('_lipid_regions'),
                '_lipid_window':     m.get('_lipid_window'),
                '_integ_mode':       _integ,
                '_ann_score':        m.get('_ann_score', 0.0),
                '_nh':               m.get('_nh', 1.0),
                '_blk_mean_area':    m.get('_blk_mean_area', 0.0),
                '_proton_corr':      proton_corr,
                '_use_blk_sub':      use_blk_sub,
                '_multiplet_window': m.get('_multiplet_window'),
                '_is_singlet':       ((_sel_hit.get('multiplicity')
                                       or _sel_hit.get('type') or '')
                                      .strip().lower() in _SINGLET_MULTS),
                '_has_anchor':       _has_anchor,
                '_peak_rank':        _peak_rank_val,
                '_used_j_resolve':   _used_jr,
                '_j_resolve_frac':   m.get('_j_resolve_frac'),
                '_jr_line_fracs':    m.get('_jr_line_fracs'),
                '_n_eligible':       _n_elig,
                '_peak_align_shifts': m.get('_align_shifts'),
                '_hits':             m.get('hits', []),
            })

        # ── Lipid composition indices (lipidomics mode only) ──────────────────
        if is_lipidomics_mode:
            _lipid_idx_entries = self._compute_lipid_composition_indices(
                ppm_arr, all_int, n_s, groups)
            self._quant_results.extend(_lipid_idx_entries)

        # ── IS + PQN post-processing ──────────────────────────────────────────
        # Apply PQN correction on top of IS-normalised mM concentrations.
        # For each sample, the PQN factor = median(conc_i_s / mean_conc_i).
        # Dividing by the factor corrects residual dilution differences while
        # preserving the mM scale anchored by the IS.
        if norm_mode == 'IS+PQN' and self._quant_results:
            # Build matrix: rows = metabolites, cols = samples (skip derived indices)
            conc_matrix = np.array([r['_concs_per_sample']
                                    for r in self._quant_results
                                    if not r.get('_is_lipid_index')], dtype=float)
            n_res = conc_matrix.shape[0]
            mean_concs  = np.nanmean(conc_matrix, axis=1)  # (n_metabolites,)
            pqn_factors = np.ones(n_s, dtype=float)
            for s in range(n_s):
                quotients = []
                for i in range(n_res):
                    if mean_concs[i] > 0 and np.isfinite(conc_matrix[i, s]):
                        quotients.append(conc_matrix[i, s] / mean_concs[i])
                if len(quotients) >= 3:
                    pqn_factors[s] = float(np.median(quotients))
            pqn_factors[pqn_factors <= 0] = 1.0
            self._pqn_factor = float(np.median(pqn_factors))
            # Rescale all results (skip ratio-based lipid indices)
            for r in self._quant_results:
                if r.get('_is_lipid_index'):
                    continue
                old_concs = np.array(r['_concs_per_sample'], dtype=float)
                new_concs = old_concs / pqn_factors
                r['_concs_per_sample'] = new_concs.tolist()
                r['conc_mm'] = float(np.nanmean(new_concs))
                # Recompute cv_by_group and concs_by_group
                for g, idxs in groups.items():
                    vals = [float(new_concs[i]) for i in idxs
                            if i < n_s and np.isfinite(new_concs[i])]
                    r['cv_by_group'][g] = _cv(vals)
                    r['_concs_by_group'][g] = vals

        if ref_peak is not None and norm_mode in ('IS', 'IS+PQN') and ref_areas is not None:
            lo_r = min(ref_peak.get('start_idx', 0), ref_peak.get('end_idx', 0))
            hi_r = max(ref_peak.get('start_idx', 0), ref_peak.get('end_idx', 0))
            is_areas_raw = np.array([
                abs(float(_trapz(all_int[lo_r:hi_r+1, s], ppm_arr[lo_r:hi_r+1])))
                for s in range(n_s)])
            is_concs = np.array([
                ((is_areas_raw[s] / ref_nh) / ref_areas[s] * ref_conc
                 if ref_areas[s] > 0 else float('nan'))
                for s in range(n_s)])
            is_cv_by_group    = {}
            is_concs_by_group = {}
            for g, idxs in groups.items():
                vals = [float(is_concs[i]) for i in idxs
                        if i < n_s and np.isfinite(is_concs[i]) and is_concs[i] > 0]
                is_cv_by_group[g]    = _cv(vals)
                is_concs_by_group[g] = vals
            is_all_vals = [v for g, vv in is_concs_by_group.items()
                           if g != 'BLK' for v in vv if np.isfinite(v) and v > 0]
            is_mean_conc = float(np.mean(is_all_vals)) if is_all_vals else float('nan')
            is_cvs = [v for v in is_cv_by_group.values() if np.isfinite(v)]
            is_worst_cv = max(is_cvs) if is_cvs else float('nan')
            is_status = ('good' if np.isfinite(is_worst_cv) and is_worst_cv < 20
                         else 'warn' if np.isfinite(is_worst_cv) and is_worst_cv < 40
                         else 'poor')
            _is_name = self.ref_std_var.get() or 'IS'
            self._quant_results.append({
                'metabolite':        f'{_is_name} [IS]',
                'hmdb_id':           '__IS__',
                'ppm':               ref_peak.get('ppm', 0.0),
                'n_peaks':           1,
                'raw_area':          float(np.mean(is_areas_raw)),
                'corr_area':         float(np.mean(is_areas_raw)),
                'conc_mm':           is_mean_conc,
                'cv_by_group':       is_cv_by_group,
                'snr':               ref_peak.get('snr', 0),
                'status':            is_status,
                '_peak':             ref_peak,
                '_concs_per_sample': is_concs.tolist(),
                '_concs_by_group':   is_concs_by_group,
                '_used_mcr':         False,
                '_lorentz':          None,
                '_mcr':              None,
                '_integ_mode':       _integ,
                '_ann_score':        1.0,
                '_is_is':            True,
            })

        # ── IS Recovery log ───────────────────────────────────────────────────────
        if _dbg_f is not None and ref_peak is not None and norm_mode in ('IS', 'IS+PQN'):
            _dbg(_dbg_f, '')
            _dbg(_dbg_f, '── IS Recovery ─────────────────────────────────────────')
            _dbg(_dbg_f, f'  IS name      : {self.ref_std_var.get() or "IS"}  @{ref_peak.get("ppm", 0):.4f} ppm')
            _dbg(_dbg_f, f'  expected     : {ref_conc:.4f} mM  (nH={ref_nh})')
            _dbg(_dbg_f, f'  measured     : {is_mean_conc:.4f} mM')
            _rec = is_mean_conc / ref_conc * 100 if ref_conc > 0 and np.isfinite(is_mean_conc) else float('nan')
            _rec_flag = ('  ⚠ LOW (<70%)' if np.isfinite(_rec) and _rec < 70 else
                         '  ⚠ HIGH (>130%)' if np.isfinite(_rec) and _rec > 130 else
                         '  ✓ OK' if np.isfinite(_rec) else '')
            _dbg(_dbg_f, f'  recovery     : {_rec:.1f}%{_rec_flag}')
            _dbg(_dbg_f, f'  IS snr       : {ref_peak.get("snr", 0):.1f}')
            _dbg(_dbg_f, f'  use_ext_is   : {use_ext_is}')
            if use_ext_is:
                _dbg(_dbg_f, f'  ext_IS_idx   : {self._ext_is_indices}')
                _dbg(_dbg_f, f'  ref_area/nH  : {ref_areas[0]:.4e}  (fixed mean from ext IS samples)')
            else:
                _dbg(_dbg_f, f'  ref_areas/nH : {[f"{v:.4e}" for v in ref_areas]}')
            for _g, _gv in is_concs_by_group.items():
                _gcv = is_cv_by_group.get(_g, float('nan'))
                _dbg(_dbg_f, f'  {_g:<8s}     : {[f"{v:.4f}" for v in _gv]} mM  CV={_gcv:.1f}%')

        # ── All-candidates quantification (for Quant Peaks window) ──────────────
        # Quantify every (mid, peak_rank, ppm_rounded) triplet in annotation results.
        # Sub-peaks share the same peak_rank but have distinct ppm → kept separate.
        # Stored as self._all_peak_quant = {mid: {(peak_rank, ppm_r): {...}} }
        # Also stores {mid: {peak_rank: {...}}} for backward compat (first sub-peak wins).
        self._all_peak_quant = {}
        if not is_lipidomics_mode and not _fs_mode:
            _seen_apq = set()
            for _r in results:
                _apq_mid = str(_r.get('_mid', ''))
                _apq_pr  = _r.get('peak_rank')
                if not _apq_mid or _apq_pr is None:
                    continue
                _apq_hit_ppm = float(_r.get('peak_ppm', _r.get('ppm', 0)) or 0)
                _apq_ppm_r   = round(_apq_hit_ppm, 3)
                _apq_key = (_apq_mid, _apq_pr, _apq_ppm_r)
                if _apq_key in _seen_apq:
                    continue
                _seen_apq.add(_apq_key)

                _apq_p = (_pk_by_rank.get(_r.get('peak_rank')) or
                          self._find_peak(_r.get('peak_ppm', _r.get('ppm', 0))))
                if _apq_p is None:
                    continue

                _apq_lo = min(_apq_p.get('start_idx', 0), _apq_p.get('end_idx', 0))
                _apq_hi = max(_apq_p.get('start_idx', 0), _apq_p.get('end_idx', 0))
                if _apq_hi <= _apq_lo:
                    _apq_hi = _apq_lo + 1
                _apq_hi = min(_apq_hi, all_int.shape[0] - 1)

                _apq_det_ppm = float(_r.get('peak_ppm', _r.get('ppm', _apq_p.get('ppm', 0))))
                # Fill protons from DB fallback for old annotation results.
                if not _r.get('protons') and _db_protons:
                    _ar_id  = str(_r.get('hmdb_id') or _r.get('bmrb_id') or '')
                    _ar_ppm = round(float(
                        _r.get('ppm_center') or _r.get('db_ppm') or _r.get('ppm') or 0), 3)
                    _ar_p = _db_protons.get((_ar_id, _ar_ppm))
                    if _ar_p:
                        _r = dict(_r)   # don't mutate original
                        _r['protons'] = _ar_p
                # Use fractional nH whenever the hit represents a specific line of a
                # multiplet (has multiple predicted_lines), regardless of is_sub_peak.
                # is_sub_peak only reflects peak-detector clustering, not multiplicity.
                _apq_pred_lines_n = len(_r.get('predicted_lines', []))
                if _apq_pred_lines_n > 1:
                    _apq_nh = max(1e-6, _effective_nh(_r, _apq_det_ppm, _spec_mhz))
                else:
                    _apq_nh = max(1.0, float(_r.get('protons', 1) or 1))


                if use_lorentz:
                    _ppm_seg = ppm_arr[_apq_lo:_apq_hi + 1]
                    _int_seg = np.mean(all_int[_apq_lo:_apq_hi + 1, :], axis=1)
                    _A_l, _cen_l, _wid_l, _apq_bl_y0, _apq_bl_slope = \
                        _fit_lorentz_mean_with_bl(_ppm_seg, _int_seg)
                    _max_sh = max(abs(float(_ppm_seg[-1] - _ppm_seg[0])) * 0.15, 0.005)
                    _apq_shifts = []
                    _apq_avec = np.zeros(n_s, dtype=float)
                    for _s in range(n_s):
                        _res_s = _fit_lorentz_sample(
                            _ppm_seg, all_int[_apq_lo:_apq_hi + 1, _s],
                            _A_l, _cen_l, _wid_l, max_shift_ppm=_max_sh)
                        if _res_s is not None:
                            _apq_avec[_s] = _res_s[0]
                            _apq_shifts.append(_res_s[2] - _cen_l)
                        else:
                            _apq_avec[_s] = abs(float(_trapz(
                                all_int[_apq_lo:_apq_hi + 1, _s], _ppm_seg)))
                            _apq_shifts.append(0.0)
                    _apq_lorentz = {
                        'cen':      _cen_l,        'wid':      _wid_l,
                        'shifts':   _apq_shifts,
                        'bl_y0':    _apq_bl_y0,    'bl_slope': _apq_bl_slope,
                        'bl_x0':    float(_ppm_seg[0]),
                    }
                else:
                    _apq_avec = np.array([
                        abs(float(_trapz(all_int[_apq_lo:_apq_hi + 1, s],
                                         ppm_arr[_apq_lo:_apq_hi + 1])))
                        for s in range(n_s)
                    ], dtype=float)

                # Mirror main pipeline: divide by nH only when IS mode or proton_corr
                if norm_mode in ('IS', 'IS+PQN') or proton_corr:
                    _apq_avec /= _apq_nh

                if use_blk_sub and blk_indices:
                    _apq_blk = float(np.mean([_apq_avec[i] for i in blk_indices
                                              if i < len(_apq_avec)]))
                    _apq_avec -= _apq_blk

                if norm_mode in ('IS', 'IS+PQN') and ref_areas is not None:
                    _apq_concs = np.array([
                        (_apq_avec[s] / ref_areas[s] * ref_conc
                         if ref_areas[s] > 0 else float('nan'))
                        for s in range(n_s)
                    ])
                else:
                    _apq_concs = _apq_avec * _sample_dil

                _apq_cbg = {}
                for _g, _idxs in groups.items():
                    _gv = [float(_apq_concs[i]) for i in _idxs
                           if i < n_s and np.isfinite(_apq_concs[i]) and _apq_concs[i] > 0]
                    _apq_cbg[_g] = _gv

                _apq_all = [v for _gv in _apq_cbg.values() for v in _gv]
                _apq_mean = float(np.mean(_apq_all)) if _apq_all else float('nan')

                _apq_entry = {
                    'conc_mm':           _apq_mean,
                    'concs_by_group':    _apq_cbg,
                    'ppm':               float(_apq_p.get('ppm', 0)),
                    '_lorentz':          _apq_lorentz if use_lorentz else None,
                    '_concs_per_sample': _apq_concs.tolist(),
                    '_peak':             _apq_p,
                    '_nh':               _apq_nh,
                }
                _mid_dict = self._all_peak_quant.setdefault(_apq_mid, {})
                # Primary key: (peak_rank, ppm_rounded) — unique per sub-peak line
                _mid_dict[(_apq_pr, _apq_ppm_r)] = _apq_entry
                # Backward-compat key: peak_rank alone (first sub-peak wins)
                if _apq_pr not in _mid_dict:
                    _mid_dict[_apq_pr] = _apq_entry
                _dbg(_dbg_f, f'  [APQ] {_apq_mid}  peak={_apq_pr}  hit_ppm={_apq_ppm_r}'
                             f'  det_ppm={_apq_p.get("ppm"):.4f}'
                             f'  nh={_apq_nh:.4f}  mean_mM={_apq_mean:.4f}'
                             f'  by_group={ {g: f"{np.mean(v):.4f}" if v else "—" for g, v in _apq_cbg.items()} }')

        # Extend _all_peak_quant with multiplet lines from annotation results tree.
        # _iid_to_subchild has (met, dp, mp_ref, line_dict, line_idx, nearest_pk)
        # already resolved — no extra search needed.
        _annot_win = getattr(getattr(self, 'app', None), '_annot_win', None)
        _iid_to_sc = getattr(_annot_win, '_iid_to_subchild', {}) or {}
        for _sc in _iid_to_sc.values():
            try:
                _sc_met, _sc_dp, _sc_mp, _sc_line, _sc_li, _sc_pk = _sc
            except (ValueError, TypeError):
                continue
            if _sc_pk is None:
                continue
            _sc_mid = str(_sc_met.get('id', ''))
            _sc_pr  = _sc_pk.get('rank')
            if not _sc_mid or _sc_pr is None:
                continue
            _sc_ppm_r = round(float(_sc_pk.get('ppm', 0)), 3)
            _mid_dict = self._all_peak_quant.setdefault(_sc_mid, {})
            # Skip if already computed from results loop above
            if (_sc_pr, _sc_ppm_r) in _mid_dict:
                continue
            _sc_lo = min(_sc_pk.get('start_idx', 0), _sc_pk.get('end_idx', 0))
            _sc_hi = max(_sc_pk.get('start_idx', 0), _sc_pk.get('end_idx', 0))
            if _sc_hi <= _sc_lo:
                _sc_hi = _sc_lo + 1
            _sc_hi = min(_sc_hi, all_int.shape[0] - 1)
            # Build a hit-like dict from the subchild data for _effective_nh.
            # dp (DB peak dict) has protons, type, j_coupling, predicted_lines.
            _sc_det_ppm = float(_sc_pk.get('ppm', 0))
            _sc_hit = dict(_sc_dp or {})
            _sc_hit['anchor_line_idx'] = _sc_li
            _sc_pred = _sc_dp.get('predicted_lines', []) if _sc_dp else []
            if not _sc_pred and _sc_line:
                # Build minimal pred_lines from what we know about this line
                _sc_pred = [{'ppm': float(_sc_line.get('ppm', 0)),
                             'rel_intensity': float(_sc_line.get('rel_intensity', 1.0))}]
            _sc_hit['predicted_lines'] = _sc_pred
            if len(_sc_pred) > 1:
                _sc_nh = max(1e-6, _effective_nh(_sc_hit, _sc_det_ppm, _spec_mhz))
            else:
                _sc_nh = max(1.0, float((_sc_mp or {}).get('protons', 1) or 1))
            _sc_avec = np.array([
                abs(float(_trapz(all_int[_sc_lo:_sc_hi + 1, s],
                                 ppm_arr[_sc_lo:_sc_hi + 1])))
                for s in range(n_s)
            ], dtype=float)
            if norm_mode in ('IS', 'IS+PQN') or proton_corr:
                _sc_avec /= _sc_nh
            if use_blk_sub and blk_indices:
                _sc_avec -= float(np.mean([_sc_avec[i] for i in blk_indices
                                           if i < len(_sc_avec)]))
            if norm_mode in ('IS', 'IS+PQN') and ref_areas is not None:
                _sc_concs = np.array([
                    (_sc_avec[s] / ref_areas[s] * ref_conc
                     if ref_areas[s] > 0 else float('nan'))
                    for s in range(n_s)
                ])
            else:
                _sc_concs = _sc_avec * _sample_dil
            _sc_cbg = {}
            for _g, _idxs in groups.items():
                _gv = [float(_sc_concs[i]) for i in _idxs
                       if i < n_s and np.isfinite(_sc_concs[i]) and _sc_concs[i] > 0]
                _sc_cbg[_g] = _gv
            _sc_all  = [v for _gv in _sc_cbg.values() for v in _gv]
            _sc_mean = float(np.mean(_sc_all)) if _sc_all else float('nan')
            _sc_entry = {'conc_mm': _sc_mean, 'concs_by_group': _sc_cbg,
                         'ppm': float(_sc_pk.get('ppm', 0))}
            _mid_dict[(_sc_pr, _sc_ppm_r)] = _sc_entry
            if _sc_pr not in _mid_dict:
                _mid_dict[_sc_pr] = _sc_entry

        # ── Fill-in loop: quantify ALL matched peaks from annotation metabolites ──
        # The two loops above only reach peaks in `results` (anchor lines) and
        # peaks already in _iid_to_subchild (requires annotation window open).
        # This loop ensures every sibling line of a multiplet gets APQ data,
        # using _effective_nh for correct fractional-proton weighting.
        _results_by_rank = {}
        for _r2 in results:
            _r2_mid = str(_r2.get('_mid', ''))
            _r2_pr  = _r2.get('peak_rank')
            if _r2_mid and _r2_pr is not None:
                _results_by_rank[(_r2_mid, _r2_pr)] = _r2

        _fill_mets = getattr(getattr(self, 'annotation_stage', None),
                             '_metabolites', None) or []
        for _fm in _fill_mets:
            _fm_mid = str(_fm.get('id', ''))
            if not _fm_mid:
                continue
            for _fmp in _fm.get('matched_peaks', []):
                _fm_pr = _fmp.get('peak_rank')
                if _fm_pr is None:
                    continue
                _mid_dict = self._all_peak_quant.setdefault(_fm_mid, {})
                # Skip if already computed by the two loops above
                if _fm_pr in _mid_dict:
                    continue
                # Prefer rank-based lookup; fall back to DB ppm proximity.
                _fm_db_ppm = float(_fmp.get('db_ppm', _fmp.get('ppm', 0)) or 0)
                _fm_p = _pk_by_rank.get(_fm_pr) or self._find_peak(_fm_db_ppm)
                if _fm_p is None:
                    continue
                _fm_ppm_r = round(float(_fm_p.get('ppm', 0)), 3)
                if (_fm_pr, _fm_ppm_r) in _mid_dict:
                    _mid_dict[_fm_pr] = _mid_dict[(_fm_pr, _fm_ppm_r)]
                    continue
                _fm_lo = min(_fm_p.get('start_idx', 0), _fm_p.get('end_idx', 0))
                _fm_hi = max(_fm_p.get('start_idx', 0), _fm_p.get('end_idx', 0))
                if _fm_hi <= _fm_lo:
                    _fm_hi = _fm_lo + 1
                _fm_hi = min(_fm_hi, all_int.shape[0] - 1)
                _fm_det_ppm = float(_fm_p.get('ppm', _fm_db_ppm))
                # Prefer result entry for this (mid, rank) — has anchor_line_idx
                # and predicted_lines needed for _effective_nh.
                _fm_hit = _results_by_rank.get((_fm_mid, _fm_pr), _fmp)
                # Fill protons from DB fallback for old annotation results.
                if not _fm_hit.get('protons') and _db_protons:
                    _fh_id  = str(_fm_hit.get('hmdb_id') or _fm_hit.get('bmrb_id') or '')
                    _fh_ppm = round(float(
                        _fm_hit.get('ppm_center') or _fm_hit.get('db_ppm') or
                        _fm_hit.get('ppm') or 0), 3)
                    _fh_p = _db_protons.get((_fh_id, _fh_ppm))
                    if _fh_p:
                        _fm_hit = dict(_fm_hit)
                        _fm_hit['protons'] = _fh_p
                _fm_pred_n = len(_fm_hit.get('predicted_lines', []))
                if _fm_pred_n > 1:
                    _fm_nh = max(1e-6, _effective_nh(_fm_hit, _fm_det_ppm, _spec_mhz))
                else:
                    _fm_nh = max(1.0, float(_fm_hit.get('protons', 1) or 1))

                if use_lorentz:
                    _fm_seg  = ppm_arr[_fm_lo:_fm_hi + 1]
                    _fm_iseg = np.mean(all_int[_fm_lo:_fm_hi + 1, :], axis=1)
                    _fm_A, _fm_cen, _fm_wid, _fm_bl_y0, _fm_bl_slope = \
                        _fit_lorentz_mean_with_bl(_fm_seg, _fm_iseg)
                    _fm_max_sh = max(abs(float(_fm_seg[-1] - _fm_seg[0])) * 0.15, 0.005)
                    _fm_avec   = np.zeros(n_s, dtype=float)
                    _fm_shifts = []
                    for _s in range(n_s):
                        _fm_rs = _fit_lorentz_sample(
                            _fm_seg, all_int[_fm_lo:_fm_hi + 1, _s],
                            _fm_A, _fm_cen, _fm_wid, max_shift_ppm=_fm_max_sh)
                        if _fm_rs is not None:
                            _fm_avec[_s] = _fm_rs[0]
                            _fm_shifts.append(_fm_rs[2] - _fm_cen)
                        else:
                            _fm_avec[_s] = abs(float(_trapz(
                                all_int[_fm_lo:_fm_hi + 1, _s], _fm_seg)))
                            _fm_shifts.append(0.0)
                    _fm_lorentz = {'cen': _fm_cen, 'wid': _fm_wid,
                                   'shifts': _fm_shifts, 'bl_y0': _fm_bl_y0,
                                   'bl_slope': _fm_bl_slope,
                                   'bl_x0': float(_fm_seg[0])}
                else:
                    _fm_avec = np.array([
                        abs(float(_trapz(all_int[_fm_lo:_fm_hi + 1, s],
                                         ppm_arr[_fm_lo:_fm_hi + 1])))
                        for s in range(n_s)
                    ], dtype=float)
                    _fm_lorentz = None

                if norm_mode in ('IS', 'IS+PQN') or proton_corr:
                    _fm_avec /= _fm_nh
                if use_blk_sub and blk_indices:
                    _fm_blk = float(np.mean([_fm_avec[i] for i in blk_indices
                                             if i < len(_fm_avec)]))
                    _fm_avec -= _fm_blk
                if norm_mode in ('IS', 'IS+PQN') and ref_areas is not None:
                    _fm_concs = np.array([
                        (_fm_avec[s] / ref_areas[s] * ref_conc
                         if ref_areas[s] > 0 else float('nan'))
                        for s in range(n_s)
                    ])
                else:
                    _fm_concs = _fm_avec * _sample_dil

                _fm_cbg = {}
                for _g, _idxs in groups.items():
                    _gv = [float(_fm_concs[i]) for i in _idxs
                           if i < n_s and np.isfinite(_fm_concs[i])
                           and _fm_concs[i] > 0]
                    _fm_cbg[_g] = _gv
                _fm_all  = [v for _gv in _fm_cbg.values() for v in _gv]
                _fm_mean = float(np.mean(_fm_all)) if _fm_all else float('nan')

                _fm_entry = {'conc_mm': _fm_mean, 'concs_by_group': _fm_cbg,
                             'ppm': float(_fm_p.get('ppm', 0)),
                             '_lorentz': _fm_lorentz}
                _mid_dict[(_fm_pr, _fm_ppm_r)] = _fm_entry
                if _fm_pr not in _mid_dict:
                    _mid_dict[_fm_pr] = _fm_entry
                _dbg(_dbg_f, f'  [APQ-FILL] {_fm_mid}  rank={_fm_pr}  det={_fm_det_ppm:.4f}'
                             f'  nh={_fm_nh:.4f}  mean_mM={_fm_mean:.4f}')

        # ── APQ fallback for metabolites excluded by peak competition ─────────
        # Metabolites that lost peak competition are absent from _quant_results.
        # If APQ computed a valid concentration, add them here so they appear in
        # the output, flagged as _from_apq (status=warn, integ_mode=APQ).
        # APQ fallback only with J-Resolve: without spectral deconvolution, adding
        # metabolites that lost peak competition double-counts the same spectral region.
        if not is_lipidomics_mode and not _fs_mode and use_j_resolve:
            _qr_mids_seen = {
                str(r.get('hmdb_id') or r.get('bmrb_id') or '') for r in self._quant_results
            }
            for _apq_mid_fb, _apq_mid_dict in self._all_peak_quant.items():
                if _apq_mid_fb in _qr_mids_seen or not _apq_mid_fb:
                    continue
                if _apq_mid_fb not in good_met_ids:
                    continue
                _best_apq  = None
                _best_conc = float('-inf')
                for _apq_e in _apq_mid_dict.values():
                    if not isinstance(_apq_e, dict):
                        continue
                    _ec = float(_apq_e.get('conc_mm', float('nan')) or float('nan'))
                    if np.isfinite(_ec) and _ec > _best_conc:
                        _best_conc = _ec
                        _best_apq  = _apq_e
                if _best_apq is None or not np.isfinite(_best_conc):
                    continue
                _apq_cps = np.array(_best_apq.get('_concs_per_sample', []), dtype=float)
                if _apq_cps.size != n_s:
                    continue
                _apq_cbg2 = _best_apq.get('concs_by_group', {})
                _apq_cv2  = {}
                for _g2, _gv2 in _apq_cbg2.items():
                    _apq_cv2[_g2] = (float(np.std(_gv2) / np.mean(_gv2) * 100)
                                     if len(_gv2) >= 2 and np.mean(_gv2) > 0
                                     else float('nan'))
                _apq_pk2  = _best_apq.get('_peak')
                _apq_name = _apq_mid_fb
                for _r_fb in results:
                    if str(_r_fb.get('_mid', '')) == _apq_mid_fb:
                        _apq_name = _r_fb.get('name', _apq_mid_fb)
                        break
                self._quant_results.append({
                    'metabolite':         _apq_name,
                    'hmdb_id':            _apq_mid_fb,
                    'ppm':                float(_best_apq.get('ppm', 0)),
                    'n_peaks':            1,
                    'raw_area':           float(np.nanmean(_apq_cps)),
                    'corr_area':          float(np.nanmean(_apq_cps)),
                    'conc_mm':            _best_conc,
                    'cv_by_group':        _apq_cv2,
                    'snr':                float(_apq_pk2.get('snr', 0)) if _apq_pk2 else 0.0,
                    'status':             '●○○',
                    '_confidence':        1,
                    '_status_reasons':    ['APQ fallback: peak shared with higher-score metabolite'],
                    '_peak':              _apq_pk2,
                    '_areas_per_sample':  _apq_cps.tolist(),
                    '_concs_per_sample':  _apq_cps.tolist(),
                    '_concs_by_group':    _apq_cbg2,
                    '_used_mcr':          False,
                    '_lorentz':           _best_apq.get('_lorentz'),
                    '_mcr':               None,
                    '_lipid_regions':     None,
                    '_lipid_window':      None,
                    '_integ_mode':        'APQ',
                    '_ann_score':         _mid_ann_score.get(str(_apq_mid_fb), 0.0),
                    '_nh':                float(_best_apq.get('_nh', 1.0)),
                    '_blk_mean_area':     0.0,
                    '_proton_corr':       proton_corr,
                    '_use_blk_sub':       use_blk_sub,
                    '_multiplet_window':  None,
                    '_has_anchor':        False,
                    '_peak_rank':         _apq_pk2.get('rank') if _apq_pk2 else None,
                    '_used_j_resolve':    False,
                    '_j_resolve_frac':    None,
                    '_n_eligible':        1,
                    '_peak_align_shifts': None,
                    '_hits':              [],
                    '_from_apq':          True,
                })

        if _dbg_f is not None:
            _dbg(_dbg_f, '')
            _dbg(_dbg_f, '═' * 70)
            _dbg(_dbg_f, 'SUMMARY TABLE  (sorted by mean concentration)')
            _dbg(_dbg_f, '═' * 70)
            # Header
            _grp_names = [g for g in groups if g != 'BLK']
            _hdr_grps  = ''.join(f'  {g[:8]:<10s}' for g in _grp_names)
            _dbg(_dbg_f,
                 f'  {"#":>3}  {"Metabolite":<28s}  {"ppm":>6}  {"mean mM":>8}'
                 + _hdr_grps
                 + f'  {"conf":>4}  {"CV%":>5}  {"SNR":>6}  warnings')
            _dbg(_dbg_f, '  ' + '-' * 120)
            _sorted_res = sorted(
                (r for r in self._quant_results if not r.get('_is_is')),
                key=lambda r: r['conc_mm'] if np.isfinite(r['conc_mm']) else -1,
                reverse=True)
            for _ri, _fr in enumerate(_sorted_res, 1):
                _fr_grp_str = ''
                _fr_warns   = []
                _fr_cbg     = _fr.get('_concs_by_group', {})
                for _g in _grp_names:
                    _gv  = _fr_cbg.get(_g, [])
                    _gmn = float(np.mean(_gv)) if _gv else float('nan')
                    _gcv = _fr.get('cv_by_group', {}).get(_g, float('nan'))
                    _fr_grp_str += f'  {_gmn:8.3f}(CV{_gcv:.0f}%)'
                # Warnings
                _fr_cvs = [v for g, v in _fr.get('cv_by_group', {}).items()
                           if g != 'BLK' and np.isfinite(v)]
                _wc = max(_fr_cvs) if _fr_cvs else float('nan')
                if np.isfinite(_wc) and _wc > 50: _fr_warns.append(f'CV>{_wc:.0f}%')
                if _fr.get('snr', 0) < 5:         _fr_warns.append('low-SNR')
                # Group ratio
                _fr_gmns = {g: float(np.mean(v)) for g, v in _fr_cbg.items()
                            if g != 'BLK' and v and np.isfinite(np.mean(v))}
                if len(_fr_gmns) >= 2:
                    _fmx = max(_fr_gmns.values()); _fmn = min(_fr_gmns.values())
                    if _fmn > 0 and _fmx / _fmn > 3:
                        _fr_warns.append(f'ratio{_fmx/_fmn:.1f}×')
                _dbg(_dbg_f,
                     f'  {_ri:>3}.  {_fr["metabolite"]:<28s}'
                     f'  {_fr.get("ppm", 0):>6.3f}'
                     f'  {_fr["conc_mm"]:>8.4f}'
                     + _fr_grp_str
                     + f'  {str(_fr.get("_confidence","?")):>4}'
                     + f'  {_wc:>5.1f}%'
                     + f'  {_fr.get("snr", 0):>6.1f}'
                     + (f'  ⚠ {", ".join(_fr_warns)}' if _fr_warns else ''))
            # IS row
            _is_row = next((r for r in self._quant_results if r.get('_is_is')), None)
            if _is_row:
                _dbg(_dbg_f, '  ' + '-' * 120)
                _is_grp_str = ''
                for _g in _grp_names:
                    _gv  = _is_row.get('_concs_by_group', {}).get(_g, [])
                    _gmn = float(np.mean(_gv)) if _gv else float('nan')
                    _gcv = _is_row.get('cv_by_group', {}).get(_g, float('nan'))
                    _is_grp_str += f'  {_gmn:8.3f}(CV{_gcv:.0f}%)'
                _is_rec = (_is_row['conc_mm'] / ref_conc * 100
                           if ref_conc > 0 and np.isfinite(_is_row['conc_mm']) else float('nan'))
                _dbg(_dbg_f,
                     f'  {"IS":>3}.  {_is_row["metabolite"]:<28s}'
                     f'  {_is_row.get("ppm", 0):>6.3f}'
                     f'  {_is_row["conc_mm"]:>8.4f}'
                     + _is_grp_str
                     + f'  {"—":>4}  {"—":>5}  {_is_row.get("snr", 0):>6.1f}'
                     + f'  recovery={_is_rec:.1f}%')
            _dbg(_dbg_f, '')
            _dbg(_dbg_f, f'norm_mode={norm_mode}  integ={_integ}  '
                         f'proton_corr={proton_corr}  blk_sub={use_blk_sub}  '
                         f'j_resolve={use_j_resolve}  spectrometer={_spec_mhz}MHz')
            _dbg(_dbg_f, 'END')
            _dbg_f.close()

        self._quant_results.sort(
            key=lambda r: r['conc_mm'] if np.isfinite(r['conc_mm']) else 0,
            reverse=True)

        # Sync _all_peak_quant best-hit entry with authoritative _quant_results
        # value so Quant Peaks window shows same conc_mm as Peak Info.
        #
        # _all_peak_quant keys: (peak_rank, annotation_hit_ppm) OR peak_rank alone.
        # _quant_results['ppm'] is the DETECTED peak ppm, which can differ from the
        # annotation hit ppm used as key during APQ construction.  Overwriting only
        # (pr, detected_ppm) misses the (pr, annotation_ppm) lookup that
        # _gather_candidates uses first — so we overwrite ALL keys for this pr.
        for _qr in self._quant_results:
            _qr_mid = str(_qr.get('hmdb_id') or _qr.get('bmrb_id') or
                          _qr.get('id', _qr.get('metabolite', '')))
            _qr_pr  = _qr.get('_peak_rank')
            if not _qr_mid or _qr_pr is None:
                continue
            _qr_entry = {
                'conc_mm':        _qr['conc_mm'],
                'concs_by_group': _qr.get('_concs_by_group', {}),
                'ppm':            float(_qr.get('ppm', 0)),
            }
            _mid_d = self._all_peak_quant.setdefault(_qr_mid, {})
            # Overwrite every key whose peak_rank matches (covers all annotation ppms)
            for _k in list(_mid_d.keys()):
                if _k == _qr_pr or (isinstance(_k, tuple) and _k[0] == _qr_pr):
                    _mid_d[_k] = _qr_entry
            # Ensure both key forms exist (sync may have created the dict fresh)
            _mid_d[(_qr_pr, round(float(_qr.get('ppm', 0)), 3))] = _qr_entry
            _mid_d[_qr_pr] = _qr_entry

        # ── All UI updates scheduled on the main thread ──────────────────────
        # Computation ran in a background thread; tkinter widgets must only be
        # touched from the main thread — schedule via after(0, ...).
        _norm_f     = norm_mode
        _use_mcr_f  = use_mcr
        _use_lor_f  = use_lorentz

        def _finish_ui():
            self._prog_apply(93, 'Computing VIP scores…')
            try:
                self._compute_vip_scores()
            except Exception as _vip_err:
                print(f'[Quant] VIP scores skipped: {_vip_err}')
                for _r in self._quant_results:
                    _r.setdefault('vip_score', float('nan'))

            self._prog_apply(99, 'Done.')
            _note     = (f' PQN×{self._pqn_factor:.3f}' if _norm_f == 'PQN' else '')
            _mcr_note = (' +MCR' if _use_mcr_f else ' +Lorentz' if _use_lor_f else '')
            self.run_status.config(
                text=f'{len(self._quant_results)} metabolites '
                     f'[{_norm_f}{_note}{_mcr_note}]',
                foreground='green')
            try:
                self.btn_csv.config(state='normal')
                self.btn_metaboanalyst.config(state='normal')
                self.btn_pdf.config(state='normal')
            except Exception:
                pass
            self._status_ok(f'Quantification: {len(self._quant_results)} metabolites')

            self._update_peak_list_labels()

            if getattr(self, '_warn_no_sdj', False):
                messagebox.showwarning(
                    'J-quality scores unavailable',
                    'Multiplet J-quality scores (s_dj) were not available for this '
                    'run.\n\n'
                    'Quantification could NOT prioritise confirmed multiplets over '
                    'singlets when two analytes competed for the same peak — '
                    'candidates were ranked by ppm proximity only.\n\n'
                    'Open Annotation Results once (so scores are computed) and '
                    're-run Quantification for the full multiplet-aware selection.')

            self._close_progress_dialog()
            self._show_right_panel()
            if self._tree:
                children = self._tree.get_children()
                if children:
                    try:
                        self._tree.selection_set(children[0])
                        self._tree.focus(children[0])
                        self._on_tree_select()
                    except Exception:
                        pass

        self.app.root.after(0, _finish_ui)

    # ── Peak List labels ─────────────────────────────────────────────────────

    def _update_peak_list_labels(self):
        """After quantification, relabel each quantified peak in the Peak List
        with the metabolite that actually won it for integration.

        Annotation Results is intentionally NOT touched — it keeps its own
        score-based ownership (_assignment_sort_key). Only the Peak List follows
        the quantification winner, written as 'assigned_metabolite' (the highest
        priority field in PeakListWindow._populate). The '_quant_assigned' marker
        lets a later run clear only the labels this method set, never an
        annotation- or user-provided assignment.

        Must run on the main thread (called from _finish_ui) because it refreshes
        a tkinter window.
        """
        try:
            peaks = (self.peak_stage.peaks or []) if self.peak_stage else []
            if not peaks:
                return
            pk_by_rank = {p['rank']: p for p in peaks if p.get('rank') is not None}

            # Clear labels written by a previous quantification run only.
            for p in peaks:
                if p.pop('_quant_assigned', False):
                    p.pop('assigned_metabolite', None)

            rank_to_name = {}
            for r in (self._quant_results or []):
                if r.get('_is_is'):
                    continue
                name = r.get('metabolite')
                if not name:
                    continue
                pr = r.get('_peak_rank')
                if pr is not None and pr in pk_by_rank:
                    rank_to_name[pr] = name
                # Multiplet member peaks: every detected peak inside the winning
                # hit's predicted-line window belongs to the same metabolite,
                # mirroring the greedy claim in _run so the Peak List shows the
                # full multiplet under one analyte.
                for h in r.get('_hits', []):
                    _pls = [float(l['ppm']) for l in (h.get('predicted_lines') or [])
                            if 'ppm' in l]
                    if len(_pls) < 2:
                        continue
                    _pk = pk_by_rank.get(pr, {})
                    _w = float(_pk.get('width_ppm', _pk.get('width_at_half', 0.008)) or 0.008)
                    _mar = max(_w * 0.7, 0.010)
                    _lo, _hi = min(_pls) - _mar, max(_pls) + _mar
                    for _orank, _op in pk_by_rank.items():
                        if _lo <= float(_op.get('ppm', -999)) <= _hi:
                            rank_to_name.setdefault(_orank, name)

            for _orank, _nm in rank_to_name.items():
                _p = pk_by_rank.get(_orank)
                if _p is not None:
                    _p['assigned_metabolite'] = _nm
                    _p['_quant_assigned'] = True

            plw = getattr(self.app, '_peak_list_win', None)
            if plw is not None and plw.win.winfo_exists():
                plw.peaks = self.peak_stage.peaks
                plw._populate()
        except Exception as _pll_err:
            print(f'[Quant] peak-list relabel skipped: {_pll_err}')

    # ── VIP scores ───────────────────────────────────────────────────────────

    def _compute_vip_scores(self):
        """Compute PLS-DA VIP scores and store in each result dict as 'vip_score'."""
        if not self._quant_results:
            return

        si      = self.app.samples_info or []
        results = self._quant_results

        n_s = max((len(r.get('_concs_per_sample', [])) for r in results), default=0)
        if n_s < 3:
            for r in results:
                r['vip_score'] = float('nan')
            return

        X = np.array([r.get('_concs_per_sample', [float('nan')] * n_s)
                      for r in results], dtype=float).T

        blk_idx = set(getattr(self, '_last_blk_indices', []))
        groups  = getattr(self, '_last_groups', {})

        non_blk_groups = sorted(g for g in groups if g != 'BLK')
        if len(non_blk_groups) < 2:
            for r in results:
                r['vip_score'] = float('nan')
            return

        g_label = {g: i for i, g in enumerate(non_blk_groups)}
        y_full  = np.full(n_s, float('nan'))
        for g, idxs in groups.items():
            if g == 'BLK':
                continue
            lbl = g_label.get(g, 0)
            for i in idxs:
                if i < n_s:
                    y_full[i] = lbl

        mask = np.isfinite(y_full)
        if mask.sum() < 3:
            for r in results:
                r['vip_score'] = float('nan')
            return

        X_sub = X[mask, :]
        y_sub = y_full[mask]

        n_comp = min(2, len(non_blk_groups) - 1, X_sub.shape[0] - 1, X_sub.shape[1])
        n_comp = max(1, n_comp)

        vip = _pls_vip(X_sub, y_sub, n_components=n_comp)

        for r, v in zip(results, vip):
            r['vip_score'] = float(v) if np.isfinite(v) else float('nan')

    # ── Per-sample area update (from Peak View manual adjustment) ────────────

    def update_sample_area(self, peak_obj, sample_idx, lo_ppm, hi_ppm):
        """Called by PeakSampleView after the user manually adjusts a region."""
        if not self._quant_results:
            return

        r = next((x for x in self._quant_results
                  if x.get('_peak') is peak_obj), None)
        if r is None:
            peak_ppm = peak_obj.get('ppm', None)
            if peak_ppm is not None:
                r = next((x for x in self._quant_results
                          if abs(x.get('ppm', float('nan')) - peak_ppm) < 0.002), None)
        if r is None:
            return

        all_int = self.app.all_intensities
        ppm_arr = self.app.ppm
        n_s     = all_int.shape[1] if all_int is not None else 0

        if '_areas_per_sample' not in r:
            r['_areas_per_sample'] = [r.get('raw_area', 0.0)] * n_s

        integ_mode  = r.get('_integ_mode', 'Area')
        used_mcr    = r.get('_used_mcr', False)
        nh          = r.get('_nh', 1.0)
        blk_mean    = r.get('_blk_mean_area', 0.0)
        proton_corr = r.get('_proton_corr', False)
        use_blk_sub = r.get('_use_blk_sub', False)

        lo_mask = ppm_arr >= min(lo_ppm, hi_ppm)
        hi_mask = ppm_arr <= max(lo_ppm, hi_ppm)
        mask    = lo_mask & hi_mask
        if not mask.any():
            return
        idxs    = np.where(mask)[0]
        lo_i_n  = int(idxs.min())
        hi_i_n  = int(idxs.max())
        col     = all_int[lo_i_n:hi_i_n+1, sample_idx]
        ppm_sub = ppm_arr[lo_i_n:hi_i_n+1]

        if used_mcr:
            mcr_info = r.get('_mcr')
            s_vec = mcr_info.get('S') if mcr_info else None
            orig_lo = mcr_info.get('lo_i', 0) if mcr_info else 0
            orig_hi = mcr_info.get('hi_i', 0) if mcr_info else 0
            if s_vec is not None:
                s_lo = max(0, lo_i_n - orig_lo)
                s_hi = min(len(s_vec) - 1, hi_i_n - orig_lo)
                if s_hi > s_lo:
                    s_sub = s_vec[s_lo:s_hi+1]
                    c_sub = col[:len(s_sub)]
                    denom = float(s_sub @ s_sub)
                    C_s   = max(0.0, float(s_sub @ c_sub) / denom) if denom > 1e-12 else 0.0
                    s_area_scale = abs(float(_trapz(s_vec, ppm_arr[orig_lo:orig_hi+1])))
                    area_raw = C_s * (s_area_scale if s_area_scale > 1e-12 else 1.0)
                else:
                    area_raw = abs(float(_trapz(col, ppm_sub))) if col.size > 1 else 0.0
            else:
                area_raw = abs(float(_trapz(col, ppm_sub))) if col.size > 1 else 0.0

        elif integ_mode == 'Lorentz':
            try:
                _, cen_l, wid_l = _fit_lorentz_mean(ppm_sub, col)
                A_s = _lorentz_amplitude(ppm_sub, col, cen_l, wid_l)
                area_raw = float(A_s) * (np.pi * float(wid_l))
            except Exception:
                area_raw = abs(float(_trapz(col, ppm_sub))) if col.size > 1 else 0.0

        else:
            area_raw = abs(float(_trapz(col, ppm_sub))) if col.size > 1 else 0.0

        new_area = area_raw
        if proton_corr and nh > 0:
            new_area /= nh
        if use_blk_sub:
            new_area -= blk_mean

        r['_areas_per_sample'][sample_idx] = new_area

        s_areas = np.array(r['_areas_per_sample'], dtype=float)

        norm_mode_used = getattr(self, '_last_norm_mode', self.norm_var.get())
        ref_areas_used = getattr(self, '_last_ref_areas', None)
        ref_conc_used  = getattr(self, '_last_ref_conc',  1.0)
        _dil           = getattr(self, '_last_sample_dil', np.ones(n_s))

        if norm_mode_used == 'IS' and ref_areas_used is not None:
            s_concs = np.array([
                (s_areas[s] / ref_areas_used[s] * ref_conc_used * _dil[s]
                 if ref_areas_used[s] > 0 else float('nan'))
                for s in range(n_s)])
        else:
            s_concs = s_areas * _dil

        groups     = getattr(self, '_last_groups', {})
        blk_indices = getattr(self, '_last_blk_indices', [])

        cv_by_group    = {}
        concs_by_group = {}
        ci_by_group    = {}
        for g, idxs in groups.items():
            vals = [float(s_concs[i]) for i in idxs
                    if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
            cv_by_group[g]    = _cv(vals)
            concs_by_group[g] = vals
            _pt, _lo, _hi = _bootstrap_fraction_ci(vals)
            ci_by_group[g] = (_lo, _hi)

        if blk_indices:
            blk_vals = [float(s_concs[i]) for i in blk_indices
                        if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
            if blk_vals:
                concs_by_group['BLK'] = blk_vals
                cv_by_group['BLK']    = _cv(blk_vals)

        all_vals = [v for g, vv in concs_by_group.items()
                    if g != 'BLK' for v in vv if np.isfinite(v) and v > 0]
        mean_conc = float(np.mean(all_vals)) if all_vals else float('nan')

        non_blk_indices = getattr(self, '_last_non_blk_indices', list(range(n_s)))
        p = r.get('_peak')
        _snr_ps, _noise_lvl = _per_sample_snr(p, all_int, n_s)
        _nb_snrs = [_snr_ps[i] for i in non_blk_indices
                    if i < n_s and np.isfinite(_snr_ps[i])]
        _mean_snr = (float(np.mean(_nb_snrs)) if _nb_snrs
                     else (p.get('snr', 0) if p else 0))
        _det_min_snr = self._detection_min_snr()
        # Unknown SNR (NaN) stays "detected" — only suppress with evidence of absence.
        _detected_ps = [bool((not np.isfinite(v)) or v >= _det_min_snr)
                        for v in _snr_ps]
        _apk = (p or {}).get('aligned_per_sample')
        if _apk is not None and len(_apk) == len(_detected_ps):
            _detected_ps = [bool(d and a) for d, a in zip(_detected_ps, _apk)]

        cvs_finite = [v for g, v in cv_by_group.items()
                      if g != 'BLK' and np.isfinite(v)]
        worst_cv   = max(cvs_finite) if cvs_finite else float('nan')
        _mcr_met   = (r.get('_mcr') or {}).get('metrics', {})
        _lof       = _mcr_met.get('lof', float('nan'))
        _r2        = _mcr_met.get('r2',  float('nan'))
        if not np.isfinite(_r2) and p:
            _r2 = float(p.get('gaussian_r2', float('nan')))

        _criteria, _reasons = [], []
        if np.isfinite(worst_cv):
            _s = 0 if worst_cv < 30 else 1 if worst_cv < 50 else 2
            _criteria.append(_s)
            _lbl = 'Good' if _s == 0 else 'Warn' if _s == 1 else 'Poor'
            _reasons.append(f'CV%: {worst_cv:.1f}%  →  {_lbl}')
        if np.isfinite(_lof):
            _s = 0 if _lof < 10 else 1 if _lof < 20 else 2
            _criteria.append(_s)
            _lbl = 'Good' if _s == 0 else 'Warn' if _s == 1 else 'Poor'
            _reasons.append(f'LOF: {_lof:.1f}%  →  {_lbl}')
        if np.isfinite(_r2):
            _s = 0 if _r2 > 0.95 else 1 if _r2 > 0.90 else 2
            _criteria.append(_s)
            _lbl = 'Good' if _s == 0 else 'Warn' if _s == 1 else 'Poor'
            _reasons.append(f'R²: {_r2:.4f}  →  {_lbl}')
        _worst = max(_criteria) if _criteria else 1
        status = 'good' if _worst == 0 else 'warn' if _worst == 1 else 'poor'
        if not _reasons:
            _reasons = ['All parameters within good range']

        r['raw_area']          = float(np.mean(s_areas))
        r['conc_mm']           = mean_conc
        r['cv_by_group']       = cv_by_group
        r['snr']               = _mean_snr
        r['status']            = status
        r['_status_reasons']   = _reasons
        r['_snr_per_sample']      = _snr_ps
        r['_detected_per_sample'] = _detected_ps
        r['_concs_per_sample'] = s_concs.tolist()
        r['_concs_by_group']   = concs_by_group
        r['_conc_ci_by_group'] = ci_by_group

        if self._right_widget:
            self._refresh_tree_row(r)

    # ── Tree row refresh ─────────────────────────────────────────────────────

    def _refresh_tree_row(self, r):
        """Update the single tree row that corresponds to result r."""
        tree = self._tree
        iid = r.get('hmdb_id', r['metabolite'])
        if not tree.exists(iid):
            return
        _nm   = getattr(self, 'norm_var', None)
        _norm = _nm.get() if _nm else 'IS'
        conc  = r.get('conc_mm', float('nan'))
        if not np.isfinite(conc):
            conc_s = 'N/A'
        elif _norm == 'IS':
            conc_s = f'{conc:.4f}'
        else:
            conc_s = f'{conc:.3e}'
        stat = r.get('status', 'good')
        sym  = '✓' if stat == 'good' else ('△' if stat == 'warn' else '✗')
        peak_rank = r.get('_peak', {})
        peak_rank = peak_rank.get('rank', '–') if peak_rank else '–'
        vip = r.get('vip_score', float('nan'))
        vip_s = f'{vip:.2f}' if np.isfinite(vip) else '—'
        tree.item(iid, tags=(stat,), values=(
            peak_rank,
            r['metabolite'],
            f"{r.get('ppm', 0):.4f}",
            conc_s,
            f"{r['raw_area']:.4e}",
            f"{r['snr']:.1f}",
            vip_s,
            f'{sym} {stat.capitalize()}',
        ))

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, mode):
        if not self._quant_results:
            messagebox.showinfo('Export', 'No results to export.')
            return

        si      = self.app.samples_info or []
        n_s     = (self.app.all_intensities.shape[1]
                   if self.app.all_intensities is not None else 0)
        metabolites = [r['metabolite'] for r in self._quant_results]

        if mode == 'csv':
            path = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[('CSV', '*.csv')],
                initialfile='quantification_results.csv')
            if not path:
                return
            try:
                # Collect all group names (excluding BLK), preserving insertion order
                all_groups = []
                for r in self._quant_results:
                    for g in r.get('cv_by_group', {}):
                        if g != 'BLK' and g not in all_groups:
                            all_groups.append(g)

                # Normalisation mode label
                norm_mode_lbl = getattr(self, '_last_norm_mode', '') or ''

                # Sample names and labels
                snames  = [si[i].get('name',  f'S{i+1}') if i < len(si) else f'S{i+1}'
                           for i in range(n_s)]
                slabels = [si[i].get('label', 'Unknown') if i < len(si) else 'Unknown'
                           for i in range(n_s)]

                # Quality columns (fixed), then one column per sample
                cv_hdrs  = [f'CV%_{g}' for g in all_groups]
                # Non-parametric bootstrap CI95 of each group mean (Bootstrap-2)
                ci_hdrs  = [h for g in all_groups
                            for h in (f'CI95lo_{g}', f'CI95hi_{g}')]
                quality_cols = (['PPM', 'Width(ppm)', 'SNR', 'R2', 'Score',
                                 'IntegMode', 'Status', 'Norm'] + cv_hdrs + ci_hdrs)

                def _fmt(v):
                    try:
                        return f'{float(v):.4f}' if np.isfinite(float(v)) else 'NA'
                    except Exception:
                        return 'NA'

                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)

                    # Header row: Metabolite | HMDB_ID | quality cols | sample names
                    w.writerow(['Metabolite', 'HMDB_ID'] + quality_cols + snames)

                    # Group row: blank for quality cols | sample group labels
                    w.writerow(['Group', ''] + [''] * len(quality_cols) + slabels)

                    # One row per analyte (skip IS internal standard row)
                    for r in self._quant_results:
                        if r.get('_is_is'):
                            continue
                        p = r.get('_peak') or {}
                        _im = r.get('_integ_mode', '')

                        ppm_val   = r.get('ppm', p.get('ppm', float('nan')))
                        # Bug 5 fix: for Multiplet mode use actual integration span
                        _mw = r.get('_multiplet_window')
                        if _im == 'Multiplet' and _mw and _mw[0] is not None:
                            width_val = abs(float(_mw[1]) - float(_mw[0]))
                        else:
                            width_val = p.get('width_ppm', float('nan'))
                        snr_val   = r.get('snr', p.get('snr', float('nan')))
                        # Bug 4 fix: use Lorentz R² when available, else gaussian_r2
                        _lz = r.get('_lorentz') or {}
                        if _im == 'Lorentz' and _lz.get('r2') is not None:
                            r2_val = float(_lz['r2'])
                        else:
                            r2_val = p.get('gaussian_r2', float('nan'))
                        score_val = r.get('_ann_score', float('nan'))
                        status    = r.get('status', '')
                        hmdb_id   = r.get('hmdb_id', '')

                        cv_vals = [_fmt(r.get('cv_by_group', {}).get(g, float('nan')))
                                   for g in all_groups]
                        _ci_g   = r.get('_conc_ci_by_group', {})
                        ci_vals = []
                        for g in all_groups:
                            _lo, _hi = _ci_g.get(g, (float('nan'), float('nan')))
                            ci_vals += [_fmt(_lo), _fmt(_hi)]

                        qual_row = [
                            _fmt(ppm_val),
                            _fmt(width_val),
                            _fmt(snr_val),
                            _fmt(r2_val),
                            _fmt(score_val),
                            _im or norm_mode_lbl,
                            status,
                            norm_mode_lbl,
                        ] + cv_vals + ci_vals

                        concs = r.get('_concs_per_sample', [])
                        det   = r.get('_detected_per_sample', [])
                        # Samples below the per-sample detection limit: report
                        # 'n.d.' instead of the noise-derived area — fabricating a
                        # concentration where the analyte is not detected would
                        # misrepresent the data.
                        sample_vals = []
                        for i in range(n_s):
                            if i < len(det) and det[i] is False:
                                sample_vals.append('n.d.')
                            elif i < len(concs):
                                sample_vals.append(_fmt(concs[i]))
                            else:
                                sample_vals.append('NA')

                        w.writerow([r['metabolite'], hmdb_id] + qual_row + sample_vals)

                self._status_ok(f'CSV → {os.path.basename(path)}')
                messagebox.showinfo('Export', f'Saved:\n{path}')
            except Exception as e:
                messagebox.showerror('Export Error', str(e))

        elif mode == 'metaboanalyst':
            path = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[('CSV', '*.csv')],
                initialfile='metaboanalyst_input.csv')
            if not path:
                return
            try:
                # Bug 7 fix: exclude BLK samples — MetaboAnalyst uses only
                # experimental samples. Indices of non-BLK samples only.
                _non_blk_idx = [
                    i for i, s in enumerate(si)
                    if str(s.get('type', s.get('label', ''))).upper() != 'BLK'
                    and i < n_s
                ]
                names  = [si[i].get('name',  f'S{i+1}') for i in _non_blk_idx]
                labels = [si[i].get('label', 'Unknown')  for i in _non_blk_idx]
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    # MetaboAnalyst format: row1=Sample+names, row2=Label+groups
                    w.writerow(['Sample'] + names)
                    w.writerow(['Label']  + labels)
                    for r in self._quant_results:
                        # Bug 6 fix: skip IS internal standard and lipid index rows
                        if r.get('_is_is') or r.get('_is_lipid_index'):
                            continue
                        concs = r.get('_concs_per_sample', [])
                        det   = r.get('_detected_per_sample', [])
                        # Below-detection samples → NA (missing), never a
                        # noise-derived value: MetaboAnalyst must not treat noise
                        # as a measured concentration.
                        vals = []
                        for i in _non_blk_idx:
                            if i < len(det) and det[i] is False:
                                vals.append('NA')
                            elif i < len(concs) and np.isfinite(concs[i]):
                                vals.append(f'{concs[i]:.6e}')
                            else:
                                vals.append('NA')
                        w.writerow([r['metabolite']] + vals)
                self._status_ok(f'MetaboAnalyst → {os.path.basename(path)}')
                messagebox.showinfo('Export', f'Saved:\n{path}')
            except Exception as e:
                messagebox.showerror('Export Error', str(e))

    def _export_pdf(self):
        if not self._quant_results:
            messagebox.showinfo('Export PDF', 'No quantification results to export.')
            return

        path = filedialog.asksaveasfilename(
            defaultextension='.pdf',
            filetypes=[('PDF', '*.pdf')],
            initialfile='nmr_quantification_report.pdf',
            title='Export PDF Report')
        if not path:
            return

        self._status('Generating PDF report…')
        self.frame.update_idletasks()

        try:
            from gui.widgets.pdf_report import generate_pdf
            ann_stage = getattr(self, 'annotation_stage', None)
            generate_pdf(path, self, self.app, annotation_stage=ann_stage)
            self._status_ok(f'PDF saved: {path}')
            messagebox.showinfo('PDF Export', f'Report saved:\n{path}')
        except Exception as exc:
            self._status_err(f'PDF export failed: {exc}')
            messagebox.showerror('PDF Export Error', str(exc))

    # ── Lipid composition indices (Box 1, Li et al. 2017) ─────────────────────

    def _compute_lipid_composition_indices(self, ppm_arr, all_int, n_s, groups):
        """
        Compute 1H NMR lipid composition indices following Li et al. 2017,
        Prog Lipid Res 68:37-56, Box 1 (FA composition from diagnostic peaks).

        Diagnostic windows (CDCl3 chemical shifts):
          a  5.25-5.45  =CH-CH= olefinic          2H per C=C bond
          b  4.05-4.40  TG glycerol CH2 (sn-1,3)  4H per TG molecule
          c  2.72-2.88  bis-allylic =CH-CH2-CH=   2H per group
          d  2.22-2.38  alpha-CH2 adjacent to C=O  2H per FA chain (6H per TG)
          e  1.97-2.07  allylic =CH-CH2            2H per group
          h  0.84-0.92  terminal CH3 (non-n3)      3H per FA chain
          i  0.95-1.01  terminal CH3 n-3 ALA-type  3H per chain (linolenic-specific;
                        EPA/DHA terminal CH3 overlaps with h at 0.88 ppm)

        U = (b/4 + d/6 + (h+i)/9) / 3  — intensity per FA chain (3-estimator avg)

        Results are dimensionless ratios (%) or double bonds per chain.
        Not subject to IS/PQN normalisation.
        """
        _WINDOWS = {
            'a': (5.25, 5.45),
            'b': (4.05, 4.40),
            'c': (2.72, 2.88),
            'd': (2.22, 2.38),
            'e': (1.97, 2.07),
            'h': (0.84, 0.92),
            'i': (0.95, 1.01),
        }

        raw = {}
        for key, (lo, hi) in _WINDOWS.items():
            mask = (ppm_arr >= lo) & (ppm_arr <= hi)
            if mask.any():
                raw[key] = np.array([
                    abs(float(_trapz(all_int[mask, s], ppm_arr[mask])))
                    for s in range(n_s)], dtype=float)
            else:
                raw[key] = np.zeros(n_s, dtype=float)

        a  = raw['a']
        b  = raw['b']
        c  = raw['c']
        d  = raw['d']
        e  = raw['e']
        h  = raw['h']
        i_ = raw['i']

        # Box 1, Eq (1) — intensity per FA chain (proton-normalised average)
        U = (b / 4.0 + d / 6.0 + (h + i_) / 9.0) / 3.0
        eps = np.finfo(float).eps

        with np.errstate(divide='ignore', invalid='ignore'):
            # n-3 PUFA %: linolenic-type ALA (18:3n-3) only.
            # i/9 = ALA chain contribution per TG; divide by U = per-chain baseline.
            n3_pct = np.where(U > eps, np.clip((i_ / 9.0) / U * 100.0, 0.0, 100.0), 0.0)

            # n-6 PUFA %: from bis-allylic signal.
            # Linolenic contributes 2 bis-allylic CH2 groups per chain;
            # linoleic contributes 1. Subtract ALA contribution: c/2 - 2*(i/9).
            n6_bisallylic = np.maximum(0.0, c / 2.0 - 2.0 * (i_ / 9.0))
            n6_pct = np.where(U > eps, np.clip(n6_bisallylic / U * 100.0, 0.0, 100.0), 0.0)

            # MUFA %: from allylic signal after subtracting PUFA contributions.
            # Each MUFA chain contributes 4H to e (2 allylic CH2 groups);
            # n-3 and n-6 chains also contribute to e, accounted for here.
            mufa_allylic = np.maximum(0.0, e / 4.0 - 2.0 * (i_ / 9.0) - n6_bisallylic)
            mufa_pct = np.where(U > eps, np.clip(mufa_allylic / U * 100.0, 0.0, 100.0), 0.0)

            # Saturated FA % — remainder; 0 when U≈0 (no signal → no composition reported)
            sat_pct = np.where(U > eps,
                               np.clip(100.0 - n3_pct - n6_pct - mufa_pct, 0.0, 100.0),
                               0.0)

            # Degree of Unsaturation: average C=C double bonds per FA chain.
            # a/2 = moles of C=C (each has 2 olefinic H); U*3 = moles of FA chains.
            du = np.where(U > eps, (a / 2.0) / (U * 3.0), 0.0)

            # Unsaturation Index: weighted average (MUFA=1, n-6=2, n-3=3 double bonds)
            ui = np.clip((mufa_pct + 2.0 * n6_pct + 3.0 * n3_pct) / 100.0, 0.0, None)

        def _make_index_entry(name, values, unit='%'):
            vals = np.asarray(values, dtype=float)
            mean_val = float(np.nanmean(vals))
            cv_by_group = {}
            concs_by_group = {}
            for g, idxs in groups.items():
                gv = [float(vals[idx]) for idx in idxs
                      if idx < n_s and np.isfinite(vals[idx])]
                from ._quant_helpers import _cv as _cv_fn
                cv_by_group[g] = _cv_fn(gv)
                concs_by_group[g] = gv
            return {
                'metabolite':          name,
                'hmdb_id':             '__LIPID_INDEX__',
                'ppm':                 0.0,
                'n_peaks':             0,
                'raw_area':            mean_val,
                'corr_area':           mean_val,
                'conc_mm':             mean_val,
                'cv_by_group':         cv_by_group,
                'snr':                 0.0,
                'status':              '—',
                '_confidence':         0,
                '_status_reasons':     [f'Derived index ({unit}) — Li et al. 2017 Box 1'],
                '_peak':               None,
                '_areas_per_sample':   vals.tolist(),
                '_concs_per_sample':   vals.tolist(),
                '_concs_by_group':     concs_by_group,
                '_used_mcr':           False,
                '_lorentz':            None,
                '_mcr':                None,
                '_lipid_regions':      None,
                '_lipid_window':       None,
                '_integ_mode':         'LipidIndex',
                '_ann_score':          1.0,
                '_nh':                 1.0,
                '_blk_mean_area':      0.0,
                '_proton_corr':        False,
                '_use_blk_sub':        False,
                '_has_anchor':         False,
                '_peak_rank':          None,
                '_used_j_resolve':     False,
                '_n_eligible':         0,
                '_peak_align_shifts':  None,
                '_is_lipid_index':     True,
            }

        return [
            _make_index_entry('n-3 PUFA — ALA index (%)',         n3_pct,  '%'),
            _make_index_entry('n-6 PUFA — bis-allylic index (%)', n6_pct,  '%'),
            _make_index_entry('MUFA — allylic index (%)',          mufa_pct, '%'),
            _make_index_entry('Saturated FA (%)',                  sat_pct, '%'),
            _make_index_entry('Degree of Unsaturation (C=C/chain)', du,    'ratio'),
            _make_index_entry('Unsaturation Index (UI)',           ui,      'ratio'),
        ]
