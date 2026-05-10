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
_QUANT_DEBUG = False
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

from ._quant_helpers import (
    _trapz, _pls_vip, _cv, _pqn, _effective_nh,
    _mcr_als, _fit_lorentz_mean, _fit_lorentz_mean_with_bl,
    _fit_lorentz_sample, _fit_multi_lorentz,
    _align_sample_center, _lorentz_amplitude, _lorentz_shape,
)
from processing.multiplet_line_selector import select_clean_lines


class _QuantCompMixin:
    """Mixin — mixed into QuantificationStage.  Uses self.* freely."""

    # ── Core quantification ───────────────────────────────────────────────────

    def _run(self):
        # ── Full reset before every run ───────────────────────────────────────
        self._quant_results          = []
        self._all_peak_quant         = {}
        self._quant_peak_overrides   = {}
        self._peak_competitor_count  = {}
        self._pqn_factor             = 1.0
        self._last_groups            = {}
        self._last_blk_indices       = []
        # Clear tree UI immediately so stale data is never visible during run
        try:
            self.run_status.config(text='Running…', foreground='blue')
            self._show_placeholder()
        except Exception:
            pass

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
        use_area      = (_integ == 'Area')
        use_lorentz   = (_integ == 'Lorentz')
        use_multiplet = (_integ == 'Multiplet')
        proton_corr   = self.proton_corr_var.get()
        use_mcr       = self.mcr_var.get()
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
        if use_j_resolve:
            _ann_st = getattr(self, 'annotation_stage', None)
            _j_resolve_contribs = getattr(_ann_st, '_j_resolve_contributions', {}) or {}
        norm_mode   = self.norm_var.get()

        if is_lipidomics_mode:
            # Lipidomics path is intentionally simple: integrate full ppm regions only.
            use_area = True
            use_lorentz = False
            use_mcr = False

        if self.ext_is_var.get() and self._ext_is_indices:
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
        _sample_dil = np.ones(n_s)
        for i, s in enumerate(si):
            if i < n_s:
                lbl = s.get('label', 'Unknown')
                _sample_dil[i] = _app_dil.get(lbl, 1.0)

        use_ext_is  = self.ext_is_var.get() and bool(self._ext_is_indices)
        ref_areas   = None
        if ref_peak is not None:
            lo_r = min(ref_peak.get('start_idx', 0), ref_peak.get('end_idx', 0))
            hi_r = max(ref_peak.get('start_idx', 0), ref_peak.get('end_idx', 0))
            if use_ext_is:
                ext_areas = []
                for s in self._ext_is_indices:
                    if s < n_s:
                        a = abs(float(_trapz(all_int[lo_r:hi_r+1, s],
                                             ppm_arr[lo_r:hi_r+1])))
                        ext_areas.append(a / ref_nh)
                if ext_areas:
                    _fixed_ref = float(np.mean(ext_areas))
                    ref_areas = [_fixed_ref] * n_s
                else:
                    use_ext_is = False
            if not use_ext_is:
                ref_areas = []
                for s in range(n_s):
                    a = abs(float(_trapz(all_int[lo_r:hi_r+1, s],
                                         ppm_arr[lo_r:hi_r+1])))
                    ref_areas.append(a / ref_nh)

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

        quality_marks = getattr(self.annotation_stage, '_quality_marks', {}) or {}

        auto_quality_marks = {}
        try:
            mets = list(getattr(self.annotation_stage, '_metabolites', []) or [])
            for met in mets:
                mid = str(met.get('id', ''))
                for mp in met.get('matched_peaks', []):
                    pr = mp.get('peak_rank')
                    qv = mp.get('annotation_quality')
                    if pr is None or qv is None:
                        continue
                    auto_quality_marks[(mid, pr)] = qv
        except Exception:
            auto_quality_marks = {}

        by_peak = {}
        for r in results:
            mid = r.get('hmdb_id') or r.get('bmrb_id') or r.get('id', r.get('name', 'unknown'))
            r['_mid'] = mid
            pr = r.get('peak_rank')
            by_peak.setdefault(pr, []).append(r)

        def _effective_quality(h):
            """Return the quality for a result hit.
            Manual mark (quality_marks) takes priority over auto mark
            (annotation_quality field set by _auto_mark_good)."""
            manual = quality_marks.get((h['_mid'], h.get('peak_rank')))
            if manual is not None:
                return manual
            auto_q = auto_quality_marks.get((str(h['_mid']), h.get('peak_rank')))
            if auto_q is not None:
                return auto_q
            return h.get('annotation_quality')   # 'good' / 'bad' / None

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
                    if _eff_q == 'good':
                        _peak_assignments[_pr] = (_met, _mp)

        winning_hits = []
        if is_lipidomics_mode:
            # For lipidomics, keep all region definitions and quantify by ppm region.
            winning_hits = list(results)
        else:
            for pr, candidates in by_peak.items():
                # Always exclude anything explicitly or automatically marked bad
                good_hits = [h for h in candidates if _effective_quality(h) == 'good']
                if good_hits:
                    best = None
                    # Prefer the metabolite explicitly assigned in _peak_assignments
                    # (respects deltaJ ranking and user window adjustments).
                    if pr in _peak_assignments:
                        assigned_met, _ = _peak_assignments[pr]
                        assigned_mid = str(assigned_met.get('id', ''))
                        best = next(
                            (h for h in good_hits if str(h['_mid']) == assigned_mid),
                            None)
                    if best is None:
                        best = max(good_hits, key=_quant_sort_key)
                    winning_hits.append(best)
                # If no good hits exist for this peak rank, skip it entirely

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

        tmp_map = {}
        for h in winning_hits:
            mid = h['_mid']
            tmp_map.setdefault(mid, {'name': h.get('name', mid), 'hits': []})
            tmp_map[mid]['hits'].append(h)

        # Build rank → detected peak for anchor and NNLS lookups below.
        _pk_by_rank = {p['rank']: p
                       for p in (self.peak_stage.peaks or [])
                       if p.get('rank') is not None}
        _ann_stage_q = getattr(self, 'annotation_stage', None)

        # ── Populate _line_selection on every hit before sorting ─────────────
        # Always recompute — never cache across _run() calls, since peak data,
        # NNLS contributions, or use_j_resolve may have changed since last run.
        _ppm_min = float(self.app.ppm.min()) if self.app.ppm is not None else 0.0
        _ppm_max = float(self.app.ppm.max()) if self.app.ppm is not None else 10.0
        for _sh in results:
            _sh['_line_selection'] = select_clean_lines(
                hit=_sh,
                pk_by_rank=_pk_by_rank,
                j_resolve_contribs=_j_resolve_contribs if use_j_resolve else {},
                mid_str=str(_sh.get('_mid', '')),
                use_nnls=use_j_resolve,
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
        for mid, info in tmp_map.items():
            if is_lipidomics_mode or len(info['hits']) == 1:
                met_map[mid] = {'name': info['name'], 'hits': info['hits']}
            else:
                mid_str = str(mid)
                best = min(info['hits'],
                           key=lambda h, _m=mid_str: _hit_sort_key(h, _m))
                met_map[mid] = {'name': info['name'], 'hits': [best]}

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
                    _nh_k = max(1, _effective_nh(h0, det_ppm, _spec_mhz_l))
                if proton_corr and _nh_k > 0:
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
                    '_ann_score':    cv.get('corr', 0.0),
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

        for mid, info in met_map.items():
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
                    '_ann_score': 1.0,
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
            for hit in info['hits']:
                # Prefer rank-based lookup: annotation peak_rank maps directly to
                # the detected peak used during annotation, bypassing ppm proximity
                # search which can land on the wrong peak when lines are close.
                p = _pk_by_rank.get(hit.get('peak_rank')) or \
                    self._find_peak(hit.get('peak_ppm', hit.get('ppm', 0)))
                if p is None:
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
                # For sub-peaks (a single line of a multiplet detected separately),
                # use the fractional nH for that specific line.  For full-peak
                # integration (the whole multiplet detected as one peak) the
                # anchor_line_idx from _anchor_lines is metadata only — the integral
                # already covers all protons, so use the full proton count.
                if use_multiplet:
                    # Multiplet whole: always total nH regardless of sub-peak status
                    nh = max(1.0, float(hit.get('protons', 1) or 1))
                elif p.get('is_sub_peak', False):
                    nh = _effective_nh(hit, detected_ppm, _spec_mhz)
                else:
                    nh = max(1.0, float(hit.get('protons', 1) or 1))
                nh = max(0.01, nh)

                if use_multiplet:
                    _m_start = hit.get('ppm_start')
                    _m_end   = hit.get('ppm_end')
                    if _m_start is not None and _m_end is not None:
                        _m_lo = min(float(_m_start), float(_m_end))
                        _m_hi = max(float(_m_start), float(_m_end))
                    else:
                        _p_lo = float(ppm_arr[lo_i])
                        _p_hi = float(ppm_arr[hi_i])
                        _m_lo, _m_hi = min(_p_lo, _p_hi), max(_p_lo, _p_hi)
                    _sm_mult = (ppm_arr >= _m_lo) & (ppm_arr <= _m_hi)
                    if not _sm_mult.any():
                        _sm_mult = np.zeros(len(ppm_arr), dtype=bool)
                        _sm_mult[lo_i:hi_i + 1] = True
                    _x_mult = ppm_arr[_sm_mult]
                    _a_list_m = []
                    for s in range(n_s):
                        seg = all_int[_sm_mult, s].copy()
                        if local_baseline and len(seg) >= 2:
                            bl = np.linspace(max(float(seg[0]), 0.0),
                                             max(float(seg[-1]), 0.0), len(seg))
                            seg = seg - bl
                        _a_list_m.append(abs(float(_trapz(seg, _x_mult))))
                    a_vec = np.array(_a_list_m)
                elif use_area and 'area_per_sample' in p:
                    a_vec = np.array(p['area_per_sample'])
                    if len(a_vec) < n_s:
                        a_vec = np.pad(a_vec, (0, n_s - len(a_vec)))
                    elif len(a_vec) > n_s:
                        a_vec = a_vec[:n_s]
                elif use_lorentz and (hi_i - lo_i) >= 1:
                    x_r      = ppm_arr[lo_i:hi_i + 1]
                    y_mean_r = np.mean(all_int[lo_i:hi_i+1, :], axis=1)
                    A_l, cen_l, wid_l, _bl_y0, _bl_slope = \
                        _fit_lorentz_mean_with_bl(x_r, y_mean_r)
                    max_shift = max(abs(float(x_r[-1] - x_r[0])) * 0.15, 0.005)
                    shifts = []
                    a_list = []
                    for s in range(n_s):
                        y_s  = all_int[lo_i:hi_i+1, s]
                        res  = _fit_lorentz_sample(x_r, y_s, A_l, cen_l, wid_l,
                                                   max_shift_ppm=max_shift)
                        if res is not None:
                            area_s, _, cen_s, _ = res
                            shifts.append(cen_s - cen_l)
                        else:
                            area_s = abs(float(
                                _trapz(np.clip(y_s, 0, None), x_r)))
                            shifts.append(0.0)
                        a_list.append(area_s)
                    a_vec = np.array(a_list)
                    _lorentz_params[mid] = {
                        'cen':      cen_l,   'wid':      wid_l,
                        'shifts':   shifts,
                        'bl_y0':    _bl_y0,  'bl_slope': _bl_slope,
                        'bl_x0':    float(x_r[0]),
                    }
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
                    _pa_shifts = []
                    a_list = []
                    for s in range(n_s):
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
                else:
                    _a_list_tb = []
                    for s in range(n_s):
                        seg = all_int[lo_i:hi_i + 1, s].copy()
                        if local_baseline and len(seg) >= 2:
                            bl = np.linspace(max(float(seg[0]), 0.0),
                                             max(float(seg[-1]), 0.0), len(seg))
                            seg = seg - bl
                        _a_list_tb.append(abs(float(_trapz(seg, ppm_arr[lo_i:hi_i + 1]))))
                    a_vec = np.array(_a_list_tb)

                # ── Auto multi-Lorentz when gaussian_r2 is low ────────────────
                # If the detected peak has poor shape quality (overlap / shoulder),
                # try deconvolving the full cluster with a sum-of-N-Lorentzians.
                # Only replace a_vec if the multi-Lorentz fit improves R² enough.
                _gr2  = float(p.get('gaussian_r2', 1.0) or 1.0)
                _n_cl = int(p.get('n_cluster_peaks', 1) or 1)

                if _gr2 < _R2_MULTI_TRIGGER and _n_cl >= 2:
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
                _ls = hit.get('_line_selection', {})
                _ls_mode = _ls.get('mode')
                _ls_sf   = float(_ls.get('scale_factor', 1.0))
                if _ls_mode == 'skip':
                    a_vec = np.zeros(n_s)
                    hit['_line_sel_skip'] = True
                elif _ls_mode in ('nnls', 'ratio') and _ls_sf > 1.0 and _ls.get('clean_ranks'):
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
                        _dbg(_dbg_f,
                             f'    line_sel   : mode={_ls_mode}'
                             f' clean={_ls["clean_ranks"]}'
                             f' scale={_ls_sf:.3f}'
                             f' cif={_ls["clean_intensity_fraction"]:.3f}')
                # ─────────────────────────────────────────────────────────────

                # For IS mode: always divide by nH so the IS ratio is proton-corrected.
                # For non-IS modes: only divide if user enabled proton_corr.
                if norm_mode in ('IS', 'IS+PQN') or proton_corr:
                    a_vec = a_vec / nh
                # J-Resolve fractional area correction
                if use_j_resolve and p is not None:
                    _peak_rank = p.get('rank')
                    _jrc = _j_resolve_contribs.get(_peak_rank, {})
                    _frac = _jrc.get(mid)
                    if _frac is not None and 0.0 < _frac <= 1.0:
                        a_vec = a_vec * _frac
                        hit['_j_resolve_frac'] = _frac

                _blk_mean_area = 0.0
                if use_blk_sub and blk_indices:
                    _blk_mean_area = float(np.mean([a_vec[i] for i in blk_indices
                                                    if i < len(a_vec)]))
                    a_vec = a_vec - _blk_mean_area
                _dbg(_dbg_f, '')
                _dbg(_dbg_f, f'  [{info["name"]}]  mid={mid}')
                _dbg(_dbg_f, f'    peak       : rank={p.get("rank")} ppm={p.get("ppm"):.4f} '
                             f'snr={p.get("snr", 0):.1f} width={p.get("width_ppm", p.get("width_at_half", 0)):.4f}')
                _dbg(_dbg_f, f'    is_sub_peak: {p.get("is_sub_peak", False)}')
                _dbg(_dbg_f, f'    nH used    : {nh:.4f}  (raw protons={hit.get("protons")} pattern={hit.get("pattern")})')
                _dbg(_dbg_f, f'    integ mode : {"area_per_sample" if use_area and "area_per_sample" in p else "lorentz" if use_lorentz else "mcr" if use_mcr else "trapz"}')
                _dbg(_dbg_f, f'    area/nH    : {[f"{v:.4e}" for v in a_vec.tolist()]}')
                if use_blk_sub and blk_indices:
                    _dbg(_dbg_f, f'    blk_mean   : {_blk_mean_area:.4e}')
                if use_j_resolve and hit.get('_j_resolve_frac') is not None:
                    _dbg(_dbg_f, f'    j_frac     : {hit["_j_resolve_frac"]:.4f}')
                s_areas += a_vec
                _ann_score = max(
                    (float(h.get('composite_score', h.get('multiplet_score', 0)) or 0)
                     for h in info['hits']), default=0.0)
                _used_jr = use_j_resolve and any(
                    h.get('_j_resolve_frac') is not None for h in info['hits'])
                meta_info[mid] = {'name': info['name'], 'hits': info['hits'],
                                  '_peak': p, '_used_mcr': use_mcr and found,
                                  '_lorentz': _lorentz_params.get(mid),
                                  '_mcr': _mcr_info.get(mid),
                                  '_ann_score': _ann_score,
                                  '_nh': nh,
                                  '_blk_mean_area': _blk_mean_area,
                                  '_has_anchor': _hit_has_anchor,
                                  '_used_j_resolve': _used_jr,
                                  '_align_shifts': _align_shifts.get(mid)}

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
            mean_areas = {mid: float(np.mean(v)) for mid, v in raw_areas.items()}
            ref_for_pqn = dict(mean_areas)
            normed, self._pqn_factor = _pqn(mean_areas, ref_for_pqn)
            for mid in raw_areas:
                if mid in normed:
                    scale = normed[mid] / (mean_areas[mid] + 1e-12)
                    raw_areas[mid] = raw_areas[mid] * scale
        elif norm_mode == 'Total':
            totals = np.sum(np.stack(list(raw_areas.values())), axis=0)
            totals[totals == 0] = 1
            for mid in raw_areas:
                raw_areas[mid] = raw_areas[mid] / totals * 100

        self._last_norm_mode      = norm_mode
        self._last_ref_areas      = list(ref_areas) if ref_areas is not None else None
        self._last_ref_conc       = ref_conc
        self._last_sample_dil     = np.array(_sample_dil, dtype=float)
        self._last_groups         = dict(groups)
        self._last_blk_indices    = list(blk_indices) if blk_indices else []
        self._last_non_blk_indices = list(non_blk_indices) if non_blk_indices else []

        self._quant_results = []
        for mid, s_areas in raw_areas.items():
            m    = meta_info.get(mid, {})
            p    = m.get('_peak')
            lw   = m.get('_lipid_window')
            if p is not None:
                ppm_val = float(p.get('ppm', 0.0))
            elif lw and lw[0] is not None and lw[1] is not None:
                ppm_val = (float(lw[0]) + float(lw[1])) / 2.0
            else:
                ppm_val = 0.0

            if norm_mode in ('IS', 'IS+PQN') and ref_areas is not None:
                # IS quantification:
                #   conc_analyte = (area_A / nH_A) / (area_IS / nH_IS) × conc_IS
                # Both areas are already divided by their respective nH above.
                # For internal IS (same tube), dilution cancels in the ratio — do NOT
                # apply _sample_dil.  For external IS, ref_areas is a fixed mean so
                # dilution does NOT cancel and must be applied.
                _apply_dil = use_ext_is   # only external IS needs dilution correction
                s_concs = np.array([
                    (s_areas[s] / ref_areas[s] * ref_conc
                     * (_sample_dil[s] if _apply_dil else 1.0)
                     if ref_areas[s] > 0 else float('nan'))
                    for s in range(n_s)])
            else:
                s_concs = np.array(s_areas, dtype=float) * _sample_dil

            cv_by_group    = {}
            concs_by_group = {}
            for g, idxs in groups.items():
                vals = [float(s_concs[i]) for i in idxs
                        if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
                cv_by_group[g]    = _cv(vals)
                concs_by_group[g] = vals

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

            _dbg(_dbg_f, f'    s_areas    : {[f"{v:.4e}" for v in s_areas.tolist()]}')
            _dbg(_dbg_f, f'    s_concs    : {[f"{v:.4f}" for v in s_concs.tolist()]}')
            _dbg(_dbg_f, f'    concs/group: { {g: [f"{v:.4f}" for v in vv] for g, vv in concs_by_group.items()} }')
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
            _mean_snr = p.get('snr', 0) if p else 0
            if p and ppm_arr is not None and all_int is not None:
                _lo_i = min(p.get('start_idx', 0), p.get('end_idx', 0))
                _hi_i = max(p.get('start_idx', 0), p.get('end_idx', 0))
                _noise_lvl = p.get('noise', None)
                if _noise_lvl is None and p.get('snr', 0) > 0:
                    _peak_max_all = float(np.max(np.mean(all_int[_lo_i:_hi_i+1, :], axis=1)))
                    _noise_lvl = _peak_max_all / p['snr']
                if _noise_lvl and _noise_lvl > 0 and non_blk_indices:
                    _snrs = []
                    for _si in non_blk_indices:
                        if _si < n_s:
                            _pm = float(np.max(all_int[_lo_i:_hi_i+1, _si]))
                            _snrs.append(_pm / _noise_lvl)
                    if _snrs:
                        _mean_snr = float(np.mean(_snrs))

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

            # Count eligible candidates sharing this peak rank
            _elig_overrides = getattr(getattr(self, 'app', None),
                                      'eligible_overrides', {})
            _jr_stage   = getattr(getattr(self, 'app', None),
                                  'stages', {}).get('j_resolve')
            _peak_overlap = getattr(_jr_stage, '_peak_overlap', {})
            _nnls_rejected = getattr(_jr_stage, '_nnls_rejected', set())

            _all_mets = getattr(_jr_stage, '_metabolites', None) or []
            if _peak_rank_val is not None:
                _n_elig = sum(
                    1 for _cm in _all_mets
                    for _cmp in _cm.get('matched_peaks', [])
                    if _cmp.get('peak_rank') == _peak_rank_val
                    and _cm.get('id') != mid
                    and _elig_overrides.get(
                        f'{_cm.get("id")}__{_peak_rank_val}', True)
                ) + 1
            else:
                _n_elig = 1

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

            # Warn when selected hit has contaminated/skipped line selection
            _sel_hit = (m.get('hits') or [{}])[0]
            _sel_ls  = _sel_hit.get('_line_selection', {})
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
                '_concs_by_group':   concs_by_group,
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
                '_has_anchor':       _has_anchor,
                '_peak_rank':        _peak_rank_val,
                '_used_j_resolve':   _used_jr,
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

                _apq_p = self._find_peak(_r.get('peak_ppm', _r.get('ppm', 0)))
                if _apq_p is None:
                    continue

                _apq_lo = min(_apq_p.get('start_idx', 0), _apq_p.get('end_idx', 0))
                _apq_hi = max(_apq_p.get('start_idx', 0), _apq_p.get('end_idx', 0))
                if _apq_hi <= _apq_lo:
                    _apq_hi = _apq_lo + 1
                _apq_hi = min(_apq_hi, all_int.shape[0] - 1)

                _apq_det_ppm = float(_r.get('peak_ppm', _r.get('ppm', _apq_p.get('ppm', 0))))
                # Mirror main pipeline nH logic exactly:
                # sub-peaks use fractional nH; full peaks use DB proton count.
                if _apq_p.get('is_sub_peak', False):
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
                    'conc_mm':        _apq_mean,
                    'concs_by_group': _apq_cbg,
                    'ppm':            float(_apq_p.get('ppm', 0)),
                    '_lorentz':       _apq_lorentz if use_lorentz else None,
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

        _dbg(_dbg_f, '')
        _dbg(_dbg_f, '── Final results ───────────────────────────────────────')
        for _fr in sorted(self._quant_results, key=lambda r: r['metabolite']):
            _dbg(_dbg_f, f'  {_fr["metabolite"]:<30s} peak={_fr.get("_peak_rank")}  '
                         f'ppm={_fr.get("ppm", 0):.4f}  mM={_fr["conc_mm"]:.4f}  '
                         f'conf={_fr.get("_confidence")}  status={_fr.get("status")}')
        _dbg(_dbg_f, '')
        _dbg(_dbg_f, 'END')
        if _dbg_f is not None:
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

        note = (f' PQN×{self._pqn_factor:.3f}' if norm_mode == 'PQN' else '')
        mcr_note = (' +MCR' if use_mcr else ' +Lorentz' if use_lorentz else '')
        self.run_status.config(
            text=f'{len(self._quant_results)} metabolites '
                 f'[{norm_mode}{note}{mcr_note}]',
            foreground='green')
        self.btn_csv.config(state='normal')
        self.btn_metaboanalyst.config(state='normal')
        self._status_ok(
            f'Quantification: {len(self._quant_results)} metabolites')

        try:
            self._compute_vip_scores()
        except Exception as _vip_err:
            print(f'[Quant] VIP scores skipped: {_vip_err}')
            for _r in self._quant_results:
                _r.setdefault('vip_score', float('nan'))

        # Always go through _show_right_panel so the widget is guaranteed
        # to be both visible and repopulated, regardless of prior pack state.
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
        for g, idxs in groups.items():
            vals = [float(s_concs[i]) for i in idxs
                    if i < n_s and np.isfinite(s_concs[i]) and s_concs[i] > 0]
            cv_by_group[g]    = _cv(vals)
            concs_by_group[g] = vals

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
        _mean_snr = p.get('snr', 0) if p else 0
        if p and ppm_arr is not None and all_int is not None:
            _lo_i = min(p.get('start_idx', 0), p.get('end_idx', 0))
            _hi_i = max(p.get('start_idx', 0), p.get('end_idx', 0))
            _noise_lvl = p.get('noise', None)
            if _noise_lvl is None and p.get('snr', 0) > 0:
                _peak_max_all = float(np.max(np.mean(all_int[_lo_i:_hi_i+1, :], axis=1)))
                _noise_lvl = _peak_max_all / p['snr']
            if _noise_lvl and _noise_lvl > 0 and non_blk_indices:
                _snrs = [float(np.max(all_int[_lo_i:_hi_i+1, _si])) / _noise_lvl
                         for _si in non_blk_indices if _si < n_s]
                if _snrs:
                    _mean_snr = float(np.mean(_snrs))

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
        r['_concs_per_sample'] = s_concs.tolist()
        r['_concs_by_group']   = concs_by_group

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
                cv_hdrs = [f'CV%_{g}' for g in all_groups]
                quality_cols = ['PPM', 'Width(ppm)', 'SNR', 'R2', 'Score', 'Status', 'Norm'] + cv_hdrs

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

                    # One row per analyte
                    for r in self._quant_results:
                        p = r.get('_peak') or {}

                        ppm_val   = r.get('ppm', p.get('ppm', float('nan')))
                        width_val = p.get('width_ppm', float('nan'))
                        snr_val   = r.get('snr', p.get('snr', float('nan')))
                        r2_val    = p.get('gaussian_r2', float('nan'))
                        score_val = r.get('_ann_score', float('nan'))
                        status    = r.get('status', '')
                        hmdb_id   = r.get('hmdb_id', '')

                        cv_vals = [_fmt(r.get('cv_by_group', {}).get(g, float('nan')))
                                   for g in all_groups]

                        qual_row = [
                            _fmt(ppm_val),
                            _fmt(width_val),
                            _fmt(snr_val),
                            _fmt(r2_val),
                            _fmt(score_val),
                            status,
                            norm_mode_lbl,
                        ] + cv_vals

                        concs = r.get('_concs_per_sample', [])
                        sample_vals = [
                            _fmt(concs[i]) if i < len(concs) else 'NA'
                            for i in range(n_s)
                        ]

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
                names  = [s.get('name',  f'S{i+1}') for i, s in enumerate(si)]
                labels = [s.get('label', 'Unknown') for s in si]
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['Sample', 'HMDB_ID'] + names)
                    w.writerow(['Label', ''] + labels)
                    for r in self._quant_results:
                        concs = r.get('_concs_per_sample', [])
                        vals  = [f'{c:.6e}' if np.isfinite(c) else 'NA'
                                 for c in concs]
                        w.writerow([r['metabolite'], r.get('hmdb_id', '')] + vals)
                self._status_ok(f'MetaboAnalyst → {os.path.basename(path)}')
                messagebox.showinfo('Export', f'Saved:\n{path}')
            except Exception as e:
                messagebox.showerror('Export Error', str(e))

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
