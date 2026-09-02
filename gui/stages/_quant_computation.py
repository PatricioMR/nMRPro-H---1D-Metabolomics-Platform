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

from processing import lipidomics

# ── Debug logging ─────────────────────────────────────────────────────────────
# Set to True to write a detailed quant_debug.log in the project root.
_QUANT_DEBUG = True
_QUANT_DEBUG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'quant_debug.log')

def _dbg_open():
    """Open the debug log without allowing logging to break quantification."""
    if not _QUANT_DEBUG:
        return None

    # The frozen application may live under Program Files, which is normally
    # read-only. Try the project/bundle path first for development, then the
    # per-user NMRPro-H data directory used by installed builds.
    candidates = [_QUANT_DEBUG_PATH]
    try:
        from app_paths import user_data_dir
        candidates.insert(
            0, os.path.join(user_data_dir(), 'logs', 'quant_debug.log'))
    except Exception:
        pass

    for path in candidates:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return open(path, 'w', encoding='utf-8')
        except (OSError, PermissionError):
            continue

    # Debug logging is optional; a read-only installation must still quantify.
    return None

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
    multiplet_nh,
    _mcr_als, _fit_multi_lorentz, _align_sample_center,
    bootstrap_fraction_ci as _bootstrap_fraction_ci,
    jr_pointwise_multiplet_area, jr_analyte_fraction, _anchor_lines_complete,
    is_unquantified_conc, multiplet_tier_penalty, multiplet_sdj_missing,
    NQ_REASON_LOST_COMPETITION, NQ_REASON_NO_SDJ,
    NQ_REASON_UNVERIFIED_MULTIPLET, NQ_REASON_ZERO_NAN,
)
from processing.multiplet_line_selector import (
    select_clean_lines, matched_line_ranks, line_match_tolerance,
    multiplet_line_peaks as _multiplet_line_peaks)
from processing.annotation_scoring import (
    candidate_key as _ann_candidate_key,
    is_competitor as _ann_is_competitor,
    DJ_GOLD_HZ_DEFAULT as _DJ_GOLD_DEFAULT)


def _multiplet_window_from_peaks(hit, pk_by_rank):
    """Integration window = union of the multiplet line peaks' Peak-List
    boundaries (left foot of the leftmost line … right foot of the rightmost),
    i.e. exactly the peaks shown in Annotation Results. Returns [lo, hi] ppm or
    None when no line peak can be matched."""
    matched = _multiplet_line_peaks(hit, pk_by_rank)
    if not matched:
        return None
    lo = min(min(float(pk.get('start_ppm', pk.get('ppm', 0))),
                 float(pk.get('end_ppm',   pk.get('ppm', 0)))) for pk in matched)
    hi = max(max(float(pk.get('start_ppm', pk.get('ppm', 0))),
                 float(pk.get('end_ppm',   pk.get('ppm', 0)))) for pk in matched)
    return [float(lo), float(hi)]


DB_WINDOW_MIN_MARGIN = 0.010   # ppm — same floor as the _margin rule below
DB_WINDOW_WIDTH_FRAC = 0.7     # ppm — same fraction of the peak width


def db_gapfill_window(line_ppms, width_ppm=0.008):
    """Integration window for a DB-predicted multiplet with no detected peak.

    ``[min(line), max(line)]`` runs from the CENTRE of the outermost left line
    to the CENTRE of the outermost right line, so half of each outer line's
    area falls outside — a doublet integrates ~47% of its area while nH stays
    the full multiplet count, halving the reported concentration. A singlet
    degenerates further: min == max gives a zero-width window and an area of
    exactly 0.

    Both are fixed by padding with the same margin the detected-peak branch
    uses (``max(width_ppm * 0.7, 0.010)`` ppm), so the outer lines' tails are
    inside the window. Returns ``(lo_ppm, hi_ppm)``; ``None`` when there are no
    lines to build a window from.
    """
    ppms = [float(v) for v in (line_ppms or []) if v is not None]
    if not ppms:
        return None
    margin = max(float(width_ppm or 0.0) * DB_WINDOW_WIDTH_FRAC,
                 DB_WINDOW_MIN_MARGIN)
    return (min(ppms) - margin, max(ppms) + margin)


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
        # NOTE: _quant_peak_overrides is intentionally NOT reset here — user
        # multiplet choices (Quant Peaks window) must survive a re-quantification.
        # They are cleared only when annotation is re-run (_annotation_search).
        if not hasattr(self, '_quant_peak_overrides'):
            self._quant_peak_overrides = {}
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
        # Withdrawn before the competition, not beaten in it. Declared here so
        # the not-quantified pass can tell the two apart whichever branch below
        # ran — an empty set simply means nothing was withdrawn.
        _blocked_no_sdj: set = set()
        _blocked_unverified: set = set()
        # Who actually entered the competition. Empty when the branch that runs
        # one does not — an analyte cannot then be told it was outscored.
        _competed_mids: set = set()
        _blocked_ambiguous: set = set()

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
            lipid_solvent = args.get('lipid_solvent') or None
            if not results:
                messagebox.showinfo('Info', 'Annotation results are empty.')
                return
        else:
            results = []
            is_lipidomics_mode = False
            lipid_solvent = None

        _integ        = self.integ_var.get()
        # 'Area' (single-line trapz) is temporarily disabled (under development).
        # The UI radiobutton is greyed out, but a loaded project could still carry
        # 'Area' — block it here and fall back to Multiplet so no run uses it.
        if _integ == 'Lorentz':
            # 'Lorentz Fit' integration mode was removed — migrate old projects
            # to Multiplicity so a loaded 'Lorentz' selection still runs.
            self.integ_var.set('Multiplet')
            _integ = 'Multiplet'
        if _integ == 'Area':
            messagebox.showinfo(
                'Single-Line Area em desenvolvimento',
                'O método "Single-Line Area" está em desenvolvimento e foi '
                'temporariamente desativado.\n\n'
                'A quantificação vai usar "Multiplicity".')
            self.integ_var.set('Multiplet')
            _integ = 'Multiplet'
        use_area      = (_integ == 'Area')
        use_multiplet = (_integ == 'Multiplet')
        proton_corr   = self.proton_corr_var.get()
        use_mcr       = self.mcr_var.get()
        if use_mcr and use_multiplet:
            messagebox.showwarning(
                'MCR + Multiplicity incompatíveis',
                'MCR Deconvolution está activado mas o modo de integração é Multiplicity.\n'
                'MCR nunca será executado — Multiplicity tem prioridade.\n\n'
                'Desactive MCR ou mude o modo para Single-Line Area.')
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

        # ── Global baseline (faithful qNMR integration) ───────────────────────
        # qNMR integrates the signal above a flat baseline. Integrating raw data
        # from 0 biases regions sitting on a sloping/elevated baseline (valleys,
        # macromolecule humps, residual-water wings). The global per-sample
        # estimate is the honest one; the local-linear per-window option is an
        # inferior fallback that can subtract real signal when a window edge
        # lands on a neighbour's skirt.
        #
        # The method and λ are INHERITED from Signal Detection. Hard-coding
        # arPLS here meant the areas in the Signal List and the areas being
        # quantified could come from two different baseline models for the same
        # signal, with nothing saying so.
        #
        # Applied to a WORKING COPY only (app state untouched, reversible).
        # Skipped when the spectrum was already globally baseline-corrected
        # upstream (Visualization step) to avoid double correction.
        _global_baseline_applied = False
        _bl_model = getattr(self.app, 'baseline_model', None)
        if _bl_model is not None and _bl_model.enabled:
            try:
                _corr = self.app.corrected_intensities()
                if _corr is not None and _corr is not all_int:
                    all_int = np.asarray(_corr, dtype=float)
                    _global_baseline_applied = True
                    print(f'[Quant] baseline: {_bl_model.describe()} '
                          f'— project model')
            except Exception as _bl_e:
                print(f'[Quant] global baseline skipped: {_bl_e}')

        # Global arPLS already flattened the baseline → the per-window local
        # linear subtraction would double-correct (and remove real signal at
        # edges). Disable it whenever a global baseline is in effect.
        if _global_baseline_applied or getattr(self.app, 'baseline_corrected', False):
            local_baseline = False

        # Expose the exact matrix the integration uses so the quant visualizer
        # draws/shades the SAME baseline-corrected spectrum (otherwise the fill
        # would show a raw pedestal that no longer matches the integrated value).
        self._quant_int = all_int

        # ── Water-suppression exclusion ───────────────────────────────────────
        # Mirror the annotation-stage 'Exclude water region' option in
        # quantification: when active, anything whose ppm falls inside the
        # residual-water zone is ignored, so solvent-suppression artefacts never
        # enter a quantified area (CLAUDE.md: never fabricate data).
        _ann_for_water = getattr(self, 'annotation_stage', None)

        def _in_water_zone(ppm_val):
            try:
                if not bool(_ann_for_water.exclude_water_var.get()):
                    return False
                _wlo = float(_ann_for_water.water_lo_var.get())
                _whi = float(_ann_for_water.water_hi_var.get())
            except Exception:
                return False
            if _wlo > _whi:
                _wlo, _whi = _whi, _wlo
            try:
                return _wlo <= float(ppm_val) <= _whi
            except (TypeError, ValueError):
                return False
        self._in_water_zone = _in_water_zone

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
                        # float, not int: BMRB stores fractional per-signal nH
                        # (0.5, 0.75, 1.55…). int() truncated 0.75 → 0, and the
                        # falsy-0 guard below then dropped the fallback entirely.
                        _db_protons[(str(_row[0]), round(float(_row[1]), 3))] = float(_row[2])
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
        _dbg(_dbg_f, f'global baseline: applied={_global_baseline_applied}  '
                     f'already_corrected={getattr(self.app, "baseline_corrected", False)}')
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
            self._status('Pre-align: per-signal window alignment active', 'blue')

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

            def _is_area(s):
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
                    # A peak with no verdict is no longer promoted here. It used
                    # to pass on the strength of a companion multiplet (RANSY
                    # area ratio + predicted position), which proved too variable
                    # to carry a quantification decision; such peaks now need an
                    # explicit Good, manual or automatic.
        except Exception as _aqm_exc:
            import traceback
            print(f'[Quant] auto_quality_marks build failed: {_aqm_exc}')
            traceback.print_exc()
            auto_quality_marks = {}

        # Which metabolites are eligible to be quantified at all.
        # NOT the same question as the annotation tree's row colour
        # (processing.annotation_quality.metabolite_tier), which aggregates
        # every signal to say what the metabolite looks like overall. Here a
        # single usable signal is enough: a metabolite contested on one peak
        # can still quantify on another. Who owns a contested peak is settled
        # in the competition below, not here.
        # Ensures only peaks belonging to Good metabolites are quantified.
        good_met_ids: set = set()
        try:
            _all_mets_gmi = list(getattr(self.annotation_stage, '_metabolites', []) or [])
            # _anchor_lines keys are (met_id, peak_rank).
            _anchor_lines = getattr(self.annotation_stage, '_anchor_lines', {}) or {}
            _anchor_met_ids = {str(k[0]) for k in _anchor_lines}
            for _met_gmi in _all_mets_gmi:
                _mid_gmi = str(_met_gmi.get('id', ''))
                _mps_gmi = _met_gmi.get('matched_peaks', [])

                # An anchor only makes the metabolite Good when the anchoring
                # peak is itself quality-'good'. This
                # blocks an incomplete multiplet (J unconfirmed, not all lines
                # found → marked 'bad' upstream) from being quantified via a
                # stale auto_anchor flag.
                _has_anchor_gmi = (
                    _mid_gmi in _anchor_met_ids
                    or any(mp.get('annotation_quality') == 'good'
                           and mp.get('auto_anchor')
                           for mp in _mps_gmi)
                )
                if _has_anchor_gmi:
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

        # Drop any annotation hit whose peak sits in the water-exclusion zone.
        if results:
            _n_before_w = len(results)
            results = [r for r in results
                       if not _in_water_zone(
                           r.get('peak_ppm', r.get('ppm', r.get('db_ppm'))))]
            _n_water = _n_before_w - len(results)
            if _n_water:
                print(f'[Quant] water exclusion: dropped {_n_water} hit(s) inside '
                      f'water zone')
                _dbg(_dbg_f, f'water exclusion: dropped {_n_water} hit(s)')

        by_peak = {}
        for r in results:
            mid = r.get('hmdb_id') or r.get('bmrb_id') or r.get('id', r.get('name', 'unknown'))
            r['_mid'] = mid
            pr = r.get('peak_rank')
            by_peak.setdefault(pr, []).append(r)

        try:
            from gui.stages._annotation_helpers import ambiguous_peak_ranks
            _ann_for_amb = getattr(self, 'annotation_stage', None)
            _mets_for_amb = getattr(_ann_for_amb, '_metabolites', []) or []
            _manual_for_amb = {
                (str(k[0]), k[1]): v for k, v in (quality_marks or {}).items()
            }
            _ambiguous_ranks = ambiguous_peak_ranks(
                _mets_for_amb, manual_quality=_manual_for_amb,
                dj_gold_hz=float(
                    (getattr(self.annotation_stage, '_annotation_params', None)
                     or {}).get('dj_gold_hz', _DJ_GOLD_DEFAULT)
                    or _DJ_GOLD_DEFAULT))
        except Exception as _amb_exc:
            print(f'[Quant] ambiguity detection unavailable: {_amb_exc}')
            _ambiguous_ranks = set()

        def _effective_quality(h):
            """Return the quality for a result hit.
            Manual mark (quality_marks) takes priority over auto_quality_marks
            built from post-processed _metabolites. Raw h['annotation_quality']
            is NOT used — it is stale (set before post-processing overwrites the
            matched_peak dicts via _group_by_metabolite)."""
            manual = quality_marks.get((h['_mid'], h.get('peak_rank')))
            if manual is None:
                manual = quality_marks.get((str(h['_mid']), h.get('peak_rank')))
            if manual is not None:
                return manual
            if h.get('peak_rank') in _ambiguous_ranks:
                return 'ambiguous'
            return auto_quality_marks.get((str(h['_mid']), h.get('peak_rank')))

        _SINGLET_MULTS = {'s', 'br s', 'brs', 'bs', ''}

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
                    if (_eff_q == 'good'
                            and _pr not in _ambiguous_ranks):
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
                'No signals are marked as Good in Annotation Results.\n\n'
                'Open Annotation Results, select the correct metabolite '
                'for each signal and press Ctrl+1 (Good) before quantifying.')
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

            # Anchor priority is intentionally DISABLED: anchors were forcing the
            # quantified peak to a single J-confirmed leg that did not match the
            # signal shown in Annotation Results / Quant Peaks. Selection now goes
            # purely by purity / contamination / competitors / width / score.

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

            # 6. SNR tie-break (qNMR): among otherwise-equivalent candidates prefer
            #    the stronger signal — a higher-SNR line yields lower quantification
            #    uncertainty.  Used only as a tie-break; never vetoes a weak line.
            _snr = float(pk.get('snr', 0) or 0)

            # Sort ascending: lower tuple = better hit
            return (
                _ls.get('mode') == 'skip',  # skip-flagged hits last
                -own_frac,                  # higher NNLS purity first
                _contamination_penalty,     # lower intra-multiplet contamination first
                n_jr_cands,                 # fewer NNLS competitors first
                n_cands,                    # fewer annotation competitors first
                _pk_width,                  # narrower peak (less overlap) first
                -_snr,                      # higher SNR first (qNMR tie-break)
                -score,                     # higher composite score first
            )

        met_map = {}
        _assigned_peaks = set()   # shared by greedy + rescue
        # Regions already being integrated, in ppm. A gap-fill candidate has no
        # detected peak and therefore no rank to claim, so _assigned_peaks says
        # nothing about it: six of them were integrating the histidine region
        # at [7.0876, 7.1075] that L-histidine had won on peak 304, and three
        # more shared [8.1091, 8.1290]. Reserving in ppm is the only space a
        # candidate without a peak can be reserved in.
        _claimed_ppm: list = []
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
            # ΔJ band width: the user's Annotation Parameters value when set,
            # otherwise the scoring core's default. Never a literal here — the
            # threshold must have one owner.
            _DJ_GOLD_HZ = float(
                (getattr(self.annotation_stage, '_annotation_params', None) or {})
                .get('dj_gold_hz', _DJ_GOLD_DEFAULT) or _DJ_GOLD_DEFAULT)

            def _ann_candidate(h, mid_str):
                """Merge a search hit with the Annotation Results verdict.

                The raw hit never carried the cosine fingerprint similarity or
                the confirmed line count — those are computed by the Annotation
                Results window. Reading them here is what makes quantification
                elect the same winner the user is looking at, manual marks
                included.
                """
                _mp = _ann_mp.get((mid_str, h.get('peak_rank'))) or {}
                _manual = quality_marks.get((h.get('_mid'), h.get('peak_rank')))
                if _manual is None:
                    _manual = quality_marks.get((mid_str, h.get('peak_rank')))
                _dj = h.get('delta_j_hz')
                if _dj is None:
                    _dj = _mp.get('delta_j_hz')
                return {
                    'multiplicity': h.get('multiplicity') or _mp.get('type'),
                    'delta_j_hz':   _dj,
                    'ppm_diff':     h.get('ppm_diff', h.get('ppm_error')),
                    'n_found':      _mp.get('_n_found_lines'),
                    'n_expected':   _mp.get('_n_exp_lines'),
                    'intensity_agreement': _mp.get('intensity_agreement'),
                    'cosine':       _mp.get('_s_cosine'),
                    'manual_quality': _manual,
                    'quality':      _effective_quality(h),
                    'score':        _mp.get('composite_score',
                                            h.get('composite_score', 0.0)),
                }

            def _compete_key(h, mid_str):
                # ── Which signal quantifies this metabolite ────────────────────
                # Delegates to the single ranking rule shared with annotation
                # (processing.annotation_scoring.candidate_key): manual mark →
                # evidence tier → ΔJ band → Δppm band → cosine. Banding is what
                # makes a technical tie a tie instead of a coin toss decided by
                # noise. _hit_sort_key (NNLS purity, contamination, competitor
                # count, width, SNR, score) is appended only to keep the order
                # deterministic once every physical criterion has tied.
                _key = _ann_candidate_key(
                    _ann_candidate(h, mid_str),
                    dj_gold_hz=_DJ_GOLD_HZ)
                return _key + (_hit_sort_key(h, mid_str),)

            # s_dj drives the multiplet/singlet tier in _compete_key. When it is
            # missing (project loaded before scores were computed, or the
            # Annotation Results window never post-processed _metabolites) a
            # multiplet's J cannot be confirmed, so it must NOT be quantified —
            # quantifying it would fabricate a tier from unknown J (CLAUDE.md:
            # never silence data warnings). Such multiplet hits are blocked below
            # (singlets need no J and are unaffected); the blocked set is surfaced.
            # Annotation-assigned (metabolite, peak_rank) pairs — exactly the
            # peaks the Annotation Results window shows for each metabolite.
            # Quantification must compete ONLY on these; stray candidate hits the
            # DB search produced at other peaks (never assigned to the analyte)
            # must not enter — e.g. a spurious peak winning over Alanine's
            # annotated doublet (peaks 49/51).
            _assigned_pairs = set()
            _mids_with_assign = set()
            # Evidence the Annotation Results window computed but the raw search
            # hits never carried — cosine above all, plus the line-count that
            # decides whether a multiplet is confirmed. Quantification must read
            # the annotation's verdict rather than re-deriving its own, so that
            # what the user sees and what gets quantified are the same decision.
            _ann_mp = {}
            for _met_a in (getattr(self.annotation_stage, '_metabolites', []) or []):
                _mid_a = str(_met_a.get('id', ''))
                for _mp_a in _met_a.get('matched_peaks', []):
                    _r_a = _mp_a.get('peak_rank')
                    if _r_a is not None:
                        _assigned_pairs.add((_mid_a, _r_a))
                        _mids_with_assign.add(_mid_a)
                        _ann_mp[(_mid_a, _r_a)] = _mp_a

            _all_non_bad = []
            for _h in results:
                _h_mid = str(_h.get('_mid', ''))
                if _h_mid not in good_met_ids:
                    continue
                _h_q = _effective_quality(_h)
                if _h_q == 'bad':
                    continue
                # An ambiguous rank has no winner. rank_candidates says so and
                # the annotation window shows it, but quantification filtered
                # only 'bad', so the final tie-break elected one anyway — a
                # number where the evidence says we cannot tell which analyte
                # this is. It competes again the moment the analyst marks one
                # Good, which is what resolves the rank.
                if _h_q == 'ambiguous':
                    _blocked_ambiguous.add(_h_mid)
                    continue
                # Only the peaks annotation assigned to this metabolite may
                # compete (peak_rank None = DB gap-fill, allowed). When the
                # metabolite has no assigned peaks at all, don't filter.
                _pr_h = _h.get('peak_rank')
                if (_pr_h is not None and _h_mid in _mids_with_assign
                        and (_h_mid, _pr_h) not in _assigned_pairs):
                    continue
                # Block multiplet hits whose s_dj (J quality) is unknown — a
                # multiplet with unconfirmed J cannot anchor quantification. Only
                # gate hits anchored on a detected peak (peak_rank not None);
                # gap-fill candidates are handled by their own path.
                if _pr_h is not None and multiplet_sdj_missing(
                        _h.get('multiplicity'),
                        (_h_mid, _pr_h) in _s_dj_map,
                        _SINGLET_MULTS):
                    _blocked_no_sdj.add(_h_mid)
                    continue
                # An unverified multiplet (expected lines not all found, so J
                # was never measured) did not identify a signal and must not
                # compete for the peak — see annotation_scoring.is_competitor.
                _cand_h = _ann_candidate(_h, _h_mid)
                if not _ann_is_competitor(_cand_h):
                    _blocked_unverified.add(_h_mid)
                    continue
                _all_non_bad.append((_compete_key(_h, _h_mid), _h_mid, _h))
            _all_non_bad.sort(key=lambda x: x[0])
            _competed_mids = {str(_m) for _, _m, _ in _all_non_bad}

            if _blocked_unverified:
                print(f'[Quant] {len(_blocked_unverified)} multiplet(s) withdrawn '
                      f'— expected lines not all found, J unverified: '
                      f'{sorted(_blocked_unverified)}')
                _dbg(_dbg_f, f'blocked (unverified multiplet): '
                             f'{sorted(_blocked_unverified)}')

            if _blocked_no_sdj:
                self._warn_no_sdj = True
                print(f'[Quant] WARNING: {len(_blocked_no_sdj)} multiplet(s) blocked '
                      f'— s_dj (J-quality) unavailable, J unconfirmed: '
                      f'{sorted(_blocked_no_sdj)}. Reopen Annotation Results to '
                      f'recompute scores before quantifying these analytes.')
                _dbg(_dbg_f, f'blocked (no s_dj): {sorted(_blocked_no_sdj)}')

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
                # The anchor was checked; the LINES were not. A multiplet claims
                # its line peaks below, but the next candidate was only ever
                # tested on its anchor — so a different anchor over the same
                # lines walked straight in. On the reference study L-valine,
                # L-isoleucine and 2-hydroxy-3-methylbutyric acid were all
                # integrating the peaks at 0.9912 and 1.0028, from anchors 90,
                # 58 and 83; butanol and N-acetyl-L-alanine shared 1.3238 and
                # 1.3354. Whatever the anchor, the same physical lines cannot be
                # quantified twice.
                _my_lines = ([l for l in (_h.get('predicted_lines') or [])
                              if 'ppm' in l])
                # Claimed with the SAME matcher that builds the integration
                # window. matched_line_ranks takes its candidates from
                # hit['matched_peaks'], a key these result rows do not carry,
                # so it fell back to the anchor alone: 196 of 206 multiplets
                # claimed less than they integrated — one claimed nothing and
                # integrated two peaks. Reserving in one space and integrating
                # in another is why 31 of 86 quantified analytes shared an
                # integration window while no anchor ever collided.
                _my_ranks = ({p.get('rank') for p in
                              _multiplet_line_peaks(_h, _pk_by_rank)
                              if p.get('rank') is not None}
                             if len(_my_lines) >= 2 else set())
                if _my_ranks & _assigned_peaks:
                    continue
                _assigned_mets.add(_h_mid)
                if _pr is not None:
                    _assigned_peaks.add(_pr)
                met_map[_h_mid] = {'name': _h.get('name', _h_mid), 'hits': [_h]}

                # Non-singlet: claim only THIS metabolite's own multiplet legs —
                # the detected peaks matched to its predicted lines — so the other
                # legs of its multiplet cannot be re-used by a competitor. A peak
                # that merely falls inside the ppm window but belongs to a
                # NEIGHBOURING resonance is NOT claimed (the old window scan stole
                # such peaks from their rightful metabolite).
                _pl_gr = _h.get('predicted_lines') or []
                if len([l for l in _pl_gr if 'ppm' in l]) >= 2:
                    for _pk_gr in _multiplet_line_peaks(_h, _pk_by_rank):
                        if _pk_gr.get('rank') is not None:
                            _assigned_peaks.add(_pk_gr['rank'])

            print(f'[Quant] greedy: {len(met_map)}/{len(good_met_ids)} mets placed '
                  f'using {len(_assigned_peaks)} claimed peak slots')

            # Diagnostic: assigned (annotation) vs chosen peak per metabolite.
            if _dbg_f is not None:
                _dbg(_dbg_f, '')
                _dbg(_dbg_f, '── PEAK SELECTION (assigned vs chosen) ───────────────')
                for _mid_d, _info_d in met_map.items():
                    _nm_d = _info_d.get('name', _mid_d)
                    _chosen = (_info_d.get('hits') or [{}])[0].get('peak_rank')
                    _assigned_d = sorted(
                        r for (m, r) in _assigned_pairs if m == str(_mid_d))
                    _dbg(_dbg_f, f'  {_nm_d} ({_mid_d}): chosen_peak={_chosen}  '
                                 f'annotation_assigned={_assigned_d}')

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
                def _free_of_claims(_r):
                    """Nothing this hit would integrate is already someone's.

                    The rescue tested the anchor alone, like the greedy pass
                    once did, so a rescued analyte could take the very lines a
                    winner was quantified on — different anchor, same signal.
                    """
                    if _r.get('peak_rank') in _assigned_peaks:
                        return False
                    _rl = [l for l in (_r.get('predicted_lines') or [])
                           if 'ppm' in l]
                    if len(_rl) < 2:
                        return True
                    return not ({p.get('rank') for p in
                                 _multiplet_line_peaks(_r, _pk_by_rank)}
                                & _assigned_peaks)

                for _rmid in _rescued_mids:
                    _rhits = [_r for _r in results
                              if str(_r.get('_mid', '')) == _rmid
                              and _effective_quality(_r) == 'good'
                              and _free_of_claims(_r)]
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
                    # Claim what it took, or the next rescue sees the peak as
                    # free and two rescued analytes share one signal.
                    if _rpr is not None:
                        _assigned_peaks.add(_rpr)
                    if len([l for l in (_best_r.get('predicted_lines') or [])
                            if 'ppm' in l]) >= 2:
                        for _pk_r in _multiplet_line_peaks(_best_r, _pk_by_rank):
                            if _pk_r.get('rank') is not None:
                                _assigned_peaks.add(_pk_r['rank'])
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
            if _in_water_zone(pk.get('ppm')):
                continue
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

        # Analytes with a detected peak are integrated first, so their windows
        # are on record before any gap-fill candidate is offered a region. A
        # gap-fill has no apex to justify its claim; whoever has one gets the
        # region, and processing order should not decide that.
        met_map = dict(sorted(
            met_map.items(),
            key=lambda kv: (kv[1]['hits'][0].get('peak_rank') is None,
                            str(kv[0]))))

        _mcr_info        = {}
        _align_shifts    = {}   # mid → list[float] per-sample ppm shift

        # ── Multiplet integration window ──────────────────────────────────
        # The window is the UNION of the Peak-List boundaries of the peaks
        # composing the multiplet (left foot of the leftmost line … right foot
        # of the rightmost) — see _multiplet_window_from_peaks. It is NOT
        # clipped against neighbouring analytes.
        #
        # A clipping pre-pass used to cut the window of whichever metabolite had
        # the lower composite_score. It was wrong three ways: it re-ran the
        # candidate competition with a criterion (composite_score, 45% Δppm)
        # that contradicts the ΔJ-first rule used to elect the winner; it cut
        # the geometry while leaving nH untouched, so a partial area was divided
        # by the full proton count; and when the loser's window sat inside the
        # winner's it collapsed to zero width and fell back to the anchor peak —
        # which for a multiplet is documented as unreliable, and in practice was
        # often not one of the multiplet's own lines at all.
        #
        # There is nothing left to clip: only one metabolite wins a contested
        # peak (the ranking is shared with annotation), and an ambiguous contest
        # elects nobody, so the shared region is never double-counted.

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
            _area_ok = False   # a hit produced a real integrated area
            _hit_has_anchor = False
            _met_synthetic_db = False   # quantified from DB window (no detected peak)
            _j_frac_this_met = None
            _jr_multiplet_applied = False   # per-line JG frac already folded into a_vec
            _jr_line_fracs_store  = None    # [(line_ppm, frac)] for PDF per-line paint
            for hit in info['hits']:
                # Prefer rank-based lookup: annotation peak_rank maps directly to
                # the detected peak used during annotation, bypassing ppm proximity
                # search which can land on the wrong peak when lines are close.
                # A hit with no peak_rank competed as a DB gap-fill: it claimed
                # no peak slot, precisely because it has no detected apex. The
                # ppm-proximity fallback would then hand it whichever peak sits
                # nearest — including one another metabolite already won, which
                # is how o-cresol came to be quantified on 3-(2-hydroxyphenyl)-
                # propionic acid's peak 214. Such a hit is integrated over its
                # own predicted window below, or not at all.
                _rank_h = hit.get('peak_rank')
                p = (_pk_by_rank.get(_rank_h) if _rank_h is not None else None)
                if p is None and _rank_h is not None:
                    # A rank the peak list no longer holds — detection re-ran
                    # since annotation. Proximity is the only way back.
                    p = self._find_peak(hit.get('peak_ppm', hit.get('ppm', 0)))
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
                        # Pad past the outer line centres, otherwise half of each
                        # outer line sits outside the window (and a singlet gets
                        # a zero-width window integrating to exactly 0).
                        _gf_lo, _gf_hi = db_gapfill_window(_pl0_ppms, 0.008)
                        _si = int(np.argmin(np.abs(ppm_arr - _gf_hi)))
                        _ei = int(np.argmin(np.abs(ppm_arr - _gf_lo)))
                        p = {'ppm': _cen,
                             'start_ppm': float(_gf_lo),
                             'end_ppm':   float(_gf_hi),
                             'start_idx': min(_si, _ei),
                             'end_idx':   max(_si, _ei),
                             'width_ppm': 0.008,
                             'rank':      hit.get('peak_rank')}
                        # A window with no detected apex must not sit on a
                        # region another analyte is already integrating. It has
                        # no peak of its own to justify the claim, so the one
                        # that does wins.
                        if any(_gf_lo < _c_hi and _c_lo < _gf_hi
                               for _c_lo, _c_hi in _claimed_ppm):
                            _dbg(_dbg_f, '    SKIP: gap-fill window overlaps a '
                                         'region already being integrated')
                            continue
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
                    if _p_fb is not None and _p_fb > 0:
                        hit['protons'] = _p_fb
                # use_multiplet integrates the whole multiplet window → the nH
                # of that DB signal, which may be FRACTIONAL: BMRB splits a
                # group's protons across the signals it is observed in (e.g.
                # D-Fructose H13-H16 = 4 H over 5 signals → 0.8 each; they sum
                # back to total_protons). Flooring at 1.0 reports D-Glucose at
                # 0.47× its true concentration — the window's area really does
                # correspond to 0.5 H, not 1.
                # Otherwise, use fractional nH whenever predicted_lines > 1:
                # each annotation hit corresponds to one specific DB line, and
                # _effective_nh matches detected ppm to the closest predicted line.
                if use_multiplet:
                    nh = multiplet_nh(hit)
                elif len(hit.get('predicted_lines', [])) > 1:
                    nh = _effective_nh(hit, detected_ppm, _spec_mhz)
                else:
                    nh = max(1.0, float(hit.get('protons', 1) or 1))
                nh = max(0.01, nh)

                _mcr_aps = p.get('area_per_sample')
                if p.get('mcr_component_idx') is not None and _mcr_aps is not None \
                        and len(_mcr_aps) == n_s:
                    # This peak is an MCR Deconvolution sub-peak the user
                    # Accepted for an overlapping region (K>=2). area_per_sample
                    # was integrated from the MCR-isolated component (S[:,k]*C
                    # per sample) — a raw start_idx/end_idx reintegration of the
                    # shared window would mix this analyte back with whatever
                    # neighbour caused the overlap, exactly what MCR was run to
                    # separate. Use it as-is instead of any of the branches below.
                    a_vec = np.asarray(_mcr_aps, dtype=float)
                    _dbg(_dbg_f, f'    integ(mcr sub-peak): component #{p.get("mcr_component_idx")}'
                                 f'  areas={[f"{v:.3e}" for v in a_vec.tolist()]}')
                elif use_multiplet:
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

                    # Integration window = union of the Peak-List boundaries of
                    # the multiplet's peaks (left foot of the leftmost line …
                    # right foot of the rightmost). Never clipped: only one
                    # metabolite wins a contested peak, so there is no shared
                    # region left to double-count.
                    _full = _multiplet_window_from_peaks(hit, _pk_by_rank)
                    if _full is not None:
                        _m_lo, _m_hi = _full
                    else:
                        _m_lo = float(min(ppm_arr[lo_i], ppm_arr[hi_i]))
                        _m_hi = float(max(ppm_arr[lo_i], ppm_arr[hi_i]))
                    _dbg(_dbg_f, f'    integ window  : [{_m_lo:.4f}, {_m_hi:.4f}] ppm  (margin={_margin:.4f})')
                    # Every region actually integrated is recorded, so a later
                    # candidate without a peak of its own cannot be handed it.
                    _claimed_ppm.append((float(min(_m_lo, _m_hi)),
                                         float(max(_m_lo, _m_hi))))
                    # Store multiplet window for visualization/export
                    hit['_multiplet_window'] = (_m_lo, _m_hi)
                    _sm_mult = (ppm_arr >= _m_lo) & (ppm_arr <= _m_hi)
                    # A window too narrow to integrate, or one holding none of
                    # the multiplet's own predicted lines, cannot yield this
                    # analyte's area. The old code fell back to the anchor peak
                    # here — but for a multiplet the anchor is the CENTRE and
                    # frequently a stray peak, so that fallback reported a
                    # concentration measured on somebody else's signal. Report
                    # nothing instead (CLAUDE.md: never fabricate data).
                    _lines_in_win = [_lp for _lp in _pl_ppms
                                     if _m_lo <= _lp <= _m_hi]
                    if _sm_mult.sum() < 2 or (_pl_ppms and not _lines_in_win):
                        _dbg(_dbg_f, '    SKIP: window holds no predicted line '
                                     f'({len(_lines_in_win)}/{len(_pl_ppms)})')
                        continue
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
                            # Per-line fraction for THIS analyte (1.0 if JG never
                            # deconvolved the line's peak, the analyte's share if
                            # it has one, 0.0 if JG resolved the peak without it —
                            # see jr_analyte_fraction for the double-count rule).
                            _line_fr.append(jr_analyte_fraction(
                                _lr_q, mid, _j_resolve_contribs,
                                _j_resolve_contribs_ps, n_s))
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

                # Skip in Multiplet mode: there we integrate the WHOLE multiplet
                # window (trapz) by design, so per-cluster multi-Lorentz curve_fit
                # would both contradict that and dominate runtime.
                if ((not use_multiplet)
                        and (_gr2 < _R2_MULTI_TRIGGER)
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
                # Skip-zeroing only applies to single-line integration (Area/
                # Lorentz), where a fully contaminated multiplet has no clean line
                # to integrate. In Multiplet mode the whole window is integrated
                # with per-line J-Resolve fractions already folded into a_vec
                # (jr_pointwise), so hard-zeroing here would discard a valid area —
                # the contamination is still surfaced as a warning downstream.
                if use_j_resolve and _ls_mode == 'skip' and not use_multiplet:
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
                    # Apply the JG share only when JG actually deconvolved this
                    # peak; otherwise leave the area untouched and report no
                    # J-Resolve fraction. jr_analyte_fraction returns 0.0 when JG
                    # resolved the peak without this analyte, so a non-contributor
                    # no longer double-counts the same ppm at 100%.
                    _resolved = (_peak_rank in _j_resolve_contribs_ps
                                 or _peak_rank in _j_resolve_contribs)
                    if _resolved:
                        _fr = jr_analyte_fraction(
                            _peak_rank, mid, _j_resolve_contribs,
                            _j_resolve_contribs_ps, len(a_vec))
                        _fr = np.clip(np.asarray(_fr, dtype=float), 0.0, 1.0)
                        a_vec = a_vec * _fr
                        _j_frac_this_met = float(np.mean(_fr))

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
                    # Integration mode
                    _im = ("line_sel+trapz" if _line_sel_applied
                           else "mcr" if use_mcr else "trapz")
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
                _area_ok = True
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
                                  '_lorentz': None,
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

            # `found` only means a detected peak was located. An analyte whose
            # every hit was skipped (window holding none of its predicted lines)
            # has no area of its own — it must fall through to the not-quantified
            # list, not be reported with a zero.
            if not found or not _area_ok:
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
                messagebox.showinfo('Info', 'No signals matched annotation results.')
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
        # competitors after filtering; Average/Poor don't affect quantification.
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
            # Average/Poor metabolites excluded — only Good competitors matter.
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

            # Representative peak for navigation ("Open peak in Peak List"): the
            # detected peak INSIDE the integrated multiplet window nearest the
            # reported centre, so it matches what Quant Peaks shows and the
            # boundaries actually integrated — not necessarily the competition
            # anchor (_peak_rank), which can sit on one leg only.
            _disp_rank = _peak_rank_val
            try:
                _line_pks = _multiplet_line_peaks(_sel_hit, _pk_by_rank)
                _cands_d = [(abs(float(_pk.get('ppm', 1e9)) - ppm_val),
                             _pk.get('rank'))
                            for _pk in _line_pks if _pk.get('rank') is not None]
                if _cands_d:
                    _disp_rank = min(_cands_d)[1]
            except Exception:
                pass

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
                # No detected apex: the window came from the database's line
                # positions, not from anything the spectrum showed. The number
                # is real, what justifies it is weaker, and the table should be
                # able to say so.
                '_synthetic_db':     bool(m.get('_synthetic_db', False)),
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
                '_display_peak_rank': _disp_rank,
                '_used_j_resolve':   _used_jr,
                '_j_resolve_frac':   m.get('_j_resolve_frac'),
                '_jr_line_fracs':    m.get('_jr_line_fracs'),
                '_n_eligible':       _n_elig,
                '_peak_align_shifts': m.get('_align_shifts'),
                '_hits':             m.get('hits', []),
            })

        # ── Lipid composition indices (Box 1) — CDCl3/Li extract only ─────────
        # FA composition (Li et al. 2017) relies on CDCl3 chemical shifts and is
        # invalid for aqueous (D2O) plasma spectra, so it is skipped there.
        if is_lipidomics_mode and lipidomics.supports_composition(lipid_solvent):
            _lipid_df = lipidomics.load_library(lipid_solvent)
            _lipid_idx_entries = lipidomics.composition_indices(
                ppm_arr, all_int, n_s, groups, _lipid_df)
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
                    '_lorentz':          None,
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
                # Only peaks nobody won. _all_peak_quant holds every matched
                # peak of every metabolite, winners included, so picking freely
                # from it hands a loser the very signal it lost — which is what
                # put two metabolites on one region in the results table.
                _free_apq = {}
                for _apq_rank, _apq_e in _apq_mid_dict.items():
                    if not isinstance(_apq_e, dict):
                        continue
                    if _apq_rank in _assigned_peaks:
                        continue
                    _free_apq[_apq_rank] = _apq_e
                _best_apq  = None
                _best_conc = float('-inf')
                for _apq_e in _free_apq.values():
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
                # Use the shared _cv helper (sample SD, ddof=1) so this fallback
                # path reports CV% identically to the main quantification path —
                # an inline np.std here defaulted to ddof=0 and diverged.
                _apq_cv2  = {_g2: _cv(_gv2) for _g2, _gv2 in _apq_cbg2.items()}
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

        # ── Não-quantificados (cinzento na lista, fora de export/estatística) ──
        # (1) Metabolitos Good que perderam TODAS as competições de picos — nunca
        #     entraram em met_map e não foram recuperados/APQ. Sem eles a lista
        #     escondia silenciosamente que o metabolito existia mas não ganhou
        #     sinal; CLAUDE.md exige tornar isto visível ao utilizador.
        # (2) Metabolitos que ganharam pico mas cuja concentração é 0/NaN em todas
        #     as amostras (integraram só ruído).
        # Ambos ficam marcados _not_quantified=True com a razão em _status_reasons
        # (mostrada no tooltip da coluna Conf). São excluídos de export e stats.
        if not is_lipidomics_mode and not _fs_mode:
            _name_by_mid = {}
            for _r_nm in results:
                _rm = str(_r_nm.get('_mid', ''))
                if _rm and _rm not in _name_by_mid and _r_nm.get('name'):
                    _name_by_mid[_rm] = _r_nm.get('name')
            for _met_nm in (getattr(self.annotation_stage, '_metabolites', []) or []):
                _mid_nm = str(_met_nm.get('id', ''))
                if _mid_nm and _mid_nm not in _name_by_mid and _met_nm.get('name'):
                    _name_by_mid[_mid_nm] = _met_nm.get('name')

            _quantified_mids = {
                str(r.get('hmdb_id') or r.get('bmrb_id') or '')
                for r in self._quant_results
            }
            _ambiguous_mids = {
                str(m.get('id', ''))
                for m in (getattr(self.annotation_stage, '_metabolites', []) or [])
                if any(mp.get('peak_rank') in _ambiguous_ranks
                       for mp in m.get('matched_peaks', []))
            }
            def _nq_reason(mid):
                """Why this metabolite has no number, in its own words.

                A multiplet whose lines were not all found never entered the
                competition, and neither did one whose J quality was missing.
                Telling either of them they were outscored hides the only
                actionable difference: those come back if detection resolves
                the lines, a metabolite beaten by a rival does not.
                """
                if mid in _ambiguous_mids or mid in _blocked_ambiguous:
                    return 'Ambiguous annotation - manual confirmation required'
                # A metabolite can have one signal withdrawn and another that
                # competed and lost; then it did lose, and saying otherwise
                # would send the analyst looking at the wrong signal.
                if mid not in _competed_mids:
                    if mid in _blocked_unverified:
                        return NQ_REASON_UNVERIFIED_MULTIPLET
                    if mid in _blocked_no_sdj:
                        return NQ_REASON_NO_SDJ
                return NQ_REASON_LOST_COMPETITION

            _n_lost = 0
            for _lost_mid in sorted(good_met_ids):
                if str(_lost_mid) in _quantified_mids:
                    continue
                self._quant_results.append({
                    'metabolite':        _name_by_mid.get(str(_lost_mid), str(_lost_mid)),
                    'hmdb_id':           _lost_mid,
                    'ppm':               0.0,
                    'n_peaks':           0,
                    'raw_area':          float('nan'),
                    'corr_area':         float('nan'),
                    'conc_mm':           float('nan'),
                    'cv_by_group':       {},
                    'snr':               0.0,
                    'status':            'n.q.',
                    '_confidence':       0,
                    '_status_reasons':   [_nq_reason(str(_lost_mid))],
                    '_peak':             None,
                    '_areas_per_sample': [],
                    '_concs_per_sample': [],
                    '_concs_by_group':   {},
                    '_integ_mode':       _integ,
                    '_ann_score':        _mid_ann_score.get(str(_lost_mid), 0.0),
                    '_hits':             [],
                    '_not_quantified':   True,
                })
                _n_lost += 1

            # Flag won-but-empty analytes (0/NaN across all samples).
            for _r_nq in self._quant_results:
                if (_r_nq.get('_is_is') or _r_nq.get('_is_lipid_index')
                        or _r_nq.get('_not_quantified')):
                    continue
                if is_unquantified_conc(_r_nq.get('conc_mm')):
                    _r_nq['_not_quantified'] = True
                    _r_nq['status'] = 'n.q.'
                    _reasons_nq = _r_nq.get('_status_reasons') or []
                    if NQ_REASON_ZERO_NAN not in _reasons_nq:
                        _r_nq['_status_reasons'] = [NQ_REASON_ZERO_NAN] + _reasons_nq
            _n_nq = sum(1 for r in self._quant_results if r.get('_not_quantified'))
            print(f'[Quant] não-quantificados: {_n_nq} '
                  f'({_n_lost} perderam competição, {_n_nq - _n_lost} 0/NaN) '
                  f'— excluídos de export e estatística')

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
            _mcr_note = ' +MCR' if _use_mcr_f else ''
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
            # The list holds every analyte that was considered, quantified or
            # not, so its length is not the number of results. Reporting it as
            # "N metabolites" read as N measurements when a third of them were
            # rows explaining why there was no measurement.
            _n_all = len(self._quant_results)
            _n_nq_s = sum(1 for _r_s in self._quant_results
                          if _r_s.get('_not_quantified'))
            _n_ok_s = _n_all - _n_nq_s
            _n_db_s = sum(1 for _r_s in self._quant_results
                          if _r_s.get('_synthetic_db')
                          and not _r_s.get('_not_quantified'))
            _msg_s = f'Quantification: {_n_ok_s} quantified'
            if _n_db_s:
                _msg_s += f' ({_n_db_s} without a detected peak)'
            if _n_nq_s:
                _msg_s += f'  ·  {_n_nq_s} not quantified'
            self._status_ok(_msg_s)

            self._update_peak_list_labels()

            if getattr(self, '_warn_no_sdj', False):
                messagebox.showwarning(
                    'J-quality scores unavailable',
                    'Multiplet J-quality scores (s_dj) were not available for this '
                    'run.\n\n'
                    'Quantification could NOT prioritise confirmed multiplets over '
                    'singlets when two analytes competed for the same signal — '
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

        # Per-sample vectors can differ in length across analytes (e.g. an analyte
        # quantified before a sample was added). Pad/truncate each to exactly n_s
        # so the matrix is rectangular — a ragged list raises "inhomogeneous shape"
        # under numpy and silently skipped all VIP scores.
        rows = []
        for r in results:
            c = list(r.get('_concs_per_sample', []) or [])
            if len(c) < n_s:
                c = c + [float('nan')] * (n_s - len(c))
            elif len(c) > n_s:
                c = c[:n_s]
            rows.append(c)
        X = np.array(rows, dtype=float).T

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

        if norm_mode_used in ('IS', 'IS+PQN') and ref_areas_used is not None:
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
        elif _norm in ('IS', 'IS+PQN'):
            conc_s = f'{conc:.4f}'
        elif _norm == 'Total':
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
                quality_cols = (['ChemShift(ppm)', 'Width(ppm)', 'SNR', 'R2', 'Score',
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

                    # One row per analyte (skip IS internal standard row and
                    # não-quantificados — these carry no measurable signal).
                    for r in self._quant_results:
                        if r.get('_is_is') or r.get('_not_quantified'):
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
                        r2_val    = p.get('gaussian_r2', float('nan'))
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
                        # Bug 6 fix: skip IS internal standard and lipid index rows.
                        # Skip não-quantificados — no measurable signal to export.
                        if (r.get('_is_is') or r.get('_is_lipid_index')
                                or r.get('_not_quantified')):
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
