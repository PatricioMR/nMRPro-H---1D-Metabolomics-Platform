# gui/project_manager.py
"""
NMRAnalysis Project Manager.

Saves and restores the full application state to a .nmrproj file (pickle).
Auto-saves every 5 minutes when a project is open.

Saved state:
  - Raw data (ppm, all_intensities, samples_info, group_colors, source file path)
  - Peak detection (noise_std, peaks list, overlapping regions)
  - Deconvolution (queued peaks, results)
  - Annotation stage (db_path, results)
  - UI state (active stage, axes settings, plot settings)
"""
import os
import pickle
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from processing.spectral_regions import (
    format_regions, parse_regions)

from app_paths import APP_VERSION

PROJECT_EXT   = '.nmrproj'
AUTOSAVE_SECS = 300  # 5 minutes


def _safe_var(obj, attr):
    """Return the value of a tkinter Variable attribute, or None if missing."""
    try:
        return getattr(obj, attr).get()
    except Exception:
        return None


def _set_var(obj, attr, value):
    """Set a tkinter Variable attribute if it exists and value is not None."""
    if value is None:
        return
    try:
        getattr(obj, attr).set(value)
    except Exception:
        pass


class ProjectManager:
    def __init__(self, app):
        self.app          = app
        self._project_path = None   # current open project file
        self._dirty        = False  # unsaved changes
        self._autosave_job = None

    # ── Path helpers ──────────────────────────────────────────────────────────

    @property
    def project_path(self):
        return self._project_path

    @project_path.setter
    def project_path(self, value):
        self._project_path = value
        title = APP_VERSION
        if value:
            title = f'{APP_VERSION} — {os.path.basename(value)}'
        try:
            self.app.root.title(title)
        except Exception:
            pass

    def mark_dirty(self):
        self._dirty = True

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _collect_state(self):
        """Gather everything needed to restore the session."""
        app = self.app
        state = {
            'version':         6,
            # Raw data
            'ppm':             app.ppm,
            'ppm_calibration': getattr(app, '_ppm_calibration', None),
            'all_intensities':     app.all_intensities,
            # The project's single baseline decision. Saved as the CHOICE, not
            # the corrected numbers: all_intensities stays raw, so reopening
            # reproduces the same correction instead of baking one in.
            'baseline_model': (app.baseline_model.to_dict()
                               if getattr(app, 'baseline_model', None) is not None
                               else None),
            'baseline_corrected':  getattr(app, 'baseline_corrected',  False),
            'alignment_corrected': getattr(app, 'alignment_corrected', False),
            # The rigid shift detection applied to each sample, kept so the
            # alignment baked into all_intensities stays auditable.
            'alignment_shifts_ppm': getattr(app, 'alignment_shifts_ppm', None),
            'samples_info':        app.samples_info,
            'group_colors':    app.group_colors,
            'source_file':     app.current_file,
            'group_dilution_factors': dict(getattr(app, 'group_dilution_factors', {})),
            # UI
            'active_stage':    getattr(app, 'active_stage', 'visualization'),
            'axes_settings':   getattr(app, 'axes_settings', {}),
            'plot_settings':   getattr(app, 'plot_settings', {}),
            'show_legend':     getattr(app, 'show_legend', tk.BooleanVar()).get(),
            # Cross-stage state
            'peak_list_filters': dict(getattr(app, '_peak_list_filters', {})),
            'eligible_overrides': dict(getattr(app, 'eligible_overrides', {})),
        }

        # ── Peak detection ────────────────────────────────────────────────────
        pd = app.stages.get('peak_detection')
        if pd:
            state['pd'] = {
                'noise_std':       pd.noise_std,
                'noise_source':    getattr(pd, 'noise_source', None),
                'peaks':           list(pd.peaks),
                '_overlaps':       list(getattr(pd, '_overlaps', [])),
                'step1_completed': getattr(pd, 'step1_completed', False),
                'noise_start_var': _safe_var(pd, 'noise_start_var'),
                'noise_end_var':   _safe_var(pd, 'noise_end_var'),
                'min_snr_var':     _safe_var(pd, 'min_snr_var'),
                'min_width_var':   _safe_var(pd, 'min_width_var'),
                # Which regions were excluded is part of what the peak list
                # means. Reopening a project with a different exclusion would
                # silently change which signals exist in it.
                'excluded_regions': _safe_var(pd, 'excluded_regions_var'),
                # What the run actually applied, not what the field now reads.
                # Without it a reopened project cannot say which regions its
                # peaks were detected under, and quantification silently
                # measures detection limits inside the suppressed solvent.
                'excluded_regions_applied': format_regions(
                    getattr(pd, '_excluded_regions', None) or ()) or None,
                'per_sample_detection': bool(_safe_var(pd, 'per_sample_var')
                                             if _safe_var(pd, 'per_sample_var') is not None
                                             else True),
                'detect_shoulders': bool(_safe_var(pd, 'detect_shoulders_var')
                                         if _safe_var(pd, 'detect_shoulders_var') is not None
                                         else False),
                'shoulder_k_var':  _safe_var(pd, 'shoulder_k_var'),
                'min_group_presence_var': _safe_var(pd, 'min_group_presence_var'),
            }
            ctrl = getattr(pd, '_deconv_controller', None)
            if ctrl:
                state['pd']['deconv'] = {
                    'peaks':            list(ctrl.peaks),
                    'results':          dict(ctrl.results),
                    'excluded_samples': {k: list(v)
                                         for k, v in ctrl.excluded_samples.items()},
                }

        # ── Annotation ────────────────────────────────────────────────────────
        ann = app.stages.get('annotation')
        if ann:
            # Capture live metabolites from the open results window (most up-to-date)
            _live_win = getattr(app, '_annot_win', None)
            _mets = None
            if _live_win is not None:
                try:
                    if _live_win.win.winfo_exists():
                        _mets = getattr(_live_win, 'metabolites', None)
                except Exception:
                    pass
            if _mets is None:
                _mets = getattr(ann, '_metabolites', None)

            state['ann'] = {
                'db_path':               getattr(ann, '_db_path', None),
                'last_results_args':     getattr(ann, '_last_results_args', None),
                'annotations':           dict(getattr(ann, 'annotations', {})),
                'tolerance':             _safe_var(ann, 'tolerance_var'),
                # The field every predicted J line is placed with. Not
                # saving it reopened a 500 MHz study at 600 and moved every
                # line a fifth of its splitting.
                'spectrometer_mhz':      _safe_var(ann, 'spectrometer_var'),
                'align':                 _safe_var(ann, 'align_var'),
                'exclude_water':         _safe_var(ann, 'exclude_water_var'),
                'water_lo':              _safe_var(ann, 'water_lo_var'),
                'water_hi':              _safe_var(ann, 'water_hi_var'),
                'internal_library_mode': _safe_var(ann, 'internal_library_mode_var'),
                'lipid_solvent':         _safe_var(ann, 'lipid_solvent_var'),
                # Scoring thresholds and the allowed signal grades. Without
                # these a reloaded project re-annotates under different rules
                # than the ones its marks were produced with.
                'annotation_params':     (dict(getattr(ann, '_annotation_params', None))
                                          if getattr(ann, '_annotation_params', None)
                                          else None),
                'quality_marks':         dict(getattr(ann, '_quality_marks', {})),
                'child_quality_marks':   dict(getattr(ann, '_child_quality_marks', {})),
                'anchor_lines':          dict(getattr(ann, '_anchor_lines', {})),
                'subline_quality_marks': dict(getattr(ann, '_subline_quality_marks', {})),
                'metabolites':           _mets,
                # Applied J-Resolve fractions consumed by Quantification.
                # Without these the corrected areas are lost on reload.
                'jr_contributions':            getattr(ann, '_j_resolve_contributions', None),
                'jr_contributions_per_sample': getattr(ann, '_j_resolve_contributions_per_sample', None),
                'jr_status':                   getattr(ann, '_j_resolve_status', None),
                'jr_conc_est':                 getattr(ann, '_j_resolve_conc_est', None),
            }

        # ── J-Resolve ─────────────────────────────────────────────────────────
        jr = app.stages.get('j_resolve')
        if jr:
            state['jr'] = {
                'conc_est':          dict(getattr(jr, '_conc_est', {})),
                'anchor_peaks':      dict(getattr(jr, '_anchor_peaks', {})),
                'nnls_rejected':     list(getattr(jr, '_nnls_rejected', set())),
                'j_validated':       list(getattr(jr, '_j_validated', set())),
                'peak_overlap':      dict(getattr(jr, '_peak_overlap', {})),
                'contributions':           dict(getattr(jr, '_contributions', {})),
                'contributions_per_sample': dict(getattr(jr, '_contributions_per_sample', {})),
                'status_map':        dict(getattr(jr, '_status_map', {})),
                'anchor_confidence': dict(getattr(jr, '_anchor_confidence', {})),
                'anchor_foldrange':  dict(getattr(jr, '_anchor_foldrange', {})),
                'anchor_ci':         dict(getattr(jr, '_anchor_ci', {})),
                'residual_attributed': dict(getattr(jr, '_residual_attributed', {})),
                'residual_diagnostic': dict(getattr(jr, '_residual_diagnostic', {})),
                'anchor_tol':        _safe_var(jr, '_anchor_tol'),
                'dj_filter':         _safe_var(jr, '_dj_filter_var'),
                'dj_tol':            _safe_var(jr, '_dj_tol'),
                'singlet_anchor':    _safe_var(jr, '_singlet_anchor_var'),
            }

        # ── Quantification ────────────────────────────────────────────────────
        qnt = app.stages.get('quantification')
        if qnt:
            _dil = getattr(qnt, '_last_sample_dil', None)
            state['qnt'] = {
                'quant_results':    list(getattr(qnt, '_quant_results', [])),
                'quant_peak_overrides': {
                    str(k): v for k, v in
                    (getattr(qnt, '_quant_peak_overrides', {}) or {}).items()},
                'deconv_overrides': {
                    str(k): dict(v) for k, v in
                    (getattr(qnt, '_deconv_overrides', {}) or {}).items()},
                'integ':            _safe_var(qnt, 'integ_var'),
                'norm':             _safe_var(qnt, 'norm_var'),
                'mcr':              _safe_var(qnt, 'mcr_var'),
                'blk_sub':          _safe_var(qnt, 'blk_sub_var'),
                'local_baseline':   _safe_var(qnt, 'local_baseline_var'),
                'baseline_model_quant': _safe_var(qnt, 'bl_model_quant_var'),
                'shape_fit':        _safe_var(qnt, 'shape_var'),
                'shared_deconv':    _safe_var(qnt, 'shared_var'),
                'singlet_min_snr':  _safe_var(qnt, 'singlet_snr_var'),
                'cascade':          _safe_var(qnt, 'cascade_var'),
                'joint_fit':        _safe_var(qnt, 'joint_fit_var'),
                'max_disagreement': _safe_var(qnt, 'max_disagreement_var'),
                # Saved under its old key: the setting is the same question
                # and an old project that answered it must keep its answer.
                'j_resolve':        _safe_var(qnt, 'deconv_var'),
                'ext_is':           _safe_var(qnt, 'ext_is_var'),
                'ext_is_indices':   [int(x) for x in getattr(qnt, '_ext_is_indices', [])],
                'ref_ppm':          _safe_var(qnt, 'ref_ppm_var'),
                'ref_nh':           _safe_var(qnt, 'ref_nh_var'),
                'ref_conc':         _safe_var(qnt, 'ref_conc_var'),
                'ref_std':          _safe_var(qnt, 'ref_std_var'),
                'ref_rank':         _safe_var(qnt, 'ref_rank_var'),
                'last_norm_mode':   getattr(qnt, '_last_norm_mode',   None),
                'last_ref_areas':   getattr(qnt, '_last_ref_areas',   None),
                'last_ref_conc':    getattr(qnt, '_last_ref_conc',    None),
                'last_sample_dil':  _dil.tolist() if _dil is not None else None,
                'last_groups':      getattr(qnt, '_last_groups',      None),
                'last_blk_indices': getattr(qnt, '_last_blk_indices', None),
                'last_non_blk':     getattr(qnt, '_last_non_blk_indices', None),
            }

        # ── Notation stage ────────────────────────────────────────────────────
        ns = app.stages.get('notation')
        if ns:
            state['ns'] = {
                'peaks': list(getattr(ns, 'peaks', [])),
            }

        # ── Full-spectrum MCR ─────────────────────────────────────────────────
        fs = app.stages.get('fullspectrum_mcr')
        if fs:
            state['fs'] = {
                'mcr_regions':    list(getattr(fs, '_mcr_regions', [])),
                'db_path':        getattr(fs, '_db_path', None),
                'norm_factors':   getattr(fs, '_norm_factors', None),
                'blank_enabled':  _safe_var(fs, '_blank_enabled'),
                'norm_enabled':   _safe_var(fs, '_norm_enabled'),
                'norm_method':    _safe_var(fs, '_norm_method'),
            }

        # ── Statistics ────────────────────────────────────────────────────────
        st = app.stages.get('statistics')
        if st:
            state['st'] = {
                'method':              _safe_var(st, '_method_var'),
                'mode':                _safe_var(st, '_mode_var'),
                'scale':               _safe_var(st, '_scale_var'),
                'log_pre':             _safe_var(st, '_log_pre_var'),
                'corr_method':         _safe_var(st, '_corr_method_var'),
                'corr_plot':           _safe_var(st, '_corr_plot_var'),
                'corr_x':              _safe_var(st, '_corr_x_var'),
                'corr_y':              _safe_var(st, '_corr_y_var'),
                'excluded_samples':    list(getattr(st, '_excluded_samples', set())),
                'excluded_metabolites': list(getattr(st, '_excluded_metabolites', set())),
                'pie_style':           dict(getattr(st, '_pie_style', {})),
                'box_style':           dict(getattr(st, '_box_style', {})),
                'box_config':          dict(getattr(st, '_box_config', {})),
                'volcano_style':       dict(getattr(st, '_volcano_style', {})),
                'pathway': {
                    'plot':     _safe_var(st, '_pw_plot_var'),
                    'category': _safe_var(st, '_pw_cat_var'),
                    'p_thr':    _safe_var(st, '_pw_p_thr_var'),
                    'min_mets': _safe_var(st, '_pw_minmets_var'),
                    'top_n':    _safe_var(st, '_pw_topn_var'),
                    'use_fdr':  _safe_var(st, '_pw_use_fdr_var'),
                    'jaccard':  _safe_var(st, '_pw_jaccard_var'),
                    'sig_only': _safe_var(st, '_pw_sig_only_var'),
                    'cmap':     _safe_var(st, '_pw_cmap_var'),
                    'group_a':  _safe_var(st, '_grp_a_var'),
                    'group_b':  _safe_var(st, '_grp_b_var'),
                    'group_ca': _safe_var(st, '_grp_ca_var'),
                    'group_cb': _safe_var(st, '_grp_cb_var'),
                },
            }

        return state

    def _restore_state(self, state):
        """Restore application state from a saved dict."""
        import numpy as _np
        app = self.app

        # ── Close stale annotation results window from a previous project ─────
        _old_annot_win = getattr(app, '_annot_win', None)
        if _old_annot_win is not None:
            try:
                _old_annot_win.win.destroy()
            except Exception:
                pass
            app._annot_win = None

        # ── Raw data ──────────────────────────────────────────────────────────
        app.ppm                  = state.get('ppm')
        app.all_intensities      = state.get('all_intensities')
        app._ppm_calibration     = state.get('ppm_calibration')
        _bl_state = state.get('baseline_model')
        if _bl_state is not None:
            from processing.baseline_model import BaselineModel
            app.baseline_model = BaselineModel.from_dict(_bl_state)
        try:
            app.invalidate_baseline_cache()
        except Exception:
            pass
        app.baseline_corrected   = state.get('baseline_corrected',  False)
        app.alignment_corrected  = state.get('alignment_corrected', False)
        app.alignment_shifts_ppm = state.get('alignment_shifts_ppm')
        app.samples_info     = state.get('samples_info', [])
        app.group_colors     = state.get('group_colors', app.group_colors)
        app.current_file     = state.get('source_file')
        app.group_dilution_factors = state.get('group_dilution_factors',
                                               getattr(app, 'group_dilution_factors', {}))
        app._peak_list_filters = state.get('peak_list_filters',
                                           getattr(app, '_peak_list_filters', {}))
        app.eligible_overrides = state.get('eligible_overrides',
                                           getattr(app, 'eligible_overrides', {}))

        # ── UI ────────────────────────────────────────────────────────────────
        if state.get('axes_settings'):
            app.axes_settings.update(state['axes_settings'])
        if state.get('plot_settings'):
            app.plot_settings.update(state['plot_settings'])
        try:
            app.show_legend.set(state.get('show_legend', True))
        except Exception:
            pass
        try:
            fname = os.path.basename(app.current_file) if app.current_file else 'No file loaded'
            app.file_label.config(text=fname)
        except Exception:
            pass
        try:
            app.stages['visualization'].refresh_sample_list()
            app.btn_select_all.config(state='normal')
            app.btn_clear_all.config(state='normal')
            app.btn_update.config(state='normal')
        except Exception:
            pass

        # Sync processing-flag warnings in peak detection UI after data is loaded
        try:
            app.stages['peak_detection']._sync_baseline_corrected_ui()
        except Exception:
            pass

        # ── Peak detection ────────────────────────────────────────────────────
        pd_state = state.get('pd')
        pd       = app.stages.get('peak_detection')
        if pd_state and pd:
            pd.noise_std       = pd_state.get('noise_std')
            pd.noise_source    = pd_state.get('noise_source')
            pd.peaks           = pd_state.get('peaks', [])
            pd._overlaps       = pd_state.get('_overlaps', [])
            pd.step1_completed = pd_state.get('step1_completed', False)
            _set_var(pd, 'noise_start_var', pd_state.get('noise_start_var'))
            _set_var(pd, 'noise_end_var',   pd_state.get('noise_end_var'))
            _set_var(pd, 'min_snr_var',     pd_state.get('min_snr_var'))
            _set_var(pd, 'min_width_var',   pd_state.get('min_width_var'))
            if pd_state.get('excluded_regions') is not None:
                _set_var(pd, 'excluded_regions_var',
                         pd_state.get('excluded_regions'))
            _applied = pd_state.get('excluded_regions_applied')
            if _applied:
                try:
                    pd._excluded_regions = parse_regions(_applied)
                except (ValueError, AttributeError):
                    pass
            _set_var(pd, 'per_sample_var',  pd_state.get('per_sample_detection', True))
            _set_var(pd, 'detect_shoulders_var', pd_state.get('detect_shoulders', False))
            _set_var(pd, 'shoulder_k_var',  pd_state.get('shoulder_k_var'))
            _set_var(pd, 'min_group_presence_var', pd_state.get('min_group_presence_var'))
            # Baseline lives on the app as a single model; the panel mirrors it.
            try:
                pd.pull_baseline_model()
            except Exception:
                pass

            if pd.processor and app.ppm is not None:
                try:
                    if pd.noise_std:
                        pd.processor.set_noise(pd.noise_std)
                    pd.processor.ppm             = app.ppm
                    pd.processor.peaks           = list(pd.peaks)
                    pd.processor.all_intensities = app.all_intensities
                    from gui.stages.peak_detection_stage import _no_blk_group_mean
                    pd.processor.mean_intensity  = _no_blk_group_mean(
                        app.all_intensities, app.samples_info)
                except Exception as _e:
                    print(f'  processor restore: {_e}')

            try:
                if pd.noise_std is not None:
                    pd.step1_completed = True
                    pd.noise_value_label.config(text=f'Noise Std Dev: {pd.noise_std:.4f}')
                    pd.step1_status.config(text='✅ Completed', foreground='green')
                    pd.btn_detect.config(state='normal')
                if pd.peaks:
                    pd.peaks_count_label.config(text=f'Found {len(pd.peaks)} signals')
                    pd.step2_status.config(
                        text=f'✅ {len(pd.peaks)} signals detected', foreground='green')
                    try: app.btn_peak_list.config(state='normal')
                    except Exception: pass
            except Exception as _e:
                print(f'  pd UI restore: {_e}')

            deconv_state = pd_state.get('deconv')
            if deconv_state and pd._overlaps:
                try:
                    from gui.notation.deconvolution import DeconvController
                    ctrl = DeconvController(pd.processor)
                    ctrl.peaks            = deconv_state.get('peaks', [])
                    ctrl.results          = deconv_state.get('results', {})
                    ctrl.excluded_samples = {k: set(v) for k, v in
                                             deconv_state.get('excluded_samples', {}).items()}
                    pd._deconv_controller = ctrl
                except Exception as _e:
                    print(f'  deconv controller restore: {_e}')

        # ── Annotation ────────────────────────────────────────────────────────
        ann_state = state.get('ann')
        ann       = app.stages.get('annotation')
        if ann_state and ann:
            try:
                ann._db_path = ann_state.get('db_path') or getattr(ann, '_db_path', None)

                if ann_state.get('last_results_args'):
                    ann._last_results_args = ann_state['last_results_args']
                    try: app.btn_annot_shortcut.config(state='normal')
                    except Exception: pass

                if ann_state.get('annotations'):
                    ann.annotations = ann_state['annotations']

                _set_var(ann, 'tolerance_var', ann_state.get('tolerance'))
                if ann_state.get('spectrometer_mhz'):
                    _set_var(ann, 'spectrometer_var',
                             ann_state.get('spectrometer_mhz'))
                _set_var(ann, 'align_var',     ann_state.get('align'))
                _set_var(ann, 'exclude_water_var', ann_state.get('exclude_water'))
                _set_var(ann, 'water_lo_var',      ann_state.get('water_lo'))
                _set_var(ann, 'water_hi_var',      ann_state.get('water_hi'))
                _set_var(ann, 'internal_library_mode_var',
                         ann_state.get('internal_library_mode'))
                _set_var(ann, 'lipid_solvent_var', ann_state.get('lipid_solvent'))
                # Reflect lipidomics solvent selector visibility after restore.
                if hasattr(ann, '_on_lib_mode_change'):
                    try: ann._on_lib_mode_change()
                    except Exception: pass

                _ann_p = ann_state.get('annotation_params')
                ann._annotation_params = dict(_ann_p) if _ann_p else None
                ann._quality_marks         = ann_state.get('quality_marks', {})
                ann._child_quality_marks   = ann_state.get('child_quality_marks', {})
                ann._anchor_lines          = ann_state.get('anchor_lines', {})
                ann._subline_quality_marks = ann_state.get('subline_quality_marks', {})

                # Restore applied J-Resolve fractions so Quantification reproduces
                # the corrected areas without re-running J-Resolve.
                if ann_state.get('jr_contributions') is not None:
                    ann._j_resolve_contributions = ann_state['jr_contributions']
                if ann_state.get('jr_contributions_per_sample') is not None:
                    ann._j_resolve_contributions_per_sample = \
                        ann_state['jr_contributions_per_sample']
                if ann_state.get('jr_status') is not None:
                    ann._j_resolve_status = ann_state['jr_status']
                if ann_state.get('jr_conc_est') is not None:
                    ann._j_resolve_conc_est = ann_state['jr_conc_est']

                if ann_state.get('metabolites'):
                    ann._metabolites = ann_state['metabolites']
                    # Re-apply quality marks into the saved metabolites' matched_peaks
                    # so the results window shows correct colours without re-scoring.
                    for met in ann._metabolites:
                        mid = met.get('id', met.get('name', ''))
                        for mp in met.get('matched_peaks', []):
                            key = (mid, mp.get('peak_rank'))
                            if key in ann._quality_marks:
                                mp['annotation_quality'] = ann._quality_marks[key]
                else:
                    # No saved metabolites — re-apply quality marks into flat results
                    # so they are picked up when the window is rebuilt from scratch.
                    _results = (ann._last_results_args or {}).get('results', [])
                    for r in _results:
                        mid = (r.get('hmdb_id') or r.get('bmrb_id')
                               or r.get('id', r.get('name', '')))
                        key = (mid, r.get('peak_rank'))
                        if key in ann._quality_marks:
                            r['annotation_quality'] = ann._quality_marks[key]

            except Exception as _e:
                print(f'  annotation restore: {_e}')

        # ── J-Resolve ─────────────────────────────────────────────────────────
        jr_state = state.get('jr')
        jr       = app.stages.get('j_resolve')
        if jr_state and jr:
            try:
                jr._conc_est     = jr_state.get('conc_est', {})
                jr._anchor_peaks = jr_state.get('anchor_peaks', {})
                jr._nnls_rejected = set(jr_state.get('nnls_rejected', []))
                jr._j_validated  = set(jr_state.get('j_validated', []))
                jr._peak_overlap = jr_state.get('peak_overlap', {})
                jr._contributions = jr_state.get('contributions', {})
                jr._contributions_per_sample = jr_state.get(
                    'contributions_per_sample', {})
                jr._status_map = jr_state.get('status_map', {})
                jr._anchor_confidence = jr_state.get('anchor_confidence', {})
                jr._anchor_foldrange = jr_state.get('anchor_foldrange', {})
                jr._anchor_ci = jr_state.get('anchor_ci', {})
                jr._residual_attributed = jr_state.get('residual_attributed', {})
                jr._residual_diagnostic = jr_state.get('residual_diagnostic', {})
                _set_var(jr, '_anchor_tol',        jr_state.get('anchor_tol'))
                _set_var(jr, '_dj_filter_var',     jr_state.get('dj_filter'))
                _set_var(jr, '_dj_tol',            jr_state.get('dj_tol'))
                _set_var(jr, '_singlet_anchor_var', jr_state.get('singlet_anchor'))
                # Share the same metabolites list as annotation
                if ann is not None and getattr(ann, '_metabolites', None):
                    jr._metabolites = ann._metabolites
                if pd is not None:
                    jr._peaks = list(pd.peaks)
                # Enable the Run button if we have conc_est results
                if jr._conc_est:
                    try: jr._btn_apply.config(state='normal')
                    except Exception: pass
            except Exception as _e:
                print(f'  j_resolve restore: {_e}')

        # ── Quantification ────────────────────────────────────────────────────
        qnt_state = state.get('qnt')
        qnt       = app.stages.get('quantification')
        if qnt_state and qnt:
            try:
                results = qnt_state.get('quant_results', [])
                if results:
                    qnt._quant_results = results
                    try:
                        qnt.btn_csv.config(state='normal')
                        qnt.btn_metaboanalyst.config(state='normal')
                        qnt.btn_pdf.config(state='normal')
                    except Exception:
                        pass
                qnt._quant_peak_overrides = dict(
                    qnt_state.get('quant_peak_overrides', {}) or {})
                qnt._deconv_overrides = {
                    str(k): dict(v) for k, v in
                    (qnt_state.get('deconv_overrides', {}) or {}).items()}
                # Rows and overrides both restored: the button reads them.
                app.refresh_deconv_button()
                _integ_loaded = qnt_state.get('integ')
                if _integ_loaded == 'Lorentz':
                    _integ_loaded = 'Multiplet'   # 'Lorentz Fit' mode was removed
                _set_var(qnt, 'integ_var',      _integ_loaded)
                _set_var(qnt, 'norm_var',        qnt_state.get('norm'))
                _set_var(qnt, 'mcr_var',         qnt_state.get('mcr'))
                _set_var(qnt, 'blk_sub_var',     qnt_state.get('blk_sub'))
                _set_var(qnt, 'ext_is_var',        qnt_state.get('ext_is'))
                _set_var(qnt, 'local_baseline_var', qnt_state.get('local_baseline'))
                _set_var(qnt, 'bl_model_quant_var',
                         qnt_state.get('baseline_model_quant'))
                _set_var(qnt, 'shape_var',        qnt_state.get('shape_fit'))
                _set_var(qnt, 'shared_var',       qnt_state.get('shared_deconv'))
                _set_var(qnt, 'singlet_snr_var',  qnt_state.get('singlet_min_snr'))
                _set_var(qnt, 'cascade_var',      qnt_state.get('cascade'))
                _set_var(qnt, 'joint_fit_var',    qnt_state.get('joint_fit'))
                _set_var(qnt, 'max_disagreement_var',
                         qnt_state.get('max_disagreement'))
                # saved under 'j_resolve' since the JG-Deconv box; the
                # variable is deconv_var, and 'j_resolve_var' never existed
                _set_var(qnt, 'deconv_var',       qnt_state.get('j_resolve'))
                _set_var(qnt, 'ref_ppm_var',     qnt_state.get('ref_ppm'))
                _set_var(qnt, 'ref_nh_var',      qnt_state.get('ref_nh'))
                _set_var(qnt, 'ref_conc_var',    qnt_state.get('ref_conc'))
                _set_var(qnt, 'ref_std_var',     qnt_state.get('ref_std'))
                _set_var(qnt, 'ref_rank_var',    qnt_state.get('ref_rank'))
                ei = qnt_state.get('ext_is_indices')
                if ei is not None:
                    qnt._ext_is_indices = [int(x) for x in ei]
                    # Recompute label from indices + current samples_info
                    # (more robust than relying on saved label string)
                    _si  = getattr(app, 'samples_info', None) or []
                    _n   = len(qnt._ext_is_indices)
                    if _n and _si:
                        _names = ', '.join(
                            str(_si[_i].get('label', _i))
                            for _i in qnt._ext_is_indices[:3] if _i < len(_si))
                        _sfx = '…' if _n > 3 else ''
                        _lbl_txt = f'{_n} sample(s): {_names}{_sfx}'
                        _fg = '#1a7a1a'
                    elif _n:
                        _lbl_txt = f'{_n} sample(s) selected'
                        _fg = '#1a7a1a'
                    else:
                        _lbl_txt = '(none selected)'
                        _fg = 'gray'
                    try:
                        qnt._ext_is_label.config(text=_lbl_txt, foreground=_fg)
                    except Exception:
                        pass
                if qnt_state.get('last_norm_mode') is not None:
                    qnt._last_norm_mode = qnt_state['last_norm_mode']
                if qnt_state.get('last_ref_areas') is not None:
                    qnt._last_ref_areas = qnt_state['last_ref_areas']
                if qnt_state.get('last_ref_conc') is not None:
                    qnt._last_ref_conc = qnt_state['last_ref_conc']
                if qnt_state.get('last_sample_dil') is not None:
                    qnt._last_sample_dil = _np.array(qnt_state['last_sample_dil'])
                if qnt_state.get('last_groups') is not None:
                    qnt._last_groups = qnt_state['last_groups']
                if qnt_state.get('last_blk_indices') is not None:
                    qnt._last_blk_indices = qnt_state['last_blk_indices']
                if qnt_state.get('last_non_blk') is not None:
                    qnt._last_non_blk_indices = qnt_state['last_non_blk']
                # Repopulate the results table so it's visible immediately
                if results:
                    try: qnt._populate_tree()
                    except Exception: pass
            except Exception as _e:
                print(f'  quantification restore: {_e}')

        # ── Notation ─────────────────────────────────────────────────────────
        ns_state = state.get('ns')
        ns       = app.stages.get('notation')
        if ns_state and ns:
            try:
                ns.peaks = ns_state.get('peaks', [])
            except Exception as _e:
                print(f'  notation restore: {_e}')

        # ── Full-spectrum MCR ─────────────────────────────────────────────────
        fs_state = state.get('fs')
        fs       = app.stages.get('fullspectrum_mcr')
        if fs_state and fs:
            try:
                fs._mcr_regions  = fs_state.get('mcr_regions', [])
                fs._db_path      = fs_state.get('db_path')
                fs._norm_factors = fs_state.get('norm_factors')
                _set_var(fs, '_blank_enabled', fs_state.get('blank_enabled'))
                _set_var(fs, '_norm_enabled',  fs_state.get('norm_enabled'))
                _set_var(fs, '_norm_method',   fs_state.get('norm_method'))
                if fs._db_path:
                    try:
                        fs._db_label.config(
                            text=os.path.basename(fs._db_path), foreground='black')
                        fs.btn_run.config(state='normal')
                    except Exception:
                        pass
                if fs._mcr_regions:
                    n_k1 = sum(1 for r in fs._mcr_regions
                               if r.get('mcr_annotation', {}).get('K', 0) == 1)
                    n_k2 = sum(1 for r in fs._mcr_regions
                               if r.get('mcr_annotation', {}).get('K', 0) >= 2)
                    try:
                        fs._result_label.config(
                            text=(f'{len(fs._mcr_regions)} region(s) — {n_k1} single, '
                                  f'{n_k2} overlapping.'),
                            foreground='black')
                    except Exception:
                        pass
            except Exception as _e:
                print(f'  fullspectrum_mcr restore: {_e}')

        # ── Statistics ────────────────────────────────────────────────────────
        st_state = state.get('st')
        st       = app.stages.get('statistics')
        if st_state and st:
            try:
                _set_var(st, '_method_var', st_state.get('method'))
                _set_var(st, '_scale_var',  st_state.get('scale'))
                _set_var(st, '_log_pre_var', st_state.get('log_pre'))
                st._excluded_samples    = set(st_state.get('excluded_samples', []))
                st._excluded_metabolites = set(st_state.get('excluded_metabolites', []))
                # Pathway-mode params: stash so the lazily-built panel seeds from
                # them, then rebuild the left panel for the restored mode.
                if st_state.get('pathway'):
                    st._pw_saved = dict(st_state['pathway'])
                _mode = st_state.get('mode')
                if _mode and hasattr(st, '_mode_var') and hasattr(st, '_left_inner'):
                    _set_var(st, '_mode_var', _mode)
                    try:
                        st._build_left(st._left_inner)
                    except Exception:
                        pass
                if st_state.get('pie_style'):
                    st._pie_style.update(st_state['pie_style'])
                if st_state.get('box_style'):
                    st._box_style.update(st_state['box_style'])
                if st_state.get('box_config'):
                    st._box_config.update(st_state['box_config'])
                    if hasattr(st, '_update_box_cfg_status'):
                        st._update_box_cfg_status()
                # Correlation selectors (set after the panel rebuild so the
                # freshly-created vars are not left at their defaults)
                for _attr, _key in (('_corr_method_var', 'corr_method'),
                                    ('_corr_plot_var',   'corr_plot'),
                                    ('_corr_x_var',      'corr_x'),
                                    ('_corr_y_var',      'corr_y')):
                    _set_var(st, _attr, st_state.get(_key))
                if st_state.get('volcano_style'):
                    try:
                        st._ensure_volcano_style().update(st_state['volcano_style'])
                    except Exception:
                        pass
            except Exception as _e:
                print(f'  statistics restore: {_e}')

        # ── Redraw & switch stage ─────────────────────────────────────────────
        try:
            app.update_plot()
            if pd and pd.peaks:
                pd.color_peaks()
        except Exception as _e:
            print(f'  redraw after restore: {_e}')

        try:
            saved_stage = state.get('active_stage', 'visualization')
            if saved_stage in app.stages:
                app.switch_stage(saved_stage)
        except Exception:
            pass


    # ── File operations ───────────────────────────────────────────────────────

    def save_project(self, path=None):
        """Save to path. If None, opens Save-As dialog."""
        if path is None:
            path = filedialog.asksaveasfilename(
                title='Save Project',
                defaultextension=PROJECT_EXT,
                filetypes=[('NMR Project', f'*{PROJECT_EXT}'),
                           ('All files', '*.*')])
            if not path:
                return False

        try:
            state = self._collect_state()
            with open(path, 'wb') as f:
                pickle.dump(state, f, protocol=5)
            self.project_path = path
            self._dirty       = False
            # Begin (or restart) autosave now that a project path exists.
            # Previously autosave only started on open_project, so a freshly
            # created+saved project never auto-saved.
            self._start_autosave()
            self.app.root.after(
                0, lambda: self.app.status_label.config(
                    text=f'Project saved: {os.path.basename(path)}',
                    foreground='green'))
            return True
        except Exception as e:
            messagebox.showerror('Save Error',
                                 f'Could not save project:\n{e}')
            return False

    def open_project(self, path=None):
        """Open a project file, restoring full state."""
        if self._dirty:
            resp = messagebox.askyesnocancel(
                'Alterações não guardadas',
                'O projeto atual tem alterações não guardadas.\n'
                'Deseja guardar antes de abrir outro projeto?\n\n'
                '  Sim      → Guardar e continuar\n'
                '  Não      → Abrir sem guardar (as alterações serão perdidas)\n'
                '  Cancelar → Voltar sem abrir')
            if resp is None:        # Cancelar → abortar
                return False
            if resp:                # Sim → guardar; abortar se falhar
                if not self.save_or_saveas():
                    return False

        if path is None:
            path = filedialog.askopenfilename(
                title='Open Project',
                filetypes=[('NMR Project', f'*{PROJECT_EXT}'),
                           ('All files', '*.*')])
            if not path:
                return False

        try:
            self.app.status_label.config(
                text='Loading project…', foreground='blue')
            self.app.root.update_idletasks()

            with open(path, 'rb') as f:
                state = pickle.load(f)

            self._restore_state(state)
            self.project_path = path
            self._dirty       = False
            self.app.status_label.config(
                text=f'Project opened: {os.path.basename(path)}',
                foreground='green')
            self._start_autosave()
            return True
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror('Open Error',
                                 f'Could not open project:\n{e}')
            return False

    def save_or_saveas(self):
        """Save to current path, or Save-As if no project open."""
        if self._project_path:
            return self.save_project(self._project_path)
        return self.save_project()

    # ── Auto-save ─────────────────────────────────────────────────────────────

    def _start_autosave(self):
        self._stop_autosave()
        self._schedule_autosave()

    def _stop_autosave(self):
        if self._autosave_job is not None:
            try:
                self.app.root.after_cancel(self._autosave_job)
            except Exception:
                pass
            self._autosave_job = None

    def _schedule_autosave(self):
        self._autosave_job = self.app.root.after(
            AUTOSAVE_SECS * 1000, self._autosave_tick)

    def _autosave_tick(self):
        if self._project_path:
            # Save to a .autosave companion file
            autosave_path = self._project_path + '.autosave'
            try:
                state = self._collect_state()
                with open(autosave_path, 'wb') as f:
                    pickle.dump(state, f, protocol=5)
                self.app.status_label.config(
                    text=f'Auto-saved  {time.strftime("%H:%M")}',
                    foreground='gray')
            except Exception as e:
                print(f'  autosave failed: {e}')
        self._schedule_autosave()  # reschedule