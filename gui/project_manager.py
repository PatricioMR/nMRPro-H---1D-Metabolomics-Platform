# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/project_manager.py
"""
NMRAnalysis Project Manager.

Saves and restores the full application state to a .nmrproj file (format and
compression: data.project_file). Auto-saves every 5 minutes to a companion
file, in the background, and offers it back when it is newer.

Saved state:
  - EVERY variable of every stage and of the app (gui.project_vars), found by
    looking at them and not by a list: a parameter nobody listed would come
    back at its default and give other numbers. Opening a project says which
    it could not put back, and when the options now set differ from the ones
    the saved results were produced with.
  - Raw data (ppm, all_intensities, samples_info, group_colors, source file path)
  - Peak detection (noise_std, peaks list, overlapping regions)
  - Deconvolution (queued peaks, results)
  - Annotation stage (db_path, results)
  - UI state (active stage, axes settings, plot settings)
"""
import hashlib
import os
import pickle
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from processing.spectral_regions import (
    format_regions, parse_regions)

from app_paths import APP_VERSION, user_data_dir
from data import project_file
from gui.project_vars import (
    apply_vars, collect_vars, diff_vars, engine_fingerprint)

PROJECT_EXT     = '.nmrproj'
AUTOSAVE_SUFFIX = '.autosave'
AUTOSAVE_SECS   = 300  # 5 minutes


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
        self._dirty        = False  # a change content comparison cannot see
        self._autosave_job = None
        self._saved_digest     = None   # content of the project file
        self._saved_content    = None   # the same, without the stage on screen
        self._autosaved_digest = None   # content of the last autosave
        self._recovery_path    = None   # autosave of a session never saved
        self._writer        = None      # the autosave thread
        self._writer_result = None

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
            'version':         7,
            # What wrote it, so a project opened on another version can say so.
            'software':        {'version': APP_VERSION,
                                'engine': engine_fingerprint()},
            # Every variable of the app and of each stage, by name. One a stage
            # has not built yet is still the project's: its saved value waits
            # in _pending_vars and is saved again from there.
            'vars':            {name: {**(getattr(obj, '_pending_vars', None) or {}),
                                       **collect_vars(obj)}
                                for name, obj in {'app': app,
                                                  **app.stages}.items()},
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
            'region_alignments': list(getattr(app, 'region_alignments', [])),
            'import_reference': getattr(app, 'import_reference', None),
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
                'mcr_r2_threshold':            getattr(ann, 'mcr_r2_threshold', None),
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
                'auto_anchored':     set(getattr(jr, '_auto_anchored', set())),
                'cascade_estimates': dict(getattr(jr, '_cascade_estimates', {})),
                'nnls_cross_cv':     dict(getattr(jr, '_nnls_cross_cv', {})),
                'nnls_coverage':     dict(getattr(jr, '_nnls_coverage', {})),
                'calib_cv_thresh':   getattr(jr, '_calib_cv_thresh', None),
                'calib_cov_thresh':  getattr(jr, '_calib_cov_thresh', None),
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
                # The options the saved results were produced with: they are
                # not the options now set when the analyst changed one after
                # running.
                'last_run_settings': getattr(qnt, '_last_run_settings', None),
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
        app.region_alignments = list(state.get('region_alignments') or [])
        app.import_reference = state.get('import_reference')
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

                # Replaced even when empty, like the quantification results.
                ann._last_results_args = ann_state.get('last_results_args')
                if ann._last_results_args:
                    try: app.btn_annot_shortcut.config(state='normal')
                    except Exception: pass

                ann.annotations = ann_state.get('annotations') or {}

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
                if ann_state.get('mcr_r2_threshold') is not None:
                    ann.mcr_r2_threshold = ann_state['mcr_r2_threshold']

                ann._metabolites = ann_state.get('metabolites') or None
                if ann._metabolites:
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
                jr._auto_anchored = set(jr_state.get('auto_anchored', ()))
                jr._cascade_estimates = jr_state.get('cascade_estimates', {})
                jr._nnls_cross_cv = jr_state.get('nnls_cross_cv', {})
                jr._nnls_coverage = jr_state.get('nnls_coverage', {})
                if jr_state.get('calib_cv_thresh') is not None:
                    jr._calib_cv_thresh = jr_state['calib_cv_thresh']
                if jr_state.get('calib_cov_thresh') is not None:
                    jr._calib_cov_thresh = jr_state['calib_cov_thresh']
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
                # Always replaced: a project without results must not show the
                # ones of the project open before it.
                results = qnt_state.get('quant_results', [])
                qnt._quant_results = results
                if results:
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
                # Saved since v7 but never put back: the first save after
                # opening lost which options the results came from.
                qnt._last_run_settings = qnt_state.get('last_run_settings')
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
                try:
                    if results:
                        qnt._populate_tree()
                    elif getattr(qnt, '_tree', None) is not None:
                        qnt._tree.delete(*qnt._tree.get_children())
                except Exception:
                    pass
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

        # ── Every other variable, by name ─────────────────────────────────────
        self._restore_all_vars(state)

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


    def _restore_all_vars(self, state):
        """Put back every variable the project saved, and say what did not
        come back. A variable a stage has not built yet waits in the stage's
        ``_pending_vars`` and is set when the stage is opened."""
        app = self.app
        saved = state.get('vars') or {}
        targets = {'app': app, **app.stages}
        failed = {}
        for name, sv in saved.items():
            obj = targets.get(name)
            if obj is None:
                continue
            res = apply_vars(obj, sv)
            obj._pending_vars = {k: sv[k] for k in res['missing']}
            if res['failed']:
                failed[name] = res['failed']
        lines = []
        for stage, errs in failed.items():
            lines += [f'  {stage}.{k}: {v}' for k, v in errs.items()]
        # The options now set against the ones the saved results came from.
        qnt = app.stages.get('quantification')
        last = (state.get('qnt') or {}).get('last_run_settings')
        if qnt is not None and last and (state.get('qnt') or {}).get('quant_results'):
            changed = diff_vars(last, collect_vars(qnt))
            if changed:
                lines.append('')
                lines.append('The quantification options now set are not the ones '
                             'the saved results were produced with; running again '
                             'gives other numbers:')
                lines += [f'  {k}: {a!r} (results) -> {b!r} (now)'
                          for k, (a, b) in sorted(changed.items())]
        sw = state.get('software') or {}
        eng_now = engine_fingerprint()
        if (sw.get('version') and sw['version'] != APP_VERSION) or (
                sw.get('engine') and eng_now and sw['engine'] != eng_now):
            lines.append('')
            lines.append(
                f"Saved by {sw.get('version')} (engine {sw.get('engine')}); "
                f"this is {APP_VERSION} (engine {eng_now}). The saved results "
                "are the older engine's until they are run again.")
        if lines:
            try:
                messagebox.showwarning('Project', '\n'.join(
                    ['Opening this project:'] + [''] + lines))
            except Exception:
                pass
        return failed

    # ── File operations ───────────────────────────────────────────────────────

    def _snapshot(self):
        return project_file.snapshot(self._collect_state())

    @staticmethod
    def _content_digest(state):
        """The digest of what the analysis is. Which stage is on screen is
        saved but looking around is not a change, and the variables are
        compared on their own (``_vars_changed``). Each part is pickled apart:
        the bytes of one pickle depend on which objects two parts share, and
        that changes with what the windows did, not with the analysis."""
        h = hashlib.blake2b(digest_size=16)
        for k in sorted(state, key=str):
            if k in ('active_stage', 'vars'):
                continue
            v = state[k]
            parts = (sorted(v.items(), key=lambda kv: str(kv[0]))
                     if isinstance(v, dict) else [(None, v)])
            for sk, sv in parts:
                h.update(repr((k, sk)).encode())
                h.update(pickle.dumps(sv, protocol=5))
        return h.hexdigest()

    @staticmethod
    def _vars_changed(saved, now):
        """A variable that was saved and now reads otherwise. One a panel
        built after the save has its default and is not a change."""
        return any(k in now.get(name, {}) and now[name][k] != v
                   for name, sv in saved.items() for k, v in sv.items())

    def _mark_saved(self, snap, state):
        self._saved_digest = snap.digest
        self._saved_content = (self._content_digest(state),
                               dict(state.get('vars') or {}))

    def has_unsaved_changes(self):
        """Whether closing now would lose something: the session differs from
        the project file as last saved or opened. Compared by content, so
        running a stage counts without every stage having to say so."""
        if self._project_path is None and getattr(self.app, 'ppm', None) is None:
            return False
        if self._dirty or self._saved_content is None:
            return True
        try:
            state = self._collect_state()
            digest, saved_vars = self._saved_content
            return (self._content_digest(state) != digest
                    or self._vars_changed(saved_vars, state.get('vars') or {}))
        except Exception:
            return True

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
            self.app.status_label.config(text='Saving project…', foreground='blue')
            self.app.root.update_idletasks()
        except Exception:
            pass
        try:
            state = self._collect_state()
            snap = project_file.snapshot(state)
            size = project_file.write(path, snap)
        except Exception as e:
            messagebox.showerror('Save Error',
                                 f'Could not save project:\n{e}')
            return False
        # The project now holds what the autosaves held.
        self._discard_autosaves()
        self.project_path = path
        self._dirty        = False
        self._mark_saved(snap, state)
        self._start_autosave()
        self.app.root.after(
            0, lambda: self.app.status_label.config(
                text=f'Project saved: {os.path.basename(path)} '
                     f'({size / 1e6:.1f} MB)',
                foreground='green'))
        return True

    def open_project(self, path=None):
        """Open a project file, restoring full state."""
        if self.has_unsaved_changes():
            resp = messagebox.askyesnocancel(
                'Unsaved Changes',
                'The current project has unsaved changes.\n'
                'Save them before opening another project?\n\n'
                '  Yes    → save and continue\n'
                '  No     → open without saving (the changes are lost)\n'
                '  Cancel → go back')
            if resp is None:
                return False
            if resp and not self.save_or_saveas():
                return False

        if path is None:
            path = filedialog.askopenfilename(
                title='Open Project',
                filetypes=[('NMR Project', f'*{PROJECT_EXT}'),
                           ('Autosave', f'*{AUTOSAVE_SUFFIX}'),
                           ('All files', '*.*')])
            if not path:
                return False

        # Opening an autosave directly is recovering it into its project.
        project = (path[:-len(AUTOSAVE_SUFFIX)]
                   if path.endswith(AUTOSAVE_SUFFIX) else path)
        source = path
        if source == project:
            auto = project + AUTOSAVE_SUFFIX
            if (os.path.exists(auto)
                    and os.path.getmtime(auto) > os.path.getmtime(project)):
                resp = messagebox.askyesnocancel(
                    'Autosave found',
                    f'{os.path.basename(project)} has an autosave newer than '
                    f'the saved project:\n\n'
                    f'  saved     {_when(project)}\n'
                    f'  autosave  {_when(auto)}\n\n'
                    '  Yes    → open the autosave (save to keep it in the project)\n'
                    '  No     → open the project as last saved\n'
                    '  Cancel → go back')
                if resp is None:
                    return False
                if resp:
                    source = auto
        recovered = source != project
        return self._load(source, project_path=(
            project if os.path.exists(project) or recovered else None),
            recovered=recovered)

    def _load(self, source, *, project_path, recovered):
        try:
            self.app.status_label.config(
                text='Loading project…', foreground='blue')
            self.app.root.update_idletasks()
            state = project_file.read(source)
            # The session being left was saved or declined a moment ago; the
            # autosave being recovered stays until the project is saved.
            self._discard_autosaves(keep=source)
            self._restore_state(state)
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror('Open Error',
                                 f'Could not open project:\n{e}')
            return False
        self.project_path = project_path
        self._dirty = recovered
        self._saved_digest = self._saved_content = None
        if not recovered:
            try:
                state = self._collect_state()
                self._mark_saved(project_file.snapshot(state), state)
            except Exception:
                pass
        if recovered:
            msg = (f'Recovered autosave of {os.path.basename(project_path)} — '
                   'save to keep it' if project_path else
                   'Recovered unsaved work — save it as a project to keep it')
            colour = 'darkorange'
        else:
            msg = f'Project opened: {os.path.basename(source)}'
            colour = 'green'
            if not project_file.is_current_format(source):
                msg += ' (older, larger format: saving converts it)'
        self.app.status_label.config(text=msg, foreground=colour)
        self._start_autosave()
        return True

    def save_or_saveas(self):
        """Save to current path, or Save-As if no project open."""
        if self._project_path:
            return self.save_project(self._project_path)
        return self.save_project()

    # ── Auto-save ─────────────────────────────────────────────────────────────
    # Every AUTOSAVE_SECS the session goes to a companion file, never over the
    # project: saving stays the analyst's checkpoint. A project saved by name
    # autosaves beside it (<name>.nmrproj.autosave), offered when it is opened
    # and newer; work never saved autosaves to the recovery folder, offered
    # when the application starts. The state is frozen on the GUI thread and
    # compressed and written on another, so the window does not stop.

    def start(self):
        """Begin autosaving and offer what a session that ended without
        saving left behind."""
        self._start_autosave()
        self.app.root.after(500, self.offer_recovery)

    def _autosave_target(self):
        if self._project_path:
            return self._project_path + AUTOSAVE_SUFFIX
        if getattr(self.app, 'ppm', None) is None:
            return None
        if self._recovery_path is None:
            self._recovery_path = os.path.join(
                recovery_dir(),
                f'untitled-{time.strftime("%Y%m%d-%H%M%S")}'
                f'{PROJECT_EXT}{AUTOSAVE_SUFFIX}')
        return self._recovery_path

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
        try:
            self.autosave_now()
        finally:
            self._schedule_autosave()

    def autosave_now(self):
        """Write the session to its autosave file in the background, when it
        differs from what is already on disk. Returns the writing thread, or
        None when there was nothing to write."""
        if self._writer is not None and self._writer.is_alive():
            return None
        target = self._autosave_target()
        if target is None:
            return None
        try:
            snap = self._snapshot()
        except Exception as e:
            self._autosave_failed(e)
            return None
        if snap.digest in (self._saved_digest, self._autosaved_digest):
            return None

        def work():
            try:
                size = project_file.write(target, snap, keep_backup=False)
                self._writer_result = (snap.digest, size, None)
            except Exception as e:
                self._writer_result = (snap.digest, 0, e)

        self._writer_result = None
        self._writer = threading.Thread(target=work, name='autosave', daemon=True)
        self._writer.start()
        self._poll_writer()
        return self._writer

    def _poll_writer(self):
        if self._writer is not None and self._writer.is_alive():
            self.app.root.after(200, self._poll_writer)
            return
        res, self._writer_result = self._writer_result, None
        if res is None:
            return
        digest, size, err = res
        if err is not None:
            self._autosave_failed(err)
            return
        self._autosaved_digest = digest
        try:
            self.app.status_label.config(
                text=f'Auto-saved {time.strftime("%H:%M")} ({size / 1e6:.1f} MB)',
                foreground='gray')
        except Exception:
            pass

    def _autosave_failed(self, err):
        print(f'  autosave failed: {err}')
        try:
            self.app.status_label.config(
                text=f'Autosave failed: {err}', foreground='red')
        except Exception:
            pass

    def _wait_for_writer(self):
        if self._writer is not None:
            self._writer.join()
            self._poll_writer()

    def _discard_autosaves(self, keep=None):
        """Remove the autosave of this session: called when the project file
        or an explicit choice of the analyst has superseded it."""
        self._wait_for_writer()
        paths = [self._recovery_path]
        if self._project_path:
            paths.append(self._project_path + AUTOSAVE_SUFFIX)
        for p in filter(None, paths):
            if keep and os.path.abspath(p) == os.path.abspath(keep):
                continue
            for f in (p, p + '.tmp'):
                try:
                    os.remove(f)
                except OSError:
                    pass
        self._recovery_path = None
        self._autosaved_digest = None

    def shutdown(self):
        """The session ends saved or with its changes declined: stop
        autosaving and leave no autosave behind to be offered again."""
        self._stop_autosave()
        self._discard_autosaves()

    def offer_recovery(self):
        """Offer the work of a session that ended without saving. The newest
        first; what is declined is deleted, what is postponed stays."""
        for path in recovery_files():
            resp = messagebox.askyesnocancel(
                'Recover unsaved work',
                'A session that was never saved as a project left this '
                f'autosave ({_when(path)}).\n\n'
                '  Yes    → open it (then save it as a project)\n'
                '  No     → delete it\n'
                '  Cancel → keep it for later')
            if resp is None:
                return
            if resp:
                if self._load(path, project_path=None, recovered=True):
                    self._recovery_path = path
                return
            try:
                os.remove(path)
            except OSError:
                pass


def recovery_dir():
    path = os.path.join(user_data_dir(), 'recovery')
    os.makedirs(path, exist_ok=True)
    return path


def recovery_files():
    """Autosaves of sessions never saved as a project, newest first."""
    d = recovery_dir()
    files = [os.path.join(d, f) for f in os.listdir(d)
             if f.endswith(AUTOSAVE_SUFFIX)]
    return sorted(files, key=os.path.getmtime, reverse=True)


def _when(path):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))

