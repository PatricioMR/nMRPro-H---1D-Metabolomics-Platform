# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/stages/annotation_stage.py
"""
Metabolites Annotation Stage — Step 3.

Algorithm
---------
1. Align spectrum to 0 ppm (DSS/TSP) using the same anchor as peak_alignment.
2. For each detected peak, query DB candidates within tolerance.
3. Score each candidate:
   a. PPM distance score (Gaussian, σ = tolerance/2)
   b. Multiplet pattern score: find surrounding peaks that form the expected
      multiplet pattern using J from DB, score position + relative intensity.
4. Rank by composite score = 0.4×ppm_score + 0.6×multiplet_score.
5. Open results window showing ranked candidates with fingerprint overlay.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
try:
    import certifi
except ImportError:
    certifi = None

if certifi is not None:
    os.environ['SSL_CERT_FILE']      = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from processing.signal_quality import SIGNAL_GOOD, normalize_tiers
from controllers.analysis_defaults import ANNOTATION
from processing.spectral_regions import WATER_REGION
from .base_stage import BaseStage
from ._annotation_search import _AnnotationSearchMixin


class AnnotationStage(BaseStage, _AnnotationSearchMixin):

    def __init__(self, parent, app, peak_stage):
        super().__init__(parent, app)
        self.peak_stage         = peak_stage
        self.annotations        = {}
        self._last_results_args = None
        self._metabolites       = None   # live reference set by AnnotationResultsWindow
        self._quality_marks       = {}   # {(met_id, peak_rank): 'good'|'bad'} — persisted
        self._child_quality_marks = {}   # {(met_id, round(dp_ppm,3)): 'good'|'bad'} — persisted
        self._anchor_lines        = {}   # {(met_id, peak_rank): line_idx} — anchor line for quantification
        self._subline_quality_marks = {} # {(met_id, peak_rank, line_idx): 'good'|'bad'} — persisted
        self.mcr_r2_threshold   = 0.95   # used by annotation MCR deconvolution
        self._annotation_params = None   # pre-annotation params; None = use defaults
        self.create_widgets()

    def annotatable_tiers(self):
        """Signal-quality grades the analyst allows annotation to use.

        Configured in the Annotation Parameters dialog. Defaults to Good and
        Average — a Poor signal has an unreliable apex, so it is opt-in.
        """
        params = self._annotation_params or {}
        return normalize_tiers(params.get('annotatable_tiers'))

    # ── Status helpers ────────────────────────────────────────────────────────

    def _status(self, msg, color='blue'):
        try:
            self.app.status_label.config(text=msg, foreground=color)
            self.app.root.update_idletasks()
        except Exception:
            pass

    def _status_ok(self,  m): self._status(m, 'green')
    def _status_err(self, m): self._status(m, 'red')
    def _status_ready(self):  self._status('Ready', 'gray')

    @property
    def peaks(self):
        return self.peak_stage.peaks

    # ── Widgets ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        ttk.Label(self.frame, text='Metabolites Annotation',
                  font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Separator(self.frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=5)

        s3 = ttk.LabelFrame(self.frame, text='📡 Database Search', padding=5)
        s3.pack(fill=tk.X, padx=5, pady=5)

        tf = ttk.Frame(s3); tf.pack(fill=tk.X, pady=2)
        ttk.Label(tf, text='Tolerance (ppm):', width=18).pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=ANNOTATION['tolerance_ppm'])
        self._tolerance_entry = ttk.Entry(tf, textvariable=self.tolerance_var, width=7)
        self._tolerance_entry.pack(side=tk.LEFT, padx=2)

        sf = ttk.Frame(s3); sf.pack(fill=tk.X, pady=2)
        ttk.Label(sf, text='Spectrometer (MHz):', width=18).pack(side=tk.LEFT)
        self.spectrometer_var = tk.StringVar(
            value=f"{ANNOTATION['fallback_field_mhz']:g}")
        self._spectrometer_entry = ttk.Entry(sf, textvariable=self.spectrometer_var, width=7)
        self._spectrometer_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(sf, text='for J Hz→ppm',
                  font=('Arial', 7), foreground='gray').pack(
                      side=tk.LEFT, padx=4)

        # Water-suppression exclusion zone — standard 1H-NMR metabolomics
        # practice: the residual water region (~4.70–4.90 ppm) is dominated by
        # solvent-suppression artefacts and must not be annotated (Emwas 2019;
        # Chenomx/MetaboAnalyst exclusion lists).
        wf = ttk.Frame(s3); wf.pack(fill=tk.X, pady=2)
        self.exclude_water_var = tk.BooleanVar(value=True)
        self._exclude_water_cb = ttk.Checkbutton(
            wf, text='Exclude water region', variable=self.exclude_water_var)
        self._exclude_water_cb.pack(side=tk.LEFT)
        self.water_lo_var = tk.DoubleVar(value=WATER_REGION[0])
        self.water_hi_var = tk.DoubleVar(value=WATER_REGION[1])
        self._water_lo_entry = ttk.Entry(wf, textvariable=self.water_lo_var, width=5)
        self._water_lo_entry.pack(side=tk.LEFT, padx=(6, 1))
        ttk.Label(wf, text='–').pack(side=tk.LEFT)
        self._water_hi_entry = ttk.Entry(wf, textvariable=self.water_hi_var, width=5)
        self._water_hi_entry.pack(side=tk.LEFT, padx=1)
        ttk.Label(wf, text='ppm', font=('Arial', 7),
                  foreground='gray').pack(side=tk.LEFT)

        mf = ttk.Frame(s3); mf.pack(fill=tk.X, pady=2)
        ttk.Label(mf, text='Internal library:', width=18).pack(side=tk.LEFT)
        self.internal_library_mode_var = tk.StringVar(value='Metabolomics (default)')
        self.internal_library_mode = ttk.Combobox(
            mf,
            textvariable=self.internal_library_mode_var,
            state='readonly',
            width=24,
            values=['Metabolomics (default)', 'Lipidomics'])
        self.internal_library_mode.pack(side=tk.LEFT, padx=2)

        # Lipidomics solvent selector — chemical shifts (and the validity of the
        # Box 1 FA-composition indices) depend on the acquisition solvent, so the
        # solvent picks which cited library is used. Shown only in Lipidomics mode.
        from processing import lipidomics as _lp
        self._lp_solvent_keys = list(_lp.SOLVENT_LIBRARIES.keys())
        self.lipid_solvent_var = tk.StringVar(value=_lp.SOLVENT_LABELS[_lp.DEFAULT_SOLVENT])
        self._lp_mf = mf  # anchor for pack(after=…)
        self.lipid_solvent_frame = ttk.Frame(s3)
        ttk.Label(self.lipid_solvent_frame, text='Solvent:',
                  width=18).pack(side=tk.LEFT)
        self.lipid_solvent_combo = ttk.Combobox(
            self.lipid_solvent_frame,
            textvariable=self.lipid_solvent_var,
            state='readonly',
            width=36,
            values=[_lp.SOLVENT_LABELS[k] for k in self._lp_solvent_keys])
        self.lipid_solvent_combo.pack(side=tk.LEFT, padx=2)
        self.internal_library_mode_var.trace_add('write', self._on_lib_mode_change)

        bf = ttk.Frame(s3); bf.pack(fill=tk.X, pady=5)
        self.btn_search_internal = ttk.Button(
            bf, text='🔍 Search in Database',
            command=self.search_internal_db)
        self.btn_search_internal.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_search_online = ttk.Button(
            bf, text='📚 Internal Library',
            command=self.search_builtin_library)
        self.btn_search_online.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        mcr_bf = ttk.Frame(s3); mcr_bf.pack(fill=tk.X, pady=(0, 5))
        self.mcr_run_var = tk.BooleanVar(value=False)
        self._mcr_run_cb = ttk.Checkbutton(
            mcr_bf, text='🔬 Run MCR Deconvolution',
            variable=self.mcr_run_var)
        self._mcr_run_cb.pack(side=tk.LEFT, padx=2)
        # Deconvolution is not part of this release (app_paths.
        # DECONVOLUTION_ENABLED, as the JG-Deconv stage): shown, not offered.
        from app_paths import DECONVOLUTION_ENABLED
        if not DECONVOLUTION_ENABLED:
            self._mcr_run_cb.state(['disabled'])

        self.internal_status = ttk.Label(
            s3, text='Internal DB: ready', foreground='gray')
        self.internal_status.pack(anchor=tk.W, pady=2)
        self.hmdb_status = ttk.Label(
            s3, text='Internal DB: ready', foreground='gray')
        self.hmdb_status.pack(anchor=tk.W, pady=2)

        info_frame = ttk.LabelFrame(
            self.frame, text='📊 Current Signals', padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        self.peaks_info_label = ttk.Label(
            info_frame, text='No signals detected yet.', foreground='gray')
        self.peaks_info_label.pack(anchor=tk.W)
        ttk.Label(info_frame,
                  text='All detected signals are annotated regardless of quality.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        mf = ttk.LabelFrame(self.frame, text='🐭 Mouse Controls', padding=5)
        mf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(mf,
                  text='• Scroll: Zoom in/out\n'
                       '• Drag: Pan\n'
                       '• R: Reset view',
                  font=('Arial', 8), foreground='blue',
                  justify=tk.LEFT).pack(anchor=tk.W, padx=5)

    # ── Lipidomics solvent selector ────────────────────────────────────────────

    def _on_lib_mode_change(self, *_):
        """Show the solvent selector only in Lipidomics mode."""
        is_lipid = (self.internal_library_mode_var.get() == 'Lipidomics')
        if is_lipid:
            self.lipid_solvent_frame.pack(fill=tk.X, pady=2, after=self._lp_mf)
        else:
            self.lipid_solvent_frame.pack_forget()
        self._apply_lipidomics_ui_state()
        # Refresh nav-bar availability (JG-Deconv blocked in lipidomics).
        _upd = getattr(self.app, '_update_stage_availability', None)
        if callable(_upd):
            try: _upd()
            except Exception: pass

    _LIPID_DISABLED_MSG = (
        'Desativado em modo Lipidomica:\n'
        'integração por região do espectro — sem deteção/correspondência de\n'
        'picos, acoplamento J, deconvolução MCR ou base de dados externa.')

    def _attach_lipid_tooltip(self, widget):
        """Tooltip shown only while the widget is disabled, explaining the
        lipidomics restriction (disabled ttk widgets give no other feedback)."""
        def _show(_e=None):
            try:
                if str(widget.cget('state')) != 'disabled':
                    return
            except Exception:
                return
            self._hide_lipid_tooltip()
            try:
                tw = tk.Toplevel(widget)
                tw.wm_overrideredirect(True)
                x = widget.winfo_rootx() + 10
                y = widget.winfo_rooty() + widget.winfo_height() + 2
                tw.wm_geometry(f'+{x}+{y}')
                tk.Label(tw, text=self._LIPID_DISABLED_MSG, justify=tk.LEFT,
                         background='#ffffe0', relief=tk.SOLID, borderwidth=1,
                         font=('Arial', 8), padx=4, pady=2).pack()
                self._lipid_tooltip_win = tw
            except Exception:
                self._lipid_tooltip_win = None
        widget.bind('<Enter>', _show, add='+')
        widget.bind('<Leave>', self._hide_lipid_tooltip, add='+')

    def _hide_lipid_tooltip(self, _e=None):
        tw = getattr(self, '_lipid_tooltip_win', None)
        if tw is not None:
            try: tw.destroy()
            except Exception: pass
            self._lipid_tooltip_win = None

    def _apply_lipidomics_ui_state(self):
        """Disable peak/J/MCR/external-DB options that do not apply to the
        region-integration lipidomics workflow."""
        is_lipid = (self.internal_library_mode_var.get() == 'Lipidomics')
        targets = [getattr(self, n, None) for n in (
            '_tolerance_entry', '_spectrometer_entry', '_exclude_water_cb',
            '_water_lo_entry', '_water_hi_entry', '_mcr_run_cb',
            'btn_search_internal')]
        if not getattr(self, '_lipid_tips_attached', False):
            for w in targets:
                if w is not None:
                    self._attach_lipid_tooltip(w)
            self._lipid_tips_attached = True
        for w in targets:
            if w is not None:
                w.configure(state='disabled' if is_lipid else 'normal')

    def lipid_solvent(self):
        """Return the selected solvent key ('D2O' / 'CDCl3')."""
        from processing import lipidomics as _lp
        label = self.lipid_solvent_var.get()
        for k, v in _lp.SOLVENT_LABELS.items():
            if v == label:
                return k
        return _lp.DEFAULT_SOLVENT

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        self._apply_lipidomics_ui_state()
        n      = len(self.peaks)
        n_good = sum(1 for p in self.peaks
                     if p.get('status') == SIGNAL_GOOD)
        if n == 0:
            self.peaks_info_label.config(
                text='No signals detected yet.', foreground='gray')
        elif n_good == 0:
            self.peaks_info_label.config(
                text=f'{n} signals — 0 marked Good. Mark in Signal List first.',
                foreground='orange')
        else:
            self.peaks_info_label.config(
                text=f'{n} signals loaded — {n_good} Good ({n - n_good} other quality).',
                foreground='blue')

    def on_deactivate(self):
        self.frame.pack_forget()

