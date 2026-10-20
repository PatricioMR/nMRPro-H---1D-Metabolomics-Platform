# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/stages/quantification_stage.py
"""
Quantification Stage — Step 4.

Left  : IS config, normalisation, MCR toggle, run, export
Right : PanedWindow (vertical)
  Top    : Treeview — analytes × quality columns (CV per group)
  Middle : Overlay plot (per-sample spectra, coloured by group)
           + bar chart (mean ± SD per group)
  Right  : Quality card

Integration: trapezoidal, per sample.
MCR option : simplified NMF-ALS per peak region.
CV         : calculated within each group independently.

Split into focused modules:
  _quant_helpers.py      — standalone helper functions + _STD_PRESETS
  _quant_visualization.py — _QuantVisMixin  (detail plots / tooltips)
  _quant_computation.py  — _QuantCompMixin  (_run, VIP, update, export)
"""
import tkinter as tk

from tkinter import ttk, filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from controllers.analysis_defaults import QUANTIFICATION, cascade_advice
from .base_stage import BaseStage
from ._quant_helpers import _STD_PRESETS
from ._quant_visualization import _QuantVisMixin
from ._quant_computation import _QuantCompMixin


class QuantificationStage(BaseStage, _QuantVisMixin, _QuantCompMixin):

    def __init__(self, parent, app, peak_stage, annotation_stage):
        super().__init__(parent, app)
        self.peak_stage       = peak_stage
        self.annotation_stage = annotation_stage
        self._quant_results   = []
        # User multiplet overrides (hmdb_id → anchor peak_rank). Persist across
        # re-runs (cleared only when annotation is re-run); never reset in _run().
        self._quant_peak_overrides = {}
        # What the analyst decided in the deconvolution review window, by
        # owner; cleared with the peak overrides when annotation re-runs.
        self._deconv_overrides = {}
        self._pqn_factor      = 1.0
        self._right_widget    = None
        self._detail_fig      = None
        self._detail_canvas   = None
        self._sort_col        = 'conc_mm'
        self._sort_rev        = True
        self._search_var      = None  # set in _build_right_panel
        self._placeholder     = None  # set in _build_right_panel
        self._tree            = None  # set in _build_right_panel
        self.create_widgets()

    def _is_lipidomics_mode(self):
        """True when the active annotation run used the Lipidomics library."""
        args = getattr(self.annotation_stage, '_last_results_args', None) or {}
        return str(args.get('internal_library_mode', '')) == 'Lipidomics'

    def concentration_unit(self):
        """Unit of the quantified values, matching the results-table header.

        IS / IS+PQN normalisation gives absolute concentrations in mM; 'Total'
        gives relative area percent; anything else is arbitrary area units.
        Lipidomics is semi-quantitative, so never mM."""
        _nm = getattr(self, 'norm_var', None)
        _norm = _nm.get() if _nm else 'IS'
        if self._is_lipidomics_mode():
            return '%' if _norm == 'Total' else 'a.u.'
        if _norm in ('IS', 'IS+PQN'):
            return 'mM'
        if _norm == 'Total':
            return '%'
        return 'a.u.'

    _LIPID_DISABLED_MSG = (
        'Desativado em modo Lipidomica:\n'
        'integração por região (trapz), semi-quantitativa — sem deteção de\n'
        'picos, deconvolução, ajuste de multipleto ou correção de protões.')

    def _attach_lipid_tooltip(self, widget):
        """Tooltip that appears only while the widget is disabled, explaining the
        lipidomics restriction (a disabled ttk widget gives no feedback)."""
        def _show(_e=None):
            try:
                if str(widget.cget('state')) != 'disabled':
                    return
            except Exception:
                return
            self._hide_dev_tooltip()
            try:
                tw = tk.Toplevel(widget)
                tw.wm_overrideredirect(True)
                x = widget.winfo_rootx() + 10
                y = widget.winfo_rooty() + widget.winfo_height() + 2
                tw.wm_geometry(f'+{x}+{y}')
                tk.Label(tw, text=self._LIPID_DISABLED_MSG, justify=tk.LEFT,
                         background='#ffffe0', relief=tk.SOLID, borderwidth=1,
                         font=('Arial', 8), padx=4, pady=2).pack()
                self._dev_tooltip_win = tw
            except Exception:
                self._dev_tooltip_win = None
        widget.bind('<Enter>', _show, add='+')
        widget.bind('<Leave>', self._hide_dev_tooltip, add='+')

    def _tip(self, widget, text):
        """A hover tooltip: the long explanations live here, not in the
        panel, so the menu fits."""
        def _show(_e=None):
            self._hide_dev_tooltip()
            try:
                tw = tk.Toplevel(widget)
                tw.wm_overrideredirect(True)
                tw.wm_geometry(f'+{widget.winfo_rootx() + 10}'
                               f'+{widget.winfo_rooty() + widget.winfo_height() + 2}')
                tk.Label(tw, text=text, justify=tk.LEFT, wraplength=340,
                         background='#ffffe0', relief=tk.SOLID, borderwidth=1,
                         font=('Arial', 8), padx=4, pady=2).pack()
                self._dev_tooltip_win = tw
            except Exception:
                self._dev_tooltip_win = None
        widget.bind('<Enter>', _show, add='+')
        widget.bind('<Leave>', self._hide_dev_tooltip, add='+')

    def _baseline_choice_of(self, local, model) -> str:
        for name, pair in self._baseline_choices.items():
            if pair == (bool(local), bool(model)):
                return name
        return 'None'

    def _baseline_from_choice(self, *_):
        local, model = self._baseline_choices.get(
            self.baseline_choice_var.get(), (False, False))
        if self.local_baseline_var.get() != local:
            self.local_baseline_var.set(local)
        if self.bl_model_quant_var.get() != model:
            self.bl_model_quant_var.set(model)

    def _baseline_to_choice(self, *_):
        name = self._baseline_choice_of(self.local_baseline_var.get(),
                                        self.bl_model_quant_var.get())
        if self.baseline_choice_var.get() != name:
            self.baseline_choice_var.set(name)

    def _apply_lipidomics_ui_state(self):
        """Disable options that do not apply to lipidomics region integration
        (multiplet integration, proton correction, JG-Deconv overlap)."""
        is_lipid = self._is_lipidomics_mode()
        targets = [self._integ_rbs.get('Multiplet'),
                   getattr(self, '_proton_corr_cb', None),
                   getattr(self, '_shape_windows_cb', None)]
        if not getattr(self, '_lipid_tips_attached', False):
            for w in targets:
                if w is not None:
                    self._attach_lipid_tooltip(w)
            self._lipid_tips_attached = True
        for w in targets:
            if w is None:
                continue
            w.configure(state='disabled' if is_lipid else 'normal')
        # In lipidomics the integration mode is forced to region area; point the
        # selector at a sane value so the disabled multiplet radio is not active.
        if is_lipid and self.integ_var.get() == 'Multiplet':
            self.integ_var.set('Area')

    # ── Status helpers ────────────────────────────────────────────────────────

    def _status(self, msg, color='blue'):
        try:
            self.app.status_label.config(text=msg, foreground=color)
            self.app.root.update_idletasks()
        except Exception:
            pass

    def _status_ok(self,  m): self._status(m, 'green')
    def _status_err(self, m): self._status(m, 'red')

    def _find_peak(self, ppm_target, tol=0.01):
        best, best_d = None, tol
        for p in (self.peak_stage.peaks or []):
            d = abs(p['ppm'] - ppm_target)
            if d < best_d:
                best, best_d = p, d
        return best

    # ── Left panel ────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)
        ttk.Label(self.frame, text='Quantification',
                  font=('Arial', 12, 'bold')).pack(pady=8)
        ttk.Separator(self.frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=4)

        cv_canvas = tk.Canvas(self.frame, borderwidth=0, highlightthickness=0)
        sb = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=cv_canvas.yview)
        sf = ttk.Frame(cv_canvas)
        _sf_win = cv_canvas.create_window((0, 0), window=sf, anchor='nw')

        def _on_sf_configure(e):
            cv_canvas.configure(scrollregion=cv_canvas.bbox('all'))
        def _on_canvas_resize(e):
            cv_canvas.itemconfig(_sf_win, width=e.width)

        sf.bind('<Configure>', _on_sf_configure)
        cv_canvas.bind('<Configure>', _on_canvas_resize)
        cv_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_mousewheel(event):
            cv_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        def _bind_mousewheel(widget):
            widget.bind('<MouseWheel>', _on_mousewheel)
            for child in widget.winfo_children():
                _bind_mousewheel(child)
        sf.bind('<Map>', lambda e: _bind_mousewheel(sf))

        # 1. Reference Signal
        s1 = ttk.LabelFrame(sf, text='1 · Reference Signal (IS)', padding=6)
        s1.pack(fill=tk.X, padx=6, pady=4)

        r0 = ttk.Frame(s1); r0.pack(fill=tk.X, pady=1)
        ttk.Label(r0, text='Standard:', width=14).pack(side=tk.LEFT)
        self.ref_std_var = tk.StringVar(value='TSP')
        cb = ttk.Combobox(r0, textvariable=self.ref_std_var,
                          values=list(_STD_PRESETS.keys()),
                          state='readonly', width=12)
        cb.pack(side=tk.LEFT, padx=2)
        cb.bind('<<ComboboxSelected>>', self._on_std_select)

        r2 = ttk.Frame(s1); r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text='Signal δ (ppm):', width=20).pack(side=tk.LEFT)
        self.ref_ppm_var = tk.StringVar(value='0.0')
        ttk.Entry(r2, textvariable=self.ref_ppm_var, width=7).pack(
            side=tk.LEFT, padx=2)
        ttk.Label(r2, text='±0.05 ppm', font=('Arial', 7),
                  foreground='gray').pack(side=tk.LEFT)

        r1 = ttk.Frame(s1); r1.pack(fill=tk.X, pady=1)
        ttk.Label(r1, text='— or Signal #:', width=20).pack(side=tk.LEFT)
        self.ref_rank_var = tk.StringVar(value='')
        ttk.Entry(r1, textvariable=self.ref_rank_var, width=6).pack(
            side=tk.LEFT, padx=2)

        r3 = ttk.Frame(s1); r3.pack(fill=tk.X, pady=1)
        ttk.Label(r3, text='Concentration (mM):', width=20).pack(side=tk.LEFT)
        self.ref_conc_var = tk.StringVar(value='1.0')
        ttk.Entry(r3, textvariable=self.ref_conc_var, width=7).pack(
            side=tk.LEFT, padx=2)

        r4 = ttk.Frame(s1); r4.pack(fill=tk.X, pady=1)
        ttk.Label(r4, text='nH of IS:', width=20).pack(side=tk.LEFT)
        self.ref_nh_var = tk.StringVar(value='9')
        ttk.Entry(r4, textvariable=self.ref_nh_var, width=7).pack(
            side=tk.LEFT, padx=2)

        ttk.Label(s1,
                  text='TSP/DSS/TMSP: 0.00 ppm | nH=9\n'
                       'Maleic acid: 6.00 ppm | nH=2\n'
                       'Formate: 8.46 ppm | nH=1',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        ttk.Separator(s1, orient='horizontal').pack(fill=tk.X, pady=4)
        ext_row = ttk.Frame(s1); ext_row.pack(fill=tk.X, pady=1)
        self.ext_is_var = tk.BooleanVar(value=False)
        self._ext_is_cb = ttk.Checkbutton(ext_row, text='External IS',
                                           variable=self.ext_is_var)
        self._ext_is_cb.pack(side=tk.LEFT)
        ttk.Button(ext_row, text='Select samples…',
                   command=self._open_ext_is_dialog).pack(side=tk.LEFT, padx=6)
        self._ext_is_label = ttk.Label(ext_row, text='(none selected)',
                                        font=('Arial', 7), foreground='gray')
        self._ext_is_label.pack(side=tk.LEFT)
        ttk.Label(s1,
                  text='External IS: uses IS area from selected samples\n'
                       '(e.g. BLK) as fixed reference for all analytical samples.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # 2. Integration
        s2 = ttk.LabelFrame(sf, text='2 · Integration', padding=6)
        s2.pack(fill=tk.X, padx=6, pady=4)

        mf = ttk.Frame(s2); mf.pack(fill=tk.X, pady=1)
        # One integration mode: the whole multiplet, divided by its own proton
        # count. Single-Line Area was shown disabled for months, which is a
        # promise the software was not keeping; it comes back with the engine
        # that will implement it. The variable stays because the saved projects
        # and the result rows carry the mode, and dropping it would rewrite
        # what an old project says it did.
        ttk.Label(mf, text='Signal: whole multiplet ÷ nH').pack(side=tk.LEFT)
        self.integ_var = tk.StringVar(value=QUANTIFICATION['integration'])
        self._integ_rbs = {}

        # Always on: against an internal or external standard the division
        # by nH is not optional (quant_concentration.concentrations), so
        # there is no box. Kept as a variable for the code that reads it.
        self.proton_corr_var = tk.BooleanVar(value=QUANTIFICATION['proton_correction'])

        # What the areas are measured on. Two choices that used to be two
        # boxes: a linear baseline under each window, and the project's
        # baseline model (the one detection always uses). Measured on the
        # GISSMO bench, whose baselines are flat, either one only costs:
        # the default is neither.
        self.local_baseline_var = tk.BooleanVar(
            value=QUANTIFICATION['local_baseline'])
        self.bl_model_quant_var = tk.BooleanVar(
            value=QUANTIFICATION['baseline_model'])
        self._baseline_choices = {
            'None': (False, False), 'Local per window': (True, False),
            'Baseline model': (False, True), 'Local + model': (True, True)}
        self.baseline_choice_var = tk.StringVar(
            value=self._baseline_choice_of(self.local_baseline_var.get(),
                                           self.bl_model_quant_var.get()))
        blf = ttk.Frame(s2); blf.pack(fill=tk.X, pady=1)
        ttk.Label(blf, text='Baseline:', font=('Arial', 8),
                  width=10).pack(side=tk.LEFT)
        _bl_cb = ttk.Combobox(blf, textvariable=self.baseline_choice_var,
                              values=list(self._baseline_choices), width=18,
                              state='readonly')
        _bl_cb.pack(side=tk.LEFT)
        self._tip(_bl_cb, 'What the areas are measured on.\n'
                          'None: the spectrum as it is (default).\n'
                          'Local per window: a straight line under each window, '
                          'read beyond the cluster of overlapping signals.\n'
                          'Baseline model: the project baseline model that '
                          'detection already uses.\n'
                          'Only worth turning on where the spectrum has a real '
                          'baseline; on a flat one each costs signal.')
        self.baseline_choice_var.trace_add('write', self._baseline_from_choice)
        self.local_baseline_var.trace_add('write', self._baseline_to_choice)
        self.bl_model_quant_var.trace_add('write', self._baseline_to_choice)

        mf2 = ttk.Frame(s2); mf2.pack(fill=tk.X, pady=1)
        self._ext_is_indices = []
        self.mcr_var = tk.BooleanVar(value=False)   # kept for internal MCR-MCR path
        # Group A -- one window resolved by the shapes of what sits in it:
        # the balance of a window two winners contest (split_shared,
        # ADR-010/011), a window holding another accepted analyte's signal
        # (resolve_shared, ADR-015), and complex multiplets judged by their
        # shape (quantify_shapes, ADR-013). One question for the analyst, and
        # one switch: an m* not confirmed line by line is only offered to the
        # remainders the shared-window split leaves, so m* on with the split
        # off lost most of them without a word. The three variables stay, as
        # the engine, the saved projects and the report read them.
        self.deconv_var = tk.BooleanVar(value=QUANTIFICATION['split_shared'])
        self.shape_var = tk.BooleanVar(value=QUANTIFICATION['quantify_shapes'])
        self.shared_var = tk.BooleanVar(value=QUANTIFICATION['resolve_shared'])
        self.shape_windows_var = tk.BooleanVar(value=self._shape_parts_on())
        self._syncing_shape_windows = False
        self._shape_windows_cb = ttk.Checkbutton(
            mf2, text='Resolve windows by shapes',
            variable=self.shape_windows_var,
            command=self._on_deconv_change)
        self._j_resolve_status_lbl = ttk.Label(
            mf2, text='(not run yet)', font=('Arial', 7), foreground='gray')
        self._shape_windows_cb.pack(side=tk.LEFT)
        self._j_resolve_status_lbl.pack(side=tk.LEFT, padx=4)
        self._tip(self._shape_windows_cb,
                  'On: a window two analytes sit in is divided between them '
                  'by their shapes (contested windows and windows holding '
                  "another analyte's signal), and complex multiplets (m*) "
                  'are identified and measured by their shape.\n'
                  "Off: every window is its owner's whole and complex "
                  'multiplets are left out, saying so.')
        self.shape_windows_var.trace_add(
            'write', lambda *_: self._sync_shape_windows(source=self.shape_windows_var))
        for _v in (self.deconv_var, self.shape_var, self.shared_var):
            _v.trace_add('write', lambda *_, v=_v: self._sync_shape_windows(source=v))
        # A singlet on its own: position is its whole evidence, and the
        # signal-to-noise it must reach is the study's call (ADR-025): ten
        # suited a synthetic bench, five an organoid study whose real
        # singlets stood at 5-10.
        mf5 = ttk.Frame(s2); mf5.pack(fill=tk.X, pady=1)
        _l = ttk.Label(mf5, text='  Min SNR, lone singlet:', font=('Arial', 8),
                       width=26)
        _l.pack(side=tk.LEFT)
        self.singlet_snr_var = tk.DoubleVar(value=QUANTIFICATION['singlet_min_snr'])
        ttk.Spinbox(mf5, from_=0.0, to=50.0, increment=1.0, width=6,
                    textvariable=self.singlet_snr_var).pack(side=tk.LEFT)
        self._tip(_l, 'Below this signal-to-noise, a singlet with no other '
                      'signal seen is not quantified (a Good mark overrides).')
        # Overlapping analytes measured together (processing.quant_joint_fit):
        # a window counts its neighbours' area as its own; the fit divides
        # the region by the analytes' shapes, with room for peaks nothing
        # in the library explains. Slower on a crowded spectrum.
        jff = ttk.Frame(s2); jff.pack(fill=tk.X, pady=1)
        self.joint_fit_var = tk.BooleanVar(value=QUANTIFICATION['joint_fit'])
        _c = ttk.Checkbutton(jff, text='Joint fit (overlaps)',
                             variable=self.joint_fit_var)
        _c.pack(side=tk.LEFT)
        self._tip(_c, 'On: analytes whose signals overlap are fitted '
                      'together, sample by sample, and the area of the '
                      'region is divided between their shapes. Peaks that '
                      'nothing in the library explains are kept apart so '
                      'they do not inflate a known analyte.\n'
                      'Analytes that overlap nothing keep their integral.\n'
                      'Slower on a crowded spectrum.')
        # The cascade (ADR-028): after the election, every quantified
        # analyte is subtracted at its measured amount and the analytes
        # whose whole listing best explains what is left are added, one
        # amount for all their signals. A second opinion that only adds.
        mf6 = ttk.Frame(s2); mf6.pack(fill=tk.X, pady=1)
        self.cascade_var = tk.BooleanVar(value=QUANTIFICATION['cascade'])
        _c = ttk.Checkbutton(mf6, text='Cascade', variable=self.cascade_var)
        _c.pack(side=tk.LEFT)
        self._tip(_c, 'Adds analytes the election missed: the quantified '
                      'analytes are subtracted and the residual is explained '
                      'jointly, one amount per analyte.\n'
                      'It only adds, at two dots at most. On the validation '
                      'mixtures it finds two to three times more compounds, '
                      'with more false rows.')
        self.cascade_advice_label = ttk.Label(
            s2, text='', font=('Arial', 8), foreground='#b26a00',
            wraplength=360, justify=tk.LEFT)
        self.cascade_advice_label.pack(fill=tk.X, padx=24)
        self.cascade_var.trace_add('write',
                                   lambda *_: self._refresh_cascade_advice())
        # Kept for the projects saved with it: the cascade's identification
        # is decided by its score now (quant_cascade.cascade_score), and this
        # no longer reaches the engine.
        self.max_disagreement_var = tk.DoubleVar(
            value=QUANTIFICATION['max_disagreement_pct'])
        ttk.Button(s2, text='Review deconvolution…',
                   command=lambda: self._open_deconvolution_window()).pack(
                       anchor=tk.W, padx=4, pady=(1, 3))

        ttk.Separator(s2, orient='horizontal').pack(fill=tk.X, pady=(6, 2))
        ic_row0 = ttk.Frame(s2); ic_row0.pack(fill=tk.X, pady=1)
        self.icoshift_var = tk.BooleanVar(value=QUANTIFICATION['align'])
        ttk.Checkbutton(ic_row0, text='Pre-align with icoshift before quantification',
                        variable=self.icoshift_var).pack(side=tk.LEFT)
        ic_row1 = ttk.Frame(s2); ic_row1.pack(fill=tk.X, pady=1)
        ttk.Label(ic_row1, text='  Max shift (ppm):', font=('Arial', 8),
                  width=20).pack(side=tk.LEFT)
        self.icoshift_max_shift_var = tk.DoubleVar(value=0.02)
        ttk.Spinbox(ic_row1, from_=0.003, to=0.10, increment=0.003, width=7,
                    textvariable=self.icoshift_max_shift_var,
                    format='%.3f').pack(side=tk.LEFT, padx=2)
        ttk.Label(s2,
                  text='Aligns each sample to mean spectrum (icoshift)\n'
                       'before integration. Quantification only — raw data unchanged.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # 3. Normalisation
        s4 = ttk.LabelFrame(sf, text='3 · Normalisation', padding=6)
        s4.pack(fill=tk.X, padx=6, pady=4)

        self.norm_var = tk.StringVar(value=QUANTIFICATION['normalisation'])
        for lbl, val in [('Internal std (IS)',    'IS'),
                         ('IS + PQN',             'IS+PQN'),
                         ('PQN',                  'PQN'),
                         ('Total area %',         'Total'),
                         ('None',                 'None')]:
            ttk.Radiobutton(s4, text=lbl, variable=self.norm_var,
                            value=val).pack(anchor=tk.W)
        ttk.Separator(s4, orient='horizontal').pack(fill=tk.X, pady=4)
        self.blk_sub_var = tk.BooleanVar(value=QUANTIFICATION['subtract_blanks'])
        ttk.Checkbutton(s4, text='Subtract BLK mean (blank correction)',
                        variable=self.blk_sub_var).pack(anchor=tk.W)
        ttk.Label(s4, text='Removes mean BLK area from each\nsample before quantification.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # Run
        ttk.Separator(sf, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=6)
        self.btn_run = ttk.Button(
            sf, text='⚗ Run Quantification',
            command=self._safe_run, state='disabled')
        self.btn_run.pack(fill=tk.X, padx=6, pady=2)
        self.run_status = ttk.Label(
            sf, text='Run Annotation first',
            foreground='gray', font=('Arial', 8))
        self.run_status.pack(pady=2)

        ttk.Separator(sf, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=4)

        # Export
        ttk.Label(sf, text='Export',
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=6)
        self.btn_csv = ttk.Button(
            sf, text='📋 Export CSV',
            command=lambda: self._export('csv'), state='disabled')
        self.btn_csv.pack(fill=tk.X, padx=6, pady=2)
        self.btn_metaboanalyst = ttk.Button(
            sf, text='📊 Export MetaboAnalyst',
            command=lambda: self._export('metaboanalyst'), state='disabled')
        self.btn_metaboanalyst.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(sf,
                  text='CSV: Sample × Group × Metabolite (mM)\n'
                       'MetaboAnalyst: samples × metabolites',
                  font=('Arial', 7), foreground='gray').pack(
                      anchor=tk.W, padx=6)

        ttk.Separator(sf, orient='horizontal').pack(fill='x', padx=6, pady=4)
        self.btn_pdf = ttk.Button(
            sf, text='📄 Export PDF Report',
            command=self._export_pdf, state='disabled')
        self.btn_pdf.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(sf, text='Technical report: spectrum + per-analyte\ncrop + quantification parameters',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W, padx=6)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        has_ann = (hasattr(self.annotation_stage, '_last_results_args')
                   and bool(self.annotation_stage._last_results_args))
        _fs_stage   = (getattr(self.app, 'stages', {}) or {}).get('fullspectrum_mcr')
        has_fs      = bool(getattr(_fs_stage, '_mcr_regions', None))
        ready       = has_ann or has_fs
        self.btn_run.config(state='normal' if ready else 'disabled')
        self.run_status.config(
            text='Ready — configure and run' if ready
                 else 'Run Full Spectrum MCR or Annotation first',
            foreground='blue' if ready else 'gray')
        self._hide_canvas()
        self._show_right_panel()
        self._refresh_deconv_label()
        self._apply_lipidomics_ui_state()
        self._refresh_cascade_advice()

    def _refresh_cascade_advice(self):
        """Say next to the cascade checkbox when the spectrum is too sparse
        for it (controllers.analysis_defaults.cascade_advice)."""
        n = len(getattr(self.peak_stage, 'peaks', None) or ())
        self.cascade_advice_label.config(
            text=cascade_advice(n, enabled=bool(self.cascade_var.get())))

    def on_deactivate(self):
        if self.frame:
            self.frame.pack_forget()
        self._restore_canvas()

    def _hide_canvas(self):
        try:
            self.app.canvas.get_tk_widget().pack_forget()
        except Exception:
            pass
        try:
            self.app.toolbar.pack_forget()
        except Exception:
            pass

    # ── Progress dialog ───────────────────────────────────────────────────────

    def _build_progress_dialog(self):
        _root = self.app.root
        win = tk.Toplevel(_root)
        win.title('Quantification')
        win.resizable(False, False)
        win.protocol('WM_DELETE_WINDOW', lambda: None)
        win.transient(_root)

        tk.Label(win, text='Running quantification…',
                 font=('Arial', 10, 'bold')).pack(pady=(18, 4), padx=28)

        self._prog_step_var = tk.StringVar(value='Initializing…')
        tk.Label(win, textvariable=self._prog_step_var,
                 font=('Arial', 9), foreground='#555555').pack(padx=28)

        self._prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(win, variable=self._prog_var, maximum=100.0,
                        length=340, mode='determinate').pack(
            pady=(10, 4), padx=28)

        self._prog_pct_var = tk.StringVar(value='0%')
        tk.Label(win, textvariable=self._prog_pct_var,
                 font=('Arial', 8), foreground='#888888').pack(pady=(0, 18))

        win.update_idletasks()
        pw = win.winfo_reqwidth()
        ph = win.winfo_reqheight()
        rx = _root.winfo_rootx() + (_root.winfo_width()  - pw) // 2
        ry = _root.winfo_rooty() + (_root.winfo_height() - ph) // 2
        win.geometry(f'+{max(rx, 0)}+{max(ry, 0)}')

        win.update()
        win.grab_set()
        win.lift()

        self._prog_win = win

    def _prog_update(self, pct, step=''):
        """Thread-safe: schedule on main thread via app.root.after(0, ...)."""
        self.app.root.after(0, lambda p=float(pct), s=str(step): self._prog_apply(p, s))

    def _prog_apply(self, pct, step):
        try:
            self._prog_var.set(pct)
            self._prog_pct_var.set(f'{int(pct)}%')
            if step:
                self._prog_step_var.set(step)
        except Exception:
            pass

    def _close_progress_dialog(self):
        try:
            if getattr(self, '_prog_win', None):
                self._prog_win.grab_release()
                self._prog_win.destroy()
                self._prog_win = None
        except Exception:
            pass

    def _restore_canvas(self):
        if self._right_widget:
            try: self._right_widget.pack_forget()
            except Exception: pass
        try:
            self.app.toolbar.pack_forget()
        except Exception:
            pass
        try:
            self.app.canvas.get_tk_widget().pack_forget()
        except Exception:
            pass
        try:
            self.app.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        except Exception:
            pass
        try:
            self.app.canvas.get_tk_widget().pack(
                side=tk.TOP, fill=tk.BOTH, expand=True)
        except Exception:
            pass

    # ── Right panel ───────────────────────────────────────────────────────────

    def _show_right_panel(self):
        if self._right_widget is None:
            self._build_right_panel()
        self._right_widget.pack(fill=tk.BOTH, expand=True)
        if self._quant_results:
            self._populate_tree()
        else:
            self._show_placeholder()

    def _build_right_panel(self):
        self._right_widget = ttk.Frame(self.app.right_panel)

        paned = tk.PanedWindow(self._right_widget, orient=tk.VERTICAL,
                               sashwidth=6, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        # Top: Treeview
        top = ttk.Frame(paned)
        paned.add(top, minsize=240)

        sf2 = ttk.Frame(top)
        sf2.pack(fill=tk.X, padx=2, pady=(2, 0))
        ttk.Label(sf2, text='🔍', font=('Arial', 9)).pack(side=tk.LEFT)
        self._search_var = tk.StringVar()
        self._search_var.trace_add('write', lambda *_: self._populate_tree())
        ttk.Entry(sf2, textvariable=self._search_var,
                  width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(sf2, text='✕', width=2,
                   command=lambda: self._search_var.set('')
                   ).pack(side=tk.LEFT)

        cols = ('peak_num', 'metabolite', 'ppm', 'conc_mm',
                'raw_area', 'snr', 'vip_score', 'status')
        self._tree = ttk.Treeview(top, columns=cols, show='headings',
                                   selectmode='browse')
        for cid, hd, w in [
            ('peak_num',    '#',               35),
            ('metabolite',  'Metabolite',     210),
            ('ppm',         'PPM',             70),
            ('conc_mm',     'Mean Conc (mM)', 110),
            ('raw_area',    'Mean Area',        90),
            ('snr',         'SNR (group)',  80),
            ('vip_score',   'VIP',              55),
            ('status',      'Quality',          70),
        ]:
            self._tree.heading(cid, text=hd,
                               command=lambda c=cid: self._sort_tree(c))
            self._tree.column(cid, width=w, minwidth=40)

        self._tree.tag_configure('good', background='#c8f7c5')
        self._tree.tag_configure('warn', background='#fef9c3')
        self._tree.tag_configure('poor', background='#ffd6d6')
        self._tree.tag_configure('conf3', background='#c8f7c5', foreground='#155724')
        self._tree.tag_configure('conf2', background='#fef9c3', foreground='#856404')
        self._tree.tag_configure('conf1', background='#ffd6d6', foreground='#721c24')
        # Não-quantificados: cinzento, sem destaque. Razão no tooltip da coluna Conf.
        self._tree.tag_configure('nq', background='#eeeeee', foreground='#9a9a9a')

        vsb = ttk.Scrollbar(top, orient=tk.VERTICAL,
                             command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree.bind('<Motion>',  self._on_tree_motion)
        self._tree.bind('<Leave>',   self._hide_tooltip)
        self._tree.bind('<Button-3>', self._on_tree_right_click)
        self._tooltip_win  = None
        self._tooltip_iid  = None

        self._placeholder = ttk.Label(
            top, text='Run Quantification to see results.',
            font=('Arial', 11), foreground='gray')

        # Bottom: horizontal split — left (spec+bar) | right (card)
        bot = tk.PanedWindow(paned, orient=tk.HORIZONTAL,
                             sashwidth=5, sashrelief=tk.RAISED,
                             bg='#d9d9d9')
        paned.add(bot, minsize=400)

        left_paned = tk.PanedWindow(bot, orient=tk.VERTICAL,
                                    sashwidth=5, sashrelief=tk.RAISED,
                                    bg='#d9d9d9')
        bot.add(left_paned, stretch='always')
        self._bot_paned = bot

        # Spectrum frame
        f_spec = ttk.Frame(left_paned)
        left_paned.add(f_spec, minsize=180, stretch='always')
        self._fig_spec = plt.Figure(facecolor='#f8f8f8', tight_layout=True)
        self._ax_spec  = self._fig_spec.add_subplot(111)
        self._cv_spec  = FigureCanvasTkAgg(self._fig_spec, master=f_spec)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self._tb_spec = NavigationToolbar2Tk(self._cv_spec, f_spec, pack_toolbar=False)
        self._tb_spec.update()
        self._tb_spec.pack(side=tk.BOTTOM, fill=tk.X)
        self._cv_spec.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bar chart frame
        f_bar = ttk.Frame(left_paned)
        left_paned.add(f_bar, minsize=150, stretch='always')
        self._fig_bar = plt.Figure(facecolor='#f8f8f8', tight_layout=True)
        self._ax_bar  = self._fig_bar.add_subplot(111)
        self._cv_bar  = FigureCanvasTkAgg(self._fig_bar, master=f_bar)
        self._cv_bar.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Quality card — right panel
        f_card = ttk.Frame(bot)
        bot.add(f_card, minsize=180, stretch='never')
        def _set_sash(*_):
            total = self._bot_paned.winfo_width()
            if total > 10:
                self._bot_paned.sash_place(0, int(total * 0.70), 0)
        bot.bind('<Configure>', _set_sash, add='+')
        self._fig_card = plt.Figure(facecolor='#f8f8f8')
        self._fig_card.set_constrained_layout(True)
        gs_card = self._fig_card.add_gridspec(2, 1, height_ratios=[1.4, 1], hspace=0.15)
        self._ax_card     = self._fig_card.add_subplot(gs_card[0])
        self._ax_card_bot = self._fig_card.add_subplot(gs_card[1])
        self._cv_card  = FigureCanvasTkAgg(self._fig_card, master=f_card)
        self._cv_card.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._card_row_tips    = {}
        self._card_tooltip_win = None
        self._fig_card.canvas.mpl_connect('motion_notify_event', self._on_card_motion)
        self._fig_card.canvas.mpl_connect('axes_leave_event',    self._hide_card_tooltip)
        for _ax in (self._ax_card, self._ax_card_bot):
            _ax.axis('off')
        self._ax_card.text(0.5, 0.5, 'Select a metabolite\nto see quality info',
                           ha='center', va='center', fontsize=8, color='gray',
                           transform=self._ax_card.transAxes)

        self._detail_fig    = self._fig_spec
        self._detail_canvas = self._cv_spec

    def _show_placeholder(self):
        if self._tree: self._tree.pack_forget()
        if self._placeholder: self._placeholder.pack(expand=True)

    def _populate_tree(self):
        if self._tree is None or self._placeholder is None:
            return
        self._placeholder.pack_forget()
        self._tree.delete(*self._tree.get_children())
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        _nm = getattr(self, 'norm_var', None)
        _norm = _nm.get() if _nm else 'IS'
        if self._is_lipidomics_mode():
            # Lipidomics is semi-quantitative: region areas use nH=1, so even IS
            # normalisation yields relative (a.u.) values, never absolute mM.
            _conc_hdr = ('Mean Area %' if _norm == 'Total'
                         else 'Mean (rel., a.u.)')
        else:
            _conc_hdr = ('Mean Conc (mM)' if _norm in ('IS', 'IS+PQN')
                         else 'Mean Area %'  if _norm == 'Total'
                         else 'Mean Area (a.u.)')
        labels = {'peak_num': '#', 'metabolite': 'Metabolite', 'ppm': 'PPM',
                  'conc_mm': _conc_hdr, 'raw_area': 'Mean Area',
                  'snr': 'SNR (group)', 'vip_score': 'VIP', 'status': 'Conf'}
        for cid, lbl in labels.items():
            arrow = (' ▲' if not self._sort_rev else ' ▼') \
                    if cid == self._sort_col else ''
            self._tree.heading(cid, text=lbl + arrow)

        q = getattr(self, '_search_var', None)
        q = q.get().strip().lower() if q else ''

        def _sort_key(r):
            v = r.get(self._sort_col, 0) or 0
            if self._sort_col == 'metabolite':
                return str(v).lower()
            try: return float(v)
            except Exception: return str(v).lower()

        filtered = [r for r in self._quant_results
                    if not q or q in r.get('metabolite', '').lower()
                    or q in f"{r.get('ppm', 0):.4f}"]
        results = sorted(filtered, key=_sort_key, reverse=self._sort_rev)

        _is_mm = (_norm in ('IS', 'IS+PQN'))
        _used_iids = set()
        for r in results:
            conc = r.get('conc_mm', float('nan'))
            if not np.isfinite(conc):
                conc_s = 'N/A'
            elif _is_mm:
                conc_s = f'{conc:.4f}'
            else:
                conc_s = f'{conc:.3e}'
            area = r.get('raw_area', 0)
            snr  = r.get('snr', 0)
            stat   = r.get('status', '●○○')
            _conf  = r.get('_confidence', 1)
            if r.get('_not_quantified'):
                conf_tag = 'nq'
                conc_s = 'n.q.'
            else:
                conf_tag = f'conf{_conf}'
            peak_rank = r.get('_peak', {})
            peak_rank = peak_rank.get('rank', '–') if peak_rank else '–'
            vip = r.get('vip_score', float('nan'))
            vip_s = f'{vip:.2f}' if np.isfinite(vip) else '—'
            _mw_r = r.get('_multiplet_window')
            _im_r = r.get('_integ_mode', '')
            if _im_r in ('Multiplet', 'Cascade') and _mw_r and _mw_r[0] is not None:
                _ppm_str = (f"{min(float(_mw_r[0]), float(_mw_r[1])):.4f}"
                            f"–{max(float(_mw_r[0]), float(_mw_r[1])):.4f}")
            else:
                _ppm_str = f"{r.get('ppm', 0):.4f}"
            # Ensure IID is a unique non-empty string
            raw_iid = r.get('hmdb_id') or r.get('metabolite', 'unknown')
            iid = str(raw_iid)
            if iid in _used_iids:
                iid = f'{iid}_{id(r)}'
            _used_iids.add(iid)
            try:
                self._tree.insert(
                    '', tk.END,
                    iid=iid,
                    tags=(conf_tag,),
                    values=(peak_rank,
                            r['metabolite'],
                            _ppm_str,
                            conc_s,
                            f'{area:.4e}',
                            f'{snr:.1f}',
                            vip_s,
                            stat))
            except Exception as _ins_err:
                print(f'[Quant tree] insert failed for {iid!r}: {_ins_err}')

    def _on_tree_right_click(self, event):
        """Show context menu on right-click in quantification tree."""
        tree = self._tree
        iid = tree.identify_row(event.y)
        if not iid:
            return
        tree.selection_set(iid)
        r = next((x for x in self._quant_results
                  if str(x.get('hmdb_id', x['metabolite'])) == iid
                  or f"{x.get('hmdb_id', x['metabolite'])}_{id(x)}" == iid), None)
        if r is None:
            return
        menu = tk.Menu(tree, tearoff=0)
        # Navigate to the peak inside the integrated multiplet window (matches
        # Quant Peaks and the boundaries actually integrated), not the bare
        # competition anchor.
        peak_rank = r.get('_display_peak_rank') or r.get('_peak_rank')
        menu.add_command(
            label='Quant Signals',
            command=lambda: self._open_quant_peaks_window(r))
        from processing.deconv_review import kind_of
        if kind_of(r) or r.get('_deconv') or r.get('_unshared_reason'):
            menu.add_command(
                label='Deconvolution review…',
                command=lambda: self._open_deconvolution_window(
                    str(r.get('hmdb_id', r['metabolite']))))
        menu.add_separator()
        if peak_rank is not None:
            menu.add_command(
                label='Open signal in Signal List',
                command=lambda: self._open_peak_in_list(peak_rank))
        else:
            menu.add_command(label='No signal available', state='disabled')
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _safe_run(self, on_done=None):
        """Run quantification in a worker thread.

        on_done : optional callable invoked on the main thread after the run
                  finishes (used by the Quant Peaks window to refresh itself
                  after a multiplet override re-quantifies all analytes).
        """
        import threading
        # askyesno must run on main thread — it needs a user answer before
        # the computation thread starts.
        if self.ext_is_var.get() and self._ext_is_indices:
            if not messagebox.askyesno(
                    'External IS — Confirm',
                    'External IS mode is active.\n\n'
                    'Please confirm that the IS concentration and nH are the same\n'
                    'for both the External IS samples and the analytical samples.\n\n'
                    'Proceed with quantification?'):
                return
        self._ext_is_confirmed = True
        self.btn_run.config(state='disabled')
        self.run_status.config(text='Running…', foreground='blue')
        self._show_placeholder()
        self._build_progress_dialog()

        def _worker():
            try:
                self._run()
            except Exception as _run_exc:
                import traceback
                traceback.print_exc()
                self.app.root.after(0, lambda e=_run_exc: messagebox.showerror(
                    'Quantification Error',
                    f'Unexpected error:\n{type(e).__name__}: {e}\n\n'
                    f'Check quant_debug.log for details.'))
            finally:
                self._ext_is_confirmed = False
                def _cleanup():
                    self._close_progress_dialog()
                    self.btn_run.config(state='normal')
                    if on_done is not None:
                        try:
                            on_done()
                        except Exception:
                            import traceback
                            traceback.print_exc()
                self.app.root.after(0, _cleanup)

        threading.Thread(target=_worker, daemon=True).start()

    def _open_quant_peaks_window(self, result):
        """Open or reuse the Quant Peaks window for an analyte."""
        from gui.widgets.quant_peaks_window import QuantPeaksWindow
        try:
            existing = getattr(self, '_quant_peaks_win', None)
            if existing is not None:
                try:
                    if not existing.win.winfo_exists():
                        raise RuntimeError('window destroyed')
                    existing.load_result(result)
                    existing.win.lift()
                    return
                except Exception:
                    self._quant_peaks_win = None
            win = QuantPeaksWindow(self.frame, self, result)
            self._quant_peaks_win = win
            close_cb = lambda: self._on_quant_peaks_close(win)
            win._close_cb = close_cb
            win.win.protocol('WM_DELETE_WINDOW', win._do_close)
        except Exception:
            import traceback
            traceback.print_exc()

    def _open_deconvolution_window(self, select=None):
        """Open or raise the review of the windows resolved by shape."""
        from gui.widgets.deconvolution_window import DeconvolutionWindow
        existing = getattr(self, '_deconv_win', None)
        if existing is not None:
            try:
                if existing.win.winfo_exists():
                    existing.refresh(select=select)
                    existing.win.lift()
                    return
            except Exception:
                pass
            self._deconv_win = None
        try:
            self._deconv_win = DeconvolutionWindow(self.frame, self,
                                                   select=select)
        except Exception:
            import traceback
            traceback.print_exc()

    def _on_quant_peaks_close(self, win):
        win.win.destroy()
        if getattr(self, '_quant_peaks_win', None) is win:
            self._quant_peaks_win = None

    def _open_peak_in_list(self, peak_rank):
        """Open the Peak List window and focus the given peak rank."""
        try:
            self.app.open_peak_list()
            pl_win = getattr(self.app, '_peak_list_win', None)
            if pl_win and hasattr(pl_win, 'focus_peak_rank'):
                pl_win.focus_peak_rank(peak_rank)
        except Exception:
            import traceback
            traceback.print_exc()

    def _sort_tree(self, col):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = col not in ('metabolite',)
        if self._quant_results:
            self._populate_tree()

    # ── Integration mode change ───────────────────────────────────────────────


    def _hide_dev_tooltip(self, _e=None):
        tw = getattr(self, '_dev_tooltip_win', None)
        if tw is not None:
            try:
                tw.destroy()
            except Exception:
                pass
            self._dev_tooltip_win = None


    def _on_deconv_change(self, *_):
        self._refresh_deconv_label()

    def _shape_parts_on(self) -> bool:
        return bool(self.deconv_var.get() and self.shape_var.get()
                    and self.shared_var.get())

    def _sync_shape_windows(self, *, source):
        """Keep group A's switch and its three variables one setting: whichever
        was set -- the switch, or one variable on its own as a project saved
        before the group restores them -- the other three take its value. A
        mix (m* on, the split it depends on off) is not a setting the engine
        supports."""
        if self._syncing_shape_windows:
            return
        self._syncing_shape_windows = True
        try:
            on = bool(source.get())
            for v in (self.shape_windows_var, self.deconv_var, self.shape_var,
                      self.shared_var):
                if bool(v.get()) != on:
                    v.set(on)
        finally:
            self._syncing_shape_windows = False
        self._refresh_deconv_label()

    def _refresh_deconv_label(self):
        """What the last run's split actually did, beside the box that asked.

        Not what the setting is -- the box already says that. A correction that
        is switched on and silently splits nothing looks exactly like one that
        is not wired in, which cost a whole afternoon of this project.
        """
        if not hasattr(self, '_j_resolve_status_lbl'):
            return
        if not bool(self.deconv_var.get()):
            self._j_resolve_status_lbl.config(
                text='(off — winners keep whole windows)', foreground='gray')
            return
        sh = (getattr(self, '_quant_diagnostics', None) or {}).get('shared')
        if not sh or sh.get('off'):
            self._j_resolve_status_lbl.config(
                text='(not run yet)', foreground='gray')
            return
        n, total = int(sh.get('resolved', 0)), int(sh.get('contested', 0))
        self._j_resolve_status_lbl.config(
            text=f'({n} of {total} shared windows split)',
            foreground='#2e7d32' if n else '#b71c1c')

    # ── Standard preset ───────────────────────────────────────────────────────

    def _on_std_select(self, event=None):
        p = _STD_PRESETS.get(self.ref_std_var.get(), {})
        if p.get('ppm'): self.ref_ppm_var.set(p['ppm'])
        if p.get('nh'):  self.ref_nh_var.set(p['nh'])

    def _open_ext_is_dialog(self):
        """Open dialog to select External IS samples."""
        si = self.app.samples_info or []
        if not si:
            messagebox.showinfo('External IS', 'Load samples first.')
            return

        dlg = tk.Toplevel(self.frame)
        dlg.title('Select External IS Samples')
        dlg.geometry('320x400')
        dlg.transient(self.frame)
        dlg.grab_set()

        ttk.Label(dlg, text='Select samples to use as External IS reference:',
                  font=('Arial', 9), wraplength=280).pack(pady=(10, 4), padx=10)

        frame_list = ttk.Frame(dlg)
        frame_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        vsb = ttk.Scrollbar(frame_list)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(frame_list, selectmode=tk.MULTIPLE,
                        yscrollcommand=vsb.set, font=('Arial', 9))
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=lb.yview)

        for i, s in enumerate(si):
            lbl  = s.get('label', f'Sample {i+1}')
            typ  = s.get('type', '')
            disp = f"{i+1}. {lbl}" + (f"  [{typ}]" if typ else '')
            lb.insert(tk.END, disp)
            if i in self._ext_is_indices:
                lb.selection_set(tk.END)

        if not self._ext_is_indices:
            for i, s in enumerate(si):
                if str(s.get('type', s.get('label', ''))).upper() == 'BLK':
                    lb.selection_set(i)

        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=10, pady=8)

        def _ok():
            self._ext_is_indices = list(lb.curselection())
            n = len(self._ext_is_indices)
            names = ', '.join(str(si[i].get('label', i)) for i in self._ext_is_indices[:3])
            suffix = '…' if n > 3 else ''
            self._ext_is_label.config(
                text=f'{n} sample(s): {names}{suffix}' if n else '(none selected)',
                foreground='#1a7a1a' if n else 'gray')
            dlg.destroy()

        ttk.Button(btn_row, text='OK',     command=_ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text='Cancel', command=dlg.destroy).pack(side=tk.RIGHT)
