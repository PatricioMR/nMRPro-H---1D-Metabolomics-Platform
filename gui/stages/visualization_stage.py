# gui/stages/visualization_stage.py
"""
Data Visualization Stage.
Samples panel: Color | Sample Name | Group | Type (Sample / IS / BLK)
Type=BLK excludes the sample from all analysis (mean, deconv, alignment).
"""
import tkinter as tk
from tkinter import ttk, colorchooser
import numpy as np
from .base_stage import BaseStage
from app_paths import APP_VERSION

SAMPLE_TYPES = ['Sample', 'IS', 'BLK']


class VisualizationStage(BaseStage):

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.sample_vars = []
        self._color_btns = []
        self._original_intensities = None
        self.scrollable_frame = None
        self.create_widgets()

    # ── Layout ────────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        # ── BOTTOM: fixed widgets reserved first ──────────────────────────────
        proc = ttk.LabelFrame(self.frame, text="⚙ Processing", padding=4)
        proc.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 4))
        self.btn_alignment = ttk.Button(
            proc, text="📐 Alignment",
            command=self._open_alignment_dialog, state="disabled")
        self.btn_alignment.pack(fill=tk.X, pady=1)
        self.btn_baseline = ttk.Button(
            proc, text="📉 Baseline Correction",
            command=self._open_baseline_dialog, state="disabled")
        self.btn_baseline.pack(fill=tk.X, pady=1)
        self.btn_quality = ttk.Button(
            proc, text="📊 Spectral Quality",
            command=self._open_quality_window, state="disabled")
        self.btn_quality.pack(fill=tk.X, pady=1)
        self.btn_digital_shim = ttk.Button(
            proc, text="🧲 Digital Shim",
            command=self._open_digital_shim_dialog, state="disabled")
        self.btn_digital_shim.pack(fill=tk.X, pady=1)
        self.btn_revert = ttk.Button(
            proc, text="↩ Revert Processing",
            command=self._revert_processing, state="disabled")
        self.btn_revert.pack(fill=tk.X, pady=1)

        self.app.btn_update = ttk.Button(
            self.frame, text="🔄 Update Plot",
            command=self.app.update_plot, state="disabled")
        self.app.btn_update.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

        ctrl = ttk.Frame(self.frame)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=1)
        self.app.btn_select_all = ttk.Button(
            ctrl, text="Select All",
            command=self.select_all_samples, state="disabled")
        self.app.btn_select_all.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.app.btn_clear_all = ttk.Button(
            ctrl, text="Clear All",
            command=self.clear_all_samples, state="disabled")
        self.app.btn_clear_all.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # ── TOP: title + file ──────────────────────────────────────────────────
        ttk.Label(self.frame, text=APP_VERSION,
                  font=("Arial", 11, "bold")).pack(side=tk.TOP, pady=(8, 2))
        ttk.Separator(self.frame, orient='horizontal').pack(
            side=tk.TOP, fill=tk.X, padx=5, pady=2)

        ff = ttk.LabelFrame(self.frame, text="File", padding=4)
        ff.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.app.btn_import = ttk.Button(
            ff, text="📂 Import Data", command=self.app.import_data)
        self.app.btn_import.pack(fill=tk.X, pady=2)
        self.app.file_label = ttk.Label(
            ff, text="No file loaded",
            font=("Arial", 8), foreground="gray", wraplength=270)
        self.app.file_label.pack(pady=1)

        ttk.Separator(self.frame, orient='horizontal').pack(
            side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # ── MIDDLE: Samples — expands to fill remaining space ────────────────
        self._samples_lf = ttk.LabelFrame(self.frame, text="Samples", padding=0)
        sf = self._samples_lf
        sf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Header row — columns: ✓ | Color | Sample | Group
        hdr = tk.Frame(sf, bg='#e8e8e8')
        hdr.pack(side=tk.TOP, fill=tk.X)
        hdr.columnconfigure(2, weight=1)
        for col, txt, w in [(0, '✓', 2), (1, 'Color', 4), (2, 'Sample', 0),
                            (3, 'Group', 5)]:
            tk.Label(hdr, text=txt, width=w, bg='#e8e8e8',
                     font=('Arial', 7, 'bold'), anchor='w').grid(
                row=0, column=col, padx=1, pady=1)
        tk.Frame(sf, height=1, bg='#cccccc').pack(side=tk.TOP, fill=tk.X)

        # Canvas + scrollbar — scrollbar only shown when content overflows
        body = tk.Frame(sf)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        _bg = ttk.Style().lookup('TFrame', 'background') or '#f0f0f0'
        self._samples_canvas = tk.Canvas(
            body, borderwidth=0, highlightthickness=0, bg=_bg)
        self._samples_vsb = ttk.Scrollbar(
            body, orient=tk.VERTICAL,
            command=self._samples_canvas.yview)
        self.scrollable_frame = tk.Frame(self._samples_canvas, bg=_bg)

        def _on_content_change(e=None):
            """Show scrollbar only when content height > canvas height."""
            self._samples_canvas.configure(
                scrollregion=self._samples_canvas.bbox("all"))
            content_h = self.scrollable_frame.winfo_reqheight()
            canvas_h = self._samples_canvas.winfo_height()
            if content_h > canvas_h:
                self._samples_vsb.grid(row=0, column=1, sticky="ns")
                self._samples_canvas.itemconfigure(
                    self._win_id, width=self._samples_canvas.winfo_width())
            else:
                self._samples_vsb.grid_remove()
                self._samples_canvas.itemconfigure(
                    self._win_id, width=self._samples_canvas.winfo_width())

        self.scrollable_frame.bind("<Configure>", _on_content_change)

        self._win_id = self._samples_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")
        self._samples_canvas.bind(
            "<Configure>", lambda e: _on_content_change())
        self._samples_canvas.configure(
            yscrollcommand=self._samples_vsb.set)

        self._samples_canvas.grid(row=0, column=0, sticky="nsew")

        # scrollbar starts hidden — shown by _on_content_change when needed

        # Mouse-wheel scroll
        def _scroll(e):
            self._samples_canvas.yview_scroll(
                int(-1 * (e.delta / 120)) if e.delta else
                (-1 if e.num == 4 else 1), "units")

        self._samples_canvas.bind("<MouseWheel>", _scroll)
        self._samples_canvas.bind("<Button-4>", _scroll)
        self._samples_canvas.bind("<Button-5>", _scroll)

    # ── Sample list refresh ───────────────────────────────────────────────────

    def refresh_sample_list(self):
        for w in self.scrollable_frame.winfo_children():
            w.destroy()
        self.sample_vars = []
        self._color_btns = []
        # Em COSY: alinhamento = calibração TSP/DSS (faz sentido); baseline e
        # qualidade são específicos de 1D e ficam desactivados.
        _is_2d = getattr(self.app, 'cosy_data', None) is not None
        _proc_state = 'disabled' if _is_2d else 'normal'
        try:
            self.btn_alignment.config(state='normal')   # calibração também em COSY
            self.btn_baseline.config(state=_proc_state)
            self.btn_quality.config(state=_proc_state)
            self.btn_digital_shim.config(state=_proc_state)  # 1D only
            if self.app.all_intensities is not None:
                self._original_intensities = self.app.all_intensities.copy()
            self.btn_revert.config(state='disabled')
        except Exception:
            pass

        for i, sample in enumerate(self.app.samples_info):
            # Ensure type field exists
            if 'type' not in sample:
                sample['type'] = (
                    'BLK' if str(sample.get('label', '')).upper() == 'BLK'
                    else 'Sample')

            vis_var = tk.BooleanVar(value=True)
            sample['excluded'] = False   # sync excluded flag with checkbox
            self.sample_vars.append(vis_var)

            _bg = self.scrollable_frame.cget('bg')
            row = tk.Frame(self.scrollable_frame, bg=_bg)
            row.pack(fill=tk.X, padx=1, pady=0)
            row.columnconfigure(2, weight=1)

            # Visibility checkbox — tk.Checkbutton for minimal appearance
            _bg2 = row.cget('bg')
            def _on_vis(i=i, v=vis_var):
                if i < len(self.app.samples_info):
                    self.app.samples_info[i]['excluded'] = not v.get()
                self.app.update_plot()
            tk.Checkbutton(row, variable=vis_var,
                           command=_on_vis,
                           bg=_bg2, activebackground=_bg2,
                           highlightthickness=1,
                           ).grid(row=0, column=0, padx=1)

            # Color swatch button
            color = self.app.group_colors.get(sample['label'], '#888888')
            btn = tk.Button(row, bg=color, width=2, height=1,
                            relief=tk.FLAT, bd=1,
                            command=lambda idx=i: self._pick_color(idx))
            btn.grid(row=0, column=1, padx=1)
            self._color_btns.append(btn)

            # Sample name — expands with panel width
            name = sample['name']
            ttk.Label(row, text=name, font=('Arial', 8),
                      anchor='w').grid(row=0, column=2, padx=1, sticky='ew')

            # Group label
            ttk.Label(row, text=str(sample.get('label', '')), width=7,
                      font=('Arial', 8), foreground='#444'
                      ).grid(row=0, column=3, padx=1)

            self._style_type_row(row, sample.get('type', 'Sample'))

    def _style_type_row(self, row, stype):
        """Dim BLK rows visually."""
        fg = '#aaaaaa' if stype == 'BLK' else '#000000'
        for child in row.winfo_children():
            try:
                child.config(foreground=fg)
            except tk.TclError:
                pass
        # Also dim the row frame background slightly
        try:
            row.config(style='BLK.TFrame' if stype == 'BLK' else 'TFrame')
        except Exception:
            pass


    def _pick_color(self, idx):
        """Open color picker for sample idx."""
        sample = self.app.samples_info[idx]
        current = self.app.group_colors.get(sample['label'], '#888888')
        color = colorchooser.askcolor(color=current,
                                      title=f"Color for {sample['name']}")[1]
        if color:
            self.app.group_colors[sample['label']] = color
            self._color_btns[idx].config(bg=color)
            self.app.update_plot()

    # ── Revert processing ─────────────────────────────────────────────────────

    def _revert_processing(self):
        """Restore all_intensities to original state (pre-alignment/baseline)."""
        if self._original_intensities is None:
            return
        from tkinter import messagebox
        if messagebox.askyesno(
                "Revert Processing",
                "This will undo ALL alignment and baseline corrections.\n"
                "Continue?", parent=self.app.root):
            self.app.all_intensities = self._original_intensities.copy()
            self.app.baseline_corrected  = False
            self.app.alignment_corrected = False
            self.app.update_plot()
            try:
                self.btn_revert.config(state='disabled')
            except Exception:
                pass

    # ── Reference deconvolution (FIDDLE / digital shim) ───────────────────────

    def _open_digital_shim_dialog(self):
        """Reference deconvolution: remove the common instrumental lineshape
        distortion (imperfect shim) using an isolated reference signal.
        Non-destructive — Apply backs onto all_intensities and the existing
        'Revert Processing' button restores the original."""
        import tkinter as tk
        from tkinter import ttk, messagebox
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        if self.app.ppm is None or self.app.all_intensities is None:
            return
        ppm = np.asarray(self.app.ppm, dtype=float)

        try:
            spec_mhz = float(self.app.stages['annotation'].spectrometer_var.get())
        except Exception:
            spec_mhz = 600.0

        dlg = tk.Toplevel(self.app.root)
        dlg.title("Digital Shim")
        dlg.transient(self.app.root)
        self._set_app_icon_safe(dlg)

        left = ttk.Frame(dlg, padding=8); left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(dlg); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Removes the lineshape distortion common to all\n"
                            "signals (imperfect shim) using one clean reference\n"
                            "signal. Areas are conserved.",
                  font=("Arial", 8), foreground="gray", justify=tk.LEFT
                  ).pack(anchor=tk.W, pady=(0, 6))

        rf = ttk.LabelFrame(left, text="Reference signal (ppm)", padding=6)
        rf.pack(fill=tk.X, pady=4)
        ttk.Label(rf, text="One sharp isolated signal (TSP/DSS 0.00, formate 8.46).",
                  font=("Arial", 8), foreground="gray", wraplength=210).pack(anchor=tk.W)
        r1 = ttk.Frame(rf); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Center (ppm):", width=14).pack(side=tk.LEFT)
        center_var = tk.StringVar(value="0.00")
        ttk.Entry(r1, textvariable=center_var, width=8).pack(side=tk.LEFT, padx=4)
        r2 = ttk.Frame(rf); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Window ± (ppm):", width=14).pack(side=tk.LEFT)
        window_var = tk.StringVar(value="0.05")
        ttk.Entry(r2, textvariable=window_var, width=8).pack(side=tk.LEFT, padx=4)

        tf = ttk.LabelFrame(left, text="Target lineshape", padding=6)
        tf.pack(fill=tk.X, pady=4)
        mode_var = tk.StringVar(value="symmetrise")
        ttk.Radiobutton(tf, text="Symmetrise only (keep width — safe)",
                        variable=mode_var, value="symmetrise").pack(anchor=tk.W)
        sharp_row = ttk.Frame(tf); sharp_row.pack(fill=tk.X)
        ttk.Radiobutton(sharp_row, text="Sharpen to FWHM (Hz):",
                        variable=mode_var, value="sharpen").pack(side=tk.LEFT)
        fwhm_var = tk.StringVar(value="1.0")
        ttk.Entry(sharp_row, textvariable=fwhm_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(tf, text="Sharpening amplifies noise — use a value near the\n"
                          "true natural linewidth.", font=("Arial", 7),
                  foreground="#a06000", wraplength=210).pack(anchor=tk.W)

        status = ttk.Label(left, text="", font=("Arial", 8),
                           foreground="blue", wraplength=210)
        status.pack(anchor=tk.W, pady=4)

        # View window (ppm) — so the user inspects a region instead of the whole
        # spectrum (where a tall TSP squashes everything).
        vf = ttk.LabelFrame(left, text="Preview view (ppm)", padding=6)
        vf.pack(fill=tk.X, pady=4)
        vr = ttk.Frame(vf); vr.pack(fill=tk.X)
        ttk.Label(vr, text="From:").pack(side=tk.LEFT)
        view_hi_var = tk.StringVar(value="")   # high ppm (left)
        ttk.Entry(vr, textvariable=view_hi_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(vr, text="To:").pack(side=tk.LEFT)
        view_lo_var = tk.StringVar(value="")   # low ppm (right)
        ttk.Entry(vr, textvariable=view_lo_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(vf, text="Blank = full. Zoom/pan also via the toolbar below.",
                  font=("Arial", 7), foreground="gray", wraplength=210).pack(anchor=tk.W)

        fig = plt.Figure(dpi=90, facecolor='#f8f8f8', tight_layout=True)
        axp = fig.add_subplot(111)
        axp.text(0.5, 0.5, 'Click Preview →', transform=axp.transAxes,
                 ha='center', va='center', color='gray')
        canvas = FigureCanvasTkAgg(fig, master=right)
        _tb = NavigationToolbar2Tk(canvas, right)
        _tb.update()
        _tb.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        _last = {'mean': None, 'corr': None}

        def _apply_view():
            """Set x-limits from the view boxes (or full) and autoscale Y to the
            visible window so a tall peak elsewhere does not flatten the rest."""
            m = _last['mean']
            if m is None:
                return
            try:
                hi = float(view_hi_var.get()) if view_hi_var.get().strip() else ppm.max()
            except ValueError:
                hi = ppm.max()
            try:
                lo = float(view_lo_var.get()) if view_lo_var.get().strip() else ppm.min()
            except ValueError:
                lo = ppm.min()
            lo, hi = min(lo, hi), max(lo, hi)
            axp.set_xlim(hi, lo)
            vis = (ppm >= lo) & (ppm <= hi)
            if vis.any():
                _top = max(float(np.max(m[vis])),
                           float(np.max(_last['corr'][vis])))
                if _top > 0:
                    axp.set_ylim(0, _top * 1.1)
            canvas.draw_idle()

        def _params():
            c = float(center_var.get()); w = float(window_var.get())
            tgt = (float(fwhm_var.get()) if mode_var.get() == "sharpen" else None)
            return c, w, tgt

        def _preview():
            from processing.digital_shim import digital_shim
            try:
                c, w, tgt = _params()
            except ValueError:
                status.config(text="Invalid numbers.", foreground="red"); return
            mean = np.mean(self.app.all_intensities, axis=1)
            try:
                corr, info = digital_shim(
                    ppm, mean, c - w, c + w, target_fwhm_hz=tgt,
                    spectrometer_mhz=spec_mhz)
            except Exception as e:
                status.config(text=f"Error: {e}", foreground="red"); return
            axp.clear()
            axp.plot(ppm, mean, color='#888', lw=0.8, label='before')
            axp.plot(ppm, corr, color='#1565c0', lw=0.8, label='after')
            axp.set_xlabel('δ (ppm)'); axp.set_ylabel('Intensity')
            axp.legend(fontsize=8)
            axp.set_title(f"area conserved: {info['area_ratio']*100:.1f}%  ·  "
                          f"ref FWHM {info['ref_fwhm_ppm']*spec_mhz:.2f} Hz  ·  "
                          "applied to the whole spectrum",
                          fontsize=9)
            _last['mean'], _last['corr'] = mean, corr
            # Default view: zoom to a window beside the reference so the effect on
            # real peaks is visible (not dominated by the reference itself).
            if not view_hi_var.get().strip() and not view_lo_var.get().strip():
                _c, _w, _ = _params()
                view_hi_var.set(f"{_c + 1.0:.2f}")
                view_lo_var.set(f"{_c + 0.2:.2f}")
            _apply_view()
            status.config(text=f"Preview OK. Area ratio {info['area_ratio']:.3f}",
                          foreground="green")

        def _apply():
            from processing.digital_shim import digital_shim_2d
            try:
                c, w, tgt = _params()
            except ValueError:
                status.config(text="Invalid numbers.", foreground="red"); return
            status.config(text="Applying to all samples…", foreground="blue")
            dlg.update_idletasks()
            try:
                corr2d, infos = digital_shim_2d(
                    ppm, self.app.all_intensities, c - w, c + w,
                    target_fwhm_hz=tgt, spectrometer_mhz=spec_mhz)
            except Exception as e:
                status.config(text=f"Error: {e}", foreground="red"); return
            if self._original_intensities is None:
                self._original_intensities = self.app.all_intensities.copy()
            self.app.all_intensities = corr2d
            self.app.update_plot()
            try:
                self.btn_revert.config(state='normal')
            except Exception:
                pass
            _ar = float(np.mean([i['area_ratio'] for i in infos]))
            messagebox.showinfo(
                "Digital Shim",
                f"Applied to {len(infos)} sample(s).\n"
                f"Mean area conservation: {_ar*100:.1f}%\n\n"
                "Use 'Revert Processing' to undo.", parent=dlg)
            dlg.destroy()

        ttk.Button(vf, text="↻ Update view", command=_apply_view).pack(
            anchor=tk.W, pady=(4, 0))

        bf = ttk.Frame(left); bf.pack(fill=tk.X, pady=8)
        ttk.Button(bf, text="👁 Preview", command=_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="✓ Apply", command=_apply).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=2)

    def _set_app_icon_safe(self, win):
        try:
            self.app._set_app_icon(win)
        except Exception:
            pass

    # ── Smooth dialog ─────────────────────────────────────────────────────────

    # ── Alignment dialog ──────────────────────────────────────────────────────

    def _open_cosy_calibration_dialog(self):
        """Calibração dos eixos COSY: referenciar TSP/DSS-d4 a 0 ppm."""
        from tkinter import messagebox
        cd = self.app.cosy_data
        if cd is None:
            return

        dlg = tk.Toplevel(self.app.root)
        dlg.title('COSY Calibration (TSP/DSS → 0 ppm)')
        dlg.transient(self.app.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text='Reference axes to internal standard',
                  font=('Arial', 10, 'bold')).pack(padx=16, pady=(12, 2))
        ttk.Label(dlg,
                  text='TSP / DSS-d4 is a singlet → diagonal peak at (0,0).\n'
                       'Auto-detects the reference signal and shifts F2 and F1.',
                  font=('Arial', 8), foreground='gray',
                  justify=tk.LEFT).pack(padx=16, pady=(0, 8))

        frm = ttk.Frame(dlg); frm.pack(padx=16, pady=2)
        ttk.Label(frm, text='Reference (ppm):', width=16).grid(
            row=0, column=0, sticky=tk.W, pady=2)
        ref_var = tk.DoubleVar(value=0.00)
        ttk.Entry(frm, textvariable=ref_var, width=8).grid(row=0, column=1, pady=2)
        ttk.Label(frm, text='Search window ±:', width=16).grid(
            row=1, column=0, sticky=tk.W, pady=2)
        win_var = tk.DoubleVar(value=0.10)
        ttk.Entry(frm, textvariable=win_var, width=8).grid(row=1, column=1, pady=2)

        status = ttk.Label(dlg, text='', font=('Arial', 8),
                           foreground='blue', wraplength=320)
        status.pack(padx=16, pady=6)

        from processing.cosy_calibration import calibrate_cosy

        def _apply():
            try:
                ref = float(ref_var.get()); win = float(win_var.get())
            except (tk.TclError, ValueError):
                status.config(text='Invalid values.', foreground='red'); return
            cal = calibrate_cosy(cd['f2_ppm'], cd['f1_ppm'], cd['data'],
                                 ref=ref, window=win)
            if cal is None:
                status.config(
                    text='Reference signal not found in window. '
                         'Widen the search window.', foreground='red')
                return
            cd['f2_ppm'] = cal['f2_ppm']
            cd['f1_ppm'] = cal['f1_ppm']
            self.app._draw_2d_spectrum(preserve_view=False)
            status.config(
                text=(f"Applied. TSP at F2={cal['f2_obs']:.3f}, "
                      f"F1={cal['f1_obs']:.3f} → 0.\n"
                      f"Shift ΔF2={cal['df2']*1000:.1f}, "
                      f"ΔF1={cal['df1']*1000:.1f} mppm."),
                foreground='green')
            dlg.after(1400, dlg.destroy)

        bf = ttk.Frame(dlg); bf.pack(pady=(2, 12))
        ttk.Button(bf, text='Auto-detect & Apply', command=_apply).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=dlg.destroy).pack(
            side=tk.LEFT, padx=4)

    def _open_alignment_dialog(self):
        """Controls (left) | interactive plot (right).

        Em COSY abre a calibração de eixos (TSP/DSS → 0 ppm) em vez do
        alinhamento 1D multi-amostra.
        """
        if getattr(self.app, 'cosy_data', None) is not None:
            self._open_cosy_calibration_dialog()
            return
        if self.app.ppm is None or self.app.all_intensities is None:
            return

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        ppm_arr = self.app.ppm
        all_int = self.app.all_intensities
        si = self.app.samples_info or []
        n_s = all_int.shape[1] if all_int.ndim == 2 else 1

        # ── Window capped to screen ─────────────────────────────────────────────
        sw = self.app.root.winfo_screenwidth()
        sh = self.app.root.winfo_screenheight()
        W = min(900, sw - 80)
        H = min(460, sh - 100)

        dlg = tk.Toplevel(self.app.root)
        dlg.title("Alignment")
        dlg.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")
        dlg.minsize(640, 360)
        dlg.transient(self.app.root)
        dlg.grab_set()
        dlg.resizable(True, True)

        # ── Main layout: left (controls) | right (plot) ────────
        pw = tk.PanedWindow(dlg, orient=tk.HORIZONTAL,
                            sashwidth=5, sashrelief=tk.RAISED)
        pw.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pw, width=260)
        right = ttk.Frame(pw)
        pw.add(left, minsize=220, stretch='never')
        pw.add(right, minsize=380, stretch='always')
        left.pack_propagate(False)

        # ── Left panel: controls ───────────────────────────────
        # Scrollable for safety on small screens
        cv_scroll = tk.Canvas(left, borderwidth=0, highlightthickness=0)
        vsb_l = ttk.Scrollbar(left, orient=tk.VERTICAL, command=cv_scroll.yview)
        sf_l = ttk.Frame(cv_scroll)
        sf_l.bind("<Configure>",
                  lambda e: cv_scroll.configure(
                      scrollregion=cv_scroll.bbox("all")))
        cv_scroll.create_window((0, 0), window=sf_l, anchor="nw")
        cv_scroll.configure(yscrollcommand=vsb_l.set)
        cv_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb_l.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Mode selector ───────────────────────────────────────────────────────
        mode_var = tk.StringVar(value='single')

        # _toggle_mode references rf and ic_frame defined below;
        # safe because it is only called after those widgets are created.
        def _toggle_mode():
            if mode_var.get() == 'single':
                ic_frame.pack_forget()
                rf.pack(fill=tk.X, padx=6, pady=6, after=mode_frame)
            else:
                rf.pack_forget()
                ic_frame.pack(fill=tk.X, padx=6, pady=4, after=mode_frame)

        mode_frame = ttk.LabelFrame(sf_l, text="Alignment Mode", padding=6)
        mode_frame.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Radiobutton(mode_frame, text="Single signal  (global shift)",
                        variable=mode_var, value='single',
                        command=_toggle_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Segment-wise  (icoshift)",
                        variable=mode_var, value='icoshift',
                        command=_toggle_mode).pack(anchor=tk.W)
        ttk.Label(mode_frame,
                  text="icoshift corrects non-uniform δ (ppm) drift across the spectrum.",
                  font=("Arial", 8), foreground="gray", wraplength=210).pack(anchor=tk.W)

        rf = ttk.LabelFrame(sf_l, text="Reference Region (ppm)", padding=6)
        rf.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(rf,
                  text="Must contain one sharp signal (e.g. TSP/DSS 0.00, formate 8.46).",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        r1 = ttk.Frame(rf);
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Center (ppm):", width=15).pack(side=tk.LEFT)
        center_var = tk.StringVar(value="0.00")
        ttk.Entry(r1, textvariable=center_var, width=8).pack(side=tk.LEFT, padx=4)
        r2 = ttk.Frame(rf);
        r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Window ± (ppm):", width=15).pack(side=tk.LEFT)
        window_var = tk.StringVar(value="0.05")
        ttk.Entry(r2, textvariable=window_var, width=8).pack(side=tk.LEFT, padx=4)

        # icoshift parameters (not packed initially — shown by _toggle_mode)
        ic_frame = ttk.LabelFrame(sf_l, text="icoshift Parameters", padding=6)
        ttk.Label(ic_frame,
                  text="Divides spectrum into segments; each aligned independently.",
                  font=("Arial", 8), foreground="gray", wraplength=210).pack(anchor=tk.W)
        ic1 = ttk.Frame(ic_frame);  ic1.pack(fill=tk.X, pady=2)
        ttk.Label(ic1, text="Segments:", width=15).pack(side=tk.LEFT)
        n_seg_var = tk.StringVar(value="50")
        ttk.Entry(ic1, textvariable=n_seg_var, width=8).pack(side=tk.LEFT, padx=4)
        ic2 = ttk.Frame(ic_frame);  ic2.pack(fill=tk.X, pady=2)
        ttk.Label(ic2, text="Max shift ± ppm:", width=15).pack(side=tk.LEFT)
        max_shift_var = tk.StringVar(value="0.030")
        ttk.Entry(ic2, textvariable=max_shift_var, width=8).pack(side=tk.LEFT, padx=4)

        sf2 = ttk.LabelFrame(sf_l, text="Reference Sample", padding=6)
        sf2.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(sf2,
                  text="Other samples shift to match this.",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        sample_names = [s.get('name', f'S{i}') for i, s in enumerate(si)]
        ref_var = tk.StringVar(value=sample_names[0] if sample_names else '')
        ttk.Combobox(sf2, textvariable=ref_var, values=sample_names,
                     state='readonly', width=26).pack(anchor=tk.W, pady=4)

        status_lbl = ttk.Label(sf_l, text="", font=("Arial", 8),
                               foreground="blue", wraplength=220)
        status_lbl.pack(padx=6, pady=4, anchor=tk.W)

        align_pb = ttk.Progressbar(sf_l, orient='horizontal', length=220,
                                   mode='indeterminate')
        # Not packed until a computation starts; packed/unpacked dynamically.

        bf = ttk.Frame(sf_l)
        bf.pack(fill=tk.X, padx=6, pady=6)
        btn_prev = ttk.Button(bf, text="👁 Preview", width=10)
        btn_apply = ttk.Button(bf, text="✓ Apply", width=10)
        btn_cancel = ttk.Button(bf, text="Cancel", width=8,
                                command=dlg.destroy)
        btn_prev.pack(side=tk.LEFT, padx=2)
        btn_apply.pack(side=tk.LEFT, padx=2)
        btn_cancel.pack(side=tk.LEFT, padx=2)

        # ── Right panel: interactive plot ────────────────────────────────
        fig = plt.Figure(dpi=90, facecolor='#f8f8f8', tight_layout=True)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#f8f8f8')
        ax.text(0.5, 0.5, 'Click Preview →',
                transform=ax.transAxes, ha='center', va='center',
                color='gray', fontsize=10)
        canvas = FigureCanvasTkAgg(fig, master=right)

        # Toolbar: BOTTOM inside plot_frame, before canvas TOP
        tb = NavigationToolbar2Tk(canvas, right)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── State ─────────────────────────────────────────────────────────────
        _state = {'aligned': None, 'shift_pts': None, 'shift_ppm': None}

        def _start_progress(msg):
            """Show indeterminate bar and disable buttons during computation."""
            status_lbl.config(text=msg, foreground="blue")
            align_pb.pack(padx=6, pady=(0, 4), anchor=tk.W)
            align_pb.start(12)
            btn_prev.config(state='disabled')
            btn_apply.config(state='disabled')
            dlg.update_idletasks()

        def _stop_progress():
            align_pb.stop()
            align_pb.pack_forget()
            btn_prev.config(state='normal')
            btn_apply.config(state='normal')

        def _preview_single():
            try:
                center = float(center_var.get())
                window = float(window_var.get())
            except ValueError:
                status_lbl.config(text="Invalid ppm values.", foreground="red")
                return
            mask = (ppm_arr >= center - window) & (ppm_arr <= center + window)
            if not np.any(mask):
                status_lbl.config(text="No data in region.", foreground="red")
                return
            win_idx = np.where(mask)[0]
            # Target index: the data point closest to the center ppm (e.g. 0.00)
            target_idx = int(np.argmin(np.abs(ppm_arr - center)))
            _start_progress("Aligning to reference signal…")
            from processing.peak_alignment import _shift_1d
            ppm_step = abs(float(ppm_arr[1] - ppm_arr[0]))
            aligned = np.empty_like(all_int)
            shifts = []
            peak_positions = []
            for s in range(n_s):
                col = all_int[:, s] if all_int.ndim == 2 else all_int
                # Find this sample's peak within the search window
                peak_i = win_idx[int(np.argmax(col[mask]))]
                peak_positions.append(float(ppm_arr[peak_i]))
                # Shift this sample so its peak lands exactly at center
                sp = target_idx - peak_i
                aligned[:, s] = _shift_1d(col, sp)
                shifts.append(sp)
            _state.update(aligned=aligned, shift_pts=shifts,
                          shift_ppm=[s * ppm_step for s in shifts])
            ax.clear()
            ax.set_facecolor('#f8f8f8')
            lo_p = center - window * 3
            hi_p = center + window * 3
            view = (ppm_arr >= lo_p) & (ppm_arr <= hi_p)
            x = ppm_arr[view]
            cmap = plt.cm.get_cmap('tab10', n_s)
            for s in range(min(n_s, 5)):
                col = all_int[view, s] if all_int.ndim == 2 else all_int[view]
                acol = aligned[view, s]
                c = cmap(s)
                ax.plot(x, col, color=c, lw=0.7, alpha=0.35, ls='--')
                ax.plot(x, acol, color=c, lw=1.1, alpha=0.9,
                        label=si[s]['name'] if s < len(si) else f'S{s}')
            # Show target (0 ppm) as a vertical reference line
            ax.axvline(center, color='red', lw=1.0, ls=':', alpha=0.8,
                       label=f'target {center:.3f} ppm')
            ax.invert_xaxis()
            ax.set_xlabel('δ (ppm)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
            ax.set_title('Dashed = before  |  Solid = after alignment', fontsize=8)
            fig.tight_layout()
            canvas.draw()
            _stop_progress()
            max_s = max(abs(v) for v in _state['shift_ppm'])
            mean_before = np.mean(peak_positions)
            status_lbl.config(
                text=(f"✓ {center:.4f} ppm  |  Max shift: {max_s * 1000:.1f} mppm"),
                foreground="green")

        def _preview_icoshift():
            import threading as _threading
            try:
                n_seg = int(n_seg_var.get())
                ms_ppm = float(max_shift_var.get())
            except ValueError:
                status_lbl.config(text="Invalid icoshift values.", foreground="red")
                return
            if n_seg < 2 or ms_ppm <= 0:
                status_lbl.config(text="Segments ≥ 2 and max shift > 0.", foreground="red")
                return
            ref_name = ref_var.get()
            ref_idx = next((i for i, s in enumerate(si)
                            if s.get('name') == ref_name), None)
            _start_progress(f"Computing icoshift ({n_seg} segments)…")

            _result = {}

            def _worker():
                try:
                    from processing.peak_alignment import icoshift_align
                    aligned, shifts_pts, seg_ppms = icoshift_align(
                        ppm_arr, all_int, ref_idx=ref_idx,
                        n_segments=n_seg, max_shift_ppm=ms_ppm)
                    _result['aligned'] = aligned
                    _result['shifts_pts'] = shifts_pts
                    _result['n_seg'] = n_seg
                    _result['ms_ppm'] = ms_ppm
                except Exception as exc:
                    _result['error'] = str(exc)

            def _done():
                _stop_progress()
                if 'error' in _result:
                    status_lbl.config(text=f"Error: {_result['error']}", foreground="red")
                    return
                aligned    = _result['aligned']
                shifts_pts = _result['shifts_pts']
                ppm_step = abs(float(ppm_arr[1] - ppm_arr[0]))
                shift_ppm_mat = shifts_pts.astype(float) * ppm_step
                _state.update(aligned=aligned, shift_pts=None,
                              shift_ppm=list(np.max(np.abs(shift_ppm_mat), axis=0)))
                ax.clear()
                ax.set_facecolor('#f8f8f8')
                cmap = plt.cm.get_cmap('tab10', n_s)
                for s in range(min(n_s, 5)):
                    col = all_int[:, s] if all_int.ndim == 2 else all_int
                    acol = aligned[:, s]
                    c = cmap(s)
                    ax.plot(ppm_arr, col, color=c, lw=0.6, alpha=0.22, ls='--')
                    ax.plot(ppm_arr, acol, color=c, lw=0.9, alpha=0.85,
                            label=si[s]['name'] if s < len(si) else f'S{s}')
                ax.invert_xaxis()
                ax.set_xlabel('δ (ppm)', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
                ax.set_title(
                    f'icoshift  ({n_seg} segments, ±{ms_ppm:.3f} ppm max)  '
                    f'Dashed = before  |  Solid = after', fontsize=8)
                fig.tight_layout()
                canvas.draw()
                max_s = float(np.max(np.abs(shift_ppm_mat))) * 1000
                mean_s = float(np.mean(np.abs(shift_ppm_mat))) * 1000
                status_lbl.config(
                    text=(f"✓ icoshift done  |  Max: {max_s:.1f} mppm  "
                          f"|  Mean: {mean_s:.1f} mppm"),
                    foreground="green")

            def _poll(t):
                if t.is_alive():
                    dlg.after(100, _poll, t)
                else:
                    dlg.after(0, _done)

            t = _threading.Thread(target=_worker, daemon=True)
            t.start()
            dlg.after(100, _poll, t)

        def _preview():
            if mode_var.get() == 'single':
                _preview_single()
            else:
                _preview_icoshift()

        def _apply():
            if _state['aligned'] is None:
                status_lbl.config(text="Run Preview first.", foreground="red")
                return
            self.app.all_intensities = _state['aligned']
            self.app.alignment_corrected = True
            self.app.update_plot()
            try:
                self.btn_revert.config(state='normal')
            except Exception:
                pass
            status_lbl.config(text="✓ Applied.", foreground="green")
            dlg.after(900, dlg.destroy)

        btn_prev.config(command=_preview)
        btn_apply.config(command=_apply)

    # ── Baseline Correction dialog ────────────────────────────────────────────

    def _open_baseline_dialog(self):
        """Controls (left) | interactive plot (right)."""
        if self.app.ppm is None or self.app.all_intensities is None:
            return

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        ppm_arr = self.app.ppm
        all_int = self.app.all_intensities
        n_s = all_int.shape[1] if all_int.ndim == 2 else 1
        si = self.app.samples_info or []

        # ── Window capped to screen ─────────────────────────────────────────────
        sw = self.app.root.winfo_screenwidth()
        sh = self.app.root.winfo_screenheight()
        W = min(1000, sw - 80)
        H = min(500, sh - 100)

        dlg = tk.Toplevel(self.app.root)
        dlg.title("Baseline Correction")
        dlg.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")
        dlg.minsize(680, 380)
        dlg.transient(self.app.root)
        dlg.grab_set()
        dlg.resizable(True, True)

        # ── Main layout: left (controls) | right (plot) ────────
        pw = tk.PanedWindow(dlg, orient=tk.HORIZONTAL,
                            sashwidth=5, sashrelief=tk.RAISED)
        pw.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pw, width=300)
        right = ttk.Frame(pw)
        pw.add(left, minsize=260, stretch='never')
        pw.add(right, minsize=380, stretch='always')
        left.pack_propagate(False)

        # ── Left panel scrollable ────────────────────────────────────────
        cv_scroll = tk.Canvas(left, borderwidth=0, highlightthickness=0)
        vsb_l = ttk.Scrollbar(left, orient=tk.VERTICAL, command=cv_scroll.yview)
        sf_l = ttk.Frame(cv_scroll)
        sf_l.bind("<Configure>",
                  lambda e: cv_scroll.configure(
                      scrollregion=cv_scroll.bbox("all")))
        cv_scroll.create_window((0, 0), window=sf_l, anchor="nw")
        cv_scroll.configure(yscrollcommand=vsb_l.set)
        cv_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb_l.pack(side=tk.RIGHT, fill=tk.Y)

        # Method toggle
        method_var = tk.StringVar(value='SNIP')
        mf = ttk.LabelFrame(sf_l, text="Method", padding=6)
        mf.pack(fill=tk.X, padx=6, pady=6)
        ttk.Radiobutton(mf, text="SNIP  (automatic)",
                        variable=method_var, value='SNIP',
                        command=lambda: _toggle_params()).pack(anchor=tk.W)
        ttk.Radiobutton(mf, text="arPLS  (advanced)",
                        variable=method_var, value='arPLS',
                        command=lambda: _toggle_params()).pack(anchor=tk.W)
        ttk.Label(mf,
                  text="SNIP: peak-clipping, good for most NMR.\n"
                       "arPLS: asymmetric fit, better for broad humps.",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)

        # Preview sample selector
        pf0 = ttk.LabelFrame(sf_l, text="Preview Sample", padding=6)
        pf0.pack(fill=tk.X, padx=6, pady=2)
        _blk_names = [s.get('name', f'S{i}') for i, s in enumerate(si)
                      if str(s.get('type', '')).upper() == 'BLK']
        _all_names = [s.get('name', f'S{i}') for i, s in enumerate(si)]
        _OPT_ALL = '— All samples —'
        _choices = [_OPT_ALL] + _all_names
        _default = _blk_names[0] if _blk_names else (_all_names[0] if _all_names else _OPT_ALL)
        preview_var = tk.StringVar(value=_default)
        ttk.Label(pf0,
                  text="Shown in preview only. Apply runs per-sample.",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
        ttk.Combobox(pf0, textvariable=preview_var,
                     values=_choices, state='readonly',
                     width=28).pack(anchor=tk.W, pady=2)

        # SNIP params (default visible) — campos numéricos, sem sliders
        snip_frame = ttk.LabelFrame(sf_l, text="SNIP Parameters", padding=6)
        snip_frame.pack(fill=tk.X, padx=6, pady=2)
        snip_iter_var = tk.StringVar(value="40")

        sf1 = ttk.Frame(snip_frame);
        sf1.pack(fill=tk.X, pady=2)
        ttk.Label(sf1, text="Iterations:", width=14).pack(side=tk.LEFT)
        ttk.Entry(sf1, textvariable=snip_iter_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(sf1, text="(10 – 200)", font=("Arial", 8),
                  foreground="gray").pack(side=tk.LEFT, padx=2)

        ttk.Label(snip_frame,
                  text="Iterations: more = deeper clipping.",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)

        # arPLS params (hidden by default) — campos numéricos, sem sliders
        params_frame = ttk.LabelFrame(sf_l, text="arPLS Parameters", padding=6)
        lam_var = tk.StringVar(value="100000")
        p_var = tk.StringVar(value="0.01")

        pf1 = ttk.Frame(params_frame);
        pf1.pack(fill=tk.X, pady=2)
        ttk.Label(pf1, text="λ smoothness:", width=14).pack(side=tk.LEFT)
        ttk.Entry(pf1, textvariable=lam_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(pf1, text="(1e3 – 1e7)", font=("Arial", 8),
                  foreground="gray").pack(side=tk.LEFT, padx=2)

        pf2 = ttk.Frame(params_frame);
        pf2.pack(fill=tk.X, pady=2)
        ttk.Label(pf2, text="p asymmetry:", width=14).pack(side=tk.LEFT)
        ttk.Entry(pf2, textvariable=p_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(pf2, text="(0.001 – 0.1)", font=("Arial", 8),
                  foreground="gray").pack(side=tk.LEFT, padx=2)

        ttk.Label(params_frame,
                  text="λ: larger = smoother.  p: smaller = stays below signals.",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)

        def _toggle_params():
            if method_var.get() == 'arPLS':
                snip_frame.pack_forget()
                params_frame.pack(fill=tk.X, padx=6, pady=2)
            else:
                params_frame.pack_forget()
                snip_frame.pack(fill=tk.X, padx=6, pady=2)

        status_lbl = ttk.Label(sf_l, text="", font=("Arial", 8),
                               foreground="blue", wraplength=240)
        status_lbl.pack(padx=6, pady=4, anchor=tk.W)

        bf = ttk.Frame(sf_l)
        bf.pack(fill=tk.X, padx=6, pady=6)
        btn_prev = ttk.Button(bf, text="👁 Preview", width=10)
        btn_apply = ttk.Button(bf, text="✓ Apply", width=10)
        btn_cancel = ttk.Button(bf, text="Cancel", width=8,
                                command=dlg.destroy)
        btn_prev.pack(side=tk.LEFT, padx=2)
        btn_apply.pack(side=tk.LEFT, padx=2)
        btn_cancel.pack(side=tk.LEFT, padx=2)

        # ── Right panel: interactive plot ────────────────────────────────
        fig = plt.Figure(dpi=90, facecolor='#f8f8f8', tight_layout=True)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#f8f8f8')
        ax.text(0.5, 0.5, 'Click Preview →',
                transform=ax.transAxes, ha='center', va='center',
                color='gray', fontsize=10)
        canvas = FigureCanvasTkAgg(fig, master=right)

        # Toolbar: BOTTOM inside plot_frame, before canvas TOP
        tb = NavigationToolbar2Tk(canvas, right)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── Algorithms ────────────────────────────────────────────────────────
        # Both come from processing/baseline_correction.py. This dialog used to
        # carry its own inline copies, which drifted: the local SNIP was missing
        # the -1 of the inverse LLS transform and zeroed any spectrum whose
        # intensities were of order 1 (normalised data), while the shared one
        # had been fixed.
        from processing.baseline_correction import snip as _snip_baseline
        from processing.baseline_correction import arPLS as _arPLS_baseline

        _state = {'corrected': None}

        def _preview():
            method = method_var.get()
            prev_name = preview_var.get()

            # Ler parâmetros dos campos de texto com validação
            try:
                if method == 'SNIP':
                    _iter = max(10, min(200, int(float(snip_iter_var.get()))))
                else:
                    _lam = max(1e3, min(1e7, float(lam_var.get())))
                    _p = max(0.001, min(0.1, float(p_var.get())))
            except ValueError as ve:
                status_lbl.config(text=f"Invalid parameter: {ve}", foreground="red")
                return

            ax.clear();
            ax.set_facecolor('#f8f8f8')
            n_neg_total = 0

            if prev_name == _OPT_ALL and all_int.ndim == 2:
                # ── Mostrar todas as amostras empilhadas ──────────────────────
                cmap_all = plt.cm.get_cmap('tab10', min(n_s, 10))
                for s in range(n_s):
                    col = all_int[:, s].astype(float)
                    try:
                        b = (_snip_baseline(col, _iter) if method == 'SNIP'
                             else _arPLS_baseline(col, _lam, _p))
                    except Exception:
                        b = np.zeros_like(col)
                    corr = col - b
                    n_neg_total += int(np.sum(corr < -1e-10))
                    c = cmap_all(s % 10)
                    name_s = si[s].get('name', f'S{s}') if s < len(si) else f'S{s}'
                    ax.plot(ppm_arr, col, color=c, lw=0.5, alpha=0.3)
                    ax.plot(ppm_arr, b, color=c, lw=0.8, ls='--', alpha=0.6)
                    ax.plot(ppm_arr, corr, color=c, lw=0.9, alpha=0.85,
                            label=name_s)
                title_str = f'{method} — all samples ({n_s})'
            else:
                # ── Amostra individual ────────────────────────────────────────
                if all_int.ndim == 2:
                    prev_idx = next((i for i, s in enumerate(si)
                                     if s.get('name', f'S{i}') == prev_name), 0)
                    col = all_int[:, prev_idx].astype(float)
                else:
                    col = all_int.ravel().astype(float)
                try:
                    b = (_snip_baseline(col, _iter) if method == 'SNIP'
                         else _arPLS_baseline(col, _lam, _p))
                except Exception as e:
                    status_lbl.config(text=f"Error: {e}", foreground="red")
                    import traceback;
                    traceback.print_exc()
                    return
                corr = col - b
                n_neg_total = int(np.sum(corr < -1e-10))
                ax.plot(ppm_arr, col, color='#aaaaaa', lw=0.7,
                        alpha=0.7, label='Original')
                ax.plot(ppm_arr, b, color='#e63946', lw=1.0,
                        ls='--', label='Baseline (dashed)')
                ax.plot(ppm_arr, corr, color='#2196F3', lw=1.0,
                        label='Corrected')
                if n_neg_total > 0:
                    ax.fill_between(ppm_arr, corr, 0,
                                    where=corr < -1e-10,
                                    color='orange', alpha=0.4, label='Negative region')
                title_str = f'{method} — {prev_name}'

            warn = f"  ⚠ {n_neg_total} neg pts" if n_neg_total > 0 else ""
            # Pre-compute per-sample correction (sempre necessário para Apply)
            # feito abaixo, mesmo para "all" preview
            ax.invert_xaxis()
            ax.set_xlabel('δ (ppm)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc='upper right', framealpha=0.8,
                      fancybox=True, shadow=False)
            ax.set_title(f'{title_str}{warn}', fontsize=8)
            fig.tight_layout();
            canvas.draw()
            corrected_all = np.empty_like(all_int, dtype=float)
            for s in range(n_s):
                col = (all_int[:, s] if all_int.ndim == 2
                       else all_int.ravel()).astype(float)
                try:
                    b = (_snip_baseline(col, _iter) if method == 'SNIP'
                         else _arPLS_baseline(col, _lam, _p))
                except Exception:
                    b = np.zeros_like(col)
                corrected_all[:, s] = col - b
            _state['corrected'] = corrected_all
            status_lbl.config(
                text=f"{n_s} samples estimated.{warn}  Click Apply to confirm.",
                foreground="blue" if not n_neg_total else "orange")

        def _apply():
            if _state['corrected'] is None:
                status_lbl.config(text="Run Preview first.", foreground="red")
                return
            # Records the CHOICE, not the corrected numbers. all_intensities
            # stays as loaded and every step reads app.corrected_intensities(),
            # so detection, quantification, statistics and MCR all see the same
            # baseline and the decision can still be changed or undone.
            from processing.baseline_model import BaselineModel
            method = method_var.get()
            try:
                if method == 'SNIP':
                    model = BaselineModel(
                        method='snip',
                        params={'niter': int(float(snip_iter_var.get()))})
                else:
                    model = BaselineModel(
                        method='arPLS',
                        params={'lam': float(lam_var.get())})
            except ValueError as ve:
                status_lbl.config(text=f"Invalid parameter: {ve}",
                                  foreground="red")
                return

            self.app.baseline_model = model
            self.app.invalidate_baseline_cache()
            # Keep the Signal Detection panel showing the same choice.
            try:
                self.app.stages['peak_detection'].pull_baseline_model()
            except Exception:
                pass
            self.app.update_plot()
            status_lbl.config(
                text=f"✓ Applied to the whole project ({model.describe()}).",
                foreground="green")
            dlg.after(900, dlg.destroy)

        btn_prev.config(command=_preview)
        btn_apply.config(command=_apply)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        if not hasattr(self.app, 'plot_manager') \
                or self.app.plot_manager is None:
            try:
                from plotting.interactive import InteractivePlotManager
                self.app.plot_manager = InteractivePlotManager(
                    self.app.figure, self.app.ax,
                    self.app.canvas, self.app)
            except Exception as e:
                print(f"PlotManager: {e}")
        if (hasattr(self.app, 'plot_manager')
                and self.app.plot_manager is not None
                and self.app.plot_manager.original_xlim is None):
            self.app.plot_manager.original_xlim = self.app.ax.get_xlim()
            self.app.plot_manager.original_ylim = self.app.ax.get_ylim()

    def on_deactivate(self):
        if self.frame:
            self.frame.pack_forget()
        # Sair da visualização → restaurar canvas 1D se o viewer 2D estava activo
        if getattr(self.app, '_2d_canvas', None) is not None:
            try:
                self.app._hide_2d_viewer()
            except Exception:
                pass

    def select_all_samples(self):
        for v in self.sample_vars:
            v.set(True)
        self.app.update_plot()

    def clear_all_samples(self):
        for v in self.sample_vars:
            v.set(False)
        self.app.update_plot()

    # ── Spectral Quality ──────────────────────────────────────────────────────

    def _open_quality_window(self):
        """Compute FWHM + Asymmetry per sample and open the quality window."""
        if self.app.ppm is None or self.app.all_intensities is None:
            return

        # Single-instance: re-focus the existing window instead of reopening.
        existing = getattr(self, '_quality_win', None)
        if existing is not None:
            try:
                if existing.win.winfo_exists():
                    existing.win.deiconify()
                    existing.win.lift()
                    existing.win.focus_set()
                    return
            except tk.TclError:
                pass
            self._quality_win = None

        # Resolve spectrometer frequency: Bruker sets app.spectrometer_mhz;
        # CSV imports may have it in the annotation stage's spectrometer_var.
        mhz = getattr(self.app, 'spectrometer_mhz', None)
        if mhz is None:
            ann = self.app.stages.get('annotation')
            if ann and hasattr(ann, 'spectrometer_var'):
                try:
                    mhz = float(ann.spectrometer_var.get())
                except (ValueError, TypeError):
                    mhz = None

        from processing.spectral_quality import assess_all_samples
        results = assess_all_samples(
            self.app.ppm,
            self.app.all_intensities,
            self.app.samples_info or [],
            spectrometer_mhz=mhz,
        )

        from gui.widgets.spectral_quality_window import SpectralQualityWindow
        self._quality_win = SpectralQualityWindow(
            self.app.root, results, spectrometer_mhz=mhz,
            ppm=self.app.ppm, intensities_2d=self.app.all_intensities,
        )