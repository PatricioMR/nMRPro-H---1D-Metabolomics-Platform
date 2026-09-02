# gui/notation/deconvolution/deconv_plot_panel.py
"""
Right panel — spectrum plot + deconvolution quality card.
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import numpy as np

from gui.utils.quality_palette import QUALITY_FG, SIGNAL_AVERAGE

# Literature thresholds (NMR metabolomics, AlpsNMR / NMR-Onion)
QUALITY_THRESHOLDS = {
    'r_squared': {'good': 0.98, 'ok': 0.90},
    'bic_delta': {'good': 10, 'ok': 2},  # lower is better vs null model
    'snr': {'good': 10, 'ok': 3},
    'linewidth': {'good': 0.02, 'ok': 0.05},  # ppm — typical 1H NMR
}


def _quality_color(metric, value):
    t = QUALITY_THRESHOLDS.get(metric, {})
    if not t:
        return '#888888'
    if metric in ('r_squared',):
        # Higher is better
        if value >= t['good']:  return '#1a7a1a'
        if value >= t['ok']:    return '#7a6000'
        return '#9a0000'
    elif metric == 'snr':
        if value >= t['good']:  return '#1a7a1a'
        if value >= t['ok']:    return '#7a6000'
        return '#9a0000'
    else:
        # Lower is better (linewidth, bic_delta)
        if value <= t['good']:  return '#1a7a1a'
        if value <= t['ok']:    return '#7a6000'
        return '#9a0000'


class DeconvPlotPanel:
    def __init__(self, parent, controller):
        self.parent = parent
        self.controller = controller
        self.current_region_index = None
        self._highlighted_child = None
        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                        padx=5, pady=5)

        # ── Plot frame (canvas + toolbar isolados dos controlos abaixo) ─────
        plot_frame = ttk.Frame(self.frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure = plt.Figure(figsize=(7, 5.2), dpi=100)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        self.ax     = self.figure.add_subplot(gs[0])
        self.ax_res = self.figure.add_subplot(gs[1], sharex=self.ax)
        self.ax_res.set_facecolor('#f5f5f5')
        self.ax_res.set_ylabel('Res.', fontsize=7)
        self.ax_res.tick_params(labelsize=6)
        self.ax_res.text(0.5, 0.5, 'residual after deconvolution',
                         ha='center', va='center',
                         transform=self.ax_res.transAxes,
                         fontsize=7, color='gray')

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()

        # Native toolbar: BOTTOM inside plot_frame, before canvas TOP
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self._toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self._toolbar.update()
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True)

        # Scroll-only zoom here; pan/drag is handled by native toolbar
        self._panning = False
        self._pan_start = None
        self._xlim_orig = None
        self._ylim_orig = None
        self._span_selector = None
        self._edit_bounds = False
        self._original_bounds = None
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        # button_press/release/motion removed — toolbar handles pan/zoom

        self.plot_info = ttk.Label(
            self.frame, text="Select a region from the list",
            foreground="gray")
        self.plot_info.pack(pady=2)

        # ── View mode toggle + Samples button ────────────────────────────
        self._build_group_checkboxes()

        # ── Edit boundaries toggle ───────────────────────────────────────────
        ef = ttk.Frame(self.frame)
        ef.pack(fill=tk.X, padx=6, pady=(0, 2))
        self._edit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ef, text='✏ Edit signal boundaries (drag)',
            variable=self._edit_var,
            command=self._toggle_edit_bounds
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(ef, text='Drag to set start/end ppm',
                  font=('Arial', 7), foreground='gray').pack(side=tk.LEFT)

        # ── Quality card ──────────────────────────────────────────────────────
        self.quality_frame = ttk.LabelFrame(
            self.frame, text="Deconvolution Quality", padding=6)
        self.quality_frame.pack(fill=tk.X, padx=4, pady=4)

        # Build metric grid: 2 columns of 3 metrics each
        # Grid layout: [label | value | ind | hint | sep | label | value | ind | hint]
        #               col 0   col 1  col 2  col 3  col4  col 5   col 6  col 7  col 8
        self._quality_labels = {}
        metrics = [
            ('r_squared', 'R²',        'Higher is better  (≥0.98 = good)'),
            ('bic',       'BIC',       'Lower is better'),
            ('snr',       'SNR',       'Per-signal SNR  (≥10 = good)'),
            ('linewidth', 'Linewidth', 'ppm  (≤0.02 = good for 1H)'),
            ('model',     'Model',     'Lineshape model used'),
        ]
        ROWS_PER_COL = 3
        for i, (key, label, hint) in enumerate(metrics):
            col_group = i // ROWS_PER_COL   # 0 → left, 1 → right
            row       = i  % ROWS_PER_COL
            base_col  = col_group * 5       # left starts at 0, right at 5 (sep at 4)

            ttk.Label(self.quality_frame, text=f"{label}:",
                      font=("Arial", 8, "bold")).grid(
                row=row, column=base_col,     sticky=tk.W, padx=(6, 2), pady=1)
            val_lbl = ttk.Label(self.quality_frame, text="—", font=("Arial", 8))
            val_lbl.grid(row=row, column=base_col + 1, sticky=tk.W, padx=2, pady=1)
            ind = tk.Label(self.quality_frame, text="●",
                           font=("Arial", 10), fg='#cccccc')
            ind.grid(row=row, column=base_col + 2, padx=2)
            ttk.Label(self.quality_frame, text=hint,
                      font=("Arial", 7), foreground='#666').grid(
                row=row, column=base_col + 3, sticky=tk.W, padx=(2, 8), pady=1)
            self._quality_labels[key] = (val_lbl, ind)

        # Vertical separator between the two metric columns
        ttk.Separator(self.quality_frame, orient='vertical').grid(
            row=0, column=4, rowspan=ROWS_PER_COL, sticky='ns', padx=4)

        # Peak-level quality (shown when child selected)
        self.peak_quality_frame = ttk.LabelFrame(
            self.frame, text="Selected Signal Details", padding=6)
        self.peak_quality_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.peak_detail_label = ttk.Label(
            self.peak_quality_frame, text="—",
            font=("Arial", 8))
        self.peak_detail_label.pack(anchor=tk.W)

    def _build_group_checkboxes(self):
        self.group_vars = {}
        proc = self.controller.processor
        app = getattr(proc, 'app', None)
        if not app or not hasattr(app, 'samples_info') or not app.samples_info:
            return

        groups = sorted(set([
            s.get('label', 'Unknown') for s in app.samples_info
            if str(s.get('type', s.get('label', ''))).upper() != 'BLK'
        ]))
        if not groups:
            return

        gf = ttk.Frame(self.frame)
        gf.pack(fill=tk.X, padx=6, pady=(0, 2))
        ttk.Label(gf, text="Show Groups:", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))

        for g in groups:
            var = tk.BooleanVar(value=True)
            self.group_vars[g] = var
            cb = ttk.Checkbutton(gf, text=g, variable=var,
                                 command=self._on_view_mode_change)
            cb.pack(side=tk.LEFT, padx=3)

    # ── Display ───────────────────────────────────────────────────────────────

    def display_region(self, peak_idx, child_index=None):
        """Display spectrum window for peak[peak_idx].
        Window = peak.start_ppm ± 0.1 ppm.
        Reference lines: centre (blue), start/end (grey 70%).
        """
        # Snapshot ONLY when switching to a different peak
        if peak_idx != self.current_region_index:
            if peak_idx is not None and peak_idx < len(self.controller.peaks):
                p = self.controller.peaks[peak_idx]
                self._original_bounds = {
                    'start_ppm': p.get('start_ppm', p['ppm']),
                    'end_ppm': p.get('end_ppm', p['ppm']),
                    'width_ppm': p.get('width_ppm', 0),
                    'width_at_half': p.get('width_at_half', 0),
                }
            else:
                self._original_bounds = None
        self.current_region_index = peak_idx
        self._highlighted_child = child_index

        all_peaks = self.controller.get_peaks()
        if peak_idx >= len(all_peaks):
            return
        pk = all_peaks[peak_idx]
        pk_ppm = pk['ppm']
        pk_start = max(pk.get('start_ppm', pk_ppm), pk.get('end_ppm', pk_ppm))
        pk_end = min(pk.get('start_ppm', pk_ppm), pk.get('end_ppm', pk_ppm))

        MARGIN = 0.1
        ppm_lo = pk_end - MARGIN
        ppm_hi = pk_start + MARGIN

        proc = self.controller.processor
        full_mask = (proc.ppm >= ppm_lo) & (proc.ppm <= ppm_hi)
        x_full = proc.ppm[full_mask]
        y_full = proc.mean_intensity[full_mask]
        if len(x_full) < 2:
            return

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        self.ax     = self.figure.add_subplot(gs[0])
        self.ax_res = self.figure.add_subplot(gs[1], sharex=self.ax)
        # SpanSelector must be re-attached to the new ax after figure.clear()
        self._span_selector = None
        if self._edit_bounds:
            self._attach_span_selector()

        # Spectrum (mode-dependent)
        shown_groups, g_colors = self._plot_spectrum(x_full, y_full, ppm_lo, ppm_hi)

        # Reference lines
        self.ax.axvline(pk_ppm, color='#1565c0', linewidth=1.4,
                        linestyle='--', alpha=1.0, zorder=7,
                        label=f'Centre {pk_ppm:.4f}')
        self.ax.axvline(pk_start, color='#666666', linewidth=1.1,
                        linestyle='--', alpha=0.70, zorder=6,
                        label=f'Start {pk_start:.4f}')
        self.ax.axvline(pk_end, color='#666666', linewidth=1.1,
                        linestyle='--', alpha=0.70, zorder=6,
                        label=f'End {pk_end:.4f}')
        self.ax.axvspan(pk_end, pk_start,
                        alpha=0.07, color='#1565c0', zorder=1)

        # Deconvolution overlay + residual
        results = self.controller.results.get(peak_idx)
        peaks = results.get('peaks', []) if results else []
        if results and peaks:
            self._plot_deconvolution({}, results, child_index,
                                     ppm_lo, ppm_hi, shown_groups, g_colors)
            self._plot_residual(results, ppm_lo, ppm_hi)
            self.plot_info.config(
                text=f"Signal #{pk.get('rank', '?')} @ {pk_ppm:.4f} ppm "
                     f"— {len(peaks)} sub-signal(s)",
                foreground='green')
            self._update_quality_card(results, child_index)
        else:
            self._clear_residual()
            self.plot_info.config(
                text=f"Signal #{pk.get('rank', '?')} @ {pk_ppm:.4f} ppm "
                     f"— pending deconvolution",
                foreground='blue')
            self._clear_quality_card()

        # xlim: zoom to sub-peak or full window
        target_ppm = None
        target_w = 0.005
        if child_index is not None:
            if results and child_index < len(peaks):
                sp = peaks[child_index]
                target_ppm = sp['ppm']
                target_w = max(sp.get('width',
                                      sp.get('width_ppm', 0.005)), 0.003)
            if target_ppm is None:
                target_ppm = pk_ppm
                target_w = max(pk.get('width_ppm',
                                      pk.get('width_at_half', 0.005)), 0.003)
        if target_ppm is not None:
            half = max(target_w * 12, 0.02)
            self.ax.set_xlim(target_ppm + half, target_ppm - half)
            self.ax.axvline(target_ppm, color='#e63946', linewidth=1.5,
                            linestyle='--', alpha=0.7, zorder=8)
        else:
            self.ax.set_xlim(ppm_hi, ppm_lo)

        # ax: hide x-tick labels (shared with ax_res below)
        plt.setp(self.ax.get_xticklabels(), visible=False)
        self.ax.set_ylabel('Intensity', fontsize=9)
        self.ax.grid(True, alpha=0.25, linestyle=':')
        self.ax.set_facecolor('#fafafa')
        self.ax.legend(loc='best', fontsize=7, ncol=2)
        # ax_res: shared x-axis carries the ppm label
        self.ax_res.set_xlabel('δ (ppm)', fontsize=9)
        self.ax_res.grid(True, alpha=0.20, linestyle=':')
        self.figure.tight_layout()
        self.canvas.draw()
        self._xlim_orig = self.ax.get_xlim()
        self._ylim_orig = self.ax.get_ylim()

    def _plot_deconvolution(self, data, results, highlight_child, ppm_lo=None, ppm_hi=None, shown_groups=None,
                            g_colors=None):
        def lorentzian(x, amp, cen, wid):
            return amp * (wid ** 2) / ((x - cen) ** 2 + wid ** 2)

        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

        def pseudo_voigt(x, amp, cen, wid, eta=0.5):
            lor = amp * (wid ** 2) / ((x - cen) ** 2 + wid ** 2)
            gau = amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))
            return eta * lor + (1 - eta) * gau

        def skewed_lorentzian(x, amp, cen, wid, gamma=0.0):
            lor = (wid ** 2) / ((x - cen) ** 2 + wid ** 2)
            skew = 1.0 + np.sin(gamma) * (x - cen) / (wid + 1e-10)
            return amp * lor * np.clip(skew, 0, None)

        model = results.get('model', 'Lorentzian')
        if model == 'Gaussian':
            model_fn = gaussian
        elif model == 'Pseudo-Voigt':
            model_fn = pseudo_voigt
        elif model == 'Skewed-Lorentzian':
            model_fn = skewed_lorentzian
        else:
            model_fn = lorentzian

        def _eval_peak(x_arr, peak):
            amp = peak['height']
            cen = peak['ppm']
            wid = peak['width']
            if model == 'Pseudo-Voigt':
                return model_fn(x_arr, amp, cen, wid, peak.get('eta', 0.5))
            elif model == 'Skewed-Lorentzian':
                return model_fn(x_arr, amp, cen, wid, peak.get('gamma', 0.0))
            return model_fn(x_arr, amp, cen, wid)

        def _area_to_amp(area, wid, peak):
            """Convert NNLS area back to amplitude for a given model."""
            if model == 'Gaussian':
                return area / (wid * np.sqrt(2 * np.pi))
            elif model == 'Lorentzian':
                return area / (wid * np.pi)
            else:
                # Pseudo-Voigt / Skewed-Lorentzian: integrate unit-amp shape numerically
                x_tmp = np.linspace(-10 * wid, 10 * wid, 500)
                if model == 'Pseudo-Voigt':
                    y_unit = pseudo_voigt(x_tmp, 1.0, 0.0, wid, peak.get('eta', 0.5))
                else:
                    y_unit = skewed_lorentzian(x_tmp, 1.0, 0.0, wid, peak.get('gamma', 0.0))
                unit_area = float(np.trapezoid(y_unit, x_tmp)
                                  if hasattr(np, 'trapezoid') else np.trapz(y_unit, x_tmp))
                return area / unit_area if unit_area > 0 else area / (wid * np.pi)

        # x_dense covers the full display window so model tails are visible.
        # Only the PPM centre of each peak is required to be within the region.
        x_lo = ppm_lo if ppm_lo is not None else 0.0
        x_hi = ppm_hi if ppm_hi is not None else 1.0
        x_dense = np.linspace(x_lo, x_hi, 600)
        colors = plt.cm.tab10(np.linspace(0, 1, len(results['peaks'])))
        cumulative = np.zeros_like(x_dense)

        # Build the most-intense group mean over the display window for dot placement
        proc = self.controller.processor
        app  = getattr(proc, 'app', None)
        raw  = getattr(app, 'all_intensities', None)
        _disp_mask = (proc.ppm >= x_lo) & (proc.ppm <= x_hi)
        _disp_ppm  = proc.ppm[_disp_mask]

        best_mean_y = proc.mean_intensity[_disp_mask]
        if shown_groups and raw is not None:
            best_peak_val = -np.inf
            for lbl, idxs in shown_groups.items():
                if not idxs:
                    continue
                gm = np.mean(raw[_disp_mask, :][:, idxs], axis=1)
                gmax = float(np.max(gm))
                if gmax > best_peak_val:
                    best_peak_val = gmax
                    best_mean_y   = gm

        for i, peak in enumerate(results['peaks']):
            status = peak.get('status', SIGNAL_AVERAGE)
            is_hl = (highlight_child == i)
            color = colors[i]

            y_peak = _eval_peak(x_dense, peak)
            cumulative += y_peak

            # Fill: highlighted peak gets deeper color
            alpha = 0.55 if is_hl else 0.20
            lw = 2.0 if is_hl else 1.0
            self.ax.fill_between(x_dense, 0, y_peak,
                                 alpha=alpha, color=color, zorder=1)
            self.ax.plot(x_dense, y_peak, '--',
                         color=color, linewidth=lw,
                         label=f'#{i + 1} {peak["ppm"]:.4f}  {(status or "—")[:4]}',
                         zorder=2)

            # Dot: x = position where the model curve actually peaks (correct for
            # all shapes including Skewed-Lorentzian), y = real spectrum intensity
            # at that x (most intense group mean), so the dot sits on the data.
            max_idx  = int(np.argmax(y_peak))
            dot_x    = float(x_dense[max_idx])
            sort_idx = np.argsort(_disp_ppm)
            dot_y    = float(np.interp(dot_x, _disp_ppm[sort_idx], best_mean_y[sort_idx]))
            dot_color = {'Good': '#1a7a1a',
                         'Poor': '#9a0000'}.get(status, '#7a6000')
            seed_kind = peak.get('seed_kind', 'top')
            marker = '*' if seed_kind == 'shoulder' else 'o'
            size = 85 if seed_kind == 'shoulder' else 40
            self.ax.scatter([dot_x], [dot_y], color=dot_color, marker=marker,
                            s=size, zorder=5)

        # Cumulative fit
        if shown_groups:
            for lbl, idxs in shown_groups.items():
                if not idxs: continue
                group_cumulative = np.zeros_like(x_dense)
                for peak in results['peaks']:
                    if 'area_per_sample' in peak:
                        areas = np.array(peak['area_per_sample'])
                        wid = float(peak['width'])
                        mean_amp = _area_to_amp(float(np.mean(areas[idxs])), wid, peak)
                        # build a temporary peak dict with the group mean amplitude
                        pk_tmp = dict(peak)
                        pk_tmp['height'] = mean_amp
                        y_sh = _eval_peak(x_dense, pk_tmp)
                        group_cumulative += y_sh
                    else:
                        group_cumulative += _eval_peak(x_dense, peak)
                mean_col = g_colors.get(lbl, '#111111') if g_colors else '#111111'
                self.ax.plot(x_dense, group_cumulative,
                             color=mean_col, linewidth=2, linestyle='--',
                             label=f'{lbl} Fit', zorder=6)
        else:
            self.ax.plot(x_dense, cumulative,
                         color='#e63946', linewidth=2,
                         label='Fit', zorder=4)

    # ── Residual subplot ──────────────────────────────────────────────────────

    def _plot_residual(self, results, ppm_lo, ppm_hi):
        """Plot residual in ax_res; color-code background by relative magnitude."""
        residual = results.get('residual')
        if residual is None:
            self._clear_residual()
            return

        proc = self.controller.processor
        mask = (proc.ppm >= ppm_lo) & (proc.ppm <= ppm_hi)
        x_reg = proc.ppm[mask]

        # Trim/interpolate residual to match x_reg length if needed
        if len(residual) != len(x_reg):
            x_src = np.linspace(ppm_lo, ppm_hi, len(residual))
            residual = np.interp(x_reg, x_src[::-1], residual[::-1])

        y_reg = proc.mean_intensity[mask]
        sig_max = float(np.max(np.abs(y_reg))) if len(y_reg) > 0 else 1.0
        rel = float(np.max(np.abs(residual))) / (sig_max + 1e-12)

        self.ax_res.cla()
        self.ax_res.axhline(0, color='#aaaaaa', lw=0.8, ls='--')
        self.ax_res.plot(x_reg, residual, color='#e53935', lw=0.9, alpha=0.85)
        self.ax_res.set_ylabel('Res.', fontsize=7)
        self.ax_res.tick_params(labelsize=6)
        self.ax_res.set_facecolor('#e8f5e9' if rel < 0.05 else '#fff3e0')

        white = results.get('residual_whiteness', {}) or {}
        lag1 = white.get('lag1_autocorr', None)
        dw = white.get('durbin_watson', None)
        white_like = bool(white.get('is_white_like', False))

        baseline_meta = results.get('baseline_validation', {}) or {}
        edge_ok = baseline_meta.get('valid', None)

        pieces = []
        if lag1 is not None and dw is not None:
            pieces.append(f"lag1={lag1:+.2f}")
            pieces.append(f"DW={dw:.2f}")
            pieces.append('white-like' if white_like else 'structured')
        if edge_ok is not None:
            pieces.append('edge baseline OK' if edge_ok else 'edge baseline FAIL')

        if pieces:
            self.ax_res.text(
                0.01, 0.96, ' | '.join(pieces),
                transform=self.ax_res.transAxes,
                ha='left', va='top', fontsize=6,
                color=('#2e7d32' if white_like else '#ef6c00'),
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='#dddddd', alpha=0.85)
            )

    def _clear_residual(self):
        if not hasattr(self, 'ax_res'):
            return
        self.ax_res.cla()
        self.ax_res.axhline(0, color='#aaaaaa', lw=0.8, ls='--')
        self.ax_res.set_facecolor('#f5f5f5')
        self.ax_res.set_ylabel('Res.', fontsize=7)
        self.ax_res.tick_params(labelsize=6)
        self.ax_res.text(0.5, 0.5, 'residual after deconvolution',
                         ha='center', va='center',
                         transform=self.ax_res.transAxes,
                         fontsize=7, color='gray')

    # ── Zoom / pan ────────────────────────────────────────────────────────

    def _open_samples_dialog(self):
        """Open a checklist dialog to include/exclude samples for the
        currently selected peak."""
        import tkinter as tk
        from tkinter import ttk
        peak_idx = self.current_region_index
        if peak_idx is None:
            return
        ctrl = self.controller
        app = getattr(ctrl.processor, 'app', None)
        if app is None:
            return
        si = getattr(app, 'samples_info', [])
        excl = set(ctrl.get_excluded_samples(peak_idx))

        dlg = tk.Toplevel(self.frame.winfo_toplevel())
        dlg.title('Sample Inclusion')
        dlg.geometry('320x420')
        dlg.resizable(True, True)
        dlg.transient(self.frame.winfo_toplevel())
        dlg.grab_set()

        ttk.Label(dlg,
                  text='Uncheck to exclude (artefacts)\n'
                       'Excluded regions zeroed on Save.',
                  foreground='gray', font=('Arial', 8)
                  ).pack(pady=(8, 4), padx=10, anchor='w')

        # Scrollable checklist
        cf = ttk.Frame(dlg)
        cf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        canvas = tk.Canvas(cf, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(cf, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind('<Configure>',
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        vars_ = []
        for i, s in enumerate(si):
            stype = s.get('type', s.get('label', 'Sample'))
            if str(stype).upper() == 'BLK':
                continue
            var = tk.BooleanVar(value=(i not in excl))
            vars_.append((i, var))
            row = ttk.Frame(inner)
            row.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(row, variable=var,
                            text=f"{s.get('name', 'S' + str(i))}  ({s.get('label', '')})"
                            ).pack(side=tk.LEFT, padx=4)

        # Select all / none
        br = ttk.Frame(dlg)
        br.pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(br, text='All',
                   command=lambda: [v.set(True) for _, v in vars_]
                   ).pack(side=tk.LEFT, padx=2)
        ttk.Button(br, text='None',
                   command=lambda: [v.set(False) for _, v in vars_]
                   ).pack(side=tk.LEFT, padx=2)

        def _ok():
            new_excl = {i for i, v in vars_ if not v.get()}
            ctrl.set_excluded_samples(peak_idx, new_excl)
            # Redraw with updated exclusions
            self.display_region(peak_idx, self._highlighted_child)
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(bf, text='Apply', command=_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=dlg.destroy).pack(side=tk.LEFT)


    def _toggle_edit_bounds(self):
        self._edit_bounds = self._edit_var.get()
        if self._edit_bounds:
            self._attach_span_selector()
        else:
            self._detach_span_selector()

    def _attach_span_selector(self):
        self._detach_span_selector()
        self._span_selector = SpanSelector(
            self.ax,
            onselect=self._on_span_selected,
            direction='horizontal',
            useblit=False,
            props=dict(alpha=0.25, facecolor='#f4a261'),
            interactive=False,
            drag_from_anywhere=True,
            minspan=0.001)

    def _detach_span_selector(self):
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
            except Exception:
                pass
            self._span_selector = None

    def _on_span_selected(self, x_min, x_max):
        """Update start/end ppm for the currently displayed peak."""
        if not self._edit_bounds: return
        ri = self.current_region_index
        if ri is None: return
        ppm = self.controller.processor.ppm
        peaks = self.controller.peaks
        if ri >= len(peaks): return
        peak = peaks[ri]
        # NMR axis inverted: x_max = left (higher ppm), x_min = right
        idx_a = int(np.argmin(np.abs(ppm - x_max)))
        idx_b = int(np.argmin(np.abs(ppm - x_min)))
        if idx_a > idx_b: idx_a, idx_b = idx_b, idx_a
        peak['start_ppm'] = float(ppm[idx_a])
        peak['end_ppm'] = float(ppm[idx_b])
        peak['width_ppm'] = float(abs(ppm[idx_b] - ppm[idx_a]))
        peak['width_at_half'] = peak['width_ppm']
        # Redraw
        self.display_region(ri, self._highlighted_child)

    def _on_view_mode_change(self):
        """Redraw when the user switches view mode."""
        if self.current_region_index is not None:
            self.display_region(
                self.current_region_index,
                self._highlighted_child)

    def _plot_spectrum(self, x_full, y_full, ppm_lo, ppm_hi):
        """Plot spectrum based on checked groups."""
        proc = self.controller.processor
        app = getattr(proc, 'app', None)
        raw = getattr(app, 'all_intensities', None)
        si = getattr(app, 'samples_info', []) if app else []
        g_colors = getattr(app, 'group_colors', {}) if app else {}

        if raw is None or raw.ndim < 2:
            self.ax.plot(x_full, y_full, color='#1a1a2e', linewidth=1.5, label='Mean', zorder=3)
            return {}, {}

        mask = (proc.ppm >= ppm_lo) & (proc.ppm <= ppm_hi)
        x_seg = proc.ppm[mask]

        shown_samples = []
        shown_groups = {}

        for i in range(raw.shape[1]):
            s = si[i] if i < len(si) else {}
            lbl = s.get('label', 'Unknown')
            if hasattr(self, 'group_vars') and self.group_vars:
                if lbl in self.group_vars and self.group_vars[lbl].get():
                    shown_samples.append(i)
                    shown_groups.setdefault(lbl, []).append(i)
            else:
                if str(s.get('type', lbl)).upper() != 'BLK':
                    shown_samples.append(i)
                    shown_groups.setdefault(lbl, []).append(i)

        cmap = plt.cm.tab20
        # Plot individual samples
        for i in shown_samples:
            s_lbl = si[i].get('label', 'Unknown') if i < len(si) else ''
            s_col = g_colors.get(s_lbl, cmap(i % 20))
            y_s = raw[mask, i]
            self.ax.plot(x_seg, y_s, linewidth=0.6, alpha=0.35, color=s_col, zorder=2)

        # Plot group means
        if not shown_groups:
            self.ax.plot(x_seg, y_full[mask] if len(y_full) == len(proc.ppm) else y_full, color='#1a1a2e',
                         linewidth=2.0, zorder=4, label='Mean')
        else:
            for lbl, idxs in shown_groups.items():
                if not idxs: continue
                y_mean = np.mean(raw[mask][:, idxs], axis=1)
                mean_col = g_colors.get(lbl, '#111111')
                self.ax.plot(x_seg, y_mean, color=mean_col, linewidth=2.0, alpha=0.9, zorder=4, label=f'{lbl} Mean')

        return shown_groups, g_colors

    def _on_scroll(self, event):
        if event.inaxes not in (self.ax, getattr(self, 'ax_res', None)):
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        factor = 0.85 if event.button == 'up' else 1 / 0.85
        xd = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
        yd = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
        self.ax.set_xlim(xd + (xlim[0] - xd) * factor, xd + (xlim[1] - xd) * factor)
        self.ax.set_ylim(yd + (ylim[0] - yd) * factor, yd + (ylim[1] - yd) * factor)
        self.canvas.draw_idle()




    def reset_view(self):
        """Restore original zoom (called by Ctrl+R in DeconvWindow)."""
        if self._xlim_orig:
            self.ax.set_xlim(self._xlim_orig)
            self.ax.set_ylim(self._ylim_orig)
            self.canvas.draw_idle()

    def _update_quality_card(self, results, child_index):
        r2 = results.get('r_squared', 0)
        bic = results.get('bic', 0)
        model = results.get('model', '—')

        # Global metrics
        self._set_metric('r_squared', f"{r2:.4f}",
                         _quality_color('r_squared', r2))
        self._set_metric('bic', f"{bic:.1f}", '#888888')
        self._set_metric('model', model, '#888888')

        # Per-peak (use selected child or mean)
        peaks = results.get('peaks', [])
        if child_index is not None and child_index < len(peaks):
            p = peaks[child_index]
            snr = p.get('snr', 0)
            lw = p.get('width', 0)
            self._set_metric('snr', f"{snr:.1f}",
                             _quality_color('snr', snr))
            self._set_metric('linewidth', f"{lw:.4f} ppm",
                             _quality_color('linewidth', lw))
            # Peak detail card
            status = p.get('status', SIGNAL_AVERAGE)
            sc = QUALITY_FG.get(status, QUALITY_FG[SIGNAL_AVERAGE])
            self.peak_detail_label.config(
                text=(f"Signal #{child_index + 1}  ·  {p['ppm']:.4f} ppm\n"
                      f"Height: {p['height']:.4f}  ·  "
                      f"Width: {p.get('width', 0):.4f} ppm\n"
                      f"SNR: {snr:.1f}  ·  "
                      f"Status: {status}"),
                foreground=sc)
        else:
            # Mean across peaks
            if peaks:
                snr_m = float(np.mean([p.get('snr', 0) for p in peaks]))
                lw_m = float(np.mean([p.get('width', 0) for p in peaks]))
                self._set_metric('snr',
                                 f"{snr_m:.1f} (mean)",
                                 _quality_color('snr', snr_m))
                self._set_metric('linewidth',
                                 f"{lw_m:.4f} ppm (mean)",
                                 _quality_color('linewidth', lw_m))
            self.peak_detail_label.config(
                text="Click a child signal to see details", foreground='gray')

    def _clear_quality_card(self):
        for key in self._quality_labels:
            val_lbl, ind = self._quality_labels[key]
            val_lbl.config(text='—')
            ind.config(fg='#cccccc')
        self.peak_detail_label.config(
            text="—", foreground='gray')

    def _set_metric(self, key, text, color):
        if key in self._quality_labels:
            val_lbl, ind = self._quality_labels[key]
            val_lbl.config(text=text)
            ind.config(fg=color)