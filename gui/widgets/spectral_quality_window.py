# gui/widgets/spectral_quality_window.py
"""
Spectral Quality Window.

Per-sample FWHM and asymmetry of the IS/reference signal (0 ppm), shown in a
cell grid where only the offending metric cell carries the grade colour.
Clicking a row plots that sample's reference peak with its FWHM (Hz) marked.
"""
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from processing.spectral_quality import (
    ASYMMETRY_THRESHOLDS,
    grade_color,
    _find_reference_apex,
    compute_fwhm_ppm,
)

# Map pre-computed grade strings → cell background colour.
_COL_UNKNOWN = '#f0f0f0'

_GRADE_TO_COLOR = {
    'good':    '#c8e6c9',
    'ok':      '#fff9c4',
    'poor':    '#ffcdd2',
    'unknown': _COL_UNKNOWN,
}

_BG_HDR    = '#e0e0e0'
_BG_CELL   = '#ffffff'
_BG_SELECT = '#cce8ff'


def _fmt(value, fmt='.4f'):
    """Format a float or return '—' for None/nan."""
    if value is None:
        return '—'
    try:
        import math
        if math.isnan(value):
            return '—'
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return '—'


class SpectralQualityWindow:
    """
    Toplevel window displaying per-sample spectral quality metrics next to a
    plot of the selected sample's reference peak.

    Parameters
    ----------
    parent           : tk parent widget (e.g. app.root)
    results          : list of dicts from assess_all_samples()
    spectrometer_mhz : if known, the FWHM (Hz) column/plot is shown
    ppm              : 1-D chemical shift axis (needed for the peak plot)
    intensities_2d   : 2-D intensity matrix (n_pts, n_samples) for the plot
    ref_ppm          : expected reference position (default 0.0)
    window_ppm       : analysis window half-width (default 0.15)
    """

    def __init__(self, parent, results: list, spectrometer_mhz=None,
                 ppm=None, intensities_2d=None,
                 ref_ppm: float = 0.0, window_ppm: float = 0.15):
        self._results  = results
        self._mhz      = spectrometer_mhz
        self._show_hz  = spectrometer_mhz is not None
        self._ref_ppm    = ref_ppm
        self._window_ppm = window_ppm

        self._ppm = None if ppm is None else np.asarray(ppm, dtype=float)
        if intensities_2d is None:
            self._intensities = None
        else:
            arr = np.asarray(intensities_2d, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            self._intensities = arr

        self._selected_row = None
        self._row_cells    = []   # [row] → list of cell Labels

        self.win = tk.Toplevel(parent)
        self.win.title('Spectral Quality — IS/Reference Signal (0 ppm)')
        self.win.geometry('1320x520')
        self.win.minsize(960, 320)
        self.win.resizable(True, True)

        self._build()
        self._populate()

        self.win.transient(parent)
        self.win.focus_set()

        if self._results:
            self._select_row(0)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        hdr = ttk.Frame(self.win, padding=(8, 6, 8, 0))
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text='Spectral Quality — IS/Reference Signal at 0 ppm',
                  font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # Column definitions (Warnings column removed)
        cols    = ['Sample', 'Group', 'IS (ppm)', 'FWHM (ppm)', 'Asymmetry']
        col_ids = ['sample', 'group', 'is_ppm', 'fwhm_ppm', 'asymmetry']
        widths  = [170, 80, 90, 110, 110]
        if self._show_hz:
            cols.insert(4, 'FWHM (Hz)')
            col_ids.insert(4, 'fwhm_hz')
            widths.insert(4, 110)

        self._cols    = cols
        self._col_ids = col_ids
        self._widths  = widths
        # Total pixel width of the table (column minsizes + 1px border each side)
        self._table_w = sum(widths) + 2 * len(widths)

        # Split: table (left, fixed width) | peak plot (right, expands)
        paned = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(paned)
        paned.add(left, weight=0)

        # Layout: header (row 0) + canvas (row 1) stacked in col 0;
        # scrollbar spans both rows in col 1 so it reaches the top.
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=0)

        # ── Frozen header (stays visible while the body scrolls) ───────────────
        self._header = tk.Frame(left, bg='white')
        self._header.grid(row=0, column=0, sticky='ew')

        # ── Vertically-scrollable cell grid ───────────────────────────────────
        self._canvas = tk.Canvas(left, highlightthickness=0, bg='white',
                                 width=self._table_w)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)

        self._canvas.grid(row=1, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, rowspan=2, sticky='ns')

        self._grid = tk.Frame(self._canvas, bg='white')
        self._canvas.create_window((0, 0), window=self._grid, anchor='nw')
        self._grid.bind('<Configure>', lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox('all')))
        self._canvas.bind_all(
            '<MouseWheel>',
            lambda e: self._canvas.yview_scroll(-1 * (e.delta // 120), 'units'))

        # Keyboard row navigation
        self.win.bind('<Up>',   lambda e: self._move_selection(-1))
        self.win.bind('<Down>', lambda e: self._move_selection(1))

        # ── Peak plot ─────────────────────────────────────────────────────────
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._fig = Figure(figsize=(4.2, 3.4), dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._plot_canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_plot()

        # ── Legend ────────────────────────────────────────────────────────────
        leg_frame = ttk.Frame(self.win, padding=(8, 0, 8, 4))
        leg_frame.pack(fill=tk.X)
        ttk.Label(leg_frame, text='Legend:', font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        for color, text in [
            ('#c8e6c9', '  Good  '),
            ('#fff9c4', '  Acceptable  '),
            ('#ffcdd2', '  Poor  '),
            (_COL_UNKNOWN, '  Not detected  '),
        ]:
            tk.Label(leg_frame, text=text, bg=color,
                     font=('Arial', 8), relief=tk.SOLID, bd=1).pack(side=tk.LEFT, padx=3)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.win, padding=(8, 2, 8, 6))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text='Export CSV',
                   command=self._export_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text='Close',
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=2)

    # ── Population ────────────────────────────────────────────────────────────

    def _populate(self):
        """Build the cell grid; colour only the offending metric cells."""
        for child in self._grid.winfo_children():
            child.destroy()
        for child in self._header.winfo_children():
            child.destroy()
        self._row_cells = []

        # Identical fixed column widths on both frames → header aligns with data
        for frame in (self._header, self._grid):
            for c, w_px in enumerate(self._widths):
                frame.columnconfigure(c, minsize=w_px, weight=0)

        # Frozen header row (built in the non-scrolling header frame)
        for c, lbl in enumerate(self._cols):
            anchor = 'w' if self._col_ids[c] in ('sample', 'group') else 'center'
            tk.Label(self._header, text=lbl, bg=_BG_HDR, font=('Arial', 9, 'bold'),
                     anchor=anchor, relief='solid', bd=1, padx=4
                     ).grid(row=0, column=c, sticky='nsew')

        # Data rows
        for ri, r in enumerate(self._results):
            has_ref = r.get('has_reference', False)

            is_ppm_str   = _fmt(r.get('apex_ppm'), '.5f')
            fwhm_ppm_str = _fmt(r.get('fwhm_ppm'), '.5f')
            asym_str     = _fmt(r.get('asymmetry'), '.4f')

            fwhm_col = (_GRADE_TO_COLOR.get(r.get('fwhm_grade', 'unknown'), _COL_UNKNOWN)
                        if has_ref else _COL_UNKNOWN)
            asym_col = (grade_color(r.get('asymmetry'), ASYMMETRY_THRESHOLDS)
                        if has_ref else _COL_UNKNOWN)

            cells = [
                (r.get('sample_name', ''),  _BG_CELL, 'w'),
                (r.get('sample_group', ''), _BG_CELL, 'w'),
                (is_ppm_str,                _BG_CELL, 'center'),
                (fwhm_ppm_str,              fwhm_col, 'center'),
            ]
            if self._show_hz:
                cells.append((_fmt(r.get('fwhm_hz'), '.2f'), fwhm_col, 'center'))
            cells.append((asym_str, asym_col, 'center'))

            row_widgets = []
            for c, (text, bg, anchor) in enumerate(cells):
                lbl = tk.Label(self._grid, text=text, bg=bg, anchor=anchor,
                               relief='solid', bd=1, padx=4)
                lbl.grid(row=ri, column=c, sticky='nsew')
                lbl.bind('<Button-1>', lambda e, row=ri: self._select_row(row))
                lbl._base_bg = bg
                row_widgets.append(lbl)
            self._row_cells.append(row_widgets)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select_row(self, row: int):
        if row < 0 or row >= len(self._row_cells):
            return
        # Restore previous row to its grade colours
        if self._selected_row is not None and self._selected_row < len(self._row_cells):
            for lbl in self._row_cells[self._selected_row]:
                lbl.configure(background=lbl._base_bg)
        # Highlight the whole selected row
        for lbl in self._row_cells[row]:
            lbl.configure(background=_BG_SELECT)
        self._selected_row = row
        self._ensure_visible(row)
        self._plot_sample(row)

    def _move_selection(self, delta: int):
        if not self._row_cells:
            return 'break'
        cur = self._selected_row if self._selected_row is not None else 0
        self._select_row(max(0, min(len(self._row_cells) - 1, cur + delta)))
        return 'break'

    def _ensure_visible(self, row: int):
        """Scroll the canvas so the selected row stays in view."""
        if not self._row_cells:
            return
        self._canvas.update_idletasks()
        cell = self._row_cells[row][0]
        top    = cell.winfo_y()
        bottom = top + cell.winfo_height()
        total  = max(1, self._grid.winfo_height())
        view_top    = self._canvas.canvasy(0)
        view_bottom = view_top + self._canvas.winfo_height()
        if top < view_top:
            self._canvas.yview_moveto(top / total)
        elif bottom > view_bottom:
            self._canvas.yview_moveto(
                (bottom - self._canvas.winfo_height()) / total)

    # ── Peak plot ───────────────────────────────────────────────────────────────

    def _draw_empty_plot(self, message='Select a sample'):
        self._ax.clear()
        self._ax.text(0.5, 0.5, message, ha='center', va='center',
                      transform=self._ax.transAxes, color='gray', fontsize=9)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._fig.tight_layout()
        self._plot_canvas.draw_idle()

    def _plot_sample(self, row: int):
        if self._ppm is None or self._intensities is None:
            self._draw_empty_plot('No spectral data')
            return
        if row >= self._intensities.shape[1]:
            self._draw_empty_plot('No spectral data')
            return

        r = self._results[row]
        if not r.get('has_reference', False):
            self._draw_empty_plot('Reference not detected')
            return

        ppm = self._ppm
        y   = self._intensities[:, row]

        apex_idx, apex_ppm = _find_reference_apex(
            ppm, y, self._ref_ppm, self._window_ppm)
        if apex_idx is None:
            self._draw_empty_plot('Reference not detected')
            return

        fres = compute_fwhm_ppm(ppm, y, apex_idx,
                                window_ppm=self._window_ppm,
                                spectrometer_mhz=self._mhz)

        # Local window around apex for display
        lo = apex_ppm - self._window_ppm
        hi = apex_ppm + self._window_ppm
        mask = (ppm >= lo) & (ppm <= hi)
        ppm_w = ppm[mask]
        y_w   = y[mask]
        order = np.argsort(ppm_w)
        ppm_w = ppm_w[order]
        y_w   = y_w[order]

        self._ax.clear()
        self._ax.plot(ppm_w, y_w, color='#1565c0', lw=1.2)

        left_ppm  = fres.get('left_ppm')
        right_ppm = fres.get('right_ppm')
        fwhm_hz   = fres.get('fwhm_hz')
        fwhm_ppm  = fres.get('fwhm_ppm')

        if left_ppm is not None and right_ppm is not None:
            # Half-max level relative to baseline-corrected apex
            apex_val = float(np.max(y_w))
            base_val = float(min(y_w[0], y_w[-1]))
            half_y   = base_val + (apex_val - base_val) / 2.0
            self._ax.hlines(half_y, min(left_ppm, right_ppm),
                            max(left_ppm, right_ppm),
                            color='#c62828', lw=1.0, linestyles='--')
            for x in (left_ppm, right_ppm):
                self._ax.axvline(x, color='#c62828', lw=0.8, alpha=0.6)

            if fwhm_hz is not None:
                label = f'FWHM = {fwhm_hz:.2f} Hz'
            else:
                label = f'FWHM = {fwhm_ppm:.4f} ppm'
            self._ax.text(
                0.02, 0.96, label,
                transform=self._ax.transAxes,
                ha='left', va='top', fontsize=9, color='#c62828',
                fontweight='bold')

        title = r.get('sample_name', '')
        self._ax.set_title(title, fontsize=9)
        self._ax.set_xlabel('δ (ppm)', fontsize=8)
        self._ax.set_ylabel('Intensity (au)', fontsize=8)
        self._ax.invert_xaxis()   # ppm convention
        self._ax.tick_params(labelsize=7)
        self._fig.tight_layout()
        self._plot_canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Export Spectral Quality',
        )
        if not path:
            return
        try:
            headers = ['Sample', 'Group', 'IS_ppm', 'FWHM_ppm', 'FWHM_grade',
                       'Asymmetry', 'Asymmetry_grade', 'Has_Reference', 'Warnings']
            if self._show_hz:
                headers.insert(4, 'FWHM_Hz')

            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                for r in self._results:
                    row = [
                        r.get('sample_name', ''),
                        r.get('sample_group', ''),
                        _fmt(r.get('apex_ppm'), '.5f'),
                        _fmt(r.get('fwhm_ppm'), '.5f'),
                        r.get('fwhm_grade', ''),
                    ]
                    if self._show_hz:
                        row.append(_fmt(r.get('fwhm_hz'), '.3f'))
                    row += [
                        _fmt(r.get('asymmetry'), '.4f'),
                        r.get('asymmetry_grade', ''),
                        str(r.get('has_reference', False)),
                        '; '.join(r.get('warnings', [])),
                    ]
                    writer.writerow(row)

            messagebox.showinfo('Export', f'Saved to:\n{path}', parent=self.win)
        except Exception as exc:
            messagebox.showerror('Export Error', str(exc), parent=self.win)
