# gui/widgets/spectral_quality_window.py
"""
Spectral Quality Window.

Shows FWHM and Asymmetry of the IS/reference signal (0 ppm) for every sample,
with a red-yellow-green colour gradient per cell.
"""
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from processing.spectral_quality import (
    ASYMMETRY_THRESHOLDS,
    grade_color,
)

# Map pre-computed grade strings → row background colour.
# FWHM grading is done in processing layer (Hz when MHz known, ppm otherwise).
_COL_UNKNOWN = '#f0f0f0'

_GRADE_TO_COLOR = {
    'good':    '#c8e6c9',
    'ok':      '#fff9c4',
    'poor':    '#ffcdd2',
    'unknown': _COL_UNKNOWN,
}


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
    Toplevel window displaying per-sample spectral quality metrics.

    Parameters
    ----------
    parent          : tk parent widget (e.g. app.root)
    results         : list of dicts from assess_all_samples()
    spectrometer_mhz: if known, the FWHM (Hz) column is shown
    """

    def __init__(self, parent, results: list, spectrometer_mhz=None):
        self._results        = results
        self._mhz            = spectrometer_mhz
        self._show_hz        = spectrometer_mhz is not None

        self.win = tk.Toplevel(parent)
        self.win.title('Spectral Quality — IS/Reference Signal (0 ppm)')
        self.win.geometry('820x480')
        self.win.minsize(600, 300)
        self.win.resizable(True, True)

        self._build()
        self._populate()

        self.win.transient(parent)
        self.win.focus_set()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Title bar ─────────────────────────────────────────────────────────
        hdr = ttk.Frame(self.win, padding=(8, 6, 8, 0))
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text='Spectral Quality — IS/Reference Signal at 0 ppm',
                  font=('Arial', 11, 'bold')).pack(side=tk.LEFT)

        # ── Columns ───────────────────────────────────────────────────────────
        cols = ['Sample', 'Group', 'IS (ppm)', 'FWHM (ppm)', 'Asymmetry']
        col_ids = ['sample', 'group', 'is_ppm', 'fwhm_ppm', 'asymmetry']
        widths  = [180, 80, 90, 110, 110]
        if self._show_hz:
            cols.insert(4, 'FWHM (Hz)')
            col_ids.insert(4, 'fwhm_hz')
            widths.insert(4, 110)
        cols.append('Warnings')
        col_ids.append('warnings')
        widths.append(200)

        self._col_ids = col_ids

        # ── Treeview + scrollbars ─────────────────────────────────────────────
        frame = ttk.Frame(self.win, padding=(8, 4))
        frame.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)

        self.tree = ttk.Treeview(
            frame,
            columns=col_ids,
            show='headings',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        for cid, lbl, w in zip(col_ids, cols, widths):
            self.tree.heading(cid, text=lbl)
            anchor = 'center' if cid not in ('sample', 'group', 'warnings') else 'w'
            self.tree.column(cid, width=w, minwidth=60, anchor=anchor)

        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

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
            lbl = tk.Label(leg_frame, text=text, bg=color,
                           font=('Arial', 8), relief=tk.SOLID, bd=1)
            lbl.pack(side=tk.LEFT, padx=3)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self.win, padding=(8, 2, 8, 6))
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text='Export CSV',
                   command=self._export_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text='Close',
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=2)

    # ── Population ────────────────────────────────────────────────────────────

    def _populate(self):
        """Fill the Treeview with results and apply per-cell background tags."""
        self.tree.delete(*self.tree.get_children())

        # ttk.Treeview does not support per-cell background natively.
        # We use row-level tags keyed by the worst metric of each row.
        # For fine-grained colouring we create one tag per unique colour.
        _tag_registry = {}

        def _get_tag(color):
            if color not in _tag_registry:
                tag = f'clr_{len(_tag_registry)}'
                self.tree.tag_configure(tag, background=color)
                _tag_registry[color] = tag
            return _tag_registry[color]

        for r in self._results:
            has_ref = r.get('has_reference', False)

            is_ppm_str   = _fmt(r.get('apex_ppm'), '.5f')
            fwhm_ppm_str = _fmt(r.get('fwhm_ppm'), '.5f')
            asym_str     = _fmt(r.get('asymmetry'), '.4f')
            warn_str     = '; '.join(r.get('warnings', [])) or ''

            # Use pre-computed fwhm_grade (graded in Hz when MHz known)
            fwhm_col = (_GRADE_TO_COLOR.get(r.get('fwhm_grade', 'unknown'), _COL_UNKNOWN)
                        if has_ref else _COL_UNKNOWN)
            asym_col = (grade_color(r.get('asymmetry'), ASYMMETRY_THRESHOLDS)
                        if has_ref else _COL_UNKNOWN)

            # Row background = worst metric colour
            # Priority: red > yellow > green > grey
            def _priority(c):
                if c == '#ffcdd2': return 3
                if c == '#f0f0f0': return 2
                if c.startswith('#fff'): return 1
                return 0

            row_col = max([fwhm_col, asym_col], key=_priority)
            row_tag = _get_tag(row_col)

            values = [
                r.get('sample_name', ''),
                r.get('sample_group', ''),
                is_ppm_str,
                fwhm_ppm_str,
            ]
            if self._show_hz:
                values.append(_fmt(r.get('fwhm_hz'), '.2f'))
            values.append(asym_str)
            values.append(warn_str)

            self.tree.insert('', tk.END, values=values, tags=(row_tag,))

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
