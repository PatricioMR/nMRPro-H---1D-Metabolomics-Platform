# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/widgets/stocsy_window.py
"""
STOCSY window -- an analyte's signals, and which peak of a window quantifies
like its reference.

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic
    1H NMR data sets. Analytical Chemistry 77(5):1282-1289.

An audit of the quantification. Pick a quantified analyte on the left: the
reference is what the quantification measured for it (its per-sample areas and
protons, whatever the mode), on the listed signal marked as the one it
measured, and its listed signals appear -- the ones the annotation found (a
multiplet whole, its lines added) and the ones it did not -- each set against
that reference. Click a signal and every detected peak
around it is read as a possible line of it, with the share of the signal's
protons the line holds: r and ratio to the reference, the ratio its share
expects, how far what it quantifies is from what the reference quantifies.
The ticked peaks make up the signal, compared whole. Tick and untick to try
other lines; "Save the ticked lines" writes them on the annotation
(``controllers.signal_lines``), to be measured at the next quantification.
Without an analyte any detected peak can be the reference, counted as 1 H.

Correlation, normalisation (PQN) and within-group centring are set once, on
the left, and apply to everything on screen alike. "Groups" draws the spectrum
as one group's mean, coloured by r over that group's samples only: whether the
signals move together inside a group, not only across the study.

Only "Save the ticked lines" and "Restore the annotation's lines" change the
project.
"""
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from controllers.analysis_defaults import ANNOTATION
from controllers.stocsy_analytes import AnalyteBrowser
from controllers.stocsy_controller import NOISE_FLOOR_FACTOR
from gui.utils.quality_palette import (
    QUALITY_LABEL, SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR, tier_from_annotation_quality)
from processing.quant_integration import MIN_TRUSTED_CAPTURE
from processing.signal_lines import peaks_at

_CMAP = 'RdBu_r'
_NORM = Normalize(-1, 1)
_GREY = (0.72, 0.72, 0.72, 0.6)
_REF, _SOUGHT, _OTHER = '#d62728', '#1f77b4', '#8c8c8c'
_LINES = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b',
          '#e377c2', '#bcbd22']
_DOTS = {0: '', 1: '●', 2: '●●', 3: '●●●'}
_MAX_LABELS = 18            # peak numbers written over the spectrum up to this many
_DRAWN = 6                  # ticked peaks drawn against the reference with no analyte


def _f(v, spec='.2f'):
    return '—' if v is None or not np.isfinite(v) else format(v, spec)


def _area(v):
    if v is None or not np.isfinite(v):
        return '—'
    return f'{v:,.0f}' if abs(v) >= 1000 else f'{v:.3g}'


def _h(v):
    return '—' if not v else f'{v:.3g}'


def _diff(row):
    d = row.get('distance')
    return '—' if d is None or not np.isfinite(d) else f'{100 * d:.0f}'


def _per_h(row):
    a, h = row.get('area_mean'), row.get('protons')
    return _area(a / h) if h and a is not None and np.isfinite(a) else '—'


def _quality(signal):
    """The annotation's verdict in the Annotation Results' words; a listed
    signal the annotation did not match has none."""
    if signal.get('mp_index') is None:
        return '—'
    return QUALITY_LABEL[tier_from_annotation_quality(signal['quality'] or None)]


def _expected(row, ref):
    """The ratio to the reference the row's protons expect."""
    h = row.get('protons')
    return h / ref.protons if h and ref is not None and ref.protons else np.nan


# Column of the lines table -> (heading, width, value to sort by)
_WIN_COLS = (
    ('use', 'In', 34, None),
    ('label', 'Peak', 150, lambda r: r['ppm']),
    ('ppm', 'δ (ppm)', 60, lambda r: r['ppm']),
    ('snr', 'SNR', 42, lambda r: r['snr']),
    ('share', 'Line share', 64, lambda r: r['share'] if r['share'] is not None else np.nan),
    ('h', 'H', 40, lambda r: r['protons']),
    ('r', 'r to ref.', 56, lambda r: r['r']),
    ('p', 'p', 50, lambda r: r['p']),
    ('ratio', 'Ratio to ref.', 74, lambda r: r['ratio']),
    ('exp', 'Expected ratio', 84, None),
    ('sd', 'Ratio spread', 74, lambda r: r['sd']),
    ('dq', 'Quant. diff (%)', 88, lambda r: r['distance']),
    ('area', 'Mean area', 70, lambda r: r['area_mean']),
    ('who', 'Annotated as', 120, None),
)
_QUANT = 'q'                # the reference is what the quantification measured
_ALL = 'All samples'        # the spectrum over the whole study


class StocsyWindow:
    """Toplevel over a ``StocsyAnalysis`` and an ``AnalyteBrowser``.

    ``view_region`` is (high, low) ppm of the main plot, where the reference is
    sought when no analyte is picked; ``group_colors`` maps a group label to
    its colour in the main plot; ``initial`` is a ``state()`` to open on.
    ``editor`` (a ``controllers.signal_lines.SignalLineEditor``) writes the
    analyst's lines; without it the window only reads. ``on_change(met_id)``
    is told when it did. ``show_signal(met_id, matched_peak)`` and
    ``show_peak(rank)`` take the analyst to the signal in the Annotation
    Results and to the peak in the Signal List (right-click); without them
    those entries are off."""

    def __init__(self, parent, analysis, *, browser=None, view_region=None,
                 group_colors=None, initial=None, editor=None, on_change=None,
                 tolerance_ppm=None, show_signal=None, show_peak=None):
        self.an = analysis
        self.browser = browser or AnalyteBrowser(analysis)
        self.editor, self.on_change = editor, on_change
        self.show_signal, self.show_peak = show_signal, show_peak
        # the annotation's match tolerance: how far from its listing a line is
        # looked for, and so how far around a signal the window looks
        self.tolerance = float(tolerance_ppm or ANNOTATION['tolerance_ppm'])
        self.quant_view = False                    # the lines the quantification used
        self.ticked = []                           # the peaks that make the signal
        self.changed = set()                       # analytes whose lines were saved here
        self.group_colors = dict(group_colors or {})
        self.met_id = None
        self.ref_i = self.sought_i = None          # signals of the analyte
        self.ref_rank = self.sought_rank = None    # peaks, with no analyte
        self.ref = self.sought = None              # what is compared (Target)
        self.focus = None                          # the candidate looked at
        self.ref_row, self.sig_rows, self.cand_rows = None, [], []
        self._sort = None
        self._sync = False
        self._view_ranks = []
        self.win = tk.Toplevel(parent)
        self.win.title('STOCSY — analyte signals')
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        self.win.geometry(f'{min(1440, sw - 60)}x{min(900, sh - 120)}+20+10')
        self._build()
        self._fill_analytes()
        self._open_on(initial or {}, view_region)
        self.win.protocol('WM_DELETE_WINDOW', self.win.destroy)
        for key, quality in (('1', 'good'), ('2', None), ('3', 'bad')):
            self.win.bind(f'<Control-Key-{key}>', lambda _e, q=quality: self._classify(q))
        self.show()
        self._proportions()

    def show(self):
        """Bring the window to the front: a window that opens behind the main
        one is a window nobody found."""
        self.win.deiconify()
        self.win.lift()
        self.win.attributes('-topmost', True)
        self.win.after(250, lambda: self.win.attributes('-topmost', False))
        self.win.focus_force()

    def state(self) -> dict:
        """Where the analyst is, to open on it again."""
        return {'met_id': self.met_id, 'ref_i': self.ref_i, 'sought_i': self.sought_i,
                'ref_rank': self.ref_rank, 'sought_rank': self.sought_rank}

    # ── layout ───────────────────────────────────────────────────────────────

    def _build(self):
        paned = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned, padding=6)
        right = ttk.Frame(paned, padding=(0, 6, 6, 6))
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        self._paned = paned
        self._build_left(left)
        self._build_right(right)

    def _proportions(self):
        """The analytes about a quarter of the width; the spectrum two fifths
        of the height under the header, the tables the rest."""
        self.win.update_idletasks()
        try:
            self._paned.sashpos(0, int(0.27 * self.win.winfo_width()))
            self._split.sashpos(0, int(0.40 * self._split.winfo_height()))
        except tk.TclError:
            pass

    def _build_left(self, left):
        af = ttk.LabelFrame(left, text='Analytes — click one', padding=4)
        af.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(af)
        top.pack(fill=tk.X)
        ttk.Label(top, text='Find').pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.filter_var, width=18).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.filter_var.trace_add('write', lambda *_: self._fill_analytes())
        self.only_q_var = tk.BooleanVar(value=self.browser.has_quantification)
        self.only_q_chk = ttk.Checkbutton(
            af, text='Quantified only', variable=self.only_q_var,
            command=self._fill_analytes)
        self.only_q_chk.pack(anchor=tk.W)
        if not self.browser.has_quantification:
            self.only_q_chk.state(['disabled'])
        tf = ttk.Frame(af)
        tf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.an_tree = ttk.Treeview(
            tf, columns=('dots', 'found', 'pairs', 'r'), show='tree headings',
            selectmode='browse', height=18)
        for c, t, w, a in (('#0', 'Analyte', 112, tk.W), ('dots', '', 36, tk.CENTER),
                           ('found', 'Found', 44, tk.CENTER),
                           ('pairs', 'Pairs', 38, tk.CENTER),
                           ('r', 'mean r', 46, tk.CENTER)):
            self.an_tree.heading(c, text=t)
            self.an_tree.column(c, width=w, anchor=a, stretch=(c == '#0'))
        sb = ttk.Scrollbar(tf, command=self.an_tree.yview)
        self.an_tree.config(yscrollcommand=sb.set)
        self.an_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.an_tree.tag_configure('unq', foreground='#888888')
        self.an_tree.bind('<<TreeviewSelect>>', lambda _e: self._on_analyte())
        tk.Label(af, justify=tk.LEFT, wraplength=300, fg='#555555', anchor='w',
                 font=('Arial', 8),
                 text=('Found = signals matched / signals listed. Pairs = pairs of '
                       f'its anchor peaks that co-vary ({self.browser.lift_rule()}): '
                       'the evidence that raises a confidence dot.')).pack(fill=tk.X)
        ttk.Button(af, text='No analyte — browse peaks',
                   command=self._no_analyte).pack(fill=tk.X, pady=(4, 0))

        of = ttk.LabelFrame(left, text='Correlation', padding=4)
        of.pack(fill=tk.X, pady=6)
        self.method_var = tk.StringVar(value='Pearson')
        self.norm_var = tk.StringVar(value='None (raw areas)')
        self.within_var = tk.BooleanVar(value=False)
        for row, (lab, var, vals) in enumerate((
                ('Method', self.method_var, ('Pearson', 'Spearman')),
                ('Normalise', self.norm_var, ('None (raw areas)', 'PQN')))):
            ttk.Label(of, text=lab).grid(row=row, column=0, sticky=tk.W)
            cb = ttk.Combobox(of, textvariable=var, values=vals, width=16,
                              state='readonly')
            cb.grid(row=row, column=1, sticky=tk.EW, padx=4, pady=1)
            cb.bind('<<ComboboxSelected>>', lambda _e: self._on_options())
        of.columnconfigure(1, weight=1)
        self.within_chk = ttk.Checkbutton(
            of, text='Remove group differences\n(correlate within groups)',
            variable=self.within_var, command=self._on_options)
        self.within_chk.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        if not self.an.can_center_by_group:
            self.within_chk.state(['disabled'])
        self.notes_lbl = tk.Label(left, text='', justify=tk.LEFT, wraplength=320,
                                  fg='#8a5a00', font=('Arial', 8), anchor='nw')
        self.notes_lbl.pack(fill=tk.X)
        self._show_notes()
        ttk.Button(left, text='Modules and samples…',
                   command=self._open_structure).pack(fill=tk.X, pady=(4, 0))

    def _build_right(self, right):
        hdr = ttk.Frame(right)
        hdr.pack(fill=tk.X)
        self.title_lbl = ttk.Label(hdr, text='', font=('Arial', 12, 'bold'))
        self.title_lbl.grid(row=0, column=0, columnspan=8, sticky=tk.W)
        self.ref_lbl = ttk.Label(hdr, text='', foreground=_REF, font=('Arial', 10))
        self.ref_lbl.grid(row=1, column=0, columnspan=8, sticky=tk.W)
        self.sought_lbl = ttk.Label(hdr, text='', foreground=_SOUGHT, font=('Arial', 10))
        self.sought_lbl.grid(row=2, column=0, columnspan=8, sticky=tk.W)
        ttk.Label(hdr, text='Window ±').grid(row=3, column=0, sticky=tk.E, pady=3)
        self.hw_var = tk.StringVar(value=f'{self.tolerance:g}')
        hw = ttk.Spinbox(hdr, from_=0.02, to=2.0, increment=0.01, width=5,
                         textvariable=self.hw_var, command=self._rebuild)
        hw.grid(row=3, column=1, padx=2)
        hw.bind('<Return>', lambda _e: self._rebuild())
        ttk.Label(hdr, text='ppm').grid(row=3, column=2, padx=(0, 14))
        ttk.Label(hdr, text='Groups').grid(row=3, column=3, padx=2)
        self.group_var = tk.StringVar(value=_ALL)
        self.group_cb = ttk.Combobox(hdr, textvariable=self.group_var, width=18,
                                     values=[_ALL] + self.an.group_names(), state='readonly')
        self.group_cb.grid(row=3, column=4, padx=(0, 10))
        self.group_cb.bind('<<ComboboxSelected>>', lambda _e: self._draw_spectrum())
        ttk.Label(hdr, text='(the spectrum below: the mean of these samples, r over them)',
                  foreground='#777777', font=('Arial', 8)).grid(row=3, column=5, padx=4)
        self.color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hdr, text='Colour the spectrum by r to the reference',
                        variable=self.color_var, command=self._draw_spectrum
                        ).grid(row=4, column=0, columnspan=4, sticky=tk.W)
        if self.editor is not None:
            ttk.Label(hdr, foreground='#0044cc', font=('Arial', 9, 'bold'), text=(
                f"Quality of the signal read:  Ctrl+1 {QUALITY_LABEL[SIGNAL_GOOD]}  ·  "
                f"Ctrl+2 {QUALITY_LABEL[SIGNAL_AVERAGE]}  ·  Ctrl+3 {QUALITY_LABEL[SIGNAL_POOR]}"
                "  —  the same mark as in the Annotation Results")
                      ).grid(row=4, column=4, columnspan=4, sticky=tk.W, padx=(10, 0))

        split = ttk.PanedWindow(right, orient=tk.VERTICAL)
        split.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self._split = split
        top = ttk.Frame(split)
        split.add(top, weight=1)
        self.fig_spec = Figure(figsize=(9.5, 2.6))
        self.cv_spec = FigureCanvasTkAgg(self.fig_spec, master=top)
        self.cv_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_spec = self.fig_spec.add_axes([0.07, 0.16, 0.86, 0.68])
        self.cax = self.fig_spec.add_axes([0.945, 0.16, 0.01, 0.68])
        self.fig_spec.colorbar(ScalarMappable(_NORM, _CMAP), cax=self.cax
                               ).set_label('r to the reference', fontsize=8)
        self.cv_spec.mpl_connect('button_press_event', self._on_spec_click)

        self.nb = ttk.Notebook(split)
        split.add(self.nb, weight=2)
        self._build_signals_tab()
        self._build_window_tab()
        self._build_values_tab()
        self._build_variation_tab()
        self._build_matrix_tab()

    def _tab(self, title, hint):
        fr = ttk.Frame(self.nb, padding=6)
        self.nb.add(fr, text=title)
        lbl = tk.Label(fr, text=hint, justify=tk.LEFT, wraplength=900, anchor='w',
                       fg='#555555', font=('Arial', 8))
        lbl.pack(fill=tk.X, pady=(0, 4))
        lbl.bind('<Configure>', lambda e: lbl.config(wraplength=max(e.width - 12, 200)))
        return fr

    def _tree(self, parent, cols, height):
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(fr, columns=[c[0] for c in cols], show='headings',
                            height=height, selectmode='browse')
        for cid, title, width in cols:
            tree.heading(cid, text=title)
            tree.column(cid, width=width, minwidth=24, anchor=tk.CENTER, stretch=False)
        vs = ttk.Scrollbar(fr, command=tree.yview)
        hs = ttk.Scrollbar(fr, orient=tk.HORIZONTAL, command=tree.xview)
        tree.config(yscrollcommand=vs.set, xscrollcommand=hs.set)
        hs.pack(side=tk.BOTTOM, fill=tk.X)          # Windows scaling widens the columns
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree.tag_configure('ref', background='#fde0dd', font=('Arial', 9, 'bold'))
        tree.tag_configure('sought', background='#e3f0fb')
        tree.tag_configure('best', background='#dff0d8')
        tree.tag_configure('weak', foreground='#999999')
        return tree

    def _build_signals_tab(self):
        fr = self._tab('Analyte signals',
                       'The signals the analyte\'s listing expects. The one the quantification measured is '
                       'marked ◉ in red: the reference is what it measured there, its per-sample areas and the '
                       'H it divided them by, and every signal is set against it. Click it to see the region '
                       'and the lines the quantification integrated; click another to read its lines (▶, blue) '
                       'in the next tab -- one not detected too: the peaks around where it is listed. '
                       'Double-click a signal to use it as the reference instead, the ◉ one to go back. The '
                       'arrow keys move through the list; Ctrl+1 / 2 / 3 mark the signal read Good / Average / '
                       'Poor, as in the Annotation Results (Good with other lines ticked saves them as its '
                       'lines); right-click takes a signal to the Annotation Results. A multiplet is its lines '
                       'added up; ✎ = lines chosen '
                       'by hand. Δδ and ΔJ are the annotation\'s: how far the signal stands from its listing '
                       'and how far its splitting is from the listed J. '
                       'Quant. diff = the typical difference, per sample, between what the signal '
                       'quantifies (area ÷ H) and what the reference quantifies: 0 means the same '
                       'quantification in every sample.')
        self.sig_tree = self._tree(fr, (
            ('mark', 'Ref', 34), ('db', 'δ listed', 66), ('ppm', 'δ found', 66),
            ('dd', 'Δδ (ppm)', 64), ('dj', 'ΔJ (Hz)', 58), ('h', 'H', 34),
            ('type', 'Type', 46), ('lines', 'Lines', 42), ('q', 'Annotation', 72),
            ('snr', 'SNR', 50), ('r', 'r to ref.', 60), ('ratio', 'Ratio to ref.', 76),
            ('dq', 'Quant. diff (%)', 92), ('why', 'Note', 160)), 8)
        self.sig_tree.bind('<<TreeviewSelect>>', lambda _e: self._on_sig_select())
        self.sig_tree.bind('<Double-1>', lambda _e: self._ref_from_signal())
        self.sig_tree.bind('<Button-3>', self._sig_menu)

    def _build_window_tab(self):
        fr = self._tab('Lines of the signal',
                       'Every detected peak in the window around the signal, read as a possible line of it. '
                       'Line share = the part of the signal\'s pattern that line holds, as the quantification '
                       'weighs it (binomial by line count, a quartet 1:3:3:1; a measured pattern\'s own heights; '
                       '— when the peak stands on no predicted line); on the first row, the part the ticked '
                       'lines hold, which must reach 75% for the quantification to scale them to the whole '
                       'signal. Expected ratio = what the reference predicts for that H; a line of the analyte '
                       'has r near 1, its ratio near the expected one and a small spread; another compound\'s '
                       'line does not. The first row is the signal the ticked lines make, whole. Click "In '
                       'signal" (or double-click, or Space) to tick or untick a peak; the numbers follow. Click '
                       'a heading to sort. Right-click a peak to see it in the Signal List: a signal split or '
                       'added there appears here at once.')
        self.win_tree = self._tree(fr, [(c, t, w) for c, t, w, _k in _WIN_COLS], 10)
        for cid, title, _w, key in _WIN_COLS:
            if key is not None:
                self.win_tree.heading(cid, text=title, command=lambda c=cid: self._sort_by(c))
        self.win_tree.column('who', anchor=tk.W)
        self.win_tree.column('label', anchor=tk.W)
        self.win_tree.tag_configure('ticked', background='#e3f0fb')
        self.win_tree.bind('<<TreeviewSelect>>', lambda _e: self._on_win_select())
        self.win_tree.bind('<Button-1>', self._on_win_click, add='+')
        self.win_tree.bind('<Double-1>', lambda _e: self._toggle_selected())
        self.win_tree.bind('<space>', lambda _e: self._toggle_selected())
        self.win_tree.bind('<Button-3>', self._win_menu)
        bf = ttk.Frame(fr)
        bf.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(bf, text='Tick / untick',
                   command=self._toggle_selected).pack(side=tk.LEFT)
        self.btn_save_lines = ttk.Button(bf, text='Save the ticked lines as this signal…',
                                         command=self._save_lines)
        self.btn_save_lines.pack(side=tk.LEFT, padx=6)
        self.btn_restore_lines = ttk.Button(bf, text="Restore the annotation's lines",
                                            command=self._restore_lines)
        self.btn_restore_lines.pack(side=tk.LEFT)
        ttk.Button(bf, text='Use the selected peak as reference',
                   command=self._ref_from_candidate).pack(side=tk.LEFT, padx=6)
        ttk.Button(bf, text='Search around the selected peak',
                   command=self._sought_from_candidate).pack(side=tk.LEFT)
        ttk.Button(bf, text='Export this table (CSV)…',
                   command=self._export).pack(side=tk.LEFT)

    def _build_values_tab(self):
        fr = self._tab('Values per sample',
                       'The numbers behind the table: the reference and each candidate in every sample. '
                       '"Area ÷ H" divides each by its H: the candidate that is the sought signal then '
                       'reads the same as the reference in every sample.')
        top = ttk.Frame(fr)
        top.pack(fill=tk.X)
        self.per_var = tk.StringVar(value='area')
        for text, val in (('Area', 'area'), ('Area ÷ H', 'proton')):
            ttk.Radiobutton(top, text=text, variable=self.per_var, value=val,
                            command=self._fill_values).pack(side=tk.LEFT, padx=6)
        self.val_frame = ttk.Frame(fr)
        self.val_frame.pack(fill=tk.BOTH, expand=True)

    def _build_variation_tab(self):
        fr = self._tab('Variation across samples',
                       'Area ÷ H of each signal in every sample, samples grouped. Lines on top of one another '
                       'quantify the same. Black: the reference. Without an analyte: the nearest candidates. '
                       'Dashed red: the peak selected in the table or the spectrum.')
        self.fig_var = Figure(figsize=(9.5, 3.2))
        self.cv_var = FigureCanvasTkAgg(self.fig_var, master=fr)
        self.cv_var.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_var = self.fig_var.add_subplot(111)

    def _build_matrix_tab(self):
        fr = self._tab('Correlation among signals',
                       'r between the analyte\'s signals (without an analyte: the reference and its nearest '
                       'candidates). Bold: p < 0.05.')
        self.fig_mat = Figure(figsize=(6.0, 3.6))
        self.cv_mat = FigureCanvasTkAgg(self.fig_mat, master=fr)
        self.cv_mat.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_mat = self.fig_mat.add_subplot(111)

    # ── opening ──────────────────────────────────────────────────────────────

    def _open_on(self, initial, view_region):
        mets = self.browser.analytes(only_quantified=self.only_q_var.get())
        met = initial.get('met_id')
        if met is None and mets and not initial.get('ref_rank'):
            met = mets[0]['id']
        if met is not None and self.an_tree.exists(str(met)):
            self.met_id = met
            self._quietly(lambda: (self.an_tree.selection_set(str(met)), self.an_tree.focus(str(met)),
                                   self.an_tree.see(str(met))))
            n = len(self.browser.signals(met))
            ref = initial.get('ref_i')
            keep = ((ref == _QUANT and self.browser.quant_target(met) is not None)
                    or (isinstance(ref, int) and ref < n))
            self.ref_i = ref if keep else self._default_reference()
            sought = initial.get('sought_i')
            if isinstance(sought, int) and sought < n:
                self._set_sought(sought)
            elif not self._show_quant():
                self._set_sought(self._default_sought())
            self._rebuild()
            return
        self.met_id = None
        self.ref_rank = initial.get('ref_rank') or self._tallest(view_region)
        self.sought_rank = initial.get('sought_rank') or self.ref_rank
        self.ticked = [self.sought_rank] if self.sought_rank is not None else []
        self._rebuild()

    def _tallest(self, region):
        ranks = [r for r in self.an.ranks if self.an.can_seed(r)]
        if region is not None:
            hi, lo = max(region), min(region)
            ranks = [r for r in ranks if lo <= self.an.ppm_of(r) <= hi] or ranks
        return max(ranks, key=lambda r: float(self.an.peaks[r].get('snr') or 0.0), default=None)

    def _default_reference(self):
        """What the quantification measured, when the analyte was quantified;
        else its clearest listed signal."""
        if self.browser.quant_target(self.met_id) is not None:
            return _QUANT
        return self.browser.default_reference(self.met_id)

    def _default_sought(self):
        """The signal read first: the listed signal nearest the δ the
        quantification reported, else the clearest one that is not the
        reference, or the reference itself when it is the only one."""
        sigs = self.browser.signals(self.met_id)
        usable = [i for i, s in enumerate(sigs) if s['usable'] and i != self.ref_i]
        row = self.browser.quantification(self.met_id)
        at = float((row or {}).get('ppm') or np.nan)
        if usable and np.isfinite(at):
            return min(usable, key=lambda i: abs(sigs[i]['ppm'] - at))
        best = None
        for i in usable:
            key = (sigs[i]['snr'], sigs[i]['protons'] or 0)
            if best is None or key > best[0]:
                best = (key, i)
        return best[1] if best else (self.ref_i if isinstance(self.ref_i, int) else None)

    def _set_sought(self, i):
        """Read the ``i``-th signal of the analyte; its own lines start ticked
        and the window is the annotation's tolerance around its pattern."""
        self.sought_i = i
        self.quant_view = False
        if not self.met_id or i is None:
            self.ticked = []
            return
        s = self.browser.signals(self.met_id)[i]
        self.ticked = list(s['lines'])
        pred = [float(v) for v in (s.get('evidence') or {}).get('predicted_ppms') or ()
                if v is not None]
        half = (max(pred) - min(pred)) / 2 if pred else 0.0
        self.hw_var.set(f'{self.tolerance + half:.3f}')

    def _quant_index(self):
        """The listed signal the quantification measured: the one nearest the
        δ it reported; None without a quantification of the analyte."""
        if not self.met_id or self.browser.quant_target(self.met_id) is None:
            return None
        row = self.browser.quantification(self.met_id) or {}
        at = float(row.get('ppm') or np.nan)
        sigs = self.browser.signals(self.met_id)
        found = [i for i, s in enumerate(sigs) if np.isfinite(s['ppm'])]
        if not np.isfinite(at) or not found:
            return None
        return min(found, key=lambda i: abs(sigs[i]['ppm'] - at))

    def _show_quant(self) -> bool:
        """What the quantification measured, in the spectrum: its signal, its
        window, and the lines it integrated ticked. False when there is none
        to show."""
        qi = self._quant_index()
        if qi is None:
            return False
        row = self.browser.quantification(self.met_id) or {}
        at = float(row.get('ppm'))
        self._set_sought(qi)
        win = row.get('_multiplet_window')
        lo, hi = (sorted(float(v) for v in win) if win and len(win) == 2
                  else (at - self._hw(), at + self._hw()))
        self.hw_var.set(f'{max(hi - at, at - lo) + self.tolerance / 10:.3f}')
        lines = [float(x) for x in row.get('_line_ppms') or () if lo <= float(x) <= hi]
        self.ticked = [int(p['rank']) for p in peaks_at(lines, self.an.peaks)]
        self.quant_view = True
        return True

    # ── analytes list ────────────────────────────────────────────────────────

    def _fill_analytes(self):
        keep = self.met_id
        self.an_tree.delete(*self.an_tree.get_children())
        quant = self.browser.has_quantification
        rows = self.browser.analytes(only_quantified=self.only_q_var.get(),
                                     text=self.filter_var.get())
        for a in rows:
            hand = any(s['manual'] for s in self.browser.signals(a['id']))
            self.an_tree.insert(
                '', tk.END, iid=str(a['id']), text=a['name'] + ('  ✎' if hand else ''),
                tags=('unq',) if quant and not a['dots'] else (),
                values=(_DOTS[a['dots']], f"{a['detected']}/{a['expected']}",
                        a['pairs'] or '', _f(a['mean_r'])))
        if not rows:
            self.an_tree.insert(
                '', tk.END, iid='-', tags=('unq',),
                text=('No annotation here: pick the peaks in the spectrum.'
                      if not self.browser.has_annotation else 'No analyte matches.'))
        if keep is not None and self.an_tree.exists(str(keep)):
            self._quietly(lambda: (self.an_tree.selection_set(str(keep)),
                                   self.an_tree.focus(str(keep))))

    def _on_analyte(self):
        sel = self.an_tree.selection()
        if self._sync or not sel or sel[0] == '-' or sel[0] == self.met_id:
            return
        self.met_id = sel[0]
        self.ref_i = self._default_reference()
        self.focus = None
        if not self._show_quant():
            self._set_sought(self._default_sought())
        self._rebuild()

    def _no_analyte(self):
        ref = self.ref.anchor if self.ref is not None else None
        sought = self.sought.anchor if self.sought is not None else None
        self.met_id = None
        self._quietly(lambda: self.an_tree.selection_remove(self.an_tree.selection()))
        self.ref_rank = ref or self._tallest(None)
        self.sought_rank = sought or self.ref_rank
        self.ticked = [self.sought_rank] if self.sought_rank is not None else []
        self._rebuild()

    def _options(self) -> dict:
        return {'method': self.method_var.get().lower(),
                'normalize': 'pqn' if self.norm_var.get() == 'PQN' else 'none',
                'within_groups': self.within_var.get()}

    def _on_options(self):
        self.an.set_options(**self._options())
        self.browser.reset()
        self._fill_analytes()
        self._show_notes()
        self._rebuild()

    def _show_notes(self):
        self.notes_lbl.config(text='\n'.join('• ' + n for n in self.an.notes()))

    # ── what is compared ─────────────────────────────────────────────────────

    def _hw(self):
        try:
            return min(max(float(self.hw_var.get()), 0.005), 5.0)
        except ValueError:
            return self.tolerance

    def _group(self):
        """The group the spectrum is drawn over; None for every sample."""
        g = self.group_var.get()
        return None if g == _ALL else g

    def _sought_signal(self):
        """The signal whose lines are read (a copy); a bare peak counts 1 H."""
        if self.met_id:
            if self.sought_i is None:
                return None
            return dict(self.browser.signals(self.met_id)[self.sought_i])
        if self.sought_rank in self.an.peaks:
            s = self.browser.peak_signal(self.sought_rank)
            s['protons'] = 1
            return s
        return None

    def _targets(self):
        """The reference, the signal read and what its ticked lines make."""
        if self.met_id:
            if self.ref_i == _QUANT:
                self.ref = self.browser.quant_target(self.met_id)
            else:
                self.ref = (self.browser.signal_target(self.met_id, self.ref_i)
                            if self.ref_i is not None else None)
        else:
            self.ref = (self.browser.peak_target(self.ref_rank, 1)
                        if self.ref_rank in self.an.peaks else None)
            if self.ref is not None and self.ref.vector is None:
                self.ref = None
        self.sought_sig = self._sought_signal()
        self.sought = (self.browser.lines_target(self.sought_sig, self.ticked)
                       if self.sought_sig is not None else None)

    def _rebuild(self):
        self._targets()
        self.ref_row = (self.browser.compare(self.ref, [self.ref])[0]
                        if self.ref is not None else None)
        self.sig_rows = (self.browser.signals_vs(self.met_id, self.ref_i, self.ref)
                         if self.met_id and self.ref is not None else [])
        self.cand_rows = (self.browser.line_rows(self.ref, self.sought_sig, self.ticked, self._hw())
                          if self.ref is not None and self.sought_sig is not None else [])
        if self.focus not in {r['rank'] for r in self.cand_rows if r['kind'] == 'peak'}:
            self.focus = None
        self._refresh_header()
        self._fill_signals()
        self._fill_window()
        self._fill_values()
        self._refresh_buttons()
        self._draw_all()

    def _refresh_header(self):
        name = ''
        if self.met_id and self.an_tree.exists(str(self.met_id)):
            item = self.an_tree.item(str(self.met_id))
            name = f"{item['text']}  {item['values'][0]}"
        self.title_lbl.config(text=name or 'No analyte — any detected peak can be the reference')
        if self.ref is None:
            self.ref_lbl.config(text='Reference: none (no signal varies across the samples)')
            self.sought_lbl.config(text='')
            return
        if self.ref.kind == 'quant':
            row = self.browser.quantification(self.met_id) or {}
            win = row.get('_multiplet_window')
            lines = ', '.join(f'{float(x):.3f}' for x in row.get('_line_ppms') or ())
            self.ref_lbl.config(text=(
                f"◉ Reference: what the quantification measured — {row.get('_integ_mode') or '?'}, "
                f"{row.get('_multiplicity') or '?'} δ {float(row.get('ppm') or np.nan):.3f}, "
                f"{_h(self.ref.protons)} H"
                + (f", window {min(win):.3f}–{max(win):.3f} ppm" if win and len(win) == 2 else '')
                + (f", lines {lines}" if lines else '')
                + '; its per-sample areas as reported'))
        else:
            src = 'from the listing' if self.met_id else 'assumed 1'
            self.ref_lbl.config(text=f"◉ Reference: {self.ref.label}  ·  {_h(self.ref.protons)} H ({src})")
        s = self.sought_sig
        if s is None:
            self.sought_lbl.config(text='▶ Signal read: none')
            return
        n = sum(1 for r in self.cand_rows if r['kind'] == 'peak')
        share = self._ticked_share()
        src = 'from the listing' if self.met_id else 'assumed 1'
        at = s['ppm'] if np.isfinite(s['ppm']) else s['db_ppm']
        self.sought_lbl.config(text=(
            f"▶ Signal read: {s['type'] or 'peak'} δ {at:.3f}  ·  {_h(s['protons'])} H ({src})  ·  "
            f"{len(self.ticked)} of {n} peaks within ±{self._hw():g} ppm ticked"
            + (f", {100 * share:.0f}% of its pattern" if share is not None else '')
            + (f" — under {100 * MIN_TRUSTED_CAPTURE:.0f}%, the quantification does not scale "
               "it to the whole signal" if share is not None and share < MIN_TRUSTED_CAPTURE else '')
            + ('  ·  ticked: the lines the quantification integrated' if self.quant_view else '')
            + (f"  ·  complex multiplet: every peak in {s['region'][1]:.3f}–{s['region'][0]:.3f} ppm "
               "(shaded) is a line" if s.get('region') else '')
            + ('  ·  ✎ lines chosen by hand' if s.get('manual') else '')
            + ('  ·  saved here: run the Quantification again to measure it'
               if self.met_id in self.changed else '')))

    def _ticked_share(self):
        """The part of the signal's pattern the ticked lines hold."""
        whole = next((r for r in self.cand_rows if r['kind'] == 'whole'), None)
        return whole.get('share') if whole is not None else None

    def _refresh_buttons(self):
        s = self.sought_sig
        editable = (self.editor is not None and self.met_id is not None and s is not None
                    and s.get('mp_index') is not None)
        self.btn_save_lines.state(['!disabled'] if editable and self.ticked else ['disabled'])
        self.btn_restore_lines.state(['!disabled'] if editable and s.get('manual') else ['disabled'])

    # ── tables ───────────────────────────────────────────────────────────────

    def _fill_signals(self):
        tree = self.sig_tree
        tree.delete(*tree.get_children())
        if not self.met_id:
            return
        qi = self._quant_index()
        mode = (self.browser.quantification(self.met_id) or {}).get('_integ_mode') or 'quantified'
        rows = dict(self.sig_rows)
        for i, s in enumerate(self.browser.signals(self.met_id)):
            row = rows.get(i)
            is_ref = self.ref is not None and (i == qi if self.ref_i == _QUANT else i == self.ref_i)
            tags = []
            if is_ref:
                tags.append('ref')
            elif i == self.sought_i:
                tags.append('sought')
            if not s['usable']:
                tags.append('weak')
            note = '✎ lines chosen by hand' if s['manual'] else s['why']
            if s.get('stored_lines'):
                note = (f"the annotation kept {s['stored_lines']} lines; the {len(s['lines'])} "
                        'nearest are shown' + (f'; {note}' if note else ''))
            if i == qi:
                note = f'measured by the quantification ({mode})' + (f'; {note}' if note else '')
            tree.insert('', tk.END, iid=str(i), tags=tags, values=(
                ('◉▶' if is_ref and i == self.sought_i else '◉' if is_ref
                 else '▶' if i == self.sought_i else ''),
                _f(s['db_ppm'], '.4f'), _f(s['ppm'], '.4f'),
                _f(s['ppm_diff'], '.4f'), _f(s['delta_j_hz'], '.2f'),
                _h(s['protons']), s['type'] or '—',
                len(s['lines']) or '—', _quality(s), _f(s['snr'], '.0f'),
                _f(row['r'], '+.2f') if row and not row['is_ref'] else '—',
                _f(row['ratio']) if row else '—', _diff(row) if row else '—', note))
        self._select_signal()

    def _select_signal(self):
        """Select the signal read and give it the keyboard focus: a refilled
        table has no focused row, and without one the arrow keys do nothing."""
        iid = str(self.sought_i)
        if self.sought_i is not None and self.sig_tree.exists(iid):
            self._quietly(lambda: (self.sig_tree.selection_set(iid), self.sig_tree.focus(iid),
                                   self.sig_tree.see(iid)))

    def _quietly(self, fn):
        """Run ``fn`` (a selection set from here) without its selection events
        counting as the analyst's: they arrive after the call, so the guard is
        dropped once they have."""
        self._sync = True
        try:
            fn()
        finally:
            self.win.after_idle(lambda: setattr(self, '_sync', False))

    def _window_rows(self):
        """The reference, the signal the ticked lines make, then the peaks
        (high δ first, or sorted when a heading was clicked)."""
        rows = list(self.cand_rows)
        whole = [r for r in rows if r['kind'] == 'whole']
        rest = [r for r in rows if r['kind'] != 'whole']
        if self._sort and rest:
            col, ascending = self._sort
            key = next(k for c, _t, _w, k in _WIN_COLS if c == col)

            def val(r):
                v = key(r)
                return None if v is None or not np.isfinite(v) else float(v)
            good = sorted((r for r in rest if val(r) is not None), key=val, reverse=not ascending)
            rest = good + [r for r in rest if val(r) is None]
        return ([self.ref_row] if self.ref_row else []) + whole + rest

    def _iid(self, r):
        return 'ref' if r['is_ref'] else 'whole' if r['kind'] == 'whole' else f"p{r['rank']}"

    def _fill_window(self):
        tree = self.win_tree
        tree.delete(*tree.get_children())
        for r in self._window_rows():
            whole = r['kind'] == 'whole'
            tags = (['ref'] if r['is_ref'] else ['sought'] if whole
                    else ['ticked'] if r.get('ticked') else [])
            if not r['usable']:
                tags = ['weak']
            share = r.get('share')
            tree.insert('', tk.END, iid=self._iid(r), tags=tags, values=(
                '◉' if r['is_ref'] else 'Σ' if whole else '☑' if r.get('ticked') else '☐',
                r['label'], f"{r['ppm']:.4f}", _f(r['snr'], '.0f'),
                '—' if r['is_ref'] else _f(share, '.3g'),
                _h(r['protons']),
                '—' if r['is_ref'] else _f(r['r'], '+.2f'),
                '—' if r['is_ref'] or not np.isfinite(r['p']) else f"{r['p']:.2g}",
                _f(r['ratio']), '—' if r['is_ref'] else _f(_expected(r, self.ref)),
                _f(r['sd']), _diff(r), _area(r['area_mean']),
                r['why'] if not r['usable'] else r['assigned']))
        self._select_rows()

    def _sort_by(self, col):
        ascending = not (self._sort and self._sort[0] == col and self._sort[1])
        self._sort = (col, ascending)
        self._fill_window()
        self._fill_values()

    def _fill_values(self):
        for w in self.val_frame.winfo_children():
            w.destroy()
        rows = [r for r in self._window_rows() if r['usable']]
        if not rows:
            return
        grid = self.browser.grid(rows, per=self.per_var.get())
        names = grid['names']
        cols = ['mark', 'label', 'ppm', 'h', 'dq'] + [f's{i}' for i in range(len(names))]
        tree = ttk.Treeview(self.val_frame, columns=cols, show='headings', height=10,
                            selectmode='browse')
        for cid, title, w in (('mark', 'In', 34), ('label', 'Peak', 150), ('ppm', 'δ (ppm)', 64),
                              ('h', 'H', 32), ('dq', 'Quant. diff (%)', 92)):
            tree.heading(cid, text=title)
            tree.column(cid, width=w, anchor=tk.W if cid == 'label' else tk.CENTER, stretch=False)
        for i, n in enumerate(names):
            tree.heading(f's{i}', text=n)
            tree.column(f's{i}', width=max(70, 9 * len(n) + 18), anchor=tk.E, stretch=False)
        vs = ttk.Scrollbar(self.val_frame, command=tree.yview)
        hs = ttk.Scrollbar(self.val_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.config(yscrollcommand=vs.set, xscrollcommand=hs.set)
        hs.pack(side=tk.BOTTOM, fill=tk.X)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree.tag_configure('ref', background='#fde0dd', font=('Arial', 9, 'bold'))
        tree.tag_configure('group', background='#eeeeee', foreground='#444444')
        tree.insert('', tk.END, tags=('group',), values=('', 'group', '', '', '', *grid['groups']))
        for r, vals in zip(rows, grid['values']):
            tree.insert('', tk.END, tags=('ref',) if r['is_ref'] else (), values=(
                '◉' if r['is_ref'] else 'Σ' if r['kind'] == 'whole' else '☑' if r.get('ticked')
                else '', r['label'], f"{r['ppm']:.4f}",
                _h(r['protons']), _diff(r), *[_area(v) for v in vals]))

    # ── selection ────────────────────────────────────────────────────────────

    def _on_sig_select(self):
        sel = self.sig_tree.selection()
        if self._sync or not sel:
            return
        i = int(sel[0])
        if i == self._quant_index():
            if not (self.quant_view and i == self.sought_i):
                self.focus = None
                self._show_quant()
                self._rebuild()
            return
        if i == self.sought_i and not self.quant_view:
            return
        # A signal not detected is read too: the peaks around where it is
        # listed, none ticked, may show it.
        self.focus = None
        self._set_sought(i)
        self._rebuild()

    def _ref_from_signal(self):
        sel = self.sig_tree.selection()
        if not sel:
            return
        i = int(sel[0])
        if i == self._quant_index():
            self.ref_i = _QUANT                 # back to what the quantification measured
        else:
            if not self.browser.signals(self.met_id)[i]['usable']:
                messagebox.showinfo('STOCSY', 'That signal was not detected, or no line of it varies '
                                    'across the samples: it cannot be the reference.', parent=self.win)
                return
            self.ref_i = i
        self._rebuild()

    def _selected_peak(self):
        sel = self.win_tree.selection()
        if not sel or not sel[0].startswith('p'):
            return None
        return int(sel[0][1:])

    def _on_win_select(self):
        rank = self._selected_peak()
        if self._sync or rank is None or rank == self.focus:
            return
        self._set_focus(rank)

    def _on_win_click(self, event):
        """A click on the "In signal" cell ticks or unticks the peak."""
        tree = self.win_tree
        if tree.identify_region(event.x, event.y) != 'cell' or tree.identify_column(event.x) != '#1':
            return None
        iid = tree.identify_row(event.y)
        if not iid.startswith('p'):
            return None
        self._toggle(int(iid[1:]))
        return 'break'

    def _toggle_selected(self):
        rank = self._selected_peak()
        if rank is not None:
            self._toggle(rank)

    def _toggle(self, rank):
        """Tick or untick a peak as a line of the signal read."""
        if rank in self.ticked:
            self.ticked = [r for r in self.ticked if r != rank]
        elif self.an.can_seed(rank):
            self.ticked = self.ticked + [rank]
        else:
            messagebox.showinfo('STOCSY', 'That peak has no per-sample areas that vary: it '
                                'cannot be read as a line.', parent=self.win)
            return
        self.focus = rank
        self._rebuild()

    def _set_focus(self, rank):
        self.focus = rank
        self._draw_spectrum()
        self._draw_variation()
        self._select_rows()

    def _select_rows(self):
        """Select the peak in focus; with none, give the first row the
        keyboard focus so the arrow keys have somewhere to start."""
        tree = self.win_tree
        if self.focus is not None and tree.exists(f'p{self.focus}'):
            iid = f'p{self.focus}'
            self._quietly(lambda: (tree.selection_set(iid), tree.focus(iid), tree.see(iid)))
        elif tree.get_children() and not tree.focus():
            tree.focus(tree.get_children()[0])

    def _ref_from_candidate(self):
        rank = self._selected_peak()
        if rank is None:
            return
        row = next((r for r in self.cand_rows if r['kind'] == 'peak' and r['rank'] == rank), None)
        if row is None or not row['usable']:
            messagebox.showinfo('STOCSY', 'That peak has no per-sample areas that vary: it cannot '
                                'be the reference.', parent=self.win)
            return
        sought = self.sought_sig['rank'] if self.sought_sig and self.sought_sig['rank'] else rank
        self.met_id = None
        self._quietly(lambda: self.an_tree.selection_remove(self.an_tree.selection()))
        self.ref_rank, self.sought_rank = rank, sought
        self.ticked = [r for r in self.ticked if r in self.an.peaks] or [sought]
        self._rebuild()

    def _sought_from_candidate(self):
        rank = self._selected_peak()
        if rank is None:
            return
        if self.met_id:
            self._no_analyte()
        self.sought_rank = rank
        self.ticked = [rank]
        self.focus = None
        self._rebuild()

    # ── the analyst's lines ──────────────────────────────────────────────────

    def _save_lines(self):
        """Write the ticked peaks on the annotation as the signal's lines."""
        s = self.sought_sig
        if self.editor is None or not self.met_id or s is None or s.get('mp_index') is None:
            messagebox.showinfo('STOCSY', 'Only a matched signal of an analyte can be given its '
                                'lines.', parent=self.win)
            return
        peaks = sorted((self.an.peaks[r] for r in self.ticked if r in self.an.peaks),
                       key=lambda p: -float(p['ppm']))
        if not peaks:
            messagebox.showinfo('STOCSY', 'Tick at least one peak.', parent=self.win)
            return
        name = self.browser.metabolite(self.met_id).get('name') or self.met_id
        at = ', '.join(f"{float(p['ppm']):.4f}" for p in peaks)
        share = self._ticked_share()
        partial = share is not None and share < MIN_TRUSTED_CAPTURE
        if not messagebox.askyesno('STOCSY', (
                f"Make these {len(peaks)} peaks ({at} ppm) the lines of the {s['type'] or 'signal'} "
                f"δ {s['db_ppm']:.3f} of {name}?\n\nThe signal is marked Good and, at the next "
                "quantification, measured on these lines: it is not left to the cascade, unless its "
                "window is shared with another analyte."
                + (f"\n\nThese lines hold {100 * share:.0f}% of the signal's pattern, under the "
                   f"{100 * MIN_TRUSTED_CAPTURE:.0f}% the quantification needs to scale them to the "
                   "whole signal: it will not quantify the analyte on them, and the cascade may "
                   "take it." if partial else '')
                + "\n\n\"Restore the annotation's lines\" undoes it."),
                icon='warning' if partial else 'question', parent=self.win):
            return
        self.editor.apply(self.met_id, s['mp_index'], peaks)
        self._after_edit()

    def _classify(self, quality):
        """Ctrl+1/2/3: the signal read is Good, Average or Poor. Good with
        lines ticked other than its own is "Save the ticked lines" (which
        marks it Good); the lines the quantification integrated are only
        shown, not taken as a choice."""
        s = self.sought_sig
        if self.editor is None or not self.met_id or s is None or s.get('mp_index') is None:
            return
        if (quality == 'good' and not self.quant_view and self.ticked
                and sorted(self.ticked) != sorted(s['lines'])):
            self._save_lines()
            return
        self.editor.classify(self.met_id, s['mp_index'], quality)
        self._after_edit()

    def reload(self, analysis, browser):
        """The project changed under the window (a signal split, added or
        removed in the Signal List): read it again, on the same analyte,
        reference and signal, with the same correlation settings."""
        state = self.state()
        self.an, self.browser = analysis, browser
        self.an.set_options(**self._options())
        names = analysis.group_names()
        self.group_cb.config(values=[_ALL] + names)
        if self._group() not in names:
            self.group_var.set(_ALL)
        self.focus = None
        self._show_notes()
        self._fill_analytes()
        self._open_on(state, None)

    # ── right-click ──────────────────────────────────────────────────────────

    def _popup(self, menu, event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _sig_menu(self, event):
        """On a listed signal: read it, then take it to the Annotation
        Results, mark it, or make it the reference."""
        iid = self.sig_tree.identify_row(event.y)
        if not iid or not self.met_id:
            return
        self.sig_tree.selection_set(iid)
        self._on_sig_select()
        s = self.browser.signals(self.met_id)[int(iid)]
        matched = s.get('mp_index') is not None
        menu = tk.Menu(self.win, tearoff=0)
        menu.add_command(label='Show this signal in the Annotation Results',
                         command=lambda: self._show_in_annotation(int(iid)),
                         state='normal' if self.show_signal and matched else 'disabled')
        menu.add_separator()
        for key, quality in (('1', 'good'), ('2', None), ('3', 'bad')):
            menu.add_command(
                label=f'Mark {QUALITY_LABEL[tier_from_annotation_quality(quality)]}',
                accelerator=f'Ctrl+{key}', command=lambda q=quality: self._classify(q),
                state='normal' if self.editor is not None and matched else 'disabled')
        menu.add_separator()
        menu.add_command(label='Use as the reference', accelerator='double-click',
                         command=self._ref_from_signal)
        self._popup(menu, event)

    def _show_in_annotation(self, i):
        s = self.browser.signals(self.met_id)[i]
        mp = self.browser.metabolite(self.met_id)['matched_peaks'][s['mp_index']]
        self.show_signal(self.met_id, mp)

    def _win_menu(self, event):
        """On a peak of the window: take it to the Signal List (to split or
        add a signal there), tick it, or make it the reference."""
        iid = self.win_tree.identify_row(event.y)
        if not iid.startswith('p'):
            return
        rank = int(iid[1:])
        self.win_tree.selection_set(iid)
        self._set_focus(rank)
        menu = tk.Menu(self.win, tearoff=0)
        menu.add_command(label=f'Show peak #{rank} in the Signal List',
                         command=lambda: self.show_peak(rank),
                         state='normal' if self.show_peak else 'disabled')
        menu.add_separator()
        menu.add_command(label='Tick / untick as a line of the signal', accelerator='Space',
                         command=lambda: self._toggle(rank))
        menu.add_command(label='Use as the reference', command=self._ref_from_candidate)
        self._popup(menu, event)

    def refresh_from_annotation(self):
        """The Annotation Results changed a verdict: read every analyte's
        signals again, so this window shows the annotation's marks."""
        self.browser.forget()
        self._fill_analytes()
        self._rebuild()

    def _restore_lines(self):
        s = self.sought_sig
        if (self.editor is None or not self.met_id or s is None or s.get('mp_index') is None
                or not self.editor.has_manual(self.met_id, s['mp_index'])):
            messagebox.showinfo('STOCSY', 'This signal keeps the annotation\'s own lines.',
                                parent=self.win)
            return
        self.editor.restore(self.met_id, s['mp_index'])
        self._after_edit()

    def _after_edit(self):
        self.browser.forget(self.met_id)
        self.changed.add(self.met_id)
        if self.on_change is not None:
            self.on_change(self.met_id)
        self._set_sought(self.sought_i)
        self._fill_analytes()
        self._rebuild()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _draw_all(self):
        self._draw_spectrum()
        self._draw_variation()
        self._draw_matrix()

    def _draw_spectrum(self):
        ax = self.ax_spec
        ax.clear()
        self._view_ranks = []
        s = self.sought_sig
        at = (s['ppm'] if s is not None and np.isfinite(s['ppm']) else
              s['db_ppm'] if s is not None else self.ref.ppm if self.ref is not None else np.nan)
        group = self._group()
        mean = self.an.mean_spectrum(group)
        if not np.isfinite(at) or mean.size == 0:
            ax.text(0.5, 0.5, 'No peak to show', transform=ax.transAxes, ha='center')
            self.cax.set_visible(False)
            self.cv_spec.draw_idle()
            return
        hw = self._hw()
        hi, lo = at + hw, at - hw
        ppm = self.an.ppm
        sel = np.flatnonzero((ppm <= hi) & (ppm >= lo))
        if sel.size < 2:
            ax.text(0.5, 0.5, 'No spectral points here', transform=ax.transAxes, ha='center')
            self.cax.set_visible(False)
            self.cv_spec.draw_idle()
            return
        x, y = ppm[sel], mean[sel]
        basis = self.an.basis if group is None else self.an.group_basis(group)
        crit = basis.r_critical
        tr = None
        if self.ref is not None:
            key = ('ref', self.met_id, self.ref.kind, self.ref.label, tuple(self.ref.ranks))
            tr = self.an.trace_of(self.ref.vector, key, group)['r'][sel]
        colour = self.color_var.get() and tr is not None
        self.cax.set_visible(bool(colour))
        if colour:
            rgba = colormaps[_CMAP](_NORM(np.nan_to_num(tr)))
            rgba[:, 3] = np.where(np.abs(tr) >= crit, 1.0, 0.35)
            rgba[np.isnan(tr)] = _GREY
            pts = np.column_stack([x, y])
            ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1),
                                             colors=rgba[:-1], lw=1.5))
        else:
            ax.plot(x, y, color='#444444', lw=1.0)
        span = float(np.nanmax(y) - np.nanmin(y)) or 1.0
        ax.set_ylim(float(np.nanmin(y)) - 0.04 * span, float(np.nanmax(y)) + 0.22 * span)
        ax.set_xlim(hi, lo)                                  # δ decreases left to right
        ax.set_xlabel('δ (ppm)', fontsize=8)
        ax.set_ylabel('Mean intensity (au)' if group is None else f'Mean of {group} (au)', fontsize=8)
        ax.tick_params(labelsize=7)
        if tr is not None and not np.isfinite(tr).any():
            ax.text(0.5, 0.80, f'grey: nothing here varies more than {NOISE_FLOOR_FACTOR:g}× the noise '
                    'across samples — no r to read', transform=ax.transAxes, ha='center',
                    fontsize=7, color='#555555')
        ref_r = set(self.ref.ranks) if self.ref is not None else set()
        sought_r = set(self.ticked)
        inside = sorted((r for r, p in self.an.peaks.items() if lo <= float(p['ppm']) <= hi),
                        key=lambda r: -self.an.ppm_of(r))
        self._view_ranks = inside
        mark = ax.get_xaxis_transform()                      # x in δ, y in axes fraction
        for r in inside:
            p = float(self.an.peaks[r]['ppm'])
            col = _REF if r in ref_r else _SOUGHT if r in sought_r else _OTHER
            big = r in ref_r or r in sought_r
            ax.plot([p], [0.93], marker='v', ms=9 if big else 5, color=col, transform=mark,
                    clip_on=False, mec='black' if r == self.focus else col,
                    mew=1.2 if r == self.focus else 0.6)
            if len(inside) <= _MAX_LABELS or big or r == self.focus:
                ax.text(p, 0.97, f'#{r}', fontsize=7, color=col, ha='center', va='bottom',
                        transform=mark, clip_on=False,
                        fontweight='bold' if big or r == self.focus else 'normal')
        region = (s or {}).get('region')
        if region:
            ax.axvspan(region[1], region[0], color=_SOUGHT, alpha=0.10, lw=0)
        if self.focus is not None:
            ax.axvline(self.an.ppm_of(self.focus), color='black', lw=0.8, ls=':')
        scope = (f'all {basis.n} samples' if group is None
                 else f'group {group}: mean and r of its {basis.n} samples')
        ax.set_title(f'{scope}, |r| ≥ {crit:.2f} significant · red ▼ reference lines · '
                     'blue ▼ ticked lines of the signal read · grey ▼ other peaks · '
                     'click a peak to see its numbers', fontsize=7, pad=14)
        self.cv_spec.draw_idle()

    def _on_spec_click(self, event):
        if event.inaxes is not self.ax_spec or event.xdata is None or event.button != 1:
            return
        best = min(self._view_ranks, key=lambda r: abs(self.an.ppm_of(r) - event.xdata), default=None)
        if best is None or abs(self.an.ppm_of(best) - event.xdata) > max(0.004, 0.07 * self._hw()):
            return
        if self.win_tree.exists(f'p{best}'):
            self._set_focus(best)

    def _compared(self):
        """The rows drawn on the variation tab: the reference, the signal the
        ticked lines make and every signal of the analyte (without one, the
        ticked peaks); and the peak in focus."""
        if self.ref_row is None:
            return []
        rows = [self.ref_row] + [r for r in self.cand_rows if r['kind'] == 'whole']
        if self.met_id:
            rows += [r for _i, r in self.sig_rows if r is not None and not r['is_ref']]
        else:
            rows += [r for r in self.cand_rows if r['usable'] and r['kind'] == 'peak'
                     and r.get('ticked')][:_DRAWN]
        extra = next((r for r in self.cand_rows if r['kind'] == 'peak' and r['rank'] == self.focus
                      and r['usable']), None) if self.focus is not None else None
        if extra is not None and all(r['label'] != extra['label'] for r in rows):
            rows.append(dict(extra, focus=True))
        return rows

    def _draw_variation(self):
        ax = self.ax_var
        ax.clear()
        rows = self._compared()
        if not rows:
            ax.text(0.5, 0.5, 'Pick an analyte or a reference peak', transform=ax.transAxes,
                    ha='center')
            self.cv_var.draw_idle()
            return
        grid = self.browser.grid(rows, per='proton')
        n = len(grid['names'])
        xs = np.arange(n)
        groups = grid['groups']
        for i, g in enumerate(dict.fromkeys(groups)):
            idx = [j for j, x in enumerate(groups) if x == g]
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=self.group_colors.get(
                g, ['#e8eef7', '#f7e8ee', '#e8f7ee', '#f7f3e8'][i % 4]), alpha=0.18, lw=0)
            ax.text((idx[0] + idx[-1]) / 2, 1.0, g, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7)
        k = 0
        for r, vals in zip(rows, grid['values']):
            label = f"{r['label']}  ({_h(r['protons'])} H)" + ('  — reference' if r['is_ref'] else '')
            if r['is_ref']:
                ax.plot(xs, vals, color='black', lw=2.6, marker='o', ms=4, label=label, zorder=5)
            elif r.get('focus'):
                ax.plot(xs, vals, color=_REF, lw=1.6, ls='--', marker='s', ms=3, label=label, zorder=4)
            else:
                ax.plot(xs, vals, color=_LINES[k % len(_LINES)], lw=1.1, marker='.', ms=4,
                        label=label, alpha=0.9)
                k += 1
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(grid['names'], rotation=90 if n > 12 else 0, fontsize=6 if n > 24 else 7)
        ax.set_ylabel('Area ÷ H (au / H)', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
        self.fig_var.subplots_adjust(left=0.07, right=0.76, top=0.9, bottom=0.2)
        self.cv_var.draw_idle()

    def _draw_matrix(self):
        ax = self.ax_mat
        ax.clear()
        if self.met_id:
            targets = ([self.ref] if self.ref is not None and self.ref.kind == 'quant' else []) + [
                t for t in (self.browser.signal_target(self.met_id, i)
                            for i in range(len(self.browser.signals(self.met_id)))) if t]
        else:
            targets = ([self.ref] if self.ref is not None else []) + [
                r['target'] for r in self.cand_rows
                if r['usable'] and r['kind'] == 'peak' and r.get('ticked')][:_DRAWN]
        labels, R, P = self.browser.matrix(targets)
        if len(labels) < 2:
            ax.text(0.5, 0.5, 'Fewer than two signals can be correlated here',
                    transform=ax.transAxes, ha='center', va='center', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            self.cv_mat.draw_idle()
            return
        ax.imshow(R, cmap=_CMAP, vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if np.isfinite(R[i, j]):
                    ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=8,
                            fontweight='bold' if P[i, j] < 0.05 else 'normal',
                            color='black' if abs(R[i, j]) < 0.6 else 'white')
        self.fig_mat.subplots_adjust(left=0.25, right=0.97, top=0.97, bottom=0.32)
        self.cv_mat.draw_idle()

    # ── modules, export ──────────────────────────────────────────────────────

    def _open_structure(self):
        from gui.widgets.stocsy_structure_window import StocsyStructureWindow
        self._structure = StocsyStructureWindow(self.win, self.an, on_send=self._send_signals)

    def _send_signals(self, ranks):
        """Show a block of signals here: the clearest is the reference and the
        window opens around the block."""
        usable = [r for r in ranks if self.an.can_seed(r)]
        if not usable:
            return
        self.met_id = None
        self._quietly(lambda: self.an_tree.selection_remove(self.an_tree.selection()))
        best = max(usable, key=lambda r: float(self.an.peaks[r].get('snr') or 0.0))
        self.ref_rank = self.sought_rank = best
        ppms = [self.an.ppm_of(r) for r in usable]
        self.hw_var.set(f'{max(max(ppms) - min(ppms), 0.05) / 2 + 0.05:.2f}')
        self.sought_rank = min(usable, key=lambda r: abs(self.an.ppm_of(r) - np.mean(ppms)))
        self.ticked = [self.sought_rank]
        self._rebuild()
        self.nb.select(1)
        self.show()

    def _export(self):
        rows = [r for r in self._window_rows()]
        if not rows or self.ref is None:
            messagebox.showinfo('STOCSY', 'Pick a reference first.', parent=self.win)
            return
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension='.csv', filetypes=[('CSV', '*.csv')],
            title='Export the lines against the reference')
        if not path:
            return
        out = self.browser.export_rows(rows, self.ref, per=self.per_var.get())
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
