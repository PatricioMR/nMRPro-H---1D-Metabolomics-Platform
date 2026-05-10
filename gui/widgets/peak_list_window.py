# gui/widgets/peak_list_window.py
"""
Peak List Window.

Panel A : sortable table — Rank, PPM, Start, End, Width, SNR, Quality, Overlap
          • Quality column: Good (green) / Average (yellow) / Poor (red)
          • Ctrl+1 = Good, Ctrl+2 = Average, Ctrl+3 = Poor
          • Save button (bottom-left) writes changes back to peak_stage.peaks
Panel B : peak region mini-plot
          • Scroll to zoom
          • Click+drag to redefine start/end boundaries
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from .peak_sample_view import PeakSampleView

# Status definitions
STATUS_GOOD      = 'Good'
STATUS_UNDEFINED = 'Average'
STATUS_POOR      = 'Poor'

STATUS_COLORS = {
    STATUS_GOOD:      '#c8f7c5',  # light green
    STATUS_UNDEFINED: '#fef9c3',  # light yellow (Average)
    STATUS_POOR:      '#ffd6d6',  # light red
}
STATUS_FG = {
    STATUS_GOOD:      '#1a7a1a',
    STATUS_UNDEFINED: '#7a6000',
    STATUS_POOR:      '#9a0000',
}


class PeakListWindow:
    def __init__(self, parent, peaks, ppm, all_intensities, app=None):
        self.peaks           = peaks
        self.ppm             = ppm
        self.all_intensities = all_intensities
        self.app             = app
        self._sort_col       = 'rank'
        self._sort_reverse   = False
        self._current_peak   = None
        self._span_selector  = None

        # Picos sem status ficam sem marcação (None).
        # O utilizador pode alterar Good/Average/Poor manualmente.
        for p in self.peaks:
            p.setdefault('status', None)

        self.win = tk.Toplevel(parent)
        self.win.title(f"Peak List  ({len(peaks)} peaks)")
        self.win.geometry("1200x560")
        self.win.minsize(900, 400)

        self._build()
        self._populate()
        self._bind_shortcuts()
        self._bind_ctrl_r()
        self._center()

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self._on_select()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        paned = tk.PanedWindow(
            self.win, orient=tk.HORIZONTAL,
            sashwidth=6, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Panel A ───────────────────────────────────────────────────────────
        frame_a = ttk.Frame(paned)
        paned.add(frame_a, minsize=430)

        ttk.Label(frame_a, text="Peak List",
                  font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(4, 2), padx=4)

        n_overlap = sum(1 for p in self.peaks if p.get('overlapping', False))
        summary   = (f"{len(self.peaks)} peaks"
                     + (f"  ·  {n_overlap} overlapping" if n_overlap else "")
                     + "  ·  Ctrl+1 Good  Ctrl+2 Average  Ctrl+3 Poor")
        ttk.Label(frame_a, text=summary,
                  font=("Arial", 7), foreground="gray").pack(
            anchor=tk.W, padx=4)

        # Filter checkboxes
        filt_row = ttk.Frame(frame_a)
        filt_row.pack(fill=tk.X, padx=4, pady=(2, 0))
        ttk.Label(filt_row, text='Show:', font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 4))
        _fs = getattr(self.app, '_peak_list_filters', {})
        self._filt_good  = tk.BooleanVar(value=_fs.get('good',  True))
        self._filt_undef = tk.BooleanVar(value=_fs.get('undef', True))
        self._filt_poor  = tk.BooleanVar(value=_fs.get('poor',  True))
        self._filt_none  = tk.BooleanVar(value=_fs.get('none',  True))
        for text, var, bg, fg in [
            ('✓ Good',      self._filt_good,  '#c8f7c5', '#1a7a1a'),
            ('~ Average',  self._filt_undef, '#fef9c3', '#7a6000'),
            ('✗ Poor',      self._filt_poor,  '#ffd6d6', '#9a0000'),
            ('— No status', self._filt_none,  '#f0f0f0', '#444444'),
        ]:
            cb = tk.Checkbutton(filt_row, text=text, variable=var,
                                bg=bg, fg=fg, selectcolor=bg,
                                activebackground=bg,
                                font=('Arial', 8),
                                command=self._on_filter_change)
            cb.pack(side=tk.LEFT, padx=2)

        # Treeview
        cols = ('rank', 'ppm', 'start', 'end', 'width', 'snr', 'area', 'metabolite', 'status', 'overlap')
        tree_frame = ttk.Frame(frame_a)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        self.tree = ttk.Treeview(
            tree_frame, columns=cols, show='headings',
            selectmode='browse',
            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        col_defs = [
            ('rank',       '#',          50,  tk.CENTER),
            ('ppm',        'PPM',        80,  tk.CENTER),
            ('start',      'Start',      80,  tk.CENTER),
            ('end',        'End',        80,  tk.CENTER),
            ('width',      'Width',      65,  tk.CENTER),
            ('snr',        'SNR',        60,  tk.CENTER),
            ('area',       'Area',       85,  tk.CENTER),
            ('metabolite', 'Metabolite',160,  tk.W),
            ('status',     'Quality',   110,  tk.CENTER),
            ('overlap',    'Overlap',    60,  tk.CENTER),
        ]
        for cid, heading, width, anchor in col_defs:
            self.tree.heading(cid, text=heading,
                              command=lambda c=cid: self._sort(c))
            self.tree.column(cid, width=width, anchor=anchor, minwidth=35)

        # Row tags for overlap
        self.tree.tag_configure('overlap',          background='#ffe4e4')
        self.tree.tag_configure('normal',           background='white')
        # Row tags for status (overlap rows keep overlap bg; status shown via Status col)
        self.tree.bind('<<TreeviewSelect>>', lambda e: self._on_select())
        self.tree.bind('<Button-3>', self._on_right_click)

        # Status buttons row
        btn_row = ttk.Frame(frame_a)
        btn_row.pack(fill=tk.X, padx=4, pady=(0, 2))

        ttk.Label(btn_row, text="Set status:",
                  font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 6))

        for label, status, bg, fg, shortcut in [
            ("✓ Good",      STATUS_GOOD,      '#c8f7c5', '#1a7a1a', "Ctrl+1"),
            ("~ Average",   STATUS_UNDEFINED, '#fef9c3', '#7a6000', "Ctrl+2"),
            ("✗ Poor",      STATUS_POOR,      '#ffd6d6', '#9a0000', "Ctrl+3"),
        ]:
            tk.Button(
                btn_row, text=f"{label}  ({shortcut})",
                bg=bg, fg=fg,
                font=("Arial", 8), relief=tk.RAISED,
                padx=6, pady=2,
                command=lambda s=status: self._set_status(s)
            ).pack(side=tk.LEFT, padx=3)


        # ── Panel B ───────────────────────────────────────────────────────────
        frame_b = ttk.Frame(paned)
        paned.add(frame_b, minsize=360)

        ttk.Label(frame_b, text="Peak Region",
                  font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(4, 2), padx=4)

        self.peak_info_label = ttk.Label(
            frame_b, text="", font=("Arial", 8), foreground="#444")
        self.peak_info_label.pack(anchor=tk.W, padx=4)

        ttk.Label(frame_b,
                  text="Right-drag: redefine boundaries  ·  Toolbar: zoom/pan/home  ·  R: reset",
                  font=("Arial", 7), foreground="#888").pack(anchor=tk.W, padx=4)

        self.fig = plt.Figure(figsize=(5, 3.5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('#f8f8f8')

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=frame_b)
        self.plot_canvas.draw()

        # Toolbar nativa — home/zoom/pan/save, mesma da janela principal
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self._pk_toolbar = NavigationToolbar2Tk(self.plot_canvas, frame_b)
        self._pk_toolbar.update()
        self._pk_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._build_group_checkboxes(frame_b)

        self.plot_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.plot_canvas.get_tk_widget().bind("<MouseWheel>", self._on_scroll_win)
        self.plot_canvas.get_tk_widget().bind("<Button-4>",   self._on_scroll_up)
        self.plot_canvas.get_tk_widget().bind("<Button-5>",   self._on_scroll_down)


        # Bind R for quick reset
        self.win.bind('<r>', lambda e: self._reset_zoom())
        self.win.bind('<R>', lambda e: self._reset_zoom())

    def _build_group_checkboxes(self, parent):
        self.group_vars = {}
        if not self.app or not hasattr(self.app, 'samples_info') or not self.app.samples_info:
            return

        groups = sorted(set([
            s.get('label', 'Unknown') for s in self.app.samples_info
            if str(s.get('type', s.get('label', ''))).upper() != 'BLK'
        ]))
        if not groups:
            return

        gf = ttk.Frame(parent)
        gf.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)
        ttk.Label(gf, text="Show Groups:", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        
        # Determine initial selection state (all ON)
        for g in groups:
            var = tk.BooleanVar(value=True)
            self.group_vars[g] = var
            cb = ttk.Checkbutton(gf, text=g, variable=var,
                                 command=lambda: self._reset_zoom())
            cb.pack(side=tk.LEFT, padx=3)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _bind_ctrl_r(self):
        self.win.bind('<Control-r>', lambda e: self._reset_zoom())
        self.win.bind('<Control-R>', lambda e: self._reset_zoom())

    def _reset_zoom(self):
        if self._current_peak:
            self._plot_peak(self._current_peak)

    def _bind_shortcuts(self):
        self.win.bind('<Control-Key-1>', lambda e: self._set_status(STATUS_GOOD))
        self.win.bind('<Control-Key-2>', lambda e: self._set_status(STATUS_UNDEFINED))
        self.win.bind('<Control-Key-3>', lambda e: self._set_status(STATUS_POOR))

    # ── Table ─────────────────────────────────────────────────────────────────

    def _status_symbol(self, status, is_anchor=False, r2=None):
        label = {'Good': 'Good', 'Average': 'Avg', 'Poor': 'Poor'}.get(status, None)
        if label is None:
            return '—'
        r2_str = f' {r2:.2f}' if (r2 is not None and r2 == r2) else ''  # nan check
        dot = '⚓' if is_anchor else '●'
        return f'{dot} {label}{r2_str}'

    def _in_deconv_queue(self, peak):
        """Check if this peak's region is in the deconv controller queue."""
        if self.app is None:
            return False
        stage = getattr(self.app, 'stages', {}).get('peak_detection')
        if stage is None or stage._deconv_controller is None:
            return False
        ctrl   = stage._deconv_controller
        p_lo   = min(peak['start_ppm'], peak['end_ppm'])
        p_hi   = max(peak['start_ppm'], peak['end_ppm'])
        return any(
            abs(min(s,e) - p_lo) < 0.001 and abs(max(s,e) - p_hi) < 0.001
            for s, e in ctrl.regions)

    def scroll_to_rank(self, rank):
        """Scroll the tree to show the peak with the given rank."""
        if rank is None:
            return
        for iid in self.tree.get_children(''):
            vals = self.tree.item(iid, 'values')
            if vals and str(vals[0]) == str(rank):
                self.tree.selection_set(iid)
                self.tree.focus(iid)
                self.tree.see(iid)
                return

    def focus_peak_rank(self, rank):
        """Select and scroll to the peak with given rank; bring window to front."""
        self.scroll_to_rank(rank)
        try:
            self.win.deiconify()
            self.win.lift()
            self.win.focus_force()
        except Exception:
            pass

    def _on_filter_change(self):
        if self.app is not None:
            self.app._peak_list_filters = {
                'good':  self._filt_good.get(),
                'undef': self._filt_undef.get(),
                'poor':  self._filt_poor.get(),
                'none':  self._filt_none.get(),
            }
        self._populate()

    def _populate(self):
        self.tree.delete(*self.tree.get_children())
        self.tree.tag_configure('sub_peak', background='#e8f5e9',
                                            foreground='#2e7d32')
        # Build set of allowed statuses from filter checkboxes
        _allowed = set()
        if getattr(self, '_filt_good',  None) and self._filt_good.get():
            _allowed.add(STATUS_GOOD)
        if getattr(self, '_filt_undef', None) and self._filt_undef.get():
            _allowed.add(STATUS_UNDEFINED)
        if getattr(self, '_filt_poor',  None) and self._filt_poor.get():
            _allowed.add(STATUS_POOR)
        _show_none = not getattr(self, '_filt_none', None) or self._filt_none.get()

        # Build metabolite lookup: one label per peak, best candidate wins.
        #
        # Eligibility (same rules as annotation_results_window):
        #   • annotation_quality == 'bad'  → discard
        #   • singlet                      → always eligible
        #   • non-singlet                  → eligible only when ΔJ < 0.5 Hz
        #
        # Priority (one metabolite per rank):
        #   composite_score is the unified tiebreaker — it already encodes
        #   ΔJ quality (60 %) and ppm closeness (10 %), so:
        #     · best ΔJ wins among multiplets
        #     · closest ppm wins among singlets
        #     · J-confirmed multiplet beats any singlet at the same rank
        _met_by_rank = {}  # rank → metabolite name
        _ann = getattr(self.app, 'stages', {}).get('annotation', None) if self.app else None
        if _ann:
            _metabolites = getattr(_ann, '_metabolites', None)
            if _metabolites:
                def _is_singlet_mp(mp):
                    mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                    return mult in ('s', 'singlet', '', 'm', 'multiplet')

                # Collect all valid candidates: {rank: [(score, name), ...]}
                _candidates = {}
                for _met in _metabolites:
                    _mname = _met.get('name', '')
                    for _mp in _met.get('matched_peaks', []):
                        if _mp.get('annotation_quality') == 'bad':
                            continue
                        _pr = _mp.get('peak_rank')
                        if _pr is None:
                            continue
                        if _is_singlet_mp(_mp):
                            _eligible = True
                        else:
                            _dj = _mp.get('delta_j_hz')
                            _eligible = (_dj is not None and float(_dj) < 0.5)
                        if not _eligible:
                            continue
                        _score = float(_mp.get('composite_score', 0) or 0)
                        _candidates.setdefault(_pr, []).append((_score, _mname))

                # Pick highest-scoring candidate per rank
                for _pr, _cands in _candidates.items():
                    _met_by_rank[_pr] = max(_cands, key=lambda x: x[0])[1]
            else:
                # Fallback: raw results list (no ΔJ filtering possible)
                _last = getattr(_ann, '_last_results_args', None) or {}
                for _r in _last.get('results', []):
                    _pr = _r.get('peak_rank')
                    if _pr is not None and _pr not in _met_by_rank:
                        _met_by_rank[_pr] = _r.get('name', _r.get('metabolite', ''))

        for p in self._sorted_peaks():
            _st = p.get('status')
            if _st is None:
                if not _show_none:
                    continue
            elif _st not in _allowed:
                continue
            is_sub   = p.get('is_sub_peak', False)
            st_sym   = self._status_symbol(p.get('status', STATUS_UNDEFINED),
                                             is_anchor=p.get('is_anchor_line', False),
                                             r2=p.get('gaussian_r2'))
            if is_sub:
                ovlp_sym = '🔧'
            elif p.get('overlapping', False):
                ovlp_sym = '⚠'
            else:
                ovlp_sym = '—'
            _st  = p.get('status')
            tag  = ('sub_peak' if is_sub
                    else ('status_none' if _st is None else f'status_{_st}'))
            area_val = p.get('area', 0) or 0
            met_name = (p.get('assigned_metabolite')
                        or p.get('metabolite')
                        or _met_by_rank.get(p.get('rank'), '—'))
            self.tree.insert('', tk.END, iid=str(p['rank']), tags=(tag,),
                             values=(
                                 p['rank'],
                                 f"{p['ppm']:.4f}",
                                 f"{p.get('start_ppm', p['ppm']):.4f}",
                                 f"{p.get('end_ppm',   p['ppm']):.4f}",
                                 f"{p.get('width_ppm', p.get('width', 0)):.4f}",
                                 f"{p.get('snr', 0):.1f}",
                                 f"{area_val:.2f}",
                                 met_name,
                                 st_sym,
                                 ovlp_sym,
                             ))
        self._apply_status_colors()

    def _apply_status_colors(self):
        """Cor da linha determinada exclusivamente pelo status.
        None (sem status) → branco, sem marcação."""
        self.tree.tag_configure('status_none', background='white')
        for status, bg in STATUS_COLORS.items():
            self.tree.tag_configure(f'status_{status}', background=bg)

        for p in self.peaks:
            iid = str(p['rank'])
            if not self.tree.exists(iid):
                continue
            status = p.get('status')   # pode ser None
            if status is None:
                tag = 'status_none'
            else:
                tag = f'status_{status}'
            self.tree.item(iid, tags=(tag,))

    def _refresh_row(self, peak):
        iid = str(peak['rank'])
        if not self.tree.exists(iid):
            return
        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        # Same single-winner logic as _populate: best composite_score per rank
        _metabolites_rr = getattr(ann, '_metabolites', None)
        _met_by_rank = {}
        if _metabolites_rr:
            def _is_sing_rr(mp):
                mult = str(mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
                return mult in ('s', 'singlet', '', 'm', 'multiplet')
            _cands_rr = {}
            for _met_rr in _metabolites_rr:
                for _mp_rr in _met_rr.get('matched_peaks', []):
                    if _mp_rr.get('annotation_quality') == 'bad':
                        continue
                    _pr_rr = _mp_rr.get('peak_rank')
                    if _pr_rr is None:
                        continue
                    if _is_sing_rr(_mp_rr):
                        _elig = True
                    else:
                        _dj_rr = _mp_rr.get('delta_j_hz')
                        _elig = (_dj_rr is not None and float(_dj_rr) < 0.5)
                    if not _elig:
                        continue
                    _sc_rr = float(_mp_rr.get('composite_score', 0) or 0)
                    _cands_rr.setdefault(_pr_rr, []).append((_sc_rr, _met_rr.get('name', '')))
            for _pr_rr, _cl in _cands_rr.items():
                _met_by_rank[_pr_rr] = max(_cl, key=lambda x: x[0])[1]
        met_name = (peak.get('assigned_metabolite')
                    or peak.get('metabolite')
                    or _met_by_rank.get(peak.get('rank'), '—'))
        self.tree.item(iid, values=(
            peak['rank'],
            f"{peak['ppm']:.4f}",
            f"{peak['start_ppm']:.4f}",
            f"{peak['end_ppm']:.4f}",
            f"{peak['width_ppm']:.4f}",
            f"{peak['snr']:.1f}",
            f"{peak.get('area', 0) or 0:.2f}",
            met_name,
            self._status_symbol(peak.get('status', STATUS_UNDEFINED),
                                is_anchor=peak.get('is_anchor_line', False),
                                r2=peak.get('gaussian_r2')),
            "⚠" if peak.get('overlapping', False) else "—",
        ))
        self._apply_status_colors()

    def _sorted_peaks(self):
        key_map = {
            'rank':    lambda p: p['rank'],
            'ppm':     lambda p: p['ppm'],
            'start':   lambda p: p['start_ppm'],
            'end':     lambda p: p['end_ppm'],
            'width':   lambda p: p['width_ppm'],
            'snr':     lambda p: p['snr'],
            'status':  lambda p: (p.get('status') or ''),
            'overlap': lambda p: int(p.get('overlapping', False)),
        }
        return sorted(self.peaks,
                      key=key_map.get(self._sort_col, lambda p: p['rank']),
                      reverse=self._sort_reverse)

    def _sort(self, col):
        if self._sort_col == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col     = col
            self._sort_reverse = False
        labels = dict(rank='#', ppm='PPM', start='Start', end='End',
                      width='Width', snr='SNR', status='Quality', overlap='Overlap')
        for c in labels:
            arrow = (' ▲' if not self._sort_reverse else ' ▼') if c == col else ''
            self.tree.heading(c, text=labels[c] + arrow)
        self._populate()

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, status):
        """Set status on the currently selected peak."""
        sel = self.tree.selection()
        if not sel:
            return
        try:
            rank = int(sel[0])
        except (ValueError, TypeError):
            return
        peak = next((p for p in self.peaks if p['rank'] == rank), None)
        if peak is None:
            return
        peak['status'] = status
        self._refresh_row(peak)
        # Update info bar
        if self._current_peak and self._current_peak['rank'] == rank:
            self._update_info_bar(peak)
        # Undefined/Poor → queue for deconvolution
        if status in (STATUS_UNDEFINED, STATUS_POOR):
            self._queue_for_deconv(peak)
        # Good → remove from deconv queue if present
        elif status == STATUS_GOOD:
            self._dequeue_deconv(peak)

    def _queue_for_deconv(self, peak):
        """
        Add a peak marked Average/Poor to the Deconvolution Tool.
        Uses the peak's start_ppm/end_ppm as the region to deconvolve.
        The peak keeps its status in the list; after deconvolution the
        user can choose to accept or reject the result.
        """
        if self.app is None:
            return
        stage = getattr(self.app, 'stages', {}).get('peak_detection')
        if stage is None:
            return
        # Initialise controller if needed
        if stage._deconv_controller is None:
            from gui.notation.deconvolution import DeconvController
            stage._deconv_controller = DeconvController(stage.processor)
        ctrl = stage._deconv_controller
        # Build a proper peak dict and use add_peak to avoid duplicates
        # (ctrl.regions and ctrl._region_meta are computed @properties —
        # appending to them would silently be discarded)
        import copy
        pk = copy.copy(peak)
        pk['start_ppm']          = float(max(peak['start_ppm'], peak['end_ppm']))
        pk['end_ppm']            = float(min(peak['start_ppm'], peak['end_ppm']))
        pk['overlapping']        = True
        pk['from_overlap_search'] = False
        ctrl.add_peak(pk, target_ppm=peak['ppm'])
        # Enable deconv shortcut & button
        stage._overlaps = list(ctrl.regions)  # now reflects the added peak
        try:
            stage.btn_open_deconv.config(state='normal')
            self.app.btn_deconv_shortcut.config(state='normal')
        except Exception:
            pass
        stage.deconv_status.config(
            text=f"{len(ctrl.peaks)} region(s) queued for deconvolution",
            foreground='orange')

    def _dequeue_deconv(self, peak):
        """Remove a peak's region from the deconv queue when marked Good."""
        if self.app is None:
            return
        stage = getattr(self.app, 'stages', {}).get('peak_detection')
        if stage is None or stage._deconv_controller is None:
            return
        ctrl   = stage._deconv_controller
        new_lo = min(peak['start_ppm'], peak['end_ppm'])
        new_hi = max(peak['start_ppm'], peak['end_ppm'])
        # Remove the matching peak from ctrl.peaks (regions is a computed property)
        keep_idx = []
        for i, p in enumerate(ctrl.peaks):
            lo = min(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm']))
            hi = max(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm']))
            if abs(lo - new_lo) < 0.001 and abs(hi - new_hi) < 0.001:
                continue  # drop this one
            keep_idx.append(i)
        ctrl.peaks = [ctrl.peaks[i] for i in keep_idx]
        # Note: ctrl._region_meta is a computed @property — no need to update it
        stage._overlaps = list(ctrl.regions)

    def _update_info_bar(self, peak):
        n_samples   = self.all_intensities.shape[1]
        overlap_txt = "  ⚠ overlapping" if peak.get('overlapping') else ""
        status_txt  = f"  ·  {peak.get('status') or '—'}"
        r2_val      = peak.get('shape_r2')
        r2_txt      = (f"  ·  R²={r2_val:.3f}" if r2_val is not None
                       else "  ·  R²=n/a")
        self.peak_info_label.config(
            text=(f"Rank {peak['rank']}  ·  {peak['ppm']:.4f} ppm  ·  "
                  f"[{peak['start_ppm']:.4f} → {peak['end_ppm']:.4f}]  ·  "
                  f"Width {peak['width_ppm']:.4f} ppm  ·  "
                  f"SNR {peak['snr']:.1f}  ·  "
                  f"{n_samples} spectra"
                  f"{status_txt}"
                  f"{r2_txt}"
                  f"{overlap_txt}"),
            foreground=STATUS_FG.get(peak.get('status'), '#444'))

    def _open_region_alignment(self, peak):
        from gui.widgets.region_alignment_window import RegionAlignmentWindow

        def _on_apply(aligned, updated_peak):
            # Persist aligned data in this window so subsequent plots use it
            self.all_intensities = aligned
            # Redraw the current peak with updated data and position
            if self._current_peak is not None:
                self._plot_peak(self._current_peak)

        RegionAlignmentWindow(
            self.win,
            self.ppm,
            self.all_intensities,
            peak,
            self.app,
            on_apply=_on_apply)

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _on_right_click(self, event):
        row = self.tree.identify_row(event.y)
        if not row: return
        self.tree.selection_set(row)
        try: rank = int(row)
        except (ValueError, TypeError): return
        peak = next((p for p in self.peaks if p['rank'] == rank), None)
        if peak is None: return
        menu = tk.Menu(self.win, tearoff=0)
        menu.add_command(label='🔧 View in Deconvolution Tool',
                         command=lambda: self._send_to_deconv(peak))
        menu.add_command(label='📊 View per Sample',
                         command=lambda: self._open_per_sample(peak))
        menu.add_command(label='📐 Align region…',
                         command=lambda: self._open_region_alignment(peak))
        menu.add_separator()
        menu.add_command(label='🔀 View Candidates',
                         command=lambda: self._view_candidates(peak))
        menu.add_separator()
        menu.add_command(label='● Mark Good',
            command=lambda: self._set_status_on(peak, STATUS_GOOD))
        menu.add_command(label='~ Mark Average',
            command=lambda: self._set_status_on(peak, STATUS_UNDEFINED))
        menu.add_command(label='✗ Mark Poor',
            command=lambda: self._set_status_on(peak, STATUS_POOR))
        try: menu.tk_popup(event.x_root, event.y_root)
        finally: menu.grab_release()

    def _view_candidates(self, peak):
        """Open PeakResolveWindow with all metabolites that matched this peak rank."""
        if self.app is None:
            return
        ann_stage = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        metabolites = getattr(ann_stage, '_metabolites', None) if ann_stage else None
        if not metabolites and ann_stage and getattr(ann_stage, '_last_results_args', None):
            # Project was loaded from disk — reconstruct the annotation results window
            try:
                self.app.open_annot_shortcut()
                metabolites = getattr(ann_stage, '_metabolites', None)
            except Exception:
                pass
        if not metabolites:
            tk.messagebox.showinfo('View Candidates',
                                   'No annotation results available.\nRun annotation first.',
                                   parent=self.win)
            return
        rank = peak.get('rank')
        peak_ppm = float(peak.get('ppm', 0))
        candidates = []
        for met in metabolites:
            for mp in met.get('matched_peaks', []):
                if mp.get('peak_rank') == rank:
                    candidates.append((met, mp))
                    break
        # Also include nearby DB peaks (within 0.02 ppm) not matched to this rank
        if not candidates:
            for met in metabolites:
                for mp in met.get('matched_peaks', []):
                    if abs(float(mp.get('peak_ppm', mp.get('ppm_center', 0)) or 0) - peak_ppm) <= 0.02:
                        candidates.append((met, mp))
                        break
        if not candidates:
            tk.messagebox.showinfo('View Candidates',
                                   f'No candidates found for Peak #{rank} @ {peak_ppm:.4f} ppm.',
                                   parent=self.win)
            return
        from gui.widgets.peak_resolve_window import PeakResolveWindow
        _ai = self.all_intensities
        if _ai is not None:
            mean_intensity = (np.mean(_ai, axis=1) if _ai.ndim == 2
                              else np.asarray(_ai, dtype=float))
        else:
            mean_intensity = None
        PeakResolveWindow(
            self.win,
            candidates=candidates,
            peak_rank=rank,
            peak_ppm=peak_ppm,
            ppm=self.ppm,
            mean_intensity=mean_intensity,
            on_assign=None,
            peaks_list=self.peaks,
            eligible_overrides=getattr(self.app, 'eligible_overrides', {}))

    def _set_status_on(self, peak, status):
        peak['status'] = status
        self._refresh_row(peak)
        self._update_info_bar(peak)
        # Average/Poor → queue silenciosa para deconv (sem abrir janela)
        if status in (STATUS_UNDEFINED, STATUS_POOR):
            self._queue_for_deconv(peak)
        elif status == STATUS_GOOD:
            self._dequeue_deconv(peak)

    def _open_per_sample(self, peak):
        if self.app is None or self.all_intensities is None:
            return
        PeakSampleView(self.win, peak, self.ppm,
                       self.all_intensities, self.app)

    def _send_to_deconv(self, peak):
        """Envia o pico para o Deconv Tool e navega directamente para ele."""
        if self.app is None: return
        stage = getattr(self.app, 'stages', {}).get('peak_detection')
        if stage is None: return

        # Criar controller se não existir
        if stage._deconv_controller is None:
            from gui.notation.deconvolution import DeconvController
            stage._deconv_controller = DeconvController(stage.processor)
            stage._deconv_controller.set_peaks([])
        ctrl = stage._deconv_controller

        # Usar add_peak — retorna (region_idx, child_idx) e evita duplicados
        import copy
        p = copy.copy(peak)
        p['from_overlap_search'] = False
        p['overlapping']         = True
        region_idx, child_idx = ctrl.add_peak(p, target_ppm=peak['ppm'])

        # Actualizar estado do botão
        try:
            stage.btn_open_deconv.config(state='normal')
            stage._overlaps = list(ctrl.regions)
            stage.deconv_status.config(
                text=f'{len(ctrl.peaks)} peak(s) queued — click Open Deconv Tool',
                foreground='orange')
            self.app.btn_deconv_shortcut.config(state='normal')
        except Exception:
            pass

        # Abrir (ou reutilizar) o DeconvWindow e navegar directamente para o pico
        from gui.notation.deconvolution import DeconvWindow
        dw = DeconvWindow(self.app.root, ctrl, stage)
        # Aguardar que a janela seja desenhada antes de seleccionar
        dw.window.after(80, lambda: dw.select_region(region_idx, child_idx))

    def _on_select(self):
        sel = self.tree.selection()
        if not sel:
            return
        peak = next((p for p in self.peaks if p['rank'] == int(sel[0])), None)
        if peak:
            self._current_peak = peak
            self._plot_peak(peak)

    def _plot_peak(self, peak):
        self.ax.clear()

        center    = peak['ppm']
        p_lo      = min(peak['start_ppm'], peak['end_ppm'])
        p_hi      = max(peak['start_ppm'], peak['end_ppm'])
        peak_span = p_hi - p_lo
        fwhm      = peak.get('width_at_half', peak.get('width_ppm', peak_span))
        margin    = max(peak_span * 0.30, fwhm * 1.5, 0.005)

        lo = p_lo - margin
        hi = p_hi + margin

        mask = (self.ppm >= lo) & (self.ppm <= hi)
        if mask.sum() < 5:
            lo   = p_lo - peak_span * 0.5
            hi   = p_hi + peak_span * 0.5
            mask = (self.ppm >= lo) & (self.ppm <= hi)

        x_region = self.ppm[mask]
        if len(x_region) < 2:
            self.ax.text(0.5, 0.5, "Not enough data points",
                         transform=self.ax.transAxes,
                         ha='center', va='center', color='gray')
            self.plot_canvas.draw_idle()
            return

        n_samples     = self.all_intensities.shape[1]
        sample_colors = self._get_sample_colors(n_samples)

        y_max_visible = 0.0
        
        si = getattr(self.app, 'samples_info', []) if self.app else []
        g_colors = getattr(self.app, 'group_colors', {}) if self.app else {}

        shown_samples = []
        shown_groups = {}
        
        for i in range(n_samples):
            s = si[i] if i < len(si) else {}
            lbl = s.get('label', 'Unknown')
            if hasattr(self, 'group_vars') and self.group_vars: # Use checkboxes if they exist
                if lbl in self.group_vars and self.group_vars[lbl].get():
                    shown_samples.append(i)
                    shown_groups.setdefault(lbl, []).append(i)
            else: # Fallback to non-BLK
                if str(s.get('type', lbl)).upper() != 'BLK':
                    shown_samples.append(i)
                    shown_groups.setdefault(lbl, []).append(i)

        is_sub  = peak.get('is_sub_peak', False)
        fit_max = 0.0  # updated by sub-peak fit block

        # Raw spectra — dimmed for sub-peaks (context only)
        raw_alpha  = 0.12 if is_sub else 0.35
        mean_alpha = 0.25 if is_sub else 0.90
        for i in shown_samples:
            s_lbl = si[i].get('label', 'Unknown') if i < len(si) else ''
            s_col = g_colors.get(s_lbl, sample_colors[i])
            y_sample = self.all_intensities[mask, i]
            if len(y_sample) > 0:
                y_max_visible = max(y_max_visible, float(np.max(y_sample)))
            self.ax.plot(x_region, y_sample,
                         color=s_col, linewidth=0.8,
                         alpha=raw_alpha, zorder=2)

        # Group means — dimmed for sub-peaks
        for lbl, idxs in shown_groups.items():
            if not idxs: continue
            y_mean   = np.mean(self.all_intensities[mask][:, idxs], axis=1)
            mean_col = g_colors.get(lbl, '#111111')
            self.ax.plot(x_region, y_mean, color=mean_col,
                         linewidth=1.8 if not is_sub else 0.9,
                         alpha=mean_alpha, zorder=3,
                         label=f'{lbl} {"(raw)" if is_sub else "Mean"}')

        # Peak fill — skip for sub-peaks (fit fill is drawn later)
        if not is_sub:
            fill_mask = (self.ppm >= p_lo) & (self.ppm <= p_hi)
            x_fill    = self.ppm[fill_mask]
            if shown_samples and len(x_fill) >= 2:
                y_fill = np.max(
                    self.all_intensities[fill_mask][:, shown_samples], axis=1)
                fc = '#ffaaaa' if peak.get('overlapping', False) else '#aad4ff'
                self.ax.fill_between(x_fill, 0, y_fill,
                                     facecolor=fc, alpha=0.25,
                                     edgecolor='none', zorder=1)

        # Boundary lines
        for x_bound in (peak['start_ppm'], peak['end_ppm']):
            self.ax.axvline(x_bound,
                            color='#2e7d32' if is_sub else '#444444',
                            linewidth=1.2 if is_sub else 1.0,
                            linestyle='--', alpha=0.9, zorder=4)

        # Centre line — label in axes coords to stay inside ylim regardless
        self.ax.axvline(center,
                        color='#e63946', linewidth=1.2,
                        linestyle='--', alpha=0.6, zorder=4)
        self.ax.text(0.50, 0.97, f"δ {center:.4f} ppm",
                     transform=self.ax.transAxes,
                     ha='center', va='top',
                     fontsize=8, color='#e63946',
                     fontweight='bold', zorder=5)

        # Deconvolved sub-peak: fitted shape + fill + quality box
        if peak.get('is_sub_peak') and peak.get('height') and peak.get('width_ppm'):
            amp   = float(peak['height'])
            wid   = float(peak['width_ppm'])
            model = peak.get('model', 'Lorentzian')
            gamma = float(peak.get('gamma', 0.0))
            x_fit = np.linspace(lo, hi, 800)

            def _model_y(x, a, c, w, g=0.0):
                if model == 'Gaussian':
                    return a * np.exp(-(x - c)**2 / (2 * w**2))
                elif model == 'Pseudo-Voigt':
                    eta = float(peak.get('eta', 0.5))
                    lor = a * w**2 / ((x - c)**2 + w**2)
                    gau = a * np.exp(-(x - c)**2 / (2 * w**2))
                    return eta * lor + (1 - eta) * gau
                elif model in ('Skewed-Lorentzian', 'Skewed_Lorentzian'):
                    dx    = x - c
                    w_eff = np.maximum(w * (1 + g * np.sign(dx)), w * 0.05)
                    lor   = 1.0 / (1 + (dx / w_eff)**2)
                    skew  = np.clip(1.0 + np.sin(g)*dx/(w+1e-10), 0, None)
                    return a * lor * skew
                else:  # Lorentzian
                    return a * w**2 / ((x - c)**2 + w**2)

            fit_max = 0.0
            if 'area_per_sample' in peak and shown_groups:
                # Per-group mean fit using NNLS amplitudes
                areas = np.array(peak['area_per_sample'])
                import math
                for lbl, idxs in shown_groups.items():
                    if not idxs: continue
                    mean_area = float(np.mean(areas[idxs]))
                    if model == 'Gaussian':
                        mean_amp = mean_area / (wid * math.sqrt(2*math.pi))
                    else:
                        mean_amp = mean_area / (wid * math.pi) if wid > 0 else amp
                    mean_col = g_colors.get(lbl, '#2e7d32')
                    y_fit = _model_y(x_fit, mean_amp, center, wid, gamma)
                    fit_max = max(fit_max, float(np.max(y_fit)))
                    self.ax.plot(x_fit, y_fit, color=mean_col, linewidth=2.2,
                                 linestyle='--', alpha=0.95, zorder=6,
                                 label=f'{lbl} fit ({model})')
                    self.ax.fill_between(x_fit, 0, y_fit,
                                         color=mean_col, alpha=0.22, zorder=3)
            else:
                # Global fit amplitude
                y_fit = _model_y(x_fit, amp, center, wid, gamma)
                fit_max = float(np.max(y_fit))
                self.ax.plot(x_fit, y_fit, color='#2e7d32', linewidth=2.2,
                             linestyle='--', alpha=0.95, zorder=6,
                             label=f'{model} fit')
                self.ax.fill_between(x_fit, 0, y_fit,
                                     color='#2e7d32', alpha=0.28, zorder=3)
            snr  = peak.get('snr', 0)
            r2   = peak.get('r_squared', None)
            bic  = peak.get('bic', None)
            unc  = peak.get('cen_uncertainty', None)
            stat = peak.get('status', 'Undefined')
            snr_q = '● Good' if snr>=10 else ('◐ OK' if snr>=3 else '○ Low')
            lines = [f'Model : {model}',
                     f'SNR   : {snr:.1f}  {snr_q}',
                     f'Width : {wid:.4f} ppm (HWHM)']
            if r2  is not None:
                r2_q = '● Good' if r2>=0.98 else ('◐ OK' if r2>=0.90 else '○ Poor')
                lines.append(f'R²    : {r2:.4f}  {r2_q}')
            if bic is not None: lines.append(f'BIC   : {bic:.1f}')
            if unc is not None: lines.append(f'Δppm  : ±{unc:.4f}')
            lines.append(f'Status: {stat}')
            bc = ('#e8f5e9' if stat=='Good' else
                  '#fff3cd' if stat=='Undefined' else '#ffd6d6')
            self.ax.text(0.02, 0.97, '\n'.join(lines),
                         transform=self.ax.transAxes, va='top', ha='left',
                         fontsize=7.5, fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor=bc, alpha=0.88))
            self.ax.legend(loc='upper right', fontsize=7,
                           ncol=1, framealpha=0.7,
                           handlelength=1.5, handletextpad=0.4)

        self.ax.set_xlim(hi, lo)
        if is_sub:
            # Scale to show both the fit and the raw context spectra
            _fh = fit_max if fit_max > 0 else float(peak.get('height', 1.0))
            # Never clip the raw spectra — include y_max_visible so context is visible
            _top = max(_fh * 1.5, y_max_visible * 1.15)
            self.ax.set_ylim(bottom=0, top=_top)
        else:
            self.ax.set_ylim(bottom=0,
                             top=max(peak['height'] * 1.3, y_max_visible * 1.15))
        self.ax.set_xlabel('δ (ppm)', fontsize=9)
        self.ax.set_ylabel('Intensity', fontsize=9)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, alpha=0.25, linestyle=':')
        self.ax.set_facecolor('#fafafa')
        try:
            self.fig.tight_layout()
        except Exception:
            pass

        self.plot_canvas.draw()          # draw completo para blit capturar background
        self._attach_span_selector()      # recriado APÓS draw — blit correcto
        self._update_info_bar(peak)

    # ── Span selector ─────────────────────────────────────────────────────────

    def _attach_span_selector(self):
        """Recria o SpanSelector depois de ax.clear() + draw().
        Deve ser chamado APÓS plot_canvas.draw() para que o blit
        capture o background correcto."""
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
            except Exception:
                pass
            self._span_selector = None
        self._span_selector = SpanSelector(
            self.ax,
            onselect=self._on_span_selected,
            direction='horizontal',
            useblit=False,
            props=dict(alpha=0.30, facecolor='#f4a261'),
            interactive=False,
            drag_from_anywhere=True,
            minspan=0.001,
            button=3,   # right-click drag changes boundaries
        )

    def _on_span_selected(self, x_min, x_max):
        if self._current_peak is None or abs(x_max - x_min) < 0.001:
            return
        peak = self._current_peak

        # Axis is inverted: x_max = left (higher ppm), x_min = right (lower ppm)
        new_lo = min(x_min, x_max)
        new_hi = max(x_min, x_max)
        idx_a  = int(np.argmin(np.abs(self.ppm - new_lo)))
        idx_b  = int(np.argmin(np.abs(self.ppm - new_hi)))
        if idx_a > idx_b:
            idx_a, idx_b = idx_b, idx_a

        peak['start_ppm'] = float(self.ppm[idx_a])
        peak['end_ppm']   = float(self.ppm[idx_b])
        peak['start_idx'] = idx_a
        peak['end_idx']   = idx_b
        peak['width_ppm'] = float(abs(self.ppm[idx_b] - self.ppm[idx_a]))

        self._refresh_row(peak)
        # Defer redraw so SpanSelector finishes its own event cleanup first
        self.win.after(50, lambda: self._plot_peak(peak))

    # ── Scroll zoom ───────────────────────────────────────────────────────────

    def _zoom(self, factor, event=None):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if event is not None:
            try:
                xdata, _ = self.ax.transData.inverted().transform(
                    (event.x, event.y))
            except Exception:
                xdata = (xlim[0] + xlim[1]) / 2
        else:
            xdata = (xlim[0] + xlim[1]) / 2
        self.ax.set_xlim(xdata + (xlim[0] - xdata) * factor,
                         xdata + (xlim[1] - xdata) * factor)
        self.ax.set_ylim(ylim[0], ylim[1] * factor)
        self.plot_canvas.draw_idle()

    def _on_scroll_win(self, event):
        self._zoom(0.85 if event.delta > 0 else 1 / 0.85, event)

    def _on_scroll_up(self, event):
        self._zoom(0.85, event)

    def _on_scroll_down(self, event):
        self._zoom(1 / 0.85, event)


    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_sample_colors(self, n_samples):
        try:
            if (self.app is not None
                    and hasattr(self.app, 'samples_info')
                    and self.app.samples_info):
                palette = plt.cm.tab20(np.linspace(0, 1, max(n_samples, 1)))
                colors  = []
                for i in range(n_samples):
                    if i < len(self.app.samples_info):
                        label = self.app.samples_info[i].get('label', '')
                        c     = self.app.group_colors.get(label)
                        if c:
                            colors.append(c)
                            continue
                    colors.append(palette[i % len(palette)])
                return colors
        except Exception:
            pass
        palette = plt.cm.tab20(np.linspace(0, 1, max(n_samples, 1)))
        return [palette[i % len(palette)] for i in range(n_samples)]

    def _center(self):
        self.win.update_idletasks()
        w  = self.win.winfo_width()
        h  = self.win.winfo_height()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")