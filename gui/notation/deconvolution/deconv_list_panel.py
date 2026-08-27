# gui/notation/deconvolution/deconv_list_panel.py
"""
Left panel of the Deconvolution Tool.

Tree layout (show='tree headings'):
  #0 column  = "Region"  → shows region label + expand/collapse arrow
  data cols  = #  |  PPM  |  Start  |  End  |  SNR  |  W  |  Status

  ▼ ⚠ Region 1                     (parent — collapsible)
        3   3.2100  3.2050  3.2150  12.4  0.005  ~ Average
  ▼ Region 2
        7   1.3950  1.3900  1.4000   8.1  0.006  ~ Average
  ▼ ✅ Region 1  (after deconv)
        —   3.2180  ...                    ~ Average
        —   3.2040  ...                    ● Good

Rules:
  • Parent #0 text = "Region N"  (no number in the # column)
  • Parent data columns are ALL EMPTY except Status = n peaks
  • ⚠ on parent  → region contains a manually-Average peak (from Peak List)
  • ⚠ on child   → child status is Average (not yet resolved)
  • Green child  → only after marking Good (Ctrl+1)
  • Ctrl+1/2/3   → Good / Average / Poor on selected child
  • Ctrl+click   → multi-select regions
  • Ctrl+G       → group selected regions into one merged entry
"""
import tkinter as tk
from tkinter import ttk, messagebox

from gui.utils.quality_palette import (
    QUALITY_FG, QUALITY_LABEL, SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR)

# Same three tiers as the Peak List and the annotation window.
STATUS_GOOD      = SIGNAL_GOOD
STATUS_UNDEFINED = SIGNAL_AVERAGE
STATUS_POOR      = SIGNAL_POOR

# Projects saved before the vocabulary was unified stored this literal. It still
# means "no verdict reached", so it must keep counting as a status.
LEGACY_UNDEFINED = 'Undefined'

STATUS_COLORS = dict(QUALITY_FG)
STATUS_SYMBOL = dict(QUALITY_LABEL)


class DeconvListPanel:
    def __init__(self, parent, controller,
                 on_select_callback, on_apply_callback, on_save_callback,
                 on_apply_all_callback=None):
        self.parent          = parent
        self.controller      = controller
        self.on_select       = on_select_callback
        self.on_apply        = on_apply_callback
        self.on_save         = on_save_callback
        self.on_apply_all_cb = on_apply_all_callback

        self._current_region = None
        self._current_child  = None
        self._row_map        = {}   # iid → (region_idx, peak_idx, row_type)

        self.create_widgets()
        self.refresh_list()
        self._bind_keys()

    # ── Widgets ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent, width=540)
        self.frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.frame.pack_propagate(False)

        ttk.Label(self.frame, text="Signals for Deconvolution",
                  font=("Arial", 11, "bold")).pack(pady=5)

        self.info_label = ttk.Label(self.frame, text="", foreground="gray")
        self.info_label.pack(pady=2)

        # ── Treeview ──────────────────────────────────────────────────────────
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=('num', 'ppm', 'start', 'end', 'snr', 'width', 'status'),
            show='tree headings',
            selectmode='extended',   # Ctrl+click multi-select
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.tree.heading('#0', text='Region',
                          command=lambda: self._sort_by('#0'))
        self.tree.column('#0', width=110, minwidth=80, stretch=False)

        for cid, hd, w in [
            ('num',    '#',      48),
            ('ppm',    'δ (ppm)', 82),
            ('start',  'Start',  82),
            ('end',    'End',    82),
            ('snr',    'SNR',    58),
            ('width',  'W',      62),
            ('status', 'Status', 90),
        ]:
            self.tree.heading(cid, text=hd,
                              command=lambda c=cid: self._sort_by(c))
            self.tree.column(cid, width=w, anchor=tk.CENTER, minwidth=35)

        self._sort_col = None
        self._sort_rev = False

        # Row tags
        self.tree.tag_configure('region_overlap',
                                background='#fff3cd', foreground='#856404')
        self.tree.tag_configure('region_manual',
                                background='#fff0f8', foreground='#880055')
        self.tree.tag_configure('region_done',
                                background='#f0f0f0', foreground='#444')
        self.tree.tag_configure('region_saved',
                                background='#e8f5e9', foreground='#1b5e20')
        self.tree.tag_configure('region_grouped',
                                background='#e3f2fd', foreground='#0d47a1')
        self.tree.tag_configure('child_good',
                                background='#c8f7c5', foreground='#1a7a1a')
        self.tree.tag_configure('child_undefined',
                                background='#fef9c3', foreground='#7a6000')
        self.tree.tag_configure('child_poor',
                                background='#ffd6d6', foreground='#9a0000')
        self.tree.tag_configure('child_queued',
                                background='#e8f4ff', foreground='#0055aa')

        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.tree.bind('<Button-3>',         self._on_right_click if hasattr(self, '_on_right_click') else (lambda e: None))
        self.tree.bind('<Control-g>',        lambda e: self._group_selected())
        self.tree.bind('<Control-G>',        lambda e: self._group_selected())

        # ── Scrollable section: Parameters + Buttons ──────────────────────────
        bottom_outer = ttk.Frame(self.frame)
        bottom_outer.pack(fill=tk.X, side=tk.BOTTOM)

        _vsb_bottom = ttk.Scrollbar(bottom_outer, orient='vertical')
        _vsb_bottom.pack(side=tk.RIGHT, fill=tk.Y)

        self._bottom_canvas = tk.Canvas(
            bottom_outer, highlightthickness=0,
            yscrollcommand=_vsb_bottom.set, height=360)
        self._bottom_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        _vsb_bottom.config(command=self._bottom_canvas.yview)

        _inner = ttk.Frame(self._bottom_canvas)
        _win_id = self._bottom_canvas.create_window((0, 0), window=_inner, anchor='nw')

        def _on_inner_resize(event):
            self._bottom_canvas.configure(
                scrollregion=self._bottom_canvas.bbox('all'))
        def _on_canvas_resize(event):
            self._bottom_canvas.itemconfig(_win_id, width=event.width)
        _inner.bind('<Configure>', _on_inner_resize)
        self._bottom_canvas.bind('<Configure>', _on_canvas_resize)

        def _scroll_bottom(event):
            self._bottom_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        def _bind_scroll(event):
            self._bottom_canvas.bind_all('<MouseWheel>', _scroll_bottom)

        def _unbind_scroll(event):
            self._bottom_canvas.unbind_all('<MouseWheel>')

        self._bottom_canvas.bind('<Enter>', _bind_scroll)
        self._bottom_canvas.bind('<Leave>', _unbind_scroll)
        _inner.bind('<Enter>', _bind_scroll)
        _inner.bind('<Leave>', _unbind_scroll)

        # ── Parameters ────────────────────────────────────────────────────────
        ttk.Separator(_inner, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(_inner, text="Deconvolution Parameters",
                  font=("Arial", 10, "bold")).pack(pady=(0, 4))

        pf = ttk.Frame(_inner)
        pf.pack(fill=tk.X, padx=5)

        # ── row 0: Min SNR ────────────────────────────────────────────────────
        ttk.Label(pf, text="Min SNR:").grid(
            row=0, column=0, sticky=tk.W, pady=2, padx=4)
        self.min_snr_var = tk.DoubleVar(value=3.0)
        ttk.Entry(pf, textvariable=self.min_snr_var, width=7).grid(
            row=0, column=1, pady=2, padx=4, sticky=tk.W)
        ttk.Label(pf, text="Minimum SNR to accept a sub-signal (default: 3)",
                  font=("Arial", 7), foreground="#888").grid(
            row=0, column=2, columnspan=2, sticky=tk.W, padx=2)

        # ── row 1: Max Peaks ──────────────────────────────────────────────────
        ttk.Label(pf, text="Max Signals:").grid(
            row=1, column=0, sticky=tk.W, pady=2, padx=4)
        self.max_peaks_var = tk.IntVar(value=6)
        ttk.Spinbox(pf, from_=1, to=15,
                    textvariable=self.max_peaks_var, width=6).grid(
            row=1, column=1, pady=2, padx=4, sticky=tk.W)
        ttk.Label(pf, text="Maximum sub-signals per region (default: 6)",
                  font=("Arial", 7), foreground="#888").grid(
            row=1, column=2, columnspan=2, sticky=tk.W, padx=2)

        # ── row 2: Peak Model ─────────────────────────────────────────────────
        ttk.Label(pf, text="Lineshape Model:").grid(
            row=2, column=0, sticky=tk.W, pady=2, padx=4)
        self.model_var = tk.StringVar(value="Lorentzian")
        cb_model = ttk.Combobox(
            pf, textvariable=self.model_var,
            values=["Lorentzian", "Gaussian", "Pseudo-Voigt",
                    "Skewed-Lorentzian", "Onion (Auto-BIC)",
                    "MCR (Constrained)"],
            width=16, state="readonly")
        cb_model.grid(row=2, column=1, pady=2, padx=4, sticky=tk.W)
        cb_model.bind('<<ComboboxSelected>>', self._on_model_changed)

        _MODEL_HINTS = {
            "Lorentzian":       "Default for solution-state NMR; use in most cases",
            "Gaussian":         "Broadened peaks from inhomogeneity (rare in 1H)",
            "Pseudo-Voigt":     "Lorentzian+Gaussian blend; useful for asymmetric peaks",
            "Skewed-Lorentzian":"Asymmetric peak (residual J effect, exchange)",
            "Onion (Auto-BIC)": "Automatically selects the best model; slower",
            "MCR (Constrained)":"Constrained fit with theoretical J profiles (MCR-ALS)",
        }
        self._model_hint_lbl = ttk.Label(
            pf, text=_MODEL_HINTS["Lorentzian"],
            font=("Arial", 7), foreground="#888", wraplength=280)
        self._model_hint_lbl.grid(
            row=3, column=0, columnspan=4, sticky=tk.W, padx=4, pady=(0, 2))
        self._model_hints_map = _MODEL_HINTS

        def _update_model_hint(*_):
            h = self._model_hints_map.get(self.model_var.get(), "")
            self._model_hint_lbl.config(text=h)
        self.model_var.trace_add('write', _update_model_hint)

        # ── row 4: Min Width ──────────────────────────────────────────────────
        ttk.Label(pf, text="Min Width (ppm):").grid(
            row=4, column=0, sticky=tk.W, pady=2, padx=4)
        self.min_width_var = tk.DoubleVar(value=0.002)
        ttk.Entry(pf, textvariable=self.min_width_var, width=7).grid(
            row=4, column=1, pady=2, padx=4, sticky=tk.W)
        ttk.Label(pf, text="Minimum width (ppm); prevents fitting noise (default: 0.002)",
                  font=("Arial", 7), foreground="#888").grid(
            row=4, column=2, columnspan=2, sticky=tk.W, padx=2)

        # ── row 5: Max Width ──────────────────────────────────────────────────
        ttk.Label(pf, text="Max Width (ppm):").grid(
            row=5, column=0, sticky=tk.W, pady=2, padx=4)
        self.max_width_var = tk.DoubleVar(value=0.010)
        ttk.Entry(pf, textvariable=self.max_width_var, width=7).grid(
            row=5, column=1, pady=2, padx=4, sticky=tk.W)
        ttk.Label(pf, text="Maximum signal HWHM (ppm); limits broad fits (default: 0.010)",
                  font=("Arial", 7), foreground="#888").grid(
            row=5, column=2, columnspan=2, sticky=tk.W, padx=2)

        # ── Advanced Parameters + Reset buttons (row 6) ─────────────────────
        btn_row = ttk.Frame(pf)
        btn_row.grid(row=6, column=0, columnspan=4, sticky=tk.W, pady=(2, 0), padx=4)
        ttk.Button(btn_row, text='⚙ Advanced Parameters',
                   command=self._open_model_params_dialog).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text='↺ Reset Parameters',
                   command=self._reset_params).pack(side=tk.LEFT)

        # ── row 7: Baseline ───────────────────────────────────────────────────
        ttk.Label(pf, text="Baseline:").grid(
            row=7, column=0, sticky=tk.W, pady=(6, 2), padx=4)
        self.baseline_var = tk.StringVar(value="Linear")
        ttk.Combobox(
            pf, textvariable=self.baseline_var,
            values=["Linear", "Whittaker"],
            width=16, state="readonly").grid(
            row=7, column=1, pady=(6, 2), padx=4, sticky=tk.W)
        ttk.Label(pf,
                   text="Linear: fast and robust (default)  |  "
                       "Whittaker: better for curved baseline",
                  font=("Arial", 7), foreground="#888").grid(
            row=7, column=2, columnspan=2, sticky=tk.W, padx=2)

        # ── row 8: Candidate seeds ──────────────────────────────────────────
        ttk.Label(pf, text="Candidate Seeds:").grid(
            row=8, column=0, sticky=tk.W, pady=(6, 2), padx=4)
        det_row = ttk.Frame(pf)
        det_row.grid(row=8, column=1, columnspan=3, sticky=tk.W, pady=(6, 2), padx=4)
        self.detect_tops_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(det_row, text="Local maxima",
                        variable=self.detect_tops_var).pack(side=tk.LEFT, padx=(0, 10))
        self.detect_shoulders_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(det_row, text="Shoulders",
                        variable=self.detect_shoulders_var).pack(side=tk.LEFT)
        ttk.Label(pf,
                  text="Choose whether to seed the fit from local maxima, shoulders, or both.",
                  font=("Arial", 7), foreground="#888").grid(
            row=9, column=1, columnspan=3, sticky=tk.W, padx=4)

        # Advanced params — not shown inline, exposed via Parameters dialog
        self.shoulder_snr_var     = tk.DoubleVar(value=0.25)
        self.n_boot_var           = tk.IntVar(value=50)
        self._params_dlg          = None

        # ── Buttons ───────────────────────────────────────────────────────────
        ttk.Separator(_inner, orient='horizontal').pack(fill=tk.X, pady=6)

        self.apply_btn = ttk.Button(
            _inner, text="🔧 Deconvolve Selected  (Enter)",
            command=self._apply_selected, state="disabled")
        self.apply_btn.pack(fill=tk.X, padx=5, pady=2)

        self.group_btn = ttk.Button(
            _inner, text="🔗 Agroup  (Ctrl+G)",
            command=self._group_selected, state="disabled")
        self.group_btn.pack(fill=tk.X, padx=5, pady=2)

        self.revert_btn = ttk.Button(
            _inner, text="↩ Revert Selected Region",
            command=self._revert_selected, state="disabled")
        self.revert_btn.pack(fill=tk.X, padx=5, pady=2)

        self.revert_all_btn = ttk.Button(
            _inner, text="↩ Revert All",
            command=self._revert_all, state="disabled")
        self.revert_all_btn.pack(fill=tk.X, padx=5, pady=2)

        self.save_btn = ttk.Button(
            _inner, text="💾 Save Sub-signals to Signal List",
            command=self.on_save, state="disabled")
        self.save_btn.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(
            _inner,
            text="Ctrl+1 Good  ·  Ctrl+2 Average  ·  Ctrl+3 Poor\n"
                 "Ctrl+R Reset view  ·  Enter Deconvolve",
            font=("Arial", 7), foreground="gray").pack(pady=4)

        self.status_label = ttk.Label(
            _inner, text="No region selected",
            foreground="gray", font=("Arial", 8))
        self.status_label.pack(pady=2)

    # ── Key bindings ──────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.tree.bind('<Up>',     self._key_up)
        self.tree.bind('<Down>',   self._key_down)
        self.tree.bind('<Return>', self._key_enter)

    def _key_up(self, event):
        sel = self.tree.selection()
        if sel:
            prev = self.tree.prev(sel[0])
            if prev:
                self.tree.selection_set(prev)
                self.tree.focus(prev)
                self.tree.see(prev)
                self._on_tree_select(None)
        return "break"

    def _key_down(self, event):
        sel = self.tree.selection()
        if sel:
            nxt = self.tree.next(sel[0])
            if not nxt:
                ch = self.tree.get_children(sel[0])
                nxt = ch[0] if ch else None
            if nxt:
                self.tree.selection_set(nxt)
                self.tree.focus(nxt)
                self.tree.see(nxt)
                self._on_tree_select(None)
        return "break"

    def _key_enter(self, event):
        if (self._current_region is not None
                and str(self.apply_btn['state']) == 'normal'):
            self._apply_selected()

    # ── Tree population ───────────────────────────────────────────────────────

    def refresh_list(self):
        # Remember current selection to restore after refresh
        _prev_sel = self.tree.selection()
        _prev_iid = _prev_sel[0] if _prev_sel else None

        self._row_map = {}
        self.tree.delete(*self.tree.get_children(""))

        regions     = self.controller.get_regions()
        valid_count = 0

        for i, r in enumerate(regions):
            data = self.controller.get_region_data(i)
            if not data or len(data['ppm']) < 3:
                continue
            valid_count += 1

            done         = r['status'] == 'deconvolved'
            result       = self.controller.results.get(i, {})
            is_saved     = result.get('_saved', False)
            region_peaks = r.get('peaks', [])
            is_grouped   = any(p.get('_grouped_from') for p in region_peaks)

            has_manual = any(
                not p.get('from_overlap_search', True)
                for p in region_peaks)

            if done and is_saved:
                p_sym        = '💾'
                p_tag        = 'region_saved'
                p_status_col = 'saved'
            elif done:
                p_sym        = '✅'
                p_tag        = 'region_done'
                p_status_col = 'deconvolved'
            elif is_grouped:
                p_sym        = '🔗'
                p_tag        = 'region_grouped'
                p_status_col = f'{len(region_peaks)} pk (grouped)'
            elif has_manual:
                p_sym        = '⚠'
                p_tag        = 'region_manual'
                p_status_col = f'{len(region_peaks)} pk'
            else:
                p_sym        = ''
                p_tag        = 'region_overlap'
                p_status_col = f'{len(region_peaks)} pk'

            p_label = f'{p_sym} Region {valid_count}'.strip()
            p_iid   = f'region_{i}'

            # Compute inline values for the parent row from the first/main peak
            _rp = region_peaks[0] if region_peaks else {}
            _r_ppm   = f"{_rp.get('ppm', 0):.4f}"   if region_peaks else ''
            _r_start = f"{max(_rp.get('start_ppm', _rp.get('ppm', 0)),
                               _rp.get('end_ppm',   _rp.get('ppm', 0))):.4f}" if region_peaks else ''
            _r_end   = f"{min(_rp.get('start_ppm', _rp.get('ppm', 0)),
                               _rp.get('end_ppm',   _rp.get('ppm', 0))):.4f}" if region_peaks else ''
            _r_snr   = f"{_rp.get('snr', 0):.1f}"          if region_peaks else ''
            _r_w     = f"{_rp.get('width_ppm', _rp.get('width', 0)):.4f}" if region_peaks else ''

            try:
                self.tree.insert(
                    '', tk.END, iid=p_iid,
                    text=p_label,
                    values=(_rp.get('rank', '—'), _r_ppm, _r_start, _r_end,
                            _r_snr, _r_w, p_status_col),
                    open=True,
                    tags=(p_tag,))
            except Exception:
                pass
            self._row_map[p_iid] = (i, -1, 'region')

            if done and result.get('peaks'):
                for j, p in enumerate(result['peaks']):
                    c_iid  = f'row_{i}_{j}'
                    status = p.get('status', STATUS_UNDEFINED)
                    if status == STATUS_GOOD:
                        c_tag   = 'child_good'
                        st_cell = STATUS_SYMBOL[STATUS_GOOD]
                    elif status == STATUS_POOR:
                        c_tag   = 'child_poor'
                        st_cell = STATUS_SYMBOL[STATUS_POOR]
                    else:
                        c_tag   = 'child_undefined'
                        st_cell = STATUS_SYMBOL[STATUS_UNDEFINED]

                    pk_num = p.get('rank', f'{valid_count}.{j+1}')
                    try:
                        self.tree.insert(
                            p_iid, tk.END, iid=c_iid,
                            text='',
                            values=(
                                pk_num,
                                f"{p['ppm']:.4f}",
                                f"{p.get('start_ppm', p['ppm']):.4f}",
                                f"{p.get('end_ppm',   p['ppm']):.4f}",
                                f"{p.get('snr', 0):.1f}",
                                f"{p.get('width', p.get('width_ppm', 0)):.4f}",
                                st_cell),
                            tags=(c_tag,))
                    except Exception:
                        pass
                    self._row_map[c_iid] = (i, j, 'child')

            else:
                # Before deconvolution: no child rows — data shown inline in parent
                pass

        self.info_label.config(text=f"{valid_count} region(s) in queue")

        # Restore selection and scroll to it
        if _prev_iid and _prev_iid in self._row_map:
            try:
                self.tree.selection_set(_prev_iid)
                self.tree.focus(_prev_iid)
                self.tree.see(_prev_iid)
            except Exception:
                pass
        elif self._current_region is not None:
            candidate = f'region_{self._current_region}'
            if candidate in self._row_map:
                try:
                    self.tree.selection_set(candidate)
                    self.tree.focus(candidate)
                    self.tree.see(candidate)
                except Exception:
                    pass

        has_any = any(
            p.get('status') in (STATUS_GOOD, STATUS_UNDEFINED, STATUS_POOR,
                                LEGACY_UNDEFINED)
            for res in self.controller.results.values()
            for p in res.get('peaks', []))
        self.save_btn.config(state='normal' if has_any else 'disabled')

        has_done = any(r['status'] == 'deconvolved' for r in regions)
        self.revert_all_btn.config(state='normal' if has_done else 'disabled')
        if self._current_region is not None and self._current_region < len(regions):
            self.revert_btn.config(state='normal' if regions[self._current_region]['status'] == 'deconvolved' else 'disabled')
        else:
            self.revert_btn.config(state='disabled')

    # ── Sorting ───────────────────────────────────────────────────────────────

    def _sort_by(self, col):
        """Sort all region-level rows by the given column; toggle direction."""
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False

        # Update heading arrows
        col_names = {'#0': 'Region', 'num': '#', 'ppm': 'δ (ppm)',
                     'start': 'Start', 'end': 'End', 'snr': 'SNR',
                     'width': 'W', 'status': 'Status'}
        arrow = ' ▲' if not self._sort_rev else ' ▼'
        for c, name in col_names.items():
            suffix = arrow if c == col else ''
            if c == '#0':
                self.tree.heading(c, text=name + suffix)
            else:
                self.tree.heading(c, text=name + suffix)

        # Collect region rows and their sort key
        regions = [iid for iid in self.tree.get_children('')
                   if self._row_map.get(iid, (None, None, ''))[2] == 'region']

        def _key(iid):
            if col == '#0':
                return self.tree.item(iid, 'text')
            val = self.tree.set(iid, col)
            # For region rows, only status col has text; others are empty
            # Fall back to ppm of first child if available
            if not val:
                children = self.tree.get_children(iid)
                if children:
                    val = self.tree.set(children[0], col)
            try:
                return float(val)
            except (ValueError, TypeError):
                return val or ''

        regions.sort(key=_key, reverse=self._sort_rev)
        for i, iid in enumerate(regions):
            self.tree.move(iid, '', i)

    def _scroll_to_selected(self):
        """Scroll the treeview so the selected row is visible."""
        sel = self.tree.selection()
        if sel:
            self.tree.see(sel[-1])

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        self._scroll_to_selected()

        # Count region-level rows selected → enable group button
        n_regions = sum(
            1 for s in sel
            if self._row_map.get(s, (None, None, ''))[2] == 'region')
        self.group_btn.config(
            state='normal' if n_regions >= 2 else 'disabled')

        # Use last item for display
        iid  = sel[-1]
        info = self._row_map.get(iid)
        if info is None:
            return
        region_idx, peak_idx, row_type = info
        self._current_region = region_idx
        self.apply_btn.config(state='normal')

        if row_type == 'child':
            self._current_child = iid
            result = self.controller.results.get(region_idx, {})
            peaks  = result.get('peaks', [])
            if peak_idx < len(peaks):
                p = peaks[peak_idx]
                self.status_label.config(
                    text=f"#{p.get('rank','?')}  {p['ppm']:.4f} ppm  "
                         f"SNR {p.get('snr',0):.1f}  "
                         f"{p.get('status', STATUS_UNDEFINED)}",
                    foreground=STATUS_COLORS.get(
                        p.get('status', STATUS_UNDEFINED), 'gray'))
                self.on_select(region_idx, child_index=peak_idx)
                return
        else:
            self._current_child = None

        regions = self.controller.get_regions()
        if region_idx < len(regions):
            r = regions[region_idx]
            self.status_label.config(
                text=f"Region {region_idx+1}: {r['status']}",
                foreground='green' if r['status'] == 'deconvolved' else 'blue')
            self.revert_btn.config(state='normal' if r['status'] == 'deconvolved' else 'disabled')
        self.on_select(region_idx)

    def _on_right_click(self, event):
        row = self.tree.identify_row(event.y)
        if not row:
            return
        self.tree.selection_set(row)
        info = self._row_map.get(row)
        if info is None:
            return
        region_idx, peak_idx, row_type = info

        menu = tk.Menu(self.tree, tearoff=0)

        if row_type == 'region':
            regions = self.controller.get_regions()
            done    = (region_idx < len(regions) and
                       regions[region_idx]['status'] == 'deconvolved')
            if done:
                menu.add_command(
                    label='↩ Restore original signal',
                    command=lambda ri=region_idx: self._restore_peak(ri))
            else:
                menu.add_command(
                    label='🔧 Deconvolve',
                    command=lambda: self._apply_selected())
            menu.add_separator()
            menu.add_command(
                label='🗑 Remove from queue',
                command=lambda ri=region_idx: self._remove_peak(ri))

        elif row_type == 'child':
            menu.add_command(label='● Mark Good',
                command=lambda: self.set_child_status(STATUS_GOOD))
            menu.add_command(label=f'{STATUS_SYMBOL[STATUS_UNDEFINED]}',
                command=lambda: self.set_child_status(STATUS_UNDEFINED))
            menu.add_command(label='✗ Mark Poor',
                command=lambda: self.set_child_status(STATUS_POOR))

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _restore_peak(self, region_idx):
        """Undo deconvolution: remove result, restore original peak status."""
        self.controller.restore_peak(region_idx)
        self.refresh_list()
        self.status_label.config(
            text=f'Signal {region_idx+1} restored to original.',
            foreground='#555')

    def _remove_peak(self, region_idx):
        """Remove peak from the deconvolution queue entirely."""
        if region_idx < len(self.controller.peaks):
            self.controller.peaks.pop(region_idx)
            self.controller.results.pop(region_idx, None)
            self.controller.model_results.pop(region_idx, None)
        self.refresh_list()

    def _on_double_click(self, event):
        sel = self.tree.selection()
        if sel:
            info = self._row_map.get(sel[0])
            if info and info[2] == 'region':
                self._apply_selected()

    # ── Group selected regions ────────────────────────────────────────────────

    def _group_selected(self):
        """Merge ≥2 selected region rows into one entry spanning all their
        boundaries.  User then clicks Deconvolve manually.
        On Save, all original peaks from _grouped_from are removed.
        """
        sel = self.tree.selection()
        # Collect only top-level region rows
        region_idxs = []
        for iid in sel:
            info = self._row_map.get(iid)
            if info and info[2] == 'region':
                region_idxs.append(info[0])
        region_idxs = list(dict.fromkeys(region_idxs))  # dedupe, keep order

        if len(region_idxs) < 2:
            messagebox.showinfo(
                'Group Signals',
                'Ctrl+click 2 or more region rows to select,\n'
                'then press Ctrl+G.')
            return

        ctrl = self.controller
        # Gather all peak dicts from the selected regions
        sel_peaks = []
        for ri in region_idxs:
            meta = ctrl._region_meta
            if ri < len(meta):
                sel_peaks.extend(meta[ri].get('peaks', []))

        if not sel_peaks:
            return

        # Compute merged boundary
        lo  = min(min(p.get('start_ppm', p['ppm']),
                      p.get('end_ppm',   p['ppm'])) for p in sel_peaks)
        hi  = max(max(p.get('start_ppm', p['ppm']),
                      p.get('end_ppm',   p['ppm'])) for p in sel_peaks)

        # Build merged peak
        merged = {
            'ppm':           (lo + hi) / 2,
            'start_ppm':     hi,   # NMR axis: higher ppm = left
            'end_ppm':       lo,
            'height':        max(p['height'] for p in sel_peaks),
            'snr':           max(p.get('snr', 0) for p in sel_peaks),
            'width_ppm':     hi - lo,
            'width_at_half': hi - lo,
            'rank':          sel_peaks[0].get('rank', '?'),
            'overlapping':   True,
            'from_overlap_search': True,
            '_grouped_from': [p['ppm'] for p in sel_peaks],
        }

        # Add merged peak via controller (peak-centric API)
        ctrl.peaks.append(merged)

        # Remove the individual regions (reverse order to keep indices valid)
        for ri in sorted(region_idxs, reverse=True):
            if ri < len(ctrl.peaks) - 1:  # -1 because we just appended merged
                ctrl.peaks.pop(ri)
            ctrl.results.pop(ri, None)
            if hasattr(ctrl, 'excluded_samples'):
                ctrl.excluded_samples.pop(ri, None)

        self.refresh_list()

        # Select the new merged entry
        new_idx = len(ctrl.peaks) - 1
        new_iid = f'region_{new_idx}'
        # Find it by scanning _row_map (index may have shifted)
        for iid, info in self._row_map.items():
            if info[2] == 'region' and info[0] == new_idx:
                new_iid = iid
                break
        if self.tree.exists(new_iid):
            self.tree.selection_set(new_iid)
            self.tree.focus(new_iid)
            self.tree.see(new_iid)

        self.status_label.config(
            text=f'Grouped {len(region_idxs)} signals → 1 entry. '
                 f'Now click Deconvolve.',
            foreground='#1565c0')

    # ── Status assignment ─────────────────────────────────────────────────────

    def set_child_status(self, status):
        if not self._current_child:
            return
        info = self._row_map.get(self._current_child)
        if info is None or info[2] != 'child':
            return
        ri, pi, _ = info
        result = self.controller.results.get(ri, {})
        peaks  = result.get('peaks', [])
        if pi < len(peaks):
            peaks[pi]['status'] = status
            self.refresh_list()
            if self.tree.exists(self._current_child):
                self.tree.selection_set(self._current_child)
                self.tree.focus(self._current_child)

    # ── Apply / Save ──────────────────────────────────────────────────────────

    def _apply_selected(self):
        if self._current_region is None:
            return
        regions = self.controller.get_regions()
        # Only warn if deconvolved AND has actual sub-peak results
        _r = regions[self._current_region] if self._current_region < len(regions) else {}
        _has_results = bool(self.controller.results.get(self._current_region, {}).get('peaks'))
        if _r.get('status') == 'deconvolved' and _has_results:
            if not messagebox.askyesno(
                    "Re-deconvolve?",
                    "This region was already deconvolved.\n"
                    "This will replace the previous result.\n\nContinue?"):
                return
        self.on_apply(self._current_region, self._get_params())
        self.refresh_list()

    def _apply_all(self):
        regions = self.controller.get_regions()
        pending = [i for i, r in enumerate(regions)
                   if r['status'] != 'deconvolved']
        if not pending:
            messagebox.showinfo("Info", "All regions already deconvolved.")
            return
        if not messagebox.askyesno(
                "Deconvolve All",
                f"Deconvolve {len(pending)} pending region(s)?\n"
                "A progress window will appear.\n\nContinue?"):
            return
        params = self._get_params()
        if self.on_apply_all_cb:
            self.on_apply_all_cb(pending, params)
        else:
            self.controller.deconvolve_all(params)
            self.refresh_list()

    def _revert_selected(self):
        if self._current_region is not None:
            self._restore_peak(self._current_region)

    def _revert_all(self):
        if messagebox.askyesno("Revert All", "Are you sure you want to revert all deconvolved regions to their original state?"):
            regions = self.controller.get_regions()
            for i in range(len(regions)):
                if regions[i]['status'] == 'deconvolved':
                    self.controller.restore_peak(i)
            self.refresh_list()
            self.status_label.config(text="All regions reverted.", foreground="#555")

    def _on_model_changed(self, event=None):
        """Quando o modelo muda, mostra o resultado cacheado para esse modelo
        (se existir) — NÃO executa nova deconvolução automaticamente."""
        # Refresh the Parameters dialog if it is open
        try:
            if self._params_dlg and self._params_dlg.winfo_exists():
                self._populate_params_dialog()
        except Exception:
            pass
        if self._current_region is None:
            return
        new_model = self.model_var.get()
        cached = self.controller.get_model_result(self._current_region, new_model)
        if cached is not None:
            # Switch active result to this model's cached result and refresh display
            self.controller.results[self._current_region] = cached
            self.refresh_list()
            self.on_select(self._current_region)

    def _get_params(self):
        bsl = self.baseline_var.get() if hasattr(self, 'baseline_var') else 'Linear'
        return {
            'min_snr':            self.min_snr_var.get(),
            'max_peaks':          self.max_peaks_var.get(),
            'model':              self.model_var.get(),
            'min_width':          self.min_width_var.get(),
            'max_width':          self.max_width_var.get(),
            'detect_tops':        self.detect_tops_var.get(),
            'detect_shoulders':   self.detect_shoulders_var.get(),
            'shoulder_snr_factor':self.shoulder_snr_var.get(),
            'n_boot':             self.n_boot_var.get(),
            'baseline_method':    'whittaker' if 'Whittaker' in bsl else 'linear',
        }

    def _reset_params(self):
        self.min_snr_var.set(3.0)
        self.max_peaks_var.set(6)
        self.model_var.set("Lorentzian")
        self.min_width_var.set(0.002)
        self.max_width_var.set(0.010)
        self.detect_tops_var.set(True)
        self.detect_shoulders_var.set(True)
        self.shoulder_snr_var.set(0.25)
        self.n_boot_var.set(50)
        if hasattr(self, 'baseline_var'):
            self.baseline_var.set('Linear')
        self.status_label.config(text="Parameters reset to default", foreground="orange")

    def get_params(self):
        return self._get_params()

    # ── Parameters dialog ─────────────────────────────────────────────────────

    def _open_model_params_dialog(self):
        """Open (or raise) the advanced parameters dialog."""
        if self._params_dlg is not None:
            try:
                if self._params_dlg.winfo_exists():
                    self._params_dlg.lift()
                    self._params_dlg.focus_force()
                    return
            except Exception:
                pass
        dlg = tk.Toplevel(self.frame.winfo_toplevel())
        self._params_dlg = dlg
        dlg.title("Advanced Parameters")
        dlg.resizable(False, False)
        dlg.transient(self.frame.winfo_toplevel())
        # No grab_set — non-blocking

        self._params_dlg_frame = ttk.Frame(dlg, padding=10)
        self._params_dlg_frame.pack(fill=tk.BOTH, expand=True)
        self._populate_params_dialog()

        def _on_close():
            self._params_dlg = None
            dlg.destroy()
        dlg.protocol('WM_DELETE_WINDOW', _on_close)

        dlg.update_idletasks()
        x = self.frame.winfo_rootx() + self.frame.winfo_width() + 10
        y = self.frame.winfo_rooty() + 200
        dlg.geometry(f'+{x}+{y}')

    def _populate_params_dialog(self):
        """Rebuild the content of the Parameters dialog for the current model."""
        if self._params_dlg is None:
            return
        f = self._params_dlg_frame
        for w in f.winfo_children():
            w.destroy()
        model = self.model_var.get()

        ttk.Label(f, text=f"Model: {model}",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 6))

        # ── Shoulder detection (all models) ───────────────────────────────────
        sh = ttk.LabelFrame(f, text="Shoulder Detection", padding=6)
        sh.pack(fill=tk.X, pady=(0, 6))

        ttk.Checkbutton(sh, text="Detect local maxima",
                        variable=self.detect_tops_var).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(sh, text="Detect shoulders (inflection points)",
                        variable=self.detect_shoulders_var).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(sh, text="SNR factor:").grid(
            row=2, column=0, sticky=tk.W, padx=(18, 4), pady=2)
        ttk.Entry(sh, textvariable=self.shoulder_snr_var, width=7).grid(
            row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(sh,
                  text="How tall a shoulder must be relative to noise×SNR.\n"
                       "0.1 = strict  ·  0.4 = default  ·  0.8 = very sensitive",
                  font=("Arial", 8), foreground="gray").grid(
            row=3, column=0, columnspan=2, sticky=tk.W, padx=(18, 0))

        # ── Model-specific section ─────────────────────────────────────────────
        if model == "Pseudo-Voigt":
            mv = ttk.LabelFrame(f, text="Pseudo-Voigt", padding=6)
            mv.pack(fill=tk.X, pady=(0, 6))
            ttk.Label(mv, text="Initial η  (0 = Gaussian · 1 = Lorentzian):").grid(
                row=0, column=0, sticky=tk.W, padx=4, pady=2)
            if not hasattr(self, 'eta_init_var'):
                self.eta_init_var = tk.DoubleVar(value=0.5)
            ttk.Entry(mv, textvariable=self.eta_init_var, width=7).grid(
                row=0, column=1, sticky=tk.W, padx=4, pady=2)

        elif model == "Onion (Auto-BIC)":
            mv = ttk.LabelFrame(f, text="Onion — Bootstrap", padding=6)
            mv.pack(fill=tk.X, pady=(0, 6))
            ttk.Label(mv, text="Bootstrap samples:").grid(
                row=0, column=0, sticky=tk.W, padx=4, pady=2)
            ttk.Spinbox(mv, from_=10, to=500, increment=10,
                        textvariable=self.n_boot_var, width=7).grid(
                row=0, column=1, sticky=tk.W, padx=4, pady=2)
            ttk.Label(mv, text="More samples → more accurate, slower",
                      font=("Arial", 8), foreground="gray").grid(
                row=1, column=0, columnspan=2, sticky=tk.W, padx=4)

        ttk.Button(f, text="Close",
                   command=lambda: self._params_dlg.destroy() or
                   setattr(self, '_params_dlg', None)
                   ).pack(anchor=tk.E, pady=(8, 0))

    # ── Samples dialog ────────────────────────────────────────────────────────

    def _open_samples_dialog(self):
        """Checklist to include/exclude samples for the selected peak."""
        ri  = self._current_region
        app = getattr(self, '_app', None)
        if app is None or ri is None:
            return
        si   = getattr(app, 'samples_info', [])
        excl = set(self.controller.get_excluded_samples(ri))

        dlg = tk.Toplevel(self.frame.winfo_toplevel())
        dlg.title('Sample Inclusion')
        dlg.geometry('320x420')
        dlg.resizable(True, True)
        dlg.transient(self.frame.winfo_toplevel())
        dlg.grab_set()

        ttk.Label(dlg,
                  text='Uncheck to exclude sample (artefact).\n'
                       'Excluded regions are zeroed on Save.',
                  foreground='gray', font=('Arial', 8)
                  ).pack(pady=(8, 4), padx=10, anchor='w')

        cf = ttk.Frame(dlg)
        cf.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        canvas = tk.Canvas(cf, borderwidth=0, highlightthickness=0)
        vsb    = ttk.Scrollbar(cf, orient=tk.VERTICAL, command=canvas.yview)
        inner  = ttk.Frame(canvas)
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
            ttk.Checkbutton(
                row, variable=var,
                text=f"{s.get('name', 'S'+str(i))}  ({s.get('label', '')})"
            ).pack(side=tk.LEFT, padx=4)

        br = ttk.Frame(dlg)
        br.pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(br, text='All',
                   command=lambda: [v.set(True)  for _, v in vars_]
                   ).pack(side=tk.LEFT, padx=2)
        ttk.Button(br, text='None',
                   command=lambda: [v.set(False) for _, v in vars_]
                   ).pack(side=tk.LEFT, padx=2)

        def _ok():
            new_excl = {i for i, v in vars_ if not v.get()}
            self.controller.set_excluded_samples(ri, new_excl)
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(bf, text='Apply', command=_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=dlg.destroy).pack(side=tk.LEFT)