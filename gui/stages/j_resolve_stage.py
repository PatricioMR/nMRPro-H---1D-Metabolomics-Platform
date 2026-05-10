# gui/stages/j_resolve_stage.py
"""
J-Resolve Stage — Overlap Decomposition (step 4 in the workflow).

Embedded stage: left panel = controls + metabolite table,
right panel (app canvas area) = contribution heatmap.

Algorithm (4 phases, same as ResolvingMethodWindow):
  Phase 1  – User-marked 'Good' peaks → concentration proxy
  Phase 1b – Auto-anchor: clean (non-overlapping) matched peaks
  Phase 2  – Subtract known contributions from overlapping peaks
  Phase 3  – Iterative cascade: residual → new anchor
  Phase 4  – Status assignment

Apply button writes results to annotation_stage so Quantification
can use fractional areas when 'J-Resolve correction' is enabled.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import re as _re
import os
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .base_stage import BaseStage

# Default log directory — next to the project root
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')

# ── Colour constants (same as ResolvingMethodWindow) ─────────────────────────
_COL_ANCHOR      = '#2196F3'
_COL_AUTO        = '#00BCD4'
_COL_RESOLVED    = '#4CAF50'
_COL_RESOLVED_NLS= '#9C27B0'   # purple for NNLS-resolved
_COL_RESIDUAL    = '#FF9800'
_COL_UNKNOWN     = '#BDBDBD'


class JResolveStage(BaseStage):

    def __init__(self, parent, app, annotation_stage):
        super().__init__(parent, app)
        self.annotation_stage = annotation_stage

        # Data (refreshed on activate)
        self._metabolites  = []
        self._peaks        = []
        self._ppm          = None
        self._all_int      = None
        self._mean_int     = None

        # Algorithm results
        self._peak_overlap  = {}
        self._conc_est      = {}
        self._contributions = {}
        self._status_map    = {}   # met_id → status string
        self._anchor_peaks  = {}
        self._auto_anchored = set()
        self._j_validated   = set()   # met_ids with ≥1 multiplet ΔJ ≤ threshold

        # Canvas / figure (built once, reused)
        self._fig      = None
        self._ax_spec  = None   # top subplot — spectrum view
        self._ax_bar   = None   # bottom subplot — per-metabolite contribution bars
        self._canvas   = None

        # NNLS quality metrics (populated by _phase_nnls)
        self._nnls_residual_norm = None   # ‖Ax-b‖/‖b‖
        self._nnls_coverage      = {}     # {rank: b_hat_i/b_i}
        self._nnls_cond          = None   # condition number of A matrix
        self._nnls_rejected      = set()  # met_ids rejected by quality filters
        self._calib_cv_thresh    = None   # data-driven CV threshold (from anchors)
        self._calib_cov_thresh   = None   # data-driven coverage threshold

        # Logging
        self._log      = []          # list of strings for current run
        self._log_path = None        # path of last written log file

        # Currently selected metabolite in the table
        self._selected_mid = None

        self.create_widgets()

    # ─────────────────────────────────────────────────────────────────────────
    # BaseStage interface
    # ─────────────────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        sf = tk.Frame(self.frame, width=260, bg='#f0f0f0')
        sf.pack(side=tk.LEFT, fill=tk.Y)
        sf.pack_propagate(False)

        # Scrollable left panel
        canvas_sf = tk.Canvas(sf, bg='#f0f0f0', highlightthickness=0)
        vsb_sf = ttk.Scrollbar(sf, orient=tk.VERTICAL, command=canvas_sf.yview)
        canvas_sf.configure(yscrollcommand=vsb_sf.set)
        vsb_sf.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_sf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas_sf)
        _win_id = canvas_sf.create_window((0, 0), window=inner, anchor='nw')
        # Keep inner width == canvas width so widgets don't overflow right edge
        canvas_sf.bind('<Configure>',
                       lambda e, wid=_win_id: canvas_sf.itemconfig(
                           wid, width=e.width))
        inner.bind('<Configure>',
                   lambda e: canvas_sf.configure(
                       scrollregion=canvas_sf.bbox('all')))

        # ── Title ─────────────────────────────────────────────────────────────
        ttk.Label(inner, text='J-Resolve', font=('Arial', 12, 'bold')
                  ).pack(anchor=tk.W, padx=8, pady=(8, 0))
        ttk.Label(inner,
                  text='Overlap deconvolution via\nJ-ratio-corrected cascade',
                  font=('Arial', 8), foreground='gray'
                  ).pack(anchor=tk.W, padx=8, pady=(0, 6))
        ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=6, pady=2)

        # ── Options ───────────────────────────────────────────────────────────
        opt = ttk.LabelFrame(inner, text='Options', padding=6)
        opt.pack(fill=tk.X, padx=6, pady=4)

        self._iterate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text='Phase 3 — iterative cascade',
                        variable=self._iterate_var).pack(anchor=tk.W)

        tol_row = ttk.Frame(opt)
        tol_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(tol_row, text='Anchor tol (ppm):', font=('Arial', 8)
                  ).pack(side=tk.LEFT)
        self._anchor_tol = tk.DoubleVar(value=0.02)
        ttk.Spinbox(tol_row, from_=0.005, to=0.10, increment=0.005, width=7,
                    textvariable=self._anchor_tol,
                    format='%.3f').pack(side=tk.LEFT, padx=4)

        # ΔJ filter
        ttk.Separator(opt, orient='horizontal').pack(fill=tk.X, pady=(6, 2))
        self._dj_filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text='ΔJ filter for multiplets',
                        variable=self._dj_filter_var).pack(anchor=tk.W)
        dj_row = ttk.Frame(opt)
        dj_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(dj_row, text='  Max ΔJ (Hz):', font=('Arial', 8)
                  ).pack(side=tk.LEFT)
        self._dj_tol = tk.DoubleVar(value=1.5)
        ttk.Spinbox(dj_row, from_=0.1, to=5.0, increment=0.1, width=6,
                    textvariable=self._dj_tol,
                    format='%.1f').pack(side=tk.LEFT, padx=4)

        # Singlet anchoring
        self._singlet_anchor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text='Singlet→multiplet anchoring',
                        variable=self._singlet_anchor_var).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(opt, text='  (singlets locked to J-validated mets)',
                  font=('Arial', 7), foreground='#888'
                  ).pack(anchor=tk.W, padx=4)

        # Solver
        ttk.Separator(opt, orient='horizontal').pack(fill=tk.X, pady=(6, 2))
        ttk.Label(opt, text='Solver:', font=('Arial', 8, 'bold')).pack(anchor=tk.W)
        self._solver_var = tk.StringVar(value='cascade')
        for _val, _txt in [('cascade', 'Cascade only'),
                            ('nnls',    'Global NNLS'),
                            ('both',    'Cascade + NNLS fallback')]:
            ttk.Radiobutton(opt, text=_txt, variable=self._solver_var,
                            value=_val).pack(anchor=tk.W, padx=8)

        # NNLS quality thresholds
        ttk.Separator(opt, orient='horizontal').pack(fill=tk.X, pady=(6, 2))
        ttk.Label(opt, text='NNLS quality filters:',
                  font=('Arial', 8, 'bold')).pack(anchor=tk.W)
        ttk.Label(opt, text='  (upper bounds — auto-calibrated from anchors)',
                  font=('Arial', 7), foreground='#888').pack(anchor=tk.W)
        _qrow1 = ttk.Frame(opt)
        _qrow1.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(_qrow1, text='  Max CV:', font=('Arial', 8)
                  ).pack(side=tk.LEFT)
        self._nnls_cv_thr = tk.DoubleVar(value=0.50)
        ttk.Spinbox(_qrow1, from_=0.10, to=1.0, increment=0.05, width=5,
                    textvariable=self._nnls_cv_thr,
                    format='%.2f').pack(side=tk.LEFT, padx=4)
        self._calib_cv_lbl = ttk.Label(_qrow1, text='', font=('Arial', 7),
                                       foreground='#1565c0')
        self._calib_cv_lbl.pack(side=tk.LEFT, padx=2)
        _qrow2 = ttk.Frame(opt)
        _qrow2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(_qrow2, text='  Max cov:', font=('Arial', 8)
                  ).pack(side=tk.LEFT)
        self._nnls_cov_thr = tk.DoubleVar(value=1.40)
        ttk.Spinbox(_qrow2, from_=1.0, to=3.0, increment=0.05, width=5,
                    textvariable=self._nnls_cov_thr,
                    format='%.2f').pack(side=tk.LEFT, padx=4)
        self._calib_cov_lbl = ttk.Label(_qrow2, text='', font=('Arial', 7),
                                        foreground='#1565c0')
        self._calib_cov_lbl.pack(side=tk.LEFT, padx=2)

        # ── Run ───────────────────────────────────────────────────────────────
        ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=6, pady=4)
        self._btn_run = ttk.Button(inner, text='▶ Run J-Resolve',
                                   command=self._run, state='disabled')
        self._btn_run.pack(fill=tk.X, padx=8, pady=2)
        self._summary_lbl = ttk.Label(inner, text='', font=('Arial', 8),
                                      foreground='#555', wraplength=220,
                                      justify=tk.LEFT)
        self._summary_lbl.pack(anchor=tk.W, padx=8, pady=(0, 4), fill=tk.X)

        # ── Apply ─────────────────────────────────────────────────────────────
        ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=6, pady=2)
        self._btn_apply = ttk.Button(inner, text='✓ Apply to Quantification',
                                     command=self._apply_to_quant,
                                     state='disabled')
        self._btn_apply.pack(fill=tk.X, padx=8, pady=2)
        self._apply_lbl = ttk.Label(inner, text='', font=('Arial', 8),
                                    foreground='#2e7d32', wraplength=230)
        self._apply_lbl.pack(anchor=tk.W, padx=8, pady=(0, 4))

        # ── Export ────────────────────────────────────────────────────────────
        ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(inner, text='📋 Export CSV',
                   command=self._export).pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(inner, text='📄 View Log',
                   command=self._open_log).pack(fill=tk.X, padx=8, pady=2)
        self._log_path_lbl = ttk.Label(inner, text='', font=('Arial', 7),
                                       foreground='gray', wraplength=230)
        self._log_path_lbl.pack(anchor=tk.W, padx=8, pady=(0, 2))

        ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=6, pady=6)

        # ── Metabolite table ──────────────────────────────────────────────────
        ttk.Label(inner, text='Metabolites with overlap',
                  font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=8)
        ttk.Label(inner,
                  text='✓ Quantifiable  ⊙ Auto-anchored\n⚠ Resolved  ✗ Unresolvable',
                  font=('Arial', 7), foreground='#888'
                  ).pack(anchor=tk.W, padx=8, pady=(0, 2))
        self._anchor_lbl = ttk.Label(inner, text='', font=('Arial', 7),
                                     foreground='#1a6e1a', wraplength=230)
        self._anchor_lbl.pack(anchor=tk.W, padx=8, pady=(0, 2))

        tbl_frame = ttk.Frame(inner)
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 8))

        COLS = [
            ('status',   'Status',     90, tk.CENTER),
            ('name',     'Metabolite', 130, tk.W),
            ('rel_conc', 'Rel.conc',   65, tk.CENTER),
        ]
        vsb_t = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL)
        self._tree = ttk.Treeview(
            tbl_frame,
            columns=[c[0] for c in COLS],
            show='headings',
            selectmode='browse',
            height=12,
            yscrollcommand=vsb_t.set)
        vsb_t.config(command=self._tree.yview)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb_t.pack(side=tk.LEFT, fill=tk.Y)
        self._tree.bind('<<TreeviewSelect>>', self._on_met_select)

        for cid, hd, w, anch in COLS:
            self._tree.heading(cid, text=hd)
            self._tree.column(cid, width=w, anchor=anch, minwidth=30,
                              stretch=(cid == 'name'))

        self._tree.tag_configure('quant',    background='#e8ffe8',
                                 font=('Arial', 9, 'bold'))
        self._tree.tag_configure('auto',     background='#e0f7fa')
        self._tree.tag_configure('resolved', background='#fff8e0')
        self._tree.tag_configure('res_nnls',    background='#f3e5f5')
        self._tree.tag_configure('nnls_rej',    background='#fff3e0',
                                 foreground='#e65100')
        self._tree.tag_configure('unresolv',    background='#ffe8e8',
                                 foreground='#880000')

        # Deconvolve button — enabled when a metabolite is selected
        self._btn_deconv = ttk.Button(
            inner, text='🔬 Deconvolve region…',
            command=self._open_deconvolution_dialog,
            state='disabled')
        self._btn_deconv.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Placeholder (shown when no data)
        self._placeholder_lbl = ttk.Label(
            self.frame,
            text='Run Annotation Results first,\nthen come back to J-Resolve.',
            font=('Arial', 11), foreground='gray', justify=tk.CENTER)

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        self._refresh_data()
        self._build_canvas()
        if self._metabolites:
            self._placeholder_lbl.pack_forget()
            self._btn_run.config(state='normal')
            self._run()
        else:
            self._placeholder_lbl.pack(expand=True)
            self._btn_run.config(state='disabled')

    def on_deactivate(self):
        self.frame.pack_forget()
        self._teardown_canvas()

    # ─────────────────────────────────────────────────────────────────────────
    # Canvas management (embedded in app.canvas area)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_canvas(self):
        """Create/reuse matplotlib figure in the app's right-panel canvas slot."""
        try:
            self.app.toolbar.pack_forget()
        except Exception:
            pass
        try:
            self.app.canvas.get_tk_widget().pack_forget()
        except Exception:
            pass

        if self._fig is None:
            self._fig = plt.Figure(facecolor='#fafafa')
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 1, figure=self._fig,
                          height_ratios=[2, 1], hspace=0.45)
            self._ax_spec = self._fig.add_subplot(gs[0])
            self._ax_bar  = self._fig.add_subplot(gs[1])
            self._canvas  = FigureCanvasTkAgg(
                self._fig, master=self.app.right_panel)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _teardown_canvas(self):
        """Hide our canvas, restore the app's main canvas."""
        if self._canvas:
            try:
                self._canvas.get_tk_widget().pack_forget()
            except Exception:
                pass
        # Unpack both first, then re-pack in correct order (BOTTOM before TOP)
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

    # ─────────────────────────────────────────────────────────────────────────
    # Data refresh
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_data(self):
        ann = self.annotation_stage
        self._metabolites = list(getattr(ann, '_metabolites', None) or [])
        self._peaks       = list(getattr(ann, 'peaks', None) or
                                 getattr(getattr(self.app, 'stages', {})
                                         .get('peak_detection'), 'peaks', None) or [])
        self._ppm         = getattr(self.app, 'ppm', None)
        self._all_int     = getattr(self.app, 'all_intensities', None)
        self._mean_int    = self._build_mean()

    def _build_mean(self):
        if self._all_int is None:
            return None
        si  = getattr(self.app, 'samples_info', []) or []
        blk = {i for i, s in enumerate(si)
               if str(s.get('type', '')).upper() == 'BLK'}
        idxs = [i for i in range(self._all_int.shape[1]) if i not in blk]
        if not idxs:
            idxs = list(range(self._all_int.shape[1]))
        return np.mean(self._all_int[:, idxs], axis=1)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _met_by_id(self, mid):
        return next((m for m in self._metabolites if m['id'] == mid), None)

    def _met_by_name(self, name):
        return next((m for m in self._metabolites if m['name'] == name), None)

    def _peak_by_rank(self, rank):
        return next((p for p in self._peaks if p.get('rank') == rank), None)

    def _get_area(self, peak):
        if self._mean_int is None or self._ppm is None:
            return 0.0
        sp = peak.get('start_ppm', peak['ppm'])
        ep = peak.get('end_ppm',   peak['ppm'])
        lo, hi = min(sp, ep), max(sp, ep)
        if lo == hi:
            lo -= 0.01; hi += 0.01
        mask = (self._ppm >= lo) & (self._ppm <= hi)
        if mask.sum() < 2:
            return 0.0
        return float(abs(np.trapezoid(self._mean_int[mask], self._ppm[mask])))

    def _get_protons(self, met, mp):
        try:
            n = int(mp.get('protons') or 0)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
        db_ppm = float(mp.get('db_ppm', 0))
        entry  = next(
            (p for p in met.get('db_peaks', [])
             if abs(p.get('ppm_center', p.get('ppm', 0)) - db_ppm) < 0.02), None)
        if entry:
            try:
                n = int(entry.get('protons') or 0)
                if n > 0:
                    return n
            except (TypeError, ValueError):
                pass
        return 1

    def _effective_nh(self, met, mp):
        _pascal = {
            's':   [1],           'd':   [1, 1],
            't':   [1, 2, 1],     'q':   [1, 3, 3, 1],
            'p':   [1, 4, 6, 4, 1], 'quin': [1, 4, 6, 4, 1],
            'h':   [1, 5, 10, 10, 5, 1],
            'sex': [1, 5, 10, 10, 5, 1],
            'sep': [1, 6, 15, 20, 15, 6, 1],
            'dd':  [1, 1, 1, 1],  'ddd': [1, 1, 1, 1, 1, 1, 1, 1],
            'dt':  [1, 1, 2, 2, 1, 1],   'td':  [1, 2, 1, 1, 2, 1],
            'dq':  [1, 1, 3, 3, 3, 3, 1, 1],
        }
        n_h = self._get_protons(met, mp)
        pred = mp.get('predicted_lines') or []
        if len(pred) > 1:
            ris   = [float(l.get('rel_intensity', 1.0)) for l in pred]
            total = sum(ris)
            if total > 0:
                peak_ppm  = float(mp.get('peak_ppm') or mp.get('ppm') or 0.0)
                closest   = min(range(len(pred)),
                                key=lambda i: abs(
                                    float(pred[i].get('ppm', peak_ppm)) - peak_ppm))
                return n_h * ris[closest] / total
        mult   = (mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
        coeffs = _pascal.get(mult)
        if coeffs and len(coeffs) > 1:
            j_str  = str(mp.get('j_coupling') or '').strip()
            j_nums = [float(x) for x in _re.findall(r'\d+\.?\d*', j_str)
                      if float(x) > 0]
            if j_nums:
                mhz     = float(getattr(self.app, 'spectrometer_mhz', None) or 600.0)
                spacing = j_nums[0] / mhz
                centre  = float(mp.get('ppm_center') or mp.get('db_ppm') or 0.0)
                n_lines = len(coeffs)
                positions = [centre + (i - (n_lines - 1) / 2.0) * spacing
                             for i in range(n_lines)]
                peak_ppm  = float(mp.get('peak_ppm') or mp.get('ppm') or centre)
                dists     = [abs(peak_ppm - pos) for pos in positions]
                closest   = dists.index(min(dists))
                return n_h * coeffs[closest] / sum(coeffs)
        return float(n_h)

    def _build_line_index(self):
        import bisect
        entries = []
        for met in self._metabolites:
            mid = met['id']
            for mp in met['matched_peaks']:
                pred = mp.get('predicted_lines') or []
                if pred:
                    for line in pred:
                        ppm = float(line.get('ppm', 0) or 0)
                        if ppm:
                            entries.append((ppm, mid))
                else:
                    ppm = float(mp.get('peak_ppm') or mp.get('ppm') or 0)
                    if ppm:
                        entries.append((ppm, mid))
        entries.sort(key=lambda x: x[0])
        return entries, [e[0] for e in entries]

    def _is_clean_position(self, peak_ppm, exclude_mid, tolerance, entries, ppms):
        import bisect
        lo = bisect.bisect_left(ppms,  peak_ppm - tolerance)
        hi = bisect.bisect_right(ppms, peak_ppm + tolerance)
        for _, mid in entries[lo:hi]:
            if mid != exclude_mid:
                return False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Algorithm
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _mp_is_singlet(mp):
        mult = (mp.get('type') or mp.get('multiplicity') or 's').lower().strip()
        return mult in ('s', 'br s', 'brs', 'bs', '')

    def _build_overlap(self):
        """Build peak→metabolite overlap map, optionally applying ΔJ and singlet-anchor filters."""
        use_dj   = self._dj_filter_var.get()
        dj_tol   = float(self._dj_tol.get())
        use_sinc = self._singlet_anchor_var.get()

        # ── Step 1: identify J-validated metabolites ──────────────────────────
        # A metabolite is J-validated if it has ≥1 multiplet peak (anywhere in
        # the spectrum, including overlapping peaks) with ΔJ ≤ tolerance.
        # This includes peaks that are in the overlap map — we just need at
        # least one multiplet that confirms the metabolite's identity via J.
        self._j_validated = set()
        for met in self._metabolites:
            for mp in met['matched_peaks']:
                if self._mp_is_singlet(mp):
                    continue
                delta_j = mp.get('delta_j_hz')
                # If delta_j_hz is None, the annotation didn't measure J (e.g.
                # sub-peak without J measurement).  Fall back to checking if
                # the peak multiplicity matches the DB multiplicity as a proxy.
                if delta_j is not None:
                    if delta_j <= dj_tol:
                        self._j_validated.add(met['id'])
                        break
                else:
                    # No ΔJ measured but it IS a multiplet → treat as
                    # provisionally validated (we can't penalise the absence
                    # of J measurement data)
                    self._j_validated.add(met['id'])
                    break

        # ── Step 2: build overlap map with filters ────────────────────────────
        # First pass: record which metabolites are rejected per rank
        _dj_rejected = {}   # rank → {met_name}
        _all_by_rank = {}   # rank → [met_name]  (pre-filter)
        for met in self._metabolites:
            for mp in met['matched_peaks']:
                rank = mp.get('peak_rank')
                if rank is None:
                    continue
                _all_by_rank.setdefault(rank, []).append(met['name'])
                is_sing = self._mp_is_singlet(mp)
                if use_dj and not is_sing:
                    delta_j = mp.get('delta_j_hz')
                    if delta_j is not None and delta_j > dj_tol:
                        _dj_rejected.setdefault(rank, set()).add(met['name'])

        # Second pass: apply filters
        om = {}
        self._dj_skip_log = []   # for logging
        for met in self._metabolites:
            for mp in met['matched_peaks']:
                rank = mp.get('peak_rank')
                if rank is None:
                    continue
                is_sing = self._mp_is_singlet(mp)
                mid     = met['id']

                # ΔJ filter for multiplets
                if use_dj and not is_sing:
                    if met['name'] in _dj_rejected.get(rank, set()):
                        delta_j = mp.get('delta_j_hz', 0.0) or 0.0
                        self._dj_skip_log.append(
                            (rank, met['name'], delta_j, 'ΔJ'))
                        continue

                # Singlet-anchor filter: for singlet peaks, keep only J-validated
                # metabolites if at least one J-validated candidate exists
                if use_sinc and is_sing:
                    candidates_so_far = _all_by_rank.get(rank, [])
                    # Check if any candidate at this rank is J-validated
                    any_jval = any(
                        (m := self._met_by_name(n)) is not None
                        and m['id'] in self._j_validated
                        for n in candidates_so_far
                    )
                    if any_jval and mid not in self._j_validated:
                        self._dj_skip_log.append(
                            (rank, met['name'], None, 'singlet_not_jval'))
                        continue

                om.setdefault(rank, []).append(met['name'])

        self._peak_overlap = om

    def _run(self):
        self._refresh_data()
        if not self._metabolites:
            return
        self._build_overlap()
        self._conc_est.clear()
        self._contributions.clear()
        self._status_map.clear()
        self._anchor_peaks.clear()
        self._auto_anchored.clear()
        self._nnls_rejected.clear()
        self._nnls_residual_norm = None
        self._nnls_coverage      = {}
        self._nnls_cond          = None
        self._calib_cv_thresh    = None
        self._calib_cov_thresh   = None

        # ── Init log ──────────────────────────────────────────────────────────
        self._log = []
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._log_line(f'J-Resolve run  {ts}')
        self._log_line(f'Metabolites   : {len(self._metabolites)}')
        self._log_line(f'Peaks         : {len(self._peaks)}')
        self._log_line(f'Anchor tol    : {self._anchor_tol.get():.3f} ppm')
        self._log_line(f'Phase 3 iter  : {self._iterate_var.get()}')
        self._log_line(f'ΔJ filter     : {self._dj_filter_var.get()} '
                       f'(tol={self._dj_tol.get():.1f} Hz)')
        self._log_line(f'Singlet anchor: {self._singlet_anchor_var.get()}')
        self._log_line(f'Solver        : {self._solver_var.get()}')
        self._log_line()

        # Overlap map summary
        ovlp = {r: ns for r, ns in self._peak_overlap.items() if len(ns) > 1}
        self._log_line(f'=== Overlap map ({len(ovlp)} overlapping peaks) ===')
        for r, ns in sorted(ovlp.items()):
            pk  = self._peak_by_rank(r)
            ppm = pk['ppm'] if pk else '?'
            self._log_line(f'  Peak #{r:>4}  {ppm:.4f} ppm  →  {", ".join(ns)}')
        self._log_line()

        # Log filter summary
        n_jval = len(self._j_validated)
        n_dj_skip = len([x for x in self._dj_skip_log if x[3] == 'ΔJ'])
        n_sg_skip = len([x for x in self._dj_skip_log if x[3] == 'singlet_not_jval'])
        self._log_line(f'J-validated metabolites : {n_jval} '
                       f'(ΔJ ≤ {self._dj_tol.get():.1f} Hz on ≥1 multiplet)')
        self._log_line(f'ΔJ filter removals      : {n_dj_skip}')
        self._log_line(f'Singlet-anchor removals : {n_sg_skip}')
        if self._dj_skip_log:
            self._log_line('  Detail:')
            for rank, name, dj, reason in self._dj_skip_log[:200]:
                pk  = self._peak_by_rank(rank)
                ppm = pk['ppm'] if pk else 0.0
                if reason == 'ΔJ':
                    self._log_line(
                        f'  SKIP_ΔJ  {name:40} rank={rank:>4}  ppm={ppm:.4f}  '
                        f'ΔJ={dj:.2f} Hz > {self._dj_tol.get():.1f} Hz')
                else:
                    self._log_line(
                        f'  SKIP_SNG {name:40} rank={rank:>4}  ppm={ppm:.4f}  '
                        f'(not J-validated; J-validated candidate exists)')
            if len(self._dj_skip_log) > 200:
                self._log_line(f'  ... and {len(self._dj_skip_log)-200} more')
        self._log_line()

        self._phase1()
        self._phase1b()
        self._phase2()
        solver = self._solver_var.get()
        if solver in ('cascade', 'both') and self._iterate_var.get():
            self._phase3()
        if solver in ('nnls', 'both'):
            self._calibrate_nnls_thresholds()
            self._phase_nnls()
        self._phase4()

        # ── Final summary log ─────────────────────────────────────────────────
        self._log_line()
        self._log_line('=== Final status ===')
        for met in sorted(self._metabolites, key=lambda m: m['name']):
            mid  = met['id']
            st   = self._status_map.get(mid, '?')
            ce   = self._conc_est.get(mid)
            anch = self._anchor_peaks.get(mid, [])
            ce_s = f'{ce:.5g}' if ce else '—'
            self._log_line(
                f'  {st:<14}  {met["name"]:<35}  rel.conc={ce_s}  '
                f'anchors={anch}')
        self._log_line()
        self._log_line('=== Contributions per peak ===')
        for rank, info in sorted(self._contributions.items()):
            pk  = self._peak_by_rank(rank)
            ppm = pk['ppm'] if pk else '?'
            obs = info.get('_observed', 0)
            res = info.get('_residual_pct', 0)
            parts = [f'{n}: {v:.1f}%'
                     for n, v in info.items()
                     if not str(n).startswith('_')]
            self._log_line(
                f'  Peak #{rank:>4}  {ppm:.4f} ppm  obs={obs:.4g}'
                f'  [{", ".join(parts)}]  residual={res:.1f}%')

        self._write_log()
        self._refresh_ui()
        self._btn_apply.config(state='normal')

    def _phase1(self):
        """User-marked Good peaks → conc_est, BUT only from non-overlapping peaks.

        A 'Good' peak that is in an overlapping region cannot be used as a
        direct anchor because its observed area includes signal from other
        metabolites.  Such peaks are left for Phase 3 cascade to resolve.

        Exception: J-anchored singlets carry anchor_expected_I (derived from the
        companion J-confirmed multiplet) which represents the known contribution of
        this metabolite.  That value is used as the concentration probe regardless
        of overlap — the multiplet already anchored the concentration separately.
        """
        self._log_line('--- Phase 1: user-marked Good peaks → conc_est (clean peaks only) ---')
        for met in self._metabolites:
            mid = met['id']
            probes, anchors = [], []
            skipped_overlapped = []
            for mp in met['matched_peaks']:
                if mp.get('annotation_quality') != 'good':
                    continue
                rank = mp.get('peak_rank')
                peak = self._peak_by_rank(rank)
                if peak is None:
                    self._log_line(f'  SKIP {met["name"]} rank={rank}: peak not found')
                    continue
                # ── Reject peaks in the overlap map — unless J-anchor gives us
                # the exact expected contribution independent of overlap ────────
                competitors = [n for n in self._peak_overlap.get(rank, [])
                               if n != met['name']]
                if competitors:
                    # J-anchor bypass: use anchor_expected_I as probe if available
                    a_exp_I = mp.get('anchor_expected_I')
                    if (a_exp_I is not None
                            and mp.get('anchor_status') == 'clean'
                            and mp.get('j_met_confirmed')):
                        eff = self._effective_nh(met, mp)
                        if eff > 0:
                            probes.append(a_exp_I / eff)
                            anchors.append(rank)
                            self._log_line(
                                f'  {met["name"]:35} rank={rank:>4}  '
                                f'ppm={peak["ppm"]:.4f}  J-ANCHOR-SINGLET  '
                                f'anchor_I={a_exp_I:.4g}  eff_nH={eff:.3f}  '
                                f'probe={a_exp_I/eff:.4g}  '
                                f'(overlaps: {", ".join(competitors[:2])})')
                    else:
                        skipped_overlapped.append((rank, competitors))
                        self._log_line(
                            f'  SKIP_OVL {met["name"]:35} rank={rank:>4}  '
                            f'ppm={peak["ppm"]:.4f}  '
                            f'overlaps with: {", ".join(competitors[:3])}'
                            + (f' +{len(competitors)-3} more'
                               if len(competitors) > 3 else ''))
                    continue
                area = self._get_area(peak)
                if area <= 0:
                    self._log_line(f'  SKIP {met["name"]} rank={rank}: area=0')
                    continue
                eff  = self._effective_nh(met, mp)
                mult = mp.get('type') or mp.get('multiplicity', 's')
                nH   = self._get_protons(met, mp)
                self._log_line(
                    f'  {met["name"]:35} rank={rank:>4}  '
                    f'ppm={peak["ppm"]:.4f}  area={area:.4g}  '
                    f'mult={mult}  nH={nH}  eff_nH={eff:.3f}  '
                    f'probe={area/eff:.4g}')
                if eff > 0:
                    probes.append(area / eff)
                    anchors.append(rank)
            if probes:
                self._conc_est[mid]     = float(np.median(probes))
                self._anchor_peaks[mid] = anchors
                self._log_line(
                    f'  → ANCHORED {met["name"]}  '
                    f'conc_est={self._conc_est[mid]:.4g}  (from {len(probes)} clean probe(s))')
            else:
                good_count = sum(1 for mp in met['matched_peaks']
                                 if mp.get('annotation_quality') == 'good')
                reason = (f'all {len(skipped_overlapped)} Good peak(s) are overlapped'
                          if skipped_overlapped
                          else f'good_peaks={good_count}')
                self._log_line(
                    f'  → no anchor for {met["name"]}  ({reason})')
        self._log_line()

    def _phase1b(self):
        tol           = float(self._anchor_tol.get())
        entries, ppms = self._build_line_index()
        self._log_line(f'--- Phase 1b: auto-anchor (clean peaks, tol={tol:.3f} ppm) ---')

        # Ranks already claimed in Phase 1 — cannot be reused for a different met
        claimed_ranks = {r for ranks in self._anchor_peaks.values() for r in ranks}

        # ── Pass 1: collect all clean candidates across all metabolites ──────────
        # Each entry: (n_overlap_competitors, -area, met, mp, rank, area, eff, nH)
        # n_overlap_competitors = #metabolites sharing the detected peak (from overlap map)
        # Lower is better → most-exclusive peaks float to the top after sort.
        all_candidates = []
        for met in self._metabolites:
            mid = met['id']
            if mid in self._conc_est:
                continue
            for mp in met['matched_peaks']:
                rank = mp.get('peak_rank')
                if rank is None:
                    continue
                peak = self._peak_by_rank(rank)
                if peak is None:
                    continue
                pred_lines  = mp.get('predicted_lines') or []
                ppms_pred_b = [float(l['ppm']) for l in pred_lines if 'ppm' in l]
                if len(ppms_pred_b) >= 2:
                    half_span = (max(ppms_pred_b) - min(ppms_pred_b)) / 2.0
                    eff_tol   = min(tol, half_span + 0.007)
                else:
                    eff_tol   = tol
                # Overlap map competitor count (detected-peak level).
                # Only count competitors that are themselves anchor-quality
                # (ΔJ < 0.5 Hz or singlet).  Competitors with poor J-match are
                # disqualified as anchors and should not block this metabolite.
                _anchor_dj_thr = 0.5
                ovl_competitors = []
                for _cn in self._peak_overlap.get(rank, []):
                    if _cn == met['name']:
                        continue
                    _cmet = self._met_by_name(_cn)
                    if _cmet is None:
                        ovl_competitors.append(_cn)
                        continue
                    _cmp_mp = next(
                        (mp2 for mp2 in _cmet.get('matched_peaks', [])
                         if mp2.get('peak_rank') == rank), None)
                    if _cmp_mp is None:
                        ovl_competitors.append(_cn)
                        continue
                    _cdj = _cmp_mp.get('delta_j_hz')
                    _csing = self._mp_is_singlet(_cmp_mp)
                    # Competitor blocks only if: singlet (no J to disprove)
                    # OR ΔJ explicitly measured and good (< threshold).
                    # Multiplets with ΔJ=None are unvalidated — cannot confirm
                    # their presence, so they don't block anchoring.
                    if _csing or (_cdj is not None and _cdj < _anchor_dj_thr):
                        ovl_competitors.append(_cn)
                if ovl_competitors:
                    self._log_line(
                        f'  SKIP_OVL {met["name"]:35} rank={rank:>4}  '
                        f'ppm={peak["ppm"]:.4f}  — shared peak with: '
                        f'{", ".join(ovl_competitors[:3])}')
                    continue
                # DB line index competitor check (predicted-position level)
                clean = self._is_clean_position(
                    float(peak['ppm']), mid, eff_tol, entries, ppms)
                if not clean:
                    self._log_line(
                        f'  SKIP {met["name"]:35} rank={rank:>4}  '
                        f'ppm={peak["ppm"]:.4f}  — competitor within {eff_tol:.3f} ppm')
                    continue
                area = self._get_area(peak)
                if area <= 0:
                    continue
                eff = self._effective_nh(met, mp)
                if eff <= 0:
                    continue
                nH = self._get_protons(met, mp)
                n_comp = len(ovl_competitors)  # 0 for unique peaks
                all_candidates.append((n_comp, -area, met, mp, rank, area, eff, nH))

        # Sort: fewest detected-peak competitors first (unique peaks claimed first),
        # then highest area (most reliable signal).  Most-constrained metabolites
        # (fewest candidate peaks) are served before being squeezed out.
        all_candidates.sort(key=lambda c: (c[0], c[1]))

        # ── Pass 2: greedy exclusive assignment — each rank to at most one met ───
        met_probes  = {}   # mid → [probe_values]
        met_anchors = {}   # mid → [ranks]
        met_obj     = {}   # mid → met dict
        for (n_comp, _neg_area, met, mp, rank, area, eff, nH) in all_candidates:
            mid = met['id']
            if rank in claimed_ranks:
                self._log_line(
                    f'  SKIP_CLAIMED {met["name"]:35} rank={rank:>4}  '
                    f'ppm={(self._peak_by_rank(rank) or {}).get("ppm", 0):.4f}'
                    f'  — already claimed by another analyte')
                continue
            claimed_ranks.add(rank)
            met_probes.setdefault(mid, []).append(area / eff)
            met_anchors.setdefault(mid, []).append(rank)
            met_obj[mid] = met
            self._log_line(
                f'  CLEAN {met["name"]:35} rank={rank:>4}  '
                f'ppm={(self._peak_by_rank(rank) or {}).get("ppm", 0):.4f}  '
                f'area={area:.4g}  nH={nH}  eff_nH={eff:.3f}')

        # ── Commit concentration estimates ────────────────────────────────────────
        anchored_mids = set()
        for mid, probes in met_probes.items():
            met = met_obj[mid]
            self._conc_est[mid]     = float(np.median(probes))
            self._anchor_peaks[mid] = met_anchors[mid]
            self._auto_anchored.add(mid)
            anchored_mids.add(mid)
            self._log_line(
                f'  → AUTO-ANCHOR {met["name"]}  '
                f'conc_est={self._conc_est[mid]:.4g}  '
                f'(from {len(probes)} peak(s))')

        # Log metabolites that still have no anchor
        for met in self._metabolites:
            mid = met['id']
            if mid in self._conc_est:
                continue
            if not any(mp.get('peak_rank') is not None
                       for mp in met['matched_peaks']):
                self._log_line(f'  → no matched peaks for {met["name"]}')
            else:
                self._log_line(f'  → all peaks of {met["name"]} have competitors or were claimed')
        self._log_line()

    def _phase2(self, _cascade_iter=None):
        label = ('--- Phase 2: contribution subtraction ---'
                 if _cascade_iter is None
                 else f'--- Phase 2 (cascade iter {_cascade_iter}) ---')
        self._log_line(label)
        for rank, met_names in self._peak_overlap.items():
            if len(met_names) <= 1:
                continue
            peak = self._peak_by_rank(rank)
            if peak is None:
                continue
            obs_area = self._get_area(peak)
            if obs_area <= 0:
                continue
            contribs = {}
            unknown_names = []
            for name in met_names:
                met = self._met_by_name(name)
                if met is None:
                    continue
                mid = met['id']
                if mid not in self._conc_est:
                    unknown_names.append(name)
                    continue
                mp = next((m for m in met['matched_peaks']
                           if m.get('peak_rank') == rank), None)
                if mp is None:
                    continue
                eff = self._effective_nh(met, mp)
                contribs[mid] = self._conc_est[mid] * eff
            total_known = sum(contribs.values())
            residual    = max(0.0, obs_area - total_known)
            result = {mid: v / obs_area * 100 for mid, v in contribs.items()}
            result['_residual_pct'] = residual / obs_area * 100
            result['_observed']     = obs_area
            self._contributions[rank] = result

            pk_ppm = peak['ppm']
            known_str = ', '.join(
                f'{(self._met_by_id(mid) or {}).get("name","?")}: '
                f'{v/obs_area*100:.1f}%'
                for mid, v in contribs.items())
            self._log_line(
                f'  Peak #{rank:>4}  {pk_ppm:.4f} ppm  obs={obs_area:.4g}'
                f'  known=[{known_str}]'
                f'  residual={residual/obs_area*100:.1f}%'
                f'  unknowns={unknown_names}')
        self._log_line()

    def _phase3(self):
        self._log_line('--- Phase 3: iterative cascade ---')
        _iter = 0
        changed = True
        while changed:
            changed = False
            _iter  += 1
            self._log_line(f'  Cascade iteration {_iter}')
            for rank, info in list(self._contributions.items()):
                resid_pct = info.get('_residual_pct', 100)
                if resid_pct < 5:
                    continue
                met_names = self._peak_overlap.get(rank, [])
                unknowns  = [
                    self._met_by_name(n) for n in met_names
                    if self._met_by_name(n) is not None
                    and self._met_by_name(n)['id'] not in self._conc_est]
                if len(unknowns) != 1:
                    if unknowns:
                        self._log_line(
                            f'    Peak #{rank}: residual={resid_pct:.1f}%  '
                            f'{len(unknowns)} unknowns still — cannot resolve yet')
                    continue
                met = unknowns[0]
                mid = met['id']
                mp  = next((m for m in met['matched_peaks']
                            if m.get('peak_rank') == rank), None)
                if mp is None:
                    continue
                peak     = self._peak_by_rank(rank)
                obs_area = self._get_area(peak) if peak else 0
                known    = sum(
                    self._conc_est[self._met_by_name(n)['id']]
                    * self._effective_nh(
                        self._met_by_name(n),
                        next((m for m in self._met_by_name(n)['matched_peaks']
                              if m.get('peak_rank') == rank), {}))
                    for n in self._peak_overlap.get(rank, [])
                    if self._met_by_name(n) is not None
                    and self._met_by_name(n)['id'] in self._conc_est)
                residual_area = max(0.0, obs_area - known)
                eff = self._effective_nh(met, mp)
                if eff > 0 and residual_area > 0:
                    prev = self._conc_est.get(mid)
                    new_est = residual_area / eff
                    self._conc_est[mid] = (
                        float(np.mean([prev, new_est])) if prev else new_est)
                    self._anchor_peaks.setdefault(mid, []).append(rank)
                    changed = True
                    self._log_line(
                        f'    Peak #{rank:>4}  {(peak or {}).get("ppm",0):.4f} ppm  '
                        f'obs={obs_area:.4g}  known={known:.4g}  '
                        f'residual_area={residual_area:.4g}  '
                        f'eff_nH={eff:.3f}  '
                        f'→ NEW conc_est({met["name"]})={self._conc_est[mid]:.4g}'
                        + (f'  (prev={prev:.4g})' if prev else '  (first estimate)'))
                else:
                    self._log_line(
                        f'    Peak #{rank}: single unknown={met["name"]} '
                        f'but eff={eff:.3f} or residual_area={residual_area:.4g} ≤ 0')
            if changed:
                self._phase2(_cascade_iter=_iter)
        if _iter == 1 and not changed:
            self._log_line('  No cascade possible (no single-unknown peaks with residual ≥5%)')
        self._log_line()

    def _calibrate_nnls_thresholds(self):
        """Derive CV and coverage thresholds from anchor metabolites.

        Logic:
          For each anchor metabolite (conc_est known, has ≥2 non-overlapping
          matched peaks), estimate concentration independently from every peak:
              conc_i = area_i / eff_nH_i
          This gives an empirical CV for a metabolite we KNOW is present.
          The NNLS validation should use a threshold calibrated from this
          distribution, not an arbitrary constant.

        Threshold rules:
          cv_thresh  = max(mean_cv + 2·std_cv, 0.25)  capped at user spinbox value
          cov_thresh = max(max_anchor_cov · 1.25, 1.20) capped at user spinbox value

        Falls back to spinbox values if <3 usable anchors.
        """
        self._log_line('--- Calibrating NNLS thresholds from anchors ---')

        # Metabolites with a known conc_est from Phase 1 / 1b
        # (not NNLS candidates — those haven't been solved yet)
        anchor_mets = [m for m in self._metabolites
                       if m['id'] in self._conc_est]

        # Non-overlapping peaks: ranks shared by only 1 metabolite
        single_occ = {r for r, ns in self._peak_overlap.items() if len(ns) == 1}

        anchor_cvs  = []
        anchor_covs = []

        for met in anchor_mets:
            mid = met['id']
            ce  = self._conc_est[mid]

            per_peak = []
            for mp in met.get('matched_peaks', []):
                rank = mp.get('peak_rank')
                if rank is None:
                    continue
                # Only use non-overlapping peaks for unbiased estimates
                if rank not in single_occ and rank in self._peak_overlap:
                    continue
                peak = self._peak_by_rank(rank)
                if peak is None:
                    continue
                obs = self._get_area(peak)
                eff = self._effective_nh(met, mp)
                if obs <= 0 or eff <= 0:
                    continue
                conc_i = obs / eff
                per_peak.append(conc_i)
                # Coverage: how well does the anchor's conc_est predict this peak?
                anchor_covs.append(ce * eff / obs)

            if len(per_peak) >= 2:
                arr = np.array(per_peak)
                cv  = arr.std() / arr.mean() if arr.mean() > 0 else 0.0
                anchor_cvs.append(cv)
                self._log_line(
                    f'  anchor  {met["name"]:40}  n_peaks={len(per_peak)}'
                    f'  CV={cv:.3f}')

        user_cv  = self._nnls_cv_thr.get()
        user_cov = self._nnls_cov_thr.get()

        if len(anchor_cvs) >= 3:
            arr_cv   = np.array(anchor_cvs)
            cal_cv   = float(np.mean(arr_cv) + 2.0 * np.std(arr_cv))
            cal_cv   = max(cal_cv, 0.20)          # never tighter than 0.20
            # Don't go looser than the user's explicit setting
            cal_cv   = min(cal_cv, user_cv)
            self._calib_cv_thresh = cal_cv
            self._log_line(
                f'  CV distribution  mean={np.mean(arr_cv):.3f}'
                f'  std={np.std(arr_cv):.3f}'
                f'  → threshold={cal_cv:.3f}  (user cap={user_cv:.2f})')
        else:
            self._calib_cv_thresh = user_cv
            self._log_line(
                f'  Too few anchors with ≥2 peaks ({len(anchor_cvs)})'
                f' — using user value {user_cv:.2f}')

        if len(anchor_covs) >= 3:
            cal_cov = float(np.max(anchor_covs) * 1.25)
            cal_cov = max(cal_cov, 1.15)
            cal_cov = min(cal_cov, user_cov)
            self._calib_cov_thresh = cal_cov
            self._log_line(
                f'  Coverage  max_anchor={np.max(anchor_covs):.3f}'
                f'  → threshold={cal_cov:.3f}  (user cap={user_cov:.2f})')
        else:
            self._calib_cov_thresh = user_cov
            self._log_line(
                f'  Too few anchor peak coverages ({len(anchor_covs)})'
                f' — using user value {user_cov:.2f}')

        # Update UI labels to show calibrated values
        if self._calib_cv_thresh is not None:
            self._calib_cv_lbl.config(
                text=f'→ {self._calib_cv_thresh:.3f}')
        if self._calib_cov_thresh is not None:
            self._calib_cov_lbl.config(
                text=f'→ {self._calib_cov_thresh:.3f}')
        self._log_line()

    def _phase_nnls(self):
        """Global non-negative least squares solver.

        Builds a matrix A (n_overlapping_peaks × n_unknown_metabolites) where
        A[i,j] = effective_nH contribution of metabolite j at peak i.
        Known contributions (from Phase 1/1b/3 anchors) are subtracted from
        the observed areas first.  Solves NNLS(A, b) and stores results.
        """
        try:
            from scipy.optimize import nnls as _nnls
        except ImportError:
            self._log_line('--- Phase NNLS: scipy not available, skipping ---')
            return

        self._log_line('--- Phase NNLS: global non-negative least squares ---')

        # Collect peaks that have ≥2 candidates and still have unknowns
        overlap_ranks = sorted(
            r for r, ns in self._peak_overlap.items() if len(ns) > 1)
        if not overlap_ranks:
            self._log_line('  No overlapping peaks — nothing to do.')
            self._log_line()
            return

        # Identify unknown metabolites (not yet anchored) that appear in
        # at least one overlapping peak AND have been confirmed by annotation.
        # Restricting to annotation-confirmed ('good') metabolites prevents
        # unconfirmed yellow/None candidates from being spuriously "quantified"
        # by the NNLS solver in crowded biological spectra.
        #
        # Inclusion criterion: metabolite must have at least one matched_peak
        # with annotation_quality == 'good'.  Yellow competitors at the same
        # peak rank are also included so the NNLS can resolve genuine overlaps
        # between a confirmed analyte and a plausible competing candidate.
        _good_ranks = {
            mp.get('peak_rank')
            for m2 in self._metabolites
            for mp in m2.get('matched_peaks', [])
            if mp.get('annotation_quality') == 'good'
            and mp.get('peak_rank') is not None
        }

        unknown_mets = []
        _seen = set()
        for rank in overlap_ranks:
            for name in self._peak_overlap.get(rank, []):
                met = self._met_by_name(name)
                if met is None or met['id'] in _seen:
                    continue
                _seen.add(met['id'])
                if met['id'] in self._conc_est:
                    continue
                # Only include annotation-confirmed metabolites or direct
                # competitors at a peak rank where a 'good' candidate exists.
                _has_good = any(
                    mp.get('annotation_quality') == 'good'
                    for mp in met.get('matched_peaks', []))
                _competes_at_good_rank = any(
                    mp.get('peak_rank') in _good_ranks
                    for mp in met.get('matched_peaks', []))
                if _has_good or _competes_at_good_rank:
                    unknown_mets.append(met)

        if not unknown_mets:
            self._log_line('  All overlapping metabolites already anchored.')
            self._log_line()
            return

        n_peaks = len(overlap_ranks)
        n_mets  = len(unknown_mets)
        self._log_line(
            f'  Matrix: {n_peaks} peaks × {n_mets} unknown metabolites')

        met_idx = {met['id']: j for j, met in enumerate(unknown_mets)}

        A = np.zeros((n_peaks, n_mets))
        b = np.zeros(n_peaks)

        for i, rank in enumerate(overlap_ranks):
            peak = self._peak_by_rank(rank)
            if peak is None:
                continue
            obs = self._get_area(peak)
            # Subtract known contributions from anchored metabolites
            known = 0.0
            for name in self._peak_overlap.get(rank, []):
                met = self._met_by_name(name)
                if met is None or met['id'] not in self._conc_est:
                    continue
                mp = next((m for m in met['matched_peaks']
                           if m.get('peak_rank') == rank), None)
                if mp is None:
                    continue
                eff = self._effective_nh(met, mp)
                known += self._conc_est[met['id']] * eff
            b[i] = max(0.0, obs - known)

            # Fill matrix column for each unknown metabolite
            for name in self._peak_overlap.get(rank, []):
                met = self._met_by_name(name)
                if met is None or met['id'] not in met_idx:
                    continue
                mp = next((m for m in met['matched_peaks']
                           if m.get('peak_rank') == rank), None)
                if mp is None:
                    continue
                eff = self._effective_nh(met, mp)
                j = met_idx[met['id']]
                A[i, j] = eff

        # Remove all-zero rows (peaks with no unknown candidates)
        nonzero_rows = np.any(A > 0, axis=1)
        A_sub = A[nonzero_rows]
        b_sub = b[nonzero_rows]

        if A_sub.size == 0 or A_sub.shape[0] < 1:
            self._log_line('  Matrix empty after filtering — no solution.')
            self._log_line()
            return

        # Solve NNLS
        try:
            x, residual_norm = _nnls(A_sub, b_sub)
        except Exception as exc:
            self._log_line(f'  NNLS failed: {exc}')
            self._log_line()
            return

        self._log_line(f'  NNLS residual norm: {residual_norm:.4g}')

        # ── Quality metrics ───────────────────────────────────────────────────
        b_norm = float(np.linalg.norm(b_sub))
        self._nnls_residual_norm = residual_norm / b_norm if b_norm > 0 else None
        b_hat = A_sub @ x
        sub_ranks = [r for i, r in enumerate(overlap_ranks) if nonzero_rows[i]]
        self._nnls_coverage = {}
        for i, rank in enumerate(sub_ranks):
            bi = b_sub[i]
            self._nnls_coverage[rank] = float(b_hat[i] / bi) if bi > 0 else 0.0
        try:
            self._nnls_cond = float(np.linalg.cond(A_sub))
        except Exception:
            self._nnls_cond = None
        if self._nnls_residual_norm is not None:
            self._log_line(f'  Normalised residual: {self._nnls_residual_norm:.3f}')
        if self._nnls_cond is not None:
            self._log_line(f'  Matrix condition #:  {self._nnls_cond:.2g}')

        # Store candidate estimates (positive solutions only)
        _nnls_candidates = {}   # mid → (met, conc_est_value)
        for j, met in enumerate(unknown_mets):
            if x[j] > 0:
                _nnls_candidates[met['id']] = (met, float(x[j]))

        # ── Validate candidates before committing ──────────────────────────────
        _cv_src  = 'calibrated' if self._calib_cv_thresh  is not None else 'user'
        _cov_src = 'calibrated' if self._calib_cov_thresh is not None else 'user'
        self._log_line(
            f'  Validation thresholds: '
            f'CV≤{self._calib_cv_thresh or self._nnls_cv_thr.get():.3f} ({_cv_src})  '
            f'coverage≤{self._calib_cov_thresh or self._nnls_cov_thr.get():.3f} ({_cov_src})')
        validated, rejected = self._validate_nnls_solution(
            _nnls_candidates, overlap_ranks, met_idx, A_sub, b_sub)

        n_new = 0
        for mid, (met, ce) in validated.items():
            self._conc_est[mid] = ce
            self._anchor_peaks.setdefault(mid, [])
            n_new += 1
            self._log_line(f'  NNLS ✓ {met["name"]:40}  conc_est={ce:.4g}')

        for mid, (met, ce, reason) in rejected.items():
            self._nnls_rejected.add(mid)
            self._log_line(
                f'  NNLS ✗ {met["name"]:40}  conc_est={ce:.4g}  [{reason}]')

        # Re-run Phase 2 with validated estimates so contributions are updated
        if n_new > 0:
            self._phase2(_cascade_iter='nnls')

        self._log_line(
            f'  → {n_new} accepted, {len(rejected)} rejected by quality filters')
        self._log_line()

    def _validate_nnls_solution(self, candidates, overlap_ranks, met_idx,
                                A_sub, b_sub):
        """Quality-filter NNLS candidates before committing concentrations.

        For each candidate metabolite:
          1. Estimate concentration independently from each of its peaks
             (conc_i = area_i / eff_nH_i) — no NNLS involved.
          2. Multi-peak consistency: CV = std/mean across per-peak estimates.
             Reject if CV ≥ CV_THRESH (unless J-confirmed multiplet → relaxed).
          3. Coverage check: predicted contribution at each peak must be ≤ MAX_COV.
             Any peak where (conc_est * eff_nH / obs_area) > MAX_COV → reject.
          4. Single-peak metabolites: only coverage check applies.

        Returns (validated, rejected) dicts:
          validated: {mid: (met, conc_est)}
          rejected:  {mid: (met, conc_est, reason_string)}
        """
        # Use data-driven calibrated thresholds when available;
        # fall back to UI spinbox values.
        CV_THRESH  = (self._calib_cv_thresh
                      if self._calib_cv_thresh is not None
                      else self._nnls_cv_thr.get())
        MAX_COV    = (self._calib_cov_thresh
                      if self._calib_cov_thresh is not None
                      else self._nnls_cov_thr.get())
        CV_J_THRESH = min(CV_THRESH + 0.20, 0.90)   # relaxed for J-confirmed
        MIN_CONC_FRAC = 0.01   # reject if conc_est < 1% of largest anchor conc

        # Largest anchored concentration for plausibility floor
        anchor_concs = [ce for mid, ce in self._conc_est.items()
                        if mid not in candidates]
        conc_floor = max(anchor_concs) * MIN_CONC_FRAC if anchor_concs else 0.0

        validated = {}
        rejected  = {}

        for mid, (met, ce) in candidates.items():
            name = met['name']

            # J-confirmation: has ≥1 multiplet with ΔJ ≤ threshold
            j_confirmed = mid in self._j_validated

            # Peaks in the overlap set where this metabolite appears
            my_ranks = [r for r in overlap_ranks
                        if name in self._peak_overlap.get(r, [])]

            # ── Per-peak independent concentration estimates ──────────────────
            per_peak_concs = []
            coverage_ok    = True

            for rank in my_ranks:
                peak = self._peak_by_rank(rank)
                if peak is None:
                    continue
                obs = self._get_area(peak)
                if obs <= 0:
                    continue
                mp = next((m for m in met.get('matched_peaks', [])
                           if m.get('peak_rank') == rank), None)
                if mp is None:
                    continue
                eff = self._effective_nh(met, mp)
                if eff <= 0:
                    continue

                conc_i = obs / eff
                per_peak_concs.append(conc_i)

                # Coverage check: does ce explain too much of this peak?
                cov = ce * eff / obs
                if cov > MAX_COV:
                    coverage_ok = False
                    self._log_line(
                        f'    coverage fail  {name}  rank={rank}'
                        f'  cov={cov:.2f}  (>{MAX_COV})')

            if not coverage_ok:
                rejected[mid] = (met, ce, f'coverage>{MAX_COV:.0%}')
                continue

            # ── Multi-peak consistency ────────────────────────────────────────
            if len(per_peak_concs) >= 2:
                arr  = np.array(per_peak_concs)
                mean = arr.mean()
                cv   = arr.std() / mean if mean > 0 else 999.0
                thresh = CV_J_THRESH if j_confirmed else CV_THRESH
                if cv > thresh:
                    rejected[mid] = (
                        met, ce,
                        f'CV={cv:.2f}>{thresh:.2f}'
                        + (' (J-relaxed)' if j_confirmed else ''))
                    self._log_line(
                        f'    CV fail  {name}  CV={cv:.2f}  thresh={thresh:.2f}')
                    continue

            # ── Plausibility floor ────────────────────────────────────────────
            if ce < conc_floor:
                rejected[mid] = (met, ce, f'conc<floor({conc_floor:.3g})')
                continue

            validated[mid] = (met, ce)

        # Summary breakdown
        by_reason = {}
        for mid, (met, ce, reason) in rejected.items():
            key = reason.split('=')[0].split('<')[0].split('>')[0]
            by_reason[key] = by_reason.get(key, 0) + 1
        if by_reason:
            self._log_line('  Rejection summary: ' +
                           ', '.join(f'{k}:{v}' for k, v in sorted(by_reason.items())))

        return validated, rejected

    def _phase4(self):
        solver = self._solver_var.get()
        for met in self._metabolites:
            mid = met['id']
            if mid in self._conc_est:
                anchors = self._anchor_peaks.get(mid, [])
                unambiguous = [
                    r for r in anchors
                    if len([n for n in self._peak_overlap.get(r, [])
                            if n != met['name']]) == 0]
                if unambiguous:
                    self._status_map[mid] = (
                        'auto_anchored' if mid in self._auto_anchored
                        else 'quantifiable')
                elif anchors:
                    self._status_map[mid] = 'resolved'
                else:
                    # No anchor peaks recorded → came from NNLS global solve
                    self._status_map[mid] = 'resolved_nnls'
            elif mid in self._nnls_rejected:
                self._status_map[mid] = 'nnls_rejected'
            else:
                self._status_map[mid] = 'unresolvable'

    # ─────────────────────────────────────────────────────────────────────────
    # UI refresh
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_ui(self):
        self._update_anchor_label()
        self._populate_table()
        self._update_summary()
        self._draw_chart()

    def _on_met_select(self, _event=None):
        sel = self._tree.selection()
        if sel:
            mid_str = sel[0]
            self._selected_mid = mid_str
            met = next((m for m in self._metabolites
                        if str(m['id']) == mid_str), None)
            if met is not None:
                self._btn_deconv.config(state='normal')
                self._draw_met_detail(met)
                return
        self._btn_deconv.config(state='disabled')
        self._draw_overview()

    def _draw_chart(self):
        if self._selected_mid:
            met = next((m for m in self._metabolites
                        if str(m['id']) == self._selected_mid), None)
            if met:
                self._draw_met_detail(met)
                return
        self._draw_overview()

    def _overlapping_met_ids(self):
        ids = set()
        for rank, names in self._peak_overlap.items():
            if len(names) > 1:
                for name in names:
                    met = self._met_by_name(name)
                    if met:
                        ids.add(met['id'])
        return ids

    def _update_summary(self):
        ovlp_ids = self._overlapping_met_ids()
        n_q  = sum(1 for mid, v in self._status_map.items()
                   if v == 'quantifiable'  and mid in ovlp_ids)
        n_a  = sum(1 for mid, v in self._status_map.items()
                   if v == 'auto_anchored' and mid in ovlp_ids)
        n_r  = sum(1 for mid, v in self._status_map.items()
                   if v == 'resolved'      and mid in ovlp_ids)
        n_rn = sum(1 for mid, v in self._status_map.items()
                   if v == 'resolved_nnls' and mid in ovlp_ids)
        n_rj = sum(1 for mid, v in self._status_map.items()
                   if v == 'nnls_rejected' and mid in ovlp_ids)
        n_u  = sum(1 for mid, v in self._status_map.items()
                   if v == 'unresolvable'  and mid in ovlp_ids)
        parts = [f"✓ {n_q} quantifiable  ⊙ {n_a} auto-anchored",
                 f"⚠ {n_r} cascade-resolved"]
        if n_rn:
            parts.append(f"∑ {n_rn} NNLS-resolved")
        if n_rj:
            parts.append(f"⊘ {n_rj} NNLS-rejected (low quality)")
        parts.append(f"✗ {n_u} unresolvable")
        self._summary_lbl.config(text='\n'.join(parts))

    def _update_anchor_label(self):
        ovlp_ids = self._overlapping_met_ids()
        anchors = [m['name'] for m in self._metabolites
                   if m['id'] in self._conc_est and m['id'] not in ovlp_ids]
        auto_names = [m['name'] for m in self._metabolites
                      if m['id'] in self._auto_anchored
                      and m['id'] not in ovlp_ids]
        if anchors or auto_names:
            parts = []
            if anchors:
                parts.append('User: ' + ', '.join(anchors))
            if auto_names:
                parts.append('Auto: ' + ', '.join(auto_names))
            self._anchor_lbl.config(text=' | '.join(parts))
        else:
            n_good = sum(1 for m in self._metabolites
                         for mp in m['matched_peaks']
                         if mp.get('annotation_quality') == 'good')
            if n_good == 0 and not self._auto_anchored:
                self._anchor_lbl.config(
                    text='⚠ No anchors — mark peaks Good in Annotation Results')
            else:
                self._anchor_lbl.config(
                    text=f'{n_good} Good + {len(self._auto_anchored)} auto-anchored')

    def _populate_table(self):
        self._tree.delete(*self._tree.get_children(''))
        ovlp_ids = self._overlapping_met_ids()
        order = {'quantifiable': 0, 'auto_anchored': 1,
                 'resolved': 2, 'resolved_nnls': 3,
                 'nnls_rejected': 4, 'unresolvable': 5}
        mets = sorted(
            [m for m in self._metabolites if m['id'] in ovlp_ids],
            key=lambda m: (
                order.get(self._status_map.get(m['id'], 'unresolvable'), 5),
                m['name']))
        for met in mets:
            mid    = met['id']
            status = self._status_map.get(mid, 'unresolvable')
            ce     = self._conc_est.get(mid)
            tag    = {'quantifiable': 'quant', 'auto_anchored': 'auto',
                      'resolved': 'resolved', 'resolved_nnls': 'res_nnls',
                      'nnls_rejected': 'nnls_rej',
                      'unresolvable': 'unresolv'}.get(status, 'unresolv')
            sym    = {'quantifiable':  '✓ Quantifiable',
                      'auto_anchored': '⊙ Auto-anchored',
                      'resolved':      '⚠ Resolved (cascade)',
                      'resolved_nnls': '∑ Resolved (NNLS)',
                      'nnls_rejected': '⊘ NNLS-rejected',
                      'unresolvable':  '✗ Unresolvable'}.get(status, status)
            self._tree.insert('', tk.END, iid=str(mid), tags=(tag,),
                              values=(sym, met['name'],
                                      f"{ce:.4g}" if ce else '—'))
        # Select first item by default to trigger detail view
        children = self._tree.get_children('')
        if children:
            self._tree.selection_set(children[0])
            self._tree.see(children[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _clear_axes(self):
        for ax in (self._ax_spec, self._ax_bar):
            if ax is not None:
                ax.clear()
                ax.set_facecolor('#fafafa')

    # ── Overview (no metabolite selected) ────────────────────────────────────

    def _draw_overview(self):
        if self._ax_spec is None:
            return
        self._clear_axes()
        ax_t = self._ax_spec
        ax_b = self._ax_bar

        if not self._contributions:
            ax_t.text(0.5, 0.5,
                      'Run J-Resolve, then select a metabolite in the list.',
                      transform=ax_t.transAxes, ha='center', va='center',
                      color='gray', fontsize=11)
            ax_b.set_visible(False)
            self._canvas.draw_idle()
            return

        ax_b.set_visible(True)

        # ── Top: coverage per peak (bar of nnls_coverage or residual) ─────────
        ovlp_ranks = sorted(
            [r for r, ns in self._peak_overlap.items() if len(ns) > 1],
            key=lambda r: (self._peak_by_rank(r) or {}).get('ppm', 0),
            reverse=True)

        if ovlp_ranks:
            residuals = [self._contributions.get(r, {}).get('_residual_pct', 0)
                         for r in ovlp_ranks]
            covered   = [100.0 - res for res in residuals]
            labels    = [f"#{r}  {(self._peak_by_rank(r) or {}).get('ppm', r):.3f}"
                         for r in ovlp_ranks]
            y = np.arange(len(ovlp_ranks))
            ax_t.barh(y, covered,   color=_COL_RESOLVED,  height=0.55, label='Explained')
            ax_t.barh(y, residuals, left=covered, color=_COL_RESIDUAL, height=0.55, label='Residual')
            ax_t.set_yticks(y)
            ax_t.set_yticklabels(labels, fontsize=7)
            ax_t.set_xlim(0, 105)
            ax_t.axvline(100, color='#ccc', linewidth=0.8, linestyle='--')
            ax_t.set_xlabel('Explained  (%  of observed area)', fontsize=8)
            ax_t.set_title('J-Resolve — Overview: signal explanation per peak', fontsize=9, pad=6)
            for spine in ('top', 'right'):
                ax_t.spines[spine].set_visible(False)
            ax_t.legend(fontsize=7, loc='lower right', framealpha=0.8,
                        handlelength=1.0)

        # ── Bottom: NNLS quality stats text ───────────────────────────────────
        ax_b.axis('off')
        lines = ['NNLS Quality']
        if self._nnls_residual_norm is not None:
            pct = self._nnls_residual_norm * 100
            lines.append(f'Normalised residual:  {pct:.1f}%'
                         + ('  ✓' if pct < 15 else '  ⚠' if pct < 35 else '  ✗'))
        if self._nnls_cond is not None:
            lines.append(f'Condition number:  {self._nnls_cond:.2g}'
                         + ('  ✓' if self._nnls_cond < 100 else
                            '  ⚠' if self._nnls_cond < 1000 else '  ✗ (near-singular)'))
        if self._nnls_coverage:
            mean_cov = np.mean(list(self._nnls_coverage.values()))
            lines.append(f'Mean peak coverage:  {mean_cov:.2f}'
                         + ('  ✓' if 0.7 <= mean_cov <= 1.3 else '  ⚠'))
        n_res = sum(1 for s in self._status_map.values() if s == 'resolved_nnls')
        if n_res:
            lines.append(f'NNLS-resolved metabolites:  {n_res}')
        ax_b.text(0.02, 0.95, '\n'.join(lines), transform=ax_b.transAxes,
                  va='top', ha='left', fontsize=9, linespacing=1.7,
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff',
                            edgecolor='#c5cae9', linewidth=0.8))

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

    # ── Per-metabolite detail view ────────────────────────────────────────────

    def _draw_met_detail(self, met):
        """Two-panel view for the selected metabolite.

        Top  — mean NMR spectrum with the metabolite's assigned peaks highlighted
               and NNLS/cascade contribution shaded.
        Bottom — per-peak horizontal contribution bars + quality metrics.
        """
        if self._ax_spec is None:
            return
        self._clear_axes()
        self._ax_b = self._ax_bar
        self._ax_b.set_visible(True)
        ax_spec = self._ax_spec
        ax_bar  = self._ax_bar

        mid    = met['id']
        name   = met['name']
        status = self._status_map.get(mid, 'unresolvable')
        ce     = self._conc_est.get(mid)

        # Peaks where this metabolite is involved (overlap peaks only)
        my_ranks = sorted(
            [r for r, ns in self._peak_overlap.items()
             if name in ns],
            key=lambda r: (self._peak_by_rank(r) or {}).get('ppm', 0),
            reverse=True)

        # ── Status colour ─────────────────────────────────────────────────────
        status_col = {
            'quantifiable':  _COL_ANCHOR,
            'auto_anchored': _COL_AUTO,
            'resolved':      _COL_RESOLVED,
            'resolved_nnls': _COL_RESOLVED_NLS,
            'nnls_rejected': '#FF6F00',   # amber — estimated but rejected
            'unresolvable':  _COL_RESIDUAL,
        }.get(status, _COL_UNKNOWN)

        # ══════════════════════════════════════════════════════════════════════
        # TOP — spectrum with peak regions highlighted
        # ══════════════════════════════════════════════════════════════════════
        if self._ppm is not None and self._mean_int is not None:
            ppm_arr = np.asarray(self._ppm)
            int_arr = np.asarray(self._mean_int)

            # Determine zoom window: 0.4 ppm around each assigned peak,
            # expanded to cover all peaks of this metabolite.
            if my_ranks:
                ppms_of_peaks = [(self._peak_by_rank(r) or {}).get('ppm', 0)
                                 for r in my_ranks]
                margin = 0.5
                lo = min(ppms_of_peaks) - margin
                hi = max(ppms_of_peaks) + margin
                mask = (ppm_arr >= lo) & (ppm_arr <= hi)
            else:
                mask = np.ones(len(ppm_arr), dtype=bool)

            ppm_w = ppm_arr[mask]
            int_w = int_arr[mask]

            ax_spec.plot(ppm_w, int_w, color='#aaaaaa', linewidth=0.8, zorder=1)
            ax_spec.invert_xaxis()
            ax_spec.set_xlabel('δ (ppm)', fontsize=8)
            ax_spec.set_ylabel('Intensity', fontsize=8)
            ax_spec.tick_params(labelsize=7)
            for spine in ('top', 'right'):
                ax_spec.spines[spine].set_visible(False)

            # Shade each peak region and draw contribution
            for rank in my_ranks:
                peak = self._peak_by_rank(rank)
                if peak is None:
                    continue
                p_lo = float(peak.get('start_ppm', peak['ppm'] - 0.03))
                p_hi = float(peak.get('end_ppm',   peak['ppm'] + 0.03))
                pmask = (ppm_arr >= p_lo) & (ppm_arr <= p_hi)
                ppm_seg = ppm_arr[pmask]
                int_seg = int_arr[pmask]

                pct  = self._contributions.get(rank, {}).get(mid, 0)
                cov  = self._nnls_coverage.get(rank)

                # Background shade for the peak region
                ax_spec.fill_between(ppm_seg, 0, int_seg,
                                     alpha=0.18, color=status_col, zorder=2)
                # Foreground — fraction attributed to this metabolite
                if pct > 0 and int_seg.size > 0:
                    ax_spec.fill_between(ppm_seg, 0, int_seg * (pct / 100.0),
                                         alpha=0.50, color=status_col, zorder=3)

                # Annotation above peak
                peak_ppm = float(peak.get('ppm', 0))
                peak_top = float(np.max(int_seg)) if int_seg.size else 0
                is_anchor = rank in self._anchor_peaks.get(mid, [])
                lbl = f"#{rank}\n{peak_ppm:.3f}\n{pct:.0f}%"
                if is_anchor:
                    lbl += '\n✓'
                ax_spec.annotate(lbl,
                                 xy=(peak_ppm, peak_top),
                                 xytext=(0, 6), textcoords='offset points',
                                 ha='center', fontsize=6, color=status_col,
                                 fontweight='bold', zorder=5)

            # Title with quality badge
            status_label = {
                'quantifiable':  '✓ Quantifiable',
                'auto_anchored': '⊙ Auto-anchored',
                'resolved':      '⚠ Cascade-resolved',
                'resolved_nnls': '∑ NNLS-resolved',
                'nnls_rejected': '⊘ NNLS-rejected',
                'unresolvable':  '✗ Unresolvable',
            }.get(status, status)
            title = f'{name}  —  {status_label}'
            if ce:
                title += f'   (conc_est = {ce:.3g})'
            ax_spec.set_title(title, fontsize=9, pad=5, color=status_col)
        else:
            ax_spec.text(0.5, 0.5, 'No spectrum data available.',
                         transform=ax_spec.transAxes, ha='center', va='center',
                         color='gray', fontsize=10)

        # ══════════════════════════════════════════════════════════════════════
        # BOTTOM — per-peak contribution bars + quality metrics
        # ══════════════════════════════════════════════════════════════════════
        if not my_ranks:
            ax_bar.text(0.5, 0.5,
                        'No overlapping peaks for this metabolite.',
                        transform=ax_bar.transAxes, ha='center', va='center',
                        color='gray', fontsize=9)
        else:
            y      = np.arange(len(my_ranks))
            bar_h  = min(0.6, 5.0 / max(len(my_ranks), 1))
            labels = []

            for bi, rank in enumerate(my_ranks):
                info   = self._contributions.get(rank, {})
                pct    = info.get(mid, 0.0)
                resid  = info.get('_residual_pct', 0.0)
                cov    = self._nnls_coverage.get(rank)
                is_anch = rank in self._anchor_peaks.get(mid, [])
                is_auto = is_anch and mid in self._auto_anchored

                col = (_COL_AUTO if is_auto else
                       _COL_ANCHOR if is_anch else status_col)

                ax_bar.barh(bi, pct, height=bar_h, color=col,
                            edgecolor='white', linewidth=0.5)
                if resid > 0.5:
                    ax_bar.barh(bi, resid, left=pct, height=bar_h,
                                color=_COL_RESIDUAL, edgecolor='white',
                                linewidth=0.5, alpha=0.7)

                # Label on bar
                if pct >= 5:
                    ax_bar.text(pct / 2, bi, f'{pct:.0f}%',
                                ha='center', va='center', fontsize=7,
                                color='white', fontweight='bold')

                # Coverage annotation to the right
                right_val = pct + (resid if resid > 0.5 else 0)
                cov_txt = f'  cov={cov:.2f}' if cov is not None else ''
                ax_bar.text(right_val + 1, bi, cov_txt, va='center',
                            fontsize=6, color='#555')

                peak = self._peak_by_rank(rank)
                ppm_v = peak['ppm'] if peak else rank
                star = ' ✓' if is_anch else ''
                labels.append(f'#{rank}  {ppm_v:.3f} ppm{star}')

            ax_bar.set_yticks(y)
            ax_bar.set_yticklabels(labels, fontsize=7)
            ax_bar.set_xlim(0, 115)
            ax_bar.axvline(100, color='#ccc', linewidth=0.8, linestyle='--')
            ax_bar.set_xlabel('Contribution to peak  (%)', fontsize=8)
            ax_bar.set_title('Peak contributions & residuals', fontsize=8, pad=4)
            ax_bar.tick_params(axis='y', length=0)
            for spine in ('top', 'right'):
                ax_bar.spines[spine].set_visible(False)

            # Quality metrics text box in lower-right of bar plot
            q_lines = []

            # Rejection reason (shown prominently if applicable)
            if status == 'nnls_rejected':
                q_lines.append('⊘ NNLS solution REJECTED')

            if self._nnls_residual_norm is not None:
                pct = self._nnls_residual_norm * 100
                flag = '✓' if pct < 15 else ('⚠' if pct < 35 else '✗')
                q_lines.append(f'Global residual: {pct:.1f}%  {flag}')
            if self._nnls_cond is not None:
                flag = '✓' if self._nnls_cond < 100 else ('⚠' if self._nnls_cond < 1000 else '✗')
                q_lines.append(f'Condition #: {self._nnls_cond:.2g}  {flag}')
            my_covs = [self._nnls_coverage[r] for r in my_ranks
                       if r in self._nnls_coverage]
            if my_covs:
                mean_cov = np.mean(my_covs)
                flag = '✓' if 0.6 <= mean_cov <= 1.15 else '✗'
                q_lines.append(f'Local coverage: {mean_cov:.2f}  {flag}'
                               f'  (range {min(my_covs):.2f}–{max(my_covs):.2f})')

            # Per-peak independent conc estimates → CV
            mp_data = {mp['peak_rank']: mp for mp in met.get('matched_peaks', [])
                       if 'peak_rank' in mp}
            per_peak_c = []
            for rank in my_ranks:
                peak = self._peak_by_rank(rank)
                mp   = mp_data.get(rank)
                if peak and mp:
                    obs = self._get_area(peak)
                    eff = self._effective_nh(met, mp)
                    if obs > 0 and eff > 0:
                        per_peak_c.append(obs / eff)
            if len(per_peak_c) >= 2:
                arr = np.array(per_peak_c)
                cv  = arr.std() / arr.mean() if arr.mean() > 0 else 999
                flag = '✓' if cv < 0.30 else ('⚠' if cv < 0.50 else '✗')
                q_lines.append(f'Multi-peak CV: {cv:.2f}  {flag}')

            # ΔJ values from matched peaks
            dj_lines = []
            for rank in my_ranks:
                mp = mp_data.get(rank)
                if mp:
                    dj   = mp.get('delta_j_hz')
                    mult = mp.get('multiplicity', mp.get('type', ''))
                    if dj is not None:
                        flag = '✓' if dj < 0.5 else ('⚠' if dj < 1.5 else '✗')
                        dj_lines.append(f'  #{rank} {mult}: ΔJ={dj:.2f} Hz {flag}')
                    elif mult and mult.lower() not in ('s', 'br s', 'brs', 'bs', ''):
                        dj_lines.append(f'  #{rank} {mult}: ΔJ=n/a')
            if dj_lines:
                q_lines.append('J-coupling:')
                q_lines.extend(dj_lines)

            if q_lines:
                ax_bar.text(0.98, 0.02, '\n'.join(q_lines),
                            transform=ax_bar.transAxes,
                            va='bottom', ha='right', fontsize=6.5,
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round,pad=0.4',
                                      facecolor='#f0f4ff',
                                      edgecolor='#c5cae9', linewidth=0.7))

        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_line(self, text=''):
        self._log.append(text)

    def _write_log(self):
        """Persist the current run log to disk and update the path label."""
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(_LOG_DIR, f'j_resolve_{ts}.log')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self._log))
            self._log_path = path
            self._log_path_lbl.config(text=os.path.basename(path))
        except Exception as e:
            self._log_path_lbl.config(text=f'Log error: {e}')

    def _open_log(self):
        """Open the last log file in the system text editor, or show in a dialog."""
        if not self._log:
            messagebox.showinfo('J-Resolve Log', 'No log yet — run the algorithm first.')
            return
        # Show in a scrollable text window
        win = tk.Toplevel()
        win.title('J-Resolve Log')
        win.geometry('820x600')
        txt = tk.Text(win, font=('Courier', 9), wrap='none')
        vsb = ttk.Scrollbar(win, orient=tk.VERTICAL,   command=txt.yview)
        hsb = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=txt.xview)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert('1.0', '\n'.join(self._log))
        txt.config(state='disabled')
        # Save button
        def _save():
            fn = filedialog.asksaveasfilename(
                defaultextension='.log',
                filetypes=[('Log', '*.log'), ('Text', '*.txt'), ('All', '*.*')],
                initialfile='j_resolve.log',
                parent=win)
            if fn:
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self._log))
        ttk.Button(win, text='💾 Save…', command=_save).pack(
            side=tk.BOTTOM, pady=4)

    # ─────────────────────────────────────────────────────────────────────────
    # Apply / Export
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_to_quant(self):
        if not self._contributions:
            messagebox.showinfo('J-Resolve', 'Run the algorithm first (▶ Run).')
            return
        ann = self.annotation_stage
        stored = {}
        for rank, info in self._contributions.items():
            frac = {}
            for k, v in info.items():
                if k.startswith('_'):
                    continue
                met = self._met_by_id(k)
                if met is None:
                    # k might already be an id — keep as is
                    frac[k] = float(v) / 100.0
                else:
                    frac[met['id']] = float(v) / 100.0
            stored[rank] = frac

        ann._j_resolve_contributions = stored
        ann._j_resolve_status        = dict(self._status_map)
        ann._j_resolve_conc_est      = dict(self._conc_est)

        n = len(stored)
        self._apply_lbl.config(text=f'✓ Applied ({n} peaks)')

        # Refresh label in quantification stage if it's built
        q_stage = getattr(self.app, 'stages', {}).get('quantification')
        if q_stage and hasattr(q_stage, '_refresh_j_resolve_label'):
            q_stage._refresh_j_resolve_label()

        messagebox.showinfo(
            'J-Resolve applied',
            f'{n} overlapping peak(s) registered.\n\n'
            'Enable "J-Resolve correction" in Quantification (step 5)\n'
            'to use these fractional areas.')

    def _export(self):
        import csv
        fn = filedialog.asksaveasfilename(
            title='Export J-Resolve Results',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')],
            initialfile='j_resolve_results.csv')
        if not fn:
            return
        try:
            with open(fn, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['Metabolite', 'ID', 'Status',
                            'Anchor peaks', 'Rel. conc estimate',
                            'Peak rank', 'Peak PPM',
                            'Contribution %', 'Residual %'])
                for met in self._metabolites:
                    mid    = met['id']
                    status = self._status_map.get(mid, '?')
                    ce     = self._conc_est.get(mid, '')
                    for rank, info in self._contributions.items():
                        if met['name'] not in self._peak_overlap.get(rank, []):
                            continue
                        peak  = self._peak_by_rank(rank)
                        ppm   = peak['ppm'] if peak else ''
                        pct   = info.get(mid, '')
                        resid = info.get('_residual_pct', '')
                        w.writerow([
                            met['name'], mid, status,
                            '; '.join(str(r) for r in
                                      self._anchor_peaks.get(mid, [])),
                            f"{ce:.6g}" if ce else '',
                            rank,
                            f"{ppm:.5f}" if ppm else '',
                            f"{pct:.1f}" if pct != '' else '',
                            f"{resid:.1f}" if resid != '' else '',
                        ])
            messagebox.showinfo('Exported', f"Saved to:\n{fn}")
        except Exception as e:
            messagebox.showerror('Error', str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # NNLS Deconvolution Dialog
    # ─────────────────────────────────────────────────────────────────────────

    def _open_deconvolution_dialog(self):
        """Open the interactive NNLS deconvolution dialog for the selected met."""
        if not self._selected_mid:
            return
        met = next((m for m in self._metabolites
                    if str(m['id']) == self._selected_mid), None)
        if met is None:
            return
        # Collect all metabolites that share any overlapping peak with this one
        name = met['name']
        shared_ranks = [r for r, ns in self._peak_overlap.items() if name in ns]
        if not shared_ranks:
            messagebox.showinfo('Deconvolve', 'No overlapping peaks for this metabolite.')
            return
        related_mets = []
        seen_ids = set()
        for rank in shared_ranks:
            for n in self._peak_overlap.get(rank, []):
                m2 = self._met_by_name(n)
                if m2 and m2['id'] not in seen_ids:
                    seen_ids.add(m2['id'])
                    related_mets.append(m2)

        NNLSDeconvolutionDialog(
            parent       = self.app.root,
            stage        = self,
            focus_met    = met,
            candidates   = related_mets,
            shared_ranks = shared_ranks,
        )


# ─────────────────────────────────────────────────────────────────────────────
# NNLS Deconvolution Dialog — standalone Toplevel
# ─────────────────────────────────────────────────────────────────────────────

class NNLSDeconvolutionDialog:
    """Interactive NNLS deconvolution for a spectral region.

    Shows:
      • Left panel  — candidate list with Accept/Reject checkboxes + metrics
      • Right panel — 3-subplot figure:
          top    = observed spectrum + candidate predicted lines (before)
          middle = NNLS fit: each accepted candidate contribution as filled area
          bottom = residual (observed − fitted)

    The user can toggle candidates on/off, re-run NNLS, and Accept to write
    results back into the J-Resolve stage.
    """

    # Lorentzian linewidth default (Hz) — used for synthetic peak shapes
    _LW_HZ = 1.5

    def __init__(self, parent, stage, focus_met, candidates, shared_ranks):
        self._stage   = stage
        self._focus   = focus_met
        self._cands   = candidates          # list of met dicts
        self._ranks   = sorted(shared_ranks,
                               key=lambda r: (stage._peak_by_rank(r) or {}).get('ppm', 0),
                               reverse=True)

        # Per-candidate accept/reject state (default: all on)
        self._accept_vars = {str(m['id']): tk.BooleanVar(value=True)
                             for m in candidates}

        # Solution (updated by _run_nnls)
        self._solution    = {}    # mid → conc_est
        self._metrics     = {}    # mid → dict with cv, coverage, etc.
        self._j_validation = {}  # mid → dict with delta_j_fit, j_valid, lines_found
        self._residual    = None  # scalar normalised residual
        self._zoomed_met      = None  # metabolite currently zoomed in the plot
        self._selected_peak    = None  # (met, mp) for per-peak zoom, or None
        self._selected_line_ppm = None  # float ppm when zoomed to a single multiplet line
        self._accept_peak_vars = {}   # pk_iid → BooleanVar (per-peak accept)
        self._row_data         = {}   # iid → {'type':'met'|'peak', 'met':…, 'mp':…}

        self._win = tk.Toplevel(parent)
        self._win.title(f'NNLS Deconvolution — {focus_met["name"]}')
        self._win.geometry('1100x680')
        self._win.minsize(800, 500)

        self._build_ui()
        self._run_nnls()
        self._draw()

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Left panel ────────────────────────────────────────────────────────
        lf = tk.Frame(self._win, width=310, bg='#f0f0f0')
        lf.pack(side=tk.LEFT, fill=tk.Y, padx=0)
        lf.pack_propagate(False)

        ttk.Label(lf, text='Candidates', font=('Arial', 11, 'bold')
                  ).pack(anchor=tk.W, padx=10, pady=(10, 0))
        ttk.Label(lf,
                  text='Click row → zoom · click ☑ column → toggle active',
                  font=('Arial', 7), foreground='gray'
                  ).pack(anchor=tk.W, padx=10, pady=(0, 3))

        # ── Compact legend ────────────────────────────────────────────────────
        leg_f = tk.Frame(lf, bg='#e8eaf6', relief='flat', bd=1)
        leg_f.pack(fill=tk.X, padx=8, pady=(0, 5))
        tk.Label(leg_f, bg='#e8eaf6', font=('Arial', 7),
                 text=(
                     'Metabolite row: color = J-Resolve status  |  ☑ = include in model\n'
                     'Multiplet lines: d/t/q L1 L2… = individual lines of that multiplet\n'
                     'Source: pk#N = detected by peak picker  |  NNLS = deconvolved only\n'
                     'J≈X.XHz = predicted spacing  |  J=X→YHz = predicted→measured after NNLS\n'
                     'J: ✓ ΔJ<0.5Hz  ✗ fail  · singlet  |  Cov ideal 0.8–1.2'
                 ),
                 justify=tk.LEFT, anchor='w', wraplength=285
                 ).pack(padx=5, pady=3, fill=tk.X)

        # ── Treeview table ────────────────────────────────────────────────────
        tv_frame = tk.Frame(lf, bg='#f0f0f0')
        tv_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        cols = ('src', 'cb', 'j', 'cov', 'dj')
        self._tree_cands = ttk.Treeview(tv_frame, columns=cols,
                                        show='tree headings',
                                        selectmode='browse')
        _vsb_tree = ttk.Scrollbar(tv_frame, orient=tk.VERTICAL,
                                   command=self._tree_cands.yview)
        self._tree_cands.configure(yscrollcommand=_vsb_tree.set)

        self._tree_cands.heading('#0',   text='Metabolite / Peak', anchor='w')
        self._tree_cands.heading('src',  text='Source',   anchor='center')
        self._tree_cands.heading('cb',   text='☑',        anchor='center')
        self._tree_cands.heading('j',    text='J',         anchor='center')
        self._tree_cands.heading('cov',  text='Cov',       anchor='center')
        self._tree_cands.heading('dj',   text='ΔJ / SNR',  anchor='center')

        self._tree_cands.column('#0',   width=115, stretch=True,  minwidth=80, anchor='w')
        self._tree_cands.column('src',  width=55,  stretch=False, anchor='center')
        self._tree_cands.column('cb',   width=28,  stretch=False, anchor='center')
        self._tree_cands.column('j',    width=28,  stretch=False, anchor='center')
        self._tree_cands.column('cov',  width=45,  stretch=False, anchor='center')
        self._tree_cands.column('dj',   width=65,  stretch=False, anchor='center')

        # Metabolite-row background tags (status colour)
        self._tree_cands.tag_configure('quantifiable',  background='#c8e6c9', foreground='#1b5e20')
        self._tree_cands.tag_configure('auto_anchored', background='#e3f2fd', foreground='#0d47a1')
        self._tree_cands.tag_configure('resolved',      background='#fff9c4', foreground='#6d4c00')
        self._tree_cands.tag_configure('resolved_nnls', background='#e8f5e9', foreground='#1b5e20')
        self._tree_cands.tag_configure('nnls_rejected', background='#fff3e0', foreground='#bf360c')
        self._tree_cands.tag_configure('unresolvable',  background='#fce4ec', foreground='#880e4f')
        self._tree_cands.tag_configure('focus',         font=('Arial', 9, 'bold'))
        self._tree_cands.tag_configure('inactive',      foreground='#aaaaaa')
        self._tree_cands.tag_configure('bad_annot',     background='#eeeeee', foreground='#999999')
        # Peak child-row tags
        self._tree_cands.tag_configure('peak_existing', background='#f5f5f5', foreground='#333')
        self._tree_cands.tag_configure('peak_nnls',     background='#e8f5e9', foreground='#1b5e20')
        self._tree_cands.tag_configure('peak_rejected', background='#fff3e0', foreground='#aaa')

        _vsb_tree.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree_cands.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tree_cands.bind('<ButtonRelease-1>', self._on_tree_click)
        self._tree_cands.bind('<<TreeviewSelect>>', self._on_tree_select)

        self._populate_cand_table()

        # Metrics area (updated after each NNLS run)
        ttk.Separator(lf, orient='horizontal').pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(lf, text='Quality metrics', font=('Arial', 9, 'bold')
                  ).pack(anchor=tk.W, padx=10)
        self._metrics_lbl = ttk.Label(lf, text='—', font=('Courier', 8),
                                      foreground='#333', wraplength=290,
                                      justify=tk.LEFT)
        self._metrics_lbl.pack(anchor=tk.W, padx=10, pady=(2, 6))

        # Buttons
        ttk.Separator(lf, orient='horizontal').pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(lf, text='▶ Re-run NNLS',
                   command=self._on_rerun).pack(fill=tk.X, padx=10, pady=2)
        ttk.Button(lf, text='✓ Accept & apply',
                   command=self._on_accept).pack(fill=tk.X, padx=10, pady=2)
        ttk.Button(lf, text='✗ Close',
                   command=self._win.destroy).pack(fill=tk.X, padx=10, pady=(2, 10))

        # ── Right panel — matplotlib ───────────────────────────────────────────
        rf = tk.Frame(self._win)
        rf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        from matplotlib.gridspec import GridSpec
        self._fig = plt.Figure(facecolor='#fafafa')
        gs = GridSpec(3, 1, figure=self._fig,
                      height_ratios=[2, 2, 1], hspace=0.55)
        self._ax_obs  = self._fig.add_subplot(gs[0])   # observed + candidates
        self._ax_fit  = self._fig.add_subplot(gs[1])   # NNLS fit
        self._ax_res  = self._fig.add_subplot(gs[2])   # residual
        self._canvas  = FigureCanvasTkAgg(self._fig, master=rf)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _populate_cand_table(self):
        """Populate / refresh the Treeview with metabolite parents and peak children."""
        stage    = self._stage
        ppm_lo, ppm_hi = self._spectral_window()
        _STATUS_TAG = {
            'quantifiable':  'quantifiable',
            'auto_anchored': 'auto_anchored',
            'resolved':      'resolved',
            'resolved_nnls': 'resolved_nnls',
            'nnls_rejected': 'nnls_rejected',
            'unresolvable':  'unresolvable',
        }
        prev_sel = self._tree_cands.selection()
        self._tree_cands.delete(*self._tree_cands.get_children())
        self._row_data = {}

        for met in self._cands:
            mid    = str(met['id'])
            status = stage._status_map.get(met['id'], 'unresolvable')
            s_tag  = _STATUS_TAG.get(status, 'unresolvable')
            active = self._accept_vars[mid].get()
            cb_sym = '☑' if active else '☐'

            ce  = self._solution.get(mid, 0)
            jv  = self._j_validation.get(mid, {})
            m_  = self._metrics.get(mid, {})

            j_valid = jv.get('j_valid')
            j_sym   = ('✓' if j_valid is True else
                       '✗' if j_valid is False else '·') if ce > 0 else '—'
            covs    = m_.get('coverages', [])
            cov_txt = f'{float(np.mean(covs)):.2f}' if covs and ce > 0 else '—'

            name_d  = met['name'] if len(met['name']) <= 20 else met['name'][:19] + '…'

            # Gray if ALL peaks at shared ranks are annotated 'bad'
            _shared_rank_set = set(self._ranks)
            _mps_at_ranks = [mp for mp in met.get('matched_peaks', [])
                             if mp.get('peak_rank') in _shared_rank_set]
            _all_bad = (bool(_mps_at_ranks)
                        and all(mp.get('annotation_quality') == 'bad'
                                for mp in _mps_at_ranks))

            par_tags = [s_tag]
            if _all_bad:
                par_tags.append('bad_annot')
            if met['id'] == self._focus['id']:
                par_tags.append('focus')
            if not active:
                par_tags.append('inactive')

            # ── Parent row (metabolite) ──────────────────────────────────────
            self._tree_cands.insert('', tk.END, iid=mid,
                                    text=name_d,
                                    values=('', cb_sym, j_sym, cov_txt, ''),
                                    open=True,
                                    tags=tuple(par_tags))
            self._row_data[mid] = {'type': 'met', 'met': met, 'mp': None}

            # ── Child rows ──────────────────────────────────────────────────
            # Singlets → one row per matched_peak (unchanged).
            # Multiplets (≥2 predicted_lines) → one row per individual line,
            # showing pk#N if the line was detected or NNLS if deconvolved.
            mps = self._met_peaks_in_window(met, ppm_lo, ppm_hi)
            # Build lookup: rank → per-peak result from j_validation
            pk_jval = {pk.get('rank'): pk for pk in jv.get('peaks', [])}
            per_peak_cov = m_.get('per_peak', {})

            seen_pk_iids = set()
            for i, mp in enumerate(mps):
                rank   = mp.get('peak_rank')
                pk_key = str(rank) if rank is not None else f'p{i}'
                pk_iid = f'{mid}__pk__{pk_key}'
                if pk_iid in seen_pk_iids:
                    continue   # _met_peaks_in_window returned same mp twice
                seen_pk_iids.add(pk_iid)

                # Per-peak accept var (default True, shared by all lines of mp)
                if pk_iid not in self._accept_peak_vars:
                    self._accept_peak_vars[pk_iid] = tk.BooleanVar(value=True)
                pk_accepted = self._accept_peak_vars[pk_iid].get()
                pk_cb       = '☑' if pk_accepted else '☐'

                # Per-peak J/SNR from j_validation
                pk_res = pk_jval.get(rank, {})
                pk_jv  = pk_res.get('j_valid')
                pk_j   = ('✓' if pk_jv is True else
                          '✗' if pk_jv is False else '·') if ce > 0 else '—'
                dj  = pk_res.get('delta_j')
                snr = pk_res.get('snr')
                if dj is not None and ce > 0:
                    pk_dj = f'{dj:.2f}Hz'
                elif snr is not None and ce > 0:
                    pk_dj = f'SNR {snr:.1f}'
                else:
                    pk_dj = '—'

                # Per-peak coverage
                cov_v  = per_peak_cov.get(pk_key)
                pk_cov = f'{cov_v:.2f}' if cov_v is not None and ce > 0 else '—'

                mult  = (mp.get('multiplicity') or mp.get('type') or 's').lower()
                ppm_c = float(mp.get('ppm_center') or mp.get('db_ppm') or
                              mp.get('peak_ppm') or mp.get('ppm', 0))

                pred  = mp.get('predicted_lines') or []
                _singlet_mult = mult in ('s', 'singlet', '', 'br s', 'brs', 'm', 'multiplet')

                if len(pred) >= 2 and not _singlet_mult:
                    # ── Per-line rows for multiplets ─────────────────────────
                    # Predicted J spacing from DB line positions (always available)
                    mhz_val  = float(getattr(stage.app, 'spectrometer_mhz', None) or 600.0)
                    pred_ppms = sorted([float(l.get('ppm', ppm_c)) for l in pred], reverse=True)
                    j_spacings = [abs(pred_ppms[k] - pred_ppms[k+1]) * mhz_val
                                  for k in range(len(pred_ppms)-1)]
                    j_pred_hz = float(sum(j_spacings) / len(j_spacings)) if j_spacings else None
                    # Measured J after NNLS (from j_validation j_actual)
                    j_meas_hz = pk_res.get('j_actual') if ce > 0 else None
                    # Build ΔJ string for L1: "J=X.X→Y.YHz" or "J≈X.XHz"
                    if j_meas_hz is not None and j_pred_hz is not None:
                        l1_dj = f'J={j_pred_hz:.1f}→{j_meas_hz:.1f}Hz'
                    elif j_meas_hz is not None:
                        l1_dj = f'J={j_meas_hz:.1f}Hz'
                    elif j_pred_hz is not None:
                        l1_dj = f'J≈{j_pred_hz:.1f}Hz'
                    else:
                        l1_dj = pk_dj  # fallback

                    for li, line in enumerate(pred):
                        line_ppm  = float(line.get('ppm', ppm_c))
                        det_peak  = self._find_peak_for_line(line_ppm)
                        line_rank = det_peak.get('rank') if det_peak else None
                        src_l     = f'pk#{line_rank}' if line_rank is not None else 'NNLS'
                        line_iid  = f'{pk_iid}__ln__{li}'

                        # ΔJ, J badge, Cov — only on first line (multiplet props)
                        l_j   = pk_j   if li == 0 else ''
                        l_cov = pk_cov if li == 0 else ''
                        l_dj  = l1_dj  if li == 0 else ''
                        l_cb  = pk_cb  if li == 0 else ''  # ☑ only on first line

                        if not active:
                            l_tag = 'inactive'
                        elif not pk_accepted:
                            l_tag = 'peak_rejected'
                        elif line_rank is None and ce > 0:
                            l_tag = 'peak_nnls'      # deconvolved (new) line
                        else:
                            l_tag = 'peak_existing'  # independently detected

                        self._tree_cands.insert(mid, tk.END, iid=line_iid,
                                                text=f'{mult} L{li+1}  {line_ppm:.4f}',
                                                values=(src_l, l_cb, l_j, l_cov, l_dj),
                                                tags=(l_tag,))
                        self._row_data[line_iid] = {
                            'type': 'line', 'met': met, 'mp': mp,
                            'pk_key': pk_key, 'pk_iid': pk_iid,
                            'line_idx': li, 'line_ppm': line_ppm,
                            'line_peak': det_peak,
                        }
                else:
                    # ── Single row for singlets / unresolved multiplets ───────
                    in_pl  = rank is not None and stage._peak_by_rank(rank) is not None
                    src_txt = f'pk#{rank}' if in_pl else 'NNLS'
                    ch_tag  = 'peak_nnls' if (not in_pl and ce > 0) else 'peak_existing'
                    if not pk_accepted:
                        ch_tag = 'peak_rejected'
                    if not active:
                        ch_tag = 'inactive'

                    self._tree_cands.insert(mid, tk.END, iid=pk_iid,
                                            text=f'{mult}  {ppm_c:.3f}',
                                            values=(src_txt, pk_cb, pk_j, pk_cov, pk_dj),
                                            tags=(ch_tag,))
                    self._row_data[pk_iid] = {'type': 'peak', 'met': met,
                                              'mp': mp, 'pk_key': pk_key}

        if prev_sel:
            try:
                self._tree_cands.selection_set(prev_sel)
            except Exception:
                pass

    # ── NNLS logic ────────────────────────────────────────────────────────────

    def _active_cands(self):
        """Return list of candidate mets that are currently checked."""
        return [m for m in self._cands
                if self._accept_vars[str(m['id'])].get()]

    def _spectral_window(self):
        """Return (ppm_lo, ppm_hi) covering all shared ranks + margin."""
        stage = self._stage
        ppms  = [(stage._peak_by_rank(r) or {}).get('ppm', 0) for r in self._ranks]
        if not ppms:
            return 0.0, 0.0
        margin = 0.4
        return min(ppms) - margin, max(ppms) + margin

    def _met_peaks_in_window(self, met, ppm_lo=None, ppm_hi=None):
        """Return matched-peak dicts for `met` that contribute to the window.

        Strategy (order matters — union of two passes):
        1. Shared-rank lookup (reliable): any rank in self._ranks where this
           metabolite appears in _peak_overlap.  This is always correct.
        2. Extra peaks by ppm_center: any matched_peak whose centre falls
           within [ppm_lo, ppm_hi] but was NOT already picked up by step 1.
        """
        if ppm_lo is None:
            ppm_lo, ppm_hi = self._spectral_window()
        stage   = self._stage
        result  = []
        seen    = set()

        # Pass 1 — shared ranks (primary, always reliable)
        for rank in self._ranks:
            if met['name'] not in stage._peak_overlap.get(rank, []):
                continue
            mp = next((m for m in met.get('matched_peaks', [])
                       if m.get('peak_rank') == rank), None)
            if mp is not None and id(mp) not in seen:
                result.append(mp)
                seen.add(id(mp))

        # Pass 2 — additional peaks in window (catches non-shared peaks)
        for mp in met.get('matched_peaks', []):
            if id(mp) in seen:
                continue
            ctr = (mp.get('ppm_center') or mp.get('db_ppm') or
                   mp.get('peak_ppm') or mp.get('ppm'))
            if ctr is None:
                continue
            ctr = float(ctr)
            if ctr == 0.0:
                continue   # 0.0 is almost certainly a missing value
            if ppm_lo <= ctr <= ppm_hi:
                result.append(mp)
                seen.add(id(mp))

        return result

    @staticmethod
    def _lorentzian(ppm_arr, center, lw_ppm):
        """Normalised Lorentzian centred at `center` with FWHM `lw_ppm`."""
        half = lw_ppm / 2.0
        return (half**2) / ((ppm_arr - center)**2 + half**2)

    def _synthetic_spectrum(self, met, mp, ppm_arr, conc, mhz=600.0):
        """Build synthetic contribution of met at peak mp on ppm_arr.

        Uses predicted_lines positions if available; otherwise a single
        Lorentzian at the predicted ppm centre.
        """
        lw_ppm = self._LW_HZ / mhz
        result = np.zeros_like(ppm_arr, dtype=float)
        pred   = mp.get('predicted_lines') or []
        n_h    = self._stage._get_protons(met, mp)

        if pred:
            ris   = [float(l.get('rel_intensity', 1.0)) for l in pred]
            total = sum(ris) if sum(ris) > 0 else 1.0
            for line, ri in zip(pred, ris):
                lc = float(line.get('ppm', mp.get('ppm_center', 0)))
                result += conc * n_h * (ri / total) * self._lorentzian(ppm_arr, lc, lw_ppm)
        else:
            ctr = float(mp.get('ppm_center') or mp.get('db_ppm') or mp.get('ppm', 0))
            result += conc * n_h * self._lorentzian(ppm_arr, ctr, lw_ppm)

        return result

    def _run_nnls(self):
        """Run local NNLS on the spectral region using active candidates."""
        from scipy.optimize import nnls as _nnls
        stage  = self._stage
        log    = stage._log_line   # shorthand
        active = self._active_cands()

        log('')
        log(f'[NNLS-Dialog] Region: {self._focus["name"]}  '
            f'({len(self._cands)} candidates, {len(active)} active)')

        if not active or stage._ppm is None or stage._mean_int is None:
            log('[NNLS-Dialog] Skipped — no active candidates or no spectrum')
            self._solution = {}
            self._residual = None
            return

        ppm_lo, ppm_hi = self._spectral_window()
        log(f'[NNLS-Dialog] Window: {ppm_hi:.4f}–{ppm_lo:.4f} ppm')

        ppm_arr = np.asarray(stage._ppm)
        int_arr = np.asarray(stage._mean_int)
        mask    = (ppm_arr >= ppm_lo) & (ppm_arr <= ppm_hi)
        if mask.sum() < 4:
            log(f'[NNLS-Dialog] Skipped — only {mask.sum()} points in window')
            self._solution = {}
            self._residual = None
            return

        ppm_w = ppm_arr[mask]
        obs_w = int_arr[mask]
        mhz   = float(getattr(stage.app, 'spectrometer_mhz', None) or 600.0)
        log(f'[NNLS-Dialog] {mask.sum()} spectral points  |  {mhz:.0f} MHz')

        # Build A matrix: one column per candidate using ALL peaks in window.
        # _met_peaks_in_window first checks shared ranks (reliable), then adds
        # any additional peaks by ppm_center (catches non-shared peaks).
        A_cols    = []
        col_mets  = []
        col_peaks = []
        for met in active:
            mps = self._met_peaks_in_window(met, ppm_lo, ppm_hi)
            log(f'[NNLS-Dialog]   {met["name"]}: {len(mps)} peak(s) in window'
                + ('' if mps else ' — EXCLUDED from A matrix'))
            if not mps:
                continue
            col = np.zeros(mask.sum(), dtype=float)
            for mp in mps:
                col += self._synthetic_spectrum(met, mp, ppm_w, 1.0, mhz)
            col_max = col.max()
            log(f'[NNLS-Dialog]     synthetic col max={col_max:.3e}')
            if col_max > 0:
                A_cols.append(col)
                col_mets.append(met)
                col_peaks.append(mps)

        if not A_cols:
            log('[NNLS-Dialog] No valid columns — NNLS skipped')
            self._solution = {}
            self._residual = None
            return

        A = np.column_stack(A_cols)
        log(f'[NNLS-Dialog] A matrix: {A.shape[0]} rows × {A.shape[1]} cols  '
            f'cond={np.linalg.cond(A):.1f}')

        # Subtract contributions of inactive-but-anchored metabolites
        b = obs_w.copy()
        for met in self._cands:
            if met in active:
                continue
            ce = stage._conc_est.get(met['id'], 0)
            if ce <= 0:
                continue
            for mp in self._met_peaks_in_window(met, ppm_lo, ppm_hi):
                b -= self._synthetic_spectrum(met, mp, ppm_w, ce, mhz)
            log(f'[NNLS-Dialog]   subtracted inactive {met["name"]} (ce={ce:.3e})')
        b = np.maximum(b, 0)

        try:
            x, resid = _nnls(A, b)
        except Exception as exc:
            log(f'[NNLS-Dialog] NNLS solver error: {exc}')
            self._solution = {}
            self._residual = None
            return

        b_norm = np.linalg.norm(b)
        self._residual = float(resid / b_norm) if b_norm > 0 else None
        log(f'[NNLS-Dialog] NNLS residual: {self._residual*100:.1f}% of window signal'
            if self._residual is not None else '[NNLS-Dialog] residual: n/a')

        # Store solution
        self._solution = {}
        for met, xi in zip(col_mets, x):
            if xi > 0:
                self._solution[str(met['id'])] = float(xi)
        log(f'[NNLS-Dialog] Solution ({sum(1 for v in x if v > 0)}/{len(x)} non-zero):')
        for met, xi in zip(col_mets, x):
            log(f'[NNLS-Dialog]   {met["name"]}: xi={xi:.4e}')

        # Per-metabolite metrics
        # Coverage = xi × |∫synth(xi=1) dppm|  /  obs_area
        # abs() because ppm_w may be in descending order → negative trapz
        self._metrics = {}
        try:
            for met, xi, mps in zip(col_mets, x, col_peaks):
                mid  = str(met['id'])
                mkey = {}
                coverages  = []
                per_peak_c = []
                per_peak   = {}   # pk_key → coverage value
                for idx, mp in enumerate(mps):
                    rank       = mp.get('peak_rank')
                    pk_key     = str(rank) if rank is not None else f'p{idx}'
                    peak       = stage._peak_by_rank(rank) if rank is not None else None
                    _raw_area  = stage._get_area(peak) if peak is not None else None
                    obs_area   = float(_raw_area) if _raw_area else 0.0
                    synth_col  = self._synthetic_spectrum(met, mp, ppm_w, 1.0, mhz)
                    synth_area = abs(float(np.trapz(synth_col, ppm_w)))
                    # Fallback: use window obs spectrum under the synth peak's footprint
                    if obs_area == 0 and synth_area > 0:
                        s_max = synth_col.max()
                        if s_max > 0:
                            foot = synth_col >= s_max * 0.05
                            if foot.sum() >= 2:
                                obs_area = abs(float(np.trapz(obs_w[foot], ppm_w[foot])))
                    log(f'[NNLS-Dialog]   {met["name"]} pk#{rank}: '
                        f'obs_area={obs_area:.3e}  synth_area={synth_area:.3e}')
                    if synth_area > 0:
                        if obs_area > 0:
                            cov_val = xi * synth_area / obs_area
                            per_peak_c.append(obs_area / synth_area)
                            coverages.append(cov_val)
                        else:
                            cov_val = 0.0   # synth predicts signal but none observed
                        per_peak[pk_key] = cov_val
                mkey['conc']     = xi
                mkey['coverages'] = coverages
                mkey['per_peak'] = per_peak
                if len(per_peak_c) >= 2:
                    arr = np.array(per_peak_c)
                    mkey['cv'] = float(arr.std() / arr.mean()) if arr.mean() > 0 else 999
                else:
                    mkey['cv'] = None
                cov_s = f'{np.mean(coverages):.3f}' if coverages else 'n/a'
                cv_s  = f'{mkey["cv"]:.3f}' if mkey['cv'] is not None else '1pk'
                log(f'[NNLS-Dialog]   {met["name"]}: cov={cov_s}  cv={cv_s}')
                self._metrics[mid] = mkey
        except Exception as exc:
            import traceback as _tb
            log(f'[NNLS-Dialog] Metrics error: {exc}')
            log(_tb.format_exc())

        # J-coupling validation from isolated spectrum
        try:
            self._validate_j_coupling(ppm_w, obs_w, col_mets, col_peaks, mhz)
        except Exception as exc:
            import traceback as _tb
            log(f'[NNLS-Dialog] J-validation error: {exc}')
            log(_tb.format_exc())

        self._update_candidate_quality()
        self._update_metrics_label()

    def _validate_j_coupling(self, ppm_w, obs_w, col_mets, col_peaks, mhz):
        """Validate J-coupling of multiplet candidates from their isolated spectrum.

        For each accepted multiplet candidate:
          1. Build isolated spectrum = obs - Σ synthetic(j, conc_j) for all j ≠ candidate
             using ALL peaks of each metabolite in the window (not just shared ranks)
          2. For each predicted J-line, find the actual intensity maximum in a
             search window of ±3×linewidth around the predicted position
          3. Compute actual J-spacing in Hz from measured peak positions
          4. ΔJ_fit = |J_actual − J_database|
          Stores results in self._j_validation[mid].
        """
        from scipy.signal import find_peaks as _find_peaks

        lw_ppm = self._LW_HZ / mhz
        search = lw_ppm * 3.0   # ±3 linewidths search window

        self._j_validation = {}

        for met, mps in zip(col_mets, col_peaks):
            mid = str(met['id'])
            ce  = self._solution.get(mid, 0)
            if ce <= 0:
                self._j_validation[mid] = {'j_valid': None, 'reason': 'not resolved'}
                continue

            # Build isolated spectrum: subtract ALL other candidates' full window contributions
            isolated = obs_w.copy().astype(float)
            for other, other_mps in zip(col_mets, col_peaks):
                if other is met:
                    continue
                ce_other = self._solution.get(str(other['id']), 0)
                if ce_other <= 0:
                    continue
                for mp2 in other_mps:
                    isolated -= self._synthetic_spectrum(other, mp2, ppm_w, ce_other, mhz)
            isolated = np.maximum(isolated, 0)

            # Check each of this metabolite's peaks in the window
            multiplet_results = []

            for mp in mps:
                rank = mp.get('peak_rank')
                mult = (mp.get('multiplicity') or mp.get('type') or 's').lower().strip()
                pred = mp.get('predicted_lines') or []

                # ── Singlet quality via isolated peak SNR ─────────────────────
                if mult in ('s', 'br s', 'brs', 'bs', '') or len(pred) < 2:
                    ctr = float(mp.get('ppm_center') or mp.get('db_ppm') or
                                mp.get('peak_ppm') or mp.get('ppm', 0))
                    win = (np.abs(ppm_w - ctr) <= search * 2)
                    if win.sum() < 2:
                        multiplet_results.append({
                            'rank': rank, 'type': 'singlet',
                            'j_valid': None, 'reason': 'no points in window'})
                    else:
                        peak_h = float(isolated[win].max())
                        # Noise = std of regions away from this peak
                        noise_mask = np.abs(ppm_w - ctr) > search * 4
                        noise = float(isolated[noise_mask].std()) if noise_mask.sum() > 5 else 1.0
                        snr   = peak_h / noise if noise > 0 else 0.0
                        # Singlet is valid if SNR > 3 in isolated spectrum
                        s_valid = snr >= 3.0
                        multiplet_results.append({
                            'rank': rank, 'type': 'singlet',
                            'j_valid': s_valid,
                            'snr': snr,
                            'reason': f'SNR={snr:.1f} in isolated spec'
                                      + (' ✓' if s_valid else ' ✗ (<3)')})
                    continue

                # Find actual maxima near each predicted line
                found_ppms = []
                for line in pred:
                    lc  = float(line.get('ppm', 0))
                    win = (np.abs(ppm_w - lc) <= search)
                    if win.sum() < 2:
                        continue
                    local_max_idx = np.argmax(isolated[win])
                    actual_ppm = float(ppm_w[win][local_max_idx])
                    found_ppms.append(actual_ppm)

                if len(found_ppms) < 2:
                    multiplet_results.append({
                        'rank': rank, 'type': mult,
                        'j_valid': False, 'delta_j': None,
                        'reason': f'only {len(found_ppms)}/{len(pred)} lines found'})
                    continue

                # Compute J from adjacent line distances
                found_ppms_sorted = sorted(found_ppms, reverse=True)
                j_spacings_hz = [abs(found_ppms_sorted[i] - found_ppms_sorted[i+1]) * mhz
                                 for i in range(len(found_ppms_sorted) - 1)]
                j_actual = float(np.median(j_spacings_hz))

                # Expected J from DB
                j_str  = str(mp.get('j_coupling') or '').strip()
                import re as _re2
                j_nums = [float(v) for v in _re2.findall(r'\d+\.?\d*', j_str)
                          if float(v) > 0]
                j_exp  = j_nums[0] if j_nums else None

                if j_exp is not None:
                    delta_j = abs(j_actual - j_exp)
                    j_valid = delta_j < 0.5   # < 0.5 Hz per literature standard
                    multiplet_results.append({
                        'rank': rank, 'type': mult,
                        'j_valid': j_valid,
                        'j_actual': j_actual, 'j_expected': j_exp,
                        'delta_j': delta_j,
                        'lines_found': len(found_ppms),
                        'lines_expected': len(pred),
                        'reason': f'ΔJ={delta_j:.2f} Hz'
                                  + (' ✓' if j_valid else ' ✗')})
                else:
                    multiplet_results.append({
                        'rank': rank, 'type': mult,
                        'j_valid': True,   # no DB J to compare → provisionally ok
                        'j_actual': j_actual, 'j_expected': None,
                        'delta_j': None,
                        'lines_found': len(found_ppms),
                        'reason': 'J measured but no DB reference'})

            # Overall J-validity for this metabolite
            if not multiplet_results:
                overall = {'j_valid': None, 'peaks': [], 'reason': 'no peaks in region'}
            else:
                failed = [r for r in multiplet_results
                          if r.get('j_valid') is False]
                overall = {
                    'j_valid': len(failed) == 0,
                    'peaks': multiplet_results,
                    'reason': (f'{len(failed)} peak(s) failed J-validation'
                               if failed else 'all peaks J-validated')}
            self._j_validation[mid] = overall
            self._stage._log_line(
                f'[NNLS-Dialog] J-val {met["name"]}: '
                f'{overall["reason"]}  '
                f'({"✓" if overall["j_valid"] else "✗" if overall["j_valid"] is False else "·"})'
            )

    def _update_candidate_quality(self):
        """Refresh the candidate table after NNLS completes."""
        self._populate_cand_table()

    def _update_metrics_label(self):
        lines = []
        if self._residual is not None:
            flag = '✓' if self._residual < 0.15 else ('⚠' if self._residual < 0.35 else '✗')
            lines.append(f'Global residual: {self._residual*100:.1f}%  {flag}')
        lines.append('')
        for met in self._active_cands():
            mid  = str(met['id'])
            m    = self._metrics.get(mid, {})
            ce   = m.get('conc', 0)
            cv   = m.get('cv')
            covs = m.get('coverages', [])
            jv   = self._j_validation.get(mid, {})
            j_valid = jv.get('j_valid')

            cov_str  = f'cov={np.mean(covs):.2f}' if covs else 'cov=n/a'
            cv_str   = f'CV={cv:.2f}' if cv is not None else 'CV=1pk'
            j_str    = ('J:✓' if j_valid is True else
                        'J:✗' if j_valid is False else 'J:·')
            name_s   = met['name'][:16]
            lines.append(f'{name_s:16}  {j_str}  {cv_str}  {cov_str}')

            # Per-peak ΔJ / SNR details
            for pk in jv.get('peaks', []):
                if pk.get('type') == 'singlet':
                    snr   = pk.get('snr')
                    valid = pk.get('j_valid', False)
                    flag  = '✓' if valid else '✗'
                    snr_s = f'{snr:.1f}' if snr is not None else 'n/a'
                    lines.append(f'  pk#{pk["rank"]}  singlet  SNR={snr_s}  {flag}')
                    continue
                dj  = pk.get('delta_j')
                exp = pk.get('j_expected')
                act = pk.get('j_actual')
                lf  = pk.get('lines_found', '?')
                le  = pk.get('lines_expected', '?')
                if dj is not None:
                    lines.append(f'  pk#{pk["rank"]}  {pk["type"]}  '
                                 f'J_exp={exp:.1f}  J_fit={act:.1f}  ΔJ={dj:.2f}Hz'
                                 + (' ✓' if pk.get('j_valid') else ' ✗'))
                elif act is not None:
                    lines.append(f'  pk#{pk["rank"]}  {pk["type"]}  '
                                 f'J_fit={act:.1f} Hz  ({lf}/{le} lines)')
        self._metrics_lbl.config(text='\n'.join(lines) if lines else '—')

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        if not hasattr(self, '_ax_obs'):
            return   # axes not yet created (called before _build_ui finishes)
        stage   = self._stage
        active  = self._active_cands()
        ppm_lo, ppm_hi = self._spectral_window()

        if stage._ppm is None or stage._mean_int is None:
            return

        ppm_arr = np.asarray(stage._ppm)
        int_arr = np.asarray(stage._mean_int)
        mask    = (ppm_arr >= ppm_lo) & (ppm_arr <= ppm_hi)
        ppm_w   = ppm_arr[mask]
        obs_w   = int_arr[mask]
        mhz     = float(getattr(stage.app, 'spectrometer_mhz', None) or 600.0)

        cmap    = plt.get_cmap('tab10')
        met_col = {str(m['id']): cmap(i % 10) for i, m in enumerate(self._cands)}

        for ax in (self._ax_obs, self._ax_fit, self._ax_res):
            ax.clear()
            ax.set_facecolor('#fafafa')
            ax.invert_xaxis()
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)

        # Pre-build all synthetic spectra using ALL peaks in the window
        synthetics = {}   # mid → np.array
        fit_total  = np.zeros_like(obs_w, dtype=float)
        for met in active:
            mid = str(met['id'])
            ce  = self._solution.get(mid, 0)
            if ce <= 0:
                synthetics[mid] = np.zeros_like(ppm_w)
                continue
            s = np.zeros_like(ppm_w, dtype=float)
            for mp in self._met_peaks_in_window(met, ppm_lo, ppm_hi):
                s += self._synthetic_spectrum(met, mp, ppm_w, ce, mhz)
            synthetics[mid] = s
            fit_total += s

        # ─────────────────────────────────────────────────────────────────────
        # Panel 1 — individual deconvolved components on top of observed
        # Each accepted metabolite shown as its own colored Lorentzian overlay
        # so the user can see what NNLS extracted for each one individually.
        # ─────────────────────────────────────────────────────────────────────
        self._ax_obs.plot(ppm_w, obs_w, color='#555', linewidth=1.2,
                          label='Observed', zorder=5)

        legend_handles = [plt.Line2D([0], [0], color='#555', lw=1.2,
                                     label='Observed')]

        for met in active:
            mid = str(met['id'])
            ce  = self._solution.get(mid, 0)
            col = met_col[mid]
            s   = synthetics[mid]
            jv  = self._j_validation.get(mid, {})
            j_valid = jv.get('j_valid')

            # Dimming hierarchy: full-window < met-zoom < peak-zoom
            is_met_focus = (self._zoomed_met is None or
                            met['id'] == (self._zoomed_met or {}).get('id'))
            is_focus     = is_met_focus
            alpha_fill   = 0.45 if is_focus else 0.07
            alpha_line   = 1.0  if is_focus else 0.20
            lw_line      = 1.2  if is_focus else 0.6

            # Line style: solid=J-valid, dashed=singlet/unknown, dotted=J-failed
            ls = '-' if j_valid is True else (':' if j_valid is False else '--')

            if ce > 0:
                self._ax_obs.fill_between(ppm_w, 0, s,
                                          alpha=alpha_fill, color=col, zorder=2)
                self._ax_obs.plot(ppm_w, s, color=col, linewidth=lw_line,
                                  linestyle=ls, alpha=alpha_line, zorder=3)
                # Annotate predicted J-lines as tick marks at bottom
                scale     = obs_w.max() * 0.12 if obs_w.size else 1.0
                _sel_mp   = self._selected_peak[1] if self._selected_peak else None
                for mp in self._met_peaks_in_window(met, ppm_lo, ppm_hi):
                    is_sel_pk = (_sel_mp is not None and mp is _sel_mp)
                    pred = mp.get('predicted_lines') or []
                    ris  = ([float(l.get('rel_intensity', 1.0)) for l in pred]
                            if pred else [1.0])
                    total_ri = sum(ris) or 1.0
                    tick_alpha = (1.0 if is_sel_pk else
                                  0.8 if is_focus else 0.15)
                    tick_lw    = 2.0 if is_sel_pk else 0.8
                    for line, ri in zip(pred or [mp], ris):
                        lc = float(line.get('ppm',
                                   mp.get('ppm_center') or mp.get('db_ppm', 0)))
                        h  = scale * ri / total_ri
                        det_pk = self._find_peak_for_line(lc)
                        if det_pk is not None:
                            # Independently detected by peak picker — solid colored tick
                            self._ax_obs.plot([lc, lc], [-h * 0.05, -h],
                                              color=col, lw=tick_lw,
                                              alpha=tick_alpha,
                                              clip_on=False, zorder=4)
                        else:
                            # NNLS-only line — dashed orange tick + small label
                            nnls_alpha = min(tick_alpha * 1.2, 1.0)
                            self._ax_obs.plot([lc, lc], [-h * 0.05, -h],
                                              color='#FF6F00', lw=max(tick_lw, 1.2),
                                              linestyle='--', alpha=nnls_alpha,
                                              clip_on=False, zorder=4)
                            if is_focus:
                                self._ax_obs.text(
                                    lc, -h * 1.15, 'NNLS',
                                    ha='center', va='top', fontsize=5,
                                    color='#FF6F00', fontweight='bold',
                                    clip_on=False)

                lbl = met['name']
                if j_valid is True:
                    lbl += ' ✓J'
                elif j_valid is False:
                    lbl += ' ✗J'
                legend_handles.append(
                    plt.Line2D([0], [0], color=col, lw=2, linestyle=ls,
                               label=lbl))

        # Add tick legend entries once
        legend_handles.append(
            plt.Line2D([0], [0], color='#555', lw=1.2,
                       linestyle='none', marker='|', markersize=8,
                       label='Detected (pk#N)'))
        legend_handles.append(
            plt.Line2D([0], [0], color='#FF6F00', lw=1.2,
                       linestyle='--', marker='|', markersize=8,
                       label='NNLS-only'))

        self._ax_obs.set_ylabel('Intensity', fontsize=8)
        if self._selected_peak:
            _sp_met, _sp_mp = self._selected_peak
            _sp_mult = (_sp_mp.get('multiplicity') or 's').lower()
            _sp_ppm  = float(_sp_mp.get('ppm_center') or _sp_mp.get('db_ppm') or 0)
            _title_obs = (f'Peak zoom: {_sp_met["name"]}  {_sp_mult} {_sp_ppm:.3f} ppm'
                          f'  (— J-valid  · · J-fail  – – singlet)')
        elif self._zoomed_met:
            _title_obs = (f'Zoom: {self._zoomed_met["name"]}'
                          f'  (— J-valid  · · J-fail  – – singlet)')
        else:
            _title_obs = (f'Deconvolved components — {ppm_lo:.3f}–{ppm_hi:.3f} ppm  '
                          f'(— J-valid  · · J-fail  – – singlet)')
        self._ax_obs.set_title(_title_obs, fontsize=8, pad=4)
        self._ax_obs.tick_params(labelsize=7)
        self._ax_obs.legend(handles=legend_handles, fontsize=6,
                            loc='upper right', framealpha=0.85,
                            ncol=min(3, len(legend_handles)))

        # ─────────────────────────────────────────────────────────────────────
        # Panel 2 — total fit vs observed
        # Shows how well the sum of all NNLS components reconstructs the signal
        # ─────────────────────────────────────────────────────────────────────
        self._ax_fit.plot(ppm_w, obs_w, color='#555', linewidth=1.2,
                          label='Observed', zorder=3)
        self._ax_fit.plot(ppm_w, fit_total, color='#c62828', linewidth=1.0,
                          linestyle='--', label='NNLS fit (sum)', zorder=4)

        # Stacked fills to show each contribution within the total
        base = np.zeros_like(ppm_w, dtype=float)
        for met in active:
            mid = str(met['id'])
            s   = synthetics[mid]
            if s.max() > 0:
                is_f = (self._zoomed_met is None or
                        met['id'] == (self._zoomed_met or {}).get('id'))
                self._ax_fit.fill_between(ppm_w, base, base + s,
                                          alpha=0.45 if is_f else 0.08,
                                          color=met_col[mid], zorder=2)
                base += s

        self._ax_fit.set_ylabel('Intensity', fontsize=8)
        self._ax_fit.set_title('Total NNLS fit vs observed', fontsize=8, pad=4)
        self._ax_fit.tick_params(labelsize=7)
        self._ax_fit.legend(fontsize=6, loc='upper right', framealpha=0.85)

        # ─────────────────────────────────────────────────────────────────────
        # Panel 3 — residual + quality badges
        # ─────────────────────────────────────────────────────────────────────
        residual_w = obs_w - fit_total
        resid_pct  = (abs(residual_w).sum() /
                      (abs(obs_w).sum() + 1e-12) * 100)
        self._ax_res.plot(ppm_w, residual_w, color=_COL_RESIDUAL,
                          linewidth=0.9, zorder=2)
        self._ax_res.axhline(0, color='#ccc', linewidth=0.7,
                             linestyle='--', zorder=1)
        self._ax_res.fill_between(ppm_w, 0, residual_w, alpha=0.3,
                                  color=_COL_RESIDUAL, zorder=2)

        # J-validation summary text inside residual panel
        j_lines = []
        for met in active:
            mid = str(met['id'])
            jv  = self._j_validation.get(mid, {})
            for pk in jv.get('peaks', []):
                if pk.get('type') == 'singlet':
                    continue
                dj = pk.get('delta_j')
                if dj is not None:
                    flag = '✓' if pk.get('j_valid') else '✗'
                    j_lines.append(
                        f"{met['name'][:12]} pk#{pk['rank']}: "
                        f"ΔJ={dj:.2f}Hz {flag}")
        if j_lines:
            self._ax_res.text(0.01, 0.97, '\n'.join(j_lines),
                              transform=self._ax_res.transAxes,
                              va='top', ha='left', fontsize=6,
                              fontfamily='monospace',
                              bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='#fff8e1',
                                        edgecolor='#f9a825', lw=0.7))

        qual = '✓ good' if resid_pct < 15 else ('⚠ moderate' if resid_pct < 35 else '✗ poor')
        self._ax_res.set_ylabel('Residual', fontsize=8)
        self._ax_res.set_xlabel('δ (ppm)', fontsize=8)
        self._ax_res.set_title(
            f'Windowed residual  ({resid_pct:.1f}% of window signal)  {qual}',
            fontsize=8, pad=4)
        self._ax_res.tick_params(labelsize=7)

        # Shared x limits: line zoom > per-peak zoom > metabolite zoom > full window
        if self._selected_peak is not None:
            if self._selected_line_ppm is not None:
                # Tight zoom to a single multiplet line (selected via per-line row)
                lp      = self._selected_line_ppm
                _margin = max(0.06, self._LW_HZ / mhz * 10)
                xlo, xhi = lp - _margin, lp + _margin
            else:
                _met_sp, _mp_sp = self._selected_peak
                pred_sp = _mp_sp.get('predicted_lines') or []
                sp_ppms = [float(l['ppm']) for l in pred_sp if 'ppm' in l]
                if not sp_ppms:
                    ctr = float(_mp_sp.get('ppm_center') or _mp_sp.get('db_ppm') or 0)
                    if ctr:
                        sp_ppms = [ctr]
                if sp_ppms:
                    _margin = max(0.08, self._LW_HZ / mhz * 12)
                    xlo = min(sp_ppms) - _margin
                    xhi = max(sp_ppms) + _margin
                else:
                    xlo, xhi = ppm_lo, ppm_hi
        elif self._zoomed_met is not None:
            zm_ppms = self._met_ppms(self._zoomed_met)
            if zm_ppms:
                xlo = min(zm_ppms) - 0.15
                xhi = max(zm_ppms) + 0.15
            else:
                xlo, xhi = ppm_lo, ppm_hi
        else:
            xlo, xhi = ppm_lo, ppm_hi

        for ax in (self._ax_obs, self._ax_fit, self._ax_res):
            ax.set_xlim(xhi + 0.05, xlo - 0.05)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _met_ppms(self, met):
        """Return ppm positions for all of `met`'s peaks/lines in the window.
        Uses _met_peaks_in_window so unique (non-shared) peaks are included too."""
        ppms = []
        for mp in self._met_peaks_in_window(met):
            ctr = float(mp.get('ppm_center') or mp.get('db_ppm') or
                        mp.get('peak_ppm') or mp.get('ppm', 0))
            ppms.append(ctr)
            for ln in (mp.get('predicted_lines') or []):
                lc = ln.get('ppm')
                if lc is not None:
                    ppms.append(float(lc))
        return ppms

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_tree_click(self, event):
        """Handle click on the candidate Treeview.

        ☑ column (#2) on parent row  → toggle metabolite active in model, re-run NNLS.
        ☑ column (#2) on child row   → toggle per-peak accept, refresh table.
        Any other click on child row → zoom to that specific peak.
        Any other click on parent    → zoom to all peaks of that metabolite.
        """
        region = self._tree_cands.identify_region(event.x, event.y)
        col    = self._tree_cands.identify_column(event.x)
        iid    = self._tree_cands.identify_row(event.y)
        if not iid:
            return
        data = self._row_data.get(iid, {})

        if region == 'cell' and col == '#2':   # ☑ column (src=#1, cb=#2)
            if data.get('type') == 'met':
                mid = str(data['met']['id'])
                var = self._accept_vars.get(mid)
                if var is not None:
                    var.set(not var.get())
                    self._on_toggle()
            elif data.get('type') == 'peak':
                var = self._accept_peak_vars.get(iid)
                if var is not None:
                    var.set(not var.get())
                    self._populate_cand_table()
            elif data.get('type') == 'line':
                # Line rows share the parent matched_peak's BooleanVar
                var = self._accept_peak_vars.get(data.get('pk_iid'))
                if var is not None:
                    var.set(not var.get())
                    self._populate_cand_table()
        else:
            if data.get('type') == 'peak':
                self._zoom_to_peak(data['met'], data['mp'])
            elif data.get('type') == 'line':
                self._zoom_to_line(data['met'], data['mp'], data['line_ppm'])
            elif data.get('type') == 'met':
                self._zoom_to_met(data['met'])

    def _on_tree_select(self, event):
        """Row selected via keyboard — dispatch zoom."""
        sel = self._tree_cands.selection()
        if not sel:
            return
        data = self._row_data.get(sel[0], {})
        if data.get('type') == 'peak':
            self._zoom_to_peak(data['met'], data['mp'])
        elif data.get('type') == 'line':
            self._zoom_to_line(data['met'], data['mp'], data['line_ppm'])
        elif data.get('type') == 'met':
            self._zoom_to_met(data['met'])

    def _find_peak_for_line(self, line_ppm, tolerance_ppm=0.015):
        """Return the stage peak whose ppm is nearest to line_ppm, or None."""
        best, best_d = None, tolerance_ppm
        for pk in self._stage._peaks:
            d = abs(float(pk.get('ppm', 0)) - line_ppm)
            if d < best_d:
                best_d, best = d, pk
        return best

    def _zoom_to_line(self, met, mp, line_ppm):
        """Zoom tightly to a single spectral line of a multiplet."""
        self._zoomed_met       = met
        self._selected_peak    = (met, mp)
        self._selected_line_ppm = float(line_ppm)
        self._draw()

    def _zoom_to_met(self, met):
        """Zoom to all peaks of `met`; clear peak selection."""
        ppms = self._met_ppms(met)
        if not ppms:
            return
        self._zoomed_met        = met
        self._selected_peak     = None
        self._selected_line_ppm = None
        self._draw()

    def _zoom_to_peak(self, met, mp):
        """Zoom to the specific peak `mp` of `met` and highlight it."""
        pred  = mp.get('predicted_lines') or []
        ppms  = [float(l['ppm']) for l in pred if 'ppm' in l]
        if not ppms:
            ctr = float(mp.get('ppm_center') or mp.get('db_ppm') or
                        mp.get('peak_ppm') or mp.get('ppm', 0))
            if ctr:
                ppms = [ctr]
        if not ppms:
            return
        self._zoomed_met        = met
        self._selected_peak     = (met, mp)
        self._selected_line_ppm = None   # full predicted_lines span
        self._draw()

    def _on_toggle(self):
        """Checkbox toggled — re-run and redraw."""
        self._run_nnls()
        self._draw()

    def _on_rerun(self):
        self._zoomed_met    = None
        self._selected_peak = None
        self._run_nnls()
        self._draw()

    def _on_accept(self):
        """Write accepted NNLS deconvolution results back into the J-Resolve stage.

        For each active metabolite with xi > 0:
          - Writes _conc_est[mid] = xi (NNLS concentration estimate)
          - For each accepted peak, adds rank to _anchor_peaks[mid]
          - Marks matched_peaks as annotation_quality='good' + nnls_resolved=True
            so they are treated as reliable by the downstream quantification stage.
        """
        stage    = self._stage
        ppm_lo, ppm_hi = self._spectral_window()
        n_mets   = 0
        n_peaks  = 0

        for met in self._active_cands():
            mid = met['id']
            ce  = self._solution.get(str(mid), 0)
            if ce <= 0:
                continue

            mps = self._met_peaks_in_window(met, ppm_lo, ppm_hi)
            accepted_ranks = []
            for i, mp in enumerate(mps):
                rank   = mp.get('peak_rank')
                pk_key = str(rank) if rank is not None else f'p{i}'
                pk_iid = f'{str(mid)}__pk__{pk_key}'
                var    = self._accept_peak_vars.get(pk_iid)
                # Default accept if var was never created (table not yet refreshed)
                if var is None or var.get():
                    # Mark as reliable for quantification (NNLS deconvolution accepted)
                    mp['annotation_quality'] = 'good'
                    mp['nnls_resolved']      = True
                    mp['nnls_conc_est']      = float(ce)
                    if rank is not None:
                        accepted_ranks.append(rank)

            if not accepted_ranks:
                continue

            stage._conc_est[mid] = ce
            current = list(stage._anchor_peaks.get(mid, []))
            for r in accepted_ranks:
                if r not in current:
                    current.append(r)
            stage._anchor_peaks[mid] = current
            stage._nnls_rejected.discard(mid)
            n_mets  += 1
            n_peaks += len(accepted_ranks)

        if n_mets:
            stage._phase2(_cascade_iter='nnls')
            stage._phase4()
            stage._refresh_ui()

        self._win.destroy()
        if n_mets:
            messagebox.showinfo(
                'Deconvolution applied',
                f'{n_mets} metabolite(s), {n_peaks} peak(s) written to J-Resolve.\n'
                'Press "Apply to Quantification" to use them.')
