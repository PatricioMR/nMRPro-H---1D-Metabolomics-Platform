# gui/stages/fullspectrum_mcr_stage.py
"""
Full-Spectrum MCR Stage.

Bypasses peak detection entirely. Instead:
  1. Queries the DB for ALL metabolite signals within the spectrum ppm range.
  2. Clusters overlapping DB hits into spectral regions (greedy interval merge).
  3. Filters regions by experimental SNR (skips below noise floor).
  4. Runs MCR-ALS per region using DB-simulated profiles as S_init.
  5. Opens MCRWindow with every region that had K≥1 component.

The resulting "virtual peaks" have the same mcr_annotation structure as peaks
produced by the peak-detection → annotation pipeline, so MCRWindow and
the quantification stage work unchanged.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sqlite3
import threading
import time

from .base_stage import BaseStage


def _split_at_gaps(lo, hi, multiplet_spans, max_width):
    """
    Recursively split (lo, hi) at the largest inter-multiplet gap until all
    sub-regions are <= max_width.

    multiplet_spans : list of (span_lo, span_hi) for each DB multiplet whose
                      centre falls in (lo, hi).  Splits only happen in the
                      gaps *between* multiplets, never inside a multiplet.
    """
    if hi - lo <= max_width:
        return [(lo, hi)]
    spans = [(a, b) for a, b in multiplet_spans if a >= lo and b <= hi]
    if not spans:
        # No multiplet info — fall back to centre-based split
        mid = (lo + hi) / 2
        return [(lo, mid), (mid, hi)]
    spans = sorted(spans)
    if len(spans) == 1:
        # Only one multiplet — clip region around it
        half = max_width / 2
        c = (spans[0][0] + spans[0][1]) / 2
        return [(max(lo, c - half), min(hi, c + half))]

    # Gaps are the spaces *between* consecutive multiplet spans
    gaps = []
    for i in range(len(spans) - 1):
        gap_lo = spans[i][1]     # end of multiplet i
        gap_hi = spans[i + 1][0] # start of multiplet i+1
        if gap_hi > gap_lo:
            gaps.append((gap_hi - gap_lo, gap_lo, gap_hi))

    if not gaps:
        return [(lo, hi)]

    # Split at the midpoint of the largest inter-multiplet gap
    _, g_lo, g_hi = max(gaps)
    split = (g_lo + g_hi) / 2
    left_spans  = [(a, b) for a, b in spans if b <= split]
    right_spans = [(a, b) for a, b in spans if a >= split]
    left  = _split_at_gaps(lo, split, left_spans, max_width)
    right = _split_at_gaps(split, hi, right_spans, max_width)
    return left + right


def _spectrum_guided_split(lo, hi, multiplet_spans, ppm_arr, mean_int,
                            valley_rel=0.60):
    """
    Split region (lo, hi) at inter-multiplet boundaries where the experimental
    spectrum has a valley below valley_rel × (local peak in region).

    valley_rel: fraction of local max below which a point is considered a valley.
                0.60 = split if minimum between multiplets < 60% of region peak.
                0    = never split on spectrum.
                1    = always split between any two multiplets.
    """
    spans = sorted((a, b) for a, b in multiplet_spans
                   if a >= lo - 1e-6 and b <= hi + 1e-6)
    if len(spans) < 2:
        return [(lo, hi)]

    ascending = ppm_arr[0] < ppm_arr[-1]
    ppm_lo_v, ppm_hi_v = (lo, hi) if ascending else (hi, lo)
    mask = ((ppm_arr >= min(lo, hi)) & (ppm_arr <= max(lo, hi)))
    if mask.sum() < 4:
        return [(lo, hi)]
    reg_ppm = ppm_arr[mask]
    reg_int = mean_int[mask]
    reg_max = float(reg_int.max())
    if reg_max < 1e-12:
        return [(lo, hi)]

    # Sort spans in ascending ppm order for gap detection
    spans_asc = sorted(spans, key=lambda s: s[0])

    result = []
    current_lo = lo
    for i in range(len(spans_asc) - 1):
        gap_lo = spans_asc[i][1]
        gap_hi = spans_asc[i + 1][0]
        if gap_hi <= gap_lo:
            continue
        split = (gap_lo + gap_hi) / 2.0
        # Find spectrum values in the inter-multiplet gap
        gap_mask = (reg_ppm >= min(gap_lo, gap_hi)) & (reg_ppm <= max(gap_lo, gap_hi))
        if gap_mask.sum() == 0:
            # No points in gap — interpolate
            ppm_for_interp = reg_ppm if ascending else reg_ppm[::-1]
            int_for_interp = reg_int if ascending else reg_int[::-1]
            valley_val = float(np.interp(split, ppm_for_interp, int_for_interp))
        else:
            valley_val = float(reg_int[gap_mask].min())

        if valley_val < valley_rel * reg_max:
            result.append((current_lo, split))
            current_lo = split

    result.append((current_lo, hi))
    return result


def _merge_regions(intervals, multiplet_spans=None, gap=0.0, max_width=0.20,
                   ppm_arr=None, mean_int=None, valley_rel=0.60):
    """
    Merge overlapping/adjacent (lo, hi) intervals, then split using the
    experimental spectrum shape (spectrum-guided split) and max_width safety net.

    gap            : two signals within this distance are merged.
    max_width      : hard upper limit; wider regions are recursively split at
                     inter-multiplet gaps regardless of spectrum shape.
    multiplet_spans: (span_lo, span_hi) for each DB entry's multiplet footprint.
    ppm_arr        : experimental ppm axis (for spectrum-guided split).
    mean_int       : mean experimental intensity (for spectrum-guided split).
    valley_rel     : fraction of local max for valley detection (0–1).
    """
    if not intervals:
        return []
    ivs = sorted([(min(a, b) - gap / 2, max(a, b) + gap / 2) for a, b in intervals])
    # Greedy merge (gap=0 means only truly overlapping intervals merge)
    merged = [list(ivs[0])]
    for lo, hi in ivs[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    spans = sorted(multiplet_spans or [])

    # Step 1: spectrum-guided split
    after_spectrum = []
    for lo, hi in merged:
        region_spans = [(a, b) for a, b in spans
                        if a >= lo - 1e-6 and b <= hi + 1e-6]
        if (ppm_arr is not None and mean_int is not None
                and valley_rel > 0 and len(region_spans) >= 2):
            subs = _spectrum_guided_split(lo, hi, region_spans,
                                          ppm_arr, mean_int, valley_rel)
        else:
            subs = [(lo, hi)]
        after_spectrum.extend(subs)

    # Step 2: max_width safety net (recursive split at largest inter-multiplet gap)
    result = []
    for lo, hi in after_spectrum:
        if hi - lo <= max_width:
            result.append((lo, hi))
        else:
            region_spans = [(a, b) for a, b in spans
                            if a >= lo - 1e-6 and b <= hi + 1e-6]
            result.extend(_split_at_gaps(lo, hi, region_spans, max_width))

    return sorted(result)


def _adaptive_region_mask(ppm_arr, mean_int, lo, hi, entries, noise,
                          min_snr=3.0, field_mhz=500.0, max_expand=0.25):
    """
    Expand a DB-based region to a realistic experimental footprint.
    Uses DB span hints + local signal envelope around region center.
    """
    lo = float(min(lo, hi))
    hi = float(max(lo, hi))
    width = max(hi - lo, 1e-4)
    margin = max(0.006, width * 0.6)
    lo -= margin
    hi += margin

    n_lines_map = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'quin': 5, 'dd': 4, 'ddd': 8, 'dt': 6, 'td': 6}
    for e in (entries or []):
        try:
            cp = float(e.get('ppm_center') or e.get('ppm') or (lo + hi) * 0.5)
            ps = e.get('ppm_start')
            pe = e.get('ppm_end')
            if ps is not None and pe is not None:
                lo = min(lo, float(ps) - 0.006)
                hi = max(hi, float(pe) + 0.006)
            else:
                mult = str(e.get('type') or e.get('multiplicity') or 's').lower().strip()
                n_lines = n_lines_map.get(mult, 2)
                j_hz = 0.0
                try:
                    j_raw = str(e.get('j_coupling') or '')
                    j_hz = float(j_raw.split(',')[0].strip()) if j_raw else 0.0
                except Exception:
                    j_hz = 0.0
                split_ppm = (j_hz / max(field_mhz, 1e-9)) * max(1, n_lines - 1)
                side = max(0.006, split_ppm * 0.8)
                lo = min(lo, cp - side)
                hi = max(hi, cp + side)
        except Exception:
            continue

    center = (lo + hi) * 0.5
    search_lo = max(float(np.min(ppm_arr)), center - max_expand)
    search_hi = min(float(np.max(ppm_arr)), center + max_expand)
    search_mask = (ppm_arr >= search_lo) & (ppm_arr <= search_hi)
    if np.sum(search_mask) >= 4:
        seg_ppm = ppm_arr[search_mask]
        seg = mean_int[search_mask]
        local_max = float(np.max(seg))
        thresh = max(local_max * 0.08, noise * min_snr * 0.4)
        above = seg >= thresh
        if np.any(above):
            c_idx = int(np.argmin(np.abs(seg_ppm - center)))
            l_idx = c_idx
            while l_idx > 0 and above[l_idx - 1]:
                l_idx -= 1
            r_idx = c_idx
            while r_idx < len(above) - 1 and above[r_idx + 1]:
                r_idx += 1
            peak_w = abs(float(seg_ppm[r_idx]) - float(seg_ppm[l_idx]))
            sig_margin = max(0.008, peak_w * 0.2)
            lo = min(lo, float(seg_ppm[l_idx]) - sig_margin)
            hi = max(hi, float(seg_ppm[r_idx]) + sig_margin)

    lo = max(float(np.min(ppm_arr)), lo)
    hi = min(float(np.max(ppm_arr)), hi)
    if hi <= lo:
        return (ppm_arr >= min(search_lo, search_hi)) & (ppm_arr <= max(search_lo, search_hi)), lo, hi
    return (ppm_arr >= lo) & (ppm_arr <= hi), lo, hi


class FullSpectrumMCRStage(BaseStage):

    def __init__(self, parent, app, annotation_stage=None):
        super().__init__(parent, app)
        self.annotation_stage = annotation_stage
        self._mcr_regions     = []   # list of virtual-peak dicts with mcr_annotation
        self._db_path         = None
        self._norm_factors    = None  # per-sample normalisation factors (for back-transform)
        self.create_widgets()

    # ── Widgets ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        # Outer frame — holds the scrollable canvas
        self.frame = ttk.Frame(self.parent)

        # Scrollable inner panel
        _outer_canvas = tk.Canvas(self.frame, bd=0, highlightthickness=0)
        _vsb = ttk.Scrollbar(self.frame, orient=tk.VERTICAL,
                             command=_outer_canvas.yview)
        _outer_canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        _outer_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(_outer_canvas)
        _inner_win = _outer_canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_configure(e):
            _outer_canvas.configure(scrollregion=_outer_canvas.bbox('all'))
        def _on_canvas_resize(e):
            _outer_canvas.itemconfig(_inner_win, width=e.width)
        inner.bind('<Configure>', _on_inner_configure)
        _outer_canvas.bind('<Configure>', _on_canvas_resize)

        def _on_wheel(e, c=_outer_canvas):
            if e.num == 4:   c.yview_scroll(-1, 'units')
            elif e.num == 5: c.yview_scroll(1, 'units')
            else:            c.yview_scroll(int(-e.delta / 60), 'units')
        _outer_canvas.bind('<MouseWheel>', _on_wheel)
        _outer_canvas.bind('<Button-4>',   _on_wheel)
        _outer_canvas.bind('<Button-5>',   _on_wheel)
        inner.bind('<MouseWheel>', _on_wheel)
        inner.bind('<Button-4>',   _on_wheel)
        inner.bind('<Button-5>',   _on_wheel)

        # From here on, pack widgets into `inner` instead of `self.frame`
        # We alias `inner` as the content frame used below.
        _cf = inner   # content frame

        ttk.Label(_cf, text='MCR Analysis',
                  font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Separator(_cf, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=5)

        # ── DB selection ──────────────────────────────────────────────────────
        db_frame = ttk.LabelFrame(_cf, text='📂 Database', padding=5)
        db_frame.pack(fill=tk.X, padx=5, pady=5)

        self._db_label = ttk.Label(db_frame, text='No database selected',
                                   foreground='gray')
        self._db_label.pack(anchor=tk.W)
        ttk.Button(db_frame, text='Select DB…',
                   command=self._select_db).pack(anchor=tk.W, pady=2)
        ttk.Label(db_frame,
                  text='Opens the same DB picker used in Annotation.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # ── Parameters ────────────────────────────────────────────────────────
        pf = ttk.LabelFrame(_cf, text='⚙ Parameters', padding=5)
        pf.pack(fill=tk.X, padx=5, pady=5)

        rows = [
            ('Region gap (ppm):',       'gap_var',       tk.DoubleVar(value=0.03)),
            ('Max region width (ppm):',  'max_width_var', tk.DoubleVar(value=0.15)),
            ('Valley threshold:',        'valley_var',    tk.DoubleVar(value=0.60)),
            ('Min SNR:',                'snr_var',       tk.DoubleVar(value=2.0)),
            ('Min DB candidates:',      'min_cand_var',  tk.IntVar(value=1)),
            ('MCR R² threshold:',       'r2_var',        tk.DoubleVar(value=0.90)),
            ('Max K components:',       'max_k_var',     tk.IntVar(value=5)),
            ('PPM shift tol (ppm):',    'shift_tol_var', tk.DoubleVar(value=0.02)),
            ('Spectrometer (MHz):',     'mhz_var',       tk.StringVar(value='600')),
        ]
        for lbl_txt, attr, var in rows:
            rf = ttk.Frame(pf); rf.pack(fill=tk.X, pady=1)
            ttk.Label(rf, text=lbl_txt, width=22).pack(side=tk.LEFT)
            ttk.Entry(rf, textvariable=var, width=8).pack(side=tk.LEFT)
            setattr(self, attr, var)

        mode_row = ttk.Frame(pf); mode_row.pack(fill=tk.X, pady=1)
        ttk.Label(mode_row, text='Region mode:', width=22).pack(side=tk.LEFT)
        self.region_mode_var = tk.StringVar(value='hybrid_adaptive')
        mode_combo = ttk.Combobox(
            mode_row, textvariable=self.region_mode_var, width=20, state='readonly',
            values=['peak_bounded', 'db_driven', 'hybrid_adaptive'])
        mode_combo.pack(side=tk.LEFT)

        # Water-suppression exclusion zone — this stage bypasses peak detection
        # entirely (queries the DB directly), so it must apply its own water
        # exclusion instead of inheriting AnnotationStage._exclude_water_peaks.
        wf = ttk.Frame(pf); wf.pack(fill=tk.X, pady=(4, 1))
        self.exclude_water_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(wf, text='Exclude water region',
                        variable=self.exclude_water_var).pack(side=tk.LEFT)
        self.water_lo_var = tk.DoubleVar(value=4.70)
        self.water_hi_var = tk.DoubleVar(value=4.90)
        ttk.Entry(wf, textvariable=self.water_lo_var, width=5).pack(
            side=tk.LEFT, padx=(6, 1))
        ttk.Label(wf, text='–').pack(side=tk.LEFT)
        ttk.Entry(wf, textvariable=self.water_hi_var, width=5).pack(
            side=tk.LEFT, padx=1)
        ttk.Label(wf, text='ppm').pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(pf,
                  text='Region gap: DB signals within this distance are grouped.\n'
                       'Max width: hard upper limit for any region (safety net).\n'
                       'Valley threshold: split two DB signals if the spectrum between\n'
                       '  them drops below this fraction of the local peak (0=off, 1=always).\n'
                       'PPM shift tol: each DB profile is shifted ±tol before MCR (0=off).\n'
                       'Region mode: peak_bounded (detected peaks), db_driven (DB only),\n'
                       '  hybrid_adaptive (recommended).',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W, pady=2)

        # ── Preprocessing ─────────────────────────────────────────────────────
        ppf = ttk.LabelFrame(_cf, text='⚗ Preprocessing (optional)', padding=5)
        ppf.pack(fill=tk.X, padx=5, pady=5)

        # Blank subtraction
        bk_row = ttk.Frame(ppf); bk_row.pack(fill=tk.X, pady=2)
        self._blank_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(bk_row, text='Blank subtraction',
                        variable=self._blank_enabled,
                        command=self._toggle_blank).pack(side=tk.LEFT)
        self._blank_path   = None
        self._blank_label  = ttk.Label(bk_row, text='No blank selected',
                                       foreground='gray', font=('Arial', 8))
        self._blank_label.pack(side=tk.LEFT, padx=6)
        self._blank_btn = ttk.Button(bk_row, text='Select blank…',
                                     command=self._select_blank, state='disabled')
        self._blank_btn.pack(side=tk.LEFT)
        ttk.Label(ppf,
                  text='Subtracts a blank/buffer spectrum from all samples before MCR.\n'
                       'Use BLK samples already loaded, or select a separate file.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # Auto-detect BLK from loaded samples
        bk_auto_row = ttk.Frame(ppf); bk_auto_row.pack(fill=tk.X, pady=(0, 4))
        self._blank_use_loaded = tk.BooleanVar(value=True)
        ttk.Checkbutton(bk_auto_row,
                        text='Use BLK sample(s) from loaded data if available',
                        variable=self._blank_use_loaded).pack(side=tk.LEFT)

        ttk.Separator(ppf, orient='horizontal').pack(fill=tk.X, pady=4)

        # Normalization
        norm_row = ttk.Frame(ppf); norm_row.pack(fill=tk.X, pady=2)
        self._norm_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(norm_row, text='Normalization:',
                        variable=self._norm_enabled).pack(side=tk.LEFT)
        self._norm_method = tk.StringVar(value='PQN')
        ttk.Radiobutton(norm_row, text='PQN', variable=self._norm_method,
                        value='PQN').pack(side=tk.LEFT, padx=(6, 2))
        ttk.Radiobutton(norm_row, text='TSA', variable=self._norm_method,
                        value='TSA').pack(side=tk.LEFT, padx=2)
        ttk.Label(ppf,
                  text='PQN: Probabilistic Quotient — robust to dilution differences.\n'
                       'TSA: Total Spectral Area — simple sum normalisation.',
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # ── Run button ────────────────────────────────────────────────────────
        bf = ttk.Frame(_cf); bf.pack(fill=tk.X, padx=5, pady=8)
        self.btn_run = ttk.Button(bf, text='▶  Run MCR Analysis',
                                  command=self._run, state='disabled')
        self.btn_run.pack(fill=tk.X)

        self._status_label = ttk.Label(_cf, text='',
                                       foreground='blue', wraplength=260)
        self._status_label.pack(anchor=tk.W, padx=5)

        # ── Results summary ───────────────────────────────────────────────────
        rf2 = ttk.LabelFrame(_cf, text='Results', padding=5)
        rf2.pack(fill=tk.X, padx=5, pady=5)
        self._result_label = ttk.Label(rf2, text='No results yet.',
                                       foreground='gray')
        self._result_label.pack(anchor=tk.W)

        self.btn_open = None   # no longer shown in panel

        # Bind mousewheel on all child widgets after creation
        self._mcr_wheel_fn = _on_wheel
        _cf.after(100, self._bind_wheel_to_children)

    def _bind_wheel_to_children(self):
        """Propagate mousewheel scrolling to the canvas from all child widgets."""
        fn = getattr(self, '_mcr_wheel_fn', None)
        if fn is None:
            return
        try:
            def _bind_recursive(w):
                try:
                    w.bind('<MouseWheel>', fn)
                    w.bind('<Button-4>',   fn)
                    w.bind('<Button-5>',   fn)
                except Exception:
                    pass
                for child in w.winfo_children():
                    _bind_recursive(child)
            _bind_recursive(self.frame)
        except Exception:
            pass

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        if not self._db_path:
            # Try to inherit DB from annotation stage (if user ran it before)
            ann = self.annotation_stage
            if ann and getattr(ann, '_last_results_args', None):
                db = ann._last_results_args.get('db_path')
                if db:
                    self._set_db(db)
            # Also check app-level last DB path
            if not self._db_path:
                db = getattr(self.app, '_last_db_path', None)
                if db:
                    self._set_db(db)
        if self._db_path:
            self.btn_run.config(state='normal')
        # Show data info
        ppm_arr = getattr(self.app, 'ppm', None)
        all_int = getattr(self.app, 'all_intensities', None)
        if ppm_arr is not None and all_int is not None:
            n_s = all_int.shape[1] if all_int.ndim == 2 else 1
            self._set_status(
                f'Data: {n_s} sample(s)  |  '
                f'{ppm_arr.min():.2f}–{ppm_arr.max():.2f} ppm', 'gray')

    # ── Preprocessing helpers ─────────────────────────────────────────────────

    def _toggle_blank(self):
        state = 'normal' if self._blank_enabled.get() else 'disabled'
        self._blank_btn.config(state=state)

    def _select_blank(self):
        path = filedialog.askopenfilename(
            title='Select blank/buffer spectrum',
            filetypes=[('Text/CSV', '*.txt *.csv *.dat'), ('All', '*.*')])
        if not path:
            return
        self._blank_path = path
        import os
        self._blank_label.config(text=os.path.basename(path), foreground='black')

    def _get_blank_spectrum(self, ppm_arr):
        """Return blank intensity vector (n_pts,) on ppm_arr grid, or None."""
        samples_info = getattr(self.app, 'samples_info', []) or []
        all_int      = getattr(self.app, 'all_intensities', None)

        # Option 1: use BLK samples already loaded
        if self._blank_use_loaded.get() and all_int is not None:
            blk_idx = [i for i, s in enumerate(samples_info)
                       if s.get('type', 'Sample') == 'BLK']
            if blk_idx:
                blk = np.mean(all_int[:, blk_idx], axis=1)
                return blk

        # Option 2: file selected by user
        if self._blank_path:
            try:
                data = np.loadtxt(self._blank_path, delimiter=None, comments='#')
                if data.ndim == 2 and data.shape[1] >= 2:
                    ppm_b, int_b = data[:, 0], data[:, 1]
                    # Interpolate onto spectrum ppm grid
                    return np.interp(ppm_arr, ppm_b[::-1], int_b[::-1])
            except Exception as e:
                print(f'Blank load error: {e}')
        return None

    @staticmethod
    def _apply_blank(all_int, blank):
        """Subtract blank from every sample column; clip to zero."""
        corrected = all_int - blank[:, np.newaxis]
        return np.clip(corrected, 0, None)

    @staticmethod
    def _normalize_pqn(all_int):
        """
        Probabilistic Quotient Normalization (Dieterle et al. 2006).
        Reference = median spectrum across all samples.
        Factor per sample = median(sample / reference) at non-zero reference points.
        Returns (normalized array, factors array).
        """
        ref = np.median(all_int, axis=1)
        ref_nz = ref > 1e-12
        factors = np.ones(all_int.shape[1])
        for i in range(all_int.shape[1]):
            quotients = all_int[ref_nz, i] / ref[ref_nz]
            if len(quotients):
                factors[i] = np.median(quotients)
        factors = np.where(factors < 1e-12, 1.0, factors)
        return all_int / factors[np.newaxis, :], factors

    @staticmethod
    def _normalize_tsa(all_int):
        """
        Total Spectral Area normalization.
        Each sample divided by its total area, then rescaled to mean area.
        Returns (normalized array, factors array).
        """
        areas   = all_int.sum(axis=0)
        areas   = np.where(areas < 1e-12, 1.0, areas)
        normed  = all_int / areas[np.newaxis, :]
        mean_a  = areas.mean()
        return normed * mean_a, areas

    def on_deactivate(self):
        self.frame.pack_forget()

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _select_db(self):
        from gui.stages.annotation_stage import _pick_database
        path = _pick_database(self.app.root)
        if path:
            self._set_db(path)

    def _set_db(self, path):
        self._db_path = path
        import os
        self._db_label.config(text=os.path.basename(path), foreground='black')
        self.btn_run.config(state='normal')
        # Share with other stages
        try:
            self.app._last_db_path = path
        except Exception:
            pass

    def _set_status(self, msg, color='blue'):
        try:
            self._status_label.config(text=msg, foreground=color)
            self.app.root.update_idletasks()
        except Exception:
            pass

    # ── Main run ─────────────────────────────────────────────────────────────

    def _run(self):
        if not self._db_path:
            messagebox.showinfo('Info', 'Select a database first.')
            return
        ppm_arr = getattr(self.app, 'ppm', None)
        all_int = getattr(self.app, 'all_intensities', None)
        if ppm_arr is None or all_int is None:
            messagebox.showinfo('Info', 'No spectral data loaded.')
            return

        try:
            gap           = float(self.gap_var.get())
            max_width     = float(self.max_width_var.get())
            valley_rel    = float(self.valley_var.get())
            min_snr       = float(self.snr_var.get())
            min_cand      = int(self.min_cand_var.get())
            r2_thr        = float(self.r2_var.get())
            max_k         = int(self.max_k_var.get())
            shift_tol     = float(self.shift_tol_var.get())
            mhz           = float(self.mhz_var.get())
            region_mode   = str(self.region_mode_var.get()).strip().lower()
            exclude_water = bool(self.exclude_water_var.get())
            water_lo      = float(self.water_lo_var.get())
            water_hi      = float(self.water_hi_var.get())
            if water_lo > water_hi:
                water_lo, water_hi = water_hi, water_lo
        except Exception:
            messagebox.showerror('Error', 'Invalid parameter values.')
            return

        # ── Preprocessing ─────────────────────────────────────────────────────
        all_int = all_int.copy().astype(float)
        if all_int.ndim == 1:
            all_int = all_int[:, np.newaxis]

        preproc_log = []

        if self._blank_enabled.get():
            blank = self._get_blank_spectrum(ppm_arr)
            if blank is not None:
                all_int = self._apply_blank(all_int, blank)
                preproc_log.append('blank subtracted')
            else:
                messagebox.showwarning(
                    'Blank not found',
                    'Blank subtraction enabled but no blank spectrum found.\n'
                    'Load a BLK sample or select a blank file.')

        self._norm_factors = None
        if self._norm_enabled.get():
            method = self._norm_method.get()
            if method == 'PQN':
                all_int, self._norm_factors = self._normalize_pqn(all_int)
            else:
                all_int, self._norm_factors = self._normalize_tsa(all_int)
            preproc_log.append(f'{method} normalised')

        if preproc_log:
            self._set_status('Preprocessing: ' + ', '.join(preproc_log))

        self.btn_run.config(state='disabled')
        self._set_status('Starting…' if not preproc_log
                         else f'Preprocessing done ({", ".join(preproc_log)}). Starting MCR…')

        def worker():
            try:
                regions = self._run_worker(
                    ppm_arr, all_int, self._db_path,
                    gap, max_width, valley_rel, min_snr, min_cand,
                    r2_thr, max_k, shift_tol, mhz, region_mode,
                    exclude_water, water_lo, water_hi)
                self.app.root.after(0, lambda: self._on_done(regions))
            except Exception as exc:
                import traceback; traceback.print_exc()
                self.app.root.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, regions):
        self._mcr_regions = regions
        n_k1 = sum(1 for r in regions if r.get('mcr_annotation', {}).get('K', 0) == 1)
        n_k2 = sum(1 for r in regions if r.get('mcr_annotation', {}).get('K', 0) >= 2)

        # Push K=1 regions into peak_detection stage as normal annotated peaks
        self._push_k1_to_peak_list(regions)

        preproc = []
        if self._blank_enabled.get():
            preproc.append('blank-subtracted')
        if self._norm_enabled.get():
            preproc.append(self._norm_method.get())
        preproc_str = f'  [{", ".join(preproc)}]' if preproc else ''
        self._result_label.config(
            text=(f'{len(regions)} region(s) — {n_k1} single (→ Peak List), '
                  f'{n_k2} overlapping (→ Deconvolution).{preproc_str}'),
            foreground='black')
        self.btn_run.config(state='normal')
        self._set_status(
            f'Done. {n_k1} peak(s) in Peak List — {n_k2} overlapping region(s) to review.',
            'green')
        try:
            self.app.btn_peak_list.config(state='normal')
        except Exception:
            pass
        try:
            if n_k2 > 0:
                self.app.btn_deconv_shortcut.config(state='normal')
        except Exception:
            pass

    def _push_k1_to_peak_list(self, regions):
        """Convert K=1 MCR regions into normal peak dicts and inject into peak_stage.peaks."""
        import numpy as np
        peak_stage = (getattr(self.app, 'stages', {}) or {}).get('peak_detection')
        if peak_stage is None:
            return

        # Remove any previously pushed MCR peaks before re-pushing
        existing = [p for p in (getattr(peak_stage, 'peaks', None) or [])
                    if not p.get('_from_mcr')]

        ppm_arr = getattr(self.app, 'ppm', None)
        all_int = getattr(self.app, 'all_intensities', None)

        new_peaks = []
        for r in regions:
            mcr_ann = r.get('mcr_annotation', {})
            if mcr_ann.get('K', 0) != 1:
                continue

            cands   = mcr_ann.get('candidates', [])
            assigns = mcr_ann.get('assignments', [])
            ci      = assigns[0][0] if assigns else 0
            corr    = float(assigns[0][1]) if assigns else 0.0
            cand    = cands[ci] if ci < len(cands) else {}

            # Height: peak max from spectrum at region
            height = 0.0
            if ppm_arr is not None and all_int is not None:
                lo, hi = r.get('start_ppm', r['ppm']), r.get('end_ppm', r['ppm'])
                mask = (ppm_arr >= min(lo, hi)) & (ppm_arr <= max(lo, hi))
                if mask.any():
                    if all_int.ndim == 2:
                        height = float(np.mean(all_int[mask, :].max(axis=0)))
                    else:
                        height = float(all_int[mask].max())

            C = mcr_ann.get('C')
            area = float(np.mean(np.abs(C[:, 0]))) if C is not None and C.shape[1] > 0 else 0.0

            quality = 'good' if corr >= 0.7 else ('warn' if corr >= 0.4 else 'poor')

            pk = {
                'rank':            r['rank'],
                'ppm':             r['ppm'],
                'start_ppm':       r.get('start_ppm', r['ppm']),
                'end_ppm':         r.get('end_ppm',   r['ppm']),
                'width_ppm':       abs(r.get('end_ppm', r['ppm']) - r.get('start_ppm', r['ppm'])),
                'height':          height,
                'area':            area,
                'snr':             mcr_ann.get('metrics', {}).get('r2', 0.0) * 10,
                'overlapping':     False,
                'status':          None,
                'mcr_annotation':  mcr_ann,
                'annotation_quality': quality,
                '_from_mcr':       True,
            }
            if cand:
                pk['annotation'] = {
                    'name':    cand.get('name', ''),
                    'hmdb_id': cand.get('hmdb_id', cand.get('bmrb_id', '')),
                    'corr':    corr,
                    'source':  'MCR',
                }
            new_peaks.append(pk)

        peak_stage.peaks = existing + new_peaks

        # Re-number ranks to avoid collisions
        for i, p in enumerate(peak_stage.peaks):
            p['rank'] = i + 1

    def _on_error(self, msg):
        self.btn_run.config(state='normal')
        self._set_status(f'Error: {msg}', 'red')
        messagebox.showerror('MCR Analysis Error', msg)

    def _open_window(self):
        if not self._mcr_regions:
            messagebox.showinfo('Info', 'Run the analysis first.')
            return
        # Only show overlapping regions (K≥2) — single-peak regions are not
        # useful for manual deconvolution review
        overlap_regions = [r for r in self._mcr_regions
                           if r.get('mcr_annotation', {}).get('K', 0) >= 2]
        if not overlap_regions:
            messagebox.showinfo('Info',
                'No overlapping regions found (K≥2).\n'
                'All detected regions contain a single component.')
            return
        from gui.widgets.mcr_window import MCRWindow
        items = [{'kind': 'region', 'met': None, 'mp': None, 'peak': r}
                 for r in overlap_regions]
        win = MCRWindow(
            self.app.root,
            items,
            self.app,
            ann_stage=self)
        self.app._deconv_win = win

    # ── Worker (background thread) ────────────────────────────────────────────

    def _run_worker(self, ppm_arr, all_int, db_path,
                    gap, max_width, valley_rel, min_snr, min_cand,
                    r2_thr, max_k, shift_tol, mhz, region_mode='hybrid_adaptive',
                    exclude_water=True, water_lo=4.70, water_hi=4.90):
        """
        1. Query DB for all signals in spectrum range.
        2. Cluster into regions.
        3. Per region: SNR filter → MCR.
        Returns list of virtual-peak dicts with mcr_annotation.
        """
        ppm_lo = float(ppm_arr.min())
        ppm_hi = float(ppm_arr.max())

        # ── 1. Query DB ───────────────────────────────────────────────────────
        self._update_status('Querying database…')
        db_entries = self._query_db(db_path, ppm_lo, ppm_hi)
        if not db_entries:
            raise RuntimeError('No DB entries found in this ppm range.')

        # Water-suppression exclusion zone — dropped here (DB-entry level)
        # since this stage bypasses peak detection and never goes through
        # AnnotationStage._exclude_water_peaks.
        if exclude_water:
            _n0 = len(db_entries)
            db_entries = [e for e in db_entries
                         if not (water_lo <= float(e['ppm_center']) <= water_hi)]
            if not db_entries:
                raise RuntimeError(
                    'All DB entries fall inside the water exclusion zone '
                    f'({water_lo:.2f}-{water_hi:.2f} ppm).')
            _n_skip = _n0 - len(db_entries)
            if _n_skip:
                print(f'[fullspectrum_mcr] water exclusion {water_lo:.2f}-'
                      f'{water_hi:.2f} ppm: skipped {_n_skip} DB signal(s)')

        self._update_status(f'{len(db_entries)} DB signals loaded. Clustering…')

        # ── 2. Cluster into regions ───────────────────────────────────────────
        # Use the full multiplet span as the interval so region boundaries
        # never cut through a doublet/triplet.  For singlets the span equals
        # the centre point; for coupled multiplets it covers all stick positions.
        from gui.widgets.annotation_results_window import _build_stick_positions
        intervals       = []   # (lo, hi) expanded by shift_tol — for merging
        multiplet_spans = []   # (lo, hi) of raw multiplet footprint — for split guard
        for e in db_entries:
            dp_ppm = float(e['ppm_center'])
            mult   = e.get('type') or e.get('multiplicity') or 's'
            try:
                nh = int(e.get('protons') or 1)
            except Exception:
                nh = 1
            try:
                _ps = e.get('ppm_start'); _pe = e.get('ppm_end')
                _span = abs(float(_pe) - float(_ps)) if (_ps is not None and _pe is not None) else None
                sticks = _build_stick_positions(
                    dp_ppm, mult, nh, e.get('j_coupling'),
                    field_mhz=mhz, ppm_span=_span)
            except Exception:
                sticks = [(dp_ppm, 1.0)]
            if not sticks:
                sticks = [(dp_ppm, 1.0)]
            ppms = [s[0] for s in sticks]
            raw_lo, raw_hi = min(ppms), max(ppms)
            multiplet_spans.append((raw_lo, raw_hi))
            # Expand by shift_tol so shifted peaks still fall inside the region
            intervals.append((raw_lo - shift_tol, raw_hi + shift_tol))
        # Compute mean_int early so spectrum-guided split can use it
        mean_int = (np.mean(all_int, axis=1)
                    if all_int.ndim == 2 else all_int.copy())

        merged = _merge_regions(intervals, multiplet_spans=multiplet_spans,
                                gap=0.0, max_width=max_width,
                                ppm_arr=ppm_arr, mean_int=mean_int,
                                valley_rel=valley_rel)

        # Map each DB entry to its region
        region_entries = [[] for _ in merged]
        for e in db_entries:
            p = float(e['ppm_center'])
            for i, (lo, hi) in enumerate(merged):
                if lo <= p <= hi:
                    region_entries[i].append(e)
                    break

        # Filter by min_cand
        regions_to_process = [
            (i, lo, hi, entries)
            for i, ((lo, hi), entries) in enumerate(zip(merged, region_entries))
            if len(entries) >= min_cand
        ]

        self._update_status(
            f'{len(regions_to_process)} candidate region(s). Computing SNR…')

        # ── 3. SNR filter ─────────────────────────────────────────────────────
        # (mean_int already computed above for spectrum-guided split)
        # Noise: MAD of the top 10% ppm range (high-field, typically empty)
        _tail_mask = ppm_arr <= np.percentile(ppm_arr, 10)
        _tail = mean_int[_tail_mask]
        noise = float(np.median(np.abs(_tail - np.median(_tail)))) * 1.4826 + 1e-12

        viable = []
        for reg_idx, lo, hi, entries in regions_to_process:
            if region_mode == 'peak_bounded':
                mask, lo, hi = self._mask_from_detected_peaks(ppm_arr, lo, hi)
            elif region_mode == 'db_driven':
                db_margin = max(0.004, abs(hi - lo) * 0.30)
                lo = max(float(np.min(ppm_arr)), float(lo - db_margin))
                hi = min(float(np.max(ppm_arr)), float(hi + db_margin))
                mask = (ppm_arr >= lo) & (ppm_arr <= hi)
            else:
                # hybrid_adaptive (default)
                mask, lo, hi = _adaptive_region_mask(
                    ppm_arr, mean_int, lo, hi, entries, noise,
                    min_snr=min_snr, field_mhz=mhz, max_expand=max_width)
            if mask.sum() < 4:
                continue
            region_sig = mean_int[mask]
            region_max = float(region_sig.max())
            region_min = float(region_sig.min())
            # Peak-presence check: require a meaningful local variation above noise.
            peak_prominence = region_max - region_min
            if region_max / noise < min_snr:
                continue
            if peak_prominence < noise * 1.5:
                # Region is flat — no real peak present, skip to avoid spurious fits
                continue
            viable.append((lo, hi, entries, mask, noise))

        self._update_status(
            f'{len(viable)} region(s) above SNR={min_snr:.1f}. Running MCR…')

        # ── 4. MCR per region ─────────────────────────────────────────────────
        from gui.utils.mcr_utils import mcr_annotation_region, simulate_lorentzian_profile

        virtual_peaks = []
        rank_counter  = 1

        for ridx, (lo, hi, entries, mask, region_noise) in enumerate(viable):
            self._update_status(
                f'MCR region {ridx+1}/{len(viable)}  '
                f'[{lo:.3f}–{hi:.3f} ppm]…')

            ppm_reg = ppm_arr[mask]
            X_reg   = (all_int[mask, :] if all_int.ndim == 2
                       else all_int[mask, np.newaxis])

            # Deduplicate entries: one per (hmdb_id, ppm_center)
            seen_keys = {}
            for e in entries:
                key = (e.get('hmdb_id') or e.get('bmrb_id', ''),
                       round(float(e['ppm_center']), 3))
                if key not in seen_keys:
                    seen_keys[key] = e
            uniq_entries = list(seen_keys.values())

            # Build simulated profiles
            db_profiles = []
            for e in uniq_entries:
                try:
                    dp_ppm = float(e['ppm_center'])
                    mult   = e.get('type') or e.get('multiplicity') or 's'
                    try:
                        nh = int(e.get('protons') or 1)
                    except Exception:
                        nh = 1
                    ppm_span = None
                    try:
                        ps = e.get('ppm_start'); pe = e.get('ppm_end')
                        if ps is not None and pe is not None:
                            ppm_span = abs(float(pe) - float(ps))
                    except Exception:
                        pass
                    sticks = _build_stick_positions(
                        dp_ppm, mult, nh, e.get('j_coupling'),
                        field_mhz=mhz, ppm_span=ppm_span)
                    if not sticks:
                        sticks = [(dp_ppm, 1.0)]
                    prof = simulate_lorentzian_profile(sticks, ppm_reg)
                except Exception:
                    prof = np.zeros(len(ppm_reg))
                db_profiles.append(prof)

            if not db_profiles:
                continue

            try:
                S, C, mcr_m, K, assignments, aligned_profiles, cand_scores = \
                    mcr_annotation_region(
                        X_reg, db_profiles, r2_threshold=r2_thr,
                        ppm_arr=ppm_reg, shift_tol_ppm=shift_tol,
                        use_baseline=True, db_entries=uniq_entries,
                        field_mhz=mhz, max_k_override=max_k)
            except Exception as e:
                print(f'MCR error region [{lo:.3f}–{hi:.3f}]: {e}')
                continue

            # Flat-region R² guard: if the total spectral variance in this region
            # is below 4× noise², R² is meaningless (trivially high on flat signal).
            X_mean = X_reg.mean(axis=1)
            region_variance = float(np.var(X_mean))
            if region_variance < (region_noise * 2.0) ** 2:
                print(f'Skipping flat region [{lo:.3f}–{hi:.3f}]: '
                      f'variance={region_variance:.2e} < noise threshold')
                continue

            center_ppm = float((lo + hi) / 2)

            vp = {
                'rank':      rank_counter,
                'ppm':       center_ppm,
                'start_ppm': lo,
                'end_ppm':   hi,
                'mcr_annotation': {
                    'S':           S,
                    'C':           C,
                    'K':           K,
                    'metrics':     mcr_m,
                    'assignments': assignments,
                    'candidates':        uniq_entries,
                    'db_profiles':       aligned_profiles,   # shift-aligned
                    'db_profiles_raw':   db_profiles,        # original (no shift)
                    'cand_scores':       cand_scores,        # per-candidate score info
                    'ppm_region':        ppm_reg,
                    'r2_threshold':      r2_thr,
                    'shift_tol_ppm':     shift_tol,
                    'region_mode':       region_mode,
                },
            }
            virtual_peaks.append(vp)
            rank_counter += 1

        return virtual_peaks

    def _mask_from_detected_peaks(self, ppm_arr, lo, hi):
        """
        Build a region mask from peak-detection bounds overlapping [lo, hi].
        Falls back to DB bounds when no detected peaks overlap.
        """
        pd_stage = getattr(self.app, 'stages', {}).get('peak_detection')
        peaks = getattr(pd_stage, 'peaks', []) if pd_stage is not None else []
        lo = float(min(lo, hi)); hi = float(max(lo, hi))
        overlaps = []
        for p in peaks:
            try:
                p_lo = float(min(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm'])))
                p_hi = float(max(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm'])))
            except Exception:
                continue
            if p_hi >= lo and p_lo <= hi:
                overlaps.append((p_lo, p_hi))

        if overlaps:
            lo = min(a for a, _ in overlaps)
            hi = max(b for _, b in overlaps)
            margin = max(0.002, (hi - lo) * 0.10)
            lo -= margin
            hi += margin
        else:
            margin = max(0.004, (hi - lo) * 0.30)
            lo -= margin
            hi += margin

        lo = max(float(np.min(ppm_arr)), lo)
        hi = min(float(np.max(ppm_arr)), hi)
        return (ppm_arr >= lo) & (ppm_arr <= hi), lo, hi

    def _update_status(self, msg):
        try:
            self.app.root.after(0, lambda m=msg: self._set_status(m))
        except Exception:
            pass

    # ── DB query ──────────────────────────────────────────────────────────────

    def _query_db(self, db_path, ppm_lo, ppm_hi):
        """Return list of dicts with hmdb_id/bmrb_id, name, ppm_center, type, protons, j_coupling."""
        results = []
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur  = conn.cursor()
            tables = {r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

            if 'shifts_1h' in tables:
                # BMRB schema
                cur.execute(
                    "SELECT s.bmrb_id as hmdb_id, met.name,"
                    " s.ppm as ppm_center, s.multiplicity as type,"
                    " s.j_coupling, 1 as protons"
                    " FROM shifts_1h s"
                    " JOIN metabolites met ON s.bmrb_id = met.bmrb_id"
                    " WHERE s.ppm BETWEEN ? AND ?",
                    (ppm_lo, ppm_hi))
                for row in cur.fetchall():
                    results.append(dict(row))

            if 'multiplets' in tables:
                # HMDB schema
                cur.execute(
                    "SELECT m.hmdb_id, met.name, m.ppm_center,"
                    " m.ppm_start, m.ppm_end,"
                    " m.type, m.protons, m.j_coupling"
                    " FROM multiplets m"
                    " JOIN metabolites met ON m.hmdb_id = met.hmdb_id"
                    " WHERE m.ppm_center BETWEEN ? AND ?",
                    (ppm_lo, ppm_hi))
                for row in cur.fetchall():
                    results.append(dict(row))

            if 'peaks' in tables and 'multiplets' not in tables \
                    and 'shifts_1h' not in tables:
                # Fallback: generic peaks table
                cur.execute(
                    "SELECT hmdb_id, ppm as ppm_center,"
                    " 's' as type, 1 as protons, NULL as j_coupling"
                    " FROM peaks"
                    " WHERE ppm BETWEEN ? AND ?",
                    (ppm_lo, ppm_hi))
                for row in cur.fetchall():
                    results.append(dict(row))

            conn.close()
        except Exception as e:
            print(f'DB query error: {e}')
        return results

    # Expose mcr_r2_threshold and _rerun_mcr_for_peak for MCRWindow
    @property
    def mcr_r2_threshold(self):
        try:
            return float(self.r2_var.get())
        except Exception:
            return 0.90

    @mcr_r2_threshold.setter
    def mcr_r2_threshold(self, value):
        try:
            self.r2_var.set(round(float(value), 2))
        except Exception:
            pass

    def _rerun_mcr_for_peak(self, peak, r2_threshold):
        """Re-run MCR on a single virtual peak with a new R² threshold."""
        mcr_ann = peak.get('mcr_annotation')
        if not mcr_ann:
            return
        from gui.utils.mcr_utils import mcr_annotation_region, simulate_lorentzian_profile
        from gui.widgets.annotation_results_window import _build_stick_positions
        cands    = mcr_ann['candidates']
        ppm_lo   = float(peak['start_ppm'])
        ppm_hi   = float(peak['end_ppm'])
        ppm_arr  = getattr(self.app, 'ppm', None)
        all_int  = getattr(self.app, 'all_intensities', None)
        if ppm_arr is None or all_int is None:
            return
        mean_int = (np.mean(all_int, axis=1) if all_int.ndim == 2 else all_int)
        _tail_mask = ppm_arr <= np.percentile(ppm_arr, 10)
        _tail = mean_int[_tail_mask]
        noise = float(np.median(np.abs(_tail - np.median(_tail)))) * 1.4826 + 1e-12
        try:
            min_snr = float(self.snr_var.get())
        except Exception:
            min_snr = 3.0
        try:
            max_width = float(self.max_width_var.get())
        except Exception:
            max_width = 0.15
        try:
            mhz = float(self.mhz_var.get())
        except Exception:
            mhz = 500.0

        mask, ppm_lo, ppm_hi = _adaptive_region_mask(
            ppm_arr, mean_int, ppm_lo, ppm_hi, cands, noise,
            min_snr=min_snr, field_mhz=mhz, max_expand=max_width)
        if mask.sum() < 4:
            return
        ppm_reg_new = ppm_arr[mask]
        X_reg = (all_int[mask, :] if all_int.ndim == 2
                 else all_int[mask, np.newaxis])
        db_profiles = []
        for c in cands:
            try:
                dp_ppm = float(c.get('ppm_center') or c.get('ppm') or 0)
                mult   = c.get('type') or 's'
                try:
                    nh = int(c.get('protons') or 1)
                except Exception:
                    nh = 1
                sticks = _build_stick_positions(dp_ppm, mult, nh, c.get('j_coupling'))
                if not sticks:
                    sticks = [(dp_ppm, 1.0)]
                prof = simulate_lorentzian_profile(sticks, ppm_reg_new)
            except Exception:
                prof = np.zeros(len(ppm_reg_new))
            db_profiles.append(prof)
        if not db_profiles:
            return
        S, C, mcr_m, K, assignments, aligned_profiles, cand_scores = \
            mcr_annotation_region(
                X_reg, db_profiles, r2_threshold=r2_threshold,
                ppm_arr=ppm_reg_new, shift_tol_ppm=mcr_ann.get('shift_tol_ppm', 0.0),
                use_baseline=True, db_entries=cands, field_mhz=mhz)
        mcr_ann.update({
            'S': S, 'C': C, 'K': K,
            'metrics': mcr_m,
            'assignments': assignments,
            'db_profiles': aligned_profiles,
            'cand_scores': cand_scores,
            'ppm_region': ppm_reg_new,
            'r2_threshold': r2_threshold,
        })
        peak['start_ppm'] = float(ppm_lo)
        peak['end_ppm'] = float(ppm_hi)
