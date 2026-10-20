# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/widgets/peak_sample_view.py
"""
Per-sample view for a single NMR peak.

Layout
------
┌─────────────────────────────────────────────────────┐
│  Header (peak info)                                  │
│  Nav bar                                             │
├─────────────────────────────────────────────────────┤
│  TOP: Treeview table (scrollable, one row/sample)   │
├──────────────────────────┬──────────────────────────┤
│  BOTTOM-LEFT: spectrum   │  BOTTOM-RIGHT: reserved  │
│  of selected sample      │  (annotation info)       │
└──────────────────────────┴──────────────────────────┘

Double-click a row → opens PeakRegionEditor to fine-tune
integration boundaries for that sample (quantification only).
Stored in peak['sample_regions'][sample_idx].
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from processing.signal_quality import (
    SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR)
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)


# ── Column definitions ────────────────────────────────────────────────────────
COLS = [
    ('sample',    'Sample',    150, tk.W),
    ('group',     'Group',      80, tk.CENTER),
    ('snr',       'SNR',        65, tk.CENTER),
    ('intensity', 'Intensity',  90, tk.CENTER),
    ('area',      'Area',       95, tk.CENTER),
    ('metabolite','Metabolite',140, tk.W),
    ('ppm',       'δ (ppm)',    75, tk.CENTER),
    ('start',     'Start',      75, tk.CENTER),
    ('end',       'End',        75, tk.CENTER),
    ('status',    'Status',     70, tk.CENTER),
    ('adj',       'Adj.',       40, tk.CENTER),
]


class PeakSampleView:

    def __init__(self, parent, peak, ppm, all_intensities, app):
        self.peak            = peak
        self.ppm             = ppm
        self.all_intensities = all_intensities
        self.app             = app
        self._sample_data    = []
        self._selected_idx   = None
        self._fig            = None
        self._ax             = None
        self._canvas         = None
        self._drag_x0        = None
        self._drag_shade     = None
        self._drag_vlo       = None
        self._drag_vhi       = None

        peak.setdefault('sample_regions', {})

        si = getattr(app, 'samples_info', []) or []
        n  = all_intensities.shape[1]

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"Signal #{peak.get('rank','?')}  @  {peak['ppm']:.4f} ppm — per sample")
        self.win.geometry('1000x680')
        self.win.minsize(750, 450)

        self._precompute(si, n)
        self._build()
        self._populate()

        # Select first row
        if self._sample_data:
            self._select(0)

        self._center()

        # On close: refresh quantification table + detail panel
        self.win.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── Data ─────────────────────────────────────────────────────────────────

    def _precompute(self, si, n):
        noise_std = (getattr(
            getattr(self.app, 'stages', {}).get('peak_detection', None),
            'noise_std', None) or 1.0)
        gc = getattr(self.app, 'group_colors', {})

        p = self.peak
        p_lo = min(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm']))
        p_hi = max(p.get('start_ppm', p['ppm']), p.get('end_ppm', p['ppm']))
        self._p_lo = p_lo
        self._p_hi = p_hi
        self._is_sub = p.get('is_sub_peak', False)

        # For sub-peaks, use a wider view centred on the parent region
        # so the user sees the sub-peak in its spectral context
        if self._is_sub:
            wid = max(p.get('width_ppm', p.get('width', 0.01)), 0.01)
            margin = max(wid * 8, 0.15)
        else:
            margin = max((p_hi - p_lo) * 2.0, 0.08)
        view_lo  = p['ppm'] - margin
        view_hi  = p['ppm'] + margin
        self._view_lo = view_lo
        self._view_hi = view_hi

        # Build metabolite lookup: peak_rank → metabolite name (from annotation results)
        _met_by_rank = {}
        ann_stage = getattr(self.app, 'stages', {}).get('annotation', None)
        if ann_stage is not None:
            _last = getattr(ann_stage, '_last_results_args', None) or {}
            for _res in _last.get('results', []):
                _pr = _res.get('peak_rank')
                if _pr is not None and _pr not in _met_by_rank:
                    _met_by_rank[_pr] = _res.get('name', _res.get('metabolite', ''))
        _peak_rank = p.get('rank')
        # Check if this peak is the IS reference
        _quant_stage_tmp = getattr(self.app, 'stages', {}).get('quantification', None)
        _is_peak = getattr(_quant_stage_tmp, 'ref_peak', None)
        _is_name = None
        if _is_peak is not None and _is_peak is p:
            _is_name = getattr(_quant_stage_tmp, 'ref_std_var', None)
            _is_name = (_is_name.get() if _is_name else None) or 'IS'
        _metabolite = _is_name if _is_name else _met_by_rank.get(_peak_rank, '—')

        def _trapz_local(y, x):
            try:    return float(np.trapezoid(y, x=x))
            except: return float(np.trapz(y, x=x))

        # Fetch quantification results for this peak (if quantification has run)
        _quant_stage = getattr(self.app, 'stages', {}).get('quantification', None)
        _quant_results = getattr(_quant_stage, '_quant_results', None) or []
        _quant_r = (_quant_stage._row_of_peak(p)
                    if _quant_stage is not None
                    and hasattr(_quant_stage, '_row_of_peak') else None)
        _quant_areas = _quant_r.get('_areas_per_sample') if _quant_r else None
        self._quant_result = _quant_r   # store for callback
        self._quant_stage  = _quant_stage
        self._quant_available = _quant_r is not None

        for s in range(n):
            inf   = si[s] if s < len(si) else {}
            sname = inf.get('name', f'S{s}')
            grp   = inf.get('label', '—')
            stype = str(inf.get('type', 'Sample')).upper()
            color = gc.get(grp, f'C{s % 10}')

            col_s = self.all_intensities[:, s]
            adj   = self.peak['sample_regions'].get(s)
            lo_s  = adj['start_ppm'] if adj else p_lo
            hi_s  = adj['end_ppm']   if adj else p_hi

            pk_m  = (self.ppm >= lo_s) & (self.ppm <= hi_s)
            lint  = col_s[pk_m]

            # Use quantified area if available, else raw trapezoid
            if _quant_areas is not None and s < len(_quant_areas):
                area_s = float(_quant_areas[s])
            else:
                area_s = abs(_trapz_local(col_s[pk_m], self.ppm[pk_m])) if pk_m.any() else 0.0

            # For deconvolved sub-peaks: use the real NNLS amplitude per sample
            hps = p.get('height_per_sample')
            if self._is_sub and hps is not None and s < len(hps):
                inten = float(hps[s])
            else:
                inten = float(np.max(lint)) if lint.size else 0.0

            snr_s = inten / noise_std if noise_std > 0 else 0.0
            # Prefer the per-sample apex computed at detection (bounded + noise-
            # gated, so absent samples keep the reference position instead of
            # chasing noise); fall back to the in-window argmax for older peaks.
            _pps = p.get('ppm_per_sample')
            _ppk = p.get('aligned_per_sample')
            if self._is_sub:
                lppm = float(p['ppm'])
            elif (_pps is not None and s < len(_pps)
                  and not (_ppk is not None and s < len(_ppk) and not _ppk[s])):
                lppm = float(_pps[s])
            else:
                lppm = (float(self.ppm[pk_m][np.argmax(lint)])
                        if lint.size else p['ppm'])

            bg = {SIGNAL_GOOD:    '#f0fff0',
                  SIGNAL_POOR:    '#fff0f0',
                  SIGNAL_AVERAGE: '#fffce0'}.get(p.get('status'), '#ffffff')
            if stype == 'BLK':
                bg = '#f5f5f5'

            self._sample_data.append({
                'idx': s, 'name': sname, 'group': grp,
                'type': stype, 'color': color, 'bg': bg,
                'snr': snr_s, 'intensity': inten, 'area': area_s,
                'metabolite': _metabolite,
                'ppm': lppm, 'lo_s': lo_s, 'hi_s': hi_s,
                'col': col_s, 'adjusted': adj is not None,
                'view_lo': view_lo, 'view_hi': view_hi,
            })

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Header ───────────────────────────────────────────────────────────
        hdr = ttk.Frame(self.win, padding=(6, 4))
        hdr.pack(side=tk.TOP, fill=tk.X)
        p = self.peak
        sub_tag = (f"  [sub-peak of {p.get('parent_ppm', '?'):.4f} ppm  ·  "
                   f"model: {p.get('model','Lorentz')}]"
                   if self._is_sub else "")
        ttk.Label(hdr,
                  text=(f"Signal #{p.get('rank','?')}  ·  "
                        f"{p['ppm']:.4f} ppm  ·  "
                        f"[{self._p_lo:.4f} → {self._p_hi:.4f}]  ·  "
                        f"Width {p.get('width_ppm',0):.4f} ppm  ·  "
                        f"Status: {p.get('status') or '—'}"
                        f"{sub_tag}"),
                  font=('Arial', 9, 'bold'),
                  foreground='#2e7d32' if self._is_sub else 'black'
                  ).pack(side=tk.LEFT)
        if self._quant_available:
            hint = 'Right-drag on plot → adjust region  |  Double-click row → editor  |  Area = quantified'
            hint_col = '#1a5f1a'
        else:
            hint = 'Run Quantification to enable region editing  |  Area = raw trapz'
            hint_col = '#888'
        ttk.Label(hdr, text=hint,
                  font=('Arial', 7), foreground=hint_col).pack(
            side=tk.RIGHT, padx=6)

        # ── Nav bar ───────────────────────────────────────────────────────────
        nav = ttk.Frame(self.win, padding=(6, 2))
        nav.pack(side=tk.TOP, fill=tk.X)
        for txt, cmd in [('▲', self._prev), ('▼', self._next),
                         ('⌂ Top', self._top), ('↓ Bottom', self._bottom)]:
            ttk.Button(nav, text=txt, width=max(2, len(txt)+1),
                       command=cmd).pack(side=tk.LEFT, padx=2)
        self._nav_lbl = ttk.Label(nav, text='', foreground='#555',
                                  font=('Arial', 8))
        self._nav_lbl.pack(side=tk.LEFT, padx=8)
        ttk.Button(nav, text='✎ Reset adj.',
                   command=self._reset_adj).pack(side=tk.RIGHT, padx=4)

        ttk.Separator(self.win, orient='horizontal').pack(
            fill=tk.X, padx=4, pady=2)

        # ── Vertical split: top=table, bottom=detail ──────────────────────────
        vsplit = tk.PanedWindow(self.win, orient=tk.VERTICAL,
                                sashwidth=6, sashrelief=tk.RAISED)
        vsplit.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # ── TOP: Treeview ─────────────────────────────────────────────────────
        top_frame = ttk.Frame(vsplit)
        vsplit.add(top_frame, minsize=120, stretch='always')

        vsb = ttk.Scrollbar(top_frame, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(top_frame, orient=tk.HORIZONTAL)
        self.tree = ttk.Treeview(
            top_frame,
            columns=[c[0] for c in COLS],
            show='headings',
            selectmode='browse',
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        top_frame.rowconfigure(0, weight=1)
        top_frame.columnconfigure(0, weight=1)

        for cid, heading, width, anchor in COLS:
            # without a quantification row the column is a raw integral of the
            # window, not the area the engine measured: it says so
            if cid == 'area' and not self._quant_available:
                heading = 'Raw area'
            self.tree.heading(cid, text=heading)
            self.tree.column(cid, width=width, anchor=anchor,
                             minwidth=30, stretch=False)

        for tag, bg in [('good', '#f0fff0'), ('poor', '#fff0f0'),
                        ('undef', '#fffce0'), ('blk', '#f5f5f5'),
                        ('none', 'white')]:
            self.tree.tag_configure(tag, background=bg)
        self.tree.tag_configure('sel_hl', background='#cce5ff')

        self.tree.bind('<<TreeviewSelect>>', self._on_tree_sel)
        self.tree.bind('<Double-Button-1>',  self._on_double)

        # ── BOTTOM: horizontal split ──────────────────────────────────────────
        bot_frame = ttk.Frame(vsplit)
        vsplit.add(bot_frame, minsize=180, stretch='always')

        hsplit = tk.PanedWindow(bot_frame, orient=tk.HORIZONTAL,
                                sashwidth=5, sashrelief=tk.RAISED)
        hsplit.pack(fill=tk.BOTH, expand=True)

        # Bottom-left: spectrum
        self._spec_frame = ttk.LabelFrame(
            hsplit, text='Sample spectrum', padding=2)
        hsplit.add(self._spec_frame, minsize=300, stretch='always')

        # Bottom-right: peak quality info
        self._info_frame = ttk.LabelFrame(
            hsplit, text='Signal Quality', padding=6)
        hsplit.add(self._info_frame, minsize=200, stretch='always')
        self._build_info_panel()

        self._build_spectrum_panel()

        # Keyboard nav
        self.win.bind('<Up>',   lambda e: self._prev())
        self.win.bind('<Down>', lambda e: self._next())
        self.win.bind('<Home>', lambda e: self._top())
        self.win.bind('<End>',  lambda e: self._bottom())

    def _build_spectrum_panel(self):
        """Build the matplotlib figure in the bottom-left panel."""
        self._fig = plt.Figure(figsize=(5, 3), dpi=100)
        self._fig.patch.set_facecolor('#fafafa')
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor('#fafafa')
        self._ax.text(0.5, 0.5, 'Select a sample above',
                      transform=self._ax.transAxes,
                      ha='center', va='center', color='gray', fontsize=10)
        self._ax.set_xticks([]); self._ax.set_yticks([])

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._spec_frame)
        self._canvas.draw()

        tb = NavigationToolbar2Tk(self._canvas, self._spec_frame)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._canvas.get_tk_widget().bind('<Button-3>',        self._rb_press)
        self._canvas.get_tk_widget().bind('<B3-Motion>',       self._rb_drag)
        self._canvas.get_tk_widget().bind('<ButtonRelease-3>', self._rb_release)

    def _build_info_panel(self):
        """Populate bottom-right panel with quality info for this peak."""
        p  = self.peak
        fr = self._info_frame

        def _row(label, value, fg='#222'):
            r = ttk.Frame(fr)
            r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=f'{label}:', width=16, anchor=tk.W,
                      font=('Arial', 8, 'bold'), foreground='#555').pack(side=tk.LEFT)
            ttk.Label(r, text=str(value), anchor=tk.W,
                      font=('Arial', 8), foreground=fg).pack(side=tk.LEFT)

        def _sep():
            ttk.Separator(fr, orient='horizontal').pack(fill=tk.X, pady=3)

        # Peak identity
        _row('Rank',    f"#{p.get('rank', '?')}")
        _row('δ (ppm)', f"{p['ppm']:.4f}")
        _row('Width',   f"{p.get('width_ppm', p.get('width', 0)):.4f} ppm")
        _row('Status',  p.get('status') or '—')
        _sep()

        # SNR
        snr = p.get('snr', 0)
        snr_col = '#1a7a1a' if snr > 10 else '#856404' if snr > 5 else '#9a0000'
        _row('SNR', f'{snr:.1f}', snr_col)

        # Area — shows selected sample's area (updates on row select)
        _area_row = ttk.Frame(fr)
        _area_row.pack(fill=tk.X, pady=1)
        ttk.Label(_area_row, text='Area:', width=16, anchor=tk.W,
                  font=('Arial', 8, 'bold'), foreground='#555').pack(side=tk.LEFT)
        self._info_area_var = tk.StringVar(value='—')
        ttk.Label(_area_row, textvariable=self._info_area_var,
                  anchor=tk.W, font=('Arial', 8)).pack(side=tk.LEFT)

        _snr_row = ttk.Frame(fr)
        _snr_row.pack(fill=tk.X, pady=1)
        ttk.Label(_snr_row, text='Sample SNR:', width=16, anchor=tk.W,
                  font=('Arial', 8, 'bold'), foreground='#555').pack(side=tk.LEFT)
        self._info_snr_var = tk.StringVar(value='—')
        ttk.Label(_snr_row, textvariable=self._info_snr_var,
                  anchor=tk.W, font=('Arial', 8)).pack(side=tk.LEFT)
        _sep()

        # Annotation
        ann_stage = getattr(self.app, 'stages', {}).get('annotation', None)
        _last = getattr(ann_stage, '_last_results_args', None) or {} if ann_stage else {}
        _rank = p.get('rank')
        _matched = [r for r in _last.get('results', []) if r.get('peak_rank') == _rank]
        if _matched:
            best = max(_matched, key=lambda r: float(
                r.get('composite_score', r.get('multiplet_score', 0)) or 0))
            score = float(best.get('composite_score', best.get('multiplet_score', 0)) or 0)
            sc_col = '#1a7a1a' if score >= 0.8 else '#856404' if score >= 0.5 else '#9a0000'
            _row('Metabolite', best.get('name', best.get('metabolite', '—')))
            _row('Ann. Score',  f'{score:.3f}', sc_col)
            _mult = (best.get('multiplicity') or best.get('type') or '—')
            _nh   = best.get('protons') or best.get('num_protons')
            _nh_s = str(int(_nh)) if _nh and str(_nh).replace('.','').isdigit() else '—'
            _row('Pattern', _mult)
            _row('nH',      _nh_s)
        else:
            ttk.Label(fr, text='No annotation yet.',
                      foreground='gray', font=('Arial', 8)).pack(anchor=tk.W, pady=2)

        # MCR metrics (from quantification results if available)
        _sep()
        quant_stage = getattr(self.app, 'stages', {}).get('quantification', None)
        _qr = getattr(quant_stage, '_quant_results', []) if quant_stage else []
        _qres = next((r for r in _qr if r.get('_peak', {}).get('rank') == _rank), None)
        if _qres:
            _mcr = _qres.get('_mcr')
            if _mcr:
                met = _mcr.get('metrics', {})
                lof = met.get('lof', float('nan'))
                r2  = met.get('r2',  float('nan'))
                ni  = met.get('n_iter', '?')
                lof_col = '#1a7a1a' if (np.isfinite(lof) and lof < 5) else \
                          '#856404' if (np.isfinite(lof) and lof < 15) else '#9a0000'
                r2_col  = '#1a7a1a' if (np.isfinite(r2) and r2 > 0.99) else \
                          '#856404' if (np.isfinite(r2) and r2 > 0.95) else '#9a0000'
                _row('MCR LOF',  f'{lof:.2f}%' if np.isfinite(lof) else 'N/A', lof_col)
                _row('MCR R²',   f'{r2:.4f}'   if np.isfinite(r2)  else 'N/A', r2_col)
                _row('MCR Iter', str(ni))
            reasons = _qres.get('_status_reasons', [])
            if reasons:
                _sep()
                ttk.Label(fr, text='Quality breakdown:',
                          font=('Arial', 8, 'bold'), foreground='#555').pack(anchor=tk.W)
                for reason in reasons:
                    col = ('#1a7a1a' if 'Good' in reason
                           else '#856404' if 'Warn' in reason else '#9a0000')
                    ttk.Label(fr, text=f'  {reason}',
                              font=('Arial', 7), foreground=col).pack(anchor=tk.W)
        else:
            ttk.Label(fr, text='Run Quantification\nto see MCR metrics.',
                      foreground='gray', font=('Arial', 8)).pack(anchor=tk.W, pady=2)

    # ── Populate ──────────────────────────────────────────────────────────────

    def _populate(self):
        status = self.peak.get('status')
        for d in self._sample_data:
            tag = {SIGNAL_GOOD: 'good', SIGNAL_POOR: 'poor',
                   SIGNAL_AVERAGE: 'undef'}.get(status, 'none')
            if d['type'] == 'BLK':
                tag = 'blk'
            self.tree.insert('', tk.END, iid=str(d['idx']),
                             tags=(tag,),
                             values=(
                                 d['name'],
                                 d['group'],
                                 f"{d['snr']:.1f}",
                                 f"{d['intensity']:.0f}",
                                 f"{d['area']:.3e}",
                                 d.get('metabolite', '—'),
                                 f"{d['ppm']:.4f}",
                                 f"{d['lo_s']:.4f}",
                                 f"{d['hi_s']:.4f}",
                                 status or '—',
                                 '✎' if d['adjusted'] else '',
                             ))

    # ── Spectrum panel ────────────────────────────────────────────────────────

    def _draw_spectrum(self, d):
        ax = self._ax
        ax.clear()
        ax.set_facecolor('#fafafa')

        vmask  = (self.ppm >= d['view_lo']) & (self.ppm <= d['view_hi'])
        x_view = self.ppm[vmask]
        y_view = d['col'][vmask]

        # Raw spectrum
        ax.plot(x_view, y_view, color=d['color'], lw=1.1, alpha=0.75,
                label=f"{d['name']} [{d['group']}]")

        # Integration region
        lo_s, hi_s = d['lo_s'], d['hi_s']
        pk_m = (self.ppm >= lo_s) & (self.ppm <= hi_s)
        if pk_m.any():
            ax.fill_between(self.ppm[pk_m], 0, d['col'][pk_m],
                            color=d['color'], alpha=0.20, zorder=1)
        ax.axvline(lo_s, color='#1565C0', lw=1.0, ls='--', alpha=0.70)
        ax.axvline(hi_s, color='#1565C0', lw=1.0, ls='--', alpha=0.70)
        ax.axvline(d['ppm'], color='#e63946', lw=0.9, ls=':', alpha=0.7)

        # ── Sub-peak: draw all deconvolved components for this sample ─────────
        p = self.peak
        if self._is_sub:
            siblings = p.get('region_sub_peaks') or []
            model    = p.get('model', 'Lorentzian')
            s_idx    = d['idx']

            # ── Local model functions (same as algorithm) ──────────────────
            def _lor(x_a, amp, cen, wid):
                return amp * wid**2 / ((x_a - cen)**2 + wid**2)

            def _gau(x_a, amp, cen, wid):
                return amp * np.exp(-(x_a - cen)**2 / (2 * wid**2))

            def _pv(x_a, amp, cen, wid, eta=0.5):
                return eta * _lor(x_a, amp, cen, wid) + \
                       (1 - eta) * _gau(x_a, amp, cen, wid)

            def _skl(x_a, amp, cen, wid, gamma=0.0):
                lor  = wid**2 / ((x_a - cen)**2 + wid**2)
                skew = np.clip(
                    1.0 + np.sin(gamma) * (x_a - cen) / (wid + 1e-10), 0, None)
                return amp * lor * skew

            def _eval(sp_dict, x_a, amp_override=None):
                amp  = amp_override if amp_override is not None else sp_dict['height']
                cen  = sp_dict['ppm']
                wid  = float(sp_dict.get('width', sp_dict.get('width_ppm', 0.005)))
                m    = sp_dict.get('model', model)
                if m == 'Gaussian':
                    return _gau(x_a, amp, cen, wid)
                elif m == 'Pseudo-Voigt':
                    return _pv(x_a, amp, cen, wid, sp_dict.get('eta', 0.5))
                elif m in ('Skewed-Lorentzian', 'Skewed_Lorentzian'):
                    return _skl(x_a, amp, cen, wid, sp_dict.get('gamma', 0.0))
                return _lor(x_a, amp, cen, wid)

            x_fit     = np.linspace(d['view_lo'], d['view_hi'], 600)
            cumulative = np.zeros_like(x_fit)
            colors_sp  = plt.cm.tab10(np.linspace(0, 1, max(len(siblings), 1)))

            this_ppm = float(p['ppm'])
            for j, sp in enumerate(siblings):
                cen_j = float(sp['ppm'])

                # Use per-sample NNLS amplitude if available, else mean height
                hps = sp.get('height_per_sample')
                if hps is not None and s_idx < len(hps):
                    amp_j = float(hps[s_idx])
                else:
                    amp_j = float(sp['height'])

                y_j = _eval(sp, x_fit, amp_j)
                cumulative += y_j

                is_this = abs(cen_j - this_ppm) < 0.0005
                col        = colors_sp[j]
                fill_alpha = 0.40 if is_this else 0.05
                line_alpha = 1.0  if is_this else 0.18
                lw         = 2.2  if is_this else 0.9
                ls         = '-'  if is_this else ':'
                label = (f'This peak  {cen_j:.4f} ppm  amp={amp_j:.0f}'
                         if is_this else
                         f'#{j+1}  {cen_j:.4f} ppm  amp={amp_j:.0f}')

                ax.fill_between(x_fit, 0, y_j,
                                color=col, alpha=fill_alpha, zorder=2)
                ax.plot(x_fit, y_j, ls=ls, color=col, lw=lw,
                        alpha=line_alpha, zorder=5, label=label)
                # Dot only for the current peak
                if is_this:
                    max_idx = int(np.argmax(y_j))
                    ax.scatter([x_fit[max_idx]], [float(y_j[max_idx])],
                               color=col, s=45, zorder=6)

            # Cumulative fit — should overlay the raw spectrum faithfully
            ax.plot(x_fit, cumulative, color='#e63946', lw=2.0, ls='-',
                    alpha=0.85, zorder=7, label=f'Sum fit  amp_tot={cumulative.max():.0f}')

        if d['adjusted']:
            ax.axvline(self._p_lo, color='#aaa', lw=0.7,
                       ls=':', alpha=0.5, label='Global boundary')
            ax.axvline(self._p_hi, color='#aaa', lw=0.7, ls=':', alpha=0.5)

        ax.invert_xaxis()
        ax.set_xlabel('δ (ppm)', fontsize=8)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, ls=':')
        adj_txt   = '  [✎ adjusted]' if d['adjusted'] else ''
        sub_txt   = '  [deconvolved sub-peak]' if self._is_sub else ''
        ax.set_title(
            f"{d['name']}  [{d['group']}]  —  "
            f"SNR {d['snr']:.1f}  |  {d['intensity']:.0f} a.u."
            f"{adj_txt}{sub_txt}",
            fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        self._fig.tight_layout()
        self._canvas.draw()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select(self, idx):
        if idx < 0 or idx >= len(self._sample_data):
            return
        self._selected_idx = idx
        iid = str(idx)
        self.tree.selection_set(iid)
        self.tree.see(iid)
        d = self._sample_data[idx]
        self._nav_lbl.config(
            text=f"{idx+1}/{len(self._sample_data)}  ·  "
                 f"{d['name']}  [{d['group']}]")
        self._draw_spectrum(d)
        if hasattr(self, '_info_area_var'):
            area = d.get('area', float('nan'))
            self._info_area_var.set(f"{area:.3e}" if area == area else '—')
        if hasattr(self, '_info_snr_var'):
            snr = d.get('snr', float('nan'))
            self._info_snr_var.set(f"{snr:.1f}" if snr == snr else '—')

    def _on_tree_sel(self, e=None):
        sel = self.tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except ValueError:
            return
        if idx != self._selected_idx:
            self._select(idx)

    def _on_double(self, e=None):
        if not self._quant_available:
            return
        if self._selected_idx is not None:
            self._open_editor(self._selected_idx)

    def _prev(self):
        cur = self._selected_idx if self._selected_idx is not None else 1
        self._select(max(0, cur - 1))

    def _next(self):
        cur = self._selected_idx if self._selected_idx is not None else -1
        self._select(min(len(self._sample_data) - 1, cur + 1))

    def _top(self):    self._select(0)
    def _bottom(self): self._select(len(self._sample_data) - 1)

    # ── Region editor ─────────────────────────────────────────────────────────

    def _open_editor(self, idx):
        qs = self._quant_stage
        if self._quant_available and qs is not None:
            ok, why = qs.sample_area_editable(self.peak)
            if not ok:
                messagebox.showinfo('Sample window', why, parent=self.win)
                return
        d = self._sample_data[idx]
        PeakRegionEditor(
            self.win, idx, d, self.ppm, self._p_lo, self._p_hi,
            on_save=lambda lo, hi: self._on_saved(idx, lo, hi))

    def _on_saved(self, s, lo, hi):
        """The analyst set the window of sample ``s``. The number the engine
        measured for that sample is scaled by the change of the integral
        (``update_sample_area``); with no quantification yet the window waits
        on the signal and is applied by the next run."""
        qs = self._quant_stage
        if self._quant_available and qs is not None:
            why = qs.update_sample_area(self.peak, s, lo, hi)
            if why:
                messagebox.showinfo('Sample window', why, parent=self.win)
                return
        self.peak['sample_regions'][s] = {'start_ppm': lo, 'end_ppm': hi}
        self._show_sample(s, lo, hi, adjusted=True)

    def _show_sample(self, s, lo, hi, *, adjusted):
        """Refresh one sample's row and figure for the window ``[lo, hi]``."""
        d = self._sample_data[s]
        d['lo_s'] = lo; d['hi_s'] = hi; d['adjusted'] = adjusted
        noise_std = (getattr(
            getattr(self.app, 'stages', {}).get('peak_detection', None),
            'noise_std', None) or 1.0)
        pk_m = (self.ppm >= lo) & (self.ppm <= hi)
        lint = d['col'][pk_m]
        d['intensity'] = float(np.max(lint)) if lint.size else 0.0
        d['snr']       = d['intensity'] / noise_std if noise_std > 0 else 0.0
        if lint.size:
            d['ppm'] = float(self.ppm[pk_m][np.argmax(lint)])
        # the area: the engine's row when there is one, else the raw integral
        r = self._quant_result
        areas = (r.get('_areas_per_sample') if r else None) or []
        if self._quant_available and s < len(areas):
            d['area'] = float(areas[s])
        else:
            try:
                tr = float(np.trapezoid(lint, x=self.ppm[pk_m]))
            except AttributeError:
                tr = float(np.trapz(lint, x=self.ppm[pk_m]))
            d['area'] = abs(tr) if lint.size > 1 else 0.0
        self.tree.item(str(s), values=(
            d['name'], d['group'],
            f"{d['snr']:.1f}", f"{d['intensity']:.0f}",
            f"{d['area']:.3e}",
            d.get('metabolite', '—'),
            f"{d['ppm']:.4f}", f"{lo:.4f}", f"{hi:.4f}",
            self.peak.get('status') or '—', '✎' if adjusted else ''))
        if self._selected_idx == s:
            self._draw_spectrum(d)
            if hasattr(self, '_info_area_var'):
                self._info_area_var.set(f"{d['area']:.3e}")

    def _reset_adj(self):
        idx = self._selected_idx
        if idx is None:
            return
        self.peak['sample_regions'].pop(idx, None)
        qs = self._quant_stage
        lo, hi = self._p_lo, self._p_hi
        if self._quant_available and qs is not None:
            qs.reset_sample_area(self.peak, idx)
            win = (self._quant_result or {}).get('_multiplet_window')
            if win:
                lo, hi = min(win), max(win)
        self._show_sample(idx, lo, hi, adjusted=False)

    # ── Right-click drag to adjust region directly on the spectrum ────────────

    def _px_to_ppm(self, event):
        try:
            return float(self._ax.transData.inverted().transform(
                (event.x, event.y))[0])
        except Exception:
            return None

    def _rb_press(self, e):
        if self._selected_idx is None or self._ax is None:
            return
        self._drag_x0 = self._px_to_ppm(e)

    def _rb_drag(self, e):
        if self._drag_x0 is None:
            return
        x1 = self._px_to_ppm(e)
        if x1 is None:
            return
        lo, hi = sorted([self._drag_x0, x1])
        # Remove previous drag overlays
        for attr in ('_drag_shade', '_drag_vlo', '_drag_vhi'):
            artist = getattr(self, attr)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
        self._drag_shade = self._ax.axvspan(
            lo, hi, color='#FF6B00', alpha=0.22, zorder=3)
        self._drag_vlo = self._ax.axvline(
            lo, color='#FF6B00', lw=1.5, ls='--', alpha=0.9, zorder=4)
        self._drag_vhi = self._ax.axvline(
            hi, color='#FF6B00', lw=1.5, ls='--', alpha=0.9, zorder=4)
        self._canvas.draw_idle()

    def _rb_release(self, e):
        if self._drag_x0 is None:
            return
        x1 = self._px_to_ppm(e)
        drag_start = self._drag_x0
        # Clear drag state before redraw
        self._drag_x0 = None
        self._drag_shade = None
        self._drag_vlo = None
        self._drag_vhi = None
        if x1 is None or self._selected_idx is None:
            if self._selected_idx is not None:
                self._draw_spectrum(self._sample_data[self._selected_idx])
            return
        lo, hi = sorted([drag_start, x1])
        if abs(hi - lo) < 0.001:
            self._draw_spectrum(self._sample_data[self._selected_idx])
            return
        self._on_saved(self._selected_idx, lo, hi)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _center(self):
        self.win.update_idletasks()
        w = self.win.winfo_width(); h = self.win.winfo_height()
        sw = self.win.winfo_screenwidth(); sh = self.win.winfo_screenheight()
        self.win.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')

    def _on_close(self):
        """Refresh what depends on the numbers the analyst changed."""
        try:
            qs = self._quant_stage
            if qs is not None and self._quant_available:
                try:
                    qs._compute_vip_scores()
                except Exception:
                    pass
                if getattr(qs, '_right_widget', None):
                    qs._populate_tree()
        except Exception:
            pass
        self.win.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  PeakRegionEditor
# ══════════════════════════════════════════════════════════════════════════════

class PeakRegionEditor:
    """Fine-tune integration boundaries for one sample (quantification only)."""

    def __init__(self, parent, sample_idx, d, ppm,
                 global_lo, global_hi, on_save):
        self.d          = d
        self.ppm        = ppm
        self.global_lo  = global_lo
        self.global_hi  = global_hi
        self.on_save    = on_save
        self._new_lo    = d['lo_s']
        self._new_hi    = d['hi_s']
        self._drag_x0   = None

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"Adjust region — {d['name']}  [{d['group']}]")
        self.win.geometry('720x440')
        self.win.minsize(560, 320)
        self.win.grab_set()

        self._build()
        self._center()

    def _build(self):
        d = self.d

        info = ttk.Frame(self.win, padding=(8, 4))
        info.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(info,
                  text=(f"Sample: {d['name']}  ·  Group: {d['group']}  ·  "
                        f"SNR: {d['snr']:.1f}  ·  Intensity: {d['intensity']:.0f}"),
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT)

        note = ttk.Frame(self.win, padding=(8, 1))
        note.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(note,
                  text=('Right-click & drag to set integration region  '
                        '(quantification only — identification uses global boundaries)'),
                  font=('Arial', 7), foreground='#666').pack(side=tk.LEFT)

        # Plot
        pf = ttk.Frame(self.win)
        pf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        margin  = max((self.global_hi - self.global_lo) * 3, 0.15)
        vlo     = self.global_lo - margin
        vhi     = self.global_hi + margin
        vmask   = (self.ppm >= vlo) & (self.ppm <= vhi)
        x_view  = self.ppm[vmask]
        y_view  = d['col'][vmask]

        self.fig = plt.Figure(figsize=(7, 3.6), dpi=100)
        self.fig.patch.set_facecolor('#fafafa')
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor('#fafafa')
        self.ax.plot(x_view, y_view, color=d['color'], lw=1.2, alpha=0.9)

        self._shade = self.ax.axvspan(
            self._new_lo, self._new_hi, color='#1565C0', alpha=0.15)
        self._vlo = self.ax.axvline(
            self._new_lo, color='#1565C0', lw=1.2, ls='--', alpha=0.8)
        self._vhi = self.ax.axvline(
            self._new_hi, color='#1565C0', lw=1.2, ls='--', alpha=0.8)
        # Global boundaries (grey reference)
        self.ax.axvline(self.global_lo, color='#aaa', lw=0.8,
                        ls=':', alpha=0.6)
        self.ax.axvline(self.global_hi, color='#aaa', lw=0.8,
                        ls=':', alpha=0.6)

        self.ax.invert_xaxis()
        self.ax.set_xlabel('δ (ppm)', fontsize=9)
        self.ax.set_ylabel('Intensity', fontsize=9)
        self.ax.grid(True, alpha=0.2, ls=':')
        self._update_title()
        self.fig.tight_layout()

        self.cv = FigureCanvasTkAgg(self.fig, master=pf)
        self.cv.draw()
        tb = NavigationToolbar2Tk(self.cv, pf)
        tb.update(); tb.pack(side=tk.BOTTOM, fill=tk.X)
        self.cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cv.get_tk_widget().bind('<Button-3>',        self._rb_press)
        self.cv.get_tk_widget().bind('<B3-Motion>',       self._rb_drag)
        self.cv.get_tk_widget().bind('<ButtonRelease-3>', self._rb_release)

        # Bottom bar
        bf = ttk.Frame(self.win, padding=(8, 4))
        bf.pack(side=tk.BOTTOM, fill=tk.X)
        self._rlbl = ttk.Label(
            bf,
            text=f"Region: {self._new_lo:.4f} → {self._new_hi:.4f} ppm",
            font=('Arial', 8))
        self._rlbl.pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Reset to global',
                   command=self._reset).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bf, text='✓ Confirm',
                   command=self._confirm).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bf, text='Cancel',
                   command=self.win.destroy).pack(side=tk.RIGHT, padx=4)

    def _px_to_ppm(self, event):
        try:
            return float(self.ax.transData.inverted().transform(
                (event.x, event.y))[0])
        except Exception:
            return None

    def _rb_press(self, e):
        self._drag_x0 = self._px_to_ppm(e)

    def _rb_drag(self, e):
        if self._drag_x0 is None:
            return
        x1 = self._px_to_ppm(e)
        if x1 is None:
            return
        lo, hi = sorted([self._drag_x0, x1])
        self._set_region(lo, hi)

    def _rb_release(self, e):
        self._drag_x0 = None

    def _set_region(self, lo, hi):
        self._new_lo = lo; self._new_hi = hi
        self._shade.remove()
        self._shade = self.ax.axvspan(lo, hi, color='#1565C0', alpha=0.15)
        self._vlo.set_xdata([lo, lo])
        self._vhi.set_xdata([hi, hi])
        self._update_title()
        self._rlbl.config(text=f"Region: {lo:.4f} → {hi:.4f} ppm")
        self.cv.draw_idle()

    def _update_title(self):
        self.ax.set_title(
            f"{self.d['name']}  —  right-drag to set region  "
            f"({self._new_lo:.4f} → {self._new_hi:.4f} ppm)",
            fontsize=8)

    def _reset(self):
        self._set_region(self.global_lo, self.global_hi)

    def _confirm(self):
        lo, hi = sorted([self._new_lo, self._new_hi])
        if abs(hi - lo) < 0.001:
            messagebox.showwarning('Too narrow',
                'Region too narrow. Drag a wider selection.',
                parent=self.win)
            return
        self.on_save(lo, hi)
        self.win.destroy()

    def _center(self):
        self.win.update_idletasks()
        w = self.win.winfo_width(); h = self.win.winfo_height()
        sw = self.win.winfo_screenwidth(); sh = self.win.winfo_screenheight()
        self.win.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')