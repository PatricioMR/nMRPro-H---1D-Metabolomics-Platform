# gui/widgets/mcr_deconv_window.py
"""
MCR Deconvolution Results window.

Opened after the Annotation stage when peaks have MCR decomposition data.

Left  : Treeview of MCR-annotated peaks
         ├─ Peak #3  2.3401 ppm  K=2  R²=0.971
         │    ├─ C1: Alanine       r=0.93  ✓ OK
         │    └─ C2: Lactate       r=0.81  —
         └─ ...

Right : Deconvolution plot (exp + total fit + per-component)
        R² threshold spinbox + Recompute button
        Per-component Accept / Reject
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

COMP_COLORS = ['#1565c0', '#e65100', '#2e7d32', '#6a1b9a', '#880e4f']


class MCRDeconvWindow:
    def __init__(self, parent, peaks, app, ann_stage=None):
        """
        Parameters
        ----------
        parent    : tk widget (root or Toplevel)
        peaks     : list of detected peaks (full list; only MCR ones are shown)
        app       : main App instance (for ppm / all_intensities)
        ann_stage : AnnotationStage instance (for _rerun_mcr_for_peak)
        """
        self.peaks     = peaks
        self.app       = app
        self.ann_stage = ann_stage

        # Only overlapping regions (K≥2) — caller is expected to pre-filter,
        # but guard here as well so the window is never populated with singles.
        self.mcr_peaks = [p for p in peaks
                          if p.get('mcr_annotation', {}).get('K', 0) >= 2]

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"MCR Deconvolution — {len(self.mcr_peaks)} overlapping region(s)")
        self.win.geometry("1500x900")
        self.win.minsize(1000, 650)
        self._center()

        self._selected_peak    = None
        self._focused_k        = None
        self._preview_candidate = None   # (ci, pk) being previewed
        # Pre-create figure so tree-select events during _build_ui never fail
        self._fig    = Figure(figsize=(6, 1.9), dpi=95, facecolor='#fafafa')
        self._canvas = None   # replaced in _build_right

        self._build_ui()
        if self.mcr_peaks:
            self._select_peak(self.mcr_peaks[0])

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        # Info bar
        ttk.Label(
            main,
            text=(f"  {len(self.mcr_peaks)} overlapping region(s)  "
                  "— accept or reject each deconvolved component"),
            font=("Arial", 9, "italic"), foreground="#444"
        ).pack(anchor=tk.W, pady=(0, 4))

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel first (correct visual order)
        left = ttk.Frame(paned, padding=2)
        paned.add(left, weight=1)
        self._build_list(left)

        right = ttk.Frame(paned, padding=2)
        paned.add(right, weight=3)
        self._build_right(right)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_list(self, parent):
        # Filter bar
        ff = ttk.Frame(parent)
        ff.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(ff, text="Show:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._filt_var = tk.StringVar(value='all')
        for lbl, val in [('All', 'all'), ('Pending', 'pending'),
                         ('Accepted', 'accepted'), ('Rejected', 'rejected')]:
            ttk.Radiobutton(ff, text=lbl, variable=self._filt_var,
                            value=val,
                            command=self._populate_tree).pack(side=tk.LEFT, padx=2)

        cols = ('ppm', 'K', 'r2', 'status')
        self.tree = ttk.Treeview(parent, columns=cols, show='tree headings',
                                  selectmode='browse')
        self.tree.heading('#0',    text='Peak')
        self.tree.heading('ppm',   text='δ (ppm)')
        self.tree.heading('K',     text='K')
        self.tree.heading('r2',    text='R²')
        self.tree.heading('status',text='Status')

        self.tree.column('#0',    width=70,  stretch=False)
        self.tree.column('ppm',   width=80,  stretch=False)
        self.tree.column('K',     width=30,  stretch=False)
        self.tree.column('r2',    width=55,  stretch=False)
        self.tree.column('status',width=80,  stretch=True)

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        # Quick-action buttons
        bf = ttk.Frame(parent)
        bf.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(bf, text='✓ Accept all regions',
                   command=self._accept_all_overlap).pack(fill=tk.X, pady=1)

        self._populate_tree()

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        filt = getattr(self, '_filt_var', None)
        filt_val = filt.get() if filt else 'all'

        # Tag colours: simple=blue row, overlap=orange row, done=green, rejected=red
        self.tree.tag_configure('simple',   foreground='#1565c0')
        self.tree.tag_configure('overlap',  foreground='#e65100')
        self.tree.tag_configure('accepted', foreground='#1a7a1a')
        self.tree.tag_configure('rejected', foreground='#9a0000')
        self.tree.tag_configure('pending',  foreground='#555555')

        for pk in self.mcr_peaks:
            mcr  = pk['mcr_annotation']
            K    = mcr['K']
            r2   = mcr.get('metrics', {}).get('r2', float('nan'))
            conf = mcr.get('confirmed_assignments', {})

            # Determine overall status
            n_accepted = sum(1 for v in conf.values() if v and v.get('accepted'))
            n_rejected = sum(1 for v in conf.values() if v and not v.get('accepted'))
            n_done     = n_accepted + n_rejected
            is_simple  = (K == 1)

            if n_done == 0:
                st_txt, row_tag = '—', 'pending'
            elif n_rejected == K:
                st_txt, row_tag = '✗ all', 'rejected'
            elif n_accepted == K:
                st_txt, row_tag = '✓ all', 'accepted'
            else:
                st_txt, row_tag = f'{n_accepted}/{K} ✓', 'pending'

            # Apply filter
            if filt_val == 'pending'  and n_done == K:      continue
            if filt_val == 'accepted' and n_accepted != K:  continue
            if filt_val == 'rejected' and n_rejected != K:  continue

            p_iid = f'pk_{pk["rank"]}'
            self.tree.insert('', tk.END, iid=p_iid,
                             text=f'⊕ #{pk["rank"]}',
                             values=(f'{pk["ppm"]:.4f}', K,
                                     ('—' if (r2 != r2) else f'{r2:.3f}'), st_txt),
                             tags=(row_tag,),
                             open=False)

            # Component children
            cands   = mcr.get('candidates', [])
            assigns = mcr.get('assignments', [])
            for k, assign in enumerate(assigns):
                ci, corr = assign[0], assign[1]
                cand   = cands[ci] if ci < len(cands) else {}
                name   = cand.get('name', f'C{k+1}')[:22]
                c_conf = conf.get(k)
                if c_conf is None:
                    c_st, c_tag = '—', 'pending'
                elif c_conf['accepted']:
                    c_st, c_tag = '✓ OK', 'accepted'
                else:
                    c_st, c_tag = '✗ Rejected', 'rejected'
                self.tree.insert(p_iid, tk.END,
                                 iid=f'comp_{pk["rank"]}_{k}',
                                 text=f'  C{k+1}',
                                 values=(f'r={corr:.2f}', '', '', c_st),
                                 tags=(c_tag,))

    def _on_tree_select(self, event=None):
        sel = self.tree.focus()
        if not sel:
            return
        # iid formats: pk_<rank>  or  comp_<rank>_<k>
        if sel.startswith('comp_'):
            parts = sel.split('_')          # ['comp', rank, k]
            rank  = int(parts[1])
            k     = int(parts[2])
            pk    = next((p for p in self.mcr_peaks if p['rank'] == rank), None)
            if pk:
                self._focused_k = k
                if pk is not self._selected_peak:
                    self._selected_peak = pk
                self._refresh_right(pk)
            return
        # Peak row selected — clear component focus
        pk_iid = sel
        if not pk_iid.startswith('pk_'):
            return
        rank = int(pk_iid.split('_')[1])
        pk   = next((p for p in self.mcr_peaks if p['rank'] == rank), None)
        if pk:
            self._focused_k = None
            if pk is not self._selected_peak:
                self._select_peak(pk)
            else:
                self._refresh_right(pk)

    def _select_peak(self, pk):
        self._selected_peak = pk
        # Highlight in tree
        iid = f'pk_{pk["rank"]}'
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)
        self._refresh_right(pk)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self, parent):
        # Toolbar: R² threshold + Recompute
        tb = ttk.Frame(parent)
        tb.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(tb, text="R² min:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._r2_var = tk.DoubleVar(value=getattr(self.ann_stage, 'mcr_r2_threshold', 0.95))
        ttk.Spinbox(tb, from_=0.50, to=1.00, increment=0.01,
                    textvariable=self._r2_var, width=5, format='%.2f'
                    ).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Button(tb, text='Recompute MCR',
                   command=self._recompute_all).pack(side=tk.LEFT)

        # Plot canvas (subplots created dynamically in _draw_plot)
        self._fig    = Figure(figsize=(9, 3), dpi=95, facecolor='#fafafa',
                              layout='constrained')
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Navigation toolbar (pan / zoom / home / save)
        self._toolbar = NavigationToolbar2Tk(self._canvas, parent, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(fill=tk.X, side=tk.BOTTOM)

        # Component table — scrollable container, fixed height
        lf = ttk.LabelFrame(parent, text='Components', padding=4)
        lf.pack(fill=tk.X, pady=(4, 0))

        _vsb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._comp_canvas = tk.Canvas(lf, bd=0, highlightthickness=0,
                                      height=160,
                                      yscrollcommand=_vsb.set)
        _vsb.configure(command=self._comp_canvas.yview)
        self._comp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._comp_frame = tk.Frame(self._comp_canvas, bg='white')
        self._comp_win   = self._comp_canvas.create_window(
            (0, 0), window=self._comp_frame, anchor='nw')

        def _on_comp_configure(e):
            self._comp_canvas.configure(
                scrollregion=self._comp_canvas.bbox('all'))
        def _on_canvas_resize(e):
            self._comp_canvas.itemconfig(self._comp_win, width=e.width)

        self._comp_frame.bind('<Configure>', _on_comp_configure)
        self._comp_canvas.bind('<Configure>', _on_canvas_resize)

        # Mouse-wheel scroll
        def _on_wheel(e, c=self._comp_canvas):
            if e.num == 4:   c.yview_scroll(-1, 'units')
            elif e.num == 5: c.yview_scroll(1,  'units')
            else:            c.yview_scroll(int(-e.delta / 60), 'units')
        self._comp_canvas.bind('<MouseWheel>', _on_wheel)
        self._comp_canvas.bind('<Button-4>',   _on_wheel)
        self._comp_canvas.bind('<Button-5>',   _on_wheel)
        self._wheel_fn   = _on_wheel   # keep ref to bind on child widgets

        self._comp_rows  = []   # keep refs to destroy on refresh

    def _refresh_right(self, pk):
        """Redraw plot and component table for peak pk."""
        self._draw_plot(pk)
        self._draw_comp_table(pk)

    def _draw_plot(self, pk, focused_k=None):
        if self._canvas is None:
            return
        if focused_k is None:
            focused_k = self._focused_k

        mcr = pk['mcr_annotation']
        self._fig.clear()

        K       = min(mcr['K'], mcr['S'].shape[1])
        S       = mcr['S']           # (n_pts_reg, K)
        C       = mcr['C']           # (n_samples, K)
        ppm_reg = mcr['ppm_region']  # ppm grid for the MCR region
        c_mean  = C.mean(axis=0)[:K]
        assigns = mcr.get('assignments', [])
        cands   = mcr.get('candidates', [])
        r2      = mcr.get('metrics', {}).get('r2', float('nan'))
        conf    = mcr.get('confirmed_assignments', {})

        # ── Experimental spectrum ─────────────────────────────────────────────
        ppm_full = getattr(self.app, 'ppm', None)
        all_int  = getattr(self.app, 'all_intensities', None)
        int_full = None
        if all_int is not None:
            int_full = (np.mean(all_int, axis=1)
                        if all_int.ndim == 2 else all_int.copy())

        reg_lo = float(min(ppm_reg)); reg_hi = float(max(ppm_reg))
        reg_w  = reg_hi - reg_lo

        exp_ctx_ppm = exp_ctx_int = None
        exp_reg_ppm = exp_reg_int = None
        e_max = 1.0

        if ppm_full is not None and int_full is not None:
            # MCR region only — for residual and amplitude reference
            reg_mask = ((ppm_full >= min(reg_lo, reg_hi)) &
                        (ppm_full <= max(reg_lo, reg_hi)))
            if reg_mask.sum() > 1:
                exp_reg_ppm = ppm_full[reg_mask]
                exp_reg_int = int_full[reg_mask]
                # Robust scale to avoid extreme outlier dominating amplitude.
                e_max = float(np.percentile(exp_reg_int, 99.0))
                if e_max <= 0:
                    e_max = float(exp_reg_int.max()) if exp_reg_int.max() > 0 else 1.0

        # ── MCR fit at real intensity scale ───────────────────────────────────
        # Scale so total fit peak matches the experimental peak amplitude
        raw_fit = sum(S[:, k] * c_mean[k] for k in range(K))
        fit_peak = raw_fit.max() if raw_fit.max() > 0 else 1.0
        amp_scale = e_max / fit_peak

        # Build per-component info (intensities in real units, not normalised)
        components = []
        total_area = sum((S[:, k] * c_mean[k]).sum() for k in range(K)) + 1e-12
        for k in range(K):
            sk = S[:, k] * c_mean[k] * amp_scale   # real intensity units
            _a = assigns[k] if k < len(assigns) else (0, 0.0)
            ci, corr = int(_a[0]), float(_a[1])
            cand   = cands[ci] if ci < len(cands) else {}
            lbl    = cand.get('name', f'C{k+1}')[:24]
            color  = COMP_COLORS[k % len(COMP_COLORS)]
            c_conf = conf.get(k)
            accepted = c_conf is None or c_conf['accepted']
            if focused_k is not None:
                alpha = 0.92 if k == focused_k else 0.10
                lw_c  = 2.5  if k == focused_k else 0.5
            else:
                alpha = 0.88 if accepted else 0.28
                lw_c  = 2.0  if accepted else 0.8
            pct = 100.0 * (S[:, k] * c_mean[k]).sum() / total_area
            components.append((k, sk, f'C{k+1}: {lbl}  r={corr:.2f}  ({pct:.0f}%)',
                                color, alpha, lw_c, ci))

        fit_scaled = np.clip(sum(c[1] for c in components), 0, None)

        # ── Sort ppm_reg ascending — needed for curve_fit and np.interp ─────
        sort_idx   = np.argsort(ppm_reg)
        ppm_sorted = ppm_reg[sort_idx]          # ascending, within MCR region

        def _sort(vec):
            return vec[sort_idx]

        # Context window for the experimental spectrum background only.
        # Components and fit are plotted on ppm_sorted (exact MCR output).
        ctx_margin = max(0.06, reg_w * 1.5)
        ctx_lo = reg_lo - ctx_margin
        ctx_hi = reg_hi + ctx_margin

        if ppm_full is not None and int_full is not None:
            ctx_mask = ((ppm_full >= min(ctx_lo, ctx_hi)) &
                        (ppm_full <= max(ctx_lo, ctx_hi)))
            if ctx_mask.sum() > 1:
                exp_ctx_ppm = ppm_full[ctx_mask]
                exp_ctx_int = int_full[ctx_mask]

        if exp_ctx_ppm is not None and exp_ctx_ppm[0] > exp_ctx_ppm[-1]:
            exp_ctx_ppm = exp_ctx_ppm[::-1]
            exp_ctx_int = exp_ctx_int[::-1]

        # ── Subplots ──────────────────────────────────────────────────────────
        gs  = self._fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.04)
        ax1 = self._fig.add_subplot(gs[0])
        ax2 = self._fig.add_subplot(gs[1], sharex=ax1)

        # 1. Experimental spectrum — context window background
        if exp_ctx_ppm is not None and exp_ctx_int is not None:
            ax1.fill_between(exp_ctx_ppm, exp_ctx_int,
                             alpha=0.13, color='#444444', zorder=1)
            ax1.plot(exp_ctx_ppm, exp_ctx_int,
                     color='#333333', lw=1.1, alpha=0.75,
                     label='Experimental (mean)', zorder=2)

        # 2. DB fingerprint profiles — raw model output on ppm_sorted
        db_profiles = mcr.get('db_profiles', [])
        drawn_db: set = set()
        for k, sk, lbl, color, alpha, lw_c, ci in components:
            if ci in drawn_db or ci >= len(db_profiles):
                continue
            prof = db_profiles[ci]
            if len(prof) != len(ppm_reg):
                continue
            sk_sorted  = np.clip(_sort(sk), 0, None)
            db_sorted  = np.clip(_sort(np.asarray(prof)), 0, None)
            db_alpha   = 0.85 if (focused_k is None or focused_k == k) else 0.22
            comp_peak  = float(sk_sorted.max()) if sk_sorted.max() > 0 else e_max
            db_peak    = float(db_sorted.max()) if db_sorted.max() > 0 else 1.0
            ax1.plot(ppm_sorted, db_sorted * comp_peak / db_peak,
                     color=color, lw=1.5, ls=':', alpha=db_alpha, zorder=3,
                     label=f'DB: {lbl.split(":")[1].strip().split()[0]}')
            drawn_db.add(ci)

        # Preview candidate overlay — raw on ppm_sorted
        if self._preview_candidate is not None:
            prev_ci, prev_pk = self._preview_candidate
            if prev_pk is pk and prev_ci < len(db_profiles):
                prev_prof = db_profiles[prev_ci]
                if len(prev_prof) == len(ppm_reg):
                    prev_sorted = np.clip(_sort(np.asarray(prev_prof)), 0, None)
                    prev_peak   = float(prev_sorted.max()) if prev_sorted.max() > 0 else 1.0
                    pname = (cands[prev_ci].get('name', '?')
                             if prev_ci < len(cands) else '?')
                    ax1.plot(ppm_sorted, prev_sorted * e_max / prev_peak,
                             color='#555555', lw=1.4, ls='--', alpha=0.80,
                             label=f'Preview: {pname}', zorder=4)

        # 3. Per-component fills and lines — exact MCR output on ppm_sorted
        for k, sk, lbl, color, alpha, lw_c, ci in components:
            sk_sorted = np.clip(_sort(sk), 0, None)
            ax1.fill_between(ppm_sorted, sk_sorted,
                             alpha=max(0.04, alpha * 0.22),
                             color=color, zorder=2)
            ax1.plot(ppm_sorted, sk_sorted,
                     color=color, lw=lw_c, alpha=alpha,
                     label=lbl, zorder=4)

        # 4. MCR total fit — only within the MCR region (no tail extension).
        # The individual deconvolved components already show full natural peak
        # shapes with tails; the fit line is just the sum within ppm_reg.
        fit_in_reg = np.clip(
            sum(_sort(sk) for k, sk, lbl, color, alpha, lw_c, ci in components),
            0, None)
        ax1.plot(ppm_sorted, fit_in_reg,
                 color='#c62828', lw=2.0, ls='--', alpha=0.90, zorder=5,
                 label=f'MCR fit  R²={"—" if np.isnan(r2) else f"{r2:.3f}"}')

        ax1.tick_params(labelsize=7, labelbottom=False)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=7)
        ax1.set_title(
            f'Peak #{pk["rank"]}  {pk["ppm"]:.4f} ppm  K={K}  '
            f'R²={"—" if np.isnan(r2) else f"{r2:.3f}"}',
            fontsize=8, pad=3)
        ax1.legend(fontsize=6.5, loc='upper left',
                   framealpha=0.85, ncol=min(2, K + 3))
        ax1.set_facecolor('#fafafa')
        ax1.set_xlim(max(ctx_lo, ctx_hi), min(ctx_lo, ctx_hi))  # NMR: high ppm left
        comp_peaks = [float(np.max(np.clip(_sort(sk), 0, None)))
                      for k, sk, *_ in components]
        y_top = max(e_max, max(comp_peaks) if comp_peaks else 1.0)
        ax1.set_ylim(bottom=-y_top * 0.05, top=y_top * 1.20)

        # 5. Residual — TRUE MCR residual, only within ppm_reg.
        # Outside the fitted region the residual is undefined; show a flat zero.
        if exp_reg_ppm is not None and exp_reg_int is not None:
            if exp_reg_ppm[0] > exp_reg_ppm[-1]:
                rrp = exp_reg_ppm[::-1]; rri = exp_reg_int[::-1]
            else:
                rrp = exp_reg_ppm.copy(); rri = exp_reg_int.copy()
            fit_reg = np.zeros_like(rrp, dtype=float)
            for k, sk, lbl, color, alpha, lw_c, ci in components:
                fit_reg += np.interp(rrp, ppm_sorted,
                                     np.clip(_sort(sk), 0, None))
            residual = rri - fit_reg
            ax2.fill_between(rrp, residual, alpha=0.28, color='#888888')
            ax2.plot(rrp, residual, color='#555555', lw=0.9)
        ax2.axhline(0, color='#aaaaaa', lw=0.7, ls='--')
        ax2.set_ylabel('Residual\n(MCR region)', fontsize=7)
        ax2.set_xlabel('δ (ppm)', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.set_facecolor('#f5f5f5')

        self._canvas.draw_idle()

    def _draw_comp_table(self, pk):
        # Destroy old rows
        for w in self._comp_rows:
            try: w.destroy()
            except Exception: pass
        self._comp_rows = []

        mcr     = pk['mcr_annotation']
        K       = mcr['K']
        assigns = mcr.get('assignments', [])
        cands   = mcr.get('candidates', [])
        conf    = mcr.get('confirmed_assignments', {})
        parent  = self._comp_frame

        # Header
        hdr_bg = '#e8eaf6'
        hdr = tk.Frame(parent, bg=hdr_bg)
        hdr.pack(fill=tk.X, pady=(0, 1))
        self._comp_rows.append(hdr)
        for col_txt, col_w in [
            ('#',     3), ('Metabolite', 22),
            ('Corr',  6), ('Δppm',        6),
            ('J',     5), ('Conf',        6), ('Status', 9), ('Action', 10),
        ]:
            tk.Label(hdr, text=col_txt, font=('Arial', 8, 'bold'),
                     bg=hdr_bg, width=col_w, anchor='w',
                     padx=4).pack(side=tk.LEFT)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=1)
        self._comp_rows.append(parent.winfo_children()[-1])

        for k, assign in enumerate(assigns):
            ci        = assign[0]
            corr      = assign[1]
            shift_ppm = assign[2] if len(assign) > 2 else 0.0
            j_score   = assign[3] if len(assign) > 3 else None
            meas_j    = assign[4] if len(assign) > 4 else None
            exp_j     = assign[5] if len(assign) > 5 else None
            combined  = assign[6] if len(assign) > 6 else None
            cand    = cands[ci] if ci < len(cands) else {}
            name    = cand.get('name', f'Component {k+1}')
            color   = COMP_COLORS[k % len(COMP_COLORS)]
            c_conf  = conf.get(k)

            if c_conf is None:
                row_bg, st_txt, st_fg = 'white', '—', '#888888'
            elif c_conf['accepted']:
                row_bg, st_txt, st_fg = '#e8f5e9', '✓ Accepted', '#1a7a1a'
            else:
                row_bg, st_txt, st_fg = '#ffebee', '✗ Rejected', '#9a0000'

            row = tk.Frame(parent, bg=row_bg, bd=0)
            row.pack(fill=tk.X, pady=1)
            self._comp_rows.append(row)

            # Color swatch + number
            tk.Label(row, text=f'C{k+1}', font=('Arial', 9, 'bold'),
                     fg=color, bg=row_bg, width=3, anchor='w',
                     padx=4).pack(side=tk.LEFT)
            # Name
            tk.Label(row, text=name, font=('Arial', 9),
                     bg=row_bg, width=26, anchor='w',
                     padx=2).pack(side=tk.LEFT)
            # Correlation
            corr_fg = ('#1a7a1a' if corr >= 0.85 else
                       '#856404' if corr >= 0.60 else '#9a0000')
            tk.Label(row, text=f'{corr:.3f}', font=('Arial', 9),
                     fg=corr_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)
            # Δppm shift applied
            shift_txt = f'{shift_ppm:+.3f}' if abs(shift_ppm) > 1e-4 else '0'
            shift_fg  = '#c62828' if abs(shift_ppm) > 0.015 else '#444444'
            tk.Label(row, text=shift_txt, font=('Arial', 9),
                     fg=shift_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)
            # J score
            if j_score is None:
                j_txt, j_fg = '—', '#888888'
            else:
                mult = (cand.get('type') or cand.get('multiplicity') or 's').lower()
                if mult == 's':
                    j_txt, j_fg = 's', '#888888'   # singlet: not applicable
                else:
                    j_txt = f'{j_score:.2f}'
                    j_fg  = ('#1a7a1a' if j_score >= 0.75 else
                              '#856404' if j_score >= 0.45 else '#9a0000')
                    if meas_j and exp_j:
                        j_txt = f'{j_score:.2f}\n({meas_j:.1f}/{exp_j:.1f}Hz)'
            tk.Label(row, text=j_txt, font=('Arial', 8),
                     fg=j_fg, bg=row_bg, width=5, anchor='w',
                     justify='left').pack(side=tk.LEFT)
            # Combined confidence
            if combined is not None:
                conf_txt = f'{combined:.2f}'
                conf_fg  = ('#1a7a1a' if combined >= 0.75 else
                             '#856404' if combined >= 0.50 else '#9a0000')
            else:
                conf_txt, conf_fg = '—', '#888888'
            tk.Label(row, text=conf_txt, font=('Arial', 9, 'bold'),
                     fg=conf_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)
            # Status
            tk.Label(row, text=st_txt, font=('Arial', 9, 'bold'),
                     fg=st_fg, bg=row_bg, width=10, anchor='w').pack(side=tk.LEFT)
            # Accept / Reject buttons
            def _acc(c=cand, k_=k, p=pk):
                self._set_component(p, k_, c, accepted=True)
            def _rej(c=cand, k_=k, p=pk):
                self._set_component(p, k_, c, accepted=False)

            acc_bg = '#c8e6c9' if (c_conf and c_conf['accepted']) else '#f5f5f5'
            rej_bg = '#ffcdd2' if (c_conf and not c_conf['accepted']) else '#f5f5f5'
            acc_rel = 'sunken' if (c_conf and c_conf['accepted']) else 'raised'
            rej_rel = 'sunken' if (c_conf and not c_conf['accepted']) else 'raised'

            tk.Button(row, text='✓ Accept', font=('Arial', 8),
                      bg=acc_bg, relief=acc_rel,
                      command=_acc).pack(side=tk.LEFT, padx=3)
            tk.Button(row, text='✗ Reject', font=('Arial', 8),
                      bg=rej_bg, relief=rej_rel,
                      command=_rej).pack(side=tk.LEFT, padx=1)

            # Forward mouse-wheel from row widgets to the scroll canvas
            _wf = getattr(self, '_wheel_fn', None)
            if _wf:
                for w in [row] + list(row.winfo_children()):
                    w.bind('<MouseWheel>', _wf)
                    w.bind('<Button-4>',   _wf)
                    w.bind('<Button-5>',   _wf)

        # ── Other candidates panel ────────────────────────────────────────────
        assigned_cis = {assign[0] for assign in assigns}
        other_cands  = [(ci, cands[ci]) for ci in range(len(cands))
                        if ci not in assigned_cis]
        if other_cands:
            sep2 = ttk.Separator(parent, orient=tk.HORIZONTAL)
            sep2.pack(fill=tk.X, pady=(6, 2))
            self._comp_rows.append(sep2)

            sec_lbl = tk.Label(parent,
                               text='Other candidates (not assigned by MCR)',
                               font=('Arial', 8, 'italic'), fg='#555555', bg='white',
                               anchor='w', padx=6)
            sec_lbl.pack(fill=tk.X)
            self._comp_rows.append(sec_lbl)

            for ci, cand in other_cands:
                self._draw_other_candidate_row(parent, pk, ci, cand)

        # Scroll to top after rebuild
        self._comp_canvas.after_idle(
            lambda: self._comp_canvas.yview_moveto(0))

    def _draw_other_candidate_row(self, parent, pk, ci, cand):
        """Render one row in the Other candidates section."""
        mcr         = pk['mcr_annotation']
        K           = mcr['K']
        cand_scores = mcr.get('cand_scores', [])
        sc          = cand_scores[ci] if ci < len(cand_scores) else {}

        corr      = sc.get('corr',    0.0)
        shift_ppm = sc.get('shift',   0.0)
        j_score   = sc.get('j_score', None)
        meas_j    = sc.get('meas_j',  None)
        exp_j     = sc.get('exp_j',   None)
        combined  = sc.get('combined', None)

        is_prev = (self._preview_candidate is not None
                   and self._preview_candidate == (ci, pk))
        row_bg = '#f0f4ff' if is_prev else '#fafafa'

        row = tk.Frame(parent, bg=row_bg, bd=0)
        row.pack(fill=tk.X, pady=1)
        self._comp_rows.append(row)

        name  = cand.get('name', f'Cand {ci}')
        mult  = (cand.get('type') or cand.get('multiplicity') or 's').lower()

        # Name
        tk.Label(row, text=f'  {name}', font=('Arial', 9),
                 bg=row_bg, width=26, anchor='w').pack(side=tk.LEFT)
        # Corr
        corr_fg = ('#1a7a1a' if corr >= 0.85 else
                   '#856404' if corr >= 0.60 else '#9a0000')
        tk.Label(row, text=f'{corr:.3f}', font=('Arial', 9),
                 fg=corr_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)
        # Δppm
        shift_txt = f'{shift_ppm:+.3f}' if abs(shift_ppm) > 1e-4 else '0'
        shift_fg  = '#c62828' if abs(shift_ppm) > 0.015 else '#444444'
        tk.Label(row, text=shift_txt, font=('Arial', 9),
                 fg=shift_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)
        # J
        if j_score is None or mult == 's':
            j_txt, j_fg = ('s' if mult == 's' else '—'), '#888888'
        else:
            j_fg  = ('#1a7a1a' if j_score >= 0.75 else
                     '#856404' if j_score >= 0.45 else '#9a0000')
            j_txt = f'{j_score:.2f}'
            if meas_j and exp_j:
                j_txt = f'{j_score:.2f}\n({meas_j:.1f}/{exp_j:.1f}Hz)'
        tk.Label(row, text=j_txt, font=('Arial', 8),
                 fg=j_fg, bg=row_bg, width=5, anchor='w',
                 justify='left').pack(side=tk.LEFT)
        # Conf
        if combined is not None:
            conf_txt = f'{combined:.2f}'
            conf_fg  = ('#1a7a1a' if combined >= 0.75 else
                        '#856404' if combined >= 0.50 else '#9a0000')
        else:
            conf_txt, conf_fg = '—', '#888888'
        tk.Label(row, text=conf_txt, font=('Arial', 9, 'bold'),
                 fg=conf_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)

        # Preview toggle button
        prev_txt  = '◉ Previewing' if is_prev else '○ Preview'
        prev_color = '#1565c0' if is_prev else '#444444'
        def _toggle_preview(ci_=ci, pk_=pk):
            if self._preview_candidate == (ci_, pk_):
                self._preview_candidate = None
            else:
                self._preview_candidate = (ci_, pk_)
            self._refresh_right(pk_)
        tk.Button(row, text=prev_txt, font=('Arial', 8),
                  fg=prev_color, bg=row_bg, relief='flat', bd=1,
                  command=_toggle_preview).pack(side=tk.LEFT, padx=4)

        # Promote → Ck combobox + button
        if K >= 1:
            k_var = tk.StringVar(value='C1')
            k_choices = [f'C{k+1}' for k in range(K)]
            ttk.Combobox(row, textvariable=k_var, values=k_choices,
                         width=4, state='readonly').pack(side=tk.LEFT, padx=2)
            def _promote(ci_=ci, pk_=pk, kv=k_var):
                k_idx = int(kv.get()[1:]) - 1
                self._promote_candidate(pk_, ci_, k_idx)
            tk.Button(row, text='Promote', font=('Arial', 8),
                      bg='#e3f2fd', relief='raised',
                      command=_promote).pack(side=tk.LEFT, padx=2)

        # Mouse-wheel forwarding
        _wf = getattr(self, '_wheel_fn', None)
        if _wf:
            for w in [row] + list(row.winfo_children()):
                w.bind('<MouseWheel>', _wf)
                w.bind('<Button-4>',   _wf)
                w.bind('<Button-5>',   _wf)

    def _promote_candidate(self, pk, ci, target_k):
        """Swap the assignment of component target_k to candidate ci."""
        mcr         = pk['mcr_annotation']
        assigns     = mcr.get('assignments', [])
        db_profiles = mcr.get('db_profiles', [])
        cand_scores = mcr.get('cand_scores', [])
        S           = mcr.get('S')
        C           = mcr.get('C')

        if target_k >= len(assigns):
            return
        if ci >= len(db_profiles) or S is None:
            return

        # Recompute Pearson between the MCR component and the new DB profile
        s_comp  = S[:, target_k] * C.mean(axis=0)[target_k]
        db_prof = db_profiles[ci]
        if len(db_prof) == len(s_comp) and s_comp.std() > 0 and db_prof.std() > 0:
            corr = float(np.corrcoef(s_comp, db_prof)[0, 1])
        else:
            corr = 0.0

        # Pull pre-computed J / Conf from cand_scores (computed against exp_mean
        # during MCR, so still valid as a label for this candidate)
        sc       = cand_scores[ci] if ci < len(cand_scores) else {}
        shift    = sc.get('shift',   0.0)
        j_score  = sc.get('j_score', None)
        meas_j   = sc.get('meas_j',  None)
        exp_j    = sc.get('exp_j',   None)
        combined = 0.60 * max(0.0, corr) + 0.40 * (j_score or 0.5)

        assigns[target_k] = (ci, corr, shift, j_score, meas_j, exp_j, combined)

        # Clear any preview for this peak
        if self._preview_candidate and self._preview_candidate[1] is pk:
            self._preview_candidate = None

        self._refresh_right(pk)
        self._update_tree_status(pk)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _set_component(self, pk, k, cand, accepted):
        """Record accept/reject decision for component k of peak pk."""
        mcr      = pk['mcr_annotation']
        confirmed = mcr.setdefault('confirmed_assignments', {})
        assigns  = mcr.get('assignments', [])
        assign   = assigns[k] if k < len(assigns) else (0, 0.0)
        ci, corr = assign[0], assign[1]
        cname     = cand.get('name', f'Component {k+1}')
        confirmed[k] = {
            'metabolite': cname,
            'hmdb_id':    cand.get('hmdb_id') or cand.get('bmrb_id', ''),
            'corr':       corr,
            'accepted':   accepted,
        }
        if accepted:
            self._push_to_annotation(pk, k, cand, corr)
            # Keep explicit metabolite label on the peak object for Peak List.
            if int(mcr.get('K', 1)) <= 1:
                pk['assigned_metabolite'] = cname
            self._sync_peak_list_from_mcr(pk)
        # Refresh plot (dims rejected components) + table + tree
        self._refresh_right(pk)
        self._update_tree_status(pk)
        self._refresh_peak_list_window()

    def _push_to_annotation(self, pk, k, cand, corr):
        """Push an accepted MCR result into the annotation stage's annotation list."""
        ann_stage = self.ann_stage
        if ann_stage is None:
            return
        # Source can be AnnotationStage or FullSpectrumMCRStage. In the latter
        # case, annotations live in ann_stage.annotation_stage.annotations.
        target_stage = ann_stage
        if not hasattr(target_stage, 'annotations'):
            target_stage = getattr(ann_stage, 'annotation_stage', None)
        if target_stage is None or not hasattr(target_stage, 'annotations'):
            return
        cand_name = cand.get('name', f'Component {k+1}')
        entry = {
            'metabolite': cand_name,
            'hmdb_id':    cand.get('hmdb_id') or cand.get('bmrb_id', ''),
            'corr':       corr,
            'source':     'MCR',
            'ppm':        pk['ppm'],
        }
        existing = target_stage.annotations.get(pk['rank'])
        if isinstance(existing, list):
            existing.append(entry)
        else:
            target_stage.annotations[pk['rank']] = entry

    def _refresh_peak_list_window(self):
        """Refresh Peak List window if currently open."""
        try:
            plw = getattr(self.app, '_peak_list_win', None)
            if plw is not None and plw.win.winfo_exists():
                plw._populate()
        except Exception:
            pass

    def _build_mcr_subpeak(self, pk, k, cand):
        """Create a peak-list-compatible sub-peak from MCR component k."""
        mcr = pk.get('mcr_annotation', {})
        S = mcr.get('S')
        C = mcr.get('C')
        ppm_reg = mcr.get('ppm_region')
        if S is None or C is None or ppm_reg is None:
            return None
        if k >= S.shape[1] or len(ppm_reg) != S.shape[0]:
            return None

        c_mean = C.mean(axis=0)
        prof = np.clip(S[:, k] * c_mean[k], 0, None)
        if prof.max() <= 0:
            return None

        i_pk = int(np.argmax(prof))
        ppm_c = float(ppm_reg[i_pk])
        h = float(prof[i_pk])
        half = h * 0.5
        idx_half = np.where(prof >= half)[0]
        if len(idx_half) >= 2:
            w = float(abs(ppm_reg[idx_half[-1]] - ppm_reg[idx_half[0]]))
        else:
            w = float(abs(ppm_reg[min(i_pk + 1, len(ppm_reg) - 1)] - ppm_reg[max(i_pk - 1, 0)]))
        w = max(w, 1e-4)

        start_ppm = float(ppm_c + 2.0 * w)
        end_ppm = float(ppm_c - 2.0 * w)
        snr = float(pk.get('snr', 0.0))
        if h > 0 and pk.get('height'):
            try:
                snr = float(pk.get('snr', 0.0)) * (h / max(float(pk.get('height')), 1e-12))
            except Exception:
                pass

        return {
            'rank':            0,  # assigned after list rebuild
            'ppm':             ppm_c,
            'start_ppm':       start_ppm,
            'end_ppm':         end_ppm,
            'width_ppm':       w,
            'width':           w,
            'height':          h,
            'snr':             snr,
            'area':            float(np.trapz(prof, ppm_reg)),
            'overlapping':     False,
            'is_sub_peak':     True,
            'deconvolved':     False,
            'parent_ppm':      float(pk.get('ppm', ppm_c)),
            '_mcr_parent_ppm': float(pk.get('ppm', ppm_c)),
            '_mcr_parent_rank': int(pk.get('rank', 0)),
            'assigned_metabolite': cand.get('name', f'Component {k+1}'),
            'status':          pk.get('status', None),
            'mcr_component_idx': int(k),
        }

    def _sync_peak_list_from_mcr(self, pk):
        """
        Replace parent peak by currently accepted MCR components (K>=2).
        Uses in-place list mutation so Peak List references remain valid.
        """
        if self.ann_stage is None or not hasattr(self.ann_stage, 'peak_stage'):
            return
        peak_stage = self.ann_stage.peak_stage
        peak_list = getattr(peak_stage, 'peaks', None)
        if not isinstance(peak_list, list):
            return

        mcr = pk.get('mcr_annotation', {})
        K = int(mcr.get('K', 1))
        if K <= 1:
            return

        conf = mcr.get('confirmed_assignments', {})
        assigns = mcr.get('assignments', [])
        cands = mcr.get('candidates', [])
        accepted_ks = sorted([k for k, v in conf.items() if v and v.get('accepted')])
        if not accepted_ks:
            return

        sub_peaks = []
        for k in accepted_ks:
            _a = assigns[k] if k < len(assigns) else (0, 0.0)
            ci = int(_a[0]) if len(_a) > 0 else 0
            cand = cands[ci] if ci < len(cands) else {'name': f'Component {k+1}'}
            sp = self._build_mcr_subpeak(pk, k, cand)
            if sp is not None:
                sub_peaks.append(sp)
        if not sub_peaks:
            return

        parent_ppm = float(pk.get('ppm', 0.0))
        parent_rank = int(pk.get('rank', 0))
        tol = 5e-4

        def _to_remove(p):
            if p.get('is_sub_peak', False):
                if int(p.get('_mcr_parent_rank', -1)) == parent_rank:
                    return True
                if abs(float(p.get('_mcr_parent_ppm', 999.0)) - parent_ppm) < tol:
                    return True
                if abs(float(p.get('parent_ppm', 999.0)) - parent_ppm) < tol:
                    return True
                return False
            return abs(float(p.get('ppm', 999.0)) - parent_ppm) < tol

        kept = [p for p in peak_list if not _to_remove(p)]
        kept.extend(sub_peaks)
        kept.sort(key=lambda p: abs(float(p.get('ppm', 0.0))))
        for i, p in enumerate(kept, 1):
            p['rank'] = i

        peak_list.clear()
        peak_list.extend(kept)
        if hasattr(peak_stage, 'processor') and getattr(peak_stage, 'processor', None):
            peak_stage.processor.peaks = peak_list

    def _update_tree_status(self, pk):
        mcr    = pk['mcr_annotation']
        K      = mcr['K']
        conf   = mcr.get('confirmed_assignments', {})
        n_done = sum(1 for v in conf.values() if v is not None)
        st_txt = f'{n_done}/{K} ✓' if n_done else '—'
        iid = f'pk_{pk["rank"]}'
        if self.tree.exists(iid):
            vals = list(self.tree.item(iid, 'values'))
            if len(vals) >= 4:
                vals[3] = st_txt
                self.tree.item(iid, values=vals)
        # Update child rows
        assigns = mcr.get('assignments', [])
        for k in range(K):
            c_iid  = f'comp_{pk["rank"]}_{k}'
            c_conf = conf.get(k)
            if c_conf is None:
                c_st = '—'
            elif c_conf['accepted']:
                c_st = '✓ OK'
            else:
                c_st = '✗ Rejected'
            if self.tree.exists(c_iid):
                _a = assigns[k] if k < len(assigns) else (0, 0.0); ci, corr = _a[0], _a[1]
                self.tree.item(c_iid, values=(f'r={corr:.2f}', '', '', c_st))  # noqa

    def _recompute_all(self):
        """Re-run MCR with the new R² threshold on all MCR peaks."""
        r2 = self._r2_var.get()
        if self.ann_stage is None:
            return
        self.ann_stage.mcr_r2_threshold = r2
        for pk in self.mcr_peaks:
            try:
                self.ann_stage._rerun_mcr_for_peak(pk, r2)
            except Exception as e:
                print(f'Recompute MCR error (peak {pk["rank"]}): {e}')
        # Refresh tree and current view
        self._populate_tree()
        if self._selected_peak is not None:
            self._refresh_right(self._selected_peak)

    def _accept_all_simple(self):
        """Accept all components for peaks with K=1 (no deconvolution needed)."""
        for pk in self.mcr_peaks:
            mcr = pk.get('mcr_annotation', {})
            if mcr.get('K', 0) != 1:
                continue
            assigns = mcr.get('assignments', [])
            cands   = mcr.get('candidates', [])
            for k in range(1):
                _a = assigns[k] if k < len(assigns) else (0, 0.0); ci, corr = _a[0], _a[1]
                cand = cands[ci] if ci < len(cands) else {}
                self._set_component(pk, k, cand, accepted=True)
        self._populate_tree()

    def _accept_all_overlap(self):
        """Accept all components for peaks with K>1 (overlapping regions)."""
        for pk in self.mcr_peaks:
            mcr = pk.get('mcr_annotation', {})
            K   = mcr.get('K', 0)
            if K <= 1:
                continue
            assigns = mcr.get('assignments', [])
            cands   = mcr.get('candidates', [])
            for k in range(K):
                _a = assigns[k] if k < len(assigns) else (0, 0.0); ci, corr = _a[0], _a[1]
                cand = cands[ci] if ci < len(cands) else {}
                self._set_component(pk, k, cand, accepted=True)
        self._populate_tree()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _center(self):
        self.win.update_idletasks()
        w  = self.win.winfo_width()
        h  = self.win.winfo_height()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')
