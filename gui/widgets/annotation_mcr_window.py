# gui/widgets/annotation_mcr_window.py
"""
MCR Deconvolution for Annotation Undef Peaks.

Opened from AnnotationResultsWindow via "Deconvolve Undef".

Left panel : list of peaks marked as Undef in annotation
              shows Peak #, Metabolite, ppm, Score, current Quality
Right panel : matplotlib figure (experimental + MCR components + fit + residual)
              "▶ Deconvolute" button — runs MCR on the selected peak's region
              Component table with Accept / Reject

Accepting a component matching the annotated metabolite promotes the peak
quality to 'good' in the annotation results (Ctrl+1 equivalent).
Rejecting sets it to 'bad' (Ctrl+3 equivalent).
"""
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

COMP_COLORS = ['#1565c0', '#e65100', '#2e7d32', '#6a1b9a', '#880e4f']


class AnnotationMCRWindow:
    """
    Parameters
    ----------
    parent      : tk widget
    undef_items : list of (met_dict, mp_dict, peak_dict)
                  where met_dict  = metabolite group from AnnotationResultsWindow
                        mp_dict   = matched-peak entry within that metabolite
                        peak_dict = peak from PeakDetectionStage.peaks
    app         : main App instance (for ppm / all_intensities)
    ann_stage   : AnnotationStage (for db_path, settings)
    ann_results : AnnotationResultsWindow instance (optional — to refresh quality)
    """

    def __init__(self, parent, undef_items, app,
                 ann_stage=None, ann_results=None):
        self.undef_items = undef_items      # [(met, mp, peak), …]
        self.app         = app
        self.ann_stage   = ann_stage
        self.ann_results = ann_results

        self._selected_idx  = None
        self._mcr_running   = False
        self._preview_candidate = None

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"MCR Deconvolution — {len(undef_items)} Undef peak(s)")
        self.win.geometry("1450x850")
        self.win.minsize(950, 600)
        self._center()
        self._build_ui()
        if undef_items:
            self._select_item(0)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _center(self):
        self.win.update_idletasks()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        w, h = 1450, 850
        self.win.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=6)
        main.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            main,
            text=(f"  {len(self.undef_items)} Undef peak(s)  "
                  "— select a peak, then click Deconvolute"),
            font=("Arial", 9, "italic"), foreground="#444"
        ).pack(anchor=tk.W, pady=(0, 4))

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=2)
        paned.add(left, weight=1)
        self._build_left(left)

        right = ttk.Frame(paned, padding=2)
        paned.add(right, weight=3)
        self._build_right(right)

    # ── Left panel — Undef peak list ─────────────────────────────────────────

    def _build_left(self, parent):
        ttk.Label(parent, text="Annotation Undef Peaks",
                  font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))

        cols = ('met', 'ppm', 'score', 'status')
        self.tree = ttk.Treeview(parent, columns=cols,
                                  show='tree headings', selectmode='browse')
        self.tree.heading('#0',     text='Peak #')
        self.tree.heading('met',    text='Metabolite')
        self.tree.heading('ppm',    text='ppm')
        self.tree.heading('score',  text='Score')
        self.tree.heading('status', text='Status')

        self.tree.column('#0',     width=55,  stretch=False)
        self.tree.column('met',    width=160, stretch=True)
        self.tree.column('ppm',    width=72,  stretch=False)
        self.tree.column('score',  width=52,  stretch=False)
        self.tree.column('status', width=82,  stretch=False)

        self.tree.tag_configure('done_good', foreground='#1a7a1a',
                                background='#e8f5e9')
        self.tree.tag_configure('done_bad',  foreground='#9a0000',
                                background='#ffebee')
        self.tree.tag_configure('mcr_done',  foreground='#1565c0',
                                background='#e3f2fd')
        self.tree.tag_configure('pending',   foreground='#555555')

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        self._populate_tree()

    def _populate_tree(self):
        sel = self.tree.focus()
        self.tree.delete(*self.tree.get_children())
        for i, (met, mp, peak) in enumerate(self.undef_items):
            rank  = peak.get('rank', i + 1)
            ppm   = float(peak.get('ppm', mp.get('peak_ppm', 0)))
            score = float(mp.get('composite_score', 0))
            name  = met.get('name', '?')[:28]
            qual  = mp.get('annotation_quality')
            mcr   = peak.get('mcr_annotation')

            if qual == 'good':
                st, tag = '✓ Good', 'done_good'
            elif qual == 'bad':
                st, tag = '✗ Bad', 'done_bad'
            elif mcr:
                r2 = mcr.get('metrics', {}).get('r2', float('nan'))
                st  = f'R²={r2:.2f}' if not np.isnan(r2) else 'MCR done'
                tag = 'mcr_done'
            else:
                st, tag = '—', 'pending'

            self.tree.insert('', tk.END, iid=str(i),
                             text=f'#{rank}',
                             values=(name, f'{ppm:.4f}',
                                     f'{score:.2f}', st),
                             tags=(tag,))
        # Restore selection if possible
        if sel and self.tree.exists(sel):
            self.tree.selection_set(sel)

    def _on_tree_select(self, _event=None):
        sel = self.tree.focus()
        if not sel:
            return
        try:
            idx = int(sel)
        except ValueError:
            return
        if idx != self._selected_idx:
            self._selected_idx = idx
            self._preview_candidate = None
            self._refresh_right()

    def _select_item(self, idx):
        self._selected_idx = idx
        iid = str(idx)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)
        self._refresh_right()

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self, parent):
        # Toolbar row
        tb = ttk.Frame(parent)
        tb.pack(fill=tk.X, pady=(0, 4))

        self._btn_deconv = ttk.Button(tb, text='▶  Deconvolute',
                                       command=self._run_mcr)
        self._btn_deconv.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(tb, text='R² min:', font=("Arial", 9)).pack(side=tk.LEFT)
        self._r2_var = tk.DoubleVar(
            value=getattr(self.ann_stage, 'mcr_r2_threshold', 0.90))
        ttk.Spinbox(tb, from_=0.50, to=1.00, increment=0.01,
                    textvariable=self._r2_var, width=5,
                    format='%.2f').pack(side=tk.LEFT, padx=(2, 12))

        self._status_lbl = ttk.Label(
            tb, text='Select a peak and click Deconvolute.',
            foreground='gray', font=("Arial", 8))
        self._status_lbl.pack(side=tk.LEFT)

        # Matplotlib figure
        self._fig = Figure(figsize=(9, 6), dpi=95, facecolor='#fafafa',
                           layout='constrained')
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._nav = NavigationToolbar2Tk(self._canvas, parent, pack_toolbar=False)
        self._nav.update()
        self._nav.pack(fill=tk.X, side=tk.BOTTOM)

        # Component table (scrollable)
        lf = ttk.LabelFrame(parent, text='Components', padding=4)
        lf.pack(fill=tk.X, pady=(4, 0))
        _vsb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._comp_canvas = tk.Canvas(lf, bd=0, highlightthickness=0,
                                      height=130, yscrollcommand=_vsb.set)
        _vsb.configure(command=self._comp_canvas.yview)
        self._comp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._comp_frame = tk.Frame(self._comp_canvas, bg='white')
        self._comp_cwin  = self._comp_canvas.create_window(
            (0, 0), window=self._comp_frame, anchor='nw')
        self._comp_frame.bind(
            '<Configure>',
            lambda e: self._comp_canvas.configure(
                scrollregion=self._comp_canvas.bbox('all')))
        self._comp_canvas.bind(
            '<Configure>',
            lambda e: self._comp_canvas.itemconfig(
                self._comp_cwin, width=e.width))
        self._comp_rows = []

    # ── MCR computation ───────────────────────────────────────────────────────

    def _run_mcr(self):
        if self._selected_idx is None or self._mcr_running:
            return
        self._mcr_running = True
        self._btn_deconv.config(state='disabled')
        self._set_status('Running MCR…', 'blue')
        met, mp, peak = self.undef_items[self._selected_idx]
        idx = self._selected_idx
        threading.Thread(
            target=self._mcr_thread,
            args=(idx, met, mp, peak),
            daemon=True).start()

    def _mcr_thread(self, idx, met, mp, peak):
        try:
            mcr_ann = self._compute_mcr(met, mp, peak)
            self.win.after(0, lambda: self._on_mcr_done(idx, peak, mcr_ann))
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.win.after(0, lambda: self._on_mcr_error(str(exc)))

    def _compute_mcr(self, met, mp, peak):
        """Run MCR-ALS on the peak's region, using annotation DB candidates."""
        from gui.utils.mcr_utils import (mcr_annotation_region,
                                          simulate_lorentzian_profile)
        from gui.widgets.annotation_results_window import _build_stick_positions

        ppm_arr = getattr(self.app, 'ppm', None)
        all_int = getattr(self.app, 'all_intensities', None)
        if ppm_arr is None or all_int is None:
            raise RuntimeError('No spectrum data loaded.')

        r2_thr    = float(self._r2_var.get())
        shift_tol = 0.02
        mhz       = 600

        # ── Region bounds ────────────────────────────────────────────────────
        lo = float(min(peak.get('start_ppm', peak['ppm']),
                       peak.get('end_ppm',   peak['ppm'])))
        hi = float(max(peak.get('start_ppm', peak['ppm']),
                       peak.get('end_ppm',   peak['ppm'])))
        margin = max(0.006, (hi - lo) * 0.25)
        lo = max(float(ppm_arr.min()), lo - margin)
        hi = min(float(ppm_arr.max()), hi + margin)

        mask = (ppm_arr >= min(lo, hi)) & (ppm_arr <= max(lo, hi))
        if mask.sum() < 4:
            raise RuntimeError(
                f'Too few spectrum points in [{lo:.3f}, {hi:.3f}] ppm.')

        ppm_reg = ppm_arr[mask]
        X_reg   = (all_int[mask, :] if all_int.ndim == 2
                   else all_int[mask, np.newaxis])

        # ── Candidate DB entries ─────────────────────────────────────────────
        # Collect ALL annotation matches that land in this region — these
        # become MCR's S_init candidates, revealing possible overlapping signals.
        raw_results = []
        ann = self.ann_stage
        if ann and hasattr(ann, '_last_results_args'):
            raw_results = (ann._last_results_args or {}).get('results', [])

        region_entries = []
        for r in raw_results:
            rp = float(r.get('ppm_center', r.get('ppm', 0)))
            if min(lo, hi) - 0.01 <= rp <= max(lo, hi) + 0.01:
                region_entries.append(r)

        # Always include the primary annotation match
        if not region_entries:
            region_entries = [{
                'ppm_center': float(mp.get('db_ppm', peak.get('ppm', 0))),
                'type':       mp.get('type', 's'),
                'protons':    mp.get('protons', 1),
                'j_coupling': mp.get('j_coupling', ''),
                'name':       met.get('name', '?'),
                'hmdb_id':    met.get('id', ''),
            }]

        # Deduplicate by (hmdb_id, rounded ppm)
        seen: dict = {}
        for r in region_entries:
            key = (r.get('hmdb_id') or r.get('bmrb_id', ''),
                   round(float(r.get('ppm_center', 0)), 3))
            if key not in seen:
                seen[key] = r
        uniq = list(seen.values())

        # ── Build simulated DB profiles ──────────────────────────────────────
        db_profiles = []
        for e in uniq:
            try:
                dp_ppm   = float(e.get('ppm_center', e.get('ppm', peak['ppm'])))
                mult     = e.get('type') or e.get('multiplicity') or 's'
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
            raise RuntimeError('Could not build any DB profiles for this region.')

        # ── Run MCR ──────────────────────────────────────────────────────────
        S, C, mcr_m, K, assignments, aligned_profiles, cand_scores = \
            mcr_annotation_region(
                X_reg, db_profiles,
                r2_threshold=r2_thr,
                ppm_arr=ppm_reg,
                shift_tol_ppm=shift_tol,
                use_baseline=True,
                db_entries=uniq,
                field_mhz=mhz,
                max_k_override=min(len(db_profiles), 5))

        return {
            'S':              S,
            'C':              C,
            'K':              K,
            'metrics':        mcr_m,
            'assignments':    assignments,
            'candidates':     uniq,
            'db_profiles':    aligned_profiles,
            'db_profiles_raw': db_profiles,
            'cand_scores':    cand_scores,
            'ppm_region':     ppm_reg,
            'r2_threshold':   r2_thr,
            'confirmed_assignments': {},
        }

    def _on_mcr_done(self, idx, peak, mcr_ann):
        self._mcr_running = False
        self._btn_deconv.config(state='normal')
        peak['mcr_annotation'] = mcr_ann
        r2 = mcr_ann.get('metrics', {}).get('r2', float('nan'))
        K  = mcr_ann.get('K', '?')
        self._set_status(
            f'Done — K={K}  R²={"—" if np.isnan(r2) else f"{r2:.3f}"}',
            '#1a7a1a')
        self._populate_tree()
        if self._selected_idx == idx:
            self._draw_plot(peak)
            met, mp, _ = self.undef_items[idx]
            self._draw_comp_table(peak, met, mp)

    def _on_mcr_error(self, msg):
        self._mcr_running = False
        self._btn_deconv.config(state='normal')
        self._set_status(f'Error: {msg}', 'red')

    def _set_status(self, msg, color='gray'):
        self._status_lbl.config(text=msg, foreground=color)

    # ── Right panel refresh ───────────────────────────────────────────────────

    def _refresh_right(self):
        if self._selected_idx is None:
            return
        met, mp, peak = self.undef_items[self._selected_idx]
        if peak.get('mcr_annotation'):
            self._draw_plot(peak)
            self._draw_comp_table(peak, met, mp)
        else:
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    f'Peak #{peak.get("rank","?")}  '
                    f'{peak.get("ppm", 0):.4f} ppm\n'
                    f'Metabolite: {met.get("name","?")}  '
                    f'(score={mp.get("composite_score",0):.2f})\n\n'
                    'Click "Deconvolute" to run MCR on this region.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#555')
            ax.set_axis_off()
            self._canvas.draw_idle()
            self._clear_comp_table()

    # ── Plot ─────────────────────────────────────────────────────────────────

    def _draw_plot(self, peak):
        """Plot MCR results: experimental background + raw model components + fit + residual."""
        if self._canvas is None:
            return
        mcr = peak.get('mcr_annotation')
        if not mcr:
            return

        self._fig.clear()
        K       = min(mcr['K'], mcr['S'].shape[1])
        S       = mcr['S']
        C       = mcr['C']
        ppm_reg = mcr['ppm_region']
        c_mean  = C.mean(axis=0)[:K]
        assigns = mcr.get('assignments', [])
        cands   = mcr.get('candidates', [])
        r2      = mcr.get('metrics', {}).get('r2', float('nan'))
        conf    = mcr.get('confirmed_assignments', {})

        ppm_full = getattr(self.app, 'ppm', None)
        all_int  = getattr(self.app, 'all_intensities', None)
        int_full = (np.mean(all_int, axis=1) if all_int is not None and all_int.ndim == 2
                    else (all_int.copy() if all_int is not None else None))

        reg_lo = float(min(ppm_reg)); reg_hi = float(max(ppm_reg))
        reg_w  = reg_hi - reg_lo

        # xlim tight around the MCR region — small padding so edge peaks
        # are not flush against the axis boundary.
        ctx_margin = max(0.003, reg_w * 0.05)
        ctx_lo = reg_lo - ctx_margin
        ctx_hi = reg_hi + ctx_margin

        exp_ctx_ppm = exp_ctx_int = None
        exp_reg_ppm = exp_reg_int = None
        e_max = 1.0

        if ppm_full is not None and int_full is not None:
            ctx_mask = ((ppm_full >= min(ctx_lo, ctx_hi)) &
                        (ppm_full <= max(ctx_lo, ctx_hi)))
            if ctx_mask.sum() > 1:
                exp_ctx_ppm = ppm_full[ctx_mask]
                exp_ctx_int = int_full[ctx_mask]
                if exp_ctx_ppm[0] > exp_ctx_ppm[-1]:
                    exp_ctx_ppm = exp_ctx_ppm[::-1]
                    exp_ctx_int = exp_ctx_int[::-1]
            reg_mask = ((ppm_full >= min(reg_lo, reg_hi)) &
                        (ppm_full <= max(reg_lo, reg_hi)))
            if reg_mask.sum() > 1:
                exp_reg_ppm = ppm_full[reg_mask]
                exp_reg_int = int_full[reg_mask]
                e_max = float(exp_reg_int.max()) if exp_reg_int.max() > 0 else 1.0

        raw_fit   = sum(S[:, k] * c_mean[k] for k in range(K))
        fit_peak  = raw_fit.max() if raw_fit.max() > 0 else 1.0
        amp_scale = e_max / fit_peak

        # Build components: sk is the exact MCR output (S[:,k]*C_mean[k]*amp_scale)
        components = []
        total_area = sum((S[:, k] * c_mean[k]).sum() for k in range(K)) + 1e-12
        for k in range(K):
            sk     = S[:, k] * c_mean[k] * amp_scale
            _a     = assigns[k] if k < len(assigns) else (0, 0.0)
            ci, corr = int(_a[0]), float(_a[1])
            cand   = cands[ci] if ci < len(cands) else {}
            lbl    = cand.get('name', f'C{k+1}')[:24]
            color  = COMP_COLORS[k % len(COMP_COLORS)]
            c_conf = conf.get(k)
            accepted = c_conf is None or c_conf.get('accepted', True)
            alpha  = 0.88 if accepted else 0.28
            lw_c   = 2.0  if accepted else 0.8
            pct    = 100.0 * (S[:, k] * c_mean[k]).sum() / total_area
            components.append(
                (k, sk, f'C{k+1}: {lbl}  r={corr:.2f}  ({pct:.0f}%)',
                 color, alpha, lw_c, ci))

        sort_idx   = np.argsort(ppm_reg)
        ppm_sorted = ppm_reg[sort_idx]

        def _sort(v):
            return v[sort_idx]

        # ── Subplots ──────────────────────────────────────────────────────────
        ax1 = self._fig.add_subplot(111)

        # 1. Experimental background — context window
        if exp_ctx_ppm is not None and exp_ctx_int is not None:
            ax1.fill_between(exp_ctx_ppm, exp_ctx_int,
                             alpha=0.13, color='#444444', zorder=1)
            ax1.plot(exp_ctx_ppm, exp_ctx_int,
                     color='#333333', lw=1.1, alpha=0.75,
                     label='Experimental', zorder=2)

        # 2. DB fingerprint profiles — re-simulated on the full ctx window
        #    so Lorentzian tails are not truncated at ppm_reg boundaries.
        try:
            from gui.utils.mcr_utils import simulate_lorentzian_profile
            from gui.widgets.annotation_results_window import _build_stick_positions
            _can_resim = (exp_ctx_ppm is not None and len(exp_ctx_ppm) > 1)
        except Exception:
            _can_resim = False

        db_profiles_list = mcr.get('db_profiles', [])
        drawn_db: set = set()
        for k, sk, lbl, color, alpha, lw_c, ci in components:
            if ci in drawn_db or ci >= len(cands):
                continue
            cand_c   = cands[ci]
            sk_sorted = np.clip(_sort(sk), 0, None)
            comp_peak = float(sk_sorted.max()) if sk_sorted.max() > 0 else e_max

            if _can_resim:
                try:
                    dp_ppm  = float(cand_c.get('ppm_center',
                                               cand_c.get('ppm', peak.get('ppm', 0))))
                    mult    = cand_c.get('type') or cand_c.get('multiplicity') or 's'
                    nh      = int(cand_c.get('protons') or 1)
                    sticks  = _build_stick_positions(
                        dp_ppm, mult, nh, cand_c.get('j_coupling'), field_mhz=600)
                    if not sticks:
                        sticks = [(dp_ppm, 1.0)]
                    db_ctx  = simulate_lorentzian_profile(sticks, exp_ctx_ppm)
                    db_pk   = float(db_ctx.max()) if db_ctx.max() > 0 else 1.0
                    ax1.plot(exp_ctx_ppm, np.clip(db_ctx, 0, None) * comp_peak / db_pk,
                             color=color, lw=1.5, ls=':', alpha=0.75, zorder=3,
                             label=f'DB: {lbl.split(":")[1].strip().split()[0]}')
                    drawn_db.add(ci)
                    continue
                except Exception:
                    pass

            # Fallback: use stored profile on ppm_reg
            if ci < len(db_profiles_list):
                prof = db_profiles_list[ci]
                if len(prof) == len(ppm_reg):
                    db_sorted = np.clip(_sort(np.asarray(prof)), 0, None)
                    db_peak   = float(db_sorted.max()) if db_sorted.max() > 0 else 1.0
                    ax1.plot(ppm_sorted, db_sorted * comp_peak / db_peak,
                             color=color, lw=1.5, ls=':', alpha=0.75, zorder=3,
                             label=f'DB: {lbl.split(":")[1].strip().split()[0]}')
            drawn_db.add(ci)

        # Preview candidate overlay — re-simulated on ctx if possible
        if self._preview_candidate is not None:
            prev_ci, prev_peak = self._preview_candidate
            if prev_peak is peak and prev_ci < len(cands):
                pname = cands[prev_ci].get('name', '?')
                cand_p = cands[prev_ci]
                plotted = False
                if _can_resim:
                    try:
                        dp_ppm = float(cand_p.get('ppm_center',
                                                   cand_p.get('ppm', peak.get('ppm', 0))))
                        sticks = _build_stick_positions(
                            dp_ppm,
                            cand_p.get('type') or 's',
                            int(cand_p.get('protons') or 1),
                            cand_p.get('j_coupling'), field_mhz=600)
                        if not sticks:
                            sticks = [(dp_ppm, 1.0)]
                        pr_ctx = simulate_lorentzian_profile(sticks, exp_ctx_ppm)
                        pr_pk  = float(pr_ctx.max()) if pr_ctx.max() > 0 else 1.0
                        ax1.plot(exp_ctx_ppm,
                                 np.clip(pr_ctx, 0, None) * e_max / pr_pk,
                                 color='#555555', lw=1.4, ls='--', alpha=0.80,
                                 label=f'Preview: {pname}', zorder=4)
                        plotted = True
                    except Exception:
                        pass
                if not plotted and prev_ci < len(db_profiles_list):
                    prev_prof = db_profiles_list[prev_ci]
                    if len(prev_prof) == len(ppm_reg):
                        prev_sorted = np.clip(_sort(np.asarray(prev_prof)), 0, None)
                        prev_pk = float(prev_sorted.max()) if prev_sorted.max() > 0 else 1.0
                        ax1.plot(ppm_sorted, prev_sorted * e_max / prev_pk,
                                 color='#555555', lw=1.4, ls='--', alpha=0.80,
                                 label=f'Preview: {pname}', zorder=4)

        # 3. Components — MCR output on ppm_sorted
        for k, sk, lbl, color, alpha, lw_c, ci in components:
            sk_sorted = np.clip(_sort(sk), 0, None)
            ax1.fill_between(ppm_sorted, sk_sorted,
                             alpha=max(0.04, alpha * 0.22),
                             color=color, zorder=2)
            ax1.plot(ppm_sorted, sk_sorted,
                     color=color, lw=lw_c, alpha=alpha,
                     label=lbl, zorder=4)

        # 4. MCR total fit — sum of components on ppm_sorted only
        fit_in_reg = np.clip(
            sum(_sort(sk) for k, sk, *_ in components), 0, None)
        ax1.plot(ppm_sorted, fit_in_reg,
                 color='#c62828', lw=2.0, ls='--', alpha=0.90, zorder=5,
                 label=f'MCR fit  R²={"—" if np.isnan(r2) else f"{r2:.3f}"}')

        ax1.tick_params(labelsize=7)
        ax1.set_ylabel('Intensity', fontsize=7)
        ax1.set_xlabel('ppm', fontsize=8)
        ax1.set_title(
            f'Peak #{peak.get("rank","?")}  '
            f'{peak.get("ppm", 0):.4f} ppm  K={K}  '
            f'R²={"—" if np.isnan(r2) else f"{r2:.3f}"}',
            fontsize=8, pad=3)
        ax1.legend(fontsize=6.5, loc='upper left',
                   framealpha=0.85, ncol=min(2, K + 3))
        ax1.set_facecolor('#fafafa')
        ax1.set_xlim(max(ctx_lo, ctx_hi), min(ctx_lo, ctx_hi))  # NMR: high ppm left

        # Y-axis: zoom in on the peak variation, not the full 0→max range.
        # Use the experimental range within ctx as reference.
        if exp_ctx_int is not None and len(exp_ctx_int) > 0:
            y_base = float(exp_ctx_int.min())
            y_top  = float(exp_ctx_int.max())
        else:
            comp_peaks = [float(np.max(np.clip(_sort(sk), 0, None)))
                          for k, sk, *_ in components]
            y_base = 0.0
            y_top  = max(e_max, max(comp_peaks) if comp_peaks else 1.0)
        y_span = max(y_top - y_base, 1.0)
        ax1.set_ylim(bottom=y_base - y_span * 0.08, top=y_top + y_span * 0.15)

        self._canvas.draw_idle()

    # ── Component table ───────────────────────────────────────────────────────

    def _clear_comp_table(self):
        for w in self._comp_rows:
            try: w.destroy()
            except Exception: pass
        self._comp_rows = []

    def _draw_comp_table(self, peak, met, mp):
        self._clear_comp_table()
        mcr = peak.get('mcr_annotation')
        if not mcr:
            return

        K       = mcr['K']
        assigns = mcr.get('assignments', [])
        cands   = mcr.get('candidates', [])
        conf    = mcr.get('confirmed_assignments', {})
        parent  = self._comp_frame

        # Primary metabolite (to mark matching component with ★)
        primary_name = met.get('name', '').lower()

        # ── Header ───────────────────────────────────────────────────────────
        hdr_bg = '#e8eaf6'
        hdr = tk.Frame(parent, bg=hdr_bg)
        hdr.pack(fill=tk.X, pady=(0, 1))
        self._comp_rows.append(hdr)
        for col_txt, col_w in [
            ('#', 3), ('Metabolite', 22), ('Corr', 6), ('Δppm', 6),
            ('J', 5), ('Conf', 6), ('Status', 9), ('Action', 10),
        ]:
            tk.Label(hdr, text=col_txt, font=('Arial', 8, 'bold'),
                     bg=hdr_bg, width=col_w, anchor='w',
                     padx=4).pack(side=tk.LEFT)

        sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=1)
        self._comp_rows.append(sep)

        # ── One row per MCR component ─────────────────────────────────────────
        for k, assign in enumerate(assigns):
            ci        = assign[0]
            corr      = assign[1]
            shift_ppm = assign[2] if len(assign) > 2 else 0.0
            j_score   = assign[3] if len(assign) > 3 else None
            meas_j    = assign[4] if len(assign) > 4 else None
            exp_j     = assign[5] if len(assign) > 5 else None
            combined  = assign[6] if len(assign) > 6 else None
            cand      = cands[ci] if ci < len(cands) else {}
            name      = cand.get('name', f'Component {k+1}')
            color     = COMP_COLORS[k % len(COMP_COLORS)]
            c_conf    = conf.get(k)

            is_primary = bool(primary_name and primary_name in name.lower())

            if c_conf is None:
                row_bg = '#fff8e1' if is_primary else 'white'
                st_txt, st_fg = '—', '#888888'
            elif c_conf.get('accepted'):
                row_bg, st_txt, st_fg = '#e8f5e9', '✓ Accepted', '#1a7a1a'
            else:
                row_bg, st_txt, st_fg = '#ffebee', '✗ Rejected', '#9a0000'

            row = tk.Frame(parent, bg=row_bg, bd=0)
            row.pack(fill=tk.X, pady=1)
            self._comp_rows.append(row)

            # C# label
            tk.Label(row, text=f'C{k+1}', font=('Arial', 9, 'bold'),
                     fg=color, bg=row_bg, width=3, anchor='w',
                     padx=4).pack(side=tk.LEFT)
            # Name (★ if matches annotation metabolite)
            lbl_name = name + (' ★' if is_primary else '')
            tk.Label(row, text=lbl_name,
                     font=('Arial', 9, 'bold' if is_primary else 'normal'),
                     bg=row_bg, width=26, anchor='w',
                     padx=2).pack(side=tk.LEFT)
            # Correlation
            corr_fg = ('#1a7a1a' if corr >= 0.85 else
                       '#856404' if corr >= 0.60 else '#9a0000')
            tk.Label(row, text=f'{corr:.3f}', font=('Arial', 9),
                     fg=corr_fg, bg=row_bg, width=6,
                     anchor='w').pack(side=tk.LEFT)
            # Δppm
            shift_txt = f'{shift_ppm:+.3f}' if abs(shift_ppm) > 1e-4 else '0'
            shift_fg  = '#c62828' if abs(shift_ppm) > 0.015 else '#444444'
            tk.Label(row, text=shift_txt, font=('Arial', 9),
                     fg=shift_fg, bg=row_bg, width=6,
                     anchor='w').pack(side=tk.LEFT)
            # J score
            if j_score is None:
                j_txt, j_fg = '—', '#888888'
            else:
                mult = (cand.get('type') or cand.get('multiplicity') or 's').lower()
                if mult == 's':
                    j_txt, j_fg = 's', '#888888'
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
                     fg=conf_fg, bg=row_bg, width=6,
                     anchor='w').pack(side=tk.LEFT)
            # Status
            tk.Label(row, text=st_txt, font=('Arial', 9, 'bold'),
                     fg=st_fg, bg=row_bg, width=10,
                     anchor='w').pack(side=tk.LEFT)

            # Accept / Reject
            def _acc(c=cand, k_=k, p=peak, is_p=is_primary, mp_=mp):
                self._set_component(p, k_, c, accepted=True,
                                    is_primary=is_p, mp=mp_)
            def _rej(c=cand, k_=k, p=peak, is_p=is_primary, mp_=mp):
                self._set_component(p, k_, c, accepted=False,
                                    is_primary=is_p, mp=mp_)

            acc_bg  = '#c8e6c9' if (c_conf and c_conf.get('accepted'))    else '#f5f5f5'
            rej_bg  = '#ffcdd2' if (c_conf and not c_conf.get('accepted')) else '#f5f5f5'
            acc_rel = 'sunken' if (c_conf and c_conf.get('accepted'))     else 'raised'
            rej_rel = 'sunken' if (c_conf and not c_conf.get('accepted')) else 'raised'
            tk.Button(row, text='✓ Accept', font=('Arial', 8),
                      bg=acc_bg, relief=acc_rel,
                      command=_acc).pack(side=tk.LEFT, padx=3)
            tk.Button(row, text='✗ Reject', font=('Arial', 8),
                      bg=rej_bg, relief=rej_rel,
                      command=_rej).pack(side=tk.LEFT, padx=1)

            # Quality shortcut for the primary metabolite component
            if is_primary:
                def _mark_good(mp_=mp):
                    mp_['annotation_quality'] = 'good'
                    self._sync_ann_quality(mp_, 'good')
                    self._populate_tree()
                def _mark_bad(mp_=mp):
                    mp_['annotation_quality'] = 'bad'
                    self._sync_ann_quality(mp_, 'bad')
                    self._populate_tree()
                tk.Label(row, text='|', bg=row_bg,
                         fg='#ccc').pack(side=tk.LEFT, padx=2)
                tk.Button(row, text='✓ Good', font=('Arial', 8),
                          bg='#c8e6c9', relief='raised',
                          command=_mark_good).pack(side=tk.LEFT, padx=2)
                tk.Button(row, text='✗ Bad', font=('Arial', 8),
                          bg='#ffcdd2', relief='raised',
                          command=_mark_bad).pack(side=tk.LEFT, padx=1)

            self._bind_wheel(row)

        # ── Other candidates (not assigned by MCR) ────────────────────────────
        assigned_cis = {a[0] for a in assigns}
        other_cands  = [(ci, cands[ci]) for ci in range(len(cands))
                        if ci not in assigned_cis]
        if other_cands:
            sep2 = ttk.Separator(parent, orient=tk.HORIZONTAL)
            sep2.pack(fill=tk.X, pady=(6, 2))
            self._comp_rows.append(sep2)
            sec_lbl = tk.Label(parent,
                               text='Other candidates (not assigned by MCR)',
                               font=('Arial', 8, 'italic'),
                               fg='#555555', bg='white', anchor='w', padx=6)
            sec_lbl.pack(fill=tk.X)
            self._comp_rows.append(sec_lbl)
            for ci, cand in other_cands:
                self._draw_other_cand_row(parent, peak, ci, cand)

        self._comp_canvas.after_idle(
            lambda: self._comp_canvas.yview_moveto(0))

    def _draw_other_cand_row(self, parent, peak, ci, cand):
        """Other-candidates section — same layout as MCRDeconvWindow."""
        mcr         = peak['mcr_annotation']
        K           = mcr['K']
        assigns     = mcr.get('assignments', [])
        cand_scores = mcr.get('cand_scores', [])
        sc          = cand_scores[ci] if ci < len(cand_scores) else {}

        corr      = sc.get('corr',    0.0)
        shift_ppm = sc.get('shift',   0.0)
        j_score   = sc.get('j_score', None)
        meas_j    = sc.get('meas_j',  None)
        exp_j     = sc.get('exp_j',   None)
        combined  = sc.get('combined', None)

        is_prev = (self._preview_candidate is not None
                   and self._preview_candidate == (ci, peak))
        row_bg = '#f0f4ff' if is_prev else '#fafafa'

        row = tk.Frame(parent, bg=row_bg, bd=0)
        row.pack(fill=tk.X, pady=1)
        self._comp_rows.append(row)

        name = cand.get('name', f'Cand {ci}')
        mult = (cand.get('type') or cand.get('multiplicity') or 's').lower()

        tk.Label(row, text=f'  {name}', font=('Arial', 9),
                 bg=row_bg, width=26, anchor='w').pack(side=tk.LEFT)
        corr_fg = ('#1a7a1a' if corr >= 0.85 else
                   '#856404' if corr >= 0.60 else '#9a0000')
        tk.Label(row, text=f'{corr:.3f}', font=('Arial', 9),
                 fg=corr_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)
        shift_txt = f'{shift_ppm:+.3f}' if abs(shift_ppm) > 1e-4 else '0'
        shift_fg  = '#c62828' if abs(shift_ppm) > 0.015 else '#444444'
        tk.Label(row, text=shift_txt, font=('Arial', 9),
                 fg=shift_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)
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
        if combined is not None:
            conf_txt = f'{combined:.2f}'
            conf_fg  = ('#1a7a1a' if combined >= 0.75 else
                        '#856404' if combined >= 0.50 else '#9a0000')
        else:
            conf_txt, conf_fg = '—', '#888888'
        tk.Label(row, text=conf_txt, font=('Arial', 9, 'bold'),
                 fg=conf_fg, bg=row_bg, width=6, anchor='w').pack(side=tk.LEFT)

        # Preview toggle
        prev_txt   = '◉ Previewing' if is_prev else '○ Preview'
        prev_color = '#1565c0' if is_prev else '#444444'
        def _toggle(ci_=ci, pk_=peak):
            self._preview_candidate = (
                None if self._preview_candidate == (ci_, pk_) else (ci_, pk_))
            self._draw_plot(pk_)
            if self._selected_idx is not None:
                met2, mp2, _ = self.undef_items[self._selected_idx]
                self._draw_comp_table(pk_, met2, mp2)
        tk.Button(row, text=prev_txt, font=('Arial', 8),
                  fg=prev_color, bg=row_bg, relief='flat', bd=1,
                  command=_toggle).pack(side=tk.LEFT, padx=4)

        # Promote → Ck
        if K >= 1:
            k_var     = tk.StringVar(value='C1')
            k_choices = [f'C{k+1}' for k in range(K)]
            ttk.Combobox(row, textvariable=k_var, values=k_choices,
                         width=4, state='readonly').pack(side=tk.LEFT, padx=2)
            def _promote(ci_=ci, pk_=peak, kv=k_var):
                self._promote_candidate(pk_, ci_, int(kv.get()[1:]) - 1)
            tk.Button(row, text='Promote', font=('Arial', 8),
                      bg='#e3f2fd', relief='raised',
                      command=_promote).pack(side=tk.LEFT, padx=2)

        self._bind_wheel(row)

    def _promote_candidate(self, peak, ci, target_k):
        """Swap assignment of component target_k to candidate ci."""
        mcr         = peak['mcr_annotation']
        assigns     = mcr.get('assignments', [])
        db_profiles = mcr.get('db_profiles', [])
        cands       = mcr.get('candidates', [])
        cand_scores = mcr.get('cand_scores', [])
        S = mcr.get('S'); C = mcr.get('C')

        if target_k >= len(assigns) or ci >= len(db_profiles) or S is None:
            return

        s_comp  = S[:, target_k] * C.mean(axis=0)[target_k]
        db_prof = db_profiles[ci]
        if (len(db_prof) == len(s_comp)
                and s_comp.std() > 0 and db_prof.std() > 0):
            corr = float(np.corrcoef(s_comp, db_prof)[0, 1])
        else:
            corr = 0.0

        sc       = cand_scores[ci] if ci < len(cand_scores) else {}
        shift    = sc.get('shift',   0.0)
        j_score  = sc.get('j_score', None)
        meas_j   = sc.get('meas_j',  None)
        exp_j    = sc.get('exp_j',   None)
        combined = 0.60 * max(0.0, corr) + 0.40 * (j_score or 0.5)

        assigns[target_k] = (ci, corr, shift, j_score, meas_j, exp_j, combined)
        self._preview_candidate = None

        self._draw_plot(peak)
        if self._selected_idx is not None:
            met2, mp2, _ = self.undef_items[self._selected_idx]
            self._draw_comp_table(peak, met2, mp2)

    def _bind_wheel(self, widget):
        """Forward mouse-wheel events from widget (and children) to comp_canvas."""
        def _wh(e, c=self._comp_canvas):
            if e.num == 4:   c.yview_scroll(-1, 'units')
            elif e.num == 5: c.yview_scroll(1,  'units')
            else:            c.yview_scroll(int(-e.delta / 60), 'units')
        for w in [widget] + list(widget.winfo_children()):
            try:
                w.bind('<MouseWheel>', _wh)
                w.bind('<Button-4>',   _wh)
                w.bind('<Button-5>',   _wh)
            except Exception:
                pass

    def _set_component(self, peak, k, cand, accepted,
                       is_primary=False, mp=None):
        """Accept/Reject a single MCR component.

        Primary component (is_primary=True): propagate quality to X's mp.
        Non-primary accepted (Y wins): mark Y as good, X as bad.
        Non-primary rejected: store decision only, no quality change.
        """
        mcr       = peak['mcr_annotation']
        confirmed = mcr.setdefault('confirmed_assignments', {})
        assigns   = mcr.get('assignments', [])
        assign    = assigns[k] if k < len(assigns) else (0, 0.0)
        ci, corr  = assign[0], assign[1]
        confirmed[k] = {
            'metabolite': cand.get('name', f'C{k+1}'),
            'hmdb_id':    cand.get('hmdb_id', cand.get('bmrb_id', '')),
            'corr':       corr,
            'accepted':   accepted,
        }

        peak_rank = peak.get('rank')

        if is_primary and mp is not None:
            # Accept/reject the annotated metabolite directly
            quality = 'good' if accepted else 'bad'
            mp['annotation_quality'] = quality
            self._sync_ann_quality(mp, quality)
            self._populate_tree()

        elif accepted and not is_primary:
            # User says this peak belongs to candidate Y, not to primary X.
            # 1. Find or create Y's mp in ann_results
            y_met, y_mp = self._find_candidate_mp(cand, peak_rank)
            if y_mp is None:
                y_met, y_mp = self._add_candidate_to_results(cand, peak)
            if y_mp is not None:
                y_mp['annotation_quality'] = 'good'
                self._sync_ann_quality(y_mp, 'good')
                # Persist quality mark
                try:
                    ann = self.ann_stage
                    if ann is not None:
                        if not hasattr(ann, '_quality_marks') or ann._quality_marks is None:
                            ann._quality_marks = {}
                        if y_met is not None:
                            ann._quality_marks[(y_met['id'], peak_rank)] = 'good'
                        pm = getattr(self.app, 'project_manager', None)
                        if pm is not None:
                            pm.mark_dirty()
                except Exception:
                    pass
            # 2. Mark primary X as bad
            if mp is not None:
                mp['annotation_quality'] = 'bad'
                self._sync_ann_quality(mp, 'bad')
            # 3. Refresh annotation results tree to show new/updated entry
            arw = self.ann_results
            if arw is not None:
                try:
                    win = getattr(arw, 'win', None)
                    if win and win.winfo_exists():
                        arw._populate()
                except Exception:
                    pass
            self._populate_tree()

        self._draw_plot(peak)
        if self._selected_idx is not None:
            met2, mp2, _ = self.undef_items[self._selected_idx]
            self._draw_comp_table(peak, met2, mp2)

    def _find_candidate_mp(self, cand, peak_rank):
        """Search ann_results.metabolites for the mp matching candidate Y.

        Matches by hmdb_id (or name as fallback) and peak_rank.
        Returns (met, mp) if found, else (None, None).
        """
        arw = self.ann_results
        if arw is None:
            return None, None
        hmdb_id   = cand.get('hmdb_id', cand.get('bmrb_id', ''))
        cand_name = cand.get('name', '').strip().lower()
        for met in getattr(arw, 'metabolites', []):
            met_id   = met.get('id', '')
            met_name = met.get('name', '').strip().lower()
            id_match   = hmdb_id and met_id == hmdb_id
            name_match = cand_name and cand_name == met_name
            if not (id_match or name_match):
                continue
            for mp in met.get('matched_peaks', []):
                if peak_rank is None or mp.get('peak_rank') == peak_rank:
                    return met, mp
        return None, None

    def _add_candidate_to_results(self, cand, peak):
        """Add candidate Y to ann_results as a new MCR-discovered annotation.

        If Y already exists as a metabolite but has no mp for this peak_rank,
        a new mp is appended.  If Y is not in the list at all, a stub met is
        created and inserted.  Returns the (met, mp) tuple that was added.
        """
        arw = self.ann_results
        if arw is None:
            return None, None

        hmdb_id   = cand.get('hmdb_id', cand.get('bmrb_id', ''))
        cand_name = cand.get('name', '').strip().lower()
        peak_rank = peak.get('rank')
        peak_ppm  = float(peak.get('ppm', cand.get('ppm_center', 0)))
        dp_ppm    = float(cand.get('ppm_center', peak_ppm))

        # ── Find existing met entry ──────────────────────────────────────────
        target_met = None
        for met in getattr(arw, 'metabolites', []):
            mid  = met.get('id', '')
            mnam = met.get('name', '').strip().lower()
            if (hmdb_id and mid == hmdb_id) or (cand_name and cand_name == mnam):
                target_met = met
                break

        # ── Create met stub if not found ─────────────────────────────────────
        if target_met is None:
            target_met = {
                'id':            hmdb_id or cand.get('name', 'MCR'),
                'name':          cand.get('name', 'Unknown'),
                'matched_peaks': [],
                'db_peaks':      [{
                    'ppm_center': dp_ppm,
                    'ppm_start':  dp_ppm + 0.01,
                    'ppm_end':    dp_ppm - 0.01,
                    'type':       cand.get('type', 's'),
                    'protons':    cand.get('protons', 1),
                }],
                'coverage':      100.0,
                'cosine':        0.0,
                'source':        'MCR',
                'best_score':    0.0,
                'composite_score': 0.0,
            }
            # Try to fetch full fingerprint from annotation DB
            try:
                ann = self.ann_stage
                if ann and hasattr(ann, '_last_results_args'):
                    db_path = (ann._last_results_args or {}).get('db_path')
                    if db_path:
                        import sqlite3
                        con = sqlite3.connect(db_path)
                        rows = con.execute(
                            "SELECT ppm_center, ppm_start, ppm_end, type, protons "
                            "FROM multiplets WHERE hmdb_id=? OR bmrb_id=?",
                            (hmdb_id, hmdb_id)).fetchall()
                        con.close()
                        if rows:
                            target_met['db_peaks'] = [
                                {'ppm_center': r[0], 'ppm_start': r[1],
                                 'ppm_end': r[2], 'type': r[3], 'protons': r[4]}
                                for r in rows]
            except Exception:
                pass
            arw.metabolites.append(target_met)

        # ── Add mp for this peak if not already present ──────────────────────
        existing_mp = next(
            (m for m in target_met['matched_peaks']
             if m.get('peak_rank') == peak_rank), None)
        if existing_mp is not None:
            return target_met, existing_mp

        new_mp = {
            'peak_ppm':         peak_ppm,
            'peak_rank':        peak_rank,
            'db_ppm':           dp_ppm,
            'ppm_diff':         abs(peak_ppm - dp_ppm),
            'protons':          cand.get('protons', 1),
            'type':             cand.get('type', 's'),
            'predicted_lines':  [],
            'multiplet_score':  0.0,
            'composite_score':  0.0,
            'multiplet_detail': 'MCR',
            'j_coupling':       cand.get('j_coupling', ''),
            'is_sub_peak':      False,
            'annotation_quality': None,
        }
        target_met['matched_peaks'].append(new_mp)
        return target_met, new_mp

    def _sync_ann_quality(self, mp, quality):
        """Propagate quality change back to AnnotationResultsWindow if open."""
        try:
            arw = self.ann_results
            if arw is None:
                return
            win = getattr(arw, 'win', None)
            if win is None or not win.winfo_exists():
                return
            for iid, row_data in getattr(arw, '_iid_to_row', {}).items():
                rank, _, mp2, _ = row_data
                if mp2 is mp:
                    arw._classify_one(iid, quality)
                    break
        except Exception:
            pass
