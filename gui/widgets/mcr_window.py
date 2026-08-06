# gui/widgets/mcr_window.py
"""
Unified MCR Deconvolution window.

Replaces the former AnnotationMCRWindow (Undef-peak deconvolution) and
MCRDeconvWindow (overlapping-region results) with a single window that
handles both cases, since MCR-ALS produces enough diagnostic values
(R², LOF%, Pearson r per component, ppm shift, J-coupling score,
combined confidence) to warrant a dedicated, complete view.

Reference: Tauler, R. "Multivariate curve resolution applied to second
order data." Chemom. Intell. Lab. Syst. 30(1), 133-146 (1995).

Items
-----
Each entry in `items` is a dict with a 'kind' key:
  {'kind': 'undef',  'met': met_dict, 'mp': mp_dict, 'peak': peak_dict}
      A peak marked Undef in annotation — MCR may not have run yet.
      Accept/Reject on the primary (annotated) component promotes/demotes
      mp['annotation_quality']; accepting a different candidate promotes it
      instead (creates/reuses its own metabolite entry in AnnotationResultsWindow).
  {'kind': 'region', 'met': None, 'mp': None, 'peak': peak_dict}
      A peak/region with 'mcr_annotation' already computed (K>=1),
      typically an overlapping-signal region from full-spectrum MCR or
      the annotation stage.  Accept/Reject pushes into
      target_stage.annotations and rebuilds synthetic sub-peaks in the
      Peak List (for K>=2).

Both kinds share the same 'mcr_annotation' dict schema (S, C, K, metrics,
assignments, candidates, ppm_region, r2_threshold, confirmed_assignments)
produced by gui.utils.mcr_utils.mcr_annotation_region.
"""
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

COMP_COLORS = ['#1565c0', '#e65100', '#2e7d32', '#6a1b9a', '#880e4f']


def _trapz(y, x):
    """np.trapz was removed in NumPy 2.0 — use np.trapezoid when available."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x)
    return np.trapz(y, x=x)


class MCRWindow:
    """
    Parameters
    ----------
    parent      : tk widget
    items       : list of dicts, see module docstring
    app         : main App instance (for ppm / all_intensities / project_manager)
    ann_stage   : AnnotationStage or FullSpectrumMCRStage (for r2 threshold,
                  recompute, candidate pool, peak_stage / annotations)
    ann_results : AnnotationResultsWindow instance, optional — refreshed when
                  'undef' items change quality/metabolite assignment
    """

    def __init__(self, parent, items, app, ann_stage=None, ann_results=None):
        self.items       = items
        self.app         = app
        self.ann_stage   = ann_stage
        self.ann_results = ann_results

        self._selected_idx      = None
        self._focused_k         = None
        self._mcr_running        = False
        self._preview_candidate = None   # (ci, peak) being previewed

        self.win = tk.Toplevel(parent)
        self.win.title(f"MCR Deconvolution — {len(items)} item(s)")
        self.win.geometry("1500x900")
        self.win.minsize(1000, 650)
        self._center()

        self._fig    = Figure(figsize=(9, 5), dpi=95, facecolor='#fafafa',
                              layout='constrained')
        self._canvas = None

        self._build_ui()
        if items:
            self._select_item(0)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _center(self):
        self.win.update_idletasks()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        w, h = 1500, 900
        self.win.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=6)
        main.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            main,
            text=(f"  {len(self.items)} item(s)  "
                  "— select one, Deconvolute if pending, Accept/Reject each component"),
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

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left(self, parent):
        ff = ttk.Frame(parent)
        ff.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(ff, text="Show:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._filt_var = tk.StringVar(value='all')
        for lbl, val in [('All', 'all'), ('Pending', 'pending'),
                         ('Accepted', 'accepted'), ('Rejected', 'rejected')]:
            ttk.Radiobutton(ff, text=lbl, variable=self._filt_var,
                            value=val,
                            command=self._populate_tree).pack(side=tk.LEFT, padx=2)

        cols = ('met', 'ppm', 'K', 'r2', 'status')
        self.tree = ttk.Treeview(parent, columns=cols, show='tree headings',
                                  selectmode='browse')
        self.tree.heading('#0',     text='Item')
        self.tree.heading('met',    text='Metabolite')
        self.tree.heading('ppm',    text='δ (ppm)')
        self.tree.heading('K',      text='K')
        self.tree.heading('r2',     text='R²')
        self.tree.heading('status', text='Status')

        self.tree.column('#0',     width=60,  stretch=False)
        self.tree.column('met',    width=150, stretch=True)
        self.tree.column('ppm',    width=72,  stretch=False)
        self.tree.column('K',      width=28,  stretch=False)
        self.tree.column('r2',     width=55,  stretch=False)
        self.tree.column('status', width=80,  stretch=False)

        self.tree.tag_configure('accepted', foreground='#1a7a1a')
        self.tree.tag_configure('rejected', foreground='#9a0000')
        self.tree.tag_configure('pending',  foreground='#555555')
        self.tree.tag_configure('partial',  foreground='#856404')
        self.tree.tag_configure('nomcr',    foreground='#888888')

        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        bf = ttk.Frame(parent)
        bf.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(bf, text='✓ Accept all regions (K>1)',
                   command=self._accept_all_regions).pack(fill=tk.X, pady=1)

        self._populate_tree()

    def _item_status(self, item):
        """Return (K, r2, status_text, tag) for a tree row."""
        peak = item['peak']
        mcr  = peak.get('mcr_annotation')
        if not mcr:
            return None, None, '— pending compute —', 'nomcr'
        K    = mcr.get('K', 0)
        r2   = mcr.get('metrics', {}).get('r2', float('nan'))
        conf = mcr.get('confirmed_assignments', {})
        n_accepted = sum(1 for v in conf.values() if v and v.get('accepted'))
        n_rejected = sum(1 for v in conf.values() if v and not v.get('accepted'))
        n_done     = n_accepted + n_rejected
        if n_done == 0:
            return K, r2, '—', 'pending'
        if n_rejected == K:
            return K, r2, '✗ all', 'rejected'
        if n_accepted == K:
            return K, r2, '✓ all', 'accepted'
        return K, r2, f'{n_accepted}/{K} ✓', 'partial'

    def _item_name(self, item):
        if item.get('met') is not None:
            return item['met'].get('name', '?')[:28]
        peak = item['peak']
        return peak.get('assigned_metabolite', '—')

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        filt_val = self._filt_var.get() if hasattr(self, '_filt_var') else 'all'

        for i, item in enumerate(self.items):
            peak = item['peak']
            K, r2, st_txt, tag = self._item_status(item)
            n_accepted = st_txt == '✓ all'
            n_rejected = st_txt == '✗ all'

            if filt_val == 'pending'  and tag in ('accepted', 'rejected'): continue
            if filt_val == 'accepted' and not n_accepted:                  continue
            if filt_val == 'rejected' and not n_rejected:                  continue

            iid = f'it_{i}'
            ppm = float(peak.get('ppm', 0))
            self.tree.insert('', tk.END, iid=iid,
                             text=f'⊕ #{peak.get("rank", i+1)}',
                             values=(self._item_name(item), f'{ppm:.4f}',
                                     K if K is not None else '',
                                     ('—' if (r2 is None or r2 != r2) else f'{r2:.3f}'),
                                     st_txt),
                             tags=(tag,), open=False)

            mcr = peak.get('mcr_annotation')
            if not mcr:
                continue
            cands   = mcr.get('candidates', [])
            assigns = mcr.get('assignments', [])
            conf    = mcr.get('confirmed_assignments', {})
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
                self.tree.insert(iid, tk.END, iid=f'comp_{i}_{k}',
                                 text=f'  C{k+1}',
                                 values=(name, f'r={corr:.2f}', '', '', c_st),
                                 tags=(c_tag,))

    def _on_tree_select(self, _event=None):
        sel = self.tree.focus()
        if not sel:
            return
        if sel.startswith('comp_'):
            _, i_str, k_str = sel.split('_')
            idx, k = int(i_str), int(k_str)
            self._focused_k = k
            if idx != self._selected_idx:
                self._selected_idx = idx
            self._refresh_right()
            return
        if not sel.startswith('it_'):
            return
        idx = int(sel.split('_')[1])
        self._focused_k = None
        if idx != self._selected_idx:
            self._selected_idx = idx
            self._preview_candidate = None
        self._refresh_right()

    def _select_item(self, idx):
        self._selected_idx = idx
        self._focused_k = None
        iid = f'it_{idx}'
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)
        self._refresh_right()

    def _selected_item(self):
        if self._selected_idx is None or self._selected_idx >= len(self.items):
            return None
        return self.items[self._selected_idx]

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self, parent):
        tb = ttk.Frame(parent)
        tb.pack(fill=tk.X, pady=(0, 2))

        self._btn_deconv = ttk.Button(tb, text='▶  Deconvolute',
                                       command=self._run_mcr)
        self._btn_deconv.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(tb, text='R² min:', font=("Arial", 9)).pack(side=tk.LEFT)
        self._r2_var = tk.DoubleVar(
            value=getattr(self.ann_stage, 'mcr_r2_threshold', 0.90))
        ttk.Spinbox(tb, from_=0.50, to=1.00, increment=0.01,
                    textvariable=self._r2_var, width=5,
                    format='%.2f').pack(side=tk.LEFT, padx=(2, 10))

        ttk.Button(tb, text='Recompute all',
                   command=self._recompute_all).pack(side=tk.LEFT)

        self._status_lbl = ttk.Label(
            tb, text='Select an item and click Deconvolute.',
            foreground='gray', font=("Arial", 8))
        self._status_lbl.pack(side=tk.LEFT, padx=(12, 0))

        # Stats header — K / R² / LOF% (LOF was never surfaced in the UI before)
        self._stats_lbl = ttk.Label(parent, text='', font=("Arial", 9, "bold"),
                                    foreground='#333')
        self._stats_lbl.pack(fill=tk.X, pady=(0, 2))

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, parent, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(fill=tk.X, side=tk.BOTTOM)

        lf = ttk.LabelFrame(parent, text='Components', padding=4)
        lf.pack(fill=tk.X, pady=(4, 0))
        _vsb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._comp_canvas = tk.Canvas(lf, bd=0, highlightthickness=0,
                                      height=160, yscrollcommand=_vsb.set)
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
            lambda e: self._comp_canvas.itemconfig(self._comp_cwin, width=e.width))

        def _on_wheel(e, c=self._comp_canvas):
            if e.num == 4:   c.yview_scroll(-1, 'units')
            elif e.num == 5: c.yview_scroll(1,  'units')
            else:            c.yview_scroll(int(-e.delta / 60), 'units')
        self._comp_canvas.bind('<MouseWheel>', _on_wheel)
        self._comp_canvas.bind('<Button-4>',   _on_wheel)
        self._comp_canvas.bind('<Button-5>',   _on_wheel)
        self._wheel_fn  = _on_wheel
        self._comp_rows = []

    def _set_status(self, msg, color='gray'):
        self._status_lbl.config(text=msg, foreground=color)

    def _refresh_right(self):
        item = self._selected_item()
        if item is None:
            return
        peak = item['peak']
        if peak.get('mcr_annotation'):
            self._draw_stats_header(peak)
            self._draw_plot(item)
            self._draw_comp_table(item)
        else:
            self._stats_lbl.config(text='')
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    f'Item #{peak.get("rank","?")}  '
                    f'{peak.get("ppm", 0):.4f} ppm\n'
                    f'{self._item_name(item)}\n\n'
                    'Click "Deconvolute" to run MCR on this region.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#555')
            ax.set_axis_off()
            self._canvas.draw_idle()
            self._clear_comp_table()

    def _draw_stats_header(self, peak):
        mcr = peak.get('mcr_annotation') or {}
        m   = mcr.get('metrics', {})
        K       = mcr.get('K', '?')
        r2      = m.get('r2', float('nan'))
        r2_med  = m.get('r2_median', float('nan'))
        lof     = m.get('lof', float('nan'))
        flat_txt = '  (flat region — R² undefined)' if m.get('flat') else ''
        r2_txt     = '—' if (r2 != r2) else f'{r2:.4f}'
        r2_med_txt = '—' if (r2_med != r2_med) else f'{r2_med:.4f}'
        lof_txt    = '—' if (lof != lof) else f'{lof:.2f}%'
        self._stats_lbl.config(
            text=f'  K={K}   R² pooled={r2_txt}   R² per-sample (median)={r2_med_txt}'
                 f'   LOF={lof_txt}{flat_txt}')

    # ── MCR computation ──────────────────────────────────────────────────────

    def _run_mcr(self):
        item = self._selected_item()
        if item is None or self._mcr_running:
            return
        self._mcr_running = True
        self._btn_deconv.config(state='disabled')
        self._set_status('Running MCR…', 'blue')
        idx = self._selected_idx
        threading.Thread(target=self._mcr_thread, args=(idx, item),
                          daemon=True).start()

    def _mcr_thread(self, idx, item):
        try:
            mcr_ann = self._compute_mcr(item)
            self.win.after(0, lambda: self._on_mcr_done(idx, item, mcr_ann))
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.win.after(0, lambda: self._on_mcr_error(str(exc)))

    def _compute_mcr(self, item):
        """Run MCR-ALS on the item's region, using annotation DB candidates."""
        from gui.utils.mcr_utils import (mcr_annotation_region,
                                          simulate_lorentzian_profile)
        from gui.widgets.annotation_results_window import _build_stick_positions

        peak = item['peak']
        met  = item.get('met')
        mp   = item.get('mp')

        ppm_arr = getattr(self.app, 'ppm', None)
        all_int = getattr(self.app, 'all_intensities', None)
        if ppm_arr is None or all_int is None:
            raise RuntimeError('No spectrum data loaded.')

        r2_thr    = float(self._r2_var.get())
        shift_tol = 0.02
        mhz       = 600

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
        raw_results = []
        ann = self.ann_stage
        if ann and hasattr(ann, '_last_results_args'):
            raw_results = (ann._last_results_args or {}).get('results', [])

        region_entries = []
        for r in raw_results:
            rp = float(r.get('ppm_center', r.get('ppm', 0)))
            if min(lo, hi) - 0.01 <= rp <= max(lo, hi) + 0.01:
                region_entries.append(r)

        if not region_entries:
            if mp is not None:
                region_entries = [{
                    'ppm_center': float(mp.get('db_ppm', peak.get('ppm', 0))),
                    'type':       mp.get('type', 's'),
                    'protons':    mp.get('protons', 1),
                    'j_coupling': mp.get('j_coupling', ''),
                    'name':       met.get('name', '?') if met else '?',
                    'hmdb_id':    met.get('id', '') if met else '',
                }]
            else:
                existing = peak.get('mcr_annotation', {})
                region_entries = existing.get('candidates') or [{
                    'ppm_center': float(peak.get('ppm', 0)),
                    'type':       's',
                    'protons':    1,
                    'j_coupling': '',
                    'name':       peak.get('assigned_metabolite', 'Unknown'),
                    'hmdb_id':    '',
                }]

        seen: dict = {}
        for r in region_entries:
            key = (r.get('hmdb_id') or r.get('bmrb_id', ''),
                   round(float(r.get('ppm_center', 0)), 3))
            if key not in seen:
                seen[key] = r
        uniq = list(seen.values())

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
            'S':               S,
            'C':               C,
            'K':               K,
            'metrics':         mcr_m,
            'assignments':     assignments,
            'candidates':      uniq,
            'db_profiles':     aligned_profiles,
            'db_profiles_raw': db_profiles,
            'cand_scores':     cand_scores,
            'ppm_region':      ppm_reg,
            'r2_threshold':    r2_thr,
            'confirmed_assignments': {},
        }

    def _on_mcr_done(self, idx, item, mcr_ann):
        self._mcr_running = False
        self._btn_deconv.config(state='normal')
        peak = item['peak']
        peak['mcr_annotation'] = mcr_ann
        r2 = mcr_ann.get('metrics', {}).get('r2', float('nan'))
        K  = mcr_ann.get('K', '?')
        self._set_status(
            f'Done — K={K}  R²={"—" if r2 != r2 else f"{r2:.3f}"}', '#1a7a1a')
        self._populate_tree()
        if self._selected_idx == idx:
            self._refresh_right()

    def _on_mcr_error(self, msg):
        self._mcr_running = False
        self._btn_deconv.config(state='normal')
        self._set_status(f'Error: {msg}', 'red')

    def _recompute_all(self):
        """Re-run MCR with the current R² threshold on every already-computed item."""
        r2 = float(self._r2_var.get())
        if self.ann_stage is not None:
            try:
                self.ann_stage.mcr_r2_threshold = r2
            except Exception:
                pass
        for item in self.items:
            peak = item['peak']
            if not peak.get('mcr_annotation'):
                continue
            try:
                rerun = getattr(self.ann_stage, '_rerun_mcr_for_peak', None)
                if item['kind'] == 'region' and rerun is not None:
                    rerun(peak, r2)
                else:
                    peak['mcr_annotation'] = self._compute_mcr(item)
            except Exception as e:
                print(f'Recompute MCR error (item {peak.get("rank","?")}): {e}')
        self._populate_tree()
        self._refresh_right()

    # ── Plot ─────────────────────────────────────────────────────────────────

    def _draw_plot(self, item):
        if self._canvas is None:
            return
        peak = item['peak']
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
        focused_k = self._focused_k

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
            reg_mask = ((ppm_full >= min(reg_lo, reg_hi)) &
                        (ppm_full <= max(reg_lo, reg_hi)))
            if reg_mask.sum() > 1:
                exp_reg_ppm = ppm_full[reg_mask]
                exp_reg_int = int_full[reg_mask]
                e_max = float(np.percentile(exp_reg_int, 99.0))
                if e_max <= 0:
                    e_max = float(exp_reg_int.max()) if exp_reg_int.max() > 0 else 1.0

        raw_fit   = sum(S[:, k] * c_mean[k] for k in range(K))
        fit_peak  = raw_fit.max() if raw_fit.max() > 0 else 1.0
        amp_scale = e_max / fit_peak

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
            if focused_k is not None:
                alpha = 0.92 if k == focused_k else 0.10
                lw_c  = 2.5  if k == focused_k else 0.5
            else:
                alpha = 0.88 if accepted else 0.28
                lw_c  = 2.0  if accepted else 0.8
            pct = 100.0 * (S[:, k] * c_mean[k]).sum() / total_area
            components.append((k, sk, f'C{k+1}: {lbl}  r={corr:.2f}  ({pct:.0f}%)',
                                color, alpha, lw_c, ci))

        sort_idx   = np.argsort(ppm_reg)
        ppm_sorted = ppm_reg[sort_idx]

        def _sort(v):
            return v[sort_idx]

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

        gs  = self._fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.04)
        ax1 = self._fig.add_subplot(gs[0])
        ax2 = self._fig.add_subplot(gs[1], sharex=ax1)

        if exp_ctx_ppm is not None and exp_ctx_int is not None:
            ax1.fill_between(exp_ctx_ppm, exp_ctx_int,
                             alpha=0.13, color='#444444', zorder=1)
            ax1.plot(exp_ctx_ppm, exp_ctx_int,
                     color='#333333', lw=1.1, alpha=0.75,
                     label='Experimental (mean)', zorder=2)

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

        if self._preview_candidate is not None:
            prev_ci, prev_peak = self._preview_candidate
            if prev_peak is peak and prev_ci < len(db_profiles):
                prev_prof = db_profiles[prev_ci]
                if len(prev_prof) == len(ppm_reg):
                    prev_sorted = np.clip(_sort(np.asarray(prev_prof)), 0, None)
                    prev_pk     = float(prev_sorted.max()) if prev_sorted.max() > 0 else 1.0
                    pname = (cands[prev_ci].get('name', '?')
                             if prev_ci < len(cands) else '?')
                    ax1.plot(ppm_sorted, prev_sorted * e_max / prev_pk,
                             color='#555555', lw=1.4, ls='--', alpha=0.80,
                             label=f'Preview: {pname}', zorder=4)

        for k, sk, lbl, color, alpha, lw_c, ci in components:
            sk_sorted = np.clip(_sort(sk), 0, None)
            ax1.fill_between(ppm_sorted, sk_sorted,
                             alpha=max(0.04, alpha * 0.22),
                             color=color, zorder=2)
            ax1.plot(ppm_sorted, sk_sorted,
                     color=color, lw=lw_c, alpha=alpha,
                     label=lbl, zorder=4)

        fit_in_reg = np.clip(
            sum(_sort(sk) for k, sk, *_ in components), 0, None)
        ax1.plot(ppm_sorted, fit_in_reg,
                 color='#c62828', lw=2.0, ls='--', alpha=0.90, zorder=5,
                 label=f'MCR fit  R²={"—" if r2 != r2 else f"{r2:.3f}"}')

        ax1.tick_params(labelsize=7, labelbottom=False)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=7)
        ax1.set_title(
            f'Item #{peak.get("rank","?")}  {peak.get("ppm", 0):.4f} ppm  K={K}  '
            f'R²={"—" if r2 != r2 else f"{r2:.3f}"}',
            fontsize=8, pad=3)
        ax1.legend(fontsize=6.5, loc='upper left',
                   framealpha=0.85, ncol=min(2, K + 3))
        ax1.set_facecolor('#fafafa')
        ax1.set_xlim(max(ctx_lo, ctx_hi), min(ctx_lo, ctx_hi))  # NMR: high ppm left
        comp_peaks = [float(np.max(np.clip(_sort(sk), 0, None)))
                      for k, sk, *_ in components]
        y_top = max(e_max, max(comp_peaks) if comp_peaks else 1.0)
        ax1.set_ylim(bottom=-y_top * 0.05, top=y_top * 1.20)

        if exp_reg_ppm is not None and exp_reg_int is not None:
            if exp_reg_ppm[0] > exp_reg_ppm[-1]:
                rrp = exp_reg_ppm[::-1]; rri = exp_reg_int[::-1]
            else:
                rrp = exp_reg_ppm.copy(); rri = exp_reg_int.copy()
            fit_reg = np.zeros_like(rrp, dtype=float)
            for k, sk, lbl, color, alpha, lw_c, ci in components:
                fit_reg += np.interp(rrp, ppm_sorted, np.clip(_sort(sk), 0, None))
            residual = rri - fit_reg
            ax2.fill_between(rrp, residual, alpha=0.28, color='#888888')
            ax2.plot(rrp, residual, color='#555555', lw=0.9)
        ax2.axhline(0, color='#aaaaaa', lw=0.7, ls='--')
        ax2.set_ylabel('Residual\n(MCR region)', fontsize=7)
        ax2.set_xlabel('δ (ppm)', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.set_facecolor('#f5f5f5')

        self._canvas.draw_idle()

    # ── Component table ──────────────────────────────────────────────────────

    def _clear_comp_table(self):
        for w in self._comp_rows:
            try: w.destroy()
            except Exception: pass
        self._comp_rows = []

    def _draw_comp_table(self, item):
        self._clear_comp_table()
        peak = item['peak']
        mcr  = peak.get('mcr_annotation')
        if not mcr:
            return

        assigns = mcr.get('assignments', [])
        cands   = mcr.get('candidates', [])
        conf    = mcr.get('confirmed_assignments', {})
        parent  = self._comp_frame

        met = item.get('met')
        mp  = item.get('mp')
        primary_name = (met.get('name', '') if met else '').lower()

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

        # Rows are shown ranked by combined confidence (descending), not by
        # raw MCR component index — rank 0 is what the algorithm would pick
        # first; the rest is the reliability-ordered list of alternatives.
        # C{k+1} labels still reference the true MCR/S,C column index k, since
        # Accept/Reject/_promote_candidate key off it (confirmed_assignments,
        # assigns[k]) — only the display ORDER changes, not the identity.
        def _conf_key(kk):
            a = assigns[kk]
            v = a[6] if len(a) > 6 else None
            return v if v is not None else -1.0
        order = sorted(range(len(assigns)), key=_conf_key, reverse=True)

        for rank_pos, k in enumerate(order):
            assign    = assigns[k]
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
            is_top    = (rank_pos == 0)

            is_primary = bool(primary_name and primary_name in name.lower())

            if c_conf is None:
                row_bg = '#fff8e1' if is_primary else 'white'
                st_txt, st_fg = ('🥇 Top pick', '#7a6000') if is_top else ('—', '#888888')
            elif c_conf.get('accepted'):
                row_bg, st_txt, st_fg = '#e8f5e9', '✓ Accepted', '#1a7a1a'
            else:
                row_bg, st_txt, st_fg = '#ffebee', '✗ Rejected', '#9a0000'

            row = tk.Frame(parent, bg=row_bg, bd=0)
            row.pack(fill=tk.X, pady=1)
            self._comp_rows.append(row)

            tk.Label(row, text=f'C{k+1}', font=('Arial', 9, 'bold'),
                     fg=color, bg=row_bg, width=3, anchor='w',
                     padx=4).pack(side=tk.LEFT)
            lbl_name = name + (' ★' if is_primary else '')
            tk.Label(row, text=lbl_name,
                     font=('Arial', 9, 'bold' if is_primary else 'normal'),
                     bg=row_bg, width=26, anchor='w',
                     padx=2).pack(side=tk.LEFT)
            corr_fg = ('#1a7a1a' if corr >= 0.85 else
                       '#856404' if corr >= 0.60 else '#9a0000')
            tk.Label(row, text=f'{corr:.3f}', font=('Arial', 9),
                     fg=corr_fg, bg=row_bg, width=6,
                     anchor='w').pack(side=tk.LEFT)
            shift_txt = f'{shift_ppm:+.3f}' if abs(shift_ppm) > 1e-4 else '0'
            shift_fg  = '#c62828' if abs(shift_ppm) > 0.015 else '#444444'
            tk.Label(row, text=shift_txt, font=('Arial', 9),
                     fg=shift_fg, bg=row_bg, width=6,
                     anchor='w').pack(side=tk.LEFT)
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
            if combined is not None:
                conf_txt = f'{combined:.2f}'
                conf_fg  = ('#1a7a1a' if combined >= 0.75 else
                             '#856404' if combined >= 0.50 else '#9a0000')
            else:
                conf_txt, conf_fg = '—', '#888888'
            tk.Label(row, text=conf_txt, font=('Arial', 9, 'bold'),
                     fg=conf_fg, bg=row_bg, width=6,
                     anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=st_txt, font=('Arial', 9, 'bold'),
                     fg=st_fg, bg=row_bg, width=10,
                     anchor='w').pack(side=tk.LEFT)

            def _acc(c=cand, k_=k, it=item, is_p=is_primary):
                self._set_component(it, k_, c, accepted=True, is_primary=is_p)
            def _rej(c=cand, k_=k, it=item, is_p=is_primary):
                self._set_component(it, k_, c, accepted=False, is_primary=is_p)

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

            if is_primary and mp is not None:
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
                self._draw_other_cand_row(parent, item, ci, cand)

        self._comp_canvas.after_idle(
            lambda: self._comp_canvas.yview_moveto(0))

    def _draw_other_cand_row(self, parent, item, ci, cand):
        peak        = item['peak']
        mcr         = peak['mcr_annotation']
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

        prev_txt   = '◉ Previewing' if is_prev else '○ Preview'
        prev_color = '#1565c0' if is_prev else '#444444'
        def _toggle(ci_=ci, pk_=peak, it=item):
            self._preview_candidate = (
                None if self._preview_candidate == (ci_, pk_) else (ci_, pk_))
            self._draw_plot(it)
            self._draw_comp_table(it)
        tk.Button(row, text=prev_txt, font=('Arial', 8),
                  fg=prev_color, bg=row_bg, relief='flat', bd=1,
                  command=_toggle).pack(side=tk.LEFT, padx=4)

        if K >= 1:
            k_var     = tk.StringVar(value='C1')
            k_choices = [f'C{k+1}' for k in range(K)]
            ttk.Combobox(row, textvariable=k_var, values=k_choices,
                         width=4, state='readonly').pack(side=tk.LEFT, padx=2)
            def _promote(ci_=ci, it=item, kv=k_var):
                self._promote_candidate(it, ci_, int(kv.get()[1:]) - 1)
            tk.Button(row, text='Promote', font=('Arial', 8),
                      bg='#e3f2fd', relief='raised',
                      command=_promote).pack(side=tk.LEFT, padx=2)

        self._bind_wheel(row)

    def _promote_candidate(self, item, ci, target_k):
        peak        = item['peak']
        mcr         = peak['mcr_annotation']
        assigns     = mcr.get('assignments', [])
        db_profiles = mcr.get('db_profiles', [])
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

        self._draw_plot(item)
        self._draw_comp_table(item)
        self._populate_tree()

    def _bind_wheel(self, widget):
        _wf = getattr(self, '_wheel_fn', None)
        if not _wf:
            return
        for w in [widget] + list(widget.winfo_children()):
            try:
                w.bind('<MouseWheel>', _wf)
                w.bind('<Button-4>',   _wf)
                w.bind('<Button-5>',   _wf)
            except Exception:
                pass

    # ── Accept / Reject ──────────────────────────────────────────────────────

    def _set_component(self, item, k, cand, accepted, is_primary=False):
        """Accept/Reject a single MCR component; behaviour depends on item kind.

        kind='undef' : primary component propagates annotation_quality to mp;
                       accepting a non-primary candidate promotes it instead.
        kind='region': accepted components are pushed into the annotation
                       stage and rebuilt as synthetic sub-peaks in Peak List.
        """
        peak      = item['peak']
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

        if item['kind'] == 'undef':
            self._set_component_undef(item, k, cand, accepted, is_primary)
        else:
            self._set_component_region(item, k, cand, accepted)
            self._refresh_peak_list_window()

        self._populate_tree()
        self._draw_plot(item)
        self._draw_comp_table(item)

    def _set_component_undef(self, item, k, cand, accepted, is_primary):
        peak      = item['peak']
        mp        = item.get('mp')
        peak_rank = peak.get('rank')

        if is_primary and mp is not None:
            quality = 'good' if accepted else 'bad'
            mp['annotation_quality'] = quality
            self._sync_ann_quality(mp, quality)

        elif accepted and not is_primary:
            y_met, y_mp = self._find_candidate_mp(cand, peak_rank)
            if y_mp is None:
                y_met, y_mp = self._add_candidate_to_results(cand, peak)
            if y_mp is not None:
                y_mp['annotation_quality'] = 'good'
                self._sync_ann_quality(y_mp, 'good')
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
            if mp is not None:
                mp['annotation_quality'] = 'bad'
                self._sync_ann_quality(mp, 'bad')
            arw = self.ann_results
            if arw is not None:
                try:
                    win = getattr(arw, 'win', None)
                    if win and win.winfo_exists():
                        arw._populate()
                except Exception:
                    pass

    def _set_component_region(self, item, k, cand, accepted):
        peak = item['peak']
        mcr  = peak['mcr_annotation']
        if accepted:
            self._push_to_annotation(peak, k, cand, mcr['assignments'][k][1]
                                      if k < len(mcr.get('assignments', [])) else 0.0)
            if int(mcr.get('K', 1)) <= 1:
                peak['assigned_metabolite'] = cand.get('name', f'Component {k+1}')
            self._sync_peak_list_from_mcr(peak)

    def _push_to_annotation(self, peak, k, cand, corr):
        ann_stage = self.ann_stage
        if ann_stage is None:
            return
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
            'ppm':        peak['ppm'],
        }
        existing = target_stage.annotations.get(peak['rank'])
        if isinstance(existing, list):
            existing.append(entry)
        else:
            target_stage.annotations[peak['rank']] = entry

    def _refresh_peak_list_window(self):
        try:
            plw = getattr(self.app, '_peak_list_win', None)
            if plw is not None and plw.win.winfo_exists():
                plw._populate()
        except Exception:
            pass

    def _build_mcr_subpeak(self, peak, k, cand):
        mcr = peak.get('mcr_annotation', {})
        S = mcr.get('S'); C = mcr.get('C'); ppm_reg = mcr.get('ppm_region')
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
        snr = float(peak.get('snr', 0.0))
        if h > 0 and peak.get('height'):
            try:
                snr = float(peak.get('snr', 0.0)) * (h / max(float(peak.get('height')), 1e-12))
            except Exception:
                pass

        # Index bounds into the FULL spectrum arrays — Quantification reads
        # start_idx/end_idx (falls back to 0 when absent, which degenerates
        # to a zero-width window at the start of the spectrum).
        ppm_full = getattr(self.app, 'ppm', None)
        start_idx = end_idx = 0
        if ppm_full is not None and len(ppm_full) > 0:
            start_idx = int(np.argmin(np.abs(ppm_full - start_ppm)))
            end_idx   = int(np.argmin(np.abs(ppm_full - end_ppm)))
            start_idx, end_idx = min(start_idx, end_idx), max(start_idx, end_idx)

        # Per-sample area/height — this is what actually answers "how much of
        # this region belongs to THIS analyte": S[:,k] is the component shape,
        # C[:,k] its per-sample concentration (mcr_annotation_region ran MCR-ALS
        # on the full per-sample matrix, not a mean spectrum). Integrating
        # S[:,k]*C[j,k] per sample j isolates this analyte from whatever
        # neighbour caused the overlap — a raw start_idx/end_idx reintegration
        # of the shared window cannot do that. Quantification uses this array
        # directly (see gui/stages/_quant_computation.py) instead of
        # reintegrating the raw window for accepted MCR sub-peaks.
        area_per_sample   = np.array([float(_trapz(S[:, k] * C[j, k], ppm_reg))
                                      for j in range(C.shape[0])])
        height_per_sample = np.clip(S[i_pk, k] * C[:, k], 0, None)

        return {
            'rank':             0,
            'ppm':              ppm_c,
            'start_ppm':        start_ppm,
            'end_ppm':          end_ppm,
            'start_idx':        start_idx,
            'end_idx':          end_idx,
            'width_ppm':        w,
            'width':            w,
            'height':           h,
            'snr':              snr,
            'area':             float(_trapz(prof, ppm_reg)),
            'area_per_sample':  area_per_sample,
            'height_per_sample': height_per_sample,
            'overlapping':      False,
            'is_sub_peak':      True,
            'deconvolved':      False,
            'parent_ppm':       float(peak.get('ppm', ppm_c)),
            '_mcr_parent_ppm':  float(peak.get('ppm', ppm_c)),
            '_mcr_parent_rank': int(peak.get('rank', 0)),
            'assigned_metabolite': cand.get('name', f'Component {k+1}'),
            'status':           peak.get('status', None),
            'mcr_component_idx': int(k),
        }

    def _sync_peak_list_from_mcr(self, peak):
        if self.ann_stage is None or not hasattr(self.ann_stage, 'peak_stage'):
            return
        peak_stage = self.ann_stage.peak_stage
        peak_list = getattr(peak_stage, 'peaks', None)
        if not isinstance(peak_list, list):
            return

        mcr = peak.get('mcr_annotation', {})
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
            sp = self._build_mcr_subpeak(peak, k, cand)
            if sp is not None:
                sub_peaks.append(sp)
        if not sub_peaks:
            return

        parent_ppm = float(peak.get('ppm', 0.0))
        parent_rank = int(peak.get('rank', 0))
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

        kept_existing = [p for p in peak_list if not _to_remove(p)]
        # Snapshot ranks BEFORE the resort/renumber below — only for peaks that
        # already existed (the new sub_peaks have no prior rank to remap from).
        old_rank_of = {id(p): p.get('rank') for p in kept_existing}

        kept = kept_existing + sub_peaks
        kept.sort(key=lambda p: abs(float(p.get('ppm', 0.0))))
        rank_map = {}
        for i, p in enumerate(kept, 1):
            old_r = old_rank_of.get(id(p))
            if old_r is not None and old_r != i:
                rank_map[old_r] = i
            p['rank'] = i

        peak_list.clear()
        peak_list.extend(kept)
        if hasattr(peak_stage, 'processor') and getattr(peak_stage, 'processor', None):
            peak_stage.processor.peaks = peak_list

        if rank_map:
            self._remap_peak_ranks(rank_map)

    def _remap_peak_ranks(self, rank_map):
        """Fix up every structure keyed by peak_rank after the renumbering
        above shifted ranks for peaks that were NOT part of the MCR split.
        Without this, quality/anchor marks and quant overrides silently point
        at the wrong physical peak (or none) after an Accept.
        """
        try:
            ann = self.ann_stage
            if ann is not None and not hasattr(ann, '_quality_marks'):
                ann = getattr(self.ann_stage, 'annotation_stage', None)
            if ann is not None:
                for attr in ('_quality_marks', '_anchor_lines'):
                    d = getattr(ann, attr, None)
                    if d:
                        setattr(ann, attr, {
                            (mid, rank_map.get(rk, rk)): v
                            for (mid, rk), v in d.items()})
                d = getattr(ann, '_subline_quality_marks', None)
                if d:
                    ann._subline_quality_marks = {
                        (mid, rank_map.get(rk, rk), li): v
                        for (mid, rk, li), v in d.items()}
                _args = getattr(ann, '_last_results_args', None)
                if _args:
                    for r in _args.get('results', []):
                        _pr = r.get('peak_rank')
                        if _pr in rank_map:
                            r['peak_rank'] = rank_map[_pr]
        except Exception as e:
            print(f'   _remap_peak_ranks (annotation) error: {e}')

        try:
            qstage = (getattr(self.app, 'stages', {}) or {}).get('quantification')
            overrides = getattr(qstage, '_quant_peak_overrides', None) if qstage else None
            if overrides:
                for mid, rk in list(overrides.items()):
                    if rk in rank_map:
                        overrides[mid] = rank_map[rk]
        except Exception as e:
            print(f'   _remap_peak_ranks (quant overrides) error: {e}')

    def _find_candidate_mp(self, cand, peak_rank):
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
        arw = self.ann_results
        if arw is None:
            return None, None

        hmdb_id   = cand.get('hmdb_id', cand.get('bmrb_id', ''))
        cand_name = cand.get('name', '').strip().lower()
        peak_rank = peak.get('rank')
        peak_ppm  = float(peak.get('ppm', cand.get('ppm_center', 0)))
        dp_ppm    = float(cand.get('ppm_center', peak_ppm))

        target_met = None
        for met in getattr(arw, 'metabolites', []):
            mid  = met.get('id', '')
            mnam = met.get('name', '').strip().lower()
            if (hmdb_id and mid == hmdb_id) or (cand_name and cand_name == mnam):
                target_met = met
                break

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

    def _accept_all_regions(self):
        """Accept every component of every 'region' item with K>1."""
        for item in self.items:
            if item['kind'] != 'region':
                continue
            peak = item['peak']
            mcr = peak.get('mcr_annotation', {})
            K = mcr.get('K', 0)
            if K <= 1:
                continue
            assigns = mcr.get('assignments', [])
            cands   = mcr.get('candidates', [])
            for k in range(K):
                _a = assigns[k] if k < len(assigns) else (0, 0.0)
                ci = _a[0]
                cand = cands[ci] if ci < len(cands) else {}
                self._set_component(item, k, cand, accepted=True)
        self._populate_tree()
