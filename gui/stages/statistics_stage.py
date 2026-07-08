# gui/stages/statistics_stage.py
"""
Statistics Stage — multivariate analysis and visualisation.

Left panel  : method selector + options
Right panel : plot canvas with save / export controls
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# ---------- helpers ----------------------------------------------------------

def _get_quantification_matrix(app):
    """
    Build a (n_samples, n_features) matrix from the app state.
    Priority: quantification results → Good-annotated peaks → None.
    Returns (X, sample_names, feature_labels, group_labels).
    """
    quant_stage = (getattr(app, 'stages', {}) or {}).get('quantification')
    ann_stage   = (getattr(app, 'stages', {}) or {}).get('annotation')
    si          = getattr(app, 'samples_info', []) or []
    sample_names = [s.get('name', f'S{i}') for i, s in enumerate(si)]
    group_labels = [s.get('label', s.get('group', 'Unknown')) for s in si]
    n_s          = len(si)

    # ── Path 1: quantification stage results ─────────────────────────────────
    # _quant_results is a list of dicts, each with:
    #   'metabolite'        — display name
    #   '_concs_per_sample' — list[float] indexed by sample position
    quant_results = getattr(quant_stage, '_quant_results', None)
    if quant_results:
        feature_labels = []
        X_cols = []
        for met in quant_results:
            # Não-quantificados (perderam competição ou 0/NaN): sem sinal
            # mensurável — excluídos da matriz estatística.
            if met.get('_not_quantified'):
                continue
            concs = met.get('_concs_per_sample') or []
            if not concs:
                continue
            # Align length to n_s
            col = list(concs[:n_s]) + [float('nan')] * max(0, n_s - len(concs))
            feature_labels.append(
                met.get('metabolite') or met.get('name') or '?')
            X_cols.append(col)
        if X_cols:
            X = np.array(X_cols, dtype=float).T   # (n_samples, n_features)
            return X, sample_names, feature_labels, group_labels

    # ── Path 2: Good-status peaks from peak detection ─────────────────────────
    ppm_arr = getattr(app, 'ppm', None)
    all_int = getattr(app, 'all_intensities', None)
    if ppm_arr is None or all_int is None:
        return None, None, None, None

    # Build rank→name from annotation results
    _ann_args    = getattr(ann_stage, '_last_results_args', None) or {}
    _rank_to_name = {}
    for r in (_ann_args.get('results') or []):
        rank = r.get('peak_rank')
        if rank is not None and r.get('name'):
            _rank_to_name.setdefault(rank, r['name'])

    det_stage = (getattr(app, 'stages', {}) or {}).get('peak_detection')
    feature_labels = []
    X_cols = []
    for pk in (getattr(det_stage, 'peaks', None) or []):
        if pk.get('status') == 'Good':
            idx = int(np.argmin(np.abs(ppm_arr - pk['ppm'])))
            col = (all_int[idx, :].tolist()
                   if all_int.ndim == 2 else [float(all_int[idx])])
            name = (_rank_to_name.get(pk.get('rank'))
                    or pk.get('annotation', {}).get('name')
                    or f'{pk["ppm"]:.4f} ppm')
            feature_labels.append(name)
            X_cols.append(col)

    if not X_cols:
        return None, None, None, None

    X = np.array(X_cols, dtype=float).T
    return X, sample_names, feature_labels, group_labels


def _group_to_int(group_labels):
    """Convert group label strings to integer class array y."""
    unique = sorted(set(group_labels))
    mapping = {g: i for i, g in enumerate(unique)}
    return np.array([mapping[g] for g in group_labels], dtype=int), unique


_CM_PER_IN = 2.54


def _fit_page(w, h):
    """Scale (w, h) inches down proportionally so it fits one A4 page.

    Orientation follows the content: a wider-than-tall figure is bounded by
    landscape A4 (11.69 × 8.27 in), otherwise by portrait A4 (8.27 × 11.69).
    Never scales up — small figures are left untouched.
    """
    A4_LONG, A4_SHORT = 11.69, 8.27
    w = max(float(w), 0.5)
    h = max(float(h), 0.5)
    if w >= h:
        wmax, hmax = A4_LONG, A4_SHORT
    else:
        wmax, hmax = A4_SHORT, A4_LONG
    scale = min(1.0, wmax / w, hmax / h)
    return w * scale, h * scale


def _resolve_box_selection(selected, feat_raw, capacity, x_raw=None,
                           fallback_order=None):
    """Map user-selected metabolite names to column indices in `feat_raw`.

    Explicit selections are returned in `feat_raw` order (stable across renders)
    and capped to `capacity` (= grid rows × cols). When nothing valid is
    selected, fall back to `fallback_order` (a precomputed ranking of column
    indices, e.g. by significance or abundance); if that is not given, fall back
    to highest-variance columns (requires `x_raw`). Names not present in
    `feat_raw` (e.g. globally excluded) are dropped rather than fabricated.
    """
    capacity = max(int(capacity), 0)
    name_set = set(selected or [])
    idx = [i for i, f in enumerate(feat_raw) if f in name_set]
    if not idx:
        if fallback_order is not None:
            idx = [int(i) for i in fallback_order]
        elif x_raw is not None and getattr(x_raw, 'shape', (0, 0))[1] > 0:
            scores = np.nanvar(x_raw, axis=0)
            idx = list(np.argsort(scores)[::-1])
    return idx[:capacity]


# ---------- main class -------------------------------------------------------

class StatisticsStage:
    def __init__(self, parent, app):
        self.parent = parent
        self.app    = app
        self._fig          = None
        self._ax           = None
        self._canvas       = None
        self._toolbar      = None
        self._right_widget = None   # widget inside app.right_panel
        self._current_method = None
        self._last_plot_fig  = None   # for export
        self._plot_export    = None   # {'method','headers','rows'} for CSV export
        self._cb_a         = None
        self._cb_b         = None
        self._excluded_samples     = set()
        self._excluded_metabolites = set()
        self._pie_style = {
            'title':            'Metabolic Profile by Group',
            'title_size':       11,
            'title_bold':       True,
            'group_titles':     {},   # {group_name: str}
            'group_title_size': 9,
            'group_title_bold': True,
            'edge_color':       '#ffffff',
            'edge_width':       0.8,
            'tight_pad':        1.0,
            'met_colors':       {},   # {met_name: hex} — user overrides
        }
        self._pie_export_figsize = None   # (w, h) inches — set by _plot_metabolic_profile
        self._box_style = {
            'title':        'Distribution by Group',
            'title_size':   11,
            'title_bold':   True,
            'xlabel':       'Metabolite',
            'ylabel':       '',   # blank = auto unit from the quantification method
            'label_size':   9,
            'tick_size':    7,
            'xlabel_rot':   45,
            'linewidth':    1.0,
            'tight_pad':    1.2,
            'group_colors': {},   # {group_name: hex}
        }
        # Unified Box/Violin layout config (grid of per-metabolite panels).
        # Persisted via project_manager. `selected` is an ordered metabolite list.
        self._box_config = {
            'kind':       'box',    # 'box' | 'violin'
            'rows':       3,        # panel grid rows
            'cols':       3,        # panel grid cols
            'panel_w':    6.5,      # cm per panel (width)
            'panel_h':    6.0,      # cm per panel (height)
            'selected':   [],       # metabolite names to plot (empty -> top variance)
            'points':     True,     # overlay raw data points
            'point_size': 24,       # scatter marker area
            'mean':       False,    # box only
            'inner':      'box',    # violin only: box|quartile|point|stick|none
            'test':       'none',   # 'none' | 't' (Welch) | 'mann' (Mann-Whitney)
            'sig_line':   False,    # draw significance brackets between groups
        }
        self._box_export_figsize = None
        self.create_widgets()

    # ── Layout ───────────────────────────────────────────────────────────────

    def create_widgets(self):
        self.frame = ttk.Frame(self.parent)

        # Top toolbar
        tb = ttk.Frame(self.frame)
        tb.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(tb, text='Statistics', font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        ttk.Button(tb, text='Run',     command=self._run   ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(tb, text='Export…', command=self._export).pack(side=tk.RIGHT, padx=2)

        # Scrollable left panel
        left_outer = ttk.Frame(self.frame)
        left_outer.pack(fill=tk.BOTH, expand=True)

        _v = ttk.Scrollbar(left_outer, orient=tk.VERTICAL)
        _v.pack(side=tk.RIGHT, fill=tk.Y)
        _lc = tk.Canvas(left_outer, yscrollcommand=_v.set,
                        highlightthickness=0, bg='#f0f0f0')
        _lc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _v.config(command=_lc.yview)

        left = ttk.Frame(_lc, padding=4)
        _win = _lc.create_window((0, 0), window=left, anchor='nw')
        left.bind('<Configure>',
                  lambda e: _lc.configure(scrollregion=_lc.bbox('all')))
        # Stretch inner frame to fill canvas width
        _lc.bind('<Configure>',
                 lambda e: _lc.itemconfig(_win, width=e.width))
        _lc.bind('<MouseWheel>',
                 lambda e: _lc.yview_scroll(-1 * (e.delta // 120), 'units'))
        left.bind('<MouseWheel>',
                  lambda e: _lc.yview_scroll(-1 * (e.delta // 120), 'units'))

        self._left_inner = left
        self._build_left(left)

    def _build_left(self, parent):
        # Clear (rebuilt when the Statistics/Pathway mode toggles)
        for w in parent.winfo_children():
            w.destroy()

        # ── Mode toggle: Statistics | Pathway ────────────────────────────────
        if not hasattr(self, '_mode_var'):
            self._mode_var = tk.StringVar(value='statistics')
        try:
            _style = ttk.Style()
            _style.configure('Mode.Toolbutton', borderwidth=1,
                             relief='raised', padding=4)
            _style.map('Mode.Toolbutton',
                       relief=[('selected', 'sunken')])
            _mode_style = 'Mode.Toolbutton'
        except Exception:
            _mode_style = 'Toolbutton'
        mode_bar = ttk.Frame(parent, relief='groove', borderwidth=1)
        mode_bar.pack(fill=tk.X, pady=(0, 6))
        # In lipidomics projects metabolic pathways do not apply to lipid
        # classes, so the third mode becomes "FA ratios" instead of "Pathway".
        _fa = self._is_lipidomics_project()
        ttk.Radiobutton(mode_bar, text='Statistics', value='statistics',
                        variable=self._mode_var, command=self._on_mode_change,
                        style=_mode_style).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Radiobutton(mode_bar, text=('FA ratios' if _fa else 'Pathway'),
                        value='pathway',
                        variable=self._mode_var, command=self._on_mode_change,
                        style=_mode_style).pack(side=tk.LEFT, expand=True, fill=tk.X)

        if self._mode_var.get() == 'pathway':
            if _fa:
                self._build_left_fa_ratios(parent)
            else:
                self._build_left_pathway(parent)
        else:
            self._build_left_statistics(parent)

    def _is_lipidomics_project(self):
        """True when the active annotation run used the Lipidomics library."""
        ann = (getattr(self.app, 'stages', {}) or {}).get('annotation')
        args = getattr(ann, '_last_results_args', None) or {}
        return str(args.get('internal_library_mode', '')) == 'Lipidomics'

    def _on_mode_change(self):
        self._build_left(self._left_inner)

    def _build_left_statistics(self, parent):
        _SEC = dict(padding=(6, 4, 6, 4), relief='groove')
        _RB  = dict(anchor=tk.W, padx=2, pady=1)

        # ── Method ───────────────────────────────────────────────────────────
        mf = ttk.LabelFrame(parent, text='Method', **_SEC)
        mf.pack(fill=tk.X, pady=(0, 4))
        self._method_var = tk.StringVar(value='PCA')
        for m in ['PCA', 'PLS-DA', 'OPLS-DA', 'Heatmap', 'Correlation',
                  'VIP Score', 'Volcano', 'Metabolic Profile',
                  'Box / Violin']:
            ttk.Radiobutton(mf, text=m, variable=self._method_var,
                            value=m, command=self._on_method_change
                            ).pack(**_RB)

        # ── Options (rebuilt per method) ──────────────────────────────────────
        self._opt_frame = ttk.LabelFrame(parent, text='Options', **_SEC)
        self._opt_frame.pack(fill=tk.X, pady=(0, 4))
        self._build_options('PCA')

        # ── Scaling ───────────────────────────────────────────────────────────
        self._build_scaling_frame(parent, _SEC, _RB)

        # ── Filters ───────────────────────────────────────────────────────────
        ff = ttk.LabelFrame(parent, text='Filters', **_SEC)
        ff.pack(fill=tk.X, pady=(0, 4))
        self._filter_samples_btn = ttk.Button(
            ff, text='Samples', command=self._open_sample_selector)
        self._filter_samples_btn.pack(fill=tk.X, padx=4, pady=2)
        self._filter_metabolites_btn = ttk.Button(
            ff, text='Metabolites', command=self._open_metabolite_selector)
        self._filter_metabolites_btn.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(ff, text='Normality check',
                   command=self._open_normality_popup
                   ).pack(fill=tk.X, padx=4, pady=2)
        self._filter_status_lbl = ttk.Label(
            ff, text='', font=('Arial', 7), foreground='#666')
        self._filter_status_lbl.pack(anchor=tk.W, padx=4, pady=(0, 2))

    def _build_scaling_frame(self, parent, _SEC, _RB):
        """Scaling/normalization controls. Shared by Statistics and Pathway
        panels so the same preprocessing applies in both modes; the vars persist
        across mode toggles (guarded by hasattr)."""
        sf = ttk.LabelFrame(parent, text='Scaling', **_SEC)
        sf.pack(fill=tk.X, pady=(0, 4))
        if not hasattr(self, '_log_pre_var'):
            self._log_pre_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, text='Log10 pre-transform',
                        variable=self._log_pre_var).pack(**_RB)
        ttk.Separator(sf, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=2, pady=3)
        if not hasattr(self, '_scale_var'):
            self._scale_var = tk.StringVar(value='pareto')
        for lbl, val in [
            ('None',           'none'),
            ('UV (unit var.)', 'UV'),
            ('Pareto',         'pareto'),
            ('Cube root',      'cbrt'),
            ('Median center',  'median'),
        ]:
            ttk.Radiobutton(sf, text=lbl, variable=self._scale_var,
                            value=val).pack(**_RB)

    # ── Pathway mode left panel ──────────────────────────────────────────────

    def _build_left_pathway(self, parent):
        _SEC = dict(padding=(6, 4, 6, 4), relief='groove')
        _RB  = dict(anchor=tk.W, padx=2, pady=1)

        # Persistent state (survives mode toggles; seeded from a restored
        # project via self._pw_saved when present)
        _sv = getattr(self, '_pw_saved', {}) or {}
        if not hasattr(self, '_pw_plot_var'):
            self._pw_plot_var = tk.StringVar(value=_sv.get('plot') or 'Overview enrichment')
        if not hasattr(self, '_pw_cat_var'):
            self._pw_cat_var = tk.StringVar(value=_sv.get('category') or 'metabolic')
        if not hasattr(self, '_pw_p_thr_var'):
            self._pw_p_thr_var = tk.DoubleVar(value=_sv.get('p_thr') or 0.05)
        if not hasattr(self, '_pw_minmets_var'):
            self._pw_minmets_var = tk.IntVar(value=_sv.get('min_mets') or 2)
        if not hasattr(self, '_pw_topn_var'):
            self._pw_topn_var = tk.IntVar(value=_sv.get('top_n') or 25)
        if not hasattr(self, '_pw_use_fdr_var'):
            self._pw_use_fdr_var = tk.BooleanVar(value=bool(_sv.get('use_fdr', False)))
        if not hasattr(self, '_pw_jaccard_var'):
            self._pw_jaccard_var = tk.DoubleVar(value=_sv.get('jaccard') or 0.20)
        if not hasattr(self, '_pw_sig_only_var'):
            self._pw_sig_only_var = tk.BooleanVar(value=bool(_sv.get('sig_only', False)))
        if not hasattr(self, '_pw_cmap_var'):
            self._pw_cmap_var = tk.StringVar(value=_sv.get('cmap') or 'YlOrRd_r')

        # ── Plot type ────────────────────────────────────────────────────────
        pf = ttk.LabelFrame(parent, text='Plot', **_SEC)
        pf.pack(fill=tk.X, pady=(0, 4))
        for m in ['Overview enrichment', 'Network', 'Dot plot', 'Lollipop',
                  'Dumbbell']:
            ttk.Radiobutton(pf, text=m, variable=self._pw_plot_var,
                            value=m, command=self._on_pw_plot_change).pack(**_RB)

        # ── Groups (A vs B; + Control row, shown only for Dumbbell) ──────────
        gf = ttk.LabelFrame(parent, text='Comparison', **_SEC)
        gf.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(gf, text='Group A:', font=('Arial', 8)
                  ).grid(row=0, column=0, sticky=tk.W)
        if not hasattr(self, '_grp_a_var'):
            self._grp_a_var = tk.StringVar(value=_sv.get('group_a') or '')
        self._cb_a = ttk.Combobox(gf, textvariable=self._grp_a_var,
                                  width=14, state='readonly')
        self._cb_a.grid(row=0, column=1, sticky=tk.W, padx=4, pady=1)
        ttk.Label(gf, text='Group B:', font=('Arial', 8)
                  ).grid(row=1, column=0, sticky=tk.W)
        if not hasattr(self, '_grp_b_var'):
            self._grp_b_var = tk.StringVar(value=_sv.get('group_b') or '')
        self._cb_b = ttk.Combobox(gf, textvariable=self._grp_b_var,
                                  width=14, state='readonly')
        self._cb_b.grid(row=1, column=1, sticky=tk.W, padx=4, pady=1)
        # Control rows — one per treatment; gridded/removed by
        # _on_pw_plot_change (Dumbbell only). A vs Control-A, B vs Control-B.
        self._pw_ctrla_lbl = ttk.Label(gf, text='Control A:', font=('Arial', 8))
        if not hasattr(self, '_grp_ca_var'):
            self._grp_ca_var = tk.StringVar(value=_sv.get('group_ca')
                                            or _sv.get('group_c') or '')
        self._cb_ca = ttk.Combobox(gf, textvariable=self._grp_ca_var,
                                   width=14, state='readonly')
        self._pw_ctrlb_lbl = ttk.Label(gf, text='Control B:', font=('Arial', 8))
        if not hasattr(self, '_grp_cb_var'):
            self._grp_cb_var = tk.StringVar(value=_sv.get('group_cb') or '')
        self._cb_cb = ttk.Combobox(gf, textvariable=self._grp_cb_var,
                                   width=14, state='readonly')
        self._pw_ctrl_hint = ttk.Label(
            gf, text='(Dumbbell: A vs Control-A, B vs Control-B)',
            font=('Arial', 7), foreground='#666')
        self._refresh_groups()
        self._on_pw_plot_change()

        # ── Options ──────────────────────────────────────────────────────────
        of = ttk.LabelFrame(parent, text='Options', **_SEC)
        of.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(of, text='Category:', font=('Arial', 8)
                  ).grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(of, textvariable=self._pw_cat_var,
                     values=['metabolic', 'drug', 'disease', 'all'],
                     width=11, state='readonly'
                     ).grid(row=0, column=1, sticky=tk.W, padx=4, pady=1)
        ttk.Label(of, text='Sig. p <:', font=('Arial', 8)
                  ).grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(of, from_=0.001, to=0.50, increment=0.01, format='%.3f',
                    textvariable=self._pw_p_thr_var, width=6
                    ).grid(row=1, column=1, sticky=tk.W, padx=4, pady=1)
        ttk.Label(of, text='Min mets:', font=('Arial', 8)
                  ).grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(of, from_=1, to=20, textvariable=self._pw_minmets_var,
                    width=6).grid(row=2, column=1, sticky=tk.W, padx=4, pady=1)
        ttk.Label(of, text='Top N:', font=('Arial', 8)
                  ).grid(row=3, column=0, sticky=tk.W)
        ttk.Spinbox(of, from_=3, to=100, textvariable=self._pw_topn_var,
                    width=6).grid(row=3, column=1, sticky=tk.W, padx=4, pady=1)
        ttk.Label(of, text='Jaccard ≥:', font=('Arial', 8)
                  ).grid(row=4, column=0, sticky=tk.W)
        ttk.Spinbox(of, from_=0.05, to=1.0, increment=0.05, format='%.2f',
                    textvariable=self._pw_jaccard_var, width=6
                    ).grid(row=4, column=1, sticky=tk.W, padx=4, pady=1)
        ttk.Label(of, text='Color map:', font=('Arial', 8)
                  ).grid(row=5, column=0, sticky=tk.W)
        ttk.Combobox(of, textvariable=self._pw_cmap_var,
                     values=['YlOrRd_r', 'RdPu_r', 'RdYlBu_r', 'viridis',
                             'plasma', 'magma', 'coolwarm', 'seismic'],
                     width=11, state='readonly'
                     ).grid(row=5, column=1, sticky=tk.W, padx=4, pady=1)
        ttk.Checkbutton(of, text='Color by FDR (else raw p)',
                        variable=self._pw_use_fdr_var
                        ).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        ttk.Checkbutton(of, text='Significant only (< 0.05)',
                        variable=self._pw_sig_only_var
                        ).grid(row=7, column=0, columnspan=2, sticky=tk.W)

        # ── Filters (shared sample/metabolite exclusion) ─────────────────────
        ff = ttk.LabelFrame(parent, text='Filters', **_SEC)
        ff.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(ff, text='Samples', command=self._open_sample_selector
                   ).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(ff, text='Metabolites', command=self._open_metabolite_selector
                   ).pack(fill=tk.X, padx=4, pady=2)
        self._filter_status_lbl = ttk.Label(
            ff, text='', font=('Arial', 7), foreground='#666')
        self._filter_status_lbl.pack(anchor=tk.W, padx=4, pady=(0, 2))
        self._update_filter_status()

        # Same scaling/normalization controls as Statistics — values persist and
        # are applied to the per-metabolite significance test below.
        self._build_scaling_frame(parent, _SEC, _RB)

        ttk.Label(parent,
                  text='ORA (Fisher) on metabolites significant\n'
                       'between Group A and B (t-test / Mann-Whitney).\n'
                       'Needs HMDB IDs on quantified metabolites.',
                  font=('Arial', 7), foreground='#666', justify=tk.LEFT
                  ).pack(anchor=tk.W, padx=4, pady=(2, 0))

    def _on_method_change(self):
        self._build_options(self._method_var.get())

    def _build_options(self, method):
        self._cb_a = None
        self._cb_b = None
        for w in self._opt_frame.winfo_children():
            w.destroy()

        if method in ('PCA',):
            ttk.Label(self._opt_frame, text='Components:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._n_comp_var = tk.IntVar(value=2)
            ttk.Spinbox(self._opt_frame, from_=2, to=10,
                        textvariable=self._n_comp_var, width=4
                        ).grid(row=0, column=1, sticky=tk.W, padx=4)
            self._biplot_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self._opt_frame, text='Show loadings (biplot)',
                            variable=self._biplot_var
                            ).grid(row=1, column=0, columnspan=2, sticky=tk.W)
            self._ellipse_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='95% confidence ellipses',
                            variable=self._ellipse_var
                            ).grid(row=2, column=0, columnspan=2, sticky=tk.W)

        elif method in ('PLS-DA', 'OPLS-DA'):
            ttk.Label(self._opt_frame, text='Components:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._n_comp_var = tk.IntVar(value=2)
            ttk.Spinbox(self._opt_frame, from_=1, to=10,
                        textvariable=self._n_comp_var, width=4
                        ).grid(row=0, column=1, sticky=tk.W, padx=4)
            self._cv_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='Show Q² (cross-validated)',
                            variable=self._cv_var
                            ).grid(row=1, column=0, columnspan=2, sticky=tk.W)
            self._ellipse_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='95% confidence ellipses',
                            variable=self._ellipse_var
                            ).grid(row=2, column=0, columnspan=2, sticky=tk.W)
            ttk.Label(self._opt_frame, text='Permutations:',
                      font=('Arial', 8)).grid(row=3, column=0, sticky=tk.W)
            self._n_perm_var = tk.IntVar(value=200)
            ttk.Spinbox(self._opt_frame, from_=0, to=2000, increment=100,
                        textvariable=self._n_perm_var, width=6
                        ).grid(row=3, column=1, sticky=tk.W, padx=4)

        elif method == 'Heatmap':
            ttk.Label(self._opt_frame, text='Clustering:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._clust_var = tk.StringVar(value='ward')
            ttk.Combobox(self._opt_frame, textvariable=self._clust_var,
                         values=['ward', 'complete', 'average', 'single'],
                         width=10, state='readonly'
                         ).grid(row=0, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Color map:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
            self._cmap_var = tk.StringVar(value='RdYlBu_r')
            ttk.Combobox(self._opt_frame, textvariable=self._cmap_var,
                         values=['RdYlBu_r', 'seismic', 'coolwarm', 'viridis',
                                 'plasma', 'RdBu_r'],
                         width=10, state='readonly'
                         ).grid(row=1, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Top N (VIP):',
                      font=('Arial', 8)).grid(row=2, column=0, sticky=tk.W)
            self._hm_topn_var = tk.StringVar(value='All')
            ttk.Combobox(self._opt_frame, textvariable=self._hm_topn_var,
                         values=['All', '10', '20', '30', '40', '50'],
                         width=10
                         ).grid(row=2, column=1, sticky=tk.W, padx=4)

            ttk.Label(self._opt_frame, text='Plot by:',
                      font=('Arial', 8)).grid(row=3, column=0, sticky=tk.W, pady=(4, 0))
            self._hm_plot_by_var = tk.StringVar(value='Samples')
            ttk.Combobox(self._opt_frame, textvariable=self._hm_plot_by_var,
                         values=['Samples', 'Group mean'],
                         width=12, state='readonly'
                         ).grid(row=3, column=1, sticky=tk.W, padx=4, pady=(4, 0))

        elif method == 'Correlation':
            ttk.Label(self._opt_frame, text='Coefficient:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            if not hasattr(self, '_corr_method_var'):
                self._corr_method_var = tk.StringVar(value='spearman')
            ttk.Combobox(self._opt_frame, textvariable=self._corr_method_var,
                         values=['spearman', 'pearson'],
                         width=10, state='readonly'
                         ).grid(row=0, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Linkage:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
            self._corr_link_var = tk.StringVar(value='average')
            ttk.Combobox(self._opt_frame, textvariable=self._corr_link_var,
                         values=['average', 'ward', 'complete', 'single'],
                         width=10, state='readonly'
                         ).grid(row=1, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Top N analytes:',
                      font=('Arial', 8)).grid(row=2, column=0, sticky=tk.W)
            self._corr_topn_var = tk.StringVar(value='30')
            ttk.Combobox(self._opt_frame, textvariable=self._corr_topn_var,
                         values=['All', '15', '20', '30', '40', '50'],
                         width=10
                         ).grid(row=2, column=1, sticky=tk.W, padx=4)
            self._corr_sig_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='Significant only (FDR<0.05)',
                            variable=self._corr_sig_var
                            ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
            # Differential panel: pick the two groups to contrast.
            _cg = sorted(set(s.get('label', s.get('group', 'Unknown'))
                             for s in getattr(self.app, 'samples_info', [])
                             if str(s.get('label', s.get('group', ''))).upper() != 'BLK'))
            ttk.Label(self._opt_frame, text='Diff. Group A:',
                      font=('Arial', 8)).grid(row=4, column=0, sticky=tk.W, pady=(4, 0))
            self._corr_g1_var = tk.StringVar(value=_cg[0] if _cg else '')
            self._corr_cb1 = ttk.Combobox(self._opt_frame, textvariable=self._corr_g1_var,
                                          values=_cg, width=10, state='readonly')
            self._corr_cb1.grid(row=4, column=1, sticky=tk.W, padx=4, pady=(4, 0))
            ttk.Label(self._opt_frame, text='Diff. Group B:',
                      font=('Arial', 8)).grid(row=5, column=0, sticky=tk.W)
            self._corr_g2_var = tk.StringVar(value=_cg[1] if len(_cg) > 1 else '')
            self._corr_cb2 = ttk.Combobox(self._opt_frame, textvariable=self._corr_g2_var,
                                          values=_cg, width=10, state='readonly')
            self._corr_cb2.grid(row=5, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Top N ranks by # significant\ncorrelations. Differential panel\ncontrasts the two groups above.',
                      font=('Arial', 7), foreground='#666'
                      ).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(3, 0))

            ttk.Separator(self._opt_frame, orient=tk.HORIZONTAL).grid(
                row=7, column=0, columnspan=2, sticky=tk.EW, pady=4)
            ttk.Label(self._opt_frame, text='Plot:',
                      font=('Arial', 8)).grid(row=8, column=0, sticky=tk.W)
            if not hasattr(self, '_corr_plot_var'):
                self._corr_plot_var = tk.StringVar(value='Heatmap')
            ttk.Combobox(self._opt_frame, textvariable=self._corr_plot_var,
                         values=['Heatmap', 'Linear (X vs Y)'],
                         width=14, state='readonly'
                         ).grid(row=8, column=1, sticky=tk.W, padx=4)
            _, _, _feat_corr, _ = _get_quantification_matrix(self.app)
            _feat_corr = _feat_corr or []
            ttk.Label(self._opt_frame, text='X analyte:',
                      font=('Arial', 8)).grid(row=9, column=0, sticky=tk.W, pady=(4, 0))
            if not hasattr(self, '_corr_x_var'):
                self._corr_x_var = tk.StringVar(value=_feat_corr[0] if _feat_corr else '')
            ttk.Combobox(self._opt_frame, textvariable=self._corr_x_var,
                         values=_feat_corr, width=14, state='readonly'
                         ).grid(row=9, column=1, sticky=tk.W, padx=4, pady=(4, 0))
            ttk.Label(self._opt_frame, text='Y analyte:',
                      font=('Arial', 8)).grid(row=10, column=0, sticky=tk.W)
            if not hasattr(self, '_corr_y_var'):
                self._corr_y_var = tk.StringVar(
                    value=_feat_corr[1] if len(_feat_corr) > 1 else '')
            ttk.Combobox(self._opt_frame, textvariable=self._corr_y_var,
                         values=_feat_corr, width=14, state='readonly'
                         ).grid(row=10, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame,
                      text='Linear: scatter X vs Y with OLS trend\n'
                           'line + 95% band; r/p per coefficient.',
                      font=('Arial', 7), foreground='#666'
                      ).grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(3, 0))

        elif method == 'VIP Score':
            ttk.Label(self._opt_frame, text='Top N features:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._top_n_var = tk.IntVar(value=20)
            ttk.Spinbox(self._opt_frame, from_=5, to=100, increment=5,
                        textvariable=self._top_n_var, width=5
                        ).grid(row=0, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='VIP threshold:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
            self._vip_thr_var = tk.DoubleVar(value=1.5)
            ttk.Spinbox(self._opt_frame, from_=0.5, to=3.0, increment=0.1,
                        textvariable=self._vip_thr_var, width=5, format='%.1f'
                        ).grid(row=1, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Class color map:',
                      font=('Arial', 8)).grid(row=2, column=0, sticky=tk.W)
            self._vip_cmap_var = tk.StringVar(value='RdBu_r')
            ttk.Combobox(self._opt_frame, textvariable=self._vip_cmap_var,
                         values=['RdBu_r', 'RdYlBu_r', 'seismic', 'coolwarm',
                                 'viridis', 'plasma'],
                         width=10, state='readonly'
                         ).grid(row=2, column=1, sticky=tk.W, padx=4)

        elif method == 'Volcano':
            ttk.Label(self._opt_frame, text='Group A:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            # Reuse an existing var so the user's group choice survives an options
            # rebuild (recreating it empty would silently reset both groups to
            # the defaults → every Run compares the same pair).
            if not hasattr(self, '_grp_a_var'):
                self._grp_a_var = tk.StringVar()
            self._cb_a = ttk.Combobox(self._opt_frame, textvariable=self._grp_a_var,
                                       width=12, state='readonly')
            self._cb_a.grid(row=0, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Group B:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
            if not hasattr(self, '_grp_b_var'):
                self._grp_b_var = tk.StringVar()
            self._cb_b = ttk.Combobox(self._opt_frame, textvariable=self._grp_b_var,
                                       width=12, state='readonly')
            self._cb_b.grid(row=1, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='FC threshold:',
                      font=('Arial', 8)).grid(row=2, column=0, sticky=tk.W)
            self._fc_thr_var = tk.DoubleVar(value=1.5)
            ttk.Spinbox(self._opt_frame, from_=1.0, to=10.0, increment=0.5,
                        textvariable=self._fc_thr_var, width=5, format='%.1f'
                        ).grid(row=2, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='p-value thr:',
                      font=('Arial', 8)).grid(row=3, column=0, sticky=tk.W)
            self._p_thr_var = tk.DoubleVar(value=0.05)
            ttk.Spinbox(self._opt_frame, from_=0.001, to=0.20, increment=0.01,
                        textvariable=self._p_thr_var, width=5, format='%.3f'
                        ).grid(row=3, column=1, sticky=tk.W, padx=4)
            self._p_adj_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='FDR correction (BH)',
                            variable=self._p_adj_var
                            ).grid(row=4, column=0, columnspan=2, sticky=tk.W)
            # Persistent label/colour style (survives option rebuilds)
            self._ensure_volcano_style()
            # Gear button pinned to the top-right corner of the Options frame
            _gear = ttk.Button(self._opt_frame, text='⚙', width=3,
                               command=self._open_volcano_style_dialog)
            _gear.place(relx=1.0, rely=0.0, anchor='ne', x=-2, y=-2)
            # Populate group comboboxes
            self._refresh_groups()

        elif method == 'Metabolic Profile':
            ttk.Label(self._opt_frame, text='Chart type:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._pie_type_var = tk.StringVar(value='Donut')
            ttk.Combobox(self._opt_frame, textvariable=self._pie_type_var,
                         values=['Donut', 'Pie'], width=8, state='readonly'
                         ).grid(row=0, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Top N metabolites:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
            self._pie_topn_var = tk.IntVar(value=10)
            ttk.Spinbox(self._opt_frame, from_=3, to=50, increment=1,
                        textvariable=self._pie_topn_var, width=5
                        ).grid(row=1, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='"Other" threshold (%):',
                      font=('Arial', 8)).grid(row=2, column=0, sticky=tk.W)
            self._pie_other_var = tk.DoubleVar(value=3.0)
            ttk.Spinbox(self._opt_frame, from_=0.0, to=20.0, increment=0.5,
                        textvariable=self._pie_other_var, width=5, format='%.1f'
                        ).grid(row=2, column=1, sticky=tk.W, padx=4)
            self._pie_pct_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='Show percentages',
                            variable=self._pie_pct_var
                            ).grid(row=3, column=0, columnspan=2, sticky=tk.W)
            self._pie_legend_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='Show legend',
                            variable=self._pie_legend_var
                            ).grid(row=4, column=0, columnspan=2, sticky=tk.W)
            ttk.Button(self._opt_frame, text='Edit Style…',
                       command=self._open_pie_style_dialog
                       ).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))

        elif method == 'Box / Violin':
            ttk.Label(self._opt_frame,
                      text='Pick the panel grid and which\n'
                           'metabolites to plot, then Apply.',
                      font=('Arial', 8), foreground='#555', justify=tk.LEFT
                      ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
            self._box_cfg_status = ttk.Label(
                self._opt_frame, text='', font=('Arial', 7), foreground='#666')
            self._box_cfg_status.grid(row=1, column=0, columnspan=2, sticky=tk.W,
                                      pady=(2, 0))
            ttk.Button(self._opt_frame, text='Configure…',
                       command=self._open_box_config_dialog
                       ).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
            ttk.Button(self._opt_frame, text='Edit Style…',
                       command=self._open_box_style_dialog
                       ).grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))
            self._update_box_cfg_status()

    def _on_pw_plot_change(self):
        """Show the two Control rows only for the Dumbbell plot; hide otherwise."""
        is_db = (getattr(self, '_pw_plot_var', None)
                 and self._pw_plot_var.get() == 'Dumbbell')
        for w, r, c, kw in (
            (getattr(self, '_pw_ctrla_lbl', None), 2, 0, dict(sticky=tk.W)),
            (getattr(self, '_cb_ca', None),        2, 1, dict(sticky=tk.W, padx=4, pady=1)),
            (getattr(self, '_pw_ctrlb_lbl', None), 3, 0, dict(sticky=tk.W)),
            (getattr(self, '_cb_cb', None),        3, 1, dict(sticky=tk.W, padx=4, pady=1)),
            (getattr(self, '_pw_ctrl_hint', None), 4, 0,
             dict(columnspan=2, sticky=tk.W)),
        ):
            if w is None:
                continue
            try:
                if is_db:
                    w.grid(row=r, column=c, **kw)
                else:
                    w.grid_remove()
            except Exception:
                pass

    def _dumbbell_pathway_map(self, feature_labels, category, min_mets=2):
        """metabolite name → representative pathway name (within ``category``).
        Each metabolite is assigned to the pathway it shares with the most other
        analytes; pathways with fewer than ``min_mets`` of our metabolites
        collapse to 'Other'."""
        import os, sqlite3
        from collections import Counter
        from app_paths import databases_dir
        try:
            qr = self._pathway_quant_results(feature_labels)
            id_by_name = {r['metabolite']: str(r.get('hmdb_id') or '') for r in qr}
        except Exception:
            id_by_name = {}
        pw_of = {n: [] for n in feature_labels}
        cat = str(category).lower()
        try:
            conn = sqlite3.connect(os.path.join(databases_dir(), 'hmdb_pathways.db'))
            cur = conn.cursor()
            for name in feature_labels:
                hid = id_by_name.get(name, '')
                if not hid.upper().startswith('HMDB'):
                    continue
                if cat in ('', 'all'):
                    q = ('SELECT p.name FROM metabolite_pathways mp '
                         'JOIN pathways p ON p.id = mp.pathway_id '
                         'WHERE mp.hmdb_id = ?')
                    args = (hid,)
                else:
                    q = ('SELECT p.name FROM metabolite_pathways mp '
                         'JOIN pathways p ON p.id = mp.pathway_id '
                         'WHERE mp.hmdb_id = ? AND p.category = ?')
                    args = (hid, cat)
                pw_of[name] = [r[0] for r in cur.execute(q, args).fetchall() if r[0]]
            conn.close()
        except Exception:
            pass
        freq = Counter()
        for ps in pw_of.values():
            for p in set(ps):
                freq[p] += 1
        assign = {}
        for name, ps in pw_of.items():
            best, bf = None, 0
            for p in ps:
                if freq[p] > bf or (freq[p] == bf and (best is None or p < best)):
                    bf, best = freq[p], p
            assign[name] = (best if (best is not None
                                     and freq[best] >= max(1, int(min_mets)))
                            else 'Other')
        return assign

    def _dumbbell_category_map(self, feature_labels):
        """metabolite name → pathway category (Metabolic/Disease/Drug/Other)."""
        import os, sqlite3
        from app_paths import databases_dir
        try:
            qr = self._pathway_quant_results(feature_labels)
            id_by_name = {r['metabolite']: str(r.get('hmdb_id') or '') for r in qr}
        except Exception:
            id_by_name = {}
        cat = {n: 'Other' for n in feature_labels}
        try:
            conn = sqlite3.connect(os.path.join(databases_dir(), 'hmdb_pathways.db'))
            cur = conn.cursor()
            for name in feature_labels:
                hid = id_by_name.get(name, '')
                if hid.upper().startswith('HMDB'):
                    row = cur.execute(
                        'SELECT p.category, COUNT(*) c FROM metabolite_pathways mp '
                        'JOIN pathways p ON p.id = mp.pathway_id '
                        'WHERE mp.hmdb_id = ? GROUP BY p.category '
                        'ORDER BY c DESC LIMIT 1', (hid,)).fetchone()
                    if row and row[0]:
                        cat[name] = str(row[0]).capitalize()
            conn.close()
        except Exception:
            pass
        return cat

    def _plot_dumbbell(self, X, feature_labels, group_labels,
                       ctrl_a, ctrl_b, ga, gb, p_thr=0.05, category='all'):
        """Dumbbell plot: log2 FC of treatment A vs Control-A and B vs Control-B
        per metabolite, joined by a line; filled ▲ = significant (p<thr vs its
        own control). Restricted to one pathway ``category`` (or 'all')."""
        from scipy import stats as spstats
        if not ga or not gb or not ctrl_a or not ctrl_b:
            raise ValueError('Pick Group A, Group B and a Control for each.')
        if ga == gb:
            raise ValueError('Group A and Group B must differ.')

        # Treatment colours from the Options colormap (low end = A, high = B).
        try:
            import matplotlib as _mpl
            _cm = _mpl.colormaps[self._pw_cmap_var.get()]
            col_a, col_b = _cm(0.15), _cm(0.85)
        except Exception:
            col_a, col_b = '#d1495b', '#1565c0'

        gl = list(group_labels)
        idx_ca = [i for i, g in enumerate(gl) if g == ctrl_a]
        idx_cb = [i for i, g in enumerate(gl) if g == ctrl_b]
        idx_a = [i for i, g in enumerate(gl) if g == ga]
        idx_b = [i for i, g in enumerate(gl) if g == gb]
        # Name exactly which role is short (and its count) — the usual cause is a
        # sample exclusion that dropped a group below 2, which the generic message
        # hid. A count of 0 also flags a stale selection no longer in the data.
        _short = [f'{role}="{name}": {len(idx)} sample(s)'
                  for role, name, idx in (('Group A', ga, idx_a),
                                          ('Group B', gb, idx_b),
                                          ('Control A', ctrl_a, idx_ca),
                                          ('Control B', ctrl_b, idx_cb))
                  if len(idx) < 2]
        if _short:
            raise ValueError(
                'Each of Group A / B / Control A / Control B needs ≥2 samples.\n'
                'Too few after sample exclusion:\n  ' + '\n  '.join(_short))

        # Group metabolites by PATHWAY (within the chosen category); a band needs
        # ≥ Min mets (Options) metabolites, else they go to 'Other'.
        try:
            _min_mets = int(self._pw_minmets_var.get())
        except Exception:
            _min_mets = 2
        pmap = self._dumbbell_pathway_map(feature_labels, category, _min_mets)
        keep = list(range(len(feature_labels)))
        if category and str(category).lower() != 'all':
            keep = [j for j, n in enumerate(feature_labels)
                    if pmap.get(n, 'Other') != 'Other']
            if not keep:
                raise ValueError(f'No metabolites with a "{category}" pathway.')

        def _fc_p(treat_idx, ctrl_idx, j):
            c = X[ctrl_idx, j]; t = X[treat_idx, j]
            c = c[np.isfinite(c)]; t = t[np.isfinite(t)]
            if c.size < 2 or t.size < 2:
                return 0.0, 1.0
            mc, mt = float(np.mean(c)), float(np.mean(t))
            l2 = float(np.log2(mt / mc)) if (mc > 0 and mt > 0) else 0.0
            try:
                if c.size >= 3 and t.size >= 3:
                    _, p = spstats.ttest_ind(t, c, equal_var=False)
                else:
                    _, p = spstats.mannwhitneyu(t, c, alternative='two-sided')
                p = float(p)
                if not np.isfinite(p):
                    p = 1.0
            except Exception:
                p = 1.0
            return l2, p

        rows = []   # (name, pathway, l2a, pa, l2b, pb)
        for j in keep:
            name = feature_labels[j]
            l2a, pa = _fc_p(idx_a, idx_ca, j)
            l2b, pb = _fc_p(idx_b, idx_cb, j)
            rows.append([name, pmap.get(name, 'Other'), l2a, pa, l2b, pb])
        if not rows:
            raise ValueError('No metabolites to plot.')

        # Cap (Options → Top N) keeping the strongest by |log2FC|.
        try:
            MAXN = int(self._pw_topn_var.get())
        except Exception:
            MAXN = 45
        MAXN = max(3, MAXN)
        if len(rows) > MAXN:
            rows = sorted(rows, key=lambda r: -max(abs(r[2]), abs(r[4])))[:MAXN]

        # CSV result: per-metabolite log2 fold change + p vs each control.
        self._plot_export = {
            'method': 'Dumbbell',
            'headers': ['metabolite', 'pathway',
                        f'log2fc_{ga}_vs_{ctrl_a}', f'p_{ga}',
                        f'log2fc_{gb}_vs_{ctrl_b}', f'p_{gb}'],
            'rows': [[r[0], r[1], float(r[2]), float(r[3]),
                      float(r[4]), float(r[5])] for r in rows],
        }

        # Order: pathway bands ('Other' last); within a band by mean log2FC.
        from collections import Counter
        from itertools import groupby
        _bsize = Counter(r[1] for r in rows)
        def _band_key(pw):
            return (pw == 'Other', -_bsize[pw], pw)
        rows.sort(key=lambda r: (_band_key(r[1]), -((r[2] + r[4]) / 2.0)))

        # One stacked panel (facet) per pathway, small gap between them.
        bands = [(pw, list(grp)) for pw, grp in groupby(rows, key=lambda r: r[1])]
        nb = len(bands)

        # Figure height scales with content so panels/labels are not squashed
        # (the viewer scrolls). Width follows the container.
        _tot = sum(len(b[1]) for b in bands)
        self._dumbbell_export_figsize = (
            9.5, max(6.0, _tot * 0.34 + nb * 0.55 + 1.2))
        all_l2 = [v for r in rows for v in (r[2], r[4])]
        xlo, xhi = (min(all_l2 + [0.0]), max(all_l2 + [0.0]))
        pad = max((xhi - xlo) * 0.12, 0.05)
        xlo, xhi = xlo - pad, xhi + pad

        gs = self._fig.add_gridspec(
            nb, 1, hspace=0.45,
            height_ratios=[len(b[1]) + 0.7 for b in bands])
        self._fig.suptitle(
            f'{ga} (A) vs {ctrl_a}   ·   {gb} (B) vs {ctrl_b}',
            fontsize=10, fontweight='bold')

        from matplotlib.lines import Line2D
        ax0 = None
        for k, (pw, brows) in enumerate(bands):
            axk = self._fig.add_subplot(gs[k, 0])
            if ax0 is None:
                ax0 = axk
            ny = len(brows)
            yp = list(range(ny))[::-1]
            for y, r in zip(yp, brows):
                _, _, l2a, pa, l2b, pb = r
                axk.plot([l2a, l2b], [y, y], color='#bbbbbb', lw=1.2, zorder=2)
                for l2, p, col in ((l2a, pa, col_a), (l2b, pb, col_b)):
                    if p < p_thr:
                        axk.scatter(l2, y, marker='^', s=50, color=col,
                                    edgecolors=col, zorder=3)
                    else:
                        axk.scatter(l2, y, marker='o', s=38, facecolors='white',
                                    edgecolors=col, linewidths=1.3, zorder=3)
            axk.axvline(0.0, color='#888', ls='--', lw=1.0, zorder=0)
            axk.set_xlim(xlo, xhi)
            axk.set_ylim(-0.6, ny - 0.4)
            axk.set_yticks(yp)
            axk.set_yticklabels([r[0] for r in brows], fontsize=7)
            axk.set_title(pw, loc='left', fontsize=8, fontweight='bold',
                          color='#1a1a2e', pad=2)
            axk.tick_params(labelsize=7, labelbottom=True)   # x scale on every panel

        # Single shared x-axis label at the figure bottom (always visible).
        self._fig.supxlabel('log2 Fold Change vs Control', fontsize=9)

        # Legends once, close to the top-right of the figure.
        sig_leg = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='#555', markersize=8, label='ns'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#555',
                   markeredgecolor='#555', markersize=9, label=f'p < {p_thr:g}'),
        ]
        trt_leg = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=col_a,
                   markersize=9, label=f'A: {ga}'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=col_b,
                   markersize=9, label=f'B: {gb}'),
        ]
        if ax0 is not None:
            leg1 = ax0.legend(handles=sig_leg, title='Significance', fontsize=7,
                              title_fontsize=8, loc='upper left',
                              bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0,
                              framealpha=0.9)
            ax0.add_artist(leg1)
            ax0.legend(handles=trt_leg, title='Treatment', fontsize=7,
                       title_fontsize=8, loc='upper left',
                       bbox_to_anchor=(1.005, 0.4), borderaxespad=0.0,
                       framealpha=0.9)

    def _open_pie_style_dialog(self):
        from tkinter import colorchooser
        st = self._pie_style

        dlg = tk.Toplevel(self.frame)
        dlg.title('Edit Metabolic Profile Style')
        dlg.resizable(True, True)
        dlg.grab_set()
        dlg.update_idletasks()
        px = self.frame.winfo_rootx() + self.frame.winfo_width() // 2
        py = self.frame.winfo_rooty() + self.frame.winfo_height() // 2
        dlg.geometry(f'480x620+{px - 240}+{py - 310}')

        nb = ttk.Notebook(dlg)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Tab 1: Chart title ────────────────────────────────────────────────
        tab_title = ttk.Frame(nb, padding=10)
        nb.add(tab_title, text='Title')

        ttk.Label(tab_title, text='Chart title text:').grid(row=0, column=0, sticky=tk.W, pady=3)
        _title_var = tk.StringVar(value=st['title'])
        ttk.Entry(tab_title, textvariable=_title_var, width=32).grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=4)

        ttk.Label(tab_title, text='Font size:').grid(row=1, column=0, sticky=tk.W, pady=3)
        _title_size_var = tk.IntVar(value=st['title_size'])
        ttk.Spinbox(tab_title, from_=6, to=28, textvariable=_title_size_var, width=5
                    ).grid(row=1, column=1, sticky=tk.W, padx=4)
        _title_bold_var = tk.BooleanVar(value=st['title_bold'])
        ttk.Checkbutton(tab_title, text='Bold', variable=_title_bold_var
                        ).grid(row=1, column=2, sticky=tk.W)

        ttk.Separator(tab_title, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=8)

        ttk.Label(tab_title, text='Group subtitle size:').grid(row=3, column=0, sticky=tk.W, pady=3)
        _grp_title_size_var = tk.IntVar(value=st['group_title_size'])
        ttk.Spinbox(tab_title, from_=6, to=20, textvariable=_grp_title_size_var, width=5
                    ).grid(row=3, column=1, sticky=tk.W, padx=4)
        _grp_title_bold_var = tk.BooleanVar(value=st['group_title_bold'])
        ttk.Checkbutton(tab_title, text='Bold', variable=_grp_title_bold_var
                        ).grid(row=3, column=2, sticky=tk.W)

        tab_title.columnconfigure(1, weight=1)

        # ── Tab 2: Per-group titles ───────────────────────────────────────────
        tab_groups = ttk.Frame(nb, padding=10)
        nb.add(tab_groups, text='Group Labels')

        ttk.Label(tab_groups, text='Custom label for each group\n(leave blank to use original name)',
                  font=('Arial', 8), foreground='#555').pack(anchor=tk.W, pady=(0, 6))

        si = getattr(self.app, 'samples_info', []) or []
        _all_groups = sorted(set(s.get('label', s.get('group', 'Unknown')) for s in si))
        _grp_title_vars = {}
        grp_scroll_frame = ttk.Frame(tab_groups)
        grp_scroll_frame.pack(fill=tk.BOTH, expand=True)
        for i, grp in enumerate(_all_groups):
            ttk.Label(grp_scroll_frame, text=grp, width=18, anchor=tk.W
                      ).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=st['group_titles'].get(grp, ''))
            ttk.Entry(grp_scroll_frame, textvariable=var, width=22
                      ).grid(row=i, column=1, sticky=tk.EW, padx=4)
            _grp_title_vars[grp] = var
        grp_scroll_frame.columnconfigure(1, weight=1)

        # ── Tab 3: Border & layout ────────────────────────────────────────────
        tab_border = ttk.Frame(nb, padding=10)
        nb.add(tab_border, text='Border & Layout')

        ttk.Label(tab_border, text='Wedge border color:').grid(row=0, column=0, sticky=tk.W, pady=4)
        _edge_color_var = tk.StringVar(value=st['edge_color'])
        _edge_swatch = tk.Label(tab_border, bg=st['edge_color'], width=4, relief='solid')
        _edge_swatch.grid(row=0, column=1, padx=4, sticky=tk.W)
        def _pick_edge():
            c = colorchooser.askcolor(color=_edge_color_var.get(), parent=dlg, title='Border color')
            if c and c[1]:
                _edge_color_var.set(c[1])
                _edge_swatch.config(bg=c[1])
        ttk.Button(tab_border, text='Choose…', command=_pick_edge
                   ).grid(row=0, column=2, padx=4, sticky=tk.W)

        ttk.Label(tab_border, text='Border width:').grid(row=1, column=0, sticky=tk.W, pady=4)
        _edge_width_var = tk.DoubleVar(value=st['edge_width'])
        ttk.Spinbox(tab_border, from_=0.0, to=5.0, increment=0.2, format='%.1f',
                    textvariable=_edge_width_var, width=6
                    ).grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=4)

        ttk.Separator(tab_border, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=8)

        ttk.Label(tab_border, text='Layout padding (tight_pad):').grid(row=3, column=0, sticky=tk.W, pady=4)
        _tight_pad_var = tk.DoubleVar(value=st['tight_pad'])
        ttk.Spinbox(tab_border, from_=0.1, to=5.0, increment=0.1, format='%.1f',
                    textvariable=_tight_pad_var, width=6
                    ).grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=4)
        ttk.Label(tab_border, text='(smaller = tighter, more space for legend)',
                  font=('Arial', 7), foreground='#666'
                  ).grid(row=4, column=0, columnspan=3, sticky=tk.W)

        tab_border.columnconfigure(0, weight=1)

        # ── Tab 4: Metabolite colors ──────────────────────────────────────────
        tab_colors = ttk.Frame(nb, padding=6)
        nb.add(tab_colors, text='Metabolite Colors')

        ttk.Label(tab_colors, text='Click a swatch to change color:',
                  font=('Arial', 8), foreground='#555').pack(anchor=tk.W, pady=(0, 4))

        outer = ttk.Frame(tab_colors)
        outer.pack(fill=tk.BOTH, expand=True)
        cvs = tk.Canvas(outer, highlightthickness=0)
        sb2 = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=cvs.yview)
        cvs.configure(yscrollcommand=sb2.set)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner2 = ttk.Frame(cvs)
        cw2 = cvs.create_window((0, 0), window=inner2, anchor='nw')
        inner2.bind('<Configure>', lambda e: (cvs.configure(scrollregion=cvs.bbox('all')),
                                              cvs.itemconfig(cw2, width=cvs.winfo_width())))
        cvs.bind('<Configure>', lambda e: cvs.itemconfig(cw2, width=e.width))
        cvs.bind('<Enter>', lambda e: cvs.bind_all('<MouseWheel>',
            lambda ev: cvs.yview_scroll(-1*(ev.delta//120), 'units')))
        cvs.bind('<Leave>', lambda e: cvs.unbind_all('<MouseWheel>'))

        # Get metabolite list from quantification matrix
        _, _, _feat, _ = _get_quantification_matrix(self.app)
        _feat = _feat or []

        # Build random colors the same way as the plot (seed=42)
        import colorsys
        rng2 = __import__('numpy').random.default_rng(seed=42)
        def _rnd():
            h = rng2.uniform(0, 1); s = rng2.uniform(0.45, 0.85); v = rng2.uniform(0.65, 0.92)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        _auto_colors = {f: _rnd() for f in _feat}

        _met_color_vars = {}
        _swatches = {}

        def _make_met_row(i, met_name):
            cur = st['met_colors'].get(met_name, _auto_colors.get(met_name, '#888888'))
            var = tk.StringVar(value=cur)
            _met_color_vars[met_name] = var
            row_f = ttk.Frame(inner2)
            row_f.pack(fill=tk.X, pady=1)
            sw = tk.Label(row_f, bg=cur, width=3, relief='solid', cursor='hand2')
            sw.pack(side=tk.LEFT, padx=(2, 4))
            _swatches[met_name] = sw
            ttk.Label(row_f, text=met_name, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
            def _pick_met(mn=met_name, sv=var, s=sw):
                c = colorchooser.askcolor(color=sv.get(), parent=dlg, title=f'Color: {mn}')
                if c and c[1]:
                    sv.set(c[1]); s.config(bg=c[1])
            sw.bind('<Button-1>', lambda e, fn=_pick_met: fn())
            ttk.Button(row_f, text='Reset', width=5,
                       command=lambda mn=met_name, sv=var, s=sw: (
                           sv.set(_auto_colors.get(mn, '#888888')),
                           s.config(bg=_auto_colors.get(mn, '#888888'))
                       )).pack(side=tk.RIGHT, padx=2)

        for i, f in enumerate(_feat):
            _make_met_row(i, f)

        ttk.Button(tab_colors, text='Reset all to auto',
                   command=lambda: [(_met_color_vars[m].set(_auto_colors.get(m, '#888888')),
                                     _swatches[m].config(bg=_auto_colors.get(m, '#888888')))
                                    for m in _met_color_vars]
                   ).pack(side=tk.BOTTOM, anchor=tk.E, pady=4, padx=4)

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        def _apply():
            st['title']            = _title_var.get()
            st['title_size']       = int(_title_size_var.get())
            st['title_bold']       = _title_bold_var.get()
            st['group_title_size'] = int(_grp_title_size_var.get())
            st['group_title_bold'] = _grp_title_bold_var.get()
            st['group_titles']     = {g: v.get() for g, v in _grp_title_vars.items() if v.get()}
            st['edge_color']       = _edge_color_var.get()
            st['edge_width']       = float(_edge_width_var.get())
            st['tight_pad']        = float(_tight_pad_var.get())
            st['met_colors']       = {m: v.get() for m, v in _met_color_vars.items()
                                      if v.get() != _auto_colors.get(m)}

        def _ok():
            _apply(); dlg.destroy()

        ttk.Button(btn_row, text='OK',     command=_ok      ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text='Apply',  command=_apply   ).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_row, text='Cancel', command=dlg.destroy).pack(side=tk.RIGHT)

    def _refresh_groups(self):
        si = getattr(self.app, 'samples_info', [])
        groups = sorted(set(s.get('label', s.get('group', 'Unknown')) for s in si))
        for cb in (getattr(self, '_cb_a', None), getattr(self, '_cb_b', None),
                   getattr(self, '_cb_ca', None), getattr(self, '_cb_cb', None)):
            if cb is not None:
                try:
                    if cb.winfo_exists():
                        cb['values'] = groups
                except Exception:
                    pass
        if groups:
            if hasattr(self, '_grp_a_var') and not self._grp_a_var.get():
                self._grp_a_var.set(groups[0])
            if hasattr(self, '_grp_b_var') and not self._grp_b_var.get():
                self._grp_b_var.set(groups[-1] if len(groups) > 1 else groups[0])
            if hasattr(self, '_grp_ca_var') and not self._grp_ca_var.get():
                self._grp_ca_var.set(groups[1] if len(groups) > 2 else groups[0])
            if hasattr(self, '_grp_cb_var') and not self._grp_cb_var.get():
                self._grp_cb_var.set(groups[2] if len(groups) > 3 else
                                     (groups[1] if len(groups) > 1 else groups[0]))

    def _update_filter_status(self):
        parts = []
        if self._excluded_samples:
            parts.append(f'{len(self._excluded_samples)} sample(s) excluded')
        if self._excluded_metabolites:
            parts.append(f'{len(self._excluded_metabolites)} metabolite(s) excluded')
        lbl = getattr(self, '_filter_status_lbl', None)
        if lbl:
            try:
                lbl.config(text=', '.join(parts) if parts else '')
            except Exception:
                pass

    def _open_sample_selector(self):
        si = getattr(self.app, 'samples_info', []) or []
        all_names = [s.get('name', f'S{i}') for i, s in enumerate(si)]
        if not all_names:
            messagebox.showinfo('No samples', 'No samples loaded.', parent=self.frame)
            return
        self._open_selector_dialog(
            title='Select Samples to Include',
            all_items=all_names,
            excluded_set=self._excluded_samples,
            on_confirm=lambda excl: self._set_excluded_samples(excl),
        )

    def _open_metabolite_selector(self):
        X, sample_names, feature_labels, group_labels = _get_quantification_matrix(self.app)
        all_mets = feature_labels or []
        if not all_mets:
            messagebox.showinfo('No metabolites', 'No metabolite data found.', parent=self.frame)
            return
        self._open_selector_dialog(
            title='Select Metabolites to Include',
            all_items=all_mets,
            excluded_set=self._excluded_metabolites,
            on_confirm=lambda excl: self._set_excluded_metabolites(excl),
        )

    def _set_excluded_samples(self, excl):
        self._excluded_samples = excl
        self._update_filter_status()

    def _set_excluded_metabolites(self, excl):
        self._excluded_metabolites = excl
        self._update_filter_status()

    def _open_selector_dialog(self, title, all_items, excluded_set, on_confirm):
        dlg = tk.Toplevel(self.frame)
        dlg.title(title)
        dlg.resizable(True, True)
        dlg.grab_set()

        # Center over parent
        dlg.update_idletasks()
        px = self.frame.winfo_rootx() + self.frame.winfo_width() // 2
        py = self.frame.winfo_rooty() + self.frame.winfo_height() // 2
        w, h = 320, 400
        dlg.geometry(f'{w}x{h}+{px - w//2}+{py - h//2}')

        ttk.Label(dlg, text='Check items to INCLUDE in analysis:',
                  font=('Arial', 8)).pack(anchor=tk.W, padx=8, pady=(8, 2))

        # Search bar
        search_var = tk.StringVar()
        search_entry = ttk.Entry(dlg, textvariable=search_var)
        search_entry.pack(fill=tk.X, padx=8, pady=(0, 4))

        # Scrollable checkbox list
        container = ttk.Frame(dlg)
        container.pack(fill=tk.BOTH, expand=True, padx=8)
        canvas = tk.Canvas(container, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_resize(e):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
        inner.bind('<Configure>', _on_inner_resize)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>',
            lambda ev: canvas.yview_scroll(-1*(ev.delta//120), 'units')))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

        check_vars = {}
        row_frames = {}

        def _populate(filter_text=''):
            for w in inner.winfo_children():
                w.destroy()
            ft = filter_text.lower()
            for item in all_items:
                if ft and ft not in item.lower():
                    continue
                var = check_vars.setdefault(item, tk.BooleanVar(value=(item not in excluded_set)))
                f = ttk.Frame(inner)
                f.pack(fill=tk.X)
                ttk.Checkbutton(f, text=item, variable=var).pack(anchor=tk.W, padx=4)
                row_frames[item] = f

        _populate()
        search_var.trace_add('write', lambda *a: _populate(search_var.get()))

        # Select all / none
        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=8, pady=(2, 0))
        def _select_all():
            for v in check_vars.values(): v.set(True)
        def _select_none():
            for v in check_vars.values(): v.set(False)
        ttk.Button(btn_row, text='All',  command=_select_all ).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text='None', command=_select_none).pack(side=tk.LEFT, padx=2)

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        def _confirm():
            new_excluded = {item for item, var in check_vars.items() if not var.get()}
            on_confirm(new_excluded)
            dlg.destroy()

        bottom = ttk.Frame(dlg)
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(bottom, text='OK',     command=_confirm       ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text='Cancel', command=dlg.destroy    ).pack(side=tk.RIGHT)

    def _open_normality_popup(self):
        """Per-sample profile QC + Grubbs outlier detection within each group.

        Shapiro-Wilk on each sample's log-transformed analyte profile is a
        whole-sample QC metric (is this sample's overall distribution as
        expected?), NOT a parametric-vs-nonparametric gate — the group tests
        choose by sample size, not by this flag.

        Grubbs' G on the Euclidean distance from the group centroid (log+z-score
        space) flags samples with anomalous metabolite profiles regardless of
        overall concentration magnitude. Because distances are non-negative,
        the test is one-sided (upper tail) and iterative — see
        processing/outlier_detection.
        """
        from scipy import stats as spstats
        import csv

        X, sample_names, feature_labels, group_labels = _get_quantification_matrix(self.app)
        if X is None or X.shape[0] == 0:
            messagebox.showinfo('No data', 'No sample data available.', parent=self.frame)
            return

        X, feature_labels = self._remove_nan(X, feature_labels)

        si = getattr(self.app, 'samples_info', []) or []
        all_names = [s.get('name', f'S{i}') for i, s in enumerate(si)]
        while len(all_names) < X.shape[0]:
            all_names.append(f'S{len(all_names)}')
        all_names = all_names[:X.shape[0]]

        all_groups = list(group_labels[:X.shape[0]])
        while len(all_groups) < X.shape[0]:
            all_groups.append('Unknown')

        if self._excluded_metabolites:
            keep = [i for i, f in enumerate(feature_labels) if f not in self._excluded_metabolites]
            X = X[:, keep]
            feature_labels = [feature_labels[i] for i in keep]

        if self._excluded_samples:
            keep = [i for i, n in enumerate(all_names) if n not in self._excluded_samples]
            if len(keep) >= 2:
                X          = X[keep, :]
                all_names  = [all_names[i]  for i in keep]
                all_groups = [all_groups[i] for i in keep]

        n_samples  = X.shape[0]
        n_analytes = X.shape[1]
        if n_samples == 0:
            messagebox.showinfo('No data', 'No samples to display.', parent=self.frame)
            return

        # ── Log-transform for normality assessment ────────────────────────────
        # log(1+x) handles zeros; metabolomics concentrations are log-normal.
        X_log = np.log1p(np.maximum(X, 0.0))

        # ── Shapiro-Wilk per sample on log-transformed profile ────────────────
        sw_stats, sw_pvals, sw_normal = [], [], []
        for i in range(n_samples):
            row = X_log[i, :]
            row = row[np.isfinite(row)]
            if len(row) < 3:
                sw_stats.append(float('nan'))
                sw_pvals.append(float('nan'))
                sw_normal.append(None)
            else:
                try:
                    w, p = spstats.shapiro(row[:5000])
                    sw_stats.append(float(w))
                    sw_pvals.append(float(p))
                    sw_normal.append(p >= 0.05)
                except Exception:
                    sw_stats.append(float('nan'))
                    sw_pvals.append(float('nan'))
                    sw_normal.append(None)

        # ── Grubbs' test: Euclidean distance to group centroid ────────────────
        # Normalize each feature across ALL samples (z-score in log space) so
        # all metabolites contribute equally regardless of concentration range.
        col_std = np.nanstd(X_log, axis=0, ddof=1)
        col_std[col_std < 1e-10] = 1.0
        X_z = (X_log - np.nanmean(X_log, axis=0)) / col_std

        # Per-sample Euclidean distance from its group centroid in z-score space.
        sample_dist = np.zeros(n_samples)
        unique_groups = sorted(set(all_groups))
        for grp in unique_groups:
            idx_g = [i for i, g in enumerate(all_groups) if g == grp]
            centroid = np.nanmean(X_z[idx_g, :], axis=0)
            for i in idx_g:
                diff = X_z[i, :] - centroid
                sample_dist[i] = float(np.sqrt(np.nansum(diff ** 2)))

        # Grubbs on distance-to-centroid: distances are non-negative, so only the
        # UPPER tail (far from centroid) is an outlier — a one-sided test avoids
        # flagging samples that are unusually CLOSE to the centroid. Iterative
        # removal prevents a single outlier from masking itself by inflating the
        # group SD. See processing/outlier_detection.
        from processing.outlier_detection import grubbs_outliers

        grubbs_g    = [float('nan')] * n_samples
        grubbs_crit = [float('nan')] * n_samples
        grubbs_flag = [False]        * n_samples

        for grp in unique_groups:
            idx_g = [i for i, g in enumerate(all_groups) if g == grp]
            if len(idx_g) < 3:
                continue
            flags_g, gstat_g, gcrit_g = grubbs_outliers(
                sample_dist[idx_g], alpha=0.05, side='upper')
            for k, i in enumerate(idx_g):
                grubbs_g[i]    = float(gstat_g[k])
                grubbs_crit[i] = float(gcrit_g[k])
                grubbs_flag[i] = bool(flags_g[k])

        # ── Build results rows ────────────────────────────────────────────────
        rows = []
        for i in range(n_samples):
            p_v  = sw_pvals[i]
            norm = sw_normal[i]
            rows.append({
                'sample':     all_names[i],
                'group':      all_groups[i],
                'n':          n_analytes,
                'sw_stat':    f'{sw_stats[i]:.4f}' if np.isfinite(sw_stats[i]) else '—',
                'sw_p':       f'{p_v:.4f}'          if np.isfinite(p_v)         else '—',
                'parametric': ('Yes' if norm else ('No' if norm is not None else '—')),
                'dist':       f'{sample_dist[i]:.3f}',
                'g_stat':     f'{grubbs_g[i]:.4f}'    if np.isfinite(grubbs_g[i])    else '—',
                'g_crit':     f'{grubbs_crit[i]:.4f}' if np.isfinite(grubbs_crit[i]) else '—',
                'outlier':    ('⚠ Yes' if grubbs_flag[i] else 'No'),
                '_norm_flag': norm,
                '_out_flag':  grubbs_flag[i],
            })

        # ── Dialog ───────────────────────────────────────────────────────────
        dlg = tk.Toplevel(self.frame)
        dlg.title('Sample Normality & Grubbs Outlier Check')
        dlg.resizable(True, True)
        dlg.update_idletasks()
        dw, dh = 960, 480
        sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        dlg.geometry(f'{dw}x{dh}+{(sw - dw) // 2}+{(sh - dh) // 2}')

        cols   = ('sample', 'group', 'n', 'sw_stat', 'sw_p',
                  'parametric', 'dist', 'g_stat', 'g_crit', 'outlier')
        hdrs   = ('Sample', 'Group', 'n', 'SW stat', 'SW p (log)',
                  'Profile normal?', 'Profile dist.', 'G stat', 'G critical', 'Outlier?')
        widths = (150, 90, 42, 78, 85, 88, 95, 78, 88, 78)

        frm = ttk.Frame(dlg)
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))

        vsb = ttk.Scrollbar(frm, orient='vertical')
        hsb = ttk.Scrollbar(frm, orient='horizontal')
        tree = ttk.Treeview(frm, columns=cols, show='headings',
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)

        for col, hdr, w in zip(cols, hdrs, widths):
            tree.heading(col, text=hdr)
            tree.column(col, width=w, minwidth=36, anchor='center')
        tree.column('sample', anchor='w')
        tree.column('group',  anchor='w')

        tree.tag_configure('non_normal', background='#fff3e0', foreground='#e65100')
        tree.tag_configure('outlier',    background='#fce4ec', foreground='#c62828')
        tree.tag_configure('both',       background='#fce4ec', foreground='#880e4f')
        tree.tag_configure('ok',         background='#f1f8e9', foreground='#33691e')

        for r in rows:
            vals = tuple(r[c] for c in cols)
            if r['_out_flag']:
                tag = 'both' if r['_norm_flag'] is False else 'outlier'
            elif r['_norm_flag'] is False:
                tag = 'non_normal'
            else:
                tag = 'ok'
            tree.insert('', tk.END, values=vals, tags=(tag,))

        # ── Bottom bar ────────────────────────────────────────────────────────
        bot = ttk.Frame(dlg)
        bot.pack(fill=tk.X, padx=6, pady=(2, 4))

        def _export_csv():
            path = filedialog.asksaveasfilename(
                parent=dlg, defaultextension='.csv',
                filetypes=[('CSV', '*.csv')], title='Save normality results')
            if not path:
                return
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                csv.writer(fh).writerow(list(hdrs))
                for r in rows:
                    csv.writer(fh).writerow([r[c] for c in cols])
            messagebox.showinfo('Saved', f'Exported to {path}', parent=dlg)

        ttk.Button(bot, text='Export CSV', command=_export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            bot, text='📈 Normality plots',
            command=lambda: self._open_normality_plots(
                X, X_log, all_names, all_groups)
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            bot,
            text=(f'{n_samples} samples · {n_analytes} analytes · '
                  'Profile normal? = Shapiro-Wilk on each sample log-profile '
                  '(QC of the sample as a whole; group test choice is by size) · '
                  'Profile dist. = Euclidean distance to group centroid (log+z-score) · '
                  'Grubbs one-sided upper, iterative (α=0.05) → possible outlier'),
            font=('Arial', 7), foreground='#555'
        ).pack(side=tk.LEFT, padx=8)

    def _open_normality_plots(self, X, X_log, all_names, all_groups):
        """MetaboAnalyst-style normality diagnostics.

        Three panels:
          • Density before  — per-sample KDE of raw concentrations
                               (typically right-skewed → non-normal).
          • Density after    — per-sample KDE of log-transformed values
                               (should approach a symmetric bell → normal).
          • Q-Q plot         — pooled, mean-centred per sample on the log
                               scale, against the normal quantiles. Points on
                               the diagonal ⇒ normal.

        Per-sample lines are coloured by group so batch/group effects are
        visible, matching MetaboAnalyst's normalization QC view.
        """
        from scipy import stats as spstats
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        dlg = tk.Toplevel(self.frame)
        dlg.title('Normality Plots')
        dlg.resizable(True, True)
        dlg.update_idletasks()
        dw, dh = 1100, 460
        sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        dlg.geometry(f'{dw}x{dh}+{(sw - dw) // 2}+{(sh - dh) // 2}')

        g_colors = getattr(self.app, 'group_colors', {}) if self.app else {}

        def _kde_curve(ax, values, color, label):
            v = values[np.isfinite(values)]
            if v.size < 2 or np.std(v) < 1e-12:
                return
            try:
                kde = spstats.gaussian_kde(v)
                xs  = np.linspace(float(v.min()), float(v.max()), 200)
                ax.plot(xs, kde(xs), color=color, lw=1.0, alpha=0.75, label=label)
            except Exception:
                ax.hist(v, bins=20, density=True, color=color, alpha=0.25)

        fig = Figure(figsize=(11, 4.4), dpi=100, layout='constrained')
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        seen_groups = set()
        for i in range(X.shape[0]):
            grp = all_groups[i] if i < len(all_groups) else 'Unknown'
            col = g_colors.get(grp, '#1f77b4')
            lbl = grp if grp not in seen_groups else None
            seen_groups.add(grp)
            _kde_curve(ax1, X[i, :],     col, lbl)
            _kde_curve(ax2, X_log[i, :], col, lbl)

        ax1.set_title('Density — raw', fontsize=9)
        ax1.set_xlabel('Concentration (au)', fontsize=8)
        ax1.set_ylabel('Density', fontsize=8)
        ax2.set_title('Density — log(1+x)', fontsize=9)
        ax2.set_xlabel('log concentration', fontsize=8)
        for ax in (ax1, ax2):
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25, linestyle=':')
        if seen_groups:
            ax2.legend(fontsize=6, framealpha=0.7)

        # Q-Q on pooled, per-sample mean-centred log values (removes overall
        # concentration offset so the test reflects distribution shape).
        pooled = (X_log - np.nanmean(X_log, axis=1, keepdims=True)).ravel()
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size >= 3:
            spstats.probplot(pooled, dist='norm', plot=ax3)
            ax3.get_lines()[0].set_markersize(2.5)
            ax3.get_lines()[0].set_alpha(0.4)
            ax3.get_lines()[1].set_color('#c62828')
        ax3.set_title('Q-Q plot (log, pooled)', fontsize=9)
        ax3.set_xlabel('Theoretical quantiles', fontsize=8)
        ax3.set_ylabel('Sample quantiles', fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.grid(True, alpha=0.25, linestyle=':')

        canvas = FigureCanvasTkAgg(fig, master=dlg)
        canvas.draw()
        NavigationToolbar2Tk(canvas, dlg).update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_right_panel(self):
        """Build the statistics plot area inside app.right_panel."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self._right_widget = ttk.Frame(self.app.right_panel)

        # Export bar at the bottom
        eb = ttk.Frame(self._right_widget)
        eb.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        ttk.Label(eb, text='DPI:', font=('Arial', 8)).pack(side=tk.LEFT, padx=(4, 2))
        self._dpi_var = tk.IntVar(value=1200)
        ttk.Spinbox(eb, from_=72, to=1200, increment=100,
                    textvariable=self._dpi_var, width=5
                    ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(eb, text='Format:', font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 2))
        self._fmt_var = tk.StringVar(value='PNG')
        ttk.Combobox(eb, textvariable=self._fmt_var,
                     values=['PNG', 'SVG', 'PDF', 'TIFF', 'EPS'],
                     width=6, state='readonly').pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(eb, text='💾 Save figure', command=self._export
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(eb, text='📄 CSV result', command=self._export_csv_result
                   ).pack(side=tk.LEFT, padx=4)

        # Scrollable display canvas
        plot_outer = ttk.Frame(self._right_widget)
        plot_outer.pack(fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(plot_outer, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(plot_outer, orient=tk.VERTICAL)
        self._scroll_canvas = tk.Canvas(plot_outer,
                                        xscrollcommand=h_scroll.set,
                                        yscrollcommand=v_scroll.set,
                                        highlightthickness=0, bg='white')
        h_scroll.config(command=self._scroll_canvas.xview)
        v_scroll.config(command=self._scroll_canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self._scroll_canvas.pack(fill=tk.BOTH, expand=True)

        # Matplotlib non-interactive renderer (no tk widget, no artifacts)
        self._fig = Figure(figsize=(8, 5), dpi=100, layout='constrained')
        self._canvas = FigureCanvasAgg(self._fig)

        # PIL display state
        self._zoom_level  = 1.0
        self._base_image  = None   # PIL image at container size
        self._photo       = None   # PhotoImage reference (keep alive)
        self._image_id    = None   # canvas item id
        self._resize_job  = None

        # Initial placeholder
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Select a method and click Run',
                transform=ax.transAxes, ha='center', va='center',
                color='#888888', fontsize=12)
        ax.set_axis_off()

        self._scroll_canvas.bind('<Configure>', self._on_container_resize)
        self._scroll_canvas.bind('<MouseWheel>', self._on_scroll_zoom)
        self._scroll_canvas.after(80, self._render_and_display)

    # ── PIL image zoom ────────────────────────────────────────────────────────

    def _render_and_display(self):
        """Render figure at current container size and display as PIL image.

        Uses savefig(bbox_inches='tight') so rotated axis labels and other
        artists that extend outside the axes area are never clipped.
        """
        import io
        from PIL import Image as _PILImage
        if self._fig is None or getattr(self, '_scroll_canvas', None) is None:
            return                      # panel not built yet — nothing to render
        w_px = self._scroll_canvas.winfo_width()
        h_px = self._scroll_canvas.winfo_height()
        if w_px < 50 or h_px < 50:
            self._scroll_canvas.after(80, self._render_and_display)
            return
        dpi = self._fig.dpi
        # Dumbbell (Pathway) renders at a content-scaled height so its stacked
        # facets are readable (the canvas scrolls); other plots fill the panel.
        _db = (getattr(self, '_mode_var', None) and self._mode_var.get() == 'pathway'
               and getattr(self, '_pw_plot_var', None)
               and self._pw_plot_var.get() == 'Dumbbell'
               and getattr(self, '_dumbbell_export_figsize', None))
        # Compact Volcano is a portrait plot (taller than wide) — fill the height
        # and make the width ~0.72× of it, instead of filling the wide canvas.
        _vol_compact = (
            getattr(self, '_method_var', None) is not None
            and self._method_var.get() == 'Volcano'
            and getattr(self, '_volcano_style', None)
            and self._volcano_style.get('theme') == 'Compact')
        # Box/Violin renders at its configured panel size (A4-capped), not the
        # canvas size, so the panel W/H controls the on-screen graph size too
        # (the canvas scrolls when the figure is larger than the viewport).
        _is_box = (getattr(self, '_method_var', None) is not None
                   and self._method_var.get() == 'Box / Violin'
                   and getattr(self, '_box_export_figsize', None))
        if _db:
            self._fig.set_size_inches(max(w_px / dpi, _db[0]), _db[1])
        elif _vol_compact:
            _h = h_px / dpi
            self._fig.set_size_inches(min(_h * 0.72, w_px / dpi), _h)
        elif _is_box:
            self._fig.set_size_inches(*self._box_export_figsize)
        else:
            self._fig.set_size_inches(w_px / dpi, h_px / dpi)
        buf = io.BytesIO()
        try:
            self._fig.savefig(buf, format='png', dpi=dpi,
                              bbox_inches='tight', facecolor='white')
        except Exception:
            # Tight bbox + constrained_layout can fail hard: a non-finite size
            # from a stray artist, or "axes collapsed to zero" when the layout
            # solver can't fit legend/labels. If we let this propagate the panel
            # keeps showing the PREVIOUS plot (looks like swapping groups does
            # nothing). Retry with the plain box and no layout engine, and fall
            # back to an error placeholder so the image ALWAYS refreshes.
            try:
                self._fig.set_layout_engine('none')
            except Exception:
                pass
            buf.seek(0); buf.truncate(0)
            try:
                self._fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
            except Exception as _re:
                import traceback; traceback.print_exc()
                self._fig.clear()
                _ax = self._fig.add_subplot(111)
                _ax.text(0.5, 0.5, f'Render failed:\n{_re}',
                         transform=_ax.transAxes, ha='center', va='center',
                         color='red', fontsize=9, wrap=True)
                _ax.set_axis_off()
                buf.seek(0); buf.truncate(0)
                self._fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        buf.seek(0)
        self._base_image = _PILImage.open(buf).copy()
        buf.close()
        self._display_image()

    def _display_image(self):
        """Scale stored base image by zoom level and put in scroll canvas."""
        from PIL import Image as _PILImage, ImageTk as _ImageTk
        if self._base_image is None:
            return
        new_w = max(1, int(self._base_image.width  * self._zoom_level))
        new_h = max(1, int(self._base_image.height * self._zoom_level))
        img = self._base_image.resize((new_w, new_h), _PILImage.LANCZOS)
        self._photo = _ImageTk.PhotoImage(img)
        if self._image_id is None:
            self._image_id = self._scroll_canvas.create_image(
                0, 0, anchor='nw', image=self._photo)
        else:
            self._scroll_canvas.itemconfig(self._image_id, image=self._photo)
        self._scroll_canvas.configure(scrollregion=(0, 0, new_w, new_h))

    def _on_container_resize(self, event):
        if self._zoom_level != 1.0:
            return
        if self._resize_job:
            self._scroll_canvas.after_cancel(self._resize_job)
        self._resize_job = self._scroll_canvas.after(150, self._render_and_display)

    def _on_scroll_zoom(self, event):
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._zoom_level = max(self._zoom_level * factor, 0.3)
        self._display_image()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_activate(self):
        self.frame.pack(fill=tk.BOTH, expand=True)
        self._refresh_groups()
        self._hide_canvas()
        self._show_right_panel()

    def on_deactivate(self):
        if self.frame:
            self.frame.pack_forget()
        self._restore_canvas()

    def _hide_canvas(self):
        try:
            self.app.canvas.get_tk_widget().pack_forget()
        except Exception:
            pass
        try:
            self.app.toolbar.pack_forget()
        except Exception:
            pass

    def _restore_canvas(self):
        if self._right_widget:
            try:
                self._right_widget.pack_forget()
            except Exception:
                pass
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

    def _show_right_panel(self):
        # Rebuild if never built OR if a previous build aborted before creating
        # the figure (leaves _right_widget set but _fig None → later crashes).
        if self._right_widget is None or self._fig is None:
            if self._right_widget is not None:
                try:
                    self._right_widget.destroy()
                except Exception:
                    pass
                self._right_widget = None
            self._build_right_panel()
        self._right_widget.pack(fill=tk.BOTH, expand=True)

    # ── Run ──────────────────────────────────────────────────────────────────

    def _run(self):
        # Ensure the plot panel/figure exist (a prior build may have aborted,
        # leaving self._fig None).
        if self._fig is None:
            self._show_right_panel()
        if self._fig is None:
            messagebox.showerror('Statistics',
                                 'Plot panel failed to initialise.',
                                 parent=self.frame)
            return
        if getattr(self, '_mode_var', None) and self._mode_var.get() == 'pathway':
            if self._is_lipidomics_project():
                self._run_fa_ratios()
            else:
                self._run_pathway()
            return
        method = self._method_var.get()
        X, sample_names, feature_labels, group_labels = _get_quantification_matrix(self.app)
        _single_sample_ok = method in ('Metabolic Profile', 'Box / Violin')
        _min_samples = 1 if _single_sample_ok else 2
        if X is None or X.shape[0] < _min_samples or X.shape[1] < 2:
            messagebox.showwarning(
                'No data',
                'Not enough data.\n'
                'Run Annotation (mark peaks as Good) or Quantification first.',
                parent=self.frame)
            return

        # Apply sample filter
        if self._excluded_samples:
            keep_rows = [i for i, n in enumerate(sample_names) if n not in self._excluded_samples]
            if len(keep_rows) < _min_samples:
                messagebox.showwarning('Too few samples',
                    'Less than 2 samples remain after filtering.', parent=self.frame)
                return
            X            = X[keep_rows, :]
            sample_names = [sample_names[i] for i in keep_rows]
            group_labels = [group_labels[i] for i in keep_rows]

        # Apply metabolite filter
        if self._excluded_metabolites:
            keep_cols = [i for i, f in enumerate(feature_labels) if f not in self._excluded_metabolites]
            if len(keep_cols) < 2:
                messagebox.showwarning('Too few metabolites',
                    'Less than 2 metabolites remain after filtering.', parent=self.frame)
                return
            X              = X[:, keep_cols]
            feature_labels = [feature_labels[i] for i in keep_cols]

        X, feature_labels = self._remove_nan(X, feature_labels)
        if X.shape[1] < 2:
            messagebox.showwarning(
                'No data',
                'Not enough valid features after removing missing values.',
                parent=self.frame)
            return

        X_scaled = self._scale(X)

        self._zoom_level = 1.0
        self._base_image = None
        self._fig.set_layout_engine('constrained')
        self._fig.clear()
        # Reset per-plot export data; each _plot_* sets it with the values shown.
        self._plot_export = None
        try:
            if method == 'PCA':
                self._plot_pca(X_scaled, sample_names, feature_labels, group_labels)
            elif method == 'PLS-DA':
                self._plot_plsda(X_scaled, sample_names, feature_labels, group_labels, opls=False)
            elif method == 'OPLS-DA':
                self._plot_plsda(X_scaled, sample_names, feature_labels, group_labels, opls=True)
            elif method == 'Heatmap':
                self._plot_heatmap(X_scaled, sample_names, feature_labels, group_labels)
            elif method == 'Correlation':
                self._plot_correlation(X, sample_names, feature_labels, group_labels)
            elif method == 'VIP Score':
                self._plot_vip(X_scaled, sample_names, feature_labels, group_labels)
            elif method == 'Volcano':
                self._plot_volcano(X, sample_names, feature_labels, group_labels)
            elif method == 'Metabolic Profile':
                self._plot_metabolic_profile(X, sample_names, feature_labels, group_labels)
            elif method == 'Box / Violin':
                self._plot_box_violin_grid()
        except Exception as e:
            import traceback; traceback.print_exc()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error:\n{e}',
                    transform=ax.transAxes, ha='center', va='center',
                    color='red', fontsize=9, wrap=True)
            ax.set_axis_off()

        self._render_and_display()

    # ── FA ratios mode (lipidomics) ──────────────────────────────────────────

    def _build_left_fa_ratios(self, parent):
        _SEC = dict(padding=(6, 4, 6, 4), relief='groove')
        info = ttk.LabelFrame(parent, text='FA ratios', **_SEC)
        info.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            info,
            text=('Fatty-acid composition ratios derived from the\n'
                  'Box 1 indices (Li et al. 2017) — CDCl3 lipid\n'
                  'extracts only. Box plots compare groups per ratio.'),
            font=('Arial', 8), foreground='#444', justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=4, pady=2)
        ttk.Label(
            info, text='n-3 index is ALA-only (EPA/DHA not resolved).',
            font=('Arial', 7, 'italic'), foreground='#a60',
            justify=tk.LEFT, wraplength=210,
        ).pack(anchor=tk.W, padx=4, pady=(0, 2))
        ttk.Label(
            info, text='Press Run to draw the box plots.',
            font=('Arial', 8, 'italic'), foreground='#666',
        ).pack(anchor=tk.W, padx=4, pady=(2, 2))

        # Sample filter is shared/honoured by the FA-ratios run.
        ff = ttk.LabelFrame(parent, text='Filters', **_SEC)
        ff.pack(fill=tk.X, pady=(0, 4))
        self._filter_samples_btn = ttk.Button(
            ff, text='Samples', command=self._open_sample_selector)
        self._filter_samples_btn.pack(fill=tk.X, padx=4, pady=2)
        self._filter_status_lbl = ttk.Label(
            ff, text='', font=('Arial', 7), foreground='#666')
        self._filter_status_lbl.pack(anchor=tk.W, padx=4, pady=(0, 2))

    def _run_fa_ratios(self):
        from processing import lipidomics as _lp
        import matplotlib.pyplot as plt

        quant = (getattr(self.app, 'stages', {}) or {}).get('quantification')
        results = getattr(quant, '_quant_results', None) or []
        ratios = _lp.fa_ratio_series(results)
        if not ratios:
            self._pathway_status(
                'No FA composition ratios available.\n\n'
                'FA ratios are built from the Box 1 indices, computed only for '
                'CDCl3 (Li) lipidomics runs. Run Quantification in Lipidomics '
                'mode with the CDCl3 solvent first.')
            return

        si = getattr(self.app, 'samples_info', []) or []
        sample_names = [s.get('name', f'S{i}') for i, s in enumerate(si)]
        group_labels = [s.get('label', s.get('group', 'Unknown')) for s in si]
        keep = [i for i, n in enumerate(sample_names)
                if n not in self._excluded_samples]
        groups = {}
        for i in keep:
            groups.setdefault(group_labels[i], []).append(i)
        if not groups:
            self._pathway_status('No samples remain after filtering.')
            return
        order = list(groups.keys())

        _app_colors = getattr(self.app, 'group_colors', {})
        cmap = plt.cm.get_cmap('tab10', max(len(order), 1))
        gcol = {g: (_app_colors.get(g) or cmap(i)) for i, g in enumerate(order)}

        self._zoom_level = 1.0
        self._base_image = None
        self._fig.set_layout_engine('constrained')
        self._fig.clear()

        names = list(ratios.keys())
        ncol = 3
        nrow = int(np.ceil(len(names) / ncol))
        for k, name in enumerate(names):
            ax = self._fig.add_subplot(nrow, ncol, k + 1)
            arr = ratios[name]
            data = []
            for g in order:
                vals = [float(arr[i]) for i in groups[g]
                        if i < len(arr) and np.isfinite(arr[i])]
                data.append(vals)
            if not any(data):
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='#999', fontsize=8)
                ax.set_title(name, fontsize=9)
                ax.set_axis_off()
                continue
            try:
                bp = ax.boxplot(data, tick_labels=order, patch_artist=True,
                                showfliers=False)
            except TypeError:                      # matplotlib < 3.9
                bp = ax.boxplot(data, labels=order, patch_artist=True,
                                showfliers=False)
            for patch, g in zip(bp['boxes'], order):
                patch.set_facecolor(gcol[g])
                patch.set_alpha(0.6)
            rng = np.random.RandomState(0)
            for xi, (g, vals) in enumerate(zip(order, data), start=1):
                if vals:
                    jit = (rng.rand(len(vals)) - 0.5) * 0.15
                    ax.scatter(np.full(len(vals), xi) + jit, vals, s=12,
                               color=gcol[g], edgecolor='k',
                               linewidths=0.3, zorder=3)
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_ylabel('%' if name.endswith('(%)') else 'ratio', fontsize=7)

        self._fig.suptitle(
            'Fatty-acid composition ratios (Box 1, Li et al. 2017)', fontsize=10)
        self._render_and_display()

    # ── Pathway mode run ─────────────────────────────────────────────────────

    def _pathway_status(self, msg):
        """Show a centred status message in the right viewer."""
        if self._fig is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='#444', wrap=True)
        ax.set_axis_off()
        self._render_and_display()

    def _run_pathway(self):
        # Reset so a run that aborts early (status message) cannot export the
        # previous plot's stale table; each successful path sets it again.
        self._plot_export = None
        ga = self._grp_a_var.get() if hasattr(self, '_grp_a_var') else ''
        gb = self._grp_b_var.get() if hasattr(self, '_grp_b_var') else ''
        if not ga or not gb or ga == gb:
            messagebox.showwarning('Pathway',
                'Select two different groups (Group A ≠ Group B).',
                parent=self.frame)
            return

        X, sample_names, feature_labels, group_labels = \
            _get_quantification_matrix(self.app)
        if X is None or X.shape[0] < 2 or X.shape[1] < 2:
            messagebox.showwarning('No data',
                'Not enough data. Run Quantification (or mark Good peaks) first.',
                parent=self.frame)
            return

        # Sample / metabolite exclusion filters
        if self._excluded_samples:
            keep = [i for i, n in enumerate(sample_names)
                    if n not in self._excluded_samples]
            X = X[keep, :]
            sample_names = [sample_names[i] for i in keep]
            group_labels = [group_labels[i] for i in keep]
        if self._excluded_metabolites:
            keep = [i for i, f in enumerate(feature_labels)
                    if f not in self._excluded_metabolites]
            X = X[:, keep]
            feature_labels = [feature_labels[i] for i in keep]

        X, feature_labels = self._remove_nan(X, feature_labels)
        if X.shape[1] < 2:
            messagebox.showwarning('No data',
                'Not enough valid metabolites after filtering.', parent=self.frame)
            return

        # ── Dumbbell: A vs Control-A and B vs Control-B, RAW data (no ORA) ────
        if self._pw_plot_var.get() == 'Dumbbell':
            _ctrl_a = self._grp_ca_var.get()
            _ctrl_b = self._grp_cb_var.get()
            self._zoom_level = 1.0; self._base_image = None
            self._fig.set_layout_engine('constrained'); self._fig.clear()
            try:
                self._plot_dumbbell(
                    X, feature_labels, group_labels, _ctrl_a, _ctrl_b, ga, gb,
                    p_thr=float(self._pw_p_thr_var.get()),
                    category=self._pw_cat_var.get())
            except Exception as _e:
                _ax = self._fig.add_subplot(111)
                _ax.text(0.5, 0.5, f'Error:\n{_e}', transform=_ax.transAxes,
                         ha='center', va='center', color='#c0392b')
                _ax.axis('off')
            self._render_and_display()
            return

        if ga not in group_labels or gb not in group_labels:
            messagebox.showwarning('Pathway',
                'Selected groups are not present in the (filtered) samples.',
                parent=self.frame)
            return

        # Apply the same normalization/scaling as Statistics for the significance
        # test, but keep the raw matrix for fold-change direction (scaling
        # mean-centres the data → ratios on centred values are meaningless).
        X_raw = X.copy()
        X = self._scale(X)
        self._pw_X_raw = X_raw

        # ── Build membership (cached; fetch in background thread) ─────────────
        quant_results = self._pathway_quant_results(feature_labels)
        n_with_id = sum(1 for r in quant_results
                        if str(r.get('hmdb_id') or '').upper().startswith('HMDB'))
        if n_with_id == 0:
            self._pathway_status(
                'No metabolite has an HMDB ID.\n'
                'Pathway analysis needs HMDB IDs from annotation/quantification.')
            return

        cache_key = frozenset(feature_labels)
        use_api = False   # internal only — local pathway DB, no external HMDB calls
        cached = getattr(self, '_pw_membership_cache', None)
        if (cached is not None and cached.get('key') == cache_key
                and cached.get('use_api') == use_api):
            self._finish_pathway(cached['table'], cached['skipped'],
                                 X, feature_labels, group_labels, ga, gb,
                                 X_fc=X_raw)
            return

        self._pathway_status('Fetching pathway membership…')

        import threading
        from data import pathway_membership as pwm

        def _worker():
            try:
                table, skipped = pwm.build_membership(
                    quant_results, use_api=use_api)
            except Exception as e:
                self.frame.after(0, lambda: self._pathway_status(
                    f'Membership lookup failed:\n{e}'))
                return
            self._pw_membership_cache = {
                'key': cache_key, 'use_api': use_api,
                'table': table, 'skipped': skipped}
            self.frame.after(0, lambda: self._finish_pathway(
                table, skipped, X, feature_labels, group_labels, ga, gb,
                X_fc=X_raw))

        threading.Thread(target=_worker, daemon=True).start()

    def _pathway_quant_results(self, feature_labels):
        """Build [{metabolite, hmdb_id}] for the given features, pulling HMDB
        IDs from the quantification stage results when available."""
        quant_stage = (getattr(self.app, 'stages', {}) or {}).get('quantification')
        raw = getattr(quant_stage, '_quant_results', None) or []
        id_map = {}
        for r in raw:
            if r.get('_not_quantified'):
                continue
            nm = r.get('metabolite') or r.get('name')
            if nm:
                id_map[nm] = r.get('hmdb_id') or r.get('id') or ''
        return [{'metabolite': f, 'hmdb_id': id_map.get(f, '')}
                for f in feature_labels]

    def _finish_pathway(self, pathway_table, skipped, X, feature_labels,
                        group_labels, ga, gb, X_fc=None):
        """Compute ORA and render the selected pathway plot (main thread).

        X is the scaled matrix used for the significance test; X_fc (raw) is used
        for fold-change direction so scaling's mean-centring does not corrupt it."""
        from data import pathway_membership as pwm
        from processing import pathway_enrichment as pwe

        cat = self._pw_cat_var.get()
        table = pwm.filter_by_category(pathway_table, cat)
        if not table:
            self._pathway_status(
                f'No "{cat}" pathways found for these metabolites.\n'
                'Try Category = all.')
            return

        test_res = pwe.per_metabolite_test(X, group_labels, ga, gb, feature_labels,
                                           X_fc=X_fc)
        p_thr = float(self._pw_p_thr_var.get())
        hits = pwe.select_hits(test_res, p_thr)
        if not hits:
            self._pathway_status(
                f'No metabolite is significant at p < {p_thr:g}\n'
                f'between {ga} and {gb}. Relax the threshold.')
            return

        bg_sizes = pwm.pathway_background_sizes(list(table.keys()))
        rows = pwe.compute_ora(table, hits, background_sizes=bg_sizes,
                               n_background=pwm.N_BACKGROUND)
        if not rows:
            self._pathway_status(
                'No pathway is enriched in the significant metabolites.')
            return

        # Min mets = minimum significant hits (k) a pathway must contain.
        min_mets = int(self._pw_minmets_var.get())
        rows = [r for r in rows if r['k'] >= min_mets]
        if not rows:
            self._pathway_status(
                f'No pathway has ≥ {min_mets} significant metabolite(s).\n'
                'Lower "Min mets" or relax the significance threshold.')
            return

        if self._pw_sig_only_var.get():
            _pk = 'fdr' if self._pw_use_fdr_var.get() else 'p'
            sig = [r for r in rows if r[_pk] < 0.05]
            if not sig:
                self._pathway_status(
                    'No pathway is significant (< 0.05).\n'
                    'Uncheck "Significant only" to see all enriched pathways.')
                return
            rows = sig

        top_n = int(self._pw_topn_var.get())
        rows = rows[:top_n]
        self._pw_last_rows = rows
        self._pw_skipped   = skipped

        # CSV result: the ORA table behind the overview/dot/lollipop/network plot.
        self._plot_export = {
            'method': 'Pathway',
            'headers': ['pathway', 'hits_k', 'pathway_size_K', 'sig_total_n',
                        'enrichment_ratio', 'p_value', 'fdr_bh', 'smpdb_id',
                        'hit_metabolites'],
            'rows': [[r['name'], r['k'], r['K'], r['n'],
                      float(r['enrichment_ratio']), float(r['p']),
                      float(r.get('fdr', float('nan'))), r.get('smpdb', ''),
                      '; '.join(r.get('mets', []))]
                     for r in rows],
        }

        self._zoom_level = 1.0
        self._base_image = None
        self._fig.set_layout_engine('constrained')
        self._fig.clear()
        plot = self._pw_plot_var.get()
        try:
            if plot == 'Network':
                self._plot_pathway_network(rows, ga, gb)
            elif plot == 'Dot plot':
                self._plot_pathway_dotplot(rows, ga, gb)
            elif plot == 'Lollipop':
                self._plot_pathway_lollipop(rows, ga, gb)
            else:
                self._plot_pathway_overview(rows, ga, gb)
        except Exception as e:
            import traceback; traceback.print_exc()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error:\n{e}', transform=ax.transAxes,
                    ha='center', va='center', color='red', fontsize=9, wrap=True)
            ax.set_axis_off()
        self._render_and_display()

    # ── Pathway plots ────────────────────────────────────────────────────────

    def _pathway_color(self, rows):
        """Return (values, LogNorm, cmap, label) for colouring by p or FDR."""
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        key   = 'fdr' if self._pw_use_fdr_var.get() else 'p'
        label = 'BH-FDR' if key == 'fdr' else 'p-value'
        vals  = [max(r[key], 1e-10) for r in rows]
        vmin  = min(vals)
        norm  = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=1.0)
        _cm = getattr(self, '_pw_cmap_var', None)
        cmap  = cm.get_cmap(_cm.get() if _cm else 'YlOrRd_r')
        return vals, norm, cmap, label

    def _pathway_colorbar(self, mappable, label):
        from matplotlib.ticker import FuncFormatter
        cb = self._fig.colorbar(mappable, ax=self._fig.axes,
                                fraction=0.046, pad=0.04, shrink=0.5)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # The p-value / FDR scale is LogNorm; matplotlib's default log ticks are
        # scientific notation (1e-2, …). Show plain decimals at the familiar
        # significance thresholds instead (0.05, 0.01, …) for readability.
        try:
            vmin = float(getattr(mappable.norm, 'vmin', None) or 1e-3)
        except (TypeError, ValueError):
            vmin = 1e-3
        ticks = [t for t in (1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 1e-4, 1e-5)
                 if t >= vmin * 0.999]
        if ticks:
            cb.set_ticks(ticks)
        cb.minorticks_off()                          # drop sci-notation minor labels

        def _decimal(v, _pos):
            if v <= 0:
                return ''
            return f'{v:.10f}'.rstrip('0').rstrip('.')
        cb.ax.yaxis.set_major_formatter(FuncFormatter(_decimal))

    def _plot_pathway_overview(self, rows, ga, gb):
        import numpy as np
        import matplotlib.cm as cm
        rows = sorted(rows, key=lambda r: r['enrichment_ratio'])
        names = [r['name'] for r in rows]
        ratios = [r['enrichment_ratio'] for r in rows]
        vals, norm, cmap, label = self._pathway_color(rows)
        colors = [cmap(norm(v)) for v in vals]

        ax = self._fig.add_subplot(111)
        y = np.arange(len(rows))
        ax.barh(y, ratios, color=colors, edgecolor='#888', linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([n[:42] for n in names], fontsize=7.5, fontweight='bold')
        ax.set_xlabel('Enrichment Ratio', fontsize=8)
        ax.set_title(f'Overview of Enriched Metabolite Sets\n{ga} vs {gb}',
                     fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7.5)
        ax.grid(axis='x', ls='--', alpha=0.4)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        self._pathway_colorbar(sm, label)

    def _plot_pathway_dotplot(self, rows, ga, gb):
        import numpy as np
        import matplotlib.cm as cm
        rows = sorted(rows, key=lambda r: -np.log10(max(r['p'], 1e-10)))
        names = [r['name'] for r in rows]
        xs = np.array([-np.log10(max(r['p'], 1e-10)) for r in rows], dtype=float)
        ratios = np.array([r['enrichment_ratio'] for r in rows], dtype=float)
        vals, norm, cmap, label = self._pathway_color(rows)
        colors = [cmap(norm(v)) for v in vals]

        rmax = ratios.max() if ratios.size and ratios.max() > 0 else 1.0
        # Dot area ∝ enrichment ratio; capped so neighbouring rows don't overlap.
        sizes = 70 + 240 * (ratios / rmax)

        ax = self._fig.add_subplot(111)
        y = np.arange(len(rows)) * 1.0
        ax.scatter(xs, y, s=sizes, c=colors, edgecolors='#444',
                   linewidths=0.7, alpha=0.95, zorder=3)
        # k/K (hits found / pathway size) written inside each dot
        for xi, yi, r in zip(xs, y, rows):
            ax.text(xi, yi, f"{r['k']}/{r['K']}", ha='center', va='center',
                    fontsize=5.5, color='#111', zorder=4, clip_on=False)
        ax.set_yticks(y)
        ax.set_yticklabels([n[:42] for n in names], fontsize=7.5)
        ax.set_xlabel('-log10(p-value)', fontsize=8)
        ax.set_title(f'Overview of Enriched Metabolite Sets\n{ga} vs {gb}',
                     fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7.5)
        ax.grid(axis='x', ls='--', alpha=0.4)
        # Pad x so capped dots near the right edge stay clear of the spine
        _xpad = max((xs.max() if xs.size else 1.0) * 0.12, 0.2)
        ax.set_xlim(-_xpad, (xs.max() if xs.size else 1.0) + _xpad)
        ax.set_ylim(-0.8, len(rows) - 0.2)
        # Size legend — markers scaled down (×0.6) and widely spaced so the
        # circles never touch each other.
        from matplotlib.lines import Line2D
        legend_ratios = sorted({round(r, 1) for r in
                                (rmax * 0.33, rmax * 0.66, rmax) if r > 0})
        handles = [Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor='#bbb', markeredgecolor='#555',
                          markersize=np.sqrt(70 + 240 * (lr / rmax)) * 0.6,
                          label=f'{lr:.1f}') for lr in legend_ratios]
        if handles:
            ax.legend(handles=handles, title='Enrichment Ratio',
                      fontsize=6.5, title_fontsize=7, loc='lower right',
                      framealpha=0.9, labelspacing=2.4, borderpad=1.0,
                      handletextpad=1.4)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        self._pathway_colorbar(sm, label)

    def _plot_pathway_lollipop(self, rows, ga, gb):
        import numpy as np
        import matplotlib.cm as cm
        rows = sorted(rows, key=lambda r: r['enrichment_ratio'])
        names = [r['name'] for r in rows]
        ratios = [r['enrichment_ratio'] for r in rows]
        vals, norm, cmap, label = self._pathway_color(rows)
        colors = [cmap(norm(v)) for v in vals]

        ax = self._fig.add_subplot(111)
        y = np.arange(len(rows))
        ax.hlines(y, 0, ratios, color='#bbb', lw=1.2, zorder=1)
        ax.scatter(ratios, y, s=120, c=colors, edgecolors='#555',
                   linewidths=0.7, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([n[:42] for n in names], fontsize=7.5)
        ax.set_xlabel('Enrichment Ratio', fontsize=8)
        ax.set_title(f'Enriched Metabolite Sets\n{ga} vs {gb}',
                     fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7.5)
        ax.grid(axis='x', ls='--', alpha=0.4)
        ax.margins(y=0.04)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        self._pathway_colorbar(sm, label)

    def _plot_pathway_network(self, rows, ga, gb):
        import numpy as np
        import matplotlib.cm as cm
        try:
            import networkx as nx
        except ImportError:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    'networkx not installed.\n'
                    'Install it to use the Network view, or pick another plot.',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#c62828', fontsize=10, wrap=True)
            ax.set_axis_off()
            return

        vals, norm, cmap, label = self._pathway_color(rows)
        jac_thr = float(self._pw_jaccard_var.get())

        G = nx.Graph()
        for r in rows:
            G.add_node(r['name'], k=r['k'],
                       pcol=('fdr' if self._pw_use_fdr_var.get() else 'p'))
        for i, r1 in enumerate(rows):
            s1 = set(r1.get('mets', []))
            for r2 in rows[i + 1:]:
                s2 = set(r2.get('mets', []))
                u = s1 | s2
                if not u:
                    continue
                j = len(s1 & s2) / len(u)
                if j >= jac_thr:
                    G.add_edge(r1['name'], r2['name'], weight=j)

        ax = self._fig.add_subplot(111)
        ax.set_axis_off()
        ax.set_title(f'Pathway Network  (Jaccard ≥ {jac_thr:.2f})\n{ga} vs {gb}',
                     fontsize=10, fontweight='bold')

        node_list = list(G.nodes)
        if not node_list:
            ax.text(0.5, 0.5, 'No pathways to display.',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#666', fontsize=10)
            return
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=1.4)

        pkey   = 'fdr' if self._pw_use_fdr_var.get() else 'p'
        rmap   = {r['name']: r for r in rows}
        ncolor = [cmap(norm(max(rmap[n][pkey], 1e-10))) for n in node_list]
        kmax   = max((rmap[n]['k'] for n in node_list), default=1) or 1
        nsize  = [180 + 520 * (rmap[n]['k'] / kmax) for n in node_list]

        if G.edges:
            ew = [G[u][v]['weight'] for u, v in G.edges]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#999',
                                   width=[2.5 * w for w in ew], alpha=0.4)
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, ax=ax,
                               node_color=ncolor, node_size=nsize,
                               edgecolors='#444', linewidths=1.2, alpha=0.95)
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={n: (n[:18] + '…' if len(n) > 18 else n) for n in node_list},
            font_size=6.5)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = self._fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
        cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)

    # ── NaN cleanup ──────────────────────────────────────────────────────────

    def _remove_nan(self, X, feature_labels):
        """Drop all-NaN columns, then impute remaining NaN with column mean."""
        X = np.array(X, dtype=float)
        # Drop columns where every sample is NaN
        valid_cols = ~np.all(np.isnan(X), axis=0)
        X = X[:, valid_cols]
        feature_labels = [f for f, v in zip(feature_labels, valid_cols) if v]
        # Impute remaining NaN with column mean
        col_means = np.nanmean(X, axis=0)
        nan_mask  = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return X, feature_labels

    # ── Scaling ───────────────────────────────────────────────────────────────

    def _scale(self, X):
        mode = getattr(self, '_scale_var', None)
        mode = mode.get() if mode else 'UV'
        Xs = X.copy().astype(float)

        # Log10 pre-transform (checkbox — combinável com qualquer scaling)
        if getattr(self, '_log_pre_var', None) and self._log_pre_var.get():
            Xs = np.log10(np.clip(Xs, 0, None) + 1)

        # ── Transformações (antes de centrar) ────────────────────────────────
        if mode == 'log':
            # log10(x + 1) — shift garante x >= 0 antes da transformação
            Xs = np.log10(np.clip(Xs, 0, None) + 1)
            mode = 'UV'

        elif mode == 'cbrt':
            Xs = np.cbrt(Xs)
            mode = 'UV'

        elif mode == 'median':
            Xs -= np.median(Xs, axis=0)
            return Xs   # centrado na mediana, sem divisão

        # ── Mean-centre + divisão de escala ──────────────────────────────────
        Xs -= Xs.mean(axis=0)
        if mode == 'UV':
            std = Xs.std(axis=0)
            std[std == 0] = 1.0
            Xs /= std
        elif mode == 'pareto':
            std = Xs.std(axis=0)
            std[std == 0] = 1.0
            Xs /= np.sqrt(std)
        # mode == 'none': só mean-centre
        return Xs

    # ── Plot methods ─────────────────────────────────────────────────────────

    def _group_colors(self, unique_groups):
        """Return color dict for groups using app colors or a tab10 fallback."""
        import matplotlib.cm as cm
        app_gc = getattr(self.app, 'group_colors', {})
        cmap   = cm.get_cmap('tab10', max(len(unique_groups), 1))
        return {g: app_gc.get(g, cmap(i)) for i, g in enumerate(unique_groups)}

    def _confidence_ellipse(self, ax, x, y, color, n_std=2.0, **kw):
        """Draw a confidence ellipse for 2-D data (x, y)."""
        if len(x) < 3:
            return
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h  = 2 * n_std * np.sqrt(np.abs(vals))
        ell   = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h,
                        angle=theta, facecolor=color, alpha=0.12,
                        edgecolor=color, linewidth=1.2, **kw)
        ax.add_patch(ell)

    def _plot_pca(self, X, sample_names, feature_labels, group_labels):
        from sklearn.decomposition import PCA
        n_comp = getattr(self, '_n_comp_var', None)
        n_comp = n_comp.get() if n_comp else 2
        n_comp = min(n_comp, X.shape[0], X.shape[1])

        pca    = PCA(n_components=n_comp)
        scores = pca.fit_transform(X)
        ev     = pca.explained_variance_ratio_ * 100

        # CSV export: PCA scores per sample (explained variance in the header).
        self._plot_export = {
            'method': 'PCA',
            'headers': (['sample', 'group']
                        + ['PC%d (%.1f%%)' % (c + 1, ev[c]) for c in range(n_comp)]),
            'rows': [([sample_names[i], str(group_labels[i])]
                      + [float(scores[i, c]) for c in range(n_comp)])
                     for i in range(scores.shape[0])],
        }

        y, unique = _group_to_int(group_labels)
        gcols = self._group_colors(unique)

        biplot  = getattr(self, '_biplot_var',  None)
        biplot  = biplot.get()  if biplot  else False
        ellipse = getattr(self, '_ellipse_var', None)
        ellipse = ellipse.get() if ellipse else True

        if n_comp >= 3:
            # 2×2 grid: PC1v2, PC1v3, PC2v3, scree
            axes = [self._fig.add_subplot(2, 2, i+1) for i in range(4)]
            pairs = [(0, 1), (0, 2), (1, 2)]
            for idx, (a, b) in enumerate(pairs):
                ax = axes[idx]
                for gi, grp in enumerate(unique):
                    mask = y == gi
                    ax.scatter(scores[mask, a], scores[mask, b],
                               color=gcols[grp], label=grp, s=55, alpha=0.85,
                               edgecolors='white', linewidths=0.5)
                    if ellipse and mask.sum() >= 3:
                        self._confidence_ellipse(ax, scores[mask, a],
                                                 scores[mask, b], gcols[grp])
                ax.set_xlabel(f'PC{a+1} ({ev[a]:.1f}%)', fontsize=8)
                ax.set_ylabel(f'PC{b+1} ({ev[b]:.1f}%)', fontsize=8)
                ax.axhline(0, color='#cccccc', lw=0.5); ax.axvline(0, color='#cccccc', lw=0.5)
                ax.set_facecolor('#f9f9f9')
                if idx == 0:
                    ax.legend(fontsize=7, loc='best')
            # Scree plot
            ax4 = axes[3]
            ax4.bar(range(1, n_comp+1), ev, color='#1565c0', alpha=0.7)
            ax4.set_xlabel('Component', fontsize=8)
            ax4.set_ylabel('Explained variance (%)', fontsize=8)
            ax4.set_title('Scree', fontsize=8)
        else:
            ax = self._fig.add_subplot(111)
            for gi, grp in enumerate(unique):
                mask = y == gi
                ax.scatter(scores[mask, 0], scores[mask, 1],
                           color=gcols[grp], label=grp, s=60, alpha=0.88,
                           edgecolors='white', linewidths=0.5)
                if ellipse and mask.sum() >= 3:
                    self._confidence_ellipse(ax, scores[mask, 0],
                                             scores[mask, 1], gcols[grp])
            if biplot:
                loadings = pca.components_.T
                scale = np.max(np.abs(scores[:, :2])) / np.max(np.abs(loadings[:, :2]) + 1e-9)
                for i, feat in enumerate(feature_labels):
                    ax.annotate('', xy=(loadings[i,0]*scale, loadings[i,1]*scale),
                                xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='#c62828', lw=0.8))
                    ax.text(loadings[i,0]*scale*1.05, loadings[i,1]*scale*1.05,
                            feat, fontsize=5, color='#c62828', ha='center')
            for i, name in enumerate(sample_names):
                ax.annotate(name, (scores[i,0], scores[i,1]),
                            fontsize=5.5, ha='center', va='bottom',
                            xytext=(0, 4), textcoords='offset points', color='#444')
            ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)', fontsize=9)
            ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)', fontsize=9)
            ax.set_title(f'PCA Score Plot  —  {X.shape[1]} features', fontsize=10)
            ax.axhline(0, color='#cccccc', lw=0.6); ax.axvline(0, color='#cccccc', lw=0.6)
            ax.legend(fontsize=8, loc='best', framealpha=0.85)
            ax.set_facecolor('#f9f9f9')

    def _plot_plsda(self, X, sample_names, feature_labels, group_labels, opls=False):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import label_binarize

        y_int, unique = _group_to_int(group_labels)
        n_classes = len(unique)
        if n_classes < 2:
            raise ValueError('Need at least 2 groups for PLS-DA.')
        # Binarize Y
        if n_classes == 2:
            Y = y_int.reshape(-1, 1).astype(float)
        else:
            Y = label_binarize(y_int, classes=list(range(n_classes))).astype(float)

        n_comp = getattr(self, '_n_comp_var', None)
        n_comp = n_comp.get() if n_comp else 2
        n_comp = min(n_comp, X.shape[0]-1, X.shape[1])

        from processing.multivariate import opls_da, pls_r2x_per_component

        gcols   = self._group_colors(unique)
        ellipse = getattr(self, '_ellipse_var', None)
        ellipse = ellipse.get() if ellipse else True

        if opls:
            # Real OPLS-DA (Trygg & Wold, J. Chemometrics 2002): one predictive
            # component vs Y-orthogonal components. Defined for two classes — the
            # previous "OPLS" merely Gram-Schmidt'd two already-orthogonal PLS
            # scores (a no-op that produced a mislabeled PLS-DA plot).
            if n_classes > 2:
                raise ValueError(
                    'OPLS-DA is defined for two classes only.\n'
                    'Use PLS-DA for more than two groups.')
            res = opls_da(X, y_int.astype(float), n_ortho=max(1, n_comp - 1))
            plot_x = res['t_pred']
            t_o    = res['t_ortho']
            plot_y = t_o[:, 0] if t_o.shape[1] else np.zeros_like(plot_x)
            ev_pred = res['r2x_pred']
            ev_orth = float(res['r2x_ortho'][0]) if res['r2x_ortho'].size else 0.0
            xlabel = f'Predictive t  (R²X = {ev_pred:.1f}%)'
            ylabel = f'Orthogonal t  (R²X = {ev_orth:.1f}%)'
            title  = 'OPLS-DA Score Plot'
        else:
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X, Y)
            scores = pls.transform(X)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            # Correct %X variance per LV: R2X_k = ||t_k||²·||p_k||² / ||X||²
            # (the dropped loading-norm factor under-reported the explained var).
            ev_x = pls_r2x_per_component(scores, pls.x_loadings_, X)
            plot_x = scores[:, 0]
            plot_y = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(plot_x)
            xlabel = f'LV1 ({ev_x[0]:.1f}% X var.)'
            ylabel = f'LV2 ({ev_x[1]:.1f}% X var.)' if len(ev_x) > 1 else 'LV2'
            title  = 'PLS-DA Score Plot'

        ax = self._fig.add_subplot(111)
        for gi, grp in enumerate(unique):
            mask = y_int == gi
            ax.scatter(plot_x[mask], plot_y[mask],
                       color=gcols[grp], label=grp, s=60, alpha=0.88,
                       edgecolors='white', linewidths=0.5)
            if ellipse and mask.sum() >= 3:
                self._confidence_ellipse(ax, plot_x[mask], plot_y[mask], gcols[grp])
        for i, name in enumerate(sample_names):
            ax.annotate(name, (plot_x[i], plot_y[i]),
                        fontsize=5.5, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', color='#444')

        # Validation fits the SAME binarized Y as the plotted model (ordinal
        # y_int would impose a false class ordering for ≥3 groups). Folds are
        # stratified and n_comp is clamped per fold (see plsda_cv_q2), so no fold
        # collapses to a constant Y — that was the "y residual is constant" warning.
        from processing.multivariate import plsda_cv_q2, plsda_permutation_p
        _min_class = int(np.min(np.bincount(y_int))) if len(y_int) else 0
        _n_splits  = max(2, min(5, _min_class))

        cv = getattr(self, '_cv_var', None)
        if cv and cv.get() and _min_class >= 2:
            try:
                r2cv = plsda_cv_q2(X, y_int, n_comp, _n_splits)
                ax.set_title(f'{title}  —  Q²={r2cv:.3f}', fontsize=10)
            except Exception:
                ax.set_title(title, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)

        # Permutation test (stratified CV, label shuffling — preserves class sizes
        # so every permuted fit stays non-degenerate).
        n_perm = getattr(self, '_n_perm_var', None)
        n_perm = n_perm.get() if n_perm else 0
        if n_perm > 0 and _min_class >= 2:
            try:
                _, pval = plsda_permutation_p(X, y_int, n_comp, _n_splits, n_perm)
                ax.text(0.02, 0.02, f'Permutation p={pval:.3f}',
                        transform=ax.transAxes, fontsize=7.5,
                        color='#555', va='bottom')
            except Exception:
                pass

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color='#cccccc', lw=0.6); ax.axvline(0, color='#cccccc', lw=0.6)
        ax.legend(fontsize=8, loc='best', framealpha=0.85)
        ax.set_facecolor('#f9f9f9')

    def _plot_heatmap(self, X, sample_names, feature_labels, group_labels):
        from scipy.cluster.hierarchy import linkage, dendrogram
        import matplotlib.patches as mpatches

        clust  = getattr(self, '_clust_var',   None)
        clust  = clust.get()  if clust  else 'ward'
        cmap   = getattr(self, '_cmap_var',    None)
        cmap   = cmap.get()   if cmap   else 'RdYlBu_r'
        _topn  = getattr(self, '_hm_topn_var', None)
        _topn  = _topn.get()  if _topn  else 'All'

        # Collapse samples to group means when requested
        _plot_by = getattr(self, '_hm_plot_by_var', None)
        _plot_by = _plot_by.get() if _plot_by else 'Samples'
        if _plot_by == 'Group mean':
            _yi, _uniq = _group_to_int(group_labels)
            X            = np.array([X[_yi == gi, :].mean(axis=0) for gi in range(len(_uniq))])
            sample_names = list(_uniq)
            group_labels = list(_uniq)

        # Feature selection: Top N by VIP score or top variance
        vip_scores = getattr(self, '_last_vip_scores', None)
        try:
            top_n = int(_topn)
        except (ValueError, TypeError):
            top_n = None

        if top_n is not None and vip_scores is not None and len(vip_scores) == X.shape[1]:
            order = np.argsort(vip_scores)[::-1][:top_n]
        elif top_n is not None:
            order = np.argsort(X.var(axis=0))[::-1][:top_n]
        else:
            MAX_FEAT = 60
            order = (np.argsort(X.var(axis=0))[::-1][:MAX_FEAT]
                     if X.shape[1] > MAX_FEAT else np.arange(X.shape[1]))

        X_h  = X[:, order]
        flbl = [feature_labels[i] for i in order]

        # Cluster rows (samples) and cols (features)
        row_link = linkage(X_h,   method=clust, metric='euclidean')
        col_link = linkage(X_h.T, method=clust, metric='euclidean')
        row_ord  = dendrogram(row_link, no_plot=True)['leaves']
        col_ord  = dendrogram(col_link, no_plot=True)['leaves']
        X_ord    = X_h[np.ix_(row_ord, col_ord)]
        slbl_ord = [sample_names[i]  for i in row_ord]
        flbl_ord = [flbl[i]          for i in col_ord]
        glbl_ord = [group_labels[i]  for i in row_ord]

        n_feat = X_ord.shape[1]
        n_samp = X_ord.shape[0]

        y_int, unique = _group_to_int(glbl_ord)
        gcols = self._group_colors(unique)
        import matplotlib.colors as mcolors

        # Remove any layout engine — use manual subplots_adjust only. Must pass
        # None (not the string 'none'): switching from the 'constrained' engine
        # set in _run() to 'none' leaves an adjust-incompatible placeholder that
        # makes subplots_adjust warn and no-op; None clears it cleanly.
        self._fig.set_layout_engine(None)

        # GridSpec: 2 rows × 2 cols  (colorbar/legend placed manually)
        #   row 0: [blank | col_dendrogram]
        #   row 1: [grp   | heatmap       ]
        gs = self._fig.add_gridspec(
            2, 2,
            height_ratios=[1, 4],
            width_ratios=[0.75, 12],
            hspace=0.01, wspace=0.02)

        ax_top = self._fig.add_subplot(gs[0, 1])   # column dendrogram
        ax_g   = self._fig.add_subplot(gs[1, 0])   # group color squares
        ax_h   = self._fig.add_subplot(gs[1, 1])   # heatmap

        # ── Column dendrogram (top) ───────────────────────────────────────────
        dendrogram(col_link, ax=ax_top, orientation='top',
                   no_labels=True, color_threshold=0,
                   above_threshold_color='#555555', link_color_func=lambda k: '#555555')
        ax_top.set_axis_off()

        # ── Group colour bar ─────────────────────────────────────────────────
        ax_g.imshow(np.array([[mcolors.to_rgba(gcols[g])[:3]] for g in glbl_ord]),
                    aspect='auto', interpolation='nearest')
        ax_g.set_xticks([]); ax_g.set_yticks(range(len(slbl_ord)))
        ax_g.set_yticklabels(slbl_ord, fontsize=7)

        # ── Heatmap ──────────────────────────────────────────────────────────
        im = ax_h.imshow(X_ord, aspect='auto', cmap=cmap,
                         interpolation='nearest',
                         vmin=np.percentile(X_ord, 2),
                         vmax=np.percentile(X_ord, 98))
        ax_h.set_xticks(range(n_feat))
        _MAX = 20
        flbl_trunc = [f'{l[:_MAX]}…' if len(l) > _MAX else l for l in flbl_ord]
        ax_h.set_xticklabels(flbl_trunc, rotation=45, ha='right', fontsize=6)
        ax_h.set_yticks([])

        self._fig.suptitle('Heatmap (hierarchical clustering)', fontsize=10)

        # bbox_inches='tight' in _render_and_display captures labels that
        # extend outside the figure — bottom margin only needs to provide
        # spacing between heatmap and label text, not full label height.
        self._fig.subplots_adjust(left=0.13, right=0.78, top=0.93, bottom=0.22)

        # ── Colorbar: upper ~40% of heatmap height, just right of it ─────────
        hm  = ax_h.get_position()          # Bbox in figure fractions
        cb_w = 0.018
        cb_h = hm.height * 0.42
        cb_x = hm.x1 + 0.014
        cb_y = hm.y0 + hm.height * 0.55   # centred in upper half
        cbax = self._fig.add_axes([cb_x, cb_y, cb_w, cb_h])
        cb   = self._fig.colorbar(im, cax=cbax)
        cb.set_label('Scaled intensity', fontsize=7)
        cb.ax.tick_params(labelsize=7)

        # ── Groups legend: below colorbar ────────────────────────────────────
        patches = [mpatches.Patch(color=gcols[g], label=g) for g in unique]
        leg_y = cb_y - 0.04
        self._fig.legend(handles=patches, fontsize=7,
                         loc='upper left',
                         bbox_to_anchor=(cb_x - 0.01, leg_y),
                         bbox_transform=self._fig.transFigure,
                         framealpha=0.85, title='Groups', title_fontsize=7)

    def _vip_scores(self, X, Y, n_comp):
        """Compute VIP scores from a PLSRegression model."""
        from sklearn.cross_decomposition import PLSRegression
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X, Y)
        T = pls.x_scores_
        W = pls.x_weights_
        Q = pls.y_loadings_
        p, h = W.shape
        vips = np.zeros(p)
        s    = np.diag(T.T @ T @ Q.T @ Q)
        s_total = np.sum(s)
        for i in range(p):
            weight = np.array([(W[i, j] / np.linalg.norm(W[:, j]))**2 * s[j]
                               for j in range(h)])
            vips[i] = np.sqrt(p * weight.sum() / s_total)
        return vips

    def _plot_correlation_linear(self, X, feature_labels, group_labels):
        """Scatter of one analyte vs another across samples, with an OLS trend
        line and a 95% confidence band (regression-mean CI). The reported
        coefficient follows the Coefficient selector: Pearson (parametric) or
        Spearman (non-parametric). See processing.correlation for the maths."""
        from processing.correlation import correlation_coeff, linear_fit_ci
        import matplotlib.cm as cm

        method = getattr(self, '_corr_method_var', None)
        method = method.get() if method else 'pearson'
        xname = (getattr(self, '_corr_x_var', None).get()
                 if getattr(self, '_corr_x_var', None) else '')
        yname = (getattr(self, '_corr_y_var', None).get()
                 if getattr(self, '_corr_y_var', None) else '')

        self._fig.clear()
        ax = self._fig.add_subplot(111)

        if (xname not in feature_labels or yname not in feature_labels
                or xname == yname):
            ax.text(0.5, 0.5,
                    'Pick two different analytes for X and Y\n'
                    '(Options → X analyte / Y analyte).',
                    transform=ax.transAxes, ha='center', va='center', color='#a00')
            ax.set_axis_off()
            return

        xi = feature_labels.index(xname)
        yi = feature_labels.index(yname)
        xdata = np.asarray(X[:, xi], dtype=float)
        ydata = np.asarray(X[:, yi], dtype=float)

        st = self._box_style
        y_int, uniq = _group_to_int(group_labels)
        gcmap = cm.get_cmap('tab10', max(len(uniq), 1))
        def _auto_hex(i):
            r, g, b, _ = gcmap(i)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        for gi, g in enumerate(uniq):
            m = np.asarray(y_int) == gi
            color = st['group_colors'].get(g, _auto_hex(gi))
            ax.scatter(xdata[m], ydata[m], s=30, color=color, alpha=0.9,
                       edgecolors='black', linewidths=0.4, label=g, zorder=3)

        fit = linear_fit_ci(xdata, ydata, alpha=0.05)
        r, p = correlation_coeff(xdata, ydata, method)
        n_fin = int(np.sum(np.isfinite(xdata) & np.isfinite(ydata)))
        if fit is not None:
            ax.fill_between(fit['xs'], fit['lower'], fit['upper'],
                            color='gray', alpha=0.25, zorder=1,
                            label='95% CI')
            ax.plot(fit['xs'], fit['ys'], color='black', linewidth=1.5,
                    zorder=2, label='OLS fit')

        coef_lbl = 'Pearson r' if method == 'pearson' else 'Spearman ρ'
        if np.isfinite(r):
            txt = f'{coef_lbl} = {r:.3f}\np = {p:.3g}\nn = {n_fin}'
        else:
            txt = f'{coef_lbl}: undefined\n(n = {n_fin}, need ≥3 varying pairs)'
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white',
                                      edgecolor='#999', alpha=0.85))

        q = (getattr(self.app, 'stages', {}) or {}).get('quantification')
        unit = 'a.u.'
        if q is not None and getattr(q, '_quant_results', None) \
                and hasattr(q, 'concentration_unit'):
            try:
                unit = q.concentration_unit() or 'a.u.'
            except Exception:
                unit = 'a.u.'
        ax.set_xlabel(f'{xname} ({unit})', fontsize=st.get('label_size', 9))
        ax.set_ylabel(f'{yname} ({unit})', fontsize=st.get('label_size', 9))
        ax.set_title(f'{yname} vs {xname}', fontsize=st.get('title_size', 11),
                     fontweight='bold' if st.get('title_bold', True) else 'normal')
        ax.tick_params(labelsize=st.get('tick_size', 7))
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        for _sp in ('top', 'right'):
            ax.spines[_sp].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.85, loc='best')

        # CSV export of the plotted values
        rows = [[float(xv), float(yv), g]
                for xv, yv, g in zip(xdata, ydata, group_labels)
                if np.isfinite(xv) and np.isfinite(yv)]
        self._plot_export = {
            'method': 'Correlation (linear)',
            'headers': [xname, yname, 'group'],
            'rows': rows,
        }

    def _plot_correlation(self, X, sample_names, feature_labels, group_labels):
        """Metabolite–metabolite correlation: clustered heatmap (A) and, when
        exactly two groups are present, the differential correlation (C).

        Spearman (rank) by default — robust to the non-normal, outlier-prone
        concentrations typical of NMR. Hierarchical clustering reorders features
        so co-regulated modules form visible blocks; the differential panel
        highlights pairs whose association rewires between the two groups.
        """
        _pv = getattr(self, '_corr_plot_var', None)
        if _pv is not None and _pv.get() == 'Linear (X vs Y)':
            self._plot_correlation_linear(X, feature_labels, group_labels)
            return

        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        from scipy.stats import spearmanr, t as _tdist

        cmethod = getattr(self, '_corr_method_var', None)
        cmethod = cmethod.get() if cmethod else 'spearman'
        linkm   = getattr(self, '_corr_link_var', None)
        linkm   = linkm.get() if linkm else 'average'
        topn    = getattr(self, '_corr_topn_var', None)
        topn    = topn.get() if topn else 'All'
        sig_only = getattr(self, '_corr_sig_var', None)
        sig_only = bool(sig_only.get()) if sig_only is not None else False
        ALPHA = 0.05

        def _corr(mat):
            """(n_samples, n_features) → (R, P) feature×feature."""
            n_s, n_f = mat.shape
            if n_s < 3:
                return np.full((n_f, n_f), np.nan), np.ones((n_f, n_f))
            if cmethod == 'pearson':
                R = np.corrcoef(mat, rowvar=False)
                with np.errstate(divide='ignore', invalid='ignore'):
                    tt = R * np.sqrt((n_s - 2) / np.clip(1 - R ** 2, 1e-12, None))
                    P = 2.0 * _tdist.sf(np.abs(tt), n_s - 2)
            else:
                R, P = spearmanr(mat, axis=0)
                R = np.atleast_2d(R); P = np.atleast_2d(P)
            R = np.nan_to_num(R, nan=0.0)
            P = np.nan_to_num(P, nan=1.0)
            np.fill_diagonal(P, 0.0)
            return R, P

        def _fdr_sig(P):
            """Benjamini-Hochberg significance mask over the upper triangle."""
            n = P.shape[0]
            iu = np.triu_indices(n, k=1)
            pv = P[iu]
            m = pv.size
            mask = np.zeros((n, n), dtype=bool)
            if m == 0:
                return mask
            order_p = np.argsort(pv)
            ranked = pv[order_p]
            thresh = ALPHA * (np.arange(1, m + 1) / m)
            passed = np.where(ranked <= thresh)[0]
            crit = ranked[passed.max()] if passed.size else -1.0
            sig = pv <= crit if crit >= 0 else np.zeros(m, dtype=bool)
            mask[iu[0][sig], iu[1][sig]] = True
            mask |= mask.T
            return mask

        R, P = _corr(X)
        sig = _fdr_sig(P)
        feature_labels = list(feature_labels)

        # ── Feature selection: significant-only drop + Top N by connectivity ──
        keep = list(range(R.shape[0]))
        if sig_only:
            keep = [i for i in keep if sig[i].sum() > 0]
        if topn != 'All':
            try:
                N = int(topn)
                degree = sig.sum(axis=1) if sig.any() else np.abs(R).sum(axis=1)
                ranked_idx = sorted(keep, key=lambda i: degree[i], reverse=True)[:N]
                keep = sorted(ranked_idx)
            except (ValueError, TypeError):
                pass
        if len(keep) < 2:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No significant correlations to display\n'
                    '(loosen filters / Top N).', ha='center', va='center')
            ax.axis('off')
            return

        sub = np.ix_(keep, keep)
        R, P, sig = R[sub], P[sub], sig[sub]
        feature_labels = [feature_labels[i] for i in keep]
        X = X[:, keep]
        n_feat = len(keep)

        # Hierarchical clustering order from the (sub)matrix (1 − |r|).
        dist = 1.0 - np.abs(R)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0          # enforce symmetry for squareform
        try:
            Z = linkage(squareform(dist, checks=False), method=linkm)
            order = list(leaves_list(Z))
        except Exception:
            order = list(range(n_feat))
        feats_o = [feature_labels[i] for i in order]
        Ro = R[np.ix_(order, order)]
        if sig_only:
            sig_o = sig[np.ix_(order, order)]
            Ro = np.ma.masked_where(~sig_o & ~np.eye(n_feat, dtype=bool), Ro)

        # CSV export: clustered metabolite × metabolite correlation matrix.
        _Rf = (np.ma.filled(Ro, np.nan) if np.ma.isMaskedArray(Ro)
               else np.asarray(Ro))
        self._plot_export = {
            'method': 'Correlation',
            'headers': ['metabolite'] + list(feats_o),
            'rows': [[feats_o[i]] + [float(_Rf[i, j]) for j in range(n_feat)]
                     for i in range(n_feat)],
        }

        # Differential panel: contrast the two user-selected groups (falls back
        # to the first two present when no/invalid selection).
        groups = [g for g in dict.fromkeys(group_labels)
                  if str(g).upper() != 'BLK']
        rows_by_g = {g: [i for i, gl in enumerate(group_labels) if gl == g]
                     for g in groups}
        sel1 = getattr(self, '_corr_g1_var', None)
        sel2 = getattr(self, '_corr_g2_var', None)
        sel1 = sel1.get() if sel1 else ''
        sel2 = sel2.get() if sel2 else ''
        if sel1 in rows_by_g and sel2 in rows_by_g and sel1 != sel2:
            g1, g2 = sel1, sel2
        elif len(groups) == 2:
            g1, g2 = groups
        else:
            g1 = g2 = None
        do_diff = (g1 is not None
                   and all(len(rows_by_g[g]) >= 3 for g in (g1, g2)))

        if do_diff:
            R1, _ = _corr(X[rows_by_g[g1], :])
            R2, _ = _corr(X[rows_by_g[g2], :])
            D  = (R1 - R2)[np.ix_(order, order)]
            D  = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
            axA = self._fig.add_subplot(1, 2, 1)
            axC = self._fig.add_subplot(1, 2, 2)
        else:
            axA = self._fig.add_subplot(1, 1, 1)
            axC = None

        # ── Panel A: clustered correlation heatmap ────────────────────────────
        import matplotlib as _mpl
        _cmapA = _mpl.colormaps['RdBu_r'].copy()
        _cmapA.set_bad('#f2f2f2')             # non-significant cells → light grey
        imA = axA.imshow(Ro, cmap=_cmapA, vmin=-1, vmax=1, aspect='auto')
        axA.set_xticks(range(n_feat)); axA.set_yticks(range(n_feat))
        _fs = max(4, min(8, int(420 / max(n_feat, 1))))
        # Truncate long names (same as heatmap) so 45°-rotated labels don't pile
        # onto each other; full name stays in the y-axis order / tooltip.
        _MAX = 20
        feats_t = [f'{l[:_MAX]}…' if len(l) > _MAX else l for l in feats_o]
        axA.set_xticklabels(feats_t, rotation=45, ha='right', fontsize=_fs)
        axA.set_yticklabels(feats_t, fontsize=_fs)
        axA.set_title(f'A · Clustered correlation ({cmethod.title()})',
                      fontsize=10, fontweight='bold')
        self._fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04,
                           label='ρ' if cmethod == 'spearman' else 'r')

        # ── Panel C: differential correlation ─────────────────────────────────
        if axC is not None:
            _absD = np.abs(D)
            _finite = _absD[np.isfinite(_absD)]
            _peak = float(_finite.max()) if _finite.size else 0.0
            vmax = float(min(2.0, _peak)) if _peak > 0 else 1.0
            imC = axC.imshow(D, cmap='PRGn', vmin=-vmax, vmax=vmax, aspect='auto')
            axC.set_xticks(range(n_feat)); axC.set_yticks(range(n_feat))
            axC.set_xticklabels(feats_t, rotation=45, ha='right', fontsize=_fs)
            axC.set_yticklabels(feats_t, fontsize=_fs)
            axC.set_title(f'C · Differential ({g1} − {g2})',
                          fontsize=10, fontweight='bold')
            self._fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04, label='Δρ')
        else:
            _msg = ('Differential panel needs 2 distinct groups with ≥3 samples each'
                    if len(groups) >= 2
                    else f'Differential panel needs 2 groups (have {len(groups)})')
            axA.text(0.5, -0.18, _msg,
                     transform=axA.transAxes, ha='center', va='top',
                     fontsize=7, color='#666')

        # Spacing: wide wspace so the two panels' rotated labels don't collide,
        # and a bottom margin so the x labels clear the axes (bbox_inches='tight'
        # in _render_and_display then crops the outer whitespace).
        self._fig.subplots_adjust(left=0.13, right=0.95, top=0.92,
                                  bottom=0.28, wspace=0.55)

    def _plot_vip(self, X, sample_names, feature_labels, group_labels):
        y_int, unique = _group_to_int(group_labels)
        if len(unique) < 2:
            raise ValueError('Need at least 2 groups for VIP.')
        from sklearn.preprocessing import label_binarize
        if len(unique) == 2:
            Y = y_int.reshape(-1, 1).astype(float)
        else:
            Y = label_binarize(y_int, classes=list(range(len(unique)))).astype(float)

        n_comp = getattr(self, '_n_comp_var', None)
        n_comp = n_comp.get() if n_comp else 2
        n_comp = min(n_comp, X.shape[0]-1, X.shape[1])
        top_n  = getattr(self, '_top_n_var',  None)
        top_n  = top_n.get()  if top_n  else 20
        vip_thr = getattr(self, '_vip_thr_var', None)
        vip_thr = vip_thr.get() if vip_thr else 1.5

        vips = self._vip_scores(X, Y, n_comp)
        self._last_vip_scores = vips
        order = np.argsort(vips)[::-1][:top_n]
        vips_top  = vips[order]
        feats_top = [feature_labels[i] for i in order]

        # Per-group mean on the (already autoscaled) X for the class-direction
        # squares: >0 → above grand mean (high, red), <0 → low (blue). Mirrors
        # the coloured class strip on the right of a MetaboAnalyst VIP plot.
        gmeans = np.zeros((len(order), len(unique)))
        for gi in range(len(unique)):
            rows = np.where(y_int == gi)[0]
            if rows.size:
                gmeans[:, gi] = np.mean(X[np.ix_(rows, order)], axis=0)

        # CSV export: VIP score per metabolite + per-group mean (autoscaled).
        self._plot_export = {
            'method': 'VIP Score',
            'headers': (['metabolite', 'vip_score', 'above_threshold']
                        + ['mean_' + str(unique[gi]) for gi in range(len(unique))]),
            'rows': [([feats_top[r], float(vips_top[r]),
                       bool(vips_top[r] >= vip_thr)]
                      + [float(gmeans[r, gi]) for gi in range(len(unique))])
                     for r in range(len(feats_top))],
        }

        # Layout: VIP bars (left) + class-mean squares (right), aligned rows.
        gs = self._fig.add_gridspec(
            1, 2, width_ratios=[5, max(1.0, 0.45 * len(unique))], wspace=0.04)
        ax  = self._fig.add_subplot(gs[0, 0])
        axh = self._fig.add_subplot(gs[0, 1], sharey=ax)

        colors = ['#c62828' if v >= vip_thr else '#1565c0' for v in vips_top]
        ax.barh(range(len(feats_top)), vips_top[::-1], color=colors[::-1],
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(feats_top)))
        ax.set_yticklabels(feats_top[::-1], fontsize=7)
        ax.set_ylim(-0.5, len(feats_top) - 0.5)
        ax.axvline(vip_thr, color='#c62828', lw=1.0, ls='--',
                   label=f'VIP = {vip_thr:.1f}')
        ax.set_xlabel('VIP Score', fontsize=9)
        ax.set_title(f'VIP Scores (PLS-DA, top {top_n})', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_facecolor('#f9f9f9')

        # Right strip: one coloured square per group, rows reversed to match bars.
        _vip_cmap = getattr(self, '_vip_cmap_var', None)
        _vip_cmap = _vip_cmap.get() if _vip_cmap else 'RdBu_r'
        _vmax = float(np.nanmax(np.abs(gmeans))) or 1.0
        im = axh.imshow(gmeans[::-1, :], aspect='auto', cmap=_vip_cmap,
                        vmin=-_vmax, vmax=_vmax, origin='lower',
                        extent=(-0.5, len(unique) - 0.5,
                                -0.5, len(feats_top) - 0.5))
        axh.set_xticks(range(len(unique)))
        axh.set_xticklabels([str(g) for g in unique], rotation=90, fontsize=7)
        axh.tick_params(left=False, labelleft=False)
        # White separators → discrete-square look.
        axh.set_xticks(np.arange(-0.5, len(unique), 1), minor=True)
        axh.set_yticks(np.arange(-0.5, len(feats_top), 1), minor=True)
        axh.grid(which='minor', color='white', linewidth=1.5)
        axh.set_title('Class', fontsize=8)
        # Half-height colorbar, vertically centred on the class strip.
        cb = self._fig.colorbar(im, ax=axh, fraction=0.25, pad=0.06,
                                shrink=0.5, anchor=(0.0, 0.5),
                                ticks=[-_vmax, _vmax])
        cb.ax.set_yticklabels(['Low', 'High'], fontsize=7)

    def _ensure_volcano_style(self):
        """Lazily create the persistent volcano style dict with defaults."""
        if not hasattr(self, '_volcano_style') or not self._volcano_style:
            self._volcano_style = {
                'theme':          'Classic',   # 'Classic' | 'Compact'
                'n_labels':       10,
                'color_up':       '#c62828',
                'color_down':     '#1565c0',
                'color_ns':       '#888888',
                'dot_size':       28,
                'label_fontsize': 6.5,
                'label_bold':     False,
                'leader_lines':   True,
            }
        return self._volcano_style

    def _open_volcano_style_dialog(self):
        """Editor for volcano label/colour style. Re-renders on Apply."""
        from tkinter import colorchooser
        st = self._ensure_volcano_style()

        dlg = tk.Toplevel(self.frame)
        dlg.title('Volcano Style')
        dlg.resizable(False, False)
        dlg.transient(self.frame)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        theme_var = tk.StringVar(value=str(st.get('theme', 'Classic')))
        n_var    = tk.IntVar(value=int(st['n_labels']))
        size_var = tk.DoubleVar(value=float(st['label_fontsize']))
        dot_var  = tk.IntVar(value=int(st['dot_size']))
        bold_var = tk.BooleanVar(value=bool(st['label_bold']))
        lead_var = tk.BooleanVar(value=bool(st['leader_lines']))
        colors   = {'color_up': st['color_up'], 'color_down': st['color_down'],
                    'color_ns': st['color_ns']}

        r = 0
        ttk.Label(frm, text='Theme:').grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(frm, textvariable=theme_var, state='readonly', width=12,
                     values=['Classic', 'Compact']).grid(
            row=r, column=1, sticky=tk.W, padx=6)
        ttk.Label(frm, text='(Compact: clean axes, no labels)',
                  font=('Arial', 7), foreground='#666').grid(
            row=r, column=2, sticky=tk.W)
        r += 1
        ttk.Label(frm, text='Labels shown (top significant):').grid(
            row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(frm, from_=0, to=100, increment=1, width=6,
                    textvariable=n_var).grid(row=r, column=1, sticky=tk.W, padx=6)
        r += 1
        ttk.Label(frm, text='Label font size:').grid(
            row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(frm, from_=3.0, to=20.0, increment=0.5, width=6, format='%.1f',
                    textvariable=size_var).grid(row=r, column=1, sticky=tk.W, padx=6)
        r += 1
        ttk.Label(frm, text='Dot size:').grid(
            row=r, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(frm, from_=5, to=120, increment=1, width=6,
                    textvariable=dot_var).grid(row=r, column=1, sticky=tk.W, padx=6)
        r += 1
        ttk.Checkbutton(frm, text='Bold labels', variable=bold_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W, pady=2)
        r += 1
        ttk.Checkbutton(frm, text='Leader lines for crowded labels',
                        variable=lead_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W, pady=2)
        r += 1

        swatches = {}
        for key, label in (('color_up',   'Up dots colour'),
                           ('color_down', 'Down dots colour'),
                           ('color_ns',   'NS dots colour')):
            ttk.Label(frm, text=label + ':').grid(
                row=r, column=0, sticky=tk.W, pady=2)
            sw = tk.Label(frm, width=6, background=colors[key], relief='solid', bd=1)
            sw.grid(row=r, column=1, sticky=tk.W, padx=6)
            swatches[key] = sw

            def _pick(k=key):
                c = colorchooser.askcolor(color=colors[k], parent=dlg)[1]
                if c:
                    colors[k] = c
                    swatches[k].config(background=c)
            ttk.Button(frm, text='…', width=3,
                       command=_pick).grid(row=r, column=2, padx=2)
            r += 1

        def _apply():
            st['theme']          = str(theme_var.get())
            st['n_labels']       = int(n_var.get())
            st['label_fontsize'] = float(size_var.get())
            st['dot_size']       = int(dot_var.get())
            st['label_bold']     = bool(bold_var.get())
            st['leader_lines']   = bool(lead_var.get())
            st['color_up']       = colors['color_up']
            st['color_down']     = colors['color_down']
            st['color_ns']       = colors['color_ns']
            dlg.destroy()
            try:
                self._run()
            except Exception:
                pass

        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))
        ttk.Button(btns, text='Apply', command=_apply).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text='Cancel', command=dlg.destroy).pack(side=tk.LEFT, padx=4)

        dlg.update_idletasks()
        sw_, sh_ = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        w_, h_ = dlg.winfo_width(), dlg.winfo_height()
        dlg.geometry(f'+{(sw_ - w_) // 2}+{(sh_ - h_) // 2}')

    def _repel_labels(self, ax, xy, labels, fontsize, bold, leader, leader_color):
        """Place point labels, fanned outward, with bounded vertical de-overlap.

        Each label starts just beside its dot, fanning OUTWARD from the plot
        centre (left-half points label left, right-half right) so names never
        cross the busy middle of the volcano. Overlapping labels are then pushed
        apart VERTICALLY ONLY, in display (pixel) space, for a fixed number of
        passes — deterministic and impossible to hang. Unlike the old iterative
        repel (which mixed point/pixel units, drifted labels off-axes and tangled
        leader lines), labels never move horizontally and stay in their dot's
        column, so an optional leader is a short, near-vertical connector — no
        crossing tangle. Final positions are clamped to the axes so nothing is
        pushed outside the plotting area.
        """
        if not labels:
            return
        weight = 'bold' if bold else 'normal'
        xs = [p[0] for p in xy]
        x_mid = (min(xs) + max(xs)) / 2.0 if xs else 0.0

        fig   = ax.figure
        trans = ax.transData
        inv   = trans.inverted()

        # Freeze the data limits before adding labels. Annotations placed in data
        # coords otherwise enlarge the axes on the next draw (autoscale), shifting
        # transData and invalidating the pixel-space de-overlap below. With limits
        # locked, the geometry we measure stays valid through the final draw.
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        # Place every label beside its dot, fanned to the outward side.
        anns = []                                    # (annotation, (dot_x, dot_y), ha)
        for (x, y), lab in zip(xy, labels):
            dpx, dpy = trans.transform((x, y))
            if x < x_mid:
                ha, ox = 'right', -5.0
            else:
                ha, ox = 'left', 5.0
            tx, ty = inv.transform((dpx + ox, dpy))
            ann = ax.annotate(
                lab, xy=(x, y), xytext=(tx, ty), textcoords='data',
                fontsize=fontsize, weight=weight, color='#222',
                zorder=6, ha=ha, va='center')
            anns.append((ann, (x, y), ha))

        # De-overlap using REAL text boxes from the renderer — estimating text
        # width under-detects collisions. Labels move VERTICALLY ONLY; a fixed
        # pass cap keeps it deterministic and hang-proof. If no renderer exists
        # yet, the base placement above is kept (never worse than before).
        try:
            fig.canvas.draw()                        # realize axes geometry
            renderer = fig.canvas.get_renderer()      # (limits frozen → no shift)
        except Exception:
            renderer = None

        if renderer is not None:
            # The draw above can move the axes; transData.inverted() is a snapshot,
            # so refresh it (and use it freshly) to map pixels back to data with
            # the FINAL geometry — a stale inverse drops labels off the axes.
            inv = trans.inverted()

            def _set_center_y(ann, cpy):
                px, _ = trans.transform(ann.get_position())
                ann.set_position(inv.transform((px, cpy)))

            try:
                ab = ax.get_window_extent(renderer)
                ax_y0, ax_y1 = ab.y0, ab.y1
            except Exception:
                ax_y0, ax_y1 = None, None

            gap = 2.0
            # Each fan side is an independent vertical column. Greedy single-pass
            # stacking (top→down) guarantees a non-overlapping column without the
            # compounding pairwise pushes that previously exploded the spacing.
            for side in ('right', 'left'):
                grp = [a for a in anns if a[2] == side]
                if not grp:
                    continue
                recs = []
                for ann, (dx, dy0), _ in grp:
                    apy = trans.transform((dx, dy0))[1]    # dot anchor (px)
                    h   = ann.get_window_extent(renderer).height
                    recs.append([ann, apy, h, apy])        # last = target centre
                recs.sort(key=lambda r: -r[1])             # top (high px) first
                prev_bottom = None
                for r in recs:
                    c = r[1]
                    if prev_bottom is not None and c + r[2] / 2.0 > prev_bottom - gap:
                        c = prev_bottom - gap - r[2] / 2.0
                    r[3] = c
                    prev_bottom = c - r[2] / 2.0

                # Rigid-translate the whole column to fit the axes (preserves the
                # spacing). A column taller than the axes is top-aligned — overflow
                # is then unavoidable without shrinking the font.
                if ax_y1 is not None:
                    blk_top = max(r[3] + r[2] / 2.0 for r in recs)
                    blk_bot = min(r[3] - r[2] / 2.0 for r in recs)
                    dy = 0.0
                    if blk_top > ax_y1:
                        dy = ax_y1 - blk_top
                    if blk_bot + dy < ax_y0:
                        dy += ax_y0 - (blk_bot + dy)
                else:
                    dy = 0.0
                for ann, _, _, c in recs:
                    _set_center_y(ann, c + dy)

            # Short connector from dot to a label that ended up displaced.
            if leader:
                for ann, (dx, dy0), _ in anns:
                    tx2, ty2 = ann.get_position()
                    if abs(trans.transform((tx2, ty2))[1]
                           - trans.transform((dx, dy0))[1]) > fontsize:
                        ax.plot([dx, tx2], [dy0, ty2], color=leader_color,
                                lw=0.5, alpha=0.7, zorder=5)

    def _plot_volcano(self, X_raw, sample_names, feature_labels, group_labels):
        from scipy import stats
        ga = getattr(self, '_grp_a_var', None)
        gb = getattr(self, '_grp_b_var', None)
        ga = ga.get() if ga else ''
        gb = gb.get() if gb else ''
        if not ga or not gb or ga == gb:
            raise ValueError('Select two different groups for Volcano plot.')

        fc_thr = getattr(self, '_fc_thr_var', None)
        fc_thr = fc_thr.get() if fc_thr else 1.5
        p_thr  = getattr(self, '_p_thr_var', None)
        p_thr  = p_thr.get() if p_thr else 0.05
        p_adj  = getattr(self, '_p_adj_var', None)
        p_adj  = p_adj.get() if p_adj else True

        idx_a = [i for i, g in enumerate(group_labels) if g == ga]
        idx_b = [i for i, g in enumerate(group_labels) if g == gb]
        if not idx_a or not idx_b:
            raise ValueError(f'No samples found for group "{ga}" or "{gb}".')

        Xa = X_raw[idx_a, :]
        Xb = X_raw[idx_b, :]

        eps = 1e-9
        fc  = (Xb.mean(axis=0) + eps) / (Xa.mean(axis=0) + eps)
        log2fc = np.log2(fc)

        # Drop features whose fold change is non-finite — happens when a group
        # mean is zero/negative (e.g. after baseline/normalization), giving
        # fc <= 0 → log2(fc) = nan/-inf. Plotting those points (or annotating
        # them) makes the tight-bbox export compute an infinite figure size,
        # which RendererAgg rejects and the whole plot fails to render. We never
        # interpolate the value — the feature is excluded and the user warned.
        finite = np.isfinite(log2fc)
        n_dropped = int((~finite).sum())
        if not finite.any():
            raise ValueError(
                'No features with a finite fold change between the two groups.\n'
                'Check that concentrations are positive (Volcano needs raw, '
                'non-normalized values).')
        if n_dropped:
            messagebox.showwarning(
                'Volcano',
                f'{n_dropped} feature(s) excluded: group mean ≤ 0 gives a '
                'non-finite fold change. They are not plotted.',
                parent=self.frame)
            Xa = Xa[:, finite]
            Xb = Xb[:, finite]
            log2fc = log2fc[finite]
            feature_labels = [l for l, ok in zip(feature_labels, finite) if ok]

        pvals = np.array([stats.ttest_ind(Xa[:, j], Xb[:, j],
                                          equal_var=False).pvalue
                          for j in range(Xa.shape[1])])
        pvals = np.nan_to_num(pvals, nan=1.0)

        if p_adj:
            # Benjamini-Hochberg FDR
            n   = len(pvals)
            ord_i = np.argsort(pvals)
            pvals_adj = pvals.copy()
            for rank, idx in enumerate(ord_i):
                pvals_adj[idx] = min(pvals[idx] * n / (rank + 1), 1.0)
            # Enforce monotonicity
            min_val = 1.0
            for idx in ord_i[::-1]:
                pvals_adj[idx] = min(pvals_adj[idx], min_val)
                min_val = pvals_adj[idx]
            pvals_plot = pvals_adj
        else:
            pvals_plot = pvals

        neg_log_p = -np.log10(np.clip(pvals_plot, 1e-15, 1.0))

        style = self._ensure_volcano_style()
        c_up   = style['color_up']
        c_down = style['color_down']
        c_ns   = style['color_ns']
        dot_s  = float(style['dot_size'])

        ax = self._fig.add_subplot(111)
        sig_up   = (log2fc > np.log2(fc_thr))  & (pvals_plot < p_thr)
        sig_down = (log2fc < -np.log2(fc_thr)) & (pvals_plot < p_thr)
        ns       = ~(sig_up | sig_down)

        # Capture ONLY the significant features (sig_up or sig_down) for CSV
        # export — those past the FC and p thresholds, i.e. the hits the user
        # wants to replot. fc is recomputed from the (filtered) log2fc so it
        # aligns with feature_labels after non-finite features were dropped.
        _fc_used = np.power(2.0, log2fc)
        self._plot_export = {
            'method': 'Volcano',
            'headers': ['metabolite', 'mean_' + ga, 'mean_' + gb,
                        'log2_fold_change', 'fold_change', 'p_value',
                        'p_value_adjusted' if p_adj else 'p_value_raw',
                        'neg_log10_p', 'direction'],
            'rows': [[feature_labels[j],
                      float(Xa[:, j].mean()), float(Xb[:, j].mean()),
                      float(log2fc[j]), float(_fc_used[j]),
                      float(pvals[j]), float(pvals_plot[j]),
                      float(neg_log_p[j]),
                      'up in ' + gb if sig_up[j] else 'down in ' + gb]
                     for j in range(len(feature_labels))
                     if sig_up[j] or sig_down[j]],
        }

        if str(style.get('theme', 'Classic')) == 'Compact':
            # Compact/clean publication theme: light (Negative) / dark (Positive) /
            # grey (Insignificant) classes, minimal axes (no top/right spines, white
            # background, no threshold guides), simple legend, no per-feature labels.
            h_ns = ax.scatter(log2fc[ns], neg_log_p[ns], color=c_ns,
                              s=dot_s * 0.8, alpha=0.9, edgecolors='none')
            h_dn = ax.scatter(log2fc[sig_down], neg_log_p[sig_down], color=c_down,
                              s=dot_s, alpha=0.95, edgecolors='none')
            h_up = ax.scatter(log2fc[sig_up], neg_log_p[sig_up], color=c_up,
                              s=dot_s, alpha=0.95, edgecolors='none')
            for _sp in ('top', 'right'):
                ax.spines[_sp].set_visible(False)
            ax.set_facecolor('white')
            ax.set_xlabel('Fold change (log₂)', fontsize=11)
            ax.set_ylabel('−log₁₀ (P-value)', fontsize=11)
            ax.legend([h_dn, h_up, h_ns], ['Negative', 'Positive', 'Insignificant'],
                      fontsize=9, loc='upper right', frameon=True,
                      framealpha=1.0, edgecolor='#333', handletextpad=0.4)
            ax.set_title(f'{ga} vs {gb}', fontsize=12, loc='left', fontweight='bold')
        else:
            ax.scatter(log2fc[ns],       neg_log_p[ns],       color=c_ns, s=dot_s * 0.7,
                       alpha=0.5, label='NS', edgecolors='none')
            ax.scatter(log2fc[sig_up],   neg_log_p[sig_up],   color=c_up, s=dot_s,
                       alpha=0.85, label=f'Up in {gb}', edgecolors='white', lw=0.4)
            ax.scatter(log2fc[sig_down], neg_log_p[sig_down], color=c_down, s=dot_s,
                       alpha=0.85, label=f'Up in {ga}', edgecolors='white', lw=0.4)

            ax.axhline(-np.log10(p_thr), color='#555', lw=0.8, ls='--',
                       label=f'p={p_thr}')
            ax.axvline( np.log2(fc_thr), color='#999', lw=0.6, ls=':')
            ax.axvline(-np.log2(fc_thr), color='#999', lw=0.6, ls=':')
            ax.set_xlabel(f'log₂ FC ({gb} / {ga})', fontsize=9)
            p_label = 'FDR-adjusted p' if p_adj else 'p-value'
            ax.set_ylabel(f'−log₁₀({p_label})', fontsize=9)
            ax.set_title(f'Volcano: {ga} vs {gb}  '
                         f'({sig_up.sum()} up, {sig_down.sum()} down)', fontsize=10)
            ax.legend(fontsize=7.5, loc='best', framealpha=0.85)
            ax.set_facecolor('#f9f9f9')

            # Labels LAST — after axes/legend/title fix the final geometry — so
            # the renderer-based de-overlap (see _repel_labels) measures the true
            # text positions. Top-N significant features, capped for readability.
            n_lab = min(int(style['n_labels']), 40)
            if n_lab > 0:
                sig_mask = pvals_plot < p_thr
                cand = [i for i in np.argsort(neg_log_p)[::-1] if sig_mask[i]][:n_lab]
                self._repel_labels(
                    ax,
                    xy=[(log2fc[i], neg_log_p[i]) for i in cand],
                    labels=[feature_labels[i] for i in cand],
                    fontsize=float(style['label_fontsize']),
                    bold=bool(style['label_bold']),
                    leader=bool(style['leader_lines']),
                    leader_color='#555')

    # ── Export ────────────────────────────────────────────────────────────────

    def _plot_metabolic_profile(self, X, sample_names, feature_labels, group_labels):
        """Pie / Donut chart of mean concentration per metabolite, one subplot per group."""
        chart_type  = getattr(self, '_pie_type_var',   None)
        top_n       = getattr(self, '_pie_topn_var',   None)
        other_thr   = getattr(self, '_pie_other_var',  None)
        show_pct    = getattr(self, '_pie_pct_var',    None)
        show_legend = getattr(self, '_pie_legend_var', None)

        chart_type  = chart_type.get()  if chart_type  else 'Donut'
        top_n       = int(top_n.get())  if top_n       else 10
        other_thr   = other_thr.get()   if other_thr   else 3.0
        show_pct    = show_pct.get()    if show_pct    else True
        show_legend = show_legend.get() if show_legend else True
        is_donut    = (chart_type == 'Donut')

        y, unique_groups = _group_to_int(group_labels)

        # Re-fetch raw (unscaled) concentrations — pie charts need absolute values.
        X_raw, _, feat_raw, gl_raw = _get_quantification_matrix(self.app)
        if X_raw is not None:
            X_raw, feat_raw = self._remove_nan(X_raw, feat_raw)
            if self._excluded_samples:
                _si = getattr(self.app, 'samples_info', []) or []
                _sn = [s.get('name', f'S{i}') for i, s in enumerate(_si)]
                _kr = [i for i, n in enumerate(_sn) if n not in self._excluded_samples]
                X_raw  = X_raw[_kr, :]
                gl_raw = [gl_raw[i] for i in _kr]
            if self._excluded_metabolites:
                _kc = [i for i, f in enumerate(feat_raw) if f not in self._excluded_metabolites]
                X_raw    = X_raw[:, _kc]
                feat_raw = [feat_raw[i] for i in _kc]
            y_raw, _ = _group_to_int(gl_raw)
        else:
            X_raw    = np.abs(X)
            feat_raw = feature_labels
            y_raw    = y
            gl_raw   = group_labels

        n_mets = len(feat_raw)

        # Random colors — one per metabolite, reproducible seed, consistent across groups.
        # User overrides from _pie_style['met_colors'] take precedence.
        import colorsys
        rng = np.random.default_rng(seed=42)
        def _rand_color_hsv():
            h = rng.uniform(0, 1); s = rng.uniform(0.45, 0.85); v = rng.uniform(0.65, 0.92)
            return colorsys.hsv_to_rgb(h, s, v)
        _auto_met_colors = [_rand_color_hsv() for _ in range(n_mets)]
        _override = self._pie_style.get('met_colors', {})
        def _hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
        met_colors = [
            _hex_to_rgb(_override[feat_raw[i]]) if feat_raw[i] in _override else _auto_met_colors[i]
            for i in range(n_mets)
        ]

        n_groups = len(unique_groups)
        # Max 2 columns — extra groups stack into new rows so the figure grows
        # downward without losing per-pie proportion/quality.
        ncols = min(n_groups, 2)
        nrows = (n_groups + ncols - 1) // ncols
        # Store desired export size (not applied to canvas — fits window as-is)
        self._pie_export_figsize = (max(7.0 * ncols, 7), max(5.0 * nrows, 4.5))
        self._fig.clear()

        for gi, grp in enumerate(unique_groups):
            mask  = np.array(y_raw) == gi
            Xg    = X_raw[mask, :]
            means = np.nanmean(Xg, axis=0)

            # Sort descending by mean concentration; drop zero/negative entries
            order  = np.argsort(means)[::-1]
            labels = [feat_raw[i] for i in order]
            vals   = means[order]
            colors = [met_colors[i] for i in order]

            # Remove metabolites with no signal in this group
            pos_mask = vals > 0
            labels = [l for l, m in zip(labels, pos_mask) if m]
            colors = [c for c, m in zip(colors, pos_mask) if m]
            vals   = vals[pos_mask]
            if vals.sum() <= 0:
                continue
            total  = float(vals.sum())
            pcts   = vals / total * 100.0

            # Keep top_n slices; everything beyond → "Other"
            n_keep      = min(top_n, len(vals))
            final_labels = list(labels[:n_keep])
            final_vals   = list(vals[:n_keep])
            final_colors = list(colors[:n_keep])
            other_val    = float(vals[n_keep:].sum())
            if other_val > 0:
                final_labels.append('Other')
                final_vals.append(other_val)
                final_colors.append('#cccccc')

            final_total = sum(final_vals)

            ax = self._fig.add_subplot(nrows, ncols, gi + 1)

            autopct_fn = (lambda p: f'{p:.1f}%' if p >= other_thr else '') if show_pct else None

            st = self._pie_style
            edge_color  = st.get('edge_color', '#ffffff')
            edge_width  = st.get('edge_width', 0.8)
            grp_size    = st.get('group_title_size', 9)
            grp_weight  = 'bold' if st.get('group_title_bold', True) else 'normal'
            grp_label   = st.get('group_titles', {}).get(grp, grp)

            result = ax.pie(
                final_vals,
                labels=None,
                colors=final_colors,
                autopct=autopct_fn,
                startangle=90,
                wedgeprops=dict(width=0.55 if is_donut else 1.0,
                                edgecolor=edge_color, linewidth=edge_width),
                pctdistance=0.78 if is_donut else 0.62,
            )
            wedges = result[0]
            if len(result) > 2:
                for t in result[2]:
                    t.set_fontsize(6.5)

            ax.set_title(grp_label, fontsize=grp_size, fontweight=grp_weight, pad=6)

            if show_legend:
                legend_labels = [
                    f'{lbl}  ({v / final_total * 100:.1f}%)'
                    for lbl, v in zip(final_labels, final_vals)
                ]
                ax.legend(wedges, legend_labels,
                          loc='center left',
                          bbox_to_anchor=(1.02, 0.5),
                          fontsize=7.8, frameon=False)

        _st = self._pie_style
        self._fig.suptitle(
            _st.get('title', 'Metabolic Profile by Group'),
            fontsize=_st.get('title_size', 11),
            fontweight='bold' if _st.get('title_bold', True) else 'normal')
        # Clear the 'constrained' engine set in _run() before tight_layout, else
        # matplotlib warns about the engine switch (None clears it cleanly).
        self._fig.set_layout_engine(None)
        self._fig.tight_layout(pad=_st.get('tight_pad', 1.0), rect=[0, 0, 1, 0.95])

    # ── Box / Violin style dialog ─────────────────────────────────────────────

    def _open_box_style_dialog(self):
        from tkinter import colorchooser
        st = self._box_style

        dlg = tk.Toplevel(self.frame)
        dlg.title('Edit Box / Violin Style')
        dlg.resizable(True, True)
        dlg.grab_set()
        dlg.update_idletasks()
        px = self.frame.winfo_rootx() + self.frame.winfo_width() // 2
        py = self.frame.winfo_rooty() + self.frame.winfo_height() // 2
        dlg.geometry(f'420x500+{px - 210}+{py - 250}')

        nb = ttk.Notebook(dlg)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Tab 1: Title & labels ─────────────────────────────────────────────
        tab_t = ttk.Frame(nb, padding=10)
        nb.add(tab_t, text='Title & Labels')

        fields = [
            ('Chart title:',  '_tv',  st['title'],     0),
            ('X-axis label:', '_xv',  st['xlabel'],    2),
            ('Y-axis label:', '_yv',  st['ylabel'],    4),
        ]
        _vars = {}
        for lbl, key, val, row in fields:
            ttk.Label(tab_t, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=2)
            v = tk.StringVar(value=val)
            _vars[key] = v
            ttk.Entry(tab_t, textvariable=v, width=28).grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=4)

        ttk.Label(tab_t, text='Title size:').grid(row=1, column=0, sticky=tk.W, pady=2)
        _title_size_v = tk.IntVar(value=st['title_size'])
        ttk.Spinbox(tab_t, from_=6, to=28, textvariable=_title_size_v, width=5
                    ).grid(row=1, column=1, sticky=tk.W, padx=4)
        _title_bold_v = tk.BooleanVar(value=st['title_bold'])
        ttk.Checkbutton(tab_t, text='Bold', variable=_title_bold_v
                        ).grid(row=1, column=2, sticky=tk.W)

        ttk.Label(tab_t, text='Axis label size:').grid(row=3, column=0, sticky=tk.W, pady=2)
        _label_size_v = tk.IntVar(value=st['label_size'])
        ttk.Spinbox(tab_t, from_=6, to=20, textvariable=_label_size_v, width=5
                    ).grid(row=3, column=1, sticky=tk.W, padx=4)

        ttk.Label(tab_t, text='Tick label size:').grid(row=5, column=0, sticky=tk.W, pady=2)
        _tick_size_v = tk.IntVar(value=st['tick_size'])
        ttk.Spinbox(tab_t, from_=4, to=16, textvariable=_tick_size_v, width=5
                    ).grid(row=5, column=1, sticky=tk.W, padx=4)

        ttk.Label(tab_t, text='X label rotation:').grid(row=6, column=0, sticky=tk.W, pady=2)
        _rot_v = tk.IntVar(value=st['xlabel_rot'])
        ttk.Spinbox(tab_t, from_=0, to=90, increment=15, textvariable=_rot_v, width=5
                    ).grid(row=6, column=1, sticky=tk.W, padx=4)

        ttk.Label(tab_t, text='(blank Y-axis label = auto unit from quantification)',
                  font=('Arial', 7), foreground='#666'
                  ).grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

        tab_t.columnconfigure(1, weight=1)

        # ── Tab 2: Group colors ───────────────────────────────────────────────
        tab_c = ttk.Frame(nb, padding=10)
        nb.add(tab_c, text='Group Colors')

        ttk.Label(tab_c, text='Click a swatch to change group color:',
                  font=('Arial', 8), foreground='#555').pack(anchor=tk.W, pady=(0, 6))

        si = getattr(self.app, 'samples_info', []) or []
        _all_groups = sorted(set(s.get('label', s.get('group', 'Unknown')) for s in si))
        import matplotlib.cm as _cm2
        _gcmap = _cm2.get_cmap('tab10', max(len(_all_groups), 1))
        def _grp_auto_hex(i):
            r, g, b, _ = _gcmap(i)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        _grp_color_vars = {}
        _grp_swatches   = {}
        for i, grp in enumerate(_all_groups):
            auto_hex = _grp_auto_hex(i)
            cur_hex  = st['group_colors'].get(grp, auto_hex)
            var = tk.StringVar(value=cur_hex)
            _grp_color_vars[grp] = var
            row_f = ttk.Frame(tab_c)
            row_f.pack(fill=tk.X, pady=2)
            sw = tk.Label(row_f, bg=cur_hex, width=3, relief='solid', cursor='hand2')
            sw.pack(side=tk.LEFT, padx=(2, 6))
            _grp_swatches[grp] = sw
            ttk.Label(row_f, text=grp, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
            def _pick(gn=grp, sv=var, s=sw):
                c = colorchooser.askcolor(color=sv.get(), parent=dlg, title=f'Color: {gn}')
                if c and c[1]: sv.set(c[1]); s.config(bg=c[1])
            sw.bind('<Button-1>', lambda e, fn=_pick: fn())
            ttk.Button(row_f, text='Reset', width=5,
                       command=lambda gn=grp, idx=i, sv=var, s=sw: (
                           sv.set(_grp_auto_hex(idx)), s.config(bg=_grp_auto_hex(idx))
                       )).pack(side=tk.RIGHT, padx=2)

        # ── Tab 3: Layout ─────────────────────────────────────────────────────
        tab_l = ttk.Frame(nb, padding=10)
        nb.add(tab_l, text='Layout')

        ttk.Label(tab_l, text='Line width:').grid(row=0, column=0, sticky=tk.W, pady=4)
        _lw_v = tk.DoubleVar(value=st['linewidth'])
        ttk.Spinbox(tab_l, from_=0.3, to=4.0, increment=0.2, format='%.1f',
                    textvariable=_lw_v, width=6).grid(row=0, column=1, sticky=tk.W, padx=4)

        ttk.Label(tab_l, text='Tight padding:').grid(row=1, column=0, sticky=tk.W, pady=4)
        _pad_v = tk.DoubleVar(value=st['tight_pad'])
        ttk.Spinbox(tab_l, from_=0.1, to=5.0, increment=0.1, format='%.1f',
                    textvariable=_pad_v, width=6).grid(row=1, column=1, sticky=tk.W, padx=4)
        ttk.Label(tab_l, text='(smaller = tighter)',
                  font=('Arial', 7), foreground='#666'
                  ).grid(row=2, column=0, columnspan=2, sticky=tk.W)

        tab_l.columnconfigure(1, weight=1)

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        def _apply():
            st['title']       = _vars['_tv'].get()
            st['xlabel']      = _vars['_xv'].get()
            st['ylabel']      = _vars['_yv'].get()
            st['title_size']  = int(_title_size_v.get())
            st['title_bold']  = _title_bold_v.get()
            st['label_size']  = int(_label_size_v.get())
            st['tick_size']   = int(_tick_size_v.get())
            st['xlabel_rot']  = int(_rot_v.get())
            st['linewidth']   = float(_lw_v.get())
            st['tight_pad']   = float(_pad_v.get())
            st['group_colors'] = {g: v.get() for g, v in _grp_color_vars.items()}

        ttk.Button(btn_row, text='OK',     command=lambda: (_apply(), dlg.destroy())
                   ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text='Apply',  command=_apply).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_row, text='Cancel', command=dlg.destroy).pack(side=tk.RIGHT)

    # ── Box / Violin config dialog ─────────────────────────────────────────────

    def _concentration_axis_label(self):
        """Y-axis label whose unit follows the quantification method: mM for
        IS/IS+PQN, percent for Total, arbitrary units otherwise (and when the
        matrix comes from raw Good-peak intensities rather than quantification)."""
        q = (getattr(self.app, 'stages', {}) or {}).get('quantification')
        has_quant = bool(getattr(q, '_quant_results', None)) if q else False
        unit = 'a.u.'
        if has_quant and hasattr(q, 'concentration_unit'):
            try:
                unit = q.concentration_unit() or 'a.u.'
            except Exception:
                unit = 'a.u.'
        return 'Relative area (%)' if unit == '%' else f'Concentration ({unit})'

    def _box_dims(self):
        """Panel grid as (rows, cols). Falls back to a legacy square 'grid' key
        so projects saved before the free rows×cols layout still load."""
        cfg = self._box_config
        if 'rows' in cfg and 'cols' in cfg:
            return max(int(cfg['rows']), 1), max(int(cfg['cols']), 1)
        g = max(int(cfg.get('grid', 3)), 1)
        return g, g

    def _update_box_cfg_status(self):
        """Refresh the one-line summary under the Options 'Configure…' button."""
        lbl = getattr(self, '_box_cfg_status', None)
        if lbl is None:
            return
        cfg = self._box_config
        rows, cols = self._box_dims()
        n_sel = len([m for m in cfg.get('selected', [])])
        kind = cfg.get('kind', 'box')
        cap = rows * cols
        pick = f'{n_sel} picked' if n_sel else 'auto'
        try:
            lbl.config(text=f'{kind.capitalize()} · {rows}×{cols} · {pick}/{cap}')
        except tk.TclError:
            pass

    def _open_box_config_dialog(self):
        """Non-modal window to pick the panel grid + which metabolites to plot.

        Left: searchable metabolite checklist. Right: plot type, grid size and
        per-type options. Apply renders onto the main canvas without closing, so
        the user can tweak and re-Apply while the main stage stays usable.
        """
        cfg = self._box_config

        _, _, feat_all, _ = _get_quantification_matrix(self.app)
        all_mets = [f for f in (feat_all or [])
                    if f not in self._excluded_metabolites]
        if not all_mets:
            messagebox.showinfo('No metabolites',
                                'No metabolite data found.', parent=self.frame)
            return

        dlg = tk.Toplevel(self.frame)
        dlg.title('Configure Box / Violin')
        dlg.resizable(True, True)          # non-modal: no grab_set (stages unlocked)
        dlg.update_idletasks()
        px = self.frame.winfo_rootx() + self.frame.winfo_width() // 2
        py = self.frame.winfo_rooty() + self.frame.winfo_height() // 2
        w, h = 640, 560
        dlg.geometry(f'{w}x{h}+{px - w//2}+{py - h//2}')

        # Button bar reserved at the bottom first so it can never be pushed off
        # the window by the (taller) options column.
        btn_row = ttk.Frame(dlg)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

        body = ttk.Frame(dlg, padding=8)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── Left: metabolite checklist ────────────────────────────────────────
        left = ttk.LabelFrame(body, text='Metabolites', padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        search_var = tk.StringVar()
        ttk.Entry(left, textvariable=search_var).pack(fill=tk.X, pady=(0, 4))

        lc_container = ttk.Frame(left)
        lc_container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(lc_container, highlightthickness=0)
        sb = ttk.Scrollbar(lc_container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas)
        cwin = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: (
            canvas.configure(scrollregion=canvas.bbox('all')),
            canvas.itemconfig(cwin, width=canvas.winfo_width())))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(cwin, width=e.width))
        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>',
            lambda ev: canvas.yview_scroll(-1 * (ev.delta // 120), 'units')))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

        check_vars = {m: tk.BooleanVar(value=(m in cfg.get('selected', [])))
                      for m in all_mets}

        def _populate(filter_text=''):
            for wch in inner.winfo_children():
                wch.destroy()
            ft = filter_text.lower()
            for m in all_mets:
                if ft and ft not in m.lower():
                    continue
                ttk.Checkbutton(inner, text=m, variable=check_vars[m],
                                command=_refresh_counter
                                ).pack(anchor=tk.W, padx=2)
        search_var.trace_add('write', lambda *a: _populate(search_var.get()))

        # ── Right: layout / type controls ─────────────────────────────────────
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.Y)

        type_f = ttk.LabelFrame(right, text='Plot type', padding=6)
        type_f.pack(fill=tk.X, pady=(0, 6))
        kind_var = tk.StringVar(value=cfg.get('kind', 'box'))
        ttk.Radiobutton(type_f, text='Boxplot', value='box',
                        variable=kind_var, command=lambda: _toggle_type()
                        ).pack(anchor=tk.W)
        ttk.Radiobutton(type_f, text='Violin', value='violin',
                        variable=kind_var, command=lambda: _toggle_type()
                        ).pack(anchor=tk.W)

        _rows0, _cols0 = self._box_dims()
        grid_f = ttk.LabelFrame(right, text='Grid layout', padding=6)
        grid_f.pack(fill=tk.X, pady=(0, 6))
        _rc = ttk.Frame(grid_f)
        _rc.pack(anchor=tk.W)
        ttk.Label(_rc, text='Rows:', font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
        rows_var = tk.IntVar(value=_rows0)
        ttk.Spinbox(_rc, from_=1, to=8, increment=1, width=4, textvariable=rows_var
                    ).grid(row=0, column=1, padx=(4, 10))
        ttk.Label(_rc, text='Cols:', font=('Arial', 8)).grid(row=0, column=2, sticky=tk.W)
        cols_var = tk.IntVar(value=_cols0)
        ttk.Spinbox(_rc, from_=1, to=8, increment=1, width=4, textvariable=cols_var
                    ).grid(row=0, column=3, padx=4)
        rows_var.trace_add('write', lambda *a: _refresh_counter())
        cols_var.trace_add('write', lambda *a: _refresh_counter())

        ttk.Label(grid_f, text='Panel size (cm):',
                  font=('Arial', 8)).pack(anchor=tk.W, pady=(6, 0))
        _sz = ttk.Frame(grid_f)
        _sz.pack(anchor=tk.W)
        ttk.Label(_sz, text='W:', font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
        panelw_var = tk.DoubleVar(value=float(cfg.get('panel_w', 6.5)))
        ttk.Spinbox(_sz, from_=2.5, to=20.0, increment=0.5, format='%.1f', width=5,
                    textvariable=panelw_var).grid(row=0, column=1, padx=(4, 10))
        ttk.Label(_sz, text='H:', font=('Arial', 8)).grid(row=0, column=2, sticky=tk.W)
        panelh_var = tk.DoubleVar(value=float(cfg.get('panel_h', 6.0)))
        ttk.Spinbox(_sz, from_=2.5, to=20.0, increment=0.5, format='%.1f', width=5,
                    textvariable=panelh_var).grid(row=0, column=3, padx=4)
        ttk.Label(grid_f, text='(figure is capped to one A4 page)',
                  font=('Arial', 7), foreground='#666').pack(anchor=tk.W)

        opt_f = ttk.LabelFrame(right, text='Options', padding=6)
        opt_f.pack(fill=tk.X, pady=(0, 6))
        points_var = tk.BooleanVar(value=cfg.get('points', True))
        _psize_row = ttk.Frame(opt_f)
        ttk.Checkbutton(opt_f, text='Overlay data points', variable=points_var,
                        command=lambda: _toggle_points()).pack(anchor=tk.W)
        _psize_row.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))
        ttk.Label(_psize_row, text='Point size:',
                  font=('Arial', 8)).pack(side=tk.LEFT)
        psize_var = tk.IntVar(value=int(cfg.get('point_size', 24)))
        _psize_sb = ttk.Spinbox(_psize_row, from_=4, to=80, increment=2,
                                textvariable=psize_var, width=5)
        _psize_sb.pack(side=tk.LEFT, padx=4)
        mean_var = tk.BooleanVar(value=cfg.get('mean', False))
        _mean_cb = ttk.Checkbutton(opt_f, text='Show means', variable=mean_var)
        _mean_cb.pack(anchor=tk.W)
        _inner_row = ttk.Frame(opt_f)
        _inner_row.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
        ttk.Label(_inner_row, text='Violin inner:',
                  font=('Arial', 8)).pack(side=tk.LEFT)
        inner_var = tk.StringVar(value=cfg.get('inner', 'box'))
        _inner_cb = ttk.Combobox(_inner_row, textvariable=inner_var, width=9,
                                 state='readonly',
                                 values=['box', 'quartile', 'point', 'stick', 'none'])
        _inner_cb.pack(side=tk.LEFT, padx=4)

        # ── Significance testing ──────────────────────────────────────────────
        sig_f = ttk.LabelFrame(right, text='Significance', padding=6)
        sig_f.pack(fill=tk.X, pady=(0, 6))
        _test_row = ttk.Frame(sig_f)
        _test_row.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(_test_row, text='Test:', font=('Arial', 8)).pack(side=tk.LEFT)
        _test_labels = {'None': 'none', 't-test (Welch)': 't',
                        'Mann-Whitney': 'mann'}
        _test_rev = {v: k for k, v in _test_labels.items()}
        test_disp_var = tk.StringVar(
            value=_test_rev.get(cfg.get('test', 'none'), 'None'))
        ttk.Combobox(_test_row, textvariable=test_disp_var, width=14,
                     state='readonly', values=list(_test_labels)
                     ).pack(side=tk.LEFT, padx=4)
        sig_line_var = tk.BooleanVar(value=cfg.get('sig_line', False))
        ttk.Checkbutton(sig_f, text='Connect groups with significance bars',
                        variable=sig_line_var).pack(anchor=tk.W, pady=(2, 0))

        counter_lbl = ttk.Label(right, text='', font=('Arial', 8), foreground='#333')
        counter_lbl.pack(anchor=tk.W, pady=(2, 4))

        def _toggle_points():
            _psize_sb.state(['!disabled'] if points_var.get() else ['disabled'])

        def _rc_dims():
            try:
                r = max(int(rows_var.get()), 1)
            except (tk.TclError, ValueError):
                r = 1
            try:
                c = max(int(cols_var.get()), 1)
            except (tk.TclError, ValueError):
                c = 1
            return r, c

        def _n_checked():
            return sum(1 for v in check_vars.values() if v.get())

        def _refresh_counter():
            r, c = _rc_dims()
            cap = r * c
            n = _n_checked()
            over = '  (over capacity — extras ignored)' if n > cap else ''
            counter_lbl.config(text=f'Selected: {n} / {cap}{over}')

        def _toggle_type():
            is_box = kind_var.get() == 'box'
            _mean_cb.state(['!disabled'] if is_box else ['disabled'])
            _inner_cb.state(['disabled'] if is_box else ['!disabled'])

        # Select all / none (respect current search filter)
        sel_row = ttk.Frame(left)
        sel_row.pack(fill=tk.X, pady=(4, 0))
        def _visible_mets():
            ft = search_var.get().lower()
            return [m for m in all_mets if not ft or ft in m.lower()]
        def _sel_all():
            for m in _visible_mets():
                check_vars[m].set(True)
            _refresh_counter()
        def _sel_none():
            for m in _visible_mets():
                check_vars[m].set(False)
            _refresh_counter()
        ttk.Button(sel_row, text='All',  command=_sel_all ).pack(side=tk.LEFT, padx=2)
        ttk.Button(sel_row, text='None', command=_sel_none).pack(side=tk.LEFT, padx=2)

        # ── Bottom buttons (btn_row created up front, populated here) ──────────
        def _apply():
            rows, cols = _rc_dims()
            selected = [m for m in all_mets if check_vars[m].get()]
            if not selected:
                messagebox.showinfo(
                    'No metabolites',
                    'No metabolites selected — auto-picking by significance '
                    '(when a test is active) or abundance.', parent=dlg)
            cap = rows * cols
            if len(selected) > cap:
                messagebox.showwarning(
                    'Too many selected',
                    f'{len(selected)} metabolites selected but the {rows}×{cols} '
                    f'grid holds {cap}. Only the first {cap} will be plotted.',
                    parent=dlg)
            cfg['kind']       = kind_var.get()
            cfg['rows']       = rows
            cfg['cols']       = cols
            cfg.pop('grid', None)          # migrate off the legacy square key
            cfg['panel_w']    = round(float(panelw_var.get()), 2)
            cfg['panel_h']    = round(float(panelh_var.get()), 2)
            cfg['selected']   = selected
            cfg['points']     = points_var.get()
            cfg['point_size'] = int(psize_var.get())
            cfg['mean']       = mean_var.get()
            cfg['inner']      = inner_var.get()
            cfg['test']       = _test_labels.get(test_disp_var.get(), 'none')
            cfg['sig_line']   = sig_line_var.get()
            self._update_box_cfg_status()
            if hasattr(self, '_method_var'):
                self._method_var.set('Box / Violin')
            self._run()

        ttk.Button(btn_row, text='Edit Style…',
                   command=self._open_box_style_dialog).pack(side=tk.LEFT)
        ttk.Button(btn_row, text='Close', command=dlg.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text='Apply', command=_apply).pack(side=tk.RIGHT, padx=2)

        _populate()
        _toggle_type()
        _toggle_points()
        _refresh_counter()

    # ── Box / Violin plot (grid of per-metabolite panels) ──────────────────────

    def _plot_box_violin_grid(self):
        import matplotlib.cm as cm
        from matplotlib.colors import to_rgba as _to_rgba

        cfg       = self._box_config
        kind      = cfg.get('kind', 'box')
        rows, cols = self._box_dims()
        panel_w   = float(cfg.get('panel_w', 6.5)) / _CM_PER_IN   # cm -> inches
        panel_h   = float(cfg.get('panel_h', 6.0)) / _CM_PER_IN
        selected  = list(cfg.get('selected', []))
        show_pts  = bool(cfg.get('points', True))
        pt_size   = int(cfg.get('point_size', 24))
        show_mean = bool(cfg.get('mean', False))
        vio_inner = cfg.get('inner', 'box')
        vio_inner = None if vio_inner == 'none' else vio_inner
        test      = cfg.get('test', 'none')
        sig_line  = bool(cfg.get('sig_line', False))
        st        = self._box_style

        # Fetch raw concentrations and apply the active sample/metabolite filters
        X_raw, _, feat_raw, gl_raw = _get_quantification_matrix(self.app)
        if X_raw is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No quantification data available.',
                    transform=ax.transAxes, ha='center', va='center', color='#a00')
            ax.set_axis_off()
            return
        # Drop all-NaN columns only. Do NOT mean-impute (as _remove_nan does):
        # imputing a not-detected value with the column mean fabricates data and
        # pulls both groups toward a shared mean, which flattens every group test
        # to non-significant. Missing entries stay NaN and are excluded per group.
        X_raw = np.asarray(X_raw, dtype=float)
        _valid = ~np.all(np.isnan(X_raw), axis=0)
        X_raw    = X_raw[:, _valid]
        feat_raw = [f for f, v in zip(feat_raw, _valid) if v]
        if self._excluded_samples:
            _si = getattr(self.app, 'samples_info', []) or []
            _sn = [s.get('name', f'S{i}') for i, s in enumerate(_si)]
            _kr = [i for i, n in enumerate(_sn) if n not in self._excluded_samples]
            X_raw  = X_raw[_kr, :]
            gl_raw = [gl_raw[i] for i in _kr]
        if self._excluded_metabolites:
            _kc = [i for i, f in enumerate(feat_raw)
                   if f not in self._excluded_metabolites]
            X_raw    = X_raw[:, _kc]
            feat_raw = [feat_raw[i] for i in _kc]
        y_raw, unique_groups = _group_to_int(gl_raw)
        n_groups = len(unique_groups)

        # Auto-pick order when no metabolite is chosen: by significance when a
        # test is active (most-separated groups first), else by abundance.
        fallback_order = None
        if test != 'none' and n_groups >= 2:
            from processing.group_stats import metabolite_min_pvalues
            pvals = metabolite_min_pvalues(X_raw, y_raw, test=test)
            # finite p-values first (ascending), NaN (untestable) last
            fallback_order = list(np.argsort(
                np.where(np.isnan(pvals), np.inf, pvals)))
        else:
            abundance = np.nanmean(X_raw, axis=0)
            fallback_order = list(np.argsort(
                np.where(np.isnan(abundance), -np.inf, abundance))[::-1])

        capacity = rows * cols
        sel_idx  = _resolve_box_selection(selected, feat_raw, capacity,
                                          x_raw=X_raw, fallback_order=fallback_order)
        feat_sel = [feat_raw[i] for i in sel_idx]
        n_mets   = len(feat_sel)
        if n_mets == 0:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No metabolites to plot.',
                    transform=ax.transAxes, ha='center', va='center', color='#a00')
            ax.set_axis_off()
            return

        # Group colors (user overrides win, else tab10)
        gcmap = cm.get_cmap('tab10', max(n_groups, 1))
        def _auto_grp_hex(i):
            r, g, b, _ = gcmap(i)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        grp_colors = [st['group_colors'].get(g, _auto_grp_hex(i))
                      for i, g in enumerate(unique_groups)]
        lw = st.get('linewidth', 1.0)

        # Figure size = panel size × grid, capped to one A4 page. Stored for export.
        self._box_export_figsize = _fit_page(cols * panel_w, rows * panel_h)
        self._fig.clear()
        axes = self._fig.subplots(rows, cols, squeeze=False)
        flat = [ax for row in axes for ax in row]

        ylabel = st.get('ylabel') or self._concentration_axis_label()

        for k, met_i in enumerate(sel_idx):
            ax = flat[k]
            col_all = X_raw[:, met_i]
            vals_by_group = []
            for gi in range(n_groups):
                mask  = (np.asarray(y_raw) == gi)
                color = grp_colors[gi]
                vals  = col_all[mask]
                vals  = vals[np.isfinite(vals)]
                vals_by_group.append(vals)
                pos   = gi

                if kind == 'box' and len(vals) >= 1:
                    # Fill = group color at 60%; edge, whiskers/caps, median all
                    # solid black. Error bars raised above the box (zorder).
                    face = _to_rgba(color, 0.60)
                    bp = ax.boxplot(
                        [vals], positions=[pos], widths=0.6, patch_artist=True,
                        showmeans=show_mean,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=4),
                        medianprops=dict(color='black', linewidth=1.4),
                        boxprops=dict(facecolor=face, edgecolor='black', linewidth=lw),
                        whiskerprops=dict(color='black', linewidth=lw),
                        capprops=dict(color='black', linewidth=lw),
                        flierprops=dict(marker='o', markerfacecolor=color,
                                        markeredgecolor='black', markersize=3, alpha=0.6),
                        manage_ticks=False)
                    for _b in bp['boxes']:
                        _b.set_zorder(2)
                    for _key in ('whiskers', 'caps', 'medians', 'means'):
                        for _art in bp.get(_key, []):
                            _art.set_zorder(3)
                elif kind == 'violin' and len(vals) >= 2:
                    # violinplot needs ≥2 points; groups with fewer are shown
                    # only as scatter below — never padded with fake values.
                    vp = ax.violinplot([vals], positions=[pos], widths=0.6,
                                       showmedians=True, showextrema=True)
                    for pc in vp['bodies']:
                        pc.set_facecolor(_to_rgba(color, 0.60))
                        pc.set_edgecolor('black'); pc.set_linewidth(lw)
                    for part in ('cmedians', 'cmaxes', 'cmins', 'cbars'):
                        if part in vp:
                            vp[part].set_color('black'); vp[part].set_linewidth(lw)

                if show_pts and len(vals):
                    # Marker fill = group color at full opacity, black edge to
                    # separate overlapping points; box/violin fill stays at 80%.
                    jit = np.random.default_rng(seed=met_i * 97 + gi
                          ).uniform(-0.12, 0.12, len(vals))
                    ax.scatter(pos + jit, vals, color=color, s=pt_size, alpha=1.0,
                               edgecolors='black', linewidths=0.4, zorder=3)

            # Significance brackets between groups (explicit test, never faked)
            if sig_line and test != 'none' and n_groups >= 2:
                self._draw_significance(ax, vals_by_group, test)

            ax.set_title(feat_sel[k], fontsize=st.get('tick_size', 7) + 1)
            ax.set_ylabel(ylabel, fontsize=max(st.get('label_size', 9) - 1, 6))
            ax.set_xticks(range(n_groups))
            ax.set_xticklabels(unique_groups,
                               rotation=st.get('xlabel_rot', 45),
                               ha='right' if st.get('xlabel_rot', 45) > 0 else 'center',
                               fontsize=st.get('tick_size', 7))
            ax.tick_params(axis='y', labelsize=st.get('tick_size', 7))
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
            # Only left (Y) and bottom (X) spines; drop the top/right frame lines.
            for _sp in ('top', 'right'):
                ax.spines[_sp].set_visible(False)
            for _sp in ('left', 'bottom'):
                ax.spines[_sp].set_color('black')
                ax.spines[_sp].set_linewidth(0.8)

        # Turn off unused cells
        for k in range(n_mets, len(flat)):
            flat[k].set_axis_off()

        # Reserve a bottom strip for the shared legend (constrained_layout does
        # not account for figure legends) and a top strip for the title.
        try:
            self._fig.get_layout_engine().set(rect=(0.0, 0.07, 1.0, 0.94))
        except Exception:
            pass

        # Shared group legend at the bottom of the figure
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=_to_rgba(grp_colors[i], 0.60),
                         edgecolor='black', label=g)
                   for i, g in enumerate(unique_groups)]
        self._fig.legend(handles=handles, loc='lower center',
                         ncol=min(n_groups, 6), fontsize=8, framealpha=0.85,
                         bbox_to_anchor=(0.5, 0.005))

        title_w = 'bold' if st.get('title_bold', True) else 'normal'
        self._fig.suptitle(st.get('title', 'Distribution by Group'),
                           fontsize=st.get('title_size', 11), fontweight=title_w)

    def _draw_significance(self, ax, vals_by_group, test):
        """Draw stacked significance brackets between group pairs on `ax`.

        Uses processing.group_stats (Welch t or Mann-Whitney). Pairs whose test
        is undefined (NaN p) are skipped rather than annotated with a fake bar.
        """
        from processing.group_stats import pairwise_significance
        pairs = [pw for pw in pairwise_significance(vals_by_group, test=test)
                 if np.isfinite(pw[2])]
        if not pairs:
            return
        finite = np.concatenate([v for v in vals_by_group if len(v)]) \
            if any(len(v) for v in vals_by_group) else np.array([0.0, 1.0])
        data_max = float(np.max(finite))
        data_min = float(np.min(finite))
        span = (data_max - data_min) or (abs(data_max) or 1.0)
        top = data_max
        for level, (i, j, p, stars) in enumerate(pairs):
            y0 = data_max + span * 0.10 * (level + 1)
            h  = span * 0.02
            ax.plot([i, i, j, j], [y0, y0 + h, y0 + h, y0],
                    color='black', linewidth=0.8, clip_on=False)
            ax.text((i + j) / 2.0, y0 + h, stars,
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold' if stars != 'ns' else 'normal')
            top = max(top, y0 + h)
        ax.set_ylim(top=top + span * 0.08)

    def _export_csv_result(self):
        """Export the underlying values of the currently displayed plot as CSV
        so they can be replotted in another tool (e.g. Volcano: log2FC, p, padj)."""
        exp    = getattr(self, '_plot_export', None)
        # Prefer the method the export itself records (in pathway mode _method_var
        # still holds the Statistics method, which would mislabel the file).
        method = (exp.get('method') if exp else None) or (
            self._method_var.get() if hasattr(self, '_method_var') else '')
        if not exp or not exp.get('rows'):
            messagebox.showinfo(
                'CSV result',
                f'No exportable values for "{method}".\n'
                'Render the plot first. CSV export is available for the plots '
                'that produce tabular values (e.g. Volcano, VIP, Correlation, PCA, '
                'Pathway, Dumbbell).',
                parent=self.frame)
            return
        fn = filedialog.asksaveasfilename(
            title='Save CSV result',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')],
            initialfile=f'statistics_{method.replace(" ", "_")}_data.csv',
            parent=self.frame)
        if not fn:
            return
        try:
            import csv as _csv
            with open(fn, 'w', newline='', encoding='utf-8-sig') as f:
                w = _csv.writer(f)
                w.writerow(exp['headers'])
                w.writerows(exp['rows'])
            messagebox.showinfo(
                'CSV result',
                f'Saved {len(exp["rows"])} row(s) to:\n{fn}', parent=self.frame)
        except Exception as e:
            messagebox.showerror('CSV result',
                                 f'Failed to save CSV:\n{e}', parent=self.frame)

    def _export(self):
        fmt  = getattr(self, '_fmt_var', None)
        fmt  = fmt.get().lower() if fmt else 'png'
        dpi  = getattr(self, '_dpi_var', None)
        dpi  = dpi.get() if dpi else 150
        ext  = fmt if fmt != 'tiff' else 'tif'
        fn = filedialog.asksaveasfilename(
            title='Save figure',
            defaultextension=f'.{ext}',
            filetypes=[(fmt.upper(), f'*.{ext}'), ('All', '*.*')],
            initialfile=f'statistics_{self._method_var.get().replace(" ","_")}.{ext}',
            parent=self.frame)
        if not fn:
            return
        try:
            # For Metabolic Profile: temporarily resize to the full export size,
            # save, then restore the original canvas size so the window isn't affected.
            _orig_size = self._fig.get_size_inches()
            _method    = self._method_var.get()
            # Metabolic Profile (donut) is NOT resized on export: it must match what
            # the user sees on screen (the figure is already sized to the canvas in
            # _render_and_display). Forcing _pie_export_figsize made the figure
            # larger than the canvas, so the point-sized legend shrank relative to
            # the donut. Export at the on-screen size keeps legend/donut proportion.
            _is_box    = (_method == 'Box / Violin' and
                          getattr(self, '_box_export_figsize', None) is not None)
            _is_db     = (getattr(self, '_mode_var', None)
                          and self._mode_var.get() == 'pathway'
                          and getattr(self, '_pw_plot_var', None)
                          and self._pw_plot_var.get() == 'Dumbbell'
                          and getattr(self, '_dumbbell_export_figsize', None) is not None)
            if _is_box:
                # Grid panels use the figure's constrained layout engine; just
                # resize (tight_layout would clash with it).
                self._fig.set_size_inches(self._box_export_figsize)
            elif _is_db:
                self._fig.set_size_inches(self._dumbbell_export_figsize)
            self._fig.savefig(fn, dpi=dpi, format=fmt,
                              bbox_inches='tight', facecolor='white')
            if _is_box or _is_db:
                self._fig.set_size_inches(_orig_size)
                self._render_and_display()
            messagebox.showinfo('Saved', f'Figure saved to:\n{fn}', parent=self.frame)
        except Exception as e:
            messagebox.showerror('Export error', str(e), parent=self.frame)
