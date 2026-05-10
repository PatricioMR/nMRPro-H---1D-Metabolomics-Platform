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
            'ylabel':       'Concentration',
            'label_size':   9,
            'tick_size':    7,
            'xlabel_rot':   45,
            'linewidth':    1.0,
            'tight_pad':    1.2,
            'group_colors': {},   # {group_name: hex}
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

        # Left panel controls go directly into self.frame
        left = ttk.Frame(self.frame, padding=4)
        left.pack(fill=tk.BOTH, expand=True)
        self._build_left(left)

    def _build_left(self, parent):
        ttk.Label(parent, text='Method', font=('Arial', 9, 'bold')).pack(anchor=tk.W)

        self._method_var = tk.StringVar(value='PCA')
        methods = ['PCA', 'PLS-DA', 'OPLS-DA', 'Heatmap',
                   'VIP Score', 'Volcano', 'Metabolic Profile',
                   'Boxplot', 'Violin']
        for m in methods:
            ttk.Radiobutton(parent, text=m, variable=self._method_var,
                            value=m, command=self._on_method_change
                            ).pack(anchor=tk.W, pady=1)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Options frame — rebuilt per method
        self._opt_frame = ttk.LabelFrame(parent, text='Options', padding=4)
        self._opt_frame.pack(fill=tk.X)
        self._build_options('PCA')

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Scaling
        ttk.Label(parent, text='Scaling', font=('Arial', 8, 'bold')).pack(anchor=tk.W)
        self._scale_var = tk.StringVar(value='UV')
        for lbl, val in [('None', 'none'), ('UV (unit var.)', 'UV'),
                         ('Pareto', 'pareto'), ('Log  ln(1+x), base e', 'log')]:
            ttk.Radiobutton(parent, text=lbl, variable=self._scale_var,
                            value=val).pack(anchor=tk.W, pady=1)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Filter buttons
        ttk.Label(parent, text='Filters', font=('Arial', 8, 'bold')).pack(anchor=tk.W)
        self._filter_samples_btn = ttk.Button(
            parent, text='🧪 Samples', command=self._open_sample_selector)
        self._filter_samples_btn.pack(fill=tk.X, pady=2)
        self._filter_metabolites_btn = ttk.Button(
            parent, text='🔬 Metabolites', command=self._open_metabolite_selector)
        self._filter_metabolites_btn.pack(fill=tk.X, pady=2)
        ttk.Button(
            parent, text='📊 Normality check', command=self._open_normality_popup
        ).pack(fill=tk.X, pady=2)
        self._filter_status_lbl = ttk.Label(parent, text='', font=('Arial', 7), foreground='#666')
        self._filter_status_lbl.pack(anchor=tk.W)

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
            ttk.Checkbutton(self._opt_frame, text='Show CV R²/Q²',
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

        elif method == 'VIP Score':
            ttk.Label(self._opt_frame, text='Top N features:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._top_n_var = tk.IntVar(value=20)
            ttk.Spinbox(self._opt_frame, from_=5, to=100, increment=5,
                        textvariable=self._top_n_var, width=5
                        ).grid(row=0, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='VIP threshold:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
            self._vip_thr_var = tk.DoubleVar(value=1.0)
            ttk.Spinbox(self._opt_frame, from_=0.5, to=3.0, increment=0.1,
                        textvariable=self._vip_thr_var, width=5, format='%.1f'
                        ).grid(row=1, column=1, sticky=tk.W, padx=4)

        elif method == 'Volcano':
            ttk.Label(self._opt_frame, text='Group A:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._grp_a_var = tk.StringVar()
            self._cb_a = ttk.Combobox(self._opt_frame, textvariable=self._grp_a_var,
                                       width=12, state='readonly')
            self._cb_a.grid(row=0, column=1, sticky=tk.W, padx=4)
            ttk.Label(self._opt_frame, text='Group B:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
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

        elif method in ('Boxplot', 'Violin'):
            ttk.Label(self._opt_frame, text='Top N metabolites:',
                      font=('Arial', 8)).grid(row=0, column=0, sticky=tk.W)
            self._box_topn_var = tk.IntVar(value=15)
            ttk.Spinbox(self._opt_frame, from_=1, to=100, increment=1,
                        textvariable=self._box_topn_var, width=5
                        ).grid(row=0, column=1, sticky=tk.W, padx=4)

            ttk.Label(self._opt_frame, text='Sort by:',
                      font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
            self._box_sort_var = tk.StringVar(value='variance')
            ttk.Combobox(self._opt_frame, textvariable=self._box_sort_var,
                         values=['variance', 'mean', 'name'], width=10, state='readonly'
                         ).grid(row=1, column=1, sticky=tk.W, padx=4)

            self._box_points_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._opt_frame, text='Overlay data points',
                            variable=self._box_points_var
                            ).grid(row=2, column=0, columnspan=2, sticky=tk.W)

            if method == 'Boxplot':
                self._box_notch_var = tk.BooleanVar(value=False)
                ttk.Checkbutton(self._opt_frame, text='Notched boxes',
                                variable=self._box_notch_var
                                ).grid(row=3, column=0, columnspan=2, sticky=tk.W)
                self._box_mean_var = tk.BooleanVar(value=False)
                ttk.Checkbutton(self._opt_frame, text='Show means',
                                variable=self._box_mean_var
                                ).grid(row=4, column=0, columnspan=2, sticky=tk.W)
            else:  # Violin
                ttk.Label(self._opt_frame, text='Inner:',
                          font=('Arial', 8)).grid(row=3, column=0, sticky=tk.W)
                self._violin_inner_var = tk.StringVar(value='box')
                ttk.Combobox(self._opt_frame, textvariable=self._violin_inner_var,
                             values=['box', 'quartile', 'point', 'stick', 'none'],
                             width=10, state='readonly'
                             ).grid(row=3, column=1, sticky=tk.W, padx=4)

            ttk.Button(self._opt_frame, text='Edit Style…',
                       command=self._open_box_style_dialog
                       ).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))

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
        for cb in (getattr(self, '_cb_a', None), getattr(self, '_cb_b', None)):
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
        """Normality check per sample.

        For each sample, the distribution of its analyte values (one value per
        metabolite) is plotted as a histogram with a fitted normal curve and a
        Shapiro-Wilk p-value.  This tests whether the overall metabolite profile
        of each sample is normally distributed — useful for detecting outlier
        samples or skewed batches.
        """
        from scipy import stats as spstats
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        X, sample_names, feature_labels, group_labels = _get_quantification_matrix(self.app)
        if X is None or X.shape[0] == 0:
            messagebox.showinfo('No data', 'No sample data available.', parent=self.frame)
            return

        X, feature_labels = self._remove_nan(X, feature_labels)

        # Apply active filters
        if self._excluded_metabolites:
            keep = [i for i, f in enumerate(feature_labels) if f not in self._excluded_metabolites]
            X = X[:, keep]
            feature_labels = [feature_labels[i] for i in keep]

        # Build per-sample names and apply sample filter
        si = getattr(self.app, 'samples_info', []) or []
        all_names = [s.get('name', f'S{i}') for i, s in enumerate(si)]
        # Pad or trim to match X rows
        while len(all_names) < X.shape[0]:
            all_names.append(f'S{len(all_names)}')
        all_names = all_names[:X.shape[0]]

        if self._excluded_samples:
            keep = [i for i, n in enumerate(all_names) if n not in self._excluded_samples]
            if len(keep) >= 2:
                X         = X[keep, :]
                all_names = [all_names[i] for i in keep]

        n_samples = X.shape[0]
        if n_samples == 0:
            messagebox.showinfo('No data', 'No samples to display.', parent=self.frame)
            return

        # Grid layout: up to MAX_COLS columns
        MAX_COLS = 4
        ncols = min(MAX_COLS, n_samples)
        nrows = (n_samples + ncols - 1) // ncols
        fig_h = max(3.0, nrows * 2.4)
        fig_w = ncols * 2.8

        dlg = tk.Toplevel(self.frame)
        dlg.title('Normality Check — Sample Distributions')
        dlg.resizable(True, True)
        dlg.update_idletasks()
        px = self.frame.winfo_rootx() + self.frame.winfo_width() // 2
        py = self.frame.winfo_rooty() + self.frame.winfo_height() // 2
        dw = min(int(fig_w * 96), 1200)
        dh = min(int(fig_h * 96 + 80), 900)
        dlg.geometry(f'{dw}x{dh}+{px - dw//2}+{py - dh//2}')

        fig = Figure(figsize=(fig_w, fig_h), dpi=96, layout='constrained')

        for i, sname in enumerate(all_names):
            ax  = fig.add_subplot(nrows, ncols, i + 1)
            row = X[i, :]
            row = row[np.isfinite(row)]

            if len(row) < 3:
                ax.set_title(sname, fontsize=6.5)
                ax.text(0.5, 0.5, 'n<3', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color='#999')
                ax.set_axis_off()
                continue

            # Shapiro-Wilk (n ≤ 5000)
            try:
                _, p_sw    = spstats.shapiro(row[:5000])
                sw_txt     = f'W p={p_sw:.3f}'
                sw_color   = '#c62828' if p_sw < 0.05 else '#2e7d32'
            except Exception:
                sw_txt, sw_color = '', '#555'

            # Histogram
            ax.hist(row, bins='auto', density=True,
                    color='#1565c0', alpha=0.55, edgecolor='white', linewidth=0.4)

            # Fitted normal curve
            mu, sigma = row.mean(), row.std()
            if sigma > 0:
                x_range = np.linspace(row.min(), row.max(), 200)
                ax.plot(x_range, spstats.norm.pdf(x_range, mu, sigma),
                        color='#e65100', linewidth=1.4)

            short = sname if len(sname) <= 16 else sname[:14] + '…'
            ax.set_title(short, fontsize=6.5, pad=2)
            ax.tick_params(labelsize=5)
            ax.set_xlabel('Analyte value', fontsize=5.5)
            ax.set_ylabel('Density',       fontsize=5.5)
            if sw_txt:
                ax.text(0.97, 0.95, sw_txt, transform=ax.transAxes,
                        ha='right', va='top', fontsize=5.5,
                        color=sw_color, fontweight='bold')

        canvas = FigureCanvasTkAgg(fig, master=dlg)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, dlg, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw()

        note = ttk.Label(
            dlg,
            text=(f'{n_samples} samples  ·  {X.shape[1]} analytes  ·  '
                  'Red p-value = non-normal (Shapiro-Wilk p < 0.05)  ·  '
                  'Orange curve = fitted normal'),
            font=('Arial', 7), foreground='#555')
        note.pack(side=tk.BOTTOM, pady=(0, 4))

    def _build_right_panel(self):
        """Build the statistics plot area inside app.right_panel."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self._right_widget = ttk.Frame(self.app.right_panel)

        # Export bar at the bottom
        eb = ttk.Frame(self._right_widget)
        eb.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        ttk.Label(eb, text='DPI:', font=('Arial', 8)).pack(side=tk.LEFT, padx=(4, 2))
        self._dpi_var = tk.IntVar(value=150)
        ttk.Spinbox(eb, from_=72, to=600, increment=25,
                    textvariable=self._dpi_var, width=5
                    ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(eb, text='Format:', font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 2))
        self._fmt_var = tk.StringVar(value='PNG')
        ttk.Combobox(eb, textvariable=self._fmt_var,
                     values=['PNG', 'SVG', 'PDF', 'TIFF', 'EPS'],
                     width=6, state='readonly').pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(eb, text='💾 Save figure', command=self._export
                   ).pack(side=tk.LEFT, padx=4)

        # Plot canvas
        self._fig = Figure(figsize=(8, 5), dpi=100, layout='constrained')
        self._ax  = self._fig.add_subplot(111)
        self._ax.text(0.5, 0.5, 'Select a method and click Run',
                      transform=self._ax.transAxes,
                      ha='center', va='center', color='#888888', fontsize=12)
        self._ax.set_axis_off()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._right_widget)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._toolbar = NavigationToolbar2Tk(
            self._canvas, self._right_widget, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)

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
        if self._right_widget is None:
            self._build_right_panel()
        self._right_widget.pack(fill=tk.BOTH, expand=True)

    # ── Run ──────────────────────────────────────────────────────────────────

    def _run(self):
        method = self._method_var.get()
        X, sample_names, feature_labels, group_labels = _get_quantification_matrix(self.app)
        _single_sample_ok = method in ('Metabolic Profile', 'Boxplot', 'Violin')
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

        self._fig.clear()
        try:
            if method == 'PCA':
                self._plot_pca(X_scaled, sample_names, feature_labels, group_labels)
            elif method == 'PLS-DA':
                self._plot_plsda(X_scaled, sample_names, feature_labels, group_labels, opls=False)
            elif method == 'OPLS-DA':
                self._plot_plsda(X_scaled, sample_names, feature_labels, group_labels, opls=True)
            elif method == 'Heatmap':
                self._plot_heatmap(X_scaled, sample_names, feature_labels, group_labels)
            elif method == 'VIP Score':
                self._plot_vip(X_scaled, sample_names, feature_labels, group_labels)
            elif method == 'Volcano':
                self._plot_volcano(X, sample_names, feature_labels, group_labels)
            elif method == 'Metabolic Profile':
                self._plot_metabolic_profile(X, sample_names, feature_labels, group_labels)
            elif method == 'Boxplot':
                self._plot_box_violin(X, sample_names, feature_labels, group_labels, kind='box')
            elif method == 'Violin':
                self._plot_box_violin(X, sample_names, feature_labels, group_labels, kind='violin')
        except Exception as e:
            import traceback; traceback.print_exc()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error:\n{e}',
                    transform=ax.transAxes, ha='center', va='center',
                    color='red', fontsize=9, wrap=True)
            ax.set_axis_off()

        self._canvas.draw_idle()

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
        # Replace zeros/negatives with small positive for log
        if mode == 'log':
            Xs = np.log1p(np.clip(Xs, 0, None))
            mode = 'UV'   # then UV scale the log-transformed data
        # Mean-centre always
        Xs -= Xs.mean(axis=0)
        if mode == 'UV':
            std = Xs.std(axis=0)
            std[std == 0] = 1.0
            Xs /= std
        elif mode == 'pareto':
            std = Xs.std(axis=0)
            std[std == 0] = 1.0
            Xs /= np.sqrt(std)
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
        from sklearn.model_selection import cross_val_score

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

        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X, Y)
        scores = pls.transform(X)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        total_var_X = np.sum(np.var(X, axis=0))  # sum of per-feature variances
        ev_x = [np.var(scores[:, k]) / total_var_X * 100 for k in range(min(2, scores.shape[1]))]
        gcols   = self._group_colors(unique)
        ellipse = getattr(self, '_ellipse_var', None)
        ellipse = ellipse.get() if ellipse else True

        if opls:
            # OPLS-DA approximation: orthogonal correction
            # Regress out the variance orthogonal to Y from T[:,0]
            t1 = scores[:, 0]
            t_orth = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(t1)
            proj = np.dot(t_orth, t1) / (np.dot(t1, t1) + 1e-12)
            t_orth_corr = t_orth - proj * t1
            plot_x, plot_y = t1, t_orth_corr
            ev_pred  = np.var(t1)          / total_var_X * 100
            ev_orth  = np.var(t_orth_corr) / total_var_X * 100
            xlabel = f'T score (predictive, {ev_pred:.1f}% X var.)'
            ylabel = f'T orthogonal ({ev_orth:.1f}% X var.)'
            title = 'OPLS-DA Score Plot'
        else:
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

        cv = getattr(self, '_cv_var', None)
        if cv and cv.get():
            try:
                from sklearn.model_selection import StratifiedKFold
                cv_scores = cross_val_score(
                    PLSRegression(n_components=n_comp, scale=False),
                    X, y_int.astype(float),
                    cv=StratifiedKFold(n_splits=min(5, X.shape[0])),
                    scoring='r2')
                r2cv = cv_scores.mean()
                ax.set_title(f'{title}  —  CV R²={r2cv:.3f}', fontsize=10)
            except Exception:
                ax.set_title(title, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)

        # Permutation test
        n_perm = getattr(self, '_n_perm_var', None)
        n_perm = n_perm.get() if n_perm else 0
        if n_perm > 0:
            try:
                from sklearn.model_selection import permutation_test_score
                _, _, pval = permutation_test_score(
                    PLSRegression(n_components=n_comp, scale=False),
                    X, y_int.astype(float),
                    n_permutations=n_perm, scoring='r2',
                    cv=min(5, X.shape[0]), n_jobs=1)
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

        clust = getattr(self, '_clust_var', None)
        clust = clust.get() if clust else 'ward'
        cmap  = getattr(self, '_cmap_var',  None)
        cmap  = cmap.get()  if cmap  else 'RdYlBu_r'

        # Limit features for legibility
        MAX_FEAT = 60
        if X.shape[1] > MAX_FEAT:
            var_order = np.argsort(X.var(axis=0))[::-1][:MAX_FEAT]
            X_h  = X[:, var_order]
            flbl = [feature_labels[i] for i in var_order]
        else:
            X_h  = X
            flbl = feature_labels

        # Cluster rows (samples) and cols (features)
        row_link = linkage(X_h,   method=clust, metric='euclidean')
        col_link = linkage(X_h.T, method=clust, metric='euclidean')
        row_ord  = dendrogram(row_link, no_plot=True)['leaves']
        col_ord  = dendrogram(col_link, no_plot=True)['leaves']
        X_ord    = X_h[np.ix_(row_ord, col_ord)]
        slbl_ord = [sample_names[i]  for i in row_ord]
        flbl_ord = [flbl[i]          for i in col_ord]
        glbl_ord = [group_labels[i]  for i in row_ord]

        gs   = self._fig.add_gridspec(1, 2, width_ratios=[1, 12], wspace=0.02)
        ax_g = self._fig.add_subplot(gs[0])
        ax_h = self._fig.add_subplot(gs[1])

        y_int, unique = _group_to_int(glbl_ord)
        gcols = self._group_colors(unique)
        import matplotlib.colors as mcolors
        group_arr = np.array([[mcolors.to_rgba(gcols[g])[:3] for g in glbl_ord]])
        ax_g.imshow(np.array([[mcolors.to_rgba(gcols[g])[:3]] for g in glbl_ord]),
                    aspect='auto', interpolation='nearest')
        ax_g.set_xticks([]); ax_g.set_yticks(range(len(slbl_ord)))
        ax_g.set_yticklabels(slbl_ord, fontsize=7)

        im = ax_h.imshow(X_ord, aspect='auto', cmap=cmap,
                         interpolation='nearest',
                         vmin=np.percentile(X_ord, 2),
                         vmax=np.percentile(X_ord, 98))
        ax_h.set_xticks(range(len(flbl_ord)))
        ax_h.set_xticklabels(flbl_ord, rotation=90, fontsize=6)
        ax_h.set_yticks([])
        self._fig.colorbar(im, ax=ax_h, fraction=0.02, pad=0.01,
                           label='Scaled intensity')
        self._fig.suptitle('Heatmap (hierarchical clustering)', fontsize=10)

        patches = [mpatches.Patch(color=gcols[g], label=g) for g in unique]
        ax_h.legend(handles=patches, fontsize=7, loc='upper right',
                    bbox_to_anchor=(1.18, 1), framealpha=0.85)

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
        vip_thr = vip_thr.get() if vip_thr else 1.0

        vips = self._vip_scores(X, Y, n_comp)
        order = np.argsort(vips)[::-1][:top_n]
        vips_top  = vips[order]
        feats_top = [feature_labels[i] for i in order]

        ax = self._fig.add_subplot(111)
        colors = ['#c62828' if v >= vip_thr else '#1565c0' for v in vips_top]
        ax.barh(range(len(feats_top)), vips_top[::-1], color=colors[::-1],
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(feats_top)))
        ax.set_yticklabels(feats_top[::-1], fontsize=7)
        ax.axvline(vip_thr, color='#c62828', lw=1.0, ls='--',
                   label=f'VIP = {vip_thr:.1f}')
        ax.set_xlabel('VIP Score', fontsize=9)
        ax.set_title(f'VIP Scores (PLS-DA, top {top_n})', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_facecolor('#f9f9f9')

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

        pvals = np.array([stats.ttest_ind(Xa[:, j], Xb[:, j],
                                          equal_var=False).pvalue
                          for j in range(X_raw.shape[1])])
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

        ax = self._fig.add_subplot(111)
        sig_up   = (log2fc > np.log2(fc_thr))  & (pvals_plot < p_thr)
        sig_down = (log2fc < -np.log2(fc_thr)) & (pvals_plot < p_thr)
        ns       = ~(sig_up | sig_down)

        ax.scatter(log2fc[ns],       neg_log_p[ns],       color='#888888', s=20,
                   alpha=0.5, label='NS', edgecolors='none')
        ax.scatter(log2fc[sig_up],   neg_log_p[sig_up],   color='#c62828', s=28,
                   alpha=0.85, label=f'Up in {gb}', edgecolors='white', lw=0.4)
        ax.scatter(log2fc[sig_down], neg_log_p[sig_down], color='#1565c0', s=28,
                   alpha=0.85, label=f'Up in {ga}', edgecolors='white', lw=0.4)

        # Label top significant
        top_idx = np.argsort(neg_log_p)[::-1][:10]
        for i in top_idx:
            if pvals_plot[i] < p_thr:
                ax.annotate(feature_labels[i], (log2fc[i], neg_log_p[i]),
                            fontsize=5.5, xytext=(3, 2),
                            textcoords='offset points', color='#333')

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
        ncols = min(n_groups, 4)
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
                          fontsize=6.5, frameon=False)

        _st = self._pie_style
        self._fig.suptitle(
            _st.get('title', 'Metabolic Profile by Group'),
            fontsize=_st.get('title_size', 11),
            fontweight='bold' if _st.get('title_bold', True) else 'normal')
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

    # ── Box / Violin plot ─────────────────────────────────────────────────────

    def _plot_box_violin(self, X, sample_names, feature_labels, group_labels, kind='box'):
        import matplotlib.cm as cm

        top_n      = int(getattr(self, '_box_topn_var',      tk.IntVar(value=15)).get())
        sort_by    = getattr(self, '_box_sort_var',           None)
        sort_by    = sort_by.get() if sort_by else 'variance'
        show_pts   = getattr(self, '_box_points_var',         None)
        show_pts   = show_pts.get() if show_pts else True

        notch      = getattr(self, '_box_notch_var',          None)
        notch      = notch.get() if notch else False
        show_mean  = getattr(self, '_box_mean_var',           None)
        show_mean  = show_mean.get() if show_mean else False
        vio_inner  = getattr(self, '_violin_inner_var',       None)
        vio_inner  = vio_inner.get() if vio_inner else 'box'
        if vio_inner == 'none':
            vio_inner = None

        st = self._box_style
        y, unique_groups = _group_to_int(group_labels)
        n_groups = len(unique_groups)

        # Re-fetch raw concentrations
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

        # Select top N metabolites by chosen criterion
        if sort_by == 'variance':
            scores = np.nanvar(X_raw, axis=0)
        elif sort_by == 'mean':
            scores = np.nanmean(X_raw, axis=0)
        else:
            scores = np.array([0.0] * X_raw.shape[1])   # name sort below

        if sort_by == 'name':
            order = np.array(sorted(range(len(feat_raw)), key=lambda i: feat_raw[i]))
        else:
            order = np.argsort(scores)[::-1]

        order    = order[:top_n]
        feat_sel = [feat_raw[i] for i in order]
        X_sel    = X_raw[:, order]
        n_mets   = len(feat_sel)

        # Group colors
        gcmap = cm.get_cmap('tab10', max(n_groups, 1))
        def _auto_grp_hex(i):
            r, g, b, _ = gcmap(i)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        grp_colors = [
            st['group_colors'].get(g, _auto_grp_hex(i))
            for i, g in enumerate(unique_groups)
        ]

        # Figure size: scale with number of metabolites, stored for export
        fig_w = max(1.1 * n_mets * n_groups + 2, 8)
        fig_h = 5.5
        self._box_export_figsize = (fig_w, fig_h)
        self._fig.clear()

        ax = self._fig.add_subplot(111)

        # Positions: each metabolite group shares a cluster of n_groups boxes
        spacing   = n_groups + 1.5
        positions = np.arange(n_mets) * spacing
        width     = 0.8

        for gi, grp in enumerate(unique_groups):
            mask  = np.array(y_raw) == gi
            color = grp_colors[gi]
            lw    = st.get('linewidth', 1.0)

            grp_pos = positions + (gi - (n_groups - 1) / 2) * width

            data_cols = [X_sel[mask, mi] for mi in range(n_mets)]
            data_cols = [col[np.isfinite(col)] for col in data_cols]

            if kind == 'box':
                bp = ax.boxplot(
                    data_cols,
                    positions=grp_pos,
                    widths=width * 0.85,
                    patch_artist=True,
                    notch=notch,
                    showmeans=show_mean,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor=color, markersize=5),
                    medianprops=dict(color='white', linewidth=1.5),
                    boxprops=dict(facecolor=color, color=color, linewidth=lw, alpha=0.80),
                    whiskerprops=dict(color=color, linewidth=lw),
                    capprops=dict(color=color, linewidth=lw),
                    flierprops=dict(marker='o', markerfacecolor=color,
                                    markeredgecolor=color, markersize=3, alpha=0.5),
                    manage_ticks=False,
                )
            else:  # violin
                valid_cols = [(col if len(col) >= 2 else np.array([0.0, 0.0]))
                              for col in data_cols]
                vp = ax.violinplot(
                    valid_cols,
                    positions=grp_pos,
                    widths=width * 0.85,
                    showmedians=True,
                    showextrema=True,
                )
                for pc in vp['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.75)
                    pc.set_linewidth(lw)
                for part in ('cmedians', 'cmeans', 'cmaxes', 'cmins', 'cbars'):
                    if part in vp:
                        vp[part].set_color(color)
                        vp[part].set_linewidth(lw)

            if show_pts:
                for mi, col in enumerate(data_cols):
                    if len(col) == 0:
                        continue
                    jitter = np.random.default_rng(seed=mi + gi * 1000
                             ).uniform(-width * 0.18, width * 0.18, len(col))
                    ax.scatter(grp_pos[mi] + jitter, col,
                               color=color, s=14, alpha=0.6,
                               edgecolors='white', linewidths=0.3, zorder=3)

            # Invisible scatter for legend
            ax.scatter([], [], color=color, label=grp, s=60)

        ax.set_xticks(positions)
        ax.set_xticklabels(feat_sel,
                           rotation=st.get('xlabel_rot', 45),
                           ha='right' if st.get('xlabel_rot', 45) > 0 else 'center',
                           fontsize=st.get('tick_size', 7))
        ax.tick_params(axis='y', labelsize=st.get('tick_size', 7))
        ax.set_xlabel(st.get('xlabel', 'Metabolite'), fontsize=st.get('label_size', 9))
        ax.set_ylabel(st.get('ylabel', 'Concentration'), fontsize=st.get('label_size', 9))
        ax.legend(fontsize=8, framealpha=0.85, loc='upper right')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

        title_w = 'bold' if st.get('title_bold', True) else 'normal'
        self._fig.suptitle(st.get('title', 'Distribution by Group'),
                           fontsize=st.get('title_size', 11), fontweight=title_w)
        self._fig.tight_layout(pad=st.get('tight_pad', 1.2), rect=[0, 0, 1, 0.95])

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
            _is_pie    = (_method == 'Metabolic Profile' and
                          getattr(self, '_pie_export_figsize', None) is not None)
            _is_box    = (_method in ('Boxplot', 'Violin') and
                          getattr(self, '_box_export_figsize', None) is not None)
            if _is_pie:
                self._fig.set_size_inches(self._pie_export_figsize)
                _st = self._pie_style
                self._fig.tight_layout(pad=_st.get('tight_pad', 1.0), rect=[0, 0, 1, 0.95])
            elif _is_box:
                self._fig.set_size_inches(self._box_export_figsize)
                _st = self._box_style
                self._fig.tight_layout(pad=_st.get('tight_pad', 1.2), rect=[0, 0, 1, 0.95])
            self._fig.savefig(fn, dpi=dpi, format=fmt,
                              bbox_inches='tight', facecolor='white')
            if _is_pie or _is_box:
                self._fig.set_size_inches(_orig_size)
                self._canvas.draw_idle()
            messagebox.showinfo('Saved', f'Figure saved to:\n{fn}', parent=self.frame)
        except Exception as e:
            messagebox.showerror('Export error', str(e), parent=self.frame)
