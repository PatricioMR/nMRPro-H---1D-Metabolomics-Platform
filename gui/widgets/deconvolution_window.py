# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Deconvolution review.

Everything the quantification resolves where signals overlap, in one place,
and the analyst's say over it:

  Shared window   a window two accepted analytes share, split by their shapes
  Joint fit       analytes fitted together with the unassigned peaks
  Cascade         analytes fitted to what the quantified ones leave
  No number       identified, but the fit could not place it / the part that
                  is its own is too small (a candidate, not an identification)

For each one the figure draws the spectrum of a group, the analyte filled, the
others that stand there dotted, the baseline, the sum, and what the fit leaves
unexplained -- the curves the fit records carry, the ones the report draws. The
analyst can accept (quantify it over what the engine refused), reject (leave
it out of the election, the fit and the cascade) or keep the engine's choice;
for a shared window also leave an analyte out of it or keep it whole.

Decisions are collected, then applied together: one re-quantification, every
number still from one engine run, every row saying what was decided. They are
kept in the project with the other analyst decisions (``_deconv_overrides``).

Opened from the Quantification stage ("Review deconvolution…"), by right-click
on a row, or from the toolbar. With no results yet it offers to run the
quantification to show what will be deconvolved.
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from processing.deconv_review import (
    KIND_CANDIDATE, KIND_CASCADE, KIND_HELP, KIND_JOINT, KIND_REJECTED,
    KIND_SHARED, KIND_TEXT, clean_decision, counts, decisions_changed,
    change_summary, change_text, describe_decision, item_window, region_curves,
    review_items, set_decision, unexplained_share, VIEW_PAD_PPM)
from processing.quant_shared import (
    SEPARABLE_MIN, component_curves, refit_record)
from processing.sample_selection import included_sample_indices
from plotting.fit_curve import fit_group

_OWNER = '#2e7d32'
_OTHER = '#c62828'
_SUM = '#333333'
_LEFT = '#8d6e63'
_BASE = '#ff8a65'
_DOTS = {0: '—', 1: '●', 2: '●●', 3: '●●●'}

# What a quantified row at each confidence looks like in the list.
_KIND_ORDER = (KIND_SHARED, KIND_JOINT, KIND_CASCADE, KIND_CANDIDATE,
               KIND_REJECTED)


class DeconvolutionWindow:
    """
    Parameters
    ----------
    parent      : tk widget
    quant_stage : QuantificationStage (rows in ``_quant_results``,
                  decisions in ``_deconv_overrides``)
    select      : hmdb_id of the row to show first, optional
    """

    def __init__(self, parent, quant_stage, select=None):
        self.qs = quant_stage
        self.win = tk.Toplevel(parent)
        self.win.title('Deconvolution review')
        self.win.geometry('1240x760')
        self.win.minsize(980, 600)
        self._items = []
        self._by_id = {}
        self._rows = []
        self._current = None
        self._pending = {}
        self._kind_vars = {k: tk.BooleanVar(value=True) for k in _KIND_ORDER}
        self._look_var = tk.BooleanVar(value=False)
        self._search_var = tk.StringVar(value='')
        self._view_var = tk.StringVar(value='')
        self._include = {}
        self._off_var = tk.BooleanVar(value=False)
        self._preview = None
        self._build_ui()
        self.refresh(select=select)

    # ── Data ──────────────────────────────────────────────────────────────

    def _applied(self) -> dict:
        return {k: dict(v) for k, v in
                (getattr(self.qs, '_deconv_overrides', {}) or {}).items()}

    def _matrix(self):
        """The spectrum the run measured on: the corrected one only when the
        run integrated on the baseline model."""
        app = self.qs.app
        ppm = np.asarray(app.ppm, dtype=float)
        settings = getattr(self.qs, '_last_run_settings', None) or {}
        use_model = bool(settings.get(
            'bl_model_quant_var', self.qs.bl_model_quant_var.get()))
        X = app.corrected_intensities() if use_model else app.all_intensities
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        inc = included_sample_indices(app.samples_info or [], X.shape[1])
        return ppm, X, inc

    def _groups(self) -> dict:
        g = getattr(self.qs, '_last_groups', None) or {}
        return {str(k): list(v) for k, v in g.items() if k != 'BLK'}

    def _sample_names(self, n):
        si = self.qs.app.samples_info or []
        return [str((si[i] if i < len(si) else {}).get('name', f'S{i + 1}'))
                for i in range(n)]

    def refresh(self, select=None):
        """Re-read the stage's rows and decisions."""
        self._rows = list(getattr(self.qs, '_quant_results', []) or [])
        applied = self._applied()
        self._pending = {k: dict(v) for k, v in applied.items()}
        self._items = review_items(self._rows, self._pending)
        self._by_id = {i['id']: i for i in self._items}
        self._row_of = {str(r.get('hmdb_id')): r for r in self._rows}
        self._fill_tree()
        pick = (select if select in self._by_id else
                self._current if self._current in self._by_id else None)
        kids = self._tree.get_children()
        if pick is None and kids:
            pick = kids[0]
        if pick is not None and self._tree.exists(pick):
            self._tree.selection_set(pick)
            self._tree.see(pick)
            self._show(pick)
        else:
            self._show(None)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=6, pady=(6, 2))
        self._summary = ttk.Label(top, text='', font=('Arial', 9, 'bold'))
        self._summary.pack(side=tk.LEFT)
        self._run_btn = ttk.Button(top, text='Run quantification',
                                   command=self._run_now)
        self._run_btn.pack(side=tk.RIGHT)

        filt = ttk.Frame(self.win)
        filt.pack(fill=tk.X, padx=6)
        for k in _KIND_ORDER:
            ttk.Checkbutton(filt, text=KIND_TEXT[k], variable=self._kind_vars[k],
                            command=self._fill_tree).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(filt, text='Needs a look (● or no number)',
                        variable=self._look_var,
                        command=self._fill_tree).pack(side=tk.LEFT, padx=10)
        ttk.Label(filt, text='Find:').pack(side=tk.LEFT, padx=(12, 2))
        ent = ttk.Entry(filt, textvariable=self._search_var, width=18)
        ent.pack(side=tk.LEFT)
        self._search_var.trace_add('write', lambda *_: self._fill_tree())

        body = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)

        left = ttk.Frame(body)
        body.add(left, weight=2)
        cols = ('name', 'kind', 'win', 'conf', 'left', 'decision')
        self._tree = ttk.Treeview(left, columns=cols, show='headings',
                                  selectmode='extended', height=10)
        for cid, lbl, wd, anc in (('name', 'Analyte', 160, tk.W),
                                  ('kind', 'Method', 90, tk.W),
                                  ('win', 'Window (ppm)', 100, tk.CENTER),
                                  ('conf', 'Conf.', 45, tk.CENTER),
                                  ('left', 'Left', 50, tk.CENTER),
                                  ('decision', 'Decision', 120, tk.W)):
            self._tree.heading(cid, text=lbl)
            self._tree.column(cid, width=wd, anchor=anc, stretch=(cid == 'name'))
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self._tree.tag_configure('candidate', background='#fff8e1')
        self._tree.tag_configure('rejected', foreground='#9e9e9e')
        self._tree.tag_configure('accepted', foreground='#1b5e20')
        self._tree.bind('<<TreeviewSelect>>', self._on_select)
        self._tree.bind('<a>', lambda e: self._decide('accept'))
        self._tree.bind('<r>', lambda e: self._decide('reject'))
        self._tree.bind('<u>', lambda e: self._decide(None))
        ttk.Label(left, text='A accept · R reject · U engine\'s choice  '
                             '(Ctrl/Shift-click to select several)',
                  foreground='#777', font=('Arial', 7)).grid(
                      row=1, column=0, columnspan=2, sticky='w')

        right = ttk.Frame(body)
        body.add(right, weight=5)
        self._head = ttk.Label(right, text='', font=('Arial', 10, 'bold'))
        self._head.pack(anchor=tk.W, padx=4)
        bar = ttk.Frame(right)
        bar.pack(fill=tk.X, padx=4)
        ttk.Label(bar, text='Spectrum of:').pack(side=tk.LEFT)
        self._view_box = ttk.Combobox(bar, textvariable=self._view_var,
                                      state='readonly', width=26)
        self._view_box.pack(side=tk.LEFT, padx=4)
        self._view_box.bind('<<ComboboxSelected>>', lambda _e: self._draw())

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        self.fig = plt.Figure(figsize=(5.5, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self._toolbar = NavigationToolbar2Tk(self.canvas, right,
                                             pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                         expand=True, padx=4, pady=2)

        lower = ttk.Frame(self.win)
        lower.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        comp = ttk.LabelFrame(lower, text='Fitted together in this region')
        comp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ccols = ('inc', 'analyte', 'nh', 'amp', 'share', 'note')
        self._ctree = ttk.Treeview(comp, columns=ccols, show='headings',
                                   selectmode='none', height=5)
        for cid, lbl, wd, anc in (('inc', 'In', 34, tk.CENTER),
                                  ('analyte', 'Analyte', 210, tk.W),
                                  ('nh', 'nH', 40, tk.CENTER),
                                  ('amp', 'Amplitude', 80, tk.E),
                                  ('share', 'Share', 55, tk.E),
                                  ('note', 'Note', 170, tk.W)):
            self._ctree.heading(cid, text=lbl)
            self._ctree.column(cid, width=wd, anchor=anc,
                               stretch=(cid == 'analyte'))
        self._ctree.pack(fill=tk.BOTH, expand=True, padx=3, pady=2)
        self._ctree.bind('<Button-1>', self._on_component_click)
        why = ttk.LabelFrame(lower, text='Why')
        why.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        wsb = ttk.Scrollbar(why, orient=tk.VERTICAL)
        self._why = tk.Text(why, height=6, wrap=tk.WORD, font=('Arial', 8),
                            relief=tk.FLAT, background='#fafafa',
                            yscrollcommand=wsb.set)
        wsb.config(command=self._why.yview)
        wsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._why.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3, pady=2)
        self._why.config(state=tk.DISABLED)

        act = ttk.Frame(self.win)
        # the footer is reserved first, so a tall list or figure cannot push
        # the buttons off the screen
        act.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6, before=lower)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)
        self._accept_btn = ttk.Button(act, text='✓ Accept', command=lambda: self._decide('accept'))
        self._reject_btn = ttk.Button(act, text='✗ Reject', command=lambda: self._decide('reject'))
        self._engine_btn = ttk.Button(act, text='↺ Engine\'s choice', command=lambda: self._decide(None))
        for b in (self._accept_btn, self._reject_btn, self._engine_btn):
            b.pack(side=tk.LEFT, padx=2)
        self._whole_chk = ttk.Checkbutton(
            act, text='Keep this shared window whole', variable=self._off_var,
            command=self._on_edit)
        self._whole_chk.pack(side=tk.LEFT, padx=12)
        self._preview_lbl = ttk.Label(act, text='', foreground='#1a3a5c',
                                      wraplength=420, justify=tk.LEFT)
        self._preview_lbl.pack(side=tk.LEFT, padx=8)
        ttk.Button(act, text='Close', command=self.win.destroy).pack(
            side=tk.RIGHT, padx=2)
        self._discard_btn = ttk.Button(act, text='Discard changes',
                                       command=self._discard, state='disabled')
        self._discard_btn.pack(side=tk.RIGHT, padx=2)
        self._apply_btn = ttk.Button(act, text='Apply & re-quantify',
                                     command=self._apply, state='disabled')
        self._apply_btn.pack(side=tk.RIGHT, padx=2)

    # ── The list ──────────────────────────────────────────────────────────

    def _visible(self):
        q = self._search_var.get().strip().lower()
        out = []
        for i in self._items:
            if not self._kind_vars[i['kind']].get():
                continue
            if self._look_var.get() and not (
                    i['kind'] == KIND_CANDIDATE
                    or (i['quantified'] and i['confidence'] <= 1)):
                continue
            if q and q not in i['name'].lower():
                continue
            out.append(i)
        return out

    @staticmethod
    def _method_text(i) -> str:
        """What produced (or tried to produce) the number."""
        via = KIND_TEXT.get(i.get('via'), KIND_TEXT[i['kind']])
        return f'{via} · no number' if i['kind'] == KIND_CANDIDATE else via

    def _decision_text(self, mid) -> str:
        t = describe_decision(self._pending.get(mid))
        if clean_decision(self._pending.get(mid)) != clean_decision(self._applied().get(mid)):
            t = (t or "engine's choice") + ' *'
        return t

    def _fill_tree(self):
        keep = set(self._tree.selection())
        self._tree.delete(*self._tree.get_children())
        for i in self._visible():
            w = i['window']
            dec = clean_decision(self._pending.get(i['id'])).get('decision')
            tags = (('rejected',) if i['kind'] == KIND_REJECTED or dec == 'reject'
                    else ('accepted',) if dec == 'accept'
                    else ('candidate',) if i['kind'] == KIND_CANDIDATE else ())
            self._tree.insert('', tk.END, iid=i['id'], tags=tags, values=(
                i['name'], self._method_text(i),
                f'{w[0]:.3f}–{w[1]:.3f}' if w else '—',
                _DOTS[i['confidence']] if i['quantified'] else '—',
                f"{100 * i['unexplained']:.0f}%" if i['unexplained'] is not None else '',
                self._decision_text(i['id'])))
        for m in keep:
            if self._tree.exists(m):
                self._tree.selection_add(m)
        self._update_summary()

    def _update_summary(self):
        c = counts(self._items)
        n = len(decisions_changed(self._applied(), self._pending))
        if not self._rows:
            self._summary.config(text='No quantification yet.')
        else:
            self._summary.config(text=(
                f"{c[KIND_SHARED]} shared window(s) · {c[KIND_JOINT]} joint fit · "
                f"{c[KIND_CASCADE]} cascade · {c[KIND_CANDIDATE]} with no number"
                + (f" · {c[KIND_REJECTED]} rejected" if c[KIND_REJECTED] else '')
                + (f"   —   {n} decision(s) waiting" if n else '')))
        self._apply_btn.config(
            text=f'Apply {n} decision(s) & re-quantify' if n else 'Apply & re-quantify',
            state='normal' if n else 'disabled')
        self._discard_btn.config(state='normal' if n else 'disabled')
        self._run_btn.config(text='Re-run quantification' if self._rows
                             else 'Run quantification')

    # ── Showing one ───────────────────────────────────────────────────────

    def _on_select(self, _e=None):
        sel = self._tree.selection()
        if sel:
            self._show(sel[-1] if len(sel) == 1 else sel[0])
        self._update_buttons()

    def _update_buttons(self):
        have = bool(self._tree.selection())
        st = 'normal' if have else 'disabled'
        for b in (self._accept_btn, self._reject_btn, self._engine_btn):
            b.config(state=st)

    def _set_why(self, text):
        self._why.config(state=tk.NORMAL)
        self._why.delete('1.0', tk.END)
        self._why.insert('1.0', text)
        self._why.config(state=tk.DISABLED)

    def _show(self, mid):
        self._current = mid
        self._preview = None
        self._preview_lbl.config(text='')
        self._ctree.delete(*self._ctree.get_children())
        self._include = {}
        if mid is None or mid not in self._by_id:
            self._head.config(text='')
            self._why_empty()
            self.ax.clear()
            self.canvas.draw_idle()
            self._view_box['values'] = []
            self._whole_chk.state(['disabled'])
            return
        it = self._by_id[mid]
        r = self._row_of[mid]
        d = r.get('_deconv') or {}
        dots = _DOTS[it['confidence']] if it['quantified'] else 'no number'
        self._head.config(text=f"{it['name']}   ·   {self._method_text(it)}   ·   {dots}")
        # who can be looked at: the groups; and the samples of a shared window
        groups = self._groups()
        views = [f'group {g}' for g in groups]
        if d and it['kind'] == KIND_SHARED:
            try:
                _p, X, inc = self._matrix()
                names = self._sample_names(X.shape[1])
                views += [f'sample {i + 1}: {names[i]}' for i in inc]
            except Exception:
                pass
        self._view_box['values'] = views
        fc = r.get('_fit_curve') or {}
        default = None
        if fc:
            g = fit_group(fc, r.get('_snr_group'))
            default = f'group {g}' if f'group {g}' in views else None
        if self._view_var.get() not in views:
            self._view_var.set(default or (views[0] if views else ''))
        elif default and it['kind'] != KIND_SHARED:
            self._view_var.set(default)
        # the shared window's own decisions
        ov = clean_decision(self._pending.get(mid))
        self._off_var.set(bool(ov.get('off')))
        excluded = {str(x) for x in ov.get('exclude', ())}
        self._whole_chk.state(['!disabled'] if (d and it['kind'] == KIND_SHARED) else ['disabled'])
        self._fill_components(it, r, d, excluded)
        self._fill_why(it, r)
        self._draw()
        self._update_buttons()

    def _why_empty(self):
        if not self._rows:
            self._set_why('Nothing has been quantified yet. "Run quantification" '
                          'shows here every window that will be deconvolved, '
                          'with the fit drawn, for you to accept, reject or '
                          'change before it goes in a result.')
        else:
            self._set_why('Nothing was deconvolved, or the filters hide it all.')

    def _fill_components(self, it, r, d, excluded):
        """The analytes fitted in this region, each with its share of the fit
        in the window. A shared window's components can be left out."""
        tot = None
        rows = []
        if d and it['kind'] == KIND_SHARED:
            amps = d.get('amplitudes') or {}
            med = {m: float(np.median(list(a.values()))) for m, a in amps.items() if a}
            tot = sum(med.get(c['met_id'], 0.0) * float(c.get('shape_area') or 0)
                      for c in d.get('components', ()))
            for c in d.get('components', ()):
                m = c['met_id']
                a = med.get(m)
                area = (a or 0.0) * float(c.get('shape_area') or 0)
                sep = c.get('separability')
                note = ('this analyte' if m == it['id'] else '')
                if c.get('fixed'):
                    note += ' · fixed'
                if sep is not None and sep < SEPARABLE_MIN:
                    note += ' · inseparable'
                rows.append((m, m not in excluded, c.get('label', m),
                             f"{float(c.get('nh') or 0):g}",
                             f'{a:.3g}' if a is not None else '—',
                             f'{100 * area / tot:.0f}%' if tot else '—', note.strip(' ·')))
        else:
            ppm, X, inc = self._matrix()
            win = it['window']
            if win:
                lo, hi = win
                g = self._view_group()
                curves = [c for c in region_curves(self._rows, ppm, lo, hi, g)
                          if c[0] != 'baseline'] if g else []
                areas = {c[0]: float(np.trapezoid(c[3], c[2])) for c in curves}
                tot = sum(areas.values())
                for mid, name, _x, _y in curves:
                    other = self._by_id.get(mid)
                    note = (KIND_TEXT[other['kind']] if other else '')
                    if mid == it['id']:
                        note = 'this analyte · ' + note
                    rows.append((mid, None, name, '', '',
                                 f'{100 * areas[mid] / tot:.0f}%' if tot else '—',
                                 note.strip(' ·')))
        for m, inc_on, label, nh, amp, share, note in rows:
            if inc_on is not None:
                self._include[m] = tk.BooleanVar(value=inc_on)
            self._ctree.insert('', tk.END, iid=m, values=(
                '' if inc_on is None else ('☑' if inc_on else '☐'),
                label, nh, amp, share, note))

    def _fill_why(self, it, r):
        lines = [KIND_HELP[it['kind']]]
        if it['why']:
            lines.append('Identity: ' + it['why'] + '.')
        if it['score'] is not None:
            lines.append(f"The cascade's model scores it {100 * float(it['score']):.0f}% likely to be there.")
        if it['unexplained'] is not None:
            lines.append(f"The shapes leave {100 * it['unexplained']:.0f}% of the window (median over samples).")
        c = it.get('covar')
        if c and np.isfinite(c.get('r_min', float('nan'))):
            lines.append(
                f"Across the {c['n_samples']} samples its {c['n_signals']} clear "
                f"signals " + ("move together" if c['r_min'] >= 0.8 else
                              "do not clearly move together")
                + f" (lowest pairwise r {c['r_min']:+.2f}). Information only: "
                "signals that move together belong to one molecule, not "
                "necessarily this one, and a constant analyte has nothing to "
                "correlate.")
        lines += [f'⚠ {w}.' for w in it['warnings']]
        for t in (r.get('_status_reasons') or [])[:2]:
            lines.append(str(t))
        dec = self._decision_text(it['id'])
        if dec:
            lines.append('Decision: ' + dec)
        self._set_why('\n'.join(lines))

    # ── The figure ────────────────────────────────────────────────────────

    def _view_group(self):
        v = self._view_var.get()
        return v[len('group '):] if v.startswith('group ') else None

    def _view_sample(self):
        v = self._view_var.get()
        if v.startswith('sample ') and ':' in v:
            try:
                return int(v[len('sample '):].split(':', 1)[0]) - 1
            except ValueError:
                return None
        return None

    def _draw(self):
        ax = self.ax
        ax.clear()
        mid = self._current
        if mid is None or mid not in self._by_id:
            self.canvas.draw_idle()
            return
        it = self._by_id[mid]
        r = self._row_of[mid]
        d = self._preview or r.get('_deconv') or {}
        win = it['window']
        try:
            ppm, X, inc = self._matrix()
        except Exception as exc:
            ax.set_title(f'No spectra: {exc}', fontsize=9)
            self.canvas.draw_idle()
            return
        if not win:
            self.canvas.draw_idle()
            return
        lo, hi = win
        pad = max(VIEW_PAD_PPM, 0.6 * (hi - lo))
        sel = (ppm >= lo - pad) & (ppm <= hi + pad)
        s_idx = self._view_sample()
        g = self._view_group()
        groups = self._groups()
        if s_idx is not None:
            trace = X[:, s_idx]
            who = self._view_var.get()
        else:
            idx = [i for i in groups.get(g, ()) if 0 <= i < X.shape[1]] or inc
            trace = np.mean(X[:, idx], axis=1)
            who = f'group {g}' if g else 'mean of included samples'
        order = np.argsort(ppm[sel])
        ax.plot(ppm[sel][order], trace[sel][order], color='#1f77b4', lw=1.3,
                label=f'spectrum ({who})', zorder=3)
        ax.axvspan(lo, hi, color='#000000', alpha=0.05, zorder=0)
        total = None
        base = None
        fit_like = it['kind'] in (KIND_JOINT, KIND_CASCADE) or not d
        if fit_like and g:
            for m, name, x, y in region_curves(self._rows, ppm, lo - pad, hi + pad, g):
                if m == 'baseline':
                    base = y
                    continue
                if m == mid:
                    ax.fill_between(x, 0 if base is None else base, y if base is None else y + base,
                                    alpha=0.3, color=_OWNER, lw=0, zorder=2)
                    ax.plot(x, y if base is None else y + base, color=_OWNER,
                            lw=1.5, ls='--', zorder=5, label=name)
                else:
                    ax.plot(x, y if base is None else y + base, color=_OTHER, lw=1.0,
                            ls=':', zorder=4, label=name)
                total = y if total is None else total + y
            if base is not None and total is not None:
                ax.plot(x, base, color=_BASE, lw=0.8, ls='--', alpha=0.8,
                        zorder=3, label='baseline of the fit')
        elif d:
            curves = component_curves(d, ppm, sample=s_idx)
            for rec, x, y in curves:
                if rec['met_id'] == mid:
                    ax.fill_between(x, 0, y, alpha=0.3, color=_OWNER, lw=0, zorder=2)
                    ax.plot(x, y, color=_OWNER, lw=1.5, ls='--', zorder=5,
                            label=rec.get('label', mid))
                else:
                    ax.plot(x, y, color=_OTHER, lw=1.0, ls=':', zorder=4,
                            label=rec.get('label', rec['met_id']))
                total = y if total is None else total + y
        if total is not None:
            xs = x
            full = total if base is None else total + base
            ax.plot(xs, full, color=_SUM, lw=0.9, ls='--', alpha=0.7, zorder=4,
                    label='sum of the fit')
            t_in = np.interp(xs, ppm[np.argsort(ppm)], trace[np.argsort(ppm)])
            ax.plot(xs, np.clip(t_in - full, 0, None), color=_LEFT, lw=0.9,
                    zorder=1, label='what the fit leaves')
            inside = (xs >= lo) & (xs <= hi)
            left = unexplained_share(t_in[inside], full[inside], xs[inside])
            if left is not None:
                ax.text(0.02, 0.96, f'fit leaves {100 * left:.0f}% of this window',
                        transform=ax.transAxes, va='top', fontsize=8, color='#444',
                        bbox=dict(fc='white', alpha=0.8, lw=0))
        # a candidate has no curve: where the cascade saw it
        if it['kind'] == KIND_CANDIDATE:
            for ppm_s, label, nh, snr in (r.get('_cascade') or {}).get('signals', ()):
                ax.axvline(ppm_s, color='#f9a825', lw=1.2, ls='-.', zorder=4)
                ax.text(ppm_s, 0.92, f'{nh:g} H', rotation=90, fontsize=7,
                        color='#b26a00', transform=ax.get_xaxis_transform(),
                        ha='right', va='top')
        tv = trace[sel]
        if tv.size:
            top = float(np.max(tv)) * 1.15
            ax.set_ylim(min(0.0, float(np.min(tv)) * 1.1), top if top > 0 else 1.0)
        ax.set_xlim(hi + pad, lo - pad)          # δ decreases to the right
        ax.set_xlabel('Chemical shift (ppm)')
        ax.set_ylabel('Intensity (au)')
        title = f"{it['name']}   {lo:.3f}–{hi:.3f} ppm"
        if self._preview:
            title += '   (preview of the edit)'
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── Deciding ──────────────────────────────────────────────────────────

    def _decide(self, decision):
        ids = [m for m in self._tree.selection() if m in self._by_id]
        if not ids:
            return
        for m in ids:
            # a shared window's own settings stay; the decision is the part
            # the buttons change
            self._pending = set_decision(self._pending, m, decision)
        self._after_decision(ids)

    def _after_decision(self, ids, *, effect=True):
        self._items = review_items(self._rows, self._pending)
        self._by_id = {i['id']: i for i in self._items}
        sel = list(self._tree.selection())
        self._fill_tree()
        for m in sel:
            if self._tree.exists(m):
                self._tree.selection_add(m)
        if self._current in self._by_id:
            self._fill_why(self._by_id[self._current], self._row_of[self._current])
        if effect:
            self._preview_lbl.config(text=self._decision_effect(ids))
        self._update_summary()

    def _decision_effect(self, ids) -> str:
        decs = [clean_decision(self._pending.get(m)).get('decision') for m in ids]
        if len(ids) > 1:
            return f'{len(ids)} analytes changed. Apply to re-quantify once.'
        d = decs[0]
        if d == 'accept':
            return ('Applying quantifies it over what the engine refused (its '
                    "confidence stays the engine's), the row saying you "
                    'accepted it.')
        if d == 'reject':
            return ('Applying leaves it out of the election, the fit and the '
                    'cascade: the others are measured without it.')
        return "The engine's choice stands."

    def _on_component_click(self, event):
        if self._ctree.identify_column(event.x) != '#1':
            return
        iid = self._ctree.identify_row(event.y)
        mid = self._current
        if (not iid or mid is None or iid == mid or iid not in self._include):
            return
        var = self._include[iid]
        var.set(not var.get())
        vals = list(self._ctree.item(iid, 'values'))
        vals[0] = '☑' if var.get() else '☐'
        self._ctree.item(iid, values=vals)
        self._on_edit()

    def _pending_window(self):
        if self._off_var.get():
            return True, []
        return False, sorted(m for m, v in self._include.items() if not v.get())

    def _on_edit(self):
        """The shared window's own edit: keep it whole, or leave analytes out."""
        mid = self._current
        if mid is None:
            return
        off, excl = self._pending_window()
        self._pending = set_decision(
            self._pending, mid,
            clean_decision(self._pending.get(mid)).get('decision'),
            off=off, exclude=excl)
        d = self._row_of[mid].get('_deconv') or {}
        self._preview = None
        text = ''
        if off:
            text = 'Preview: the window is integrated whole, as if not shared.'
        elif excl and d:
            try:
                ppm, X, inc = self._matrix()
                fit = refit_record(d, ppm, X, exclude=excl, included=inc)
            except Exception as exc:
                fit, text = None, f'Preview failed: {exc}'
            if fit is None:
                text = text or ('Preview: nothing left to resolve against; the '
                                'window is integrated whole.')
            elif fit.get('refused'):
                text = f"Preview: refused — {fit['refused']}."
            else:
                self._preview = {**fit, 'skipped': []}
                areas = {s: a * float(next((c.get('shape_area') or 0 for c in d.get('components', ())
                                            if c['met_id'] == mid), 0.0))
                         for s, a in ((d.get('amplitudes') or {}).get(mid) or {}).items()}
                old = np.median(list(areas.values())) if areas else 0.0
                new = np.median(list(fit['areas'][mid].values()))
                rem = np.median(list(fit['remainder'].values()))
                pct = 100 * (new / old - 1) if old else float('nan')
                text = (f"Preview: this analyte's area {pct:+.0f}%, "
                        f"remainder {100 * rem:.0f}%.")
        self._after_decision([mid], effect=False)
        self._preview_lbl.config(text=text)
        self._draw()

    def _discard(self):
        self._pending = self._applied()
        self._items = review_items(self._rows, self._pending)
        self._by_id = {i['id']: i for i in self._items}
        self._fill_tree()
        if self._current:
            self._show(self._current if self._current in self._by_id else None)

    def _apply(self):
        changed = decisions_changed(self._applied(), self._pending)
        if not changed:
            return
        names = ', '.join(self._by_id[m]['name'] if m in self._by_id else m
                          for m in changed[:6]) + (' …' if len(changed) > 6 else '')
        if not messagebox.askyesno(
                'Re-quantify all analytes',
                f'{len(changed)} decision(s): {names}\n\nApplying them re-runs the '
                'quantification for EVERY analyte, once, with the decisions '
                'recorded on their rows. Numbers of other analytes may change.\n\n'
                'Continue?', parent=self.win):
            return
        self.qs._deconv_overrides = {k: dict(v) for k, v in self._pending.items()}
        self._apply_btn.config(state='disabled')
        self._preview_lbl.config(text='Re-quantifying…')
        keep = self._current
        self._before_rows = [dict(r) for r in self._rows]
        self.qs._safe_run(on_done=lambda: self._after_requant(keep))

    def _run_now(self):
        if self._rows and not messagebox.askyesno(
                'Run quantification',
                'Run the quantification again with the decisions applied so far?',
                parent=self.win):
            return
        self._preview_lbl.config(text='Quantifying…')
        keep = self._current
        self._before_rows = [dict(r) for r in self._rows]
        self.qs._safe_run(on_done=lambda: self._after_requant(keep))

    def _after_requant(self, mid):
        try:
            if not self.win.winfo_exists():
                return
        except Exception:
            return
        before = getattr(self, '_before_rows', None)
        self.refresh(select=mid)
        if before:
            self.last_change = change_summary(before, self._rows)
            self._preview_lbl.config(text=change_text(self.last_change))
        else:
            self._preview_lbl.config(text='Quantified.')
