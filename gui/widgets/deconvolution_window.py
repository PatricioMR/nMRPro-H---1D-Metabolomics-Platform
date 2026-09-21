"""Deconvolution review window.

Every window the quantification resolved by shape (ADR-015), with the fit
drawn over the trace: this analyte's component filled, the others' drawn,
their sum dashed, and what the shapes leave. The analyst can leave an
analyte out of a window or switch a window's resolution off, see the fit
that would give, and apply it -- which re-runs the quantification with the
decision recorded, so every number still comes from one engine run and
the row says what was decided.

Opened from the Quantification stage ("Review deconvolution…") or by
right-click on a resolved row.
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from processing.quant_shared import (
    SEPARABLE_MIN, component_curves, refit_record)
from processing.sample_selection import included_sample_indices

_OWNER = '#2e7d32'
_OTHER = '#c62828'
_SUM = '#333333'
_LEFT = '#9e9e9e'


class DeconvolutionWindow:
    """
    Parameters
    ----------
    parent      : tk widget
    quant_stage : QuantificationStage (rows in ``_quant_results``,
                  overrides in ``_deconv_overrides``)
    select      : hmdb_id of the row to show first, optional
    """

    def __init__(self, parent, quant_stage, select=None):
        self.qs = quant_stage
        self.win = tk.Toplevel(parent)
        self.win.title('Deconvolution — windows resolved by shape')
        self.win.geometry('1100x720')
        self.win.resizable(True, True)
        self._rows = {}          # hmdb_id -> row
        self._current = None     # hmdb_id shown
        self._include = {}       # met_id -> tk.BooleanVar for the shown row
        self._off_var = tk.BooleanVar(value=False)
        self._sample_var = tk.StringVar(value='mean')
        self._preview = None
        self._build_ui()
        self.refresh(select=select)

    # ── Data ──────────────────────────────────────────────────────────────

    def _spectra(self):
        app = self.qs.app
        ppm = np.asarray(app.ppm, dtype=float)
        try:
            X = app.corrected_intensities()
        except Exception:
            X = app.all_intensities
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        inc = included_sample_indices(app.samples_info or [], X.shape[1])
        return ppm, X, inc

    def _sample_names(self, n):
        si = self.qs.app.samples_info or []
        return [str((si[i] if i < len(si) else {}).get('name', f'S{i + 1}'))
                for i in range(n)]

    def refresh(self, select=None):
        """Re-read the stage's rows: what was resolved, and what the analyst
        switched off."""
        rows = getattr(self.qs, '_quant_results', []) or []
        self._rows = {}
        for r in rows:
            mid = str(r.get('hmdb_id', r.get('metabolite', '')))
            if r.get('_deconv') or self._override(mid):
                self._rows[mid] = r
        self._tree.delete(*self._tree.get_children())
        for mid, r in sorted(self._rows.items(),
                             key=lambda kv: str(kv[1].get('metabolite', ''))):
            d = r.get('_deconv') or {}
            w = d.get('window') or r.get('_multiplet_window') or (np.nan, np.nan)
            others = [c['label'] for c in d.get('components', ())
                      if c.get('met_id') != mid]
            rem = list((d.get('remainder') or {}).values())
            ov = self._override(mid)
            state = ('off' if ov.get('off') else
                     f"−{len(ov.get('exclude', ()))}" if ov.get('exclude')
                     else '')
            self._tree.insert('', tk.END, iid=mid, values=(
                r.get('metabolite', mid),
                f'{min(w):.3f}–{max(w):.3f}' if np.isfinite(w[0]) else '—',
                ', '.join(others) if others else
                ('(kept whole)' if not d else '—'),
                f'{100 * float(np.median(rem)):.0f}%' if rem else '—',
                state))
        if not self._rows:
            self._info.config(
                text='No window was resolved by shape. Switch "Deconvolve '
                     'windows two accepted analytes share" on in the '
                     'quantification stage and run it again.')
            self._show(None)
            return
        pick = select if select in self._rows else (
            self._current if self._current in self._rows
            else next(iter(self._rows)))
        self._tree.selection_set(pick)
        self._tree.see(pick)
        self._show(pick)

    def _override(self, mid) -> dict:
        return dict((getattr(self.qs, '_deconv_overrides', {}) or {})
                    .get(str(mid)) or {})

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        top.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(top)
        top.add(left, weight=1)
        cols = ('owner', 'window', 'with', 'rem', 'edit')
        self._tree = ttk.Treeview(left, columns=cols, show='headings',
                                  selectmode='browse', height=12)
        for cid, lbl, wd in (('owner', 'Analyte', 150),
                             ('window', 'Window (ppm)', 95),
                             ('with', 'Resolved with', 170),
                             ('rem', 'Remainder', 70),
                             ('edit', 'Edited', 50)):
            self._tree.heading(cid, text=lbl)
            self._tree.column(cid, width=wd, anchor=tk.W, stretch=(cid == 'with'))
        self._tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._tree.bind('<<TreeviewSelect>>', self._on_select)

        right = ttk.Frame(top)
        top.add(right, weight=3)

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        bar = ttk.Frame(right)
        bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(bar, text='Sample:').pack(side=tk.LEFT)
        self._sample_box = ttk.Combobox(bar, textvariable=self._sample_var,
                                        state='readonly', width=22)
        self._sample_box.pack(side=tk.LEFT, padx=4)
        self._sample_box.bind('<<ComboboxSelected>>', lambda _e: self._draw())
        self._toolbar = NavigationToolbar2Tk(self.canvas, right,
                                             pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                         expand=True, padx=4, pady=4)

        # components
        comp = ttk.LabelFrame(self.win, text='Components')
        comp.pack(fill=tk.X, padx=6, pady=4)
        ccols = ('inc', 'analyte', 'nh', 'amp', 'share', 'ceiling', 'sep')
        self._ctree = ttk.Treeview(comp, columns=ccols, show='headings',
                                   selectmode='none', height=5)
        for cid, lbl, wd, anc in (('inc', 'In', 36, tk.CENTER),
                                  ('analyte', 'Analyte · signal', 240, tk.W),
                                  ('nh', 'nH', 50, tk.CENTER),
                                  ('amp', 'Amplitude', 90, tk.E),
                                  ('share', 'Share of fit', 90, tk.E),
                                  ('ceiling', 'Ceiling', 110, tk.CENTER),
                                  ('sep', 'Separability', 90, tk.CENTER)):
            self._ctree.heading(cid, text=lbl)
            self._ctree.column(cid, width=wd, anchor=anc,
                               stretch=(cid == 'analyte'))
        self._ctree.pack(fill=tk.X, padx=4, pady=2)
        self._ctree.bind('<Button-1>', self._on_component_click)
        self._info = ttk.Label(comp, text='', wraplength=1060,
                               justify=tk.LEFT, foreground='#444')
        self._info.pack(fill=tk.X, padx=6, pady=(0, 4))

        # actions
        act = ttk.Frame(self.win)
        act.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Checkbutton(act, text='Keep this window whole (resolution off)',
                        variable=self._off_var,
                        command=self._on_edit).pack(side=tk.LEFT)
        self._preview_lbl = ttk.Label(act, text='', foreground='#1a3a5c')
        self._preview_lbl.pack(side=tk.LEFT, padx=12)
        ttk.Button(act, text='Close', command=self.win.destroy).pack(
            side=tk.RIGHT, padx=2)
        self._reset_btn = ttk.Button(act, text='Reset window',
                                     command=self._reset, state='disabled')
        self._reset_btn.pack(side=tk.RIGHT, padx=2)
        self._apply_btn = ttk.Button(act, text='Apply & re-quantify',
                                     command=self._apply, state='disabled')
        self._apply_btn.pack(side=tk.RIGHT, padx=2)

    # ── Showing a window ──────────────────────────────────────────────────

    def _on_select(self, _e=None):
        sel = self._tree.selection()
        if sel:
            self._show(sel[0])

    def _show(self, mid):
        self._current = mid
        self._ctree.delete(*self._ctree.get_children())
        self._include = {}
        self._preview = None
        self._preview_lbl.config(text='')
        if mid is None:
            self.ax.clear()
            self.canvas.draw_idle()
            self._apply_btn.config(state='disabled')
            self._reset_btn.config(state='disabled')
            return
        r = self._rows[mid]
        d = r.get('_deconv') or {}
        ov = self._override(mid)
        self._off_var.set(bool(ov.get('off')))
        excluded = {str(x) for x in ov.get('exclude', ())}
        amps = d.get('amplitudes') or {}
        med = {m: float(np.median(list(a.values()))) for m, a in amps.items()
               if a}
        total = sum(med.get(c['met_id'], 0.0) * float(c.get('shape_area') or 0)
                    for c in d.get('components', ()))
        for c in d.get('components', ()):
            m = c['met_id']
            a = med.get(m)
            area = (a or 0.0) * float(c.get('shape_area') or 0)
            ceil = c.get('ceiling')
            at_ceiling = (ceil is not None and a is not None
                          and a >= 0.99 * float(ceil))
            sep = c.get('separability')
            self._include[m] = tk.BooleanVar(value=m not in excluded)
            self._ctree.insert('', tk.END, iid=m, values=(
                '☑' if m not in excluded else '☐',
                c.get('label', m) + ('  (this analyte)' if m == mid else '')
                + ('  (by remainder)' if c.get('by_residual') else ''),
                f"{float(c.get('nh') or 0):g}",
                f'{a:.3g}' if a is not None else '—',
                f'{100 * area / total:.0f}%' if total > 0 else '—',
                ('fixed' if c.get('fixed') else
                 'free' if ceil is None else
                 'at ceiling' if at_ceiling else f'{float(ceil):.3g}'),
                ('—' if sep is None else
                 f'{float(sep):.2f}' + ('' if sep >= SEPARABLE_MIN else ' ✗'))))
        notes = []
        if d.get('skipped'):
            notes.append('Left out: ' + ', '.join(d['skipped']) + '.')
        rem = list((d.get('remainder') or {}).values())
        if rem:
            notes.append(f'The shapes leave {100 * float(np.median(rem)):.0f}% '
                         f'of the window (median over samples).')
        if r.get('_unshared_reason'):
            notes.append('Kept whole: ' + str(r['_unshared_reason']) + '.')
        notes.append('Untick an analyte to leave it out of this window, or '
                     'keep the window whole; the fit below previews it, '
                     '"Apply" re-runs the quantification with the decision '
                     'recorded on the row.')
        self._info.config(text=' '.join(notes))
        self._reset_btn.config(state='normal' if ov else 'disabled')
        self._apply_btn.config(state='disabled')
        # samples
        try:
            _ppm, X, inc = self._spectra()
            names = self._sample_names(X.shape[1])
            self._sample_box['values'] = ['mean'] + [
                f'{i + 1}: {names[i]}' for i in inc]
        except Exception:
            self._sample_box['values'] = ['mean']
        if self._sample_var.get() not in self._sample_box['values']:
            self._sample_var.set('mean')
        self._draw()

    def _sample_index(self):
        v = self._sample_var.get()
        if v == 'mean' or ':' not in v:
            return None
        try:
            return int(v.split(':', 1)[0]) - 1
        except ValueError:
            return None

    def _draw(self):
        ax = self.ax
        ax.clear()
        mid = self._current
        if mid is None:
            self.canvas.draw_idle()
            return
        r = self._rows[mid]
        d = self._preview or r.get('_deconv') or {}
        try:
            ppm, X, inc = self._spectra()
        except Exception as exc:
            ax.set_title(f'No spectra: {exc}', fontsize=9)
            self.canvas.draw_idle()
            return
        w = d.get('window') or r.get('_multiplet_window')
        if not w:
            self.canvas.draw_idle()
            return
        lo, hi = float(min(w)), float(max(w))
        pad = 0.25 * (hi - lo)
        sel = (ppm >= lo - pad) & (ppm <= hi + pad)
        s_idx = self._sample_index()
        if s_idx is None:
            trace = np.mean(X[:, inc], axis=1) if inc else np.mean(X, axis=1)
            who = 'mean of included samples'
        else:
            trace = X[:, s_idx]
            who = self._sample_var.get()
        ax.plot(ppm[sel], trace[sel], color='#1f77b4', lw=1.2,
                label=f'trace ({who})', zorder=3)
        ax.axvspan(lo, hi, color='#000000', alpha=0.04, zorder=0)
        curves = component_curves(d, ppm, sample=s_idx)
        total = None
        for rec, x, y in curves:
            if rec['met_id'] == mid:
                ax.fill_between(x, 0, y, alpha=0.3, color=_OWNER, lw=0,
                                zorder=2)
                ax.plot(x, y, color=_OWNER, lw=1.4, ls='--', zorder=5,
                        label=rec.get('label', mid))
            else:
                ax.plot(x, y, color=_OTHER, lw=1.1, ls=':', zorder=4,
                        label=rec.get('label', rec['met_id']))
            total = y if total is None else total + y
        if total is not None and curves:
            x = curves[0][1]
            ax.plot(x, total, color=_SUM, lw=0.9, ls='--', alpha=0.7,
                    zorder=4, label='sum of shapes')
            t_in = np.interp(x, ppm[np.argsort(ppm)], trace[np.argsort(ppm)])
            ax.plot(x, np.clip(t_in - total, 0, None), color=_LEFT, lw=0.9,
                    zorder=1, label='what the shapes leave')
        ax.set_xlim(hi + pad, lo - pad)          # δ decreases to the right
        ax.set_xlabel('Chemical shift (ppm)')
        ax.set_ylabel('Intensity (au)')
        title = f"{r.get('metabolite', mid)}  {lo:.3f}–{hi:.3f} ppm"
        if self._preview:
            title += '   (preview of the edit)'
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── Editing ───────────────────────────────────────────────────────────

    def _on_component_click(self, event):
        if self._ctree.identify_column(event.x) != '#1':
            return
        iid = self._ctree.identify_row(event.y)
        if not iid or iid == self._current or iid not in self._include:
            return
        var = self._include[iid]
        var.set(not var.get())
        vals = list(self._ctree.item(iid, 'values'))
        vals[0] = '☑' if var.get() else '☐'
        self._ctree.item(iid, values=vals)
        self._on_edit()

    def _pending(self) -> dict:
        """The override the current edit amounts to; empty when none."""
        if self._off_var.get():
            return {'off': True}
        excl = sorted(m for m, v in self._include.items() if not v.get())
        return {'exclude': excl} if excl else {}

    def _on_edit(self):
        mid = self._current
        if mid is None:
            return
        r = self._rows[mid]
        d = r.get('_deconv') or {}
        pend = self._pending()
        changed = pend != self._override(mid)
        self._apply_btn.config(state='normal' if changed else 'disabled')
        self._preview = None
        text = ''
        if pend.get('off'):
            text = 'Preview: the window is integrated whole, as if not shared.'
        elif pend.get('exclude') and d:
            try:
                ppm, X, inc = self._spectra()
                fit = refit_record(d, ppm, X, exclude=pend['exclude'],
                                   included=inc)
            except Exception as exc:
                fit = None
                text = f'Preview failed: {exc}'
            if fit is None:
                text = text or ('Preview: nothing left to resolve against; '
                                'the window is integrated whole.')
            elif fit.get('refused'):
                text = f"Preview: refused — {fit['refused']}."
            else:
                self._preview = {**fit, 'skipped': []}
                old = np.median(list((d.get('areas') or
                                      self._areas_of(d, mid)).values()))
                new = np.median(list(fit['areas'][mid].values()))
                rem = np.median(list(fit['remainder'].values()))
                pct = 100 * (new / old - 1) if old else float('nan')
                text = (f"Preview: this analyte's area {pct:+.0f}%, "
                        f"remainder {100 * rem:.0f}%.")
        self._preview_lbl.config(text=text)
        self._draw()

    @staticmethod
    def _areas_of(d, mid) -> dict:
        area = next((float(c.get('shape_area') or 0)
                     for c in d.get('components', ()) if c['met_id'] == mid),
                    0.0)
        return {s: a * area for s, a in
                ((d.get('amplitudes') or {}).get(mid) or {}).items()}

    def _apply(self):
        mid = self._current
        if mid is None:
            return
        pend = self._pending()
        if not messagebox.askyesno(
                'Re-quantify all analytes',
                'Applying this decision re-runs the quantification for '
                'EVERY analyte, with the decision recorded on the row.\n\n'
                'Continue?', parent=self.win):
            return
        if getattr(self.qs, '_deconv_overrides', None) is None:
            self.qs._deconv_overrides = {}
        if pend:
            self.qs._deconv_overrides[str(mid)] = pend
        else:
            self.qs._deconv_overrides.pop(str(mid), None)
        self._apply_btn.config(state='disabled')
        self._preview_lbl.config(text='Re-quantifying…')
        self.qs._safe_run(on_done=lambda: self._after_requant(mid))

    def _reset(self):
        mid = self._current
        if mid is None or not self._override(mid):
            return
        if not messagebox.askyesno(
                'Re-quantify all analytes',
                'Dropping this decision re-runs the quantification for '
                'EVERY analyte.\n\nContinue?', parent=self.win):
            return
        self.qs._deconv_overrides.pop(str(mid), None)
        self._preview_lbl.config(text='Re-quantifying…')
        self.qs._safe_run(on_done=lambda: self._after_requant(mid))

    def _after_requant(self, mid):
        try:
            if not self.win.winfo_exists():
                return
        except Exception:
            return
        self.refresh(select=mid)
        self._preview_lbl.config(text='Re-quantified.')
