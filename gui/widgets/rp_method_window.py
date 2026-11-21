# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""RP-Method: the resolution plot.

The spectrum of a group above the axis, everything the quantification
resolved of it mirrored below -- each analyte as the curve that gave its
number, in the colour it has in the Quantification viewer, the report and
the deconvolution review -- and the residual (spectrum less the fits'
baselines less the resolved curves) over the spectrum. Where the residual
stands above the noise a signal is missing; where it stands below, a fit
claims more than the spectrum holds. The computation is
``processing.resolution_plot``.

The table lists those regions with their numbers; picking one zooms on it
and lists the annotation's analytes with a signal there, each with what the
annotation already measured of it -- multiplicity, the reference's J, the J
it observed, their difference, the lines found -- nothing measured again.
Picking candidates lays their reference fingerprints over the residual, so
the analyst sees which multiplet fits. A candidate's signal can be marked
Good -- the analyst's validation, which the next quantification takes into
the election (``controllers.signal_lines``) -- and the quantification re-run
from here.

Opened from Analysis > RP-Method.
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from controllers.analysis_defaults import QUANTIFICATION
from plotting import contributions
from processing import resolution_plot as rp

_INK = '#1a1a19'
_RESIDUAL = '#c62828'
_MISSING_FILL = '#f6c9c9'
_OVER_FILL = '#c9daf6'
_ZOOM_PAD_PPM = 0.04
# the window the quantification looks for its reference in (``_find_peak``
# with ``tol=0.05``): the reference is not an analyte to be explained
_REF_TOL_PPM = 0.05
# beyond this many analytes in a zoomed view, names are left to the table
_MAX_LABELS = 14
_QUALITY = {'good': 'Good', 'bad': 'Poor', None: 'Average'}


def _hz(v) -> str:
    """A J in Hz as the annotation stored it, two decimals; '—' when none."""
    if v is None or v == '':
        return '—'
    try:
        return f'{float(v):.2f}'
    except (TypeError, ValueError):
        return str(v)


_HELP = (
    "Above the axis: the spectrum (black) and the residual (red) -- the spectrum less "
    "the baselines the fits stood on, less everything resolved.\n\n"
    "Below the axis, mirrored: what the quantification resolved, each analyte in its "
    "colour. A fit is drawn whole; an integral (hatched) is the data it summed; an "
    "integral or a shape models only the signal its number came from.\n\n"
    "Subtract confirmed analytes' other signals: for every analyte at two dots or more, "
    "its other signals -- a multiplet confirmed by the annotation (all lines, dJ and "
    "d-ppm within the quality parameters), a singlet only when STOCSY linked it -- are "
    "taken off as its own number predicts them (dotted), only up to what the residual "
    "holds.\n\n"
    "Down-triangle: a signal nothing explains (missing). Up-triangle: the resolved "
    "standing above the spectrum (over-claim). Regions of at least the lone-singlet SNR, "
    "on the detection noise.\n\n"
    "Picking candidates fits each into the residual (dashed: the largest copy of its "
    "reference that fits under it); for one already quantified, dotted, what its own "
    "number predicts there.")


class RPMethodWindow:
    def __init__(self, app, quant_stage, ann_stage=None):
        self.app = app
        self.qs = quant_stage
        self.ann = ann_stage
        self.win = tk.Toplevel(app.root)
        self.win.title('RP-Method — resolution plot')
        self.win.geometry('1320x820')
        self.win.minsize(1000, 640)
        self._group_var = tk.StringVar(value='')
        self._regions = []
        self._cands = []
        self._view = None
        self._out = None
        self._marked = 0
        self._hide_other_var = tk.BooleanVar(value=False)
        self._secondary_var = tk.BooleanVar(value=True)
        self._summary_base = ''
        self._region_shown = None
        self._sigma = None
        self._shown = []
        self._build_ui()
        self.refresh()

    # ── Data ──────────────────────────────────────────────────────────────

    def _inputs(self):
        from gui.stages._quant_computation import excluded_regions_of
        from gui.widgets.deconvolution_window import measured_matrix, run_groups
        ppm, X, inc = measured_matrix(self.qs)
        groups = {g: [i for i in idx if i in set(inc)] for g, idx in run_groups(self.qs).items()}
        groups = {g: idx for g, idx in groups.items() if idx} or {'all samples': list(inc)}
        rows = list(getattr(self.qs, '_quant_results', []) or [])
        excluded = list(excluded_regions_of(getattr(self.qs, 'peak_stage', None))[0] or [])
        for r in rows:
            if r.get('_is_is'):
                sp = contributions.number_region(r, ppm)
                if sp:
                    excluded.append(sp)
        try:
            ref = float(self.qs.ref_ppm_var.get())
            excluded.append((ref - _REF_TOL_PPM, ref + _REF_TOL_PPM))
        except (AttributeError, tk.TclError, ValueError):
            pass
        sigma = getattr(getattr(self.qs, 'peak_stage', None), 'noise_std', None)
        try:
            min_snr = float(self.qs.singlet_snr_var.get())
        except (AttributeError, tk.TclError, ValueError):
            min_snr = float(QUANTIFICATION['singlet_min_snr'])
        return ppm, X, groups, rows, excluded, sigma, min_snr

    def _metabolites(self):
        return (getattr(self.ann, '_metabolites', None) or []) if self.ann else []

    def _tolerance(self) -> float:
        try:
            return float(self.ann.tolerance_var.get())
        except (AttributeError, tk.TclError, ValueError):
            from controllers.analysis_defaults import ANNOTATION
            return float(ANNOTATION['tolerance_ppm'])

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(top, text='Spectrum of:').pack(side=tk.LEFT)
        self._group_box = ttk.Combobox(top, textvariable=self._group_var,
                                       state='readonly', width=16)
        self._group_box.pack(side=tk.LEFT, padx=4)
        self._group_box.bind('<<ComboboxSelected>>', lambda _e: self.refresh())
        ttk.Button(top, text='Whole spectrum',
                   command=lambda: self._zoom(None)).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text='◀ Previous region', command=lambda: self._step(-1)).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(top, text='Next region ▶', command=lambda: self._step(1)).pack(
            side=tk.LEFT, padx=2)
        ttk.Checkbutton(top, text="Subtract confirmed analytes' other signals",
                        variable=self._secondary_var,
                        command=self.refresh).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(top, text="Hide residuals on a quantified analyte's other signal",
                        variable=self._hide_other_var,
                        command=self._refill).pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text='Close', command=self.win.destroy).pack(side=tk.RIGHT)
        self._run_btn = ttk.Button(top, text='Re-run quantification',
                                   command=self._rerun)
        self._run_btn.pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text='?', width=3, command=self._help).pack(side=tk.RIGHT, padx=2)
        # one short line of numbers; the warning, when there is one, beside it
        line = ttk.Frame(self.win)
        line.pack(fill=tk.X, padx=8)
        self._summary = ttk.Label(line, text='', font=('Arial', 9, 'bold'))
        self._summary.pack(side=tk.LEFT)
        self._warn = ttk.Label(line, text='', font=('Arial', 8), foreground='#b26a00')
        self._warn.pack(side=tk.LEFT, padx=12)

        body = ttk.PanedWindow(self.win, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        fig_frame = ttk.Frame(body)
        body.add(fig_frame, weight=3)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        self.fig = plt.Figure(figsize=(9, 3.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        tb = NavigationToolbar2Tk(self.canvas, fig_frame, pack_toolbar=False)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_click)

        low = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
        body.add(low, weight=2)

        reg = ttk.LabelFrame(low, text='Residual: signals nothing explains, and over-claims')
        low.add(reg, weight=2)
        cols = (('n', '#', 32, tk.CENTER), ('kind', 'Kind', 58, tk.CENTER),
                ('range', 'δ (ppm)', 96, tk.CENTER), ('apex', 'Apex', 52, tk.CENTER),
                ('snr', 'σ', 50, tk.E), ('share', 'Share', 48, tk.E),
                ('other', 'Analytes there', 150, tk.W),
                ('best', 'Nearest candidate', 140, tk.W))
        self._rtree = self._table(reg, cols, height=7)
        self._rtree.bind('<<TreeviewSelect>>', self._on_region)

        cand = ttk.LabelFrame(low, text='Candidates: the annotation\'s analytes with a signal there')
        low.add(cand, weight=3)
        ccols = (('name', 'Analyte', 150, tk.W), ('mult', 'Mult', 38, tk.CENTER),
                 ('jref', 'J ref (Hz)', 64, tk.E), ('jobs', 'J obs (Hz)', 64, tk.E),
                 ('dj', 'ΔJ (Hz)', 54, tk.E), ('lines', 'Lines', 44, tk.CENTER),
                 ('listed', 'Listed δ', 58, tk.CENTER), ('delta', 'Δδ (mppm)', 68, tk.E),
                 ('nh', 'nH', 30, tk.CENTER), ('quality', 'Annot.', 58, tk.CENTER),
                 ('status', 'Quantification', 170, tk.W))
        self._ctree = self._table(cand, ccols, height=6, selectmode='extended')
        self._ctree.bind('<<TreeviewSelect>>', self._on_candidates)
        act = ttk.Frame(cand)
        act.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(2, 0))
        self._good_btn = ttk.Button(act, text='✓ Mark signal Good (include)',
                                    command=lambda: self._mark('good'))
        self._good_btn.pack(side=tk.LEFT, padx=2)
        self._clear_btn = ttk.Button(act, text='Clear mark',
                                     command=lambda: self._mark(None))
        self._clear_btn.pack(side=tk.LEFT, padx=2)
        ttk.Label(act, text='Ctrl/Shift-click: several fingerprints at once',
                  foreground='#777', font=('Arial', 7)).pack(side=tk.LEFT, padx=6)
        self._note = ttk.Label(act, text='', foreground='#1a3a5c', wraplength=420,
                               justify=tk.LEFT, font=('Arial', 8))
        self._note.pack(side=tk.LEFT, padx=8)
        self._update_buttons()

    def _help(self):
        messagebox.showinfo('How to read the resolution plot', _HELP, parent=self.win)

    @classmethod
    def _table(cls, parent, cols, height, selectmode='browse'):
        tree = ttk.Treeview(parent, columns=[c[0] for c in cols], show='headings',
                            selectmode=selectmode, height=height)
        for cid, lbl, wd, anc in cols:
            tree.heading(cid, text=lbl, command=lambda c=cid: cls._sort(tree, c))
            tree.column(cid, width=wd, anchor=anc, stretch=(anc == tk.W))
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        sb.grid(row=0, column=1, sticky='ns')
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        return tree

    @staticmethod
    def _sort(tree, col):
        """Sort the rows on a column, numbers as numbers; a second click on
        the same column reverses."""
        def key(v):
            t = str(v).replace('%', '').replace('+', '').strip()
            try:
                return (0, abs(float(t)) if col in ('delta', 'dj') else float(t))
            except ValueError:
                return (1, t.lower())
        items = [(key(tree.set(k, col)), k) for k in tree.get_children('')]
        rev = getattr(tree, '_sorted', None) == col
        items.sort(reverse=rev)
        for i, (_v, k) in enumerate(items):
            tree.move(k, '', i)
        tree._sorted = None if rev else col

    # ── Computing and drawing ─────────────────────────────────────────────

    def refresh(self):
        """Read the project as it is now and recompute everything."""
        from gui.widgets.pdf_report import _group_amplitude
        try:
            ppm, X, groups, rows, excluded, sigma, min_snr = self._inputs()
        except Exception as exc:
            self._summary.config(text=f'No spectra: {exc}')
            return
        if not rows:
            self._summary.config(text='Nothing quantified yet: run the quantification first.')
            return
        names = list(groups)
        self._group_box['values'] = names
        if self._group_var.get() not in names:
            self._group_var.set(names[0])
        g = self._group_var.get()
        trace = np.mean(X[:, groups[g]], axis=1)
        si = self.app.samples_info or []
        out = rp.resolve(rows, ppm, trace, g, lambda a: _group_amplitude(a, si, g))
        mets, tol = self._metabolites(), self._tolerance()
        sec = []
        if self._secondary_var.get() and sigma:
            sec = rp.secondary_signals(rows, mets, ppm, out['residual'], g, sigma,
                                       field_mhz=self._field(),
                                       study_lw_hz=self._study_linewidth(),
                                       tolerance_ppm=tol)
            for c in sec:
                v = rp.on_axis(c['x'], c['y'], ppm)
                out['resolved'] = out['resolved'] + v
                out['residual'] = out['residual'] - v
            out['curves'] = out['curves'] + sec
        self._ppm, self._trace, self._rows, self._out = ppm, trace, rows, out
        self._palette = contributions.stable_palette(rows, ppm)
        self._excluded = excluded
        self._sigma = sigma
        self._regions = (rp.residual_regions(ppm, trace, out['residual'], sigma,
                                             min_snr=min_snr, excluded=excluded)
                         if sigma else [])
        share = rp.resolved_share(ppm, trace, out['resolved'], out['baseline'], excluded)
        self._summary_base = (
            (f'Group {g}: resolved {100 * share:.0f}%' if share is not None
             else f'Group {g}: resolved —')
            + (self._secondary_text(sec) if sec else ''))
        warn = []
        if not sigma:
            warn.append('no noise level from detection: run Signal Detection')
        if self._secondary_var.get() and not any(r.get('_signal_links') for r in rows):
            warn.append('singlets not subtracted: no STOCSY links in this run '
                        '(re-run the quantification)')
        self._warn.config(text='  ·  '.join(warn))
        for reg in self._regions:
            missing = reg['kind'] == rp.MISSING
            reg['other'] = rp.quantified_signals_in(reg, mets, rows, ppm) if missing else []
            reg['claimed'] = [] if missing else rp.claimed_by(reg, out['curves'], sigma)
            c = rp.candidates(reg, mets, rows, tol, ppm) if missing else []
            reg['best'] = c[0]['name'] if c else ''
        self._share = share
        self._refill()

    @staticmethod
    def _secondary_text(sec):
        pred = sum(c['predicted'] for c in sec)
        taken = sum(c['taken'] for c in sec)
        return (f"  ·  {len(sec)} other signals of {len({c['id'] for c in sec})} confirmed "
                f"analytes subtracted ({100 * taken / pred:.0f}% fit)" if pred > 0 else '')

    def _refill(self):
        """The table and the figure for the regions the filter keeps."""
        if self._out is None:
            return
        hide = self._hide_other_var.get()
        self._shown = [g for g in self._regions if not (hide and g['other'])]
        n_miss = sum(1 for g in self._shown if g['kind'] == rp.MISSING)
        n_other = sum(1 for g in self._regions if g['other'])
        txt = self._summary_base + (
            f'  ·  {n_miss} missing  ·  {len(self._shown) - n_miss} over'
            + (f'  ·  {n_other} on another signal' + (' (hidden)' if hide else '')
               if n_other else ''))
        self._summary.config(text=txt)
        self._fill_regions()
        self._draw()

    def _fill_regions(self):
        self._rtree.delete(*self._rtree.get_children())
        for i, g in enumerate(self._shown):
            self._rtree.insert('', tk.END, iid=str(i), values=(
                i + 1, 'missing' if g['kind'] == rp.MISSING else 'over',
                f"{g['lo']:.3f}–{g['hi']:.3f}", f"{g['apex_ppm']:.3f}",
                f"{g['snr']:.0f}",
                f"{100 * g['share']:.0f}%" if g['share'] is not None else '—',
                (('other signal of: ' + ', '.join(g['other'])) if g['other'] else
                 ('claimed by: ' + ', '.join(g['claimed'])) if g.get('claimed') else ''),
                g['best'] or ('—' if g['kind'] == rp.MISSING else '')),
                tags=(g['kind'],))
        self._rtree.tag_configure(rp.MISSING, foreground='#8e1b1b')
        self._rtree.tag_configure(rp.OVER, foreground='#1b3f8e')
        self._ctree.delete(*self._ctree.get_children())
        self._cands = []
        self._region_shown = None

    def _draw(self):
        ax = self.ax
        ax.clear()
        if self._out is None:
            self.canvas.draw_idle()
            return
        ppm, trace, out = self._ppm, self._trace, self._out
        o = np.argsort(ppm)
        x = ppm[o]
        sel = self._selected_region()
        if sel is not None:
            ax.axvspan(sel['lo'], sel['hi'], color=_MISSING_FILL if sel['kind'] == rp.MISSING
                       else _OVER_FILL, zorder=0, lw=0)
        for kind, col, mk in ((rp.MISSING, _RESIDUAL, 'v'), (rp.OVER, '#1b3f8e', '^')):
            xs = [g['apex_ppm'] for g in self._shown if g['kind'] == kind]
            if xs:
                ax.scatter(xs, [0.97] * len(xs), marker=mk, s=18, color=col, zorder=6,
                           transform=ax.get_xaxis_transform(), clip_on=False,
                           label='missing' if kind == rp.MISSING else 'over-claim')
        ax.plot(x, trace[o], color=_INK, lw=0.8, zorder=4, label='spectrum')
        ax.plot(x, out['residual'][o], color=_RESIDUAL, lw=0.7, zorder=5,
                label='residual')
        for c in out['curves']:
            col = (contributions.MUTED if c['method'] == 'unreported'
                   else self._palette.get(c['id'], contributions.MUTED))
            if c['method'] == 'secondary':
                # predicted by the analyte's own number, not fitted: lighter,
                # dotted
                ax.fill_between(c['x'], 0, -c['y'], color=col, lw=0, zorder=2, alpha=0.25)
                ax.plot(c['x'], -c['y'], color=col, lw=0.9, ls=':', zorder=3)
                continue
            data = c['method'] == 'integral'
            ax.fill_between(c['x'], 0, -c['y'], color=col, lw=0, zorder=2,
                            alpha=0.25 if data else contributions.FILL_ALPHA,
                            hatch='////' if data else None,
                            edgecolor=col if data else None)
        ax.plot(x, -out['resolved'][o], color='#555555', lw=0.6, zorder=3,
                label='resolved ↓ (hatched: integral · dotted: predicted other signals)')
        ax.axhline(0, color='#999999', lw=0.6, zorder=1)
        self._draw_fingerprints(sel)
        # outside the axes: inside it covers the spectrum
        # a strip under the axes: beside them it took a third of the width
        ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4,
                  frameon=False, handlelength=1.6, columnspacing=1.2)
        ax.set_xlabel('δ ¹H (ppm)')
        ax.set_ylabel('Intensity (a.u.)  —  ↑ spectrum   ↓ resolved')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._apply_view()

    def _field(self):
        try:
            return float(self.ann.spectrometer_var.get())
        except (AttributeError, tk.TclError, ValueError):
            return None

    def _draw_fingerprints(self, region):
        """The picked candidates fitted into the residual of the region: the
        largest copy of each reference shape, at the study's linewidth,
        that fits under the residual (``resolution_plot.fit_in_residual``),
        dashed in a colour of its own, with the part of the residual it
        accounts for; for one already quantified by a fit, dotted, what its
        own number says this signal holds (``expected_from_number``). A candidate without what that needs is shown as its
        reference fingerprint scaled to the residual's apex, and says so."""
        cands = self._selected_candidates()
        if region is None or not cands:
            return
        inside = (self._ppm >= region['lo']) & (self._ppm <= region['hi'])
        top = float(self._out['residual'][inside].max()) if inside.any() else 0.0
        if top <= 0:
            return
        field = self._field()
        lw = self._study_linewidth()
        for k, c in enumerate(cands[:len(contributions.SLOTS)]):
            col = contributions.SLOTS[k]
            head = f"{c['name'][:22]} {c['mult']} J {_hz(c['j_ref'])}"
            fit = rp.fit_in_residual(c, self._ppm, self._out['residual'], self._sigma,
                                     region, field, lw, self._tolerance())
            row = next((r for r in self._rows if str(r.get('hmdb_id')) == c['met_id']), None)
            exp = rp.expected_from_number(c, row, self._ppm, self._out['residual'],
                                          self._group_var.get(), region, field, lw,
                                          origin=fit['origin'] if fit else None)
            if exp is not None:
                self.ax.plot(exp['x'], exp['y'], color=col, lw=1.3, ls=':', zorder=7,
                             label=f"{c['name'][:22]}: its number -> "
                                   + (f"{100 * exp['ratio']:.0f}% of residual"
                                      if exp['ratio'] is not None else 'predicted'))
            if fit is not None:
                share = fit['share']
                self.ax.fill_between(fit['x'], 0, fit['y'], color=col, alpha=0.15, lw=0,
                                     zorder=6)
                self.ax.plot(fit['x'], fit['y'], color=col, lw=1.5, ls='--', zorder=7,
                             label=head + (f": fits {100 * share:.0f}%" if share is not None
                                           else '')
                             + f" ({1000 * fit['shift_ppm']:+.0f} mppm)")
                continue
            fp = rp.fingerprint(c, field)
            if fp is None:
                continue
            x, h, kind = fp
            label = head + ' (reference, not fitted)'
            if kind == 'profile':
                self.ax.plot(x, top * h, color=col, lw=1.2, ls=':', zorder=7, label=label)
            else:
                self.ax.vlines(x, 0, top * h, colors=col, lw=1.6, linestyles=':',
                               zorder=7, label=label)

    def _study_linewidth(self):
        """The study's linewidth as the annotation measured it (any matched
        signal carries it): for a candidate whose own signal has none."""
        for met in self._metabolites():
            for q in met.get('matched_peaks') or ():
                if q.get('study_linewidth_hz'):
                    return float(q['study_linewidth_hz'])
        return None

    def _apply_view(self):
        ax, ppm, trace = self.ax, self._ppm, self._trace
        if self._view is None:
            lo, hi = float(ppm.min()), float(ppm.max())
        else:
            lo, hi = self._view
        sel = (ppm >= lo) & (ppm <= hi) & ~rp.excluded_mask(ppm, self._excluded)
        top = float(np.max(trace[sel])) if sel.any() else 1.0
        bot = float(np.max(self._out['resolved'][sel])) if sel.any() else 1.0
        m = max(top, bot, 1e-12) * 1.1
        ax.set_xlim(hi, lo)                     # δ decreases to the right
        ax.set_ylim(-m, m)
        for t in list(ax.texts):
            t.remove()
        if self._view is not None:
            self._label_curves(lo, hi, m)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _label_curves(self, lo, hi, m):
        """In a zoomed view, each resolved analyte named under its apex."""
        here = []
        for c in self._out['curves']:
            inside = (c['x'] >= lo) & (c['x'] <= hi)
            if not inside.any() or c['y'][inside].max() <= 0.02 * m:
                continue
            k = np.argmax(np.where(inside, c['y'], -np.inf))
            here.append((c['y'][k], c['x'][k], c))
        here.sort(key=lambda t: -t[0])
        for _y, xk, c in here[:_MAX_LABELS]:
            col = self._palette.get(c['id'], contributions.MUTED)
            name = c['name'][:22] + (' (other signal)' if c['method'] == 'secondary' else '')
            self.ax.text(xk, -_y - 0.03 * m, name, rotation=90, fontsize=7,
                         ha='center', va='top', color=col, zorder=6)

    def _zoom(self, region):
        self._view = (None if region is None else
                      (region['lo'] - _ZOOM_PAD_PPM, region['hi'] + _ZOOM_PAD_PPM))
        if self._out is not None:
            self._apply_view()

    # ── Regions and candidates ────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or not self._shown:
            return
        if self.canvas.toolbar is not None and self.canvas.toolbar.mode:
            return                              # zooming or panning, not picking
        regs = self._shown
        i = min(range(len(regs)), key=lambda k: min(
            abs(event.xdata - regs[k]['lo']), abs(event.xdata - regs[k]['hi']),
            0.0 if regs[k]['lo'] <= event.xdata <= regs[k]['hi'] else 1e9))
        self._rtree.selection_set(str(i))
        self._rtree.see(str(i))

    def _on_region(self, _e=None, *, force=False):
        g = self._selected_region()
        if g is None:
            return
        # Tk can deliver one selection twice: the same region keeps its list
        # and the candidates picked on it
        if g is self._region_shown and not force:
            return
        self._region_shown = g
        self._draw()
        self._zoom(g)
        self._ctree.delete(*self._ctree.get_children())
        self._cands = (rp.candidates(g, self._metabolites(), self._rows, self._tolerance(),
                                     self._ppm) if g['kind'] == rp.MISSING else [])
        for i, c in enumerate(self._cands):
            row = next((r for r in self._rows if str(r.get('hmdb_id')) == c['met_id']), None)
            status = c['status']
            if status == 'quantified' and row is not None:
                status += f" ({contributions.method_name(row) or row.get('_integ_mode', '')})"
            taken = self._taken_here(c['met_id'], g)
            if taken is not None:
                share, why = taken
                status += (f" — this signal subtracted ({100 * share:.0f}% of what its number "
                           f"predicts; {why})")
            if c['detail']:
                status += f" — {c['detail']}"
            self._ctree.insert('', tk.END, iid=str(i), values=(
                c['name'], c['mult'], _hz(c['j_ref']), _hz(c['j_obs']), _hz(c['delta_j']),
                f"{c['lines'][0]}/{c['lines'][1]}" if c['lines'] else '—',
                f"{c['listed_ppm']:.3f}", f"{1000 * c['delta_ppm']:+.0f}",
                '' if c['protons'] is None else f"{c['protons']:g}",
                _QUALITY.get(c['quality'], str(c['quality'])) if c['mp_index'] is not None
                else 'no match', status))
        if g['kind'] == rp.OVER:
            self._note.config(text='An over-claim: the fits here hold more than the spectrum. '
                                   'Review it in Quantification > Review deconvolution.')
        elif not self._cands:
            self._note.config(text='No analyte of the annotation lists a signal here.')
        else:
            self._note.config(text='')
        self._update_buttons()

    def _taken_here(self, mid, region):
        """``(share, why)``: the part of what a confirmed analyte's number
        predicts that was subtracted on a secondary signal of it inside
        ``region``, and the confirmation that allowed it; None when none of
        its subtracted signals is there."""
        hits = [c for c in self._out['curves']
                if c['method'] == 'secondary' and c['id'] == mid and len(c['x'])
                and c['x'][0] <= region['hi'] and c['x'][-1] >= region['lo']]
        pred = sum(c['predicted'] for c in hits)
        if not hits or pred <= 0:
            return None
        return sum(c['taken'] for c in hits) / pred, '; '.join(c['why'] for c in hits)

    def _selected_region(self):
        sel = self._rtree.selection()
        if not sel or int(sel[0]) >= len(self._shown):
            return None
        return self._shown[int(sel[0])]

    def _selected_candidates(self):
        return [self._cands[int(k)] for k in self._ctree.selection()
                if int(k) < len(self._cands)]

    def _on_candidates(self, _e=None):
        self._update_buttons()
        if self._out is not None:
            self._draw()

    def _step(self, d):
        """The previous or next region of the table, in the order shown."""
        kids = list(self._rtree.get_children(''))
        if not kids:
            return
        sel = self._rtree.selection()
        i = kids.index(sel[0]) + d if sel and sel[0] in kids else 0
        k = kids[max(0, min(len(kids) - 1, i))]
        self._rtree.selection_set(k)
        self._rtree.see(k)

    def _update_buttons(self):
        cs = self._selected_candidates()
        ok = any(c['mp_index'] is not None for c in cs) and self._editor() is not None
        for b in (self._good_btn, self._clear_btn):
            b.config(state='normal' if ok else 'disabled')
        if cs and all(c['mp_index'] is None for c in cs):
            self._note.config(text='The annotation matched no peak to this signal: there is '
                                   'nothing to mark. Split or add the peak in the Signal List '
                                   'first.')

    def _editor(self):
        if self.ann is None or not hasattr(self.ann, '_quality_marks'):
            return None
        from controllers.signal_lines import SignalLineEditor
        return SignalLineEditor(lambda: self.ann._metabolites, lambda: self.ann._quality_marks)

    def _mark(self, quality):
        ed = self._editor()
        cs = [c for c in self._selected_candidates() if c['mp_index'] is not None]
        if ed is None or not cs:
            return
        for c in cs:
            ed.classify(c['met_id'], c['mp_index'], quality)
        self._marked += len(cs)
        sel = self._ctree.selection()
        self._on_region(force=True)
        keep = [k for k in sel if self._ctree.exists(k)]
        if keep:
            self._ctree.selection_set(keep)
        names = ', '.join(f"{c['name']} ({c['listed_ppm']:.3f} ppm)" for c in cs)
        self._note.config(text=(
            f"Marked {'Good' if quality else 'Average'}: {names}. The numbers change only "
            f"when the quantification runs again ({self._marked} change(s) waiting)."))
        self._run_btn.config(text=f'Re-run quantification ({self._marked})')

    def _rerun(self):
        if not messagebox.askyesno(
                'Re-run quantification',
                'Run the quantification again for every analyte, with the marks made '
                'so far? Numbers of other analytes may change.', parent=self.win):
            return
        self._summary.config(text='Quantifying…')

        def _done():
            self._marked = 0
            self._run_btn.config(text='Re-run quantification')
            self.refresh()
        self.qs._safe_run(on_done=_done, on_cancel=self.refresh)
