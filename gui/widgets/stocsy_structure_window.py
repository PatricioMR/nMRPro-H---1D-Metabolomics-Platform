# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/widgets/stocsy_structure_window.py
"""
Modules and samples -- how the study's signals and samples are organised.

The STOCSY window asks what moves with one signal. This one asks the whole
matrix, on the same correlation (Pearson/Spearman, PQN, within groups) the
STOCSY window is set to:

  Modules  blocks of signals that move together (CLASSY: Robinette et al.,
           reviewed in Anal Chem 2013;85:5297), with what each block does
           between the groups. A block that is a molecule's own signals, a pool
           of molecules that follow a per-sample factor, a reference or
           water-tail artefact, or a group signature each reads differently.
  Axes     the main axes of variation of the samples, their group tests and
           the signals that load on them.
  Samples  each sample's PQN factor, its scores and how far it stands from
           the rest.

Read-only: nothing here changes the project or the annotation.
"""
import tkinter as tk
from tkinter import ttk

import numpy as np

from processing.signal_structure import STRUCTURE_MIN_SNR


def _p(v):
    return '—' if v is None or not np.isfinite(v) else f'{v:.2g}'


class StocsyStructureWindow:
    """Toplevel over a ``StocsyAnalysis``. ``on_send(ranks)`` is called with
    the members of the chosen module when the analyst sends them to the STOCSY
    window."""

    def __init__(self, parent, analysis, *, on_send=None):
        self.an = analysis
        self.on_send = on_send
        self._modules = []
        self.win = tk.Toplevel(parent)
        self.win.title('STOCSY — modules and samples')
        self.win.geometry('1180x720')
        self._build()
        self.refresh()
        self.win.protocol('WM_DELETE_WINDOW', self.win.destroy)

    # ── layout ───────────────────────────────────────────────────────────────

    def _build(self):
        top = ttk.Frame(self.win, padding=6)
        top.pack(fill=tk.X)
        self.basis_lbl = ttk.Label(top, text='', font=('Arial', 9, 'bold'))
        self.basis_lbl.pack(side=tk.LEFT)
        self.cut_var = tk.DoubleVar(value=0.8)
        self.snr_var = tk.DoubleVar(value=STRUCTURE_MIN_SNR)
        ttk.Button(top, text='Recompute', command=self.refresh).pack(
            side=tk.RIGHT, padx=4)
        ttk.Spinbox(top, from_=0.0, to=500.0, increment=5, width=5,
                    textvariable=self.snr_var).pack(side=tk.RIGHT)
        ttk.Label(top, text='min SNR').pack(side=tk.RIGHT, padx=(8, 2))
        ttk.Spinbox(top, from_=0.5, to=0.99, increment=0.05, width=5,
                    textvariable=self.cut_var).pack(side=tk.RIGHT)
        ttk.Label(top, text='block r ≥').pack(side=tk.RIGHT, padx=(8, 2))
        self.notes_lbl = tk.Label(self.win, text='', justify=tk.LEFT,
                                  wraplength=1120, fg='#8a5a00',
                                  font=('Arial', 8), anchor='w')
        self.notes_lbl.pack(fill=tk.X, padx=8)

        self.nb = ttk.Notebook(self.win)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_modules()
        self._build_axes()
        self._build_samples()

    def _tree(self, parent, cols, height=12):
        fr = ttk.Frame(parent)
        tree = ttk.Treeview(fr, columns=[c[0] for c in cols], show='headings',
                            height=height)
        for key, title, width in cols:
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor=tk.CENTER)
        sb = ttk.Scrollbar(fr, command=tree.yview)
        tree.config(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        return fr, tree

    def _build_modules(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text='Signal modules')
        cols = [('id', '#', 36), ('n', 'signals', 56), ('r', 'mean r', 60),
                ('pc', 'PC1 %', 52), ('pg', 'p groups', 72),
                ('pa', 'p factor A', 72), ('pb', 'p factor B', 72),
                ('pab', 'p A×B', 66), ('hi', 'highest in', 120),
                ('kind', 'reads as', 90), ('ppm', 'strongest δ', 170),
                ('names', 'annotated as', 280)]
        fr, self.mod_tree = self._tree(tab, cols, height=11)
        fr.pack(fill=tk.BOTH, expand=True)
        self.mod_tree.tag_configure('artefact', foreground='#a33')
        self.mod_tree.tag_configure('group', background='#fff3d6')
        self.mod_tree.bind('<<TreeviewSelect>>', lambda _e: self._on_module())
        bot = ttk.Frame(tab)
        bot.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        ttk.Label(bot, text='Members of the selected module').pack(anchor=tk.W)
        fr2, self.mem_tree = self._tree(bot, [
            ('rank', 'signal', 70), ('ppm', 'δ (ppm)', 90), ('snr', 'SNR', 80),
            ('name', 'annotated as', 260)], height=8)
        fr2.pack(fill=tk.BOTH, expand=True)
        self.means_lbl = ttk.Label(bot, text='', font=('Arial', 8))
        self.means_lbl.pack(anchor=tk.W, pady=2)
        ttk.Button(bot, text='Send these signals to the STOCSY window',
                   command=self._send).pack(anchor=tk.W)

    def _build_axes(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text='Sample axes')
        cols = [('id', 'axis', 50), ('var', 'variance %', 80),
                ('pg', 'p groups', 72), ('pa', 'p factor A', 72),
                ('pb', 'p factor B', 72), ('pab', 'p A×B', 66),
                ('rp', 'r with PQN factor', 110), ('means', 'score by group', 330),
                ('load', 'biggest loadings (δ)', 360)]
        fr, self.axis_tree = self._tree(tab, cols, height=14)
        fr.pack(fill=tk.BOTH, expand=True)
        ttk.Label(tab, text='Each axis is a direction in which the samples '
                  'differ over the signals standing over the SNR above. The '
                  'sign of an axis is arbitrary.', font=('Arial', 8)
                  ).pack(anchor=tk.W, pady=4)

    def _build_samples(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text='Samples')
        self.dil_lbl = ttk.Label(tab, text='', font=('Arial', 9))
        self.dil_lbl.pack(anchor=tk.W, pady=(0, 4))
        cols = [('name', 'sample', 150), ('group', 'group', 130),
                ('f', 'PQN factor', 80), ('s1', 'axis 1', 70),
                ('s2', 'axis 2', 70), ('s3', 'axis 3', 70),
                ('d', 'distance', 80)]
        fr, self.sam_tree = self._tree(tab, cols, height=18)
        fr.pack(fill=tk.BOTH, expand=True)
        ttk.Label(tab, text='Distance: how many standard deviations the '
                  'sample stands from the others, summed over the axes. A '
                  'screening number, not a test.', font=('Arial', 8)
                  ).pack(anchor=tk.W, pady=4)

    # ── content ──────────────────────────────────────────────────────────────

    def refresh(self):
        b = self.an.basis
        self.basis_lbl.config(
            text=f'{b.method}, '
                 f'{"PQN" if b.factors is not None else "raw areas"}'
                 f'{", within groups" if b.within_groups else ""}, '
                 f'n = {b.n} samples  (set in the STOCSY window)')
        self.notes_lbl.config(text='\n'.join('• ' + n for n in self.an.notes()))
        try:
            cut = float(self.cut_var.get())
            snr = float(self.snr_var.get())
        except (ValueError, tk.TclError):
            cut, snr = 0.8, STRUCTURE_MIN_SNR
        self._fill_modules(cut, snr)
        self._fill_axes(snr)
        self._fill_samples(snr)

    def _fill_modules(self, cut, snr):
        self._modules = self.an.modules(r_cut=cut, min_snr=snr)
        self.mod_tree.delete(*self.mod_tree.get_children())
        for m in self._modules:
            means = m['means']
            hi = max(means, key=means.get) if means else '—'
            p = m['p']
            tags = []
            if m['artefact']:
                tags.append('artefact')
            elif np.isfinite(p.get('group', np.nan)) and p['group'] < 0.01:
                tags.append('group')
            names = ', '.join(f'{k} ×{n}' if n > 1 else k for k, n in
                              sorted(m['names'].items(), key=lambda t: -t[1])[:3])
            self.mod_tree.insert(
                '', tk.END, iid=str(m['id']), tags=tags,
                values=(m['id'], m['n'], f'{m["mean_r"]:.2f}',
                        f'{100 * m["share"]:.0f}', _p(p.get('group')),
                        _p(p.get('A')), _p(p.get('B')), _p(p.get('AxB')), hi,
                        m['artefact'] or ('group signal' if 'group' in tags
                                          else 'sample factor / molecule'),
                        ', '.join(f'{self.an.ppm_of(r):.2f}' for r in m['ranks'][:5]),
                        names or '—'))
        self.mem_tree.delete(*self.mem_tree.get_children())
        self.means_lbl.config(text='')

    def _fill_axes(self, snr):
        self.axis_tree.delete(*self.axis_tree.get_children())
        for c in self.an.components(4, snr):
            p = c['p']
            self.axis_tree.insert('', tk.END, values=(
                f'PC{c["id"]}', f'{100 * c["variance"]:.0f}', _p(p.get('group')),
                _p(p.get('A')), _p(p.get('B')), _p(p.get('AxB')),
                '—' if not np.isfinite(c['r_pqn']) else f'{c["r_pqn"]:+.2f}',
                '  '.join(f'{k[:12]} {v:+.1f}' for k, v in c['means'].items()),
                ', '.join(f'{ppm:.2f} ({ld:+.2f})' for _r, ppm, ld in c['loadings'][:6])))

    def _fill_samples(self, snr):
        self.sam_tree.delete(*self.sam_tree.get_children())
        div = self.an.dilution_by_group()
        self.dil_lbl.config(
            text=('Mean PQN factor by group: ' + ', '.join(
                f'{g} {v:.2f}' for g, v in div['by_group'].items())
                + f'  (highest / lowest = {div["ratio"]:.2f}'
                + (' — PQN is taking a group difference out)' if div['diverges']
                   else ')')) if div else '')
        rows = sorted(self.an.samples(3, snr), key=lambda r: -r['distance'])
        for r in rows:
            sc = (r['scores'] + [float('nan')] * 3)[:3]
            self.sam_tree.insert('', tk.END, values=(
                r['name'], r['group'], f'{r["pqn_factor"]:.2f}',
                *[('—' if not np.isfinite(x) else f'{x:+.1f}') for x in sc],
                f'{r["distance"]:.1f}'))

    # ── interaction ──────────────────────────────────────────────────────────

    def _selected_module(self):
        sel = self.mod_tree.selection()
        return next((m for m in self._modules if str(m['id']) == sel[0]), None) \
            if sel else None

    def _on_module(self):
        m = self._selected_module()
        self.mem_tree.delete(*self.mem_tree.get_children())
        if not m:
            return
        for r in m['ranks']:
            pk = self.an.peaks[r]
            self.mem_tree.insert('', tk.END, values=(
                f'#{r}', f'{self.an.ppm_of(r):.4f}', f'{float(pk.get("snr") or 0):.0f}',
                pk.get('assigned_metabolite') or '—'))
        self.means_lbl.config(text='Module score by group: ' + '   '.join(
            f'{k} {v:+.2f}' for k, v in m['means'].items()))

    def _send(self):
        m = self._selected_module()
        if m and self.on_send:
            self.on_send(list(m['ranks']))
