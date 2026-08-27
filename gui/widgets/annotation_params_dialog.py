# gui/widgets/annotation_params_dialog.py
"""The Annotation Parameters dialog — one implementation, two entry points.

It is reachable from the menu before annotation runs and from the results window
afterwards. Those were two separate dialogs carrying their own copies of the
parameter list, and they had already drifted: a threshold removed from one
lingered in the other. The list itself now lives in
``processing.annotation_scoring.PARAMETER_SPEC`` — the same module that owns the
defaults and the scoring functions that read them.

'Tolerance (ppm)' is deliberately absent: it is entered on the Annotation panel
and owns both the search window and the Δppm gate.
"""
import tkinter as tk
from tkinter import ttk, messagebox

from processing.annotation_scoring import (
    PARAMETER_SPEC, SCORING_DEFAULTS, migrate_params)


def current_params(stored=None) -> dict:
    """Defaults overlaid with *stored*, migrated off any retired keys."""
    params = dict(SCORING_DEFAULTS)
    params.update(migrate_params(stored or {}))
    return params


def open_annotation_params(parent, stored, on_apply, *,
                           tier_builder=None, tier_reader=None,
                           tier_note='', title='Annotation Parameters'):
    """Show the dialog. ``on_apply(new_params)`` receives the edited values.

    ``tier_builder``/``tier_reader`` add the signal-grade chooser when the caller
    can supply one; the parameter rows work without it.
    """
    current = current_params(stored)

    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.transient(parent)
    dlg.resizable(False, False)

    nb = ttk.Notebook(dlg)
    nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    groups: dict[str, list] = {}
    for group, label, key, helptext in PARAMETER_SPEC:
        groups.setdefault(group, []).append((label, key, helptext))

    entries: dict[str, tk.StringVar] = {}
    for group, rows in groups.items():
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text=group)
        for r, (label, key, helptext) in enumerate(rows):
            ttk.Label(frm, text=label, width=26).grid(
                row=r, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=str(current.get(key, '')))
            entries[key] = var
            ttk.Entry(frm, textvariable=var, width=10).grid(
                row=r, column=1, sticky='w', padx=(4, 10))
            if helptext:
                ttk.Label(frm, text=helptext, foreground='#555',
                          wraplength=420, justify='left').grid(
                    row=r, column=2, sticky='w')

    tier_vars = None
    if tier_builder is not None:
        frm = ttk.Frame(nb, padding=10)
        nb.add(frm, text='Signal Quality')
        ttk.Label(frm, text='Grades allowed to be annotated:',
                  font=('Arial', 9, 'bold')).grid(row=0, column=0,
                                                  sticky='w', pady=(0, 6))
        hold = ttk.Frame(frm)
        hold.grid(row=1, column=0, sticky='w')
        tier_vars = tier_builder(hold, current.get('annotatable_tiers'),
                                 note=tier_note)

    bf = ttk.Frame(dlg)
    bf.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _apply():
        new_p, errs = {}, []
        for key, var in entries.items():
            raw = var.get().strip()
            try:
                new_p[key] = float(raw)
            except ValueError:
                errs.append(f'{key}: "{raw}" is not a number')
        if tier_vars is not None and tier_reader is not None:
            tiers = tier_reader(tier_vars)
            if not tiers:
                messagebox.showerror(
                    'No grade selected',
                    'Annotation needs at least one signal grade to work with.',
                    parent=dlg)
                return
            new_p['annotatable_tiers'] = tiers
        if errs:
            messagebox.showerror('Invalid value', '\n'.join(errs), parent=dlg)
            return
        on_apply(new_p)
        dlg.destroy()

    def _reset():
        for key, var in entries.items():
            var.set(str(SCORING_DEFAULTS.get(key, '')))

    ttk.Button(bf, text='Reset defaults', command=_reset).pack(side=tk.LEFT)
    ttk.Button(bf, text='Cancel', command=dlg.destroy).pack(side=tk.RIGHT)
    ttk.Button(bf, text='Apply', command=_apply).pack(side=tk.RIGHT, padx=6)
    return dlg
