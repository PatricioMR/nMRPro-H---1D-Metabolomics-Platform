# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/utils/quality_palette.py
"""One quality vocabulary and colour palette for every window.

The peak list grades a *signal* (can this integral be trusted?) and the
annotation window grades a *match* (is this the right metabolite?). They are
different judgements, but the analyst reads them as one scale, so both use the
same three tiers and the same colours:

    Good     green   — confirmed
    Average  yellow  — plausible, needs the analyst's eye
    Poor     red     — rejected

The annotation pipeline stores its verdict as ``annotation_quality`` with the
internal values ``'good'`` / ``'bad'`` / ``None``. Those keys are persisted in
saved projects and are deliberately left untouched; only the presentation maps
onto the shared tiers.
"""
from processing.signal_quality import (
    SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR, SIGNAL_TIERS)

__all__ = [
    'SIGNAL_GOOD', 'SIGNAL_AVERAGE', 'SIGNAL_POOR', 'SIGNAL_TIERS',
    'QUALITY_BG', 'QUALITY_FG', 'QUALITY_LABEL',
    'ANNOTATION_QUALITY_TIER', 'tier_from_annotation_quality',
]

# Row/checkbox background per tier.
QUALITY_BG = {
    SIGNAL_GOOD:    '#c8f7c5',   # light green
    SIGNAL_AVERAGE: '#fef9c3',   # light yellow
    SIGNAL_POOR:    '#ffd6d6',   # light red
}

# Foreground that stays legible on the matching background.
QUALITY_FG = {
    SIGNAL_GOOD:    '#1a7a1a',
    SIGNAL_AVERAGE: '#7a6000',
    SIGNAL_POOR:    '#9a0000',
}

# Glyph + name used on buttons and filters.
QUALITY_LABEL = {
    SIGNAL_GOOD:    '✓ Good',
    SIGNAL_AVERAGE: '~ Average',
    SIGNAL_POOR:    '✗ Poor',
}

# Internal annotation verdict → displayed tier. None means the pipeline reached
# no verdict, which is exactly what Average communicates: plausible, unconfirmed.
ANNOTATION_QUALITY_TIER = {
    'good': SIGNAL_GOOD,
    'bad':  SIGNAL_POOR,
    None:   SIGNAL_AVERAGE,
}


def tier_from_annotation_quality(quality) -> str:
    """Tier to display for an ``annotation_quality`` value.

    Unknown values fall back to Average rather than raising: an unexpected
    verdict must still be visible in the tree, flagged as unconfirmed.
    """
    return ANNOTATION_QUALITY_TIER.get(quality, SIGNAL_AVERAGE)


def build_tier_filter_row(parent, tier_vars, command, label='Show:'):
    """Build the three-tier filter checkbox row.

    Shared by the Peak List and the annotation window so the two filters are
    the same control, not two look-alikes that drift apart.

    Args:
        parent:     container frame; the widgets pack themselves LEFT into it.
        tier_vars:  {tier: BooleanVar} for the three tiers.
        command:    callback fired when any checkbox is toggled.
        label:      leading text.

    Returns:
        {tier: Checkbutton} so the caller can append live counts.
    """
    import tkinter as tk
    from tkinter import ttk

    ttk.Label(parent, text=label,
              font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 4))
    boxes = {}
    for tier in SIGNAL_TIERS:
        bg = QUALITY_BG[tier]
        cb = tk.Checkbutton(parent, text=QUALITY_LABEL[tier],
                            variable=tier_vars[tier],
                            bg=bg, fg=QUALITY_FG[tier], selectcolor=bg,
                            activebackground=bg, font=('Arial', 8),
                            command=command)
        cb.pack(side=tk.LEFT, padx=2)
        boxes[tier] = cb
    return boxes


def update_tier_counts(boxes, counts):
    """Show how many rows each tier holds, on the filter labels themselves."""
    for tier, cb in (boxes or {}).items():
        try:
            cb.config(text=f'{QUALITY_LABEL[tier]} ({counts.get(tier, 0)})')
        except Exception:
            pass


def build_tier_choice(parent, selected, note=None):
    """Checkbox column for choosing which quality grades a step may use.

    Shares the filter row's vocabulary and colours so "Good" means the same
    thing when filtering a view and when restricting what gets annotated.

    Args:
        parent:   container frame; widgets are gridded into it.
        selected: iterable of tier names to start checked.
        note:     optional explanatory line under the boxes.

    Returns:
        {tier: BooleanVar} — read after the dialog closes.
    """
    import tkinter as tk
    from tkinter import ttk

    # A missing or empty selection means "use the default", never "nothing
    # ticked" — an all-blank dialog reads as a broken control and, if applied,
    # would be rejected as an empty selection.
    from processing.signal_quality import normalize_tiers
    chosen = {t.lower() for t in normalize_tiers(selected)}
    variables = {}
    for row, tier in enumerate(SIGNAL_TIERS):
        var = tk.BooleanVar(value=tier.lower() in chosen)
        bg = QUALITY_BG[tier]
        tk.Checkbutton(parent, text=QUALITY_LABEL[tier], variable=var,
                       bg=bg, fg=QUALITY_FG[tier], selectcolor=bg,
                       activebackground=bg, anchor='w',
                       font=('Arial', 9)).grid(
            row=row, column=0, sticky='we', pady=2)
        variables[tier] = var
    if note:
        ttk.Label(parent, text=note, foreground='gray',
                  font=('Arial', 8), wraplength=320, justify='left').grid(
            row=len(SIGNAL_TIERS), column=0, sticky='w', pady=(8, 0))
    return variables


def read_tier_choice(variables):
    """Tiers ticked in a build_tier_choice block, in canonical order."""
    return [t for t in SIGNAL_TIERS if variables[t].get()]
