# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/utils/verified_dot.py
"""The blue dot: an analyte the analyst verified.

The engine's dots (one to three) say how far the data identify an analyte.
The blue dot beside them says the analyst confirmed it -- a signal marked
Good in the Annotation Results, or lines saved in the STOCSY window -- on
an analyte the engine had quantified at any level, or on one that came in
only through the mark (ADR-107). The row carries it as
``_annotation_manual`` (``_quant_adapter.mark_analyst_confirmed``).

Text that takes colour (the report, the card) shows it as a blue ●. A
Treeview cannot colour part of a cell, nor show an image outside its first
column, so the tables write it as ◎, the first of the Quality dots: ◎●●○.
(◉ is taken: it marks the signal the quantification measured, in STOCSY.)
"""
VERIFIED_BLUE = '#1565c0'
VERIFIED_MARK = '◎'
VERIFIED_TEXT = 'verified by the analyst'
VERIFIED_TIP = (f'{VERIFIED_MARK} the analyst verified this analyte (a signal marked Good in '
                'the Annotation Results, or lines saved in the STOCSY window), whatever the '
                "number of the engine's dots.")


def is_verified(row) -> bool:
    return bool((row or {}).get('_annotation_manual'))


def with_mark(row, dots) -> str:
    """The Quality text of a table: ``dots`` with the analyst's mark first
    when the analyte was verified."""
    return f'{VERIFIED_MARK}{dots}' if is_verified(row) else str(dots)
