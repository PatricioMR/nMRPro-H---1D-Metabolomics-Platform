# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Where a small dialog opens over its parent window."""


def center_on(win, parent):
    """Place *win* centred across *parent*, a third of the way down."""
    win.update_idletasks()
    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - h) // 3
    win.geometry(f'+{max(x, 0)}+{max(y, 0)}')
