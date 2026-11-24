# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Where a window opens, and that a fixed-size one shows all it holds."""


def center_on(win, parent):
    """Place *win* centred across *parent*, a third of the way down."""
    win.update_idletasks()
    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - h) // 3
    win.geometry(f'+{max(x, 0)}+{max(y, 0)}')


def grow_to_fit(win):
    """Enlarge *win* to the size its content asks for, when that is larger
    than the size it was given; never shrink it, and keep it on the screen.

    Dialogs are sized in pixels measured on Windows. macOS draws the same
    widgets taller and wider, so a dialog that cannot be resized hid what did
    not fit: the Import and Cancel buttons of "Bruker Import Options" were
    below its edge on a Mac, and no spectrum could be imported (v0.89).
    Returns whether the window grew.
    """
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    nw = min(max(w, win.winfo_reqwidth()), sw)
    nh = min(max(h, win.winfo_reqheight()), sh)
    if (nw, nh) == (w, h):
        return False
    x = min(max(win.winfo_x(), 0), max(sw - nw, 0))
    y = min(max(win.winfo_y(), 0), max(sh - nh, 0))
    win.geometry(f'{nw}x{nh}+{x}+{y}')
    return True


def fit_fixed_windows(root):
    """Make every window that cannot be resized grow to fit its content when
    it first appears (see grow_to_fit). Windows the analyst can resize keep
    the size they were given."""

    def on_map(event):
        win = event.widget
        if getattr(win, '_nmrpro_fitted', False):
            return
        try:
            fixed = tuple(int(v) for v in win.resizable()) == (0, 0)
        except Exception:              # not a toplevel (or already gone)
            return
        win._nmrpro_fitted = True
        if fixed:
            win.after_idle(lambda: win.winfo_exists() and grow_to_fit(win))

    root.bind_class('Toplevel', '<Map>', on_map, add='+')
