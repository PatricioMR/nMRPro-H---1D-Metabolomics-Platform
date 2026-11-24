# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Window-manager options that exist on one windowing system only.

Tk raises TclError on an option the running system lacks: the splash's
``-transparentcolor`` (Windows only) closed the macOS and Linux builds before
the first window, and ``state('zoomed')`` does not exist on X11. Every such
option goes through here, keyed by ``tk windowingsystem`` (win32, aqua, x11).
"""


def windowing_system(win):
    return str(win.tk.call('tk', 'windowingsystem'))


def make_color_transparent(win, color):
    """Show *color* as see-through where the system can (Windows only).

    Elsewhere the window keeps *color* as an opaque background. Returns
    whether the colour was made transparent.
    """
    if windowing_system(win) != 'win32':
        return False
    win.attributes('-transparentcolor', color)
    return True


def maximize(win):
    """Maximize *win*: ``zoomed`` state on Windows and macOS, the ``-zoomed``
    attribute on X11."""
    if windowing_system(win) == 'x11':
        win.attributes('-zoomed', True)
    else:
        win.state('zoomed')
