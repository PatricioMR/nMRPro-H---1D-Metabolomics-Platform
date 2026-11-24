# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""A window that cannot be resized shows everything it holds.

v0.89 on a Mac: "Bruker Import Options" (360x270, not resizable) hid its
Import and Cancel buttons below its edge, so nothing could be imported.
"""
import tkinter as tk

import pytest

from gui.utils.window_place import fit_fixed_windows, grow_to_fit


@pytest.fixture(scope='module')
def root():
    # One interpreter for the module: creating Tk() again and again fails now
    # and then on Windows ("Can't find a usable tk.tcl"). The Bruker test,
    # which enlarges every widget, runs last.
    try:
        r = tk.Tk()
    except tk.TclError as exc:                    # no display
        pytest.skip(f'Tk unavailable: {exc}')
    r.withdraw()
    yield r
    r.destroy()


def _settle(win):
    for _ in range(20):
        win.update()


def _toplevel(root, size, content, resizable):
    win = tk.Toplevel(root)
    win.geometry(f'{size[0]}x{size[1]}+20+20')
    win.resizable(resizable, resizable)
    tk.Frame(win, width=content[0], height=content[1]).pack()
    return win


def test_a_window_too_small_for_its_content_grows_to_fit(root):
    win = _toplevel(root, (120, 60), (400, 300), resizable=False)
    _settle(win)
    assert grow_to_fit(win)
    _settle(win)
    assert win.winfo_width() >= 400 and win.winfo_height() >= 300


def test_a_window_larger_than_its_content_is_not_shrunk(root):
    win = _toplevel(root, (500, 400), (100, 80), resizable=False)
    _settle(win)
    assert not grow_to_fit(win)
    _settle(win)
    assert (win.winfo_width(), win.winfo_height()) == (500, 400)


def test_only_fixed_windows_grow_when_they_appear(root):
    fit_fixed_windows(root)
    fixed = _toplevel(root, (120, 60), (400, 300), resizable=False)
    free = _toplevel(root, (120, 60), (400, 300), resizable=True)
    _settle(root)
    assert fixed.winfo_height() >= 300
    assert free.winfo_height() == 60


def test_bruker_import_buttons_are_inside_the_dialog_with_larger_widgets(root):
    # Tk scaling 2x draws every font and widget about twice as large, as the
    # macOS theme does for the sizes measured on Windows.
    from gui._bruker_import_dialog import _BrukerImportDialog
    root.tk.call('tk', 'scaling', 2.0 * root.winfo_fpixels('1i') / 72.0)
    root.deiconify()
    root.geometry('800x600+0+0')
    fit_fixed_windows(root)
    seen = {}

    def check():
        dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
        dlg.update()
        buttons = [b for b in _descendants(dlg)
                   if b.winfo_class() in ('TButton', 'Button')
                   and b.cget('text') in ('Import', 'Cancel')]
        seen['buttons'] = len(buttons)
        bottom = dlg.winfo_rooty() + dlg.winfo_height()
        seen['inside'] = all(
            b.winfo_ismapped() and b.winfo_rooty() + b.winfo_height() <= bottom
            for b in buttons)
        dlg.destroy()

    root.after(800, check)
    _BrukerImportDialog(root)
    assert seen['buttons'] == 2
    assert seen['inside']


def _descendants(w):
    for c in w.winfo_children():
        yield c
        yield from _descendants(c)
