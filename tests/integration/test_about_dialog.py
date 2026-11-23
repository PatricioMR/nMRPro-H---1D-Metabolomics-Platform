# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_about_dialog.py
"""Help → About, with a real Tk: the contact channels open where they say,
and the GNU AGPL notice is still shown."""
import pytest

tk = pytest.importorskip('tkinter')

from app_paths import (APP_VERSION, CONTACT_EMAIL, COPYRIGHT, ISSUES_URL,
                       LICENSE_TERMS, LINKEDIN_URL, SOURCE_URL)


@pytest.fixture(scope='module')
def root():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    r.geometry('200x200+-30000+-30000')   # mapped, so clicks and keys reach it
    r.update()
    yield r
    r.destroy()


@pytest.fixture
def dialog(root):
    from gui.widgets.about_dialog import AboutDialog
    d = AboutDialog(root)
    d.update()
    yield d
    if d.winfo_exists():
        d.destroy()


def _texts(widget):
    out = []
    for child in widget.winfo_children():
        try:
            out.append(str(child.cget('text')))
        except tk.TclError:
            pass
        out.extend(_texts(child))
    return out


def test_shows_version_copyright_licence_and_channels(dialog):
    shown = _texts(dialog)
    for expected in (APP_VERSION, COPYRIGHT, LICENSE_TERMS, CONTACT_EMAIL,
                     ISSUES_URL, SOURCE_URL, LINKEDIN_URL):
        assert expected in shown
    assert any('bug' in t.lower() and 'suggestion' in t.lower() for t in shown)
    assert any('collaborations are welcome' in t for t in shown)


def test_links_open_their_targets(dialog, monkeypatch):
    from gui.widgets import about_dialog
    opened = []
    monkeypatch.setattr(about_dialog.webbrowser, 'open', opened.append)
    clickable = []

    def walk(w):
        for c in w.winfo_children():
            if isinstance(c, tk.Label) and c.cget('cursor') == 'hand2':
                clickable.append(c)
            walk(c)
    walk(dialog)
    for label in clickable:
        label.event_generate('<Button-1>')
    assert opened == [f'mailto:{CONTACT_EMAIL}', ISSUES_URL, SOURCE_URL,
                      LINKEDIN_URL]


def test_escape_closes(dialog):
    # a key goes where the focus is: after the other windows of a full run
    # it is not this dialog unless it takes it
    dialog.focus_force()
    dialog.update()
    dialog.event_generate('<Escape>')
    dialog.update()
    assert not dialog.winfo_exists()
