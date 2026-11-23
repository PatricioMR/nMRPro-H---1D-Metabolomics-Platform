# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_update_notice.py
"""The new-release notice, with a real Tk: when it opens, what its buttons
do, and that a startup check stays quiet when GitHub cannot be reached."""
import time

import pytest

tk = pytest.importorskip('tkinter')

from data import update_check as uc


@pytest.fixture(scope='module')
def root():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    r.withdraw()
    yield r
    r.destroy()


_NEWER = {'tag': 'v99.0-beta', 'name': 'NMRPro-H v99.0 Beta',
          'url': 'https://example.org/release', 'notes': 'Line one.\nLine two.'}
_SAME = {'tag': 'v0.0', 'name': 'NMRPro-H v0.0', 'url': 'u', 'notes': ''}


@pytest.fixture
def host(root, monkeypatch, tmp_path):
    from gui import _update_check as mod

    monkeypatch.setattr(uc, '_prefs_path', lambda: str(tmp_path / 'p.json'))
    told = []
    monkeypatch.setattr(mod.messagebox, 'showinfo',
                        lambda *a, **k: told.append(('info', a)))
    monkeypatch.setattr(mod.messagebox, 'showwarning',
                        lambda *a, **k: told.append(('warning', a)))
    dialogs = []
    real = mod.UpdateDialog

    def opened(*a, **k):
        dialogs.append(real(*a, **k))
        return dialogs[-1]

    monkeypatch.setattr(mod, 'UpdateDialog', opened)

    class Host(mod._UpdateCheckMixin):
        pass

    h = Host()
    h.root = root
    h.told, h.dialogs = told, dialogs
    yield h
    for d in dialogs:
        if d.winfo_exists():
            d.destroy()


def _serve(monkeypatch, release=None, error=None):
    def fetch():
        if error is not None:
            raise uc.UpdateCheckError(error)
        return release
    monkeypatch.setattr(uc, 'fetch_latest_release', fetch)


def _run(host, manual):
    host._start_update_check(manual=manual)
    deadline = time.time() + 5
    while host._update_check_running and time.time() < deadline:
        host.root.update()
        time.sleep(0.02)
    assert not host._update_check_running


def test_newer_release_opens_the_notice(host, monkeypatch):
    _serve(monkeypatch, _NEWER)
    _run(host, manual=False)
    assert len(host.dialogs) == 1 and host.told == []
    assert host.dialogs[0].title() == 'Update available'


def test_download_opens_the_release_page(host, monkeypatch):
    from gui.widgets import update_dialog
    opened = []
    monkeypatch.setattr(update_dialog.webbrowser, 'open', opened.append)
    _serve(monkeypatch, _NEWER)
    _run(host, manual=False)
    host.dialogs[0]._download()
    assert opened == ['https://example.org/release']
    assert not host.dialogs[0].winfo_exists()


def test_skipped_release_is_not_shown_again_at_startup(host, monkeypatch):
    _serve(monkeypatch, _NEWER)
    _run(host, manual=False)
    host.dialogs[0]._skip()
    assert uc.load_skipped_tag() == 'v99.0-beta'

    _run(host, manual=False)
    assert len(host.dialogs) == 1                 # quiet at startup
    _run(host, manual=True)
    assert len(host.dialogs) == 2                 # asked for: shown anyway


def test_offline_startup_is_quiet_manual_check_warns(host, monkeypatch):
    _serve(monkeypatch, error='offline')
    _run(host, manual=False)
    assert host.told == [] and host.dialogs == []
    _run(host, manual=True)
    assert [kind for kind, _ in host.told] == ['warning']


def test_up_to_date_is_said_only_when_asked(host, monkeypatch):
    _serve(monkeypatch, _SAME)
    _run(host, manual=False)
    assert host.told == []
    _run(host, manual=True)
    assert [kind for kind, _ in host.told] == ['info']


def test_startup_check_can_be_switched_off(host, monkeypatch):
    scheduled = []
    monkeypatch.setattr(host.root, 'after', lambda *a: scheduled.append(a),
                        raising=False)
    monkeypatch.setenv(uc.DISABLE_ENV, '1')
    host._schedule_startup_update_check()
    assert scheduled == []
    monkeypatch.delenv(uc.DISABLE_ENV)
    host._schedule_startup_update_check()
    assert len(scheduled) == 1
