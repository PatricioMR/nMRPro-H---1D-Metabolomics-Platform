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


def test_update_now_opens_the_release_page(host, monkeypatch):
    # Run from source, the program cannot update itself: the release page is
    # all that opens, and the window says to download from there.
    from gui.widgets import update_dialog
    opened = []
    monkeypatch.setattr(update_dialog.webbrowser, 'open', opened.append)
    _serve(monkeypatch, _NEWER)
    _run(host, manual=False)
    host.dialogs[0]._update_now()
    assert opened == ['https://example.org/release']
    texts = [w.cget('text') for w in host.dialogs[0]._progress_frame.winfo_children()
             if w.winfo_class() == 'TLabel']
    assert any('cannot update itself' in t for t in texts)


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


# ── updating from the notice ─────────────────────────────────────────────────

_SETUP = {'name': 'NMRPro-H-v99.0-Setup.exe', 'url': 'https://x/Setup.exe',
          'size': 1000, 'sha256': 'ab' * 32}
_INSTALLABLE = dict(_NEWER, assets=[_SETUP], mandatory=False)


def _dialog(root, release, monkeypatch, confirm=True, fail=None):
    from controllers import self_update
    from gui.widgets import update_dialog as ud
    seen = {'opened': [], 'launched': [], 'shutdown': 0}
    monkeypatch.setattr(ud.webbrowser, 'open', seen['opened'].append)
    monkeypatch.setattr(ud.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(self_update, 'current_target',
                        lambda: self_update.Target('windows', r'C:\P'))
    monkeypatch.setattr(self_update, 'launch', seen['launched'].append)

    def fake_download(url, dest, sha, progress=None, cancelled=None):
        for done in (250, 500, 1000):
            progress(done, 1000)
        if fail:
            raise ud.DownloadError(fail)
        return dest
    monkeypatch.setattr(ud, 'download_verified', fake_download)

    def shutdown():
        seen['shutdown'] += 1
    d = ud.UpdateDialog(root, release, '0.89', on_skip=None,
                        confirm_close=lambda: confirm, shutdown=shutdown)
    return d, seen


def _settle(root, until, timeout=5):
    deadline = time.time() + timeout
    while not until() and time.time() < deadline:
        root.update()
        time.sleep(0.02)


def _buttons(widget):
    out = []
    for w in widget.winfo_children():
        if w.winfo_class() == 'TButton':
            out.append(w.cget('text'))
        out.extend(_buttons(w))
    return out


def test_a_mandatory_update_is_announced_and_cannot_be_skipped(root, monkeypatch):
    d, _ = _dialog(root, dict(_INSTALLABLE, mandatory=True), monkeypatch)
    try:
        assert d.title() == 'Mandatory update'
        assert 'Skip this version' not in _buttons(d)
        assert 'Update now' in _buttons(d)
    finally:
        d.destroy()


def test_update_now_opens_the_page_downloads_installs_and_closes(root, monkeypatch):
    d, seen = _dialog(root, _INSTALLABLE, monkeypatch)
    try:
        d._update_now()
        _settle(root, lambda: seen['launched'])
        assert seen['opened'] == ['https://example.org/release']
        argv = seen['launched'][0]
        assert 'NMRPro-H-v99.0-Setup.exe' in argv[-1]
        assert f'Wait-Process -Id {__import__("os").getpid()}' in argv[-1]
        assert seen['shutdown'] == 1
        assert d._bar['value'] == 100
    finally:
        if d.winfo_exists():
            d.destroy()


def test_cancelling_the_save_question_postpones_the_install(root, monkeypatch):
    d, seen = _dialog(root, _INSTALLABLE, monkeypatch, confirm=False)
    try:
        d._update_now()
        _settle(root, lambda: d._action.cget('text') == 'Install')
        assert d._action.cget('text') == 'Install'
        assert seen['launched'] == [] and seen['shutdown'] == 0
    finally:
        d.destroy()


def test_a_failed_download_installs_nothing_and_says_why(root, monkeypatch):
    d, seen = _dialog(root, _INSTALLABLE, monkeypatch, fail='does not match')
    try:
        d._update_now()
        _settle(root, lambda: 'Close' in _buttons(d))
        texts = [w.cget('text') for w in d._progress_frame.winfo_children()
                 if w.winfo_class() == 'TLabel']
        assert any('does not match' in t for t in texts)
        assert seen['launched'] == [] and seen['shutdown'] == 0
    finally:
        d.destroy()
