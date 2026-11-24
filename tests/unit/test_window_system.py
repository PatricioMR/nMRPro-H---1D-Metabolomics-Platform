# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Window-manager options only reach the systems that have them."""
import pytest

from gui.utils.window_system import make_color_transparent, maximize


class _Tk:
    def __init__(self, system):
        self.system = system

    def call(self, *args):
        assert args == ('tk', 'windowingsystem')
        return self.system


class _Win:
    """Records calls; rejects options its system lacks, as Tk does."""

    _ATTRIBUTES = {
        'win32': {'-transparentcolor', '-topmost', '-alpha'},
        'aqua': {'-transparent', '-topmost', '-alpha'},
        'x11': {'-zoomed', '-topmost', '-alpha'},
    }
    _STATES = {
        'win32': {'normal', 'zoomed'},
        'aqua': {'normal', 'zoomed'},
        'x11': {'normal'},
    }

    def __init__(self, system):
        self.tk = _Tk(system)
        self.calls = []

    def attributes(self, name, value):
        if name not in self._ATTRIBUTES[self.tk.system]:
            raise RuntimeError(f'bad attribute "{name}"')
        self.calls.append(('attributes', name, value))

    def state(self, value):
        if value not in self._STATES[self.tk.system]:
            raise RuntimeError(f'bad argument "{value}"')
        self.calls.append(('state', value))


@pytest.mark.parametrize('system', ['win32', 'aqua', 'x11'])
def test_maximize_never_uses_an_option_the_system_lacks(system):
    win = _Win(system)
    maximize(win)
    assert len(win.calls) == 1


def test_maximize_uses_the_zoomed_attribute_on_x11():
    win = _Win('x11')
    maximize(win)
    assert win.calls == [('attributes', '-zoomed', True)]


def test_transparent_color_on_windows():
    win = _Win('win32')
    assert make_color_transparent(win, 'black') is True
    assert win.calls == [('attributes', '-transparentcolor', 'black')]


@pytest.mark.parametrize('system', ['aqua', 'x11'])
def test_transparent_color_is_skipped_where_it_does_not_exist(system):
    win = _Win(system)
    assert make_color_transparent(win, 'black') is False
    assert win.calls == []
