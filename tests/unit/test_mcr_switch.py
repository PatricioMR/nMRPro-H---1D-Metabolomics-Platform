# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""MCR in the annotation runs only in a release that ships deconvolution
(app_paths.DECONVOLUTION_ENABLED), whatever the checkbox or a saved project
says."""
import pytest

from gui.stages._annotation_search import mcr_requested


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Stage:
    def __init__(self, checked):
        self.mcr_run_var = _Var(checked)


@pytest.mark.parametrize('enabled, checked, runs', [
    (False, True, False),
    (False, False, False),
    (True, True, True),
    (True, False, False),
])
def test_mcr_runs_only_when_the_release_ships_it(monkeypatch, enabled, checked, runs):
    monkeypatch.setattr('app_paths.DECONVOLUTION_ENABLED', enabled)
    assert mcr_requested(_Stage(checked)) is runs


def test_a_stage_without_the_checkbox_runs_no_mcr(monkeypatch):
    monkeypatch.setattr('app_paths.DECONVOLUTION_ENABLED', True)
    assert mcr_requested(object()) is False


def test_this_release_ships_no_deconvolution():
    import app_paths
    assert app_paths.DECONVOLUTION_ENABLED is False
