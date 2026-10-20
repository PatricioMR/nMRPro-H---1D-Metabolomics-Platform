# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Deconvolution is a future stage, and one switch says so.

Everything that resolves overlap between analytes stays in the code and
tested; what the release ships is the pipeline validated end to end. The
switch lives in app_paths so a build cannot half-enable it: every entry point
reads the same name.
"""
import ast
import pathlib

from app_paths import DECONVOLUTION_ENABLED

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_the_release_ships_with_deconvolution_off():
    assert DECONVOLUTION_ENABLED is False


def test_every_entry_point_reads_the_one_switch():
    """The sidebar button, the stage switch, the peak list's and the annotation
    window's context menus."""
    for rel, needle in (
        ('gui/app.py', 'if DECONVOLUTION_ENABLED:'),
        ('gui/app.py', "stage_name in ('j_resolve', 'fullspectrum_mcr')"),
        ('gui/widgets/peak_list_window.py', 'if DECONVOLUTION_ENABLED:'),
        ('gui/widgets/annotation_results_window.py',
         'if DECONVOLUTION_ENABLED:'),
    ):
        src = (ROOT / rel).read_text(encoding='utf-8')
        assert needle in src, f'{rel} does not honour the switch: {needle}'


def test_the_engine_still_offers_it():
    """Off in the UI is not gone from the engine: quantify(deconvolve=True)
    must keep working, or turning the switch back on would find nothing."""
    from processing.quant_engine import quantify
    import inspect
    assert 'deconvolve' in inspect.signature(quantify).parameters


def test_the_windows_balance_is_a_quantification_default_of_its_own():
    """The shared-window balance left the switch (ADR-075): it is a measured
    quantification default, on, with its box shown; the unfinished stages
    stay behind DECONVOLUTION_ENABLED."""
    from controllers.analysis_defaults import QUANTIFICATION
    assert QUANTIFICATION['split_shared'] is True
    for rel, needle in (
        ('gui/stages/quantification_stage.py',
         "tk.BooleanVar(value=QUANTIFICATION['split_shared'])"),
        ('gui/stages/_quant_computation.py', "QUANTIFICATION['split_shared']"),
        ('validation/fresh_run.py', "deconvolve=d['split_shared']"),
    ):
        src = (ROOT / rel).read_text(encoding='utf-8')
        assert needle in src, f'{rel} does not read the default: {needle}'

