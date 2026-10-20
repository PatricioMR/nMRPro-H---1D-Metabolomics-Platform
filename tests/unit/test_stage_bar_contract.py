# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Structural contracts for the stage toolbar and the bottom button bar."""
import inspect
import re

from gui.app import App as NMRApp


def test_the_shipped_stages_are_numbered_one_to_five():
    src = inspect.getsource(NMRApp.create_stage_toolbar)
    numbered = re.findall(r'text="(\d)\. ', src)
    assert numbered == ['1', '2', '3', '4', '5']
    assert 'JG-Deconv' in src and '. 🔀 JG-Deconv' not in src, \
        'the parked stage carries no number'


def test_the_deconvolution_button_opens_the_quantification_review():
    """The bottom button used to open the MCR window and was enabled by
    detection, annotation and full-spectrum MCR alike. It now reviews what
    the quantification resolved by shape, and only once it has."""
    src = inspect.getsource(NMRApp.open_deconv_shortcut)
    assert '_open_deconvolution_window' in src
    assert 'MCRWindow' not in src
    assert 'quant_has_deconvolution' in inspect.getsource(
        NMRApp.refresh_deconv_button)


def test_only_the_quantification_enables_the_deconvolution_button():
    import gui.stages._annotation_search as srch
    import gui.stages.fullspectrum_mcr_stage as fs
    import gui.widgets.peak_list_window as pl
    import gui.stages._quant_computation as qc
    import gui.project_manager as pm
    for mod in (srch, fs, pl):
        assert 'btn_deconv_shortcut' not in inspect.getsource(mod), mod.__name__
    assert 'refresh_deconv_button' in inspect.getsource(qc)
    assert 'refresh_deconv_button' in inspect.getsource(pm)


def test_the_bottom_bar_follows_the_pipeline_order():
    """Signal List, Annotation Results, Deconvolution -- the order the
    analyst reaches them."""
    src = inspect.getsource(NMRApp.__init__)
    if 'btn_peak_list' not in src:
        src = inspect.getsource(NMRApp)
    order = [src.index(k) for k in ('self.btn_peak_list = ',
                                    'self.btn_annot_shortcut = ',
                                    'self.btn_deconv_shortcut = ')]
    assert order == sorted(order)
