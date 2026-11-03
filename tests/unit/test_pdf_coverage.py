# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The report's first page says how much of the spectrum the run resolved
(gui.widgets.pdf_report.coverage_text, processing.spectrum_coverage)."""
from gui.widgets.pdf_report import coverage_text


def test_it_quotes_the_peaks_and_the_area_the_model_explains():
    text = coverage_text({'peaks': 270, 'explained': 111, 'area_fraction': 0.62,
                          'n_orphans': 159, 'explained_share': 0.5,
                          'min_snr': 5.0})
    assert '<b>41%</b> of the detected peaks (111 of 270' in text
    assert '<b>62%</b> of the signal area' in text
    assert 'reaches 50% of its height' in text
    assert 'The 159 unexplained peak(s)' in text


def test_a_run_without_coverage_says_it_was_not_measured():
    assert 'not measured' in coverage_text(None)
    assert 'not measured' in coverage_text({'peaks': 'x'})
    assert 'no detected peak' in coverage_text(
        {'peaks': 0, 'explained': 0, 'area_fraction': float('nan')})
    full = coverage_text({'peaks': 10, 'explained': 10,
                          'area_fraction': float('nan'), 'n_orphans': 0})
    assert '<b>100%</b>' in full and 'signal area' not in full
    assert 'unexplained' not in full
