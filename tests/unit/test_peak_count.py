# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A line that moves between samples is one line in each sample. The merged
peak list keeps it once per position; the report must count what a sample
shows (ADR-089)."""
import numpy as np
import pytest

from processing.peak_count import per_sample_peak_count, per_sample_peak_counts
from processing.peak_detection import detect_peaks

PPM = np.linspace(10.0, 0.0, 20001)          # 0.5 mppm per point
HWHM = 0.0022                                # 2.6 Hz full width at 600 MHz
SETTINGS = dict(min_snr=6.0, min_distance=0.004)


def _line(center, height=1.0):
    return height / (1.0 + ((PPM - center) / HWHM) ** 2)


def _study(n_per_group=3, shift=0.012, blank=False):
    """Two groups; the 3.000 line stands 12 mppm higher in group B, the
    1.330 line and the water at 4.70 do not move."""
    rng = np.random.default_rng(7)
    cols, info = [], []
    for g, d in (('A', 0.0), ('B', shift)):
        for _ in range(n_per_group):
            y = _line(1.330) + _line(3.000 + d) + _line(4.700)
            cols.append(y + rng.normal(0, 0.01, PPM.size))
            info.append({'type': 'Sample', 'label': g})
    if blank:
        cols.append(rng.normal(0, 0.01, PPM.size) + 3 * _line(2.2))
        info.append({'type': 'BLK', 'label': 'BLK'})
    return np.column_stack(cols), info


def test_a_moving_line_counts_once_per_sample():
    X, info = _study()
    counts = per_sample_peak_counts(PPM, X, excluded_regions=[(4.5, 5.0)],
                                    samples_info=info, **SETTINGS)
    assert counts == [2] * 6
    assert per_sample_peak_count(PPM, X, excluded_regions=[(4.5, 5.0)],
                                 samples_info=info, **SETTINGS) == 2


def test_the_merged_list_holds_the_moving_line_twice():
    """Why the report cannot use len(peaks): detection keeps the moved line
    at both positions, each found in a whole group."""
    X, info = _study()
    merged = detect_peaks(PPM, X, noise_std=0.01, baseline_correct=False,
                          global_align=False, samples_info=info,
                          per_sample_detection=True, detect_shoulders=False,
                          min_group_presence=3, excluded_regions=[(4.5, 5.0)],
                          **SETTINGS)
    near3 = [p for p in merged if abs(p['ppm'] - 3.006) < 0.02]
    assert len(near3) == 2
    assert len(merged) > per_sample_peak_count(
        PPM, X, excluded_regions=[(4.5, 5.0)], samples_info=info, **SETTINGS)


def test_excluded_regions_and_blanks_are_left_out():
    X, info = _study(blank=True)
    with_water = per_sample_peak_counts(PPM, X, samples_info=info, **SETTINGS)
    assert len(with_water) == 6              # the blank is not a sample
    assert with_water == [3] * 6             # the water is a peak until excluded
    assert per_sample_peak_counts(PPM, X, excluded_regions=[(4.5, 5.0)],
                                  samples_info=info, **SETTINGS) == [2] * 6


def test_one_spectrum_is_one_sample():
    X, _info = _study()
    assert per_sample_peak_count(PPM, X[:, 0], **SETTINGS) == 3


class _Var:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class _Stage:
    def __init__(self, peaks):
        self.peaks = peaks
        self.min_snr_var = _Var(6.0)
        self.min_width_var = _Var(0.004)
        self.detect_shoulders_var = _Var(False)
        self.shoulder_k_var = _Var(3.0)
        self.excluded_regions_var = _Var('4.5-5.0')


class _App:
    def __init__(self, X, info):
        self.ppm, self._X, self.samples_info = PPM, X, info

    def corrected_intensities(self):
        return self._X


class TestReportCount:
    def test_the_report_counts_per_sample_not_the_merged_list(self):
        from gui.widgets.pdf_report import project_peak_count
        X, info = _study()
        quant = type('Q', (), {'peak_stage': _Stage([{}] * 7)})()
        assert project_peak_count(quant, _App(X, info)) == pytest.approx(2)

    def test_without_detection_there_is_no_count(self):
        from gui.widgets.pdf_report import project_peak_count
        X, info = _study()
        quant = type('Q', (), {'peak_stage': _Stage([])})()
        assert project_peak_count(quant, _App(X, info)) is None

    def test_a_blank_setting_gives_no_count_rather_than_a_guess(self):
        from gui.widgets.pdf_report import project_peak_count
        X, info = _study()
        stage = _Stage([{}] * 7)
        stage.min_snr_var = _Var('')
        quant = type('Q', (), {'peak_stage': stage})()
        assert project_peak_count(quant, _App(X, info)) is None
