# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""How much of the spectrum the final model explains
(processing.spectrum_coverage)."""
import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.spectrum_coverage import coverage, standard_weight

FIELD = 600.0
PPM = np.linspace(-0.5, 4.5, 50000)
FWHM = 2.6
LW = FWHM / FIELD


def _line(centre, height):
    return height * lines_model(PPM, [(0.0, 1.0)], centre, FWHM, FIELD)


def test_a_peak_the_model_reaches_is_explained_and_one_it_misses_is_an_orphan():
    trace = _line(1.0, 100.0) + _line(2.0, 50.0)
    model = _line(1.0, 100.0)
    cov = coverage(PPM, trace, model, [1.0, 2.0], noise_sigma=1.0)
    assert cov['peaks'] == 2 and cov['explained'] == 1
    assert cov['orphans'][0][0] == pytest.approx(2.0, abs=1e-3)
    assert cov['orphans'][0][1] == pytest.approx(50.0, rel=0.01)


def test_half_the_height_is_the_line():
    trace = _line(1.0, 100.0)
    assert coverage(PPM, trace, 0.55 * trace, [1.0], noise_sigma=1.0)['explained'] == 1
    assert coverage(PPM, trace, 0.45 * trace, [1.0], noise_sigma=1.0)['explained'] == 0


def test_a_model_above_the_trace_counts_only_what_the_trace_holds():
    trace = _line(1.0, 100.0)
    cov = coverage(PPM, trace, 3.0 * trace, [1.0], noise_sigma=1.0)
    assert cov['area_fraction'] == pytest.approx(1.0)


def test_peaks_below_the_floor_and_in_excluded_regions_are_not_asked():
    trace = _line(1.0, 100.0) + _line(2.0, 3.0) + _line(3.0, 80.0)
    w = np.where(np.abs(PPM - 3.0) < 0.05, 0.0, 1.0)
    cov = coverage(PPM, trace, np.zeros(PPM.size), [1.0, 2.0, 3.0],
                   noise_sigma=1.0, weight=w)
    assert cov['peaks'] == 1


def test_the_standard_and_its_carbon_satellites_are_left_out():
    w = standard_weight(PPM, 0.0, field_mhz=FIELD, linewidth_ppm=LW)
    sat = 59.0 / FIELD
    assert w[np.argmin(np.abs(PPM - 0.0))] == 0
    assert w[np.argmin(np.abs(PPM - sat))] == 0
    assert w[np.argmin(np.abs(PPM + sat))] == 0
    assert w[np.argmin(np.abs(PPM - 0.5))] == 1


def test_no_standard_leaves_everything_in():
    assert standard_weight(PPM, None, field_mhz=FIELD, linewidth_ppm=LW).min() == 1
