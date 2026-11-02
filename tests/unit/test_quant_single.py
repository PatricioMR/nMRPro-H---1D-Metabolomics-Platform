# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""One analyte quantified on the signal the analyst chose, with the run's own
conversion to concentration (processing.quant_single)."""
import numpy as np
import pytest

from processing.quant_single import quantify_chosen_signal, run_conversion

PPM = np.linspace(2.0, 4.0, 8001)
LW = 0.004


def _line(c, area):
    g = LW / 2
    return area * (g / np.pi) / ((PPM - c) ** 2 + g * g)


def _measure(lines_ppm, predicted, X, conversion, pqn=None):
    peaks = [{'rank': i + 1, 'ppm': p} for i, p in enumerate(lines_ppm)]
    return quantify_chosen_signal(
        peaks, {'protons': 2, 'predicted_ppms': predicted},
        ppm=PPM, intensities=X, included=list(range(X.shape[1])),
        local_baseline=False, align=False, blank_indices=None,
        linewidth_ppm=LW, other_peaks=peaks, noise_sigma=1.0,
        conversion=conversion, pqn_factors=pqn)


def test_the_run_conversion_is_read_off_its_rows():
    """conc = area / (nH x PQN) x k: every row of a sample carries the same
    k, a row without a number says nothing, and the median ignores one odd
    row."""
    rows = [{'_nh': 2, '_areas_per_sample': [10.0, 20.0],
             '_concs_per_sample': [1.0, 4.0], '_pqn_factors': {0: 1.0, 1: 2.0}},
            {'_nh': 1, '_areas_per_sample': [5.0, 8.0],
             '_concs_per_sample': [1.0, 3.2], '_pqn_factors': {'0': 1.0, '1': 2.0}},
            {'_nh': 1, '_areas_per_sample': [5.0, 8.0],
             '_concs_per_sample': [9.0, 9.0], '_not_quantified': True},
            {'_nh': 3, '_areas_per_sample': [1.0, 1.0],
             '_concs_per_sample': [99.0, 99.0], '_is_is': True}]
    k = run_conversion(rows, 3)
    assert k[0] == pytest.approx(0.2) and k[1] == pytest.approx(0.8)
    assert 2 not in k


def test_a_chosen_doublet_comes_back_at_its_area_per_proton():
    """Two lines of area 50 each: 100 for 2 protons, 50 per proton, times
    the conversion, over the PQN factor."""
    X = np.column_stack([_line(3.00, 50.0) + _line(3.012, 50.0),
                         _line(3.00, 80.0) + _line(3.012, 80.0)])
    got = _measure([3.00, 3.012], [3.00, 3.012], X, {0: 2.0, 1: 2.0},
                   pqn={0: 1.0, 1: 2.0})
    assert got['concs'][0] == pytest.approx(100.0, rel=0.03)
    assert got['concs'][1] == pytest.approx(80.0, rel=0.03)
    assert got['info']['window'][0] < 3.00 < 3.012 < got['info']['window'][1]


def test_a_signal_it_cannot_measure_says_why():
    X = np.column_stack([_line(3.00, 50.0)])
    with pytest.raises(ValueError, match='of the pattern'):
        # one line found of a quartet: a quarter of it, not scaled up
        _measure([3.00], [3.00, 3.012, 3.024, 3.036], X, {0: 1.0})
    with pytest.raises(ValueError, match='no detected line'):
        _measure([], [3.00], X, {0: 1.0})
    # no conversion for the sample: no concentration, said, not a zero
    with pytest.raises(ValueError, match='no positive area'):
        _measure([3.00], [3.00], X, {})
