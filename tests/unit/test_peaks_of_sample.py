# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A sample annotated on its own reads the study's peaks where that sample
holds them: only those detected in it, at its own apex, with its own share
of the height."""
import numpy as np

from validation.app_annotation import peaks_of_sample

PPM = np.linspace(10.0, 0.0, 10001)        # descending, as a spectrum is
STEP = float(PPM[1] - PPM[0])


def _peak(ppm, per, present, areas):
    i = int(np.argmin(np.abs(PPM - ppm)))
    return {'rank': 1, 'ppm': ppm, 'start_ppm': ppm + 0.01, 'end_ppm': ppm - 0.01,
            'index': i, 'start_idx': i - 10, 'end_idx': i + 10,
            'height': 100.0, 'snr': 50.0, 'area': float(np.mean(areas)),
            'ppm_per_sample': per, 'aligned_per_sample': present,
            'area_per_sample': np.asarray(areas, dtype=float)}


def test_a_peak_absent_from_the_sample_is_left_out():
    p = _peak(3.0, [3.0, 3.0], [True, False], [1.0, 1.0])
    assert len(peaks_of_sample([p], 0, PPM)) == 1
    assert peaks_of_sample([p], 1, PPM) == []


def test_the_peak_moves_to_the_samples_own_apex_window_and_all():
    p = _peak(3.0, [3.0, 3.002], [True, True], [1.0, 1.0])
    q = peaks_of_sample([p], 1, PPM)[0]
    assert q['ppm'] == 3.002
    assert np.isclose(q['start_ppm'] - q['end_ppm'], 0.02)
    assert q['index'] - p['index'] == round(0.002 / STEP)
    assert q['rank'] == p['rank']
    assert p['ppm'] == 3.0                          # the study's peak untouched


def test_height_and_snr_are_the_samples_share():
    p = _peak(3.0, [3.0, 3.0], [True, True], [1.0, 3.0])
    q0, = peaks_of_sample([p], 0, PPM)
    q1, = peaks_of_sample([p], 1, PPM)
    assert np.isclose(q0['height'], 50.0) and np.isclose(q1['height'], 150.0)
    assert np.isclose(q1['snr'], 75.0) and q1['area'] == 3.0
