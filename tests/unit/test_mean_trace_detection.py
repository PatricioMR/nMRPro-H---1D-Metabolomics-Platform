# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A resonance clear on the mean of the samples and too weak in each of them
is detected -- where nothing else is; a bump on a big peak's tail is not.

20 sigma on the mean of 18 samples is ~5 sigma in any one, under the
per-sample gate. On the virtual bench's tube of 100, 24 analytes had a clean
signal like that and no peak on it (glucose 5.23, xylose 5.19, cis-
aconitate). Detecting on the group mean everywhere made fragments on big
peaks' tails (ADR-047); only isolated apices are taken.
"""
import contextlib
import io

import numpy as np

from processing.peak_detection import (
    MEAN_MIN_SNR, _isolated_mean_apices, detect_peaks)

PPM = np.linspace(6.0, 4.0, 8001)
FWHM = 0.004
N = 18
SIGMA = 1.0


def _lor(c, h):
    g = FWHM / 2.0
    return h * g ** 2 / ((PPM - c) ** 2 + g ** 2)


def _study(weak_h):
    rng = np.random.default_rng(3)
    base = _lor(5.50, 400.0) + _lor(4.60, weak_h)
    X = np.column_stack([base * rng.lognormal(0, 0.1) + rng.normal(0, SIGMA, PPM.size)
                         for _ in range(N)])
    info = [{'index': i, 'name': f'S{i}', 'label': 'A' if i < N // 2 else 'B'}
            for i in range(N)]
    return X, info


def _peaks(X, info):
    with contextlib.redirect_stdout(io.StringIO()):
        det = detect_peaks(PPM, X, noise_std=None, min_snr=6.0,
                           min_distance=0.002, baseline_correct=False,
                           global_align=False, samples_info=info,
                           return_details=True, per_sample_detection=True,
                           detect_shoulders=False, min_group_presence=3)
    return [p['ppm'] for p in det['peaks']]


def test_a_line_seen_only_on_the_mean_is_detected():
    # ~5 sigma per sample, ~20 on the mean of 18
    X, info = _study(weak_h=5.0)
    found = _peaks(X, info)
    assert any(abs(p - 5.50) < 0.002 for p in found)
    assert any(abs(p - 4.60) < 0.002 for p in found)


def test_nothing_is_taken_near_a_detected_apex():
    y = _lor(5.50, 400.0) + _lor(5.50 + 1.5 * FWHM, 30.0)
    y = y + np.random.default_rng(0).normal(0, SIGMA, PPM.size) / np.sqrt(N)
    apex = int(np.argmin(np.abs(PPM - 5.50)))
    lw_pts = int(round(FWHM / abs(PPM[1] - PPM[0])))
    got = _isolated_mean_apices(y, [apex], lw_pts, 5)
    assert all(abs(PPM[i] - 5.50) > 3 * FWHM for i in got)


def test_the_bar_is_on_the_mean_trace_s_own_noise():
    y = np.random.default_rng(1).normal(0, 1.0, PPM.size)
    y = y + _lor(4.60, 0.5 * MEAN_MIN_SNR)            # under the bar
    assert _isolated_mean_apices(y, [], 10, 5) == []
    y = y + _lor(5.00, 3.0 * MEAN_MIN_SNR)            # over it, alone
    got = _isolated_mean_apices(y, [], 10, 5)
    assert any(abs(PPM[i] - 5.00) < 0.002 for i in got)
