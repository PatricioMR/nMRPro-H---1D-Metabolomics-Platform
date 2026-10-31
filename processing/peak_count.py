# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""How many peaks one sample of a study shows on its own.

Detection merges the peaks of every sample into one list, joining apices
closer than about the minimum separation. A line whose position moves
between samples by more than that -- the chemical environment of a real
extract changes from sample to sample -- is kept once per position, so the
merged list of a real study holds the same line two or three times. In the
simulated mixtures every sample shares one environment and the merged count
equals what each sample shows (ratio 0.99-1.03 for 50-200 analytes); in the
real studies it was 1.06-1.70 times larger (ADR-089).

What the report matches against its validation mixtures is therefore the
median over samples of the peaks each sample shows alone, found the way
per-sample detection finds them: each sample's own noise floor and the same
apex search, before any merging.
"""
from __future__ import annotations

import numpy as np

from processing.peak_detection import _detect_single, _quiet_segment_stats
from processing.sample_selection import included_sample_indices
from processing.spectral_regions import in_excluded_region


def per_sample_peak_counts(ppm, intensities, *, min_snr, min_distance,
                           excluded_regions=None, samples_info=None,
                           detect_shoulders=False,
                           shoulder_curvature_k=3.0) -> list[int]:
    """Peaks each included sample shows alone, outside the excluded regions.

    ``intensities`` is (points, samples), already baseline-corrected as
    detection reads it; blanks and excluded samples are skipped as detection
    skips them."""
    ppm = np.asarray(ppm, dtype=float)
    data = np.asarray(intensities, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    n_points = ppm.size
    # the same conversion detection makes, so the two count in one unit
    ppm_step = (np.max(ppm) - np.min(ppm)) / n_points
    distance = max(1, int(min_distance / ppm_step))
    counts = []
    for s in included_sample_indices(samples_info, data.shape[1]):
        col = data[:, s]
        med, sig = _quiet_segment_stats(col)
        idx = _detect_single(col, sig, med + sig * min_snr, min_snr, distance,
                             detect_shoulders=detect_shoulders,
                             shoulder_curvature_k=shoulder_curvature_k)
        counts.append(sum(1 for i in idx
                          if not in_excluded_region(ppm[i], excluded_regions)))
    return counts


def per_sample_peak_count(ppm, intensities, **kw) -> float | None:
    """The median of ``per_sample_peak_counts``; None without samples."""
    counts = per_sample_peak_counts(ppm, intensities, **kw)
    return float(np.median(counts)) if counts else None
