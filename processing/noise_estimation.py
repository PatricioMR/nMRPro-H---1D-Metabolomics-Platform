# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/noise_estimation.py
import numpy as np


def estimate_noise_from_region(ppm, intensities, region_start, region_end):
    """
    Estimate noise standard deviation from a specified ppm region.

    Args:
        ppm: array of ppm values
        intensities: 2D array (points x samples) or 1D array
        region_start: start of noise region (higher ppm value)
        region_end: end of noise region (lower ppm value)

    Returns:
        float: standard deviation of the noise (to be used in peak detection)
    """
    # Find indices in the region
    mask = (ppm <= region_start) & (ppm >= region_end)

    if not np.any(mask):
        raise ValueError(f"No data points found in region {region_start}-{region_end} ppm")

    noise_data = intensities[mask]

    if noise_data.ndim == 1 or noise_data.shape[1] == 1:
        return float(np.std(noise_data))

    # Per-sample std, then MEDIAN across samples.
    # Robust to groups with different noise levels:
    # one noisy group does not inflate the threshold for all.
    per_sample_std = np.std(noise_data, axis=0)
    noise_std = float(np.median(per_sample_std))

    print(f'   Noise per sample: '
          f'min={per_sample_std.min():.5f}  '
          f'max={per_sample_std.max():.5f}  '
          f'median={noise_std:.5f}')

    return noise_std