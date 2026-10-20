# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# controllers/onion_controller.py
"""
OnionController — placeholder for NMR-Onion integration.

NMR-Onion is a deep-learning deconvolution model.
This stub keeps the import chain intact while the integration
is not yet implemented; swap the body of deconvolve_region()
when the real model is available.
"""
import numpy as np


class OnionController:
    """Wraps NMR-Onion peak deconvolution (stub)."""

    def __init__(self):
        self.model_loaded = False
        print("⚠️  OnionController: running in stub mode (no model loaded)")

    def deconvolve_region(
        self,
        ppm,
        intensity,
        start_ppm: float,
        end_ppm: float,
        min_snr: float = 3.0,
        device: str = "cpu",
    ) -> dict:
        """
        Deconvolve a spectral region.

        Args:
            ppm:       full ppm array
            intensity: full intensity array (1-D mean spectrum)
            start_ppm: left boundary of region (higher value)
            end_ppm:   right boundary of region (lower value)
            min_snr:   minimum signal-to-noise ratio
            device:    'cpu' or 'cuda'

        Returns:
            dict with keys: ppm, intensity, peaks, model, bic
        """
        mask = (ppm <= start_ppm) & (ppm >= end_ppm)
        region_ppm = ppm[mask]
        region_intensity = intensity[mask]

        # ── Stub: fall back to simple peak detection ─────────────────────────
        from scipy.signal import find_peaks as _find_peaks

        noise = np.std(region_intensity) if len(region_intensity) else 1.0
        indices, _ = _find_peaks(region_intensity, height=noise * min_snr, distance=3)

        peaks = []
        for idx in indices:
            peaks.append({
                'ppm':    float(region_ppm[idx]),
                'height': float(region_intensity[idx]),
                'width':  0.005,
                'snr':    float(region_intensity[idx] / noise),
                'model':  'stub',
                'deconvolved': True,
                'start_ppm': float(region_ppm[idx]) - 0.015,
                'end_ppm':   float(region_ppm[idx]) + 0.015,
            })

        return {
            'ppm':       region_ppm,
            'intensity': region_intensity,
            'peaks':     peaks,
            'model':     'stub (NMR-Onion not loaded)',
            'bic':       0.0,
        }
