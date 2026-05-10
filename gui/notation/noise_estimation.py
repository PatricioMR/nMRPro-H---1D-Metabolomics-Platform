# gui/notation/noise_estimation.py
"""
Noise estimation module.
"""
import numpy as np


class NoiseEstimator:
    """Handles noise estimation from spectrum regions."""

    def __init__(self, app):
        self.app = app
        self.noise_std = None
        self.step1_completed = False
        print("✅ NoiseEstimator initialized")

    def estimate_from_region(self, ppm, intensities, region_start, region_end):
        """Estimate noise standard deviation from a ppm region."""
        mask = (ppm <= region_start) & (ppm >= region_end)
        noise_data = intensities[mask]

        if noise_data.ndim == 2:
            noise_data = noise_data.flatten()

        self.noise_std = np.std(noise_data)
        self.step1_completed = True
        return self.noise_std

    def get_status(self):
        """Get current noise estimation status."""
        return {
            'completed': self.step1_completed,
            'noise_std': self.noise_std
        }