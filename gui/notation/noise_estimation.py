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

