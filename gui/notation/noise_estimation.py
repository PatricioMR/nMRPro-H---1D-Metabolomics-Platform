# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
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

