# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# gui/notation/results_display.py
"""
Results display module.
"""
import tkinter as tk
from tkinter import ttk


class ResultsDisplay:
    """Handles display of search and deconvolution results."""

    def __init__(self, app):
        self.app = app
        print("✅ ResultsDisplay initialized")
