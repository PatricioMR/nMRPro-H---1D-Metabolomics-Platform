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
