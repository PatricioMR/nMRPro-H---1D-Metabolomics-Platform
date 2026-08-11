"""Shared lightweight interaction and peak-overlay manager."""
from __future__ import annotations
import numpy as np

class InteractivePlotManager:
    def __init__(self, figure, ax, canvas, app=None):
        self.figure = figure
        self.ax = ax
        self.canvas = canvas
        self.app = app
        self.original_xlim = None
        self.original_ylim = None
        self._peak_artists = []
        if ax is not None:
            self.original_xlim = ax.get_xlim() if ax.has_data() else None
            self.original_ylim = ax.get_ylim() if ax.has_data() else None

    def reset_zoom(self):
        if self.original_xlim is not None:
            self.ax.set_xlim(self.original_xlim)
        if self.original_ylim is not None:
            self.ax.set_ylim(self.original_ylim)
        self.canvas.draw_idle()

    def _clear_overlay(self):
        for artist in self._peak_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._peak_artists = []

    def draw_peak_overlay(self, peaks, mean_intensity, ppm):
        self._clear_overlay()
        for peak in peaks or []:
            x = float(peak.get("ppm", 0.0))
            artist = self.ax.axvline(x, color="#d62728", lw=0.8,
                                     alpha=0.75, zorder=8)
            self._peak_artists.append(artist)
        self.canvas.draw_idle()

    def force_redraw_peaks(self, peaks, mean_intensity, ppm):
        self.draw_peak_overlay(peaks, mean_intensity, ppm)
