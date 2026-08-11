"""Matplotlib rendering helpers used by the main application."""
from __future__ import annotations
import numpy as np

def _settings(settings):
    settings = settings or {}
    return {
        "line_width": settings.get("line_width", 0.8),
        "alpha": settings.get("alpha", 0.9),
        "grid": settings.get("grid", False),
        "xlabel": settings.get("xlabel", "Chemical shift (ppm)"),
    }

def _prepare(ax, ppm, settings):
    cfg = _settings(settings)
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("Intensity")
    ax.grid(bool(cfg["grid"]), alpha=0.25)
    if len(ppm):
        ax.set_xlim(float(np.nanmax(ppm)), float(np.nanmin(ppm)))
    return cfg

def plot_spectrum(ax, ppm, intensity, title="", color="black", label=None,
                  show_legend=True, settings=None):
    ax.clear()
    ppm = np.asarray(ppm)
    y = np.asarray(intensity)
    cfg = _prepare(ax, ppm, settings)
    ax.plot(ppm, y, color=color, lw=cfg["line_width"],
            alpha=cfg["alpha"], label=label)
    ax.set_title(title)
    if show_legend and label:
        ax.legend(loc="best", fontsize=8)
    return ax

def plot_multiple_spectra(ax, ppm, intensities, names, labels, group_colors,
                          show_legend=True, settings=None):
    ax.clear()
    ppm = np.asarray(ppm)
    y = np.asarray(intensities)
    cfg = _prepare(ax, ppm, settings)
    if y.ndim == 1:
        y = y[:, None]
    for i in range(y.shape[1]):
        group = str(labels[i]) if i < len(labels) else "Unknown"
        color = group_colors.get(group, None) if group_colors else None
        name = names[i] if i < len(names) else f"Sample {i + 1}"
        ax.plot(ppm, y[:, i], lw=cfg["line_width"], alpha=cfg["alpha"],
                color=color, label=name)
    if show_legend and y.shape[1] <= 20:
        ax.legend(loc="best", fontsize=8)
    return ax
