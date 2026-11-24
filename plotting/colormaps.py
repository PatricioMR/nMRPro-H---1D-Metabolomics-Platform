# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""One way to get a colormap, on every matplotlib the application runs on.

``matplotlib.cm.get_cmap`` was removed in matplotlib 3.11: the installers built
on GitHub's machines take the newest matplotlib, and the PCA, spectra and
group colours raised AttributeError there while the developer's 3.10 still
had it. The registry (``matplotlib.colormaps``, since 3.6) is the replacement.
"""
import matplotlib


def get_cmap(name, n=None):
    """Colormap *name*, resampled to *n* colours when *n* is given."""
    cmap = matplotlib.colormaps[name]
    return cmap.resampled(n) if n else cmap
