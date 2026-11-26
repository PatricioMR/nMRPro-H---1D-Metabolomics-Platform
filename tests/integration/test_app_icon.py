# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The Windows icon is the current logo.

Up to v0.89.1 utils/nmrpro_h.ico still held the logo of the project's former
name, so the installed program, its shortcut and its window showed it while
the PNG (macOS, Linux, splash, PDF) had the new one.
"""
import pathlib

import numpy as np
from PIL import Image

UTILS = pathlib.Path(__file__).resolve().parents[2] / 'utils'


def _rgba(img, size):
    return np.asarray(img.convert('RGBA').resize(size, Image.LANCZOS),
                      dtype=float)


def test_the_icon_is_the_logo_at_every_size():
    logo = Image.open(UTILS / 'Logo_NMRPro-H.png')
    ico = Image.open(UTILS / 'nmrpro_h.ico')
    sizes = sorted(ico.info['sizes'])
    assert (256, 256) in sizes and (16, 16) in sizes
    for size in sizes:
        ico.size = size
        diff = np.abs(_rgba(ico, size) - _rgba(logo, size)).mean()
        # 0.0 for the current icon; the old one differed by 8.2 (only the name
        # under the picture changed), so any stale frame fails this.
        assert diff < 1, f'{size}: mean difference {diff:.1f} of 255'
