# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Colormaps come from one helper that works on every matplotlib we ship."""
import pathlib
import re

from plotting.colormaps import get_cmap

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_resampled_to_the_number_of_groups():
    assert get_cmap('tab10', 3).N == 3


def test_whole_colormap_without_a_count():
    assert get_cmap('tab10').N == 10


def test_no_module_calls_the_removed_get_cmap():
    # matplotlib 3.11 removed matplotlib.cm.get_cmap; the CI-built installers
    # carry the newest matplotlib, so any call left would fail for users only.
    old = re.compile(r'\b(?:cm|_cm\w*|plt|_plt|pyplot)(?:\.cm)?\.get_cmap\(')
    offenders = [
        f'{p.relative_to(ROOT)}:{n}'
        for folder in ('gui', 'plotting', 'processing', 'controllers', 'data', 'utils')
        for p in (ROOT / folder).rglob('*.py')
        for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1)
        if old.search(line)
    ]
    assert offenders == []
