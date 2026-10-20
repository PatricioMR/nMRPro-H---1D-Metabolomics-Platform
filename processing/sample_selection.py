# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/sample_selection.py
"""Which samples a processing step is allowed to look at.

The rule is one line long and was written inside ``detect_peaks``, so detection
honoured it and nothing else did: annotation averaged every column, blanks
included, and the MCR fitted its components across them too. One definition,
imported by whoever needs it, is what keeps the steps from disagreeing about
what the study contains.
"""
from __future__ import annotations

BLANK_TYPE = 'BLK'


def is_included(info) -> bool:
    """False for a blank or a sample the analyst switched off."""
    info = info or {}
    if str(info.get('type', 'Sample')).upper() == BLANK_TYPE:
        return False
    return not bool(info.get('excluded', False))


def included_sample_indices(samples_info, n_samples=None) -> list[int]:
    """Indices of the samples a processing step should read.

    Falls back to every sample when the selection would be empty — a study with
    nothing but blanks is a labelling mistake, and refusing to process it would
    only hide that behind an empty result.
    """
    if n_samples is None:
        n_samples = len(samples_info or ())
    info = list(samples_info or ())
    keep = [i for i in range(n_samples)
            if is_included(info[i] if i < len(info) else None)]
    return keep or list(range(n_samples))
