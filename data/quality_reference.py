# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The measured meaning of the confidence levels, as the report prints it.

``data/quality_reference.json`` is written by ``validation.quality_reference``
from the crowded reference bench, keyed by the spectrometer field it was
measured at. A project is matched to the field it was acquired at; where
that field has not been measured, the nearest measured one is returned and
the caller is told it is not the project's own.
"""
from __future__ import annotations

import json
import os

from app_paths import asset_path

PATH = asset_path('data', 'quality_reference.json')


def load(path=None) -> dict:
    """The whole file, ``{}`` when it is missing or unreadable."""
    try:
        with open(path or PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def for_field(field_mhz, doc=None):
    """``(measured field, entry, exact)`` for a project at ``field_mhz``:
    the entry measured at that field, or at the nearest one measured
    (``exact`` False); ``(None, None, False)`` when nothing is measured."""
    doc = load() if doc is None else doc
    fields = sorted(int(k) for k in doc if str(k).isdigit())
    if not fields:
        return None, None, False
    try:
        want = float(field_mhz)
    except (TypeError, ValueError):
        want = float(fields[0])
    best = min(fields, key=lambda f: abs(f - want))
    return best, doc[str(best)], abs(best - want) < 1.0


def tube_for_peaks(entry, n_peaks):
    """``(tube key, beyond)``: the validation mixture whose spectrum is as
    crowded as the project's, judged by the peaks detection found -- the one
    thing a real sample tells about how many compounds it holds. The mixture
    with the nearest mean peak count; ``beyond`` is True when the project
    has more peaks than the most crowded mixture measured, whose figures are
    then the best that can be said. Without a peak count, or an entry that
    does not record one, the most crowded mixture: a biological extract
    usually holds a hundred compounds or more. ``(None, False)`` when the
    entry has no mixtures."""
    tubes = (entry or {}).get('tubes') or {}
    if not tubes:
        return None, False
    crowded = max(tubes, key=int)
    counts = {k: t.get('peaks') for k, t in tubes.items()}
    if n_peaks is None or any(v is None for v in counts.values()):
        return crowded, False
    n = float(n_peaks)
    if n > float(counts[crowded]):
        return crowded, True
    return min(counts, key=lambda k: abs(float(counts[k]) - n)), False


def level_precision(entry, level, tube=None) -> float | None:
    """The identity precision a level was measured to have: in ``tube``
    (``tube_for_peaks``) when given, else the lowest over the tubes -- what a
    row at that level can be said to be, at least."""
    tubes = (entry or {}).get('tubes', {})
    if tube is not None:
        return (tubes.get(str(tube), {}).get('levels', {})
                .get(str(level), {}).get('precision'))
    vals = [t['levels'][str(level)]['precision']
            for t in tubes.values()
            if t['levels'].get(str(level), {}).get('precision') is not None]
    return min(vals) if vals else None
