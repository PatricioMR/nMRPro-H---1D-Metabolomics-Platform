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


def level_precision(entry, level) -> float | None:
    """The lowest identity precision a level reached over the tubes: what
    a row at that level can be said to be, at least."""
    vals = [t['levels'][str(level)]['precision']
            for t in (entry or {}).get('tubes', {}).values()
            if t['levels'].get(str(level), {}).get('precision') is not None]
    return min(vals) if vals else None
