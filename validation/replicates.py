# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Replicate runs reported as they were measured: the figures of each run,
then their mean and the spread between runs (sample standard deviation).
A bench runs a tube more than once to have that spread; totals over the runs
are never reported."""
from __future__ import annotations

import statistics

import numpy as np


def reported(rows):
    """The rows the quantification reports with a number (not the internal
    standard, not the refused ones)."""
    return [r for r in rows if not r.get('_not_quantified') and not r.get('_is_is')]


def run_figures(rows, truth, errors) -> dict:
    """One run: analytes in the tube, found (real, reported) and false,
    recovery and precision, real/false at each confidence level, and the
    median and p90 of ``errors`` (relative errors of the real analytes, in
    whatever unit the bench defines: per analyte or per analyte and group)."""
    q = reported(rows)
    real = sum(1 for r in q if r['hmdb_id'] in truth)
    false = len(q) - real
    out = {'in_tube': len(truth), 'found': real, 'false': false,
           'recovery_pct': 100 * real / len(truth) if truth else np.nan,
           'precision_pct': 100 * real / len(q) if q else np.nan}
    for lvl in (3, 2, 1):
        at = [r for r in q if min(max(int(r.get('_confidence') or 1), 1), 3) == lvl]
        out[f'real{lvl}'] = sum(1 for r in at if r['hmdb_id'] in truth)
        out[f'false{lvl}'] = len(at) - out[f'real{lvl}']
    e = np.abs(np.asarray(errors, float))
    e = e[np.isfinite(e)]
    out['error_median_pct'] = 100 * float(np.median(e)) if e.size else np.nan
    out['error_p90_pct'] = 100 * float(np.percentile(e, 90)) if e.size else np.nan
    return out


def mean_sd(values, digits=1):
    """``(mean, sd)`` of the finite values, rounded; the SD is ``''`` with
    fewer than two, the mean ``''`` with none."""
    v = [float(x) for x in values if x != '' and x is not None and np.isfinite(float(x))]
    if not v:
        return '', ''
    sd = round(statistics.stdev(v), digits) if len(v) > 1 else ''
    return round(statistics.fmean(v), digits), sd


def summarize(runs, keys, digits=1) -> dict:
    """``{key_mean, key_sd}`` over the runs for each key, plus ``runs``."""
    out = {'runs': len(runs)}
    for k in keys:
        out[f'{k}_mean'], out[f'{k}_sd'] = mean_sd([r.get(k) for r in runs], digits)
    return out


def paired_change(runs, base, keys, digits=1) -> dict:
    """The change of each run against its own base run (same tube, same
    seed: ``runs[i]`` with ``base[i]``), as ``{key_change_mean,
    key_change_sd}``."""
    if len(runs) != len(base):
        raise ValueError('paired runs need the same seeds on both sides')
    out = {}
    for k in keys:
        d = [float(r[k]) - float(b[k]) for r, b in zip(runs, base)
             if r.get(k) not in ('', None) and b.get(k) not in ('', None)]
        out[f'{k}_change_mean'], out[f'{k}_change_sd'] = mean_sd(d, digits)
    return out


def pm(mean, sd, unit='') -> str:
    """``mean ± sd`` for a table cell."""
    if mean == '':
        return '—'
    return f'{mean}{unit}' if sd == '' else f'{mean} ± {sd}{unit}'
