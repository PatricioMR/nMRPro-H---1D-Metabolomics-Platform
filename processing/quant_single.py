# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/quant_single.py
"""One analyte measured on the signal the analyst chose, nothing else moved.

The analyst picks, among an analyte's signals of Good or Average quality that
no other analyte holds, the one to quantify it with. Re-running the study for
that would re-decide every other analyte too; this measures the one, with the
run's own conventions -- the same integration and capture correction as the
election (``quant_engine.measure_signal``), the run's PQN factors and the
run's own conversion to concentration -- so the row compares with the rest
of the table.
"""
from __future__ import annotations

import numpy as np

from processing.quant_engine import measure_signal


def _at(seq, s):
    """Sample ``s`` of a per-sample record, a list or a dict."""
    if isinstance(seq, dict):
        return seq.get(s, seq.get(str(s)))
    return seq[s] if seq is not None and 0 <= s < len(seq) else None


def run_conversion(rows, n_samples) -> dict:
    """``{sample: factor}`` turning an area per proton, after PQN, into the
    run's concentration -- the standard, its concentration, the dilution or
    correction -- read off the run's own rows: it is the same for every row
    of a sample, so the median over them is that factor, and a row measured
    with it compares with the table. A sample no row was measured in has
    none."""
    out = {}
    for s in range(int(n_samples)):
        ks = []
        for r in rows or ():
            if r.get('_not_quantified') or r.get('_is_is'):
                continue
            nh = r.get('_nh')
            a = _at(r.get('_areas_per_sample'), s)
            c = _at(r.get('_concs_per_sample'), s)
            f = _at(r.get('_pqn_factors') or {}, s) or 1.0
            try:
                nh, a, c, f = float(nh), float(a), float(c), float(f)
            except (TypeError, ValueError):
                continue
            if nh > 0 and a > 0 and c > 0 and f > 0 and np.isfinite(c):
                ks.append(c * nh * f / a)
        if ks:
            out[s] = float(np.median(ks))
    return out


def quantify_chosen_signal(lines, evidence, *, ppm, intensities, included,
                           local_baseline, align, blank_indices,
                           linewidth_ppm, other_peaks, noise_sigma,
                           conversion, pqn_factors=None) -> dict:
    """The analyte's numbers on the chosen signal: ``{'info': what
    ``measure_signal`` measured, 'areas': {sample: area}, 'concs': {sample:
    concentration}}``. Raises ValueError with the reason when the signal
    cannot be measured -- said to the analyst, never a silent zero.

    ``conversion`` is the run's factor per sample (``run_conversion``) and
    ``pqn_factors`` its PQN factors: the area per proton is divided by the
    PQN factor and multiplied by the conversion, as the run did for every
    analyte. A sample without a factor has no concentration (NaN).
    """
    if not lines:
        raise ValueError('The signal has no detected line to integrate.')
    info, why = measure_signal(
        lines, evidence or {}, ppm=ppm, intensities=intensities,
        included=included, local_baseline=local_baseline, align=align,
        blank_indices=blank_indices, linewidth_ppm=linewidth_ppm,
        other_peaks=other_peaks, noise_sigma=noise_sigma)
    if why == 'unmeasurable':
        raise ValueError('No integration window could be built on its lines.')
    if why == 'partial':
        found, expected, share, _missing = info
        raise ValueError(
            f'Its detected lines hold {100 * share:.0f}% of the pattern '
            f'({found} of {expected} lines): too little to scale to the '
            f'whole signal.')
    nh = float(info['nh'])
    concs = {}
    for s, a in info['areas'].items():
        k = conversion.get(s)
        f = float(_at(pqn_factors or {}, s) or 1.0)
        concs[s] = (float(a) / (nh * f) * k) if k else float('nan')
    if not any(np.isfinite(v) and v > 0 for v in concs.values()):
        raise ValueError('The window holds no positive area in any sample.')
    return {'info': info, 'areas': info['areas'], 'concs': concs}
