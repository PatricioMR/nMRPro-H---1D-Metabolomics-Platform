# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/signal_comparison.py
"""
How the quantification of a signal compares with a reference signal's.

One molecule at one concentration gives each of its signals an area
proportional to the protons the signal holds, so area / protons -- what the
signal quantifies -- is the same for all of them, in every sample. A peak of
another molecule, or an overlap, breaks that: its quantification differs from
the reference's, or varies from sample to sample.

    Wei, S. et al. (2011). Ratio analysis nuclear magnetic resonance
    spectroscopy for selective metabolite identification in complex samples.
    Analytical Chemistry 83(20):7616-7623.

For each signal the ratio of its area to the reference's (median over the
samples where both stand over the noise, ``processing.stocsy.fixed_ratio``)
and the protons of each give the gap between what it quantifies and what the
reference quantifies: ``ratio x protons of the reference / protons of the
signal - 1``. The distance joins the gap with the spread of the ratio across
samples, ``hypot(gap, sd)``: the typical relative difference per sample.

One line of a multiplet holds its share of the signal's protons, the share
the quantification itself weighs it by (``quant_integration.pattern_lines``).
A line read with its share is set against the reference like any signal: a
line of the analyte gives the reference's quantification, a line of another
compound does not, and does not keep the ratio from sample to sample.
"""
import warnings

import numpy as np

from processing.quant_integration import pattern_lines
from processing.stocsy import fixed_ratio


def quantity_gap(ratio, n_ref, n_signal) -> float:
    """How far the signal's quantification is from the reference's, as a
    fraction of the reference's (0 = the same). NaN when the ratio cannot be
    read or a count of protons is missing."""
    if not n_ref or not n_signal or not np.isfinite(ratio) or ratio <= 0:
        return np.nan
    return float(ratio) * float(n_ref) / float(n_signal) - 1.0


def compare_to_reference(ref_areas, areas, sigma_ref, sigma_rows, n_ref,
                         n_signals, *, min_valid=5) -> list:
    """One dict per row of ``areas`` (signals x samples) against the reference.

    ``sigma_ref`` and ``sigma_rows`` are the area noises (``area_noise``);
    ``n_signals`` the protons of each row (None where unknown). Keys: ``ratio``,
    ``sd`` (of the log ratio across the samples read), ``noise_sd``, ``n``
    (samples read), ``gap`` and ``distance``. Everything is NaN below
    ``min_valid`` samples read."""
    areas = np.atleast_2d(np.asarray(areas, dtype=float))
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')     # a row with no sample read is NaN, said below
        fr = fixed_ratio(ref_areas, areas, sigma_ref, np.asarray(sigma_rows, float),
                         min_valid=min_valid)
    out = []
    for i in range(areas.shape[0]):
        ratio, sd = float(fr['ratio'][i]), float(fr['sd'][i])
        gap = quantity_gap(ratio, n_ref, n_signals[i])
        out.append({'ratio': ratio, 'sd': sd, 'noise_sd': float(fr['noise_sd'][i]),
                    'n': int(fr['n'][i]), 'gap': gap,
                    'distance': float(np.hypot(gap, sd))
                    if np.isfinite(gap) and np.isfinite(sd) else np.nan})
    return out


def closest_first(distances) -> list:
    """Indices ordering ``distances`` from the nearest to the reference's
    quantification; the ones that could not be read go last."""
    d = np.asarray(distances, dtype=float)
    return sorted(range(d.size), key=lambda i: (not np.isfinite(d[i]), d[i]
                                                if np.isfinite(d[i]) else 0.0, i))


def pattern_share(evidence, line_ppms, tolerance) -> float | None:
    """The fraction of a signal's pattern the lines at ``line_ppms`` hold, as
    the quantification weighs it (``evidence`` = ``pattern_evidence``: the
    predicted lines and, for a measured pattern, their heights). None when no
    line stands on a predicted line (within ``tolerance`` for a pattern of one
    line; the pattern's own for more)."""
    pred = [float(v) for v in (evidence or {}).get('predicted_ppms') or () if v is not None]
    found = [float(v) for v in line_ppms]
    if not pred or not found:
        return None
    if len(pred) == 1:
        return 1.0 if min(abs(x - pred[0]) for x in found) <= tolerance else None
    if len(found) >= len(pred):
        return 1.0
    _p, _w, mask, share = pattern_lines(pred, found, weights=evidence.get('line_weights'))
    return share if any(mask) else None


def per_proton(areas, protons) -> np.ndarray:
    """Areas divided by the protons: what each sample's signal quantifies."""
    return np.asarray(areas, dtype=float) / float(protons)
