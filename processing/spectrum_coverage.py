# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""How much of the spectrum the analysis accounts for.

A table of concentrations says nothing about what it left out. The model
the run ended with -- every quantified analyte at its measured amount,
where the run placed each of its signals, plus what the cascade fitted --
is laid over the mean trace, and two numbers come out: the share of the
detected peaks it explains and the share of the signal area under it. What
it does not explain is listed, tallest first: that list is where an
unidentified compound, a signal of a known one standing somewhere the
reference does not put it, or a misplaced shape shows up.

A peak is explained when the model reaches EXPLAINED_SHARE of the trace's
height at its apex, the definition the extract was first measured with
(ADR-065: 41% of 270 peaks on the NMR_Murilo cell extract, 82-88% on the
GISSMO bench).
"""
from __future__ import annotations

import numpy as np

EXPLAINED_SHARE = 0.5
# A peak lower than this is not asked about: the model need not explain
# noise, and the cascade does not offer anything to a point this low.
COVERAGE_MIN_SNR = 5.0
ORPHANS_LISTED = 40
# The internal standard is not a signal to explain, and neither are its 13C
# satellites: half the one-bond 13C-1H coupling of its methyls (~118 Hz) either
# side. Left in, the standard of the NMR_Murilo extract (SNR 83 000) and its
# satellites made the explained area read 23%.
STANDARD_SATELLITE_HZ = 59.0
STANDARD_PAD_LW = 3.0


def standard_weight(ppm, standard_ppm, *, field_mhz, linewidth_ppm):
    """1 everywhere but the internal standard and its 13C satellites."""
    x = np.asarray(ppm, dtype=float)
    w = np.ones(x.size)
    if standard_ppm is None or not field_mhz:
        return w
    reach = STANDARD_SATELLITE_HZ / float(field_mhz) + STANDARD_PAD_LW * float(linewidth_ppm or 0.0)
    w[np.abs(x - float(standard_ppm)) <= reach] = 0.0
    return w


def coverage(ppm, trace, model, peak_ppms, *, noise_sigma, weight=None,
             explained_share=EXPLAINED_SHARE, min_snr=COVERAGE_MIN_SNR) -> dict:
    """``{'peaks', 'explained', 'peak_fraction', 'area_fraction',
    'orphans': [(ppm, snr, share)], ...}`` for ``model`` (same grid as
    ``trace``) over the detected ``peak_ppms``.

    ``weight`` is 0 where the spectrum does not count (the excluded
    regions); a peak there is not asked about and its area not counted.
    The area share is over the points standing ``min_snr`` noises above
    zero: the model's part of each, never more than the trace holds."""
    x = np.asarray(ppm, dtype=float)
    t = np.asarray(trace, dtype=float)
    m = np.clip(np.asarray(model, dtype=float), 0.0, None)
    w = np.ones(x.size) if weight is None else np.asarray(weight, dtype=float)
    sigma = float(noise_sigma or 0.0)
    floor = float(min_snr) * sigma
    asked, orphans = 0, []
    for p in sorted({float(v) for v in peak_ppms or ()}):
        i = int(np.argmin(np.abs(x - p)))
        h = float(t[i])
        if w[i] <= 0 or h <= floor or h <= 0:
            continue
        asked += 1
        share = min(float(m[i]) / h, 1.0)
        if share < float(explained_share):
            orphans.append((float(x[i]), h / sigma if sigma else float('inf'),
                            share))
    mask = (t > floor) & (w > 0)
    total = float(t[mask].sum())
    area = float(np.minimum(m[mask], t[mask]).sum()) / total if total > 0 else float('nan')
    orphans.sort(key=lambda o: (-o[1], o[0]))
    explained = asked - len(orphans)
    return {'peaks': asked, 'explained': explained,
            'peak_fraction': explained / asked if asked else float('nan'),
            'area_fraction': area,
            'orphans': orphans[:ORPHANS_LISTED],
            'n_orphans': len(orphans),
            'explained_share': float(explained_share),
            'min_snr': float(min_snr)}
