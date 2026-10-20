# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""What a not-detected sample (n.d.) becomes before a multivariate model.

A model (PCA, PLS-DA, a heatmap, a correlation) needs a number in every cell.
Leaving out every metabolite with one n.d. left most real studies with fewer
than two, and filling an n.d. with the column mean invents a measurement that
pulls the groups toward each other. This follows MetaboAnalyst's own
processing, as its R source does it:

1. ``remove_missing_by_percent`` -- MetaboAnalystR ``RemoveMissingByPercent``
   (default ``percent = 0.20``, ``grpWise = FALSE``): a metabolite missing in
   ``percent`` or more of the samples is left out. Group-wise, it stays if at
   least one group is under the threshold -- the modified 80 % rule (Bijlsma S
   et al., Anal Chem 2006;78:567).
2. ``impute_lod`` -- MetaboAnalystR ``ImputeMissingVar(method = "lod")``: each
   remaining n.d. becomes the metabolite's "LoD", 1/5 of its minimum positive
   value (over all samples, or within the sample's group with
   ``group_wise``). A value below anything measured, as an n.d. is.

Pang Z et al. MetaboAnalyst 6.0: towards a unified platform for metabolomics
data processing, analysis and interpretation. Nucleic Acids Res
2024;52:W398-W406. Source: github.com/xia-lab/MetaboAnalystR,
R/general_proc_utils.R and R/util_missing.R.
"""
from __future__ import annotations

import numpy as np

MISSING_PERCENT = 0.20
LOD_FRACTION = 1.0 / 5.0


def _missing(X):
    X = np.asarray(X, dtype=float)
    return ~np.isfinite(X) | (X <= 0)


def remove_missing_by_percent(X, groups=None, *, percent=MISSING_PERCENT,
                              group_wise=False) -> np.ndarray:
    """Boolean mask over the columns of ``X`` (samples x metabolites): True
    where the metabolite is kept. Missing = not finite or not positive."""
    miss = _missing(X)
    if miss.shape[0] == 0:
        return np.zeros(miss.shape[1], dtype=bool)
    if not group_wise or groups is None:
        return miss.mean(axis=0) < float(percent)
    g = np.asarray(list(groups))
    keep = np.zeros(miss.shape[1], dtype=bool)
    for lab in sorted(set(g.tolist()), key=str):
        rows = g == lab
        if rows.any():
            keep |= miss[rows].mean(axis=0) < float(percent)
    return keep


def impute_lod(X, groups=None, *, group_wise=False,
               fraction=LOD_FRACTION) -> tuple[np.ndarray, int]:
    """``(X with every missing cell at the metabolite's LoD, cells replaced)``.

    Group-wise, a group that measured nothing of a metabolite takes the
    metabolite's LoD over all samples. MetaboAnalyst puts 1e-6 there, which in
    mM is a zero that a log transform turns into an outlier; the overall LoD
    is still below anything measured. A metabolite measured nowhere keeps its
    missing cells: there is nothing to place them under."""
    X = np.array(X, dtype=float)
    miss = _missing(X)
    overall = np.array([float(X[~miss[:, j], j].min()) * float(fraction)
                        if (~miss[:, j]).any() else np.nan
                        for j in range(X.shape[1])])
    if not group_wise or groups is None:
        blocks = [np.ones(X.shape[0], dtype=bool)]
    else:
        g = np.asarray(list(groups))
        blocks = [g == lab for lab in sorted(set(g.tolist()), key=str)]
    n = 0
    for rows in blocks:
        sub, m = X[rows], miss[rows]
        for j in range(X.shape[1]):
            if not m[:, j].any() or not np.isfinite(overall[j]):
                continue
            pos = sub[~m[:, j], j]
            sub[m[:, j], j] = (float(pos.min()) * float(fraction) if pos.size
                               else overall[j])
            n += int(m[:, j].sum())
        X[rows] = sub
    return X, n
