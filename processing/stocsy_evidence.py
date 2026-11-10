# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/stocsy_evidence.py
"""
Identity evidence from co-variation across the samples: when the signals an
analyte stands on move together, that is one molecule at one concentration.

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic
    1H NMR data sets. Analytical Chemistry 77(5):1282-1289.

The rule: an analyte whose own signals (one peak per signal of its listing that
was matched) hold ``STOCSY_LIFT_MIN_PAIRS`` or more pairs that co-vary -- r of
``STOCSY_LIFT_R`` or more, p under ``STOCSY_LIFT_ALPHA``, on PQN-normalised
areas -- is raised one dot, to three at most. It is never lowered: a low
correlation is what a weak or sparsely varying analyte shows too, and is not
evidence against it. An analyte the cascade brought in is not raised here: its
lines were never confirmed one by one and the cascade caps it at two
(``signal_links`` asks that of its measured amounts instead, ADR-101).

Measured on the GISSMO bench (tubes 18/50/100, 600 MHz, standard and
extract-like regimes -- ADR-097). Chosen on seeds 0-2 among six variants, where
it moved 24 real analytes from one dot to two with 1 false one and 6 from two
to three with none (cascade rows excluded). Run on seeds 3-5, which it was not
chosen on: 10 real analytes from one dot to two and 12 from two to three, no
false one, and the precision of each level unchanged (three dots 97.8% ->
98.0%, two dots 87.1% -> 87.1%). The gain is small: 22 real analytes in about
1,600 reported.
"""
import numpy as np

from processing.quant_engine import raise_confidence
from processing.sample_selection import included_sample_indices
from processing.stocsy import (
    STOCSY_MIN_SAMPLES, build_basis, peak_vectors)

# Set to False to switch the lift off (ADR-097).
STOCSY_LIFT = True
STOCSY_LIFT_MIN_PAIRS = 2
STOCSY_LIFT_R = 0.80
STOCSY_LIFT_ALPHA = 0.05
MAX_LEVEL = 3


def own_signal_pairs(matched_peaks, row_of, R, P, *, min_r=STOCSY_LIFT_R,
                     alpha=STOCSY_LIFT_ALPHA) -> int:
    """How many pairs among the peaks the analyte matched co-vary.

    ``matched_peaks`` are the annotation's matches (each with ``peak_rank``);
    ``row_of`` maps a peak rank to its row in ``R``/``P`` (the correlation
    and p-value matrices of the detected peaks)."""
    ranks = sorted({int(mp['peak_rank']) for mp in matched_peaks or ()
                    if mp.get('peak_rank') is not None
                    and int(mp['peak_rank']) in row_of})
    n = 0
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            a, b = row_of[ranks[i]], row_of[ranks[j]]
            r, p = R[a, b], P[a, b]
            if np.isfinite(r) and r >= min_r and np.isfinite(p) and p < alpha:
                n += 1
    return n


def lift_by_covariation(rows, metabolites, peaks, samples_info, *,
                        min_pairs=None) -> list:
    """The rows with the analytes whose own signals co-vary raised one dot.

    Each raised row gets ``_stocsy_lift`` (``{'pairs', 'from'}``) and a
    sentence in ``_identity_why``. Rows are changed in place and returned.
    Returns them untouched when the study holds fewer than
    ``STOCSY_MIN_SAMPLES`` analytical samples: with fewer, a correlation
    reaches 0.80 by chance too often to count."""
    min_pairs = STOCSY_LIFT_MIN_PAIRS if min_pairs is None else int(min_pairs)
    n_cols = max((len(np.asarray(p.get('area_per_sample', ())))
                  for p in peaks or ()), default=0)
    included = included_sample_indices(samples_info, n_cols)
    if len(included) < STOCSY_MIN_SAMPLES:
        return rows
    ranks, M, _ = peak_vectors(peaks, included=included,
                               min_samples=STOCSY_MIN_SAMPLES)
    if len(ranks) < 3:
        return rows
    info = list(samples_info or ())
    groups = [str((info[i] if i < len(info) else {}).get('label', ''))
              for i in included]
    basis = build_basis(M, groups=groups, normalize='pqn')
    R = basis.correlate(M, M)
    P = np.vstack([basis.stats(row)[0] for row in R])
    row_of = {r: i for i, r in enumerate(ranks)}
    by_id = {m.get('id'): m for m in metabolites or ()}
    for row in rows:
        level = int(row.get('_confidence') or 0)
        if (row.get('_not_quantified') or row.get('_is_is')
                or row.get('_by_cascade') or level not in (1, 2)):
            continue
        met = by_id.get(row.get('hmdb_id'))
        if not met:
            continue
        pairs = own_signal_pairs(met.get('matched_peaks'), row_of, R, P)
        if pairs < min_pairs:
            continue
        raise_confidence(row, min(level + 1, MAX_LEVEL))
        row['_stocsy_lift'] = {'pairs': pairs, 'from': level}
        why = (f"its own signals move together across the {basis.n} samples: "
               f"{pairs} pairs with r of at least {STOCSY_LIFT_R:.2f} "
               f"(p < {STOCSY_LIFT_ALPHA:g}) -- one molecule has one "
               "concentration per sample")
        row['_identity_why'] = (f"{row['_identity_why']}; {why}"
                                if row.get('_identity_why') else why)
    return rows
