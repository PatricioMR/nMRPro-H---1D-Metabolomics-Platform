# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""What a quantified analyte lists inside another winner's window is
subtracted at the amount it was measured at (ADR-026): the listing places
it, the analyte's own signal sizes it, and it stays only if it fits."""
import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.quant_engine import quantify
from processing.quantification import Candidate, WINDOW_TAKEN

FIELD = 600.0
PPM = np.linspace(1.0, 4.0, 30000)
FWHM = 2.6
LW = FWHM / FIELD
SIGMA = 1.0
J = 7.0


def _line(centre, height, lines=((0.0, 1.0),)):
    return height * lines_model(PPM, list(lines), centre, FWHM, FIELD)


def _study(parts, n=3, seed=0):
    rng = np.random.default_rng(seed)
    cols = []
    for k in range(n):
        y = np.zeros(PPM.size)
        for centre, height, lines in parts:
            y += _line(centre, height * (k + 1), lines)
        cols.append(y + rng.normal(0, SIGMA, PPM.size))
    return np.column_stack(cols)


def _peak(rank, centre, half=0.006):
    return {'rank': rank, 'ppm': centre, 'start_ppm': centre - half,
            'end_ppm': centre + half, 'height': 1.0}


DOUBLET = ((-J / 2, 1.0), (J / 2, 1.0))


def _singlet_cand(mid, rank, centre, db_peaks):
    return Candidate(met_id=mid, peak_rank=rank, line_ranks=frozenset({rank}),
                     evidence={'multiplicity': 's', 'ppm_diff': 0.001,
                               'protons': 3, 'db_ppm': centre,
                               'study_drift_ppm': 0.0, 'n_found': 1,
                               'n_expected': 1, 'peak_ppm': centre,
                               'predicted_ppms': [centre],
                               'pattern_field_mhz': FIELD},
                     payload={'met': {'db_peaks': db_peaks}})


def _run(cands, peaks, X, diag=None):
    return {r['hmdb_id']: r for r in quantify(
        candidates=cands, peaks_by_rank=peaks, ppm=PPM, intensities=X,
        samples_info=[], included=[0, 1, 2], groups={'G': [0, 1, 2]},
        dj_gold_hz=0.2, noise_sigma=SIGMA, linewidth_ppm=LW,
        deconvolve=False, resolve_shared=True, diagnostics=diag)}


def test_a_quantified_analytes_listed_signal_is_subtracted_from_another_window():
    """A owns a singlet at 2.0 (3H). B is quantified on its singlet at 3.0
    (3H) and lists a doublet at 2.003 (3H) -- never a candidate -- that sits
    under A's peak. Without it A reads A + B; with it, A."""
    X = _study([(2.0, 100.0, ((0.0, 1.0),)),
                (2.003, 50.0, DOUBLET),
                (3.0, 100.0, ((0.0, 1.0),))])
    a = _singlet_cand('A', 1, 2.0, [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
                                    {'ppm_center': 3.5, 'type': 's', 'protons': 3}])
    b = _singlet_cand('B', 2, 3.0, [{'ppm_center': 3.0, 'type': 's', 'protons': 3},
                                    {'ppm_center': 2.003, 'type': 'd',
                                     'j_coupling': f'{J:.2f}', 'protons': 3}])
    # A's second listed singlet is in the study too, so its own amount holds
    X = X + np.column_stack([_line(3.5, 100.0 * (k + 1)) for k in range(3)])
    diag = {}
    rows = _run([a, b], {1: _peak(1, 2.0), 2: _peak(2, 3.0)}, X, diag)
    assert 'A' in diag['listed_intruders']
    ra, rb = rows['A'], rows['B']
    assert not ra['_not_quantified'] and not rb['_not_quantified']
    # A's area is now about B's: both 3H at the same amount per proton
    assert ra['_areas_per_sample'][0] == pytest.approx(
        rb['_areas_per_sample'][0], rel=0.25)
    assert any('lists it inside this window' in s for s in ra['_status_reasons'])


def test_a_listed_signal_that_does_not_fit_is_not_kept():
    """B lists a doublet at 2.003 but the spectrum holds no such thing: at
    B's amount it would overshoot A's peak, so it is left out."""
    X = _study([(2.0, 20.0, ((0.0, 1.0),)), (3.0, 100.0, ((0.0, 1.0),)),
                (3.5, 20.0, ((0.0, 1.0),))])
    a = _singlet_cand('A', 1, 2.0, [{'ppm_center': 2.0, 'type': 's', 'protons': 3},
                                    {'ppm_center': 3.5, 'type': 's', 'protons': 3}])
    b = _singlet_cand('B', 2, 3.0, [{'ppm_center': 3.0, 'type': 's', 'protons': 3},
                                    {'ppm_center': 2.003, 'type': 'd',
                                     'j_coupling': f'{J:.2f}', 'protons': 3}])
    diag = {}
    rows = _run([a, b], {1: _peak(1, 2.0), 2: _peak(2, 3.0)}, X, diag)
    assert 'A' not in diag['listed_intruders']
    assert not rows['A']['_not_quantified']


def test_a_window_the_listed_signals_account_for_is_not_the_owners():
    """A's peak at 2.0 is nothing but B's listed doublet: A is not
    quantified, and the row says whose the window was."""
    X = _study([(2.003, 50.0, DOUBLET), (3.0, 100.0, ((0.0, 1.0),)),
                (3.5, 10.0, ((0.0, 1.0),))])
    a = _singlet_cand('A', 1, 2.003, [{'ppm_center': 2.003, 'type': 's', 'protons': 3}])
    b = _singlet_cand('B', 2, 3.0, [{'ppm_center': 3.0, 'type': 's', 'protons': 3},
                                    {'ppm_center': 2.003, 'type': 'd',
                                     'j_coupling': f'{J:.2f}', 'protons': 3}])
    rows = _run([a, b], {1: _peak(1, 2.003), 2: _peak(2, 3.0)}, X)
    assert rows['A']['_not_quantified']
    assert rows['A']['_rejection'] == WINDOW_TAKEN
    assert 'B' in rows['A']['_status_reasons'][0]
