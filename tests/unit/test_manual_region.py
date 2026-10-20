# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The analyst's region: a right-drag on the zoom view says where a signal is,
and the measurement is redone inside it (processing.annotation_scoring).

The window stores only the interval; the lines, J, position and score follow
from the same measurement path every other signal goes through.
"""
import numpy as np
import pytest

from processing.annotation_scoring import (
    MANUAL_REGION_KEY,
    apply_mcr_dj,
    apply_per_sample_consensus,
    manual_region,
    measure_in_region,
    region_peaks,
    resolve_dj,
)

FIELD = 600.0
PARAMS = {'dj_gold_hz': 0.20, 'dj_bad_hz': 0.50, 'dj_neutral': 0.0}


def _peaks(*items):
    """``(ppm, height)`` pairs → detected-peak dicts."""
    return [{'ppm': p, 'height': h, 'rank': i} for i, (p, h) in enumerate(items)]


def _mp(**kw):
    base = {'type': 'd', 'db_ppm': 1.300, 'peak_ppm': 1.300, 'ppm_diff': 0.0,
            'j_coupling': '6.00', 'predicted_lines': []}
    base.update(kw)
    return base


def _resolve(mp, peaks, *, met=None, trace=None, sigma=None, remeasure=False):
    return resolve_dj(mp, met or {'db_peaks': []}, list(peaks),
                      mult=mp['type'], params=PARAMS, field_mhz=FIELD,
                      tolerance=0.05, remeasure=remeasure, trace=trace,
                      sigma=sigma)


# ── the interval itself ───────────────────────────────────────────────────────

def test_region_is_read_as_an_ordered_pair():
    assert manual_region({MANUAL_REGION_KEY: [1.2, 1.3]}) == (1.2, 1.3)


@pytest.mark.parametrize('bad', [None, [], [1.3], [1.3, 1.2], ['x', 1.3],
                                 [float('nan'), 1.3]])
def test_a_malformed_region_is_no_region(bad):
    assert manual_region({MANUAL_REGION_KEY: bad}) is None
    assert manual_region({}) is None


# ── which peaks are inside ────────────────────────────────────────────────────

def test_region_peaks_keeps_only_the_detected_peaks_inside():
    pk = _peaks((1.20, 5), (1.295, 10), (1.305, 10), (1.40, 5))
    inside = region_peaks(pk, (1.29, 1.31))
    assert [p['ppm'] for p in inside] == [1.295, 1.305]


def test_region_peaks_asks_the_trace_for_apices_detection_missed():
    """Two lines the detector merged are two apices on the trace."""
    x = np.linspace(1.28, 1.32, 801)
    lw = 0.0015
    y = (np.exp(-0.5 * ((x - 1.295) / lw) ** 2)
         + np.exp(-0.5 * ((x - 1.305) / lw) ** 2))
    detected = _peaks((1.295, 1.0))          # the detector listed one
    found = region_peaks(detected, (1.29, 1.31), trace=(x, y), sigma=0.01)
    ppms = [round(p['ppm'], 3) for p in found]
    assert ppms == [1.295, 1.305]
    assert found[1]['from_trace'] is True
    assert found[1]['height'] == pytest.approx(1.0, abs=0.01)


def test_region_peaks_does_not_invent_apices_from_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(1.28, 1.32, 801)
    y = rng.normal(0, 0.01, x.size)
    assert region_peaks([], (1.29, 1.31), trace=(x, y), sigma=0.01) == []


# ── measuring inside it ───────────────────────────────────────────────────────

def test_doublet_is_the_two_tallest_peaks_in_the_region():
    pk = _peaks((1.293, 2), (1.295, 10), (1.305, 10), (1.309, 1))
    m = measure_in_region(_mp(), {'db_peaks': []}, pk, (1.29, 1.31),
                          mult='d', n_expected=2, field_mhz=FIELD,
                          dj_gold_hz=0.2)
    assert m.complete
    assert m.found_ppms == [1.295, 1.305]
    assert m.observed_j_hz == pytest.approx(6.0)
    assert m.delta_j_hz == pytest.approx(0.0, abs=1e-9)


def test_fewer_peaks_than_lines_is_incomplete_not_filled_in():
    pk = _peaks((1.295, 10))
    m = measure_in_region(_mp(), {'db_peaks': []}, pk, (1.29, 1.31),
                          mult='d', n_expected=2, field_mhz=FIELD,
                          dj_gold_hz=0.2)
    assert m.measured and not m.complete
    assert m.n_found == 1
    assert m.delta_j_hz is None


def test_peaks_outside_the_region_are_never_the_lines():
    """The algorithm's doublet at 1.30; the analyst says it is at 1.40."""
    pk = _peaks((1.295, 10), (1.305, 10), (1.395, 4), (1.405, 4))
    m = measure_in_region(_mp(), {'db_peaks': []}, pk, (1.39, 1.41),
                          mult='d', n_expected=2, field_mhz=FIELD,
                          dj_gold_hz=0.2)
    assert m.found_ppms == [1.395, 1.405]


def test_singlet_is_the_tallest_peak_in_the_region():
    pk = _peaks((1.295, 3), (1.302, 10))
    m = measure_in_region(_mp(type='s'), {'db_peaks': []}, pk, (1.29, 1.31),
                          mult='s', n_expected=1, field_mhz=FIELD,
                          dj_gold_hz=0.2)
    assert m.complete and m.centre_ppm == 1.302


def test_triplet_lines_carry_the_binomial_heights():
    pk = _peaks((1.290, 5), (1.300, 10), (1.310, 5))
    m = measure_in_region(_mp(type='t'), {'db_peaks': []}, pk, (1.28, 1.32),
                          mult='t', n_expected=3, field_mhz=FIELD,
                          dj_gold_hz=0.2)
    assert [l['rel_intensity'] for l in m.predicted_lines] == [0.5, 1.0, 0.5]


# ── through resolve_dj ────────────────────────────────────────────────────────

def test_a_region_overrides_the_cached_search_measurement():
    """Without the region the stored ΔJ is reused; with it, the lines inside
    are measured and the stored numbers replaced."""
    stale = _mp(_n_found_lines=2, _n_exp_lines=2, delta_j_hz=0.05,
                observed_j_hz=6.05,
                predicted_lines=[{'ppm': 1.295}, {'ppm': 1.305}])
    pk = _peaks((1.295, 10), (1.305, 10), (1.394, 4), (1.406, 4))
    s_dj, conf = _resolve(dict(stale), pk)
    assert (s_dj, conf) == (1.0, 1.0)

    mp = dict(stale, **{MANUAL_REGION_KEY: [1.39, 1.41]})
    s_dj, conf = _resolve(mp, pk)
    assert mp['observed_j_hz'] == pytest.approx(7.2)      # 0.012 ppm × 600
    assert mp['delta_j_hz'] == pytest.approx(1.2)
    assert s_dj == 0.0 and conf == 1.0
    assert [l['ppm'] for l in mp['predicted_lines']] == [1.394, 1.406]
    assert mp['peak_ppm'] == pytest.approx(1.400)
    assert mp['ppm_diff'] == pytest.approx(0.100)


def test_an_empty_region_places_the_comb_unmeasured():
    """Nothing in the region: the reference comb sits at its centre so the
    analyst sees where the lines should have been; none counts as found."""
    mp = _mp(predicted_lines=[{'ppm': 1.295}, {'ppm': 1.305}],
             **{MANUAL_REGION_KEY: [1.39, 1.41]})
    s_dj, conf = _resolve(mp, _peaks((1.295, 10), (1.305, 10)))
    assert mp['_n_found_lines'] == 0
    assert [round(l['ppm'], 3) for l in mp['predicted_lines']] == [1.395, 1.405]
    assert all(l['measured'] is False for l in mp['predicted_lines'])
    assert mp['peak_ppm'] == pytest.approx(1.400)
    assert mp['delta_j_hz'] is None
    assert s_dj == 0.0


def test_one_line_of_a_doublet_places_the_comb_on_it():
    """One apex in the region: the comb is anchored on it, the missing leg
    drawn where the DB J puts it, and the signal stays incomplete."""
    mp = _mp(**{MANUAL_REGION_KEY: [1.39, 1.41]})
    s_dj, conf = _resolve(mp, _peaks((1.300, 10), (1.395, 4)))
    lines = mp['predicted_lines']
    assert [round(l['ppm'], 3) for l in lines] in ([1.385, 1.395], [1.395, 1.405])
    assert [l['measured'] for l in sorted(lines, key=lambda l: l['ppm'])].count(True) == 1
    assert mp['_n_found_lines'] == 1 and mp['_n_exp_lines'] == 2
    assert mp['delta_j_hz'] is None
    assert s_dj == pytest.approx(0.25)          # 1/2 × 0.5
    assert mp['peak_ppm'] != 1.300               # the position followed the region


def test_place_comb_prefers_the_alignment_the_trace_supports():
    """One found peak, a doublet: the other leg goes where the trace is
    taller."""
    from processing.annotation_scoring import place_comb
    x = np.linspace(1.38, 1.42, 401)
    y = np.exp(-0.5 * ((x - 1.395) / 0.001) ** 2) + 0.4 * np.exp(-0.5 * ((x - 1.405) / 0.001) ** 2)
    lines, hits = place_comb([1.395], [{'ppm': 1.395}], j_hz=6.0, n_lines=2,
                             field_mhz=FIELD, region=(1.39, 1.41), trace=(x, y))
    assert [round(l, 3) for l in lines] == [1.395, 1.405]
    assert hits == [True, False]


def test_place_comb_counts_candidate_hits_first():
    from processing.annotation_scoring import place_comb
    cands = [{'ppm': 1.395}, {'ppm': 1.405}]
    lines, hits = place_comb([1.395], cands, j_hz=6.0, n_lines=2,
                             field_mhz=FIELD, region=(1.39, 1.41))
    assert [round(l, 3) for l in lines] == [1.395, 1.405]
    assert hits == [True, True]


def test_a_singlet_in_a_region_moves_to_the_peak_there():
    mp = _mp(type='s', **{MANUAL_REGION_KEY: [1.39, 1.41]})
    s_dj, conf = _resolve(mp, _peaks((1.300, 10), (1.402, 4)))
    assert (s_dj, conf) == (1.0, 1.0)
    assert mp['peak_ppm'] == pytest.approx(1.402)
    assert mp['ppm_diff'] == pytest.approx(0.102)


def test_readings_taken_elsewhere_yield_to_the_region():
    mp = _mp(per_sample_consensus={'ppm_centre': 1.30, 'ppm_diff': 0.0,
                                   'observed_j_hz': 6.0, 'delta_j_hz': 0.0,
                                   'j_n': 5},
             mcr_delta_j_hz=0.0,
             **{MANUAL_REGION_KEY: [1.39, 1.41]})
    assert apply_per_sample_consensus(mp, params=PARAMS) is None
    assert apply_mcr_dj(mp, params=PARAMS) is None
    del mp[MANUAL_REGION_KEY]
    assert apply_per_sample_consensus(mp, params=PARAMS) is not None
