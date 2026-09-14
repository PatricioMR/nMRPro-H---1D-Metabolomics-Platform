"""A reference line is expected where its apex would be seen at this study's
linewidth, not where the line is."""
import numpy as np
import pytest

from processing.annotation_scoring import (
    apparent_lines, apparent_spacing_hz, measure_multiplet,
    resolvable_pattern)


def test_lines_well_apart_keep_their_positions():
    seen = apparent_lines([(-5.0, 1.0), (5.0, 1.0)], 1.0)
    assert len(seen) == 2
    assert seen[0][0] == pytest.approx(-5.0, abs=0.02)
    assert seen[1][0] == pytest.approx(5.0, abs=0.02)
    assert seen[0][1] == pytest.approx(1.0)


def test_lines_leaning_on_each_other_pull_inward():
    seen = apparent_lines([(-1.5, 1.0), (1.5, 1.0)], 2.65)
    assert len(seen) == 2
    assert 0 < seen[1][0] < 1.5
    assert seen[1][0] - seen[0][0] < 3.0


def test_lines_closer_than_resolution_are_one_apex():
    seen = apparent_lines([(-0.5, 1.0), (0.5, 1.0)], 2.65)
    assert len(seen) == 1
    assert seen[0][0] == pytest.approx(0.0, abs=0.02)


def test_unequal_heights_keep_the_ratio_where_apart():
    seen = apparent_lines([(-10.0, 1.0), (10.0, 0.5)], 1.0)
    assert seen[1][1] == pytest.approx(0.5, abs=0.02)


def test_no_width_returns_the_lines():
    lines = [(-3.0, 1.0), (3.0, 1.0)]
    assert apparent_lines(lines, None) == lines
    assert apparent_lines(lines, 0.0) == lines


def test_spacing_of_a_wide_doublet_is_its_coupling():
    assert apparent_spacing_hz(7.0, 1.0, 2) == pytest.approx(7.0, abs=0.02)


def test_spacing_of_a_narrow_doublet_is_less_than_its_coupling():
    j = apparent_spacing_hz(3.76, 2.65, 2)
    assert j is not None
    assert 3.4 < j < 3.76


def test_spacing_of_a_merged_doublet_is_none():
    assert apparent_spacing_hz(1.1, 2.65, 2) is None


def test_spacing_without_width_is_the_coupling():
    assert apparent_spacing_hz(6.5, None, 3) == 6.5
    assert apparent_spacing_hz(0.0, 2.0, 2) is None


def test_triplet_spacing_barely_moves_when_resolved():
    assert apparent_spacing_hz(6.7, 2.65, 3) == pytest.approx(6.7, abs=0.05)


def test_measure_multiplet_judges_against_the_apparent_spacing():
    field = 500.0
    j_db = 3.0
    # peaks sit where two Lorentzians of FWHM 2.65 Hz, 3.0 Hz apart, peak
    seen = apparent_lines([(-1.5, 1.0), (1.5, 1.0)], 2.65)
    centre = 5.22
    peaks = [{'ppm': centre + o / field, 'rank': i, 'height': 1.0}
             for i, (o, _r) in enumerate(seen)]
    pred = [{'ppm': centre + k / field, 'rel_intensity': 1.0}
            for k in (-1.5, 1.5)]
    plain = measure_multiplet(pred, peaks, mult='d', j_coupling='3.0',
                              db_ppm=centre, centre_hint=centre, n_expected=2,
                              tolerance=0.01, field_mhz=field)
    aware = measure_multiplet(pred, peaks, mult='d', j_coupling='3.0',
                              db_ppm=centre, centre_hint=centre, n_expected=2,
                              tolerance=0.01, field_mhz=field,
                              linewidth_hz=2.65)
    assert plain.observed_j_hz == pytest.approx(aware.observed_j_hz)
    assert plain.delta_j_hz > 0.1
    assert aware.delta_j_hz == pytest.approx(0.0, abs=0.05)
    assert aware.expected_j_hz < j_db
    assert plain.expected_j_hz == pytest.approx(j_db)


def test_resolvable_pattern_takes_the_apices_at_this_width():
    pattern = [(-9.0, 0.7), (-0.9, 0.95), (0.25, 1.0), (8.2, 0.85)]
    lines = resolvable_pattern(pattern, 0.0, fwhm_hz=2.65)
    assert len(lines) == 3
    # the two inner lines are one apex between them, and the tallest
    assert lines[1][1] == pytest.approx(1.0)
    assert -0.9 < lines[1][0] < 0.25


def test_resolvable_pattern_without_width_merges_by_floor_only():
    pattern = [(-9.0, 0.7), (-0.9, 0.95), (0.25, 1.0), (8.2, 0.85)]
    assert len(resolvable_pattern(pattern, 0.0)) == 4
    assert len(resolvable_pattern(pattern, 2.0)) == 3
