"""Per-sample view of one annotated signal.

Annotation grades on the mean spectrum, so it cannot say in how many samples a
signal is actually there. These read what detection already recorded per sample
instead of re-measuring, which is what keeps "present here" from disagreeing
with the peak list.
"""
import numpy as np
import pytest

from processing.per_sample_signal import (
    alignment_shifts, line_positions, sample_count, signal_across_samples,
    signal_in_sample)

MHZ = 600.0
J = 7.0
D = J / MHZ          # one coupling, in ppm


def _line(ppm, present, drift=None, area=1.0, n=4):
    """One detected line: same apex in every sample unless drift says otherwise."""
    return {
        'ppm': ppm,
        'ppm_per_sample': [ppm + (drift[i] if drift else 0.0) for i in range(n)],
        'aligned_per_sample': list(present),
        'area_per_sample': [area] * n,
    }


def _doublet(centre=1.47, present=(True, True, True, True), drift=None, n=4):
    return [_line(centre - D / 2, present, drift, n=n),
            _line(centre + D / 2, present, drift, n=n)]


class TestPresence:
    def test_every_line_detected_is_present(self):
        r = signal_in_sample(_doublet(), 0)
        assert r['present'] is True

    def test_one_missing_leg_is_absent(self):
        """A doublet with one leg is not a doublet in that sample."""
        lines = [_line(1.4642, (True,) * 4), _line(1.4758, (False, True, True, True))]
        assert signal_in_sample(lines, 0)['present'] is False
        assert signal_in_sample(lines, 1)['present'] is True

    def test_absent_sample_reports_no_j(self):
        """An undetected line's recorded apex is noise; a J from it is fiction."""
        lines = [_line(1.4642, (False,) * 4), _line(1.4758, (False,) * 4)]
        r = signal_in_sample(lines, 0)
        assert r['present'] is False
        assert r['j_obs_hz'] is None
        assert r['delta_j_hz'] is None

    def test_missing_presence_vector_is_not_absence(self):
        """Detection that never recorded presence must not be read as absent."""
        lines = [{'ppm': 1.4642, 'ppm_per_sample': [1.4642] * 4},
                 {'ppm': 1.4758, 'ppm_per_sample': [1.4758] * 4}]
        assert signal_in_sample(lines, 0)['present'] is True


class TestMeasurement:
    def test_j_is_the_line_spacing_in_that_sample(self):
        r = signal_in_sample(_doublet(), 0, db_j_hz=J, field_mhz=MHZ)
        assert r['j_obs_hz'] == pytest.approx(J, abs=1e-6)
        assert r['delta_j_hz'] == pytest.approx(0.0, abs=1e-6)

    def test_drift_moves_the_centre_but_not_the_coupling(self):
        """The whole pattern shifts together, so J survives a 20 Hz drift."""
        shift = 20.0 / MHZ
        lines = _doublet(drift=[0.0, shift, 0.0, 0.0])
        r = signal_in_sample(lines, 1, db_ppm=1.47, db_j_hz=J, field_mhz=MHZ)
        assert r['j_obs_hz'] == pytest.approx(J, abs=1e-6)
        assert r['ppm_diff'] == pytest.approx(shift, abs=1e-9)

    def test_singlet_has_no_j(self):
        r = signal_in_sample([_line(3.03, (True,) * 4)], 0, db_j_hz=None)
        assert r['j_obs_hz'] is None
        assert r['n_lines'] == 1

    def test_area_sums_the_lines(self):
        r = signal_in_sample(_doublet(), 0)
        assert r['area'] == pytest.approx(2.0)

    def test_no_peaks_measures_nothing(self):
        r = signal_in_sample([], 0)
        assert r['present'] is False and r['area'] is None

    def test_sample_index_out_of_range(self):
        assert signal_in_sample(_doublet(), 99)['present'] is False


class TestAcrossSamples:
    def test_counts_only_the_included_samples(self):
        """Blanks live outside `included`, the way detection already treats them."""
        lines = _doublet(present=(True, True, False, True))
        out = signal_across_samples(lines, included=[1, 2, 3])
        assert out['n_included'] == 3
        assert out['n_present'] == 2
        assert [r['sample_idx'] for r in out['per_sample']] == [1, 2, 3]

    def test_all_samples_when_nothing_excluded(self):
        out = signal_across_samples(_doublet())
        assert out['n_included'] == 4 and out['n_present'] == 4

    def test_spread_reports_the_drift(self):
        shift = 0.03
        out = signal_across_samples(_doublet(drift=[0.0, shift, 0.0, 0.0]))
        assert out['ppm_spread'] == pytest.approx(shift, abs=1e-9)

    def test_spread_needs_two_present_samples(self):
        out = signal_across_samples(_doublet(present=(True, False, False, False)))
        assert out['ppm_spread'] is None

    def test_sample_count_survives_peaks_without_vectors(self):
        assert sample_count([{'ppm': 1.0}]) == 0
        assert sample_count(_doublet()) == 4


class TestAlignment:
    def test_shift_is_the_drift_of_the_pattern_centre(self):
        shift = 15.0 / MHZ
        lines = _doublet(drift=[0.0, shift, 0.0, 0.0])
        sh = alignment_shifts(lines)
        # Reference is the mean centre, so the shifts sum to zero.
        assert sum(sh.values()) == pytest.approx(0.0, abs=1e-12)
        assert sh[1] - sh[0] == pytest.approx(shift, abs=1e-9)

    def test_subtracting_the_shift_lines_the_centres_up(self):
        lines = _doublet(drift=[0.0, 0.01, -0.004, 0.007])
        sh = alignment_shifts(lines)
        aligned = [signal_in_sample(lines, i)['ppm_centre'] - sh[i] for i in sh]
        assert max(aligned) - min(aligned) == pytest.approx(0.0, abs=1e-9)

    def test_absent_samples_get_no_shift(self):
        sh = alignment_shifts(_doublet(present=(True, False, True, True)))
        assert set(sh) == {0, 2, 3}

    def test_explicit_reference_pins_the_axis(self):
        lines = _doublet(centre=1.47, drift=[0.0, 0.01, 0.0, 0.0])
        sh = alignment_shifts(lines, reference=1.47)
        assert sh[0] == pytest.approx(0.0, abs=1e-9)
        assert sh[1] == pytest.approx(0.01, abs=1e-9)

    def test_nothing_present_yields_no_shifts(self):
        assert alignment_shifts(_doublet(present=(False,) * 4)) == {}


def test_line_positions_falls_back_to_the_master_apex():
    """A peak without a per-sample vector still has one position to offer."""
    assert line_positions([{'ppm': 2.5}], 0) == [2.5]
