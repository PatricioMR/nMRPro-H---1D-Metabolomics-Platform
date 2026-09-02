"""Foot-to-foot integration, one sample at a time.

One mode, not five. The old engine carried multiplet, area, Lorentz fit, MCR
component and a database gap-fill, each a second answer to the same question
reached by a different path and reconciled nowhere.
"""
import numpy as np
import pytest

from processing.quant_integration import (
    integrate_across_samples, integrate_sample, integrate_signal,
    multiplet_window, window_width_ppm)


def _gauss(x, c, a, w=0.002):
    return a * np.exp(-((x - c) ** 2) / (2 * w ** 2))


def _line(ppm, lo, hi):
    return {'ppm': ppm, 'start_ppm': lo, 'end_ppm': hi}


class TestWindow:
    def test_foot_to_foot_over_every_line(self):
        lines = [_line(1.47, 1.462, 1.474), _line(1.49, 1.484, 1.496)]
        assert multiplet_window(lines) == (1.462, 1.496)

    def test_boundaries_given_in_either_order(self):
        """The ppm axis runs downward, so a peak's feet may arrive reversed."""
        assert multiplet_window([_line(1.47, 1.474, 1.462)]) == (1.462, 1.474)

    def test_a_line_without_feet_falls_back_to_its_apex(self):
        assert multiplet_window([{'ppm': 2.5}]) == (2.5, 2.5)

    def test_nothing_to_bound(self):
        assert multiplet_window([]) is None

    def test_width(self):
        assert window_width_ppm(1.462, 1.496) == pytest.approx(0.034)


class TestOneSample:
    def _spectrum(self, offset=0.0, amp=1.0):
        x = np.linspace(1.40, 1.55, 3000)
        return x, _gauss(x, 1.47, amp) + offset

    def test_a_peak_integrates_to_its_area(self):
        x, y = self._spectrum()
        a = integrate_sample(x, y, 1.45, 1.49, local_baseline=False)
        # Gaussian area = amp·w·sqrt(2π)
        assert a == pytest.approx(1.0 * 0.002 * np.sqrt(2 * np.pi), rel=0.02)

    def test_the_local_baseline_removes_what_it_sits_on(self):
        x, y = self._spectrum(offset=5.0)
        with_bl = integrate_sample(x, y, 1.45, 1.49, local_baseline=True)
        clean = integrate_sample(*self._spectrum(), 1.45, 1.49,
                                 local_baseline=False)
        assert with_bl == pytest.approx(clean, rel=0.02)

    def test_without_the_local_baseline_the_offset_is_counted(self):
        x, y = self._spectrum(offset=5.0)
        assert integrate_sample(x, y, 1.45, 1.49,
                                local_baseline=False) > 0.15

    def test_a_negative_area_is_reported_as_it_comes(self):
        """A window over noise sometimes integrates below zero. Clipping would
        report an absent analyte as merely very dilute."""
        x = np.linspace(1.40, 1.55, 3000)
        y = -_gauss(x, 1.47, 1.0)
        assert integrate_sample(x, y, 1.45, 1.49, local_baseline=False) < 0

    def test_a_descending_axis_gives_the_same_area(self):
        x, y = self._spectrum()
        a = integrate_sample(x, y, 1.45, 1.49, local_baseline=False)
        b = integrate_sample(x[::-1], y[::-1], 1.45, 1.49, local_baseline=False)
        assert a == pytest.approx(b)

    def test_the_window_bounds_may_arrive_reversed(self):
        x, y = self._spectrum()
        assert integrate_sample(x, y, 1.49, 1.45, local_baseline=False) == \
            pytest.approx(integrate_sample(x, y, 1.45, 1.49,
                                           local_baseline=False))

    def test_too_few_points_is_not_an_area_of_zero(self):
        x, y = self._spectrum()
        assert integrate_sample(x, y, 1.4700, 1.47001) is None


class TestAcrossSamples:
    def _matrix(self, amps):
        x = np.linspace(1.40, 1.55, 2000)
        return x, np.column_stack([_gauss(x, 1.47, a) for a in amps])

    def test_each_sample_is_measured_on_its_own_trace(self):
        x, X = self._matrix([1.0, 2.0, 3.0])
        areas = integrate_across_samples(x, X, 1.45, 1.49, local_baseline=False)
        assert set(areas) == {0, 1, 2}
        assert areas[1] == pytest.approx(2 * areas[0], rel=1e-3)
        assert areas[2] == pytest.approx(3 * areas[0], rel=1e-3)

    def test_blanks_stay_outside_by_not_being_included(self):
        x, X = self._matrix([1.0, 2.0, 3.0])
        assert set(integrate_across_samples(x, X, 1.45, 1.49,
                                            included=[1, 2])) == {1, 2}

    def test_an_index_the_matrix_does_not_have_is_ignored(self):
        x, X = self._matrix([1.0])
        assert set(integrate_across_samples(x, X, 1.45, 1.49,
                                            included=[0, 7])) == {0}

    def test_a_sample_that_could_not_be_measured_is_absent_not_zero(self):
        """No measurement is not a measurement of nothing."""
        x, X = self._matrix([1.0, 2.0])
        assert integrate_across_samples(x, X, 1.4700, 1.47001) == {}

    def test_a_single_column_matrix(self):
        x = np.linspace(1.40, 1.55, 2000)
        y = _gauss(x, 1.47, 1.0)
        assert set(integrate_across_samples(x, y, 1.45, 1.49)) == {0}


class TestWholeSignal:
    def test_the_window_travels_with_the_areas(self):
        """A concentration whose window nobody can see is not auditable."""
        x = np.linspace(1.40, 1.55, 2000)
        X = np.column_stack([_gauss(x, 1.466, 1.0) + _gauss(x, 1.478, 1.0)] * 2)
        lines = [_line(1.466, 1.460, 1.472), _line(1.478, 1.472, 1.484)]
        out = integrate_signal(lines, x, X, local_baseline=False)
        assert out['window'] == (1.460, 1.484)
        assert out['width_ppm'] == pytest.approx(0.024)
        assert set(out['areas']) == {0, 1}
        assert out['areas'][0] > 0

    def test_a_signal_with_no_lines_yields_nothing(self):
        x = np.linspace(1.40, 1.55, 100)
        assert integrate_signal([], x, np.zeros((100, 2))) is None
