import numpy as np
import pytest

from processing.peak_alignment import global_peak_rigid_align


def _gaussian(ppm, center, width, amp=1.0):
    return amp * np.exp(-0.5 * ((ppm - center) / width) ** 2)


class TestGlobalPeakRigidAlign:
    def test_recovers_integer_global_shifts(self):
        ppm = np.linspace(0.0, 10.0, 2001)
        base = (
            _gaussian(ppm, 1.25, 0.02, 1.0)
            + _gaussian(ppm, 3.40, 0.03, 0.8)
            + _gaussian(ppm, 7.15, 0.025, 0.9)
        )

        step = float(ppm[1] - ppm[0])
        sample_0 = base
        sample_1 = np.roll(base, 3)
        sample_2 = np.roll(base, -4)
        all_intensities = np.column_stack([sample_0, sample_1, sample_2])

        aligned, shifts_pts, shifts_ppm, reference = global_peak_rigid_align(
            ppm,
            all_intensities,
            max_shift_ppm=0.04,
        )

        assert shifts_pts.tolist() == [0, -3, 4]
        assert np.allclose(shifts_ppm, np.array([0.0, -3 * step, 4 * step]))
        assert np.max(np.abs(aligned[:, 0] - aligned[:, 1])) < 1e-9
        assert np.max(np.abs(aligned[:, 0] - aligned[:, 2])) < 1e-9
        assert reference.shape == ppm.shape

    def test_ignores_blk_when_building_alignment_reference(self):
        ppm = np.linspace(0.0, 10.0, 2001)
        base = _gaussian(ppm, 2.0, 0.03, 1.0) + _gaussian(ppm, 6.0, 0.04, 0.7)
        blk = np.roll(base, 5)
        sample_a = base
        sample_b = np.roll(base, -2)
        all_intensities = np.column_stack([blk, sample_a, sample_b])
        samples_info = [
            {'type': 'BLK', 'label': 'BLK'},
            {'type': 'Sample', 'label': 'A'},
            {'type': 'Sample', 'label': 'B'},
        ]

        aligned, shifts_pts, _, _ = global_peak_rigid_align(
            ppm,
            all_intensities,
            samples_info=samples_info,
            max_shift_ppm=0.04,
        )

        assert shifts_pts[0] < shifts_pts[1]
        assert shifts_pts[1] <= shifts_pts[2]
        assert np.max(np.abs(aligned[:, 1] - aligned[:, 2])) < 1e-9

class TestParabolicApex:
    """The sub-pixel step is signed. A ppm axis descends, so a positive
    sub-sample offset must move the apex toward the NEXT index — a smaller ppm.
    An unsigned step moved it the other way by up to half a point, 0.15 Hz on a
    typical grid, which two lines of one multiplet could turn into more than
    dj_gold_hz of invented disagreement.
    """

    def _axis(self):
        import numpy as np
        return np.array([1.50, 1.49, 1.48, 1.47, 1.46])

    def test_a_symmetric_apex_does_not_move(self):
        import numpy as np
        from processing.peak_alignment import parabolic_apex
        y = np.array([0.0, 1.0, 4.0, 1.0, 0.0])
        assert parabolic_apex(self._axis(), y, 2) == pytest.approx(1.48)

    def test_it_leans_toward_the_taller_neighbour(self):
        import numpy as np
        from processing.peak_alignment import parabolic_apex
        x = self._axis()
        # Neighbour at index 3 (ppm 1.47) is the taller one → apex below 1.48.
        y = np.array([0.0, 1.0, 4.0, 3.0, 0.0])
        assert parabolic_apex(x, y, 2) < 1.48
        # And the mirror case leans the other way.
        y2 = np.array([0.0, 3.0, 4.0, 1.0, 0.0])
        assert parabolic_apex(x, y2, 2) > 1.48

    def test_it_never_leaves_the_neighbouring_points(self):
        import numpy as np
        from processing.peak_alignment import parabolic_apex
        x = self._axis()
        y = np.array([0.0, 3.9, 4.0, 0.0, 0.0])
        assert 1.485 >= parabolic_apex(x, y, 2) >= 1.475

    def test_an_edge_point_stays_where_it_is(self):
        import numpy as np
        from processing.peak_alignment import parabolic_apex
        x = self._axis()
        y = np.array([4.0, 1.0, 0.0, 0.0, 0.0])
        assert parabolic_apex(x, y, 0) == pytest.approx(1.50)

    def test_a_flat_top_stays_where_it_is(self):
        import numpy as np
        from processing.peak_alignment import parabolic_apex
        x = self._axis()
        y = np.array([0.0, 4.0, 4.0, 4.0, 0.0])
        assert parabolic_apex(x, y, 2) == pytest.approx(1.48)
