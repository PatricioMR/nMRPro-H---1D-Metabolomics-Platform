"""Multiplet-mode quantification: fractional nH and the DB gap-fill window.

Both defects made the divisor and the integrated area disagree:
  1. ``max(1.0, protons)`` floored the divisor while the window's area really
     did correspond to a fractional proton count (BMRB splits a group's protons
     across the signals it is observed in).
  2. The gap-fill window ran centre-to-centre of the outer lines, integrating
     ~47% of a doublet's area against the full nH — and exactly 0 for a singlet.
"""
import numpy as np
import pytest

from gui.stages._quant_helpers import multiplet_nh
from gui.stages._quant_computation import db_gapfill_window


def _lorentz_area_fraction(lo, hi, centre, hwhm):
    """Fraction of a unit-area Lorentzian lying inside [lo, hi]."""
    return (np.arctan((hi - centre) / hwhm)
            - np.arctan((lo - centre) / hwhm)) / np.pi


class TestMultipletNh:
    """Defect 1 — the divisor must be the DB signal's own proton count."""

    def test_integer_nh_unchanged(self):
        # L-alanine CH3: the case that already worked must keep working.
        assert multiplet_nh({'protons': 3}) == pytest.approx(3.0)

    @pytest.mark.parametrize('protons', [0.5, 0.75, 0.8, 1.55])
    def test_fractional_nh_survives(self, protons):
        assert multiplet_nh({'protons': protons}) == pytest.approx(protons)

    def test_glucose_half_proton_not_floored(self):
        # D-Glucose 5.222 (d, nH=0.5): flooring to 1.0 reported 0.47× the true
        # concentration. The divisor must halve with the area.
        assert multiplet_nh({'protons': 0.5}) == pytest.approx(0.5)

    def test_missing_nh_falls_back_to_one(self):
        assert multiplet_nh({}) == pytest.approx(1.0)
        assert multiplet_nh({'protons': None}) == pytest.approx(1.0)
        assert multiplet_nh({'protons': 0}) == pytest.approx(1.0)

    def test_non_numeric_nh_falls_back_to_one(self):
        assert multiplet_nh({'protons': 'n/a'}) == pytest.approx(1.0)

    def test_fractional_divisor_recovers_true_concentration(self):
        # Area of a window holding 0.5 H of a 1.0 mM analyte, in area units
        # where 1 H of 1.0 mM integrates to 1.0.
        area = 0.5
        assert area / multiplet_nh({'protons': 0.5}) == pytest.approx(1.0)
        assert area / max(1.0, 0.5) == pytest.approx(0.5)   # old behaviour


class TestDbGapfillWindow:
    """Defect 2 — the DB-predicted window must contain the outer lines' tails."""

    def test_doublet_pads_past_both_line_centres(self):
        lo, hi = db_gapfill_window([1.460, 1.472], width_ppm=0.008)
        assert lo < 1.460 and hi > 1.472

    def test_singlet_is_not_zero_width(self):
        # min == max previously gave a zero-width window → trapz over one point
        # → an area of exactly 0 reported as a concentration.
        lo, hi = db_gapfill_window([3.034], width_ppm=0.008)
        assert hi > lo

    def test_margin_follows_peak_width_above_the_floor(self):
        lo_n, hi_n = db_gapfill_window([2.000], width_ppm=0.008)   # → 0.010 floor
        lo_w, hi_w = db_gapfill_window([2.000], width_ppm=0.050)   # → 0.035
        assert (hi_n - lo_n) == pytest.approx(0.020)
        assert (hi_w - lo_w) == pytest.approx(0.070)

    def test_no_lines_returns_none(self):
        assert db_gapfill_window([]) is None
        assert db_gapfill_window(None) is None

    def test_doublet_captures_most_of_its_area(self):
        """Centre-to-centre kept ~47% of an alanine doublet; padded keeps >90%."""
        mhz, lw_hz = 600.0, 1.5
        hwhm = (lw_hz / 2.0) / mhz
        j_ppm = 7.20 / mhz
        lines = [(1.466 - j_ppm / 2, 0.5), (1.466 + j_ppm / 2, 0.5)]
        lps = [p for p, _ in lines]

        def captured(lo, hi):
            return sum(w * _lorentz_area_fraction(lo, hi, p, hwhm)
                       for p, w in lines)

        old = captured(min(lps), max(lps))            # centre-to-centre
        new = captured(*db_gapfill_window(lps, 0.008))
        assert old < 0.55, old
        assert new > 0.90, new

    def test_singlet_captures_most_of_its_area(self):
        mhz, lw_hz = 600.0, 1.5
        hwhm = (lw_hz / 2.0) / mhz
        lo, hi = db_gapfill_window([3.034], width_ppm=0.008)
        assert _lorentz_area_fraction(lo, hi, 3.034, hwhm) > 0.90
