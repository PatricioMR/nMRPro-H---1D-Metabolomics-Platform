"""The suppressed region is not a source of analytes.

On the reference study ATP's H37 triplet — a real database signal at 4.639 ppm,
1 H — matched the residual water and came back as the study's most abundant
metabolite at 288 mM with an SNR of 31520, a hundred times the next row. The
database has 82 signals between 4.5 and 4.9 ppm, so it was not going to be the
only one.
"""
import numpy as np
import pytest

from processing.spectral_regions import (
    DEFAULT_EXCLUDED_REGIONS, WATER_REGION, format_regions,
    in_excluded_region, parse_regions, split_excluded)


def test_the_water_region_is_excluded_by_default():
    """Not opt-in. An analyst who never opens the field must not be the one
    person quantifying the solvent."""
    assert WATER_REGION in DEFAULT_EXCLUDED_REGIONS
    assert in_excluded_region(4.639, DEFAULT_EXCLUDED_REGIONS)


def test_the_metabolite_region_is_untouched():
    for ppm in (0.0, 1.33, 3.05, 4.10, 5.25, 8.46):
        assert not in_excluded_region(ppm, DEFAULT_EXCLUDED_REGIONS)


def test_the_window_covers_the_shoulder_presaturation_flattens():
    """A narrower window leaves the wings, and the wings are what a database
    signal matches — 4.639 is a wing, not the centre."""
    lo, hi = WATER_REGION
    assert lo <= 4.60 and hi >= 4.90


class TestParsing:
    def test_a_pair(self):
        assert parse_regions('4.50-5.00') == [(4.5, 5.0)]

    def test_several(self):
        assert parse_regions('4.50-5.00, 0.00-0.10') == [(4.5, 5.0), (0.0, 0.1)]

    def test_semicolons_and_spaces(self):
        assert parse_regions(' 4.5 - 5.0 ; 0-0.1 ') == [(4.5, 5.0), (0.0, 0.1)]

    def test_reversed_bounds_are_ordered(self):
        assert parse_regions('5.00-4.50') == [(4.5, 5.0)]

    def test_empty_is_no_regions(self):
        assert parse_regions('') == []
        assert parse_regions(None) == []

    @pytest.mark.parametrize('bad', ['4.50', '4.5-5.0-6.0', 'water'])
    def test_malformed_raises_rather_than_being_skipped(self, bad):
        """An analyst who believes the water is excluded and is not would get
        the artefact back with nothing said. That is the failure this module
        exists to prevent, so it must not be reachable by a typo."""
        with pytest.raises(ValueError):
            parse_regions(bad)

    def test_it_round_trips(self):
        assert parse_regions(format_regions([(4.5, 5.0)])) == [(4.5, 5.0)]


class TestSplitting:
    def _peaks(self):
        return [{'rank': 1, 'ppm': 1.33}, {'rank': 2, 'ppm': 4.65},
                {'rank': 3, 'ppm': 8.46}, {'rank': 4, 'ppm': 4.80}]

    def test_peaks_inside_are_separated_out(self):
        kept, dropped = split_excluded(self._peaks(), DEFAULT_EXCLUDED_REGIONS)
        assert [p['ppm'] for p in kept] == [1.33, 8.46]
        assert [p['ppm'] for p in dropped] == [4.65, 4.80]

    def test_the_dropped_ones_come_back_so_they_can_be_reported(self):
        """A peak that disappears with no count attached leaves the analyst
        unable to tell an exclusion from a detection failure, and those call
        for opposite responses."""
        _, dropped = split_excluded(self._peaks(), DEFAULT_EXCLUDED_REGIONS)
        assert len(dropped) == 2

    def test_no_regions_drops_nothing(self):
        kept, dropped = split_excluded(self._peaks(), [])
        assert len(kept) == 4 and dropped == []


def test_detection_excludes_the_region_and_reports_what_it_dropped():
    """End to end through detect_peaks, because the exclusion belongs there —
    annotation, quantification and the Signal List all read that one list and
    must not be able to disagree about which peaks the study contains."""
    from processing.peak_detection import detect_peaks
    ppm = np.linspace(0.0, 10.0, 8001)

    def gauss(c, h, w=0.004):
        return h * np.exp(-0.5 * ((ppm - c) / w) ** 2)

    y = (gauss(1.33, 100.0) + gauss(4.70, 10000.0) + gauss(8.46, 80.0))
    out = detect_peaks(ppm, np.column_stack([y, y]), noise_std=1.0,
                       min_snr=5.0, baseline_correct=False, global_align=False,
                       return_details=True,
                       excluded_regions=DEFAULT_EXCLUDED_REGIONS)
    found = [p['ppm'] for p in out['peaks']]
    assert not any(4.5 <= v <= 5.0 for v in found), found
    assert out['excluded_peaks']
    assert out['excluded_regions'] == DEFAULT_EXCLUDED_REGIONS


def test_ranks_stay_contiguous_after_the_exclusion():
    """Dropped before ranking, so nothing downstream sees a hole where the
    solvent used to be."""
    from processing.peak_detection import detect_peaks
    ppm = np.linspace(0.0, 10.0, 8001)

    def gauss(c, h, w=0.004):
        return h * np.exp(-0.5 * ((ppm - c) / w) ** 2)

    y = gauss(1.33, 100.0) + gauss(4.70, 10000.0) + gauss(8.46, 80.0)
    peaks = detect_peaks(ppm, np.column_stack([y, y]), noise_std=1.0,
                         min_snr=5.0, baseline_correct=False,
                         global_align=False,
                         excluded_regions=DEFAULT_EXCLUDED_REGIONS)
    assert [p['rank'] for p in peaks] == list(range(1, len(peaks) + 1))
