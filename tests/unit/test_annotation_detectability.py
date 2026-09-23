"""Expected SNR and peak claims — the two questions asked of the spectrum."""
import numpy as np
import pytest

from processing.annotation_detectability import (
    OVERLAP_TOL_PPM, build_overlap_map, estimate_noise, mark_detectability,
    proton_count)


def _spectrum(peaks_ppm, height=1000.0, noise=1.0):
    ppm = np.linspace(10.0, 0.0, 2000)
    y = np.full_like(ppm, noise)
    for p in peaks_ppm:
        y[np.argmin(np.abs(ppm - p))] = height
    return ppm, y


class TestProtonCount:
    @pytest.mark.parametrize('value', [0.5, 0.75, 0.8, 1.55])
    def test_fractional_counts_survive(self, value):
        """BMRB splits a group's protons across the signals it is seen in.

        Reading them with int() turned 0.5 into 0, and an expected SNR built on
        a zero proton ratio is zero — which marked the signal below the
        detection limit and hid it. Measured on a real project: 86 of 114
        fractional-nH signals were hidden that way, at expected_snr 0.00.
        """
        assert proton_count({'protons': value}) == pytest.approx(value)

    def test_integer_counts_unchanged(self):
        assert proton_count({'protons': 3}) == 3.0

    def test_missing_or_zero_falls_back(self):
        assert proton_count({}) == 1.0
        assert proton_count({'protons': 0}) == 1.0
        assert proton_count({'protons': None}) == 1.0

    def test_garbage_falls_back(self):
        assert proton_count({'protons': 'n/a'}) == 1.0


class TestDetectability:
    """The estimate is reported; nothing is hidden on the strength of it.

    Signals whose expected SNR fell under the noise floor used to be flagged
    below_detection_limit and dropped from the tree, the overlap map and
    quantification. That decision belongs with singlet anchoring — a signal
    predicted to be invisible is exactly the case a companion multiplet should
    settle — and until that exists, hiding it removes evidence the analyst never
    gets to see.
    """

    def test_expected_snr_follows_the_proton_ratio(self):
        ppm, y = _spectrum([3.0, 4.0], height=1000.0)
        mets = [{'name': 'X', 'db_peaks': [],
                 'matched_peaks': [{'peak_ppm': 3.0, 'protons': 3},
                                   {'peak_ppm': 4.0, 'protons': 1}]}]
        mark_detectability(mets, ppm, y)
        strong, weak = mets[0]['matched_peaks']
        assert weak['expected_snr'] == pytest.approx(strong['expected_snr'] / 3,
                                                     rel=1e-6)

    def test_a_half_proton_signal_gets_a_real_estimate(self):
        """int() turned 0.5 into 0, and a zero proton ratio gives zero SNR."""
        ppm, y = _spectrum([3.0, 4.0], height=1000.0)
        mets = [{'name': 'X', 'db_peaks': [],
                 'matched_peaks': [{'peak_ppm': 3.0, 'protons': 3},
                                   {'peak_ppm': 4.0, 'protons': 0.5}]}]
        mark_detectability(mets, ppm, y)
        assert mets[0]['matched_peaks'][1]['expected_snr'] > 0

    def test_nothing_is_flagged_or_hidden(self):
        ppm, y = _spectrum([3.0], height=12.0)
        mets = [{'name': 'X', 'db_peaks': [{'protons': 1}],
                 'matched_peaks': [{'peak_ppm': 3.0, 'protons': 9},
                                   {'peak_ppm': 4.0, 'protons': 1}]}]
        mark_detectability(mets, ppm, y)
        for mp in mets[0]['matched_peaks']:
            assert 'below_detection_limit' not in mp
        assert 'bdl' not in mets[0]['db_peaks'][0]

    def test_db_peaks_get_an_estimate_too(self):
        ppm, y = _spectrum([3.0], height=1000.0)
        mets = [{'name': 'X',
                 'matched_peaks': [{'peak_ppm': 3.0, 'protons': 3}],
                 'db_peaks': [{'protons': 3}, {'protons': 0.5}]}]
        mark_detectability(mets, ppm, y)
        a, b = mets[0]['db_peaks']
        assert a['expected_snr'] > b['expected_snr'] > 0

    def test_noise_falls_back_when_there_is_no_tail(self):
        ppm = np.linspace(5.0, 0.0, 500)
        assert estimate_noise(ppm, np.full_like(ppm, 4.0)) > 0

    def test_noise_is_never_zero(self):
        ppm = np.linspace(10.0, 0.0, 500)
        assert estimate_noise(ppm, np.zeros_like(ppm)) == 1.0


class TestOverlap:
    def test_every_line_of_a_multiplet_claims_its_peak(self):
        """A doublet of X competes with Y even where Y overlaps one leg only."""
        peaks = [{'rank': 1, 'ppm': 1.330}, {'rank': 2, 'ppm': 1.342}]
        mets = [{'name': 'X', 'matched_peaks': [
            {'peak_rank': 1, 'predicted_lines': [{'ppm': 1.330}, {'ppm': 1.342}]}]}]
        claims = build_overlap_map(mets, peaks)
        assert claims[1] == ['X'] and claims[2] == ['X']

    def test_rivals_accumulate_per_peak(self):
        peaks = [{'rank': 1, 'ppm': 3.20}]
        mets = [{'name': 'X', 'matched_peaks': [{'peak_rank': 1}]},
                {'name': 'Y', 'matched_peaks': [{'peak_rank': 1}]}]
        assert build_overlap_map(mets, peaks)[1] == ['X', 'Y']

    def test_a_faint_rival_still_competes(self):
        """Faintness no longer removes a metabolite from the contest."""
        peaks = [{'rank': 1, 'ppm': 3.20}]
        mets = [{'name': 'X', 'matched_peaks': [{'peak_rank': 1}]},
                {'name': 'Y', 'matched_peaks': [
                    {'peak_rank': 1, 'expected_snr': 0.1}]}]
        assert build_overlap_map(mets, peaks)[1] == ['X', 'Y']

    def test_a_line_beyond_the_tolerance_claims_nothing(self):
        peaks = [{'rank': 1, 'ppm': 1.330}, {'rank': 9, 'ppm': 2.000}]
        mets = [{'name': 'X', 'matched_peaks': [
            {'peak_rank': 1,
             'predicted_lines': [{'ppm': 1.330},
                                 {'ppm': 1.330 + 2 * OVERLAP_TOL_PPM}]}]}]
        assert 9 not in build_overlap_map(mets, peaks)

    def test_a_name_is_never_listed_twice(self):
        peaks = [{'rank': 1, 'ppm': 1.330}]
        mets = [{'name': 'X', 'matched_peaks': [
            {'peak_rank': 1, 'predicted_lines': [{'ppm': 1.330}, {'ppm': 1.3301}]}]}]
        assert build_overlap_map(mets, peaks)[1] == ['X']


class TestTallestLineShare:
    """How tall a signal's tallest line should stand, which is what the
    fingerprint gate predicts against."""

    def test_a_first_order_multiplet_knows_its_own_share(self):
        from processing.annotation_detectability import tallest_line_share
        assert tallest_line_share({'type': 's'}) == 1.0
        assert tallest_line_share({'type': 'd'}) == pytest.approx(0.5)
        assert tallest_line_share({'type': 't'}) == pytest.approx(0.5)

    def test_an_unresolved_multiplet_has_no_predictable_apex(self):
        """It spreads its protons over lines nobody here can count. Saying
        one -- as this did -- expects every proton in a single line, which
        over-predicted inosine's 2H m* at 3.87 ppm by a factor of seven
        (SNR 38 against the 5 the spectrum holds) and marked the whole
        analyte contradicted."""
        from processing.annotation_detectability import tallest_line_share
        assert tallest_line_share({'type': 'm'}) is None
        assert tallest_line_share({'type': 'm*'}) is None

    def test_a_heptet_is_a_septet_under_another_spelling(self):
        """Seven lines either way. HMDB writes 'hept' and BMRB 'sept', and
        a table that knew only one of them read a real heptet as a shape
        nobody could count -- which, before the guard in
        quant_shared.fingerprint_holds, ended the run with a TypeError
        the moment a study was annotated against HMDB."""
        from processing.annotation_detectability import tallest_line_share
        assert (tallest_line_share({'type': 'hept'})
                == tallest_line_share({'type': 'sept'}))
        # 1:6:15:20:15:6:1 -- the tallest of the seven carries 20 of 64
        assert tallest_line_share({'type': 'hept'}) == pytest.approx(20 / 64)

    def test_a_measured_pattern_speaks_for_itself(self):
        """Where the reference measured the pattern, its tallest line's
        share is a measurement, not a guess -- the same rule
        quant_shared.fingerprint_holds applies at quantification time."""
        from processing.annotation_detectability import tallest_line_share
        share = tallest_line_share(
            {'type': 'm*',
             'pattern_lines': '[[-3.0, 0.5], [0.0, 1.0], [3.0, 0.5]]'})
        assert share is not None and share == pytest.approx(0.5)
