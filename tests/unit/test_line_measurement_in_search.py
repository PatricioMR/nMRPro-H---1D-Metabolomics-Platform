# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Line counting and J measurement belong to annotation, not to the view.

Whether a multiplet's expected lines are present in the spectrum is what decides
if the analyte is a candidate at all. It therefore has to happen while
candidates are being built — not when a results window opens.
"""
import pytest

from processing.annotation_scoring import (
    LINE_COUNTS, expected_line_count, is_competitor, measure_multiplet)


def _pk(ppm, rank=0):
    return {'rank': rank, 'ppm': ppm, 'height': 1.0}


class TestMultiplicityVocabulary:
    """BMRB writes quint/sext/sept; HMDB writes p/h/hept. Both must resolve.

    They did not: 'quin' was in the search's table where BMRB writes 'quint',
    so 35 quintets, 9 sextets and 7 septets in BMRB_Metabolomics_v01 fell to the
    default of 2 expected lines — and were drawn as a single line, i.e. modelled
    as singlets.
    """

    @pytest.mark.parametrize('mult,n', [
        ('d', 2), ('t', 3), ('q', 4),
        ('quint', 5), ('quintet', 5), ('p', 5), ('pentet', 5),
        ('sext', 6), ('h', 6),
        ('sept', 7), ('hept', 7),
        ('dd', 4), ('dt', 6), ('td', 6), ('tt', 9), ('ddd', 8),
    ])
    def test_expected_line_count(self, mult, n):
        assert expected_line_count(mult, fallback=2) == n

    def test_no_real_multiplicity_falls_back_silently(self):
        for mult in ('quint', 'sext', 'sept', 'p', 'h', 'hept'):
            assert mult in LINE_COUNTS, mult

    def test_unknown_label_still_uses_the_fallback(self):
        assert expected_line_count('zzz', fallback=3) == 3


class TestMeasureMultiplet:
    """The rigorous measurement the search now runs."""

    def test_complete_doublet_measures_j(self):
        j_ppm = 7.0 / 600.0
        lines = [{'ppm': 1.0 - j_ppm / 2}, {'ppm': 1.0 + j_ppm / 2}]
        peaks = [_pk(l['ppm'], i) for i, l in enumerate(lines)]
        m = measure_multiplet(lines, peaks, mult='d', j_coupling='7.00',
                              db_ppm=1.0, centre_hint=1.0, n_expected=2,
                              tolerance=0.005, field_mhz=600.0)
        assert m.complete
        assert m.observed_j_hz == pytest.approx(7.0, abs=0.1)
        assert m.delta_j_hz == pytest.approx(0.0, abs=0.1)

    def test_missing_leg_reports_no_j(self):
        """A ΔJ measured on an incomplete set is not this signal's ΔJ."""
        j_ppm = 7.0 / 600.0
        lines = [{'ppm': 1.0 - j_ppm / 2}, {'ppm': 1.0 + j_ppm / 2}]
        peaks = [_pk(lines[0]['ppm'], 0)]          # only one leg detected
        m = measure_multiplet(lines, peaks, mult='d', j_coupling='7.00',
                              db_ppm=1.0, centre_hint=1.0, n_expected=2,
                              tolerance=0.005, field_mhz=600.0)
        assert not m.complete
        assert m.delta_j_hz is None
        assert m.observed_j_hz is None

    def test_quintet_needs_five_lines_not_two(self):
        """The bug the vocabulary fix closes: 2 of 5 lines is not a quintet."""
        j_ppm = 7.0 / 600.0
        lines = [{'ppm': 2.0 + (i - 2) * j_ppm} for i in range(5)]
        peaks = [_pk(lines[i]['ppm'], i) for i in (0, 1)]     # 2 of 5 found
        n_exp = expected_line_count('quint', fallback=2)
        assert n_exp == 5
        m = measure_multiplet(lines, peaks, mult='quint', j_coupling='7.00',
                              db_ppm=2.0, centre_hint=2.0, n_expected=n_exp,
                              tolerance=0.005, field_mhz=600.0)
        assert not m.complete
        assert m.delta_j_hz is None

    def test_incomplete_multiplet_cannot_compete(self):
        """End to end: a missing line keeps the candidate out of the contest."""
        j_ppm = 7.0 / 600.0
        lines = [{'ppm': 1.0 - j_ppm / 2}, {'ppm': 1.0 + j_ppm / 2}]
        peaks = [_pk(lines[0]['ppm'], 0)]
        m = measure_multiplet(lines, peaks, mult='d', j_coupling='7.00',
                              db_ppm=1.0, centre_hint=1.0, n_expected=2,
                              tolerance=0.005, field_mhz=600.0)
        cand = {'multiplicity': 'd', 'delta_j_hz': m.delta_j_hz,
                'n_found': m.n_found, 'n_expected': m.n_expected,
                'ppm_diff': 0.0}
        assert not is_competitor(cand)


class TestSearchOwnsTheMeasurement:
    def test_search_no_longer_imports_the_biased_estimator(self):
        """_observe_j_hz picked the peak separation closest to the DB J.

        Circular: ΔJ came out small by construction. It also never counted
        lines, so a doublet missing a leg still reported a confident J.
        """
        import gui.stages._annotation_search as srch
        assert not hasattr(srch, '_observe_j_hz')

    def test_search_imports_the_rigorous_one(self):
        import gui.stages._annotation_search as srch
        assert hasattr(srch, 'measure_multiplet')
        assert hasattr(srch, 'expected_line_count')


class TestResolveDj:
    """Settling a signal's ΔJ — reuse what the search measured, or measure again."""

    @staticmethod
    def _mp(**kw):
        base = {'type': 'd', 'db_ppm': 1.0, 'peak_ppm': 1.0,
                'j_coupling': '7.00', 'predicted_lines': []}
        base.update(kw)
        return base

    PARAMS = {'dj_gold_hz': 0.20, 'dj_bad_hz': 0.50, 'dj_neutral': 0.0}

    def _resolve(self, mp, met=None, peaks=(), remeasure=False):
        from processing.annotation_scoring import resolve_dj
        return resolve_dj(mp, met or {'db_peaks': []}, list(peaks),
                          mult=mp['type'], params=self.PARAMS,
                          field_mhz=600.0, tolerance=0.005,
                          remeasure=remeasure)

    def test_a_singlet_has_no_coupling_to_validate(self):
        s_dj, conf = self._resolve(self._mp(type='s'))
        assert (s_dj, conf) == (1.0, 1.0)

    def test_the_search_measurement_is_reused(self):
        mp = self._mp(_n_found_lines=2, _n_exp_lines=2, delta_j_hz=0.05)
        s_dj, conf = self._resolve(mp)
        assert s_dj == 1.0 and conf == 1.0        # ΔJ inside the gold band

    def test_a_bad_coupling_scores_zero(self):
        mp = self._mp(_n_found_lines=2, _n_exp_lines=2, delta_j_hz=0.9)
        assert self._resolve(mp)[0] == 0.0

    def test_an_incomplete_multiplet_loses_its_dj(self):
        """A coupling measured from a subset describes other peaks."""
        mp = self._mp(_n_found_lines=1, _n_exp_lines=2, delta_j_hz=0.01)
        s_dj, _ = self._resolve(mp)
        assert mp['delta_j_hz'] is None
        assert mp['observed_j_hz'] is None
        assert s_dj == pytest.approx(0.25)        # 1 of 2 lines, capped at 0.5

    def test_nothing_to_snap_falls_back_to_the_stored_value(self):
        """No lines and no DB J — there is nothing to reconstruct from."""
        mp = self._mp(j_coupling='', delta_j_hz=0.05)
        s_dj, conf = self._resolve(mp)
        assert s_dj == 1.0 and conf == 1.0

    def test_an_unmeasured_dj_excludes_the_term(self):
        s_dj, conf = self._resolve(self._mp(j_coupling='', delta_j_hz=None))
        assert conf == 0.0

    def test_a_multiplet_with_no_detected_peaks_finds_nothing(self):
        """The DB J lets the pattern be reconstructed, but no peak matches it."""
        mp = self._mp(delta_j_hz=0.05)
        s_dj, _ = self._resolve(mp, peaks=[])
        assert mp['delta_j_hz'] is None
        assert s_dj == 0.0

    def test_remeasuring_ignores_the_stored_counts(self):
        j = 7.0 / 600.0
        lines = [{'ppm': 1.0 - j / 2}, {'ppm': 1.0 + j / 2}]
        peaks = [{'rank': i, 'ppm': l['ppm'], 'height': 1.0}
                 for i, l in enumerate(lines)]
        mp = self._mp(predicted_lines=lines,
                      _n_found_lines=1, _n_exp_lines=2, delta_j_hz=None)
        self._resolve(mp, peaks=peaks, remeasure=True)
        assert mp['_n_found_lines'] == 2
        assert mp['delta_j_hz'] == pytest.approx(0.0, abs=0.2)

    def test_expected_line_count_comes_from_the_db_entry(self):
        """mp['predicted_lines'] may be truncated to the lines that were found;
        asking a truncated list how many to expect concludes none are missing."""
        from processing.annotation_scoring import expected_lines_for
        mp = self._mp(type='q', predicted_lines=[{'ppm': 1.0}])
        met = {'db_peaks': [{'ppm_center': 1.0,
                             'predicted_lines': [{'ppm': p} for p in range(4)]}]}
        assert expected_lines_for(mp, met, mult='q') == 4
