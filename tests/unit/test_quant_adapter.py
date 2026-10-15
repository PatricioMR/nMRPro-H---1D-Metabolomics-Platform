"""Feeding the engine from what the stages hold.

The only place that knows about stage shapes, so the engine stays testable
without a GUI. What breaks here breaks silently: a wrong matrix integrates to a
perfectly plausible number.
"""
import numpy as np
import pytest

from gui.stages._quant_adapter import (
    build_candidates, quantify_project, sample_groups,
    standard_areas_from_peak)


def _mp(rank, **kw):
    d = {'peak_rank': rank, 'type': 'd', 'delta_j_hz': 0.05, 'ppm_diff': 0.001,
         '_n_found_lines': 2, '_n_exp_lines': 2, 'protons': 3.0,
         'composite_score': 0.8, 'annotation_quality': 'good',
         'predicted_lines': [{'ppm': 1.47}, {'ppm': 1.48}]}
    d.update(kw)
    return d


def _met(mid, name, mps):
    return {'id': mid, 'name': name, 'matched_peaks': mps}


PEAKS = {1: {'rank': 1, 'ppm': 1.47, 'start_ppm': 1.464, 'end_ppm': 1.476},
         2: {'rank': 2, 'ppm': 1.48, 'start_ppm': 1.474, 'end_ppm': 1.486}}


class TestCandidates:
    def test_one_candidate_per_signal(self):
        c = build_candidates([_met('A', 'Ala', [_mp(1), _mp(2)])], PEAKS)
        assert len(c) == 2
        assert {x.met_id for x in c} == {'A'}

    def test_a_bad_signal_never_becomes_a_candidate(self):
        c = build_candidates(
            [_met('A', 'Ala', [_mp(1, annotation_quality='bad')])], PEAKS)
        assert c == []

    def test_the_evidence_is_what_the_ranking_reads(self):
        """Recomputing here would ask a second time and risk a second answer."""
        c = build_candidates([_met('A', 'Ala', [_mp(1)])], PEAKS)[0]
        assert c.evidence['delta_j_hz'] == 0.05
        assert c.evidence['n_found'] == 2
        assert c.evidence['protons'] == 3.0

    def test_the_lines_under_the_signal_come_along(self):
        c = build_candidates([_met('A', 'Ala', [_mp(1)])], PEAKS)[0]
        assert c.line_ranks == {1, 2}

    def test_a_manual_mark_reaches_the_ranking(self):
        c = build_candidates([_met('A', 'Ala', [_mp(1)])], PEAKS,
                             quality_marks={('A', 1): 'good'})[0]
        assert c.evidence['manual_quality'] == 'good'

    def test_a_manual_good_mark_overrides_the_searchs_bad(self):
        c = build_candidates(
            [_met('A', 'Ala', [_mp(1, annotation_quality='bad')])], PEAKS,
            quality_marks={('A', 1): 'good'})
        assert len(c) == 1 and c[0].evidence['manual_quality'] == 'good'

    @staticmethod
    def _pattern(**kw):
        return _mp(1, type='m*', pattern_fit={'snr': 30.0, 'explained': 0.9},
                   peak_ppm=1.47, study_linewidth_hz=1.5,
                   pattern_field_mhz=600.0, **kw)

    def test_an_unmarked_complex_multiplet_goes_to_the_remainders(self):
        pool = []
        out = build_candidates([_met('G', 'Glu', [self._pattern()])], PEAKS,
                               pool=pool)
        assert out == [] and len(pool) == 1

    def test_a_complex_multiplet_marked_good_competes_on_its_peak(self):
        # Glutamate, citrate and serine marked Good on the Mice ISS study were
        # still sent to the remainders, and with the rank dropped first the
        # mark never reached the evidence
        pool = []
        out = build_candidates([_met('G', 'Glu', [self._pattern()])], PEAKS,
                               quality_marks={('G', 1): 'good'}, pool=pool)
        assert pool == [] and len(out) == 1
        c = out[0]
        assert c.peak_rank == 1
        assert c.evidence['manual_quality'] == 'good'
        assert c.evidence['shape_fit'] is not None

    def test_the_display_name_travels_as_payload(self):
        c = build_candidates([_met('A', 'Ala', [_mp(1)])], PEAKS)[0]
        assert c.payload['name'] == 'Ala'

    def test_a_metabolite_without_an_id_is_skipped(self):
        assert build_candidates([{'matched_peaks': [_mp(1)]}], PEAKS) == []


class TestTheStandard:
    def test_it_is_integrated_exactly_as_an_analyte(self):
        """A standard measured differently from what it standardises would
        carry the difference into every concentration."""
        ppm = np.linspace(1.40, 1.60, 3000)
        y = np.exp(-((ppm - 1.47) ** 2) / (2 * 0.002 ** 2))
        X = np.column_stack([y, 2 * y])
        areas = standard_areas_from_peak(PEAKS[1], ppm, X,
                                         local_baseline=False)
        assert areas[1] == pytest.approx(2 * areas[0], rel=1e-3)

    def test_no_reference_peak_is_no_standard(self):
        assert standard_areas_from_peak(None, np.linspace(0, 1, 10),
                                        np.zeros((10, 2))) == {}


class TestGroups:
    def test_samples_are_grouped_as_labelled(self):
        info = [{'label': 'FL'}, {'label': 'HC'}, {'label': 'FL'}]
        assert sample_groups(info) == {'FL': [0, 2], 'HC': [1]}

    def test_an_unlabelled_sample_still_belongs_somewhere(self):
        assert sample_groups([{}]) == {'Unknown': [0]}


class TestWholeProject:
    def _project(self, n=3):
        ppm = np.linspace(1.40, 1.60, 3000)
        y = (np.exp(-((ppm - 1.47) ** 2) / (2 * 0.002 ** 2))
             + np.exp(-((ppm - 1.48) ** 2) / (2 * 0.002 ** 2)))
        X = np.column_stack([y * (i + 1) for i in range(n)])
        return ppm, X

    def test_it_returns_rows_the_window_can_read(self):
        ppm, X = self._project()
        rows = quantify_project(
            metabolites=[_met('A', 'Ala', [_mp(1)])],
            peaks=list(PEAKS.values()), ppm=ppm, intensities=X,
            samples_info=[{'label': 'G'}] * 3)
        assert len(rows) == 1
        # The display name, not the id. The engine works in ids because two
        # analytes can share a name; the rows that leave here are read by the
        # table, the statistics matrix and the export, and those show names.
        assert rows[0]['metabolite'] == 'Ala'
        assert rows[0]['hmdb_id'] == 'A'
        assert rows[0]['conc_mm'] > 0
        assert rows[0]['_integ_mode'] == 'Multiplet'
        assert rows[0]['_peak'] is not None, 'the detected peak travels too'

    def test_the_snr_shown_is_the_signals_on_the_group_the_figure_draws(self):
        """One trace for the picture and the numbers: the elected signal's
        SNR and shape agreement are read on each group's mean, and the row
        names the group it shows -- best cosine, then highest SNR."""
        ppm, X = self._project()
        rows = quantify_project(
            metabolites=[_met('A', 'Ala', [_mp(1)])],
            peaks=list(PEAKS.values()), ppm=ppm, intensities=X,
            samples_info=[{'label': 'A'}, {'label': 'B'}, {'label': 'B'}],
            noise_sigma=0.01)
        r = rows[0]
        assert set(r['_snr_by_group']) == {'A', 'B'}
        assert r['_snr_group'] == 'B', 'B is 2.5x A on the same shape'
        assert r['snr'] == r['_snr_by_group']['B']
        assert r['_snr_by_group']['B'] > r['_snr_by_group']['A']
        assert all(0 < c <= 1 for c in r['_cosine_by_group'].values())

    def test_the_reference_pattern_travels_to_the_row(self):
        """Every line the reference lists, at the centre the annotation
        found: the plot draws them beside the lines integrated, so a doublet
        fused into one apex still shows where its two lines are."""
        ppm, X = self._project()
        mp = _mp(1, reference_lines=[{'ppm': 1.47, 'rel_intensity': 1.0},
                                     {'ppm': 1.48, 'rel_intensity': 1.0}])
        rows = quantify_project(
            metabolites=[_met('A', 'Ala', [mp])],
            peaks=list(PEAKS.values()), ppm=ppm, intensities=X,
            samples_info=[{'label': 'G'}] * 3)
        assert rows[0]['_reference_lines'] == [1.47, 1.48]

    def test_blanks_are_dropped_by_the_same_rule_detection_uses(self):
        """Three engines must not disagree about which samples a study has."""
        ppm, X = self._project()
        rows = quantify_project(
            metabolites=[_met('A', 'Ala', [_mp(1)])],
            peaks=list(PEAKS.values()), ppm=ppm, intensities=X,
            samples_info=[{'type': 'BLK'}, {'label': 'G'}, {'label': 'G'}])
        concs = rows[0]['_concs_per_sample']
        assert len(concs) == 3, 'one entry per sample, blanks included'
        assert np.isnan(concs[0]), 'the blank was not measured'
        assert all(np.isfinite(v) for v in concs[1:])

    def test_a_one_dimensional_matrix_is_one_sample(self):
        ppm, X = self._project(n=1)
        rows = quantify_project(
            metabolites=[_met('A', 'Ala', [_mp(1)])],
            peaks=list(PEAKS.values()), ppm=ppm, intensities=X[:, 0],
            samples_info=[{'label': 'G'}])
        concs = rows[0]['_concs_per_sample']
        assert len(concs) == 1 and np.isfinite(concs[0])


class TestComplexMultipletsCompeting:
    """quant_engine.M_STAR_COMPETE_MIN_GOOD: an m* of an analyte with enough
    confirmed m* signals competes for its peak; off, none does (ADR-024)."""

    @staticmethod
    def _met(n_good):
        mps = [_mp(r, type='m*', annotation_quality='good' if r <= n_good else 'bad',
                   pattern_fit={'snr': 30.0}, peak_ppm=1.47, study_linewidth_hz=1.5,
                   pattern_field_mhz=600.0) for r in (1, 2)]
        return _met('G', 'Glu', mps)

    def test_off_no_complex_multiplet_competes(self, monkeypatch):
        from processing import quant_engine as qe
        monkeypatch.setattr(qe, 'M_STAR_COMPETE_MIN_GOOD', None)
        assert build_candidates([self._met(2)], PEAKS, pool=[]) == []

    def test_with_enough_confirmed_ones_they_compete_on_their_peaks(self, monkeypatch):
        from processing import quant_engine as qe
        monkeypatch.setattr(qe, 'M_STAR_COMPETE_MIN_GOOD', 2)
        out = build_candidates([self._met(2)], PEAKS, pool=[])
        assert sorted(c.peak_rank for c in out) == [1, 2]
        assert all(c.evidence['mstar_compete'] for c in out)

    def test_one_confirmed_is_not_enough(self, monkeypatch):
        from processing import quant_engine as qe
        monkeypatch.setattr(qe, 'M_STAR_COMPETE_MIN_GOOD', 2)
        assert build_candidates([self._met(1)], PEAKS, pool=[]) == []

