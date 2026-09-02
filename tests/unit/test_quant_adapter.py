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
        assert rows[0]['metabolite'] == 'A'
        assert rows[0]['conc_mm'] > 0
        assert rows[0]['_integ_mode'] == 'Multiplet'

    def test_blanks_are_dropped_by_the_same_rule_detection_uses(self):
        """Three engines must not disagree about which samples a study has."""
        ppm, X = self._project()
        rows = quantify_project(
            metabolites=[_met('A', 'Ala', [_mp(1)])],
            peaks=list(PEAKS.values()), ppm=ppm, intensities=X,
            samples_info=[{'type': 'BLK'}, {'label': 'G'}, {'label': 'G'}])
        assert set(rows[0]['_concs_per_sample']) == {1, 2}

    def test_a_one_dimensional_matrix_is_one_sample(self):
        ppm, X = self._project(n=1)
        rows = quantify_project(
            metabolites=[_met('A', 'Ala', [_mp(1)])],
            peaks=list(PEAKS.values()), ppm=ppm, intensities=X[:, 0],
            samples_info=[{'label': 'G'}])
        assert set(rows[0]['_concs_per_sample']) == {0}
