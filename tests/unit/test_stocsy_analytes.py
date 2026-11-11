# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy_analytes.py
"""The STOCSY window by analyte: signals expected and found (a multiplet whole,
its lines added), the signals set against what the quantification measured,
a signal read line by line (each line with its share of the protons, the
ticked ones whole), and the numbers sample by sample -- on a study whose
molecules are known."""
import numpy as np
import pytest

from controllers.stocsy_analytes import AnalyteBrowser
from controllers.stocsy_controller import StocsyAnalysis

N = 14


def _study(seed=0):
    rng = np.random.default_rng(seed)
    a = 1000 * np.exp(rng.normal(0, 0.4, N))               # molecule A, one concentration
    b = 800 * np.exp(rng.normal(0, 0.4, N))                # molecule B
    c = 900 * np.exp(rng.normal(0, 0.4, N))                # molecule C
    noise = lambda: np.exp(rng.normal(0, 0.02, N))
    areas = {1: (3 * a * noise(), 1.300, 90.0),            # A: 3 H singlet
             2: (1 * a * noise(), 4.095, 60.0),            # A: 2 H doublet, line one
             8: (1 * a * noise(), 4.105, 55.0),            #               line two
             3: (2 * a * noise(), 2.050, 40.0),            # A: 2 H singlet
             4: (3 * b * noise(), 1.330, 70.0),            # B beside A's singlet
             5: (3 * c * noise(), 1.270, 30.0),            # C beside it too
             9: (1.5 * a * noise(), 1.280, 25.0),          # half the area A's 3 H would give
             10: (1 * b * noise(), 4.115, 30.0),           # B's line beside A's doublet
             6: (np.full(N, 50.0), 1.310, 20.0),           # flat: cannot vary
             7: (900 * np.exp(rng.normal(0, 0.4, N)), 6.5, 50.0)}
    peaks = [{'rank': k, 'ppm': ppm, 'snr': snr, 'start_idx': 100 * k, 'end_idx': 100 * k + 40,
              'area_per_sample': v} for k, (v, ppm, snr) in areas.items()]
    next(p for p in peaks if p['rank'] == 4)['assigned_metabolite'] = 'Other'
    info = [{'index': i, 'label': 'G1' if i < N // 2 else 'G2', 'type': 'Sample', 'name': f'S{i}'}
            for i in range(N)]
    ppm = np.linspace(10.0, -0.5, 4000)
    X = np.random.default_rng(seed + 1).normal(0, 1.0, (ppm.size, N))
    return StocsyAnalysis(ppm, X, peaks, info, noise_std=1.0)


def detection_areas(an):
    """``measure`` for the synthetic study, whose spectrum is noise: the
    detection areas of the peaks added up -- the scale its quantification
    rows are written in."""
    def measure(peaks, _evidence, _local_baseline):
        ranks = [int(p['rank']) for p in peaks]
        A = [np.asarray(an.peaks[r]['area_per_sample'], float)[an.included] for r in ranks]
        sd = float(np.sqrt(sum(an.area_sigma[an._row[r]] ** 2 for r in ranks)))
        return np.sum(A, axis=0), sd, ''
    return measure


def _ab(an, *a, **k):
    return AnalyteBrowser(an, *a, measure=detection_areas(an), **k)


def _mp(rank, db, h, t, lines, q='good'):
    return {'peak_rank': rank, 'db_ppm': db, 'protons': h, 'type': t, 'annotation_quality': q,
            'predicted_lines': [{'ppm': x, 'measured': True} for x in lines]}


def _mets():
    return [
        {'id': 'MA', 'name': 'A-like', 'composite_score': 0.9,
         'matched_peaks': [_mp(1, 1.30, 3, 's', [1.300]), _mp(2, 4.10, 2, 'd', [4.095, 4.105]),
                           _mp(3, 2.05, 2, 's', [2.050], 'bad')],
         'db_peaks': [{'ppm_center': 1.30, 'protons': 3}, {'ppm_center': 4.10, 'protons': 2},
                      {'ppm_center': 2.05, 'protons': 2}, {'ppm_center': 7.90, 'protons': 2}]},
        {'id': 'MB', 'name': 'Other', 'composite_score': 0.5,
         'matched_peaks': [_mp(4, 1.33, 3, 's', [1.330]), _mp(7, 6.5, 1, 's', [6.5])],
         'db_peaks': []},
        {'id': 'MZ', 'name': 'No matches', 'composite_score': 0.9, 'matched_peaks': [], 'db_peaks': []},
    ]


def _browser(quant_rows=None, **kw):
    rows = [{'hmdb_id': 'MA', '_confidence': 3}, {'hmdb_id': 'MB', '_confidence': 1},
            {'hmdb_id': 'IS', '_is_is': True, '_confidence': 3}] if quant_rows is None else quant_rows
    return _ab(_study(**kw), _mets(), rows)


def _index(b, met, ppm):
    return next(i for i, s in enumerate(b.signals(met)) if abs(s['db_ppm'] - ppm) < 1e-6)


class TestAnalytes:

    def test_an_analyte_without_a_matched_peak_is_not_listed(self):
        assert [a['id'] for a in _browser().analytes()] == ['MA', 'MB']

    def test_the_quantified_with_most_dots_come_first(self):
        assert [a['dots'] for a in _browser().analytes()] == [3, 1]

    def test_only_quantified_and_the_name_filter(self):
        b = _browser(quant_rows=[{'hmdb_id': 'MB', '_confidence': 2}])
        assert [a['id'] for a in b.analytes(only_quantified=True)] == ['MB']
        assert [a['id'] for a in b.analytes(text='a-li')] == ['MA']

    def test_the_standard_and_the_refused_carry_no_dots(self):
        b = _browser(quant_rows=[{'hmdb_id': 'MA', '_confidence': 3, '_not_quantified': True}])
        assert b.analytes()[0]['dots'] == 0 and not b.has_quantification

    def test_a_study_without_annotation_lists_nothing(self):
        assert not _ab(_study(), None, None).has_annotation

    def test_the_summary_counts_signals_found_and_the_pairs_the_dot_rule_reads(self):
        a = _browser().analytes(text='a-li')[0]
        assert (a['detected'], a['expected']) == (3, 4)
        assert a['pairs'] == 3 and a['mean_r'] > 0.9

    def test_one_detected_signal_has_no_mean_r(self):
        b = _ab(_study(), [{'id': 'M1', 'name': 'x', 'matched_peaks': [
            _mp(1, 1.3, 3, 's', [1.3])], 'db_peaks': []}])
        assert np.isnan(b.analytes()[0]['mean_r']) and b.analytes()[0]['pairs'] == 0


class TestSignals:

    def test_signals_are_listed_high_delta_first_and_the_unfound_one_says_so(self):
        s = _browser().signals('MA')
        assert [round(x['db_ppm'], 2) for x in s] == [7.90, 4.10, 2.05, 1.30]
        assert not s[0]['detected'] and s[0]['rank'] is None and s[0]['why'] == 'not detected'
        assert s[0]['protons'] == 2 and not s[0]['usable']

    def test_a_multiplet_is_its_lines_not_its_anchor(self):
        d = _browser().signals('MA')[1]
        assert sorted(d['lines']) == [2, 8] and d['type'] == 'd' and d['protons'] == 2
        assert d['ppm'] == pytest.approx(4.10) and d['snr'] == 60.0

    def test_a_signal_carries_the_annotation_verdict(self):
        s = _browser().signals('MA')
        assert s[2]['quality'] == 'bad' and s[3]['quality'] == 'good' and s[3]['usable']

    def test_a_signal_whose_peak_does_not_vary_is_listed_with_the_reason(self):
        b = _ab(_study(), [{'id': 'F', 'name': 'flat', 'matched_peaks': [
            _mp(6, 1.31, 1, 's', [1.31])], 'db_peaks': []}])
        s = b.signals('F')[0]
        assert s['detected'] and not s['usable'] and s['why'] and s['lines'] == []

    def test_the_default_reference_is_the_clearest_singlet(self):
        b = _browser()
        assert b.default_reference('MA') == _index(b, 'MA', 1.30)
        assert b.default_reference('unknown') is None

    def test_a_doublet_is_the_reference_only_when_no_singlet_can_be(self):
        b = _ab(_study(), [{'id': 'D', 'name': 'd', 'matched_peaks': [
            _mp(2, 4.10, 2, 'd', [4.095, 4.105])], 'db_peaks': []}])
        assert b.default_reference('D') == 0


class TestTargets:

    def test_a_signal_is_the_sum_of_its_lines(self):
        b = _browser()
        t = b.signal_target('MA', _index(b, 'MA', 4.10))
        assert t.kind == 'signal' and sorted(t.ranks) == [2, 8] and t.protons == 2
        a = lambda r: np.asarray(b.an.peaks[r]['area_per_sample'], float)
        assert np.allclose(t.vector, a(2) + a(8))
        assert t.sigma == pytest.approx(np.hypot(*(b.an.area_sigma[b.an._row[r]] for r in (2, 8))))

    def test_a_signal_not_detected_has_no_target(self):
        b = _browser()
        assert b.signal_target('MA', _index(b, 'MA', 7.90)) is None

    def test_a_peak_that_does_not_vary_is_a_stub_with_the_reason(self):
        t = _browser().peak_target(6, 3)
        assert t.vector is None and t.why and t.protons == 3

    def test_a_peak_carries_the_protons_it_is_read_with(self):
        t = _browser().peak_target(1, 2)
        assert t.protons == 2 and t.ranks == [1] and t.anchor == 1


class TestSignalsAgainstTheReference:

    def _rows(self):
        b = _browser()
        return b, dict(b.signals_vs('MA', _index(b, 'MA', 1.30)))

    def test_every_signal_of_the_molecule_quantifies_like_the_reference(self):
        b, rows = self._rows()
        for ppm in (4.10, 2.05):
            r = rows[_index(b, 'MA', ppm)]
            assert abs(r['gap']) < 0.08 and r['distance'] < 0.15 and r['r'] > 0.9

    def test_the_multiplet_is_compared_whole(self):
        """One line of the doublet alone would read as half of what 2 H calls for."""
        b, rows = self._rows()
        doublet = rows[_index(b, 'MA', 4.10)]
        assert doublet['ratio'] == pytest.approx(2 / 3, rel=0.05)
        line = b.compare(b.signal_target('MA', _index(b, 'MA', 1.30)),
                         [b.peak_target(2, 2)])[0]
        assert line['gap'] == pytest.approx(-0.5, abs=0.08)

    def test_the_reference_row_is_a_ratio_of_one_at_zero_distance(self):
        b, rows = self._rows()
        ref = rows[_index(b, 'MA', 1.30)]
        assert ref['is_ref'] and ref['ratio'] == 1.0 and ref['distance'] == 0.0

    def test_a_signal_not_detected_has_no_row(self):
        b, rows = self._rows()
        assert rows[_index(b, 'MA', 7.90)] is None

    def test_a_reference_that_cannot_be_read_gives_nothing(self):
        assert _browser().signals_vs('MA', _index(_browser(), 'MA', 7.90)) == []


class TestQuantTarget:

    def test_the_reference_is_what_the_quantification_measured(self):
        b = _browser(quant_rows=_quant_rows())
        t = b.quant_target('MA')
        assert t.kind == 'quant' and t.protons == 7 and 'Cascade' in t.label
        assert np.allclose(t.vector, _a7(b))
        assert np.isfinite(t.sigma) and t.sigma > 0

    def test_the_areas_follow_the_samples_the_quantification_kept(self):
        rows = _quant_rows()
        full = np.asarray(rows[0]['_areas_per_sample'], float)
        rows[0]['_included'] = list(range(1, N))
        rows[0]['_areas_per_sample'] = list(full[1:])
        t = _browser(quant_rows=rows).quant_target('MA')
        assert np.isnan(t.vector[0]) and np.allclose(t.vector[1:], full[1:])

    def test_without_areas_or_protons_there_is_no_quantified_reference(self):
        assert _browser().quant_target('MA') is None
        rows = _quant_rows()
        rows[0]['_nh'] = None
        assert _browser(quant_rows=rows).quant_target('MA') is None
        assert _browser(quant_rows=_quant_rows()).quant_target('MB') is None

    def test_every_signal_of_the_molecule_quantifies_like_what_was_measured(self):
        b = _browser(quant_rows=_quant_rows())
        rows = dict(b.signals_vs('MA', 'q', b.quant_target('MA')))
        for ppm in (4.10, 2.05, 1.30):
            r = rows[_index(b, 'MA', ppm)]
            assert abs(r['gap']) < 0.08 and r['r'] > 0.9


def _a7(b):
    """What a quantification of A over its 7 H would give in every sample."""
    return np.asarray(b.an.peaks[1]['area_per_sample'], float) / 3 * 7


def _quant_rows():
    b = _browser()
    return [{'hmdb_id': 'MA', '_confidence': 3, '_areas_per_sample': list(_a7(b)), '_nh': 7.0,
             '_included': list(range(N)), '_integ_mode': 'Cascade', '_multiplicity': 's',
             'ppm': 1.30, 'snr': 90.0, '_multiplet_window': [1.29, 1.31],
             '_line_ppms': [1.30, 4.10]},
            {'hmdb_id': 'MB', '_confidence': 1}]


class TestLineRows:

    def _go(self, ticked=(2, 8), hw=0.03):
        b = _browser(quant_rows=_quant_rows())
        ref = b.quant_target('MA')
        s = b.signals('MA')[_index(b, 'MA', 4.10)]
        return b, ref, b.line_rows(ref, s, list(ticked), hw)

    def test_the_signal_the_ticked_lines_make_comes_first_and_quantifies_like_the_reference(self):
        _b, _ref, rows = self._go()
        assert rows[0]['kind'] == 'whole' and rows[0]['label'] == 'ticked lines (2)'
        assert rows[0]['protons'] == 2 and abs(rows[0]['gap']) < 0.08

    def test_the_peaks_within_the_half_width_follow_high_delta_first(self):
        _b, _ref, rows = self._go()
        assert [r['rank'] for r in rows[1:]] == [10, 8, 2]

    def test_each_line_of_a_doublet_holds_half_its_protons(self):
        _b, _ref, rows = self._go()
        by = {r['rank']: r for r in rows[1:]}
        for k in (2, 8):
            assert by[k]['share'] == pytest.approx(0.5) and by[k]['protons'] == pytest.approx(1.0)
            assert abs(by[k]['gap']) < 0.08 and by[k]['r'] > 0.9 and by[k]['ticked']

    def test_another_compounds_peak_beside_the_lines_has_no_share_and_does_not_covary(self):
        _b, _ref, rows = self._go()
        foreign = next(r for r in rows if r['rank'] == 10)
        assert foreign['share'] is None and foreign['protons'] is None
        assert np.isnan(foreign['gap']) and foreign['r'] < 0.7 and not foreign['ticked']

    def test_a_signal_of_one_line_of_its_doublet_reads_half(self):
        _b, _ref, rows = self._go(ticked=(2,))
        assert rows[0]['gap'] == pytest.approx(-0.5, abs=0.08)

    def test_ticking_the_foreign_peak_moves_the_signal_away_from_the_reference(self):
        _b, _ref, good = self._go()
        _b, _ref, bad = self._go(ticked=(2, 8, 10))
        assert bad[0]['distance'] > 3 * good[0]['distance']

    def test_with_nothing_ticked_there_is_no_whole_row(self):
        _b, _ref, rows = self._go(ticked=())
        assert all(r['kind'] == 'peak' for r in rows)

    def test_a_peak_that_does_not_vary_is_kept_with_the_reason(self):
        b = _browser(quant_rows=_quant_rows())
        s = b.signals('MA')[_index(b, 'MA', 1.30)]
        rows = b.line_rows(b.quant_target('MA'), s, s['lines'], 0.2)
        flat = next(r for r in rows if r['rank'] == 6)
        assert not flat['usable'] and flat['why'] and np.isnan(flat['distance'])

    def test_the_name_the_annotation_gave_a_peak_travels_with_it(self):
        b = _browser(quant_rows=_quant_rows())
        s = b.signals('MA')[_index(b, 'MA', 1.30)]
        rows = b.line_rows(b.quant_target('MA'), s, s['lines'], 0.2)
        assert next(r for r in rows if r['rank'] == 4)['assigned'] == 'Other'

    def test_the_same_input_gives_the_same_rows(self):
        a = [(r['label'], r['distance']) for r in self._go()[2]]
        b = [(r['label'], r['distance']) for r in self._go()[2]]
        assert a == b

    def test_without_an_analyte_a_peak_is_read_with_the_protons_the_analyst_gives(self):
        b = _browser()
        ref = b.peak_target(3, 2)
        rows = b.line_rows(ref, b.peak_signal(1, 3), [1], 0.005)
        assert rows[0]['kind'] == 'whole' and rows[0]['distance'] < 0.12

    def test_a_line_holds_the_share_the_quantification_weighs_it_by(self):
        b = _browser()
        q = {'evidence': {'predicted_ppms': [4.115, 4.105, 4.095, 4.085], 'line_weights': None}}
        assert b.pattern_share(q, [8]) == pytest.approx(3 / 8)              # binomial 1:3:3:1
        m = {'evidence': {'predicted_ppms': [4.115, 4.105, 4.095, 4.085],
                          'line_weights': [1.0, 1.0, 2.0, 0.0]}}
        assert b.pattern_share(m, [8]) == pytest.approx(1 / 4)              # its own heights

    def test_the_ticked_lines_say_how_much_of_the_pattern_they_hold(self):
        _b, _ref, rows = self._go(ticked=(2,))
        assert rows[0]['share'] == pytest.approx(0.5)
        _b, _ref, rows = self._go()
        assert rows[0]['share'] == pytest.approx(1.0)


class TestGrid:

    def _rows(self):
        b = _browser(quant_rows=_quant_rows())
        ref = b.quant_target('MA')
        s = b.signals('MA')[_index(b, 'MA', 1.30)]
        return b, ref, b.line_rows(ref, s, s['lines'], 0.2)

    def test_samples_are_grouped_and_the_names_follow(self):
        b, _ref, rows = self._rows()
        g = b.grid(rows)
        assert g['groups'] == ['G1'] * 7 + ['G2'] * 7 and g['names'][0] == 'S0'
        assert len(g['values']) == len(rows) and len(g['values'][0]) == N

    def test_the_area_view_is_the_raw_numbers(self):
        b, _ref, rows = self._rows()
        g = b.grid(rows)
        assert g['values'][0] == [float(x) for x in b.an.peaks[1]['area_per_sample']]

    def test_per_proton_makes_the_signal_that_fits_read_like_the_reference(self):
        b, ref, rows = self._rows()
        both = [rows[0], b.compare(ref, [ref])[0]]
        v = np.array(b.grid(both, per='proton')['values'])
        assert np.allclose(v[0], v[1], rtol=0.1)

    def test_a_row_without_a_vector_has_no_numbers(self):
        b, _ref, rows = self._rows()
        flat = next(r for r in rows if r['rank'] == 6)
        assert np.all(np.isnan(b.grid([flat])['values'][0]))

    def test_a_row_without_protons_has_no_per_proton_value(self):
        b = _browser()
        ref = b.peak_target(3, 2)
        row = b.compare(ref, [b.peak_target(1, None)])[0]
        assert np.all(np.isnan(b.grid([row], per='proton')['values'][0]))

    def test_the_export_has_one_row_per_line_with_every_sample_and_whether_it_is_in(self):
        b, ref, rows = self._rows()
        out = b.export_rows(rows, ref)
        assert len(out) == len(rows)
        assert sum(1 for k in out[0] if k.startswith('S')) == N
        assert out[0]['reference'] == ref.label and out[0]['in_signal'] is True


class TestComplexMultiplet:
    """A measured pattern (m*) is read as the region its lines span, every
    peak in it a line: line by line its close lines fell on one apex."""

    def _pattern(self, lines):
        mets = _mets()
        mets[0]['matched_peaks'][1].update(type='m*', predicted_lines=[{'ppm': p} for p in lines])
        return _ab(_study(), mets, [{'hmdb_id': 'MA', '_confidence': 3}])

    def test_every_peak_in_the_span_of_its_lines_is_a_line(self):
        b = self._pattern([4.116, 4.104, 4.099, 4.096])               # four lines over three peaks
        s = b.signals('MA')[_index(b, 'MA', 4.10)]
        assert s['lines'] == [10, 8, 2] and s['region'] == pytest.approx((4.118, 4.094))

    def test_a_peak_outside_the_span_is_not_taken(self):
        b = self._pattern([4.106, 4.094])
        s = b.signals('MA')[_index(b, 'MA', 4.10)]
        assert sorted(s['lines']) == [2, 8]

    def test_a_letter_has_no_region(self):
        b = _browser()
        assert all(s['region'] is None for s in b.signals('MA'))


class TestLetterLines:
    """A letter shows no more lines than it has, the ones nearest where the
    signal stands; the annotation's own list is left as it is."""

    def _crowded(self, n_expected):
        mets = _mets()
        mp = mets[0]['matched_peaks'][1]                          # the doublet at 4.10
        mp.update(peak_ppm=4.10, _n_exp_lines=n_expected,
                  predicted_lines=[{'ppm': p} for p in (4.115, 4.105, 4.095)])
        return _ab(_study(), mets, [{'hmdb_id': 'MA', '_confidence': 3}]), mp

    def test_a_doublet_handed_three_lines_shows_the_two_nearest(self):
        b, mp = self._crowded(2)
        s = b.signals('MA')[_index(b, 'MA', 4.10)]
        assert s['lines'] == [8, 2] and s['stored_lines'] == 3
        assert len(mp['predicted_lines']) == 3                    # the annotation is not touched

    def test_a_letter_with_its_own_count_is_left_alone(self):
        b, _mp = self._crowded(3)
        s = b.signals('MA')[_index(b, 'MA', 4.10)]
        assert s['lines'] == [10, 8, 2] and s['stored_lines'] is None
