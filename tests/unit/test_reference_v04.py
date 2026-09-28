"""v04: a reference value is corrected only where the spectra BMRB deposited
agree with each other and the one v03 measured on is among them."""
import json
import sys

import pytest

sys.path.insert(0, 'hmdb_api_download')
import build_reference_v04 as B  # noqa: E402


def _read(set_, centre, lines, snr=500.0, field=500.0):
    # as the build names a set: <entry mirrored>/<entry deposited>/<experiment>
    return {'set': f'bmse000120/bmse000120/{set_}', 'field': field,
            'status': 'ok' if len(lines) > 1 else 'single', 'centre': centre,
            'lines': lines, 'n': len(lines), 'snr': snr}


TRIPLET = [(-6.6, 0.5), (0.0, 1.0), (6.6, 0.5)]


def _row(mult='d', ppm_meas=3.4059, meas_set='bmse000120/1H'):
    return {'id': 1, 'multiplicity': mult, 'ppm': 3.41, 'ppm_meas': ppm_meas,
            'meas_set': meas_set}


class TestConsensus:
    def test_two_spectra_that_agree_are_a_consensus(self):
        c = B.consensus([_read('1H', 3.4102, TRIPLET),
                         _read('1H.5/1', 3.4110, TRIPLET)])
        assert c is not None and c[0] == 3
        assert c[1] == pytest.approx(3.4106)

    def test_one_spectrum_is_not(self):
        assert B.consensus([_read('1H', 3.4102, TRIPLET)]) is None

    def test_a_reading_of_something_else_neither_votes_nor_vetoes(self):
        far = _read('1H_2', 3.3517, [(-3, 1.0), (3, 0.9)])
        c = B.consensus([_read('1H', 3.4102, TRIPLET),
                         _read('1H.5/1', 3.4110, TRIPLET), far])
        assert c is not None and c[0] == 3

    def test_a_weak_or_j_resolved_reading_does_not_count(self):
        assert B.consensus([_read('1H', 3.4102, TRIPLET),
                            _read('1H.5/1', 3.4110, TRIPLET, snr=5)]) is None
        assert B.consensus([_read('1H', 3.4102, TRIPLET),
                            _read('x/1H_1H_JRES', 3.4129, [(0, 1.0)])]) is None


class TestFirstOrder:
    def test_a_binomial_triplet_is_a_triplet(self):
        mult, j = B.first_order([_read('a', 3.41, TRIPLET),
                                 _read('b', 3.41, TRIPLET)])
        assert mult == 't' and j == pytest.approx(6.6)

    def test_unequal_spacings_are_not_read_as_a_triplet(self):
        odd = [(-8.0, 0.5), (0.0, 1.0), (5.0, 0.5)]
        assert B.first_order([_read('a', 3.41, odd),
                              _read('b', 3.41, odd)]) is None


class TestVerdict:
    def _reads(self):
        return [_read('1H', 3.4102, TRIPLET), _read('1H.5/1', 3.4110, TRIPLET)]

    def test_a_label_its_own_spectrum_contradicts_is_corrected(self):
        st, ch, _ev = B.verdict(_row(), self._reads(), 500.0)
        assert st == 'corrected'
        assert ch['mult_meas'] == 't'
        assert ch['ppm_meas'] == pytest.approx(3.4106)

    def test_other_depositions_disagreeing_is_a_condition_not_an_error(self):
        st, ch, _ev = B.verdict(_row(meas_set='bmse000999/1H'),
                                self._reads(), 500.0)
        assert st == 'review' and ch == {}

    def test_agreement_is_confirmed(self):
        st, ch, _ev = B.verdict(_row(mult='t', ppm_meas=3.4105),
                                self._reads(), 500.0)
        assert st == 'confirmed' and ch == {}

    def test_a_large_move_is_left_for_review(self):
        st, _ch, _ev = B.verdict(_row(ppm_meas=3.3800), self._reads(), 500.0)
        assert st == 'review'

    def test_a_generic_m_is_made_specific(self):
        st, ch, _ev = B.verdict(_row(mult='m', ppm_meas=3.4105),
                                self._reads(), 500.0)
        assert st == 'corrected' and ch['mult_meas'] == 't'


def test_the_profile_follows_the_corrected_position():
    prof = json.dumps({'start': -12.0, 'step': 0.1, 'h': [0, 1, 0]})
    moved = json.loads(B.reregister_profile(prof, 3.4059, 3.4102, 500.0))
    # the same trace, now quoted from 3.4102: 2.15 Hz further left
    assert moved['start'] == pytest.approx(-12.0 - 0.0043 * 500.0)
    assert moved['h'] == [0, 1, 0]
