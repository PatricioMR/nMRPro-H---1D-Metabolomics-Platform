# tests/unit/test_multiplet_scorer.py
"""Tests for gui/notation/multiplet_scorer.py"""
import numpy as np
import pytest
from gui.notation.multiplet_scorer import (
    _norm, _parse_j, _offsets_from_pattern, score_multiplet_match,
)


class TestNorm:
    def test_singlet_alias(self):
        assert _norm('singlet') == 's'
        assert _norm('s') == 's'

    def test_doublet_alias(self):
        assert _norm('doublet') == 'd'
        assert _norm('D') == 'd'

    def test_multiplet_alias(self):
        assert _norm('multiplet') == 'm'
        assert _norm('broad singlet') == 's'

    def test_unknown_passthrough(self):
        assert _norm('xyz') == 'xyz'


class TestParseJ:
    def test_single_value(self):
        assert _parse_j('7.2') == [7.2]

    def test_multiple_values(self):
        result = _parse_j('6.8, 13.5')
        assert len(result) == 2
        assert 6.8 in result
        assert 13.5 in result

    def test_zero_excluded(self):
        assert _parse_j('0.0') == []

    def test_none_returns_empty(self):
        assert _parse_j(None) == []

    def test_text_with_hz(self):
        assert _parse_j('J=7.0 Hz') == [7.0]


class TestOffsets:
    def test_singlet_single_line(self):
        offsets = _offsets_from_pattern('s', [], 600.0)
        assert len(offsets) == 1
        assert offsets[0][0] == pytest.approx(0.0)

    def test_doublet_two_lines_symmetric(self):
        offsets = _offsets_from_pattern('d', [7.0], 600.0)
        assert len(offsets) == 2
        pos = [o[0] for o in offsets]
        assert pos[0] == pytest.approx(-pos[1], abs=1e-6)

    def test_doublet_spacing(self):
        j_hz = 7.0
        mhz  = 600.0
        offsets = _offsets_from_pattern('d', [j_hz], mhz)
        spacing = abs(offsets[1][0] - offsets[0][0])
        assert spacing == pytest.approx(j_hz / mhz, rel=1e-3)

    def test_triplet_three_lines(self):
        offsets = _offsets_from_pattern('t', [7.0], 600.0)
        assert len(offsets) == 3

    def test_quartet_four_lines(self):
        offsets = _offsets_from_pattern('q', [7.0], 600.0)
        assert len(offsets) == 4


class TestScoreMultipletMatch:
    """Test the full scorer with synthetic spectra."""

    def _make_result(self, ppm_center, mult_type, j_hz=7.0):
        return {
            'ppm_center': ppm_center,
            'type': mult_type,
            'j_coupling': str(j_hz),
            'multiplicity': mult_type,
        }

    def test_singlet_perfect_match(self, synthetic_singlet):
        d      = synthetic_singlet
        result = self._make_result(d['centre'], 's', j_hz=0)
        score, detail, lines = score_multiplet_match(
            result, d['peaks'], tolerance=0.05, spectrometer_mhz=600.0,
            ppm_array=d['ppm'], intensity_array=d['spec'])
        assert score >= 0.0
        assert score <= 1.0

    def test_doublet_correct_j_scores_higher(self, synthetic_doublet):
        d = synthetic_doublet
        result_correct = self._make_result(d['centre'], 'd', j_hz=d['j_hz'])
        result_wrong   = self._make_result(d['centre'], 'd', j_hz=50.0)

        score_correct, _, _ = score_multiplet_match(
            result_correct, d['peaks'], tolerance=0.05, spectrometer_mhz=d['mhz'],
            ppm_array=d['ppm'], intensity_array=d['spec'])
        score_wrong, _, _ = score_multiplet_match(
            result_wrong, d['peaks'], tolerance=0.05, spectrometer_mhz=d['mhz'],
            ppm_array=d['ppm'], intensity_array=d['spec'])

        assert score_correct >= score_wrong, (
            f'Correct J={d["j_hz"]} Hz should score >= wrong J=50 Hz '
            f'(got {score_correct:.3f} vs {score_wrong:.3f})')

    def test_returns_predicted_lines_for_doublet(self, synthetic_doublet):
        d      = synthetic_doublet
        result = self._make_result(d['centre'], 'd', j_hz=d['j_hz'])
        _, _, lines = score_multiplet_match(
            result, d['peaks'], tolerance=0.05, spectrometer_mhz=d['mhz'])
        assert isinstance(lines, list)

    def test_score_range(self, synthetic_doublet):
        d      = synthetic_doublet
        result = self._make_result(d['centre'], 'd', j_hz=d['j_hz'])
        score, _, _ = score_multiplet_match(
            result, d['peaks'], tolerance=0.05, spectrometer_mhz=d['mhz'])
        assert 0.0 <= score <= 1.0

    def test_far_off_ppm_scores_low(self, synthetic_singlet):
        d      = synthetic_singlet
        result = self._make_result(d['centre'] + 2.0, 's')  # 2 ppm off
        score, _, _ = score_multiplet_match(
            result, d['peaks'], tolerance=0.05, spectrometer_mhz=600.0)
        assert score < 0.3, f'Expected low score for 2 ppm offset, got {score:.3f}'

    def test_partial_quartet_predicted_lines_match_detected_only(
            self, synthetic_quartet_partial):
        """predicted_lines must contain only the 3 experimentally matched lines,
        not all 4 predicted positions — ensures fingerprint visualization is accurate."""
        d = synthetic_quartet_partial
        result = self._make_result(d['centre'], 'q', j_hz=d['j_hz'])
        result['peak_ppm'] = d['centre']
        score, detail, lines = score_multiplet_match(
            result, d['peaks'], tolerance=0.05, spectrometer_mhz=d['mhz'])
        assert score > 0.0, 'Partial quartet must score > 0'
        assert len(lines) == d['n_detected'], (
            f'predicted_lines should have {d["n_detected"]} entries (matched only), '
            f'got {len(lines)}. Detail: {detail}')
        assert '3/4 matched' in detail, f'Detail string missing coverage count: {detail}'
