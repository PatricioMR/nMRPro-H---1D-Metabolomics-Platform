# tests/unit/test_evidence_weighting.py
"""Unit tests for gui.notation.evidence_weighting — dynamic_composite."""
import pytest
from gui.notation.evidence_weighting import (
    dynamic_composite,
    j_confidence,
    alignment_ppm_confidence,
    _BASE_WEIGHTS_SEARCH,
    _BASE_WEIGHTS_ENRICHMENT_NS,
    _BASE_WEIGHTS_ENRICHMENT_S,
)


class TestDynamicComposite:

    def test_all_components_present_equals_static(self):
        """With all confidences=1 the result equals the static formula."""
        s = {'ppm': 0.9, 'r2': 0.8, 'mult': 0.7, 'j': 0.6}
        c = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 1.0}
        expected = (0.45*0.9 + 0.30*0.8 + 0.15*0.7 + 0.10*0.6)
        assert abs(dynamic_composite(s, c) - expected) < 1e-9

    def test_j_excluded_renormalises(self):
        """When j_conf=0, J is excluded and remaining weights renormalise to 1."""
        s = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 0.0}
        c = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 0.0}
        # Renormalised: ppm=0.45, r2=0.30, mult=0.15 → total=0.90
        expected = (0.45*1.0 + 0.30*1.0 + 0.15*1.0) / 0.90
        assert abs(dynamic_composite(s, c) - expected) < 1e-9

    def test_j_excluded_score_above_static_max(self):
        """With j excluded, perfect ppm/r2/mult should give score > 0.9 (not capped)."""
        s = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 0.0}
        c = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 0.0}
        score = dynamic_composite(s, c)
        assert score == pytest.approx(1.0)

    def test_all_components_excluded_returns_zero(self):
        """If all confidences=0, score is 0 (no evidence)."""
        s = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 1.0}
        c = {'ppm': 0.0, 'r2': 0.0, 'mult': 0.0, 'j': 0.0}
        assert dynamic_composite(s, c) == 0.0

    def test_r2_excluded_when_no_spectrum(self):
        """r2_conf=0 removes R² from composite — ppm+mult+j renormalise."""
        s = {'ppm': 0.9, 'r2': 0.0, 'mult': 0.8, 'j': 0.7}
        c = {'ppm': 1.0, 'r2': 0.0, 'mult': 1.0, 'j': 1.0}
        total_w = 0.45 + 0.15 + 0.10   # r2 excluded
        expected = (0.45*0.9 + 0.15*0.8 + 0.10*0.7) / total_w
        assert abs(dynamic_composite(s, c) - expected) < 1e-9

    def test_partial_j_confidence_interpolates(self):
        """Intermediate j_conf scales j weight, rest renormalise."""
        s = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 1.0}
        c = {'ppm': 1.0, 'r2': 1.0, 'mult': 1.0, 'j': 0.5}
        j_eff = 0.10 * 0.5
        total_w = 0.45 + 0.30 + 0.15 + j_eff
        expected = (0.45 + 0.30 + 0.15 + j_eff) / total_w
        assert abs(dynamic_composite(s, c) - expected) < 1e-9

    def test_custom_base_weights(self):
        """Caller-provided base_weights override the defaults."""
        s = {'a': 0.8, 'b': 0.6}
        c = {'a': 1.0, 'b': 1.0}
        w = {'a': 0.6, 'b': 0.4}
        expected = (0.6*0.8 + 0.4*0.6) / (0.6 + 0.4)
        assert abs(dynamic_composite(s, c, base_weights=w) - expected) < 1e-9

    def test_result_bounded_zero_one(self):
        """Score is always in [0, 1] for inputs in [0, 1]."""
        import random
        rng = random.Random(42)
        for _ in range(50):
            s = {k: rng.random() for k in ('ppm', 'r2', 'mult', 'j')}
            c = {k: rng.choice([0.0, 0.5, 1.0]) for k in ('ppm', 'r2', 'mult', 'j')}
            score = dynamic_composite(s, c)
            assert 0.0 <= score <= 1.0 + 1e-9


class TestJConfidence:

    def test_j_measured(self):
        assert j_confidence(7.2) == 1.0

    def test_j_not_measured(self):
        assert j_confidence(None) == 0.0

    def test_j_zero_is_measured(self):
        """Zero Hz is a valid (unusual) measurement, not missing."""
        assert j_confidence(0.0) == 1.0


class TestAlignmentPpmConfidence:

    def test_alignment_ok(self):
        assert alignment_ppm_confidence(True) == 1.0

    def test_alignment_failed(self):
        conf = alignment_ppm_confidence(False)
        assert 0.0 < conf < 1.0   # penalised but not zero


class TestBaseWeightConsistency:

    def test_search_weights_sum_to_one(self):
        total = sum(_BASE_WEIGHTS_SEARCH.values())
        assert abs(total - 1.0) < 1e-9

    def test_enrichment_ns_weights_sum_to_one(self):
        total = sum(_BASE_WEIGHTS_ENRICHMENT_NS.values())
        assert abs(total - 1.0) < 1e-9

    def test_enrichment_s_weights_sum_to_one(self):
        total = sum(_BASE_WEIGHTS_ENRICHMENT_S.values())
        assert abs(total - 1.0) < 1e-9
