# tests/unit/test_confidence_classifier.py
"""Unit tests for gui.notation.confidence_classifier — classify_annotation."""
import pytest
from gui.notation.confidence_classifier import (
    classify_annotation,
    CLASS_COLOURS,
    CLASS_MSI_LEVEL,
    _DEFAULT_THRESHOLDS,
)


def _make_candidate(**kwargs):
    """Build a minimal candidate dict with sensible defaults."""
    defaults = {
        'composite_score': 0.0,
        'peak_coverage':   0.0,
        'coherence_score': 0.0,
        'stocsy_coherence': None,
        'mcr_confirmed':   None,
        'rri':             None,
        'fitting_r2':      0.0,
        'ppm_score':       0.0,
    }
    defaults.update(kwargs)
    return defaults


class TestClassifyAnnotation:

    # ── Class A ──────────────────────────────────────────────────────────────

    def test_class_a_stocsy_path(self):
        c = _make_candidate(
            peak_coverage=0.85,
            stocsy_coherence=0.75,
            rri=0.80,
            composite_score=0.90,
        )
        assert classify_annotation(c) == 'A'

    def test_class_a_mcr_path(self):
        c = _make_candidate(
            peak_coverage=0.90,
            mcr_confirmed=True,
            rri=0.70,
            composite_score=0.90,
        )
        assert classify_annotation(c) == 'A'

    def test_class_a_requires_rri(self):
        """Missing RRI (< threshold) prevents class A even with STOCSY."""
        c = _make_candidate(
            peak_coverage=0.85,
            stocsy_coherence=0.75,
            rri=0.30,   # below a_rri=0.65
            composite_score=0.90,
        )
        assert classify_annotation(c) != 'A'

    def test_class_a_requires_coverage(self):
        c = _make_candidate(
            peak_coverage=0.50,   # below a_coverage=0.80
            stocsy_coherence=0.75,
            rri=0.80,
        )
        assert classify_annotation(c) != 'A'

    # ── Class B ──────────────────────────────────────────────────────────────

    def test_class_b_multi_peak_coherent(self):
        c = _make_candidate(
            peak_coverage=0.70,
            coherence_score=0.60,
            composite_score=0.60,
        )
        assert classify_annotation(c) == 'B'

    def test_class_b_borderline_coverage(self):
        c = _make_candidate(
            peak_coverage=0.65,
            coherence_score=0.55,
            composite_score=0.60,
        )
        assert classify_annotation(c) == 'B'

    def test_class_b_requires_coherence(self):
        c = _make_candidate(
            peak_coverage=0.70,
            coherence_score=0.30,   # below b_coherence=0.55
        )
        assert classify_annotation(c) not in ('A', 'B')

    # ── Class C ──────────────────────────────────────────────────────────────

    def test_class_c_good_fit(self):
        c = _make_candidate(
            composite_score=0.80,
            fitting_r2=0.70,
        )
        assert classify_annotation(c) == 'C'

    def test_class_c_good_ppm(self):
        c = _make_candidate(
            composite_score=0.80,
            fitting_r2=0.40,   # below c_r2=0.60
            ppm_score=0.92,    # above c_ppm_score=0.88
        )
        assert classify_annotation(c) == 'C'

    def test_class_c_requires_score(self):
        c = _make_candidate(
            composite_score=0.60,   # below c_score=0.75
            fitting_r2=0.80,
            ppm_score=0.95,
        )
        assert classify_annotation(c) not in ('A', 'B', 'C')

    # ── Class D ──────────────────────────────────────────────────────────────

    def test_class_d_tentative(self):
        c = _make_candidate(composite_score=0.55)
        assert classify_annotation(c) == 'D'

    def test_class_d_borderline(self):
        c = _make_candidate(composite_score=0.50)
        assert classify_annotation(c) == 'D'

    # ── Class X ──────────────────────────────────────────────────────────────

    def test_class_x_low_score(self):
        c = _make_candidate(composite_score=0.30)
        assert classify_annotation(c) == 'X'

    def test_class_x_zero_score(self):
        c = _make_candidate()
        assert classify_annotation(c) == 'X'

    # ── Custom thresholds ────────────────────────────────────────────────────

    def test_custom_thresholds_override(self):
        """Caller can lower thresholds to promote a candidate to class A."""
        c = _make_candidate(
            peak_coverage=0.70,   # below default a_coverage=0.80
            stocsy_coherence=0.65,
            rri=0.70,
        )
        assert classify_annotation(c) != 'A'
        # Custom threshold relaxes coverage requirement
        assert classify_annotation(c, thresholds={'a_coverage': 0.65}) == 'A'


class TestClassColoursAndMsiLevel:

    def test_all_classes_have_colours(self):
        for cls in ('A', 'B', 'C', 'D', 'X'):
            assert cls in CLASS_COLOURS
            bg, fg = CLASS_COLOURS[cls]
            assert isinstance(bg, str)
            assert isinstance(fg, str)

    def test_all_classes_have_msi_level(self):
        for cls in ('A', 'B', 'C', 'D', 'X'):
            assert cls in CLASS_MSI_LEVEL

    def test_x_is_rejected(self):
        assert 'Rejected' in CLASS_MSI_LEVEL['X']
