"""The signal-grade selection must hold on every path into annotation.

Each test here corresponds to a way an excluded signal got annotated anyway.
They read the search source rather than driving the Tk stage, because the leaks
were in how a peak dict is *constructed*, not in the filter itself.
"""
import inspect
import re

import pytest

from processing.signal_quality import (
    SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR, effective_tier, is_annotatable,
    normalize_tiers)

import gui.stages._annotation_search as search_mod

SRC = inspect.getsource(search_mod)


# ── the filter itself ─────────────────────────────────────────────────────────

def test_anchor_selection_applies_the_configured_grades():
    """good_in_tol is the anchor pool — the peak that carries the annotation."""
    assert re.search(r'good_in_tol\s*=\s*\[p for p in local_peaks.*?'
                     r'is_annotatable\(p, _tiers\)', SRC, re.S)


def test_sibling_pool_is_not_filtered():
    """A Poor line of a doublet is still a line of that doublet."""
    assert 'local_peaks = [p for p in peaks_to_search' in SRC


# ── leak 1: the spectrum fallback ─────────────────────────────────────────────

def test_fallback_is_skipped_when_a_detected_peak_was_excluded():
    """Rebuilding an excluded peak from the raw spectrum annotates the very
    signal the grade filter ruled out."""
    guard = re.search(
        r'if not good_in_tol:.*?if any\(abs\(p\[.ppm.\] - db_ppm\) <= tolerance\s*'
        r'for p in local_peaks\):\s*continue', SRC, re.S)
    assert guard, 'fallback runs even when a detected peak was filtered out'


def test_synthetic_peak_does_not_claim_a_grade():
    """A spectrum-derived peak was never graded; claiming Good would bypass
    the selection entirely."""
    block = SRC[SRC.index('_syn_peak = {'):SRC.index('good_in_tol = [_syn_peak]')]
    assert "'status'" not in block
    assert 'if not is_annotatable(_syn_peak, _tiers)' in SRC


def test_ungraded_peak_reads_as_average():
    """Which is what makes the check above meaningful."""
    syn = {'ppm': 1.0, 'rank': None, '_synthetic': True}
    assert effective_tier(syn) == SIGNAL_AVERAGE
    assert is_annotatable(syn, normalize_tiers([SIGNAL_AVERAGE]))
    assert not is_annotatable(syn, normalize_tiers([SIGNAL_GOOD]))


# ── leak 2: deconvolution sub-peaks ───────────────────────────────────────────

def test_subpeaks_do_not_all_claim_good():
    """Stamping every sub-peak Good let a Poor parent re-enter annotation
    through its own deconvolved components."""
    assert "'status':      'Good'" not in SRC
    assert "'status':        'Good'" not in SRC
    assert '_parent_tier = effective_tier(_pk)' in SRC


def test_subpeak_inherits_the_parent_grade():
    parent = {'ppm': 1.47, 'rank': 3, 'signal_quality': SIGNAL_POOR}
    sub = {}
    inherited = sub.get('status') or effective_tier(parent)
    assert inherited == SIGNAL_POOR
    assert not is_annotatable({'status': inherited}, normalize_tiers(None))


def test_deconvolution_grade_wins_over_the_parent_grade():
    """Deconvolution measures the component directly; if it graded it, that
    grade is the better evidence."""
    parent = {'ppm': 1.47, 'rank': 3, 'signal_quality': SIGNAL_POOR}
    sub = {'status': SIGNAL_GOOD}
    inherited = sub.get('status') or effective_tier(parent)
    assert inherited == SIGNAL_GOOD


# ── the selection reaches the search ──────────────────────────────────────────

def test_search_reads_the_selection_from_the_stage():
    assert '_tiers = self.annotatable_tiers()' in SRC


@pytest.mark.parametrize('params,expected', [
    (None,                                   {SIGNAL_GOOD, SIGNAL_AVERAGE}),
    ({'annotatable_tiers': ['Good']},        {SIGNAL_GOOD}),
    ({'annotatable_tiers': ['Average']},     {SIGNAL_AVERAGE}),
    ({'annotatable_tiers': ['Good', 'Poor']}, {SIGNAL_GOOD, SIGNAL_POOR}),
])
def test_stage_accessor_reflects_the_dialog(params, expected):
    from gui.stages.annotation_stage import AnnotationStage
    stage = AnnotationStage.__new__(AnnotationStage)
    stage._annotation_params = params
    assert set(stage.annotatable_tiers()) == expected
