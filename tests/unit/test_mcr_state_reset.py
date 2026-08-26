"""MCR state must not survive a search that did not run MCR.

`mcr_annotation` lives on the detected-peak dicts, which outlive any single
search, and nothing used to remove it. Switching MCR off and re-running left the
results window reading components and verdicts produced by the previous run.
"""
import inspect
import re

import pytest

import gui.stages._annotation_search as search_mod
from gui.stages._annotation_search import MCR_RESULT_KEYS, clear_stale_mcr


def test_stale_peak_result_is_dropped():
    peaks = [{'rank': 1, 'ppm': 1.47, 'mcr_annotation': {'K': 2, 'S': object()}}]
    n_peaks, _ = clear_stale_mcr(peaks)
    assert n_peaks == 1
    assert 'mcr_annotation' not in peaks[0]
    assert peaks[0]['ppm'] == 1.47, 'only the MCR state goes'


def test_full_spectrum_mcr_peaks_keep_their_result():
    """Those peaks exist because of their MCR result — removing it would
    destroy the peak's own reason for being there."""
    peaks = [{'rank': 9, 'mcr_annotation': {'K': 1}, '_from_mcr': True}]
    n_peaks, _ = clear_stale_mcr(peaks)
    assert n_peaks == 0
    assert peaks[0]['mcr_annotation'] == {'K': 1}


def test_peaks_without_mcr_state_are_untouched():
    peaks = [{'rank': 1, 'ppm': 1.47}]
    assert clear_stale_mcr(peaks) == (0, 0)
    assert peaks == [{'rank': 1, 'ppm': 1.47}]


def test_carried_results_lose_every_mcr_field():
    row = {'name': 'Taurine', 'composite_score': 0.8}
    row.update({k: 'x' for k in MCR_RESULT_KEYS})
    _, n_results = clear_stale_mcr(None, [row])
    assert n_results == 1
    assert not (set(row) & set(MCR_RESULT_KEYS))
    assert row['composite_score'] == 0.8, 'scores are not MCR state'


def test_empty_input_is_safe():
    assert clear_stale_mcr(None, None) == (0, 0)
    assert clear_stale_mcr([], []) == (0, 0)


def test_key_list_covers_everything_the_search_writes():
    """If an MCR pass gains a new output field and this list is not updated,
    that field survives a run with MCR switched off."""
    src = inspect.getsource(search_mod)
    written = set(re.findall(r"c\['(mcr_[a-z0-9_]+)'\]\s*=", src))
    missing = written - set(MCR_RESULT_KEYS)
    assert not missing, f'MCR_RESULT_KEYS is missing {sorted(missing)}'


def test_search_clears_before_it_matches():
    """The clear has to happen in the search itself, not at window build: an
    aborted run must not leave the previous verdicts in place either."""
    src = inspect.getsource(search_mod)
    assert 'clear_stale_mcr(self.peaks)' in src


def test_carried_rows_are_cleared_only_when_mcr_is_off():
    src = inspect.getsource(search_mod)
    block = src[src.index('_mcr_on = getattr(self'):]
    block = block[:400]
    assert 'if not (_mcr_on is not None and _mcr_on.get()):' in block
    assert 'clear_stale_mcr(None, carry_results)' in block
