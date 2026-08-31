"""A gap-fill hit must not adopt a peak someone else won.

The greedy allocation in _quant_computation gives each detected peak to one
metabolite. A candidate with no detected apex — peak_rank None — competes as a
DB gap-fill and deliberately claims no slot, so that one such candidate cannot
block every other. Integration then resolved the peak with a ppm-proximity
fallback, which handed it whichever peak sat nearest: o-cresol was quantified on
peak 214, already won by 3-(2-hydroxyphenyl)propionic acid, on a study where
that was the only shared peak among 117 quantified metabolites.
"""
import inspect

import pytest

from gui.stages import _quant_computation as qc


def _resolve_src():
    src = inspect.getsource(qc)
    i = src.index("_rank_h = hit.get('peak_rank')")
    return src[i:i + 700]


def test_a_hit_without_a_rank_never_reaches_the_proximity_search():
    src = _resolve_src()
    assert "p = (_pk_by_rank.get(_rank_h) if _rank_h is not None else None)" in src
    assert 'if p is None and _rank_h is not None:' in src


def test_a_stale_rank_still_falls_back():
    """Detection re-running since annotation leaves a rank the peak list no
    longer holds; proximity is the only way back for those."""
    src = _resolve_src()
    i = src.index('if p is None and _rank_h is not None:')
    assert 'self._find_peak(' in src[i:i + 300]


def test_the_gapfill_window_is_still_reachable():
    """A hit with no peak is meant to be integrated over its predicted lines,
    which is the branch this fix routes it to."""
    src = inspect.getsource(qc)
    assert 'db_gapfill_window(' in src
    assert '_synthetic_db = True' in src


def test_find_peak_is_not_called_unconditionally_anywhere_in_the_hit_loop():
    """Regression guard: the old line resolved the peak in one expression, so
    the None case fell through to proximity without anyone noticing."""
    src = inspect.getsource(qc)
    assert "_pk_by_rank.get(hit.get('peak_rank')) or" not in src
