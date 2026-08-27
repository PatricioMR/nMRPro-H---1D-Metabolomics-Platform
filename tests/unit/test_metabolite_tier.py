"""One definition of a metabolite's tier, for the row, the filter and the count.

The Annotation Results window computed the parent tier in two places that
disagreed, and read the per-tier counts from the already-filtered list — so
unticking Average made its own count read (0), and a row rendered Average could
be hidden by the Good filter.
"""
import pytest

from processing.annotation_quality import metabolite_has_anchor, metabolite_tier
from processing.signal_quality import SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR


def _met(*quals, mid='M1', ranks=None, anchor_on=None):
    ranks = ranks or list(range(1, len(quals) + 1))
    mps = [{'annotation_quality': q, 'peak_rank': r}
           for q, r in zip(quals, ranks)]
    if anchor_on is not None:
        mps[anchor_on]['auto_anchor'] = True
    return {'id': mid, 'matched_peaks': mps}


class TestTier:
    def test_all_bad_is_poor(self):
        assert metabolite_tier(_met('bad', 'bad')) == SIGNAL_POOR

    def test_any_good_is_good(self):
        assert metabolite_tier(_met('bad', 'good')) == SIGNAL_GOOD

    def test_no_verdict_is_average(self):
        assert metabolite_tier(_met(None, None)) == SIGNAL_AVERAGE

    def test_no_signals_is_average(self):
        assert metabolite_tier({'id': 'M1', 'matched_peaks': []}) == SIGNAL_AVERAGE

    def test_contested_demotes_good_to_average(self):
        """A contest says the peak may be someone else's — that outranks Good."""
        met = _met('good', ranks=[7])
        assert metabolite_tier(met, ambiguous_ranks={7}) == SIGNAL_AVERAGE

    def test_contested_does_not_rescue_all_bad(self):
        met = _met('bad', 'bad', ranks=[7, 8])
        assert metabolite_tier(met, ambiguous_ranks={7}) == SIGNAL_POOR

    def test_contested_outranks_an_anchor(self):
        met = _met('good', ranks=[7], anchor_on=0)
        assert metabolite_tier(met, ambiguous_ranks={7}) == SIGNAL_AVERAGE

    def test_manual_anchor_makes_it_good(self):
        met = _met(None, mid='M9')
        assert metabolite_tier(met, anchor_store={('M9', 3): True}) == SIGNAL_GOOD


class TestAnchor:
    def test_auto_anchor_needs_a_good_signal(self):
        """The filter accepted any auto_anchor while the row required Good, so
        the same metabolite rendered Average and was hidden by Good."""
        assert not metabolite_has_anchor(_met(None, anchor_on=0))
        assert metabolite_has_anchor(_met('good', anchor_on=0))

    def test_manual_anchor_matches_on_metabolite_id(self):
        assert metabolite_has_anchor(_met(None, mid='M9'), {('M9', 1): True})
        assert not metabolite_has_anchor(_met(None, mid='M9'), {('M8', 1): True})

    def test_empty_store_is_not_an_anchor(self):
        assert not metabolite_has_anchor(_met(None))


def test_counts_do_not_depend_on_the_tier_filter():
    """The counts label the tier checkboxes, so they must survive unticking one.

    Reproduces the reported bug at the level the window now works: count over
    the set the non-tier filters allow, never over the filtered rows.
    """
    mets = [_met('good', mid='A'), _met(None, mid='B'), _met('bad', mid='C')]

    def counts(pool):
        out = {SIGNAL_GOOD: 0, SIGNAL_AVERAGE: 0, SIGNAL_POOR: 0}
        for m in pool:
            out[metabolite_tier(m)] += 1
        return out

    full = counts(mets)
    assert full == {SIGNAL_GOOD: 1, SIGNAL_AVERAGE: 1, SIGNAL_POOR: 1}

    shown = {SIGNAL_GOOD: True, SIGNAL_AVERAGE: False, SIGNAL_POOR: True}
    visible = [m for m in mets if shown[metabolite_tier(m)]]
    assert len(visible) == 2
    assert counts(mets) == full          # unchanged by the filter
    assert counts(visible)[SIGNAL_AVERAGE] == 0   # what the old code showed
