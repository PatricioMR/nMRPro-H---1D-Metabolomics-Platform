"""A metabolite with no number must say which kind of nothing it is.

Every "good" metabolite that ends without a concentration was told it had lost
the peak competition. Some had never been in one: a multiplet whose expected
lines were not all found is withdrawn before competing, and so is one whose J
quality was unavailable. The difference is the only actionable part — those come
back if detection resolves the lines, a metabolite beaten by a rival does not.
"""
import inspect

from gui.stages import _quant_computation as qc
from gui.stages._quant_helpers import (
    NQ_REASON_LOST_COMPETITION, NQ_REASON_NO_SDJ,
    NQ_REASON_UNVERIFIED_MULTIPLET, NQ_REASON_ZERO_NAN)


def _reason_src():
    src = inspect.getsource(qc)
    i = src.index('def _nq_reason(mid):')
    return src[i:i + 1200]


def test_the_four_outcomes_are_distinguishable():
    reasons = {NQ_REASON_LOST_COMPETITION, NQ_REASON_NO_SDJ,
               NQ_REASON_UNVERIFIED_MULTIPLET, NQ_REASON_ZERO_NAN}
    assert len(reasons) == 4


def test_a_withdrawn_multiplet_is_not_called_a_loser():
    src = _reason_src()
    assert 'NQ_REASON_UNVERIFIED_MULTIPLET' in src
    assert 'NQ_REASON_NO_SDJ' in src


def test_only_a_metabolite_that_never_competed_gets_the_withdrawn_reason():
    """One signal can be withdrawn while another competes and loses; then it
    did lose, and saying otherwise sends the analyst to the wrong signal."""
    src = _reason_src()
    assert 'if mid not in _competed_mids:' in src
    i = src.index('if mid not in _competed_mids:')
    j = src.index('return NQ_REASON_LOST_COMPETITION')
    assert i < j, 'the withdrawn check must be inside the never-competed branch'


def test_ambiguity_still_outranks_everything():
    src = _reason_src()
    assert src.index('_ambiguous_mids') < src.index('_blocked_unverified')


def test_the_sets_exist_whichever_branch_runs():
    """They are read in a different conditional from the one that fills them;
    a lipidomics run would otherwise raise NameError instead of quantifying."""
    src = inspect.getsource(qc)
    head = src[:src.index('_all_non_bad = []')]
    for name in ('_blocked_no_sdj: set = set()',
                 '_blocked_unverified: set = set()',
                 '_competed_mids: set = set()'):
        assert name in head, f'{name} must be declared before the branches'


def test_withdrawn_metabolites_still_get_a_row():
    """Withdrawn is not deleted: the analyst has to see it greyed out with the
    reason, or the analyte silently vanishes from the study."""
    src = inspect.getsource(qc)
    assert "'_not_quantified':   True," in src
    assert "'status':            'n.q.'," in src
