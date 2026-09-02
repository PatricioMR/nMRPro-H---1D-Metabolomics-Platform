"""An ambiguous rank has no winner, and quantification must not elect one.

rank_candidates decides a rank is ambiguous when its top two candidates tie on
every criterion there is — evidence tier, ΔJ band, Δppm band, quality, line
intensities — with cosine and score inside their tie widths. Its docstring says
the caller must not quantify any of them from that peak. Quantification filtered
only 'bad', so the final tie-break elected one anyway: on the reference study 3
of 112 quantified analytes sat on one of 16 ambiguous ranks.
"""
import inspect

from gui.stages import _quant_computation as qc


def _competition_src():
    src = inspect.getsource(qc)
    i = src.index('_all_non_bad = []')
    j = src.index('_all_non_bad.sort(', i)
    return src[i:j]


def test_an_ambiguous_hit_does_not_compete():
    src = _competition_src()
    assert "if _h_q == 'ambiguous':" in src
    assert '_blocked_ambiguous.add(_h_mid)' in src


def test_it_is_barred_before_the_key_is_built():
    """Ranking it and then discarding would still let it claim a peak."""
    src = _competition_src()
    assert src.index("_h_q == 'ambiguous'") < src.index('_compete_key(_h, _h_mid)')


def test_the_analyst_is_told_it_needs_a_manual_call():
    """Ambiguity is resolved by a person marking one candidate Good, so the
    reason has to say that rather than 'lost the competition'."""
    src = inspect.getsource(qc)
    i = src.index('def _nq_reason(mid):')
    body = src[i:i + 900]
    assert '_blocked_ambiguous' in body
    assert 'manual confirmation required' in body


def test_the_dead_sort_key_is_gone():
    """_quant_sort_key was never called and carried a +0.20 J-bonus it claimed
    to mirror from the annotation window, where it no longer exists."""
    assert '_quant_sort_key' not in inspect.getsource(qc)


def test_the_always_true_guard_is_gone():
    assert 'and not False' not in inspect.getsource(qc)


def test_eligibility_does_not_claim_to_mirror_a_deleted_function():
    """The comment pointed at _met_parent_qual, replaced by metabolite_tier —
    which answers a different question: the tree's row colour aggregates every
    signal, eligibility needs only one usable one."""
    src = inspect.getsource(qc)
    assert '_met_parent_qual' not in src
