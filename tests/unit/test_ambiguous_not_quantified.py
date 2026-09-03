"""An ambiguous rank has no winner, and quantification must not elect one.

rank_candidates decides a rank is ambiguous when its top two candidates tie on
every criterion there is — evidence tier, ΔJ band, Δppm band, quality, line
intensities — with cosine and score inside their tie widths. Its docstring says
the caller must not quantify any of them from that peak. The old engine filtered only 'bad', so the final tie-break elected one anyway:
on the reference study 3 of 112 quantified analytes sat on one of 16 ambiguous
ranks. These were source-text assertions on that engine; the election is now a
pure function, so the same claims are made about what it does.
"""
import inspect

from gui.stages import _quant_computation as qc


from processing.quantification import AMBIGUOUS_RANK, Candidate, elect


def _cand(mid, rank, dj=0.0):
    return Candidate(met_id=mid, peak_rank=rank, line_ranks=frozenset([rank]),
                     evidence={'multiplicity': 'd', 'delta_j_hz': dj,
                               'ppm_diff': 0.0, 'n_found': 2, 'n_expected': 2,
                               'quality': 'good', 'score': 1.0})


def test_an_ambiguous_hit_does_not_compete():
    e = elect([_cand('a', 7), _cand('b', 7)], dj_gold_hz=0.5,
              ambiguous_ranks=[7])
    assert e.winners == {}
    assert e.rejected['a'] == e.rejected['b'] == AMBIGUOUS_RANK


def test_it_is_barred_before_the_key_is_built():
    """Ranking it and then discarding would still let it claim the peak."""
    e = elect([_cand('a', 7), _cand('b', 7)], dj_gold_hz=0.5,
              ambiguous_ranks=[7])
    assert 7 not in e.claimed_peaks


def test_the_analyst_is_told_it_needs_a_manual_call():
    """Ambiguity is resolved by a person marking one candidate Good, so the
    reason has to say that rather than 'lost the competition'."""
    from processing.quantification import REJECTION_TEXT
    assert 'Good' in REJECTION_TEXT[AMBIGUOUS_RANK]


def test_a_metabolite_barred_on_one_rank_still_competes_elsewhere():
    """The rank is ambiguous, not the analyte."""
    e = elect([_cand('a', 7), _cand('b', 7), _cand('a', 9)],
              dj_gold_hz=0.5, ambiguous_ranks=[7])
    assert e.winners['a'].peak_rank == 9
    assert e.rejected['b'] == AMBIGUOUS_RANK


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
