"""The same physical lines cannot be quantified twice.

The greedy allocation blocked a candidate whose ANCHOR peak was already won. A
multiplet also claims its line peaks — but the next candidate was only ever
tested on its anchor, so a different anchor over the same lines walked straight
in. On the reference study 16 of 111 quantified metabolites were integrating
another's lines: L-alanine and hexanoic acid both on 1.4801/1.4919, L-valine,
L-isoleucine and 2-hydroxy-3-methylbutyric acid all on 0.9912/1.0028, and
nicotinamide with nicotinuric acid on 8.1938/8.2071.
"""
import inspect

from gui.stages import _quant_computation as qc


def _placement_src():
    src = inspect.getsource(qc)
    i = src.index('_assigned_mets = set()')
    return src[i:i + 2200]


def test_a_candidate_is_tested_on_its_lines_not_only_its_anchor():
    src = _placement_src()
    assert '_my_ranks & _assigned_peaks' in src


def test_the_line_check_happens_before_the_metabolite_is_placed():
    """Placed first and checked after would leave the winner recorded."""
    src = _placement_src()
    assert src.index('_my_ranks & _assigned_peaks') < \
        src.index('_assigned_mets.add(_h_mid)')


def test_the_anchor_check_is_still_there():
    src = _placement_src()
    assert '_pr is not None and _pr in _assigned_peaks' in src


def test_a_singlet_is_not_line_checked():
    """One line is the anchor itself; requiring two keeps a singlet from
    blocking itself through matched_line_ranks."""
    src = _placement_src()
    assert 'if len(_my_lines) >= 2 else set()' in src


def test_the_claim_uses_the_same_matcher_as_the_check():
    """Claiming by one rule and checking by another would let a peak be claimed
    under a name the check never looks for."""
    src = inspect.getsource(qc)
    i = src.index('_assigned_mets = set()')
    j = src.index('greedy:', i)
    assert src[i:j].count('matched_line_ranks(_h, _pk_by_rank)') == 2


def test_a_blocked_candidate_still_reaches_the_not_quantified_pass():
    """Losing here is losing the competition, which is what it will be told —
    the analyte is greyed out with a reason, not dropped."""
    src = inspect.getsource(qc)
    assert 'NQ_REASON_LOST_COMPETITION' in src
    assert "'_not_quantified':   True," in src


class TestTheOtherWaysBackIn:
    """The greedy pass was fixed; the two paths that put a metabolite back
    after it lost were not. Both run whenever JG-Deconv is enabled, which is
    how a signal reappeared under a second metabolite in the results table.
    """

    def _rescue_src(self):
        src = inspect.getsource(qc)
        i = src.index('Rescue: Good analytes that lost all peak-rank')
        return src[i:src.index('raw_areas = {}', i)]

    def _apq_src(self):
        src = inspect.getsource(qc)
        i = src.index('APQ fallback for metabolites excluded')
        return src[i:i + 2500]

    def test_a_rescued_hit_is_tested_on_its_lines(self):
        src = self._rescue_src()
        assert 'def _free_of_claims' in src
        assert 'matched_line_ranks(_r, _pk_by_rank)' in src

    def test_a_rescued_hit_claims_what_it_took(self):
        """Without this the next rescue sees the peak as free and two rescued
        analytes share one signal."""
        src = self._rescue_src()
        assert '_assigned_peaks.add(_rpr)' in src
        assert '_assigned_peaks.add(_or_r)' in src

    def test_the_apq_fallback_only_looks_at_free_peaks(self):
        """_all_peak_quant holds every matched peak of every metabolite,
        winners included."""
        src = self._apq_src()
        assert 'if _apq_rank in _assigned_peaks:' in src
        assert '_free_apq' in src

    def test_the_apq_fallback_still_picks_by_concentration_among_those(self):
        """Documented as a known weakness rather than silently changed: among
        peaks nobody won it still takes the largest number, which is magnitude
        and not evidence."""
        src = self._apq_src()
        assert '_ec > _best_conc' in src


def test_claimed_peaks_are_visible_to_every_path():
    """Declared before every branch that reads it: the greedy pass, the rescue
    and the APQ fallback each run in a different conditional, and a set defined
    inside one of them would raise NameError in the others."""
    src = inspect.getsource(qc)
    decl = src.index('_assigned_peaks = set()')
    for reader in ('_pr in _assigned_peaks',
                   '_my_ranks & _assigned_peaks',
                   '_apq_rank in _assigned_peaks'):
        assert decl < src.index(reader), reader
