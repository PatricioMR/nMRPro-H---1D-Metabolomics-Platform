"""PQN divides each sample by how concentrated it is against the median sample,
so a difference that reaches every metabolite alike is read as dilution and
taken out. On the validation bench a global x0.6 on one group was estimated at
3% and removed (ADR-062). A gap between the groups' mean factors is the sign,
and the analyst is told."""
import pytest

from processing.quant_concentration import (
    PQN_GROUP_WARN, pqn_group_divergence, pqn_group_divergence_of_rows,
    pqn_group_note)

GROUPS = {'A': [0, 1, 2], 'B': [3, 4, 5]}


def _factors(a, b):
    return {i: (a if i < 3 else b) for i in range(6)}


def test_groups_with_the_same_factors_do_not_diverge():
    d = pqn_group_divergence(_factors(1.0, 1.05), GROUPS)
    assert d['by_group'] == pytest.approx({'A': 1.0, 'B': 1.05})
    assert d['ratio'] == pytest.approx(1.05) and not d['diverges']


def test_a_group_standing_at_0_6_of_the_other_diverges():
    d = pqn_group_divergence(_factors(1.2, 0.72), GROUPS)
    assert d['ratio'] == pytest.approx(1.2 / 0.72) and d['diverges']


def test_the_bar_is_a_ratio_of_one_plus_the_threshold():
    f = _factors(1.0, 1.0 / (1.0 + PQN_GROUP_WARN) * 1.001)   # just under
    assert not pqn_group_divergence(f, GROUPS)['diverges']
    assert pqn_group_divergence(_factors(1.0, 1.25), GROUPS)['diverges']
    assert not pqn_group_divergence(_factors(1.0, 1.25), GROUPS, threshold=0.3)['diverges']


def test_one_group_or_no_factors_says_nothing():
    assert pqn_group_divergence(_factors(1.0, 2.0), {'A': [0, 1, 2]}) is None
    assert pqn_group_divergence({}, GROUPS) is None
    # samples without a factor (an external standard's tube) are not counted
    d = pqn_group_divergence({0: 1.0, 3: 1.0}, GROUPS)
    assert d['by_group'] == {'A': 1.0, 'B': 1.0}


def test_the_factors_are_read_from_the_rows_and_none_means_pqn_was_off():
    rows = [{'_pqn_factors': {}}, {'_pqn_factors': _factors(1.0, 0.5)}]
    assert pqn_group_divergence_of_rows(rows, GROUPS)['diverges']
    assert pqn_group_divergence_of_rows([{'_pqn_factors': {}}], GROUPS) is None
    assert pqn_group_divergence_of_rows([], GROUPS) is None


def test_the_note_names_the_groups_and_what_it_means():
    note = pqn_group_note(pqn_group_divergence(_factors(1.0, 0.5), GROUPS))
    assert 'A 1.00' in note and 'B 0.50' in note and 'ratio of 2.00' in note
    assert 'composition' in note and 'Dieterle' in note
