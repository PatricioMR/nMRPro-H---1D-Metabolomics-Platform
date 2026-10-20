# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Line intensities as competition evidence.

ΔJ and Δppm both come from position. A multiplicity also fixes how tall its
lines are — a triplet is 1:2:1 by the binomial theorem — so three apices of
equal height argue against a triplet claim however well their spacing fits.
That is the one thing height says, and nothing in the ranking said it.
"""
import pytest

from processing.annotation_scoring import (
    LINE_COUNTS, LINE_INTENSITIES, PASCAL_TIE, candidate_key, pascal_agreement,
    pascal_band, rank_candidates)


class TestTheTable:
    def test_every_multiplicity_agrees_on_its_line_count(self):
        """Two copies of this lived in the GUI and disagreed on which keys
        existed. One table, and it must match LINE_COUNTS entry for entry."""
        for k, pattern in LINE_INTENSITIES.items():
            assert k in LINE_COUNTS, f'{k} has intensities but no line count'
            assert len(pattern) == LINE_COUNTS[k], k

    def test_no_multiplicity_is_left_without_a_pattern(self):
        assert not set(LINE_COUNTS) - set(LINE_INTENSITIES)

    def test_the_simple_ones_are_pascals_triangle(self):
        assert LINE_INTENSITIES['t'] == (1, 2, 1)
        assert LINE_INTENSITIES['q'] == (1, 3, 3, 1)
        assert LINE_INTENSITIES['quint'] == (1, 4, 6, 4, 1)


class TestAgreement:
    def test_a_perfect_triplet(self):
        assert pascal_agreement([0, 1, 2], [1, 2, 1], 't') == pytest.approx(1.0)

    def test_three_equal_apices_are_not_a_triplet(self):
        """The case the ranking could not see: spacing perfect, shape wrong."""
        assert pascal_agreement([0, 1, 2], [1, 1, 1], 't') < 0.7

    def test_a_partial_multiplet_is_judged_on_the_lines_it_has(self):
        """The outer pair of a triplet is 1:1 and must not be scored against
        1:2:1 — that would punish a reading for being partial."""
        assert pascal_agreement([0, 2], [1, 1], 't') == pytest.approx(1.0)

    def test_the_middle_line_alone_with_an_outer_one(self):
        assert pascal_agreement([0, 1], [1, 2], 't') == pytest.approx(1.0)

    def test_scale_does_not_matter(self):
        """Concentration is not shape: the same pattern at any height agrees."""
        a = pascal_agreement([0, 1, 2], [1, 2, 1], 't')
        b = pascal_agreement([0, 1, 2], [500, 1000, 500], 't')
        assert a == pytest.approx(b)

    @pytest.mark.parametrize('bad', [
        ([0, 1], [1, 1], 'zzz'),          # multiplicity nobody knows
        ([0], [1], 't'),                  # a single line has no shape
        ([0, 1], [0, 0], 't'),            # nothing measured
        ([0, 1], [1], 't'),               # ranks and heights disagree
        ([0, 9], [1, 1], 't'),            # a rank the pattern does not have
        ([0, 1], [1, float('nan')], 't'),
        ([0, 1], [1, -3], 't'),
    ])
    def test_what_cannot_be_judged_returns_none(self, bad):
        assert pascal_agreement(*bad) is None


class TestBanding:
    def test_agreement_within_the_tie_is_one_band(self):
        assert pascal_band(1.0) == pascal_band(1.0 - PASCAL_TIE / 2)

    def test_worse_agreement_sorts_later(self):
        assert pascal_band(0.5) > pascal_band(0.95)

    def test_unjudgeable_sorts_last_but_is_not_disagreement(self):
        """A candidate that cannot be judged is not thereby wrong; it simply
        offers nothing here and the decision falls through."""
        assert pascal_band(None) > pascal_band(0.0)


class TestInTheRanking:
    def _c(self, **kw):
        c = dict(multiplicity='t', delta_j_hz=0.05, ppm_diff=0.001,
                 n_found=3, n_expected=3, cosine=0.5, composite_score=0.5)
        c.update(kw)
        return c

    def test_shape_separates_two_candidates_position_cannot(self):
        good = self._c(intensity_agreement=1.0)
        bad = self._c(intensity_agreement=0.4)
        k_good = candidate_key(good, dj_gold_hz=0.2)
        k_bad = candidate_key(bad, dj_gold_hz=0.2)
        assert k_good < k_bad

    def test_shape_never_outranks_the_coupling(self):
        """A perfect shape with a wrong J must not beat a right J."""
        shapely = self._c(intensity_agreement=1.0, delta_j_hz=3.0)
        coupled = self._c(intensity_agreement=0.3, delta_j_hz=0.05)
        assert candidate_key(coupled, dj_gold_hz=0.2) < \
            candidate_key(shapely, dj_gold_hz=0.2)

    def test_shape_outranks_the_fingerprint_cosine(self):
        """It judges THIS signal; the cosine judges the metabolite as a whole."""
        shapely = self._c(intensity_agreement=1.0, cosine=0.10)
        similar = self._c(intensity_agreement=0.3, cosine=0.99)
        assert candidate_key(shapely, dj_gold_hz=0.2) < \
            candidate_key(similar, dj_gold_hz=0.2)

    def test_a_shape_difference_resolves_an_otherwise_ambiguous_contest(self):
        _, ambiguous = rank_candidates(
            [self._c(intensity_agreement=1.0), self._c(intensity_agreement=0.3)],
            dj_gold_hz=0.2)
        assert ambiguous is False

    def test_two_unjudgeable_candidates_stay_ambiguous(self):
        _, ambiguous = rank_candidates(
            [self._c(intensity_agreement=None), self._c(intensity_agreement=None)],
            dj_gold_hz=0.2)
        assert ambiguous is True


def test_the_key_is_read_by_name_not_by_position():
    """rank_candidates used k[:5], k[5], k[6]; inserting a criterion made it
    compare the cosine against the score's tie width.

    Asserted as the property, not as a fixed field list: pinning the names in
    order would fail on exactly the change it exists to make safe, and the
    fix would be to edit this test — which teaches nobody anything.
    """
    import ast
    import inspect

    k = candidate_key({'multiplicity': 'd'}, dj_gold_hz=0.2)
    assert {'cosine', 'score'} <= set(k._fields)

    tree = ast.parse(inspect.getsource(rank_candidates))
    subscripts = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript)
                  and isinstance(n.value, ast.Name)
                  and n.value.id.startswith('k')]
    assert subscripts == []
