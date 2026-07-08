"""Unit tests for the não-quantificado predicate.

A metabolite that wins no peak (lost competition) or whose mean concentration
is 0/NaN across all samples carries no measurable signal: it must be flagged so
it is greyed in the list and excluded from exports and statistics, never
reported as a real value (CLAUDE.md: never report noise as signal).
"""
import math

from gui.stages._quant_helpers import (
    is_unquantified_conc,
    NQ_REASON_LOST_COMPETITION,
    NQ_REASON_ZERO_NAN,
)


def test_nan_is_unquantified():
    assert is_unquantified_conc(float('nan')) is True


def test_zero_is_unquantified():
    assert is_unquantified_conc(0.0) is True


def test_negative_is_unquantified():
    assert is_unquantified_conc(-1.5) is True


def test_none_is_unquantified():
    assert is_unquantified_conc(None) is True


def test_non_numeric_is_unquantified():
    assert is_unquantified_conc('') is True


def test_positive_is_quantified():
    assert is_unquantified_conc(0.0001) is False
    assert is_unquantified_conc(12.5) is False


def test_reasons_are_distinct_nonempty_strings():
    assert NQ_REASON_LOST_COMPETITION
    assert NQ_REASON_ZERO_NAN
    assert NQ_REASON_LOST_COMPETITION != NQ_REASON_ZERO_NAN
