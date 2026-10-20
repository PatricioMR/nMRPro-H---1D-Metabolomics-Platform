# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Broad/exchangeable and generic multiplicities are excluded from annotation."""
from gui.stages._annotation_search import is_unquantifiable_multiplicity as ex


def test_broad_variants_excluded():
    for m in ['br s', 'br. s.', 'brs', 'br d', 'BR S', ' br. s. ', 'br t']:
        assert ex(m) is True, m


def test_generic_multiplet_excluded():
    assert ex('m') is True
    assert ex('multiplet') is True


def test_real_multiplicities_kept():
    for m in ['s', 'd', 't', 'q', 'dd', 'ddd', 'dt', 'td', 'tt', 'dq', '']:
        assert ex(m) is False, m


def test_none_kept():
    assert ex(None) is False
