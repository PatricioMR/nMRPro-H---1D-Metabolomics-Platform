# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Complex multiplets shaped by their spin system (processing.spin_patterns)."""
import json

import pytest

from processing import spin_patterns as sp


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(sp, 'SPIN_PATTERNS', True)


def test_citrate_ab_opens_with_the_field():
    # an AB pair: the outer lines move away as the field rises -- the
    # stored 500 MHz pattern (+-23.5 / +-38.6 Hz) is not citrate at 600
    l500 = json.loads(sp.signal_pattern('Citrate', 2.579, 500.0, 0.08)[0])
    l600 = json.loads(sp.signal_pattern('Citrate', 2.579, 600.0, 0.08)[0])
    span = lambda ls: max(o for o, _ in ls) - min(o for o, _ in ls)
    assert span(l600) > span(l500) + 5.0
    # symmetric about its centre, roofed inwards
    inner = sorted(l600, key=lambda l: abs(l[0]))[:2]
    outer = sorted(l600, key=lambda l: abs(l[0]))[-2:]
    assert min(r for _, r in inner) > max(r for _, r in outer)


def test_only_complex_multiplets_of_known_systems_change():
    rows = [{'name': 'L-glutamic acid', 'multiplicity': 'm', 'ppm_center': 2.339,
             'pattern_lines': '[[-24.0, 0.05], [24.0, 0.05]]'},
            {'name': 'L-lactic acid', 'multiplicity': 'd', 'ppm_center': 1.318,
             'pattern_lines': '[[-3.5, 1.0], [3.5, 1.0]]'},
            {'name': 'Not a compound', 'multiplicity': 'm', 'ppm_center': 2.0,
             'pattern_lines': '[[-5, 1], [5, 1]]'}]
    assert sp.apply(rows, 600.71) == 1
    assert rows[0]['pattern_source'] == 'spin system'
    assert 'pattern_source' not in rows[1] and 'pattern_source' not in rows[2]
    assert rows[0]['pattern_profile'] and rows[0]['profile_fwhm_hz'] == sp.PROFILE_FWHM_HZ


def test_off_nothing_changes(monkeypatch):
    monkeypatch.setattr(sp, 'SPIN_PATTERNS', False)
    rows = [{'name': 'L-glutamic acid', 'multiplicity': 'm', 'ppm_center': 2.339,
             'pattern_lines': '[[-24.0, 0.05], [24.0, 0.05]]'}]
    assert sp.apply(rows, 600.71) == 0
    assert 'pattern_source' not in rows[0]
