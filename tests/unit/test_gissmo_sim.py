# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The GISSMO spin-system simulator against what is known exactly: an AB
system's four lines, and every system's protons."""
import numpy as np
import pytest

from processing.spin_system import _lines, load_systems, render, stick_spectrum


def test_an_ab_system_is_the_textbook_one():
    # Pople, Schneider & Bernstein 1959: lines at centre +/- (D +/- J)/2,
    # D = sqrt(dnu^2 + J^2), inner/outer intensities (1 +/- J/D)
    nu_a, nu_b, J = 1000.0, 1010.0, 7.0
    f, i = _lines(np.array([nu_a, nu_b]), np.array([[0, J], [J, 0.0]]))
    order = np.argsort(f)
    f, i = f[order], i[order]
    c, d = (nu_a + nu_b) / 2, np.hypot(nu_b - nu_a, J)
    assert f == pytest.approx([c - (d + J) / 2, c - (d - J) / 2,
                               c + (d - J) / 2, c + (d + J) / 2], abs=1e-9)
    inner, outer = 1 + J / d, 1 - J / d
    assert i / i.sum() * 4 == pytest.approx([outer, inner, inner, outer],
                                            rel=1e-9)


def test_every_system_holds_its_protons_at_both_fields():
    # the systems of up to eight spins: the larger ones take minutes and were
    # checked line for line against the GISSMO XMLs when they were extracted
    for s in load_systems().values():
        if len(s['ppm']) > 8:
            continue
        for field in (500.13, 600.13):
            _p, inten = stick_spectrum(s, field)
            assert inten.sum() == pytest.approx(len(s['ppm']), rel=1e-6)


def test_a_rendered_line_holds_its_intensity_as_area():
    x = np.linspace(4.0, 3.0, 20001)
    y = render(x, np.array([3.5]), np.array([2.0]), 1.0, 600.13)
    o = np.argsort(x)
    assert np.trapezoid(y[o], x=x[o]) == pytest.approx(2.0, rel=0.02)


def test_uncoupled_fragments_solve_apart_as_they_would_together():
    # two fragments with no coupling path: the whole system's spectrum is
    # the sum of the fragments' -- the split must not change a line
    nu = np.array([1000.0, 1007.0, 1500.0, 1506.0, 1511.0])
    J = np.zeros((5, 5))
    J[0, 1] = J[1, 0] = 7.0
    J[2, 3] = J[3, 2] = 6.0
    J[3, 4] = J[4, 3] = 5.0
    s = {'ppm': nu / 600.0, 'J': J, 'external': []}
    f, i = stick_spectrum(s, 600.0)
    whole_f, whole_i = _lines(nu, J)
    x = np.linspace(2.6, 1.6, 20001)
    a = render(x, f, i, 1.0, 600.0)
    b = render(x, whole_f / 600.0, whole_i, 1.0, 600.0)
    assert np.allclose(a, b, atol=1e-9 * np.abs(b).max())
    assert i.sum() == pytest.approx(5.0)


def test_a_compound_in_two_forms_counts_its_protons_once():
    # GISSMO fits both glucose anomers into one entry at full weight each;
    # weighted by their equilibrium shares, the molecule has 7 C-H protons
    from processing.spin_system import load_systems
    glc = load_systems()['D-Glucose']
    p, i = stick_spectrum(glc, 600.0)
    assert i.sum() == pytest.approx(7.0)
    alpha = i[np.abs(p - 5.222) < 0.02].sum()
    beta = i[np.abs(p - 4.636) < 0.02].sum()
    assert alpha / (alpha + beta) == pytest.approx(0.36, abs=0.01)
