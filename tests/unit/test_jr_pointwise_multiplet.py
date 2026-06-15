# tests/unit/test_jr_pointwise_multiplet.py
"""Unit tests for jr_pointwise_multiplet_area.

A multiplet's legs can be resolved differently by NNLS (J-Resolve). The
quantified area must weight each window point by the fraction of its nearest
predicted line, while reducing exactly to the plain whole-window integral when
fractions are uniform (so non-resolved multiplets are unchanged).
"""
import numpy as np
import pytest

from gui.stages._quant_helpers import jr_pointwise_multiplet_area, _trapz


def _doublet(x, c1, c2, a1=1.0, a2=1.0, w=0.004):
    """Two Lorentzian legs at c1, c2."""
    return (a1 * w**2 / ((x - c1) ** 2 + w**2)
            + a2 * w**2 / ((x - c2) ** 2 + w**2))


def test_uniform_fraction_equals_scaled_whole_window():
    # All lines share the same fraction → result == frac * whole-window area.
    x = np.linspace(8.40, 8.50, 400)
    y = _doublet(x, 8.44, 8.46)
    seg = y[:, None]                      # 1 sample
    lines = [8.44, 8.46]
    frac = 0.5
    out = jr_pointwise_multiplet_area(x, seg, lines, [frac, frac])
    whole = abs(float(_trapz(y, x)))
    assert out.shape == (1,)
    assert out[0] == pytest.approx(frac * whole, rel=1e-9)


def test_all_ones_equals_plain_area():
    x = np.linspace(8.40, 8.50, 400)
    y = _doublet(x, 8.44, 8.46)
    out = jr_pointwise_multiplet_area(x, y, [8.44, 8.46], [1.0, 1.0])
    assert out[0] == pytest.approx(abs(float(_trapz(y, x))), rel=1e-9)


def test_per_leg_fraction_weights_each_side():
    # Clean leg (frac=1.0) + shared leg (frac=0.5). Equal-area legs →
    # expected ≈ leg_area*1.0 + leg_area*0.5 = 1.5 * one_leg_area.
    x = np.linspace(8.40, 8.50, 4000)
    y = _doublet(x, 8.44, 8.46, a1=1.0, a2=1.0)
    out = jr_pointwise_multiplet_area(x, y, [8.44, 8.46], [1.0, 0.5])
    whole = abs(float(_trapz(y, x)))
    # Each leg ~ half the total area; weights 1.0 and 0.5 → 0.75 of the whole.
    assert out[0] == pytest.approx(0.75 * whole, rel=0.02)
    # Must lie strictly between the 0.5-uniform and 1.0-uniform results.
    lo = jr_pointwise_multiplet_area(x, y, [8.44, 8.46], [0.5, 0.5])[0]
    hi = jr_pointwise_multiplet_area(x, y, [8.44, 8.46], [1.0, 1.0])[0]
    assert lo < out[0] < hi


def test_per_sample_fraction_vector():
    # Sample 0 keeps the shared leg fully; sample 1 drops it to 0.
    x = np.linspace(8.40, 8.50, 2000)
    y = _doublet(x, 8.44, 8.46)
    seg = np.column_stack([y, y])         # 2 identical samples
    lines = [8.44, 8.46]
    fracs = [np.array([1.0, 1.0]), np.array([1.0, 0.0])]
    out = jr_pointwise_multiplet_area(x, seg, lines, fracs)
    # Sample 0 = whole; sample 1 = only the clean leg (~half).
    assert out[0] == pytest.approx(abs(float(_trapz(y, x))), rel=1e-6)
    assert out[1] < out[0]
    assert out[1] == pytest.approx(0.5 * out[0], rel=0.02)


def test_fraction_clipping():
    x = np.linspace(8.40, 8.50, 200)
    y = _doublet(x, 8.44, 8.46)
    # Out-of-range fractions are clipped to [0, 1].
    out = jr_pointwise_multiplet_area(x, y, [8.44, 8.46], [2.0, -1.0])
    clipped = jr_pointwise_multiplet_area(x, y, [8.44, 8.46], [1.0, 0.0])
    assert out[0] == pytest.approx(clipped[0], rel=1e-9)


def test_degenerate_window_falls_back_to_trapz():
    x = np.array([8.45])
    y = np.array([3.0])
    out = jr_pointwise_multiplet_area(x, y[:, None], [8.45], [0.5])
    # m < 2 → trapz of a single point is 0.
    assert out[0] == pytest.approx(0.0)
