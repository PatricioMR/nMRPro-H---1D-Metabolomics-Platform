"""The PCA plot prints the PERMANOVA the CSV export carries (lower-left)."""
import types

import numpy as np
import pytest
from matplotlib.figure import Figure

from gui.stages.statistics_stage import StatisticsStage


class _Var:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v


def _stage(n_comp, n_perm):
    st = StatisticsStage.__new__(StatisticsStage)
    st.app = types.SimpleNamespace(group_colors={})
    st._fig = Figure()
    st._n_comp_var = _Var(n_comp)
    st._n_perm_var = _Var(n_perm)
    st._biplot_var = _Var(False)
    st._ellipse_var = _Var(True)
    return st


def _data(seed=0):
    rng = np.random.default_rng(seed)
    g = ['CTRL'] * 6 + ['MSCs EVs'] * 6
    X = rng.normal(size=(12, 8))
    X[6:, :3] += 2.0
    return X - X.mean(0), [f's{i}' for i in range(12)], [f'm{i}' for i in range(8)], g


def _texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


@pytest.mark.parametrize('n_comp', [2, 3])
def test_permanova_text_on_plot_and_in_export(n_comp):
    st = _stage(n_comp, 99)
    X, names, feats, g = _data()
    st._plot_pca(X, names, feats, g)
    shown = [t for t in _texts(st._fig) if t.startswith('PERMANOVA')]
    assert len(shown) == 1
    assert 'F = ' in shown[0] and 'R² = ' in shown[0] and 'p = ' in shown[0]
    assert '99 permutations' in shown[0]
    last = st._plot_export['rows'][-1]
    assert last[0] == 'PERMANOVA' and 'F = ' in last[1]


def test_zero_permutations_prints_nothing():
    st = _stage(2, 0)
    X, names, feats, g = _data()
    st._plot_pca(X, names, feats, g)
    assert not [t for t in _texts(st._fig) if t.startswith('PERMANOVA')]
    assert st._plot_export['rows'][-1][0] != 'PERMANOVA'


def test_undefined_test_says_so():
    st = _stage(2, 99)
    X, names, feats, _ = _data()
    st._plot_pca(X, names, feats, ['CTRL'] * 12)
    assert any('PERMANOVA not defined' in t for t in _texts(st._fig))
