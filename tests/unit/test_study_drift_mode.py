"""The study's drift read where the confirmed multiplets crowd, not where
the coincidences spread."""
import numpy as np
import pytest

from processing import annotation_scoring as asc


def _met(name, shifts):
    """A metabolite with one confirmed doublet per shift."""
    return {'name': name, 'matched_peaks': [
        {'peak_rank': i, 'annotation_quality': 'good', 'type': 'd',
         'delta_j_hz': 0.1, '_n_found_lines': 2, '_n_exp_lines': 2,
         'db_ppm': 3.0 + i, 'peak_ppm': 3.0 + i + s}
        for i, s in enumerate(shifts)]}


def _study(true=0.008, n_true=20, n_bg=30, panel=None, seed=0):
    rng = np.random.default_rng(seed)
    mets = [_met('x', list(true + rng.normal(0, 0.001, n_true))),
            _met('y', list(rng.uniform(-0.05, 0.05, n_bg)))]
    if panel is not None:
        mets.append(_met('l-alanine', panel))
    return mets


@pytest.fixture
def mode_on(monkeypatch):
    monkeypatch.setattr(asc, 'DRIFT_BY_MODE', True)


def test_the_mode_finds_the_drift_the_true_matches_share():
    v = list(0.008 + np.random.default_rng(1).normal(0, 0.001, 20))
    v += list(np.linspace(-0.05, -0.01, 30))       # a skewed background
    assert asc.drift_mode(v) == pytest.approx(0.008, abs=0.001)
    assert abs(float(np.median(v)) - 0.008) > 0.01   # what the median did


def test_without_a_panel_the_mode_is_the_drift(mode_on):
    mets = _study()
    assert asc.study_drift(mets) == pytest.approx(0.008, abs=0.0015)


def test_a_panel_confirmed_by_coincidence_is_not_believed(mode_on):
    mets = _study(panel=[-0.024, -0.020, -0.031])
    assert asc.reference_drift(mets) is None
    assert asc.study_drift(mets) == pytest.approx(0.008, abs=0.0015)


def test_a_panel_that_agrees_with_the_mode_still_sets_the_drift(mode_on):
    mets = _study(panel=[0.012, 0.013, 0.011])
    assert asc.study_drift(mets) == pytest.approx(0.012)


def test_off_the_estimators_are_as_they_were(monkeypatch):
    monkeypatch.setattr(asc, 'DRIFT_BY_MODE', False)
    mets = _study(panel=[-0.024, -0.020, -0.031])
    assert asc.study_drift(mets) == pytest.approx(-0.024)
    no_panel = _study()
    shifts = asc._confirmed_shifts(no_panel)
    assert asc.study_drift(no_panel) == pytest.approx(float(np.median(shifts)))


def test_nothing_confirmed_has_no_drift(mode_on):
    assert asc.study_drift([]) is None
    assert asc.drift_mode([]) is None


def test_a_mode_nothing_crowds_on_is_not_believed(mode_on):
    """Matches spread evenly (no real cluster): the panel, which the mode
    cannot contradict, keeps the drift."""
    rng = np.random.default_rng(3)
    mets = [_met('y', list(rng.uniform(-0.05, 0.05, 60))),
            _met('l-alanine', [0.003, 0.004, 0.0035])]
    assert asc.crowded_mode(asc._confirmed_shifts(mets)) is None
    assert asc.study_drift(mets) == pytest.approx(0.0035)
