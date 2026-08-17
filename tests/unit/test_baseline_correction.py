"""Baseline correction: every offered method must preserve peak intensity.

The Signal Detection panel exposes arPLS / snip / rolling / constant. A method
that returns an all-zero spectrum, or that eats most of a peak, silently
destroys every downstream SNR and integral — so each is checked against a
synthetic spectrum with a known apex.
"""
import numpy as np
import pytest

from processing.baseline_correction import correct_baseline, snip

METHODS = ['arPLS', 'snip', 'rolling', 'constant']


def _lorentz(x, amp, cen, hwhm):
    return amp * hwhm ** 2 / ((x - cen) ** 2 + hwhm ** 2)


@pytest.fixture
def spectrum():
    n = 4000
    ppm = np.linspace(10.0, 0.0, n)
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 0.002, n)
    for cen, amp, w in [(8.0, 1.0, 0.006), (6.0, 0.8, 0.005),
                        (4.0, 1.0, 0.006), (2.0, 0.4, 0.020)]:
        y += _lorentz(ppm, amp, cen, w)
    return ppm, y


def test_lls_transform_round_trips():
    """SNIP's log-log-sqrt transform must invert exactly.

    Dropping the trailing -1 of the inverse inflates the baseline by one
    intensity unit; the clamp to y then makes baseline == y and the whole
    spectrum corrects to zero.
    """
    v = np.array([0.0, 0.5, 1.0, 10.0, 1e4])
    z = np.log(np.log(np.sqrt(v + 1) + 1) + 1)
    assert np.allclose((np.exp(np.exp(z) - 1) - 1) ** 2 - 1, v)


@pytest.mark.parametrize('method', METHODS)
def test_method_preserves_peak_height(spectrum, method):
    ppm, y = spectrum
    i_apex = int(np.argmin(np.abs(ppm - 8.0)))
    corrected, baseline = correct_baseline(ppm, y, method=method)

    assert corrected.shape == y.shape
    assert np.all(np.isfinite(corrected))
    # A flat-baseline spectrum must come back essentially intact.
    assert corrected[i_apex] > 0.85 * y[i_apex], (
        f'{method} removed {100 * (1 - corrected[i_apex] / y[i_apex]):.0f}% '
        f'of the apex')
    assert corrected.max() > 0, f'{method} returned an all-zero spectrum'


@pytest.mark.parametrize('method', METHODS)
def test_method_removes_a_broad_hump(spectrum, method):
    """A macromolecule-like hump must go; the sharp signal must stay."""
    ppm, y = spectrum
    hump = 0.15 * np.exp(-((ppm - 5.0) ** 2) / 20.0)
    i_apex = int(np.argmin(np.abs(ppm - 8.0)))
    corrected, baseline = correct_baseline(ppm, y + hump, method=method)

    assert baseline[i_apex] > 0.5 * hump[i_apex], (
        f'{method} did not follow the hump')
    assert corrected[i_apex] > 0.80 * y[i_apex]


def test_snip_scales_with_intensity(spectrum):
    """The fix must hold at raw-count magnitudes too, not only normalised ones.

    The missing -1 was invisible at 1e6 counts and fatal at unit scale, so both
    ends are pinned here.
    """
    ppm, y = spectrum
    i_apex = int(np.argmin(np.abs(ppm - 8.0)))
    for scale in (1.0, 1e6):
        corrected, _ = correct_baseline(ppm, y * scale, method='snip')
        assert corrected[i_apex] > 0.9 * y[i_apex] * scale


def test_snip_baseline_never_exceeds_signal(spectrum):
    ppm, y = spectrum
    baseline = snip(y)
    assert np.all(baseline <= y + 1e-12)
