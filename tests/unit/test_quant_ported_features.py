# tests/unit/test_quant_ported_features.py
"""The five things the old engine did that the new one had to learn.

Each is a feature the analyst can switch on in the Quantification panel. A
rewrite that dropped one silently would not fail anywhere — the run would
finish, the table would fill, and the number would be wrong in a way nothing
announces.
"""
import numpy as np
import pytest

from processing.quant_concentration import concentrations
from processing.quant_integration import (
    integrate_signal, subtract_blank, window_shifts)
from processing.quant_snr import (
    noise_level, representative_snr, window_snr)


def _grid():
    return np.linspace(0.0, 2.0, 401)


def _gauss(ppm, centre, height, width=0.01):
    return height * np.exp(-0.5 * ((ppm - centre) / width) ** 2)


def _peak(centre, **kw):
    p = {'rank': 1, 'ppm': centre,
         'start_ppm': centre - 0.05, 'end_ppm': centre + 0.05}
    p.update(kw)
    return p


# ── SNR ──────────────────────────────────────────────────────────────────────

def test_snr_is_reported_per_sample_so_absence_is_not_read_as_dilution():
    """One window is integrated in every sample. Where the analyte is absent it
    still integrates noise, and the area alone cannot say which happened."""
    ppm = _grid()
    strong = _gauss(ppm, 1.0, 100.0)
    absent = _gauss(ppm, 1.0, 0.0) + 1.0
    y = np.column_stack([strong, absent])
    snr = window_snr(ppm, y, 0.95, 1.05, noise=1.0)
    assert snr[0] > 50
    assert snr[1] < 5


def test_unknown_noise_gives_nan_not_zero():
    """A zero SNR reads as "no signal here". Nobody looked is a different fact."""
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    snr = window_snr(ppm, y, 0.95, 1.05, noise=None)
    assert np.isnan(snr[0])


def test_noise_comes_from_detection_when_detection_measured_it():
    """Quantification and the Signal List disagreeing about the noise level
    would make the same signal carry two SNRs."""
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    assert noise_level([_peak(1.0, noise=2.5)], ppm, y) == 2.5


def test_noise_is_reconstructed_from_the_peaks_own_snr_as_a_fallback():
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    n = noise_level([_peak(1.0, snr=50.0)], ppm, y)
    assert n == pytest.approx(2.0, rel=0.05)


def test_representative_snr_is_the_median_not_the_mean():
    """An analyte abundant in one group must not lend its SNR to the group it
    is absent from."""
    assert representative_snr({0: 100.0, 1: 2.0, 2: 3.0}) == 3.0


# ── Blank subtraction ────────────────────────────────────────────────────────

def test_the_blank_is_measured_in_the_window_it_corrects():
    """What the blanks integrate here is what this window collects without the
    analyte. Assuming a constant instead would correct the wrong amount."""
    ppm = _grid()
    y = np.column_stack([_gauss(ppm, 1.0, 100.0) + 5.0,
                         _gauss(ppm, 1.0, 0.0) + 5.0])
    out = integrate_signal([_peak(1.0)], ppm, y, included=[0],
                           blank_indices=[1], local_baseline=False)
    plain = integrate_signal([_peak(1.0)], ppm, y, included=[0],
                             local_baseline=False)
    assert out['areas'][0] < plain['areas'][0]
    assert out['blank_area'] > 0


def test_an_area_below_the_blank_stays_negative():
    """Clipping to zero makes absence indistinguishable from a trace."""
    assert subtract_blank({0: 1.0}, {1: 5.0})[0] == -4.0


def test_no_blanks_leaves_the_areas_alone():
    assert subtract_blank({0: 3.0}, {}) == {0: 3.0}


# ── Per-sample window alignment ──────────────────────────────────────────────

def test_the_window_follows_the_drift_detection_measured():
    """A resonance drifting with pH would otherwise be measured on its shoulder
    in half the study, and the difference would read as concentration."""
    sh = window_shifts([_peak(1.0, ppm_per_sample=[1.0, 1.02])])
    assert sh[0] == pytest.approx(0.0)
    assert sh[1] == pytest.approx(0.02)


def test_a_drift_larger_than_the_window_is_clamped_not_followed():
    """Past that the window leaves the multiplet, and a drift that large is a
    detection problem, not something integration should absorb."""
    sh = window_shifts([_peak(1.0, ppm_per_sample=[1.0, 9.0])])
    assert sh[1] == pytest.approx(0.1 / 3.0)


def test_a_sample_detection_did_not_align_keeps_its_window():
    """Moving a window onto a centre nobody trusted is worse than not moving."""
    sh = window_shifts([_peak(1.0, ppm_per_sample=[1.0, 1.02],
                              aligned_per_sample=[True, False])])
    assert 1 not in sh


def test_alignment_is_off_unless_asked_for():
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    out = integrate_signal([_peak(1.0, ppm_per_sample=[1.05])], ppm, y)
    assert out['shifts'] == {}


# ── Proton correction ────────────────────────────────────────────────────────

def test_against_a_standard_the_proton_division_is_not_optional():
    """The ratio of two areas is a ratio of concentrations only once both are
    per proton. Honouring the flag here would return a number that looks like a
    concentration and is not one."""
    with_flag = concentrations({0: 6.0}, 3.0, standard={0: 1.0},
                               standard_conc=1.0, proton_correction=True)
    without = concentrations({0: 6.0}, 3.0, standard={0: 1.0},
                             standard_conc=1.0, proton_correction=False)
    assert with_flag[0] == without[0] == pytest.approx(2.0)


def test_without_a_standard_the_flag_decides():
    """The output is an area, and whether it is per proton is the analyst's."""
    assert concentrations({0: 6.0}, 3.0, proton_correction=True)[0] == 2.0
    assert concentrations({0: 6.0}, 3.0, proton_correction=False)[0] == 6.0


# ── The internal standard's own row ──────────────────────────────────────────

def test_the_standard_gets_a_row_and_reads_back_its_nominal_concentration():
    """Recovery is the check on the whole run: the standard was added at a
    known concentration, so reading it against itself must return that. When it
    does not, every area it divided is suspect too."""
    from processing.quant_engine import quantify
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    rows = quantify(candidates=[], peaks_by_rank={}, ppm=ppm, intensities=y,
                    samples_info=[{}], included=[0], groups={'A': [0]},
                    dj_gold_hz=0.5, standard_areas={0: 10.0}, standard_nh=9.0,
                    standard_conc=1.0, standard_name='TSP', standard_ppm=0.0)
    is_row = next(r for r in rows if r.get('_is_is'))
    assert is_row['metabolite'] == 'TSP'
    assert is_row['conc_mm'] == pytest.approx(1.0)
    assert not is_row['_not_quantified']


def test_no_standard_means_no_standard_row():
    from processing.quant_engine import quantify
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    rows = quantify(candidates=[], peaks_by_rank={}, ppm=ppm, intensities=y,
                    samples_info=[{}], included=[0], groups={}, dj_gold_hz=0.5)
    assert not any(r.get('_is_is') for r in rows)


def test_an_external_standards_recovery_is_read_where_it_was_injected():
    """An externally injected standard is absent from the analytical samples by
    design. Integrating its window there measures the baseline underneath it,
    and averaging that in reports a preparation failure that did not happen —
    on a 21-sample study with 5 injections it read 57.5% recovery."""
    from processing.quant_engine import quantify
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    areas = {0: 90.0, 1: 90.0}          # the injections
    areas.update({s: 0.5 for s in range(2, 12)})   # analytical samples: no IS
    rows = quantify(candidates=[], peaks_by_rank={}, ppm=ppm, intensities=y,
                    samples_info=[{}], included=[0], groups={},
                    dj_gold_hz=0.5, standard_areas=areas, standard_nh=9.0,
                    standard_conc=5.38, standard_name='TSP',
                    external_standard_samples=[0, 1])
    is_row = next(r for r in rows if r.get('_is_is'))
    assert is_row['conc_mm'] == pytest.approx(5.38, rel=0.01)


def test_an_internal_standard_is_still_read_across_every_sample():
    """It is in every sample there, so restricting would throw away the check."""
    from processing.quant_engine import quantify
    ppm = _grid()
    y = _gauss(ppm, 1.0, 100.0).reshape(-1, 1)
    rows = quantify(candidates=[], peaks_by_rank={}, ppm=ppm, intensities=y,
                    samples_info=[{}], included=[0], groups={},
                    dj_gold_hz=0.5, standard_areas={0: 90.0, 1: 45.0},
                    standard_nh=9.0, standard_conc=1.0, standard_name='TSP')
    is_row = next(r for r in rows if r.get('_is_is'))
    assert len(is_row['_areas_per_sample']) == 2


def test_the_snr_is_a_height_not_an_absolute_intensity():
    """A window on a baseline offset would otherwise report the offset as
    signal. In the sugar region of one study the raw intensity ran from
    -1.04e5 to +3.98e4, and the SNR said more about where the baseline sat
    than about the analyte."""
    ppm = _grid()
    signal = _gauss(ppm, 1.0, 100.0)
    on_the_floor = signal
    on_a_hill = signal + 50000.0
    flat = window_snr(ppm, on_the_floor.reshape(-1, 1), 0.95, 1.05, noise=1.0)
    hill = window_snr(ppm, on_a_hill.reshape(-1, 1), 0.95, 1.05, noise=1.0)
    assert hill[0] == pytest.approx(flat[0], rel=1e-6)


def test_it_uses_the_window_endpoints_as_the_baseline():
    """The same straight line the integration subtracts, so the SNR and the
    area describe one measurement rather than two."""
    ppm = _grid()
    y = (_gauss(ppm, 1.0, 100.0) + 10.0 * ppm).reshape(-1, 1)
    snr = window_snr(ppm, y, 0.95, 1.05, noise=1.0)
    assert snr[0] == pytest.approx(100.0, rel=0.05)


def test_quantification_measures_its_snr_against_the_same_noise_as_its_limits():
    """Two answers to "how loud is the noise" put an SNR of 3149 beside a
    concentration of 0.008 mM."""
    import inspect

    from processing import quant_engine as qe
    src = inspect.getsource(qe.quantify)
    i_sigma = src.index('sigma_here = local_noise_sigma')
    i_snr = src.index('snr = window_snr(')
    assert i_sigma < i_snr, 'the noise must be settled before the SNR uses it'
    assert 'window_snr(ppm, intensities, lo, hi, sigma_here' in src


def test_a_peak_on_the_window_edge_still_has_a_height():
    """The endpoint line runs through the signal when an apex sits on a window
    edge — a spectrum that ends there, or a window clipped by the axis. On a
    synthetic study whose axis began at 0.0 ppm with the standard at 0.0, the
    line ran from the apex down to the tail and the height came out zero: the
    internal standard was reported with no signal at all."""
    ppm = _grid()
    y = _gauss(ppm, 0.0, 100.0).reshape(-1, 1)     # apex at the axis start
    snr = window_snr(ppm, y, 0.0, 0.05, noise=1.0)
    assert snr[0] > 50, 'a peak at the edge must not read as no signal'
