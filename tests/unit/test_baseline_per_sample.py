"""Baseline correction must be estimated per sample, and be auditable.

In plasma and serum the broad macromolecule envelope is real NMR signal whose
amplitude differs between individuals — protein content is the biological
variable.  Estimating one baseline on the mean spectrum and reusing it with a
per-sample constant offset cannot absorb an envelope that differs in shape, so
it injects variance into areas that should be identical across samples.
"""
import numpy as np
import pytest

from processing.peak_detection import detect_peaks

NOISE = 0.002
HUMPS = [0.0, 0.20, 0.50, 1.00]      # macromolecule envelope per sample
METHODS = ['arPLS', 'snip', 'rolling']


def _lorentz(x, amp, cen, hwhm):
    return amp * hwhm ** 2 / ((x - cen) ** 2 + hwhm ** 2)


@pytest.fixture
def varying_hump():
    """Identical analyte on top of a per-sample macromolecule envelope."""
    n = 6000
    ppm = np.linspace(10.0, 0.0, n)
    rng = np.random.default_rng(7)
    cols = []
    for h in HUMPS:
        y = rng.normal(0.0, NOISE, n)
        y += _lorentz(ppm, 1.0, 8.0, 0.006)          # same in every sample
        y += h * np.exp(-((ppm - 7.5) ** 2) / 2.0)   # differs per sample
        cols.append(y)
    samples = [{'label': 'G1', 'type': 'Sample'}] * len(HUMPS)
    return ppm, np.column_stack(cols), samples


def _analyte(ppm, Y, samples, method):
    details = detect_peaks(
        ppm, Y, noise_std=NOISE, min_snr=6.0, min_distance=0.005,
        baseline_correct=True, baseline_method=method,
        global_align=False, samples_info=samples, return_details=True,
        per_sample_detection=False, min_group_presence=1)
    hits = [p for p in details['peaks'] if abs(p['ppm'] - 8.0) < 0.05]
    assert hits, f'{method}: analyte at 8.0 ppm was not detected'
    return hits[0], details


@pytest.mark.parametrize('method', METHODS)
def test_area_is_stable_across_differing_humps(varying_hump, method):
    """A constant analyte must give a constant area whatever sits under it."""
    peak, _ = _analyte(*varying_hump, method)
    areas = np.asarray(peak['area_per_sample'], dtype=float)
    assert areas.size == len(HUMPS)
    cv = 100.0 * areas.std(ddof=1) / areas.mean()
    assert cv < 5.0, (
        f'{method}: {cv:.1f}% CV on an area that is identical by construction '
        f'— the baseline is tracking the envelope, not the analyte ({areas})')


@pytest.mark.parametrize('method', METHODS)
def test_removed_broad_component_is_recorded(varying_hump, method):
    """The subtraction of real signal must leave an audit trail."""
    peak, _ = _analyte(*varying_hump, method)
    frac = np.asarray(peak['broad_removed_frac_per_sample'], dtype=float)
    assert frac.size == len(HUMPS)
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    # Bigger envelope → larger share of the raw area removed.
    assert frac[0] < frac[-1]
    assert frac[-1] > 0.5, (
        f'{method}: only {frac[-1]:.0%} recorded as removed for the largest '
        f'hump — the audit trail understates the subtraction')
    assert 0.0 <= peak['broad_removed_frac'] <= 1.0


def test_no_correction_records_nothing_removed(varying_hump):
    ppm, Y, samples = varying_hump
    details = detect_peaks(
        ppm, Y, noise_std=NOISE, min_snr=6.0, min_distance=0.005,
        baseline_correct=False, global_align=False, samples_info=samples,
        return_details=True, per_sample_detection=False, min_group_presence=1)
    for p in details['peaks']:
        assert p['broad_removed_frac'] == 0.0


def test_details_expose_the_baseline_model(varying_hump):
    """Quantification inherits these; they must survive in the result."""
    ppm, Y, samples = varying_hump
    details = detect_peaks(
        ppm, Y, noise_std=NOISE, min_snr=6.0, min_distance=0.005,
        baseline_correct=True, baseline_method='arPLS', baseline_lam=1e5,
        global_align=False, samples_info=samples, return_details=True,
        per_sample_detection=False, min_group_presence=1)
    assert details['baseline_method'] == 'arPLS'
    assert details['baseline_lam'] == pytest.approx(1e5)
    assert details['baseline_per_sample'].shape == Y.shape
    # Each sample got its own baseline, not a shared one plus an offset.
    spread = details['baseline_per_sample'].std(axis=1)
    assert spread.max() > 0.05, 'baselines look identical across samples'


# ── The single project-wide baseline decision ─────────────────────────────────

def test_model_round_trips_through_a_dict():
    from processing.baseline_model import BaselineModel
    m = BaselineModel(method='snip', params={'niter': 77})
    assert BaselineModel.from_dict(m.to_dict()) == m
    assert BaselineModel.from_dict(m.to_dict()).params == {'niter': 77}


def test_model_fills_the_right_parameter_per_method():
    from processing.baseline_model import BaselineModel
    assert BaselineModel(method='snip').params == {'niter': 40}
    assert BaselineModel(method='arPLS').params == {'lam': 1e5}
    assert BaselineModel(method='rolling').params == {}
    # An unknown method must not produce a model that cannot run.
    assert BaselineModel(method='nonsense').method == 'snip'


def test_disabled_model_returns_the_input_untouched(varying_hump):
    from processing.baseline_model import BaselineModel
    ppm, Y, _ = varying_hump
    assert np.array_equal(BaselineModel(enabled=False).apply(ppm, Y), Y)


def test_apply_never_mutates_the_raw_matrix(varying_hump):
    """The raw matrix is the record of what was measured."""
    from processing.baseline_model import BaselineModel
    ppm, Y, _ = varying_hump
    before = Y.copy()
    BaselineModel(method='snip').apply(ppm, Y)
    assert np.array_equal(Y, before)


def test_every_reader_gets_the_same_corrected_matrix(varying_hump):
    """Detection, quantification, statistics and MCR must not disagree.

    They used to: detection corrected a working copy, quantification ran its
    own hard-coded airPLS, and statistics and MCR read the raw matrix with no
    correction at all.
    """
    from processing.baseline_model import BaselineModel, BaselineCache
    ppm, Y, _ = varying_hump
    cache = BaselineCache()
    model = BaselineModel(method='snip', params={'niter': 40})

    first = cache.corrected(ppm, Y, model)
    assert not np.array_equal(first, Y), 'nothing was corrected'
    assert cache.corrected(ppm, Y, model) is first, 'recomputed needlessly'


def test_changing_the_model_recomputes(varying_hump):
    from processing.baseline_model import BaselineModel, BaselineCache
    ppm, Y, _ = varying_hump
    cache = BaselineCache()
    shallow = cache.corrected(ppm, Y, BaselineModel(method='snip',
                                                    params={'niter': 5}))
    deep = cache.corrected(ppm, Y, BaselineModel(method='snip',
                                                 params={'niter': 150}))
    assert not np.array_equal(shallow, deep), 'the new model was ignored'


def test_replacing_the_data_recomputes(varying_hump):
    from processing.baseline_model import BaselineModel, BaselineCache
    ppm, Y, _ = varying_hump
    cache = BaselineCache()
    model = BaselineModel(method='snip')
    first = cache.corrected(ppm, Y, model)
    second = cache.corrected(ppm, Y * 2.0, model)
    assert not np.array_equal(first, second), 'stale correction reused'


def test_no_work_needed_returns_the_input_itself(varying_hump):
    """An off model, or data already corrected upstream, must cost nothing."""
    from processing.baseline_model import BaselineModel, BaselineCache
    ppm, Y, _ = varying_hump
    cache = BaselineCache()
    assert cache.corrected(ppm, Y, BaselineModel(enabled=False)) is Y
    assert cache.corrected(ppm, Y, BaselineModel(method='snip'),
                           already_corrected=True) is Y
    assert cache.corrected(ppm, None, BaselineModel()) is None


def test_broad_removed_fraction_tracks_the_envelope(varying_hump):
    from processing.baseline_model import BaselineModel, broad_removed_fraction
    ppm, Y, _ = varying_hump
    corrected = BaselineModel(method='snip').apply(ppm, Y)
    lo = int(np.argmin(np.abs(ppm - 8.05)))
    hi = int(np.argmin(np.abs(ppm - 7.95)))
    frac = broad_removed_fraction(Y, corrected, ppm, min(lo, hi), max(lo, hi))
    assert frac.shape == (len(HUMPS),)
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    assert frac[0] < frac[-1]


def test_precorrected_path_matches_the_internal_one(varying_hump):
    """Feeding detect_peaks already-corrected data must change nothing.

    The GUI corrects once via the project model and hands detect_peaks the
    baselines that were removed, instead of letting it correct a second time.
    Both routes have to agree, or the refactor moved the numbers.
    """
    from processing.baseline_model import BaselineModel, BaselineCache
    ppm, Y, samples = varying_hump

    common = dict(noise_std=NOISE, min_snr=6.0, min_distance=0.005,
                  global_align=False, samples_info=samples, return_details=True,
                  per_sample_detection=False, min_group_presence=1)

    internal = detect_peaks(ppm, Y, baseline_correct=True,
                            baseline_method='snip', **common)

    corrected = BaselineCache().corrected(ppm, Y, BaselineModel(method='snip'))
    external = detect_peaks(ppm, corrected, baseline_correct=False,
                            baseline_2d=(Y - corrected), **common)

    a = [p for p in internal['peaks'] if abs(p['ppm'] - 8.0) < 0.05][0]
    b = [p for p in external['peaks'] if abs(p['ppm'] - 8.0) < 0.05][0]
    assert np.allclose(a['area_per_sample'], b['area_per_sample'])
    assert np.allclose(a['broad_removed_frac_per_sample'],
                       b['broad_removed_frac_per_sample'])
    assert a['signal_quality'] == b['signal_quality']


def test_historical_airpls_spelling_still_resolves():
    """Projects saved before the rename must reopen on the same algorithm.

    The function was called airPLS for years, but it implements arPLS (Baek
    2015) — a different method from Zhang's airPLS (2010). The name was fixed;
    the stored spelling has to keep working.
    """
    from processing.baseline_correction import arPLS, airPLS, normalize_method
    from processing.baseline_model import BaselineModel

    assert airPLS is arPLS
    assert normalize_method('airPLS') == 'arPLS'
    assert normalize_method('AIRPLS') == 'arPLS'
    assert normalize_method('arPLS') == 'arPLS'
    # An unrecognised name must not crash a reload; it falls back to the default.
    assert normalize_method('nonsense') == 'snip'

    old = BaselineModel.from_dict(
        {'enabled': True, 'method': 'airPLS', 'params': {'lam': 2.5e4}})
    assert old.method == 'arPLS'
    assert old.params == {'lam': 2.5e4}
