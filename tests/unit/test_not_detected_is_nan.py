"""A sample the analyte was not detected in is NaN wherever the table is read.

The fit and the integral hand every sample a number whatever the data hold, a
zero is the engine saying it found nothing, and a number written where an
analyte was not seen puts a measurement where there is none (Helsel 2006,
Chemosphere 65:2434). So what the analyst is given -- the statistics, the CSV,
the MetaboAnalyst file -- reads every value through ``reported_values``, and
nothing in the statistics fills a NaN with the column's mean.
"""
import numpy as np
import pytest

from gui.stages._quant_adapter import identity_columns, reported_values
from gui.stages._quant_helpers import _pls_vip
from processing.quant_cascade import Column
from processing.quant_engine import _fit_detected

NAN = float('nan')


def test_a_value_is_a_number_only_where_the_analyte_was_detected():
    row = {'_concs_per_sample': [1.0, 2.0, 3.0, NAN, 0.0],
           '_detected_per_sample': [True, False, True, True, True]}
    vals, nd = reported_values(row, 5)
    assert vals[0] == 1.0 and vals[2] == 3.0
    assert np.isnan(vals[1]) and nd[1], 'detected False: n.d.'
    assert np.isnan(vals[3]) and not nd[3], 'never measured: NA, not n.d.'
    assert np.isnan(vals[4]) and nd[4], 'a zero is nothing found, not a measurement'


def test_a_row_with_no_detection_record_keeps_its_positive_numbers():
    row = {'_concs_per_sample': [1.0, 2.0], '_detected_per_sample': [NAN, NAN]}
    vals, nd = reported_values(row, 2)
    assert vals == [1.0, 2.0] and nd == [False, False]


def test_a_short_vector_is_padded_with_na():
    vals, nd = reported_values({'_concs_per_sample': [1.0]}, 3)
    assert vals[0] == 1.0 and np.isnan(vals[1]) and np.isnan(vals[2])
    assert nd == [False, False, False]


def test_numpy_booleans_count_as_detection_verdicts():
    row = {'_concs_per_sample': [1.0, 2.0],
           '_detected_per_sample': [np.True_, np.False_]}
    vals, nd = reported_values(row, 2)
    assert vals[0] == 1.0 and np.isnan(vals[1]) and nd[1]


def test_the_exports_carry_the_identity_of_every_row():
    assert identity_columns({'_by_cascade': True, '_confidence': 1,
                             '_identity_why': 'x'}) == (1, 'Cascade estimate', 'x')
    assert identity_columns({'_joint_fit': {'group': []}, '_confidence': 3}
                            )[:2] == (3, 'Joint fit')
    assert identity_columns({'_fit_refused': True, '_confidence': 1}
                            )[1] == 'Integral (joint fit refused)'
    assert identity_columns({'_integ_mode': 'Multiplet', '_confidence': 2}
                            )[:2] == (2, 'Multiplet')
    assert identity_columns({})[0] == ''


# ── the VIP does not fill what was not detected ──────────────────────────────

def _vip_data(seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    X = rng.normal(size=(6, 4))
    X[:, 0] += 2.0 * y
    return X, y


def test_a_feature_with_a_not_detected_sample_gets_no_vip_and_does_not_bend_the_rest():
    X, y = _vip_data()
    full = _pls_vip(X, y, n_components=1)
    Xn = X.copy()
    Xn[2, 1] = np.nan
    got = _pls_vip(Xn, y, n_components=1)
    assert np.isnan(got[1])
    # the others are what they are on the three complete features alone: the
    # missing cell was not filled with its column mean to keep four in the model
    only = _pls_vip(X[:, [0, 2, 3]], y, n_components=1)
    assert got[[0, 2, 3]] == pytest.approx(only)
    assert np.all(np.isfinite(full))


def test_all_features_incomplete_gives_nothing():
    X, y = _vip_data()
    X[0, :] = np.nan
    assert np.all(np.isnan(_pls_vip(X, y, n_components=1)))


# ── the cascade's fit is held to the noise sample by sample ──────────────────

FWHM = 2.6 / 600.0
PPM = np.linspace(4.0, 1.0, 30000)


class _Comp:
    def __init__(self, c):
        self.lo, self.hi, self.shift_ppm = c - 0.03, c + 0.03, 0.0
        self.width_scale, self.lines, self.c = 1.0, [c], c

    def model(self, x, shift, width):
        g = FWHM * width / 2.0
        return g ** 2 / ((x - self.c - shift) ** 2 + g ** 2)


def _column():
    col = Column('m', 'm')
    col.parts.append((_Comp(2.0), 1.0))
    return col


def test_the_cascades_fit_is_detected_only_where_it_stands_above_the_noise():
    col = _column()
    top = max(v.max() for _i, v in col.part_curves(PPM, 0.0))
    noise = 10.0
    # amplitude x the signal's height per unit amplitude over the noise
    amps = {0: 10.0 * noise / top, 1: 1.0 * noise / top, 2: 0.0}
    got = _fit_detected(col, PPM, {'shift_ppm': 0.0, 'per_proton': amps}, noise)
    assert got == {0: True, 1: False, 2: False}


def test_without_a_noise_or_a_column_nothing_is_claimed():
    res = {'shift_ppm': 0.0, 'per_proton': {0: 1.0}}
    assert _fit_detected(_column(), PPM, res, None) == {}
    assert _fit_detected(None, PPM, res, 10.0) == {}
