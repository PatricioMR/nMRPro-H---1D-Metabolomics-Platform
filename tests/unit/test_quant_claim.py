"""An analyte is measured on every signal its own amount predicts, not on
one: a doublet that says ten millimolar says the molecule's singlet must
say ten millimolar too, whatever its protons."""
import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.quant_shared import CLAIM_RATIO, claim_listed_signals

FIELD = 600.0
PPM = np.linspace(0.5, 4.5, 40000)
FWHM = 2.6
LW = FWHM / FIELD
SIGMA = 1.0


def _line(centre, height, lines=((0.0, 1.0),)):
    return height * lines_model(PPM, list(lines), centre, FWHM, FIELD)


def _signal(ppm, mult='s', protons=1):
    return {'ppm_center': ppm, 'type': mult, 'protons': protons}


class _Cand:
    def __init__(self, ev, met):
        self.evidence = ev
        self.payload = {'met': met}


def _measured(db_peaks, own_ppm, nh):
    met = {'db_peaks': db_peaks}
    ev = {'db_ppm': own_ppm, 'pattern_field_mhz': FIELD,
          'study_drift_ppm': 0.0, 'multiplicity': 's', 'protons': nh}
    return {'A': {'signal': _Cand(ev, met), 'nh': nh}}


def _per_proton(centre, nh, X):
    """What the analyte's own window gives, so the test compares the
    claim against a real reading and not against a made-up number."""
    from processing.quant_integration import integrate_sample
    lo, hi = centre - 3 * LW / 2, centre + 3 * LW / 2
    a = integrate_sample(PPM, X[:, 0], lo, hi, local_baseline=False)
    return float(a) / nh


def _study(parts, n=3):
    cols = []
    for k in range(n):
        y = np.zeros(PPM.size)
        for centre, height in parts:
            y += _line(centre, height)
        cols.append(y)
    return np.column_stack(cols)


def test_a_second_signal_that_agrees_is_measured_too():
    """Two singlets of one molecule, one of three protons and one of one,
    laid at the same amount per proton."""
    db = [_signal(2.0, 's', 3), _signal(3.0, 's', 1)]
    X = _study([(2.0, 300.0), (3.0, 100.0)])
    own = _per_proton(2.0, 3, X)
    ppa = {'A': {0: own, 1: own, 2: own}}
    out = claim_listed_signals(_measured(db, 2.0, 3), ppa, PPM, X,
                               included=[0, 1, 2], linewidth_ppm=LW,
                               noise_sigma=SIGMA, local_baseline=False)
    assert out['A']['used'] == 1
    assert out['A']['signals'][0][0] == pytest.approx(3.0)


def test_a_region_somebody_else_is_in_is_left_alone():
    """The singlet the reference lists at 3.0 ppm is there, and so is a
    neighbour four times its size. Averaging that in would carry the
    neighbour into the number."""
    db = [_signal(2.0, 's', 3), _signal(3.0, 's', 1)]
    X = _study([(2.0, 300.0), (3.0, 100.0 * 4)])
    own = _per_proton(2.0, 3, X)
    ppa = {'A': {0: own, 1: own, 2: own}}
    before = dict(ppa['A'])
    out = claim_listed_signals(_measured(db, 2.0, 3), ppa, PPM, X,
                               included=[0, 1, 2], linewidth_ppm=LW,
                               noise_sigma=SIGMA, local_baseline=False)
    assert out['A']['used'] == 0
    assert out['A']['rejected'] and out['A']['rejected'][0][1] > CLAIM_RATIO
    assert ppa['A'] == before          # the amount is untouched


def test_a_signal_too_faint_to_read_is_not_read():
    """Its area would be noise divided by protons."""
    db = [_signal(2.0, 's', 3), _signal(3.0, 's', 1)]
    X = _study([(2.0, 300.0)])         # nothing at 3.0
    own = _per_proton(2.0, 3, X)
    ppa = {'A': {0: own, 1: own, 2: own}}
    out = claim_listed_signals(_measured(db, 2.0, 3), ppa, PPM, X,
                               included=[0, 1, 2], linewidth_ppm=LW,
                               noise_sigma=100.0, local_baseline=False)
    assert out['A']['used'] == 0


def test_the_analyte_s_own_signal_is_not_claimed_twice():
    db = [_signal(2.0, 's', 3)]
    X = _study([(2.0, 300.0)])
    ppa = {'A': {0: _per_proton(2.0, 3, X)}}
    out = claim_listed_signals(_measured(db, 2.0, 3), ppa, PPM, X,
                               included=[0], linewidth_ppm=LW,
                               noise_sigma=SIGMA, local_baseline=False)
    assert out['A']['used'] == 0
