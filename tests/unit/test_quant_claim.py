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


class TestAnUnreadableSignal:
    """A reference may list a signal this cannot predict the height of: a
    multiplicity whose lines it does not know how to count. Such a signal
    can neither confirm nor contradict an amount, and it must not end
    the run."""

    def test_an_unknown_multiplicity_is_reported_not_raised(self):
        import numpy as np
        from processing.quant_shared import fingerprint_holds
        ppm = np.linspace(0.0, 4.0, 4001)
        trace = np.zeros_like(ppm)
        held, verdicts = fingerprint_holds(
            [{'ppm': 2.0, 'type': 'nonagonal-something', 'protons': 3}],
            per_proton_area=1.0e4, ppm=ppm, trace=trace,
            noise=100.0, linewidth_ppm=0.005)
        assert [v[3] for v in verdicts] == ['unreadable']
        # neither present nor contradicted: it says nothing either way
        assert held is True


class TestSignalsCorroborateEachOther:
    """Creatine, betaine, creatinine, glycine and acetate -- five of the
    eighteen analytes of the reference study -- carry nothing but singlets,
    because nothing in those molecules has a neighbour to couple to. No
    signal of theirs can ever be confirmed the way a multiplet is. What
    they do have is each other: two singlets of one molecule stand a fixed
    distance apart and in the fixed ratio of their proton counts."""

    @staticmethod
    def _spectrum(peaks, width=0.005):
        import numpy as np
        ppm = np.linspace(0.5, 4.5, 40001)
        y = np.zeros_like(ppm)
        for centre, area in peaks:
            hw = width / 2.0
            y += (area / np.pi) * hw / ((ppm - centre) ** 2 + hw ** 2)
        return ppm, y

    @staticmethod
    def _creatine(peak_a=3.928, peak_b=3.031):
        """Its CH2 (2H, 3.920) and its N-CH3 (3H, 3.024)."""
        return {
            'db_peaks': [{'ppm': 3.920, 'type': 's', 'protons': 2},
                         {'ppm': 3.024, 'type': 's', 'protons': 3}],
            'matched_peaks': [
                {'db_ppm': 3.920, 'type': 's', 'peak_ppm': peak_a,
                 'ppm_off_drift': 0.001},
                {'db_ppm': 3.024, 'type': 's', 'peak_ppm': peak_b,
                 'ppm_off_drift': 0.001}]}

    def test_two_singlets_in_the_listed_ratio_corroborate(self):
        from processing.quant_shared import signals_corroborate
        ppm, y = self._spectrum([(3.928, 2000.0), (3.031, 3000.0)])
        assert signals_corroborate(self._creatine(), ppm, y,
                                    linewidth_ppm=0.005)

    def test_a_ratio_the_protons_do_not_explain_does_not(self):
        """Same two positions, but the 3H singlet reads like 30H: whatever
        stands there, the pair is not this molecule."""
        from processing.quant_shared import signals_corroborate
        ppm, y = self._spectrum([(3.928, 2000.0), (3.031, 30000.0)])
        assert not signals_corroborate(self._creatine(), ppm, y,
                                        linewidth_ppm=0.005)

    def test_a_single_singlet_has_nothing_to_link_to(self):
        from processing.quant_shared import signals_corroborate
        ppm, y = self._spectrum([(3.928, 2000.0)])
        met = self._creatine()
        met['matched_peaks'] = met['matched_peaks'][:1]
        assert not signals_corroborate(met, ppm, y, linewidth_ppm=0.005)

    def test_singlets_that_are_all_somebody_elses_do_not(self):
        """Adenosine passed on inosine's evidence: purine analogues whose
        aromatic singlets sit 3 mppm apart, same 0.118 ppm separation, same
        1:1 ratio, and the one singlet that tells them apart is under the
        suppressed water. At least one of the pair has to be the analyte's
        own peak."""
        from processing.quant_shared import signals_corroborate
        ppm, y = self._spectrum([(3.928, 2000.0), (3.031, 3000.0)])
        assert not signals_corroborate(
            self._creatine(), ppm, y, linewidth_ppm=0.005,
            taken_peaks=(3.928, 3.031))
        # one of its own is enough
        assert signals_corroborate(
            self._creatine(), ppm, y, linewidth_ppm=0.005,
            taken_peaks=(3.031,))

    def test_a_singlet_outside_the_study_is_not_counted(self):
        from processing.quant_shared import signals_corroborate
        ppm, y = self._spectrum([(3.928, 2000.0), (3.031, 3000.0)])
        met = self._creatine()
        met['matched_peaks'][1]['ppm_off_drift'] = 0.043
        assert not signals_corroborate(met, ppm, y, linewidth_ppm=0.005)

    def test_a_doublet_and_a_singlet_are_asked_the_same_question(self):
        """The integral of a multiplet is nH x (area per proton) whatever
        its multiplicity -- line shares decide height, not area. So an
        analyte's singlet and its doublet can be asked the same question:
        how much of you is there? Asking it of the singlets alone promoted
        2 analytes on the reference study; asking it of every signal
        promoted 8, and no stranger either way."""
        import numpy as np
        from processing.quant_shared import signals_corroborate
        j_hz, mhz = 7.0, 500.0
        half = (j_hz / mhz) / 2.0
        # 2H in one line at 3.900, 3H split over two lines at 1.500 +- J/2
        ppm, y = self._spectrum([(3.900, 2000.0),
                                 (1.500 - half, 1500.0),
                                 (1.500 + half, 1500.0)])
        met = {
            'db_peaks': [{'ppm': 3.900, 'type': 's', 'protons': 2},
                         {'ppm': 1.500, 'type': 'd', 'protons': 3,
                          'j_coupling': str(j_hz)}],
            'matched_peaks': [
                {'db_ppm': 3.900, 'type': 's', 'peak_ppm': 3.900,
                 'ppm_off_drift': 0.0},
                {'db_ppm': 1.500, 'type': 'd', 'peak_ppm': 1.500 - half,
                 'ppm_off_drift': 0.0}]}
        assert signals_corroborate(met, ppm, y, linewidth_ppm=0.005,
                                   field_mhz=mhz)

    def test_a_signal_under_a_suppressed_region_is_not_asked(self):
        """Its area is the solvent's, not the analyte's."""
        from processing.quant_shared import signals_corroborate
        ppm, y = self._spectrum([(3.928, 2000.0), (3.031, 3000.0)])
        met = self._creatine()
        assert not signals_corroborate(met, ppm, y, linewidth_ppm=0.005,
                                       excluded_regions=[(3.00, 3.06)])
