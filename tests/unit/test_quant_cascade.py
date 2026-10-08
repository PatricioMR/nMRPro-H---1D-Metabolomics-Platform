"""The cascade adds analytes the election missed, on what the quantified
ones leave of the spectrum (ADR-028): one amplitude for all of an
analyte's signals, every signal capping it, and nothing the election
audited moved."""
import numpy as np
import pytest

from processing.pattern_shape import lines_model
from processing.quant_cascade import (
    CAP_RATIO, MAX_SIGNAL_DISAGREEMENT, MIN_FREE_TO_QUANTIFY, amplitude_cap, build_columns, cascade,
    cascade_missed, exclusion_weight, signal_protons, subtract_fixed)
from processing.quant_engine import quantify
from processing.quantification import CASCADE_CROWDED, Candidate

FIELD = 600.0
PPM = np.linspace(0.5, 4.5, 40000)
FWHM = 2.6
LW = FWHM / FIELD
SIGMA = 1.0
J = 7.0
DOUBLET = ((-J / 2, 1.0), (J / 2, 1.0))


def _line(centre, height, lines=((0.0, 1.0),)):
    return height * lines_model(PPM, list(lines), centre, FWHM, FIELD)


def _signal(ppm, mult='s', protons=1, j=None):
    return {'ppm_center': ppm, 'type': mult, 'protons': protons,
            'j_coupling': (f'{j:.2f}' if j else None)}


def _met(mid, name, signals):
    return {'id': mid, 'name': name, 'db_peaks': signals, 'matched_peaks': []}


def _study(parts, n=3, seed=0, sigma=SIGMA):
    """One column per sample, each analyte's height scaled by the sample."""
    rng = np.random.default_rng(seed)
    cols = []
    for k in range(n):
        y = np.zeros(PPM.size)
        for centre, height, lines in parts:
            y += _line(centre, height * (1.0 + 0.1 * k), lines)
        cols.append(y + rng.normal(0, sigma, PPM.size))
    return np.column_stack(cols)


# ── the column ──────────────────────────────────────────────────────────

def test_a_column_holds_every_listed_signal_at_its_proton_count():
    met = _met('m1', 'Analyte', [_signal(1.0, 's', 3), _signal(3.0, 'd', 2, J)])
    col, = build_columns([met], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.0)
    assert col.nh_total == pytest.approx(5.0)
    # one amplitude, and each signal's area is its protons
    areas = []
    for idx, val in col.part_curves(PPM, 0.0):
        order = np.argsort(PPM[idx])
        areas.append(float(np.trapezoid(val[order], x=PPM[idx][order])))
    assert sorted(areas) == pytest.approx([2.0, 3.0], rel=0.05)


def test_a_signal_the_reference_did_not_measure_leaves_no_column():
    # a bare 'm' has no shape and no coupling: nothing places it
    met = _met('m1', 'Analyte', [_signal(1.0, 'm', 2)])
    assert build_columns([met], field_mhz=FIELD, linewidth_ppm=LW,
                         drift_ppm=0.0) == []


def test_the_protons_come_from_the_profile_where_the_reference_measured_one():
    # stoichiometry, not the share measured on the reference spectrum
    assert signal_protons({'protons': 3, 'profile_nh': 3.35}) == 3.0
    assert signal_protons({'protons': 2}) == 2.0
    assert signal_protons({'protons': '', 'profile_nh': None}) is None


def test_the_column_is_laid_at_the_listed_position_plus_the_study_drift():
    met = _met('m1', 'Analyte', [_signal(2.0, 's', 1)])
    col, = build_columns([met], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.01)
    idx, val = col.curve(PPM, 0.0)
    assert PPM[idx[int(np.argmax(val))]] == pytest.approx(2.01, abs=0.002)


# ── the cap ─────────────────────────────────────────────────────────────

def test_every_signal_caps_the_amplitude_and_the_tightest_one_wins():
    """An analyte with a tall signal and a signal over an empty region
    cannot be taller than the empty region allows."""
    met = _met('m1', 'Analyte', [_signal(1.0, 's', 1), _signal(3.0, 's', 1)])
    col, = build_columns([met], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.0)
    trace = _line(1.0, 500.0)              # only the first signal is there
    w = np.ones(PPM.size)
    cap = amplitude_cap(col.part_curves(PPM, 0.0), trace, w, SIGMA)
    idx, val = col.curve(PPM, 0.0)
    # at the cap the analyte stands no higher than the empty region holds
    assert cap * float(val.max()) < 20 * SIGMA


def test_a_signal_inside_an_excluded_region_does_not_cap():
    met = _met('m1', 'Analyte', [_signal(1.0, 's', 1), _signal(3.0, 's', 1)])
    col, = build_columns([met], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.0)
    trace = _line(1.0, 500.0)
    w = exclusion_weight(PPM, [(2.9, 3.1)])
    capped = amplitude_cap(col.part_curves(PPM, 0.0), trace, w, SIGMA)
    assert capped > 100 * amplitude_cap(col.part_curves(PPM, 0.0), trace,
                                        np.ones(PPM.size), SIGMA)


def test_an_analyte_asking_far_more_than_its_cap_does_not_enter():
    """The rule of ADR-025 on the whole column: a claim its own listing
    says it cannot carry is not taken."""
    honest = _met('good', 'Honest', [_signal(1.0, 's', 1)])
    # its second signal would be as tall as the first, and nothing is there
    greedy = _met('bad', 'Greedy', [_signal(1.0, 's', 1), _signal(3.0, 's', 1)])
    cols = build_columns([honest, greedy], field_mhz=FIELD, linewidth_ppm=LW,
                         drift_ppm=0.0)
    trace = _line(1.0, 500.0)
    res = cascade(cols, PPM, trace, sigma=SIGMA, linewidth_ppm=LW)
    assert [e.col.mid for e in res.active] == ['good']
    assert 'bad' in res.contradicted
    proj, cap = res.contradicted['bad']
    assert proj > CAP_RATIO * cap


def test_nothing_below_the_bar_is_taken_in():
    met = _met('m1', 'Analyte', [_signal(1.0, 's', 1)])
    cols = build_columns([met], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.0)
    res = cascade(cols, PPM, _line(1.0, 2.0), sigma=SIGMA, linewidth_ppm=LW,
                  min_snr=10.0)
    assert res.active == []
    assert res.stopped_at is not None


# ── what the quantified analytes leave ──────────────────────────────────

def test_a_quantified_analyte_is_subtracted_and_not_offered_again():
    met = _met('m1', 'Analyte', [_signal(1.0, 's', 2)])
    cols = build_columns([met], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.0)
    X = np.column_stack([_line(1.0, 300.0)] * 2)
    idx, val = cols[0].curve(PPM, 0.0)
    amp = 300.0 / float(val.max())
    Y, free, owned = subtract_fixed(cols, PPM, X, {'m1': {0: amp, 1: amp}},
                                    sigma=SIGMA)
    assert free == []
    # what is left of it is the tails outside the modelled window, not a signal
    assert abs(Y[:, 0]).max() < 0.05 * 300.0
    assert owned[idx[int(np.argmax(val))]]     # its region is spoken for


def test_the_remainder_of_a_subtracted_signal_is_not_a_new_analyte():
    """A tall quantified peak leaves a percentage of itself behind; an
    analyte whose signals lie only there has no evidence of its own."""
    owner = _met('owner', 'Owner', [_signal(2.0, 's', 1)])
    # a would-be intruder whose signals sit on the owner's shoulders
    intruder = _met('ghost', 'Ghost', [_signal(2.004, 's', 1), _signal(1.996, 's', 1)])
    cols = build_columns([owner, intruder], field_mhz=FIELD, linewidth_ppm=LW,
                         drift_ppm=0.0)
    X = np.column_stack([_line(2.0, 1000.0)] * 2)
    oc = next(c for c in cols if c.mid == 'owner')
    _idx, val = oc.curve(PPM, 0.0)
    amp = 0.97 * 1000.0 / float(val.max())      # subtract 3% short
    Y, free, owned = subtract_fixed(cols, PPM, X, {'owner': {0: amp, 1: amp}},
                                    sigma=SIGMA)
    w = exclusion_weight(PPM)
    w[owned] = 0.0
    res = cascade(free, PPM, Y[:, 0], sigma=SIGMA, linewidth_ppm=LW, weight=w)
    assert [e.col.mid for e in res.active] == []


def _owner_off_its_listing(off_lw):
    """A quantified singlet listed at 2.0 ppm whose line stands ``off_lw``
    linewidths away, with its true amount handed to the subtraction."""
    owner = _met('owner', 'Owner', [_signal(2.0, 's', 1)])
    cols = build_columns([owner], field_mhz=FIELD, linewidth_ppm=LW,
                         drift_ppm=0.0)
    X = np.column_stack([_line(2.0 + off_lw * LW, 1000.0)] * 2)
    _idx, val = cols[0].curve(PPM, 0.0)
    amp = 1000.0 / float(val.max())
    return cols, X, {'owner': {0: amp, 1: amp}}


def test_a_quantified_line_off_its_listing_is_subtracted_where_it_stands():
    """Subtracted at the listed place, a line half a linewidth off leaves a
    remainder shaped like a derivative, hundreds of sigma tall, for a
    compound listing a line there to take."""
    cols, X, fixed = _owner_off_its_listing(0.5)
    listed, _f, _o = subtract_fixed(cols, PPM, X, fixed, sigma=SIGMA)
    settled, _f, _o = subtract_fixed(cols, PPM, X, fixed, sigma=SIGMA,
                                     linewidth_ppm=LW)
    assert listed[:, 0].max() > 300.0
    assert settled[:, 0].max() < 0.1 * listed[:, 0].max()


def test_a_line_beyond_the_reach_stays_where_it_is_listed():
    """Settling moves a signal within SETTLE_REACH_LW, not onto whatever
    peak stands further off: a neighbour two linewidths away is not
    claimed by the quantified line beside it."""
    cols, X, fixed = _owner_off_its_listing(0.0)
    X = X + np.column_stack([_line(2.0 + 2.0 * LW, 400.0)] * 2)
    settled, _f, _o = subtract_fixed(cols, PPM, X, fixed, sigma=SIGMA,
                                     linewidth_ppm=LW)
    near = np.abs(PPM - (2.0 + 2.0 * LW)) < 0.2 * LW
    assert settled[near, 0].max() > 0.9 * 400.0


# ── end to end, through the engine ──────────────────────────────────────

def _cand(mid, rank, centre, db_peaks, mult='s', protons=3, lines=None):
    return Candidate(
        met_id=mid, peak_rank=rank, line_ranks=frozenset({rank}),
        evidence={'multiplicity': mult, 'ppm_diff': 0.0005, 'protons': protons,
                  'db_ppm': centre, 'study_drift_ppm': 0.0, 'n_found': 1,
                  'n_expected': 1, 'peak_ppm': centre,
                  'predicted_ppms': lines or [centre],
                  'pattern_field_mhz': FIELD},
        payload={'met': {'db_peaks': db_peaks}, 'name': mid})


def _peak(rank, centre, half=0.01):
    return {'rank': rank, 'ppm': centre, 'start_ppm': centre - half,
            'end_ppm': centre + half, 'height': 1.0}


def test_the_cascade_adds_an_analyte_the_election_never_saw():
    """Two analytes in the spectrum; only one has a candidate. The other
    is found on what the first leaves, and gets a row with one dot."""
    seen = _met('seen', 'Seen', [_signal(2.0, 's', 3)])
    missed = _met('missed', 'Missed', [_signal(1.0, 'd', 3, J),
                                       _signal(3.0, 'd', 3, J)])
    X = _study([(2.0, 400.0, ((0.0, 1.0),)),
                (1.0, 120.0, DOUBLET), (3.0, 120.0, DOUBLET)])
    peaks = {1: _peak(1, 2.0)}
    rows = {r['hmdb_id']: r for r in quantify(
        candidates=[_cand('seen', 1, 2.0, seen['db_peaks'])],
        peaks_by_rank=peaks, ppm=PPM, intensities=X, samples_info=[],
        included=[0, 1, 2], groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
        noise_sigma=SIGMA, linewidth_ppm=LW, deconvolve=False,
        cascade=True, metabolites=[seen, missed], field_mhz=FIELD)}
    assert not rows['seen'].get('_by_cascade')
    added = rows['missed']
    assert added['_by_cascade'] and not added['_not_quantified']
    # never three: the cascade never confirmed its lines one by one. Two
    # when its own signals agree on the amount, one when nothing checked it.
    assert added['_confidence'] == 2
    assert added['_identity_level'] == 2
    assert added['_integ_mode'] == 'Cascade'
    assert 'cascade' in ' '.join(added['_status_reasons']).lower()


def test_the_cascade_does_not_move_a_quantified_analyte():
    seen = _met('seen', 'Seen', [_signal(2.0, 's', 3)])
    missed = _met('missed', 'Missed', [_signal(1.0, 'd', 3, J),
                                       _signal(3.0, 'd', 3, J)])
    X = _study([(2.0, 400.0, ((0.0, 1.0),)),
                (1.0, 120.0, DOUBLET), (3.0, 120.0, DOUBLET)])
    peaks = {1: _peak(1, 2.0)}
    kw = dict(candidates=[_cand('seen', 1, 2.0, seen['db_peaks'])],
              peaks_by_rank=peaks, ppm=PPM, intensities=X, samples_info=[],
              included=[0, 1, 2], groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
              noise_sigma=SIGMA, linewidth_ppm=LW, deconvolve=False)
    before = {r['hmdb_id']: r for r in quantify(**kw)}
    after = {r['hmdb_id']: r for r in quantify(
        cascade=True, metabolites=[seen, missed], field_mhz=FIELD, **kw)}
    assert (after['seen']['_areas_per_sample']
            == before['seen']['_areas_per_sample'])
    assert after['seen']['conc_mm'] == before['seen']['conc_mm']


def test_the_amount_couples_every_signal_of_the_analyte():
    """One amplitude for the whole listing: the analyte's area per proton
    is what both its doublets agree on, not what one of them holds."""
    missed = _met('missed', 'Missed', [_signal(1.0, 'd', 2, J),
                                       _signal(3.0, 'd', 4, J)])
    # laid at the same area per proton: the 4H signal is twice the 2H one
    X = _study([(1.0, 100.0, DOUBLET), (3.0, 200.0, DOUBLET)])
    out = cascade_missed([missed], PPM, X, [0, 1, 2], fixed_per_proton={},
                         noise_sigma=SIGMA, linewidth_ppm=LW, drift_ppm=0.0,
                         field_mhz=FIELD)
    assert set(out) == {'missed'}
    assert out['missed']['nh'] == pytest.approx(6.0)
    per = out['missed']['per_proton']
    assert all(v > 0 for v in per.values())
    # the sample with 10% more of everything gets 10% more
    assert per[1] / per[0] == pytest.approx(1.1, rel=0.05)


def test_nothing_is_added_when_the_spectrum_is_already_explained():
    seen = _met('seen', 'Seen', [_signal(2.0, 's', 3)])
    other = _met('other', 'Other', [_signal(1.0, 'd', 3, J)])
    X = _study([(2.0, 400.0, ((0.0, 1.0),))])     # only 'seen' is there
    cols = build_columns([seen], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.0)
    _idx, val = cols[0].curve(PPM, 0.0)
    amp = 400.0 / float(val.max())
    out = cascade_missed([seen, other], PPM, X, [0, 1, 2],
                         fixed_per_proton={'seen': {0: amp, 1: amp, 2: amp}},
                         noise_sigma=SIGMA, linewidth_ppm=LW, drift_ppm=0.0,
                         field_mhz=FIELD)
    assert out == {}


def test_the_diagnostics_say_what_the_cascade_did():
    missed = _met('missed', 'Missed', [_signal(1.0, 'd', 3, J)])
    X = _study([(1.0, 150.0, DOUBLET)])
    diag = {}
    cascade_missed([missed], PPM, X, [0, 1, 2], fixed_per_proton={},
                   noise_sigma=SIGMA, linewidth_ppm=LW, drift_ppm=0.0,
                   field_mhz=FIELD, diagnostics=diag)
    assert diag['cascade']['columns'] == 1
    assert diag['cascade']['order'][0][1] == 'Missed'


def test_an_analyte_mostly_under_other_analytes_says_its_amount_is_an_estimate():
    """Half its signal sits where a quantified analyte accounts for the
    trace. On the bench an amount read from the free minority missed by a
    quarter and moved the ratio between groups as much: the analyte is
    identified by the cascade's score, and the row says its amount is an
    estimate (or, below the score's minimum, it is a candidate: CASCADE_CROWDED)."""
    owner = _met('owner', 'Owner', [_signal(2.0, 's', 2)])
    crowded = _met('crowded', 'Crowded', [_signal(2.0, 's', 2),
                                          _signal(3.0, 's', 2)])
    X = _study([(2.0, 600.0, ((0.0, 1.0),)), (3.0, 60.0, ((0.0, 1.0),))])
    peaks = {1: _peak(1, 2.0)}
    rows = {r['hmdb_id']: r for r in quantify(
        candidates=[_cand('owner', 1, 2.0, owner['db_peaks'], protons=2)],
        peaks_by_rank=peaks, ppm=PPM, intensities=X, samples_info=[],
        included=[0, 1, 2], groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
        noise_sigma=SIGMA, linewidth_ppm=LW, deconvolve=False,
        cascade=True, metabolites=[owner, crowded], field_mhz=FIELD)}
    r = rows['crowded']
    assert r['_by_cascade']
    assert 0.25 <= r['_cascade']['free_share'] < MIN_FREE_TO_QUANTIFY
    if r['_not_quantified']:
        assert r['_rejection'] == CASCADE_CROWDED
    else:
        assert 'estimate' in ' '.join(r['_status_reasons'])


def test_a_free_analyte_is_quantified_and_says_how_free_it_was():
    missed = _met('missed', 'Missed', [_signal(1.0, 'd', 3, J)])
    X = _study([(1.0, 200.0, DOUBLET)])
    out = cascade_missed([missed], PPM, X, [0, 1, 2], fixed_per_proton={},
                         noise_sigma=SIGMA, linewidth_ppm=LW, drift_ppm=0.0,
                         field_mhz=FIELD)
    assert out['missed']['free_share'] == pytest.approx(1.0)


def test_the_amount_is_fitted_beside_the_quantified_analyte_that_overlaps_it():
    """A quantified analyte's tail over the newcomer's peak must not cost
    the newcomer its peak: the two are fitted together, the quantified one
    free within the precision its own measurement has."""
    # the owner is tall and far, but its wing reaches the newcomer's singlet
    owner = _met('owner', 'Owner', [_signal(2.0, 's', 2), _signal(3.02, 's', 2)])
    missed = _met('missed', 'Missed', [_signal(3.0, 's', 3)])
    X = _study([(2.0, 800.0, ((0.0, 1.0),)), (3.02, 40.0, ((0.0, 1.0),)),
                (3.0, 300.0, ((0.0, 1.0),))])
    peaks = {1: _peak(1, 2.0)}
    rows = {r['hmdb_id']: r for r in quantify(
        candidates=[_cand('owner', 1, 2.0, owner['db_peaks'], protons=2)],
        peaks_by_rank=peaks, ppm=PPM, intensities=X, samples_info=[],
        included=[0, 1, 2], groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
        noise_sigma=SIGMA, linewidth_ppm=LW, deconvolve=False,
        cascade=True, metabolites=[owner, missed], field_mhz=FIELD)}
    r = rows['missed']
    assert r['_by_cascade'] and not r['_not_quantified']
    assert r['_cascade']['free_share'] > 0.6


def test_signals_that_disagree_on_the_amount_give_no_number():
    """Two signals of one listing, and the spectrum holds them at amounts
    that cannot both be this analyte: identified, not quantified."""
    missed = _met('missed', 'Missed', [_signal(1.0, 's', 2), _signal(3.0, 's', 2)])
    # the same protons, five times the height at one of them
    X = _study([(1.0, 400.0, ((0.0, 1.0),)), (3.0, 80.0, ((0.0, 1.0),))])
    out = cascade_missed([missed], PPM, X, [0, 1, 2], fixed_per_proton={},
                         noise_sigma=SIGMA, linewidth_ppm=LW, drift_ppm=0.0,
                         field_mhz=FIELD)
    assert out['missed']['disagreement'] > MAX_SIGNAL_DISAGREEMENT


def test_one_signal_alone_has_no_error_bar_and_says_so():
    missed = _met('missed', 'Missed', [_signal(1.0, 's', 3)])
    X = _study([(1.0, 200.0, ((0.0, 1.0),))])
    out = cascade_missed([missed], PPM, X, [0, 1, 2], fixed_per_proton={},
                         noise_sigma=SIGMA, linewidth_ppm=LW, drift_ppm=0.0,
                         field_mhz=FIELD)
    assert np.isnan(out['missed']['disagreement'])


def test_how_far_its_signals_may_disagree_is_the_study_s_call():
    """Measured on the reference study: myo-inositol's four signals agree
    to 26% and it measures to within 7%, while hydroxyproline's six
    disagree by 262%. Where the line falls between them is a judgement
    about the study, not about the engine."""
    import inspect
    from processing.quant_engine import quantify
    from gui.stages._quant_adapter import quantify_project
    for fn in (quantify, quantify_project):
        assert 'max_disagreement' in inspect.signature(fn).parameters


def test_the_default_bar_is_the_measured_one():
    from processing.quant_cascade import MAX_SIGNAL_DISAGREEMENT
    assert MAX_SIGNAL_DISAGREEMENT == 0.45


class TestWhatACascadeRowIsWorth:
    """Two dots when its own signals were measured one by one and agree;
    one when there was no second signal to measure."""

    def test_signals_that_agree_are_worth_two(self):
        from processing.quant_engine import _identity_evidence
        assert _identity_evidence(cascade={'disagreement': 0.17})[0] == 2

    def test_signals_that_do_not_agree_are_worth_one(self):
        from processing.quant_engine import _identity_evidence
        assert _identity_evidence(cascade={'disagreement': 0.45})[0] == 1

    def test_nothing_to_check_against_is_worth_one(self):
        """The three false positives the cascade found on the reference
        study that a flat floor of two would have promoted had exactly
        this in common: no second signal, so nothing was ever checked."""
        from processing.quant_engine import _identity_evidence
        level, why = _identity_evidence(cascade={'disagreement': float('nan')})
        assert level == 1 and 'no second signal' in why

    def test_placed_on_the_study_s_drift_it_is_worth_two(self):
        """What is in the mixture moves with the study (ADR-042): on the
        crowded tubes every stranger the cascade took in sat two or three
        steps of its slide off the drift, the reals at one dot at zero or
        one."""
        from processing.quant_engine import _identity_evidence
        alone = _identity_evidence(cascade={'disagreement': float('nan'),
                                            'placed': True})
        assert alone[0] == 2 and 'drift' in alone[1]
        assert _identity_evidence(cascade={'disagreement': 0.28,
                                           'placed': True})[0] == 2
        # off the drift, or placed but disagreeing, nothing changes
        assert _identity_evidence(cascade={'disagreement': float('nan'),
                                           'placed': False})[0] == 1
        assert _identity_evidence(cascade={'disagreement': 0.40,
                                           'placed': False})[0] == 1

    def test_one_step_of_the_slide_is_the_grid_s_own(self):
        import numpy as np
        from processing.quant_cascade import slide_grid, slide_step
        from processing.quant_cascade import SHIFT_REACH_LW
        g = slide_grid(0.006)
        fine = g[np.abs(g) <= SHIFT_REACH_LW * 0.006 * 1.0001]
        assert slide_step(0.006) == pytest.approx(float(np.diff(fine)[0]))

    def test_the_bar_is_not_the_one_that_decides_the_number(self):
        """One threshold answers one question: whether there is a number
        at all is MAX_SIGNAL_DISAGREEMENT's."""
        from processing.quant_cascade import MAX_SIGNAL_DISAGREEMENT
        from processing.quant_engine import CASCADE_AGREEMENT_GOOD
        assert CASCADE_AGREEMENT_GOOD < MAX_SIGNAL_DISAGREEMENT


def test_the_position_band_widens_only_where_the_study_is_loose():
    from processing.quant_engine import study_position_band
    from processing.annotation_scoring import PPM_BAND_PPM

    def mets(offs):
        return [{'matched_peaks': [{'annotation_quality': 'good',
                                    'ppm_off_drift': o}]} for o in offs]
    tight = mets([0.001] * 30)
    loose = mets([0.010] * 10 + [0.035] * 20)
    wild = mets([0.2] * 30)
    assert study_position_band(tight) == PPM_BAND_PPM
    assert study_position_band(mets([0.04] * 5)) == PPM_BAND_PPM   # too few
    assert PPM_BAND_PPM < study_position_band(loose) <= 0.05
    assert study_position_band(wild) == 0.05                        # capped


def test_a_refused_joint_fit_row_is_a_candidate_with_an_upper_bound():
    from processing.quant_engine import _as_candidate
    row = {'conc_mm': 0.5, '_confidence': 1, 'status': 'x',
           '_status_reasons': ['earlier'], '_not_quantified': False}
    _as_candidate(row, '2 of 4 signals at the edge', {'A': [0.4, 0.6], 'B': []})
    assert row['_not_quantified'] and row['_confidence'] == 0
    assert row['_upper_bound_mm'] == {'A': 0.5}
    assert 'Upper bound' in row['_status_reasons'][0]
    assert row['_status_reasons'][1] == 'earlier'


def _ann_met(name, mps):
    return {'id': name, 'name': name, 'matched_peaks': mps}


def _confirmed(db, peak):
    return {'peak_rank': 1, 'annotation_quality': 'good', 'type': 'd',
            'db_ppm': db, 'peak_ppm': peak, 'delta_j_hz': 0.02,
            '_n_found_lines': 2, '_n_exp_lines': 2}


def test_the_reference_drift_is_the_panel_s_not_the_background_s():
    from processing.annotation_scoring import reference_drift, study_drift
    panel = [_ann_met(n, [_confirmed(1.0, 1.013)]) for n in
             ('L-lactic acid', 'L-alanine', 'L-valine', 'L-isoleucine')]
    strangers = [_ann_met(f'stranger{i}', [_confirmed(2.0, 2.0 + 0.001 * i)])
                 for i in range(12)]
    mets = panel + strangers
    assert reference_drift(mets) == pytest.approx(0.013)
    assert study_drift(mets) == pytest.approx(0.013)       # not the median
    assert reference_drift(strangers) is None               # no panel, no say
    assert reference_drift(panel[:2]) is None               # too few


def test_a_singlet_goes_to_the_peak_standing_where_the_references_stand():
    from gui.stages._quant_adapter import reselect_singlets
    panel = [_ann_met(n, [_confirmed(1.0, 1.014)]) for n in
             ('L-lactic acid', 'L-alanine', 'L-valine')]
    succ = _ann_met('Succinic acid', [{'peak_rank': 20, 'db_ppm': 2.3913,
                                   'peak_ppm': 2.3926, 'type': 's',
                                   'annotation_quality': None,
                                   'ppm_diff': 0.0013}])
    peaks = {20: {'rank': 20, 'ppm': 2.3926, 'snr': 69},
             13: {'rank': 13, 'ppm': 2.4073, 'snr': 160}}
    out = reselect_singlets(panel + [succ], peaks)
    mp = next(m for m in out if m['name'] == 'Succinic acid')['matched_peaks'][0]
    assert mp['peak_rank'] == 13 and mp['reselected']
    assert succ['matched_peaks'][0]['peak_rank'] == 20       # not touched


def test_a_singlet_stays_where_the_search_put_it_without_a_reference_drift():
    from gui.stages._quant_adapter import reselect_singlets
    succ = _ann_met('Succinic acid', [{'peak_rank': 20, 'db_ppm': 2.3913,
                                   'peak_ppm': 2.3926, 'type': 's'}])
    peaks = {20: {'rank': 20, 'ppm': 2.3926}, 13: {'rank': 13, 'ppm': 2.4073}}
    assert reselect_singlets([succ], peaks)[0]['matched_peaks'][0]['peak_rank'] == 20


# ── do an analyte's signals move together across the samples? ─────────────

def test_signals_of_one_molecule_move_together_across_samples():
    from processing.quant_cascade import covariation_confirms, signal_covariation
    rng = np.random.default_rng(0)
    conc = np.exp(rng.normal(0, 0.4, 12))                 # one concentration per sample
    a = 100 * conc + rng.normal(0, 3, 12)
    b = 40 * conc + rng.normal(0, 3, 12)
    cov = signal_covariation([a, b], sigma=3.0)
    assert cov['r_min'] > 0.8 and cov['n_signals'] == 2 and cov['n_samples'] == 12
    assert covariation_confirms(cov)


def test_signals_of_different_molecules_do_not():
    from processing.quant_cascade import covariation_confirms, signal_covariation
    rng = np.random.default_rng(1)
    a = 100 * np.exp(rng.normal(0, 0.4, 12))
    b = 40 * np.exp(rng.normal(0, 0.4, 12))
    cov = signal_covariation([a, b], sigma=3.0)
    assert not covariation_confirms(cov)


def test_a_signal_that_does_not_vary_has_nothing_to_correlate():
    from processing.quant_cascade import covariation_confirms, signal_covariation
    flat = np.full(12, 50.0)
    cov = signal_covariation([flat, np.arange(12.0)], sigma=1.0)
    assert cov['variation'] == 0.0 and not covariation_confirms(cov)


def test_the_variation_must_stand_over_the_noise():
    from processing.quant_cascade import covariation_confirms, signal_covariation
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 12)
    cov = signal_covariation([x, x + rng.normal(0, 0.05, 12)], sigma=10.0)
    assert cov['r_min'] > 0.9 and cov['variation'] < 1 and not covariation_confirms(cov)


def test_one_signal_or_too_few_samples_is_not_read():
    from processing.quant_cascade import signal_covariation
    assert signal_covariation([np.arange(12.0)], sigma=1.0) is None
    assert signal_covariation([np.arange(5.0), np.arange(5.0)], sigma=1.0) is None
    assert signal_covariation([np.arange(12.0), np.arange(12.0)], sigma=0) is None


def test_the_measure_is_kept_on_the_row_but_does_not_raise_it_unless_asked(monkeypatch):
    import processing.quant_engine as qe
    from processing.quant_engine import _identity_evidence
    assert qe.CASCADE_COVARIATION_RAISES is False
    res = {'score': 0.3, 'covar': {'r_min': 0.95, 'r_median': 0.95, 'variation': 9.0,
                                   'n_signals': 2, 'n_samples': 12}}
    assert _identity_evidence(cascade=res)[0] == 1
    monkeypatch.setattr(qe, 'CASCADE_COVARIATION_RAISES', True)
    level, why = _identity_evidence(cascade=res)
    assert level == 2 and 'move together' in why
    # never past the cascade's own ceiling, and never down
    assert _identity_evidence(cascade={'score': 0.65, 'covar': res['covar']})[0] == 2
    assert _identity_evidence(cascade={'score': 0.3, 'covar': {
        'r_min': -0.5, 'r_median': -0.5, 'variation': 9.0,
        'n_signals': 2, 'n_samples': 12}})[0] == 1


def test_a_cascade_row_carries_its_covariation():
    missed = _met('missed', 'Missed', [_signal(1.0, 's', 2), _signal(3.0, 's', 2)])
    rng = np.random.default_rng(3)
    cols = []
    for k in range(12):
        f = float(np.exp(rng.normal(0, 0.4)))
        y = (_line(1.0, 150.0 * f) + _line(3.0, 150.0 * f)
             + rng.normal(0, SIGMA, PPM.size))
        cols.append(y)
    X = np.column_stack(cols)
    rows = {r['hmdb_id']: r for r in quantify(
        candidates=[], peaks_by_rank={}, ppm=PPM, intensities=X, samples_info=[],
        included=list(range(12)), groups={'G': list(range(12))}, dj_gold_hz=0.2,
        noise_sigma=SIGMA, linewidth_ppm=LW, deconvolve=False,
        cascade=True, metabolites=[missed], field_mhz=FIELD)}
    cov = rows['missed']['_cascade']['covar']
    assert cov and cov['r_min'] > 0.8 and cov['n_samples'] == 12


# ── each signal of a column finds its own place (ADR-065) ─────────────────

def _two_singlets():
    met = _met('m1', 'Analyte', [_signal(1.0, 's', 1), _signal(3.0, 's', 1)])
    col, = build_columns([met], field_mhz=FIELD, linewidth_ppm=LW, drift_ppm=0.0)
    return col


def test_a_signal_off_its_listing_moves_to_where_it_stands():
    from processing.quant_cascade import part_offsets
    col = _two_singlets()
    off = 0.75 * LW
    trace = _line(1.0, 200.0) + _line(3.0 + off, 200.0)
    d = part_offsets(col, PPM, trace, 0.0, LW)
    assert d[0] == 0.0
    assert d[1] == pytest.approx(off, abs=0.15 * LW)


def test_a_signal_where_it_is_listed_stays_put():
    from processing.quant_cascade import part_offsets
    col = _two_singlets()
    assert part_offsets(col, PPM, _line(1.0, 200.0) + _line(3.0, 200.0), 0.0, LW) == [0.0, 0.0]


def test_no_signal_moves_further_than_its_reach():
    from processing.quant_cascade import CASCADE_PART_REACH_LW, part_offsets
    col = _two_singlets()
    trace = _line(1.0, 200.0) + _line(3.0 + 3.0 * LW, 200.0)
    d = part_offsets(col, PPM, trace, 0.0, LW)
    assert all(abs(x) <= CASCADE_PART_REACH_LW * LW + 1e-12 for x in d)


def test_the_offsets_ride_on_the_column_s_own_slide():
    from processing.quant_cascade import part_offsets
    col = _two_singlets()
    off = 0.5 * LW
    trace = _line(1.0 + 2 * LW, 200.0) + _line(3.0 + 2 * LW + off, 200.0)
    d = part_offsets(col, PPM, trace, 2 * LW, LW)
    assert d[0] == 0.0 and d[1] == pytest.approx(off, abs=0.15 * LW)


def test_an_offset_column_keeps_its_identity_and_leaves_the_listing_alone():
    from processing.quant_cascade import with_offsets
    col = _two_singlets()
    moved = with_offsets(col, [0.0, 0.002])
    assert moved.mid == col.mid and moved.nh_total == col.nh_total
    assert moved.parts[1][0].shift_ppm == pytest.approx(col.parts[1][0].shift_ppm + 0.002)
    assert col.parts[1][0].shift_ppm == pytest.approx(0.0)
    assert with_offsets(col, [0.0, 0.0]) is col


def test_the_cascade_measures_an_analyte_whose_signals_moved_apart():
    """A doublet listed at 3.0 that stands 0.75 linewidths off in the
    extract: measured with its signals where they stand, its amount is
    right and the row says how far each moved."""
    missed = _met('missed', 'Missed', [_signal(1.0, 'd', 3, J), _signal(3.0, 'd', 3, J)])
    off = 0.75 * LW
    X = _study([(1.0, 120.0, DOUBLET), (3.0 + off, 120.0, DOUBLET)])
    rows = {r['hmdb_id']: r for r in quantify(
        candidates=[], peaks_by_rank={}, ppm=PPM, intensities=X, samples_info=[],
        included=[0, 1, 2], groups={'G': [0, 1, 2]}, dj_gold_hz=0.2,
        noise_sigma=SIGMA, linewidth_ppm=LW, deconvolve=False,
        cascade=True, metabolites=[missed], field_mhz=FIELD)}
    c = rows['missed']['_cascade']
    placed = [c['shift_ppm'] + x for x in c['part_offsets']]
    assert placed[0] == pytest.approx(0.0, abs=0.2 * LW)
    assert placed[1] == pytest.approx(off, abs=0.2 * LW)
    assert c['disagreement'] < 0.1
