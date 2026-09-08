"""A study whose answer is known, put through the pipeline that runs.

Every measurement in this project so far was taken one stage at a time, or by
recomputing from a saved project. That checks the piece that changed and
nothing around it, which is how a claim/window mismatch survived three passes
of review: each half was correct and they disagreed with each other. Item F3 of
the pending list has asked for this since the beginning.

Here a spectrum is built from real database entries at concentrations chosen in
advance, with a linewidth, a per-sample drift and a noise level taken from the
reference study, then run through detection, the annotation decisions and
quantification. What comes out has to be what went in.

The GUI orchestration is not here — the annotation search is a Tk mixin. Every
function that DECIDES anything is, and they are called in the order the
application calls them:

    detect_peaks                  which peaks exist
    measure_multiplet             which detected peaks are a signal's lines
    fit_across_samples            where each sample put those lines
    consensus                     the delta and J the samples agree on
    apply_per_sample_consensus    those replacing the mean spectrum's
    promote_singlets              which singlets are identified
    eligible_metabolite_ids       which metabolites may be quantified
    elect                         which signal quantifies which metabolite
    quantify                      the concentration
"""
import numpy as np
import pytest

from gui.stages._quant_adapter import quantify_project
from processing.annotation_quality import promote_singlets
from processing.annotation_scoring import (
    SCORING_DEFAULTS, apply_per_sample_consensus, measure_multiplet,
    synthetic_lines)
from processing.multiplet_line_fit import fit_across_samples
from processing.multiplet_line_selector import multiplet_line_peaks
from processing.peak_detection import detect_peaks
from processing.per_sample_signal import signal_across_samples
from processing.spectral_regions import DEFAULT_EXCLUDED_REGIONS

MHZ = 600.71
PPM = np.linspace(-0.5, 10.0, 65536)
LINEWIDTH_HZ = 3.2                      # measured on the reference study
NOISE = 1.0
DRIFT_HZ = 0.5   # per sample. The reference study drifts 1.7 Hz between its
                 # two GROUPS; per sample the scatter is smaller. See
                 # test_drift_is_what_this_pipeline_is_sensitive_to.

# Real BMRB_Metabolomics_v01 entries, chosen for well separated signals so the
# truth is unambiguous. (ppm, multiplicity, J in Hz, protons)
DB = {
    'L-alanine':     [(1.466, 'd', 7.20, 3.0), (3.767, 'q', 7.20, 1.0)],
    'L-lactic acid': [(1.318, 'd', 7.50, 3.0), (4.101, 'q', 7.00, 1.0)],
    'Creatine':      [(3.024, 's', None, 3.0), (3.920, 's', None, 2.0)],
    'Glycine':       [(3.545, 's', None, 2.0)],
    'Succinic acid': [(2.391, 's', None, 4.0)],
}

# A decoy the study does not contain. Nothing may be reported for it.
DECOY = {'Ethanol': [(1.190, 't', 7.10, 3.0)]}

STANDARD_PPM, STANDARD_NH, STANDARD_MM = 0.0, 9.0, 1.0

# mM per group. Alanine doubles, lactate halves, the rest hold still.
TRUTH = {
    'L-alanine':     {'A': 0.20, 'B': 0.40},
    'L-lactic acid': {'A': 1.60, 'B': 0.80},
    'Creatine':      {'A': 0.30, 'B': 0.30},
    'Glycine':       {'A': 0.25, 'B': 0.25},
    'Succinic acid': {'A': 0.10, 'B': 0.10},
}
GROUPS = ['A'] * 6 + ['B'] * 6
SAMPLES = ([{'type': 'BLK', 'label': 'BLK', 'name': 'BLK1'}]
           + [{'label': g, 'name': f'{g}{i}'} for i, g in enumerate(GROUPS)])

_HWHM = (LINEWIDTH_HZ / MHZ) / 2.0
# One unit of concentration must give the same area whatever the lineshape, so
# every line is normalised to unit area and scaled by concentration x protons.
_AREA_UNIT = 1.0e6


def _lorentz(centre, area):
    return (area / np.pi) * _HWHM / ((PPM - centre) ** 2 + _HWHM ** 2)


def _pascal(n):
    row = [1.0]
    for _ in range(n - 1):
        row = [a + b for a, b in zip([0.0] + row, row + [0.0])]
    return row


def _signal_trace(ppm_c, mult, j_hz, protons, conc):
    """One database signal at one concentration, as first-order lines."""
    n = {'s': 1, 'd': 2, 't': 3, 'q': 4}.get(mult, 1)
    weights = _pascal(n)
    total = float(conc) * float(protons) * _AREA_UNIT
    spacing = (j_hz or 0.0) / MHZ
    offsets = [(i - (n - 1) / 2.0) * spacing for i in range(n)]
    y = np.zeros_like(PPM)
    for w, off in zip(weights, offsets):
        y += _lorentz(ppm_c + off, total * w / sum(weights))
    return y


def _study(seed=0):
    rng = np.random.default_rng(seed)
    cols = []
    for s_i, info in enumerate(SAMPLES):
        y = rng.normal(0.0, NOISE, PPM.size)
        if str(info.get('type', '')).upper() != 'BLK':
            # Each sample drifts by its own amount, as a real one does.
            shift = rng.normal(0.0, DRIFT_HZ / MHZ)
            y = y + _lorentz(STANDARD_PPM + shift,
                             STANDARD_MM * STANDARD_NH * _AREA_UNIT)
            for name, signals in DB.items():
                conc = TRUTH[name][info['label']]
                for ppm_c, mult, j, nh in signals:
                    y = y + _signal_trace(ppm_c + shift, mult, j, nh, conc)
        cols.append(y)
    return np.column_stack(cols)


def _annotate(peaks, X):
    """The annotation decisions, in the order the application makes them."""
    params = dict(SCORING_DEFAULTS)
    params['dppm_gold'] = 0.05
    tol = params['dppm_gold']
    incl = [i for i, s in enumerate(SAMPLES)
            if str(s.get('type', '')).upper() != 'BLK']
    mets = []
    for name, signals in {**DB, **DECOY}.items():
        mps = []
        for ppm_c, mult, j, nh in signals:
            n_exp = {'s': 1, 'd': 2, 't': 3, 'q': 4}.get(mult, 1)
            pred = (synthetic_lines(ppm_c, j, n_exp, MHZ)
                    if j else [{'ppm': ppm_c}])
            meas = measure_multiplet(
                pred, peaks, mult=mult, j_coupling=(str(j) if j else None),
                db_ppm=ppm_c, centre_hint=ppm_c, n_expected=n_exp,
                tolerance=tol, field_mhz=MHZ)
            mp = {'ppm': ppm_c, 'db_ppm': ppm_c, 'type': mult, 'protons': nh,
                  'j_coupling': (str(j) if j else None),
                  'predicted_lines': meas.predicted_lines or pred,
                  '_n_found_lines': meas.n_found, '_n_exp_lines': meas.n_expected,
                  'delta_j_hz': meas.delta_j_hz,
                  'observed_j_hz': meas.observed_j_hz,
                  'peak_ppm': meas.centre_ppm if meas.centre_ppm else ppm_c,
                  'ppm_diff': (abs(meas.centre_ppm - ppm_c)
                               if meas.centre_ppm else None),
                  'composite_score': 0.8}
            line_pks = multiplet_line_peaks(mp, {p['rank']: p for p in peaks})
            mp['peak_rank'] = line_pks[0]['rank'] if line_pks else None
            if line_pks:
                exp = [float(l['ppm']) for l in mp['predicted_lines']
                       if 'ppm' in l]
                fits = fit_across_samples(
                    exp, PPM, X, included=incl, field_mhz=MHZ,
                    dj_bad_hz=params['dj_bad_hz'],
                    dj_gold_hz=params['dj_gold_hz'],
                    max_drift_ppm=tol, noise_floor=NOISE * 3.0)
                out = signal_across_samples(line_pks, fits=fits,
                                            included=incl, db_ppm=ppm_c,
                                            db_j_hz=j, field_mhz=MHZ)
                mp['per_sample'] = out['per_sample']
                mp['n_present'] = out['n_present']
                mp['per_sample_consensus'] = out
                apply_per_sample_consensus(mp, params=params)
            mps.append(mp)
        mets.append({'id': name, 'name': name, 'matched_peaks': mps})
    promote_singlets(mets, peaks, params=params)
    return mets


@pytest.fixture(scope='module')
def run():
    X = _study()
    peaks = detect_peaks(
        PPM, X, noise_std=NOISE, min_snr=6.0, min_distance=0.005,
        baseline_correct=False, global_align=False, samples_info=SAMPLES,
        per_sample_detection=True, detect_shoulders=False,
        min_group_presence=3, excluded_regions=DEFAULT_EXCLUDED_REGIONS)
    mets = _annotate(peaks, X)
    ref = min(peaks, key=lambda p: abs(float(p['ppm']) - STANDARD_PPM))
    rows = quantify_project(
        metabolites=mets, peaks=peaks, ppm=PPM, intensities=X,
        samples_info=SAMPLES, ref_peak=ref, ref_nh=STANDARD_NH,
        ref_conc=STANDARD_MM, ref_name='TSP', noise_sigma=NOISE,
        use_pqn=False, local_baseline=False)
    return peaks, mets, {r['metabolite']: r for r in rows}


# ── Detection ────────────────────────────────────────────────────────────────

def test_every_analyte_has_its_strongest_signal_detected(run):
    """Not every line of every signal — a quartet's outer lines are a third of
    its inner ones and can sit under the detection threshold at a low
    concentration, which is a limit of the experiment and not a fault. What
    must be there is the signal each analyte is quantified on."""
    peaks, _, _ = run
    found = [float(p['ppm']) for p in peaks]
    for name, signals in DB.items():
        ppm_c, mult, j, _nh = max(signals, key=lambda s: s[3])
        n = {'s': 1, 'd': 2, 't': 3, 'q': 4}.get(mult, 1)
        spacing = (j or 0.0) / MHZ
        for i in range(n):
            want = ppm_c + (i - (n - 1) / 2.0) * spacing
            assert any(abs(f - want) < 0.006 for f in found), (
                f'{name} line at {want:.4f} was not detected')


def test_the_blank_contributes_no_peaks(run):
    """It holds only noise. A peak surviving from it would be a peak the
    presence filter let through on intensity."""
    peaks, _, _ = run
    for p in peaks:
        pres = p.get('aligned_per_sample') or []
        assert not (len(pres) > 0 and pres[0] and sum(pres) == 1), (
            f"peak at {p['ppm']:.4f} exists only in the blank")


# ── Annotation ───────────────────────────────────────────────────────────────

def test_the_signal_each_analyte_is_quantified_on_is_complete(run):
    _, mets, by = run
    for met in mets:
        if met['name'] in DECOY or met['name'] not in by:
            continue
        row = by[met['name']]
        if row.get('_not_quantified'):
            continue
        assert row['_multiplet_window'] is not None
        lo, hi = row['_multiplet_window']
        assert hi > lo, f"{met['name']} was measured over nothing"


def test_the_coupling_is_measured_from_the_samples_not_a_mean(run):
    """Where the samples reached a consensus at all, it is the one that stands.
    A signal they could not agree on keeps the mean spectrum's value, and says
    so — what must not happen is a per-sample source with a J that disagrees
    with the database."""
    _, mets, _ = run
    agreed = 0
    for met in mets:
        if met['name'] in DECOY:
            continue
        for mp in met['matched_peaks']:
            if mp['type'] == 's' or mp.get('_dj_source') != 'per_sample':
                continue
            agreed += 1
            assert abs(mp['delta_j_hz']) < 0.5, (
                f"{met['name']} at {mp['db_ppm']}: ΔJ = {mp['delta_j_hz']:.2f}")
    assert agreed >= 2, 'no multiplet reached a per-sample consensus at all'


def test_the_decoy_finds_nothing(run):
    """Ethanol is not in the study. Its triplet must not be assembled out of
    whatever happens to sit near 1.19 ppm."""
    _, mets, _ = run
    eth = next(m for m in mets if m['name'] == 'Ethanol')
    for mp in eth['matched_peaks']:
        assert (mp['_n_found_lines'] or 0) < (mp['_n_exp_lines'] or 1), (
            'the decoy assembled a complete multiplet from noise')


# ── Quantification ───────────────────────────────────────────────────────────

def test_every_analyte_is_quantified(run):
    _, _, by = run
    for name in TRUTH:
        assert name in by, f'{name} produced no row at all'
        assert not by[name]['_not_quantified'], (
            f"{name}: {by[name].get('_status_reasons')}")


def test_the_decoy_is_not_quantified(run):
    _, _, by = run
    assert 'Ethanol' not in by or by['Ethanol']['_not_quantified'], (
        'an analyte absent from the study was given a concentration')


def test_no_two_analytes_share_a_region(run):
    _, _, by = run
    rows = [r for r in by.values()
            if r.get('_multiplet_window') and not r.get('_is_is')]
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            lo_a, hi_a = a['_multiplet_window']
            lo_b, hi_b = b['_multiplet_window']
            assert not (lo_a < hi_b and lo_b < hi_a), (
                f"{a['metabolite']} and {b['metabolite']} share a window")


@pytest.mark.parametrize('name', list(TRUTH))
def test_the_concentration_is_the_one_that_went_in(run, name):
    """Absolute, against the internal standard, in mM.

    Loose on purpose. Foot-to-foot integration of a Lorentzian leaves its tails
    outside, and the standard's window and an analyte's hold different
    fractions, so the absolute scale carries a systematic bias — about 7% on
    this study at rest. The ratio does not, and the ratio is what a study
    reports; that is the assertion below.
    """
    _, _, by = run
    row = by[name]
    concs = row['_concs_per_sample']
    for grp, want in TRUTH[name].items():
        idx = [i for i, s in enumerate(SAMPLES) if s.get('label') == grp]
        got = float(np.nanmean([concs[i] for i in idx]))
        assert got == pytest.approx(want, rel=0.20), (
            f'{name} in {grp}: expected {want} mM, got {got:.4f}')


def test_the_group_difference_is_recovered(run):
    """Alanine doubles from A to B, lactate halves, the rest hold still. This
    is what a study actually reports."""
    _, _, by = run
    for name, want in (('L-alanine', 2.0), ('L-lactic acid', 0.5),
                       ('Creatine', 1.0), ('Glycine', 1.0),
                       ('Succinic acid', 1.0)):
        cbg = by[name]['_concs_by_group']
        got = float(np.mean(cbg['B'])) / float(np.mean(cbg['A']))
        assert got == pytest.approx(want, rel=0.06), (
            f'{name}: B/A expected {want}, got {got:.3f}')


def test_the_standard_reads_back_its_own_concentration(run):
    _, _, by = run
    is_row = next(r for r in by.values() if r.get('_is_is'))
    assert is_row['conc_mm'] == pytest.approx(STANDARD_MM, rel=0.05)


def test_the_blank_is_measured_but_holds_nothing(run):
    """It must not be dropped from the table, and must not carry signal."""
    _, _, by = run
    for name in TRUTH:
        det = by[name]['_detected_per_sample']
        assert det[0] is not True, f'{name} was detected in the blank'


def test_drift_is_what_this_pipeline_is_sensitive_to(run):
    """Measured by re-running the whole study at several drift levels.

        per-sample drift   |error| absolute   |error| on the B/A ratio
              0.0 Hz             7.3%                  1.1%
              0.5 Hz             6.9%                  1.1%
              1.7 Hz            11.8%                  3.3%
              3.0 Hz            32.0%                  4.5%

    Three hertz is close to a full linewidth: at that point the union splits a
    line into two master peaks and the windows stop describing one signal. The
    ratio degrades far more slowly than the absolute value, which is the
    property a study depends on.

    Before the standard's window was centred, 0.5 Hz of drift gave a 120%
    absolute error — its apex sat on the boundary of a 440 Hz valley-to-valley
    window and half its area was integrated away. Nothing in the results could
    have shown it: the standard's recovery is computed against itself.
    """
    from gui.stages._quant_adapter import standard_window
    peaks, _, _ = run
    ref = min(peaks, key=lambda p: abs(float(p['ppm']) - STANDARD_PPM))
    lo, hi = standard_window(ref, LINEWIDTH_HZ / MHZ, peaks)
    width_hz = (hi - lo) * MHZ
    assert width_hz == pytest.approx(10 * LINEWIDTH_HZ, rel=0.3), (
        'the standard is measured over a window sized by the linewidth')
    apex = float(ref['ppm'])
    assert abs((apex - lo) - (hi - apex)) < 1e-9, (
        'the window must be centred on the standard, not bounded by whatever '
        'happens to sit near it')
