# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The whole of quantification, on a spectrum whose answer is known.

Every change to this pipeline so far was measured by recomputing from a saved
project. That checks the piece that changed and nothing around it, which is how
a claim/window mismatch survived three passes of review: each half was correct
and they disagreed with each other.

Here a spectrum is built from known concentrations, annotated the way the
search would leave it, and quantified through the adapter — the same call the
stage makes. What comes back has to be the concentrations that went in.
"""
import numpy as np
import pytest

from gui.stages._quant_adapter import quantify_project

MHZ = 600.0
PPM = np.linspace(-0.5, 10.0, 21001)   # the standard at 0.0 needs both feet


def _lorentz(centre, height, hwhm=0.002):
    return height * hwhm ** 2 / ((PPM - centre) ** 2 + hwhm ** 2)


def _doublet(centre, j_hz, height):
    d = j_hz / MHZ / 2.0
    return _lorentz(centre - d, height) + _lorentz(centre + d, height)


def _peak(rank, centre, *, width=0.004, noise=1.0):
    """A detected peak as the Signal List holds it."""
    i = int(np.argmin(np.abs(PPM - centre)))
    return {'rank': rank, 'ppm': float(PPM[i]),
            'start_ppm': centre - width, 'end_ppm': centre + width,
            'start_idx': i - 20, 'end_idx': i + 20,
            'width_ppm': width, 'noise': noise, 'snr': 100.0}


def _signal(ppm_db, mult, protons, j_hz, rank, line_ranks, *, dj=0.05):
    """A matched_peak as the annotation search leaves it."""
    return {'ppm': ppm_db, 'type': mult, 'protons': protons,
            'j_hz': j_hz, 'delta_j_hz': dj, 'ppm_diff': 0.001,
            'peak_rank': rank, 'annotation_quality': 'good',
            '_n_found_lines': len(line_ranks),
            '_n_exp_lines': len(line_ranks),
            'composite_score': 0.9,
            'predicted_lines': [{'ppm': p} for p in line_ranks.values()],
            'matched_line_ranks': list(line_ranks)}


def _run(study, **kw):
    args = dict(metabolites=study['metabolites'], peaks=study['peaks'],
                ppm=PPM, intensities=study['intensities'],
                samples_info=study['samples_info'],
                ref_peak=study['ref_peak'], ref_nh=9.0, ref_conc=1.0,
                ref_name='TSP', dj_gold_hz=0.5, use_pqn=False)
    args.update(kw)
    return quantify_project(**args)


def _by_name(rows):
    return {r['hmdb_id']: r for r in rows}


# ── The pipeline produces the concentrations that went in ────────────────────

def test_both_analytes_are_quantified():
    rows = _run(_study())
    got = _by_name(rows)
    for mid in ('HMDB0000161', 'HMDB0000190'):
        assert not got[mid]['_not_quantified'], got[mid]['_status_reasons']


def test_the_ratio_between_samples_is_the_ratio_that_was_built_in():
    """Absolute scale depends on lineshape; the ratio does not, and it is what
    a study is actually reporting."""
    rows = _run(_study())
    conc = _by_name(rows)['HMDB0000161']['_concs_per_sample']
    assert conc[1] / conc[0] == pytest.approx(2.0, rel=0.02)
    assert conc[2] / conc[0] == pytest.approx(3.0, rel=0.02)


def _analyte_ratio(**kw):
    got = _by_name(_run(_study(), **kw))
    return (got['HMDB0000190']['_concs_per_sample'][0]
            / got['HMDB0000161']['_concs_per_sample'][0])


def test_the_ratio_between_two_analytes_survives_their_different_nh():
    """Both are 3-proton methyls, so the concentration ratio has to be the
    height ratio — 60/100. A nH dropped, floored or truncated shows up here."""
    assert _analyte_ratio(local_baseline=False) == pytest.approx(0.6, rel=0.02)


def test_the_local_baseline_no_longer_biases_that_ratio():
    """It used to, and the bias is gone because the trapezium is now modelled.

    A straight line between a lorentzian's two feet sits on the peak's own
    tails, so subtracting it removes signal, and removes proportionally more
    from the line whose window is narrower in half-widths. This read 0.564
    against a true 0.600, and was written down as the cost of the option.

    The shared-window balance had to model that trapezium to attribute a
    narrow window's area between its claimants, and the same expression
    corrects the area of a window with only one. Two details decide whether it
    is worth 0.06% or 7%: the apex refined off the grid, and the window's
    bounds taken where the integration actually put them rather than where
    they were asked for.
    """
    assert _analyte_ratio(local_baseline=True) == pytest.approx(0.600,
                                                                abs=0.01)


def test_the_two_baseline_settings_now_agree():
    """They are answering the same question about the same spectrum. That they
    used to differ by 5% was the defect, not the measurement."""
    assert _analyte_ratio(local_baseline=True) == pytest.approx(
        _analyte_ratio(local_baseline=False), rel=0.01)


def test_no_two_analytes_are_measured_on_the_same_region():
    """The defect this rewrite exists for: 16 of 111 quantified metabolites
    were integrating another's lines."""
    rows = [r for r in _run(_study())
            if r.get('_multiplet_window') and not r.get('_is_is')]
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            lo_a, hi_a = a['_multiplet_window']
            lo_b, hi_b = b['_multiplet_window']
            assert not (lo_a < hi_b and lo_b < hi_a), (
                f"{a['metabolite']} and {b['metabolite']} share a window")


def test_the_standard_reads_back_its_own_concentration():
    """Recovery. It was added at 1 mM and must come back at 1 mM, or every
    number it divided is suspect."""
    row = next(r for r in _run(_study()) if r.get('_is_is'))
    assert row['conc_mm'] == pytest.approx(1.0, rel=0.01)


def test_every_quantified_row_carries_its_snr():
    """CLAUDE.md: never omit SNR from a quantification output. A window over
    noise integrates to a plausible number and the SNR is what exposes it."""
    for r in _run(_study()):
        if not r.get('_not_quantified'):
            assert np.isfinite(r['snr']) and r['snr'] > 0


def test_every_quantified_row_says_which_region_produced_it():
    for r in _run(_study()):
        if not r.get('_not_quantified') and not r.get('_is_is'):
            lo, hi = r['_multiplet_window']
            assert hi > lo


def test_the_row_says_where_its_local_baseline_was_read_and_keeps_it():
    """The plot and the PDF draw the line the area was measured against, so
    the anchors travel on the row -- outside the window, and through a
    project save and load unchanged."""
    import pickle
    rows = [r for r in _run(_study(), local_baseline=True)
            if not r.get('_not_quantified') and not r.get('_is_is')]
    assert rows
    for r in rows:
        a, b, half = r['_baseline_anchors']
        lo, hi = r['_multiplet_window']
        assert a + half <= lo + 1e-9 and b - half >= hi - 1e-9
    back = pickle.loads(pickle.dumps(rows, protocol=5))
    assert [r['_baseline_anchors'] for r in back] == \
        [r['_baseline_anchors'] for r in rows]
    assert all(r.get('_baseline_anchors') is None
               for r in _run(_study(), local_baseline=False))


# ── The options the panel offers actually change the answer ──────────────────

def test_group_dilution_is_applied_when_the_standard_was_injected_separately():
    """Nothing cancels then, so the full dilution has to be applied. Getting it
    backwards scales every concentration in the study, uniformly and
    silently."""
    s = _study()
    plain = _by_name(_run(s))['HMDB0000161']['_concs_per_sample'][0]
    diluted = _by_name(_run(
        s, external_is_samples=[0, 1, 2],
        group_dilution={'A': 2.0}))['HMDB0000161']['_concs_per_sample'][0]
    assert diluted > plain


def test_a_bad_signal_is_never_quantified():
    """A metabolite whose every signal is Bad does not reach the election at
    all — it has no row here, and the Annotation table is where the analyst
    already sees why. Its rival is unaffected."""
    s = _study()
    s['metabolites'][0]['matched_peaks'][0]['annotation_quality'] = 'bad'
    got = _by_name(_run(s))
    assert 'HMDB0000161' not in got
    assert not got['HMDB0000190']['_not_quantified']


def test_an_ambiguous_rank_blocks_both_claimants():
    s = _study()
    got = _by_name(_run(s, ambiguous_ranks=[1]))
    assert got['HMDB0000161']['_not_quantified']
    assert not got['HMDB0000190']['_not_quantified']


def test_a_signal_with_a_peak_and_no_verdict_is_still_measured():
    """Eligibility asks whether something was found, not whether the search
    settled its identity. A peak the search left undecided is a peak that was
    found; how much the number is worth is the confidence's job."""
    s = _study()
    for mp in s['metabolites'][0]['matched_peaks']:
        mp['annotation_quality'] = 'average'
    got = _by_name(_run(s))
    assert not got['HMDB0000161']['_not_quantified']


def test_a_metabolite_whose_signals_have_no_peaks_is_not_eligible():
    """The hole that closed: a signal with no detected peak has no rivals and
    an empty window, so "alone and uncontested" promoted it to Good and it won
    the election over the metabolite's own detected signals."""
    s = _study()
    for mp in s['metabolites'][0]['matched_peaks']:
        mp['annotation_quality'] = None
        mp['peak_rank'] = None
        mp['predicted_lines'] = []
        mp['matched_line_ranks'] = []
    got = _by_name(_run(s))
    assert 'HMDB0000161' not in got


def _study():
    """Two analytes and a standard, in three samples at 1x, 2x and 3x.

    Alanine's methyl doublet at 1.480 ppm (3 H, J 7.2) and lactate's at 1.330
    (3 H, J 6.9) at 60% of its height, with TSP at 0.0 (9 H) held at 1 mM in
    every sample. Built fresh per test because several mutate it.
    """
    scale = [1.0, 2.0, 3.0]
    cols = []
    for k in scale:
        cols.append(_doublet(1.480, 7.2, 100.0 * k)
                    + _doublet(1.330, 6.9, 60.0 * k)
                    + _lorentz(0.000, 300.0))
    peaks = [_peak(1, 1.480 - 7.2 / MHZ / 2), _peak(2, 1.480 + 7.2 / MHZ / 2),
             _peak(3, 1.330 - 6.9 / MHZ / 2), _peak(4, 1.330 + 6.9 / MHZ / 2),
             _peak(9, 0.000)]
    metabolites = [
        {'id': 'HMDB0000161', 'name': 'L-Alanine',
         'matched_peaks': [_signal(1.480, 'd', 3.0, 7.2, 1,
                                   {1: 1.4740, 2: 1.4860})]},
        {'id': 'HMDB0000190', 'name': 'L-Lactic acid',
         'matched_peaks': [_signal(1.330, 'd', 3.0, 6.9, 3,
                                   {3: 1.3243, 4: 1.3358})]},
    ]
    return dict(metabolites=metabolites, peaks=peaks,
                intensities=np.column_stack(cols),
                samples_info=[{'label': 'A', 'type': 'Sample'},
                              {'label': 'A', 'type': 'Sample'},
                              {'label': 'B', 'type': 'Sample'}],
                ref_peak=peaks[-1], scale=scale)
