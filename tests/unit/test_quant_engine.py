"""Elect, integrate, concentrate — end to end, on data small enough to check."""
import numpy as np
import pytest

from processing.quant_engine import (
    STATUS_NONE, STATUS_STRONG, STATUS_WEAK, quantify)
from processing.quantification import AMBIGUOUS_RANK, Candidate

PPM = np.linspace(1.40, 1.60, 4000)


def _gauss(c, a, w=0.002):
    return a * np.exp(-((PPM - c) ** 2) / (2 * w ** 2))


def _peak(rank, centre, half=0.006):
    return {'rank': rank, 'ppm': centre,
            'start_ppm': centre - half, 'end_ppm': centre + half}


def _cand(mid, rank, lines, *, protons=1.0, dj=0.05, n_found=2, n_exp=2,
          ppm_diff=0.001, mult='d'):
    return Candidate(met_id=mid, peak_rank=rank,
                     line_ranks=frozenset(lines),
                     evidence={'multiplicity': mult, 'delta_j_hz': dj,
                               'ppm_diff': ppm_diff, 'n_found': n_found,
                               'n_expected': n_exp, 'protons': protons})


def _study(amps):
    """One doublet at 1.47/1.48, scaled per sample."""
    return np.column_stack([_gauss(1.47, a) + _gauss(1.48, a) for a in amps])


PEAKS = {1: _peak(1, 1.47), 2: _peak(2, 1.48), 9: _peak(9, 1.55)}
GROUPS = {'G': [0, 1, 2]}


def _run(cands, X, **kw):
    kw.setdefault('dj_gold_hz', 0.20)
    return quantify(candidates=cands, peaks_by_rank=PEAKS, ppm=PPM,
                    intensities=X, samples_info=[], included=[0, 1, 2],
                    groups=GROUPS, **kw)


class TestTheHappyPath:
    def test_a_row_per_metabolite_with_a_concentration(self):
        rows = _run([_cand('A', 1, [1, 2])], _study([1.0, 2.0, 3.0]))
        assert len(rows) == 1
        r = rows[0]
        assert r['_not_quantified'] is False
        assert r['conc_mm'] > 0
        assert set(r['_concs_per_sample']) == {0, 1, 2}

    def test_the_window_is_reported_with_the_number(self):
        r = _run([_cand('A', 1, [1, 2])], _study([1.0, 1.0, 1.0]))[0]
        lo, hi = r['_multiplet_window']
        # foot to foot over the lines, let out three linewidths (the
        # engine's default 0.008 ppm) past the outer apices
        assert lo == pytest.approx(1.47 - 3 * 0.008)
        assert hi == pytest.approx(1.48 + 3 * 0.008)
        assert lo <= 1.464 and hi >= 1.486

    def test_concentration_follows_the_signal_between_samples(self):
        rows = _run([_cand('A', 1, [1, 2])], _study([1.0, 2.0, 3.0]),
                    use_pqn=False)
        c = rows[0]['_concs_per_sample']
        assert c[1] == pytest.approx(2 * c[0], rel=1e-2)
        assert c[2] == pytest.approx(3 * c[0], rel=1e-2)

    def test_the_proton_count_divides(self):
        one = _run([_cand('A', 1, [1, 2], protons=1.0)],
                   _study([1.0] * 3), use_pqn=False)[0]
        three = _run([_cand('A', 1, [1, 2], protons=3.0)],
                     _study([1.0] * 3), use_pqn=False)[0]
        assert three['conc_mm'] == pytest.approx(one['conc_mm'] / 3, rel=1e-6)
        assert three['_nh'] == 3.0


class TestTheStandard:
    def test_against_a_standard_the_answer_is_in_its_units(self):
        X = _study([1.0, 1.0, 1.0])
        rows = _run([_cand('A', 1, [1, 2])], X, use_pqn=False,
                    standard_areas={0: 90.0, 1: 90.0, 2: 90.0},
                    standard_nh=9.0, standard_conc=5.0)
        r = rows[0]
        expected = r['raw_area'] / 1.0 / 10.0 * 5.0
        assert r['conc_mm'] == pytest.approx(expected, rel=1e-6)

    def test_a_sample_the_standard_failed_in_carries_no_number(self):
        rows = _run([_cand('A', 1, [1, 2])], _study([1.0] * 3), use_pqn=False,
                    standard_areas={0: 90.0, 1: 0.0, 2: 90.0},
                    standard_nh=9.0, standard_conc=5.0)
        assert np.isnan(rows[0]['_concs_per_sample'][1])


class TestWhoGetsNothing:
    def test_a_loser_still_gets_a_row_and_a_reason(self):
        """An analyte that vanishes from the table cannot be questioned."""
        rows = _run([_cand('A', 1, [1, 2], dj=0.05),
                     _cand('B', 1, [1, 2], dj=3.0)], _study([1.0] * 3))
        by = {r['hmdb_id']: r for r in rows}
        assert by['A']['_not_quantified'] is False
        assert by['B']['_not_quantified'] is True
        assert by['B']['status'] == STATUS_NONE
        assert by['B']['_status_reasons'][0].startswith('Not quantified')

    def test_an_ambiguous_rank_leaves_both_without_a_number(self):
        rows = _run([_cand('A', 1, [1, 2]), _cand('B', 1, [1, 2])],
                    _study([1.0] * 3), ambiguous_ranks={1})
        assert all(r['_not_quantified'] for r in rows)
        assert all(r['_rejection'] == AMBIGUOUS_RANK for r in rows)

    def test_sharing_the_lines_is_sharing_the_signal(self):
        rows = _run([_cand('A', 1, [1, 2], dj=0.05),
                     _cand('B', 2, [1, 2], dj=0.30)], _study([1.0] * 3))
        assert sum(1 for r in rows if not r['_not_quantified']) == 1


class TestConfidence:
    def test_a_complete_multiplet_with_a_good_coupling(self):
        r = _run([_cand('A', 1, [1, 2], dj=0.05)], _study([1.0] * 3))[0]
        assert (r['_confidence'], r['status']) == (3, STATUS_STRONG)

    def test_a_singlet_carries_no_coupling_to_agree_with(self):
        r = _run([_cand('S', 9, [9], mult='s', dj=None,
                        n_found=None, n_exp=None)],
                 _study([1.0] * 3) + _gauss(1.55, 1.0)[:, None])[0]
        assert (r['_confidence'], r['status']) == (1, STATUS_WEAK)


class TestTheContract:
    REQUIRED = {
        'hmdb_id', 'metabolite', 'ppm', 'n_peaks', 'raw_area', 'corr_area',
        'conc_mm', 'cv_by_group', 'snr', 'status', 'vip_score', '_confidence',
        '_integ_mode', '_not_quantified', '_status_reasons', '_peak',
        '_peak_rank', '_display_peak_rank', '_multiplet_window', '_nh',
        '_areas_per_sample', '_concs_per_sample', '_concs_by_group',
        '_ann_score', '_hits', '_used_j_resolve', '_j_resolve_frac',
        '_used_mcr', '_mcr', '_lorentz', '_is_is',
    }

    def test_a_quantified_row_carries_every_field_the_window_reads(self):
        r = _run([_cand('A', 1, [1, 2])], _study([1.0] * 3))[0]
        assert self.REQUIRED <= set(r)

    def test_a_rejected_row_carries_them_too(self):
        """A row missing keys makes the window fail on the rows that failed,
        which is exactly where an analyst is looking."""
        rows = _run([_cand('A', 1, [1, 2]), _cand('B', 1, [1, 2])],
                    _study([1.0] * 3), ambiguous_ranks={1})
        assert self.REQUIRED <= set(rows[0])

    def test_only_multiplet_integration_is_reported(self):
        r = _run([_cand('A', 1, [1, 2])], _study([1.0] * 3))[0]
        assert r['_integ_mode'] == 'Multiplet'

    def test_the_jg_fields_say_plainly_that_nothing_was_applied(self):
        r = _run([_cand('A', 1, [1, 2])], _study([1.0] * 3))[0]
        assert r['_used_j_resolve'] is False
        assert r['_j_resolve_frac'] == 1.0
        assert r['_used_mcr'] is False


class TestTheSignalWindow:
    def test_let_out_past_the_lines_and_never_into_a_neighbour(self):
        from processing.quant_integration import signal_window
        lines = [_peak(1, 1.47, half=0.003), _peak(2, 1.48, half=0.003)]
        lo, hi = signal_window(lines, linewidth_ppm=0.005)
        assert lo == pytest.approx(1.47 - 0.015)
        assert hi == pytest.approx(1.48 + 0.015)
        # a neighbour 0.012 above the upper line: the window stops at the
        # midpoint plus half a linewidth, whatever the feet said
        lo2, hi2 = signal_window(lines, linewidth_ppm=0.005,
                                 other_peaks=[_peak(9, 1.492)])
        assert hi2 == pytest.approx((1.492 + 1.48) / 2 + 0.0025)
        assert lo2 == lo
        # a neighbour right on the line still leaves half a linewidth
        lo3, hi3 = signal_window(lines, linewidth_ppm=0.005,
                                 other_peaks=[_peak(9, 1.4805)])
        assert hi3 == pytest.approx((1.4805 + 1.48) / 2 + 0.0025)
        assert hi3 >= 1.48 + 0.0025

    def test_without_a_linewidth_the_feet_stand(self):
        from processing.quant_integration import signal_window
        lines = [_peak(1, 1.47, half=0.003), _peak(2, 1.48, half=0.003)]
        lo, hi = signal_window(lines, linewidth_ppm=None)
        assert (lo, hi) == pytest.approx((1.467, 1.483))

    def test_the_capture_correction_applies_to_a_built_window(self):
        r = _run([_cand('A', 1, [1, 2])], _study([1.0] * 3))[0]
        assert r['_captured_fraction'] is not None
        assert 0.85 < r['_captured_fraction'] <= 1.0


class TestIdentityEvidence:
    """The identity axis, measured on the reference bench: an analyte with
    a signal confirmed was real 7 times out of 7 there, and every one of
    the eight false positives had none."""

    def test_a_confirmed_signal_carries_the_identity(self):
        r = _run([_cand('A', 1, [1, 2], dj=0.05)], _study([1.0] * 3))[0]
        assert r['_identity_level'] == 3
        assert 'confirmed' in r['_identity_why']

    def test_a_signal_missing_at_the_measured_amount_puts_it_at_one(self):
        """Its own listing says a second signal should stand here, and the
        spectrum holds nothing like it."""
        from processing.quant_engine import _identity_evidence
        level, why = _identity_evidence(confirmed=2, contradicted=1, listed=3)
        assert level == 1 and 'missing' in why

    def test_one_listed_signal_cannot_corroborate_itself(self):
        from processing.quant_engine import _identity_evidence
        level, why = _identity_evidence(confirmed=0, listed=1)
        assert level == 1 and 'one signal' in why

    def test_a_covered_listing_that_stands_is_worth_two(self):
        from processing.quant_engine import _identity_evidence
        level, _why = _identity_evidence(confirmed=0, corroborated=True, listed=3)
        assert level == 2

    def test_the_cascade_never_reaches_three(self):
        from processing.quant_engine import _identity_evidence
        assert _identity_evidence(cascade={'disagreement': 0.01})[0] == 2
        assert _identity_evidence(cascade={'disagreement': 0.5})[0] == 1
        assert _identity_evidence(cascade={'disagreement': float('nan')})[0] == 1


class TestWhatCostsADot:
    """A dot is spent on evidence that is actually missing, not on the
    length of a claimant list."""

    def test_the_coupling_is_judged_by_the_study_gate_not_a_literal(self):
        """A reference whose own J carries uncertainty cannot be judged to
        a fifth of a hertz: inosine, matched at 0.25 Hz against a reference
        measured to 0.19, was Good in the annotation window and second-rate
        in the table."""
        from processing.quant_engine import _structural_evidence

        class _S:
            evidence = {'n_found': 3, 'n_expected': 3, 'delta_j_hz': 0.25,
                        'j_sigma_hz': 0.185}
        assert _structural_evidence(_S(), 0.20) == 3
        _S.evidence = {**_S.evidence, 'j_sigma_hz': None}
        assert _structural_evidence(_S(), 0.20) == 2
        assert _structural_evidence(_S(), 0.30) == 3

    def test_a_coupling_far_outside_the_gate_stays_outside_it(self):
        from processing.quant_engine import _structural_evidence

        class _S:
            evidence = {'n_found': 3, 'n_expected': 3, 'delta_j_hz': 1.8,
                        'j_sigma_hz': 0.19}
        assert _structural_evidence(_S(), 0.20) == 2
