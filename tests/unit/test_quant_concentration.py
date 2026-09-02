"""Area to concentration, per sample."""
import numpy as np
import pytest

from processing.quant_concentration import (
    apply_pqn, concentrations, per_proton, pqn_factor, pqn_reference,
    signal_nh, standard_per_proton)


class TestProtonCount:
    def test_a_fractional_count_is_kept(self):
        """A quarter of the reference database's quantifiable signals are
        fractional; flooring them at 1.0 halves those concentrations."""
        assert signal_nh(0.5) == 0.5
        assert signal_nh(0.8) == pytest.approx(0.8)

    def test_a_whole_count_is_kept(self):
        assert signal_nh(3) == 3.0

    @pytest.mark.parametrize('bad', [None, 0, -2, 'x', float('nan')])
    def test_what_the_database_did_not_say_falls_back_to_one(self, bad):
        assert signal_nh(bad) == 1.0

    def test_per_proton_divides_every_sample(self):
        assert per_proton({0: 10.0, 1: 5.0}, 2.0) == {0: 5.0, 1: 2.5}


class TestTheStandard:
    def test_internal_standard_uses_each_sample_own(self):
        std = standard_per_proton({0: 90.0, 1: 45.0}, 9.0)
        assert std == {0: 10.0, 1: 5.0}

    def test_an_external_standard_is_one_number_for_everyone(self):
        """Injected separately, so there is no per-sample standard to use."""
        std = standard_per_proton({0: 90.0, 1: 45.0, 2: 0.0}, 9.0,
                                  external_samples=[0, 1])
        assert set(std.values()) == {7.5}

    def test_a_standard_with_no_signal_is_no_standard(self):
        assert standard_per_proton({0: 0.0}, 9.0) is None
        assert standard_per_proton({}, 9.0) is None
        assert standard_per_proton({0: 5.0}, 9.0, external_samples=[7]) is None


class TestConcentration:
    def test_the_ratio_against_the_standard(self):
        c = concentrations({0: 20.0}, 2.0, standard={0: 10.0},
                           standard_conc=5.0)
        assert c[0] == pytest.approx(20 / 2 / 10 * 5)

    def test_an_internal_standard_shares_the_dilution_so_it_cancels(self):
        """Only the correction applies; the dilution is in both areas."""
        c = concentrations({0: 20.0}, 2.0, standard={0: 10.0},
                           standard_conc=5.0, external=False,
                           dilution={0: 100.0}, correction={0: 2.0})
        assert c[0] == pytest.approx(20 / 2 / 10 * 5 * 2.0)

    def test_an_external_standard_cancels_nothing(self):
        """A fixed standard shares nothing with the sample, so the full
        dilution applies. Getting this backwards scales the whole study."""
        c = concentrations({0: 20.0}, 2.0, standard={0: 10.0},
                           standard_conc=5.0, external=True,
                           dilution={0: 100.0}, correction={0: 2.0})
        assert c[0] == pytest.approx(20 / 2 / 10 * 5 * 100.0)

    def test_without_a_standard_only_the_dilution_scales(self):
        c = concentrations({0: 20.0}, 2.0, dilution={0: 3.0})
        assert c[0] == pytest.approx(30.0)

    def test_a_sample_the_standard_failed_in_is_not_a_zero(self):
        c = concentrations({0: 20.0, 1: 20.0}, 2.0,
                           standard={0: 10.0, 1: 0.0}, standard_conc=5.0)
        assert np.isnan(c[1])
        assert c[0] > 0

    def test_a_missing_dilution_is_one_not_zero(self):
        c = concentrations({0: 20.0}, 2.0, standard={0: 10.0},
                           standard_conc=5.0, external=True)
        assert c[0] == pytest.approx(5.0)


class TestPQN:
    def test_a_uniformly_diluted_sample_reports_its_dilution(self):
        ref = {'a': 10.0, 'b': 20.0, 'c': 30.0}
        half = {'a': 5.0, 'b': 10.0, 'c': 15.0}
        assert pqn_factor(half, ref) == pytest.approx(0.5)

    def test_one_signal_moving_does_not_move_the_factor(self):
        """A biological change moves some signals; a dilution moves all."""
        ref = {'a': 10.0, 'b': 20.0, 'c': 30.0, 'd': 40.0}
        one_up = {'a': 10.0, 'b': 20.0, 'c': 30.0, 'd': 400.0}
        assert pqn_factor(one_up, ref) == pytest.approx(1.0)

    def test_too_few_signals_means_no_correction(self):
        """A median over two quotients is not a mode."""
        assert pqn_factor({'a': 5.0, 'b': 10.0}, {'a': 10.0, 'b': 20.0}) == 1.0

    def test_nothing_comparable_means_no_correction(self):
        assert pqn_factor({}, {}) == 1.0
        assert pqn_factor({'a': 5.0}, {'a': 0.0}) == 1.0

    def test_the_reference_is_the_median_sample(self):
        """Not a chosen sample, which would make the study depend on which one
        was run first."""
        ref = pqn_reference({'a': {0: 1.0, 1: 2.0, 2: 3.0}})
        assert ref['a'] == pytest.approx(2.0)

    def test_applying_it_brings_the_samples_onto_one_scale(self):
        """Sample 1 is sample 0 halved. Both are measured against the median
        profile, so they end equal to it and to each other — the point of the
        correction, and what makes two samples comparable at all."""
        data = {'a': {0: 10.0, 1: 5.0}, 'b': {0: 20.0, 1: 10.0},
                'c': {0: 30.0, 1: 15.0}}
        out, factors = apply_pqn(data)
        assert factors[0] / factors[1] == pytest.approx(2.0)
        for sig in ('a', 'b', 'c'):
            assert out[sig][0] == pytest.approx(out[sig][1])

    def test_the_factors_come_back_with_the_result(self):
        """A concentration corrected by an unseen number cannot be checked."""
        _, factors = apply_pqn({'a': {0: 1.0}, 'b': {0: 2.0}, 'c': {0: 3.0}})
        assert set(factors) == {0}
