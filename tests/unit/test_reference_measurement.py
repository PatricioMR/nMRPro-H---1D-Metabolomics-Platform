"""A reference coupling measured from the spectrum, with its uncertainty.

The databases publish peak lists to three decimals -- half a hertz at 500 MHz
-- and disagree with each other by 0.80 Hz at the median on the couplings the
annotation judges to 0.2 Hz. Measured from the raw spectrum, a coupling is
known to a few hundredths of a hertz at reference signal-to-noise, and it
comes with the number that says so.
"""
import numpy as np
import pytest

from processing.reference_measurement import (
    linewidth_ppm, measure_fingerprint, measure_signal, noise_sigma,
    reference_offset)

FIELD = 500.0
PPM = np.linspace(-1.0, 11.0, 2 ** 17)          # 0.00009 ppm per point
HWHM = 0.0012                                    # 1.2 Hz at 500 MHz


def _line(centre, height, hwhm=HWHM):
    return height * hwhm ** 2 / ((PPM - centre) ** 2 + hwhm ** 2)


def _multiplet(centre, j_hz, n, height):
    """First-order multiplet, binomial heights, lines j_hz apart."""
    row = [1.0]
    for _ in range(n - 1):
        row = [a + b for a, b in zip([0.0] + row, row + [0.0])]
    d = j_hz / FIELD
    y = np.zeros_like(PPM)
    for k, w in enumerate(row):
        y += _line(centre + (k - (n - 1) / 2.0) * d, height * w / max(row))
    return y


def _spectrum(*parts, noise=1.0, seed=0):
    y = np.random.default_rng(seed).normal(0.0, noise, PPM.size)
    for p in parts:
        y += p
    return y


def test_a_doublets_coupling_is_measured_to_hundredths():
    y = _spectrum(_multiplet(1.318, 6.93, 2, 2000.0))
    m = measure_signal(PPM, y, centre=1.318, multiplicity='d',
                       field_mhz=FIELD)
    assert m['status'] == 'ok'
    assert m['j_hz'] == pytest.approx(6.93, abs=0.03)
    assert m['j_sigma_hz'] < 0.05
    assert m['ppm'] == pytest.approx(1.318, abs=0.0005)


def test_the_database_value_would_have_failed_here():
    """7.50 Hz listed against 6.93 measured: more than ten sigma out. The
    disagreement is not within the noise; it is a wrong entry."""
    y = _spectrum(_multiplet(1.318, 6.93, 2, 2000.0))
    m = measure_signal(PPM, y, centre=1.318, multiplicity='d',
                       field_mhz=FIELD)
    assert abs(7.50 - m['j_hz']) / m['j_sigma_hz'] > 10


def test_a_quartet_gives_the_mean_spacing_and_the_pattern():
    y = _spectrum(_multiplet(4.101, 6.93, 4, 600.0))
    m = measure_signal(PPM, y, centre=4.101, multiplicity='q',
                       field_mhz=FIELD)
    assert m['status'] == 'ok'
    assert len(m['lines']) == 4
    assert m['j_hz'] == pytest.approx(6.93, abs=0.05)
    assert m['pattern_agreement'] > 0.9


def test_a_singlet_has_a_position_and_no_coupling():
    y = _spectrum(_line(3.036, 3000.0))
    m = measure_signal(PPM, y, centre=3.036, multiplicity='s',
                       field_mhz=FIELD)
    assert m['status'] == 'ok'
    assert m['j_hz'] is None
    assert m['ppm'] == pytest.approx(3.036, abs=0.0002)


def test_the_uncertainty_grows_as_the_signal_weakens():
    strong = measure_signal(PPM, _spectrum(_multiplet(1.318, 6.93, 2, 2000.0)),
                            centre=1.318, multiplicity='d', field_mhz=FIELD)
    weak = measure_signal(PPM, _spectrum(_multiplet(1.318, 6.93, 2, 40.0)),
                          centre=1.318, multiplicity='d', field_mhz=FIELD)
    assert weak['j_sigma_hz'] > 3 * strong['j_sigma_hz']
    assert abs(weak['j_hz'] - 6.93) < 3 * weak['j_sigma_hz'] + 0.05


def test_missing_lines_are_said_not_guessed():
    """Only one line where a doublet was listed: no coupling can be read off
    one position, and the status says which lines were found."""
    y = _spectrum(_line(1.318, 2000.0))
    m = measure_signal(PPM, y, centre=1.318, multiplicity='d',
                       field_mhz=FIELD)
    assert m['status'] == 'lines_missing'
    assert m['n_found'] == 1 and m['n_expected'] == 2


def test_an_unresolved_multiplet_is_not_measured():
    m = measure_signal(PPM, _spectrum(), centre=3.5, multiplicity='m',
                       field_mhz=FIELD)
    assert m['status'] == 'unresolved'


def test_uneven_spacing_widens_the_uncertainty():
    """A 'triplet' whose lines are 6.0 and 8.0 Hz apart is not a first-order
    triplet at 7 Hz. The spread of the spacings, not the noise, is then what
    the coupling is known to."""
    d1, d2 = 6.0 / FIELD, 8.0 / FIELD
    y = _spectrum(_line(2.10 - d1, 500.0), _line(2.10, 1000.0),
                  _line(2.10 + d2, 500.0))
    m = measure_signal(PPM, y, centre=2.10, multiplicity='t', field_mhz=FIELD)
    assert m['status'] == 'ok'
    assert m['j_sigma_hz'] > 1.0


def test_the_reference_line_offset_is_found_off_the_grid():
    y = _spectrum(_line(0.0037, 5000.0))
    assert reference_offset(PPM, y) == pytest.approx(0.0037, abs=0.0002)


def test_linewidth_is_the_full_width_at_half_maximum():
    y = _line(5.0, 1000.0)
    assert linewidth_ppm(PPM, y, 5.0) == pytest.approx(2 * HWHM, rel=0.05)


def test_noise_comes_from_a_quiet_region():
    y = _spectrum(_line(1.318, 2000.0), noise=3.0)
    assert noise_sigma(PPM, y) == pytest.approx(3.0, rel=0.15)


def test_a_fingerprint_keeps_the_listing_beside_the_measurement():
    y = _spectrum(_multiplet(1.318, 6.93, 2, 2000.0),
                  _multiplet(4.101, 6.93, 4, 600.0))
    out = measure_fingerprint(
        PPM, y, [{'ppm': 1.318, 'multiplicity': 'd', 'j_coupling': '7.50'},
                 {'ppm': 4.101, 'multiplicity': 'q', 'j_coupling': '7.00'}],
        field_mhz=FIELD)
    assert [o['status'] for o in out] == ['ok', 'ok']
    assert out[0]['listed']['j_coupling'] == '7.50'
    assert out[0]['j_hz'] == pytest.approx(6.93, abs=0.03)


def test_a_taller_neighbour_beyond_any_coupling_is_not_the_second_line():
    """Lipid CH2 at 1.28 ppm, 30 Hz from lactate's doublet and taller. A
    window sized to the multiplet leaves it out; and two apices further apart
    than any proton coupling are two signals, said so."""
    from processing.reference_measurement import MAX_J_HZ, search_half_width
    y = _spectrum(_multiplet(1.318, 6.93, 2, 800.0), _line(1.268, 3000.0))
    m = measure_signal(PPM, y, centre=1.318, multiplicity='d',
                       field_mhz=FIELD)
    assert m['status'] == 'ok'
    assert m['j_hz'] == pytest.approx(6.93, abs=0.05)
    assert search_half_width(2, FIELD) < 0.05
    # Forced wide, the lipid line is the tallest in the window and the
    # doublet's lines fall under the pattern floor: one line found, and said
    # so -- never a 30 Hz "coupling".
    wide = measure_signal(PPM, y, centre=1.318, multiplicity='d',
                          field_mhz=FIELD, half_width=0.06)
    assert wide['status'] in ('lines_missing', 'not_a_multiplet')
    assert wide.get('j_hz') is None or wide['j_hz'] <= MAX_J_HZ


class TestTheV02Build:
    """The build's core, on a synthetic lactate: the listing beside the
    measurement, and a disagreement flagged only beyond the measurement's own
    error plus what a curator could have rounded."""

    def _measured(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import measure_entry
        y = _spectrum(_line(0.0021, 5000.0),                  # DSS, off-grid
                      _multiplet(1.318 + 0.0021, 6.93, 2, 2000.0),
                      _multiplet(4.101 + 0.0021, 6.93, 4, 600.0))
        signals = [{'id': 1, 'ppm': 1.318, 'multiplicity': 'd',
                    'j_coupling': '7.50'},
                   {'id': 2, 'ppm': 4.101, 'multiplicity': 'q',
                    'j_coupling': '7.00'}]
        return measure_entry(PPM, y, signals, field_mhz=FIELD)

    def test_the_axis_is_referenced_on_dss_first(self):
        out = self._measured()
        assert out[0]['reference_offset_ppm'] == pytest.approx(0.0021,
                                                              abs=0.0003)
        assert out[0]['ppm'] == pytest.approx(1.318, abs=0.0005)

    def test_lactates_listed_coupling_is_flagged_and_the_quartets_is_not(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import disagreements
        rows = disagreements(self._measured())
        assert [r['multiplicity'] for r in rows] == ['d']
        assert rows[0]['disagrees_on'] == 'J'
        assert rows[0]['j_listed'] == 7.50
        assert rows[0]['j_meas'] == pytest.approx(6.93, abs=0.03)


class TestConditionsAndFieldDependence:
    """A coupling is quoted in hertz because it does not move with the field.
    An apparent splitting of a second-order system does; a rotamer-dependent
    coupling moves with the pH. The measurement carries its conditions, and
    a signal measured at two fields is tested for moving."""

    STAR = '''
save_sample_conditions_1
   loop_
      _Sample_condition_variable.Type
      _Sample_condition_variable.Val
      _Sample_condition_variable.Val_units
     pH            7.4   pH
     temperature   298   K
   stop_
save_
save_sample_1
   loop_
      _Sample_component.ID
      _Sample_component.Mol_common_name
      _Sample_component.Type
     1   'Lactic Acid'        Solute
     2   D2O                  Solvent
     3   'sodium phosphate'   Buffer
     5   DSS                  Reference
   stop_
save_
save_Bruker_DMX_500
   _NMR_spectrometer.Field_strength  500
save_
save_Bruker_DMX_400
   _NMR_spectrometer.Field_strength  400
save_
save_experiment_list
   loop_
      _Experiment.ID
      _Experiment.Name
      _Experiment.Raw_data_flag
      _Experiment.NMR_spectrometer_label
     1   '1D 1H'   yes   $Bruker_DMX_500
     2   '1D 1H'   yes   $Bruker_DMX_400
     3   '1D 13C'  yes   $Bruker_DMX_500
   stop_
   loop_
      _Experiment_file.Experiment_ID
      _Experiment_file.Name
      _Experiment_file.Type
      _Experiment_file.Directory_path
     1   1   text/directory   nmr/set01/
     2   1   text/directory   nmr/set02/
     3   1   text/directory   nmr/set03/
   stop_
save_
'''

    def test_the_conditions_are_read_off_the_entry(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from download_bmrb_fids import sample_conditions
        c = sample_conditions(self.STAR)
        assert c['ph'] == 7.4 and c['temperature'] == 298.0
        assert c['buffer'] == 'sodium phosphate' and c['reference'] == 'DSS'

    def test_each_proton_spectrum_knows_its_field(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from download_bmrb_fids import one_d_proton_dirs
        d = one_d_proton_dirs(self.STAR)
        assert [(x['path'], x['field_mhz']) for x in d] == [
            ('nmr/set01', 500.0), ('nmr/set02', 400.0)]

    def test_a_coupling_that_holds_across_fields_is_not_flagged(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import field_dependence
        dep, flag = field_dependence({500.0: (6.93, 0.02), 400.0: (6.95, 0.03)})
        assert dep == pytest.approx(0.02)
        assert flag is None

    def test_a_splitting_that_moves_with_the_field_is_second_order(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import field_dependence
        dep, flag = field_dependence({500.0: (15.9, 0.03), 400.0: (14.2, 0.04)})
        assert dep == pytest.approx(1.7)
        assert flag == 'second_order_suspect'

    def test_one_field_gives_no_verdict(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import field_dependence
        assert field_dependence({500.0: (6.93, 0.02)}) == (None, None)
