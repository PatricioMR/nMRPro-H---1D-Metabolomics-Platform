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

    def test_a_listed_window_that_misses_the_signal_is_not_used(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import measure_entry
        y = _spectrum(_line(0.0, 5000.0),
                      _multiplet(3.411, 6.7, 3, 2000.0))
        # BMRB lists taurine's 3.41 triplet with a window 0.07 ppm below it
        signals = [{'id': 1, 'ppm': 3.41, 'multiplicity': 't',
                    'j_coupling': '6.25', 'ppm_min': 3.3362,
                    'ppm_max': 3.3487, 'n_lines': 3}]
        out = measure_entry(PPM, y, signals, field_mhz=FIELD)
        assert out[0]['status'] == 'ok'
        assert out[0]['pattern']['status'] == 'ok'
        assert len(out[0]['pattern']['lines']) == 3
        lo, hi = out[0]['pattern']['window']
        assert lo < 3.411 < hi

    def test_the_signals_own_listing_is_not_a_neighbour(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import measure_entry
        # L-lysine's triplet: listed at 3.012 with a window that cuts the
        # upper line, measured at 3.0115. The midpoint to its own listing
        # is 3.0117, and the window could not grow past it.
        y = _spectrum(_line(0.0, 5000.0),
                      _multiplet(3.0115, 7.5, 3, 20000.0))
        signals = [{'id': 1, 'ppm': 3.012, 'multiplicity': 't',
                    'j_coupling': '7.55', 'ppm_min': 2.9854,
                    'ppm_max': 3.0156, 'n_lines': 3}]
        out = measure_entry(PPM, y, signals, field_mhz=FIELD)
        assert out[0]['status'] == 'ok'
        assert len(out[0]['pattern']['lines']) == 3

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

    def test_the_mirror_map_is_read_off_the_list_files(self):
        """The PDBj mirror lays entries out by compound, and the only map from
        entry to compound directory is the tarball list files."""
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from download_bmrb_fids import parse_list_files
        m = parse_list_files(['bmse000897 D_ornithine\nbmse000898 D_xylose\n',
                              'bmse000789\tcreatinine\n', 'junk line\n'])
        assert m == {'bmse000897': 'D_ornithine', 'bmse000898': 'D_xylose',
                     'bmse000789': 'creatinine'}

    def test_the_field_comes_from_the_acquisition_parameters(self, tmp_path):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import field_from_acqus
        (tmp_path / 'acqus').write_text('##$BF1= 600.13\n##$SFO1= 600.1328\n')
        assert field_from_acqus(str(tmp_path)) == pytest.approx(600.13)
        assert field_from_acqus(str(tmp_path / 'nothing')) is None

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


class TestTheAppReadsTheMeasurement:
    """A reference known to plus or minus half a hertz cannot judge a match
    to a fifth of one. Where the DB carries the measured coupling and its
    uncertainty, the search reads them and the ranking's band is never
    narrower than three sigma of the reference."""

    def test_the_band_is_the_gate_when_the_reference_is_precise(self):
        from processing.annotation_scoring import dj_band
        assert dj_band(0.15, 0.2, ref_sigma_hz=0.001) == 0
        assert dj_band(0.25, 0.2, ref_sigma_hz=0.001) == 1

    def test_the_band_widens_to_the_references_own_error(self):
        """Listed to the half hertz: sigma ~0.29 (a uniform rounding error),
        three of which is 0.87. A delta-J of 0.6 is then inside band 0."""
        from processing.annotation_scoring import dj_band
        assert dj_band(0.6, 0.2, ref_sigma_hz=0.29) == 0
        assert dj_band(0.6, 0.2) >= 2

    def test_without_a_sigma_nothing_changes(self):
        from processing.annotation_scoring import dj_band
        assert dj_band(0.6, 0.2, ref_sigma_hz=None) == dj_band(0.6, 0.2)

    def test_the_sql_prefers_the_measurement_where_the_columns_exist(self):
        import sys
        sys.path.insert(0, 'gui/stages')
        from gui.stages._annotation_search import _measured_columns_sql
        v01 = _measured_columns_sql({'ppm', 'j_coupling'})
        assert v01 == {'ppm': 's.ppm', 'j': 's.j_coupling', 'j_sigma': 'NULL',
                       'pattern': 'NULL', 'profile': 'NULL',
                       'profile_fwhm': 'NULL', 'profile_nh': 'NULL'}
        v03 = _measured_columns_sql({'ppm', 'pattern_profile',
                                     'profile_fwhm_hz'})
        assert v03['profile'] == 's.pattern_profile'
        assert v03['profile_fwhm'] == 's.profile_fwhm_hz'
        v02 = _measured_columns_sql({'ppm', 'j_coupling', 'ppm_meas', 'j_use',
                                     'j_use_source', 'j_sigma'})
        assert 'COALESCE(s.ppm_meas, s.ppm)' == v02['ppm']
        assert 'j_use' in v02['j'] and 'ELSE s.j_coupling' in v02['j']
        assert 'measured%' in v02['j_sigma']

    def test_the_sql_runs_against_a_v02_shaped_table(self, tmp_path):
        import sqlite3
        from gui.stages._annotation_search import _measured_columns_sql
        con = sqlite3.connect(':memory:')
        con.execute('create table shifts_1h (bmrb_id, ppm real, j_coupling '
                    'text, ppm_meas real, j_use real, j_use_source text, '
                    'j_sigma real)')
        con.execute("insert into shifts_1h values ('a', 1.318, '7.50', "
                    "1.3177, 6.936, 'measured', 0.001)")
        con.execute("insert into shifts_1h values ('b', 1.317, '7.20', "
                    "NULL, NULL, NULL, NULL)")
        cols = {r[1] for r in con.execute('pragma table_info(shifts_1h)')}
        m = _measured_columns_sql(cols)
        rows = con.execute(f"select s.bmrb_id, {m['ppm']}, {m['j']}, "
                           f"{m['j_sigma']} from shifts_1h s order by 1").fetchall()
        assert rows[0] == ('a', 1.3177, '6.936', 0.001)
        assert rows[1] == ('b', 1.317, '7.20', None)


class TestWhichCouplingToUse:
    """A small sigma says the lines were fitted well, not that they were the
    right lines. The measurement stands where it corrects a listing by a
    plausible amount or where the listing was never a coupling; otherwise
    the listing stays and the row is marked for review."""

    def _f(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import coupling_to_use
        return coupling_to_use

    def test_a_plausible_correction_is_used(self):
        assert self._f()(7.50, 6.936) == (6.936, 'measured')

    def test_a_listed_span_is_replaced_by_the_measurement(self):
        assert self._f()(64.05, 8.49) == (8.49, 'measured')

    def test_a_gross_disagreement_keeps_the_listing_for_review(self):
        assert self._f()(3.20, 11.59) == (3.20, 'review')

    def test_no_measurement_means_the_listing(self):
        assert self._f()(7.20, None) == (7.20, 'listed')


class TestPatterns:
    """An unresolved multiplet has no letter and no single J, but its lines
    stand where its couplings put them, and that can be measured, stored in
    hertz, and matched -- which is what lets D-glucose verify anything."""

    def test_a_dd_is_measured_as_four_offsets_in_hertz(self):
        from processing.reference_measurement import measure_pattern
        c, j1, j2 = 3.232, 8.0, 9.5
        y = _spectrum(*[_line(c + (a * j1 + b * j2) / 2 / FIELD, 500.0)
                        for a in (-1, 1) for b in (-1, 1)])
        pat = measure_pattern(PPM, y, lo=c - 0.03, hi=c + 0.03, field_mhz=FIELD)
        assert pat['status'] == 'ok' and pat['n_lines'] == 4
        offs = [o for o, _ in pat['lines']]
        assert offs == pytest.approx([-8.75, -0.75, 0.75, 8.75], abs=0.05)
        assert pat['centre_ppm'] == pytest.approx(c, abs=0.0002)

    def test_the_residual_ignores_a_shift_and_measures_the_shape(self):
        from processing.annotation_scoring import pattern_residual_hz
        pred = [3.2300, 3.2316, 3.2319, 3.2335]
        shifted = [p + 0.021 for p in pred]                # same shape, drifted
        assert pattern_residual_hz(pred, shifted, FIELD) == pytest.approx(0.0, abs=1e-6)
        wrong = [3.2300, 3.2316, 3.2319, 3.2380]           # one line off by 2.2 Hz
        assert pattern_residual_hz(pred, wrong, FIELD) > 0.5

    def test_a_pattern_multiplet_is_verified_by_its_residual(self):
        """measure_multiplet on 'm*': no J to compare, the residual stands in
        for delta-J and a complete, well-placed pattern is confirmed."""
        from processing.annotation_scoring import (
            PATTERN_MULT, candidate_tier, measure_multiplet,
            TIER_CONFIRMED_MULTIPLET)
        # A dd with J 3 and 10 Hz at 500 MHz: offsets +/-3.5 and +/-6.5 Hz.
        pred = [{'ppm': 3.232 + o / FIELD} for o in (-6.5, -3.5, 3.5, 6.5)]
        peaks = [{'rank': i, 'ppm': p['ppm'] + 0.004} for i, p in enumerate(pred)]
        m = measure_multiplet(pred, peaks, mult=PATTERN_MULT,
                              j_coupling='1.15 8.05', db_ppm=3.232,
                              centre_hint=3.236, n_expected=4,
                              tolerance=0.01, field_mhz=FIELD)
        assert m.complete and m.delta_j_hz == pytest.approx(0.0, abs=0.01)
        assert candidate_tier(PATTERN_MULT, m.delta_j_hz, m.n_found,
                              m.n_expected) == TIER_CONFIRMED_MULTIPLET

    def test_a_pattern_is_not_singlet_like(self):
        from processing.annotation_scoring import PATTERN_MULT, is_singlet_like
        assert not is_singlet_like(PATTERN_MULT)
        assert is_singlet_like('m')

    def test_the_search_lays_the_pattern_on_the_axis_at_its_field(self):
        from gui.stages._annotation_search import pattern_of
        entry = {'ppm_center': 3.232, 'pattern_lines': '[[-8.75, 0.9], [8.75, 1.0]]'}
        lines = pattern_of(entry, 600.0)
        assert lines[0][0] == pytest.approx(3.232 - 8.75 / 600.0)
        assert lines[1][0] == pytest.approx(3.232 + 8.75 / 600.0)
        assert pattern_of({'ppm_center': 3.2, 'pattern_lines': None}, 600.0) == []


class TestMatchPattern:
    """A pattern is matched as a shape: each line within two hertz of where
    the aligned pattern puts it, the alignment finding the most lines wins,
    the residual is over the found lines. The annotation's 0.05 ppm tolerance
    is for where a multiplet sits, not where its lines sit; used per line it
    matched any peak in the neighbourhood and the residual measured the
    neighbourhood -- 3.4 Hz at the median on the reference study."""

    PAT = [(-9.2, 0.4), (-5.2, 0.8), (0.1, 1.0), (4.1, 0.8), (9.6, 0.4)]

    def _peaks(self, centre, jitter_hz=0.0, extra=()):
        from processing.annotation_scoring import match_pattern
        pk = [{'ppm': centre + (o + jitter_hz * (i % 2 * 2 - 1)) / 600.0}
              for i, (o, _r) in enumerate(self.PAT)]
        pk += [{'ppm': e} for e in extra]
        return pk, match_pattern

    def test_a_pattern_present_is_found_whole_with_a_small_residual(self):
        pk, match_pattern = self._peaks(3.401, jitter_hz=0.2)
        m = match_pattern(self.PAT, pk, anchor_ppm=pk[2]['ppm'], field_mhz=600.0)
        assert m['n_found'] == 5 and m['n_expected'] == 5
        assert m['residual_hz'] < 0.5

    def test_the_anchor_may_be_any_line_of_the_pattern(self):
        """The search matched the signal on whichever line sat nearest the
        database position; the pattern must align on that line, not assume it
        is the centre."""
        pk, match_pattern = self._peaks(3.401)
        m = match_pattern(self.PAT, pk, anchor_ppm=pk[0]['ppm'], field_mhz=600.0)
        assert m['n_found'] == 5

    def test_a_drifted_pattern_still_matches(self):
        """The whole multiplet 0.03 ppm from the database is delta-ppm's
        business; the shape is unchanged and the residual says so."""
        pk, match_pattern = self._peaks(3.431, jitter_hz=0.1)
        m = match_pattern(self.PAT, pk, anchor_ppm=pk[2]['ppm'], field_mhz=600.0)
        assert m['n_found'] == 5 and m['residual_hz'] < 0.5

    def test_neighbouring_peaks_thirty_hertz_away_are_not_lines(self):
        pk, match_pattern = self._peaks(3.401)
        pk = [pk[2]] + [{'ppm': 3.401 + d} for d in (0.02, 0.035, -0.025, -0.04)]
        m = match_pattern(self.PAT, pk, anchor_ppm=3.401, field_mhz=600.0)
        # Two of those neighbours happen to sit 13 Hz apart, as two lines of
        # the pattern do; that is two lines found, never five.
        assert m['n_found'] < m['n_expected']

    def test_a_different_shape_has_a_large_residual(self):
        """Lines at roughly the right places but not the pattern's: found,
        and the residual says they are not it."""
        pk, match_pattern = self._peaks(3.401)
        pk = [{'ppm': 3.401 + o / 600.0} for o in (-9.2, -3.5, 0.1, 5.8, 9.6)]
        m = match_pattern(self.PAT, pk, anchor_ppm=pk[2]['ppm'], field_mhz=600.0)
        assert m['n_found'] == 5
        from processing.annotation_scoring import PATTERN_GOLD_HZ
        assert m['residual_hz'] > PATTERN_GOLD_HZ


class TestOnePathForPatterns:
    """resolve_dj re-measures every signal after the axis is recalibrated. For
    a pattern it snapped the stored lines with the thirty-hertz tolerance and
    wrote the result over the search's: a three-line pattern found whole came
    back as two lines seven hertz off. One function now, asked from both."""

    def test_resolve_dj_matches_the_pattern_not_the_stored_lines(self):
        from processing.annotation_scoring import (
            PATTERN_MULT, SCORING_DEFAULTS, resolve_dj)
        pat = [[-9.2, 0.4], [-5.2, 0.8], [0.1, 1.0], [4.1, 0.8], [9.6, 0.4]]
        peaks = [{'rank': i, 'ppm': 3.431 + o / 600.0}
                 for i, (o, _r) in enumerate(pat)]
        import json
        mp = {'db_ppm': 3.401, 'peak_ppm': 3.431, 'ppm_center': 3.401,
              'type': PATTERN_MULT, 'pattern_lines': json.dumps(pat),
              'predicted_lines': [{'ppm': 3.431}, {'ppm': 3.44}],  # stale
              '_n_found_lines': 2, '_n_exp_lines': 2, 'delta_j_hz': 3.0}
        met = {'db_peaks': []}
        s_dj, conf = resolve_dj(mp, met, peaks, mult=PATTERN_MULT,
                                params=SCORING_DEFAULTS, field_mhz=600.0,
                                tolerance=0.05, remeasure=True)
        assert mp['_n_found_lines'] == 5 and mp['_n_exp_lines'] == 5
        assert mp['pattern_residual_hz'] < 0.2
        assert mp['delta_j_hz'] < 0.1
        assert s_dj > 0.9


class TestResolvablePattern:
    """A reference pattern is measured on a pure compound at a linewidth of a
    hertz or two; the study's lines are wider. Two pattern lines closer than
    the study can resolve are one peak in the study, and are expected as one
    -- finding two of three was the right count of the wrong expectation."""

    def test_lines_closer_than_the_linewidth_collapse(self):
        from processing.annotation_scoring import resolvable_pattern
        pat = [(-3.2, 0.94), (-2.1, 1.0), (5.9, 0.86)]     # glucose 3.239
        out = resolvable_pattern(pat, merge_hz=3.2)
        assert len(out) == 2
        assert out[0][0] == pytest.approx((-3.2 * 0.94 + -2.1 * 1.0) / 1.94)
        assert out[1][1] < out[0][1]                       # merged pair is tallest

    def test_lines_the_study_resolves_are_kept(self):
        from processing.annotation_scoring import resolvable_pattern
        pat = [(-9.2, 0.4), (-5.2, 0.8), (0.1, 1.0), (4.1, 0.8), (9.6, 0.4)]
        assert len(resolvable_pattern(pat, merge_hz=3.2)) == 5

    def test_no_linewidth_means_no_merging(self):
        from processing.annotation_scoring import resolvable_pattern
        pat = [(-3.2, 0.9), (-2.1, 1.0)]
        assert resolvable_pattern(pat, None) == pat

    def test_the_measurement_expects_what_the_study_can_resolve(self):
        import json
        from processing.annotation_scoring import measure_pattern_signal
        pat = [[-3.2, 0.94], [-2.1, 1.0], [5.9, 0.86]]
        # In the study the close pair is one peak, at its weighted position.
        peaks = [{'ppm': 3.239 + o / 600.0} for o in (-2.63, 5.9)]
        mp = {'pattern_lines': json.dumps(pat), 'peak_ppm': peaks[1]['ppm']}
        m = measure_pattern_signal(mp, peaks, field_mhz=600.0, dj_gold_hz=0.2,
                                   merge_hz=3.2)
        assert m.n_expected == 2 and m.n_found == 2 and m.complete


class TestTheWindowLetsItselfOut:
    """The database's window for a signal was drawn by a curator, and it cut
    D-glucose's doublet of doublets at 3.232 ppm to three lines of four with
    the centre in the wrong place. A window's edge must sit on noise: while
    it stands on signal it is cutting a line and is let out -- never past the
    midpoint to the compound's neighbouring signal."""

    def _dd(self, c=3.232, j1=8.0, j2=9.5, h=500.0):
        return [_line(c + (a * j1 + b * j2) / 2 / FIELD, h)
                for a in (-1, 1) for b in (-1, 1)]

    def test_a_window_that_cuts_the_outer_lines_grows_until_it_does_not(self):
        from processing.reference_measurement import measure_pattern_extended
        y = _spectrum(*self._dd())
        # +/- 8.6 Hz: the outer lines at +/- 8.75 sit just outside
        pat = measure_pattern_extended(PPM, y, lo=3.232 - 8.6 / FIELD,
                                       hi=3.232 + 8.6 / FIELD, field_mhz=FIELD)
        assert pat['n_lines'] == 4
        assert [o for o, _ in pat['lines']] == pytest.approx(
            [-8.75, -0.75, 0.75, 8.75], abs=0.15)

    def test_it_never_crosses_into_the_neighbour(self):
        from processing.reference_measurement import measure_pattern_extended
        # a neighbouring multiplet 12 Hz away whose midpoint is 6 Hz out
        y = _spectrum(*self._dd(), _line(3.232 + 12.0 / FIELD, 800.0))
        bounds = (3.232 - 0.05, 3.232 + 6.0 / FIELD)
        pat = measure_pattern_extended(PPM, y, lo=3.232 - 8.6 / FIELD,
                                       hi=3.232 + 5.0 / FIELD, field_mhz=FIELD,
                                       bounds=bounds)
        assert pat['window'][1] <= bounds[1] + 1e-9
        assert all(o < 10.0 for o, _ in pat['lines'])

    def test_a_window_already_on_noise_stays_put(self):
        from processing.reference_measurement import measure_pattern_extended
        y = _spectrum(*self._dd())
        # edges 30 Hz out: the outer lines' tails are under the noise there
        lo, hi = 3.232 - 0.06, 3.232 + 0.06
        pat = measure_pattern_extended(PPM, y, lo=lo, hi=hi, field_mhz=FIELD)
        assert pat['window'] == (lo, hi)
        assert len(pat['lines']) == 4


def test_the_per_sample_consensus_leaves_a_patterns_residual_alone():
    """The comb fit assumes one spacing and compares it with the first
    coupling in the database's list -- 1.15 Hz for glucose's dd of 8 and 9.5.
    Its consensus overwrote a residual of 0.46 Hz with a disagreement of
    several, and a complete, well-placed pattern went Bad."""
    from processing.annotation_scoring import (
        PATTERN_MULT, SCORING_DEFAULTS, apply_per_sample_consensus)
    mp = {'type': PATTERN_MULT, 'delta_j_hz': 0.18, 'observed_j_hz': None,
          'per_sample_consensus': {'delta_j_hz': 6.5, 'observed_j_hz': 7.6,
                                   'ppm_diff': 0.003, 'ppm_centre': 3.234}}
    assert apply_per_sample_consensus(mp, params=SCORING_DEFAULTS) is None
    assert mp['delta_j_hz'] == 0.18
    assert mp['ppm_diff'] == 0.003          # the position still applies


def test_a_pattern_carries_its_own_intensity_agreement():
    """None sorts last in the ranking. A confirmed pattern with no agreement
    lost D-glucose's peak to a triplet on 'line intensities' -- the triplet
    had a number and the pattern had None, which is not disagreement."""
    import json
    from processing.annotation_scoring import measure_pattern_signal
    pat = [[-9.0, 0.5], [-0.9, 1.0], [0.2, 1.0], [8.2, 0.5]]
    peaks = [{'ppm': 3.234 + o / 600.0, 'height': h}
             for (o, _r), h in zip(pat, (52.0, 100.0, 97.0, 49.0))]
    mp = {'pattern_lines': json.dumps(pat), 'peak_ppm': peaks[1]['ppm']}
    m = measure_pattern_signal(mp, peaks, field_mhz=600.0, dj_gold_hz=0.2)
    assert m.complete
    assert m.pattern_intensity_agreement > 0.95


class TestAPatternsDrift:
    """Delta-ppm of a pattern is where its reference position fell, not
    where its centre of found lines is. Glucose's 3.234 pattern anchored on
    its lowest line read +0.002 while the pattern as a whole sat +0.017, with
    the rest of the analyte; and an asymmetric pattern's found centre is not
    its reference position even when every line is found."""

    PAT = [[-9.0, 0.7], [-0.9, 0.95], [0.2, 1.0], [8.2, 0.85]]

    def _measure(self, anchor, drift_ppm, *, merge_hz=0.0):
        import json
        from processing.annotation_scoring import (
            PATTERN_MULT, measure_pattern_signal)
        # the pattern, drifted, with two of its lines merged as the study
        # resolves them
        offs = [-9.0, -0.35, 8.2]
        peaks = [{'rank': i, 'ppm': 3.234 + drift_ppm + o / 600.0, 'height': 1.0}
                 for i, o in enumerate(offs)]
        mp = {'db_ppm': 3.234, 'type': PATTERN_MULT,
              'pattern_lines': json.dumps(self.PAT)}
        return measure_pattern_signal(mp, peaks, field_mhz=600.0,
                                      dj_gold_hz=0.2, anchor_ppm=anchor,
                                      merge_hz=2.0)

    def test_the_origin_is_the_drift_whatever_line_anchored_it(self):
        drift = 0.017
        lowest = 3.234 + drift - 9.0 / 600.0
        highest = 3.234 + drift + 8.2 / 600.0
        a = self._measure(lowest, drift)
        b = self._measure(highest, drift)
        assert a.complete and b.complete
        assert a.origin_ppm == pytest.approx(3.234 + drift, abs=0.0005)
        assert b.origin_ppm == pytest.approx(a.origin_ppm, abs=1e-6)
        # the centre of found lines is not the reference position
        assert a.centre_ppm != pytest.approx(a.origin_ppm, abs=1e-4)

    def test_resolve_dj_writes_the_drift_not_the_centre(self):
        import json
        from processing.annotation_scoring import (
            PATTERN_MULT, SCORING_DEFAULTS, resolve_dj)
        drift = 0.017
        offs = [-9.0, -0.35, 8.2]
        peaks = [{'rank': i, 'ppm': 3.234 + drift + o / 600.0, 'height': 1.0}
                 for i, o in enumerate(offs)]
        mp = {'db_ppm': 3.234, 'peak_ppm': 3.234 + drift - 9.0 / 600.0,
              'type': PATTERN_MULT, 'pattern_lines': json.dumps(self.PAT),
              'pattern_merge_hz': 2.0}
        resolve_dj(mp, {'db_peaks': []}, peaks, mult=PATTERN_MULT,
                   params=SCORING_DEFAULTS, field_mhz=600.0,
                   tolerance=0.05, remeasure=True)
        assert mp['ppm_diff'] == pytest.approx(drift, abs=0.0005)
        assert mp['pattern_centre_offset_ppm'] == pytest.approx(
            mp['peak_ppm'] - (3.234 + drift), abs=5e-5)


class TestTheMeasurementGoverns:
    """Where the measured line count contradicts the listed letter, the
    measurement says what to look for. Taurine's 3.41 ppm signal is listed
    'd' and measured as three lines at 1:2:1; judged as a doublet it found two
    of three and was Bad on every study."""

    def test_an_unresolved_m_with_a_pattern_is_governed(self):
        from processing.annotation_scoring import pattern_governs
        # a shape that is not a first-order letter
        assert pattern_governs('m', [[-9.0, 0.7], [-0.9, 0.95], [0.25, 1.0],
                                     [8.2, 0.85]])
        # one that is, is matched as the letter instead
        assert not pattern_governs('m', [[-7.0, 0.5], [0.0, 1.0], [7.0, 0.5]])

    def test_a_letter_the_pattern_agrees_with_keeps_the_letter(self):
        from processing.annotation_scoring import pattern_governs
        assert not pattern_governs('d', [[-3.5, 1.0], [3.5, 1.0]])
        assert not pattern_governs('t', [[-7.0, 0.5], [0.0, 1.0], [7.0, 0.5]])

    def test_a_letter_the_pattern_contradicts_is_overruled(self):
        import json
        from processing.annotation_scoring import pattern_governs
        # a 'd' measured as an uneven three-line shape: the shape governs
        uneven = [[-6.0, 0.663], [0.724, 1.0], [9.5, 0.456]]
        assert pattern_governs('d', uneven)
        assert pattern_governs('d', json.dumps(uneven))
        # taurine's 'd', measured as an even triplet, is a triplet instead
        taurine = [[-6.0, 0.663], [0.724, 1.0], [7.131, 0.456]]
        assert not pattern_governs('d', taurine)
        assert pattern_governs('s', [[-1.0, 0.9], [0.0, 1.0], [1.0, 0.9]])

    def test_no_pattern_no_say(self):
        from processing.annotation_scoring import pattern_governs
        assert not pattern_governs('m', None)
        assert not pattern_governs('d', [])
        assert not pattern_governs('d', [[0.0, 1.0]])


def test_a_curated_window_that_cuts_an_outer_line_is_let_out():
    """L-lysine's 3.012 ppm triplet is listed with a window of 2.9854-3.0156:
    the upper line at +7.5 Hz falls outside it. The window is full of signal,
    so its own tenth percentile is a tail and the outer line's flank at the
    edge sat below it; the floor must be the baseline around the multiplet."""
    from processing.reference_measurement import measure_pattern_extended
    y = _spectrum(_multiplet(3.0117, 7.5, 3, 20000.0))
    pat = measure_pattern_extended(PPM, y, lo=2.9854, hi=3.0156,
                                   field_mhz=FIELD)
    assert pat['status'] == 'ok'
    assert len(pat['lines']) == 3
    lo, hi = pat['window']
    assert hi > 3.0117 + 7.5 / FIELD


class TestALineTheStudyCouldNotSee:
    """The reference measured its lines at a signal-to-noise of thousands; a
    line at a twentieth of the tallest is real there and invisible in a study
    at twenty. What could not have been seen is not counted missing."""

    PAT = [(-9.0, 0.05), (-3.0, 1.0), (3.0, 0.9), (9.0, 0.05)]

    def _peaks(self):
        return [{'rank': i, 'ppm': 3.0 + o / 600.0, 'height': 100.0 * r}
                for i, (o, r) in enumerate(self.PAT) if r > 0.5]

    def test_faint_lines_are_not_expected_under_the_floor(self):
        from processing.annotation_scoring import match_pattern
        pm = match_pattern(self.PAT, self._peaks(),
                           anchor_ppm=3.0 - 3.0 / 600.0, field_mhz=600.0,
                           anchor_height=100.0, floor=20.0)
        assert pm['n_expected'] == 2 and pm['n_found'] == 2

    def test_the_same_lines_are_expected_where_they_could_be_seen(self):
        from processing.annotation_scoring import match_pattern
        pm = match_pattern(self.PAT, self._peaks(),
                           anchor_ppm=3.0 - 3.0 / 600.0, field_mhz=600.0,
                           anchor_height=100.0, floor=2.0)
        assert pm['n_expected'] == 4 and pm['n_found'] == 2

    def test_without_a_floor_every_line_is_expected(self):
        from processing.annotation_scoring import match_pattern
        pm = match_pattern(self.PAT, self._peaks(),
                           anchor_ppm=3.0 - 3.0 / 600.0, field_mhz=600.0)
        assert pm['n_expected'] == 4

    def test_the_floor_rides_on_the_matched_peak(self):
        import json
        from processing.annotation_scoring import (
            PATTERN_MULT, measure_pattern_signal)
        mp = {'db_ppm': 3.0, 'type': PATTERN_MULT,
              'pattern_lines': json.dumps(self.PAT), 'pattern_floor': 20.0}
        m = measure_pattern_signal(mp, self._peaks(), field_mhz=600.0,
                                   dj_gold_hz=0.2,
                                   anchor_ppm=3.0 - 3.0 / 600.0)
        assert m.complete and m.n_expected == 2


def test_satellites_do_not_make_a_singlet_a_pattern():
    from processing.annotation_scoring import pattern_governs
    acetate = [[-5.5, 0.07], [0.0, 1.0], [5.5, 0.06]]
    assert not pattern_governs('s', acetate)
    creatinine = [[-9.3, 0.07], [0.6, 1.0]]
    assert not pattern_governs('s', creatinine)
    cystine = [[-6.0, 0.9], [-1.3, 1.0], [1.5, 0.97], [6.3, 0.83]]
    assert pattern_governs('s', cystine)


def test_a_substantial_line_is_expected_whatever_the_anchor_height():
    """Anchored on a small peak the scale shrinks; the floor must not shrink
    the pattern to the two lines that happened to be found."""
    from processing.annotation_scoring import match_pattern
    pat = [(-9.0, 1.0), (-3.0, 0.9), (3.0, 0.8), (9.0, 0.05)]
    peaks = [{'ppm': 3.0 + o / 600.0, 'height': 1.0}
             for o, r in pat[:2]]
    pm = match_pattern(pat, peaks, anchor_ppm=3.0 - 9.0 / 600.0,
                       field_mhz=600.0, anchor_height=1.0, floor=100.0)
    assert pm['n_expected'] == 3 and pm['n_found'] == 2


class TestTheProtonsUnderAProfile:
    """What a shape is divided by is measured, not listed: the window's share
    of the compound's profiled area times the listed protons. The windows
    partition the compound's area, so a neighbour's tail inside a window is
    counted where it lies."""

    def _measured(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import measure_entry
        # a CH3 doublet (3H) and a CH quartet (1H), well apart
        y = _spectrum(_line(0.0, 5000.0),
                      _multiplet(1.318, 6.93, 2, 3000.0),
                      _multiplet(4.101, 6.93, 4, 500.0))
        signals = [{'id': 1, 'ppm': 1.318, 'multiplicity': 'd',
                    'j_coupling': '6.93', 'protons': 3},
                   {'id': 2, 'ppm': 4.101, 'multiplicity': 'q',
                    'j_coupling': '6.93', 'protons': 1}]
        return measure_entry(PPM, y, signals, field_mhz=FIELD)

    def test_the_shares_add_up_to_the_listed_protons(self):
        out = self._measured()
        nh = [m['profile_nh'] for m in out]
        assert all(v is not None for v in nh)
        assert sum(nh) == pytest.approx(4.0, abs=1e-6)

    def test_the_protons_follow_the_areas_not_the_listing(self):
        out = self._measured()
        by = {m['listed']['id']: m['profile_nh'] for m in out}
        # by construction: the doublet's lines are 3000 each (area 6000),
        # the quartet's 1:3:3:1 with the tallest at 500 (area 1333) -- so
        # the areas stand 4.5:1 whatever the listing says (3:1)
        assert by[1] / by[2] == pytest.approx(4.5, rel=0.1)

    def test_no_profile_no_share(self):
        import sys
        sys.path.insert(0, 'hmdb_api_download')
        from measure_reference_db import assign_profile_protons
        ms = [{'listed': {'protons': 2}, 'profile_area': 10.0},
              {'listed': {'protons': 5}, 'profile_area': None}]
        assign_profile_protons(ms)
        assert ms[0]['profile_nh'] == pytest.approx(2.0)
        assert ms[1]['profile_nh'] is None


class TestAMeasuredLetter:
    """A pattern whose lines are evenly spaced at binomial heights is a
    first-order multiplet, and is matched as one -- lines to find, a
    coupling to measure -- whatever the listing said."""

    def test_taurines_listed_doublet_is_a_triplet(self):
        from processing.annotation_scoring import measured_letter, pattern_governs
        pat = [[-6.0, 0.663], [0.724, 1.0], [7.131, 0.456]]
        letter, j = measured_letter(pat)
        assert letter == 't' and j == pytest.approx(6.57, abs=0.05)
        assert not pattern_governs('d', pat)

    def test_a_doublet_of_doublets_is_not_a_letter(self):
        from processing.annotation_scoring import measured_letter, pattern_governs
        pat = [[-9.035, 0.715], [-0.906, 0.944], [0.239, 1.0], [8.212, 0.862]]
        assert measured_letter(pat) is None
        assert pattern_governs('m', pat)

    def test_a_one_hertz_splitting_is_not_a_doublet_here(self):
        from processing.annotation_scoring import measured_letter
        assert measured_letter([[-0.5, 0.97], [0.5, 1.0]]) is None

    def test_uneven_spacing_or_wrong_heights_are_not_a_letter(self):
        from processing.annotation_scoring import measured_letter
        assert measured_letter([[-7.0, 1.0], [0.0, 1.0], [9.5, 1.0]]) is None
        assert measured_letter([[-7.0, 0.1], [0.0, 1.0], [7.0, 0.1]]) is None
        assert measured_letter([[-7.0, 1.0], [0.0, 0.3], [7.0, 1.0]]) is None

    def test_a_quintets_faint_outer_pair_is_not_required(self):
        from processing.annotation_scoring import measured_letter
        pat = [[-14.0, 0.17], [-7.0, 0.67], [0.0, 1.0], [7.0, 0.67], [14.0, 0.17]]
        letter, j = measured_letter(pat)
        assert letter == 't' and j == pytest.approx(7.0)
