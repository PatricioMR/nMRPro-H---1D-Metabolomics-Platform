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
                       'pattern': 'NULL'}
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
