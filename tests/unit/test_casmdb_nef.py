# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The CASMDB NEF reader on a trimmed copy of its L-alanine entry: one HMDB
peak list with its two multiplets and one GISSMO spin matrix."""
import pytest

from data.casmdb_nef import parse_frames, read_entry

ALANINE = """data_l-alanine

   save_nef_nmr_spectrum_l-alanine`1`
      _nef_nmr_spectrum.sf_category                   nef_nmr_spectrum
      _nef_nmr_spectrum.ccpn_spectrum_comment         ''
      _nef_nmr_spectrum.ccpn_spectrum_type            peak_list
      _nef_nmr_spectrum.ccpn_origin                   HMDB
      _nef_nmr_spectrum.ccpn_file_path                /Users/x/Ndf5_spectra/SP-171.ndf5
      _nef_nmr_spectrum.ccpn_similarity_score         0.9765002008

      loop_
         _nef_spectrum_dimension.dimension_id
         _nef_spectrum_dimension.axis_unit
         _nef_spectrum_dimension.spectrometer_frequency

         1  ppm  500
      stop_

      loop_
         _nef_peak.index
         _nef_peak.peak_id
         _nef_peak.height
         _nef_peak.position_1

         1  1  0.9806  1.459
         2  2  1       1.474
         3  3  0.0631  3.751
         4  4  0.2025  3.766
         5  5  0.203   3.78
         6  6  0.0649  3.795
      stop_

      loop_
         _ccpn_multiplet.multiplet_list_serial
         _ccpn_multiplet.multiplet_serial
         _ccpn_multiplet.height
         _ccpn_multiplet.volume
         _ccpn_multiplet.position

         1  1  1.9806  .  '(1.4665, 0.9903)'
         1  2  0.5335  .  '(3.7729999999999997, 0.133375)'
      stop_

      loop_
         _ccpn_multiplet_peaks.multiplet_list_serial
         _ccpn_multiplet_peaks.multiplet_serial
         _ccpn_multiplet_peaks.peak_serial

         1  1  1
         1  1  2
         1  2  3
         1  2  4
         1  2  5
         1  2  6
      stop_
   save_

   save_nef_nmr_spectrum_l-alanine_2`1`
      _nef_nmr_spectrum.ccpn_spectrum_type            spin_system
      _nef_nmr_spectrum.ccpn_origin                   GISSMO
      _nef_nmr_spectrum.ccpn_similarity_score         .

      loop_
         _ccpn_spin_system_matrix.multiplet_0
         _ccpn_spin_system_matrix.multiplet_1
         _ccpn_spin_system_matrix.multiplet_2
         _ccpn_spin_system_matrix.multiplet_3

         1.4662     -12.5      -12.5      7.2988397
         -12.5      1.4662     -12.5      7.2988397
         -12.5      -12.5      1.4662     7.2988397
         7.2988397  7.2988397  7.2988397  3.7674
      stop_
   save_

   save_ccpn_substance_l-alanine.metabolite
      _ccpn_substance.name               l-alanine
      _ccpn_substance.smiles             C[C@H](N)C(O)=O
      _ccpn_substance.inchi              InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1
      _ccpn_substance.comment            'It is an "alpha" acid; the (S) one.'
      _ccpn_substance.hmdb_accession     HMDB0000161
      _ccpn_substance.bmrb_accession     bmse000028
      _ccpn_substance.iupac_name         '(2S)-2-aminopropanoic acid'

      loop_
         _ccpn_substance_synonym.synonym

         '(S)-2-Aminopropanoic acid'
         Ala
      stop_
   save_

   save_ccpn_additional_data
      loop_
         _ccpn_internal_data.ccpn_object_pid
         _ccpn_internal_data.internal_data_string

         Project:default
;{
  "a": 1
}
;
      stop_
   save_
"""


@pytest.fixture(scope='module')
def entry():
    return read_entry(ALANINE)


def test_the_substance_is_read_with_its_identifiers(entry):
    assert entry.name == 'l-alanine'
    assert entry.hmdb_id == 'HMDB0000161' and entry.bmrb_id == 'bmse000028'
    assert entry.inchi.startswith('InChI=1S/C3H7NO2/')
    assert entry.iupac == '(2S)-2-aminopropanoic acid'
    assert entry.synonyms == ['(S)-2-Aminopropanoic acid', 'Ala']


def test_the_peak_list_keeps_origin_field_score_and_multiplets(entry):
    hmdb = next(s for s in entry.spectra if s.origin == 'HMDB')
    assert hmdb.kind == 'peak_list' and hmdb.field_mhz == 500.0
    assert hmdb.similarity == pytest.approx(0.9765002008)
    assert hmdb.source_file == 'SP-171.ndf5'
    assert len(hmdb.peaks) == 6
    doublet, quartet = hmdb.multiplets
    assert doublet['ppm'] == pytest.approx(1.4665)
    assert [p for p, _h in doublet['peaks']] == [1.459, 1.474]
    assert len(quartet['peaks']) == 4
    assert doublet['volume'] is None          # '.' is no value, not zero


def test_the_spin_matrix_splits_into_shifts_and_couplings(entry):
    g = next(s for s in entry.spectra if s.origin == 'GISSMO')
    assert g.kind == 'spin_system' and g.similarity is None
    assert g.spin_ppm == pytest.approx([1.4662, 1.4662, 1.4662, 3.7674])
    j = {(i, k): v for i, k, v in g.spin_j}
    assert j[(0, 3)] == pytest.approx(7.2988397)
    assert j[(0, 1)] == pytest.approx(-12.5)  # geminal, as GISSMO stores it
    assert len(j) == 6


def test_quoted_values_and_text_blocks_do_not_leak_into_items():
    frames = parse_frames(ALANINE)
    sub = frames['ccpn_substance_l-alanine.metabolite']['items']
    assert sub['comment'] == 'It is an "alpha" acid; the (S) one.'
    extra = frames['ccpn_additional_data']['loops']['ccpn_internal_data']
    assert extra[0]['internal_data_string'].startswith('{')
