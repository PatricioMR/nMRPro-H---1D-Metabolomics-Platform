"""A metabolite's signals stand in the proportion of their protons; the anchor
of a signal is the nearest peak among those its proportions allow."""
from processing.annotation_consistency import (
    consistent_peaks, reference_per_proton)

PEAKS = [{'rank': 1, 'ppm': 3.928, 'snr': 189.0},     # creatine CH2, 2H
         {'rank': 2, 'ppm': 3.022, 'snr': 8.8},       # noise near 3.024
         {'rank': 3, 'ppm': 3.044, 'snr': 268.0},     # creatine CH3, 3H
         {'rank': 4, 'ppm': 1.470, 'snr': 90.0}]
CRE = [{'hmdb_id': 'CRE', 'ppm_center': 3.024, 'protons': 3},
       {'hmdb_id': 'CRE', 'ppm_center': 3.920, 'protons': 2}]


def _key(e):
    return e['hmdb_id']


def test_the_reference_is_the_tallest_peak_per_proton_in_reach():
    ref = reference_per_proton(CRE, PEAKS, tolerance=0.05, met_key=_key)
    assert ref['CRE'] == max(189.0 / 2, 268.0 / 3)


def test_the_noise_peak_is_not_the_methyl_singlet():
    ref = reference_per_proton(CRE, PEAKS, tolerance=0.05, met_key=_key)
    in_tol = [p for p in PEAKS if abs(p['ppm'] - 3.024) <= 0.05]
    kept = consistent_peaks(in_tol, CRE[0], ref['CRE'])
    assert [p['rank'] for p in kept] == [3]


def test_without_a_reference_or_protons_every_peak_stays():
    in_tol = [p for p in PEAKS if abs(p['ppm'] - 3.024) <= 0.05]
    assert consistent_peaks(in_tol, CRE[0], None) == in_tol
    assert consistent_peaks(in_tol, {'protons': None}, 90.0) == in_tol
    assert consistent_peaks([], CRE[0], 90.0) == []


def test_when_no_peak_is_consistent_all_stay_as_before():
    in_tol = [p for p in PEAKS if abs(p['ppm'] - 3.024) <= 0.05]
    assert consistent_peaks(in_tol, CRE[0], 10000.0) == in_tol


def test_overlap_makes_a_peak_taller_never_shorter_so_taller_passes():
    small = {'rank': 9, 'ppm': 3.03, 'snr': 8.0}
    big = {'rank': 3, 'ppm': 3.044, 'snr': 268.0}
    kept = consistent_peaks([small, big], {'protons': 1}, 30.0)
    assert small in kept and big in kept
    kept = consistent_peaks([small, big], {'protons': 3}, 30.0)
    assert small not in kept and big in kept


def test_signals_without_protons_or_peaks_take_no_part_in_the_reference():
    entries = [{'hmdb_id': 'X', 'ppm_center': 3.024, 'protons': None},
               {'hmdb_id': 'X', 'ppm_center': 8.0, 'protons': 1}]
    assert reference_per_proton(entries, PEAKS, tolerance=0.05, met_key=_key) == {}
