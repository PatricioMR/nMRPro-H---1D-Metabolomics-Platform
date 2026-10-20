# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The Signal List's analyte column follows the quantification's result."""
from gui.stages._signal_labels import NO_LABEL, signal_labels


def _peak(rank, ppm):
    return {'rank': rank, 'ppm': ppm}


def _met(mid, name, *ranks, quality=None):
    return {'id': mid, 'name': name,
            'matched_peaks': [{'peak_rank': r, 'composite_score': 0.8,
                               'annotation_quality': quality} for r in ranks]}


def _row(mid, name, rank, *, lines=(), quantified=True):
    hit = {'peak_rank': rank,
           'predicted_lines': [{'ppm': p} for p in lines]}
    return {'hmdb_id': mid, 'metabolite': name, '_peak_rank': rank,
            '_hits': [hit], '_not_quantified': not quantified}


PEAKS = [_peak(1, 1.330), _peak(2, 1.318), _peak(3, 4.110), _peak(4, 3.030),
         _peak(5, 2.000)]


def test_before_quantification_the_annotation_names_the_signals():
    mets = [_met('L', 'Lactate', 1, 2, 3), _met('C', 'Creatine', 4)]
    lab = signal_labels(PEAKS, metabolites=mets)
    assert lab[1] == 'Lactate' and lab[4] == 'Creatine' and lab[5] == NO_LABEL


def test_an_analyte_the_quantification_turned_down_names_no_signal():
    mets = [_met('L', 'Lactate', 1, 2, 3), _met('C', 'Creatine', 4)]
    rows = [_row('L', 'Lactate', 1, lines=(1.330, 1.318)),
            _row('C', 'Creatine', 4, quantified=False)]
    lab = signal_labels(PEAKS, quant_rows=rows, metabolites=mets)
    assert lab[4] == NO_LABEL


def test_the_measured_multiplet_and_the_other_signals_of_a_quantified_analyte():
    mets = [_met('L', 'Lactate', 1, 2, 3)]
    rows = [_row('L', 'Lactate', 1, lines=(1.330, 1.318))]
    lab = signal_labels(PEAKS, quant_rows=rows, metabolites=mets)
    # the doublet it was measured on, and its quartet matched by annotation
    assert lab[1] == lab[2] == lab[3] == 'Lactate'


def test_the_measured_signal_wins_over_the_annotation_choice():
    # annotation gives peak 4 to a confirmed Creatinine, quantification
    # measured Creatine there and turned Creatinine down
    mets = [_met('C', 'Creatine', 4), _met('K', 'Creatinine', 4, quality='good')]
    rows = [_row('C', 'Creatine', 4), _row('K', 'Creatinine', 4, quantified=False)]
    lab = signal_labels(PEAKS, quant_rows=rows, metabolites=mets)
    assert lab[4] == 'Creatine'


def test_two_analytes_measured_on_one_signal_are_both_named():
    rows = [_row('A', 'Alpha', 5), _row('B', 'Beta', 5)]
    lab = signal_labels(PEAKS, quant_rows=rows, metabolites=[])
    assert lab[5] == 'Alpha / Beta'


def test_a_name_left_on_a_signal_does_not_outlive_the_quantification():
    peaks = [dict(p) for p in PEAKS]
    peaks[3]['assigned_metabolite'] = 'Creatinine'
    rows = [_row('L', 'Lactate', 1)]
    lab = signal_labels(peaks, quant_rows=rows, metabolites=[])
    assert lab[4] == NO_LABEL
    assert signal_labels(peaks, metabolites=[])[4] == 'Creatinine'


def test_the_internal_standard_is_not_an_analyte():
    rows = [dict(_row('T', 'TSP', 5), _is_is=True)]
    lab = signal_labels(PEAKS, quant_rows=rows, metabolites=[_met('T', 'TSP', 5)])
    assert lab[5] == 'TSP'   # no quantified analyte: the annotation stands


def test_an_analyte_the_cascade_took_in_names_the_window_it_was_measured_on():
    rows = [{'hmdb_id': 'F', 'metabolite': 'Formate', '_peak_rank': None,
             '_hits': [None], '_by_cascade': True,
             '_multiplet_window': [1.99, 2.01], '_not_quantified': False}]
    lab = signal_labels(PEAKS, quant_rows=rows, metabolites=[])
    assert lab[5] == 'Formate' and lab[4] == NO_LABEL
