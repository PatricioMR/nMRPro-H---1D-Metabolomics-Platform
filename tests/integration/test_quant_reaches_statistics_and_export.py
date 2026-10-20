# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""What quantification produces has to be readable by what comes after it.

The engine returns its per-sample values as ``{sample_index: value}`` — a
sample it could not measure is absent rather than present with a zero.
Everything downstream indexes them by position instead, and has since before
the engine was rewritten: the statistics matrix, the CSV export, the manual
region editor, the per-sample peak view, the lipidomics path.

Handing a dict to those readers is worse than a crash. ``list(a_dict)`` yields
its KEYS, so the export would have written 0, 1, 2 in the concentration column
and nobody would have seen anything wrong with the file. The statistics stage
would at least have raised, on ``concs[:n_s]``.
"""
import numpy as np
import pytest

from gui.stages._quant_adapter import quantify_project

MHZ = 600.0
PPM = np.linspace(0.0, 10.0, 20001)


def _lorentz(centre, height, hwhm=0.002):
    return height * hwhm ** 2 / ((PPM - centre) ** 2 + hwhm ** 2)


def _peak(rank, centre, width=0.004):
    i = int(np.argmin(np.abs(PPM - centre)))
    return {'rank': rank, 'ppm': float(PPM[i]),
            'start_ppm': centre - width, 'end_ppm': centre + width,
            'start_idx': i - 20, 'end_idx': i + 20,
            'width_ppm': width, 'noise': 1.0, 'snr': 100.0}


def _signal(ppm_db, rank, line_ppms):
    return {'ppm': ppm_db, 'type': 'd', 'protons': 3.0, 'j_hz': 7.2,
            'delta_j_hz': 0.05, 'ppm_diff': 0.001, 'peak_rank': rank,
            'annotation_quality': 'good', '_n_found_lines': len(line_ppms),
            '_n_exp_lines': len(line_ppms), 'composite_score': 0.9,
            'predicted_lines': [{'ppm': p} for p in line_ppms]}


def _study():
    """Two analytes, three analytical samples at 1x/2x/3x, and one blank."""
    cols = [np.zeros_like(PPM)]                       # the blank
    for k in (1.0, 2.0, 3.0):
        cols.append(_lorentz(1.474, 100.0 * k) + _lorentz(1.486, 100.0 * k)
                    + _lorentz(1.324, 60.0 * k) + _lorentz(1.336, 60.0 * k)
                    + _lorentz(0.0, 300.0))
    peaks = [_peak(1, 1.474), _peak(2, 1.486), _peak(3, 1.324),
             _peak(4, 1.336), _peak(9, 0.0)]
    mets = [
        {'id': 'HMDB0000161', 'name': 'L-Alanine',
         'matched_peaks': [_signal(1.480, 1, [1.474, 1.486])]},
        {'id': 'HMDB0000190', 'name': 'L-Lactic acid',
         'matched_peaks': [_signal(1.330, 3, [1.324, 1.336])]},
    ]
    samples = [{'type': 'BLK', 'label': 'BLK', 'name': 'B1'},
               {'label': 'A', 'name': 'S1'}, {'label': 'A', 'name': 'S2'},
               {'label': 'B', 'name': 'S3'}]
    return mets, peaks, np.column_stack(cols), samples, peaks[-1]


def _rows():
    mets, peaks, X, samples, ref = _study()
    return quantify_project(
        metabolites=mets, peaks=peaks, ppm=PPM, intensities=X,
        samples_info=samples, ref_peak=ref, ref_nh=9.0, ref_conc=1.0,
        dj_gold_hz=0.5, use_pqn=False), samples


# ── The shape every reader after quantification assumes ──────────────────────

@pytest.mark.parametrize('field', ['_concs_per_sample', '_areas_per_sample',
                                   '_snr_per_sample'])
def test_the_per_sample_fields_are_indexed_by_sample(field):
    rows, samples = _rows()
    for r in rows:
        assert isinstance(r[field], list), f'{field} must be indexable'
        assert len(r[field]) == len(samples), (
            f'{field} must have one entry per sample, blanks included, so '
            f'position means sample')


def test_the_statistics_matrix_leaves_the_blanks_and_switched_off_samples_out():
    """The rule detection, annotation and quantification follow
    (processing.sample_selection): a blank's n.d. would otherwise count
    against every metabolite in the missing-value filter."""
    from gui.stages.statistics_stage import _get_quantification_matrix as build
    rows, samples = _rows()
    samples = [dict(s) for s in samples]
    samples[-1]['excluded'] = True
    X, names, _f, groups = build(_FakeApp(rows, samples))
    blanks = [s.get('name') for s in samples if str(s.get('type')).upper() == 'BLK']
    assert blanks, 'the fixture holds a blank'
    assert not set(blanks) & set(names)
    assert samples[-1].get('name') not in names
    assert X.shape[0] == len(names) == len(groups)


def test_a_sample_that_was_not_measured_is_nan_and_not_zero():
    """A zero would enter a group mean as a real measurement of nothing."""
    rows, _ = _rows()
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    assert np.isnan(row['_concs_per_sample'][0]), 'the blank'
    assert all(np.isfinite(v) for v in row['_concs_per_sample'][1:])


def test_reading_it_as_a_list_gives_values_and_not_indices():
    """The failure that would not have announced itself: list() over a dict
    yields its keys, so the export column would have read 0, 1, 2."""
    rows, _ = _rows()
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    got = [v for v in row['_concs_per_sample'] if np.isfinite(v)]
    assert got and not all(float(v).is_integer() and v < 10 for v in got)


# ── The statistics matrix ────────────────────────────────────────────────────

class _FakeStage:
    def __init__(self, rows):
        self._quant_results = rows


class _FakeApp:
    def __init__(self, rows, samples):
        self.stages = {'quantification': _FakeStage(rows), 'annotation': None}
        self.samples_info = samples
        self.ppm = PPM


def test_the_statistics_matrix_is_built_from_the_quantified_rows():
    from gui.stages.statistics_stage import _get_quantification_matrix as build_feature_matrix
    rows, samples = _rows()
    X, names, features, groups = build_feature_matrix(_FakeApp(rows, samples))
    assert X is not None, 'quantification produced nothing statistics could read'
    # the blank is not a sample of the study (processing.sample_selection)
    assert X.shape == (len(samples) - 1, len(features))
    assert set(features) >= {'L-Alanine', 'L-Lactic acid'}
    assert names == ['S1', 'S2', 'S3']
    assert groups == ['A', 'A', 'B']


def test_the_matrix_carries_the_concentrations_in_sample_order():
    """Built at 1x, 2x and 3x, so the column must rise in that order."""
    from gui.stages.statistics_stage import _get_quantification_matrix as build_feature_matrix
    rows, samples = _rows()
    X, _, features, _ = build_feature_matrix(_FakeApp(rows, samples))
    col = X[:, features.index('L-Alanine')]
    # the blank is not in the matrix: rows are the 1x, 2x and 3x samples
    assert col[1] / col[0] == pytest.approx(2.0, rel=0.05)
    assert col[2] / col[0] == pytest.approx(3.0, rel=0.05)


def test_rows_with_no_number_stay_out_of_the_matrix():
    """A row the engine did not quantify has no measurement, and a number
    for it in a statistical model is a fabricated value."""
    from gui.stages.statistics_stage import _get_quantification_matrix as build_feature_matrix
    rows, samples = _rows()
    rows.append({'metabolite': 'Ghost', '_not_quantified': True,
                 '_concs_per_sample': [1.0] * len(samples)})
    _, _, features, _ = build_feature_matrix(_FakeApp(rows, samples))
    assert 'Ghost' not in features


# ── The export ───────────────────────────────────────────────────────────────

def test_the_export_reads_concentrations_and_not_sample_indices():
    """The failure that would not have announced itself. The VIP path does
    ``list(r['_concs_per_sample'])``, and over a dict that yields its keys —
    the scores would have been computed on 0, 1, 2."""
    rows, samples = _rows()
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    as_list = list(row['_concs_per_sample'])
    assert as_list == row['_concs_per_sample']
    assert as_list != list(range(len(samples)))


def test_the_blank_is_not_a_sample_of_the_study():
    """It holds the buffer, not the analyte: no number in the table."""
    from gui.stages._quant_adapter import reported_cells
    rows, samples = _rows()
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    cells = reported_cells(row, len(samples))
    assert np.isnan(cells[0][0]), 'the blank holds nothing'
    assert all(k == 'measured' for _v, k in cells[1:])


def test_every_sample_of_the_study_has_a_number_for_a_quantified_analyte():
    """The study proved the analyte, so no sample of it reads 'n.d.': the
    engine's amount where it has one, a ceiling where it has none. The
    blank is not in the study and is not measured."""
    from gui.stages._quant_adapter import reported_cells
    rows, samples = _rows()
    for r in rows:
        if r.get('_not_quantified') or r.get('_is_is') or r.get('_is_lipid_index'):
            continue
        assert '_ceiling_per_sample' in r and r.get('_included'), r['metabolite']
        cells = reported_cells(r, len(samples))
        for i in r['_included']:
            value, kind = cells[i]
            assert value > 0 and kind in ('measured', 'ceiling'), (
                r['metabolite'], i, cells[i])


def test_the_internal_standard_is_not_a_variable_in_the_model():
    """It is what every concentration was divided by: constant by
    construction, carrying no biology, and it would sit in the model as a
    variable that cannot vary."""
    from gui.stages.statistics_stage import (
        _get_quantification_matrix as build_feature_matrix)
    rows, samples = _rows()
    assert any(r.get('_is_is') for r in rows), 'the fixture has a standard'
    _, _, features, _ = build_feature_matrix(_FakeApp(rows, samples))
    assert 'Internal standard' not in features


# ── Not detected is NaN: the statistics and both exports ─────────────────────

def _with_not_detected(rows):
    """The first analyte with a zero in S3 (index 3)."""
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    row['_concs_per_sample'] = [float('nan'), 1.0, 2.0, 0.0]
    row['_confidence'] = 1
    row['_by_cascade'] = True
    row['_identity_why'] = 'the cascade took it in'
    return row


def test_a_not_detected_sample_is_nan_in_the_statistics_matrix():
    from gui.stages.statistics_stage import _get_quantification_matrix as build
    rows, samples = _rows()
    _with_not_detected(rows)
    X, _, features, _ = build(_FakeApp(rows, samples))
    col = X[:, features.index('L-Alanine')]      # blank left out: sample i is row i-1
    assert col[0] == 1.0 and col[1] == 2.0
    assert np.isnan(col[2]), 'a ceiling is not an amount'
    # asked to keep the table's numbers, the statistics read them all
    from gui.stages._quant_adapter import statistics_values
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    kept = statistics_values(row, 4, censored=False)
    assert kept[2] == 2.0 and kept[3] > 0, 'a ceiling is a number in the table'


def test_a_low_confidence_row_still_reaches_the_statistics():
    """The analyst reads the confidence; the program does not drop the row."""
    from gui.stages.statistics_stage import _get_quantification_matrix as build
    rows, samples = _rows()
    _with_not_detected(rows)
    _, _, features, _ = build(_FakeApp(rows, samples))
    assert 'L-Alanine' in features


class _FakeStats:
    def __init__(self):
        self.statuses = 0

    def _update_filter_status(self):
        self.statuses += 1


def test_the_multivariate_methods_follow_metaboanalyst():
    """An n.d. becomes 1/5 of its metabolite's minimum; a metabolite with
    n.d. in 20 % of the samples or more is left out; never the mean."""
    from gui.stages.statistics_stage import StatisticsStage
    X = np.array([[1.0, 5.0, np.nan], [2.0, np.nan, 7.0], [3.0, 7.0, np.nan],
                  [4.0, 8.0, 9.0], [5.0, 9.0, 9.5], [6.0, 10.0, 10.0]])
    fake = _FakeStats()
    out, labels = StatisticsStage._remove_nan(fake, X, ['a', 'b', 'c'],
                                              ['g1'] * 3 + ['g2'] * 3)
    # the application's share is 50 %: b (1 of 6 n.d.) and c (2 of 6) stay
    assert labels == ['a', 'b', 'c'] and fake._nan_dropped == []
    assert out[1, 1] == pytest.approx(5.0 / 5)
    assert out[0, 2] == pytest.approx(7.0 / 5)
    assert fake._nd_imputed == 3
    # at MetaboAnalyst's own 20 %, c (33 %) goes
    fake = _FakeStats()
    fake._missing_pct_var = _Var(20.0)
    out, labels = StatisticsStage._remove_nan(fake, X, ['a', 'b', 'c'],
                                              ['g1'] * 3 + ['g2'] * 3)
    assert labels == ['a', 'b'] and fake._nan_dropped == ['c']


def test_counted_per_group_a_metabolite_one_group_holds_stays():
    from gui.stages.statistics_stage import StatisticsStage
    import tkinter as tk
    X = np.array([[np.nan], [np.nan], [np.nan], [3.0], [4.0], [5.0]])
    fake = _FakeStats()
    fake._missing_grp_var = _Var(True)
    out, labels = StatisticsStage._remove_nan(fake, X, ['x'],
                                              ['ctrl'] * 3 + ['lps'] * 3)
    assert labels == ['x'] and out[0, 0] == pytest.approx(3.0 / 5)


def test_exclude_leaves_out_every_metabolite_with_an_nd_and_fills_nothing():
    from gui.stages.statistics_stage import StatisticsStage
    X = np.array([[1.0, 5.0, np.nan], [2.0, 6.0, 7.0], [3.0, 7.0, 8.0]])
    fake = _FakeStats()
    fake._missing_var = _Var('exclude')
    out, labels = StatisticsStage._remove_nan(fake, X, ['a', 'b', 'c'])
    assert labels == ['a', 'b'] and fake._nan_dropped == ['c']
    assert np.array_equal(out, X[:, :2])


class _Var:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


def test_the_per_feature_methods_keep_the_nan_and_drop_only_the_empty():
    from gui.stages.statistics_stage import StatisticsStage
    X = np.array([[1.0, np.nan, np.nan], [2.0, 6.0, np.nan], [3.0, 7.0, np.nan]])
    fake = _FakeStats()
    out, labels = StatisticsStage._remove_nan(fake, X, ['a', 'b', 'c'],
                                              complete=False)
    assert labels == ['a', 'b'] and np.isnan(out[0, 1])
    assert fake._nan_dropped == []


def _export_to(tmp_path, mode, rows, samples, monkeypatch):
    from gui.stages import _quant_computation as qc

    class _Stage(qc._QuantCompMixin):
        def __init__(self):
            self._quant_results = rows
            self.app = _FakeApp(rows, samples)
            self.app.all_intensities = np.zeros((10, len(samples)))
            self._last_norm_mode = 'IS'
            self.ok = None

        def _status_ok(self, msg):
            self.ok = msg

    path = str(tmp_path / ('out.csv'))
    monkeypatch.setattr(qc.filedialog, 'asksaveasfilename', lambda **k: path)
    shown = []
    monkeypatch.setattr(qc.messagebox, 'showinfo', lambda *a, **k: shown.append(a))
    monkeypatch.setattr(qc.messagebox, 'showerror',
                        lambda *a, **k: shown.append(('ERROR',) + a))
    _Stage()._export(mode)
    assert not any(s[0] == 'ERROR' for s in shown), shown
    return path


def _read(path):
    import csv
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def test_the_csv_marks_not_detected_and_carries_the_confidence(tmp_path, monkeypatch):
    rows, samples = _rows()
    _with_not_detected(rows)
    table = _read(_export_to(tmp_path, 'csv', rows, samples, monkeypatch))
    head = table[0]
    for col in ('Confidence', 'Origin', 'IdentityBasis'):
        assert col in head
    ala = next(r for r in table if r[0] == 'L-Alanine')
    assert ala[head.index('Confidence')] == '1'
    assert ala[head.index('Origin')] == 'Cascade estimate'
    # the four sample columns are the last four: the blank is not measured, S1 and
    # S2 carry their numbers, S3's zero is a ceiling, '<value': the most the
    # sample can hold
    assert ala[-4] == 'NA' and ala[-3] == '1.0000' and ala[-2] == '2.0000'
    assert ala[-1].startswith('<') and float(ala[-1][1:]) > 0
    assert ala[head.index('Ceiling_samples')] == 'S3'
    assert not any('LOD' in h for h in head)


def test_metaboanalyst_gets_na_for_not_detected_and_the_confidence_beside_it(
        tmp_path, monkeypatch):
    rows, samples = _rows()
    _with_not_detected(rows)
    path = _export_to(tmp_path, 'metaboanalyst', rows, samples, monkeypatch)
    table = _read(path)
    assert table[0] == ['Sample', 'S1', 'S2', 'S3'], 'the blank is not a column'
    assert table[1] == ['Label', 'A', 'A', 'B']
    ala = next(r for r in table[2:] if r[0] == 'L-Alanine')
    # a model sees measurements only: a ceiling is NA
    assert ala[1] != 'NA' and ala[2] != 'NA' and ala[3] == 'NA'
    ann = _read(str(tmp_path / 'out_annotations.csv'))
    assert ann[0][:3] == ['Feature', 'HMDB_ID', 'Confidence']
    head = ann[0]
    arow = next(r for r in ann if r[0] == 'L-Alanine')
    assert arow[2] == '1' and arow[3] == 'Cascade estimate'
    assert arow[head.index('N_measured')] == '2'
    assert not any('LOD' in h for h in head)
    assert arow[head.index('N_ceiling')] == '1'
    assert arow[head.index('Ceiling_samples')] == 'S3'


def test_metaboanalyst_leaves_out_a_sample_the_analyst_excluded(tmp_path, monkeypatch):
    rows, samples = _rows()
    samples[2] = dict(samples[2], excluded=True)
    table = _read(_export_to(tmp_path, 'metaboanalyst', rows, samples, monkeypatch))
    assert table[0] == ['Sample', 'S1', 'S3']


def test_metaboanalyst_names_are_unique_and_a_feature_never_detected_is_left_out(
        tmp_path, monkeypatch):
    rows, samples = _rows()
    dup = dict(next(r for r in rows if r['hmdb_id'] == 'HMDB0000190'),
               hmdb_id='HMDB9999999')
    ghost = dict(dup, metabolite='Ghost', hmdb_id='HMDB1',
                 _concs_per_sample=[float('nan')] * len(samples))
    rows += [dup, ghost]
    table = _read(_export_to(tmp_path, 'metaboanalyst', rows, samples, monkeypatch))
    names = [r[0] for r in table[2:]]
    assert len(names) == len(set(names)), names
    assert 'Ghost' not in names
    assert any(n.startswith('L-Lactic acid [') for n in names)


# ── The analyst sets the window of one sample ────────────────────────────────

def _editing_stage(rows, samples):
    """The quantification stage's own mixin, on the study's rows."""
    import types
    from gui.stages import _quant_computation as qc

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stage(qc._QuantCompMixin):
        pass

    st = _Stage()
    st._quant_results = rows
    st.app = types.SimpleNamespace(
        ppm=PPM, all_intensities=_study()[2],
        corrected_intensities=lambda: _study()[2])
    st.bl_model_quant_var = _V(False)
    st._last_run_settings = {'bl_model_quant_var': False}
    st._last_groups = {'A': [1, 2], 'B': [3]}
    st._last_blk_indices = [0]
    st._right_widget = None
    st.peak_stage = types.SimpleNamespace(peaks=[])
    return st


def test_the_engines_own_window_gives_the_engines_number_back():
    rows, samples = _rows()
    st = _editing_stage(rows, samples)
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    before = list(row['_concs_per_sample'])
    peak = {'rank': row['_peak_rank']}
    assert st.update_sample_area(peak, 2, *row['_multiplet_window']) is None
    assert row['_concs_per_sample'] == before
    assert not row['_manual_regions']


def test_a_narrower_window_scales_the_engines_number_by_the_change_in_the_integral():
    rows, samples = _rows()
    st = _editing_stage(rows, samples)
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    engine = list(row['_concs_per_sample'])
    lo, hi = sorted(row['_multiplet_window'])
    narrow = (lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo))
    peak = {'rank': row['_peak_rank']}
    assert st.update_sample_area(peak, 2, *narrow) is None
    i_old = st._window_integral(row, 2, lo, hi)
    i_new = st._window_integral(row, 2, *narrow)
    assert row['_concs_per_sample'][2] == pytest.approx(engine[2] * i_new / i_old)
    assert row['_concs_per_sample'][2] < engine[2]
    # the other samples are the engine's, and so is the row's identity
    assert row['_concs_per_sample'][1] == engine[1]
    assert row['_manual_regions'] == {2: pytest.approx(narrow)}
    assert row['_status_reasons'][0].startswith('Re-measured by hand in 1 sample')


def test_the_edit_is_never_compounded_and_reset_gives_the_engine_back():
    rows, samples = _rows()
    st = _editing_stage(rows, samples)
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    engine = list(row['_concs_per_sample'])
    lo, hi = sorted(row['_multiplet_window'])
    peak = {'rank': row['_peak_rank']}
    st.update_sample_area(peak, 2, lo + 0.2 * (hi - lo), hi)
    once = row['_concs_per_sample'][2]
    st.update_sample_area(peak, 2, lo + 0.2 * (hi - lo), hi)
    assert row['_concs_per_sample'][2] == pytest.approx(once), 'same edit twice'
    st.reset_sample_area(peak, 2)
    assert row['_concs_per_sample'] == pytest.approx(engine, nan_ok=True)
    assert not any(t.startswith('Re-measured') for t in row['_status_reasons'])


def test_the_group_statistics_follow_the_edited_sample():
    rows, samples = _rows()
    st = _editing_stage(rows, samples)
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    lo, hi = sorted(row['_multiplet_window'])
    st.update_sample_area({'rank': row['_peak_rank']}, 2,
                          lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo))
    assert row['_concs_by_group']['A'] == pytest.approx(
        [row['_concs_per_sample'][1], row['_concs_per_sample'][2]])
    assert row['conc_mm'] == pytest.approx(
        sum(row['_concs_per_sample'][1:]) / 3)


def test_a_number_from_a_fit_has_no_window_to_move():
    rows, samples = _rows()
    st = _editing_stage(rows, samples)
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    row['_by_cascade'] = True
    before = list(row['_concs_per_sample'])
    peak = {'rank': row['_peak_rank']}
    ok, why = st.sample_area_editable(peak)
    assert not ok and 'fit' in why
    assert st.update_sample_area(peak, 2, 1.40, 1.50) == why
    assert row['_concs_per_sample'] == before


def test_the_windows_the_analyst_set_survive_a_new_run():
    rows, samples = _rows()
    st = _editing_stage(rows, samples)
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    lo, hi = sorted(row['_multiplet_window'])
    narrow = (lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo))
    peak = {'rank': row['_peak_rank'], 'sample_regions': {
        2: {'start_ppm': narrow[0], 'end_ppm': narrow[1]}}}
    st.peak_stage.peaks = [peak]
    edited = None
    st.update_sample_area(peak, 2, *narrow)
    edited = row['_concs_per_sample'][2]
    # a fresh run: the engine's numbers again, then the windows on top
    fresh, _ = _rows()
    st._quant_results = fresh
    assert st._reapply_manual_regions() == []
    row2 = next(r for r in fresh if r['hmdb_id'] == 'HMDB0000161')
    assert row2['_concs_per_sample'][2] == pytest.approx(edited)
