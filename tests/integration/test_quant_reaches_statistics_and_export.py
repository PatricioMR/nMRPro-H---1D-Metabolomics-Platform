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
    assert X.shape == (len(samples), len(features))
    assert set(features) >= {'L-Alanine', 'L-Lactic acid'}
    assert names == ['B1', 'S1', 'S2', 'S3']
    assert groups == ['BLK', 'A', 'A', 'B']


def test_the_matrix_carries_the_concentrations_in_sample_order():
    """Built at 1x, 2x and 3x, so the column must rise in that order."""
    from gui.stages.statistics_stage import _get_quantification_matrix as build_feature_matrix
    rows, samples = _rows()
    X, _, features, _ = build_feature_matrix(_FakeApp(rows, samples))
    col = X[:, features.index('L-Alanine')]
    assert np.isnan(col[0])
    assert col[2] / col[1] == pytest.approx(2.0, rel=0.05)
    assert col[3] / col[1] == pytest.approx(3.0, rel=0.05)


def test_rows_with_no_number_stay_out_of_the_matrix():
    """A censored analyte has a bound, not a measurement, and a bound in a
    statistical model is a fabricated value."""
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


def test_a_sample_the_analyte_was_not_seen_in_is_marked_not_detected():
    """The CSV writes 'n.d.' there rather than a number. A window over noise
    integrates to a perfectly plausible concentration, and writing it would put
    a measurement where the analyte was not seen."""
    rows, samples = _rows()
    row = next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')
    det = row['_detected_per_sample']
    assert len(det) == len(samples)
    assert det[0] is not True, 'the blank holds nothing'
    assert all(d is True for d in det[1:])


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
