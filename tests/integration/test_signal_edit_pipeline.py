"""A signal split by hand in the Signal List reaches quantification like a
detected one: the two lines of a doublet that detection left in one window,
cut apart, measure the concentrations that were built in."""
import numpy as np
import pytest

from gui.stages._quant_adapter import quantify_project
from processing.peak_detection import as_app_peak
from processing.signal_edit import measure_window, split_signal, valley_near

MHZ = 600.0
PPM = np.linspace(-0.5, 10.0, 21001)
SCALE = [1.0, 2.0, 3.0]


def _lorentz(centre, height, hwhm=0.002):
    return height * hwhm ** 2 / ((PPM - centre) ** 2 + hwhm ** 2)


def _idx(ppm):
    return int(np.argmin(np.abs(PPM - ppm)))


def _signal(ppm_db, rank, line_ranks):
    return {'ppm': ppm_db, 'type': 'd', 'protons': 3.0, 'j_hz': 7.2,
            'delta_j_hz': 0.05, 'ppm_diff': 0.001, 'peak_rank': rank,
            'annotation_quality': 'good', '_n_found_lines': 2,
            '_n_exp_lines': 2, 'composite_score': 0.9,
            'predicted_lines': [{'ppm': p} for p in line_ranks.values()],
            'matched_line_ranks': list(line_ranks)}


@pytest.fixture
def split_study():
    d = 7.2 / MHZ / 2
    X = np.column_stack([_lorentz(1.480 - d, 100.0 * k)
                         + _lorentz(1.480 + d, 100.0 * k)
                         + _lorentz(0.0, 300.0) for k in SCALE])
    merged = as_app_peak(dict(measure_window(PPM, X, X, _idx(1.462),
                                             _idx(1.498), 1.0), rank=1))
    tsp = as_app_peak(dict(measure_window(PPM, X, X, _idx(-0.01),
                                          _idx(0.01), 1.0), rank=9))
    peaks = [merged, tsp]
    cut = valley_near(X.mean(axis=1), _idx(1.480), *sorted(
        (merged['start_idx'], merged['end_idx'])))
    piece = split_signal(peaks, merged, cut, PPM, X, X, 1.0)
    return peaks, merged, piece, tsp, X


def _quantify(X, peaks, rank_a, rank_b, ref):
    lines = {p['rank']: p['ppm'] for p in peaks if p['rank'] in (rank_a, rank_b)}
    metabolites = [{'id': 'HMDB0000161', 'name': 'L-Alanine',
                    'matched_peaks': [_signal(1.480, rank_a, lines)]}]
    rows = quantify_project(
        metabolites=metabolites, peaks=peaks, ppm=PPM, intensities=X,
        samples_info=[{'label': 'A', 'type': 'Sample'}] * 3,
        ref_peak=ref, ref_nh=9.0, ref_conc=1.0, ref_name='TSP',
        dj_gold_hz=0.5, use_pqn=False)
    return next(r for r in rows if r['hmdb_id'] == 'HMDB0000161')


def test_a_split_doublet_is_quantified_from_its_two_pieces(split_study):
    peaks, merged, piece, tsp, X = split_study
    assert piece['rank'] == 10
    row = _quantify(X, peaks, merged['rank'], piece['rank'], tsp)
    assert not row['_not_quantified'], row['_status_reasons']
    c = np.asarray(row['_concs_per_sample'], dtype=float)
    assert c / c[0] == pytest.approx(SCALE, rel=0.02)


def test_split_pieces_give_what_two_detected_windows_give(split_study):
    """Same windows, one pair cut by hand and one as detection would hand
    them over: the engine cannot tell them apart."""
    peaks, merged, piece, tsp, X = split_study
    by_hand = _quantify(X, peaks, merged['rank'], piece['rank'], tsp)
    detected = [as_app_peak(dict(measure_window(PPM, X, X, p['start_idx'],
                                                p['end_idx'], 1.0), rank=r))
                for p, r in ((merged, 1), (piece, 2))] + [tsp]
    by_detection = _quantify(X, detected, 1, 2, tsp)
    assert by_hand['_concs_per_sample'] == pytest.approx(
        by_detection['_concs_per_sample'], rel=1e-9)
