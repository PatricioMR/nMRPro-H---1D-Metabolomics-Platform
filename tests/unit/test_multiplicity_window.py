"""Multiplicity integration window = union of the multiplet peaks' Peak-List
boundaries (lateral limits of all lines joined into one)."""
from gui.stages._quant_computation import _multiplet_window_from_peaks


def _pk(rank, ppm, lo, hi):
    return {'rank': rank, 'ppm': ppm, 'start_ppm': lo, 'end_ppm': hi,
            'width_ppm': abs(hi - lo)}


def test_doublet_unions_both_peak_boundaries():
    # Doublet: two detected peaks at 1.330 and 1.342, each ~0.004 ppm wide.
    pk_by_rank = {
        1: _pk(1, 1.330, 1.332, 1.328),   # left line (ppm decreasing within)
        2: _pk(2, 1.342, 1.344, 1.340),   # right line
        9: _pk(9, 3.500, 3.502, 3.498),   # unrelated peak elsewhere
    }
    hit = {'peak_rank': 1,
           'predicted_lines': [{'ppm': 1.330}, {'ppm': 1.342}]}
    lo, hi = _multiplet_window_from_peaks(hit, pk_by_rank)
    # Union spans the outer feet of BOTH doublet lines, not the unrelated peak.
    assert abs(lo - 1.328) < 1e-6, lo
    assert abs(hi - 1.344) < 1e-6, hi


def test_anchor_centre_stray_peak_is_excluded():
    """The hit anchor is the multiplet CENTRE, landing on a stray peak (461)
    between the doublet lines. The window must use the line peaks (49/51), not
    the stray centre."""
    from gui.stages._quant_computation import _multiplet_line_peaks
    pk_by_rank = {
        49:  _pk(49, 1.330, 1.332, 1.328),
        51:  _pk(51, 1.342, 1.344, 1.340),
        461: _pk(461, 1.336, 1.337, 1.335),   # stray peak at the doublet centre
    }
    hit = {'peak_rank': 461,                    # anchor = centre = stray
           'predicted_lines': [{'ppm': 1.330}, {'ppm': 1.342}]}
    ranks = sorted(p['rank'] for p in _multiplet_line_peaks(hit, pk_by_rank))
    assert ranks == [49, 51], ranks            # 461 excluded
    lo, hi = _multiplet_window_from_peaks(hit, pk_by_rank)
    assert abs(lo - 1.328) < 1e-6 and abs(hi - 1.344) < 1e-6


def test_singlet_collapses_to_its_peak():
    pk_by_rank = {1: _pk(1, 3.030, 3.033, 3.027)}
    hit = {'peak_rank': 1, 'predicted_lines': [{'ppm': 3.030}]}
    lo, hi = _multiplet_window_from_peaks(hit, pk_by_rank)
    assert abs(lo - 3.027) < 1e-6 and abs(hi - 3.033) < 1e-6


def test_undetected_line_is_skipped():
    # Second predicted line has no detected peak nearby → window = first peak only.
    pk_by_rank = {1: _pk(1, 1.330, 1.332, 1.328)}
    hit = {'peak_rank': 1,
           'predicted_lines': [{'ppm': 1.330}, {'ppm': 1.900}]}
    lo, hi = _multiplet_window_from_peaks(hit, pk_by_rank)
    assert abs(lo - 1.328) < 1e-6 and abs(hi - 1.332) < 1e-6
