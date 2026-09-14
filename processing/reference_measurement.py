# processing/reference_measurement.py
"""Measure a reference compound's signals from its own spectrum.

The reference databases publish peak lists to three decimal places. At 500 MHz
one thousandth of a ppm is half a hertz, so a coupling read off two such
positions carries plus or minus half a hertz before anything is measured, and
HMDB rounds its couplings to whole hertz besides. Between BMRB and HMDB, for
the 115 metabolites both hold, the couplings disagree by 0.80 Hz at the median
-- four times the gate the annotation judges a match by. L-lactic acid is
recorded at 7.50 Hz in one and the line is 6.93; N-acetyl-L-alanine at 7.20 sits
nearer the truth than lactate's own entry, and wins lactate's peak.

Both databases also hold the raw spectra the lists were read from. This module
measures from those: where each line actually peaks, refined off the grid; the
coupling from the spacing of the lines; the width and the signal-to-noise of
the tallest line; and, from those, how well the coupling is known. A value with
its uncertainty can be compared with an observation on equal terms. A value
without one is compared with a rule of thumb.

Position precision: the standard deviation of a line's position is of the
order of its full width at half maximum divided by its signal-to-noise ratio,
the estimator for a symmetric line whose height is measured against noise
(e.g. Posener, D.W. (1974). Precision in measuring resonance spectra.
J. Magn. Reson. 14:121-128). A coupling is the difference of two positions, so
its variance is twice that, plus what the grid adds.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from processing.annotation_scoring import normalize_mult
from processing.quant_integration import refine_apex_ppm

# The database position is where to look; the line is near it, not at it. The
# listing was read from this very spectrum, so the margin is small -- and it
# must be, because further out the window starts holding another signal: in a
# plasma spectrum the lipid CH2 at 1.28 ppm sits 30 Hz from lactate's doublet
# and is taller, and a window wide enough to admit it read a 28 Hz coupling.
SEARCH_MARGIN_PPM = 0.01

# No proton-proton coupling in these molecules exceeds this. Two apices further
# apart are two signals, not one multiplet.
MAX_J_HZ = 20.0

# Lines of a first-order multiplet, by symbol.
_N_LINES = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'quint': 5, 'p': 5,
            'sext': 6, 'sept': 7, 'h': 6}

# A line the detector will accept: this many times the noise, peak to
# baseline. Reference spectra are of pure compounds at millimolar and their
# lines stand far above this; the floor is against picking a ripple.
MIN_LINE_SNR = 5.0


def _pascal(n):
    row = [1.0]
    for _ in range(max(int(n), 1) - 1):
        row = [a + b for a, b in zip([0.0] + row, row + [0.0])]
    return row


def noise_sigma(ppm, y, *, quiet=(-0.5, -0.2)) -> float:
    """Noise from a region that holds nothing, by the median absolute
    deviation, so a stray line in it does not set the scale."""
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = min(quiet), max(quiet)
    m = (ppm >= lo) & (ppm <= hi)
    seg = y[m] if m.sum() >= 50 else y[: max(50, y.size // 50)]
    mad = float(np.median(np.abs(seg - np.median(seg))))
    return mad * 1.4826 if mad > 0 else float(np.std(seg)) or 1.0


def reference_offset(ppm, y, *, expected=0.0, half_width=0.3) -> float:
    """How far the reference line (DSS, TSP) sits from where it should.

    Returns the shift to subtract from the axis. Measured on the tallest
    line near ``expected``, refined off the grid, so the whole spectrum is
    referenced to better than the grid rather than to the nearest point.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (ppm >= expected - half_width) & (ppm <= expected + half_width)
    if m.sum() < 5:
        return 0.0
    i = int(np.argmax(y[m]))
    apex = float(ppm[m][i])
    return refine_apex_ppm(ppm, y, apex) - float(expected)


def linewidth_ppm(ppm, y, apex_ppm, *, baseline=0.0) -> float | None:
    """Full width at half maximum of the line at ``apex_ppm``, walked on the
    trace with the crossing interpolated.

    ``baseline`` is what the line stands on. Half maximum is half of the height
    ABOVE it: a line on a broad hump never falls to half its absolute value,
    and the walk then runs to the next valley and reports the neighbourhood.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.argmin(np.abs(ppm - float(apex_ppm))))
    half = float(baseline) + (float(y[i]) - float(baseline)) / 2.0
    if half <= float(baseline):
        return None
    lo = i
    while lo > 0 and y[lo] > half:
        lo -= 1
    hi = i
    while hi < y.size - 1 and y[hi] > half:
        hi += 1
    if lo == 0 or hi == y.size - 1:
        return None

    def _cross(a, b):
        ya, yb = float(y[a]), float(y[b])
        if yb == ya:
            return float(ppm[a])
        t = (half - ya) / (yb - ya)
        return float(ppm[a] + t * (ppm[b] - ppm[a]))

    return abs(_cross(hi - 1, hi) - _cross(lo, lo + 1))


def _lines_in_window(ppm, y, lo, hi, sigma, n_lines):
    """The ``n_lines`` tallest apices inside ``[lo, hi]`` above the floor,
    sorted by position, refined off the grid."""
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (ppm >= lo) & (ppm <= hi)
    idx = np.flatnonzero(m)
    if idx.size < 5:
        return []
    seg = y[idx]
    apices, props = find_peaks(seg, height=MIN_LINE_SNR * sigma,
                               prominence=MIN_LINE_SNR * sigma)
    if apices.size == 0:
        return []
    heights = props['peak_heights']
    order = np.argsort(heights)[::-1]
    # A line of the multiplet is at least a fair share of its tallest line:
    # the binomial pattern's smallest member, with room for a poor pattern.
    # Without this a noise ripple on the flank of a tall line passes for the
    # second line of a doublet.
    pattern = _pascal(n_lines)
    floor = 0.5 * (min(pattern) / max(pattern)) * float(heights[order[0]])
    order = [o for o in order if heights[o] >= floor][:int(n_lines)]
    chosen = [int(idx[a]) for a in apices[order]]
    out = []
    for i in chosen:
        pos = refine_apex_ppm(ppm, y, float(ppm[i]))
        out.append({'ppm': float(pos), 'height': float(y[i]), 'index': i})
    # By position, not by index: the axis runs downhill in ppm, as an NMR
    # axis does, and a spacing taken along the index came out negative.
    return sorted(out, key=lambda ln: ln['ppm'])


def search_half_width(n_lines, field_mhz, *, margin=SEARCH_MARGIN_PPM) -> float:
    """Half the span a multiplet of ``n_lines`` can have at the largest
    plausible coupling, plus the margin for where the listing put it."""
    return (max(int(n_lines), 1) - 1) / 2.0 * MAX_J_HZ / float(field_mhz) + margin


def measure_signal(ppm, y, *, centre, multiplicity, field_mhz, sigma=None,
                   half_width=None) -> dict:
    """One database signal, measured on the spectrum it was read from.

    Returns a dict with ``status`` ('ok', 'lines_missing', 'unresolved',
    'not_a_multiplet'), and for 'ok': ``ppm`` (intensity-weighted centre),
    ``lines`` (positions), ``j_hz`` and ``j_sigma_hz`` (None for a singlet),
    ``linewidth_hz``, ``snr``, ``pattern_agreement`` (how close the line
    heights are to the binomial pattern, 1 is exact).

    Unresolved multiplets ('m') have no first-order pattern to measure and
    come back 'unresolved'.
    """
    mult = normalize_mult(multiplicity or 's')
    n = _N_LINES.get(mult)
    if n is None:
        return {'status': 'unresolved', 'multiplicity': mult}
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if sigma is None:
        sigma = noise_sigma(ppm, y)
    if half_width is None:
        half_width = search_half_width(n, field_mhz)
    lo, hi = float(centre) - half_width, float(centre) + half_width
    lines = _lines_in_window(ppm, y, lo, hi, sigma, n)
    if len(lines) < n:
        return {'status': 'lines_missing', 'multiplicity': mult,
                'n_found': len(lines), 'n_expected': n,
                'lines': [ln['ppm'] for ln in lines]}

    heights = np.array([ln['height'] for ln in lines])
    pattern = np.array(_pascal(n))
    agreement = 1.0 - float(np.abs(heights / heights.sum()
                                   - pattern / pattern.sum()).sum()) / 2.0
    positions = np.array([ln['ppm'] for ln in lines])
    centre_meas = float((positions * heights).sum() / heights.sum())

    tallest = lines[int(np.argmax(heights))]
    # The line stands on whatever the window's floor is -- flat for a pure
    # compound in buffer, a hump in a biofluid. Height and width are taken
    # above it, or a lipid hump reads as a 600 Hz creatinine line.
    win = (ppm >= lo) & (ppm <= hi)
    base = float(np.percentile(y[win], 10)) if win.sum() >= 10 else 0.0
    lw = linewidth_ppm(ppm, y, tallest['ppm'], baseline=base)
    snr = (float((tallest['height'] - base) / sigma) if sigma > 0
           else float('inf'))
    dppm = float(abs(ppm[1] - ppm[0])) if ppm.size > 1 else 0.0

    out = {'status': 'ok', 'multiplicity': mult, 'ppm': centre_meas,
           'lines': [float(p) for p in positions],
           'linewidth_hz': (lw * field_mhz) if lw else None,
           'snr': snr, 'pattern_agreement': agreement,
           'j_hz': None, 'j_sigma_hz': None}
    if n >= 2:
        spacing = np.abs(np.diff(positions))
        j = float(np.mean(spacing)) * float(field_mhz)
        if j > MAX_J_HZ:
            return {'status': 'not_a_multiplet', 'multiplicity': mult,
                    'lines': out['lines'], 'spacing_hz': j}
        # Position error of one line ~ FWHM / SNR; a spacing is a difference
        # of two, so twice the variance. The apex is refined off the grid, so
        # the grid's quantisation is not a floor here; a line too narrow to
        # walk falls back to one grid step as its width.
        pos_sigma_ppm = (lw / max(snr, 1.0)) if lw else dppm
        j_sigma = float(np.sqrt(2.0) * pos_sigma_ppm * field_mhz)
        # Lines that should be equally spaced and are not say the pattern is
        # not first order, or two signals overlap: the spread of the spacings
        # is then the honest uncertainty, whatever the noise says.
        if spacing.size > 1:
            j_sigma = max(j_sigma, float(np.std(spacing, ddof=1)) * field_mhz)
        out['j_hz'], out['j_sigma_hz'] = j, j_sigma
    return out


def measure_fingerprint(ppm, y, signals, *, field_mhz, sigma=None) -> list:
    """Every listed signal of one compound, measured on its spectrum.

    ``signals`` are ``{'ppm', 'multiplicity', ...}`` as the database lists
    them; each result carries the input under ``listed`` beside what was
    measured, so the two can be compared row by row.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if sigma is None:
        sigma = noise_sigma(ppm, y)
    out = []
    for s in signals or ():
        m = measure_signal(ppm, y, centre=float(s.get('ppm', 0.0)),
                           multiplicity=s.get('multiplicity') or s.get('type'),
                           field_mhz=field_mhz, sigma=sigma)
        m['listed'] = dict(s)
        out.append(m)
    return out


# ── Patterns: what an unresolved multiplet actually looks like ──────────────
# A third of the reference database's signals are 'm': a coupling list the
# curators could not reduce to one letter -- D-glucose's ring protons carry
# two, three, four couplings each -- and the build wrote them off as having no
# pattern to verify. Seven of glucose's nine signals are 'm'; the annotation
# never matched it, and its peaks went to singlets of other compounds.
#
# The pattern is in the spectrum. Within the signal's own window the lines
# stand where the couplings put them, and their spacings are in hertz, so
# the pattern measured at 500 MHz predicts the pattern at 600 to the extent
# the system is first order -- which the ring protons of a sugar are, nearly:
# their shift differences are tens of hertz against couplings of ten. What
# is measured here is offsets from the pattern's centre, in hertz, and
# relative heights, and the annotation snaps them to detected peaks the way
# it snaps a doublet's two lines.

# Lines closer than this, in hertz, are one line the grid split, or a shoulder
# the detector took twice; a reference pattern is coarser than that.
MIN_PATTERN_SPACING_HZ = 0.5


def measure_pattern(ppm, y, *, lo, hi, field_mhz, sigma=None, max_lines=16,
                    min_rel=0.05) -> dict:
    """Every line inside ``[lo, hi]``, as offsets from the pattern's centre.

    Returns ``{'status', 'centre_ppm', 'lines': [(offset_hz, rel_height)],
    'n_lines', 'snr'}`` -- 'ok' with two lines or more, 'single' with one,
    'none' with nothing above the floor. Lines below ``min_rel`` of the
    tallest are noise on a pattern this strong and are left out; a pattern
    is capped at ``max_lines`` because more is a region, not a multiplet.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if sigma is None:
        sigma = noise_sigma(ppm, y)
    lo, hi = (min(lo, hi), max(lo, hi))
    m = (ppm >= lo) & (ppm <= hi)
    idx = np.flatnonzero(m)
    if idx.size < 5:
        return {'status': 'none', 'lines': [], 'n_lines': 0}
    seg = y[idx]
    base = float(np.percentile(seg, 10))
    apices, props = find_peaks(seg, height=base + MIN_LINE_SNR * sigma,
                               prominence=MIN_LINE_SNR * sigma)
    if apices.size == 0:
        return {'status': 'none', 'lines': [], 'n_lines': 0}
    heights = props['peak_heights'] - base
    keep = heights >= float(min_rel) * float(heights.max())
    apices, heights = apices[keep], heights[keep]
    order = np.argsort(heights)[::-1][:int(max_lines)]
    pos = np.array([refine_apex_ppm(ppm, y, float(ppm[idx[a]]))
                    for a in apices[order]])
    hts = heights[order]
    srt = np.argsort(pos)
    pos, hts = pos[srt], hts[srt]
    # Merge what the grid split.
    merged_p, merged_h = [], []
    for pp, hh in zip(pos, hts):
        if merged_p and abs(pp - merged_p[-1]) * field_mhz < MIN_PATTERN_SPACING_HZ:
            if hh > merged_h[-1]:
                merged_p[-1], merged_h[-1] = pp, hh
            continue
        merged_p.append(float(pp))
        merged_h.append(float(hh))
    pos, hts = np.array(merged_p), np.array(merged_h)
    centre = float((pos * hts).sum() / hts.sum())
    lines = [(float((pp - centre) * field_mhz), float(hh / hts.max()))
             for pp, hh in zip(pos, hts)]
    return {'status': 'ok' if len(lines) >= 2 else 'single',
            'centre_ppm': centre, 'lines': lines, 'n_lines': len(lines),
            'snr': float(hts.max() / sigma) if sigma > 0 else float('inf')}


# A window's edge must sit on noise. While the spectrum at an edge is still
# this many sigma above the window's floor, a line is being cut and the window
# is let out. Looking at the found lines instead cannot see the one that was
# cut off entirely: it is not found.
EDGE_SIGMA = 3.0
WIDEN_STEP_HZ = 2.0
WIDEN_MAX_HZ = 15.0


def _edge_on_signal(ppm, y, edge_ppm, floor, sigma, half_hz, field_mhz):
    m = np.abs(ppm - edge_ppm) <= half_hz / float(field_mhz)
    if not m.any():
        return False
    return float(y[m].max()) > floor + EDGE_SIGMA * sigma


def measure_pattern_extended(ppm, y, *, lo, hi, field_mhz, sigma=None,
                             bounds=None, max_lines=16) -> dict:
    """The pattern, with the window let out until its edges sit on noise.

    The database lists a window per signal, and the window was drawn by a
    curator: D-glucose's doublet of doublets at 3.232 ppm spans +/-9 Hz and
    its window reaches +/-8.6, so the outer lines fell just outside it and the
    pattern came out as three lines of a four-line shape, asymmetric, with
    its centre in the wrong place. A multiplet ends where its lines stop.

    An edge standing on signal -- more than EDGE_SIGMA above the window's
    floor -- is cutting a line, and the window is let out by WIDEN_STEP_HZ on
    that side, up to WIDEN_MAX_HZ and never past ``bounds``, the midpoints
    to the compound's neighbouring signals: past those the lines are
    somebody else's.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if sigma is None:
        sigma = noise_sigma(ppm, y)
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    f = float(field_mhz)
    b_lo, b_hi = (bounds if bounds else (lo - WIDEN_MAX_HZ / f,
                                         hi + WIDEN_MAX_HZ / f))
    lo0, hi0 = lo, hi
    for _ in range(int(2 * WIDEN_MAX_HZ / WIDEN_STEP_HZ) + 2):
        m = (ppm >= lo) & (ppm <= hi)
        floor = float(np.percentile(y[m], 10)) if m.sum() >= 10 else 0.0
        grew = False
        if (_edge_on_signal(ppm, y, lo, floor, sigma, 1.0, f)
                and lo > b_lo and (lo0 - lo) * f < WIDEN_MAX_HZ):
            lo = max(b_lo, lo - WIDEN_STEP_HZ / f)
            grew = True
        if (_edge_on_signal(ppm, y, hi, floor, sigma, 1.0, f)
                and hi < b_hi and (hi - hi0) * f < WIDEN_MAX_HZ):
            hi = min(b_hi, hi + WIDEN_STEP_HZ / f)
            grew = True
        if not grew:
            break
    pat = measure_pattern(ppm, y, lo=lo, hi=hi, field_mhz=f, sigma=sigma,
                          max_lines=max_lines)
    pat['window'] = (lo, hi)
    return pat
