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
