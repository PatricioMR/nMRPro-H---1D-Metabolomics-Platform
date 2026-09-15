"""A measured line pattern fitted to the spectrum as a shape.

A pattern of fourteen lines -- the ribose protons of NAD between 4.2 and 4.3
ppm -- is never found line by line in plasma: the lines lean on each other
at the study's linewidth, on the neighbours' lines, and on the detector's
idea of where a peak begins. What can be asked of the spectrum is whether
the shape is in it. The pattern is laid down as a sum of Lorentzians of the
study's own width, slid over the tolerance, and at each position the
largest copy of it that fits UNDER the trace is taken: other analytes add
intensity, never remove it, so a line of the pattern standing higher than
the trace is a contradiction while the trace standing higher than a line is
overlap. The position where the largest copy fits is where the pattern is.

What comes out is a measurement, not a verdict: the amplitude in the
trace's units and as a signal-to-noise ratio, the fraction of the trace
over the pattern's lines that the pattern accounts for (1.0 where the
pattern is alone, less where it shares the region), and where the pattern's
apices sit, for the line-by-line reading the annotation already makes.

Lorentzian lineshape: Abragam, The Principles of Nuclear Magnetism, Oxford
1961, ch. III. Fitting a spectrum as a non-negative sum of known shapes is
the premise of Lawson & Hanson's NNLS (Solving Least Squares Problems,
1974) and of every targeted-profiling quantifier since Weljie et al., Anal.
Chem. 2006, 78, 4430; the one-sided amplitude here is the NNLS answer for
one component when the rest of the mixture is unknown.
"""
from __future__ import annotations

import numpy as np

# Points where the model stands below this fraction of its own maximum are
# between lines: the trace there is baseline and neighbours, and says
# nothing about the pattern's height.
SUPPORT_REL = 0.2

# The slack, in noise sigmas, a line may stand above the trace by before it
# is a contradiction: a line exactly as tall as the trace is a line the
# noise put half the time below it.
SLACK_SIGMA = 2.0

# The fit is refined to this fraction of a hertz around the coarse best.
FINE_STEP_HZ = 0.05


def simulate(x_ppm, lines, origin_ppm, fwhm_hz, field_mhz) -> np.ndarray:
    """The pattern's shape on ``x_ppm``: Lorentzians of FWHM ``fwhm_hz`` at
    ``origin_ppm + offset/field`` with the pattern's relative heights,
    scaled so the tallest point is 1."""
    x = np.asarray(x_ppm, dtype=float)
    y = np.zeros_like(x)
    hw = float(fwhm_hz) / 2.0 / float(field_mhz)
    if hw <= 0:
        return y
    for off, rel in lines:
        c = float(origin_ppm) + float(off) / float(field_mhz)
        y += float(rel) * hw ** 2 / ((x - c) ** 2 + hw ** 2)
    top = float(y.max()) if y.size else 0.0
    return y / top if top > 0 else y


def _amplitude(y, m, sigma):
    """The largest multiple of ``m`` that fits under ``y`` on the model's
    support, with SLACK_SIGMA of noise to spare -- 0 when the trace is
    below the noise everywhere the pattern has a line."""
    sup = m >= SUPPORT_REL
    if not sup.any():
        return 0.0
    ratio = (y[sup] + SLACK_SIGMA * float(sigma)) / m[sup]
    return float(max(0.0, ratio.min()))


def fit_pattern_shape(ppm, y, lines, *, db_ppm, field_mhz, fwhm_hz,
                      tolerance_ppm, sigma, step_hz=0.5) -> dict | None:
    """Slide the pattern over ``db_ppm +/- tolerance_ppm`` and keep the
    position where the largest copy of it fits under the trace.

    ``lines`` is ``[(offset_hz, rel_height)]``; ``ppm`` descends or ascends,
    it does not matter. Returns None when there is no pattern, no width, or
    no trace over the region; otherwise a dict with ``origin_ppm`` (where the
    pattern's reference position fell), ``shift_ppm`` (``origin - db_ppm``),
    ``amplitude`` (trace units, at the pattern's tallest point), ``snr``
    (``amplitude / sigma``), ``explained`` (the fraction of the trace over
    the pattern's lines the pattern accounts for, 0-1), ``r2`` (the fit's
    coefficient of determination on that support, negative under overlap)
    and ``apex_ppms`` (where the pattern's apices sit at this linewidth).
    """
    lines = [(float(o), float(r)) for o, r in (lines or ())]
    if len(lines) < 2 or not fwhm_hz or float(fwhm_hz) <= 0 \
            or not field_mhz or float(sigma or 0) <= 0:
        return None
    f = float(field_mhz)
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    offs = [o for o, _ in lines]
    half_span = (max(offs) - min(offs)) / 2.0 / f
    reach = float(tolerance_ppm) + half_span + 3.0 * float(fwhm_hz) / f
    sel = (ppm >= float(db_ppm) - reach) & (ppm <= float(db_ppm) + reach)
    if sel.sum() < 8:
        return None
    x = ppm[sel]
    t = y[sel]

    def _at(origin):
        m = simulate(x, lines, origin, fwhm_hz, f)
        return _amplitude(t, m, sigma), m

    best = None
    coarse = np.arange(-float(tolerance_ppm), float(tolerance_ppm) + 1e-12,
                       float(step_hz) / f)
    for d in coarse:
        a, _m = _at(float(db_ppm) + d)
        if best is None or a > best[0]:
            best = (a, float(db_ppm) + d)
    fine = np.arange(-float(step_hz), float(step_hz) + 1e-12,
                     FINE_STEP_HZ) / f
    for d in fine:
        a, _m = _at(best[1] + d)
        if a > best[0]:
            best = (a, best[1] + d)
    a, origin = best
    m = simulate(x, lines, origin, fwhm_hz, f)
    sup = m >= SUPPORT_REL
    fitted = a * m
    denom = float(t[sup].sum())
    explained = float(min(1.0, fitted[sup].sum() / denom)) if denom > 0 else 0.0
    ss_res = float(((t[sup] - fitted[sup]) ** 2).sum())
    ss_tot = float(((t[sup] - t[sup].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # where the pattern's apices are, on a fine grid of its own
    xs = np.arange(min(offs) - 3 * float(fwhm_hz),
                   max(offs) + 3 * float(fwhm_hz), FINE_STEP_HZ)
    ms = simulate(origin + xs / f, lines, origin, fwhm_hz, f)
    inner = (ms[1:-1] > ms[:-2]) & (ms[1:-1] >= ms[2:])
    apex = [float(origin + xs[i + 1] / f) for i in np.nonzero(inner)[0]]
    return {'origin_ppm': float(origin),
            'shift_ppm': float(origin - float(db_ppm)),
            'amplitude': float(a), 'snr': float(a / float(sigma)),
            'explained': explained, 'r2': float(r2),
            'apex_ppms': sorted(apex)}


def trace_apices_near(ppm, y, positions, *, tol_hz, field_mhz) -> list:
    """Which of ``positions`` (ppm) has a local maximum of the trace within
    ``tol_hz``; returns the maxima's ppm, or None per position."""
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if ppm.size < 3:
        return [None for _ in positions]
    tol = float(tol_hz) / float(field_mhz)
    inner = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    maxima = ppm[1:-1][inner]
    out = []
    for p in positions:
        if maxima.size == 0:
            out.append(None)
            continue
        i = int(np.argmin(np.abs(maxima - float(p))))
        out.append(float(maxima[i]) if abs(maxima[i] - float(p)) <= tol
                   else None)
    return out
