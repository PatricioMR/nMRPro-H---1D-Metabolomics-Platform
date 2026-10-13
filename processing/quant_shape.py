"""Quantifying a complex multiplet from its fitted shape.

A multiplet judged as a shape (ADR-013) claims no peak: its lines were never
found one by one, so there is nothing for an integration window to run foot
to foot over, and letting it hold peaks would take them from analytes whose
lines were seen. What it has instead is the shape itself, placed where the
annotation fitted it. In every sample the largest copy of that shape that
fits under the sample's own trace is taken -- the same rule the annotation
used, so what is quantified is what was identified -- and its area is the
amplitude times the shape's own area. The shape's area is known exactly, so
no capture correction applies and none is made.

Under overlap the copy that fits under the trace is bounded by the least
overlapped apex, and can stand higher than the analyte alone would: the
number is an upper bound there, and the fraction of the region the shape
explains (``pattern_fit['explained']``) travels with it as the reason.

Targeted profiling: Weljie et al., Anal. Chem. 2006, 78, 4430. One-sided
amplitude: Lawson & Hanson, Solving Least Squares Problems, 1974 (NNLS).
"""
from __future__ import annotations

import numpy as np

from processing.pattern_shape import (
    _amplitude, lines_model, parse_profile, profile_model)
from processing.quant_integration import subtract_blank

# Beyond the shape's last line, in linewidths, the window ends.
PAD_LINEWIDTHS = 3.0


def shape_model(ev, ppm):
    """``(x, m, (lo, hi))`` -- the fitted shape on the study's axis over its
    own window, scaled to 1 -- or None when the evidence cannot build one.

    ``ev`` is the candidate's evidence: ``shape_fit`` with ``origin_ppm``,
    ``field_mhz`` and ``study_linewidth_hz``, and either ``pattern_profile``
    (with ``profile_fwhm_hz``) or ``pattern_lines``.
    """
    fit = (ev or {}).get('shape_fit') or {}
    origin = fit.get('origin_ppm')
    field = fit.get('field_mhz')
    fwhm = fit.get('study_linewidth_hz')
    if origin is None or not field or not fwhm:
        return None
    f = float(field)
    profile = parse_profile(fit.get('pattern_profile'))
    lines = fit.get('pattern_lines') or []
    if profile is not None:
        offs = np.asarray(profile[0], dtype=float)
        extra = max(0.0, float(fwhm) - float(fit.get('profile_fwhm_hz') or 0.0))
        pad = PAD_LINEWIDTHS * max(extra, float(fwhm))
    elif len(lines) >= 2:
        offs = np.asarray([float(o) for o, _r in lines])
        pad = PAD_LINEWIDTHS * float(fwhm)
    else:
        return None
    lo = float(origin) + (offs.min() - pad) / f
    hi = float(origin) + (offs.max() + pad) / f
    p = np.asarray(ppm, dtype=float)
    sel = (p >= lo) & (p <= hi)
    if sel.sum() < 8:
        return None
    x = p[sel]
    if profile is not None:
        m = profile_model(x, profile, float(origin), extra, f)
    else:
        m = lines_model(x, [(float(o), float(r)) for o, r in lines],
                        float(origin), float(fwhm), f)
    if m.max() <= 0:
        return None
    return x, m, (lo, hi)


def _area_of(m, x) -> float:
    order = np.argsort(x)
    xs, ms = x[order], m[order]
    return float(np.trapezoid(ms, x=xs) if hasattr(np, 'trapezoid')
                 else np.trapz(ms, x=xs))


def measure_shape(ev, ppm, intensities, *, included=None, noise_sigma,
                  blank_indices=None) -> dict | None:
    """The shape's area in every sample the caller allows, with the limits
    and the signal-to-noise that go with it, in the form
    ``processing.quant_engine`` keeps for every measured signal.

    None when the evidence builds no shape or the noise is unknown.
    """
    if noise_sigma is None or not np.isfinite(noise_sigma) or noise_sigma <= 0:
        return None
    built = shape_model(ev, ppm)
    if built is None:
        return None
    x, m, (lo, hi) = built
    p = np.asarray(ppm, dtype=float)
    sel = (p >= lo) & (p <= hi)
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_s = arr.shape[1]
    idx = range(n_s) if included is None else [i for i in included
                                              if 0 <= i < n_s]
    a_shape = _area_of(m, x)
    sigma = float(noise_sigma)

    def _amp(s):
        a, _sup = _amplitude(arr[sel, s], m, sigma)
        return float(a)

    amps = {int(s): _amp(s) for s in idx}
    areas = {s: a * a_shape for s, a in amps.items()}
    blank_area = float('nan')
    if blank_indices:
        blanks = {int(s): _amp(s) * a_shape for s in blank_indices
                  if 0 <= int(s) < n_s}
        if blanks:
            blank_area = float(np.mean(list(blanks.values())))
            areas = subtract_blank(areas, blanks)
    snr = {s: a / sigma for s, a in amps.items()}
    med = float(np.median(list(areas.values()))) if areas else float('nan')
    fit = (ev or {}).get('shape_fit') or {}
    return {
        'window': (lo, hi),
        'width_ppm': float(hi - lo),
        'areas': areas,
        'shifts': {},
        'blank_area': blank_area,
        'snr': snr,
        'area_median': med,
        'lines': [float(v) for v in (fit.get('apex_ppms') or ())],
        'weights': [float(v) for v in (fit.get('apex_rel') or ())],
        # the shape's area is known exactly: nothing to correct, nothing
        # to trust a neighbour subtraction on
        'captured': None,
        'shape': True,
        'shape_explained': fit.get('explained'),
    }
