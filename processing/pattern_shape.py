# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A multiplet's measured shape fitted to the spectrum.

A pattern of fourteen lines -- the ribose protons of NAD between 4.2 and 4.3
ppm -- is never found line by line in plasma: the lines lean on each other
at the study's linewidth, on the neighbours' lines, and on the detector's
idea of where a peak begins. What can be asked of the spectrum is whether
the shape is in it, and how tall.

The shape is the reference's own: the trace of the pure compound over the
multiplet, on a 0.1 Hz grid, broadened to the study's linewidth (Lorentzian
widths add under convolution, so a Lorentzian of the difference does it).
Where the database holds only the lines -- positions and heights -- they are
laid down as Lorentzians of the study's width instead; on the reference
spectra themselves the profile explains 97% of the trace and the lines 76%,
so the lines are the fallback, not the model.

The shape is slid over the tolerance and at each position the largest copy
of it that fits UNDER the trace is taken, judged at its apices: other
analytes add intensity, never remove it, so an apex standing higher than the
trace is a contradiction while the trace standing higher than an apex is
overlap. The position where the largest copy fits is where the multiplet
is. Out come the amplitude (as a signal-to-noise ratio: is it here?) and the
fraction of the trace over the apices the shape accounts for (is it alone?).

Lorentzian lineshape: Abragam, The Principles of Nuclear Magnetism, Oxford
1961, ch. III. Fitting a spectrum as a non-negative sum of known shapes is
the premise of Lawson & Hanson's NNLS (Solving Least Squares Problems,
1974) and of targeted profiling (Weljie et al., Anal. Chem. 2006, 78, 4430);
the one-sided amplitude here is the NNLS answer for one component when the
rest of the mixture is unknown.
"""
from __future__ import annotations

import json

import numpy as np

# An apex of the model below this fraction of its tallest point is a minor
# line; the amplitude is not judged on it.
SUPPORT_REL = 0.2

# Around an apex, points holding this fraction of the apex's height belong
# to it; the flanks below do not -- a study a little narrower than the
# reference stands under the model's flanks and is no contradiction.
APEX_REL = 0.75

# The slack, in noise sigmas, an apex may stand above the trace by before it
# is a contradiction: an apex exactly as tall as the trace is one the noise
# put half the time below it.
SLACK_SIGMA = 2.0

# The profile's grid, in hertz, as the reference database stores it.
PROFILE_STEP_HZ = 0.1

# The fit is refined to this fraction of a hertz around the coarse best.
FINE_STEP_HZ = 0.05


def parse_profile(raw) -> tuple[np.ndarray, np.ndarray] | None:
    """``(offsets_hz, heights)`` from the database's ``pattern_profile``
    JSON ``{"start": hz, "step": hz, "h": [...]}``, or None."""
    if not raw:
        return None
    try:
        d = json.loads(raw) if isinstance(raw, str) else dict(raw)
        h = np.asarray(d['h'], dtype=float)
        start, step = float(d['start']), float(d.get('step', PROFILE_STEP_HZ))
    except (TypeError, ValueError, KeyError):
        return None
    if h.size < 3 or step <= 0 or h.max() <= 0:
        return None
    return start + step * np.arange(h.size), h


def encode_profile(offsets_hz, heights) -> str:
    """The inverse of ``parse_profile``: heights scaled to a tallest point
    of 1, four decimals, on a uniform grid."""
    h = np.asarray(heights, dtype=float)
    offs = np.asarray(offsets_hz, dtype=float)
    top = float(h.max()) if h.size else 0.0
    h = h / top if top > 0 else h
    step = float(offs[1] - offs[0]) if offs.size > 1 else PROFILE_STEP_HZ
    return json.dumps({'start': round(float(offs[0]), 3),
                       'step': round(step, 4),
                       'h': [round(float(v), 4) for v in h]},
                      separators=(',', ':'))


def lines_model(x_ppm, lines, origin_ppm, fwhm_hz, field_mhz) -> np.ndarray:
    """Lorentzians of FWHM ``fwhm_hz`` at ``origin + offset/field`` with the
    lines' relative heights, scaled so the tallest point is 1."""
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


def profile_model(x_ppm, profile, origin_ppm, extra_fwhm_hz,
                  field_mhz) -> np.ndarray:
    """The reference's profile at ``origin``, broadened by a Lorentzian of
    ``extra_fwhm_hz`` (the study's width less the reference's), scaled so
    the tallest point is 1."""
    offs, h = profile
    offs = np.asarray(offs, dtype=float)
    h = np.asarray(h, dtype=float)
    if offs.size < 3 or h.max() <= 0:
        return np.zeros(len(x_ppm))
    step = float(offs[1] - offs[0])
    if extra_fwhm_hz and float(extra_fwhm_hz) > 0:
        hw = float(extra_fwhm_hz) / 2.0
        k = np.arange(-4 * float(extra_fwhm_hz),
                      4 * float(extra_fwhm_hz) + step, step)
        kern = hw ** 2 / (k ** 2 + hw ** 2)
        # the broadened profile spreads past the cut: room for it, and
        # for a kernel longer than a short profile
        pad = k.size
        h = np.concatenate([np.zeros(pad), h, np.zeros(pad)])
        offs = np.concatenate([offs[0] - step * np.arange(pad, 0, -1), offs,
                               offs[-1] + step * np.arange(1, pad + 1)])
        h = np.convolve(h, kern / kern.sum(), mode='same')
    x_hz = (np.asarray(x_ppm, dtype=float) - float(origin_ppm)) * float(field_mhz)
    y = np.interp(x_hz, offs, h, left=0.0, right=0.0)
    top = float(y.max()) if y.size else 0.0
    return y / top if top > 0 else y


def apex_support(m) -> np.ndarray:
    """The points that belong to the model's apices: each local maximum at
    or above SUPPORT_REL, with the contiguous points holding APEX_REL of
    its height."""
    m = np.asarray(m, dtype=float)
    sup = np.zeros(m.size, dtype=bool)
    if m.size < 3:
        return sup
    inner = (m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]) & (m[1:-1] >= SUPPORT_REL)
    for i in np.nonzero(inner)[0] + 1:
        bar = APEX_REL * m[i]
        lo = i
        while lo > 0 and m[lo - 1] >= bar:
            lo -= 1
        hi = i
        while hi < m.size - 1 and m[hi + 1] >= bar:
            hi += 1
        sup[lo:hi + 1] = True
    return sup


def _amplitude(t, m, sigma):
    sup = apex_support(m)
    if not sup.any():
        return 0.0, sup
    ratio = (t[sup] + SLACK_SIGMA * float(sigma)) / m[sup]
    return float(max(0.0, ratio.min())), sup


def fit_shape(ppm, y, *, db_ppm, field_mhz, fwhm_hz, tolerance_ppm, sigma,
              profile=None, profile_fwhm_hz=None, lines=None,
              step_hz=0.5) -> dict | None:
    """Slide the shape over ``db_ppm +/- tolerance_ppm`` and keep the
    position where the largest copy of it fits under the trace.

    ``profile`` (``(offsets_hz, heights)`` with the reference's own width
    ``profile_fwhm_hz``) is the shape when given; else ``lines``
    (``[(offset_hz, rel_height)]``) laid down as Lorentzians of ``fwhm_hz``.
    ``fwhm_hz`` is the study's linewidth, ``sigma`` its noise.

    Returns None without a shape, a width or a trace over the region; else
    ``origin_ppm`` (where the reference position fell), ``shift_ppm``
    (``origin - db_ppm``), ``amplitude`` (trace units at the tallest
    point), ``snr`` (``amplitude / sigma``), ``explained`` (the fraction of
    the trace over the apices the shape accounts for, 0-1), ``r2`` (the
    fit's coefficient of determination there, negative under overlap),
    ``apex_ppms`` and ``apex_rel`` (the apices at this width and their
    heights relative to the tallest), and ``model`` -- a callable giving
    the fitted shape, scaled to 1, on any ppm axis.
    """
    f = float(field_mhz or 0)
    if f <= 0 or not fwhm_hz or float(fwhm_hz) <= 0 or float(sigma or 0) <= 0:
        return None
    if profile is not None and len(profile[0]) >= 3:
        offs = np.asarray(profile[0], dtype=float)
        extra = max(0.0, float(fwhm_hz) - float(profile_fwhm_hz or 0.0))

        def model(x, origin):
            return profile_model(x, profile, origin, extra, f)
        pad_hz = 3.0 * max(extra, float(fwhm_hz))
    elif lines and len(lines) >= 2:
        lines = [(float(o), float(r)) for o, r in lines]
        offs = np.asarray([o for o, _ in lines])

        def model(x, origin):
            return lines_model(x, lines, origin, fwhm_hz, f)
        pad_hz = 3.0 * float(fwhm_hz)
    else:
        return None
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    half_span = (offs.max() - offs.min()) / 2.0 / f
    reach = float(tolerance_ppm) + half_span + pad_hz / f
    sel = (ppm >= float(db_ppm) - reach) & (ppm <= float(db_ppm) + reach)
    if sel.sum() < 8:
        return None
    x = ppm[sel]
    t = y[sel]

    best = None
    for d in np.arange(-float(tolerance_ppm), float(tolerance_ppm) + 1e-12,
                       float(step_hz) / f):
        a, _s = _amplitude(t, model(x, float(db_ppm) + d), sigma)
        if best is None or a > best[0]:
            best = (a, float(db_ppm) + d)
    for d in np.arange(-float(step_hz), float(step_hz) + 1e-12,
                       FINE_STEP_HZ) / f:
        a, _s = _amplitude(t, model(x, best[1] + d), sigma)
        if a > best[0]:
            best = (a, best[1] + d)
    a, origin = best
    m = model(x, origin)
    _a, sup = _amplitude(t, m, sigma)
    fitted = a * m
    denom = float(t[sup].sum()) if sup.any() else 0.0
    explained = float(min(1.0, fitted[sup].sum() / denom)) if denom > 0 else 0.0
    ss_res = float(((t[sup] - fitted[sup]) ** 2).sum()) if sup.any() else 0.0
    ss_tot = float(((t[sup] - t[sup].mean()) ** 2).sum()) if sup.any() else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # the apices, on a fine grid of the shape's own
    xs = origin + np.arange(offs.min() - pad_hz, offs.max() + pad_hz,
                            FINE_STEP_HZ) / f
    ms = model(xs, origin)
    inner = (ms[1:-1] > ms[:-2]) & (ms[1:-1] >= ms[2:]) & (ms[1:-1] >= SUPPORT_REL)
    idx = np.nonzero(inner)[0] + 1
    return {'origin_ppm': float(origin),
            'shift_ppm': float(origin - float(db_ppm)),
            'amplitude': float(a), 'snr': float(a / float(sigma)),
            'explained': explained, 'r2': float(r2),
            'apex_ppms': [float(v) for v in xs[idx]],
            'apex_rel': [float(v) for v in ms[idx]],
            'model': lambda xx, _o=origin: model(np.asarray(xx, dtype=float), _o)}
