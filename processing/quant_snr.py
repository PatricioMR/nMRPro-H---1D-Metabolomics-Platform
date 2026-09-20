# processing/quant_snr.py
"""How much of an integrated window is signal, sample by sample.

One window is integrated in every sample. In a sample where the analyte is
weak or absent that window still integrates something, so an area on its own
cannot tell "present but small" from "not there at all". The SNR of the window
in each sample is what separates them, and reporting a concentration without it
asks the analyst to trust a number whose evidence was withheld.
"""
from __future__ import annotations

import numpy as np


def noise_level(line_peaks, ppm, intensities) -> float | None:
    """The noise the window's SNR is measured against.

    Detection already estimated it and stored it on the peak; that estimate is
    preferred, so quantification and the Signal List cannot disagree about how
    loud the noise is. When it is absent the peak's own SNR reconstructs it —
    the apex over the ratio detection reported — and when neither exists there
    is no honest denominator and None says so.
    """
    for pk in line_peaks or ():
        n = pk.get('noise')
        if n and float(n) > 0:
            return float(n)
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    ppm = np.asarray(ppm, dtype=float)
    for pk in line_peaks or ():
        snr = pk.get('snr')
        if not snr or float(snr) <= 0:
            continue
        lo = min(float(pk.get('start_ppm', pk.get('ppm', 0))),
                 float(pk.get('end_ppm', pk.get('ppm', 0))))
        hi = max(float(pk.get('start_ppm', pk.get('ppm', 0))),
                 float(pk.get('end_ppm', pk.get('ppm', 0))))
        m = (ppm >= lo) & (ppm <= hi)
        if not m.any():
            continue
        apex = float(np.max(np.mean(arr[m, :], axis=1)))
        if apex > 0:
            return apex / float(snr)
    return None


def window_snr(ppm, intensities, lo, hi, noise, *, included=None) -> dict:
    """``{sample: SNR}`` for one window: its height above its own baseline.

    Height, not absolute intensity. A window sitting on a baseline offset would
    otherwise report that offset as signal — in the sugar region of one study
    the raw intensity ran from -1.04e5 to +3.98e4, so the number said far more
    about where the baseline was than about the analyte. The window's own
    endpoints define the baseline, the same straight line the integration
    subtracts, so the SNR and the area describe the same measurement.

    NaN when the noise level is unknown — an unmeasured SNR is not a zero, and
    a zero would read as "no signal here" rather than "nobody looked".
    """
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    ppm = np.asarray(ppm, dtype=float)
    if lo > hi:
        lo, hi = hi, lo
    idx = (range(arr.shape[1]) if included is None
           else [i for i in included if 0 <= i < arr.shape[1]])
    if not noise or float(noise) <= 0:
        return {int(s): float('nan') for s in idx}
    m = (ppm >= lo) & (ppm <= hi)
    if not m.any():
        return {int(s): float('nan') for s in idx}
    out = {}
    for s in idx:
        seg = arr[m, s]
        if seg.size < 2:
            out[int(s)] = float('nan')
            continue
        base = np.linspace(float(seg[0]), float(seg[-1]), seg.size)
        height = float(np.max(seg - base))
        if height <= 0.0:
            # The endpoint line ran through the signal, which happens when an
            # apex sits on a window edge — a spectrum that ends there, or a
            # window clipped by the axis. Measured on a synthetic study whose
            # axis began at 0.0 ppm with the standard at 0.0: the line ran from
            # the apex down to the tail and the height came out zero, so the
            # internal standard was reported with no signal at all.
            height = float(np.max(seg) - np.min(seg))
        out[int(s)] = height / float(noise)
    return out


def representative_snr(snr_by_sample) -> float:
    """The one SNR the results table shows: the median over the samples.

    Median, not mean: a study where the analyte is absent from one group should
    not have that group's noise-level SNR pulled up by the group where it is
    abundant, nor the reverse.
    """
    v = [float(x) for x in (snr_by_sample or {}).values() if np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')


def group_evidence(ppm, intensities, groups, *, lo, hi, noise,
                   lines=(), weights=(), hwhm=None) -> dict:
    """What the elected signal looks like on each group's mean trace:
    ``{group: {'snr': float, 'cosine': float | None}}``.

    The signal, not its anchor peak: the window is the one the signal was
    integrated on, so a doublet's SNR is its higher line above the window's
    own endpoint line, against the noise detection measured for the study
    (``window_snr`` with the global noise). The cosine is the annotation's
    own shape term (``peak_shape_score``: the predicted pattern of lines and
    relative intensities against the trace in the signal's window), read on
    the group's mean rather than on the study's. Same functions, same
    numbers the analyst already reads elsewhere -- one per group.

    Blanks are not a group (``gui.stages._quant_adapter.sample_groups``).
    A group with no sample in the matrix is absent from the result.
    """
    from processing.annotation_scoring import peak_shape_score
    arr = np.asarray(intensities, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    p = np.asarray(ppm, dtype=float)
    ppms = [float(v) for v in (lines or ()) if v is not None]
    w = list(weights or ())
    if len(w) != len(ppms) or not ppms or sum(w) <= 0:
        w = [1.0] * len(ppms)
    sticks = list(zip(ppms, w))
    out = {}
    for g, idxs in (groups or {}).items():
        cols = [int(i) for i in idxs if 0 <= int(i) < arr.shape[1]]
        if not cols:
            continue
        mean = arr[:, cols].mean(axis=1)
        snr = window_snr(p, mean, lo, hi, noise).get(0, float('nan'))
        cos = None
        if sticks and hwhm and float(hwhm) > 0:
            cos = peak_shape_score(p, mean, sticks, float(np.mean(ppms)), {},
                                   hwhm_default=float(hwhm))
        out[str(g)] = {'snr': float(snr), 'cosine': cos}
    return out


def best_group(evidence, *, cosine_tie=None) -> str | None:
    """The group the figure shows and the numbers come from: the best
    cosine, and among the groups within ``cosine_tie`` of it (the
    annotation's own tie band, ``COSINE_TIE``) the highest SNR. Without a
    cosine anywhere, the highest SNR alone. None when nothing was measured.
    """
    from processing.annotation_scoring import COSINE_TIE
    tie = COSINE_TIE if cosine_tie is None else float(cosine_tie)
    rows = [(g, e) for g, e in (evidence or {}).items()
            if e and np.isfinite(e.get('snr', float('nan')))]
    if not rows:
        return None
    cosines = [e['cosine'] for _, e in rows if e.get('cosine') is not None]
    if cosines:
        top = max(cosines)
        rows = [(g, e) for g, e in rows
                if e.get('cosine') is not None and e['cosine'] >= top - tie]
    return max(rows, key=lambda ge: ge[1]['snr'])[0]
