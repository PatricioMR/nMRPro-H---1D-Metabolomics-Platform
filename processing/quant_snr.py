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
    """``{sample: SNR}`` for one window, from the window's own apex.

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
    return {int(s): float(np.max(arr[m, s])) / float(noise) for s in idx}


def representative_snr(snr_by_sample) -> float:
    """The one SNR the results table shows: the median over the samples.

    Median, not mean: a study where the analyte is absent from one group should
    not have that group's noise-level SNR pulled up by the group where it is
    abundant, nor the reverse.
    """
    v = [float(x) for x in (snr_by_sample or {}).values() if np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')
