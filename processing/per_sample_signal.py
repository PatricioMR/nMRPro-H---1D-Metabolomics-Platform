# processing/per_sample_signal.py
"""What one annotated signal looks like in each sample, one sample at a time.

Annotation reads the mean spectrum: a single verdict for the whole study. That
hides the two things an analyst actually wants from a multi-sample run — in how
many samples the signal is really there, and how far its lines drift between
them. Both are already recorded on the detected peaks: ``detect_peaks`` runs per
sample and ``refine_peak_centers_per_sample`` writes each peak's apex, area and
genuine detection presence for every sample. Nothing here re-measures the
spectrum; it reads what detection already established, so a signal can never be
"present" here and absent in the peak list.

The presence rule is the one the multiplet rule already uses: every line, or
nothing. A doublet with one leg detected is not a doublet in that sample.

Blank and excluded samples are the caller's business — pass ``included``.
"""
from __future__ import annotations

import numpy as np

# A line's apex position in a sample is only meaningful where that line was
# detected; elsewhere refine_peak_centers_per_sample locked onto noise.
_PPM_KEY = 'ppm_per_sample'
_PRESENT_KEY = 'aligned_per_sample'
_AREA_KEY = 'area_per_sample'

# Two lines of the same multiplet may be reported at the same position: the
# per-sample apex search (peak_alignment._refine_one_center) scans +-0.02 ppm,
# which is 12 Hz at 600 MHz, while the median line spacing is 7.1 Hz — so the
# taller leg wins BOTH searches and the pattern collapses onto it. Measured on
# the reference set: 64% of (multiplet, sample) pairs. A reading whose lines sit
# closer than this fraction of their master spacing has lost the pattern; its
# positions describe one leg, not the multiplet, and neither a J nor a centre
# may be reported from it.
COLLAPSE_FRACTION = 0.5


def _per_sample(peak, key, n_samples):
    """One peak's per-sample vector, padded to ``n_samples``; None when absent."""
    v = peak.get(key)
    if v is None:
        return None
    arr = list(v)
    if len(arr) < n_samples:
        arr += [None] * (n_samples - len(arr))
    return arr[:n_samples]


def _min_gap(values):
    srt = sorted(float(v) for v in values)
    return min((srt[i + 1] - srt[i] for i in range(len(srt) - 1)), default=None)


def _lines_collapsed(line_peaks, sample_ppms) -> bool:
    """Did the per-sample apex search fold this multiplet onto one line?

    Compared against the multiplet's own master spacing rather than a fixed Hz:
    what counts as collapsed for a 3 Hz doublet is not what counts for a 12 Hz
    one.
    """
    if len(sample_ppms) < 2:
        return False
    master = _min_gap(pk.get('ppm', 0.0) for pk in line_peaks)
    here = _min_gap(sample_ppms)
    if not master or master <= 0 or here is None:
        return False
    return here < COLLAPSE_FRACTION * master


def sample_count(line_peaks) -> int:
    """How many samples the peaks carry data for; 0 when they carry none."""
    for pk in line_peaks or ():
        v = pk.get(_PPM_KEY)
        if v is not None:
            return len(v)
    return 0


def line_positions(line_peaks, sample_idx) -> list[float]:
    """Apex ppm of every line of this signal, in one sample.

    Returned in the peaks' own order, not sorted — the caller decides whether
    order matters (spacings sort, a centre does not).
    """
    out = []
    n = sample_count(line_peaks)
    for pk in line_peaks or ():
        v = _per_sample(pk, _PPM_KEY, n)
        p = v[sample_idx] if (v is not None and 0 <= sample_idx < n) else None
        out.append(float(p) if p is not None else float(pk.get('ppm', 0.0)))
    return out


def signal_in_sample(line_peaks, sample_idx, *, db_ppm=None, db_j_hz=None,
                     field_mhz=600.0) -> dict:
    """Measure one annotated signal in one sample.

    ``present`` is True only when EVERY line of the signal was detected in this
    sample. ``j_obs_hz`` is the median spacing of the lines as they sit in this
    sample, and is None for a singlet or when the signal is absent — an absent
    line's recorded position is noise, and a J measured from it is fiction.
    """
    n = sample_count(line_peaks)
    lines = list(line_peaks or ())
    blank = dict(present=False, ppm_centre=None, ppm_diff=None,
                 j_obs_hz=None, delta_j_hz=None, area=None, n_lines=len(lines),
                 collapsed=False)
    if not lines or n == 0 or not (0 <= sample_idx < n):
        return blank

    present = True
    for pk in lines:
        v = _per_sample(pk, _PRESENT_KEY, n)
        if v is None:
            continue          # detection did not record presence — do not invent absence
        if not v[sample_idx]:
            present = False
            break

    ppms = line_positions(lines, sample_idx)
    collapsed = _lines_collapsed(lines, ppms)
    centre = (float(np.mean(ppms)) if (ppms and not collapsed) else None)

    area = 0.0
    for pk in lines:
        v = _per_sample(pk, _AREA_KEY, n)
        if v is not None and v[sample_idx] is not None:
            area += float(v[sample_idx])

    j_obs = None
    if present and not collapsed and len(ppms) >= 2 and field_mhz > 0:
        srt = sorted(ppms)
        seps = [abs(srt[k + 1] - srt[k]) * field_mhz for k in range(len(srt) - 1)]
        if seps:
            j_obs = float(np.median(seps))

    return dict(
        present=present,
        ppm_centre=centre,
        ppm_diff=(abs(centre - float(db_ppm))
                  if (centre is not None and db_ppm is not None) else None),
        j_obs_hz=j_obs,
        delta_j_hz=(abs(j_obs - float(db_j_hz))
                    if (j_obs is not None and db_j_hz) else None),
        area=area,
        n_lines=len(lines),
        collapsed=collapsed,
    )


def signal_across_samples(line_peaks, *, included=None, db_ppm=None,
                          db_j_hz=None, field_mhz=600.0) -> dict:
    """The same signal measured in every sample the caller allows.

    ``included`` is the list of sample indices to report — blanks and excluded
    samples belong outside it, the way detection already treats them. Omitted,
    every sample is reported.

    Returns the per-sample rows plus the summary the results table needs:
    ``n_present`` out of ``n_included``.
    """
    n = sample_count(line_peaks)
    idx = list(range(n)) if included is None else [i for i in included if 0 <= i < n]
    rows = []
    for i in idx:
        row = signal_in_sample(line_peaks, i, db_ppm=db_ppm, db_j_hz=db_j_hz,
                               field_mhz=field_mhz)
        row['sample_idx'] = i
        rows.append(row)
    present = [r for r in rows if r['present']]
    drifts = [r['ppm_centre'] for r in present if r['ppm_centre'] is not None]
    n_collapsed = sum(1 for r in rows if r.get('collapsed'))
    return dict(
        per_sample=rows,
        n_present=len(present),
        n_included=len(idx),
        ppm_spread=(float(max(drifts) - min(drifts)) if len(drifts) >= 2 else None),
        n_collapsed=n_collapsed,
    )


def alignment_shifts(line_peaks, *, included=None, reference=None) -> dict:
    """Per-sample shift that puts this signal's pattern on top of itself.

    The multiplet aligns to the multiplet: a sample's shift is how far the
    CENTRE of its own lines sits from the reference centre, so the whole
    pattern moves as one and the spacing between the lines — the coupling — is
    never touched. Alanine's doublet lines up against the other samples'
    doublets, and a doublet that drifted 20 Hz still reads J = 7 Hz.

    ``reference`` defaults to the mean centre over the present samples, which
    keeps the group near zero shift instead of pinning it to sample 0.

    Returns ``{sample_idx: shift_ppm}``. Subtract the shift from a sample's ppm
    axis to align it. Meant for display: nothing here writes to the spectrum.
    """
    n = sample_count(line_peaks)
    idx = list(range(n)) if included is None else [i for i in included if 0 <= i < n]
    centres = {}
    for i in idx:
        row = signal_in_sample(line_peaks, i)
        if row['present'] and row['ppm_centre'] is not None:
            centres[i] = row['ppm_centre']
    if not centres:
        return {}
    ref = float(reference) if reference is not None else float(
        np.mean(list(centres.values())))
    return {i: float(c - ref) for i, c in centres.items()}


def relative_sd_percent(values) -> float | None:
    """Coefficient of variation, in percent, of what was measured.

    Reported on the MEASURED quantity — the signal's δ, its observed J — never
    on the deviation from the database. A deviation is centred on zero by
    construction, so its own CV divides by ~0 and reports hundreds of percent
    for a set that agrees to three decimal places.

    None when fewer than two values, or when the mean is too close to zero for
    a ratio to mean anything.
    """
    vals = [float(v) for v in (values or ()) if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    mean = float(np.mean(vals))
    if abs(mean) < 1e-12:
        return None
    return float(np.std(vals, ddof=1) / abs(mean) * 100.0)
