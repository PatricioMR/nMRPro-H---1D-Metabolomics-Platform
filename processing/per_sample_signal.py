# processing/per_sample_signal.py
"""What one annotated signal looks like in each sample, one sample at a time.

Annotation reads the mean spectrum: a single verdict for the whole study. That
hides the two things an analyst actually wants from a multi-sample run — in how
many samples the signal is really there, and what it measured in each one.

Where the lines of the signal sit in a given sample is decided by
``multiplet_line_fit``, which fits the multiplet as a unit against the apices
that sample really has. This module reads that assignment, adds the area
detection already integrated, and aggregates the samples into the one δ and J
the signal is reported by.

The signal is in the peak list by construction: the expected line positions
handed to the fit are the ``predicted_lines`` the annotation search matched to
detected peaks, so nothing here can report a signal detection never saw.

Blank and excluded samples are the caller's business — pass ``included``.
"""
from __future__ import annotations

import numpy as np

from processing.multiplet_line_fit import observed_j_hz

# A line's area in a sample comes from detection's own integration over the
# master peak bounds, which the per-sample apex position does not enter.
_PPM_KEY = 'ppm_per_sample'
_AREA_KEY = 'area_per_sample'


def _per_sample(peak, key, n_samples):
    """One peak's per-sample vector, padded to ``n_samples``; None when absent."""
    v = peak.get(key)
    if v is None:
        return None
    arr = list(v)
    if len(arr) < n_samples:
        arr += [None] * (n_samples - len(arr))
    return arr[:n_samples]


def sample_count(line_peaks) -> int:
    """How many samples the peaks carry data for; 0 when they carry none."""
    for pk in line_peaks or ():
        v = pk.get(_PPM_KEY)
        if v is not None:
            return len(v)
    return 0


def signal_in_sample(line_peaks, sample_idx, *, fit=None, db_ppm=None,
                     db_j_hz=None, field_mhz=600.0, n_lines=None) -> dict:
    """Measure one annotated signal in one sample, from its line assignment.

    ``fit`` is what ``multiplet_line_fit.fit_lines`` returned for this sample.
    Without one there is no reading: δ and J are not guessed from the master
    positions, which belong to the study and not to this sample.

    ``present`` is True only when the fit assigned EVERY line of the signal —
    the rule the pipeline's multiplet tier already uses. δ and J are still
    reported when only some lines were assigned, measured from those alone and
    flagged ``partial``: half a multiplet is worth seeing even though it cannot
    confirm the assignment.
    """
    n = sample_count(line_peaks)
    lines = list(line_peaks or ())
    n_exp = int(n_lines if n_lines is not None else len(lines))
    fit = fit or {}
    ppms = list(fit.get('ppms') or ())
    ranks = list(fit.get('ranks') or ())
    fitted = bool(fit.get('fitted'))

    n_found = len(ppms) if fitted else 0
    centre = float(np.mean(ppms)) if n_found >= 2 else None
    j_obs = observed_j_hz(ranks, ppms, field_mhz) if n_found >= 2 else None

    area = 0.0
    for pk in lines:
        v = _per_sample(pk, _AREA_KEY, n)
        if v is not None and 0 <= sample_idx < n and v[sample_idx] is not None:
            area += float(v[sample_idx])

    return dict(
        present=bool(n_found and n_found >= n_exp),
        ppm_centre=centre,
        ppm_diff=(abs(centre - float(db_ppm))
                  if (centre is not None and db_ppm is not None) else None),
        j_obs_hz=j_obs,
        delta_j_hz=(abs(j_obs - float(db_j_hz))
                    if (j_obs is not None and db_j_hz) else None),
        area=area,
        n_lines=n_exp,
        n_found=n_found,
        partial=bool(n_found) and n_found < n_exp,
        n_alternatives=int(fit.get('n_alternatives') or 0),
        residual_hz=fit.get('residual_hz') if fitted else None,
    )


def signal_across_samples(line_peaks, *, fits=None, included=None, db_ppm=None,
                          db_j_hz=None, field_mhz=600.0, n_lines=None,
                          j_from_db=False) -> dict:
    """The same signal measured in every sample the caller allows.

    ``fits`` maps sample index to the line assignment for that sample, as
    ``multiplet_line_fit.fit_across_samples`` returns it. ``included`` is the
    list of sample indices to report — blanks and excluded samples belong
    outside it, the way detection already treats them.

    ``j_from_db`` says the expected line positions were built from the
    database J rather than from detected peaks. The fit then recovers that J by
    construction — measured at a slope of 1.00 against 0.00 on the other path —
    so the consensus reports the positions but withholds the J: a number that
    came from the database is not evidence about the database.

    Returns the per-sample rows, the summary the results table needs
    (``n_present`` out of ``n_included``), and the consensus δ and J the signal
    is reported by.
    """
    n = sample_count(line_peaks)
    idx = list(range(n)) if included is None else [i for i in included if 0 <= i < n]
    fits = fits or {}
    rows = []
    for i in idx:
        row = signal_in_sample(line_peaks, i, fit=fits.get(i), db_ppm=db_ppm,
                               db_j_hz=db_j_hz, field_mhz=field_mhz,
                               n_lines=n_lines)
        row['sample_idx'] = i
        rows.append(row)
    present = [r for r in rows if r['present']]
    drifts = [r['ppm_centre'] for r in present if r['ppm_centre'] is not None]
    out = dict(
        per_sample=rows,
        n_present=len(present),
        n_included=len(idx),
        ppm_spread=(float(max(drifts) - min(drifts)) if len(drifts) >= 2 else None),
    )
    out.update(consensus(rows, db_ppm=db_ppm, db_j_hz=db_j_hz,
                         j_from_db=j_from_db))
    return out


def consensus(rows, *, db_ppm=None, db_j_hz=None, j_from_db=False) -> dict:
    """The signal's δ and J, agreed across the samples that measured them.

    The value annotation used to report came from the mean spectrum, which is
    the pointwise maximum across samples — a spectrum none of them actually is.
    These come from the samples themselves.

    Aggregated on the MEASURED quantity, never on its distance from the
    database::

        wrong:  ΔJ = median(|J_s - J_db|)      noise cannot cancel
        right:  ΔJ = |median(J_s) - J_db|      noise cancels

    ΔJ is an absolute value, so averaging it accumulates the per-sample scatter
    instead of averaging it away: a signal whose J agrees with the database
    exactly, read with ±0.5 Hz of noise, would report ~0.4 Hz of disagreement.
    The same rule ``relative_sd_percent`` already states for the CV.

    The median, not the mean — one bad reading out of five drags a mean, and
    the pipeline chooses distribution-free statistics everywhere else for the
    same reason. ``n`` and the scatter travel with the value: a consensus over
    two samples and one over twenty are not the same evidence.

    ``j_from_db`` withholds the J entirely. When the pattern searched for was
    laid out from the database J, the spacing found among its own lines is that
    J returned: perturbing it by 4 Hz moves the result by 3.99 Hz, against 0.00
    when the pattern came from detected peaks. The positions are still a
    measurement — where the lines sit is the sample's own answer — so δ stays.
    """
    ppms = [r['ppm_centre'] for r in (rows or ())
            if r.get('ppm_centre') is not None]
    js = ([] if j_from_db else
          [r['j_obs_hz'] for r in (rows or ())
           if r.get('j_obs_hz') is not None])

    def _sd(v):
        return float(np.std(v, ddof=1)) if len(v) >= 2 else None

    ppm_c = float(np.median(ppms)) if ppms else None
    j_c = float(np.median(js)) if js else None
    return dict(
        ppm_centre=ppm_c,
        ppm_diff=(abs(ppm_c - float(db_ppm))
                  if (ppm_c is not None and db_ppm is not None) else None),
        ppm_n=len(ppms),
        ppm_sd=_sd(ppms),
        observed_j_hz=j_c,
        delta_j_hz=(abs(j_c - float(db_j_hz))
                    if (j_c is not None and db_j_hz) else None),
        j_n=len(js),
        j_sd=_sd(js),
        j_from_db=bool(j_from_db),
    )


def alignment_shifts(rows, *, reference=None) -> dict:
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
    centres = {int(r['sample_idx']): float(r['ppm_centre'])
               for r in (rows or ())
               if r.get('present') and r.get('ppm_centre') is not None
               and r.get('sample_idx') is not None}
    if not centres:
        return {}
    ref = float(reference) if reference is not None else float(
        np.mean(list(centres.values())))
    return {i: float(c - ref) for i, c in centres.items()}


def percent_deviation(value, reference) -> float | None:
    """How far one reading sits from its group's mean, in percent.

    Answers the question the table is asked in practice — is THIS sample out of
    line with the others of its group — which a coefficient of variation over
    the whole set cannot: the CV says the set is tight, not which member is not.
    """
    if value is None or reference is None:
        return None
    try:
        v, r = float(value), float(reference)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(v) and np.isfinite(r)) or abs(r) < 1e-12:
        return None
    return float((v - r) / abs(r) * 100.0)


def group_means(rows, key, group_key='group'):
    """Mean of ``key`` within each group, over the rows that carry a value.

    Groups are the analyst's own — the deviation is meant to be read against
    the samples a sample belongs with, not against a study-wide average that
    mixes conditions a chemical shift is expected to separate.
    """
    buckets: dict = {}
    for r in rows or ():
        v = r.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            buckets.setdefault(r.get(group_key), []).append(v)
    return {g: float(np.mean(v)) for g, v in buckets.items() if v}


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
