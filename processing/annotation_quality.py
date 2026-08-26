# processing/annotation_quality.py
"""Turning a score into a verdict.

Scoring says how well a candidate matches; this module decides what to do about
it — Good, Bad, or leave it to the analyst. The two were entangled in the
results window, where a score could be rewritten by the rule that read it.

Singlets are handled apart from everything else, because a singlet carries no
coupling: its whole evidence is a position, so the question is not how well it
scores but whether anything else in the region could be the same peak.
"""
from __future__ import annotations

from processing.annotation_scoring import is_singlet_like

# Verdicts a signal can receive automatically. None is the middle band: the
# evidence neither confirms nor rejects, and the analyst decides.
QUALITY_GOOD = 'good'
QUALITY_BAD = 'bad'
QUALITY_UNDECIDED = None


def quality_verdict(mp, *, params) -> tuple[str | None, str | None]:
    """Automatic quality for one signal — returns ``(quality, reason)``.

    Read in order:

      singlet             → undecided. A singlet is weak evidence on its own;
                            only the singlet rule can promote it, and only when
                            it is alone in its window and uncontested.
      lines missing       → Bad. Some lines of the multiplet were not found, so
                            the coupling was never confirmed.
      all lines + Δppm    → Good, whatever the score says. A complete multiplet
      + ΔJ agreeing         at the right shift with the right coupling is an
                            identification; a composite number cannot overrule
                            three independent agreements.
      score >= good       → Good, unless Δppm is out of tolerance — a strong
                            coupling term can carry the score while the signal
                            sits at the wrong shift.
      score <  bad        → Bad.
      otherwise           → undecided.

    A branch for "all this analyte's signals are singlets" used to sit in the
    score section; it could never run, since a singlet returns above. It carried
    the only use of the fingerprint-coverage threshold with it.
    """
    if is_singlet_like(mp.get('type') or mp.get('multiplicity') or 's'):
        return QUALITY_UNDECIDED, None

    n_found = mp.get('_n_found_lines')
    n_expected = mp.get('_n_exp_lines')
    if (n_found is not None and n_expected is not None
            and int(n_expected) >= 2 and int(n_found) < int(n_expected)):
        return QUALITY_BAD, (f'Incomplete multiplet: {n_found} of '
                             f'{n_expected} lines found')

    ppm_diff = float(mp.get('ppm_diff', 0) or 0)
    ppm_ok = ppm_diff < params['dppm_gold']
    dj = mp.get('delta_j_hz')
    dj_ok = dj is None or float(dj) < params['dj_bad_hz']

    if (ppm_ok and dj_ok and n_found is not None and n_expected is not None
            and int(n_found) >= 2 and int(n_found) >= int(n_expected)):
        return QUALITY_GOOD, 'Full-multiplet confirmed: all lines found'

    score = float(mp.get('composite_score', 0) or 0)
    if score >= params['good_score_thresh']:
        if ppm_ok:
            return QUALITY_GOOD, None
        return QUALITY_UNDECIDED, (f'Auto-good blocked: Δppm={ppm_diff:.4f} '
                                   f'>= {params["dppm_gold"]} ppm')
    if score < params['bad_score_thresh']:
        return QUALITY_BAD, None
    return QUALITY_UNDECIDED, None


def promote_singlets(metabolites, peaks, *, params, manual_marks=None):
    """Decide the quality of every singlet, in one rule.

    A singlet has no J to validate, so two kinds of ambiguity disqualify it:
    another detected signal in its window that it might equally be, and another
    metabolite laying claim to the same peak.

      Δppm >= tolerance          → Bad. No database singlet sits here.
      contested                  → undecided. One does, but which is undecided.
      alone and uncontested      → Good, and an anchor.

    The window is ±the annotation tolerance around the detected peak — the same
    number that decided the candidate was worth considering. Three rules used to
    promote singlets with two thresholds sixty times apart; neither survived the
    data, because the measured peak drift sits between them.

    Manual marks are never overwritten. Returns the number promoted.
    """
    manual_marks = manual_marks or {}
    tol = float(params['dppm_gold'])
    peak_ppms = [float(p.get('ppm', 0)) for p in (peaks or [])]

    rivals: dict = {}
    for met in metabolites:
        mid = met.get('id', met.get('name'))
        for mp in met.get('matched_peaks', []):
            rank = mp.get('peak_rank')
            if rank is None or mp.get('annotation_quality') == 'bad':
                continue
            rivals.setdefault(rank, set()).add(mid)

    promoted = 0
    for met in metabolites:
        for mp in met.get('matched_peaks', []):
            mult = mp.get('type') or mp.get('multiplicity') or 's'
            if not is_singlet_like(mult, strict=True):
                continue
            if (met.get('id'), mp.get('peak_rank')) in manual_marks:
                continue

            if float(mp.get('ppm_diff', 0) or 0) >= tol:
                mp['annotation_quality'] = QUALITY_BAD
                mp.setdefault('_status_reasons', []).append(
                    f'Singlet Δppm >= {tol} ppm — no DB singlet here')
                continue

            n_rivals = max(0, len(rivals.get(mp.get('peak_rank'), ())) - 1)
            centre = float(mp.get('peak_ppm') or mp.get('db_ppm') or 0)
            n_in_window = sum(1 for v in peak_ppms if abs(v - centre) <= tol)

            if n_rivals == 0 and n_in_window <= 1:
                mp['annotation_quality'] = QUALITY_GOOD
                mp['auto_anchor'] = True
                mp.setdefault('_status_reasons', []).append(
                    'Singlet alone in its window, uncontested')
                promoted += 1
            else:
                mp['annotation_quality'] = QUALITY_UNDECIDED
                mp.setdefault('_status_reasons', []).append(
                    f'Singlet contested: {n_rivals} rival(s), '
                    f'{n_in_window} signal(s) within {tol} ppm')
    return promoted
