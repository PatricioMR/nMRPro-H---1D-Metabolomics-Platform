# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/signal_lines.py
"""
The detected lines the analyst chose to make up a signal.

The annotation matches each predicted line of a signal to the nearest
detected peak (``multiplet_line_selector.multiplet_line_peaks``). In a crowded
region the nearest peak can be another compound's line, and a multiplet of
four lines can stand among eight. The analyst who has read the lines against
the analyte's reference (the STOCSY window) can name the ones that are the
signal: their positions ride on the matched peak as ``manual_lines_ppm`` and
every reader of the signal's lines -- the annotation tree, the quantification,
the STOCSY window -- takes those instead of the matched ones.

Positions and not peak numbers: a new detection renumbers the peaks, and a
line keeps its δ. A position no detected peak stands on any more is a line
that is gone; it is not replaced by a neighbour.

Reversible: the first edit keeps what the signal was (its anchor peak and
δ), and ``clear_manual_lines`` puts it back.
"""
import numpy as np

MANUAL_LINES_KEY = 'manual_lines_ppm'
_BEFORE = '_lines_before_manual'

# A chosen line is the detected peak standing within this of its δ: the
# finest line-match window the annotation itself uses (LINE_MATCH_FLOOR).
MATCH_PPM = 0.002


def manual_lines(mp):
    """The δ of the lines the analyst chose for the signal, high δ first, or
    None when the signal keeps the annotation's own lines."""
    v = mp.get(MANUAL_LINES_KEY) if isinstance(mp, dict) else None
    if not isinstance(v, (list, tuple)) or not v:
        return None
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return sorted((x for x in out if np.isfinite(x)), reverse=True) or None


def peaks_at(lines_ppm, pk_by_rank, tol=MATCH_PPM) -> list:
    """The detected peak at each δ (nearest within ``tol``, none used twice);
    a δ with no peak there gives nothing."""
    out, used = [], set()
    for x in lines_ppm:
        best, bd = None, float(tol)
        for pk in pk_by_rank.values():
            r = pk.get('rank')
            if r in used:
                continue
            d = abs(float(pk.get('ppm', np.inf)) - float(x))
            if d <= bd:
                best, bd = pk, d
        if best is not None:
            out.append(best)
            used.add(best.get('rank'))
    return out


def set_manual_lines(mp, peaks) -> int:
    """Make ``peaks`` (detected-peak dicts) the signal's lines.

    The signal's position (``peak_ppm``) is the centre of its pattern, as the
    annotation gives it, and the study's drift is read from these positions:
    one line's δ put there would move every analyte's expected place. So the
    centre moves only by how far the chosen lines stand from the predicted
    lines they fall nearest to (the median offset): the same lines leave it
    where it was, one line of a doublet does not make that line the centre.
    The anchor stays when it is one of the chosen lines; otherwise it moves to
    the chosen line nearest the centre, so the signal claims one of its own
    lines and not a peak the analyst left out. Returns the anchor rank."""
    if not peaks:
        raise ValueError('a signal needs at least one line')
    if _BEFORE not in mp:
        mp[_BEFORE] = {'peak_rank': mp.get('peak_rank'), 'peak_ppm': mp.get('peak_ppm')}
    before = lines_before(mp)
    ppms = [float(p['ppm']) for p in peaks]
    centre = _centre(ppms, mp, before.get('peak_ppm'))
    anchor = next((p for p in peaks if p.get('rank') == before.get('peak_rank')), None)
    if anchor is None:
        anchor = min(peaks, key=lambda p: (abs(float(p['ppm']) - centre),
                                           -float(p.get('snr') or 0.0)))
    mp[MANUAL_LINES_KEY] = sorted(ppms, reverse=True)
    mp['peak_rank'] = int(anchor['rank'])
    mp['peak_ppm'] = centre
    return int(anchor['rank'])


def _centre(ppms, mp, centre_before) -> float:
    """Where the pattern stands once its lines are ``ppms``."""
    pred = [float(l['ppm']) for l in mp.get('predicted_lines') or () if 'ppm' in l]
    if not pred or centre_before is None:
        return float(np.mean(ppms))
    offset = np.median([x - min(pred, key=lambda p: abs(p - x)) for x in ppms])
    return float(centre_before) + float(offset)


def lines_before(mp) -> dict | None:
    """What the signal was before the analyst chose its lines."""
    v = mp.get(_BEFORE) if isinstance(mp, dict) else None
    return dict(v) if isinstance(v, dict) else None


def clear_manual_lines(mp):
    """Give the signal back the annotation's own lines and anchor. Returns
    the anchor rank it had before the edit (None when there was no edit)."""
    before = mp.pop(_BEFORE, None)
    mp.pop(MANUAL_LINES_KEY, None)
    if before is None:
        return None
    mp['peak_rank'] = before.get('peak_rank')
    mp['peak_ppm'] = before.get('peak_ppm')
    return before.get('peak_rank')


def singlet_on_peak(mp, peak, common_drift=None) -> dict:
    """A copy of the matched singlet ``mp`` standing on the detected ``peak``."""
    out = dict(mp)
    x = float(peak['ppm'])
    out['peak_ppm'] = x
    out['peak_rank'] = peak['rank']
    out['ppm_diff'] = abs(x - float(mp['db_ppm']))
    out['predicted_lines'] = [{'ppm': x, 'measured': True, 'rel_intensity': 1.0}]
    if common_drift is not None:
        out['ppm_off_drift'] = abs(x - float(mp['db_ppm']) - common_drift)
    out['obs_snr'] = peak.get('snr', mp.get('obs_snr'))
    return out
