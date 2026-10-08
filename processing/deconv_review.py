"""What the deconvolution review shows, and what the analyst's decisions are.

The quantification resolves overlapping signals in three ways, each of which
puts a number on an analyte that no window of its own holds:

  shared     a window two accepted analytes share, split by their shapes
             (ADR-015)
  joint      analytes whose shapes overlap fitted together, with a free
             component on every unassigned peak (ADR-046)
  cascade    analytes the election did not take, fitted one by one to what
             the quantified ones leave of the spectrum (ADR-028)

and two ways of identifying without a number: a joint fit that could not place
a shape on its signal (``fit_misplaced``) and a cascade identification whose
score leaves it without one (``cascade_crowded``). The review lists all of them
-- ``review_items`` -- draws every fitted analyte standing in a region
(``region_curves``), and keeps what the analyst decided about each
(``describe_decision``, ``decisions_changed``). Pure: no window, no engine.

A decision is stored in the same per-analyte record the shared-window
overrides already use (``_deconv_overrides[met_id]``), so a project saves and
restores it with no new field:

  {'decision': 'accept'}   quantify it over what the engine refused
  {'decision': 'reject'}   leave it out of the election, the fit and the cascade
  {'off': True}            keep a shared window whole
  {'exclude': [met_id]}    leave these analytes out of a shared window
"""
from __future__ import annotations

import numpy as np

KIND_SHARED = 'shared'
KIND_JOINT = 'joint'
KIND_CASCADE = 'cascade'
KIND_CANDIDATE = 'candidate'
KIND_REJECTED = 'rejected'

KIND_TEXT = {
    KIND_SHARED: 'Shared window',
    KIND_JOINT: 'Joint fit',
    KIND_CASCADE: 'Cascade',
    KIND_CANDIDATE: 'No number',
    KIND_REJECTED: 'Rejected',
}

KIND_HELP = {
    KIND_SHARED: 'Two accepted analytes share this window; it was split '
                 'between them by their shapes.',
    KIND_JOINT: 'Fitted together with the analytes whose shapes overlap it, '
                'and with a free component on every unassigned peak.',
    KIND_CASCADE: 'The election did not take it; it was fitted to what the '
                  'quantified analytes leave of the spectrum.',
    KIND_CANDIDATE: 'Identified, but not given a number: the fit could not '
                    'place its shape, or the part of the spectrum that is '
                    'its own is too small or its signals disagree.',
    KIND_REJECTED: 'Left out at the analyst\'s decision.',
}

# Not given a number, but fitted or looked for: what the review shows.
_CANDIDATE_REASONS = ('fit_misplaced', 'cascade_crowded')

# A figure window is this far (ppm) around the fitted signal.
VIEW_PAD_PPM = 0.04


def kind_of(row) -> str | None:
    """The review kind of a quantification row, None when nothing about it was
    deconvolved."""
    if row.get('_is_is'):
        return None
    if row.get('_rejection') == 'analyst_rejected':
        return KIND_REJECTED
    if row.get('_not_quantified'):
        return KIND_CANDIDATE if row.get('_rejection') in _CANDIDATE_REASONS else None
    if row.get('_by_cascade'):
        return KIND_CASCADE
    if row.get('_joint_fit'):
        return KIND_JOINT
    if row.get('_deconv'):
        return KIND_SHARED
    return None


def item_window(row):
    """``(lo, hi)`` ppm the item stands in: the fitted signal, the shared
    window, the integration window, or, for a row with none, the signals the
    cascade looked at."""
    fc = row.get('_fit_curve') or {}
    if fc.get('focus'):
        a, b = fc['focus']
        return float(min(a, b)), float(max(a, b))
    d = row.get('_deconv') or {}
    for w in (d.get('window'), row.get('_multiplet_window')):
        if w:
            return float(min(w)), float(max(w))
    sig = (row.get('_cascade') or {}).get('signals') or ()
    if sig:
        # the tallest signal is the one the cascade found it by
        top = max(sig, key=lambda s: s[3])
        return float(top[0]) - 0.02, float(top[0]) + 0.02
    p = row.get('ppm')
    if p:
        return float(p) - 0.02, float(p) + 0.02
    return None


def partners(row, rows_by_id) -> list:
    """The other analytes this one was fitted with: the joint fit's group, a
    shared window's components, and the quantified analytes whose fitted
    curves stand under its signal (cascade)."""
    mid = str(row.get('hmdb_id'))
    out = []
    jf = row.get('_joint_fit') or {}
    for m in jf.get('group') or ():
        m = str(m)
        if m != mid and m in rows_by_id:
            out.append(m)
    for c in (row.get('_deconv') or {}).get('components', ()):
        m = str(c.get('met_id'))
        if m != mid and m not in out:
            out.append(m)
    return out


def unexplained_share(trace, fit, x) -> float | None:
    """What of the signal above the fit's own baseline the fit leaves: the
    positive part of ``trace - fit`` over the positive part of ``trace``,
    on the trapezoid. None when the trace holds nothing there."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(trace, dtype=float)
    f = np.asarray(fit, dtype=float)
    if x.size < 2 or t.size != x.size or f.size != x.size:
        return None
    order = np.argsort(x)
    x, t, f = x[order], t[order], f[order]
    tot = float(np.trapezoid(np.clip(t, 0.0, None), x))
    if tot <= 0:
        return None
    return float(np.trapezoid(np.clip(t - f, 0.0, None), x) / tot)


def region_curves(rows, ppm, lo, hi, group):
    """Every fitted analyte standing in ``[lo, hi]``, as the fit drew it for
    ``group``: ``[(met_id, name, x, y), ...]``, x ascending, y the analyte's
    own curve (its amplitude times its shape, without the baseline), plus the
    baseline the fits stood on as ``('baseline', ...)`` when there is one.

    Read from the fit records the rows carry (``_fit_curve``): the same
    curves the report draws, so the review and the report show one fit.
    A shared window resolved by shape has no such record and is drawn by the
    caller from its own components.
    """
    ppm = np.asarray(ppm, dtype=float)
    g = str(group)
    sel = (ppm >= lo) & (ppm <= hi)
    x = np.sort(ppm[sel])
    out, base_sum = [], None
    if x.size < 2:
        return out
    for r in rows or ():
        if r.get('_not_quantified') or r.get('_is_is'):
            continue
        fc = r.get('_fit_curve')
        if not fc:
            continue
        a = (fc.get('amp_by_group') or {}).get(g)
        unit = (fc.get('unit_by_group') or {}).get(g)
        if a is None or unit is None:
            continue
        fx = np.asarray(fc['ppm'], dtype=float)
        if fx.max() < lo or fx.min() > hi:
            continue
        order = np.argsort(fx)
        y = float(a) * np.asarray(unit, dtype=float)[order]
        yi = np.interp(x, fx[order], y, left=0.0, right=0.0)
        if not np.any(yi > 0):
            continue
        out.append((str(r.get('hmdb_id')), str(r.get('metabolite')), x, yi))
        b = (fc.get('baseline_by_group') or {}).get(g)
        if b is not None and base_sum is None:
            base_sum = np.interp(x, fx[order], np.asarray(b, dtype=float)[order])
    if base_sum is not None:
        out.append(('baseline', 'baseline', x, base_sum))
    return out


def review_items(rows, overrides=None) -> list:
    """One entry per row the review has something to say about.

    ``{'id', 'name', 'kind', 'quantified', 'confidence', 'window', 'partners',
    'ppm', 'unexplained', 'score', 'free_share', 'why', 'warnings',
    'decision'}`` -- ``decision`` as ``describe_decision`` words it; sorted by
    what most needs a look: candidates and one-dot rows first, then by
    position.
    """
    overrides = overrides or {}
    rows = list(rows or ())
    by_id = {str(r.get('hmdb_id')): r for r in rows}
    items = []
    for r in rows:
        k = kind_of(r)
        if k is None:
            continue
        mid = str(r.get('hmdb_id'))
        win = item_window(r)
        casc = r.get('_cascade') or {}
        fc = r.get('_fit_curve') or {}
        rem = list(((r.get('_deconv') or {}).get('remainder') or {}).values())
        warns = []
        if fc and fc.get('at_limit'):
            warns.append(f"{len(fc['at_limit'])} of {fc.get('n_signals')} signals "
                         "held at the position limit of the fit")
        if fc and fc.get('focus_matches_window') is False:
            warns.append('the fitted signal is not the one the election found')
        if casc.get('wide'):
            warns.append('its column was slid far from where the reference lists it')
        if k == KIND_CASCADE and casc.get('free_share') is not None \
                and float(casc['free_share']) < 0.6:
            warns.append(f"{100 * (1 - float(casc['free_share'])):.0f}% of its "
                         "signal lies where other analytes already account "
                         "for the trace")
        items.append({
            'id': mid, 'name': str(r.get('metabolite') or mid), 'kind': k,
            # what produced the number, or tried to: a candidate came from
            # the cascade or from a joint fit that could not place it
            'via': (KIND_CASCADE if r.get('_rejection') == 'cascade_crowded'
                    else KIND_JOINT) if k == KIND_CANDIDATE else k,
            'quantified': not r.get('_not_quantified'),
            'confidence': int(r.get('_confidence') or 0),
            'window': win,
            'partners': partners(r, by_id),
            'ppm': float(r.get('ppm') or (sum(win) / 2 if win else 0.0)),
            'unexplained': (float(np.median(rem)) if rem else None),
            'score': casc.get('score'),
            'covar': casc.get('covar'),
            'free_share': casc.get('free_share'),
            'why': str(r.get('_identity_why') or ''),
            'warnings': warns,
            'decision': describe_decision(overrides.get(mid)),
        })
    items.sort(key=lambda i: (i['quantified'] and i['confidence'] > 1,
                              i['kind'] != KIND_CANDIDATE, i['ppm']))
    return items


# ── Decisions ────────────────────────────────────────────────────────────

def clean_decision(ov) -> dict:
    """The part of an override that is a decision, empty when none."""
    ov = dict(ov or {})
    out = {}
    if ov.get('decision') in ('accept', 'reject'):
        out['decision'] = ov['decision']
    if ov.get('off'):
        out['off'] = True
    if ov.get('exclude'):
        out['exclude'] = sorted(str(m) for m in ov['exclude'])
    return out


def describe_decision(ov) -> str:
    """The decision in the words a table cell carries: '' for none."""
    ov = clean_decision(ov)
    parts = []
    if ov.get('decision') == 'accept':
        parts.append('✓ accepted')
    elif ov.get('decision') == 'reject':
        parts.append('✗ rejected')
    if ov.get('off'):
        parts.append('window kept whole')
    if ov.get('exclude'):
        parts.append(f"−{len(ov['exclude'])} from window")
    return ' · '.join(parts)


def set_decision(overrides, mid, decision=None, *, off=None, exclude=None) -> dict:
    """``overrides`` with ``mid``'s record changed (a new dict; the argument
    is not touched). ``decision`` 'accept' / 'reject' / None clears it;
    ``off`` and ``exclude`` are left as they were when None. A record left
    empty is dropped."""
    out = {k: dict(v) for k, v in (overrides or {}).items()}
    rec = dict(out.get(str(mid)) or {})
    if decision is None:
        rec.pop('decision', None)
    else:
        rec['decision'] = decision
    if off is not None:
        if off:
            rec['off'] = True
        else:
            rec.pop('off', None)
    if exclude is not None:
        if exclude:
            rec['exclude'] = sorted(str(m) for m in exclude)
        else:
            rec.pop('exclude', None)
    rec = clean_decision(rec)
    if rec:
        out[str(mid)] = rec
    else:
        out.pop(str(mid), None)
    return out


def decisions_changed(applied, pending) -> list:
    """The analytes whose decision in ``pending`` is not the one the last run
    was made with (``applied``), sorted."""
    ids = set(applied or {}) | set(pending or {})
    return sorted(m for m in ids
                  if clean_decision((applied or {}).get(m))
                  != clean_decision((pending or {}).get(m)))


def counts(items) -> dict:
    """``{kind: n}`` of the items, every kind present."""
    out = {k: 0 for k in KIND_TEXT}
    for i in items:
        out[i['kind']] += 1
    return out


def change_summary(before, after, *, moved=0.10) -> dict:
    """What a re-quantification changed, by metabolite id, between two row
    lists: ``{'entered': [names], 'left': [names], 'moved': [(name, ratio)],
    'n_before', 'n_after'}``. ``moved`` are the analytes quantified in both
    whose concentration changed by more than ``moved`` (a fraction).

    A decision on an analyte the joint fit or the cascade holds reshuffles
    what the cascade takes next -- it is greedy -- so the weak rows move
    around it; the review says so after applying rather than leave the analyst
    to find out."""
    def quantified(rows):
        return {str(r.get('hmdb_id')): r for r in rows or ()
                if not r.get('_not_quantified') and not r.get('_is_is')}
    a, b = quantified(before), quantified(after)
    name = lambda m: str((b.get(m) or a.get(m) or {}).get('metabolite') or m)
    mv = []
    for m in sorted(set(a) & set(b)):
        x, y = a[m].get('conc_mm'), b[m].get('conc_mm')
        try:
            ratio = float(y) / float(x)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if np.isfinite(ratio) and abs(ratio - 1.0) > moved:
            mv.append((name(m), ratio))
    return {'entered': sorted(name(m) for m in set(b) - set(a)),
            'left': sorted(name(m) for m in set(a) - set(b)),
            'moved': mv, 'n_before': len(a), 'n_after': len(b)}


def change_text(c) -> str:
    """The summary in a sentence."""
    if not (c['entered'] or c['left'] or c['moved']):
        return f"Re-quantified: nothing else changed ({c['n_after']} quantified)."
    parts = [f"{c['n_before']} → {c['n_after']} quantified"]
    if c['entered']:
        parts.append(f"{len(c['entered'])} entered")
    if c['left']:
        parts.append(f"{len(c['left'])} left")
    if c['moved']:
        parts.append(f"{len(c['moved'])} moved by more than 10%")
    return ('Re-quantified: ' + ', '.join(parts) + '. The cascade is greedy: '
            'a decision on an analyte the fit holds reshuffles the weak rows '
            'around it.')
