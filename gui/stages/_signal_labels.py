# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Which analyte the Signal List names for each detected signal.

Once quantification has run, the column is its result: a signal is named
after an analyte only if that analyte was quantified. The quantification list
also carries every analyte it considered and turned down (below the limit of
quantification, refused, not found); naming signals after those, or after the
annotation's best guess where the quantification decided otherwise, made the
list read a different set of analytes than the table beside it.

Before quantification the annotation's choice stands, as it always has.
"""
from __future__ import annotations

from gui.stages._annotation_helpers import select_metabolite_per_rank
from processing.multiplet_line_selector import multiplet_line_peaks

NO_LABEL = '—'


def quantified_rows(rows):
    """Rows of analytes that got a number (not the internal standard)."""
    return [r for r in (rows or ())
            if not r.get('_is_is') and not r.get('_not_quantified')]


def _quant_claims(rows, pk_by_rank):
    """rank -> names of the quantified analytes whose measured signal is
    there: the signal's own peak and the detected lines of its multiplet,
    matched as the quantification matched them. An analyte the cascade took
    in has no peak of its own; its signal is the window it was measured on."""
    claims: dict = {}
    for r in rows:
        name = r.get('metabolite')
        if not name:
            continue
        ranks = set()
        if r.get('_peak_rank') in pk_by_rank:
            ranks.add(r['_peak_rank'])
        hit = (r.get('_hits') or [None])[0]
        if isinstance(hit, dict):
            ranks.update(p['rank'] for p in multiplet_line_peaks(hit, pk_by_rank)
                         if p.get('rank') is not None)
        win = r.get('_multiplet_window')
        if r.get('_by_cascade') and win:
            lo, hi = sorted(float(v) for v in win)
            ranks.update(rk for rk, p in pk_by_rank.items()
                         if lo <= float(p.get('ppm', -1e9)) <= hi)
        for rk in ranks:
            claims.setdefault(rk, set()).add(name)
    return claims


def signal_labels(peaks, *, quant_rows=None, metabolites=None,
                  manual_quality=None):
    """``{rank: name}`` for every detected signal in ``peaks``.

    With quantified rows: the signal measured for an analyte is named after
    it (two analytes measured on one signal are both named); any other signal
    the annotation matched to a quantified analyte follows the annotation's
    choice among the quantified ones; the rest has no analyte.

    Without: a name already on the signal (a manual or MCR assignment), then
    the annotation's choice.
    """
    pk_by_rank = {p['rank']: p for p in (peaks or ()) if p.get('rank') is not None}
    quantified = quantified_rows(quant_rows)
    if quantified:
        ids = {str(r.get('hmdb_id', '')) for r in quantified}
        mets = [m for m in (metabolites or ()) if str(m.get('id', '')) in ids]
        by_ann = select_metabolite_per_rank(mets, manual_quality=manual_quality)
        claims = _quant_claims(quantified, pk_by_rank)
        return {rk: (' / '.join(sorted(claims[rk])) if rk in claims
                     else by_ann.get(rk, NO_LABEL))
                for rk in pk_by_rank}
    by_ann = (select_metabolite_per_rank(metabolites, manual_quality=manual_quality)
              if metabolites else {})
    return {rk: (p.get('assigned_metabolite') or p.get('metabolite')
                 or by_ann.get(rk, NO_LABEL))
            for rk, p in pk_by_rank.items()}
