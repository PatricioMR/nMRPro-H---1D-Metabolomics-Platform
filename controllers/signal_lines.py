# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# controllers/signal_lines.py
"""
The analyst's choice of a signal's lines, written on the annotation.

The lines ride on the matched peak (``processing.signal_lines``). With them
the signal is marked Good: a signal validated by hand is the analyst's
judgement, so it is quantified on its own lines -- it competes for them in
the election and is not left to the cascade, whose joint fit is for what no
signal resolves. A window it shares with another analyte is still resolved as
any shared window is. The mark and the annotation's verdict it replaced are
kept, and restoring the lines puts both back.

The quantification shown is the one made before the change: it is measured
again only when the analyst runs it.
"""
from processing.signal_lines import (
    clear_manual_lines, lines_before, manual_lines, set_manual_lines)

_MARK_BEFORE = '_mark_before_manual'


class SignalLineEditor:
    """Over the annotation's ``metabolites`` and the analyst's
    ``quality_marks`` ({(met_id, peak_rank): 'good' | 'bad' | ...}).

    Either may be a callable returning it: the annotation replaces its list
    when its window regroups the results, and its marks when a search starts
    again, and an edit must land on what the project saves."""

    def __init__(self, metabolites, quality_marks):
        self._mets = metabolites
        self._marks = quality_marks if quality_marks is not None else {}

    @property
    def marks(self) -> dict:
        return self._marks() if callable(self._marks) else self._marks

    def _mp(self, met_id, mp_index):
        mets = self._mets() if callable(self._mets) else self._mets
        met = next((m for m in mets or () if str(m.get('id')) == str(met_id)), None)
        if met is None:
            raise KeyError(f'no analyte {met_id!r} in the annotation')
        return met['matched_peaks'][int(mp_index)]

    def has_manual(self, met_id, mp_index) -> bool:
        return manual_lines(self._mp(met_id, mp_index)) is not None

    def apply(self, met_id, mp_index, peaks) -> int:
        """Make ``peaks`` the lines of the signal and mark it Good. Returns
        its anchor rank."""
        mp = self._mp(met_id, mp_index)
        mid = str(met_id)
        if _MARK_BEFORE not in mp:
            old = mp.get('peak_rank')
            mp[_MARK_BEFORE] = {'mark': self.marks.get((mid, old)),
                                'quality': mp.get('annotation_quality')}
        self.marks.pop((mid, mp.get('peak_rank')), None)
        rank = set_manual_lines(mp, peaks)
        self.marks[(mid, rank)] = 'good'
        mp['annotation_quality'] = 'good'
        return rank

    def restore(self, met_id, mp_index):
        """Give the signal back the annotation's own lines, anchor and
        verdict."""
        mp = self._mp(met_id, mp_index)
        mid = str(met_id)
        if lines_before(mp) is None:
            return
        self.marks.pop((mid, mp.get('peak_rank')), None)
        rank = clear_manual_lines(mp)
        before = mp.pop(_MARK_BEFORE, None) or {}
        if before.get('mark') is not None:
            self.marks[(mid, rank)] = before['mark']
        mp['annotation_quality'] = before.get('quality')
