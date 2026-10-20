# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Tests for lines_from_transitions (exact multiplet lines from DB transitions)."""
import sqlite3
import numpy as np

from gui.widgets.annotation_results_window import lines_from_transitions


def _db(rows, with_freq=True):
    c = sqlite3.connect(':memory:')
    if with_freq:
        c.execute('CREATE TABLE transitions (hmdb_id TEXT, ppm REAL, '
                  'intensity REAL, freq_mhz INTEGER)')
        c.executemany('INSERT INTO transitions VALUES (?,?,?,?)', rows)
    else:
        c.execute('CREATE TABLE transitions (hmdb_id TEXT, ppm REAL, intensity REAL)')
        c.executemany('INSERT INTO transitions VALUES (?,?,?)', rows)
    c.commit()
    return c


def test_triplet_lines_within_window_normalised():
    rows = [('HMDB1', 0.888, 1.0, 600), ('HMDB1', 0.900, 2.0, 600),
            ('HMDB1', 0.912, 1.0, 600),
            ('HMDB1', 3.50, 5.0, 600)]              # other resonance, out of window
    conn = _db(rows)
    lines = lines_from_transitions(conn, 'HMDB1', 0.87, 0.93, 600)
    ppms = [round(p, 3) for p, _ in lines]
    assert ppms == [0.888, 0.900, 0.912]
    # normalised to the strongest line (centre = 1.0)
    rel = dict((round(p, 3), i) for p, i in lines)
    assert abs(rel[0.900] - 1.0) < 1e-9
    assert abs(rel[0.888] - 0.5) < 1e-9


def test_freq_filter_picks_closest():
    rows = [('HMDB1', 0.90, 1.0, 500), ('HMDB1', 0.905, 1.0, 500),
            ('HMDB1', 0.900, 1.0, 600), ('HMDB1', 0.912, 1.0, 600)]
    conn = _db(rows)
    lines = lines_from_transitions(conn, 'HMDB1', 0.87, 0.93, 600)
    ppms = sorted(round(p, 3) for p, _ in lines)
    assert ppms == [0.900, 0.912]                  # only the 600 MHz rows


def test_empty_when_no_transitions_in_window():
    conn = _db([('HMDB1', 3.5, 1.0, 600)])
    assert lines_from_transitions(conn, 'HMDB1', 0.87, 0.93, 600) == []


def test_works_without_freq_column():
    conn = _db([('HMDB1', 0.895, 1.0), ('HMDB1', 0.905, 1.0)], with_freq=False)
    lines = lines_from_transitions(conn, 'HMDB1', 0.87, 0.93, 600)
    assert len(lines) == 2


def test_none_conn_returns_empty():
    assert lines_from_transitions(None, 'HMDB1', 0, 1, 600) == []
