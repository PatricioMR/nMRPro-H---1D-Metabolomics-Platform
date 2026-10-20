# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_db_queries.py
"""Test direct queries against the plasma (serum) HMDB database."""
import sqlite3
import pytest


@pytest.fixture(scope='module')
def db_conn(plasma_db_path):
    conn = sqlite3.connect(plasma_db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


class TestDBSchema:
    def test_tables_exist(self, db_conn):
        tables = {r[0] for r in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert 'metabolites' in tables
        assert 'multiplets' in tables or 'shifts_1h' in tables

    def test_metabolites_not_empty(self, db_conn):
        count = db_conn.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
        assert count > 0, 'metabolites table is empty'

    def test_metabolites_has_name(self, db_conn):
        cols = {r[1] for r in db_conn.execute(
            "PRAGMA table_info(metabolites)").fetchall()}
        assert 'name' in cols

    def test_multiplets_or_shifts_not_empty(self, db_conn):
        tables = {r[0] for r in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if 'multiplets' in tables:
            count = db_conn.execute("SELECT COUNT(*) FROM multiplets").fetchone()[0]
        else:
            count = db_conn.execute("SELECT COUNT(*) FROM shifts_1h").fetchone()[0]
        assert count > 0


class TestPPMQueries:
    """Query real ppm regions that should return hits in plasma."""

    COMMON_PLASMA_PPMS = [3.03, 1.33, 3.55, 2.03, 1.48, 3.21]

    @pytest.mark.parametrize('ppm', COMMON_PLASMA_PPMS)
    def test_db_returns_results_at_plasma_ppm(self, db_conn, ppm):
        """DB must have entries near common plasma metabolite ppm values."""
        tables = {r[0] for r in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        tol = 0.15

        if 'multiplets' in tables:
            rows = db_conn.execute(
                "SELECT met.name FROM multiplets m "
                "JOIN metabolites met ON m.hmdb_id = met.hmdb_id "
                "WHERE ABS(m.ppm_center - ?) <= ?",
                (ppm, tol)).fetchall()
        else:
            rows = db_conn.execute(
                "SELECT met.name FROM shifts_1h s "
                "JOIN metabolites met ON s.bmrb_id = met.bmrb_id "
                "WHERE ABS(s.ppm - ?) <= ?",
                (ppm, tol)).fetchall()

        assert len(rows) > 0, f'No DB results near {ppm} ppm (tol={tol})'

    def test_query_returns_results_in_aromatic_region(self, db_conn):
        tables = {r[0] for r in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if 'multiplets' in tables:
            count = db_conn.execute(
                "SELECT COUNT(*) FROM multiplets WHERE ppm_center BETWEEN 6.5 AND 9.0"
            ).fetchone()[0]
        else:
            count = db_conn.execute(
                "SELECT COUNT(*) FROM shifts_1h WHERE ppm BETWEEN 6.5 AND 9.0"
            ).fetchone()[0]
        assert count > 0, 'No aromatic region peaks in DB'

    def test_no_results_outside_nmr_range(self, db_conn):
        tables = {r[0] for r in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if 'multiplets' in tables:
            count = db_conn.execute(
                "SELECT COUNT(*) FROM multiplets WHERE ppm_center > 15.0"
            ).fetchone()[0]
        else:
            count = db_conn.execute(
                "SELECT COUNT(*) FROM shifts_1h WHERE ppm > 15.0"
            ).fetchone()[0]
        assert count == 0, 'DB has entries outside realistic 1H NMR range (>15 ppm)'
