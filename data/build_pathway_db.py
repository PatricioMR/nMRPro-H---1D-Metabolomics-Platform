# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# data/build_pathway_db.py
"""
One-time build script: harvests HMDB IDs from all project NMR databases,
fetches pathway data from the MII HMDB API (POST batch), and stores the
result in databases/hmdb_pathways.db.

Usage:
    python data/build_pathway_db.py

Config:
    API_KEY   — MII API key (expires 2026-11-20)
    BATCH_SZ  — IDs per POST request
    DB_DIR    — directory containing *.db NMR databases
    OUT_DB    — output pathway database path
    RESUME    — skip IDs already in DB (safe to re-run)
"""

import os
import re
import glob
import json
import time
import sqlite3
import requests
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY   = '81YFiiU1.IsjP29ju8C544CuvAfZE3z37WI71OKl0'
MII_URL   = f'http://35.184.189.38/api/hmdb/metabolites/pathways/?api-key={API_KEY}'
BATCH_SZ   = 10     # IDs per request (MII API max=10)
BATCH_DELAY = 2.0   # seconds between batches (avoid 429)
TIMEOUT    = 30     # HTTP timeout per batch
RESUME     = True   # skip IDs already in DB

_HERE   = Path(__file__).parent.parent
DB_DIR  = _HERE / 'databases'
OUT_DB  = DB_DIR / 'hmdb_pathways.db'

# ── HMDB ID helpers ───────────────────────────────────────────────────────────

def _norm_hmdb(raw: str):
    m = re.search(r'\d+', raw or '')
    return f'HMDB{int(m.group()):07d}' if m else None


def _collect_hmdb_ids(db_dir: Path) -> dict:
    """Return {hmdb_id: name} from all *.db NMR databases in db_dir."""
    ids = {}
    for db_path in sorted(glob.glob(str(db_dir / '*.db'))):
        if Path(db_path).name == 'hmdb_pathways.db':
            continue
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if 'metabolites' not in {r[0] for r in cur.fetchall()}:
                con.close()
                continue
            cur.execute('PRAGMA table_info(metabolites)')
            cols = {r[1] for r in cur.fetchall()}
            if 'hmdb_id' not in cols:
                con.close()
                continue
            name_col = 'name' if 'name' in cols else 'hmdb_id'
            cur.execute(f'SELECT hmdb_id, {name_col} FROM metabolites')
            for raw_id, name in cur.fetchall():
                norm = _norm_hmdb(str(raw_id or ''))
                if norm and norm not in ids:
                    ids[norm] = str(name or norm)
            con.close()
        except Exception as exc:
            print(f'  [skip] {Path(db_path).name}: {exc}')
    return ids


# ── Database ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS metabolites (
    hmdb_id TEXT PRIMARY KEY,
    name    TEXT
);

CREATE TABLE IF NOT EXISTS pathways (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    smpdb_id    TEXT DEFAULT '',
    description TEXT DEFAULT '',
    category    TEXT DEFAULT '',
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS metabolite_pathways (
    hmdb_id    TEXT    NOT NULL,
    pathway_id INTEGER NOT NULL,
    PRIMARY KEY (hmdb_id, pathway_id)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    hmdb_id    TEXT PRIMARY KEY,
    status     TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_mp_hmdb    ON metabolite_pathways(hmdb_id);
CREATE INDEX IF NOT EXISTS idx_mp_pathway ON metabolite_pathways(pathway_id);
"""


def _init_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    con.commit()
    return con


def _already_fetched(con, hmdb_id: str) -> bool:
    return con.execute(
        'SELECT 1 FROM fetch_log WHERE hmdb_id = ?', (hmdb_id,)
    ).fetchone() is not None


def _store_batch(con, results: list, name_map: dict):
    for entry in results:
        hmdb_id = entry.get('hmdb_id', '')
        name    = name_map.get(hmdb_id, hmdb_id)
        con.execute(
            'INSERT OR IGNORE INTO metabolites(hmdb_id, name) VALUES (?,?)',
            (hmdb_id, name))
        for pw in entry.get('pathways') or []:
            pw_name = pw.get('name') or '(unnamed)'
            smpdb   = pw.get('smpdb_id', '') or ''
            desc    = pw.get('description', '') or ''
            # Use API-provided category when available; fall back to heuristic
            api_cat = (pw.get('category') or pw.get('subject') or pw.get('pathway_type') or '').lower()
            if api_cat in ('drug', 'drug metabolism', 'drug action'):
                cat = 'drug'
            elif api_cat in ('disease', 'disorder'):
                cat = 'disease'
            elif api_cat in ('metabolic', 'metabolism', 'physiological', 'signaling'):
                cat = 'metabolic'
            else:
                # Heuristic from name + description
                _nd = (pw_name + ' ' + desc).lower()
                _DRUG_KW_B = {'drug', 'action pathway', 'pharmacokinetics',
                              'agonist', 'antagonist', 'antidepressant',
                              'antipsychotic', 'statin', 'antibiotic',
                              'antifungal', 'antiviral', 'xenobiotic',
                              'pharmaceutical', 'pharmacodynamic',
                              'pharmacological', 'medication',
                              'biotransformation', 'bioavailability',
                              'anaesthetic', 'anesthetic', 'analgesic',
                              'sedative', 'anticonvulsant', 'antiepileptic',
                              'antifungal', 'antimicrobial', 'antineoplastic',
                              'antihypertensive', 'antidiabetic', 'anticoagulant'}
                _DIS_KW_B  = {'deficiency', 'syndrome', 'disease', 'disorder',
                              'cancer', 'carcinoma', 'tumor', 'tumour',
                              'inborn', 'acidemia', 'aciduria', 'dystrophy'}
                if any(k in _nd for k in _DRUG_KW_B):
                    cat = 'drug'
                elif any(k in _nd for k in _DIS_KW_B):
                    cat = 'disease'
                else:
                    cat = 'metabolic'
            con.execute(
                'INSERT OR IGNORE INTO pathways(name, smpdb_id, description, category) '
                'VALUES (?,?,?,?)',
                (pw_name, smpdb, desc, cat))
            # Update category on existing rows (may have been inserted with empty cat)
            con.execute(
                'UPDATE pathways SET category = ? WHERE name = ? AND (category = "" OR category IS NULL)',
                (cat, pw_name))
            pw_id = con.execute(
                'SELECT id FROM pathways WHERE name = ?', (pw_name,)
            ).fetchone()[0]
            con.execute(
                'INSERT OR IGNORE INTO metabolite_pathways(hmdb_id, pathway_id) '
                'VALUES (?,?)',
                (hmdb_id, pw_id))
        con.execute(
            'INSERT OR REPLACE INTO fetch_log(hmdb_id, status) VALUES (?,?)',
            (hmdb_id, 'ok' if entry.get('pathways') else 'no_pathways'))


def _log_not_found(con, hmdb_ids: list):
    for hid in hmdb_ids:
        con.execute(
            'INSERT OR REPLACE INTO fetch_log(hmdb_id, status) VALUES (?,?)',
            (hid, 'not_found'))


# ── Fetch ─────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3

def _fetch_batch(hmdb_ids: list) -> tuple:
    """
    POST batch to MII API with 429 backoff.
    Returns (found_list, not_found_list) or raises on error.
    """
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            MII_URL,
            json={'hmdb_id': hmdb_ids},
            timeout=TIMEOUT)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get('Retry-After', 60))
            print(f'\n  [rate-limit] waiting {retry_after}s …', flush=True)
            time.sleep(retry_after + 5)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data.get('found', []), data.get('not_found', [])
    raise RuntimeError(f'Max retries exceeded after rate-limit')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'Building pathway DB: {OUT_DB}')
    print(f'Scanning NMR databases in: {DB_DIR}')

    name_map = _collect_hmdb_ids(DB_DIR)
    print(f'Found {len(name_map)} unique HMDB IDs.\n')
    if not name_map:
        print('No HMDB IDs found. Check DB_DIR.')
        return

    con = _init_db(OUT_DB)

    # Filter out already-fetched IDs if RESUME
    pending = [hid for hid in sorted(name_map) if not (RESUME and _already_fetched(con, hid))]
    n_skip  = len(name_map) - len(pending)
    print(f'Pending: {len(pending)}  |  Skipped (already fetched): {n_skip}\n')

    total    = len(pending)
    n_ok     = 0
    n_nf     = 0
    n_err    = 0

    for start in range(0, total, BATCH_SZ):
        batch   = pending[start:start + BATCH_SZ]
        end_idx = min(start + BATCH_SZ, total)
        print(f'[{start+1:4d}–{end_idx:4d} / {total}]  ', end='', flush=True)

        try:
            found, not_found = _fetch_batch(batch)
            _store_batch(con, found, name_map)
            _log_not_found(con, not_found)
            # Log any IDs the API silently omitted from both lists
            found_ids   = {e.get('hmdb_id') for e in found}
            missing_ids = [hid for hid in batch
                           if hid not in found_ids and hid not in not_found]
            _log_not_found(con, missing_ids)
            con.commit()

            n_pw_batch = sum(len(e.get('pathways') or []) for e in found)
            n_ok  += len(found)
            n_nf  += len(not_found)
            print(f'OK {len(found)} found, {len(not_found)} not_found, '
                  f'{n_pw_batch} pathway links')
        except Exception as exc:
            print(f'ERR {exc}')
            n_err += len(batch)
            # Log as error so RESUME skips them on retry
            for hid in batch:
                con.execute(
                    'INSERT OR REPLACE INTO fetch_log(hmdb_id, status) VALUES (?,?)',
                    (hid, 'error'))
            con.commit()

        time.sleep(BATCH_DELAY)

    con.close()

    print(f'\nDone.')
    print(f'  Fetched OK   : {n_ok}')
    print(f'  Not found    : {n_nf}')
    print(f'  Errors       : {n_err}')
    print(f'  Skipped      : {n_skip}')

    con2 = sqlite3.connect(str(OUT_DB))
    n_pw  = con2.execute('SELECT COUNT(*) FROM pathways').fetchone()[0]
    n_lnk = con2.execute('SELECT COUNT(*) FROM metabolite_pathways').fetchone()[0]
    con2.close()
    print(f'  Pathways     : {n_pw}')
    print(f'  Links        : {n_lnk}')
    print(f'\nOutput: {OUT_DB}')


if __name__ == '__main__':
    main()
