# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# data/dedupe_metabolites.py
"""Collapse duplicate metabolite entries that 1H NMR cannot tell apart.

Reference databases built from different sources carry the same compound twice
under different names — ``Alanine`` (BMSE, unspecified stereochemistry) and
``L-alanine`` (HMDB). Their measured couplings disagree (9.00 Hz vs 7.20 Hz for
alanine), so the annotation search scores both against the same signal and the
wrong one can win, reporting a ΔJ error of ~1.8 Hz on a perfectly good peak.

Two entries are collapsed only when 1H NMR in an achiral solvent genuinely
cannot separate them:

  * the same compound named with and without a stereodescriptor
    (``Acetylcarnitine`` / ``Acetyl-L-carnitine``);
  * enantiomers (``D-Methionine`` / ``L-methionine``) — mirror images have
    identical spectra outside a chiral environment.

Everything else is left alone. Diastereomers (D-Glucose / D-Galactose),
constitutional isomers (Alanine / Beta-alanine) and cis/trans pairs
(fumarate / maleate) have genuinely different spectra and must both stay: they
share an InChIKey skeleton but not a spectrum.

Survivor preference, in order: the entry with more measured shifts (never
discard measurements), then the L- form, then the one carrying both an HMDB and
a BMRB identifier. Shift count leads because the unspecified-stereochemistry
entry is often the better characterised one — for 3-hydroxybutyric acid the
generic entry holds four shifts against three for the (S) form.

Run as a script for a dry run; pass ``--apply`` to write, which timestamps a
backup of every database it touches.

References
----------
Stereochemical equivalence in achiral media: Mislow, K. & Raban, M.,
    "Stereoisomeric relationships of groups in molecules",
    Topics in Stereochemistry 1 (1967) 1-38.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass, field

# Stereodescriptors that do not change the 1H spectrum in an achiral solvent.
# Deliberately excludes alpha/beta, cis/trans, myo/muco/scyllo and n/iso/neo —
# those denote different connectivity or diastereomers with distinct spectra.
#
# Bare tokens are restricted to the sugar/amino-acid prefixes. 'R' and 'S' count
# only in parenthesised form: a bare 'S-' in a metabolite name is sulfur
# attachment (S-adenosylmethionine), not a configuration.
_BARE_STEREO  = frozenset({'d', 'l', 'dl', 'ld'})
_PAREN_STEREO = frozenset({'r', 's', 'rs', 'sr', '+', '-', '+-', '±'})

# Suffix pairs that name the same species (free acid vs conjugate base). The
# NMR entry describes the same solution species at physiological pH.
_ACID_SALT = (
    ('icacid', 'ate'),
    ('ic', 'ate'),
)


def _is_stereo_token(tok: str) -> bool:
    """True when a hyphen-delimited fragment is only a configuration label."""
    t = tok.strip().lower()
    if not t:
        return False
    m = re.fullmatch(r'\((.+)\)', t)
    if m:
        return m.group(1) in _PAREN_STEREO
    return t in _BARE_STEREO


def _drop_stereo_tokens(name: str) -> str:
    """Remove configuration labels wherever they sit in the name.

    They are not always leading: 'Acetyl-L-carnitine' and 'N-Acetyl-L-Alanine'
    carry theirs in the middle.
    """
    parts = re.split(r'[\-\s]+', name)
    kept = [p for p in parts if not _is_stereo_token(p)]
    return ' '.join(kept) if kept else name


def stereo_normalized_name(name) -> str:
    """Key under which two names denote the same NMR species.

    Lower-cases, drops leading stereodescriptors and punctuation, and maps the
    free acid onto its conjugate base so 'L-aspartic acid' and 'D-Aspartate'
    land on the same key.
    """
    n = str(name or '').strip().lower()
    n = _drop_stereo_tokens(n)
    n = re.sub(r'[^a-z0-9]', '', n)
    for acid, salt in _ACID_SALT:
        if n.endswith(acid):
            n = n[: -len(acid)] + salt
            break
    return n


def _skeleton(inchikey) -> str | None:
    """First InChIKey block — the connectivity, ignoring stereochemistry."""
    ik = str(inchikey or '').strip()
    return ik.split('-')[0] if ik else None


@dataclass
class Entry:
    """One metabolite row, reduced to what the decision needs."""
    key: str                    # identifier used for display and grouping
    name: str
    inchikey: str | None = None
    formula: str | None = None
    n_shifts: int = 0
    has_both_ids: bool = False
    # {column: value} for every identifier column the row actually fills. These
    # databases populate hmdb_id and bmrb_id inconsistently — one holds HMDB
    # accessions in its bmrb_id column — so deletion matches on all of them
    # rather than trusting one guessed column and orphaning the child rows.
    ids: dict = field(default_factory=dict)


@dataclass
class Collapse:
    """A group of entries that describe one NMR species."""
    survivor: Entry
    removed: list[Entry] = field(default_factory=list)
    reason: str = ''


def _is_l_form(name: str) -> bool:
    """True for the L (equivalently (S)) configuration.

    A bare leading 'S-' is sulfur attachment, not a configuration, so only the
    parenthesised form counts.
    """
    return bool(re.match(r'^(l|\(s\))[\-\s]', str(name or '').strip().lower()))


def _survivor_rank(e: Entry):
    """Sort key — the best survivor sorts first.

    Measurements outrank nomenclature: dropping the entry with more assigned
    shifts would lose data that no renaming can recover.
    """
    return (-e.n_shifts,
            0 if _is_l_form(e.name) else 1,
            0 if e.has_both_ids else 1,
            e.name.lower())


def _compatible(a: Entry, b: Entry) -> bool:
    """Guard the name rule with structural evidence when it is available.

    Same normalized name is necessary but not sufficient: require the same
    connectivity skeleton (or, when no InChIKey is stored, the same molecular
    formula). Entries with neither are never collapsed.
    """
    sk_a, sk_b = _skeleton(a.inchikey), _skeleton(b.inchikey)
    if sk_a and sk_b:
        return sk_a == sk_b
    fa = (a.formula or '').replace(' ', '').upper()
    fb = (b.formula or '').replace(' ', '').upper()
    if fa and fb:
        return fa == fb
    return False


def plan_collapses(entries) -> list[Collapse]:
    """Group entries that 1H NMR cannot distinguish; pick each survivor.

    Pure function — takes and returns plain data so the rule can be tested
    without a database.
    """
    by_name: dict[str, list[Entry]] = {}
    for e in entries:
        by_name.setdefault(stereo_normalized_name(e.name), []).append(e)

    plans: list[Collapse] = []
    for norm, group in sorted(by_name.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=_survivor_rank)
        survivor = ordered[0]
        removed = [e for e in ordered[1:] if _compatible(survivor, e)]
        if not removed:
            continue
        why = ('enantiomer / stereodescriptor variant of the same compound'
               if any(_is_l_form(e.name) != _is_l_form(survivor.name)
                      for e in removed)
               else 'same compound under a second name')
        plans.append(Collapse(survivor=survivor, removed=removed, reason=why))
    return plans


# ── Database access ───────────────────────────────────────────────────────────

# Identifier columns these schemas use, in the order they are preferred.
ID_COLUMNS = ('hmdb_id', 'bmrb_id')

def _metabolite_columns(con) -> list[str]:
    return [c[1] for c in con.execute('PRAGMA table_info(metabolites)')]


def _primary_key(cols: list[str]) -> str | None:
    """Identifier the child tables key their rows by."""
    for candidate in ('bmrb_id', 'hmdb_id'):
        if candidate in cols:
            return candidate
    return None


def _child_tables(con, key: str) -> list[str]:
    """Every table other than metabolites that carries any identifier column.

    Not just ``key``: HMDB_Metabolomics_v01 stores sticks_1h and hsqc keyed on
    hmdb_id while the metabolite rows are read through bmrb_id. Missing them
    leaves their rows orphaned when a metabolite is deleted.
    """
    out = []
    for (t,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT IN ('metabolites','sqlite_sequence')"):
        cols = [c[1] for c in con.execute(f'PRAGMA table_info("{t}")')]
        if any(c in cols for c in ID_COLUMNS):
            out.append(t)
    return out


def _count_shifts(con, ids: dict) -> int:
    """Shifts reachable from any identifier this row carries."""
    shift_cols = [c[1] for c in con.execute('PRAGMA table_info(shifts_1h)')]
    total = 0
    for col, val in ids.items():
        if col in shift_cols:
            total = max(total, con.execute(
                f'SELECT COUNT(*) FROM shifts_1h WHERE "{col}" = ?',
                (val,)).fetchone()[0])
    return int(total)


def read_entries(con, key: str, cols: list[str]) -> list[Entry]:
    con.row_factory = sqlite3.Row
    sel = [key, 'name']
    for opt in ('inchikey', 'formula') + ID_COLUMNS:
        if opt in cols and opt not in sel:
            sel.append(opt)
    rows = con.execute(
        f'SELECT {", ".join(chr(34) + c + chr(34) for c in sel)} '
        'FROM metabolites').fetchall()
    entries = []
    for r in rows:
        keys = r.keys()
        ids = {c: r[c] for c in ID_COLUMNS if c in keys and r[c]}
        if not ids:
            continue
        entries.append(Entry(
            key=r[key] if r[key] else next(iter(ids.values())),
            name=r['name'] or '',
            inchikey=r['inchikey'] if 'inchikey' in keys else None,
            formula=r['formula'] if 'formula' in keys else None,
            n_shifts=_count_shifts(con, ids),
            has_both_ids=len(ids) > 1,
            ids=ids,
        ))
    return entries


def plan_database(path: str):
    """Return (key, child_tables, plans) for one database, without writing."""
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        cols = _metabolite_columns(con)
        key = _primary_key(cols)
        if key is None:
            return None, [], []
        children = _child_tables(con, key)
        if 'shifts_1h' not in children:
            return None, [], []
        return key, children, plan_collapses(read_entries(con, key, cols))
    finally:
        con.close()


def apply_plans(path: str, key: str, children: list[str],
                plans: list[Collapse], stamp: str) -> dict:
    """Delete the redundant entries, after backing the database up.

    One transaction per database: either every duplicate goes or none does.
    Deletion matches on every identifier the removed row carries, so no child
    row is left orphaned — but never on a value the survivor also uses.
    """
    backup = f'{path}.bak_{stamp}'
    shutil.copy2(path, backup)

    survivor_ids = {(c, v) for p in plans for c, v in p.survivor.ids.items()}
    targets = [(c, v)
               for p in plans for e in p.removed
               for c, v in e.ids.items()
               if (c, v) not in survivor_ids]

    counts: dict[str, int] = {}
    con = sqlite3.connect(path)
    try:
        con.execute('BEGIN')
        for table in children + ['metabolites']:
            table_cols = [c[1] for c in con.execute(f'PRAGMA table_info("{table}")')]
            n = 0
            for col, val in targets:
                if col not in table_cols:
                    continue
                n += con.execute(
                    f'DELETE FROM "{table}" WHERE "{col}" = ?', (val,)).rowcount
            if n:
                counts[table] = n
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return {'backup': backup, 'deleted': counts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('databases', nargs='*', help='.db files (default: databases/*.db)')
    ap.add_argument('--apply', action='store_true',
                    help='write the changes (default: dry run)')
    args = ap.parse_args(argv)

    paths = args.databases
    if not paths:
        import glob
        paths = sorted(glob.glob(os.path.join('databases', '*.db')))

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    total = 0
    for path in paths:
        key, children, plans = plan_database(path)
        if key is None:
            print(f'{path}: no metabolite/shift schema — skipped')
            continue
        if not plans:
            print(f'{path}: no duplicates')
            continue
        print(f'\n{path}  (key={key}, child tables={children})')
        for p in plans:
            print(f'  keep   {p.survivor.name!r} [{p.survivor.key}] '
                  f'{p.survivor.n_shifts} shifts')
            for e in p.removed:
                print(f'  remove {e.name!r} [{e.key}] {e.n_shifts} shifts'
                      f'   — {p.reason}')
                total += 1
        if args.apply:
            res = apply_plans(path, key, children, plans, stamp)
            print(f'  applied: {res["deleted"]}  (backup {res["backup"]})')

    print(f'\n{total} entr{"y" if total == 1 else "ies"} '
          f'{"removed" if args.apply else "would be removed"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
