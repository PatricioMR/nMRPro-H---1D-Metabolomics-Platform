"""
Apply HMDB origin classification to HMDB_Metabolomics_v01.db.

Purpose: the DB analyses mammalian biological material — drugs, plant/dietary-only
and synthetic compounds are out of scope. This:
  1. adds origin columns to `metabolites` (transparency / re-filtering),
  2. writes a review report (_origin_report.csv): kept vs excluded + reason,
  3. copies every excluded metabolite (and all its spectra rows) into an audit DB
     HMDB_excluded_v01.db — nothing is lost,
  4. deletes excluded rows from the working DB so they cannot match during
     annotation.

Keep rule: HMDB tags it Endogenous. An endogenous metabolite that is ALSO a drug
is KEPT (e.g. caffeine) — only NON-endogenous compounds (plant/food-only,
synthetic-only, drug-only) are removed. Metabolites with no ontology record are
kept as 'unknown' (never deleted on absence of evidence — CLAUDE.md: don't
fabricate, don't hide). role_drug stays as an informational column.
"""
import json
import glob
import os
import sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
ONTO = os.path.join(ROOT, 'ontology')
DBDIR = os.path.join(os.path.dirname(ROOT), 'databases')
DB = os.path.join(DBDIR, 'HMDB_Metabolomics_v01.db')
AUDIT = os.path.join(DBDIR, 'HMDB_excluded_v01.db')
REPORT = os.path.join(ROOT, '_origin_report.csv')

SPECTRA_TABLES = ('shifts_1h', 'sticks_1h', 'shifts_13c', 'hsqc', 'transitions')


def load_class():
    out = {}
    for fp in glob.glob(os.path.join(ONTO, '*.json')):
        d = json.load(open(fp, encoding='utf-8'))
        out[d['hmdb_id']] = d
    return out


def main():
    cls = load_class()
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 1. add origin columns (idempotent)
    cols = {r[1] for r in cur.execute("PRAGMA table_info(metabolites)")}
    for c, t in [('is_endogenous', 'INTEGER'), ('is_food', 'INTEGER'),
                 ('is_microbial', 'INTEGER'), ('is_synthetic', 'INTEGER'),
                 ('role_drug', 'INTEGER'), ('primary_origin', 'TEXT'),
                 ('keep_mammalian', 'INTEGER'), ('exclude_reason', 'TEXT'),
                 ('source_hilevel', 'TEXT')]:
        if c not in cols:
            cur.execute(f"ALTER TABLE metabolites ADD COLUMN {c} {t}")

    ids = [r[0] for r in cur.execute("SELECT hmdb_id FROM metabolites")]
    keep, excl, unknown = [], [], []
    for hid in ids:
        c = cls.get(hid)
        if c is None:
            cur.execute("UPDATE metabolites SET primary_origin='unknown',"
                        " keep_mammalian=1, exclude_reason='no ontology record'"
                        " WHERE hmdb_id=?", (hid,))
            unknown.append(hid)
            continue
        # KEEP iff endogenous (drug status is informational, not exclusionary).
        keep_it = bool(c['is_endogenous'])
        if keep_it:
            primary, reason = 'endogenous', ''
        elif c['is_synthetic'] or c.get('role_industrial') or c.get('role_toxic'):
            primary, reason = 'synthetic/xenobiotic', 'not endogenous (synthetic)'
        elif c['role_drug']:
            primary, reason = 'drug', 'not endogenous (drug only)'
        elif c['is_microbial']:
            primary, reason = 'microbial', 'not endogenous (microbial)'
        elif c['is_food']:
            primary, reason = 'plant/food', 'not endogenous (dietary)'
        else:
            primary, reason = 'unknown', 'no Endogenous tag'
        c = dict(c, primary_origin=primary, exclude_reason=reason,
                 keep_mammalian=keep_it)
        cur.execute(
            "UPDATE metabolites SET is_endogenous=?,is_food=?,is_microbial=?,"
            "is_synthetic=?,role_drug=?,primary_origin=?,keep_mammalian=?,"
            "exclude_reason=?,source_hilevel=? WHERE hmdb_id=?",
            (int(c['is_endogenous']), int(c['is_food']), int(c['is_microbial']),
             int(c['is_synthetic']), int(c['role_drug']), c['primary_origin'],
             int(keep_it), c['exclude_reason'],
             ', '.join(c['source_hilevel']), hid))
        (keep if keep_it else excl).append((hid, c))
    con.commit()

    # 2. report
    import csv
    with open(REPORT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['hmdb_id', 'name', 'decision', 'primary_origin',
                    'exclude_reason', 'is_endogenous', 'role_drug',
                    'is_synthetic', 'source_hilevel'])
        for hid in ids:
            nm = cur.execute("SELECT name FROM metabolites WHERE hmdb_id=?",
                             (hid,)).fetchone()[0]
            c = cls.get(hid)
            if c is None:
                w.writerow([hid, nm, 'KEEP', 'unknown', 'no ontology record',
                            '', '', '', ''])
            else:
                w.writerow([hid, nm,
                            'KEEP' if c['keep_mammalian'] else 'EXCLUDE',
                            c['primary_origin'], c['exclude_reason'],
                            c['is_endogenous'], c['role_drug'], c['is_synthetic'],
                            ', '.join(c['source_hilevel'])])

    # 3. audit DB with excluded rows (full copy, reversible)
    excl_ids = [hid for hid, _ in excl]
    if os.path.exists(AUDIT):
        os.remove(AUDIT)
    if excl_ids:
        cur.execute(f"ATTACH DATABASE ? AS audit", (AUDIT,))
        # recreate schema in audit by copying structure + excluded rows
        ph = ','.join('?' * len(excl_ids))
        cur.execute(f"CREATE TABLE audit.metabolites AS "
                    f"SELECT * FROM metabolites WHERE hmdb_id IN ({ph})", excl_ids)
        for t in SPECTRA_TABLES:
            cur.execute(f"CREATE TABLE audit.{t} AS "
                        f"SELECT * FROM {t} WHERE hmdb_id IN ({ph})", excl_ids)
        con.commit()
        # 4. delete excluded from working DB
        for t in SPECTRA_TABLES:
            cur.execute(f"DELETE FROM {t} WHERE hmdb_id IN ({ph})", excl_ids)
        cur.execute(f"DELETE FROM metabolites WHERE hmdb_id IN ({ph})", excl_ids)
        con.commit()
        cur.execute("DETACH DATABASE audit")

    con.execute("VACUUM")
    # summary
    n_met = con.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    n_sig = con.execute("SELECT COUNT(*) FROM shifts_1h").fetchone()[0]
    con.close()
    print(f"classified={len(cls)}  kept={len(keep)}  excluded={len(excl)}  "
          f"unknown(kept)={len(unknown)}")
    print(f"working DB now: metabolites={n_met}  shifts_1h={n_sig}")
    print(f"excluded audit DB: {AUDIT}  ({len(excl_ids)} metabolites)")
    print(f"review report: {REPORT}")
    # show excluded breakdown
    from collections import Counter
    by = Counter(c['primary_origin'] for _, c in excl)
    print("excluded by origin:", dict(by))


if __name__ == '__main__':
    main()
