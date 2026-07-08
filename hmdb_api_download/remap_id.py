"""
Make the metabolite primary id the HMDB id where known, so pathway analysis works.

The annotation pipeline keys a shifts_1h DB by bmrb_id and aliases it `as hmdb_id`
in results; pathway lookup needs that to be a real HMDB id. So set
bmrb_id = hmdb_id (when known), keeping the original BMRB id in bmse_id for
provenance. Applied consistently across metabolites/shifts_1h/transitions, then
the two derived DBs are rebuilt by split_dbs.
"""
import os
import sqlite3

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'databases', 'BMRB_Metabolomics_v01.db')


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(metabolites)")}
    if 'bmse_id' not in cols:
        cur.execute("ALTER TABLE metabolites ADD COLUMN bmse_id TEXT")
    # provenance: original BMRB id
    cur.execute("UPDATE metabolites SET bmse_id=bmrb_id WHERE bmse_id IS NULL")

    # old bmrb(bmse) -> new id (hmdb_id when present)
    remap = {b: (h or b) for b, h in
             cur.execute("SELECT bmrb_id,hmdb_id FROM metabolites")}
    changed = {b: n for b, n in remap.items() if b != n}
    # guard: new ids must stay unique
    if len(set(remap.values())) != len(remap):
        from collections import Counter
        dups = [k for k, v in Counter(remap.values()).items() if v > 1]
        print(f"ABORT: id collisions would occur: {dups[:10]}")
        con.close()
        return
    for tbl in ('metabolites', 'shifts_1h', 'transitions'):
        for old, new in changed.items():
            cur.execute(f"UPDATE {tbl} SET bmrb_id=? WHERE bmrb_id=?", (new, old))
    con.commit()
    n = cur.execute("SELECT COUNT(*) FROM metabolites WHERE bmrb_id LIKE 'HMDB%'").fetchone()[0]
    tot = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    con.close()
    print(f"remapped bmrb_id -> hmdb_id for {len(changed)} metabolites; "
          f"now {n}/{tot} keyed by HMDB id")


if __name__ == '__main__':
    main()
