"""
Derive two filtered DBs from BMRB_Metabolomics_v01.db (offline origin classified):

  • BMRB_Metabolomics_General_v01.db   — mammalian/endogenous + unknown
        (keep_mammalian IN '1','?')  → excludes only the clear plant/drug/xenobiotic
  • BMRB_Metabolomics_Mammalian_v01.db — mammalian/endogenous only
        (keep_mammalian = '1')

Each keeps every spectral table (metabolites, shifts_1h, transitions) for the
selected bmrb_ids only. bmrb_id stays the real BMRB id (app queries by it).
"""
import os
import shutil
import sqlite3

DBDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'databases')
SRC = os.path.join(DBDIR, 'BMRB_Metabolomics_v01.db')
TABLES_WITH_ID = ('metabolites', 'shifts_1h', 'transitions')


def make(dst, keep_values):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(SRC, dst)
    con = sqlite3.connect(dst)
    cur = con.cursor()
    ph = ','.join('?' * len(keep_values))
    drop_ids = [r[0] for r in cur.execute(
        f"SELECT bmrb_id FROM metabolites WHERE keep_mammalian IS NULL "
        f"OR keep_mammalian NOT IN ({ph})", keep_values)]
    if drop_ids:
        dph = ','.join('?' * len(drop_ids))
        for tbl in TABLES_WITH_ID:
            cur.execute(f"DELETE FROM {tbl} WHERE bmrb_id IN ({dph})", drop_ids)
    con.commit()
    con.execute("VACUUM")
    nm = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    ns = cur.execute("SELECT COUNT(*) FROM shifts_1h").fetchone()[0]
    orph = cur.execute("SELECT COUNT(*) FROM shifts_1h WHERE bmrb_id NOT IN "
                       "(SELECT bmrb_id FROM metabolites)").fetchone()[0]
    con.close()
    print(f"{os.path.basename(dst)}: metabolites={nm} shifts_1h={ns} orphan={orph}")


def main():
    make(os.path.join(DBDIR, 'BMRB_Metabolomics_General_v01.db'), ('1', '?'))
    make(os.path.join(DBDIR, 'BMRB_Metabolomics_Mammalian_v01.db'), ('1',))


if __name__ == '__main__':
    main()
