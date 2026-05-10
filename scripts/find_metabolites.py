import sqlite3, sys

db = sys.argv[1] if len(sys.argv) > 1 else "databases/HMDB_Total_v01.db"
conn = sqlite3.connect(db)
cur = conn.cursor()

targets = ["Taurine","Caffeine","Glucose","Arabinose","Cysteine","Glutamic","Mannitol","Itaconic"]
for t in targets:
    rows = cur.execute(
        "SELECT bmrb_id, name, hmdb_id FROM metabolites WHERE name LIKE ? LIMIT 5",
        (f"%{t}%",)
    ).fetchall()
    print(f"\n{t}:")
    for r in rows:
        bmrb, name, hmdb = r
        shifts = cur.execute(
            "SELECT ppm, multiplicity, j_coupling FROM shifts_1h WHERE bmrb_id=? ORDER BY ppm DESC",
            (bmrb,)
        ).fetchall()
        print(f"  {bmrb} | {name} | {hmdb} | shifts: {shifts}")
conn.close()
