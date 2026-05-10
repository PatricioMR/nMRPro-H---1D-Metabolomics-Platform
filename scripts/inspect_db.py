import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "databases/HMDB_ValidationPipe.db"
conn = sqlite3.connect(db)
cur = conn.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)
for t in tables:
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"\n{t} ({n} rows): {cols}")
    rows = cur.execute(f"SELECT * FROM {t} LIMIT 2").fetchall()
    for r in rows:
        print(" ", r)
conn.close()
