"""
Download the full HMDB metabolite index (one-time) into a local SQLite so the
mammalian-biofluid flag never needs the API again.

Endpoint: /api/hmdb/metabolites/page/{n}/  → 100 metabolites/page, 2481 pages.
Per metabolite: hmdb_id, name, formula, mono_mass, biospecimen_location
(the mammalian-biofluid presence — Blood/Urine/CSF/...). No InChIKey in this
endpoint, so downstream mapping BMRB→HMDB is by normalized name + formula + mass.

Resumable: skips pages already stored. Index DB: hmdb_api_download/hmdb_index.db
"""
import json
import os
import socket
import time
import urllib.request
import urllib.error
import sqlite3

socket.setdefaulttimeout(40)
KEY = "81YFiiU1.IsjP29ju8C544CuvAfZE3z37WI71OKl0"
ROOT = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(ROOT, "hmdb_index.db")
PROG = os.path.join(ROOT, "_idx_progress.log")
ERR = os.path.join(ROOT, "_idx_errors.log")
API = "http://35.184.189.38/api/hmdb/metabolites/page/{}/?api-key=" + KEY


def log(p, m):
    with open(p, 'a', encoding='utf-8') as f:
        f.write(m + '\n')


def fetch(page, retries=3):
    for k in range(1, retries + 1):
        try:
            req = urllib.request.Request(API.format(page),
                                         headers={'User-Agent': 'nMRPipe/1.0'})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode('utf-8-sig'))
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, json.JSONDecodeError) as e:
            if k == retries:
                raise
            time.sleep(2 * k)
    return None


def main():
    con = sqlite3.connect(IDX)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS hmdb(
        hmdb_id TEXT PRIMARY KEY, name TEXT, name_norm TEXT, formula TEXT,
        mono_mass REAL, biofluids TEXT, n_biofluids INTEGER);
    CREATE TABLE IF NOT EXISTS pages(page INTEGER PRIMARY KEY);
    CREATE INDEX IF NOT EXISTS ix_norm ON hmdb(name_norm);
    CREATE INDEX IF NOT EXISTS ix_formula ON hmdb(formula);
    """)
    cur = con.cursor()
    done = {p for (p,) in cur.execute("SELECT page FROM pages")}

    # discover total_page from page 1
    first = fetch(1)
    total = int(first.get('total_page', 1))
    t0 = time.time()
    log(PROG, f"=== start {time.strftime('%H:%M:%S')} total_pages={total} "
              f"already_done={len(done)} ===")

    def store(page, payload):
        for m in payload.get('metabolites', []):
            bf = m.get('biospecimen_location') or []
            nm = m.get('name') or ''
            cur.execute(
                "INSERT OR REPLACE INTO hmdb VALUES(?,?,?,?,?,?,?)",
                (m.get('hmdb_id'), nm, nm.lower().strip(), m.get('moldb_formula'),
                 float(m.get('moldb_mono_mass') or 0) or None,
                 json.dumps(bf), len(bf)))
        cur.execute("INSERT OR IGNORE INTO pages VALUES(?)", (page,))

    if 1 not in done:
        store(1, first)
        con.commit()

    for page in range(2, total + 1):
        if page in done:
            continue
        try:
            payload = fetch(page)
        except Exception as e:
            log(ERR, f"page {page}\t{e}")
            continue
        if not payload:
            log(ERR, f"page {page}\tempty")
            continue
        store(page, payload)
        if page % 25 == 0 or page == total:
            con.commit()
            el = time.time() - t0
            n = cur.execute("SELECT COUNT(*) FROM hmdb").fetchone()[0]
            r = (page - len(done)) / el if el else 0
            eta = (total - page) / r / 60 if r else 0
            log(PROG, f"[{page}/{total}] metabolites={n} | {r:.1f} pg/s ETA {eta:.0f} min")
        time.sleep(0.05)

    con.commit()
    n = cur.execute("SELECT COUNT(*) FROM hmdb").fetchone()[0]
    nbf = cur.execute("SELECT COUNT(*) FROM hmdb WHERE n_biofluids>0").fetchone()[0]
    np = cur.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    con.close()
    log(PROG, f"=== done metabolites={n} with_biofluid={nbf} pages={np}/{total} "
              f"in {(time.time()-t0)/60:.1f} min ===")
    print(f"hmdb_index.db: {n} metabolites, {nbf} with biofluid, {np}/{total} pages")


if __name__ == '__main__':
    main()
