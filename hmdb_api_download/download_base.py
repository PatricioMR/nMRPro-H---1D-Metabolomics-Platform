"""
Download HMDB base metabolite records for origin classification.

The base endpoint carries the strongest origin signals (dropped from the slim
NMR files): `status`, `biospecimen_normal/abnormal` (mammalian biofluid
presence), `chemical_taxonomy` (ClassyFire — structure-derived class, the best
discriminator for plant phytochemicals vs core metabolites), and external links
(kegg/foodb/drugbank/biocyc).

Saves a slim per-metabolite JSON (origin fields only — not the huge publication
lists) to hmdb_api_download/base/{HMDB_ID}.json. Resumable.
"""
import json
import os
import socket
import sqlite3
import time
import urllib.request
import urllib.error

socket.setdefaulttimeout(35)
KEY = "81YFiiU1.IsjP29ju8C544CuvAfZE3z37WI71OKl0"
ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(ROOT), "databases", "HMDB_Metabolomics_v01.db")
OUT = os.path.join(ROOT, "base")
os.makedirs(OUT, exist_ok=True)
ERRLOG = os.path.join(ROOT, "_base_errors.log")
PROG = os.path.join(ROOT, "_base_progress.log")
API = "http://35.184.189.38/api/hmdb/metabolites/{}/?api-key=" + KEY


def log(p, m):
    with open(p, 'a', encoding='utf-8') as f:
        f.write(m + '\n')


def fetch(hid, retries=2):
    for k in range(1, retries + 1):
        try:
            req = urllib.request.Request(API.format(hid),
                                         headers={'User-Agent': 'nMRPipe/1.0'})
            with urllib.request.urlopen(req, timeout=35) as r:
                return json.loads(r.read().decode('utf-8-sig'))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == retries:
                raise
            time.sleep(2 * k)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if k == retries:
                raise
            time.sleep(2 * k)
    return None


def slim(d):
    ct = d.get('chemical_taxonomy', {}) or {}
    el = d.get('external_links', {}) or {}
    def names(key):
        return sorted({b.get('biospecimen') for b in (d.get(key) or [])
                       if isinstance(b, dict) and b.get('biospecimen')})
    return {
        'hmdb_id': d.get('hmdb_id'),
        'name': d.get('name'),
        'status': d.get('status'),
        'biospecimen_normal': names('biospecimen_normal'),
        'biospecimen_abnormal': names('biospecimen_abnormal'),
        'taxonomy': {
            'kingdom': ct.get('kingdom'), 'superclass': ct.get('superclass'),
            'class': ct.get('class'), 'subclass': ct.get('subclass'),
            'direct_parent': ct.get('direct_parent'),
        },
        'kegg_id': el.get('kegg_id'),
        'foodb_id': el.get('foodb_id'),
        'drugbank_id': el.get('drugbank_id'),
        'biocyc_id': el.get('biocyc_id'),
        'chebi_id': el.get('chebi_id'),
    }


def main():
    con = sqlite3.connect(DB)
    ids = [r[0] for r in con.execute(
        "SELECT hmdb_id FROM metabolites WHERE hmdb_id LIKE 'HMDB%' ORDER BY hmdb_id")]
    con.close()
    total = len(ids)
    t0 = time.time()
    ok = err = skip = 0
    log(PROG, f"=== start {time.strftime('%H:%M:%S')} — {total} metabolites ===")
    for i, hid in enumerate(ids, 1):
        dst = os.path.join(OUT, f"{hid}.json")
        if os.path.exists(dst):
            skip += 1
            continue
        try:
            d = fetch(hid)
        except Exception as e:
            err += 1
            log(ERRLOG, f"{hid}\t{e}")
            continue
        if not d:
            log(ERRLOG, f"{hid}\tNO_RECORD")
            continue
        json.dump(slim(d), open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
        ok += 1
        if i % 10 == 0 or i == total:
            el = time.time() - t0
            r = i / el if el else 0
            log(PROG, f"[{i}/{total}] ok={ok} err={err} skip={skip} "
                      f"| {r:.2f}/s ETA {(total-i)/r/60:.1f} min" if r else f"[{i}/{total}]")
        time.sleep(0.1)
    log(PROG, f"=== done ok={ok} err={err} skip={skip} in {(time.time()-t0)/60:.1f} min ===")


if __name__ == '__main__':
    main()
