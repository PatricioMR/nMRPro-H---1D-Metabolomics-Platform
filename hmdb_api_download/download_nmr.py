# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Download all HMDB NMR spectra for the metabolites present in HMDB_General.db.

Source: MII / HMDB API  http://35.184.189.38/api/hmdb/metabolites/{id}/spectra/
Each metabolite JSON keeps only NMR-relevant content (formula, smiles, the full
`nmr` spectra list: ppm/intensity peak lists, nucleus, frequency, solvent).

NOTE: the HMDB API NMR peaks carry ppm+intensity ONLY — no per-signal proton
count or multiplicity. This download is raw NMR data; it does NOT by itself fix
the `protons` column (nH must still be derived from structure/assignment).

Resumable: already-saved metabolites are skipped. Failures logged to _errors.log.
"""
import json
import os
import sqlite3
import time
import urllib.request
import urllib.error

KEY = "81YFiiU1.IsjP29ju8C544CuvAfZE3z37WI71OKl0"
ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(ROOT), "databases", "HMDB_General.db")
OUT = os.path.join(ROOT, "nmr")
os.makedirs(OUT, exist_ok=True)
ERRLOG = os.path.join(ROOT, "_errors.log")
PROGRESS = os.path.join(ROOT, "_progress.log")

API = "http://35.184.189.38/api/hmdb/metabolites/{}/spectra/?api-key=" + KEY


def log(path, msg):
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def fetch(hmdb_id, retries=3):
    url = API.format(hmdb_id)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NMRPro-H/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8-sig")
            return json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code == 404:
                return None  # no spectra record — not an error
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
        except json.JSONDecodeError:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return None


def main():
    conn = sqlite3.connect(DB)
    ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT hmdb_id FROM metabolites "
        "WHERE hmdb_id IS NOT NULL ORDER BY hmdb_id")]
    conn.close()

    total = len(ids)
    done = ok = no_nmr = err = skip = 0
    t0 = time.time()
    log(PROGRESS, f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')} — {total} metabolites ===")

    for i, hid in enumerate(ids, 1):
        dst = os.path.join(OUT, f"{hid}.json")
        if os.path.exists(dst):
            skip += 1
            done += 1
            continue
        try:
            d = fetch(hid)
        except Exception as e:
            err += 1
            log(ERRLOG, f"{hid}\tFETCH_FAIL\t{e}")
            done += 1
            continue

        if not d:
            no_nmr += 1
            log(ERRLOG, f"{hid}\tNO_RECORD")
            done += 1
            continue

        nmr = d.get("nmr") or []
        slim = {
            "hmdb_id": d.get("hmdb_id", hid),
            "name": d.get("name"),
            "moldb_formula": d.get("moldb_formula"),
            "moldb_smiles": d.get("moldb_smiles"),
            "n_nmr_spectra": len(nmr),
            "nmr": nmr,
        }
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False)
        if nmr:
            ok += 1
        else:
            no_nmr += 1
        done += 1

        if i % 10 == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (total - i) / rate if rate else 0
            log(PROGRESS,
                f"[{i}/{total}] ok={ok} no_nmr={no_nmr} err={err} skip={skip} "
                f"| {rate:.1f}/s ETA {eta/60:.1f} min")
        time.sleep(0.15)  # polite throttle

    log(PROGRESS,
        f"=== done ok={ok} no_nmr={no_nmr} err={err} skip={skip} "
        f"total={total} in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
