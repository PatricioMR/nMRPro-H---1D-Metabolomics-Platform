"""
Download HMDB ontology for the metabolites in HMDB_Metabolomics_v01.db and
extract ONLY the origin classification (discard the bulk: disease/location/food
sub-lists). Lets us separate endogenous metabolites from drugs / synthetic /
dietary xenobiotics.

Two HMDB ontology roots are used:
  • disposition.source  → high-level origin terms (Endogenous, Exogenous, Food,
    Plant, Animal, Microbial, Synthetic, Toxin/Pollutant, Personal care product…)
  • role                → functional roles incl. Drug / Pharmaceutical /
    Industrial / Toxin / Pesticide

Output: hmdb_api_download/ontology/{HMDB_ID}.json (tiny: id + flag terms).
Resumable. Errors → _onto_errors.log, progress → _onto_progress.log.
"""
import json
import os
import sqlite3
import time
import socket
import urllib.request
import urllib.error

socket.setdefaulttimeout(35)

KEY = "81YFiiU1.IsjP29ju8C544CuvAfZE3z37WI71OKl0"
ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(ROOT), "databases", "HMDB_Metabolomics_v01.db")
OUT = os.path.join(ROOT, "ontology")
os.makedirs(OUT, exist_ok=True)
ERRLOG = os.path.join(ROOT, "_onto_errors.log")
PROG = os.path.join(ROOT, "_onto_progress.log")
API = "http://35.184.189.38/api/hmdb/metabolites/{}/ontology/?api-key=" + KEY

# High-level origin terms in disposition.source (everything else = food/plant
# sub-sources we ignore).
SOURCE_HILEVEL = {
    'Endogenous', 'Exogenous', 'Food', 'Plant', 'Animal', 'Microbial',
    'Fungi', 'Bacteria', 'Synthetic', 'Industrial', 'Toxin/Pollutant',
    'Pollutant', 'Toxin', 'Cosmetic', 'Personal care product', 'Household',
    'Drug', 'Drug metabolite', 'Drug or steroid metabolite', 'Cigarette',
    'Naturally occurring process', 'Plant-based',
}


def _terms(o):
    out = []
    def rec(x):
        if isinstance(x, dict):
            t = x.get('name') or x.get('term')
            if t:
                out.append(str(t))
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for i in x:
                rec(i)
    rec(o)
    return out


def log(p, m):
    with open(p, 'a', encoding='utf-8') as f:
        f.write(m + '\n')


def fetch(hid, retries=2):
    for k in range(1, retries + 1):
        try:
            req = urllib.request.Request(API.format(hid),
                                         headers={'User-Agent': 'NMRPro-H/1.0'})
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


def classify(fo):
    disp = (fo or {}).get('disposition', {}) or {}
    src = disp.get('source', []) or []
    src_names = list(dict.fromkeys(s.get('name') for s in src if isinstance(s, dict)))
    hi = [n for n in src_names if n in SOURCE_HILEVEL]

    role_terms = set(_terms((fo or {}).get('role', {})))
    # 'drug' substring only — NOT 'pharmaceutical'/'medication' (those over-flag
    # endogenous metabolites used in pharma, e.g. alanine is tagged 'Pharmaceutical'
    # yet is a core metabolite). Caffeine carries 'Drug'/'Drug metabolite'.
    is_drug = any('drug' in t.lower() for t in role_terms)
    is_toxic = any(any(k in t.lower() for k in ('toxin', 'pollut', 'pesticide',
                                                'carcinogen'))
                   for t in role_terms)
    is_industrial = any('industrial' in t.lower() for t in role_terms)

    is_endo = 'Endogenous' in hi
    is_exo = 'Exogenous' in hi
    is_food = any(t in hi for t in ('Food', 'Plant', 'Animal'))
    is_microbial = any(t in hi for t in ('Microbial', 'Bacteria', 'Fungi'))
    # Strict synthetic origin (industrial/toxin). 'Personal care product'/'Cosmetic'
    # deliberately EXCLUDED here — endogenous compounds appear in cosmetics, so
    # those tags must not by themselves brand a metabolite synthetic.
    is_synth = any(t in hi for t in ('Synthetic', 'Industrial', 'Toxin/Pollutant',
                                     'Pollutant', 'Toxin'))

    # Goal: mammalian endogenous metabolome. KEEP iff HMDB tags it Endogenous and
    # it is not a drug. Food/plant co-occurrence does NOT exclude (most endogenous
    # metabolites are also dietary). Drugs, plant-only, synthetic-only, food-only
    # are excluded.
    keep = bool(is_endo and not is_drug)
    if keep:
        primary = 'endogenous'
        reason = ''
    elif is_drug:
        primary, reason = 'drug', 'role=Drug'
    elif is_synth or is_industrial or is_toxic:
        primary, reason = 'synthetic/xenobiotic', 'synthetic/industrial/toxin'
    elif is_microbial and not is_endo:
        primary, reason = 'microbial', 'microbial only (not endogenous)'
    elif is_food and not is_endo:
        primary, reason = 'plant/food', 'dietary only (not endogenous)'
    else:
        primary, reason = 'unknown', 'no Endogenous tag'

    return {
        'source_hilevel': hi,
        'role_drug': bool(is_drug),
        'role_toxic': bool(is_toxic),
        'role_industrial': bool(is_industrial),
        'is_endogenous': bool(is_endo),
        'is_exogenous': bool(is_exo),
        'is_food': bool(is_food),
        'is_microbial': bool(is_microbial),
        'is_synthetic': bool(is_synth),
        'keep_mammalian': keep,
        'exclude_reason': reason,
        'primary_origin': primary,
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
        cls = classify(d.get('functional_ontology', {}))
        cls['hmdb_id'] = d.get('hmdb_id', hid)
        cls['name'] = d.get('name')
        with open(dst, 'w', encoding='utf-8') as f:
            json.dump(cls, f, ensure_ascii=False)
        ok += 1
        if i % 10 == 0 or i == total:
            el = time.time() - t0
            r = i / el if el else 0
            log(PROG, f"[{i}/{total}] ok={ok} err={err} skip={skip} "
                      f"| {r:.2f}/s ETA {(total-i)/r/60:.1f} min" if r else f"[{i}/{total}]")
        time.sleep(0.12)
    log(PROG, f"=== done ok={ok} err={err} skip={skip} in {(time.time()-t0)/60:.1f} min ===")


if __name__ == '__main__':
    main()
