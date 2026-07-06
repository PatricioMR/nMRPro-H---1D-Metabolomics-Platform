"""
Final mammalian/endogenous consolidation for BMRB_Metabolomics_v01 — offline.

Combines every offline signal so a genuine mammalian metabolite is not lost to
one source's incomplete annotation (e.g. ChEBI tagging D-fructose only 'fungal'):

  KEEP (mammalian) if ANY of:
    • ChEBI role endogenous/mammalian (incl. fundamental metabolite)
    • HMDB ontology Endogenous (downloaded set)
    • present in a local HMDB human-biofluid DB
    • ChEBI compound_origins reports a MAMMALIAN species (Homo/Mus/Rattus/Bos/...)
  EXCLUDE only if NO keep signal AND a clear non-mammalian origin role
    (plant / drug / xenobiotic). Microbial/fungal-ONLY is NOT excluded — many
    universal metabolites (sugars, TCA acids) carry microbial roles too.
  else UNKNOWN.
"""
import gzip
import os
import sqlite3
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(ROOT, 'chebi', 'compound_origins.tsv.gz')
DB = os.path.join(os.path.dirname(ROOT), 'databases', 'BMRB_Metabolomics_v01.db')

MAMMAL_SPECIES = ('homo sapiens', 'mus musculus', 'rattus', 'bos taurus',
                  'sus scrofa', 'mammalia', 'oryctolagus', 'macaca', 'ovis aries',
                  'capra', 'equus', 'canis', 'felis', 'cavia', 'papio')


def chebi_mammal_origins():
    out = defaultdict(set)
    with gzip.open(ORIG, 'rt', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) < 4:
                continue
            sp = (c[3] or '').lower()
            if any(m in sp for m in MAMMAL_SPECIES):
                out['CHEBI:' + c[1]].add(c[3])
    return out


def main():
    cmam = chebi_mammal_origins()
    con = sqlite3.connect(DB)
    cur = con.cursor()
    for col in ('keep_mammalian', 'keep_reason', 'chebi_mammal_origin'):
        try:
            cur.execute(f"ALTER TABLE metabolites ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass

    cols = {r[1] for r in cur.execute("PRAGMA table_info(metabolites)")}
    he_col = 'is_endogenous' if 'is_endogenous' in cols else "''"
    hm_col = 'is_mammalian' if 'is_mammalian' in cols else "''"
    rows = cur.execute(
        f"SELECT bmrb_id,name,chebi_id,{he_col},{hm_col},"
        "is_endogenous_chebi,origin_chebi FROM metabolites").fetchall()
    keep = excl = unk = 0
    for bid, nm, chebi, he, hm, ce, oc in rows:
        sigs = []
        if ce == '1':
            sigs.append('chebi-role')
        if he == '1':
            sigs.append('hmdb-endogenous')
        if hm == '1':
            sigs.append('hmdb-biofluid')
        mam_org = chebi and chebi in cmam
        if mam_org:
            sigs.append('chebi-origin-mammal')
        oc = oc or ''
        has_plant_drug = any(t in oc for t in ('plant', 'drug', 'xenobiotic'))
        if sigs:
            k, reason = '1', 'mammalian: ' + ', '.join(sigs)
            keep += 1
        elif has_plant_drug:
            k, reason = '0', 'non-mammalian: ' + oc
            excl += 1
        else:
            k, reason = '?', f'unknown ({oc or "no role"})'
            unk += 1
        cur.execute("UPDATE metabolites SET keep_mammalian=?, keep_reason=?, "
                    "chebi_mammal_origin=? WHERE bmrb_id=?",
                    (k, reason, '; '.join(sorted(cmam.get(chebi, []))) or None, bid))
    con.commit()
    tot = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    con.close()
    print(f"ChEBI compounds with mammalian origin: {len(cmam)}")
    print(f"BMRB total={tot}  KEEP(mammalian)={keep}  EXCLUDE={excl}  UNKNOWN={unk}")


if __name__ == '__main__':
    main()
