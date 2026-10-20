# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Fill ChEBI classification for BMRB metabolites NOT mapped via KEGG, using the
InChIKey route (ChEBI structures.tsv.gz). Offline. Reuses chebi_classify's obo
role machinery.
"""
import csv
import gzip
import os
import sqlite3
from collections import defaultdict

import chebi_classify as cc

ROOT = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(ROOT, 'chebi', 'structures.tsv.gz')
DB = cc.DB


def load_inchikey_map():
    full, conn = {}, {}
    with gzip.open(STRUCT, 'rt', encoding='utf-8', errors='ignore', newline='') as f:
        r = csv.reader(f, delimiter='\t', quotechar='"')
        header = next(r)
        ci = header.index('compound_id')
        ki = header.index('standard_inchi_key')
        di = header.index('default_structure')
        for row in r:
            if len(row) <= max(ci, ki, di):
                continue
            ik = row[ki].strip()
            if not ik or '-' not in ik:
                continue
            chebi = 'CHEBI:' + row[ci]
            default = row[di].strip().upper() in ('Y', 'YES', 'TRUE', '1')
            # prefer default structures; don't overwrite a default with a non-default
            if ik not in full or default:
                full[ik] = chebi
            conn.setdefault(ik[:14], chebi)
    return full, conn


def main():
    id2name, isa, hasrole, same = cc.parse_obo()
    mam_sub = cc.descendants(cc.MAMMAL_ROOTS, isa)
    excl_roles = {tag: cc.descendants({root}, isa)
                  for root, tag in cc.EXCLUDE_ROOTS.items()}
    full, conn = load_inchikey_map()
    print(f"ChEBI inchikey map: full={len(full)} conn={len(conn)}")

    con = sqlite3.connect(DB)
    cur = con.cursor()
    todo = cur.execute("SELECT bmrb_id,inchikey FROM metabolites "
                       "WHERE chebi_id IS NULL AND inchikey IS NOT NULL").fetchall()
    n_fill = n_endo = 0
    for bid, ik in todo:
        chebi = full.get(ik) or conn.get(ik[:14])
        if not chebi:
            continue
        roles = cc.roles_of(chebi, hasrole, same)
        is_endo = (cc.FUNDAMENTAL in roles) or any(r in mam_sub for r in roles)
        is_mam = is_endo
        tags = [tag for tag, s in excl_roles.items() if any(r in s for r in roles)]
        names = sorted({id2name.get(r, r) for r in roles
                        if r in mam_sub or r == cc.FUNDAMENTAL
                        or any(r in s for s in excl_roles.values())})
        origin = ('endogenous/mammalian' if is_endo
                  else (','.join(tags) if tags else ('no-role' if not roles else 'other')))
        cur.execute("UPDATE metabolites SET chebi_id=?, chebi_roles=?, "
                    "is_mammalian_chebi=?, is_endogenous_chebi=?, origin_chebi=? "
                    "WHERE bmrb_id=?",
                    (chebi, '; '.join(names), '1' if is_mam else '0',
                     '1' if is_endo else '0', origin, bid))
        n_fill += 1
        n_endo += int(is_endo)
    con.commit()
    tot = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    mapped = cur.execute("SELECT COUNT(*) FROM metabolites WHERE chebi_id IS NOT NULL").fetchone()[0]
    endo = cur.execute("SELECT COUNT(*) FROM metabolites WHERE is_endogenous_chebi='1'").fetchone()[0]
    nomap = cur.execute("SELECT COUNT(*) FROM metabolites WHERE chebi_id IS NULL").fetchone()[0]
    con.close()
    print(f"filled via InChIKey: {n_fill} (endo +{n_endo})")
    print(f"TOTAL={tot} chebi_mapped={mapped} endogenous={endo} still_unmapped={nomap}")


if __name__ == '__main__':
    main()
