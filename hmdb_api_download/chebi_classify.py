# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Classify BMRB metabolites by ChEBI biological role — fully offline (no HMDB API).

Sources (downloaded once):
  • chebi.obo.gz             — role ontology: term names, is_a hierarchy, and
                               has_role edges (RO:0000087) compound→role.
  • database_accession.tsv.gz — ChEBI ↔ KEGG / HMDB cross-references.
  • structures.tsv.gz (opt)  — ChEBI ↔ InChIKey (fallback mapping).

Mapping BMRB→ChEBI: KEGG id from the entry's _Chem_comp_db_link (local), then
InChIKey fallback. Classification uses ChEBI role subtrees:
  mammalian/endogenous = descendant of 'mammalian metabolite' (CHEBI:75768) incl.
    human/mouse/rat metabolite, human blood serum/urinary metabolite, etc.
  exclusion = descendant of plant metabolite (76924), drug (23888), xenobiotic
    (35703), food (33290), bacterial (76969), fungal (76946) metabolite.
A compound may carry several roles (e.g. caffeine is mammalian AND drug/plant) —
all flags are recorded; the user keeps endogenous-mammalian even if also a drug.
"""
import gzip
import os
import re
import glob
import sqlite3
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
CHEBI = os.path.join(ROOT, 'chebi')
BMRB_DIR = r"C:\Users\migue\OneDrive\Documentos\bmrb_entries"
DB = os.path.join(os.path.dirname(ROOT), 'databases', 'BMRB_Metabolomics_v01.db')

MAMMAL_ROOTS = {'CHEBI:75768'}                       # mammalian metabolite (detected)
FUNDAMENTAL = 'CHEBI:78675'                          # fundamental metabolite (endogenous)
EXCLUDE_ROOTS = {
    'CHEBI:76924': 'plant', 'CHEBI:23888': 'drug', 'CHEBI:35703': 'xenobiotic',
    'CHEBI:33290': 'food', 'CHEBI:76969': 'bacterial', 'CHEBI:76946': 'fungal',
}
# ChEBI attaches roles to a specific protonation/tautomer form; union roles over
# these same-species links so the neutral compound inherits them (e.g. L-alanine
# gets 'fundamental metabolite' from L-alaninate).
SAME_SPECIES_REL = ('RO:0018033', 'RO:0018034', 'RO:0018035', 'RO:0018036')


def parse_obo():
    id2name, isa, hasrole = {}, defaultdict(set), defaultdict(set)
    same = defaultdict(set)            # same-species links (conjugate/tautomer)
    cur = None
    with gzip.open(os.path.join(CHEBI, 'chebi.obo.gz'), 'rt',
                   encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '[Term]':
                cur = None
            elif line.startswith('id: CHEBI:'):
                cur = line[4:].strip()
            elif cur and line.startswith('name:'):
                id2name[cur] = line[5:].strip()
            elif cur and line.startswith('is_a:'):
                isa[cur].add(line.split()[1])
            elif cur and line.startswith('relationship: RO:0000087'):
                m = re.search(r'(CHEBI:\d+)', line.split('RO:0000087', 1)[1])
                if m:
                    hasrole[cur].add(m.group(1))
            elif cur and line.startswith('relationship:'):
                parts = line.split()
                if len(parts) >= 3 and parts[1] in SAME_SPECIES_REL:
                    same[cur].add(parts[2])
                    same[parts[2]].add(cur)
    return id2name, isa, hasrole, same


def roles_of(chebi, hasrole, same):
    """Union of roles over the same-species component (conjugate/tautomer forms)."""
    seen, stack, roles = {chebi}, [chebi], set()
    while stack:
        n = stack.pop()
        roles |= hasrole.get(n, set())
        for m in same.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return roles


def descendants(roots, isa):
    """All terms that are (transitively) is_a one of roots — i.e. the subtree."""
    children = defaultdict(set)
    for c, parents in isa.items():
        for p in parents:
            children[p].add(c)
    seen = set(roots)
    stack = list(roots)
    while stack:
        n = stack.pop()
        for ch in children.get(n, ()):
            if ch not in seen:
                seen.add(ch)
                stack.append(ch)
    return seen


def load_xref():
    kegg2chebi, chebi2hmdb = {}, defaultdict(list)
    with gzip.open(os.path.join(CHEBI, 'database_accession.tsv.gz'), 'rt',
                   encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) < 3:
                continue
            cid, acc = c[1], c[2]
            chebi = 'CHEBI:' + cid
            if re.fullmatch(r'C\d{5}', acc):
                kegg2chebi.setdefault(acc, chebi)
            elif acc.startswith('HMDB'):
                chebi2hmdb[chebi].append(acc)
    return kegg2chebi, chebi2hmdb


def bmrb_kegg(ours):
    """bmse_id -> KEGG id from _Chem_comp_db_link (local BMRB)."""
    out = {}
    for fp in glob.glob(os.path.join(BMRB_DIR, 'bmse*', '*.str')):
        bid = os.path.basename(os.path.dirname(fp))
        if bid not in ours:
            continue
        t = open(fp, encoding='utf-8', errors='ignore').read()
        m = re.search(r'_Chem_comp_db_link\.Comp_ID\s*\n(.*?)\n\s*stop_', t, re.S)
        if not m:
            continue
        for line in m.group(1).strip().splitlines():
            f = re.findall(r"'[^']*'|\S+", line)
            if len(f) >= 3 and f[1].strip("'") == 'KEGG':
                out[bid] = f[2]
                break
    return out


def main():
    id2name, isa, hasrole, same = parse_obo()
    mam_roles = descendants(MAMMAL_ROOTS, isa) | {FUNDAMENTAL}
    excl_roles = {tag: descendants({root}, isa)
                  for root, tag in EXCLUDE_ROOTS.items()}
    kegg2chebi, chebi2hmdb = load_xref()

    con = sqlite3.connect(DB)
    cur = con.cursor()
    for col in ('chebi_id', 'chebi_roles', 'is_mammalian_chebi',
                'is_endogenous_chebi', 'origin_chebi'):
        try:
            cur.execute(f"ALTER TABLE metabolites ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    ours = {x for (x,) in cur.execute("SELECT bmrb_id FROM metabolites")}
    kegg = bmrb_kegg(ours)

    n_map = n_mam = n_endo = 0
    stats = defaultdict(int)
    for bid in ours:
        chebi = kegg2chebi.get(kegg.get(bid)) if kegg.get(bid) else None
        if not chebi:
            continue
        roles = roles_of(chebi, hasrole, same)
        if not roles:
            n_map += 1
            cur.execute("UPDATE metabolites SET chebi_id=?, origin_chebi='no-role' "
                        "WHERE bmrb_id=?", (chebi, bid))
            stats['no-role'] += 1
            continue
        endo = FUNDAMENTAL in roles or any(r in descendants(MAMMAL_ROOTS, isa) for r in roles)
        is_mam = any(r in mam_roles for r in roles)
        tags = [tag for tag, s in excl_roles.items() if any(r in s for r in roles)]
        # endogenous-mammalian = fundamental metabolite OR mammalian metabolite role
        is_endo = (FUNDAMENTAL in roles) or any(r in (descendants(MAMMAL_ROOTS, isa)) for r in roles)
        role_names = sorted({id2name.get(r, r) for r in roles
                             if r in mam_roles or any(r in s for s in excl_roles.values())})
        origin = ('endogenous/mammalian' if is_endo
                  else (','.join(tags) if tags else 'other'))
        cur.execute("UPDATE metabolites SET chebi_id=?, chebi_roles=?, "
                    "is_mammalian_chebi=?, is_endogenous_chebi=?, origin_chebi=? "
                    "WHERE bmrb_id=?",
                    (chebi, '; '.join(role_names), '1' if is_mam else '0',
                     '1' if is_endo else '0', origin, bid))
        n_map += 1
        n_mam += int(is_mam)
        n_endo += int(is_endo)
        stats[origin] += 1
    con.commit()
    tot = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    con.close()
    print(f"ChEBI terms={len(id2name)} mammal/endo roles={len(mam_roles)}")
    print(f"BMRB total={tot}  mapped_via_KEGG={n_map}  "
          f"mammalian={n_mam}  endogenous={n_endo}")
    print("origin breakdown:", dict(stats))


if __name__ == '__main__':
    main()
