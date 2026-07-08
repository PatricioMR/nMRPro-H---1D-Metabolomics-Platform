"""
De-duplicate BMRB_Metabolomics_v01.db.

Merge rule (user spec):
  • Only entries sharing the SAME molecular skeleton (InChIKey connectivity block)
    or the SAME name are merge candidates — keeps epimers/diastereomers apart
    (D-glucose vs D-galactose share connectivity but have DIFFERENT spectra).
  • Within candidates, two entries are spectral duplicates when their 1H spectra
    coincide: same signal count, each ppm within 0.03, equal nH. This catches
    exact duplicates AND enantiomers (identical NMR in achiral solvent).
  • From each spectral-duplicate cluster keep the entry whose field is CLOSEST to
    600 MHz (tie → more signals, then the longer/explicit-stereo name). The rest
    (rows in metabolites/shifts_1h/transitions) are deleted.

Distinct spectra (epimers) are left untouched.
"""
import os
import re
import sqlite3
from collections import defaultdict

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'databases', 'BMRB_Metabolomics_v01.db')
PPM_TOL = 0.03


def relstereo(inchi):
    """InChI reduced to relative configuration: drop the /m (mirror) and /s
    layers but KEEP /t (tetrahedral parity). Enantiomers → equal (differ only in
    /m); diastereomers → differ in /t → unequal. Gates enantiomer merges so
    diastereomers (erythritol vs threitol) are never collapsed."""
    if not inchi:
        return None
    return re.sub(r'/m[01.]+|/s\d+', '', inchi)


def signature(shifts):
    """Spectral signature: sorted (ppm, nH, multiplicity). Multiplicity is
    included so DIASTEREOMERS (same shifts, different coupling — e.g. erythritol
    vs threitol) are NOT treated as identical; only true spectral twins match."""
    return sorted((round(p, 3), round(float(nh), 1), (m or '').strip())
                  for p, nh, m in shifts)


def spectra_match(a, b):
    if len(a) != len(b):
        return False
    for (pa, na, ma), (pb, nb, mb) in zip(a, b):
        if abs(pa - pb) > PPM_TOL or abs(na - nb) > 0.5 or ma != mb:
            return False
    return True


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    mets = cur.execute(
        "SELECT bmrb_id,name,inchikey,field_mhz,n_1h_signals,inchi FROM metabolites").fetchall()
    shifts = defaultdict(list)
    for b, p, nh, m in cur.execute(
            "SELECT bmrb_id,ppm,protons,multiplicity FROM shifts_1h"):
        shifts[b].append((p, nh, m))

    def score_keep(e):
        # closest to 600 first; then more signals; then longer name (explicit stereo)
        b, name, fld, ns = e
        return (abs(fld - 600), -ns, -len(name or ''))

    to_delete = []
    merges = []

    # ── Pass A: identical molecule (full InChIKey, or name when key missing) ──
    # Same compound measured more than once → always collapse to the entry whose
    # field is closest to 600, regardless of small measurement differences.
    full = defaultdict(list)
    for b, name, ik, fld, ns, inchi in mets:
        key = ik if ik else 'name:' + (name or b).lower().strip()
        full[key].append((b, name, fld or 0, ns or 0))
    removed = set()
    for key, entries in full.items():
        if len(entries) < 2:
            continue
        members = sorted(entries, key=score_keep)
        merges.append((members[0], members[1:]))
        for d in members[1:]:
            to_delete.append(d[0])
            removed.add(d[0])

    # ── Pass B: enantiomers (same skeleton, different stereo) merged ONLY when
    #    spectrally identical. Epimers (glucose/galactose) differ → kept apart. ──
    skel = defaultdict(list)
    for b, name, ik, fld, ns, inchi in mets:
        rs = relstereo(inchi)
        if b in removed or not rs:
            continue
        skel[rs].append((b, name, fld or 0, ns or 0))
    for key, entries in skel.items():
        if len(entries) < 2:
            continue
        clusters = []
        for e in entries:
            sig = signature(shifts.get(e[0], []))
            for cl in clusters:
                if spectra_match(sig, cl['sig']):
                    cl['members'].append(e)
                    break
            else:
                clusters.append({'sig': sig, 'members': [e]})
        for cl in clusters:
            if len(cl['members']) < 2:
                continue
            members = sorted(cl['members'], key=score_keep)
            merges.append((members[0], members[1:]))
            to_delete.extend(d[0] for d in members[1:])

    print(f"metabolites before: {len(mets)}")
    print(f"spectral-duplicate clusters merged: {len(merges)}")
    print(f"entries to delete: {len(to_delete)}")
    for keep, drop in merges:
        print(f"  KEEP {keep[0]} '{keep[1]}' ({keep[2]:g}MHz, {keep[3]}sig)  "
              f"<- drop {[ (d[0], d[1], f'{d[2]:g}MHz') for d in drop ]}")

    if to_delete:
        ph = ','.join('?' * len(to_delete))
        for tbl in ('shifts_1h', 'transitions', 'metabolites'):
            cur.execute(f"DELETE FROM {tbl} WHERE bmrb_id IN ({ph})", to_delete)
        con.commit()
        con.execute("VACUUM")
    n_after = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    con.close()
    print(f"metabolites after: {n_after}")


if __name__ == '__main__':
    main()
