# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Build BMRB_Metabolomics_v01.db from local BMRB metabolite entries (NMR-STAR).

Scope: real metabolite entries (NOT the GISSMO synthetic library — those carry
_Peak_coupling.Coupling_val and are medicinal-chemistry compounds). A metabolite
qualifies when it has BOTH spectral transitions (line positions) AND assigned
atoms — 749 such entries (sugars, amino acids, nucleotides, organic acids...).

Everything is read from MEASURED data — no estimation/modelling by us:
  • ppm          — _Peak_char.Chem_shift_val
  • protons (nH) — count of _Assigned_peak_chem_shift atoms for that peak
  • multiplicity — number of transition lines grouped to that peak (s/d/t/q/...)
  • J (Hz)       — spacing between those lines × spectrometer field (J's definition)
  • transitions  — the line positions, stored verbatim (drive the app fingerprint)
  • field, solvent, name, formula, InChI

J/multiplicity from line spacings is exact for first-order multiplets; complex
patterns (uneven spacings) are labelled 'm' with every distinct J listed —
never collapsed to a single fabricated value.

Schema is annotation-pipeline compatible: a `shifts_1h` DB is treated as
BMRB-style and queried by bmrb_id — here bmrb_id is the real bmse id.
"""
import glob
import os
import re
import sqlite3
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    Chem = None

BMRB_DIR = r"C:\Users\migue\OneDrive\Documentos\bmrb_entries"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DB = os.path.join(os.path.dirname(ROOT), 'databases', 'BMRB_Metabolomics_v01.db')

_MULT_BY_N = {1: 's', 2: 'd', 3: 't', 4: 'q', 5: 'quint', 6: 'sext', 7: 'sept'}


def loop(text, prefix):
    """Rows of the first NMR-STAR loop whose columns start with prefix (dynamic
    header order). Returns list of dicts."""
    m = re.search(r'((?:\s*' + re.escape(prefix) + r'\.\S+\s*\n)+)(.*?)\n\s*stop_',
                  text, re.S)
    if not m:
        return []
    cols = re.findall(re.escape(prefix) + r'\.(\S+)', m.group(1))
    out = []
    for line in m.group(2).strip().splitlines():
        if not line.strip():
            continue
        f = re.findall(r"'[^']*'|\S+", line)
        if len(f) >= len(cols):
            out.append(dict(zip(cols, f)))
    return out


def first_val(text, tag):
    """Value of a single-line NMR-STAR tag. Handles single-quoted, double-quoted
    (used when the value contains an apostrophe, e.g. "Inosine 5'-monophosphate")
    and bare tokens — so multi-word names are NOT truncated at the first space."""
    m = re.search(re.escape(tag) + r'\s+(?:"([^"]*)"|\'([^\']*)\'|(\S+))', text)
    if not m:
        return None
    return (m.group(1) if m.group(1) is not None
            else m.group(2) if m.group(2) is not None
            else m.group(3)).strip()


def field_strength(text):
    vals = [float(x) for x in re.findall(r'_NMR_spectrometer\.Field_strength\s+([\d.]+)', text)]
    return max(vals) if vals else None


def solvent_names(text):
    """List of solvent component names (rows whose Type field == 'Solvent',
    case-insensitive). NMR-STAR Type is capitalised ('Solvent')."""
    m = re.search(r'_Sample_component\.Sample_ID\s*?\n(.*?)\n\s*stop_', text, re.S)
    out = []
    if not m:
        return out
    for line in m.group(1).strip().splitlines():
        f = re.findall(r"'[^']*'|\S+", line)
        if any(tok.strip("'").lower() == 'solvent' for tok in f[1:]):
            out.append(f[1].strip("'"))
    return out


def is_d2o(names):
    return any('d2o' in n.lower() for n in names)


def classify_multiplet(lines, field):
    """(multiplicity_label, j_string) from grouped line positions.

    First-order: even spacing → s/d/t/q/...; J = that spacing × field.
    Complex: uneven spacings → 'm', J = every distinct spacing × field listed.
    """
    n = len(lines)
    if n <= 1:
        return 's', None
    lines = sorted(lines)
    seps_hz = [(lines[i + 1] - lines[i]) * field for i in range(n - 1)]
    seps_hz = [round(s, 2) for s in seps_hz if s > 0]
    if not seps_hz:
        return 's', None
    jmin, jmax = min(seps_hz), max(seps_hz)
    even = (jmax - jmin) <= 0.15 * jmax       # within 15 %
    if even and n in _MULT_BY_N:
        return _MULT_BY_N[n], f'{sum(seps_hz)/len(seps_hz):.2f}'
    # complex: list distinct couplings (cluster within 0.6 Hz)
    distinct = []
    for s in sorted(seps_hz):
        if not distinct or abs(s - distinct[-1]) > 0.6:
            distinct.append(s)
    return 'm', ' '.join(f'{j:.2f}' for j in distinct)


def main():
    files = sorted(glob.glob(os.path.join(BMRB_DIR, 'bmse*', '*.str')))
    if os.path.exists(OUT_DB):
        os.remove(OUT_DB)
    con = sqlite3.connect(OUT_DB)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE metabolites(
        bmrb_id TEXT PRIMARY KEY, hmdb_id TEXT, name TEXT, formula TEXT,
        inchi TEXT, inchikey TEXT, smiles TEXT, solvent TEXT, field_mhz REAL,
        n_1h_signals INTEGER, total_protons INTEGER);
    CREATE TABLE shifts_1h(
        id INTEGER PRIMARY KEY AUTOINCREMENT, bmrb_id TEXT, hmdb_id TEXT,
        atom_name TEXT, ppm REAL, ppm_min REAL, ppm_max REAL, multiplicity TEXT,
        j_coupling TEXT, protons INTEGER, n_lines INTEGER, intensity REAL,
        field_mhz REAL, solvent TEXT);
    CREATE TABLE transitions(bmrb_id TEXT, hmdb_id TEXT, ppm REAL, intensity REAL,
        freq_mhz REAL);
    CREATE INDEX ix_sh_ppm ON shifts_1h(ppm);
    CREATE INDEX ix_sh_b ON shifts_1h(bmrb_id);
    CREATE INDEX ix_tr_b ON transitions(bmrb_id);
    """)

    n_built = n_sig = n_nonaq = 0
    for fp in files:
        t = open(fp, encoding='utf-8', errors='ignore').read()
        if '_Peak_coupling.Coupling_val' in t:
            continue                              # GISSMO synthetic — skip
        # D2O only: chemical shifts must match aqueous mammalian samples; entries
        # in CDCl3/DMSO/etc. have shifted resonances and are excluded.
        snames = solvent_names(t)
        if not is_d2o(snames):
            n_nonaq += 1
            continue
        peaks = loop(t, '_Peak_char')
        asg = loop(t, '_Assigned_peak_chem_shift')
        trans = loop(t, '_Spectral_transition_char')
        if not peaks or not asg or not trans:
            continue
        field = field_strength(t) or 500.0
        bid = os.path.basename(os.path.dirname(fp))
        name = first_val(t, '_Entity.Name') or first_val(t, '_Chem_comp.Name') or bid
        if name:                                  # capitalise first letter
            name = name[0].upper() + name[1:]
        formula = first_val(t, '_Chem_comp.Formula')
        inchi = (re.search(r'_Chem_comp\.InChI_code\s+(InChI=\S+)', t) or [None, None])
        inchi = inchi[1] if isinstance(inchi, list) else (inchi.group(1) if inchi else None)
        inchikey = smiles = None
        if inchi and Chem is not None:
            try:
                inchikey = Chem.InchiToInchiKey(inchi)
                mol = Chem.MolFromInchi(inchi)
                smiles = Chem.MolToSmiles(mol) if mol else None
            except Exception:
                pass
        solv = '; '.join(snames) if snames else None

        # nH per peak = assigned H atoms, but an atom assigned to k peaks (BMRB's
        # way of flagging a stereo-ambiguous pair — e.g. valine's two methyls each
        # listing all 6 methyl H) contributes 1/k to each peak. Faithful reading of
        # the ambiguous assignment, not estimation; gives valine 3+3 not 6+6.
        nh = defaultdict(set)
        atom_npeaks = defaultdict(set)
        for r in asg:
            atom = r.get('Atom_ID', '')
            pid = r.get('Peak_ID')
            if atom and atom[0].upper() == 'H':
                nh[pid].add(atom)
                atom_npeaks[atom].add(pid)

        def peak_nh(pid):
            return sum(1.0 / len(atom_npeaks[a]) for a in nh.get(pid, ()))
        # peak ppm
        pk_ppm = {}
        for r in peaks:
            pid = r.get('Peak_ID') or r.get('ID')
            try:
                pk_ppm[pid] = float(r.get('Chem_shift_val'))
            except (TypeError, ValueError):
                pass
        # transition line positions (flat) → assign each to nearest peak
        lines = []
        for r in trans:
            try:
                lines.append(float(r.get('Chem_shift_val')))
            except (TypeError, ValueError):
                pass
        peak_lines = defaultdict(list)
        for L in lines:
            pid = min(pk_ppm, key=lambda p: abs(pk_ppm[p] - L), default=None)
            if pid is not None and abs(pk_ppm[pid] - L) <= 0.08:
                peak_lines[pid].append(L)

        sig = 0
        tot_p = 0
        for pid, ppm in pk_ppm.items():
            nprot = peak_nh(pid)
            if nprot <= 0:                        # no assigned H → not a 1H analyte peak
                continue
            nprot = int(round(nprot)) if abs(nprot - round(nprot)) < 0.05 else round(nprot, 2)
            grp = peak_lines.get(pid, [ppm])
            mult, jstr = classify_multiplet(grp, field)
            cur.execute(
                "INSERT INTO shifts_1h(bmrb_id,hmdb_id,atom_name,ppm,ppm_min,ppm_max,"
                "multiplicity,j_coupling,protons,n_lines,intensity,field_mhz,solvent) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (bid, None, ','.join(sorted(nh[pid])), round(ppm, 4),
                 round(min(grp), 4), round(max(grp), 4), mult, jstr, nprot,
                 len(grp), 1.0, field, solv))
            for L in grp:
                cur.execute("INSERT INTO transitions VALUES(?,?,?,?,?)",
                            (bid, None, round(L, 4), 1.0, field))
            sig += 1
            tot_p += nprot
        if sig == 0:
            continue
        cur.execute("INSERT INTO metabolites VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (bid, None, name, formula, inchi, inchikey, smiles, solv,
                     field, sig, tot_p))
        n_built += 1
        n_sig += sig

    con.commit()
    nm = cur.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
    nj = cur.execute("SELECT COUNT(*) FROM shifts_1h WHERE j_coupling IS NOT NULL").fetchone()[0]
    ntr = cur.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
    con.close()
    print(f"metabolites built: {n_built}  (non-D2O skipped: {n_nonaq})")
    print(f"shifts_1h signals: {n_sig}  (with J: {nj})")
    print(f"transitions: {ntr}")
    print(f"DB: {OUT_DB}")


if __name__ == '__main__':
    main()
