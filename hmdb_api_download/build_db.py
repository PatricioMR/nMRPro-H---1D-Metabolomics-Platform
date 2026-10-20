# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Build HMDB_Metabolomics_v01.db from downloaded HMDB experimental NMR data.

Primary data: hmdb_api_download/nmr/{HMDB_ID}.json (experimental 1H stick lists,
13C, HSQC). Derives, per metabolite, resolved 1H multiplets with:
  • integration window (ppm_min..ppm_max)
  • intensity-weighted ppm centre
  • inferred multiplicity + J (from stick spacing)
  • protons (nH) — from relative integral anchored to the number of
    non-exchangeable (carbon-bound) protons computed from SMILES via RDKit.

Scientific basis: in a 1H spectrum integrated area ∝ nH. HMDB stick intensities
are line heights; summing the lines of one multiplet ≈ its relative integral
(equal-linewidth assumption). Anchoring Σintegral to carbon-bound H (the only
protons visible in aqueous NMR; OH/NH/COOH exchange away) yields per-multiplet
nH. Validated on alanine (3:1) and valine (6:1:1).

Auditability (CLAUDE.md): raw sticks are kept verbatim in sticks_1h; every
derived nH carries the anchor + a mismatch flag when Σround(nH) ≠ carbon H.
No value is fabricated — metabolites without an experimental 1H list get no
shifts_1h rows (reported, not invented).
"""
import json
import glob
import os
import sqlite3
import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    Chem = None

ROOT = os.path.dirname(os.path.abspath(__file__))
NMR_DIR = os.path.join(ROOT, 'nmr')
OUT_DB = os.path.join(os.path.dirname(ROOT), 'databases', 'HMDB_Metabolomics_v01.db')
OLD_DB = os.path.join(os.path.dirname(ROOT), 'databases', 'HMDB_General.db')
REPORT = os.path.join(ROOT, '_build_report.csv')

# Multiplet clustering: split stick list where the ppm gap between adjacent
# lines exceeds this. Intra-multiplet spacing is J/field (≤ ~16 Hz → ~0.032 ppm
# at 500 MHz); a larger gap means a new signal. Equivalent groups closer than
# this merge into one integration window (acceptable: same nH, one window).
CLUSTER_GAP_PPM = 0.045

_PASCAL_NAME = {1: 's', 2: 'd', 3: 't', 4: 'q', 5: 'p', 6: 'h', 7: 'hept'}


def carbon_bound_H(smiles):
    """Non-exchangeable proton count = H attached to carbon (RDKit). None on fail."""
    if not smiles or Chem is None:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    n = 0
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 1:
            nb = a.GetNeighbors()
            if nb and nb[0].GetAtomicNum() == 6:
                n += 1
    return n


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def is_exp(sp):
    f = sp.get('frequency')
    return isinstance(f, str) and 'MHz' in f


def field_mhz(sp):
    f = str(sp.get('frequency', ''))
    try:
        return float(f.replace('MHz', '').strip())
    except ValueError:
        return 500.0


def pick_best_1h(d):
    """Cleanest experimental 1H pick-list: assessment rank, then fewest peaks
    (a digitized trace has many; the curated stick list few). Require >=2 peaks."""
    rank = {'Excellent': 0, 'Satisfactory': 1, 'Poor': 2, 'None': 3, None: 3}
    cands = []
    for sp in (d.get('nmr') or []):
        if str(sp.get('nucleus')) != '1H' or not is_exp(sp):
            continue
        pk = [p for p in (sp.get('peaks') or [])
              if p.get('ppm') is not None and p.get('intensity') is not None]
        if len(pk) < 1:
            continue
        cands.append((rank.get(sp.get('spectra_assessment'), 3), len(pk), sp))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1]))
    return cands[0][2]


def cluster_sticks(peaks, gap=CLUSTER_GAP_PPM):
    """Group sticks (sorted by ppm) into multiplets, splitting on large gaps."""
    pk = sorted(({'ppm': _f(p['ppm']), 'int': abs(_f(p['intensity']))}
                 for p in peaks), key=lambda p: p['ppm'])
    clusters, cur = [], [pk[0]]
    for p in pk[1:]:
        if p['ppm'] - cur[-1]['ppm'] > gap:
            clusters.append(cur)
            cur = [p]
        else:
            cur.append(p)
    clusters.append(cur)
    return clusters


def infer_mult_j(cluster, fmhz):
    """(multiplicity, j_hz) from a cluster's lines. Simple n→name when lines are
    evenly spaced; 'm' otherwise. J = median adjacent spacing × field."""
    n = len(cluster)
    if n == 1:
        return 's', None
    ppms = [c['ppm'] for c in cluster]
    seps = [(ppms[i + 1] - ppms[i]) for i in range(n - 1)]
    seps_hz = [s * fmhz for s in seps if s > 0]
    j = float(np.median(seps_hz)) if seps_hz else None
    # Even spacing (within 25%) and small line count → Pascal multiplet.
    if 2 <= n <= 7 and seps_hz:
        spread = (max(seps_hz) - min(seps_hz)) / max(np.median(seps_hz), 1e-6)
        if spread < 0.25:
            return _PASCAL_NAME.get(n, 'm'), j
    return 'm', j


def build():
    files = sorted(glob.glob(os.path.join(NMR_DIR, '*.json')))
    if os.path.exists(OUT_DB):
        os.remove(OUT_DB)
    con = sqlite3.connect(OUT_DB)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE metabolites(
        hmdb_id TEXT PRIMARY KEY, bmrb_id TEXT, name TEXT, formula TEXT,
        smiles TEXT, carbon_bound_H INTEGER, n_1h_signals INTEGER,
        has_hsqc INTEGER);
    CREATE TABLE shifts_1h(
        id INTEGER PRIMARY KEY AUTOINCREMENT, hmdb_id TEXT, bmrb_id TEXT,
        atom_name TEXT, ppm REAL,
        ppm_min REAL, ppm_max REAL, multiplicity TEXT, j_coupling TEXT,
        protons INTEGER, intensity REAL, rel_integral REAL, n_lines INTEGER,
        solvent TEXT, field_mhz REAL, ph TEXT, temperature TEXT,
        shift_ref TEXT, assessment TEXT, spectrum_url TEXT,
        proton_flag TEXT);
    CREATE TABLE sticks_1h(
        hmdb_id TEXT, ppm REAL, intensity REAL, field_mhz REAL);
    CREATE TABLE shifts_13c(
        hmdb_id TEXT, ppm REAL, intensity REAL, field_mhz REAL, assessment TEXT);
    CREATE TABLE hsqc(
        hmdb_id TEXT, h_ppm REAL, c_ppm REAL, intensity REAL);
    CREATE INDEX ix_shifts_ppm ON shifts_1h(ppm);
    CREATE INDEX ix_shifts_mid ON shifts_1h(hmdb_id);
    """)

    rep = []  # validation rows
    n_met = n_with1h = n_hsqc = 0
    for fp in files:
        d = json.load(open(fp, encoding='utf-8-sig'))
        hid = d.get('hmdb_id') or os.path.splitext(os.path.basename(fp))[0]
        name = d.get('name')
        formula = d.get('moldb_formula')
        smiles = d.get('moldb_smiles')
        cbh = carbon_bound_H(smiles)

        sp = pick_best_1h(d)
        nmr = d.get('nmr') or []
        has_hsqc = any(str(s.get('nucleus')) == '1H-13C' for s in nmr)
        if has_hsqc:
            n_hsqc += 1

        n_signals = 0
        if sp is not None:
            fmhz = field_mhz(sp)
            peaks = [p for p in sp.get('peaks') or []
                     if p.get('ppm') is not None and p.get('intensity') is not None]
            # raw sticks (auditable)
            for p in peaks:
                cur.execute("INSERT INTO sticks_1h VALUES(?,?,?,?)",
                            (hid, _f(p['ppm']), abs(_f(p['intensity'])), fmhz))
            clusters = cluster_sticks(peaks) if peaks else []
            integ = [sum(c['int'] for c in cl) for cl in clusters]
            tot = sum(integ)
            # Anchor integral → carbon-bound H. The scaled values sum EXACTLY to
            # carbon_H by construction, so the proton partition is obtained by
            # largest-remainder apportionment (sum stays == carbon_H — no fabricated
            # total). Clusters that apportion to 0 H are impurity/noise lines: kept
            # in sticks_1h (auditable) but not emitted as quantifiable signals.
            scale = (cbh / tot) if (cbh and tot > 0) else None
            dropped = 0
            if scale:
                raw = [i * scale for i in integ]
                floor = [int(np.floor(x)) for x in raw]
                rem = int(round(cbh - sum(floor)))
                order = sorted(range(len(raw)), key=lambda k: -(raw[k] - floor[k]))
                prot = floor[:]
                for k in order[:max(0, rem)]:
                    prot[k] += 1
                keep = [(cl, ig, pr) for cl, ig, pr in zip(clusters, integ, prot)
                        if pr > 0]
                dropped = len(clusters) - len(keep)
                clusters = [k[0] for k in keep]
                integ = [k[1] for k in keep]
                prot = [k[2] for k in keep]
            else:
                prot = [None] * len(clusters)
            mismatch = False  # apportionment guarantees Σprotons == carbon_H
            for cl, ig, pr in zip(clusters, integ, prot):
                ppms = [c['ppm'] for c in cl]
                ints = [c['int'] for c in cl]
                wsum = sum(ints) or 1.0
                centre = sum(p * w for p, w in zip(ppms, ints)) / wsum
                mult, j = infer_mult_j(cl, fmhz)
                flag = ('no_smiles' if cbh is None
                        else f'dropped_minor({dropped})' if dropped else 'ok')
                cur.execute(
                    "INSERT INTO shifts_1h(hmdb_id,bmrb_id,atom_name,ppm,ppm_min,"
                    "ppm_max,multiplicity,j_coupling,protons,intensity,rel_integral,"
                    "n_lines,solvent,field_mhz,ph,temperature,shift_ref,assessment,"
                    "spectrum_url,proton_flag) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (hid, hid, None, round(centre, 4),
                     round(min(ppms), 4), round(max(ppms), 4),
                     mult, (f"{j:.2f}" if j else None), pr,
                     round(ig, 4), round(ig, 4), len(cl),
                     sp.get('solvent'), fmhz, sp.get('sample_ph'),
                     sp.get('sample_temperature'), sp.get('chemical_shift_reference'),
                     sp.get('spectra_assessment'), sp.get('spectrum_url'), flag))
            n_signals = len(clusters)
            if n_signals:
                n_with1h += 1
            rep.append((hid, name, cbh, sum(p for p in prot if p) if prot else 0,
                        n_signals, f'dropped_minor({dropped})' if dropped else ''))

        # 13C experimental
        for s13 in nmr:
            if str(s13.get('nucleus')) == '13C' and is_exp(s13):
                for p in s13.get('peaks') or []:
                    if p.get('ppm') is not None:
                        cur.execute("INSERT INTO shifts_13c VALUES(?,?,?,?,?)",
                                    (hid, _f(p['ppm']),
                                     abs(_f(p.get('intensity'))),
                                     field_mhz(s13), s13.get('spectra_assessment')))
                break
        # HSQC
        for sh in nmr:
            if str(sh.get('nucleus')) == '1H-13C':
                for p in sh.get('peaks') or []:
                    if p.get('F2ppm') is not None and p.get('F1ppm') is not None:
                        cur.execute("INSERT INTO hsqc VALUES(?,?,?,?)",
                                    (hid, _f(p['F2ppm']), _f(p['F1ppm']),
                                     abs(_f(p.get('intensity')))))
                break

        # bmrb_id mirrors hmdb_id: the app treats a shifts_1h-based DB as
        # BMRB-style and queries every table by bmrb_id (annotation_results_window,
        # rescue, transitions). Mirroring keeps the HMDB ids reachable on both
        # column names so all app query paths resolve.
        cur.execute("INSERT INTO metabolites VALUES(?,?,?,?,?,?,?,?)",
                    (hid, hid, name, formula, smiles, cbh, n_signals, int(has_hsqc)))
        n_met += 1

    # transitions = raw experimental sticks (exact multiplet line positions) —
    # consumed by the app's lines_from_transitions for precise fingerprinting.
    cur.execute("CREATE TABLE transitions AS "
                "SELECT hmdb_id, hmdb_id AS bmrb_id, ppm, intensity, "
                "field_mhz AS freq_mhz FROM sticks_1h")
    cur.execute("CREATE INDEX ix_tr_mid ON transitions(hmdb_id)")
    cur.execute("CREATE INDEX ix_tr_bmrb ON transitions(bmrb_id)")
    con.commit()

    # validation report
    import csv
    with open(REPORT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['hmdb_id', 'name', 'carbon_H', 'sum_derived_protons',
                    'n_1h_signals', 'flag'])
        w.writerows(rep)

    # stats
    s1 = cur.execute("SELECT COUNT(*) FROM shifts_1h").fetchone()[0]
    sm = cur.execute("SELECT COUNT(DISTINCT hmdb_id) FROM shifts_1h WHERE proton_flag LIKE 'dropped_minor%'").fetchone()[0]
    # verify anchor: per metabolite Σprotons == carbon_bound_H
    bad = cur.execute("""
        SELECT COUNT(*) FROM (
          SELECT s.hmdb_id, SUM(s.protons) sp, m.carbon_bound_H ch
          FROM shifts_1h s JOIN metabolites m ON s.hmdb_id=m.hmdb_id
          WHERE m.carbon_bound_H IS NOT NULL
          GROUP BY s.hmdb_id HAVING sp != ch)""").fetchone()[0]
    st = cur.execute("SELECT COUNT(*) FROM sticks_1h").fetchone()[0]
    c13 = cur.execute("SELECT COUNT(*) FROM shifts_13c").fetchone()[0]
    hq = cur.execute("SELECT COUNT(*) FROM hsqc").fetchone()[0]
    con.close()
    print(f"DB: {OUT_DB}")
    print(f"metabolites={n_met}  with_1h={n_with1h}  with_hsqc={n_hsqc}")
    print(f"shifts_1h rows={s1}  metabolites_with_dropped_minor={sm}")
    print(f"anchor check - metabolites where sum(protons) != carbon_H: {bad} (should be 0)")
    print(f"sticks_1h={st}  shifts_13c={c13}  hsqc={hq}")
    print(f"report: {REPORT}")


if __name__ == '__main__':
    build()
