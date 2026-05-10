"""
enrich_from_hmdb.py
===================
Enriquece shifts_1h.multiplicity no DB BMRB usando espectros HMDB NMR.

Fluxo:
  1. Para cada metabolito no DB com pubchem_id, consulta PubChem API
     para obter HMDB ID (cross-reference RegistryID)
  2. Guarda mapeamento pubchem→hmdb no DB (coluna metabolites.hmdb_id)
  3. Constrói índice HMDB: hmdb_id → [(ppm, multiplicity), ...]
     usando o melhor espectro 1H disponível (prioridade: 600 MHz > D2O)
  4. Actualiza shifts_1h.multiplicity onde está NULL ou inferido (peak_N)
"""

import os
import re
import sqlite3
import time
import urllib.request
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

# ─── Configuração ────────────────────────────────────────────────────────────
HMDB_NMR_DIR = r"C:\Users\migue\Downloads\hmdb_nmr_spectra"
DB_PATH      = r"C:\Users\migue\PycharmProjects\NMRAnalysis_Gui\databases\bmrb_metabolites.db"
CLUSTER_GAP  = 0.08   # ppm — gap mínimo entre grupos de transições
MATCH_TOL    = 0.05   # ppm — tolerância para cruzar HMDB↔BMRB shifts
PUBCHEM_DELAY = 0.22  # segundos entre chamadas à API (≤5 req/s)
# ─────────────────────────────────────────────────────────────────────────────

MULT_BY_LINES = {1: "s", 2: "d", 3: "t", 4: "q"}
VALID_MULT    = {"s", "d", "t", "q", "m", "dd", "dt", "td", "ddd", "tt",
                 "ddt", "tdd", "p", "sex", "sep", "quintet"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_freq(freq_str: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", freq_str or "")
    return float(m.group(1)) if m else 0.0


def _score_spectrum(root) -> float:
    freq  = _parse_freq(root.findtext("frequency") or "")
    solv  = (root.findtext("solvent") or "").lower()
    ph    = root.findtext("sample-ph") or ""
    score = min(freq / 12.0, 50.0)
    if "d2o" in solv or "water" in solv or "h2o" in solv:
        score += 30.0
    try:
        score += max(0.0, 20.0 - abs(float(ph) - 7.4) * 10.0)
    except ValueError:
        pass
    return score


def _cluster_peaks(peaks: list) -> list:
    """[(ppm, mult_label)] → [(ppm_center, multiplicity)]"""
    if not peaks:
        return []
    peaks_sorted = sorted(peaks, key=lambda x: x[0])
    groups = []
    cur = [peaks_sorted[0]]
    for ppm, mult in peaks_sorted[1:]:
        if ppm - cur[-1][0] > CLUSTER_GAP:
            groups.append(cur)
            cur = [(ppm, mult)]
        else:
            cur.append((ppm, mult))
    groups.append(cur)

    result = []
    for grp in groups:
        ppm_center = round(sum(p for p, _ in grp) / len(grp), 5)
        labels = [m for _, m in grp if m and m.strip() and m.lower() in VALID_MULT]
        mult = max(set(labels), key=labels.count) if labels else MULT_BY_LINES.get(len(grp), "m")
        result.append((ppm_center, mult))
    return result


# ── Passo 1: PubChem → HMDB ID ───────────────────────────────────────────────

def fetch_hmdb_from_pubchem(cid: str) -> str | None:
    """Consulta PubChem RegistryID cross-references e devolve HMDB ID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/RegistryID/JSON"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        ids = data["InformationList"]["Information"][0].get("RegistryID", [])
        for rid in ids:
            if rid.upper().startswith("HMDB"):
                return rid.upper()
    except Exception:
        pass
    return None


def update_hmdb_ids(db_path: str) -> dict:
    """
    Para metabolitos com pubchem_id mas sem hmdb_id, consulta PubChem.
    Devolve {bmrb_id: hmdb_id} com todos os mapeamentos conhecidos.
    """
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT bmrb_id, pubchem_id, hmdb_id FROM metabolites WHERE pubchem_id IS NOT NULL"
    ).fetchall()

    mapping = {}
    need_lookup = []
    for bmrb_id, pubchem_id, hmdb_id in rows:
        if hmdb_id:
            mapping[bmrb_id] = hmdb_id
        else:
            need_lookup.append((bmrb_id, pubchem_id))

    print(f"  {len(mapping)} já têm hmdb_id | {len(need_lookup)} precisam lookup PubChem")

    for i, (bmrb_id, cid) in enumerate(need_lookup, 1):
        if i % 100 == 0:
            print(f"  … {i}/{len(need_lookup)} lookups PubChem")
        hmdb_id = fetch_hmdb_from_pubchem(cid)
        if hmdb_id:
            con.execute("UPDATE metabolites SET hmdb_id=? WHERE bmrb_id=?", (hmdb_id, bmrb_id))
            mapping[bmrb_id] = hmdb_id
        time.sleep(PUBCHEM_DELAY)

    con.commit()
    con.close()
    return mapping


# ── Passo 2: Índice HMDB NMR ─────────────────────────────────────────────────

def build_hmdb_index(nmr_dir: str, hmdb_ids_needed: set) -> dict:
    """
    Lê ficheiros XML do HMDB, filtra pelos IDs necessários.
    Devolve {hmdb_id: [(ppm, multiplicity), ...]}.
    """
    # Normalizar IDs para formato HMDB0000251 (7 dígitos)
    def normalize_id(hid: str) -> str:
        m = re.match(r"HMDB0*(\d+)", hid.upper())
        return f"HMDB{int(m.group(1)):07d}" if m else hid.upper()

    needed_normalized = {normalize_id(h) for h in hmdb_ids_needed}

    files_by_id: dict[str, list] = defaultdict(list)
    for fn in os.listdir(nmr_dir):
        if not fn.endswith(".xml"):
            continue
        m = re.match(r"(HMDB\d+)_nmr_one_d_spectrum_", fn)
        if not m:
            continue
        nid = normalize_id(m.group(1))
        if nid in needed_normalized:
            files_by_id[nid].append(os.path.join(nmr_dir, fn))

    index = {}
    for nid, paths in files_by_id.items():
        best_score = -1.0
        best_peaks = []
        for path in paths:
            try:
                root = ET.parse(path).getroot()
                nuc = (root.findtext("nucleus") or "").strip()
                if nuc and nuc != "1H":
                    continue
                score = _score_spectrum(root)
                peaks_raw = []
                for p in root.findall(".//nmr-one-d-peak"):
                    try:
                        ppm = float(p.findtext("chemical-shift") or "")
                    except ValueError:
                        continue
                    mult = (p.findtext("multiplet-true") or "").strip().lower()
                    peaks_raw.append((ppm, mult))
                if peaks_raw and score > best_score:
                    best_score, best_peaks = score, peaks_raw
            except Exception:
                continue
        if best_peaks:
            index[nid] = _cluster_peaks(best_peaks)

    return index


# ── Passo 3: Enriquecimento DB ────────────────────────────────────────────────

def enrich_db(db_path: str, bmrb_to_hmdb: dict, hmdb_index: dict) -> tuple:
    """
    Actualiza multiplicity em shifts_1h.
    Só actualiza onde atom_name começa com 'peak_' ou multiplicity é NULL.
    """
    def normalize_id(hid):
        m = re.match(r"HMDB0*(\d+)", hid.upper())
        return f"HMDB{int(m.group(1)):07d}" if m else hid.upper()

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    updated = 0
    matched_mets = 0

    for bmrb_id, hmdb_id in bmrb_to_hmdb.items():
        nid = normalize_id(hmdb_id)
        hmdb_shifts = hmdb_index.get(nid)
        if not hmdb_shifts:
            continue
        matched_mets += 1

        db_shifts = cur.execute(
            "SELECT id, ppm, multiplicity, atom_name FROM shifts_1h WHERE bmrb_id=?",
            (bmrb_id,)
        ).fetchall()

        for sid, ppm, cur_mult, atom_name in db_shifts:
            is_inferred = (cur_mult is None) or atom_name.startswith("peak_")
            if not is_inferred:
                continue
            best = min(hmdb_shifts, key=lambda x: abs(x[0] - ppm))
            if abs(best[0] - ppm) > MATCH_TOL:
                continue
            new_mult = best[1]
            if new_mult and new_mult != cur_mult:
                cur.execute("UPDATE shifts_1h SET multiplicity=? WHERE id=?", (new_mult, sid))
                updated += 1

    con.commit()
    con.close()
    return updated, matched_mets


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HMDB NMR → enriquecimento de multiplicity")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    print("\n1. Mapeamento PubChem → HMDB ID...")
    bmrb_to_hmdb = update_hmdb_ids(DB_PATH)
    hmdb_ids_needed = set(bmrb_to_hmdb.values())
    print(f"   {len(bmrb_to_hmdb)} metabolitos com HMDB ID")

    print(f"\n2. Construindo índice HMDB ({len(hmdb_ids_needed)} IDs)...")
    index = build_hmdb_index(HMDB_NMR_DIR, hmdb_ids_needed)
    print(f"   {len(index)} IDs com espectro 1H encontrado")

    print("\n3. Actualizando multiplicity no DB...")
    updated, matched = enrich_db(DB_PATH, bmrb_to_hmdb, index)
    print(f"   Metabolitos com match HMDB: {matched}")
    print(f"   Shifts actualizados        : {updated}")

    con = sqlite3.connect(DB_PATH)
    n_null  = con.execute("SELECT COUNT(*) FROM shifts_1h WHERE multiplicity IS NULL").fetchone()[0]
    n_peak  = con.execute("SELECT COUNT(*) FROM shifts_1h WHERE atom_name LIKE 'peak_%' AND multiplicity IS NULL").fetchone()[0]
    n_total = con.execute("SELECT COUNT(*) FROM shifts_1h").fetchone()[0]
    con.close()
    print(f"\n   Total shifts        : {n_total}")
    print(f"   Multiplicity NULL   : {n_null}")
    print(f"   peak_N sem mult     : {n_peak}")
    print("\nConcluído.")


if __name__ == "__main__":
    main()
