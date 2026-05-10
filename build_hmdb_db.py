# build_hmdb_db.py
# Constroi dois SQLite a partir dos ficheiros XML HMDB NMR:
#   - hmdb_nmr_total.db      : todos os HMDB IDs com dados 1H
#   - hmdb_nmr_validation.db : subconjunto de alta qualidade

import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from collections import defaultdict

# ── Configuração ──────────────────────────────────────────────────────────────
HMDB_NMR_DIR    = r"C:\Users\migue\Downloads\hmdb_nmr_spectra"
OUT_DIR         = r"C:\Users\migue\PycharmProjects\NMRAnalysis_Gui\databases"
DB_TOTAL        = os.path.join(OUT_DIR, "hmdb_nmr_total.db")
DB_VALIDATION   = os.path.join(OUT_DIR, "hmdb_nmr_validation.db")

CLUSTER_GAP     = 0.08   # ppm — gap mínimo entre grupos
MIN_SCORE_VAL   = 60.0   # score mínimo para validation DB
MIN_LABELED_VAL = 2      # mínimo de peaks com multiplicity label para validation
# ─────────────────────────────────────────────────────────────────────────────

VALID_MULT = {"s", "d", "t", "q", "m", "dd", "dt", "td", "ddd", "tt",
              "ddt", "tdd", "p", "sex", "sep", "quintet"}
MULT_BY_LINES = {1: "s", 2: "d", 3: "t", 4: "q"}

GOOD_ASSESSMENTS = {"excellent", "good"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_hmdb(hid: str) -> str:
    m = re.match(r"HMDB0*(\d+)", hid.upper())
    return f"HMDB{int(m.group(1)):07d}" if m else hid.upper()


def parse_freq(s: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", s or "")
    return float(m.group(1)) if m else 0.0


def score_spectrum(root) -> float:
    freq  = parse_freq(root.findtext("frequency") or "")
    solv  = (root.findtext("solvent") or "").lower()
    ph    = root.findtext("sample-ph") or ""
    assess = (root.findtext("sample-assessment") or "").lower()

    score = min(freq / 12.0, 50.0)   # max 50 pts para 600 MHz

    if "d2o" in solv or "water" in solv or "h2o" in solv:
        score += 20.0
    if assess in GOOD_ASSESSMENTS:
        score += 10.0
    try:
        score += max(0.0, 20.0 - abs(float(ph) - 7.4) * 4.0)
    except ValueError:
        pass
    return score


def cluster_peaks(peaks: list) -> list:
    """[(ppm, mult_label, intensity)] → [(ppm_center, multiplicity, intensity_sum)]"""
    if not peaks:
        return []
    peaks = sorted(peaks, key=lambda x: x[0])
    groups = []
    cur = [peaks[0]]
    for p in peaks[1:]:
        if p[0] - cur[-1][0] > CLUSTER_GAP:
            groups.append(cur)
            cur = [p]
        else:
            cur.append(p)
    groups.append(cur)

    result = []
    for grp in groups:
        ppm_c   = round(sum(x[0] for x in grp) / len(grp), 5)
        inten_s = round(sum(x[2] for x in grp), 4)
        labels  = [x[1] for x in grp if x[1] and x[1].lower() in VALID_MULT]
        if labels:
            mult = max(set(labels), key=labels.count)
        else:
            mult = MULT_BY_LINES.get(len(grp), "m")
        result.append((ppm_c, mult, inten_s, bool(labels)))
    return result


# ── DDL ───────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS metabolites (
    hmdb_id          TEXT PRIMARY KEY,
    bmrb_id          TEXT,
    name             TEXT,
    formula          TEXT,
    inchi            TEXT,
    smiles           TEXT,
    kegg_id          TEXT,
    drugbank_id      TEXT,
    pubchem_id       TEXT,
    spectrum_id      TEXT,
    frequency_mhz    REAL,
    solvent          TEXT,
    ph               REAL,
    temperature_c    REAL,
    assessment       TEXT,
    score            REAL,
    peak_count       INTEGER
);

CREATE TABLE IF NOT EXISTS shifts_1h (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hmdb_id         TEXT NOT NULL,
    bmrb_id         TEXT NOT NULL,
    atom_name       TEXT,
    ppm             REAL NOT NULL,
    multiplicity    TEXT,
    j_coupling      REAL,
    intensity       REAL,
    mult_labeled    INTEGER DEFAULT 0,
    FOREIGN KEY (hmdb_id) REFERENCES metabolites(hmdb_id)
);

CREATE INDEX IF NOT EXISTS idx_shifts_hmdb ON shifts_1h(hmdb_id);
CREATE INDEX IF NOT EXISTS idx_shifts_bmrb ON shifts_1h(bmrb_id);
CREATE INDEX IF NOT EXISTS idx_shifts_ppm  ON shifts_1h(ppm);
"""


def init_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(DDL)
    con.commit()
    return con


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_spectrum(path: str):
    """
    Devolve (hmdb_id, meta_dict, peaks_clustered) ou None se não for 1H.
    meta_dict: {spectrum_id, frequency_mhz, solvent, ph, temperature_c, assessment, score}
    peaks_clustered: [(ppm, mult, intensity, labeled)]
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    nuc = (root.findtext("nucleus") or "").strip()
    if nuc and nuc != "1H":
        return None

    hmdb_id_raw = root.findtext("database-id") or ""
    if not hmdb_id_raw:
        # tentar extrair do nome do ficheiro
        m = re.search(r"(HMDB\d+)", os.path.basename(path))
        hmdb_id_raw = m.group(1) if m else ""
    if not hmdb_id_raw:
        return None

    hmdb_id = normalize_hmdb(hmdb_id_raw)
    sc      = score_spectrum(root)

    freq = parse_freq(root.findtext("frequency") or "")
    try:
        ph = float(root.findtext("sample-ph") or "")
    except (ValueError, TypeError):
        ph = None
    try:
        temp = float(root.findtext("sample-temperature") or "")
    except (ValueError, TypeError):
        temp = None

    meta = {
        "spectrum_id":   root.findtext("id") or "",
        "frequency_mhz": freq if freq else None,
        "solvent":       (root.findtext("solvent") or "").strip() or None,
        "ph":            ph,
        "temperature_c": temp,
        "assessment":    (root.findtext("sample-assessment") or "").strip() or None,
        "score":         sc,
    }

    peaks_raw = []
    for p in root.findall(".//nmr-one-d-peak"):
        try:
            ppm = float(p.findtext("chemical-shift") or "")
        except (ValueError, TypeError):
            continue
        mult  = (p.findtext("multiplet-true") or "").strip().lower() or None
        try:
            inten = float(p.findtext("intensity") or "")
        except (ValueError, TypeError):
            inten = 0.0
        peaks_raw.append((ppm, mult, inten))

    if not peaks_raw:
        return None

    peaks_c = cluster_peaks(peaks_raw)
    return hmdb_id, meta, peaks_c


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HMDB NMR -> build_hmdb_db")
    print(f"  Fonte : {HMDB_NMR_DIR}")
    print(f"  Total : {DB_TOTAL}")
    print(f"  Valid : {DB_VALIDATION}")
    print("=" * 60)

    # Recolher todos os ficheiros XML agrupados por HMDB ID
    print("\n1. Listando ficheiros XML...")
    files_by_id: dict[str, list] = defaultdict(list)
    all_files = [fn for fn in os.listdir(HMDB_NMR_DIR) if fn.endswith(".xml")]
    print(f"   {len(all_files)} ficheiros encontrados")

    for fn in all_files:
        m = re.match(r"(HMDB\d+)_nmr_one_d_spectrum_", fn)
        if m:
            nid = normalize_hmdb(m.group(1))
            files_by_id[nid].append(os.path.join(HMDB_NMR_DIR, fn))

    print(f"   {len(files_by_id)} IDs HMDB únicos")

    # Para cada ID, escolher o melhor espectro
    print("\n2. Parseando e seleccionando melhor espectro por ID...")
    best: dict[str, tuple] = {}  # hmdb_id -> (meta, peaks_c)
    skipped_nuc = 0
    skipped_empty = 0

    for i, (nid, paths) in enumerate(files_by_id.items(), 1):
        if i % 5000 == 0:
            print(f"   ... {i}/{len(files_by_id)} IDs")

        best_score = -1.0
        best_meta  = None
        best_peaks = None

        for path in paths:
            result = parse_spectrum(path)
            if result is None:
                skipped_nuc += 1
                continue
            _, meta, peaks_c = result
            if meta["score"] > best_score:
                best_score = meta["score"]
                best_meta  = meta
                best_peaks = peaks_c

        if best_peaks:
            best[nid] = (best_meta, best_peaks)
        else:
            skipped_empty += 1

    print(f"   {len(best)} IDs com espectro 1H válido")
    print(f"   Ignorados (não-1H ou vazio): {skipped_nuc + skipped_empty}")

    # Construir DBs
    print("\n3. Construindo DBs...")

    con_total = init_db(DB_TOTAL)
    con_val   = init_db(DB_VALIDATION)

    n_total = 0
    n_val   = 0

    for hmdb_id, (meta, peaks_c) in best.items():
        n_peaks   = len(peaks_c)
        n_labeled = sum(1 for p in peaks_c if p[3])

        # metabolites: hmdb_id=bmrb_id, name/formula/etc preenchidos depois pelo CSV
        row_met = (
            hmdb_id, hmdb_id,          # hmdb_id, bmrb_id
            None, None, None, None,    # name, formula, inchi, smiles
            None, None, None,          # kegg_id, drugbank_id, pubchem_id
            meta["spectrum_id"], meta["frequency_mhz"], meta["solvent"],
            meta["ph"], meta["temperature_c"], meta["assessment"],
            meta["score"], n_peaks,
        )

        def insert_met(con, row):
            con.execute(
                "INSERT OR REPLACE INTO metabolites "
                "(hmdb_id,bmrb_id,name,formula,inchi,smiles,"
                " kegg_id,drugbank_id,pubchem_id,"
                " spectrum_id,frequency_mhz,solvent,ph,temperature_c,"
                " assessment,score,peak_count) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)

        def insert_shifts(con, hmdb_id, peaks_c):
            for idx, (ppm, mult, inten, labeled) in enumerate(peaks_c, 1):
                con.execute(
                    "INSERT INTO shifts_1h "
                    "(hmdb_id,bmrb_id,atom_name,ppm,multiplicity,j_coupling,intensity,mult_labeled) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (hmdb_id, hmdb_id, f"peak_{idx}", ppm, mult, None, inten, int(labeled)))

        insert_met(con_total, row_met)
        insert_shifts(con_total, hmdb_id, peaks_c)
        n_total += 1

        if meta["score"] >= MIN_SCORE_VAL and n_labeled >= MIN_LABELED_VAL:
            insert_met(con_val, row_met)
            insert_shifts(con_val, hmdb_id, peaks_c)
            n_val += 1

    con_total.commit()
    con_val.commit()

    # Stats finais
    for label, con in [("Total", con_total), ("Validation", con_val)]:
        n_met    = con.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
        n_shifts = con.execute("SELECT COUNT(*) FROM shifts_1h").fetchone()[0]
        n_labeled_shifts = con.execute(
            "SELECT COUNT(*) FROM shifts_1h WHERE mult_labeled=1"
        ).fetchone()[0]
        print(f"\n   [{label}]")
        print(f"     Metabolitos : {n_met}")
        print(f"     Shifts      : {n_shifts}")
        print(f"     Com mult    : {n_labeled_shifts} ({100*n_labeled_shifts//max(n_shifts,1)}%)")

    con_total.close()
    con_val.close()
    print("\nConcluido.")


if __name__ == "__main__":
    main()
