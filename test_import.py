"""
BMRB NMR-STAR → SQLite para Metabolômica (apenas 1D ¹H e 600 MHz)
=================================================================
Lê arquivos .str do BMRB e popula um banco bmrb_600mhz.db com:
  - metabolites   (info do metabólito)
  - shifts_1h     (deslocamentos ¹H, multiplicidade, J)
  - transitions   (picos de alta resolução, opcional)
  - samples       (condições experimentais)

Apenas entradas com experimentos 1D ¹H e campo magnético 600 MHz são inseridas.
Para picos não atribuídos, cria-se nomes sintéticos do tipo "peak_<id>".

Uso:
    python bmrb_to_sqlite_600mhz.py
"""

import os
import re
import sqlite3
import logging
from pathlib import Path

# ─── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────
INPUT_DIR         = r"C:\Users\migue\OneDrive\Documentos\bmrb_entries"
OUTPUT_DB         = r"C:\Users\migue\PycharmProjects\NMRAnalysis_Gui\databases\bmrb_metabolites.db"
TARGET_FIELD_MHZ    = 600
ACCEPTED_SOLVENTS   = {"d2o", "h2o"}  # solventes aquosos aceitos (case-insensitive)
TARGET_PH_MIN       = 7.0
TARGET_PH_MAX       = 7.8
# Proveniências excluídas (drug libraries, não metabolitos)
EXCLUDED_PROVENANCE = {"inext"}
LOG_LEVEL         = logging.INFO
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Multiplicidades válidas (aceitas no banco)
VALID_MULT = {"s", "d", "t", "q", "m", "dd", "dt", "td", "tt", "dq", "qt",
              "sept", "sext", "br s", "br d", "br t", "br m"}


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS DE LEITURA NMR-STAR
# ═══════════════════════════════════════════════════════════════════════════════

def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def normalize(val: str | None) -> str | None:
    if val is None:
        return None
    val = val.strip().strip("'\"")
    if val in (".", "?", ""):
        return None
    return val


def to_float(val: str | None) -> float | None:
    try:
        return float(val) if val not in (None, ".", "?", "") else None
    except (ValueError, TypeError):
        return None


def to_int(val: str | None) -> int | None:
    try:
        return int(float(val)) if val not in (None, ".", "?", "") else None
    except (ValueError, TypeError):
        return None


def get_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(
        r'(?:^|\s)' + re.escape(tag) + r'\s+'
        r"(?:'([^']*)'|\"([^\"]*)\"|(\S+))",
        re.MULTILINE | re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2) or m.group(3)
    return normalize(val)


def get_save_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    pattern = re.compile(r'save_(\S+)(.*?)(?=save_|\Z)', re.DOTALL)
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        content = m.group(2)
        blocks[name] = content
    return blocks


def parse_loop(block: str, prefix: str) -> list[dict[str, str]]:
    loop_pattern = re.compile(
        r'loop_\s+((?:\s*' + re.escape(prefix) + r'\S+\s*)+)(.*?)(?=loop_|save_|\Z)',
        re.DOTALL | re.IGNORECASE,
    )
    results = []
    for lm in loop_pattern.finditer(block):
        header_text = lm.group(1)
        data_text   = lm.group(2)

        cols = re.findall(re.escape(prefix) + r'(\S+)', header_text, re.IGNORECASE)
        if not cols:
            continue

        tokens = tokenize_nmrstar(data_text)
        n = len(cols)
        for i in range(0, len(tokens) - n + 1, n):
            row = {cols[j]: normalize(tokens[i + j]) for j in range(n)}
            results.append(row)
    return results


def tokenize_nmrstar(text: str) -> list[str]:
    tokens = []
    text = re.sub(r'#.*', '', text)
    text = re.sub(r'\n;(.*?)\n;', lambda m: f"'{m.group(1).strip()}'", text, flags=re.DOTALL)
    pattern = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")
    for m in pattern.finditer(text):
        tok = m.group(0)
        if tok.lower() in ("loop_", "stop_", "save_"):
            break
        tokens.append(tok)
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
#  PARSERS ESPECÍFICOS
# ═══════════════════════════════════════════════════════════════════════════════

_STEREO_PREFIX = re.compile(
    r'^(?:'
    r'[LDRSldrs]-|'           # L-, D-, R-, S-
    r'\([LDRSldrs+-]\)-?|'    # (L)-, (D)-, (R)-, (S)-, (+)-, (-)-
    r'meso-|rac-|DL-|'
    r'(?:\([^)]{1,20}\)-?)+' # (2S)-, (R,S)-, etc.
    r')',
    re.IGNORECASE,
)


def _simplify_name(chem_comp_name: str | None,
                   common_name_rows: list[dict]) -> str | None:
    """
    Remove prefixos estereoquímicos de chem_comp_name iterativamente.
    Se o resultado stripped existir nos sinônimos, retorna o sinônimo limpo.
    Se não existir mas o stripped for um nome simples (começa com letra, len>3),
    retorna o stripped mesmo assim — é sempre mais simples que o original.
    """
    if not chem_comp_name:
        return chem_comp_name

    synonyms_map: dict[str, str] = {}
    for r in common_name_rows:
        n = normalize(r.get("Name") or "")
        if not n:
            continue
        key = n.lower()
        # Preferir versão não-all-caps quando há duplicata
        if key not in synonyms_map or (
            synonyms_map[key] == synonyms_map[key].upper() and n != n.upper()
        ):
            synonyms_map[key] = n

    last_stripped = chem_comp_name
    base = chem_comp_name
    prefix_found = False
    for _ in range(5):
        stripped = _STEREO_PREFIX.sub("", base, count=1).strip().strip('-')
        if stripped == base:
            break
        base = stripped
        prefix_found = True
        last_stripped = stripped
        if base.lower() in synonyms_map:
            found = synonyms_map[base.lower()]
            return found.title() if found == found.upper() else found

    if prefix_found and last_stripped and last_stripped[0].isalpha() and len(last_stripped) > 3:
        # Capitalizar primeira letra
        return last_stripped[0].upper() + last_stripped[1:]

    return chem_comp_name


def parse_metabolite(text: str, bmrb_id: str) -> dict:
    meta = {"bmrb_id": bmrb_id}

    chem_comp_name = get_tag(text, "_Chem_comp.Name")
    common_rows    = parse_loop(text, "_Chem_comp_common_name.")
    simplified     = _simplify_name(chem_comp_name, common_rows)

    assembly_name = None
    for row in parse_loop(text, "_Assembly."):
        n = normalize(row.get("Name"))
        if n:
            assembly_name = n
            break

    entity_name = None
    for row in parse_loop(text, "_Entity."):
        n = normalize(row.get("Name"))
        if n:
            entity_name = n
            break

    meta["name"] = (
        simplified
        or chem_comp_name
        or assembly_name
        or entity_name
        or get_tag(text, "_Entry.Title")
    )

    meta["formula"]        = get_tag(text, "_Chem_comp.Formula")
    meta["formula_weight"] = to_float(get_tag(text, "_Chem_comp.Formula_weight"))
    meta["inchi"]          = get_tag(text, "_Chem_comp.InChI_code")

    smiles = None
    smiles_rows = parse_loop(text, "_Chem_comp_SMILES.")
    for row in smiles_rows:
        s = row.get("String")
        if s:
            smiles = s
            if row.get("Type", "").lower() in ("canonical", "isomeric"):
                break
    meta["smiles"] = smiles

    meta["hmdb_id"]    = None
    meta["pubchem_id"] = None
    meta["kegg_id"]    = None
    meta["cas_number"] = None
    meta["chebi_id"]   = None
    db_rows = parse_loop(text, "_Chem_comp_db_link.")
    pubchem_cid = None
    pubchem_any = None
    for row in db_rows:
        db    = (row.get("Database_code") or row.get("Db_code") or "").upper().strip("'\"")
        acc   = normalize(row.get("Accession_code") or row.get("Entry_ID"))
        atype = (row.get("Accession_code_type") or "").lower().strip("'\"")
        if not acc:
            continue
        # HMDB usa código numérico "811" nos arquivos BMRB
        if (db == "811" or "HMDB" in db) and meta["hmdb_id"] is None:
            meta["hmdb_id"] = acc
        elif db == "KEGG" and meta["kegg_id"] is None:
            meta["kegg_id"] = acc
        elif "CAS" in db and meta["cas_number"] is None:
            meta["cas_number"] = acc
        elif db == "CHEBI" and meta["chebi_id"] is None:
            meta["chebi_id"] = acc
        elif "PUBCHEM" in db or "PUB_CHEM" in db:
            if atype == "cid":
                pubchem_cid = acc
            elif pubchem_any is None:
                pubchem_any = acc
    meta["pubchem_id"] = pubchem_cid or pubchem_any

    return meta


def parse_shifts_1h(text: str, bmrb_id: str) -> list[dict]:
    """
    Extrai deslocamentos 1H exclusivamente de blocos de picos espectrais
    de experimentos 1D 1H. Mantém apenas picos que possuem multiplicidade
    (não None). Se existirem atribuições, usa‑as; caso contrário, cria nomes
    sintéticos "peak_<id>" e tenta extrair multiplicidade de Coupling_pattern
    ou do campo Details.
    """
    shifts = []
    blocks = get_save_blocks(text)

    for name, content in blocks.items():
        if not name.startswith("spectral_peak"):
            continue
        exp_name = get_tag(content, "_Spectral_peak_list.Experiment_name") or ""
        if "1D 1H" not in exp_name and not name.startswith("spectral_peak_1H"):
            continue

        # 1. Tentar extrair atribuições (se existirem)
        assign_rows = parse_loop(content, "_Assigned_peak_chem_shift.")
        if assign_rows:
            # Mapear peak_id -> (ppm, mult, j) da dimensão 1
            peak_data = {}
            peak_rows = parse_loop(content, "_Peak_char.")
            for row in peak_rows:
                pid = normalize(row.get("Peak_ID"))
                dim = row.get("Spectral_dim_ID") or "1"
                if dim != "1":
                    continue
                ppm = to_float(row.get("Chem_shift_val"))
                if ppm is None:
                    continue
                mult = normalize(row.get("Coupling_pattern"))
                if mult:
                    mult = mult.lower()
                    if mult not in VALID_MULT:
                        mult = None
                jval = to_float(row.get("Coupling_constant_value"))
                if pid:
                    peak_data[pid] = (ppm, mult, jval)

            for row in assign_rows:
                pid = normalize(row.get("Peak_ID"))
                dim = row.get("Spectral_dim_ID") or "1"
                if dim != "1":
                    continue
                atom = normalize(row.get("Atom_ID"))
                if not pid or not atom or pid not in peak_data:
                    continue
                if not atom.upper().startswith('H'):
                    continue
                ppm, mult, jval = peak_data[pid]
                if mult is not None:
                    shifts.append({
                        "bmrb_id":      bmrb_id,
                        "atom_name":    atom,
                        "ppm":          ppm,
                        "multiplicity": mult,
                        "j_coupling":   jval,
                    })

        else:
            # 2. Sem atribuições: criar entradas por pico
            peak_mult = {}
            # Coletar dados de pico
            for row in parse_loop(content, "_Peak_char."):
                pid = normalize(row.get("Peak_ID"))
                dim = row.get("Spectral_dim_ID") or "1"
                if dim != "1":
                    continue
                ppm = to_float(row.get("Chem_shift_val"))
                if ppm is None:
                    continue
                # Multiplicidade: primeiro tenta Coupling_pattern, depois Details
                mult = normalize(row.get("Coupling_pattern"))
                if not mult:
                    details = row.get("Details") or ""
                    # Exemplo: "5_d_Q3" -> extrai 'd'
                    m = re.search(r'[a-z]+', details)
                    if m:
                        candidate = m.group(0).lower()
                        if candidate in VALID_MULT:
                            mult = candidate
                else:
                    mult = mult.lower()
                    if mult not in VALID_MULT:
                        mult = None
                # Só guarda se multiplicidade for válida
                if mult is not None:
                    peak_mult[pid] = {"ppm": ppm, "mult": mult, "j": None}

            # Constantes de acoplamento (se existirem)
            for row in parse_loop(content, "_Peak_coupling."):
                pid = normalize(row.get("Peak_ID"))
                dim = row.get("Spectral_dim_ID") or "1"
                if dim != "1":
                    continue
                jval = to_float(row.get("Coupling_val"))
                if pid and pid in peak_mult and jval is not None:
                    if peak_mult[pid]["j"] is None:
                        peak_mult[pid]["j"] = f"{jval:.2f}"
                    else:
                        peak_mult[pid]["j"] += f"/{jval:.2f}"

            for pid, data in peak_mult.items():
                j_coupling = data["j"]
                if j_coupling and '/' not in j_coupling:
                    j_coupling = to_float(j_coupling)
                shifts.append({
                    "bmrb_id":      bmrb_id,
                    "atom_name":    f"peak_{pid}",
                    "ppm":          data["ppm"],
                    "multiplicity": data["mult"],
                    "j_coupling":   j_coupling,
                })

    # Fallback: formato _Atom_chem_shift (bmse000854+ e outros arquivos modernos)
    # Estes arquivos guardam shifts em bloco separado "assigned_chemical_shifts"
    if not shifts:
        blocks2 = get_save_blocks(text)
        for name, content in blocks2.items():
            if "assigned_chemical_shift" not in name.lower():
                continue
            for row in parse_loop(content, "_Atom_chem_shift."):
                atype     = (row.get("Atom_type") or "").upper()
                isotope   = row.get("Atom_isotope_number") or "1"
                if atype != "H" or str(isotope) != "1":
                    continue
                atom = normalize(row.get("Atom_ID") or row.get("Auth_atom_ID"))
                ppm  = to_float(row.get("Val"))
                if atom and ppm is not None:
                    shifts.append({
                        "bmrb_id":      bmrb_id,
                        "atom_name":    atom,
                        "ppm":          ppm,
                        "multiplicity": None,
                        "j_coupling":   None,
                    })

    # Fallback 2: formato _Spectral_transition_char (apenas transições, sem atribuição)
    # Agrupa transições próximas em picos compostos e infere multiplicidade.
    if not shifts:
        _MULT_BY_LINES = {1: "s", 2: "d", 3: "t", 4: "q"}
        blocks3 = get_save_blocks(text)
        for name, content in blocks3.items():
            if not name.startswith("spectral_peak"):
                continue
            exp_name = get_tag(content, "_Spectral_peak_list.Experiment_name") or ""
            if "1D 1H" not in exp_name and not name.startswith("spectral_peak_1H"):
                continue
            # Coletar (transition_id → ppm)
            trans_ppm: list[float] = []
            for row in parse_loop(content, "_Spectral_transition_char."):
                dim = row.get("Spectral_dim_ID") or "1"
                if str(dim) != "1":
                    continue
                ppm = to_float(row.get("Chem_shift_val"))
                if ppm is not None:
                    trans_ppm.append(ppm)
            if not trans_ppm:
                continue
            # Agrupar por proximidade (gap > 0.15 ppm = novo grupo)
            trans_ppm.sort()
            groups: list[list[float]] = []
            current: list[float] = [trans_ppm[0]]
            for v in trans_ppm[1:]:
                if v - current[-1] > 0.08:
                    groups.append(current)
                    current = [v]
                else:
                    current.append(v)
            groups.append(current)
            for idx, grp in enumerate(groups, 1):
                n = len(grp)
                mult = _MULT_BY_LINES.get(n, "m")
                shifts.append({
                    "bmrb_id":      bmrb_id,
                    "atom_name":    f"peak_{idx}",
                    "ppm":          round(sum(grp) / n, 5),
                    "multiplicity": mult,
                    "j_coupling":   None,
                })

    return shifts


def parse_transitions(text: str, bmrb_id: str) -> list[dict]:
    transitions = []
    blocks = get_save_blocks(text)
    for name, content in blocks.items():
        if "spectral_peak" not in name.lower():
            continue
        pl_id = normalize(get_tag(content, "_Spectral_peak_list.ID") or "1")
        ints = {}
        for row in parse_loop(content, "_Spectral_transition_general_char."):
            tid = normalize(row.get("Spectral_transition_ID"))
            val = to_float(row.get("Intensity_val"))
            if tid and val is not None:
                ints[tid] = val
        for row in parse_loop(content, "_Spectral_transition_char."):
            tid = normalize(row.get("Spectral_transition_ID"))
            dim = normalize(row.get("Spectral_dim_ID") or "1")
            ppm = to_float(row.get("Chem_shift_val"))
            if not tid or dim != "1" or ppm is None:
                continue
            if not (0.0 <= ppm <= 15.0):
                continue
            transitions.append({
                "bmrb_id":       bmrb_id,
                "peak_list_id":  pl_id or "1",
                "transition_id": tid,
                "ppm":           ppm,
                "intensity":     ints.get(tid),
            })
    return transitions


def parse_sample_conditions(text: str, bmrb_id: str) -> dict:
    cond = {
        "bmrb_id":       bmrb_id,
        "ph":            None,
        "temperature":   None,
        "solvent":       None,
        "field_strength": None,
        "concentration": None,
    }

    blocks = get_save_blocks(text)

    for name, content in blocks.items():
        n = name.lower()
        if "sample_condition" not in n and "cond" not in n:
            continue
        rows = parse_loop(content, "_Sample_condition_variable.")
        for row in rows:
            tipo = (row.get("Type") or "").lower()
            val  = normalize(row.get("Val"))
            if "ph" in tipo and cond["ph"] is None:
                cond["ph"] = to_float(val)
            if "temperature" in tipo and cond["temperature"] is None:
                cond["temperature"] = to_float(val)
        if cond["ph"] is None:
            cond["ph"] = to_float(get_tag(content, "_Sample_condition_variable.Val"))
        if cond["ph"] is None:
            cond["ph"] = to_float(get_tag(content, "_Conditions.pH"))

    for name, content in blocks.items():
        if "sample" not in name.lower():
            continue
        rows = parse_loop(content, "_Sample_component.")
        for row in rows:
            tipo = (row.get("Type") or "").lower()
            comp = normalize(row.get("Mol_common_name") or row.get("Name"))
            if "solvent" in tipo and cond["solvent"] is None and comp:
                cond["solvent"] = comp
            if "solute" in tipo and cond["concentration"] is None:
                unit = (row.get("Concentration_val_units") or "").lower()
                if "mm" in unit or "m" in unit:
                    cond["concentration"] = to_float(row.get("Concentration_val"))

    for name, content in blocks.items():
        fs = (
            get_tag(content, "_NMR_spectrometer.Field_strength")
            or get_tag(content, "_Instrument.Field_strength")
            or get_tag(content, "_Spectrometer.Freq_1H")
        )
        if fs:
            cond["field_strength"] = to_int(fs)
            break

    return cond


def has_1d1h_data(text: str) -> bool:
    blocks = get_save_blocks(text)
    for name, content in blocks.items():
        if not name.startswith("spectral_peak"):
            continue
        exp_name = get_tag(content, "_Spectral_peak_list.Experiment_name") or ""
        if "1D 1H" in exp_name or name.startswith("spectral_peak_1H"):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

DDL = """
CREATE TABLE IF NOT EXISTS metabolites (
    bmrb_id        TEXT PRIMARY KEY,
    name           TEXT,
    formula        TEXT,
    hmdb_id        TEXT,
    pubchem_id     TEXT,
    inchi          TEXT,
    smiles         TEXT,
    kegg_id        TEXT,
    cas_number     TEXT,
    chebi_id       TEXT,
    formula_weight REAL
);

CREATE TABLE IF NOT EXISTS shifts_1h (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bmrb_id      TEXT,
    atom_name    TEXT,
    ppm          REAL,
    multiplicity TEXT,
    j_coupling   REAL,
    FOREIGN KEY(bmrb_id) REFERENCES metabolites(bmrb_id)
);

CREATE TABLE IF NOT EXISTS transitions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    bmrb_id        TEXT,
    peak_list_id   TEXT,
    transition_id  TEXT,
    ppm            REAL,
    intensity      REAL,
    FOREIGN KEY(bmrb_id) REFERENCES metabolites(bmrb_id)
);

CREATE TABLE IF NOT EXISTS samples (
    bmrb_id       TEXT PRIMARY KEY,
    ph            REAL,
    temperature   REAL,
    solvent       TEXT,
    field_strength INTEGER,
    concentration REAL,
    FOREIGN KEY(bmrb_id) REFERENCES metabolites(bmrb_id)
);

CREATE INDEX IF NOT EXISTS idx_shifts_ppm    ON shifts_1h(ppm);
CREATE INDEX IF NOT EXISTS idx_shifts_bmrb   ON shifts_1h(bmrb_id);
CREATE INDEX IF NOT EXISTS idx_meta_name     ON metabolites(name);
CREATE INDEX IF NOT EXISTS idx_trans_ppm      ON transitions(ppm);
CREATE INDEX IF NOT EXISTS idx_trans_bmrb     ON transitions(bmrb_id);
"""

INSERT_META = """
INSERT OR REPLACE INTO metabolites
    (bmrb_id, name, formula, hmdb_id, pubchem_id, inchi, smiles,
     kegg_id, cas_number, chebi_id, formula_weight)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_SHIFT = """
INSERT INTO shifts_1h (bmrb_id, atom_name, ppm, multiplicity, j_coupling)
VALUES (?, ?, ?, ?, ?)
"""

INSERT_TRANSITION = """
INSERT INTO transitions (bmrb_id, peak_list_id, transition_id, ppm, intensity)
VALUES (?, ?, ?, ?, ?)
"""

INSERT_SAMPLE = """
INSERT OR REPLACE INTO samples
    (bmrb_id, ph, temperature, solvent, field_strength, concentration)
VALUES (?, ?, ?, ?, ?, ?)
"""


def init_db(db_path: str) -> sqlite3.Connection:
    if os.path.exists(db_path):
        os.remove(db_path)
        log.info("DB anterior apagado: %s", db_path)
    con = sqlite3.connect(db_path)
    con.executescript(DDL)
    # Migrações (garantir colunas)
    existing_migrations = [
        ("metabolites", "inchi",          "ALTER TABLE metabolites ADD COLUMN inchi TEXT"),
        ("metabolites", "smiles",         "ALTER TABLE metabolites ADD COLUMN smiles TEXT"),
        ("metabolites", "hmdb_id",        "ALTER TABLE metabolites ADD COLUMN hmdb_id TEXT"),
        ("metabolites", "pubchem_id",     "ALTER TABLE metabolites ADD COLUMN pubchem_id TEXT"),
        ("metabolites", "formula",        "ALTER TABLE metabolites ADD COLUMN formula TEXT"),
        ("metabolites", "kegg_id",        "ALTER TABLE metabolites ADD COLUMN kegg_id TEXT"),
        ("metabolites", "cas_number",     "ALTER TABLE metabolites ADD COLUMN cas_number TEXT"),
        ("metabolites", "chebi_id",       "ALTER TABLE metabolites ADD COLUMN chebi_id TEXT"),
        ("metabolites", "formula_weight", "ALTER TABLE metabolites ADD COLUMN formula_weight REAL"),
        ("shifts_1h",   "j_coupling",     "ALTER TABLE shifts_1h ADD COLUMN j_coupling REAL"),
        ("shifts_1h",   "multiplicity",   "ALTER TABLE shifts_1h ADD COLUMN multiplicity TEXT"),
        ("samples",     "concentration",  "ALTER TABLE samples ADD COLUMN concentration REAL"),
    ]
    for table, col, sql in existing_migrations:
        try:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
            if col not in cols:
                con.execute(sql)
                log.info("Migração: coluna '%s.%s' adicionada.", table, col)
        except Exception as e:
            log.debug("Migração ignorada (%s.%s): %s", table, col, e)
    con.commit()
    return con


def insert_entry(con: sqlite3.Connection, meta: dict, shifts: list[dict],
                 trans: list, sample: dict):
    cur = con.cursor()
    cur.execute(INSERT_META, (
        meta["bmrb_id"], meta["name"], meta["formula"],
        meta["hmdb_id"], meta["pubchem_id"], meta["inchi"], meta["smiles"],
        meta["kegg_id"], meta["cas_number"], meta["chebi_id"], meta["formula_weight"],
    ))
    for s in shifts:
        cur.execute(INSERT_SHIFT, (
            s["bmrb_id"], s["atom_name"], s["ppm"],
            s["multiplicity"], s["j_coupling"],
        ))
    for t in trans:
        cur.execute(INSERT_TRANSITION, (
            t["bmrb_id"], t["peak_list_id"], t["transition_id"],
            t["ppm"], t["intensity"],
        ))
    cur.execute(INSERT_SAMPLE, (
        sample["bmrb_id"], sample["ph"], sample["temperature"],
        sample["solvent"], sample["field_strength"], sample["concentration"],
    ))
    con.commit()


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def find_str_files(root: str) -> list[Path]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".str"):
                files.append(Path(dirpath) / fn)
    return sorted(files)


def extract_bmrb_id(text: str, fallback: str) -> str:
    bid = get_tag(text, "_Entry.ID")
    if not bid:
        bid = get_tag(text, "_BMRB_accession_number")
    return bid or fallback


def process_file(path: Path):
    try:
        text = read_file(path)
    except Exception as e:
        log.warning("Erro ao ler %s: %s", path, e)
        return None

    fallback_id = path.parent.name
    bmrb_id = extract_bmrb_id(text, fallback_id)

    if not has_1d1h_data(text):
        log.debug("Sem dados 1D 1H em %s", path.name)
        return None

    meta = parse_metabolite(text, bmrb_id)
    shifts = parse_shifts_1h(text, bmrb_id)
    if not shifts:
        # Não deveria acontecer porque has_1d1h_data garantiu, mas por segurança
        log.debug("Parse_shifts_1h retornou vazio mesmo com has_1d1h_data True em %s", path.name)
        return None

    # Filtrar drug libraries (iNEXT, etc.)
    provenance = (get_tag(text, "_Chem_comp.Provenance") or "").lower()
    if provenance in EXCLUDED_PROVENANCE:
        log.debug("Proveniência excluída (%s) em %s – ignorado.", provenance, path.name)
        return ("EXCLUDED",)

    # Excluir bibliotecas farmacêuticas por prefixo de ID (bmse011xxx, bmse012xxx)
    # Esses ficheiros têm Provenance vazio mas são compostos iNEXT/drug library
    if any(bmrb_id.startswith(p) for p in ("bmse011", "bmse012")):
        log.debug("ID de drug library excluído (%s) em %s – ignorado.", bmrb_id, path.name)
        return ("EXCLUDED",)

    sample = parse_sample_conditions(text, bmrb_id)

    field = sample["field_strength"]
    if field != TARGET_FIELD_MHZ:
        log.debug("Campo %s != %d MHz em %s – ignorado.", field, TARGET_FIELD_MHZ, path.name)
        return ("FIELD_MISMATCH",)

    solvent = (sample["solvent"] or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    if ACCEPTED_SOLVENTS and not any(s in solvent for s in ACCEPTED_SOLVENTS):
        log.debug("Solvente %r não aceito em %s – ignorado.", sample["solvent"], path.name)
        return ("FIELD_MISMATCH",)

    ph = sample["ph"]
    if ph is not None and not (TARGET_PH_MIN <= ph <= TARGET_PH_MAX):
        log.debug("pH %.2f fora de [%.1f, %.1f] em %s – ignorado.",
                  ph, TARGET_PH_MIN, TARGET_PH_MAX, path.name)
        return ("FIELD_MISMATCH",)

    trans = parse_transitions(text, bmrb_id)
    return meta, shifts, trans, sample


def main():
    log.info("━━━ BMRB → SQLite (1D ¹H, metabolitos) ━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Entrada : %s", INPUT_DIR)
    log.info("Saída   : %s", OUTPUT_DB)

    files = find_str_files(INPUT_DIR)
    log.info("Arquivos .str encontrados: %d", len(files))

    if not files:
        log.error("Nenhum arquivo .str encontrado. Verifique INPUT_DIR.")
        return

    con = init_db(OUTPUT_DB)

    ok = 0
    no_1d1h = 0
    field_mismatch = 0
    errors = 0

    for i, path in enumerate(files, 1):
        if i % 500 == 0:
            log.info("  … %d / %d processados", i, len(files))
        try:
            result = process_file(path)
            if result is None:
                no_1d1h += 1
                continue
            if isinstance(result, tuple) and result[0] in ("FIELD_MISMATCH", "EXCLUDED"):
                field_mismatch += 1
                continue
            meta, shifts, trans, sample = result
            insert_entry(con, meta, shifts, trans, sample)
            ok += 1
            log.debug("  ✓ %s  (%d shifts)", meta["bmrb_id"], len(shifts))
        except Exception as e:
            log.warning("ERRO em %s: %s", path.name, e)
            errors += 1

    con.close()

    log.info("━━━ Concluído ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Inseridos (metabolitos 1D ¹H): %d", ok)
    log.info("  Ignorados (sem dados 1D ¹H): %d", no_1d1h)
    log.info("  Ignorados (campo/proveniência): %d", field_mismatch)
    log.info("  Erros      : %d", errors)
    log.info("  Banco      : %s", OUTPUT_DB)

    try:
        con2 = sqlite3.connect(OUTPUT_DB)
        n_meta  = con2.execute("SELECT COUNT(*) FROM metabolites").fetchone()[0]
        n_shift = con2.execute("SELECT COUNT(*) FROM shifts_1h").fetchone()[0]
        n_trans = con2.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        n_samp  = con2.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        con2.close()
        log.info("  metabolites : %d linhas", n_meta)
        log.info("  shifts_1h   : %d linhas", n_shift)
        log.info("  transitions : %d linhas", n_trans)
        log.info("  samples     : %d linhas", n_samp)
    except Exception:
        pass


if __name__ == "__main__":
    main()