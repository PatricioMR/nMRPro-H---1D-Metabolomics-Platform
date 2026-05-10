#!/usr/bin/env python3
"""
build_bmrb_db.py — Reconstrói o banco NMR a partir de ficheiros BMRB NMR-STAR (.str).

Filtros aplicados:
  - Apenas compostos com match InChI no HMDB_NAMES.csv (metabolitos mamíferos)
  - Apenas compostos com KEGG ID (papel metabólico documentado)
  - Sem lípidos (por keyword no nome HMDB)
  - Sem drogas/xenobióticos (por keyword na description HMDB)

Melhorias vs bancos anteriores:
  - shifts_1h.protons  → count real de H equivalentes por multiplet
  - shifts_1h.intensity → intensidade normalizada (0-1) por metabolito
  - shifts_1h.j_coupling → calculado a partir do espaçamento das transições
  - metabolites.hmdb_id, .smiles, .pubchem_cid → preenchidos via HMDB_NAMES.csv
  - Todos os entries BMRB (não só 600 MHz)

Uso:
  python scripts/build_bmrb_db.py [--bmrb-dir DIR] [--hmdb-csv CSV] [--output DB]
"""

import re
import sys
import csv
import sqlite3
import argparse
import os
from pathlib import Path
from collections import defaultdict

# ── Caminhos default ───────────────────────────────────────────────────────────
_PROJECT = Path(__file__).resolve().parent.parent
BMRB_DIR  = Path(r"C:\Users\migue\OneDrive\Documentos\bmrb_entries")
HMDB_CSV  = Path(r"C:\Users\migue\OneDrive\Documentos\NMR Databases\HMDB_NAMES.csv")
OUTPUT_DB = _PROJECT / "databases" / "HMDB_Total_v02.db"

# ── Filtros ────────────────────────────────────────────────────────────────────
LIPID_KEYWORDS = [
    'fatty acid', 'phosphatidyl', 'lysophosphatidyl', 'lysophospho',
    'ceramide', 'sphingomyelin', 'sphingosine', 'phytosphingosine',
    'triglyceride', 'triacylglycerol', 'diglyceride', 'diacylglycerol',
    'monoglyceride', 'monoacylglycerol', 'glycerophospholipid',
    'cardiolipin', 'plasmalogen', 'ether lipid',
    'eicosanoid', 'prostaglandin', 'leukotriene', 'thromboxane',
    'cholesteryl ester', 'wax ester', 'acylglycerol', 'acylcarnitine',
    'bile acid',
]

DRUG_MARKERS = [
    'is a drug', 'pharmaceutical drug', 'xenobiotic', 'over-the-counter',
    'prescribed to', 'therapeutic agent used', 'administered to patients',
    'antibiotic', 'antifungal agent', 'antiviral agent', 'antihistamine',
    'antidepressant', 'antipsychotic', 'analgesic drug', 'is a medication',
    'exogenous compound', 'food additive',
]

# ── Schema ─────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS metabolites (
    bmrb_id     TEXT PRIMARY KEY,
    name        TEXT,
    formula     TEXT,
    hmdb_id     TEXT,
    pubchem_cid TEXT,
    inchi       TEXT,
    smiles      TEXT,
    kegg_id     TEXT,
    chebi_id    TEXT
);
CREATE INDEX IF NOT EXISTS idx_met_hmdb  ON metabolites(hmdb_id);
CREATE INDEX IF NOT EXISTS idx_met_inchi ON metabolites(inchi);

CREATE TABLE IF NOT EXISTS shifts_1h (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bmrb_id      TEXT NOT NULL,
    atom_name    TEXT,
    ppm          REAL,
    multiplicity TEXT,
    j_coupling   TEXT,
    intensity    REAL DEFAULT 1.0,
    protons      INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_sh1h_bmrb ON shifts_1h(bmrb_id);
CREATE INDEX IF NOT EXISTS idx_sh1h_ppm  ON shifts_1h(ppm);

CREATE TABLE IF NOT EXISTS shifts_13c (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    bmrb_id   TEXT NOT NULL,
    atom_name TEXT,
    ppm       REAL
);
CREATE INDEX IF NOT EXISTS idx_sh13c_bmrb ON shifts_13c(bmrb_id);

CREATE TABLE IF NOT EXISTS transitions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    bmrb_id       TEXT NOT NULL,
    peak_list_id  TEXT,
    transition_id TEXT,
    ppm           REAL,
    intensity     REAL
);
CREATE INDEX IF NOT EXISTS idx_tr_bmrb ON transitions(bmrb_id);
CREATE INDEX IF NOT EXISTS idx_tr_ppm  ON transitions(ppm);

CREATE TABLE IF NOT EXISTS samples (
    bmrb_id        TEXT PRIMARY KEY,
    ph             REAL,
    temperature    REAL,
    solvent        TEXT,
    field_strength INTEGER,
    concentration  REAL
);

CREATE TABLE IF NOT EXISTS synonyms (
    bmrb_id TEXT NOT NULL,
    name    TEXT,
    type    TEXT
);
CREATE INDEX IF NOT EXISTS idx_syn_bmrb ON synonyms(bmrb_id);
"""

# ── NMR-STAR tokenizer ─────────────────────────────────────────────────────────

def _tokenize(text):
    """Tokeniza NMR-STAR 3.1. Suporta '', "", blocos ;, comentários #."""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in ' \t\r\n':
            i += 1
            continue
        if c == '#':
            while i < n and text[i] != '\n':
                i += 1
            continue
        # Bloco semicolon — só se ';' está no início da linha
        if c == ';' and (i == 0 or text[i - 1] == '\n'):
            j = i + 1
            while j < n:
                nl = text.find('\n', j)
                if nl == -1:
                    j = n
                    break
                if nl + 1 < n and text[nl + 1] == ';':
                    j = nl + 1
                    break
                j = nl + 1
            tokens.append(text[i + 1:j].strip())
            i = j + 1
            continue
        if c == "'":
            j = i + 1
            while j < n:
                if text[j] == "'" and (j + 1 >= n or text[j + 1] in ' \t\r\n'):
                    break
                j += 1
            tokens.append(text[i + 1:j])
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 1
            tokens.append(text[i + 1:j])
            i = j + 1
            continue
        j = i
        while j < n and text[j] not in ' \t\r\n':
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def _parse_frames(tokens):
    """
    Converte lista de tokens em dict de save frames.
    Retorna {frame_code: {'tags': {tag_lower: val}, 'loops': [{'cols': [...], 'rows': [...]}]}}
    """
    frames = {}
    i = 0
    n = len(tokens)
    RESERVED = {'loop_', 'stop_', 'save_', 'global_', 'data_'}

    while i < n:
        tok = tokens[i]
        tok_l = tok.lower()
        if tok_l.startswith('save_') and len(tok) > 5:
            frame_code = tok[5:]
            frame = {'tags': {}, 'loops': []}
            i += 1
            while i < n:
                tok = tokens[i]
                tok_l = tok.lower()
                if tok_l == 'save_':
                    i += 1
                    break
                if tok_l.startswith('save_') and len(tok) > 5:
                    break  # save frame aninhado (não deve acontecer)
                if tok_l == 'loop_':
                    i += 1
                    cols = []
                    while i < n and tokens[i].startswith('_'):
                        cols.append(tokens[i].lower())
                        i += 1
                    rows = []
                    nc = len(cols)
                    if nc > 0:
                        row = []
                        while i < n:
                            t = tokens[i].lower()
                            if t in RESERVED or t.startswith('save_'):
                                break
                            row.append(tokens[i])
                            if len(row) == nc:
                                rows.append(row)
                                row = []
                            i += 1
                        if i < n and tokens[i].lower() == 'stop_':
                            i += 1
                    frame['loops'].append({'cols': cols, 'rows': rows})
                elif tok.startswith('_'):
                    tag = tok.lower()
                    i += 1
                    if i < n:
                        val = tokens[i]
                        if val not in ('?', '.'):
                            frame['tags'][tag] = val
                        i += 1
                else:
                    i += 1
            frames[frame_code] = frame
        else:
            i += 1
    return frames


# ── Parser de um ficheiro .str ─────────────────────────────────────────────────

def _loop_as_dict(loop, key_col, val_cols):
    """Extrai colunas de um loop como dict. Ignora linhas com valores ? ou ."""
    cols = loop['cols']
    if key_col not in cols:
        return {}
    ki = cols.index(key_col)
    vis = {vc: cols.index(vc) for vc in val_cols if vc in cols}
    result = {}
    for row in loop['rows']:
        if len(row) <= ki or row[ki] in ('?', '.'):
            continue
        k = row[ki]
        result[k] = {vc: (row[vi] if len(row) > vi and row[vi] not in ('?', '.') else None)
                     for vc, vi in vis.items()}
    return result


def _safe_float(s):
    if s is None or s in ('?', '.', ''):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_str_file(path: Path):
    """
    Faz parse de um ficheiro BMRB NMR-STAR .str.
    Retorna dict com todos os dados extraídos, ou None se falhar.
    """
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None

    tokens = _tokenize(text)
    frames = _parse_frames(tokens)

    result = {
        'entry_id':   None,
        'name':       None,       # de _Chem_comp.Name
        'formula':    None,
        'inchi':      None,
        'smiles':     None,
        'db_links':   [],         # [(db_name, accession, acc_type)]
        'synonyms':   [],         # [(name, type)]
        'shifts_1h':  [],         # [(atom_name, ppm, multiplicity, j_hz_str, protons, intensity_raw)]
        'shifts_13c': [],         # [(atom_name, ppm)]
        'transitions': [],        # [(peak_list_id, trans_id, ppm, intensity)]
        'sample':     {},         # {ph, temperature, solvent, field_strength, concentration}
        'spectrometers': {},      # {spec_id: field_strength}
        'experiments':   {},      # {exp_id: spec_id}
    }

    # ── Passo 1: entry_id ──────────────────────────────────────────────────────
    for frame in frames.values():
        if '_entry.id' in frame['tags']:
            result['entry_id'] = frame['tags']['_entry.id']
            break
    if result['entry_id'] is None:
        result['entry_id'] = path.stem  # fallback: nome do ficheiro

    # ── Passo 2: Espectrômetros ───────────────────────────────────────────────
    for frame in frames.values():
        tags = frame['tags']
        if '_nmr_spectrometer.id' in tags and '_nmr_spectrometer.field_strength' in tags:
            sid = tags['_nmr_spectrometer.id']
            fs  = _safe_float(tags['_nmr_spectrometer.field_strength'])
            if fs:
                result['spectrometers'][sid] = int(fs)

    # ── Passo 3: Experimentos → espectrômetro ────────────────────────────────
    for frame in frames.values():
        for loop in frame['loops']:
            cols = loop['cols']
            if '_experiment.id' in cols and '_experiment.nmr_spectrometer_id' in cols:
                ei  = cols.index('_experiment.id')
                si  = cols.index('_experiment.nmr_spectrometer_id')
                for row in loop['rows']:
                    if len(row) > max(ei, si):
                        result['experiments'][row[ei]] = row[si]

    # ── Passo 4: Chem comp (identidade química) ───────────────────────────────
    for frame in frames.values():
        tags = frame['tags']
        if '_chem_comp.name' not in tags and '_chem_comp.formula' not in tags \
                and '_chem_comp.inchi_code' not in tags:
            continue
        if result['name'] is None:
            result['name'] = tags.get('_chem_comp.name')
        if result['formula'] is None:
            f = tags.get('_chem_comp.formula', '')
            result['formula'] = f.replace(' ', '') if f else None
        if result['inchi'] is None:
            result['inchi'] = tags.get('_chem_comp.inchi_code')

        for loop in frame['loops']:
            cols = loop['cols']
            # SMILES
            if '_chem_comp_smiles.type' in cols and '_chem_comp_smiles.string' in cols:
                ti = cols.index('_chem_comp_smiles.type')
                si = cols.index('_chem_comp_smiles.string')
                for row in loop['rows']:
                    if len(row) > max(ti, si) and row[si] not in ('?', '.'):
                        if result['smiles'] is None or row[ti].lower() == 'canonical':
                            result['smiles'] = row[si]
            # DB links
            if '_chem_comp_db_link.database_code' in cols \
                    and '_chem_comp_db_link.accession_code' in cols:
                di  = cols.index('_chem_comp_db_link.database_code')
                ai  = cols.index('_chem_comp_db_link.accession_code')
                ati = cols.index('_chem_comp_db_link.accession_code_type') \
                    if '_chem_comp_db_link.accession_code_type' in cols else None
                for row in loop['rows']:
                    if len(row) > max(di, ai) and row[ai] not in ('?', '.'):
                        acc_type = row[ati] if ati is not None and len(row) > ati else ''
                        result['db_links'].append((row[di], row[ai], acc_type))
            # Nomes comuns / sinónimos
            if '_chem_comp_common_name.name' in cols:
                ni  = cols.index('_chem_comp_common_name.name')
                tti = cols.index('_chem_comp_common_name.type') \
                    if '_chem_comp_common_name.type' in cols else None
                for row in loop['rows']:
                    if len(row) > ni and row[ni] not in ('?', '.'):
                        syn_t = row[tti] if tti is not None and len(row) > tti else 'synonym'
                        result['synonyms'].append((row[ni], syn_t))
            # Nomes sistemáticos
            if '_chem_comp_systematic_name.name' in cols:
                ni  = cols.index('_chem_comp_systematic_name.name')
                tti = cols.index('_chem_comp_systematic_name.naming_system') \
                    if '_chem_comp_systematic_name.naming_system' in cols else None
                for row in loop['rows']:
                    if len(row) > ni and row[ni] not in ('?', '.'):
                        ns = row[tti] if tti is not None and len(row) > tti else 'systematic'
                        result['synonyms'].append((row[ni], ns))
        break  # só o primeiro chem_comp frame

    # ── Passo 5: Condições da amostra ─────────────────────────────────────────
    for frame in frames.values():
        for loop in frame['loops']:
            cols = loop['cols']
            if '_sample_condition_variable.type' in cols \
                    and '_sample_condition_variable.val' in cols:
                ti  = cols.index('_sample_condition_variable.type')
                vi  = cols.index('_sample_condition_variable.val')
                for row in loop['rows']:
                    if len(row) <= max(ti, vi) or row[vi] in ('?', '.'):
                        continue
                    t = row[ti].lower()
                    v = row[vi]
                    if 'ph' == t and 'ph' not in result['sample']:
                        result['sample']['ph'] = _safe_float(v)
                    elif 'temperature' in t and 'temperature' not in result['sample']:
                        result['sample']['temperature'] = _safe_float(v)
                    elif 'concentration' in t and 'concentration' not in result['sample']:
                        result['sample']['concentration'] = _safe_float(v)
                    elif 'solvent' in t and 'solvent' not in result['sample']:
                        result['sample']['solvent'] = v
                break  # primeira frame de sample conditions

    # ── Passo 6: 13C shifts atribuídos ────────────────────────────────────────
    for frame in frames.values():
        for loop in frame['loops']:
            cols = loop['cols']
            if '_atom_chem_shift.atom_type' not in cols \
                    or '_atom_chem_shift.val' not in cols:
                continue
            type_i = cols.index('_atom_chem_shift.atom_type')
            iso_i  = cols.index('_atom_chem_shift.atom_isotope_number') \
                if '_atom_chem_shift.atom_isotope_number' in cols else None
            val_i  = cols.index('_atom_chem_shift.val')
            atid_i = cols.index('_atom_chem_shift.atom_id') \
                if '_atom_chem_shift.atom_id' in cols else None
            for row in loop['rows']:
                if len(row) <= max(type_i, val_i) or row[val_i] in ('?', '.'):
                    continue
                atom_t  = row[type_i].upper()
                isotope = row[iso_i] if iso_i is not None and len(row) > iso_i else None
                ppm_val = _safe_float(row[val_i])
                if ppm_val is None:
                    continue
                atom_id = row[atid_i] if atid_i and len(row) > atid_i else None
                if atom_t == 'C' and isotope in (None, '13'):
                    result['shifts_13c'].append((atom_id, ppm_val))

    # ── Passo 7: Peak lists 1H → shifts_1h + transitions ─────────────────────
    # Recolher frames de spectral_peak_list com experimento 1D 1H
    peak_list_frames = {}
    for fcode, frame in frames.items():
        tags = frame['tags']
        if '_spectral_peak_list.id' not in tags:
            continue
        plid = tags['_spectral_peak_list.id']
        exp_id = tags.get('_spectral_peak_list.experiment_id')
        n_dims = int(tags.get('_spectral_peak_list.number_of_spectral_dimensions', 1) or 1)

        # Detectar se é 1D 1H
        is_1d_1h = False
        for loop in frame['loops']:
            cols = loop['cols']
            if '_spectral_dim.atom_type' in cols:
                ai = cols.index('_spectral_dim.atom_type')
                isi = cols.index('_spectral_dim.atom_isotope_number') \
                    if '_spectral_dim.atom_isotope_number' in cols else None
                ndi = cols.index('_spectral_dim.id') if '_spectral_dim.id' in cols else None
                for row in loop['rows']:
                    if len(row) > ai:
                        dim_id = row[ndi] if ndi and len(row) > ndi else '1'
                        if dim_id == '1' and row[ai].upper() == 'H':
                            isotope_ok = (isi is None or len(row) <= isi
                                          or row[isi] in ('1', '?', '.'))
                            if isotope_ok and n_dims == 1:
                                is_1d_1h = True
                                break
        if not is_1d_1h:
            continue

        # Field strength desta peak list
        field = 0
        if exp_id and exp_id in result['experiments']:
            spec_id = result['experiments'][exp_id]
            field = result['spectrometers'].get(spec_id, 0)

        peak_list_frames[plid] = {
            'frame': frame,
            'field': field,
        }

    # Escolher o peak list com maior field strength
    if not peak_list_frames:
        # Fallback: usar _Atom_chem_shift para 1H se não há peak lists 1D
        for frame in frames.values():
            for loop in frame['loops']:
                cols = loop['cols']
                if '_atom_chem_shift.atom_type' not in cols or '_atom_chem_shift.val' not in cols:
                    continue
                type_i = cols.index('_atom_chem_shift.atom_type')
                iso_i  = cols.index('_atom_chem_shift.atom_isotope_number') \
                    if '_atom_chem_shift.atom_isotope_number' in cols else None
                val_i  = cols.index('_atom_chem_shift.val')
                atid_i = cols.index('_atom_chem_shift.atom_id') \
                    if '_atom_chem_shift.atom_id' in cols else None
                for row in loop['rows']:
                    if len(row) <= max(type_i, val_i) or row[val_i] in ('?', '.'):
                        continue
                    atom_t  = row[type_i].upper()
                    isotope = row[iso_i] if iso_i and len(row) > iso_i else None
                    ppm_val = _safe_float(row[val_i])
                    if ppm_val is None:
                        continue
                    atom_id = row[atid_i] if atid_i and len(row) > atid_i else 'H'
                    if atom_t == 'H' and isotope in (None, '1'):
                        result['shifts_1h'].append(
                            (atom_id, ppm_val, None, None, 1, None))
        return result

    # Preferir: 1) maior field_strength, 2) tem _Peak_char (multiplicity + protons)
    def _pl_score(plid):
        pl = peak_list_frames[plid]
        has_pk_char = int(any('_peak_char.chem_shift_val' in lp['cols']
                              for lp in pl['frame']['loops']))
        return (pl['field'], has_pk_char)

    best_plid = max(peak_list_frames, key=_pl_score)
    best_field = peak_list_frames[best_plid]['field']
    result['sample']['field_strength'] = best_field or None
    best_frame = peak_list_frames[best_plid]['frame']

    # Extrair Peak_char: ppm + multiplicity por peak_id
    peaks_ppm  = {}   # peak_id → ppm
    peaks_mult = {}   # peak_id → multiplicity
    peaks_int  = {}   # peak_id → intensity (de _Peak_general_char)
    for loop in best_frame['loops']:
        cols = loop['cols']
        if '_peak_char.peak_id' in cols and '_peak_char.chem_shift_val' in cols:
            pid_i  = cols.index('_peak_char.peak_id')
            ppm_i  = cols.index('_peak_char.chem_shift_val')
            coup_i = cols.index('_peak_char.coupling_pattern') \
                if '_peak_char.coupling_pattern' in cols else None
            dim_i  = cols.index('_peak_char.spectral_dim_id') \
                if '_peak_char.spectral_dim_id' in cols else None
            for row in loop['rows']:
                if len(row) <= max(pid_i, ppm_i) or row[ppm_i] in ('?', '.'):
                    continue
                if dim_i and len(row) > dim_i and row[dim_i] != '1':
                    continue  # só dimensão 1H
                ppm_val = _safe_float(row[ppm_i])
                if ppm_val is None:
                    continue
                pid = row[pid_i]
                if pid not in peaks_ppm:
                    peaks_ppm[pid]  = ppm_val
                    peaks_mult[pid] = row[coup_i].lower() \
                        if coup_i and len(row) > coup_i and row[coup_i] not in ('?', '.', 'm', 'M') \
                        else None

        if '_peak_general_char.peak_id' in cols \
                and '_peak_general_char.intensity_val' in cols:
            pid_i = cols.index('_peak_general_char.peak_id')
            int_i = cols.index('_peak_general_char.intensity_val')
            for row in loop['rows']:
                if len(row) > max(pid_i, int_i) and row[int_i] not in ('?', '.'):
                    v = _safe_float(row[int_i])
                    if v is not None and v > 0:
                        peaks_int[row[pid_i]] = peaks_int.get(row[pid_i], 0) + v

    # Protons por peak_id via _Assigned_peak_chem_shift
    peaks_protons  = defaultdict(set)  # peak_id → set of atom_ids
    peaks_atomname = {}                 # peak_id → primeiro atom_id
    for loop in best_frame['loops']:
        cols = loop['cols']
        if '_assigned_peak_chem_shift.peak_id' in cols \
                and '_assigned_peak_chem_shift.atom_id' in cols:
            pid_i  = cols.index('_assigned_peak_chem_shift.peak_id')
            atid_i = cols.index('_assigned_peak_chem_shift.atom_id')
            dim_i  = cols.index('_assigned_peak_chem_shift.spectral_dim_id') \
                if '_assigned_peak_chem_shift.spectral_dim_id' in cols else None
            for row in loop['rows']:
                if len(row) <= max(pid_i, atid_i):
                    continue
                if dim_i and len(row) > dim_i and row[dim_i] != '1':
                    continue
                pid    = row[pid_i]
                atid   = row[atid_i]
                if atid not in ('?', '.'):
                    peaks_protons[pid].add(atid)
                    if pid not in peaks_atomname:
                        peaks_atomname[pid] = atid

    # Transitions: ppm + intensity
    trans_ppm = {}   # trans_id → ppm
    trans_int = {}   # trans_id → intensity
    for loop in best_frame['loops']:
        cols = loop['cols']
        if '_spectral_transition_char.spectral_transition_id' in cols \
                and '_spectral_transition_char.chem_shift_val' in cols:
            tid_i  = cols.index('_spectral_transition_char.spectral_transition_id')
            ppm_i  = cols.index('_spectral_transition_char.chem_shift_val')
            dim_i  = cols.index('_spectral_transition_char.spectral_dim_id') \
                if '_spectral_transition_char.spectral_dim_id' in cols else None
            for row in loop['rows']:
                if len(row) <= max(tid_i, ppm_i) or row[ppm_i] in ('?', '.'):
                    continue
                if dim_i and len(row) > dim_i and row[dim_i] != '1':
                    continue
                ppm_val = _safe_float(row[ppm_i])
                if ppm_val is not None:
                    trans_ppm[row[tid_i]] = ppm_val

        if '_spectral_transition_general_char.spectral_transition_id' in cols \
                and '_spectral_transition_general_char.intensity_val' in cols:
            tid_i = cols.index('_spectral_transition_general_char.spectral_transition_id')
            int_i = cols.index('_spectral_transition_general_char.intensity_val')
            for row in loop['rows']:
                if len(row) > max(tid_i, int_i) and row[int_i] not in ('?', '.'):
                    v = _safe_float(row[int_i])
                    if v is not None:
                        trans_int[row[tid_i]] = v

    # Registar transitions (linhas individuais)
    max_trans_int = max(abs(v) for v in trans_int.values()) if trans_int else 1.0
    if max_trans_int == 0:
        max_trans_int = 1.0
    for tid, ppm_val in trans_ppm.items():
        intensity_norm = (trans_int[tid] / max_trans_int) if tid in trans_int else None
        result['transitions'].append((best_plid, tid, ppm_val, intensity_norm))

    # Construir shifts_1h: calcular j_coupling e intensity por peak
    all_trans_ppms = sorted(trans_ppm.values())
    max_peak_int = max(abs(v) for v in peaks_int.values()) if peaks_int else 1.0
    if max_peak_int == 0:
        max_peak_int = 1.0

    if not peaks_ppm:
        # Fallback sem _Peak_char: agrupar transitions por proximidade (0.05 ppm)
        used = set()
        for ppm_val in sorted(all_trans_ppms):
            if ppm_val in used:
                continue
            cluster = [p for p in all_trans_ppms
                       if abs(p - ppm_val) <= 0.08 and p not in used]
            for p in cluster:
                used.add(p)
            n_lines = len(cluster)
            mult_map = {1: 's', 2: 'd', 3: 't', 4: 'q', 5: 'quintet', 6: 'sextet'}
            mult = mult_map.get(n_lines, 'm')
            center = sum(cluster) / len(cluster) if cluster else ppm_val
            j_hz = None
            if len(cluster) >= 2 and best_field > 0:
                seps = [abs(cluster[k + 1] - cluster[k]) * best_field
                        for k in range(len(cluster) - 1)]
                j_hz = sum(seps) / len(seps) if seps else None
            j_str = f"{j_hz:.2f}" if j_hz is not None else None
            result['shifts_1h'].append(('H', center, mult, j_str, 1, None))
        return result

    for pid, ppm_center in peaks_ppm.items():
        mult     = peaks_mult.get(pid)
        protons  = len(peaks_protons[pid]) if pid in peaks_protons else 1
        if protons == 0:
            protons = 1
        atom_name = peaks_atomname.get(pid, 'H')

        # Intensidade normalizada
        raw_int = peaks_int.get(pid)
        intensity_norm = (raw_int / max_peak_int) if raw_int is not None else None
        if intensity_norm is None:
            # Fallback: somar intensidades das transitions próximas
            close_ints = [abs(trans_int[tid])
                          for tid, p in trans_ppm.items()
                          if abs(p - ppm_center) <= 0.10 and tid in trans_int]
            if close_ints:
                intensity_norm = sum(close_ints) / max_trans_int

        # J coupling: a partir do espaçamento das transitions próximas (dentro de 0.15 ppm)
        j_str = None
        if mult and mult not in ('s', 'singlet') and best_field > 0:
            close_ppms = sorted(p for p in all_trans_ppms
                                if abs(p - ppm_center) <= 0.15)
            if len(close_ppms) >= 2:
                seps_hz = [abs(close_ppms[k + 1] - close_ppms[k]) * best_field
                           for k in range(len(close_ppms) - 1)]
                seps_hz = [s for s in seps_hz if 0.5 < s < 25.0]
                if seps_hz:
                    j_median = sorted(seps_hz)[len(seps_hz) // 2]
                    j_str = f"{j_median:.2f}"

        result['shifts_1h'].append(
            (atom_name, ppm_center, mult, j_str, protons, intensity_norm))

    return result


# ── Carregar HMDB_NAMES.csv ────────────────────────────────────────────────────

def _strip_stereo(inchi: str) -> str:
    """Remove camadas de estereoquímica do InChI (/b, /t, /m, /s).
    BMRB usa ALATIS InChI sem stereo; HMDB usa InChI completo com stereo.
    """
    parts = inchi.split('/')
    return '/'.join(p for i, p in enumerate(parts)
                    if i < 2 or p[0] not in ('b', 't', 'm', 's'))


def load_hmdb_index(csv_path: Path):
    """
    Carrega HMDB_NAMES.csv e retorna dois dicts:
      index         : {inchi_exact → row_dict}
      index_nostereo: {inchi_stripped → row_dict}  (fallback para match BMRB)
    Aplica filtros: requer KEGG ID, exclui lípidos e drogas.
    """
    print(f"A carregar HMDB_NAMES.csv ({csv_path})...")
    index          = {}   # inchi exacto → row
    index_nostereo = {}   # inchi sem stereo → row (fallback)
    excluded = {'lipid': 0, 'drug': 0, 'no_kegg': 0, 'no_inchi': 0}
    total = 0

    lipid_kw  = [k.lower() for k in LIPID_KEYWORDS]
    drug_marks = [d.lower() for d in DRUG_MARKERS]

    with open(csv_path, encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 50000 == 0:
                print(f"  {total} linhas processadas, {len(index)} aprovados...",
                      flush=True)

            inchi = (row.get('inchi') or '').strip()
            if not inchi or inchi in ('?', '.', 'nan', 'None'):
                excluded['no_inchi'] += 1
                continue

            kegg = (row.get('kegg_id') or '').strip()
            if not kegg or kegg in ('?', '.', 'nan', 'None', ''):
                excluded['no_kegg'] += 1
                continue

            name_lower = (row.get('name') or '').lower()
            if any(kw in name_lower for kw in lipid_kw):
                excluded['lipid'] += 1
                continue

            desc = (row.get('description') or '')[:600].lower()
            if any(dm in desc for dm in drug_marks):
                excluded['drug'] += 1
                continue

            entry = {
                'hmdb_id':    row.get('hmdb_id', ''),
                'name':       row.get('name', ''),
                'kegg_id':    kegg,
                'smiles':     row.get('smiles', ''),
                'pubchem_id': row.get('pubchem_id', ''),
            }
            index[inchi] = entry
            stripped = _strip_stereo(inchi)
            if stripped not in index_nostereo:
                index_nostereo[stripped] = entry

    print(f"  Total: {total} | Aprovados: {len(index)}")
    print(f"  Index nostereo: {len(index_nostereo)} entradas")
    print(f"  Excluídos — sem InChI: {excluded['no_inchi']}, "
          f"sem KEGG: {excluded['no_kegg']}, "
          f"lípidos: {excluded['lipid']}, drogas: {excluded['drug']}")
    return index, index_nostereo


# ── Deduplicação por hmdb_id ───────────────────────────────────────────────────

def deduplicate(conn):
    """Remove entradas duplicadas (mesmo hmdb_id). Mantém a com mais shifts_1h."""
    cur = conn.cursor()
    cur.execute("""
        SELECT m.hmdb_id, m.bmrb_id, COALESCE(s.n, 0) as n
        FROM metabolites m
        LEFT JOIN (
            SELECT bmrb_id, COUNT(*) as n FROM shifts_1h GROUP BY bmrb_id
        ) s ON m.bmrb_id = s.bmrb_id
        WHERE m.hmdb_id IS NOT NULL AND m.hmdb_id != ''
        ORDER BY m.hmdb_id, COALESCE(s.n, 0) DESC, m.bmrb_id ASC
    """)
    rows = cur.fetchall()

    seen     = {}   # hmdb_id → bmrb_id a manter
    to_delete = []
    for hmdb_id, bmrb_id, n in rows:
        if hmdb_id not in seen:
            seen[hmdb_id] = bmrb_id
        else:
            to_delete.append(bmrb_id)

    if to_delete:
        placeholders = ','.join('?' * len(to_delete))
        for tbl in ('shifts_1h', 'shifts_13c', 'transitions', 'samples', 'synonyms'):
            cur.execute(f"DELETE FROM {tbl} WHERE bmrb_id IN ({placeholders})",
                        to_delete)
        cur.execute(f"DELETE FROM metabolites WHERE bmrb_id IN ({placeholders})",
                    to_delete)
        print(f"  Deduplicação: {len(to_delete)} entries removidos")
    else:
        print(f"  Deduplicação: sem duplicados detectados")
    conn.commit()


# ── Base de dados ──────────────────────────────────────────────────────────────

def create_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def insert_entry(conn, data, hmdb_row):
    """Insere um entry aprovado no banco."""
    bmrb_id = data['entry_id']
    cur = conn.cursor()

    # Metabolites
    cur.execute("""
        INSERT OR REPLACE INTO metabolites
            (bmrb_id, name, formula, hmdb_id, pubchem_cid, inchi, smiles, kegg_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bmrb_id,
        hmdb_row['name'],         # nome preferencial: HMDB
        data['formula'],
        hmdb_row['hmdb_id'],
        hmdb_row.get('pubchem_id') or None,
        data['inchi'],
        hmdb_row.get('smiles') or data.get('smiles'),
        hmdb_row['kegg_id'],
    ))

    # Synonyms
    added_syns = {hmdb_row['name'].lower()}
    for name, syn_type in data['synonyms']:
        if name.lower() not in added_syns:
            cur.execute(
                "INSERT INTO synonyms (bmrb_id, name, type) VALUES (?, ?, ?)",
                (bmrb_id, name, syn_type))
            added_syns.add(name.lower())

    # Shifts 1H — normalizar intensidade 0-1
    if data['shifts_1h']:
        max_int = max((abs(s[5]) for s in data['shifts_1h'] if s[5] is not None),
                      default=None)
        for atom_name, ppm, mult, j_str, protons, raw_int in data['shifts_1h']:
            intensity = None
            if raw_int is not None and max_int and max_int > 0:
                intensity = round(abs(raw_int) / max_int, 4)
            cur.execute("""
                INSERT INTO shifts_1h
                    (bmrb_id, atom_name, ppm, multiplicity, j_coupling, intensity, protons)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (bmrb_id, atom_name, round(ppm, 4),
                  mult, j_str,
                  intensity if intensity is not None else 1.0,
                  protons or 1))

    # Shifts 13C
    for atom_name, ppm in data['shifts_13c']:
        cur.execute(
            "INSERT INTO shifts_13c (bmrb_id, atom_name, ppm) VALUES (?, ?, ?)",
            (bmrb_id, atom_name, round(ppm, 4)))

    # Transitions
    for plid, tid, ppm, intensity in data['transitions']:
        cur.execute("""
            INSERT INTO transitions (bmrb_id, peak_list_id, transition_id, ppm, intensity)
            VALUES (?, ?, ?, ?, ?)
        """, (bmrb_id, plid, tid, round(ppm, 4),
              round(intensity, 4) if intensity is not None else None))

    # Samples
    s = data['sample']
    if s:
        cur.execute("""
            INSERT OR REPLACE INTO samples
                (bmrb_id, ph, temperature, solvent, field_strength, concentration)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            bmrb_id,
            s.get('ph'),
            s.get('temperature'),
            s.get('solvent'),
            s.get('field_strength'),
            s.get('concentration'),
        ))


# ── Build principal ────────────────────────────────────────────────────────────

def build_db(bmrb_dir: Path, hmdb_csv: Path, output: Path):
    print(f"\n{'='*60}")
    print(f"  build_bmrb_db.py — HMDB_Total_v02")
    print(f"{'='*60}")
    print(f"  BMRB dir : {bmrb_dir}")
    print(f"  HMDB CSV : {hmdb_csv}")
    print(f"  Output   : {output}")
    print(f"{'='*60}\n")

    # Carregar index HMDB
    hmdb_index, hmdb_index_nostereo = load_hmdb_index(hmdb_csv)

    # Preparar DB
    if output.exists():
        output.unlink()
    conn = sqlite3.connect(output)
    create_schema(conn)

    # Encontrar todos os ficheiros .str
    str_files = sorted(bmrb_dir.glob("*/bmse*.str"))
    total = len(str_files)
    print(f"\nFicheiros .str encontrados: {total}")
    print(f"A processar...\n")

    n_inserted  = 0
    n_no_match  = 0
    n_parse_err = 0

    for i, path in enumerate(str_files):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  [{i+1}/{total}] inseridos={n_inserted} "
                  f"sem_match={n_no_match} erros={n_parse_err}",
                  flush=True)

        data = parse_str_file(path)
        if data is None:
            n_parse_err += 1
            continue

        inchi = data.get('inchi')
        if not inchi:
            n_no_match += 1
            continue

        hmdb_row = hmdb_index.get(inchi)
        if hmdb_row is None:
            # Fallback: InChI sem camadas de estereoquímica (BMRB usa ALATIS)
            hmdb_row = hmdb_index_nostereo.get(_strip_stereo(inchi))
        if hmdb_row is None:
            n_no_match += 1
            continue

        try:
            insert_entry(conn, data, hmdb_row)
            n_inserted += 1
        except Exception as e:
            print(f"  ERRO ao inserir {path.stem}: {e}")
            n_parse_err += 1

        if (i + 1) % 100 == 0:
            conn.commit()

    conn.commit()

    # Deduplicar por hmdb_id
    print("\nA deduplicate por hmdb_id...")
    deduplicate(conn)

    # Stats finais
    cur = conn.cursor()
    stats = {}
    for table in ('metabolites', 'shifts_1h', 'shifts_13c', 'transitions', 'samples', 'synonyms'):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cur.fetchone()[0]

    conn.close()

    print(f"\n{'='*60}")
    print(f"  Build concluído!")
    print(f"  Ficheiros processados : {total}")
    print(f"  Metabolitos inseridos : {n_inserted}")
    print(f"  Sem match HMDB        : {n_no_match}")
    print(f"  Erros de parse        : {n_parse_err}")
    print(f"\n  Tabela          | Rows")
    print(f"  --------------- | -----")
    for t, n in stats.items():
        print(f"  {t:<15} | {n}")
    print(f"{'='*60}")
    print(f"\n  DB pronto: {output}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bmrb-dir', default=str(BMRB_DIR),
                        help='Directório com sub-pastas bmseXXXXXX/')
    parser.add_argument('--hmdb-csv', default=str(HMDB_CSV),
                        help='Caminho para HMDB_NAMES.csv')
    parser.add_argument('--output', default=str(OUTPUT_DB),
                        help='Caminho para o DB de saída (.db)')
    args = parser.parse_args()

    build_db(
        bmrb_dir=Path(args.bmrb_dir),
        hmdb_csv=Path(args.hmdb_csv),
        output=Path(args.output),
    )


if __name__ == '__main__':
    main()
