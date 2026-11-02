# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""BMRB_Metabolomics_v06: v05 untouched where it had the compound, CASMDB
for the compounds it did not, HMDB for identifiers and for which compounds
are mammalian.

v05 is BMRB alone, measured and audited signal by signal (ADR-049, ADR-060).
Every v05 row travels unchanged. What v06 adds:

1. Identifiers. A v05 compound without ``hmdb_id`` gets the HMDB accession
   whose InChIKey is its own -- the full key, stereo included, so D-carnitine
   does not become L-carnitine. Its ``bmrb_id`` becomes that HMDB id too --
   the annotation keys compounds by ``bmrb_id`` and reads it as the HMDB id
   the pathway statistics look up (remap_id.py did the same for v01); the
   bmse id stays in ``bmse_id``, the old key in ``v06_evidence``.
2. Mammalian. v05 kept in Mammalian what ChEBI places in mammals and cut
   what ChEBI left unknown (TMAO, pyroglutamate). A compound is promoted
   when HMDB lists it "Detected and Quantified" with a normal concentration
   published for a human biospecimen: it is found in people, endogenous or
   from diet and microbiota alike. The HMDB ontology cannot say more --
   its source "Endogenous" (FooDB) is set on 221 688 of 248 079 entries,
   nicotine among them, and the role "Human metabolite" on 151. Drugs
   (a DrugBank entry) stay out: quantified in patients, not metabolites.
3. New compounds from CASMDB (Hayward MW, Mureddu LG, Thompson GS, Phelan
   MM, Brooksbank EJ et al. CASMDB: An Open-Source Database of Metabolite
   Annotation Data for 1D 1H NMR-Based Metabolomics. Anal Chem 2026;98:
   12224-12233, doi:10.1021/acs.analchem.5c04525), compounds v05 does not
   have, from a spectrum CASMDB verified against its experimental one:
   cosine similarity >= 0.9, the paper's acceptance threshold; a score of
   exactly 1.0 is CASMDB's mark of a cleaned spectrum with no experimental
   check and does not count. SDBS spectra are left out (SDBS does not allow
   redistribution), and HMDB spectra recorded outside water.
   Source preference: the GISSMO spin system (shifts and couplings, exact at
   any field; Dashti H et al. Anal Chem 2017;89:12201), then the BMRB peak
   list (D2O, pH 7.4), then HMDB (+-0.005 ppm peak lists).

Protons of a new peak-list signal are its share of the summed line heights
apportioned by largest remainder to the molecule's carbon-bound protons
(the only ones seen in D2O) -- estimated, and said so in ``v06_evidence``.
A spin-system signal holds its spins: stoichiometric by construction.

Nothing is measured on our side for a new compound: the measured columns
stay NULL, ``v04_status`` is 'unverified', ``v06_status`` 'added'.

Licences: HMDB data CC BY-NC 4.0 (the application is not sold -- ADR-090),
BMRB/GISSMO CC0 1.0, CASMDB remediations CC BY 4.0; each new row names its
source in ``ref_sources``.

    python hmdb_api_download/build_reference_v06.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, REPO)
from build_bmrb_db import is_d2o, loop, solvent_names  # noqa: E402
from build_db import carbon_bound_H, cluster_sticks, infer_mult_j  # noqa: E402

from data.casmdb_nef import read_file  # noqa: E402
from processing.spin_patterns import merge_close, merge_lines, profile_of, signal_lines  # noqa: E402
from processing.spin_system import stick_spectrum  # noqa: E402

DBDIR = os.path.join(REPO, 'databases')
SRC = os.path.join(DBDIR, 'BMRB_Metabolomics_v05.db')
DST = os.path.join(DBDIR, 'BMRB_Metabolomics_v06.db')
EXT = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documentos', 'NMR Databases')
NEF_DIR = os.path.join(EXT, 'casmdb_src', 'nef')
BMRB_ENTRIES = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documentos',
                            'bmrb_entries')
CASMDB_GIT = os.path.join(EXT, 'casmdb_src', 'CASMDB')
HMDB_DIR = os.path.join(EXT, 'hmdb_bulk')
HMDB_CACHE = os.path.join(ROOT, '_hmdb_v06_index.json')
SYSTEMS_JSON = os.path.join(REPO, 'data', 'gissmo_spin_systems.json')
REPORT = os.path.join(ROOT, '_v06_report.csv')

MIN_SIMILARITY = 0.9
CLEANED_SIMILARITY = 0.99999
ORIGINS = ('GISSMO', 'BMRB', 'HMDB')
AQUEOUS = ('Water', 'D2O', 'H2O')
# a spin-system fragment above this many spins takes hours to solve
# (acetylcarnitine's 24 310 states; vault GISSMO-espectros-simulados)
MAX_FRAGMENT_SPINS = 10
# couplings below this are not resolved splittings (Hz)
MIN_J_HZ = 0.5
# first order when the shift difference is this many times the coupling
FIRST_ORDER_RATIO = 8.0
# spins this close (ppm) are one signal: GISSMO fits the three protons of a
# methyl with couplings a few tenths of a hertz apart
EQUIV_PPM = 5e-4
# spins of distinct shifts closer than this (ppm, chained) are one listed
# multiplet, as BMRB lists them: GISSMO splits a CH2's two protons
# (5-aminopentanoic acid 2.220 / 2.222)
SIGNAL_MERGE_PPM = 0.015
# a peak-list signal's share of the C-H protons must round this cleanly
MAX_PROTON_REMAINDER = 0.25
# HDO in D2O at 298 K: a singlet listed here is the solvent, not the compound
HDO_PPM = (4.70, 4.90)
PATTERN_FIELD_MHZ = 500.0
# a CASMDB multiplet wider than this is not one signal: its BMRB peak lists
# often come as a single multiplet of every line (norvaline: 46 lines)
MAX_MULTIPLET_PPM = 0.15
# a reference spectrum recorded outside this pH is another protonation
# state (xanthosine's BMRB entry at pH 10.5)
PH_RANGE = (6.0, 8.0)
_LETTER = {1: 's', 2: 'd', 3: 't', 4: 'q', 5: 'quint', 6: 'sext', 7: 'sept'}
_HMDB_TO_V05 = {'p': 'quint', 'h': 'sext', 'hept': 'sept'}
LICENCE = {'HMDB': 'CC BY-NC 4.0', 'BMRB': 'CC0 1.0', 'GISSMO': 'CC0 1.0'}


# --------------------------------------------------------------- HMDB index
def hmdb_index(path=HMDB_CACHE) -> dict:
    """``{hmdb_id: {...}}`` from the three HMDB bulk archives: status,
    InChIKey, the human biospecimens with a normal concentration, biological
    locations (tissues, cells), and the experimental 1H spectra (field,
    solvent, pH, temperature, reference, peaks). Built once (~15 min), then
    cached."""
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    idx = {}

    def each(archive):
        with tarfile.open(os.path.join(HMDB_DIR, archive), 'r:gz') as tf:
            for m in tf:
                if m.isfile() and m.name.endswith('.json'):
                    yield list(json.load(tf.extractfile(m)).values())[0]

    for d in each('hmdb_metabolites.tar.gz'):
        idx[d['hmdb_id']] = {
            'name': d.get('name'), 'status': d.get('status') or '',
            'inchikey': d.get('moldb_inchikey'),
            'biospecimens': sorted({b.get('biospecimen') for b in
                                    d.get('biospecimen_normal') or ()
                                    if b.get('biospecimen')}),
            'synonyms': list(d.get('synonyms') or ()),
            'kegg': (d.get('external_links') or {}).get('kegg_id'),
            'chebi': (d.get('external_links') or {}).get('chebi_id'),
            'drugbank': (d.get('external_links') or {}).get('drugbank_id'),
        }
    for d in each('hmdb_metabolites_ontology.tar.gz'):
        e = idx.setdefault(d['hmdb_id'], {})
        o = d.get('functional_ontology') or {}
        disp = o.get('disposition') or {}
        e['locations'] = sorted({b.get('name') for b in
                                 disp.get('biological_location') or ()
                                 if b.get('name')})
    for d in each('hmdb_metabolites_spectra.tar.gz'):
        exp = [s for s in d.get('nmr') or ()
               if s.get('nucleus') == '1H' and s.get('nmr_type') == '1D NMR'
               and 'MHz' in str(s.get('frequency') or '')]
        if exp:
            idx.setdefault(d['hmdb_id'], {})['nmr_1h'] = [{
                'field': s.get('frequency'), 'solvent': s.get('solvent'),
                'ph': s.get('sample_ph'), 'temp': s.get('sample_temperature'),
                'reference': s.get('chemical_shift_reference'),
                'peaks': sorted(float(p['ppm']) for p in s.get('peaks') or ()
                                if p.get('ppm') is not None)} for s in exp]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(idx, f)
    return idx


def is_mammalian(h: dict | None) -> bool:
    """Quantified in a human biospecimen, with a published normal value, and
    not a drug (no DrugBank entry). A drug is quantified in patients too:
    metformin's 3.03 singlet, 6 H, sits on creatinine's methyl and took its
    error from 7 to 54% in the 18-compound tube (ADR-090). Compounds v05
    already kept are not re-judged -- alanine has a DrugBank entry."""
    return bool(h) and h.get('status') == 'Detected and Quantified' \
        and bool(h.get('biospecimens')) and not h.get('drugbank')


def inchikey(inchi):
    if not inchi:
        return None
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        return Chem.InchiToInchiKey(inchi) or None
    except Exception:  # noqa: BLE001 -- RDKit raises its own on bad InChI
        return None


# ----------------------------------------------------------------- choosing
def usable(spec) -> bool:
    s = spec.similarity
    return (spec.origin in ORIGINS and s is not None
            and MIN_SIMILARITY <= s < CLEANED_SIMILARITY)


def best_spectrum(entry):
    """The spectrum a new compound is built from: the first origin of
    ``ORIGINS`` with a usable one, the highest score among those."""
    for origin in ORIGINS:
        cands = [s for s in entry.spectra if s.origin == origin and usable(s)]
        if origin == 'GISSMO':
            cands = [s for s in cands if s.spin_ppm
                     and max_fragment(len(s.spin_ppm), s.spin_j) <= MAX_FRAGMENT_SPINS]
        else:
            cands = [s for s in cands if s.peaks]
        if cands:
            return max(cands, key=lambda s: s.similarity)
    return None


def bmrb_sample(bmse, entries=BMRB_ENTRIES):
    """``{'solvent': [names], 'ph': str, 'temp': str}`` of a BMRB entry (its
    NMR-STAR sample and sample conditions), or None when the entry file is
    not at hand. A GISSMO system is fitted to
    its BMRB entry's spectrum, so it shares the entry's solvent. Most BMRB
    entries CASMDB adds over v05 are in CDCl3, DMSO or methanol -- the ones
    v01 left out for that reason (build_bmrb_db)."""
    path = os.path.join(entries, str(bmse), f'{bmse}.str')
    if not bmse or not os.path.exists(path):
        return None
    with open(path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    cond = {r.get('Type', '').strip("'").lower(): r for r in
            loop(text, '_Sample_condition_variable')}
    ph = cond.get('ph', {}).get('Val')
    temp = cond.get('temperature', {})
    return {'solvent': solvent_names(text),
            'ph': ph if ph not in (None, '.', '?') else None,
            'temp': (f"{temp['Val']} {temp.get('Val_units', '')}".strip()
                     if temp.get('Val') not in (None, '.', '?') else None)}


def ph_outside(ph) -> bool:
    """True only for a pH known and outside ``PH_RANGE``."""
    try:
        v = float(str(ph).strip())
    except (TypeError, ValueError):
        return False
    return not PH_RANGE[0] <= v <= PH_RANGE[1]


def hmdb_conditions(h: dict | None, spec):
    """The HMDB experimental spectrum a CASMDB HMDB peak list came from --
    same field, peaks nearest -- or None when none is in water."""
    best, best_d = None, None
    for s in (h or {}).get('nmr_1h') or ():
        try:
            f = float(str(s['field']).replace('MHz', ''))
        except ValueError:
            continue
        if spec.field_mhz and abs(f - spec.field_mhz) > 1 or not s['peaks']:
            continue
        ref = np.asarray(s['peaks'])
        d = float(np.mean([np.min(np.abs(ref - p)) for p, _h in spec.peaks]))
        if best_d is None or d < best_d:
            best, best_d = s, d
    if best is None or str(best.get('solvent')) not in AQUEOUS:
        return None
    return best


# ------------------------------------------------------------ spin systems
def max_fragment(n, couplings) -> int:
    """Largest set of spins linked by couplings."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, v in couplings:
        if v:
            parent[find(i)] = find(j)
    return max(Counter(find(i) for i in range(n)).values()) if n else 0


def equivalent_groups(ppm) -> list[list[int]]:
    """Spins of one shift: one signal."""
    groups, seen = [], set()
    for i in range(len(ppm)):
        if i in seen:
            continue
        g = [j for j in range(i, len(ppm))
             if j not in seen and abs(ppm[i] - ppm[j]) <= EQUIV_PPM]
        seen.update(g)
        groups.append(g)
    return groups


def listed_groups(ppm, groups) -> list[list[int]]:
    """Groups of equivalent spins chained within ``SIGNAL_MERGE_PPM``."""
    order = sorted(groups, key=lambda g: ppm[g[0]])
    out = [list(order[0])] if order else []
    for g in order[1:]:
        if ppm[g[0]] - ppm[out[-1][-1]] <= SIGNAL_MERGE_PPM:
            out[-1].extend(g)
        else:
            out.append(list(g))
    return out


def carbon_h(smiles, inchi) -> int | None:
    """Carbon-bound protons from the SMILES, or the InChI where it has none."""
    n = carbon_bound_H(smiles)
    if n or not inchi:
        return n
    from rdkit import Chem
    m = Chem.MolFromInchi(inchi)
    return carbon_bound_H(Chem.MolToSmiles(m)) if m is not None else None


def spin_signals(spec) -> list[dict]:
    """One signal per listed multiplet (``listed_groups``): protons = its
    spins; a letter where it is one equivalent group coupled first-order to
    one other, 'm' otherwise; lines from the system solved at
    ``PATTERN_FIELD_MHZ``."""
    ppm = np.asarray(spec.spin_ppm, dtype=float)
    n = len(ppm)
    J = np.zeros((n, n))
    for i, j, v in spec.spin_j:
        J[i, j] = J[j, i] = v
    groups = equivalent_groups(ppm)
    owner = {s: k for k, g in enumerate(groups) for s in g}
    sp, si = stick_spectrum({'ppm': ppm, 'J': J, 'weight': np.ones(n),
                             'external': []}, PATTERN_FIELD_MHZ)
    out = []
    for spins in listed_groups(ppm, groups):
        c = float(np.mean(ppm[spins]))
        mine = {owner[s] for s in spins}
        partners = {}
        for s in spins:
            for k in range(n):
                if owner[k] not in mine and abs(J[s, k]) >= MIN_J_HZ:
                    partners.setdefault(owner[k], set()).add(round(abs(J[s, k]), 2))
        first_order = all(
            abs(c - ppm[groups[h][0]]) * PATTERN_FIELD_MHZ >= FIRST_ORDER_RATIO * max(js)
            for h, js in partners.items())
        jvals = sorted({j for js in partners.values() for j in js})
        if not partners:
            mult, jtxt = 's', None
        elif (len(mine) == 1 and len(partners) == 1 and first_order
              and len(next(iter(partners.values()))) == 1
              and len(groups[next(iter(partners))]) + 1 in _LETTER):
            mult = _LETTER[len(groups[next(iter(partners))]) + 1]
            jtxt = f'{jvals[0]:.2f}'
        else:
            mult, jtxt = 'm', ' '.join(f'{j:.2f}' for j in jvals)
        reach = float(np.max(np.abs(ppm[spins] - c))) + EQUIV_PPM
        lines = (signal_lines(ppm, sp, si, c, PATTERN_FIELD_MHZ, reach)
                 if partners else None)
        out.append({'ppm': c, 'protons': len(spins), 'multiplicity': mult,
                    'j_coupling': jtxt, 'lines': lines,
                    'atom_name': 'spin ' + ','.join(str(s) for s in sorted(spins)),
                    'ppm_min': c + (min(h for h, _r in lines) / PATTERN_FIELD_MHZ if lines else 0),
                    'ppm_max': c + (max(h for h, _r in lines) / PATTERN_FIELD_MHZ if lines else 0),
                    'protons_source': 'spins'})
    return out


def spin_transitions(spec):
    ppm = np.asarray(spec.spin_ppm, dtype=float)
    n = len(ppm)
    J = np.zeros((n, n))
    for i, j, v in spec.spin_j:
        J[i, j] = J[j, i] = v
    sp, si = stick_spectrum({'ppm': ppm, 'J': J, 'weight': np.ones(n),
                             'external': []}, PATTERN_FIELD_MHZ)
    # the annotation reads these rows as the lines a spectrum shows: raw
    # degenerate transitions handed a doublet 223 lines to fit per sample
    lines = merge_close(np.asarray(sp) * PATTERN_FIELD_MHZ, si)
    top = max(i for _h, i in lines)
    return ([(h / PATTERN_FIELD_MHZ, i) for h, i in lines if i > 1e-3 * top],
            PATTERN_FIELD_MHZ)


# --------------------------------------------------------------- peak lists
def apportion(weights, total) -> list[int]:
    """Integers summing to ``total`` in proportion to ``weights``, by largest
    remainder."""
    w = np.asarray(weights, dtype=float)
    if total <= 0 or w.sum() <= 0:
        return [0] * len(w)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - base))[:total - base.sum()]:
        base[k] += 1
    return base.tolist()


def peak_signals(spec, total) -> list[dict]:
    """One signal per CASMDB multiplet (or per cluster of lines where the
    list has none), protons apportioned from the line heights to the
    ``total`` carbon-bound protons. Empty when a share does not round
    cleanly: summed heights stand for areas only under equal line widths,
    and a count read wrong biases every concentration (ADR-060)."""
    groups = [m['peaks'] for m in spec.multiplets if m['peaks']]
    if any(max(p for p, _h in g) - min(p for p, _h in g) > MAX_MULTIPLET_PPM
           for g in groups):
        groups = []
    if not groups:
        groups = [[(c['ppm'], c['int']) for c in cl] for cl in
                  cluster_sticks([{'ppm': p, 'intensity': h} for p, h in spec.peaks])]
    groups = [g for g in groups if not (
        len(g) == 1 and HDO_PPM[0] <= g[0][0] <= HDO_PPM[1])]
    if not total or not groups:
        return []
    weights = [sum(max(h, 0.0) for _p, h in g) for g in groups]
    if sum(weights) <= 0:
        return []
    raw = np.asarray(weights) / sum(weights) * total
    if np.any(np.abs(raw - np.round(raw)) > MAX_PROTON_REMAINDER):
        return []
    protons = apportion(weights, total)
    field = spec.field_mhz or PATTERN_FIELD_MHZ
    out = []
    for g, w, nh in zip(groups, weights, protons):
        if nh == 0 or w <= 0:
            continue
        ppms = np.array([p for p, _h in g])
        hs = np.array([max(h, 0.0) for _p, h in g])
        c = float(np.sum(ppms * hs) / hs.sum())
        mult, j = infer_mult_j([{'ppm': p} for p in sorted(ppms)], field)
        mult = _HMDB_TO_V05.get(mult, mult)
        lines = merge_lines((ppms - c) * field, hs) if len(g) > 1 else None
        out.append({'ppm': c, 'protons': nh, 'multiplicity': mult,
                    'j_coupling': None if j is None else f'{j:.2f}',
                    'lines': lines, 'atom_name': None,
                    'ppm_min': float(ppms.min()), 'ppm_max': float(ppms.max()),
                    'protons_source': f'estimated: line heights over {total} C-H'})
    return out


# -------------------------------------------------------------------- build
SCHEMA_V06 = (
    'ALTER TABLE metabolites ADD COLUMN v06_status TEXT',
    'ALTER TABLE metabolites ADD COLUMN v06_evidence TEXT',
    'ALTER TABLE shifts_1h ADD COLUMN v06_status TEXT',
    'ALTER TABLE shifts_1h ADD COLUMN v06_evidence TEXT',
    'CREATE TABLE ref_sources(bmrb_id TEXT, origin TEXT, accession TEXT, '
    'casmdb_file TEXT, spectrum_file TEXT, field_mhz REAL, similarity REAL, '
    'solvent TEXT, ph TEXT, temperature TEXT, reference TEXT, licence TEXT)',
    'CREATE TABLE synonyms(bmrb_id TEXT, synonym TEXT, source TEXT)',
    'CREATE TABLE biospecimens(bmrb_id TEXT, hmdb_id TEXT, biospecimen TEXT, '
    'kind TEXT)',
    'CREATE TABLE build_info(key TEXT PRIMARY KEY, value TEXT)',
)


def _casmdb_commit():
    try:
        return subprocess.run(['git', '-C', CASMDB_GIT, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _new_id(entry, h_id, taken):
    for cand in (h_id, entry.bmrb_id, 'CASMDB:' + entry.name):
        if cand and cand not in taken:
            return cand
    raise ValueError(f'no free id for {entry.name}')


def _display_name(entry, h):
    if h and h.get('name'):
        return h['name']
    return entry.name.replace('_', ' ')


def _add_compound(con, bid, name, entry, h_id, h, spec, signals, trans,
                  trans_field, mammalian, conditions, systems_out):
    solvent = 'D2O' if spec.origin in ('GISSMO', 'BMRB') else str(conditions['solvent'])
    ph = conditions.get('ph')
    evidence = json.dumps({'origin': spec.origin, 'similarity': round(spec.similarity, 4),
                           'casmdb_file': entry.name, 'licence': LICENCE[spec.origin]})
    con.execute(
        'insert into metabolites(bmrb_id, hmdb_id, name, formula, inchi, inchikey, '
        'smiles, solvent, field_mhz, n_1h_signals, total_protons, chebi_id, '
        'keep_mammalian, keep_reason, bmse_id, name_full, v06_status, v06_evidence) '
        'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (bid, h_id, name, entry.formula, entry.inchi, inchikey(entry.inchi),
         entry.smiles, solvent, spec.field_mhz, len(signals),
         sum(s['protons'] for s in signals), (h or {}).get('chebi'),
         '1' if mammalian else '?',
         ('mammalian: HMDB quantified in human biospecimen' if mammalian
          else 'unknown (CASMDB, not quantified in HMDB)'),
         entry.bmrb_id, name, 'added', evidence))
    for s in signals:
        lines = s['lines']
        con.execute(
            'insert into shifts_1h(bmrb_id, hmdb_id, atom_name, ppm, ppm_min, ppm_max, '
            'multiplicity, j_coupling, protons, n_lines, intensity, field_mhz, solvent, '
            'pattern_lines, pattern_n, pattern_profile, profile_fwhm_hz, v04_status, '
            'v06_status, v06_evidence) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (bid, h_id, s['atom_name'], round(s['ppm'], 4), round(s['ppm_min'], 4),
             round(s['ppm_max'], 4), s['multiplicity'], s['j_coupling'], s['protons'],
             len(lines) if lines else 1, 1.0, spec.field_mhz or PATTERN_FIELD_MHZ, solvent,
             json.dumps(lines) if lines else None, len(lines) if lines else None,
             profile_of(lines) if lines else None,
             0.5 if lines else None, 'unverified', 'added',
             json.dumps({'origin': spec.origin, 'protons': s['protons_source']})))
    con.executemany('insert into transitions(bmrb_id, hmdb_id, ppm, intensity, freq_mhz) '
                    'values (?,?,?,?,?)',
                    [(bid, h_id, p, i, trans_field) for p, i in trans])
    con.execute('insert into ref_sources values (?,?,?,?,?,?,?,?,?,?,?,?)',
                (bid, spec.origin,
                 h_id if spec.origin == 'HMDB' else entry.bmrb_id,
                 entry.name, spec.source_file, spec.field_mhz, spec.similarity,
                 solvent, ph, (conditions or {}).get('temp'),
                 (conditions or {}).get('reference') or 'DSS', LICENCE[spec.origin]))
    if spec.origin == 'GISSMO':
        systems_out.append({'name': name, 'bmse': entry.bmrb_id,
                            'gissmo_file': spec.source_file, 'gissmo_name': entry.name,
                            'fit_field_mhz': spec.field_mhz or PATTERN_FIELD_MHZ,
                            'lw_hz': None, 'ppm': list(spec.spin_ppm),
                            'J_hz': [[i, j, v] for i, j, v in spec.spin_j],
                            'external': [], 'source': 'CASMDB'})


def build(dst=DST, nef_dir=NEF_DIR, systems_json=SYSTEMS_JSON) -> Counter:
    idx = hmdb_index()
    by_key = {v['inchikey']: k for k, v in idx.items() if v.get('inchikey')}
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(SRC, dst)
    con = sqlite3.connect(dst)
    tally, report = Counter(), []
    with con:
        for sql in SCHEMA_V06:
            con.execute(sql)
        con.executemany('insert into build_info values (?,?)', [
            ('version', 'v06'), ('source_db', os.path.basename(SRC)),
            ('casmdb_commit', _casmdb_commit()),
            ('min_similarity', str(MIN_SIMILARITY))])
        old = con.execute('select bmrb_id, hmdb_id, name, inchikey, inchi, '
                          'keep_mammalian, bmse_id from metabolites').fetchall()
        keys, skeletons, hmdbs, bmses = set(), set(), set(), set()
        ids_in_use = {r[0] for r in old}
        for bid, h_id, name, ik, inchi, keep, bmse in old:
            ik = ik or inchikey(inchi)
            if ik:
                keys.add(ik)
                skeletons.add(ik.split('-')[0])
            if bmse:
                bmses.add(bmse)
            if not h_id and ik in by_key:
                h_id = by_key[ik]
                # the annotation keys every compound by bmrb_id and reads it as
                # its HMDB id (remap_id.py): only the remapped key reaches the
                # pathway statistics. The bmse id stays in bmse_id.
                new_bid = h_id if h_id not in ids_in_use else bid
                con.execute('update metabolites set hmdb_id=?, bmrb_id=?, v06_status=?, '
                            'v06_evidence=? where bmrb_id=?',
                            (h_id, new_bid, 'id_filled', json.dumps(
                                {'hmdb_id': h_id, 'by': 'InChIKey',
                                 'bmrb_id_was': bid}), bid))
                for t in ('shifts_1h', 'transitions'):
                    con.execute(f'update {t} set hmdb_id=?, bmrb_id=? '  # noqa: S608 -- fixed names
                                'where bmrb_id=?', (h_id, new_bid, bid))
                ids_in_use.discard(bid)
                ids_in_use.add(new_bid)
                bid = new_bid
                tally['hmdb_id filled'] += 1
            if h_id:
                hmdbs.add(h_id)
            h = idx.get(h_id) if h_id else None
            if keep != '1' and is_mammalian(h):
                prev = con.execute('select v06_evidence from metabolites where bmrb_id=?',
                                   (bid,)).fetchone()[0]
                ev = json.loads(prev) if prev else {}
                ev.update({'keep_mammalian_was': keep,
                           'why': 'HMDB quantified in ' + ', '.join(h['biospecimens'])
                           })
                con.execute('update metabolites set keep_mammalian=?, keep_reason=?, '
                            'v06_status=?, v06_evidence=? where bmrb_id=?',
                            ('1', 'mammalian: HMDB quantified in human biospecimen',
                             'promoted', json.dumps(ev), bid))
                tally[f'promoted from {keep!r}'] += 1
                report.append(('promoted', name, h_id, keep))
            if h:
                _store_hmdb_side(con, bid, h_id, h)
        taken = set(ids_in_use)
        systems_out = []
        for path in sorted(glob.glob(os.path.join(nef_dir, '*.nef'))):
            e = read_file(path)
            ik = inchikey(e.inchi)
            if ((ik and (ik in keys or ik.split('-')[0] in skeletons))
                    or (e.hmdb_id and e.hmdb_id in hmdbs)
                    or (e.bmrb_id and e.bmrb_id in bmses)):
                tally['casmdb already in v05'] += 1
                continue
            spec = best_spectrum(e)
            if spec is None:
                tally['casmdb: no verified spectrum'] += 1
                report.append(('no verified spectrum', e.name, e.hmdb_id, ''))
                continue
            h_id = e.hmdb_id or by_key.get(ik)
            h = idx.get(h_id) if h_id else None
            conditions = None
            if spec.origin in ('GISSMO', 'BMRB'):
                sample = bmrb_sample(e.bmrb_id)
                names = (sample or {}).get('solvent')
                if not names or not is_d2o(names):
                    why = 'solvent unknown' if not names else 'not in D2O: ' + ', '.join(names)
                    tally['casmdb: BMRB entry ' + why.split(':')[0]] += 1
                    report.append((why, e.name, e.hmdb_id, e.bmrb_id))
                    continue
                conditions = {'solvent': ', '.join(names), 'ph': sample['ph'],
                              'temp': sample['temp'], 'reference': None}
            if spec.origin == 'HMDB':
                conditions = hmdb_conditions(h, spec)
                if conditions is None:
                    tally['casmdb: HMDB spectrum not in water'] += 1
                    report.append(('not in water', e.name, h_id, ''))
                    continue
            if ph_outside((conditions or {}).get('ph')):
                tally['casmdb: pH outside 6-8'] += 1
                report.append((f"pH {conditions['ph']}", e.name, h_id, spec.origin))
                continue
            if spec.origin == 'GISSMO':
                signals = spin_signals(spec)
                if sum(s['protons'] for s in signals) != carbon_h(e.smiles, e.inchi):
                    signals = []
                trans, tfield = spin_transitions(spec)
            else:
                signals = peak_signals(spec, carbon_h(e.smiles, e.inchi))
                trans, tfield = list(spec.peaks), spec.field_mhz or PATTERN_FIELD_MHZ
            if not signals:
                tally['casmdb: protons do not match the C-H count'] += 1
                report.append(('no signal', e.name, h_id, spec.origin))
                continue
            bid = _new_id(e, h_id, taken)
            taken.add(bid)
            keys.add(ik)
            skeletons.add(ik.split('-')[0])
            if h_id:
                hmdbs.add(h_id)
            mam = is_mammalian(h)
            _add_compound(con, bid, _display_name(e, h), e, h_id, h, spec, signals,
                          trans, tfield, mam, conditions, systems_out)
            for syn in sorted(set(e.synonyms)):
                con.execute('insert into synonyms values (?,?,?)', (bid, syn, 'CASMDB'))
            if h:
                _store_hmdb_side(con, bid, h_id, h)
            tally[f'added from {spec.origin}' + (' (mammalian)' if mam else '')] += 1
            report.append(('added', _display_name(e, h), h_id, spec.origin))
    con.execute('VACUUM')
    con.close()
    _write_systems(systems_json, systems_out)
    import csv
    with open(REPORT, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([('what', 'name', 'hmdb_id', 'detail')] + report)
    return tally


def _store_hmdb_side(con, bid, h_id, h):
    con.executemany('insert into biospecimens values (?,?,?,?)',
                    [(bid, h_id, b, 'biospecimen') for b in h.get('biospecimens') or ()]
                    + [(bid, h_id, b, 'location') for b in h.get('locations') or ()])
    con.executemany('insert into synonyms values (?,?,?)',
                    [(bid, s['name'], 'HMDB/' + str(s.get('source') or ''))
                     if isinstance(s, dict) else (bid, s, 'HMDB')
                     for s in (h.get('synonyms') or ())[:60]
                     if (s.get('name') if isinstance(s, dict) else s)])


def _write_systems(path, new):
    """The new compounds' spin systems join the bundled extract under their
    database names, so ``spin_patterns`` solves them at the study's field.
    Systems already there are kept as they are."""
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    doc['systems'] = [s for s in doc['systems'] if s.get('source') != 'CASMDB']
    have = {s['name'] for s in doc['systems']}
    doc['systems'] += [s for s in new if s['name'] not in have]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


# The databases the application offers: BMRB_<matrix>_nMRPro-H.db, every one
# in D2O. General and Mammalian split by origin as v05 did (keep_mammalian);
# each matrix is the Mammalian compounds HMDB quantifies in that biospecimen
# (a published normal concentration; Wishart DS et al. HMDB 5.0. Nucleic
# Acids Res 2022;50:D622). A compound HMDB has not measured there is left
# out of the matrix, not judged absent -- it stays in Mammalian and General.
MATRICES = (
    ('Blood', 'Blood'), ('Urine', 'Urine'), ('Feces', 'Feces'),
    ('Saliva', 'Saliva'), ('CSF', 'Cerebrospinal Fluid (CSF)'),
    ('Milk', 'Breast Milk'), ('Sweat', 'Sweat'))
_SUBSET_TABLES = ('metabolites', 'shifts_1h', 'transitions', 'ref_sources',
                  'synonyms', 'biospecimens')


def output_name(matrix: str) -> str:
    return f'BMRB_{matrix}_nMRPro-H.db'


def make_subset(src, dst, keep_sql, params=()) -> int:
    """``dst``: a copy of ``src`` holding only the compounds ``keep_sql``
    (a query of bmrb_id over ``metabolites``) selects; every table follows."""
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(src, dst)
    con = sqlite3.connect(dst)
    with con:
        keep = {r[0] for r in con.execute(keep_sql, params)}
        drop = [(r[0],) for r in con.execute('select bmrb_id from metabolites')
                if r[0] not in keep]
        for t in _SUBSET_TABLES:
            con.executemany(f'delete from {t} where bmrb_id = ?', drop)  # noqa: S608 -- fixed names
    con.execute('VACUUM')
    n = con.execute('select count(*) from metabolites').fetchone()[0]
    con.close()
    return n


def split(src=DST) -> dict:
    """Every offered database from the built v06: ``{file: compounds}``."""
    out = {output_name('General'): make_subset(
               src, os.path.join(DBDIR, output_name('General')),
               "select bmrb_id from metabolites where keep_mammalian in ('1', '?')"),
           output_name('Mammalian'): make_subset(
               src, os.path.join(DBDIR, output_name('Mammalian')),
               "select bmrb_id from metabolites where keep_mammalian = '1'")}
    for matrix, biospecimen in MATRICES:
        out[output_name(matrix)] = make_subset(
            src, os.path.join(DBDIR, output_name(matrix)),
            "select m.bmrb_id from metabolites m where m.keep_mammalian = '1' "
            "and exists (select 1 from biospecimens b where b.bmrb_id = m.bmrb_id "
            "and b.kind = 'biospecimen' and b.biospecimen = ?)", (biospecimen,))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-split', action='store_true')
    args = ap.parse_args()
    tally = build()
    for k, v in sorted(tally.items()):
        print(f'{k}: {v}')
    if not args.no_split:
        for f, n in split().items():
            print(f'{f}: {n} compounds')


if __name__ == '__main__':
    main()
