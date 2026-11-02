# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Read a CASMDB entry: one metabolite per NEF file.

CASMDB collects the 1D 1H annotation data of HMDB, BMRB and GISSMO, matches
them by InChI and remediates each simulation against its experimental
spectrum (cosine similarity >= 0.9 accepted) -- Hayward MW, Mureddu LG,
Thompson GS, Phelan MM, Brooksbank EJ et al. CASMDB: An Open-Source
Database of Metabolite Annotation Data for 1D 1H NMR-Based Metabolomics.
Anal Chem 2026;98:12224-12233, doi:10.1021/acs.analchem.5c04525
(github.com/ccpnmr/CASMDB). Licences: HMDB-derived data CC BY-NC 4.0,
BMRB/GISSMO CC0 1.0, remediated data CC BY 4.0.

A NEF file is NMR-STAR text (Gutmanas A et al. NMR Exchange Format: a
unified and open standard for representation of NMR restraint data. Nat
Struct Mol Biol 2015;22:433-434): ``save_`` frames of ``_tag value`` items
and ``loop_ ... stop_`` tables. The reader below handles the subset the
CASMDB files use -- quoted values, ``;`` text blocks -- and keeps only what
a reference database needs: the substance, and per spectrum its origin,
field, similarity score, peaks, multiplets and (GISSMO) spin matrix.

The files carry no pH, temperature or solvent: those stay with the source
entry (HMDB spectrum, BMRB entry), and a caller that needs them has to look
them up there.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"'(.*?)'(?=\s|$)|\"(.*?)\"(?=\s|$)|(\S+)")


def _tokens(text: str):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(';'):
            block = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(';'):
                block.append(lines[i])
                i += 1
            yield '\n'.join(block).strip('\n'), True
            i += 1
            continue
        for m in _TOKEN.finditer(line):
            if m.group(3) is not None and m.group(3).startswith('#'):
                break
            quoted = m.group(3) is None
            yield (m.group(1) if m.group(1) is not None else
                   m.group(2) if m.group(2) is not None else m.group(3)), quoted
        i += 1


def parse_frames(text: str) -> dict:
    """``{framecode: {'items': {tag: value}, 'loops': {category: [row]}}}``
    with each loop row a ``{tag: value}`` dict; '.' and '?' read as None."""
    frames, cur, loop_tags, loop_rows = {}, None, None, None
    toks = list(_tokens(text))

    def val(v, quoted):
        return None if not quoted and v in ('.', '?') else v

    def close_loop():
        nonlocal loop_tags, loop_rows
        if loop_tags and cur is not None:
            cat = loop_tags[0].split('.')[0].lstrip('_')
            names = [t.split('.', 1)[1] for t in loop_tags]
            rows = [dict(zip(names, loop_rows[k:k + len(names)]))
                    for k in range(0, len(loop_rows), len(names))]
            cur['loops'][cat] = rows
        loop_tags = loop_rows = None

    k = 0
    while k < len(toks):
        t, quoted = toks[k]
        if not quoted and t.startswith('save_'):
            close_loop()
            if t == 'save_':
                cur = None
            else:
                cur = frames.setdefault(t[5:], {'items': {}, 'loops': {}})
        elif not quoted and t == 'loop_':
            close_loop()
            loop_tags, loop_rows = [], []
            k += 1
            while k < len(toks) and not toks[k][1] and toks[k][0].startswith('_'):
                loop_tags.append(toks[k][0])
                k += 1
            continue
        elif not quoted and t == 'stop_':
            close_loop()
        elif loop_tags is not None:
            loop_rows.append(val(t, quoted))
        elif not quoted and t.startswith('_') and cur is not None:
            tag = t.split('.', 1)[1] if '.' in t else t.lstrip('_')
            v, q = toks[k + 1] if k + 1 < len(toks) else ('.', False)
            cur['items'][tag] = val(v, q)
            k += 2
            continue
        k += 1
    close_loop()
    return frames


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Spectrum:
    """One reference spectrum of the entry, as CASMDB remediated it."""
    origin: str                      # HMDB | BMRB | GISSMO
    kind: str                        # peak_list | spin_system
    field_mhz: float | None
    similarity: float | None         # cosine vs the experimental spectrum
    source_file: str | None          # CASMDB's NDF5 name (SP-<n>)
    peaks: list = field(default_factory=list)       # [(ppm, height)]
    multiplets: list = field(default_factory=list)  # [{'ppm','volume','peaks'}]
    spin_ppm: list = field(default_factory=list)    # GISSMO diagonal
    spin_j: list = field(default_factory=list)      # [(i, j, J Hz)]


@dataclass
class Entry:
    name: str
    inchi: str | None
    smiles: str | None
    formula: str | None
    hmdb_id: str | None
    bmrb_id: str | None
    cas: str | None
    iupac: str | None
    synonyms: list
    spectra: list


def _spectrum(frame) -> Spectrum:
    it, lp = frame['items'], frame['loops']
    dims = lp.get('nef_spectrum_dimension') or [{}]
    peaks_by_id = {}
    for p in lp.get('nef_peak') or ():
        ppm, h = _float(p.get('position_1')), _float(p.get('height'))
        if ppm is not None:
            peaks_by_id[p.get('peak_id')] = (ppm, h if h is not None else 0.0)
    members = {}
    for m in lp.get('ccpn_multiplet_peaks') or ():
        members.setdefault(m.get('multiplet_serial'), []).append(m.get('peak_serial'))
    multiplets = []
    for m in lp.get('ccpn_multiplet') or ():
        pos = m.get('position')
        try:
            ppm = float(ast.literal_eval(pos)[0]) if pos else None
        except (ValueError, SyntaxError, TypeError, IndexError):
            ppm = None
        ids = members.get(m.get('multiplet_serial'), [])
        multiplets.append({'ppm': ppm, 'volume': _float(m.get('volume')),
                           'height': _float(m.get('height')),
                           'peaks': [peaks_by_id[i] for i in ids if i in peaks_by_id]})
    spin_ppm, spin_j = [], []
    rows = lp.get('ccpn_spin_system_matrix') or []
    if rows:
        cols = list(rows[0])
        mat = [[_float(r[c]) for c in cols] for r in rows]
        spin_ppm = [mat[i][i] for i in range(len(mat))]
        spin_j = [(i, j, mat[i][j]) for i in range(len(mat))
                  for j in range(i + 1, len(mat)) if mat[i][j]]
    path = it.get('ccpn_file_path') or ''
    return Spectrum(origin=it.get('ccpn_origin') or '',
                    kind=it.get('ccpn_spectrum_type') or '',
                    field_mhz=_float(dims[0].get('spectrometer_frequency')),
                    similarity=_float(it.get('ccpn_similarity_score')),
                    source_file=re.split(r'[\\/]', path)[-1] or None,
                    peaks=sorted(peaks_by_id.values()),
                    multiplets=multiplets, spin_ppm=spin_ppm, spin_j=spin_j)


def read_entry(text: str) -> Entry:
    frames = parse_frames(text)
    sub = next((f for k, f in frames.items() if k.startswith('ccpn_substance')),
               {'items': {}, 'loops': {}})
    it = sub['items']
    spectra = [_spectrum(f) for k, f in frames.items()
               if k.startswith('nef_nmr_spectrum')]
    syn = [r.get('synonym') for r in sub['loops'].get('ccpn_substance_synonym') or ()
           if r.get('synonym')]
    return Entry(name=it.get('name') or '', inchi=it.get('inchi'),
                 smiles=it.get('smiles'), formula=it.get('empirical_formula'),
                 hmdb_id=it.get('hmdb_accession'), bmrb_id=it.get('bmrb_accession'),
                 cas=it.get('cas_number'), iupac=it.get('iupac_name'),
                 synonyms=syn, spectra=spectra)


def read_file(path: str) -> Entry:
    with open(path, encoding='utf-8', errors='replace') as f:
        return read_entry(f.read())
