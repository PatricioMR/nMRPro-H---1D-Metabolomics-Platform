# validation/gissmo_extract.py
"""Build ``validation/data/gissmo_spin_systems.json`` from GISSMO XMLs.

GISSMO (gissmo.bmrb.io) publishes, per BMRB entry, one or more spin-system
fits of the compound's reference spectrum (``simulation_N/
spin_simulation.xml``): shifts in ppm and couplings in Hz, which do not
depend on the field. This keeps the complete ('merged', index 1) coupling
matrix of one simulation per compound -- ``simulation_1`` unless it was
fitted at 600 MHz and another was not, so the 600 MHz spectra BMRB acquired
can still judge the simulator (Banco-a-600-MHz in the vault) -- and the
couplings to spins outside the matrix (31P, other sub-matrices).

The XMLs live outside the repository (``--xml-dir``, by default
``NMR Databases/gissmo``); fetch a missing one with

    https://gissmo.bmrb.io/entry/<bmse>/simulation_<N>/spin_simulation.xml

Dashti H, Westler WM, Tonelli M, Wedell JR, Markley JL, Eghbalnia HR. Spin
System Modeling of Nuclear Magnetic Resonance Spectra for Applications in
Metabolomics and Small Molecule Screening. Anal Chem 2017;89(22):12201-12208.

    python -m validation.gissmo_extract
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation import mixture_bench as mb
from validation.gissmo_sim import SYSTEMS_JSON

XML_DIR = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documentos',
                       'NMR Databases', 'gissmo')


def parse_gissmo(path) -> dict:
    """The complete ('merged', index 1) coupling matrix of a GISSMO XML."""
    t = open(path, encoding='utf-8', errors='ignore').read()
    field = float(re.search(r'<field_strength>([\d.]+)', t).group(1))
    name = re.search(r'<name>(.*?)</name>', t).group(1)
    mats = t.split('<coupling_matrix>')[1:]
    m = next(x for x in mats if re.search(r'<index>1</index>', x))
    m = m.split('<spectrum>')[0]
    cs = {int(i): float(p) for i, p in
          re.findall(r'<cs index="(\d+)" ppm="([-\d.]+)"', m)}
    n = len(cs)
    J = np.zeros((n, n))
    for a, b, v in re.findall(
            r'<coupling from_index="(\d+)" to_index="(\d+)" value="([-\d.eE]+)"', m):
        a, b = int(a) - 1, int(b) - 1
        J[a, b] = J[b, a] = float(v)
    ext = {}
    for si, c, _sg, cg in re.findall(
            r'<acc spin_index="(\d+)" coupling="([-\d.]+)" '
            r'spin_group_index="(\d+)" coupling_group_index="(\d+)"', m):
        ext.setdefault(int(cg), []).append((int(si) - 1, float(c)))
    lw = float(re.search(r'<lw>([\d.]+)', m).group(1))
    return {'name': name, 'fit_field': field, 'lw': lw,
            'ppm': np.array([cs[i + 1] for i in range(n)]), 'J': J,
            'external': list(ext.values())}


def choose(xml_dir, bmse):
    """The file of ``bmse`` to use (see the module), or None."""
    cands = sorted(f for f in os.listdir(xml_dir) if f.startswith(bmse + '_'))
    for f in cands:
        if round(parse_gissmo(os.path.join(xml_dir, f))['fit_field']) != 600:
            return f
    return cands[0] if cands else None


def extract(xml_dir=XML_DIR, db_path=mb.DB_DEFAULT, out=SYSTEMS_JSON):
    """Every compound of the reference whose entry has a GISSMO XML."""
    con = sqlite3.connect(db_path)
    old = {}
    if os.path.exists(out):
        with open(out, encoding='utf-8') as f:
            old = json.load(f)
    systems = []
    for name, bid, bmse in con.execute(
            'select name, bmrb_id, bmse_id from metabolites order by name'):
        entry = bmse or bid
        if not entry or not str(entry).startswith('bmse'):
            continue
        f = choose(xml_dir, entry)
        if not f:
            continue
        g = parse_gissmo(os.path.join(xml_dir, f))
        n = len(g['ppm'])
        systems.append({
            'name': name, 'bmse': entry, 'gissmo_file': f,
            'gissmo_name': g['name'], 'fit_field_mhz': g['fit_field'],
            'lw_hz': g['lw'], 'ppm': [float(v) for v in g['ppm']],
            'J_hz': [[i, j, float(g['J'][i, j])] for i in range(n)
                     for j in range(i + 1, n) if g['J'][i, j]],
            'external': [[[int(si), float(c)] for si, c in grp]
                         for grp in g['external']]})
    doc = {k: v for k, v in old.items() if k != 'systems'}
    doc['systems'] = systems
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    return systems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--xml-dir', default=XML_DIR)
    a = ap.parse_args()
    s = extract(a.xml_dir)
    print(f'{len(s)} spin systems -> {SYSTEMS_JSON}')


if __name__ == '__main__':
    main()
