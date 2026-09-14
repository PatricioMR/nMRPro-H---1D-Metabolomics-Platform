"""Build BMRB_Metabolomics_v02: every signal measured from its own spectrum.

v01 carries what BMRB's curators listed -- ppm to three decimals, J to the
half hertz, and wrong where they were wrong (L-lactic acid's doublet at 7.50 Hz
against 6.93 in the spectrum). v02 keeps every column of v01 and adds, per
signal, what the spectrum says: where the lines are, the coupling with its
uncertainty, the width and the signal-to-noise. Nothing in v01 is altered;
the listing stays beside the measurement so the two can be compared.

    python hmdb_api_download/measure_reference_db.py [--entries bmse000208,...]

Reads the spectra mirrored by download_bmrb_fids.py under databases/bmrb_fids,
writes databases/BMRB_Metabolomics_v02.db and a CSV of every signal where the
listing and the measurement disagree beyond the measurement's own error.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)

from data.bruker_loader import _find_experiments, load_bruker_dataset  # noqa: E402
from processing.reference_measurement import (  # noqa: E402
    measure_fingerprint, noise_sigma, reference_offset)

FIDS = os.path.join(REPO, 'databases', 'bmrb_fids')
V01 = os.path.join(REPO, 'databases', 'BMRB_Metabolomics_v01.db')
V02 = os.path.join(REPO, 'databases', 'BMRB_Metabolomics_v02.db')
REPORT = os.path.join(ROOT, '_v02_disagreements.csv')

MEASURED_COLUMNS = (
    ('ppm_meas', 'REAL'), ('j_meas', 'REAL'), ('j_sigma', 'REAL'),
    ('linewidth_hz', 'REAL'), ('snr_meas', 'REAL'), ('pattern_meas', 'REAL'),
    ('meas_status', 'TEXT'), ('meas_set', 'TEXT'),
)

# Beyond this many of the measurement's own sigma, plus a floor for what a
# curator could reasonably have rounded, the listing is not a rounding of the
# measurement but a different number.
DISAGREE_SIGMAS = 3.0
J_FLOOR_HZ = 0.25
PPM_FLOOR = 0.003


def _j_listed(v):
    if v in (None, ''):
        return None
    try:
        return float(str(v).split()[0])
    except ValueError:
        return None


def measure_entry(ppm, y, signals, *, field_mhz) -> list[dict]:
    """One spectrum of one compound against its listed signals.

    The axis is referenced on the DSS line first, so a measured position is
    a position and not the spectrometer's offset. Returns one dict per listed
    signal with the measurement beside the listing.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    off = reference_offset(ppm, y)
    sigma = noise_sigma(ppm - off, y)
    out = measure_fingerprint(ppm - off, y, signals, field_mhz=field_mhz,
                              sigma=sigma)
    for m in out:
        m['reference_offset_ppm'] = float(off)
    return out


def disagreements(measured) -> list[dict]:
    """Signals whose listing the measurement contradicts."""
    rows = []
    for m in measured:
        if m.get('status') != 'ok':
            continue
        lst = m['listed']
        row = {'ppm_listed': float(lst.get('ppm', 0.0)), 'ppm_meas': m['ppm'],
               'j_listed': _j_listed(lst.get('j_coupling')),
               'j_meas': m.get('j_hz'), 'j_sigma': m.get('j_sigma_hz'),
               'multiplicity': m['multiplicity']}
        why = []
        if abs(row['ppm_meas'] - row['ppm_listed']) > PPM_FLOOR:
            why.append('ppm')
        if (row['j_listed'] is not None and row['j_meas'] is not None
                and abs(row['j_meas'] - row['j_listed'])
                > DISAGREE_SIGMAS * float(row['j_sigma'] or 0.0) + J_FLOOR_HZ):
            why.append('J')
        if why:
            row['disagrees_on'] = '+'.join(why)
            rows.append(row)
    return rows


def _ensure_v02():
    if not os.path.exists(V02):
        shutil.copyfile(V01, V02)
    con = sqlite3.connect(V02)
    have = {r[1] for r in con.execute('pragma table_info(shifts_1h)')}
    for name, typ in MEASURED_COLUMNS:
        if name not in have:
            con.execute(f'alter table shifts_1h add column {name} {typ}')
    con.commit()
    return con


def _spectrum_sets(bmse):
    """``(set_name, proc_dir)`` for every mirrored 1D 1H spectrum."""
    base = os.path.join(FIDS, bmse)
    if not os.path.isdir(base):
        return []
    out = []
    for root, dirs, files in os.walk(base):
        if '1r' in files and 'procs' in files:
            rel = os.path.relpath(root, base)
            out.append((rel.split(os.sep)[1] if os.sep in rel else rel, root))
    return sorted(out)


def _load(proc_dir):
    ppm, X, _names, _labels = load_bruker_dataset(
        proc_dir, ppm_min=-1.0, ppm_max=12.0, n_points=2 ** 17)
    return ppm, X[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entries', default='')
    args = ap.parse_args()
    con = _ensure_v02()
    mets = con.execute(
        'select bmrb_id, coalesce(bmse_id, bmrb_id), name, field_mhz '
        'from metabolites').fetchall()
    if args.entries:
        keep = {e.strip() for e in args.entries.split(',')}
        mets = [m for m in mets if m[1] in keep]
    report = []
    n_done = n_sig = 0
    for key, bmse, name, field in mets:
        sets = _spectrum_sets(bmse)
        if not sets:
            continue
        signals = [dict(zip(('id', 'ppm', 'multiplicity', 'j_coupling'), r))
                   for r in con.execute(
                       'select id, ppm, multiplicity, j_coupling from shifts_1h '
                       'where bmrb_id=? order by ppm', (key,))]
        best: dict = {}
        for set_name, proc_dir in sets:
            try:
                ppm, y = _load(proc_dir)
            except Exception as e:                      # noqa: BLE001
                print(f'{bmse} {set_name}: cannot load ({e})')
                continue
            for m in measure_entry(ppm, y, signals, field_mhz=float(field or 500.0)):
                sid = m['listed']['id']
                # Several sets are several concentrations; the measurement
                # from the strongest clean signal stands for the signal.
                if m.get('status') == 'ok' and (
                        sid not in best or best[sid].get('status') != 'ok'
                        or m['snr'] > best[sid]['snr']):
                    best[sid] = {**m, 'set': set_name}
                elif sid not in best:
                    best[sid] = {**m, 'set': set_name}
        for sid, m in best.items():
            con.execute(
                'update shifts_1h set ppm_meas=?, j_meas=?, j_sigma=?, '
                'linewidth_hz=?, snr_meas=?, pattern_meas=?, meas_status=?, '
                'meas_set=? where id=?',
                (m.get('ppm'), m.get('j_hz'), m.get('j_sigma_hz'),
                 m.get('linewidth_hz'), m.get('snr'),
                 m.get('pattern_agreement'), m.get('status'), m.get('set'),
                 sid))
            n_sig += 1
        for d in disagreements(list(best.values())):
            report.append({'bmse': bmse, 'name': name, **d})
        n_done += 1
        con.commit()
    with open(REPORT, 'w', newline='', encoding='utf-8') as f:
        cols = ['bmse', 'name', 'multiplicity', 'ppm_listed', 'ppm_meas',
                'j_listed', 'j_meas', 'j_sigma', 'disagrees_on']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in report:
            w.writerow({k: r.get(k) for k in cols})
    print(f'{n_done} entries, {n_sig} signals measured, '
          f'{len(report)} disagreements -> {REPORT}')


if __name__ == '__main__':
    main()
