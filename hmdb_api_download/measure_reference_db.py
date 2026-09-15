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
import json
import os
import re
import shutil
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)

sys.path.insert(0, ROOT)
from download_bmrb_fids import BMRB_ENTRIES, sample_conditions  # noqa: E402
from data.bruker_loader import load_bruker_dataset  # noqa: E402
from processing.reference_measurement import (  # noqa: E402
    measure_fingerprint, measure_pattern_extended, noise_sigma,
    reference_offset)

FIDS = os.path.join(REPO, 'databases', 'bmrb_fids')
V01 = os.path.join(REPO, 'databases', 'BMRB_Metabolomics_v01.db')
V02 = os.path.join(REPO, 'databases', 'BMRB_Metabolomics_v02.db')
REPORT = os.path.join(ROOT, '_v02_disagreements.csv')

MEASURED_COLUMNS = (
    ('ppm_meas', 'REAL'), ('j_meas', 'REAL'), ('j_sigma', 'REAL'),
    ('linewidth_hz', 'REAL'), ('snr_meas', 'REAL'), ('pattern_meas', 'REAL'),
    ('meas_status', 'TEXT'), ('meas_set', 'TEXT'),
    # The conditions the measurement was made under. A coupling is quoted
    # in hertz because it does not move with the field; a second-order
    # splitting does, and a rotamer-dependent one moves with the pH. Without
    # these beside the number, nobody can say which kind it is.
    ('meas_field_mhz', 'REAL'), ('meas_ph', 'REAL'), ('meas_temp_k', 'REAL'),
    ('meas_buffer', 'TEXT'),
    # The same signal at two fields. A true scalar coupling gives the same
    # hertz at both; an apparent splitting of a strongly coupled system does
    # not. This is the difference, and the flag it earns.
    ('j_field_dependence_hz', 'REAL'), ('j_flag', 'TEXT'),
    # What the annotation should use, and why. 'measured' where the
    # measurement stands, 'listed' where it does not, and 'review' where the
    # two disagree beyond what a rounding could explain.
    ('j_use', 'REAL'), ('j_use_source', 'TEXT'),
    # The measured line pattern -- offsets in hertz from the centre, relative
    # heights -- as JSON. For an unresolved multiplet this is the whole of what
    # can be verified about it; for a resolved one it is the same information
    # the multiplicity and J already give, kept for the same reason.
    ('pattern_lines', 'TEXT'), ('pattern_n', 'INTEGER'),
)

# A coupling that moves between fields by more than this, beyond the two
# measurements' own errors, is not the first-order constant it is listed as.
FIELD_DEPENDENCE_FLOOR_HZ = 0.30

# When the measurement and the listing are further apart than this, one of
# them is not a coupling: a listing above MAX_J is a line span of a strongly
# coupled system written in the J column (apocholic acid 'd 64.05'), and a
# measurement far from a plausible listing most likely picked the wrong lines
# in a crowded region (N-acetyl-D-glucosamine 'd 3.20' measured 11.59 with a
# sigma of 0.012 -- a small sigma says the lines were fitted well, not that
# they were the right lines). Neither is handed to the annotation as fact:
# the measurement stands where it corrects a listing by up to this much, or
# where the listing was never a coupling; otherwise the row is marked for
# review and the listing stays in force.
MAX_CORRECTION_HZ = 1.5
MAX_J_HZ = 20.0

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
    centres = sorted(float(s_.get('ppm', 0.0)) for s_ in signals)
    for m in out:
        m['reference_offset_ppm'] = float(off)
        lst = m['listed']
        # The pattern is measured around where the signal was found, not
        # where it was listed. Taurine's 3.41 ppm triplet is listed with a
        # window of 3.336-3.349: the resolved measurement found the triplet
        # at 3.411 and the pattern, cut in the listed window, found sixteen
        # lines of noise at a third of the noise's own apex.
        c = float(m.get('ppm') if m.get('status') == 'ok' and m.get('ppm')
                  else lst.get('ppm', 0.0))
        lo, hi = lst.get('ppm_min'), lst.get('ppm_max')
        if (lo is None or hi is None or float(hi) - float(lo) < 0.002
                or not float(lo) <= c <= float(hi)):
            # No window listed, or one that does not hold the signal: the
            # multiplet's own span at the largest plausible coupling, as
            # the resolved measurement uses.
            n = int(lst.get('n_lines') or 2)
            half = (n - 1) / 2.0 * 20.0 / float(field_mhz) + 0.01
            lo, hi = c - half, c + half
        # Never into a neighbouring signal of the same compound: the window
        # may grow to the midpoint and no further.
        # The signal's own listing is not a neighbour: measured half a
        # millippm off its listed position, the midpoint to itself pinned
        # the window at the listed value and every triplet lost a line.
        own = float(lst.get('ppm', 0.0))
        others = [v for v in centres if abs(v - own) > 1e-6]
        below = [v for v in others if v < c - 1e-6]
        above = [v for v in others if v > c + 1e-6]
        bounds = ((c + below[-1]) / 2.0 if below else float(lo) - 0.05,
                  (c + above[0]) / 2.0 if above else float(hi) + 0.05)
        pat = measure_pattern_extended(ppm - off, y, lo=float(lo), hi=float(hi),
                                       field_mhz=field_mhz, sigma=sigma,
                                       bounds=bounds)
        m['pattern'] = pat
        # An unresolved multiplet has no first-order measurement; the pattern
        # is its measurement, and its centre its position.
        if m.get('status') == 'unresolved' and pat['status'] == 'ok':
            m['status'] = 'pattern'
            m['ppm'] = pat['centre_ppm']
            m['snr'] = pat['snr']
    return out


def disagreements(measured) -> list[dict]:
    """Signals whose listing the measurement contradicts."""
    rows = []
    for m in measured:
        if m.get('status') != 'ok':
            continue
        lst = m['listed']
        jl = _j_listed(lst.get('j_coupling'))
        row = {'ppm_listed': float(lst.get('ppm', 0.0)), 'ppm_meas': m['ppm'],
               'j_listed': jl,
               'j_meas': m.get('j_hz'), 'j_sigma': m.get('j_sigma_hz'),
               'pattern': m.get('pattern_agreement'),
               'linewidth_hz': m.get('linewidth_hz'), 'snr': m.get('snr'),
               'j_use_source': coupling_to_use(jl, m.get('j_hz'),
                                               m.get('j_flag'))[1],
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


def field_from_acqus(exp_dir) -> float | None:
    """The spectrometer frequency, from the acquisition parameters written
    with the data -- not from the entry's prose, which names a spectrometer
    and not the field the spectrum was actually taken at."""
    path = os.path.join(exp_dir, 'acqus')
    if not os.path.exists(path):
        return None
    with open(path, encoding='latin-1') as f:
        m = re.search(r'##\$BF1=\s*([\d.]+)', f.read())
    return float(m.group(1)) if m else None


def _spectrum_sets(bmse):
    """``(set_name, proc_dir, field_mhz)`` for every mirrored 1D 1H
    spectrum: the extracted experiment directories under FIDS/<bmse>/."""
    base = os.path.join(FIDS, bmse)
    if not os.path.isdir(base):
        return []
    out = []
    for root, _dirs, files in os.walk(base):
        if '1r' in files and 'procs' in files:
            exp_dir = os.path.dirname(os.path.dirname(root))   # <exp>/pdata/1
            set_name = os.path.relpath(exp_dir, base).replace(os.sep, '/')
            out.append((set_name, root, field_from_acqus(exp_dir)))
    return sorted(out)


def _conditions(bmse) -> dict:
    star = os.path.join(BMRB_ENTRIES, bmse, bmse + '.str')
    if not os.path.exists(star):
        return {}
    return sample_conditions(
        open(star, encoding='utf-8', errors='replace').read())


def field_dependence(per_field) -> tuple[float | None, str | None]:
    """``per_field`` is ``{field_mhz: (j_hz, j_sigma_hz)}`` for one signal.

    Returns the largest difference in hertz between fields and, when it
    exceeds the measurements' own errors by the floor, the flag. Two fields
    are the minimum; with one there is nothing to compare and no verdict.
    """
    vals = [(f, j, s) for f, (j, s) in per_field.items() if j is not None]
    if len(vals) < 2:
        return None, None
    worst, worst_sig = 0.0, 0.0
    for i in range(len(vals)):
        for k in range(i + 1, len(vals)):
            d = abs(vals[i][1] - vals[k][1])
            if d > worst:
                worst = d
                worst_sig = float(np.hypot(vals[i][2] or 0.0,
                                           vals[k][2] or 0.0))
    flag = ('second_order_suspect'
            if worst > DISAGREE_SIGMAS * worst_sig + FIELD_DEPENDENCE_FLOOR_HZ
            else None)
    return worst, flag


def coupling_to_use(j_listed, j_meas, j_flag=None):
    """``(value, source)`` -- the coupling the annotation should read."""
    if j_meas is None:
        return (j_listed, 'listed') if j_listed is not None else (None, None)
    if j_flag == 'second_order_suspect':
        return j_meas, 'measured_second_order'
    if j_listed is None or float(j_listed) > MAX_J_HZ:
        return j_meas, 'measured'
    if abs(float(j_meas) - float(j_listed)) <= MAX_CORRECTION_HZ:
        return j_meas, 'measured'
    return j_listed, 'review'


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
        signals = [dict(zip(('id', 'ppm', 'multiplicity', 'j_coupling',
                             'ppm_min', 'ppm_max', 'n_lines'), r))
                   for r in con.execute(
                       'select id, ppm, multiplicity, j_coupling, ppm_min, '
                       'ppm_max, n_lines from shifts_1h '
                       'where bmrb_id=? order by ppm', (key,))]
        cond = _conditions(bmse)
        best: dict = {}
        per_field: dict = {}
        for set_name, proc_dir, set_field in sets:
            f_mhz = float(set_field or field or 500.0)
            try:
                ppm, y = _load(proc_dir)
            except Exception as e:                      # noqa: BLE001
                print(f'{bmse} {set_name}: cannot load ({e})')
                continue
            for m in measure_entry(ppm, y, signals, field_mhz=f_mhz):
                sid = m['listed']['id']
                if m.get('status') == 'ok' and m.get('j_hz') is not None:
                    # Best per field, for the field-dependence test.
                    prev = per_field.setdefault(sid, {}).get(f_mhz)
                    if prev is None or m['snr'] > prev[2]:
                        per_field[sid][f_mhz] = (m['j_hz'], m['j_sigma_hz'],
                                                 m['snr'])
                # Several sets are several concentrations; the measurement
                # from the strongest clean signal stands for the signal.
                if m.get('status') in ('ok', 'pattern') and (
                        sid not in best
                        or best[sid].get('status') not in ('ok', 'pattern')
                        or m.get('snr', 0) > best[sid].get('snr', 0)):
                    best[sid] = {**m, 'set': set_name, 'field': f_mhz}
                elif sid not in best:
                    best[sid] = {**m, 'set': set_name, 'field': f_mhz}
        for sid, m in best.items():
            dep, flag = field_dependence(
                {f: (j, s) for f, (j, s, _snr) in per_field.get(sid, {}).items()})
            j_sigma = m.get('j_sigma_hz')
            if flag and j_sigma is not None and dep is not None:
                # What the number is known to, once it is known to move.
                j_sigma = max(float(j_sigma), float(dep) / 2.0)
            m['j_flag'] = flag
            j_use, j_src = coupling_to_use(
                _j_listed(m['listed'].get('j_coupling')), m.get('j_hz'), flag)
            pat = m.get('pattern') or {}
            pat_json = (json.dumps([[round(o, 3), round(r, 3)]
                                    for o, r in pat.get('lines', [])])
                        if pat.get('status') == 'ok' else None)
            con.execute(
                'update shifts_1h set ppm_meas=?, j_meas=?, j_sigma=?, '
                'linewidth_hz=?, snr_meas=?, pattern_meas=?, meas_status=?, '
                'meas_set=?, meas_field_mhz=?, meas_ph=?, meas_temp_k=?, '
                'meas_buffer=?, j_field_dependence_hz=?, j_flag=?, '
                'j_use=?, j_use_source=?, pattern_lines=?, pattern_n=? '
                'where id=?',
                (m.get('ppm'), m.get('j_hz'), j_sigma,
                 m.get('linewidth_hz'), m.get('snr'),
                 m.get('pattern_agreement'), m.get('status'), m.get('set'),
                 m.get('field'), cond.get('ph'), cond.get('temperature'),
                 cond.get('buffer'), dep, flag, j_use, j_src,
                 pat_json, pat.get('n_lines') if pat_json else None, sid))
            n_sig += 1
        for d in disagreements(list(best.values())):
            report.append({'bmse': bmse, 'name': name, **d})
        n_done += 1
        con.commit()
    with open(REPORT, 'w', newline='', encoding='utf-8') as f:
        cols = ['bmse', 'name', 'multiplicity', 'ppm_listed', 'ppm_meas',
                'j_listed', 'j_meas', 'j_sigma', 'pattern', 'linewidth_hz',
                'snr', 'j_use_source', 'disagrees_on']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in report:
            w.writerow({k: r.get(k) for k in cols})
    print(f'{n_done} entries, {n_sig} signals measured, '
          f'{len(report)} disagreements -> {REPORT}')


if __name__ == '__main__':
    main()
