# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Build BMRB_Metabolomics_v04: v03 with its measurements checked against
every BMRB spectrum of the compound.

v02/v03 measured each signal on the one spectrum that gave the strongest
clean reading, and measured it AS LISTED: the listed multiplicity chose the
model the lines were fitted with. Where the curators' multiplicity was wrong
the measurement inherited it. Taurine's 3.41 ppm signal is listed as a
doublet; three BMRB acquisitions of it (bmse000120 twice at 500 MHz,
bmse000863 at 600 MHz) show a triplet centred at 3.411 ppm, J 6.5-6.7 Hz. A
doublet fitted to two of its three lines put ``ppm_meas`` at 3.4059 -- half
way between them -- and the measured profile, registered on that position,
2 Hz off its own lines.

This build re-reads every signal of every compound on every mirrored BMRB
spectrum of it, WITHOUT the listing: the lines inside the signal's window,
their centre and relative heights (``reference_measurement.measure_pattern``).
A value is changed only where at least ``MIN_SETS`` independent spectra agree
with each other and disagree with the database:

* the same number of lines in at least ``MIN_SETS`` spectra that read the
  signal cleanly (SNR >= ``MIN_SNR``) and in ``CONSENSUS_SHARE`` of them,
  centres within ``CENTRE_TOL_HZ`` of their median;
* the spectrum v03 measured the signal on among them -- a disagreement
  with OTHER depositions is a difference of conditions (pH, buffer,
  concentration), not a misreading, and is left for review;
* the database's multiplicity implies a different number of lines, or its
  position sits more than ``CENTRE_TOL_HZ`` from the consensus centre.

Then: the position becomes the consensus centre; the multiplicity is
re-read from the consensus lines only when the pattern is unambiguously
first order (s, d, t, q: equal spacings, binomial-like heights), with the
coupling from the spacing; the measured profile is re-registered on the
new position. Anything else that disagrees is marked for review and left
as it was.

Nothing is overwritten without a trace. The curators' listing
(``multiplicity``, ``ppm``, ``j_coupling``) is never touched; the measured
columns keep their v03 values in ``*_v03`` columns; every row records its
verdict (``v04_status``: confirmed / corrected / review / unverified) and
the per-spectrum evidence as JSON (``v04_evidence``). A CSV of every change
and every review item is written beside the database.

Source data: Biological Magnetic Resonance Data Bank (BMRB) metabolomics
entries and their deposited time-domain data -- Ulrich, E.L. et al. (2008).
BioMagResBank. Nucleic Acids Res. 36:D402-D408; Hoch, J.C. et al. (2023).
Biological Magnetic Resonance Data Bank. Nucleic Acids Res. 51:D368-D376.

    python hmdb_api_download/build_reference_v04.py            # measure + build
    python hmdb_api_download/build_reference_v04.py --reuse    # build from the cached readings
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, ROOT)
from measure_reference_db import (  # noqa: E402
    _load, _spectrum_sets, measure_entry)

DBDIR = os.path.join(REPO, 'databases')
V03 = os.path.join(DBDIR, 'BMRB_Metabolomics_v03.db')
V04 = os.path.join(DBDIR, 'BMRB_Metabolomics_v04.db')
READINGS = os.path.join(ROOT, '_v04_readings.jsonl')
REPORT = os.path.join(ROOT, '_v04_changes.csv')

# At least two spectra, each an independent acquisition deposited in BMRB,
# must read the signal the same way before a value is changed.
MIN_SETS = 2
# A line pattern read below this SNR is not counted, for or against.
MIN_SNR = 20.0
# Centres agreeing within one hertz: the width a rounding to 0.002 ppm at
# 500 MHz already allows, and several times a clean line's position error.
CENTRE_TOL_HZ = 1.0
# Readings further than this from the readings' median centre are of
# something else.
NEAR_HZ = 10.0
# A correction that would move a position further than this is a question
# for a person, not a rule.
MAX_MOVE_HZ = 10.0
# The share of the clean readings the consensus must hold.
CONSENSUS_SHARE = 2.0 / 3.0
# First-order spacing agreement, within one pattern and between spectra.
SPACING_TOL_HZ = 0.6

LABEL_LINES = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'quint': 5, 'sext': 6,
               'sept': 7, 'dd': 4, 'dt': 6, 'td': 6, 'ddd': 8}


def label_lines(mult) -> int | None:
    """Lines a first-order multiplicity implies; None where it implies
    nothing checkable (m, br, unresolved)."""
    return LABEL_LINES.get(str(mult or '').strip().lower())


def read_signals(con, bmrb_id):
    return [dict(zip(('id', 'ppm', 'multiplicity', 'j_coupling', 'ppm_min',
                      'ppm_max', 'n_lines', 'protons'), r))
            for r in con.execute(
                'select id, ppm, multiplicity, j_coupling, ppm_min, ppm_max, '
                'n_lines, protons from shifts_1h where bmrb_id=? order by ppm',
                (bmrb_id,))]


def measure_all(entries=None) -> None:
    """Every signal of every compound on every mirrored spectrum of it, as
    JSON lines: the listing-free pattern of each (set, signal)."""
    con = sqlite3.connect(V03)
    mets = con.execute('select bmrb_id, coalesce(bmse_id, bmrb_id), name, '
                       'field_mhz from metabolites').fetchall()
    if entries:
        mets = [m for m in mets if m[1] in entries]
    with open(READINGS, 'w', encoding='utf-8') as out:
        for k, (key, bmse, name, field) in enumerate(mets):
            signals = read_signals(con, key)
            per: dict = {s['id']: [] for s in signals}
            for set_name, proc_dir, set_field in _spectrum_sets(bmse):
                f_mhz = float(set_field or field or 500.0)
                try:
                    ppm, y = _load(proc_dir)
                except Exception as e:                      # noqa: BLE001
                    print(f'{bmse} {set_name}: cannot load ({e})')
                    continue
                for m in measure_entry(ppm, y, signals, field_mhz=f_mhz):
                    pat = m.get('pattern') or {}
                    per[m['listed']['id']].append({
                        'set': f'{bmse}/{set_name}', 'field': f_mhz,
                        'status': pat.get('status'),
                        'centre': pat.get('centre_ppm'),
                        'lines': pat.get('lines') or [],
                        'n': pat.get('n_lines'), 'snr': pat.get('snr')})
            for sid, reads in per.items():
                out.write(json.dumps({'id': sid, 'bmrb_id': key, 'name': name,
                                      'reads': reads}) + '\n')
            if k % 50 == 0:
                print(f'{k}/{len(mets)} {name}', flush=True)


def clean_reads(reads):
    """The readings that count: a line pattern read at MIN_SNR or better on
    a 1D spectrum. A J-resolved set is left out -- its projection collapses
    every multiplet to one line."""
    return [r for r in reads
            if r['status'] in ('ok', 'single') and r['centre'] is not None
            and (r['snr'] or 0) >= MIN_SNR and 'JRES' not in r['set']]


def consensus(reads):
    """``(n_lines, centre_ppm, [read, ...])`` when the most common number of
    lines is read by at least MIN_SETS clean spectra and by at least
    CONSENSUS_SHARE of them, with their centres within CENTRE_TOL_HZ of
    their median; else None. A spectrum that reads otherwise stays in the
    evidence and cannot, alone, veto what the others agree on."""
    clean = clean_reads(reads)
    if len(clean) < MIN_SETS:
        return None
    # A reading far from where most readings put the signal did not read
    # this signal (a window on another line, a spectrum referenced
    # elsewhere): it neither votes nor vetoes.
    mid = float(np.median([r['centre'] for r in clean]))
    clean = [r for r in clean
             if abs(r['centre'] - mid) * r['field'] <= NEAR_HZ]
    if len(clean) < MIN_SETS:
        return None
    counts = {}
    for r in clean:
        counts[r['n']] = counts.get(r['n'], 0) + 1
    n, k = max(counts.items(), key=lambda t: (t[1], -t[0]))
    if k < MIN_SETS or k < CONSENSUS_SHARE * len(clean):
        return None
    agree = [r for r in clean if r['n'] == n]
    centre = float(np.median([r['centre'] for r in agree]))
    if any(abs(r['centre'] - centre) * r['field'] > CENTRE_TOL_HZ
           for r in agree):
        return None
    return n, centre, agree


def first_order(clean):
    """``(multiplicity, J_hz)`` read off the consensus lines when the
    pattern is unambiguously first order, else None."""
    n = clean[0]['n']
    if n == 1:
        return 's', None
    spacings, rels = [], []
    for r in clean:
        offs = sorted(r['lines'])
        pos = [o for o, _h in offs]
        spacings.append(np.diff(pos))
        rels.append([h for _o, h in offs])
    sp = np.array(spacings)
    if np.ptp(sp, axis=0).max() > SPACING_TOL_HZ:
        return None                      # the spectra disagree on a spacing
    sp = sp.mean(axis=0)
    rel = np.array(rels).mean(axis=0)
    if n == 2:
        return ('d', float(sp[0])) if rel.min() / rel.max() >= 0.5 else None
    if np.ptp(sp) > SPACING_TOL_HZ:
        return None                      # unequal spacings: not t or q
    inner = rel[1:-1].mean()
    outer = np.array([rel[0], rel[-1]])
    if n == 3 and outer.max() / inner <= 0.85 and outer.min() / inner >= 0.25:
        return 't', float(sp.mean())
    if n == 4 and outer.max() / inner <= 0.6:
        return 'q', float(sp.mean())
    return None


def verdict(row, reads, meas_field):
    """What v04 does with one signal: ``(status, changes, evidence)``."""
    evidence = [{k: r[k] for k in ('set', 'field', 'status', 'centre', 'n',
                                   'snr')} for r in reads]
    c = consensus(reads)
    if c is None:
        clean = clean_reads(reads)
        disagree = bool(clean) and (
            label_lines(row['multiplicity']) not in (None, clean[0]['n'])
            or (row['ppm_meas'] is not None and clean[0]['centre'] is not None
                and abs(clean[0]['centre'] - row['ppm_meas'])
                * clean[0]['field'] > CENTRE_TOL_HZ))
        return ('review' if disagree else 'unverified'), {}, evidence
    n, centre, clean = c
    # A measurement is corrected only where it misread its OWN spectrum:
    # the spectrum v03 measured the signal on must be among those that
    # agree. Where it is not, v03 and the others read different spectra --
    # BMRB deposits the same compound at several pH, concentrations and
    # buffers, and a nucleotide's lines move tens of mppm with them (ATP,
    # NAD, AMP moved 20-65 mppm between depositions). That is a condition,
    # not an error, and the majority of depositions is no reason to move it.
    own = row.get('meas_set')
    if not own or not any(r['set'].split('/', 1)[1] == own for r in clean):
        return ('review' if _disagrees(row, n, centre, meas_field)
                else 'confirmed'), {}, evidence
    if (row['ppm_meas'] is not None and abs(centre - row['ppm_meas'])
            * float(meas_field or 500.0) > MAX_MOVE_HZ):
        return 'review', {}, evidence
    lab = label_lines(row['multiplicity'])
    if lab is None and n >= 2:
        # A generic label ('m', unresolved): not contradicted, but made
        # specific where every agreeing spectrum shows a first-order pattern.
        # 'm' excludes a signal from annotation and quantification.
        fo = first_order(clean)
        if fo is not None:
            ch = {'ppm_meas': centre, 'mult_meas': fo[0]}
            if fo[1] is not None:
                ch['j_use'] = fo[1]
            return 'corrected', ch, evidence
    off_hz = (abs(centre - row['ppm_meas']) * float(meas_field or 500.0)
              if row['ppm_meas'] is not None else 0.0)
    label_wrong = lab is not None and lab != n
    if not label_wrong and off_hz <= CENTRE_TOL_HZ:
        return 'confirmed', {}, evidence
    fo = first_order(clean)
    if label_wrong and fo is None:
        return 'review', {}, evidence    # the label is wrong, the fix is not clear
    changes = {'ppm_meas': centre}
    if label_wrong:
        changes['mult_meas'] = fo[0]
        if fo[1] is not None:
            changes['j_use'] = fo[1]
    return 'corrected', changes, evidence


def _disagrees(row, n, centre, meas_field) -> bool:
    lab = label_lines(row['multiplicity'])
    return ((lab is not None and lab != n)
            or (row['ppm_meas'] is not None
                and abs(centre - row['ppm_meas']) * float(meas_field or 500.0)
                > CENTRE_TOL_HZ))


def reregister_profile(profile_json, old_ppm, new_ppm, field_mhz):
    """The profile's offsets are from the position it was cut at; moved to
    the corrected one."""
    if not profile_json or old_ppm is None or not field_mhz:
        return profile_json
    d = json.loads(profile_json)
    d['start'] = round(float(d['start']) + (float(old_ppm) - float(new_ppm))
                       * float(field_mhz), 3)
    return json.dumps(d)


NEW_COLUMNS = (('mult_meas', 'TEXT'), ('ppm_meas_v03', 'REAL'),
               ('j_use_v03', 'REAL'), ('j_use_source_v03', 'TEXT'),
               ('pattern_profile_v03', 'TEXT'), ('v04_status', 'TEXT'),
               ('v04_evidence', 'TEXT'))


def build() -> None:
    shutil.copyfile(V03, V04)
    con = sqlite3.connect(V04)
    have = {r[1] for r in con.execute('pragma table_info(shifts_1h)')}
    for col, typ in NEW_COLUMNS:
        if col not in have:
            con.execute(f'alter table shifts_1h add column {col} {typ}')
    reads = {}
    with open(READINGS, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            reads[r['id']] = r
    cols = ('id', 'bmrb_id', 'multiplicity', 'ppm', 'ppm_meas', 'j_use',
            'j_use_source', 'pattern_profile', 'meas_field_mhz', 'meas_set')
    rows = [dict(zip(cols, r)) for r in con.execute(
        f'select {", ".join(cols)} from shifts_1h')]
    names = dict(con.execute('select bmrb_id, name from metabolites'))
    report, tally = [], {}
    for row in rows:
        rd = reads.get(row['id'])
        if rd is None:
            continue
        status, ch, ev = verdict(row, rd['reads'], row['meas_field_mhz'])
        tally[status] = tally.get(status, 0) + 1
        sets = {}
        if ch:
            sets.update(ppm_meas_v03=row['ppm_meas'], j_use_v03=row['j_use'],
                        j_use_source_v03=row['j_use_source'],
                        pattern_profile_v03=row['pattern_profile'],
                        ppm_meas=ch['ppm_meas'],
                        pattern_profile=reregister_profile(
                            row['pattern_profile'], row['ppm_meas'],
                            ch['ppm_meas'], row['meas_field_mhz']))
            if 'mult_meas' in ch:
                sets['mult_meas'] = ch['mult_meas']
            if 'j_use' in ch:
                sets.update(j_use=ch['j_use'],
                            j_use_source='v04_pattern_consensus')
        sets.update(v04_status=status, v04_evidence=json.dumps(ev))
        con.execute(f'update shifts_1h set '
                    f'{", ".join(k + "=?" for k in sets)} where id=?',
                    (*sets.values(), row['id']))
        if status in ('corrected', 'review'):
            report.append({
                'bmrb_id': row['bmrb_id'], 'name': names.get(row['bmrb_id']),
                'status': status, 'listed_mult': row['multiplicity'],
                'mult_v04': ch.get('mult_meas', ''),
                'ppm_listed': row['ppm'], 'ppm_meas_v03': row['ppm_meas'],
                'ppm_meas_v04': ch.get('ppm_meas', ''),
                'j_use_v03': row['j_use'], 'j_use_v04': ch.get('j_use', ''),
                'evidence': json.dumps(ev)})
    con.commit()
    con.execute('VACUUM')
    con.close()
    with open(REPORT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]) if report else ['x'])
        w.writeheader()
        w.writerows(report)
    print('v04:', tally, '->', V04)
    print('changes and review items ->', REPORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', action='store_true',
                    help='build from the cached readings')
    ap.add_argument('--entries', default='',
                    help='comma-separated bmse ids (measurement only)')
    a = ap.parse_args()
    if not a.reuse:
        measure_all({e.strip() for e in a.entries.split(',') if e.strip()})
    build()


if __name__ == '__main__':
    main()
