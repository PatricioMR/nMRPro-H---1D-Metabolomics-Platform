# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Validate the pattern shape fit where the truth is known: on the reference
spectra themselves.

Level 1, self-consistency: every pattern in BMRB_Metabolomics_v02 fitted on
the spectrum it was measured from. The pattern is there by construction, so
the fit must find it at its position, at a signal-to-noise near the
signal's own, explaining nearly all of the trace over its lines.

Level 2, cross-set: the same pattern fitted on the compound's other spectra
-- another concentration, another field, another entry. Couplings do not
move with the field; positions move with pH. A pattern that fits its own
spectrum and not its sibling's is a pattern of that spectrum, not of the
compound.

Writes one row per (pattern, spectrum) to _pattern_fit_validation.csv and a
summary to stdout. Nothing is written to the database.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, ROOT)
from measure_reference_db import _spectrum_sets, _load, V02  # noqa: E402
from processing.annotation_scoring import (  # noqa: E402
    pattern_governs, pattern_offsets_hz)
from processing.pattern_shape import (  # noqa: E402
    fit_pattern_shape, fit_profile_shape)
from processing.reference_measurement import (  # noqa: E402
    linewidth_ppm, noise_sigma, reference_offset)

TOLERANCE_PPM = 0.05
OUT = os.path.join(ROOT, '_pattern_fit_validation.csv')


def _spectrum(proc_dir, cache):
    if proc_dir not in cache:
        ppm, y = _load(proc_dir)
        off = reference_offset(ppm, y)
        ppm = ppm - off
        cache[proc_dir] = (ppm, y, noise_sigma(ppm, y))
    return cache[proc_dir]


def _width_hz(ppm, y, field, fallback):
    """The spectrum's linewidth, in Hz, read on the reference line at 0 ppm:
    one line, resolved, in every spectrum. Read on the pattern itself the
    walk crosses half maximum on the envelope of unresolved lines and
    reports the envelope's width."""
    sel = (ppm >= -0.05) & (ppm <= 0.05)
    if not sel.any():
        return fallback
    apex = float(ppm[sel][np.argmax(y[sel])])
    w = linewidth_ppm(ppm, y, apex)
    return float(w) * field if w else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(
        REPO, 'databases', 'BMRB_Metabolomics_Mammalian_v02.db'))
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        'select m.name, m.bmrb_id, m.bmse_id, s.id, s.ppm, s.ppm_meas, '
        's.multiplicity, s.pattern_lines, s.meas_set, s.meas_field_mhz, '
        's.linewidth_hz, s.snr_meas from shifts_1h s '
        'join metabolites m on m.bmrb_id = s.bmrb_id '
        'where s.pattern_lines is not null order by m.name, s.ppm')]
    rows = [r for r in rows if pattern_governs(r['multiplicity'],
                                               r['pattern_lines'])]
    if args.limit:
        rows = rows[:args.limit]
    cache: dict = {}
    widths: dict = {}
    out = []
    for r in rows:
        lines = pattern_offsets_hz({'pattern_lines': r['pattern_lines']})
        if len(lines) < 2:
            continue
        db_ppm = float(r['ppm_meas'] or r['ppm'])
        bmse = r['bmse_id'] or r['bmrb_id']
        sets = _spectrum_sets(bmse)
        # the profile: the own spectrum's trace over the pattern's span,
        # plus three linewidths, on a 0.1 Hz grid from the reference position
        profile = None
        own = next((t for t in sets if t[0] == r['meas_set']), None)
        if own is not None:
            try:
                oppm, oy, osig = _spectrum(own[1], cache)
                ofield = own[2]
                if own[1] not in widths:
                    widths[own[1]] = _width_hz(oppm, oy, ofield,
                                               r['linewidth_hz'] or 1.5)
                ow = widths[own[1]]
                offs = [o for o, _ in lines]
                grid = np.arange(min(offs) - 3 * ow, max(offs) + 3 * ow, 0.1)
                asc = oppm[0] < oppm[-1]
                prof_h = np.interp(db_ppm + grid / ofield,
                                   oppm if asc else oppm[::-1],
                                   oy if asc else oy[::-1])
                prof_h = np.clip(prof_h - np.percentile(prof_h, 5), 0, None)
                profile = (grid, prof_h, ow)
            except Exception as e:
                print(f'  no profile {bmse}: {e}', file=sys.stderr)
        for set_name, proc, field in sets:
            if 'JRES' in set_name or not field:
                continue
            try:
                ppm, y, sigma = _spectrum(proc, cache)
            except Exception as e:      # a set that does not load
                print(f'  skip {bmse} {set_name}: {e}', file=sys.stderr)
                continue
            if proc not in widths:
                widths[proc] = _width_hz(ppm, y, field,
                                         r['linewidth_hz'] or 1.5)
            fwhm = widths[proc]
            fit = fit_pattern_shape(ppm, y, lines, db_ppm=db_ppm,
                                    field_mhz=field, fwhm_hz=fwhm,
                                    tolerance_ppm=TOLERANCE_PPM, sigma=sigma)
            if fit is None:
                continue
            pfit = None
            if profile is not None:
                pfit = fit_profile_shape(
                    ppm, y, (profile[0], profile[1]), db_ppm=db_ppm,
                    field_mhz=field, fwhm_hz=fwhm, ref_fwhm_hz=profile[2],
                    tolerance_ppm=TOLERANCE_PPM, sigma=sigma)
            out.append({
                'p_shift_ppm': round(pfit['shift_ppm'], 4) if pfit else None,
                'p_snr': round(pfit['snr'], 1) if pfit else None,
                'p_explained': round(pfit['explained'], 3) if pfit else None,
                'p_r2': round(pfit['r2'], 3) if pfit else None,
                'name': r['name'], 'ppm': db_ppm, 'mult': r['multiplicity'],
                'n_lines': len(lines), 'own_set': r['meas_set'],
                'set': set_name, 'level': 1 if set_name == r['meas_set'] else 2,
                'field_mhz': field, 'fwhm_hz': round(fwhm, 2),
                'shift_ppm': round(fit['shift_ppm'], 4),
                'snr': round(fit['snr'], 1),
                'explained': round(fit['explained'], 3),
                'r2': round(fit['r2'], 3),
                'signal_snr': round(float(r['snr_meas'] or 0), 1)})
        print(f"{r['name'][:30]:30} {db_ppm:7.3f} done", file=sys.stderr)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for level in (1, 2):
        sub = [o for o in out if o['level'] == level]
        if not sub:
            continue
        print(f'level {level}: {len(sub)} fits')
        for tag, ke, ks, kn in (('sticks ', 'explained', 'shift_ppm', 'snr'),
                                ('profile', 'p_explained', 'p_shift_ppm',
                                 'p_snr')):
            got = [o for o in sub if o[ke] is not None]
            if not got:
                continue
            ex = np.array([o[ke] for o in got])
            sh = np.array([abs(o[ks]) for o in got])
            sn = np.array([o[kn] for o in got])
            print(f'  [{tag}] explained median {np.median(ex):.3f} '
                  f'p10 {np.percentile(ex, 10):.3f} under 0.7: {(ex < 0.7).sum()}'
                  f'  |  |shift| median {np.median(sh):.4f} '
                  f'p90 {np.percentile(sh, 90):.4f} over 0.01: {(sh > 0.01).sum()}'
                  f'  |  SNR median {np.median(sn):.0f} under 10: {(sn < 10).sum()}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
