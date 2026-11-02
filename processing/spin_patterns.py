# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Complex multiplets judged by their spin system at the study's field.

A complex multiplet (``m*``) is judged by its shape, and the reference
database stores that shape as measured on a BMRB spectrum -- at 500 MHz
for most, 400 or 700 for some -- in hertz from the signal's centre
(``pattern_lines``, ``pattern_profile``). Hertz offsets are field-free only
for first-order patterns; a strongly coupled system changes shape with the
field. Against the compounds' own spin systems simulated alone at 600 MHz
the stored m* shapes correlate a median 0.95 but a quarter below 0.76 and a
tenth below 0.48 (letters 0.99, singlets 1.00; glutamine 0.26, glycerol
0.36, citrate 0.38); against REAL 600 MHz BMRB spectra the spin systems fit
with R2 0.96 against 0.89 for the stored profiles, the gain all in the
second-order ones -- glutamate 0.96 vs 0.72, serine 0.92 vs 0.37, glycerol
0.97 vs 0.15 (vault: Banco-a-600-MHz).

With ``SPIN_PATTERNS`` on, every listed signal of multiplicity m / m* of a
compound with a spin system gets its lines and profile from the system
solved at the study's field (Dashti H, Westler WM, Tonelli M, et al. Spin
system modeling of nuclear magnetic resonance spectra for applications in
metabolomics and small molecule screening. Anal Chem 2017;89:12201-12208;
GISSMO). Lines are assigned to the signal by the nearest spin; the
position stays the database's.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

SPIN_PATTERNS = True

# lines closer than this are one (Hz); lines under this share of the
# tallest are dropped; the profile is drawn at this width (Hz)
MERGE_HZ = 0.2
MIN_REL = 0.01
PROFILE_FWHM_HZ = 0.5
# a database signal takes the spins listed within this of it (ppm)
SPIN_REACH_PPM = 0.03

_PATTERN_MULTS = ('m', 'm*', 'multiplet')


@lru_cache(maxsize=1)
def _systems():
    from processing.spin_system import SYSTEMS_JSON, load_systems
    return load_systems(SYSTEMS_JSON) if os.path.exists(SYSTEMS_JSON) else {}


@lru_cache(maxsize=512)
def _sticks(name, field_mhz):
    from processing.spin_system import stick_spectrum
    sys_ = _systems().get(name)
    if sys_ is None:
        return None
    sp, si = stick_spectrum(sys_, float(field_mhz))
    return (np.asarray(sys_['ppm'], dtype=float), np.asarray(sp, dtype=float),
            np.asarray(si, dtype=float))


def signal_pattern(name, ppm_listed, field_mhz, reach_ppm=None):
    """``(pattern_lines, pattern_profile, profile_fwhm_hz)`` for the signal
    of ``name`` listed at ``ppm_listed``, from its spin system at
    ``field_mhz``; None when there is no system or no spin near it.
    ``reach_ppm`` (default ``SPIN_REACH_PPM``) is how far from the listed
    centre the signal's spins may stand: citrate is listed once, 4 H at
    2.579, for an AB pair at 2.52 and 2.65."""
    st = _sticks(name, float(field_mhz))
    if st is None:
        return None
    lines = signal_lines(*st, ppm_listed, field_mhz,
                         reach_ppm or SPIN_REACH_PPM)
    if lines is None:
        return None
    return json.dumps(lines), profile_of(lines), PROFILE_FWHM_HZ


def signal_lines(spins, sp, si, ppm_listed, field_mhz, reach_ppm):
    """``[(offset Hz, rel)]`` of the stick lines ``(sp, si)`` that belong to
    the spins within ``reach_ppm`` of ``ppm_listed`` -- each line goes to its
    nearest spin -- about their intensity-weighted centre; None under two
    lines."""
    spins, sp, si = (np.asarray(a, dtype=float) for a in (spins, sp, si))
    mine = np.abs(spins - float(ppm_listed)) <= reach_ppm
    if not mine.any():
        return None
    nearest = np.argmin(np.abs(sp[:, None] - spins[None, :]), axis=1)
    keep = mine[nearest] & (si > 0)
    if keep.sum() < 2:
        return None
    lp, li = sp[keep], si[keep]
    centre = float(np.sum(lp * li) / np.sum(li))
    return merge_lines((lp - centre) * float(field_mhz), li)


def merge_close(hz, rel):
    """``[(hz, intensity)]`` in ascending hz with lines closer than
    ``MERGE_HZ`` summed into one at their intensity-weighted position.
    A spin system lists every transition: the isopropyl of
    2-isopropylmalic acid solves to 296 sticks for its two methyl doublets,
    some 0.03 Hz apart, which a spectrum shows as four lines."""
    hz, rel = np.asarray(hz, dtype=float), np.asarray(rel, dtype=float)
    order = np.argsort(hz)
    merged = []
    for h, i in zip(hz[order], rel[order]):
        if merged and h - merged[-1][0] <= MERGE_HZ:
            w = merged[-1][1] + i
            merged[-1] = ((merged[-1][0] * merged[-1][1] + h * i) / w, w)
        else:
            merged.append((float(h), float(i)))
    return merged


def merge_lines(hz, rel):
    """Lines closer than ``MERGE_HZ`` as one, normalised to the tallest,
    those under ``MIN_REL`` dropped; None under two lines."""
    merged = merge_close(hz, rel)
    top = max(i for _h, i in merged)
    lines = [(round(h, 3), round(i / top, 4)) for h, i in merged if i / top >= MIN_REL]
    return lines if len(lines) >= 2 else None


def profile_of(lines) -> str:
    """The encoded profile of ``[(offset Hz, rel)]`` drawn at
    ``PROFILE_FWHM_HZ``."""
    from processing.pattern_shape import encode_profile
    lo, hi = lines[0][0] - 10.0, lines[-1][0] + 10.0
    grid = np.arange(lo, hi + 1e-9, 0.1)
    w = PROFILE_FWHM_HZ / 2.0
    prof = sum(r * (w / np.pi) / ((grid - h) ** 2 + w ** 2) for h, r in lines)
    return encode_profile(grid, prof)


def apply(rows, field_mhz, *, name=None) -> int:
    """Replace, in place, the pattern of every m / m* row with the spin
    system's when ``SPIN_PATTERNS`` is on. ``name`` for a compound's own
    fingerprint rows; otherwise each row's ``name``. Returns how many."""
    if not SPIN_PATTERNS or not field_mhz:
        return 0
    n = 0
    for r in rows or ():
        mult = str(r.get('multiplicity') or r.get('type') or '').lower().strip()
        if mult not in _PATTERN_MULTS:
            continue
        nm = name or r.get('name')
        c = r.get('ppm_listed', r.get('ppm_center', r.get('ppm')))
        if not nm or c is None:
            continue
        # as far as the stored pattern itself reaches
        reach = SPIN_REACH_PPM
        try:
            offs = [abs(float(o)) for o, _r in json.loads(r['pattern_lines'])]
            reach = max(reach, max(offs) / float(field_mhz) + 0.005)
        except (TypeError, ValueError, KeyError):
            pass
        got = signal_pattern(str(nm), float(c), float(field_mhz), reach)
        if got is None:
            continue
        r['pattern_lines'], r['pattern_profile'], r['profile_fwhm_hz'] = got
        r['pattern_source'] = 'spin system'
        n += 1
    return n
