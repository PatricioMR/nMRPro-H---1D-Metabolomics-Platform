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
    spins, sp, si = st
    mine = np.abs(spins - float(ppm_listed)) <= (reach_ppm or SPIN_REACH_PPM)
    if not mine.any():
        return None
    nearest = np.argmin(np.abs(sp[:, None] - spins[None, :]), axis=1)
    keep = mine[nearest] & (si > 0)
    if keep.sum() < 2:
        return None
    lp, li = sp[keep], si[keep]
    f = float(field_mhz)
    centre = float(np.sum(lp * li) / np.sum(li))
    hz = (lp - centre) * f
    order = np.argsort(hz)
    hz, li = hz[order], li[order]
    merged = []
    for h, i in zip(hz, li):
        if merged and h - merged[-1][0] <= MERGE_HZ:
            w = merged[-1][1] + i
            merged[-1] = ((merged[-1][0] * merged[-1][1] + h * i) / w, w)
        else:
            merged.append((float(h), float(i)))
    top = max(i for _h, i in merged)
    lines = [(round(h, 3), round(i / top, 4)) for h, i in merged if i / top >= MIN_REL]
    if len(lines) < 2:
        return None
    from processing.pattern_shape import encode_profile
    lo, hi = lines[0][0] - 10.0, lines[-1][0] + 10.0
    grid = np.arange(lo, hi + 1e-9, 0.1)
    w = PROFILE_FWHM_HZ / 2.0
    prof = sum(r * (w / np.pi) / ((grid - h) ** 2 + w ** 2) for h, r in lines)
    return json.dumps(lines), encode_profile(grid, prof), PROFILE_FWHM_HZ


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
