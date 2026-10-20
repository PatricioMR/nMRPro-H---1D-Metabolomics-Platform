# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/conftest.py
"""
Shared fixtures for NMRAnalysis test suite.

Real data: MRP_PlasmaSamplesCurve_030226.csv  (12 samples, plasma matrix)
Real DB:   databases/HMDB_Serum_v01.db
"""
import os
import sys
import numpy as np
import pytest

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Paths ──────────────────────────────────────────────────────────────────────
NMR_CSV  = os.path.join(os.path.expanduser('~'), 'Downloads',
                        'MRP_PlasmaSamplesCurve_030226.csv')
PLASMA_DB = os.path.join(ROOT, 'databases', 'HMDB_Serum_v01.db')


# ── Real-data fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def plasma_csv_path():
    if not os.path.exists(NMR_CSV):
        pytest.skip(f'NMR CSV not found: {NMR_CSV}')
    return NMR_CSV


@pytest.fixture(scope='session')
def plasma_db_path():
    if not os.path.exists(PLASMA_DB):
        pytest.skip(f'Plasma DB not found: {PLASMA_DB}')
    return PLASMA_DB


@pytest.fixture(scope='session')
def loaded_spectrum(plasma_csv_path):
    """Load the real plasma CSV once per session."""
    from data.loader import load_all_samples
    ppm, intensities, names, labels = load_all_samples(plasma_csv_path)
    return {'ppm': ppm, 'intensities': intensities,
            'names': names, 'labels': labels}


# ── Synthetic fixtures (no files needed) ──────────────────────────────────────

@pytest.fixture
def synthetic_doublet():
    """
    A clean synthetic doublet: two lines separated by 7 Hz at 600 MHz
    (Δppm = 7/600 ≈ 0.01167 ppm), centred at 1.5 ppm.
    """
    mhz     = 600.0
    j_hz    = 7.0
    centre  = 1.5
    half    = j_hz / mhz / 2        # ≈ 0.00583 ppm

    n_pts   = 2000
    ppm     = np.linspace(2.0, 0.5, n_pts)
    spec    = np.zeros(n_pts)

    for offset in (-half, +half):
        peak_ppm = centre + offset
        lw       = 0.002             # Lorentzian half-width (ppm)
        spec    += lw**2 / ((ppm - peak_ppm)**2 + lw**2)

    # Two detected peaks matching the lines
    i_lo = int(np.argmin(np.abs(ppm - (centre - half))))
    i_hi = int(np.argmin(np.abs(ppm - (centre + half))))
    peaks = [
        {'rank': 1, 'ppm': float(ppm[i_hi]), 'height': float(spec[i_hi]),
         'start_idx': max(0, i_hi - 20), 'end_idx': min(n_pts-1, i_hi + 20),
         'start_ppm': float(ppm[i_hi]) + 0.02, 'end_ppm': float(ppm[i_hi]) - 0.02,
         'width_ppm': 0.004, 'snr': 50.0, 'status': 'Good'},
        {'rank': 2, 'ppm': float(ppm[i_lo]), 'height': float(spec[i_lo]),
         'start_idx': max(0, i_lo - 20), 'end_idx': min(n_pts-1, i_lo + 20),
         'start_ppm': float(ppm[i_lo]) + 0.02, 'end_ppm': float(ppm[i_lo]) - 0.02,
         'width_ppm': 0.004, 'snr': 50.0, 'status': 'Good'},
    ]
    return {'ppm': ppm, 'spec': spec, 'peaks': peaks,
            'centre': centre, 'j_hz': j_hz, 'mhz': mhz}


@pytest.fixture
def synthetic_singlet():
    """A clean singlet at 3.03 ppm (creatine-like)."""
    n_pts  = 2000
    ppm    = np.linspace(4.0, 1.0, n_pts)
    centre = 3.03
    lw     = 0.002
    spec   = lw**2 / ((ppm - centre)**2 + lw**2)
    i_pk   = int(np.argmin(np.abs(ppm - centre)))
    peaks  = [
        {'rank': 1, 'ppm': centre, 'height': 1.0,
         'start_idx': max(0, i_pk - 20), 'end_idx': min(n_pts-1, i_pk + 20),
         'start_ppm': centre + 0.02, 'end_ppm': centre - 0.02,
         'width_ppm': 0.004, 'snr': 80.0, 'status': 'Good'},
    ]
    return {'ppm': ppm, 'spec': spec, 'peaks': peaks, 'centre': centre}


@pytest.fixture
def synthetic_quartet_partial():
    """
    Quartet at 2.0 ppm, J=7 Hz at 600 MHz, with only 3 of 4 lines detected.
    Line positions (ppm offsets from centre): -3J/2, -J/2, +J/2, +3J/2
    The outermost high-field line (+3J/2) is missing from detected peaks.
    """
    mhz    = 600.0
    j_hz   = 7.0
    centre = 2.0
    jp     = j_hz / mhz  # ppm per J unit

    line_ppms = [centre - 1.5*jp, centre - 0.5*jp, centre + 0.5*jp, centre + 1.5*jp]
    line_rels = [1.0, 3.0, 3.0, 1.0]  # 1:3:3:1 quartet

    n_pts = 2000
    ppm   = np.linspace(2.5, 1.5, n_pts)
    spec  = np.zeros(n_pts)
    lw    = 0.002
    for lp, lr in zip(line_ppms, line_rels):
        spec += lr * lw**2 / ((ppm - lp)**2 + lw**2)

    # Provide only 3 peaks (drop the 4th line at line_ppms[3])
    peaks = []
    for rank, lp in enumerate(line_ppms[:3], 1):
        i_pk = int(np.argmin(np.abs(ppm - lp)))
        peaks.append({
            'rank': rank, 'ppm': float(ppm[i_pk]), 'height': float(spec[i_pk]),
            'start_idx': max(0, i_pk - 20), 'end_idx': min(n_pts-1, i_pk + 20),
            'start_ppm': float(ppm[i_pk]) + 0.02, 'end_ppm': float(ppm[i_pk]) - 0.02,
            'width_ppm': 0.004, 'snr': 30.0, 'status': 'Good',
        })
    return {'ppm': ppm, 'spec': spec, 'peaks': peaks, 'centre': centre,
            'j_hz': j_hz, 'mhz': mhz, 'n_lines': 4, 'n_detected': 3}
