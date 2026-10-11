# processing/spin_system.py
"""A 1H spectrum at any field from a GISSMO spin system.

GISSMO stores, per compound, the chemical shifts (ppm) and scalar couplings
(Hz) fitted to the BMRB reference spectrum: parameters that do not depend on
the field. The spectrum at another field is the eigen-solution of the
Hamiltonian at that field -- exact, second-order effects included.

    H = sum_i nu_i Iz_i + sum_{i<j} J_ij I_i . I_j        (Hz, rotating frame)
    nu_i = delta_i * SF

H conserves Fz, so it is diagonalised block by block (states with k spins
up). A line is a transition |a> (k up) -> |b> (k-1 up), at E_a - E_b with
intensity |<b|F-|a>|^2 (Pople, Schneider & Bernstein, High-resolution
Nuclear Magnetic Resonance, McGraw-Hill 1959; Levitt, Spin Dynamics, 2nd ed.,
Wiley 2008). Summed intensity is n * 2^(n-1); divided by 2^(n-1) each proton
contributes 1, so stoichiometry holds by construction.

Couplings to spins outside the matrix (GISSMO ``additional_coupling_
constants``: 31P, or protons of another sub-matrix) enter in the weak-coupling
limit: every external spin-1/2 is up or down with equal weight and shifts the
coupled proton by +/- J/2.

Spin-system parameters: Dashti H, Westler WM, Tonelli M, Wedell JR, Markley JL,
Eghbalnia HR. Spin System Modeling of Nuclear Magnetic Resonance Spectra for
Applications in Metabolomics and Small Molecule Screening. Anal Chem 2017;
89(22):12201-12208.
"""
from __future__ import annotations

import itertools
import json
import os
from itertools import combinations

import numpy as np


from app_paths import asset_path

# Bundled with the application (data/ ships with the installer).
SYSTEMS_JSON = asset_path('data', 'gissmo_spin_systems.json')


def load_systems(path=SYSTEMS_JSON) -> dict:
    """``{compound name: spin system}`` from the compact extract of the
    GISSMO XMLs (``validation/data/gissmo_spin_systems.json``): shifts in
    ppm, the coupling matrix in Hz, and the groups of couplings to spins
    outside the matrix."""
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    forms = doc.get('forms') or {}
    out = {}
    for s in doc['systems']:
        n = len(s['ppm'])
        J = np.zeros((n, n))
        for i, j, v in s['J_hz']:
            J[i, j] = J[j, i] = float(v)
        weight = np.ones(n)
        for f_ in forms.get(s['name']) or ():
            weight[list(f_['spins'])] = float(f_['fraction'])
        out[s['name']] = {'name': s['name'], 'bmse': s['bmse'],
                          'fit_field': s['fit_field_mhz'], 'lw': s['lw_hz'],
                          'ppm': np.asarray(s['ppm'], dtype=float), 'J': J,
                          # each spin's share of the molecules: below one
                          # where the entry holds several forms (anomers)
                          'weight': weight,
                          'external': [[(int(a), float(c)) for a, c in g]
                                       for g in s['external']]}
    return out


def _lines(nu, J):
    """(freq_hz, intensity) of every transition of the strongly coupled
    system ``nu`` (Hz), ``J`` (Hz)."""
    n = len(nu)
    blocks = {}
    for k in range(n + 1):
        states = [sum(1 << i for i in c) for c in combinations(range(n), k)]
        idx = {s: q for q, s in enumerate(states)}
        m = np.array([[0.5 if (s >> i) & 1 else -0.5 for i in range(n)]
                      for s in states]).reshape(len(states), n)
        H = np.diag(m @ nu + 0.5 * np.einsum('si,ij,sj->s', m, np.triu(J, 1) * 2, m))
        for q, s in enumerate(states):
            for i in range(n):
                for j in range(i + 1, n):
                    if J[i, j] and ((s >> i) & 1) != ((s >> j) & 1):
                        H[idx[s ^ (1 << i) ^ (1 << j)], q] += J[i, j] / 2.0
        E, V = np.linalg.eigh(H)
        blocks[k] = (states, idx, E, V)
    out_f, out_i = [], []
    for k in range(1, n + 1):
        sa, _ia, Ea, Va = blocks[k]
        sb, ib, Eb, Vb = blocks[k - 1]
        Fm = np.zeros((len(sb), len(sa)))
        for q, s in enumerate(sa):
            for i in range(n):
                if (s >> i) & 1:
                    Fm[ib[s ^ (1 << i)], q] = 1.0
        T = Vb.T @ Fm @ Va
        inten = T ** 2
        f = Ea[None, :] - Eb[:, None]
        keep = inten > 1e-9
        out_f.append(f[keep])
        out_i.append(inten[keep])
    f = np.concatenate(out_f)
    i = np.concatenate(out_i) / 2 ** (n - 1)
    return f, i


def _fragments(J):
    """The sets of spins coupled to one another, directly or through others:
    the connected components of the coupling graph."""
    from scipy.sparse.csgraph import connected_components
    _k, lab = connected_components((np.abs(J) > 0).astype(int),
                                   directed=False)
    return [np.flatnonzero(lab == c) for c in range(int(lab.max()) + 1)]


def stick_spectrum(sys_, field_mhz):
    """(ppm, intensity) of every line at ``field_mhz``; intensities sum to
    the number of protons, each form of the compound at its share
    (``weight``, the JSON's ``forms``).

    Solved fragment by fragment: spins with no coupling path between them
    have independent Hamiltonians, and the spectrum of the whole is the sum
    of the fragments' -- exact, not an approximation. The merged matrix of
    acetylcarnitine is 17 spins (a middle block of 24,310 states); its
    largest coupled fragment is 5. Each fragment enumerates only the
    external couplings that reach it."""
    nu0 = sys_['ppm'] * field_mhz
    J = sys_['J']
    weight = sys_.get('weight', np.ones(len(nu0)))
    fs, ins = [], []
    for frag in _fragments(J):
        pos = {int(s): k for k, s in enumerate(frag)}
        groups = [[(pos[si], c) for si, c in g if si in pos]
                  for g in sys_['external']]
        groups = [g for g in groups if g]
        Jf = J[np.ix_(frag, frag)]
        for signs in itertools.product((0.5, -0.5), repeat=len(groups)):
            nu = nu0[frag].copy()
            for g, s in zip(groups, signs):
                for si, c in g:
                    nu[si] += s * c
            f, i = _lines(nu, Jf)
            fs.append(f)
            ins.append(i * float(np.mean(weight[frag])) / 2 ** len(groups))
    return np.concatenate(fs) / field_mhz, np.concatenate(ins)


def render(ppm_axis, sticks_ppm, sticks_int, fwhm_hz, field_mhz):
    """Lorentzian lines of ``fwhm_hz`` on ``ppm_axis`` (any order); the area
    of each line in ppm units equals its intensity."""
    x = np.asarray(ppm_axis, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    step = float(np.median(np.diff(xs)))
    grid_lo = xs[0]
    ngrid = xs.size
    hist = np.zeros(ngrid)
    pos = (np.asarray(sticks_ppm) - grid_lo) / step
    ok = (pos >= 0) & (pos < ngrid - 1)
    p0 = np.floor(pos[ok]).astype(int)
    w1 = pos[ok] - p0
    np.add.at(hist, p0, sticks_int[ok] * (1 - w1))
    np.add.at(hist, p0 + 1, sticks_int[ok] * w1)
    hw = fwhm_hz / 2.0 / field_mhz
    k = (np.arange(ngrid) - ngrid // 2) * step
    kern = (hw / np.pi) / (k ** 2 + hw ** 2)
    y = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(np.fft.ifftshift(kern))))
    out = np.empty_like(y)
    out[order] = y
    return out
