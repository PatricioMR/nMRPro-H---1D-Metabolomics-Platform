# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""
Lipidomics method — isolated from the metabolomics pipeline.

Lipidomics here is deliberately simple: **pure trapezoidal integration over the
ppm regions defined in the reference library**. There is no peak detection, no
deconvolution and no line fitting — the whole database region is integrated.

Two cited reference libraries are selected by acquisition solvent:

  • ``D2O``   — SigMa plasma lipid library (aqueous, intact plasma/serum).
               Khakimov B, Mobaraki N, Trimigno A, Aru V, Engelsen SB.
               "Signature Mapping (SigMa): An efficient approach for processing
               complex human urine 1H NMR metabolomics data."
               Anal Chim Acta 2020;1108:142-151. doi:10.1016/j.aca.2020.02.025
               Relative quantification only: no proton counts, no internal
               standard, no fatty-acid composition (Box 1 is invalid in water).

  • ``CDCl3`` — Li Q, et al. "Applications of nuclear magnetic resonance in lipid
               analyses." Prog Lipid Res 2017;68:37-56. Lipid-extract spectra.
               Per-class diagnostic regions plus the Box 1 fatty-acid composition
               windows used to derive FA-composition indices.

Chemical shifts depend on the solvent/sample system, so the two libraries are
**not interchangeable**: Box 1 composition indices (CDCl3 shifts, lipid extract)
must never be applied to aqueous D2O plasma spectra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Solvent → bundled library file. Single source of truth for the UI selector.
SOLVENT_LIBRARIES = {
    'D2O':   'lipidomics_sigma_d2o.csv',
    'CDCl3': 'lipidomics_li_cdcl3.csv',
}
DEFAULT_SOLVENT = 'D2O'

# UI labels (kept here so stage and module never drift apart).
SOLVENT_LABELS = {
    'D2O':   'D2O (SigMa — plasma/serum, relative)',
    'CDCl3': 'CDCl3 (Li — lipid extract, FA composition)',
}


# ── small local DSP helpers (processing/ must not import gui/) ────────────────

def _trapz(y, x):
    try:
        return np.trapezoid(y, x=x)
    except AttributeError:
        return np.trapz(y, x=x)


def _cv(vals):
    """CV% of a 1-D iterable (sample std, ddof=1). NaN if < 2 finite positives."""
    v = [x for x in vals if np.isfinite(x) and x > 0]
    if len(v) < 2:
        return float('nan')
    m = float(np.mean(v))
    return float(np.std(v, ddof=1) / m * 100.0) if m > 0 else float('nan')


# ── library loading ──────────────────────────────────────────────────────────

def library_filename(solvent: str) -> str:
    """Bundled CSV filename for a solvent. Raises ValueError on unknown solvent."""
    try:
        return SOLVENT_LIBRARIES[solvent]
    except KeyError:
        raise ValueError(
            f"Unknown lipidomics solvent {solvent!r}; "
            f"expected one of {sorted(SOLVENT_LIBRARIES)}")


def load_library(solvent: str) -> pd.DataFrame:
    """
    Load the solvent-specific lipid region library as a DataFrame.

    Uses ``app_paths.asset_path`` so it works both from source and from the
    bundled (PyInstaller) build.
    """
    from app_paths import asset_path
    path = asset_path('data', library_filename(solvent))
    df = pd.read_csv(path)
    # Defensive: guarantee numeric region bounds.
    df['start_ppm'] = pd.to_numeric(df['start_ppm'], errors='coerce')
    df['end_ppm'] = pd.to_numeric(df['end_ppm'], errors='coerce')
    return df.dropna(subset=['start_ppm', 'end_ppm']).reset_index(drop=True)


def build_regions(df: pd.DataFrame) -> list[dict]:
    """
    Build the list of integration regions from a library DataFrame.

    Lipidomics integrates **every** library region — there is no dependency on
    detected peaks. Box 1 composition windows are excluded here; they feed the
    composition indices, not the per-region area table.
    """
    regions = []
    for idx, row in df.iterrows():
        cls = str(row.get('class', '') or '')
        if cls == 'BOX1_WINDOW':
            continue
        lo = float(min(row['start_ppm'], row['end_ppm']))
        hi = float(max(row['start_ppm'], row['end_ppm']))
        try:
            rid = str(int(row['id']))
        except Exception:
            rid = str(row.get('id', idx))
        regions.append({
            'id':         rid,
            'name':       str(row.get('name', 'Unknown')),
            'class':      cls,
            'start_ppm':  lo,
            'end_ppm':    hi,
            'ppm':        (lo + hi) / 2.0,
            'signal':     str(row.get('signal', '') or ''),
            'comment':    str(row.get('comment', '') or ''),
            'solvent':    str(row.get('solvent', '') or ''),
        })
    return regions


# ── region integration (pure trapz) ──────────────────────────────────────────

def integrate_region(ppm_arr, intens, lo_ppm, hi_ppm, n_s):
    """
    Trapezoidal area of one ppm region for every sample.

    ``intens`` is the (n_points, n_samples) intensity matrix. The ppm axis is
    inverted (δ decreases left→right); ``abs`` makes the area sign-independent.
    Returns a length-``n_s`` array, or ``None`` when the region falls outside
    the spectrum (so the caller can report "not detected", never fabricate 0).
    """
    lo = float(min(lo_ppm, hi_ppm))
    hi = float(max(lo_ppm, hi_ppm))
    mask = (ppm_arr >= lo) & (ppm_arr <= hi)
    if not mask.any():
        return None
    return np.array([
        abs(float(_trapz(intens[mask, s], ppm_arr[mask])))
        for s in range(n_s)
    ], dtype=float)


# ── Box 1 fatty-acid composition indices (CDCl3 / Li 2017 only) ───────────────

def supports_composition(solvent: str) -> bool:
    """Box 1 FA composition is only valid for CDCl3 lipid-extract spectra."""
    return solvent == 'CDCl3'


def _box1_windows(df: pd.DataFrame) -> dict:
    """
    Extract the Box 1 integration windows from the library (single source of
    truth). Each ``BOX1_WINDOW`` row carries ``box1=<key>`` in its comment.
    Returns ``{key: (lo, hi)}`` for keys a,b,c,d,e,h,i.
    """
    windows = {}
    box = df[df['class'].astype(str) == 'BOX1_WINDOW']
    for _, row in box.iterrows():
        comment = str(row.get('comment', '') or '')
        key = None
        for tok in comment.split(';'):
            tok = tok.strip()
            if tok.startswith('box1='):
                key = tok.split('=', 1)[1].strip()
                break
        if not key:
            continue
        lo = float(min(row['start_ppm'], row['end_ppm']))
        hi = float(max(row['start_ppm'], row['end_ppm']))
        windows[key] = (lo, hi)
    return windows


def composition_indices(ppm_arr, all_int, n_s, groups, df):
    """
    1H NMR lipid composition indices following Li et al. 2017,
    Prog Lipid Res 68:37-56, Box 1 (FA composition from diagnostic peaks).

    Integration windows are read from the CDCl3 library's ``BOX1_WINDOW`` rows
    (single source of truth), keyed by ``box1=`` in the comment:

      a  =CH-CH= olefinic      2H per C=C bond
      b  TG glycerol CH2       4H per TG molecule
      c  bis-allylic =CH-CH2   2H per group
      d  alpha-CH2 to C=O      2H per FA chain (6H per TG)
      e  allylic =CH-CH2       2H per group
      h  terminal CH3 (non-n3) 3H per FA chain
      i  terminal CH3 n-3 ALA  3H per chain (linolenic-specific)

    U = (b/4 + d/6 + (h+i)/9) / 3 — intensity per FA chain (3-estimator avg)

    Results are dimensionless ratios (%) / double bonds per chain and are not
    subject to IS/PQN normalisation. The n-3 index captures **only ALA** (18:3):
    EPA/DHA terminal CH3 overlaps window ``h`` at ~0.88 ppm and is therefore not
    resolved — this caveat is recorded in every n-3 entry's status reasons.
    """
    windows = _box1_windows(df)
    if not windows:
        return []

    raw = {}
    for key, (lo, hi) in windows.items():
        mask = (ppm_arr >= lo) & (ppm_arr <= hi)
        if mask.any():
            raw[key] = np.array([
                abs(float(_trapz(all_int[mask, s], ppm_arr[mask])))
                for s in range(n_s)], dtype=float)
        else:
            raw[key] = np.zeros(n_s, dtype=float)

    a = raw.get('a', np.zeros(n_s))
    b = raw.get('b', np.zeros(n_s))
    c = raw.get('c', np.zeros(n_s))
    d = raw.get('d', np.zeros(n_s))
    e = raw.get('e', np.zeros(n_s))
    h = raw.get('h', np.zeros(n_s))
    i_ = raw.get('i', np.zeros(n_s))

    # Box 1, Eq (1) — intensity per FA chain (proton-normalised average)
    U = (b / 4.0 + d / 6.0 + (h + i_) / 9.0) / 3.0
    eps = np.finfo(float).eps

    with np.errstate(divide='ignore', invalid='ignore'):
        # n-3 PUFA %: linolenic-type ALA (18:3n-3) only (see docstring caveat).
        n3_pct = np.where(U > eps, np.clip((i_ / 9.0) / U * 100.0, 0.0, 100.0), 0.0)

        # n-6 PUFA %: from bis-allylic signal. Linolenic contributes 2 bis-allylic
        # CH2 groups per chain, linoleic 1. Subtract ALA: c/2 - 2*(i/9).
        n6_bisallylic = np.maximum(0.0, c / 2.0 - 2.0 * (i_ / 9.0))
        n6_pct = np.where(U > eps, np.clip(n6_bisallylic / U * 100.0, 0.0, 100.0), 0.0)

        # MUFA %: from allylic signal after subtracting PUFA contributions.
        mufa_allylic = np.maximum(0.0, e / 4.0 - 2.0 * (i_ / 9.0) - n6_bisallylic)
        mufa_pct = np.where(U > eps, np.clip(mufa_allylic / U * 100.0, 0.0, 100.0), 0.0)

        # Saturated FA % — remainder; 0 when U≈0 (no signal → no composition).
        sat_pct = np.where(U > eps,
                           np.clip(100.0 - n3_pct - n6_pct - mufa_pct, 0.0, 100.0),
                           0.0)

        # Degree of Unsaturation: average C=C double bonds per FA chain.
        du = np.where(U > eps, (a / 2.0) / (U * 3.0), 0.0)

        # Unsaturation Index: weighted (MUFA=1, n-6=2, n-3=3 double bonds).
        ui = np.clip((mufa_pct + 2.0 * n6_pct + 3.0 * n3_pct) / 100.0, 0.0, None)

    _N3_CAVEAT = ('n-3 index captures ALA (18:3) only — EPA/DHA terminal CH3 '
                  'overlaps the 0.88 ppm window and is not resolved.')

    def _entry(name, values, unit='%', extra_reasons=None):
        vals = np.asarray(values, dtype=float)
        mean_val = float(np.nanmean(vals))
        cv_by_group, concs_by_group = {}, {}
        for g, idxs in groups.items():
            gv = [float(vals[idx]) for idx in idxs
                  if idx < n_s and np.isfinite(vals[idx])]
            cv_by_group[g] = _cv(gv)
            concs_by_group[g] = gv
        reasons = [f'Derived index ({unit}) — Li et al. 2017 Box 1']
        if extra_reasons:
            reasons.extend(extra_reasons)
        return {
            'metabolite':          name,
            'hmdb_id':             '__LIPID_INDEX__',
            'ppm':                 0.0,
            'n_peaks':             0,
            'raw_area':            mean_val,
            'corr_area':           mean_val,
            'conc_mm':             mean_val,
            'cv_by_group':         cv_by_group,
            'snr':                 0.0,
            'status':              '—',
            '_confidence':         0,
            '_status_reasons':     reasons,
            '_peak':               None,
            '_areas_per_sample':   vals.tolist(),
            '_concs_per_sample':   vals.tolist(),
            '_concs_by_group':     concs_by_group,
            '_used_mcr':           False,
            '_lorentz':            None,
            '_mcr':                None,
            '_lipid_regions':      None,
            '_lipid_window':       None,
            '_integ_mode':         'LipidIndex',
            '_ann_score':          1.0,
            '_nh':                 1.0,
            '_blk_mean_area':      0.0,
            '_proton_corr':        False,
            '_use_blk_sub':        False,
            '_has_anchor':         False,
            '_peak_rank':          None,
            '_used_j_resolve':     False,
            '_n_eligible':         0,
            '_peak_align_shifts':  None,
            '_is_lipid_index':     True,
        }

    return [
        _entry('n-3 PUFA — ALA index (%)',          n3_pct, '%', [_N3_CAVEAT]),
        _entry('n-6 PUFA — bis-allylic index (%)',  n6_pct, '%'),
        _entry('MUFA — allylic index (%)',          mufa_pct, '%'),
        _entry('Saturated FA (%)',                  sat_pct, '%'),
        _entry('Degree of Unsaturation (C=C/chain)', du,    'ratio'),
        _entry('Unsaturation Index (UI)',           ui,     'ratio'),
    ]


# ── Fatty-acid ratios for the Statistics "FA ratios" panel ────────────────────

def fa_ratio_series(index_entries):
    """
    Per-sample fatty-acid ratios derived from the Box 1 composition indices
    (CDCl3 only). Returns an ordered ``{ratio_label: np.ndarray(n_samples)}``.

    Empty when composition indices are absent (e.g. an aqueous D2O run), which
    the caller surfaces as a message rather than fabricating ratios. Division
    guards yield NaN where a denominator is non-positive — never a fake 0.
    """
    by = {e['metabolite']: np.asarray(e.get('_concs_per_sample', []), dtype=float)
          for e in (index_entries or []) if e.get('_is_lipid_index')}
    if not by:
        return {}

    n3   = by.get('n-3 PUFA — ALA index (%)')
    n6   = by.get('n-6 PUFA — bis-allylic index (%)')
    mufa = by.get('MUFA — allylic index (%)')
    sfa  = by.get('Saturated FA (%)')
    du   = by.get('Degree of Unsaturation (C=C/chain)')
    ui   = by.get('Unsaturation Index (UI)')
    if any(v is None for v in (n3, n6, mufa, sfa)):
        return {}

    def _safe_div(a, b):
        out = np.full(a.shape, np.nan, dtype=float)
        m = np.isfinite(a) & np.isfinite(b) & (b > 0)
        out[m] = a[m] / b[m]
        return out

    pufa = n3 + n6
    out = {}
    out['SFA (%)']  = sfa
    out['MUFA (%)'] = mufa
    out['PUFA (%)'] = pufa
    out['PUFA/SFA'] = _safe_div(pufa, sfa)
    out['MUFA/SFA'] = _safe_div(mufa, sfa)
    out['n-6/n-3']  = _safe_div(n6, n3)
    if ui is not None:
        out['Unsaturation Index'] = ui
    if du is not None:
        out['Degree of Unsaturation'] = du
    return out
