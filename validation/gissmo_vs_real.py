"""Is the GISSMO simulator right, and right at every field?

Three checks, each independent of the others:

``solver``  -- the application's solver (processing.spin_system, Fz blocks
              and uncoupled fragments) against a brute-force one written here
              from scratch: the whole 2^n Hilbert space built from Kronecker
              products, diagonalised in one piece. Every system of up to
              ``MAX_BRUTE_SPINS`` spins, at 400/500/600/700 MHz. The spectra
              must be the same line for line.
``params``  -- the shifts and couplings in data/gissmo_spin_systems.json
              against the GISSMO XMLs read again with a parser of this file.
``real``    -- the simulated spectrum against the REAL 1H spectra BMRB
              deposited (databases/bmrb_fids: 400, 500, 600 and 700 MHz), of
              the same compound. The question is asked three ways:
                * at the field the system was fitted at, on another entry or
                  the same one (the fit's own quality);
                * at ANOTHER field than the fit's -- the prediction GISSMO's
                  premise rests on (shifts in ppm and couplings in Hz do not
                  depend on the field, the spectrum is the Hamiltonian's
                  eigen-solution at the field);
                * against two controls on the very same spectrum: the system
                  solved at a wrong field (400 <-> 800), and solved to first
                  order (every coupling a plain splitting). A simulator that
                  is right about the field beats both where the coupling is
                  strong (|delta nu| / J small) and ties them where it is weak.
              Fit: amplitude x simulation + one offset per window, a global
              offset of +/-30 mppm (the depositions are at their own pH),
              line width and Lorentz/Gauss mix free; R^2 over the windows
              around the compound's lines, water and DSS left out.

    python -m validation.gissmo_vs_real real --jobs 2     # resumable, below-normal priority

Spin Hamiltonian and transition intensities: Pople JA, Schneider WG,
Bernstein HJ. High-resolution Nuclear Magnetic Resonance. McGraw-Hill, 1959;
Levitt MH. Spin Dynamics, 2nd ed. Wiley, 2008. Spin systems: Dashti H et al.
Anal Chem 2017;89:12201-12208.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from processing.spin_system import load_systems, stick_spectrum

REPO = pathlib.Path(__file__).resolve().parents[1]
FIDS = REPO / 'databases' / 'bmrb_fids'
BMSE_MAP = REPO / 'hmdb_api_download' / '_bmse_to_compound.tsv'
XML_DIR = pathlib.Path.home() / 'OneDrive' / 'Documentos' / 'NMR Databases' / 'gissmo'
OUT = REPO / 'validation' / 'out' / 'gissmo_vs_real'
MAX_BRUTE_SPINS = 11
FIELDS = (400.0, 500.0, 600.0, 700.0)
EXCLUDE = ((4.6, 5.0), (-0.05, 0.05), (0.60, 0.66), (1.72, 1.79), (2.88, 2.94))


# ── solver ────────────────────────────────────────────────────────────────────

def brute_lines(nu, J):
    """``(freq_hz, intensity)`` of every transition of H = sum nu_i Iz_i +
    sum_{i<j} J_ij I_i.I_j over the full 2^n space; intensities |<b|F-|a>|^2
    divided by 2^(n-1), so each proton contributes 1."""
    n = len(nu)
    eye = np.eye(2)
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    sm = sp.T

    def op(*pairs):
        """Kronecker product with the given single-spin operators at their
        positions and the identity elsewhere."""
        at = dict(pairs)
        out = np.array([[1.0]])
        for k in range(n):
            out = np.kron(out, at.get(k, eye))
        return out
    H = sum(float(nu[i]) * op((i, sz)) for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j]:
                H = H + float(J[i, j]) * (op((i, sz), (j, sz))
                                          + 0.5 * (op((i, sp), (j, sm)) + op((i, sm), (j, sp))))
    Fm_lab = sum(op((i, sm)) for i in range(n))
    E, V = np.linalg.eigh(H)
    Fm = V.T @ Fm_lab @ V
    b, a = np.nonzero(np.abs(Fm) > 1e-7)
    return E[a] - E[b], (Fm[b, a] ** 2) / 2 ** (n - 1)


def line_difference(fa, ia, fb, ib, bin_hz=0.01):
    """Largest difference between two line lists, as a share of the tallest
    line: intensities summed in bins of ``bin_hz`` and each bin read with its
    neighbours, so a line that rounding put one bin over is not a
    difference."""
    keys = np.round(np.concatenate([fa, fb]) / bin_hz).astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    va = np.bincount(inv[:fa.size], weights=ia, minlength=uniq.size)
    vb = np.bincount(inv[fa.size:], weights=ib, minlength=uniq.size)
    # neighbours: bins whose keys differ by one
    def smooth(v):
        out = v.copy()
        for d in (-1, 1):
            pos = np.searchsorted(uniq, uniq + d)
            ok = (pos < uniq.size) & (uniq[np.minimum(pos, uniq.size - 1)] == uniq + d)
            out[ok] += v[pos[ok]]
        return out
    sa, sb = smooth(va), smooth(vb)
    return float(np.abs(sa - sb).max() / max(sb.max(), 1e-12))


def check_solver(systems):
    rows = []
    for name, s in sorted(systems.items()):
        n = len(s['ppm'])
        if n > MAX_BRUTE_SPINS:
            continue
        bare = dict(s, external=[], weight=np.ones(n))
        for f in FIELDS:
            pa, ia = stick_spectrum(bare, f)
            fb, ib = brute_lines(np.asarray(s['ppm']) * f, s['J'])
            keep = ib > 1e-9
            rows.append({'compound': name, 'spins': n, 'field': f,
                         'lines': int(keep.sum()),
                         'max_rel_diff': line_difference(pa * f, ia, fb[keep], ib[keep]),
                         'protons_app': float(ia.sum()), 'protons_brute': float(ib.sum())})
    return rows


# ── parameters ────────────────────────────────────────────────────────────────

def parse_xml(path):
    """``(field, ppm list, {(i, j): J})`` of the merged matrix (index 1)."""
    root = ET.parse(path).getroot()
    field = float(root.findtext('field_strength'))
    for cm in root.iter('coupling_matrix'):
        if (cm.findtext('label') or '').strip() == 'merged' and cm.findtext('index').strip() == '1':
            ppm = [float(c.get('ppm')) for c in sorted(cm.iter('cs'), key=lambda c: int(c.get('index')))]
            J = {}
            for c in cm.iter('coupling'):
                i, j = int(c.get('from_index')) - 1, int(c.get('to_index')) - 1
                J[(min(i, j), max(i, j))] = float(c.get('value'))
            return field, ppm, J
    return None


def check_params(systems):
    import json
    doc = json.loads((REPO / 'data' / 'gissmo_spin_systems.json').read_text(encoding='utf-8'))
    rows = []
    for s in doc['systems']:
        path = XML_DIR / s['gissmo_file']
        if not path.exists():
            rows.append({'compound': s['name'], 'status': 'xml missing'})
            continue
        got = parse_xml(path)
        if got is None:
            rows.append({'compound': s['name'], 'status': 'no merged matrix'})
            continue
        field, ppm, J = got
        ours = {(min(i, j), max(i, j)): float(v) for i, j, v in s['J_hz']}
        dppm = (max(abs(a - b) for a, b in zip(ppm, s['ppm']))
                if len(ppm) == len(s['ppm']) else float('inf'))
        keys = set(J) | set(ours)
        dJ = max((abs(J.get(k, 0.0) - ours.get(k, 0.0)) for k in keys), default=0.0)
        rows.append({'compound': s['name'], 'status': 'ok',
                     'n_spins_xml': len(ppm), 'n_spins_json': len(s['ppm']),
                     'max_dppm': dppm, 'max_dJ_hz': dJ,
                     'dfield_mhz': abs(field - float(s['fit_field_mhz']))})
    return rows


# ── real spectra ──────────────────────────────────────────────────────────────

def first_order(sys_, field):
    """Every proton a multiplet of plain splittings by its couplings (equal
    shifts do not split): what a spectrum looks like when |delta nu| >> J."""
    ppm = np.asarray(sys_['ppm'], float)
    J = sys_['J']
    fs, ins = [], []
    for i in range(len(ppm)):
        lines = np.array([ppm[i] * field]); w = np.array([1.0])
        for j in range(len(ppm)):
            if j == i or not J[i, j] or abs(ppm[i] - ppm[j]) < 1e-4:
                continue
            lines = np.concatenate([lines + J[i, j] / 2, lines - J[i, j] / 2])
            w = np.concatenate([w / 2, w / 2])
        fs.append(lines / field); ins.append(w)
    return np.concatenate(fs), np.concatenate(ins)


def _read_procs(proc_dir):
    txt = open(os.path.join(proc_dir, 'procs'), encoding='latin-1').read()
    return {k: v for k, v in re.findall(r'##\$(\w+)=\s*(\S+)', txt)}


def real_spectra(bmse_list):
    """``[(bmse, experiment, field_mhz, ppm, y)]`` of the 1D 1H spectra of
    these entries."""
    from data.bruker_loader import _load_spectrum
    out = []
    for b in bmse_list:
        for proc in glob.glob(str(FIDS / b / '*' / '*' / 'pdata' / '1')):
            if not os.path.exists(os.path.join(proc, '1r')):
                continue
            try:
                sf = float(_read_procs(proc).get('SF'))
                ppm, y = _load_spectrum(proc)
            except Exception:
                continue
            exp = pathlib.Path(proc).parts[-3]
            out.append((b, exp, sf, np.asarray(ppm, float), np.asarray(y, float)))
    return out


def _windows(sticks_ppm, sticks_int, field):
    """Windows around the compound's lines (merged where they touch), with
    the excluded regions cut out."""
    keep = sticks_int > 0.002 * sticks_int.max()
    p = np.sort(sticks_ppm[keep])
    pad = 0.012 + 3.0 / field
    wins = []
    for v in p:
        if wins and v - pad <= wins[-1][1]:
            wins[-1][1] = v + pad
        else:
            wins.append([v - pad, v + pad])
    out = []
    for lo, hi in wins:
        segs = [(lo, hi)]
        for elo, ehi in EXCLUDE:
            nxt = []
            for a, b in segs:
                if ehi <= a or elo >= b:
                    nxt.append((a, b))
                else:
                    if a < elo: nxt.append((a, elo))
                    if ehi < b: nxt.append((ehi, b))
            segs = nxt
        out += [s for s in segs if s[1] - s[0] > 0.004]
    return out


def fit_r2(ppm, y, sticks, field, *, shift=0.03, step=0.002, fine_step=0.0002):
    """Best R^2 of y ~ a * simulation(shifted, width, mix) + offset per
    window, over the windows of the simulation's own lines."""
    sp, si = sticks
    wins = _windows(sp, si, field)
    if not wins:
        return np.nan
    order = np.argsort(ppm)
    xs, ys = ppm[order], y[order]
    idx = [np.flatnonzero((xs >= lo - shift) & (xs <= hi + shift)) for lo, hi in wins]
    pts = [np.flatnonzero((xs >= lo) & (xs <= hi)) for lo, hi in wins]
    if sum(len(p) for p in pts) < 20:
        return np.nan
    x_all = np.concatenate([xs[p] for p in pts])
    y_all = np.concatenate([ys[p] for p in pts])
    win_of = np.concatenate([np.full(len(p), k) for k, p in enumerate(pts)])
    offs = (win_of[:, None] == np.arange(len(pts))[None, :]).astype(float)
    sst = float(((y_all - y_all.mean()) ** 2).sum())
    if sst <= 0:
        return np.nan
    fine = np.arange(x_all.min() - shift - 0.002, x_all.max() + shift + 0.002, 0.0001)
    best = -np.inf
    for fwhm in (0.6, 1.0, 1.6, 2.5, 4.0):
        hw = fwhm / 2.0 / field
        sig = fwhm / field / (2 * np.sqrt(2 * np.log(2)))
        d = fine[:, None] - sp[None, :]
        L = ((hw / np.pi) / (d * d + hw * hw)) @ si
        G = (np.exp(-d * d / (2 * sig * sig)) / (sig * np.sqrt(2 * np.pi))) @ si
        for eta in (1.0, 0.5):
            prof = eta * L + (1 - eta) * G

            def r2_at(dlt):
                sim = np.interp(x_all - dlt, fine, prof)
                A = np.column_stack([sim, offs])
                coef, *_ = np.linalg.lstsq(A, y_all, rcond=None)
                if coef[0] <= 0:
                    return -np.inf
                r = y_all - A @ coef
                return 1.0 - float(r @ r) / sst
            # coarse, then fine around the best coarse offset
            coarse = np.arange(-shift, shift + 1e-12, step)
            vals = [r2_at(d) for d in coarse]
            d0 = float(coarse[int(np.argmax(vals))])
            for dlt in np.arange(d0 - step, d0 + step + 1e-12, fine_step):
                vals.append(r2_at(dlt))
            best = max(best, max(vals))
    return best


def coupling_strength(sys_, field):
    """min |delta nu| / |J| over the coupled pairs of distinct shift: below
    ~3 the multiplets are second order at this field."""
    ppm = np.asarray(sys_['ppm'], float); J = sys_['J']
    vals = [abs(ppm[i] - ppm[j]) * field / abs(J[i, j])
            for i in range(len(ppm)) for j in range(i + 1, len(ppm))
            if abs(J[i, j]) > 0.5 and abs(ppm[i] - ppm[j]) > 1e-4]
    return min(vals) if vals else np.inf


def _one_compound(job):
    name, sys_, entries = job
    rows = []
    for bmse, exp, sf, ppm, y in real_spectra(entries):
        alt = 400.0 if sf >= 550 else 800.0
        exact = stick_spectrum(sys_, sf)
        r_exact = fit_r2(ppm, y, exact, sf)
        r_first = fit_r2(ppm, y, first_order(sys_, sf), sf)
        sw = stick_spectrum(sys_, alt)
        r_swap = fit_r2(ppm, y, (sw[0], sw[1]), sf)
        fit_f = float(sys_['fit_field'])
        same_field = abs(sf - fit_f) < 15
        rows.append({'compound': name, 'bmse': bmse, 'experiment': exp, 'field': round(sf, 2),
                     'fit_field': fit_f, 'fit_entry': sys_['bmse'],
                     'case': ('fit spectrum' if same_field and bmse == sys_['bmse']
                              else 'same field, other entry' if same_field else 'other field'),
                     'kappa': coupling_strength(sys_, sf),
                     'r2_exact': r_exact, 'r2_first_order': r_first, 'r2_wrong_field': r_swap,
                     'wrong_field': alt})
    return rows


def _below_normal():
    """Worker initializer: below-normal priority, so the machine stays usable."""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def check_real(systems, jobs=2):
    """Resumable: each compound's rows go to real_rows.jsonl as it ends,
    and a compound already there is not computed again."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    done_path = OUT / 'real_rows.jsonl'
    done = {}
    if done_path.exists():
        for line in done_path.read_text(encoding='utf-8').splitlines():
            rec = json.loads(line)
            done[rec['compound']] = rec['rows']
    cmap = dict(line.rstrip('\n').split('\t') for line in open(BMSE_MAP, encoding='utf-8') if '\t' in line)
    by_comp: dict = {}
    for b, c in cmap.items():
        by_comp.setdefault(c, []).append(b)
    todo = []
    for name, s in systems.items():
        c = cmap.get(s['bmse'])
        if c is None:
            continue
        entries = [b for b in by_comp[c] if (FIDS / b).exists()]
        if entries and name not in done:
            todo.append((name, s, entries))
    print(f'  {len(done)} compounds already done, {len(todo)} to go', flush=True)
    with ProcessPoolExecutor(max_workers=jobs, initializer=_below_normal) as ex, \
            open(done_path, 'a', encoding='utf-8') as fh:
        for k, ((name, _s, _e), r) in enumerate(zip(todo, ex.map(_one_compound, todo)), 1):
            done[name] = r
            fh.write(json.dumps({'compound': name, 'rows': r}, default=float) + '\n')
            fh.flush()
            if k % 10 == 0:
                print(f'  {k}/{len(todo)} compounds', flush=True)
    return [row for rs in done.values() for row in rs]


# ── report ────────────────────────────────────────────────────────────────────

def _write(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(OUT / name, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def summarise_real(rows) -> str:
    L = ['| caso | campo real | espectros | R² exato (mediana) | ≥ 0,9 | 1.ª ordem | campo errado | '
         'exato > 1.ª ordem | exato > campo errado |', '|---|---|---|---|---|---|---|---|---|']
    def line(sel, case, fld):
        e = np.array([r['r2_exact'] for r in sel], float)
        f = np.array([r['r2_first_order'] for r in sel], float)
        s = np.array([r['r2_wrong_field'] for r in sel], float)
        ok = np.isfinite(e) & np.isfinite(f) & np.isfinite(s)
        if not ok.sum():
            return
        e, f, s = e[ok], f[ok], s[ok]
        L.append(f'| {case} | {fld} | {ok.sum()} | {np.median(e):.3f} | {100 * np.mean(e >= 0.9):.0f}% | '
                 f'{np.median(f):.3f} | {np.median(s):.3f} | {100 * np.mean(e > f + 0.005):.0f}% | '
                 f'{100 * np.mean(e > s + 0.005):.0f}% |')
    for case in ('fit spectrum', 'same field, other entry', 'other field'):
        sel = [r for r in rows if r['case'] == case]
        line(sel, case, 'todos')
        if case == 'other field':
            for fld in sorted({round(r['field'] / 100) * 100 for r in sel}):
                line([r for r in sel if round(r['field'] / 100) * 100 == fld], case, f'{fld:.0f}')
    L += ['', 'Só os de acoplamento forte no campo do espectro (κ = |Δν|/J < 3):', '', L[0], L[1]]
    strong = [r for r in rows if r['kappa'] < 3]
    for case in ('fit spectrum', 'same field, other entry', 'other field'):
        line([r for r in strong if r['case'] == case], case, 'todos')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('cmd', choices=('solver', 'params', 'real', 'all'))
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    systems = load_systems()
    if a.cmd in ('solver', 'all'):
        rows = check_solver(systems)
        _write(rows, 'solver.csv')
        d = np.array([r['max_rel_diff'] for r in rows])
        dp = np.array([abs(r['protons_app'] - r['protons_brute']) for r in rows])
        print(f'solver: {len(rows)} (system, field) pairs, {len({r["compound"] for r in rows})} systems '
              f'of <= {MAX_BRUTE_SPINS} spins; max relative difference {d.max():.2e}, '
              f'max proton-count difference {dp.max():.2e}', flush=True)
    if a.cmd in ('params', 'all'):
        rows = check_params(systems)
        _write(rows, 'params.csv')
        ok = [r for r in rows if r['status'] == 'ok']
        bad = [r for r in ok if r['max_dppm'] > 1e-4 or r['max_dJ_hz'] > 1e-3 or r['dfield_mhz'] > 0.5]
        print(f'params: {len(ok)} of {len(rows)} systems compared with their XML; '
              f'{len(bad)} differ (ppm > 1e-4, J > 1e-3 Hz or field > 0.5 MHz); '
              f'not compared: {[r["compound"] for r in rows if r["status"] != "ok"]}', flush=True)
        for r in bad[:10]:
            print('   ', r)
    if a.cmd in ('real', 'all'):
        rows = check_real(systems, a.jobs)
        _write(rows, 'real.csv')
        text = summarise_real(rows)
        (OUT / 'real_summary.md').write_text(text, encoding='utf-8')
        print(text, flush=True)


if __name__ == '__main__':
    main()
