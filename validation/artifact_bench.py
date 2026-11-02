# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The GISSMO tube with an artefact in some of its samples, annotated on the
study's mean trace and sample by sample.

The annotation reads every shape, coupling and fingerprint on the mean of
the samples. A sample with an artefact enters that mean: if it distorts the
lines enough, a real signal fails its gates in every sample at once. Here a
few samples of each group are spoiled after the tube is made, and the same
tube is annotated both ways (``fresh_run.prepare(annotation=...)``):

``phase``   a zero-order phase error: each column becomes
            cos(phi) * absorption + sin(phi) * dispersion, the dispersion
            being the Hilbert transform of the absorption spectrum (the
            Kramers-Kronig pair of a Lorentzian line; Ernst RR, Bodenhausen G,
            Wokaun A. Principles of Nuclear Magnetic Resonance in One and Two
            Dimensions. Oxford 1987, sec. 4.2).
``shim``    a poor shim: every line convolved with a two-Lorentzian kernel,
            broadened and with a shoulder on one side; the noise smoothed by
            the kernel is topped back up to the tube's own sigma.
``both``    the two together.

The concentrations are the tube's, so the truth per sample is the clean
tube's.

    python -m validation.artifact_bench --artifact phase shim --modes mean per_sample
    python -m validation.artifact_bench --export-dir "<validation folder>"
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import fftconvolve, hilbert

from validation import gissmo_bench as gb, mixture_bench as mb
from validation.crowded_bench import CACHE_DIR, _CACHED_SOURCES
from validation.fresh_run import prepare, quantify_prepared, score, tables

PHASE_DEG = 20.0
# Poor shim: the line, broadened, plus a shoulder this far to one side (Hz)
SHIM_KERNEL = ((0.65, 0.0, 1.5), (0.35, 3.0, 2.0))  # (weight, offset, HWHM)
# Spoiled samples per group, in order: the third of each group first.
BAD_ORDER = {'A': (2, 4, 0), 'B': (8, 10, 6)}


def bad_samples(n_per_group):
    return sorted(s for g in BAD_ORDER for s in BAD_ORDER[g][:n_per_group])


def _sigma(y):
    """The noise of a column from its point-to-point differences (signals
    are a small part of the axis)."""
    d = np.diff(y)
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2.0))


def _phase(y, deg):
    phi = np.deg2rad(deg)
    return np.cos(phi) * y + np.sin(phi) * np.imag(hilbert(y))


def _shim(y, field, rng):
    step_hz = abs(float(mb.PPM[1] - mb.PPM[0])) * field
    x = np.arange(-400, 401) * step_hz
    k = sum(w * g / (np.pi * ((x - off) ** 2 + g ** 2))
            for w, off, g in SHIM_KERNEL)
    k /= k.sum()
    sig = _sigma(y)
    out = fftconvolve(y, k, mode='same')
    left = max(1.0 - float(np.sum(k ** 2)), 0.0)
    return out + rng.normal(0.0, sig * np.sqrt(left), y.size)


def spoil(X, field, artifact, samples, seed):
    """``X`` with ``artifact`` applied to the columns ``samples``."""
    X = np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng(seed + 23_000_051)
    for s in samples:
        y = X[:, s]
        if artifact in ('phase', 'both'):
            y = _phase(y, PHASE_DEG)
        if artifact in ('shim', 'both'):
            y = _shim(y, field, rng)
        X[:, s] = y
    return X


def _key(size, field, seed, artifact, n_bad, mode):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha1(f'{size}|{field}|{seed}|{artifact}|{n_bad}|{mode}|'
                     f'{PHASE_DEG}|{SHIM_KERNEL}|{gb._data_tag()}'.encode())
    for rel in _CACHED_SOURCES:
        with open(os.path.join(root, rel), 'rb') as f:
            h.update(f.read())
    h.update(gb._quant_hash().encode())
    return h.hexdigest()[:16]


def run(size, field, seed, artifact, n_bad, mode, log=print):
    """``(score, rows, truth, prep)`` for one tube, artefact and mode."""
    X, truth = gb.tube(field, seed, size)
    if artifact != 'none':
        X = spoil(X, field, artifact, bad_samples(n_bad), seed)
    path = os.path.join(CACHE_DIR, f'artifact_{size}_{field:.0f}_s{seed}_'
                                   f'{artifact}{n_bad}_{mode}_'
                                   f'{_key(size, field, seed, artifact, n_bad, mode)}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            prep, rows = pickle.load(f)
    else:
        mb.MHZ = field
        with contextlib.redirect_stdout(io.StringIO()):
            prep = prepare(mb.PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT,
                           field_mhz=field, annotation=mode, log=lambda *a: None)
            rows = quantify_prepared(prep, mb.SAMPLES,
                                     standard_ppm=mb.STANDARD_PPM,
                                     standard_nh=mb.STANDARD_NH,
                                     standard_conc=mb.STANDARD_MM,
                                     standard_name='DSS')
        with open(path, 'wb') as f:
            pickle.dump((prep, rows), f, protocol=pickle.HIGHEST_PROTOCOL)
    return score(rows, truth), rows, truth, prep


def eligible_real(prep, truth):
    """Real analytes the annotation left eligible for quantification: a
    signal on a detected peak not marked Bad (``quantification`` rule 3)."""
    out = set()
    for met in prep['mets']:
        if str(met.get('id')) in truth and any(
                mp.get('peak_rank') is not None
                and mp.get('annotation_quality') != 'bad'
                for mp in met.get('matched_peaks') or ()):
            out.add(str(met.get('id')))
    return out


def bad_sample_error(rows, truth, per, bad):
    """Median |measured/true - 1| of the real analytes in the spoiled samples
    and in the clean ones."""
    e_bad, e_ok = [], []
    for r in rows:
        mid = r.get('hmdb_id')
        if r.get('_not_quantified') or r.get('_is_is') or mid not in truth:
            continue
        vals = r.get('_concs_per_sample')
        if vals is None or mid not in per:
            continue
        for s, v in enumerate(vals):
            t = per[mid][s] if s < len(per[mid]) else None
            if v is None or not t or not np.isfinite(v):
                continue
            (e_bad if s in bad else e_ok).append(abs(float(v) / float(t) - 1.0))
    med = lambda a: float(np.median(a)) if a else float('nan')
    return med(e_bad), med(e_ok)


def export(dest, results):
    """``dest``/10_Artefatos, made anew: each run's analytes and samples in
    the validation export's tables, and a summary."""
    import shutil
    from validation import export_validation as ev
    from validation.validation_folder import subfolder
    root = subfolder(dest, 'artifacts')
    if os.path.isdir(root):
        shutil.rmtree(root)
    names = ev._sample_names()
    summary = []
    for (size, fk, seed, art, n_bad, mode), (s, rows, truth, prep, per, extra) in results.items():
        a_rows, s_rows = ev.run_tables(rows, truth, per, names)
        d = os.path.join(root, 'corridas', f'{art}{n_bad}_{mode}',
                         f'tubo{int(size):03d}_{fk}MHz_semente{seed}')
        ev.write_csv(os.path.join(d, 'resultados_por_analito.csv'), ev.A_HEAD[4:], a_rows)
        ev.write_csv(os.path.join(d, 'concentracoes_por_amostra.csv'), ev.S_HEAD[4:], s_rows)
        summary.append([size, fk, seed, art, n_bad, mode, s['found'], s['truth'], s['fp'],
                        extra['eligible'], f"{100 * s['median']:.1f}", f"{100 * s['p90']:.1f}",
                        f"{100 * extra['err_bad']:.1f}", f"{100 * extra['err_ok']:.1f}",
                        *[s['dots'][c][i] for c in (3, 2, 1) for i in (0, 1)]])
    ev.write_csv(os.path.join(root, 'resumo.csv'),
                 ['tubo', 'campo_MHz', 'semente', 'artefato', 'ruins_por_grupo', 'anotacao',
                  'achados', 'no_tubo', 'falsos', 'elegiveis_reais', 'erro_mediano_pct',
                  'erro_p90_pct', 'erro_amostras_ruins_pct', 'erro_amostras_boas_pct',
                  'r3', 'f3', 'r2', 'f2', 'r1', 'f1'], summary)
    with open(os.path.join(root, 'LEIA-ME.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# Amostras com artefato: anotação no traço médio × por amostra

Tubo GISSMO padrão (12 amostras, dois grupos de 6). Depois de montado o tubo,
algumas amostras de cada grupo (`ruins_por_grupo`: as amostras {BAD_ORDER}
na ordem) recebem um artefato:

- `phase`: erro de fase de ordem zero de {PHASE_DEG:g}° (absorção·cos φ + dispersão·sen φ);
- `shim`: shim ruim, cada linha convoluída com dois lorentzianos {SHIM_KERNEL}
  (peso, deslocamento Hz, meia largura Hz): linha larga com ombro; o ruído
  alisado é reposto ao σ da amostra;
- `both`: os dois.

`anotacao`: `mean` = como o app faz hoje (anotação no traço médio do estudo);
`per_sample` = cada amostra anotada sozinha nos seus picos e ápices, sinal
aceito quando ≥3 amostras de um grupo aceitam (a regra de presença da
detecção). A quantificação é a mesma nos dois.

| arquivo | conteúdo |
|---|---|
| `resumo.csv` | por corrida: achados, falsos, analitos reais elegíveis depois da anotação, erro mediano/p90, erro nas amostras com artefato e nas limpas, reais/falsos por bolinha |
| `corridas/<artefato><n>_<anotacao>/tubo…/resultados_por_analito.csv` | cada analito: verdade, medido, erro, bolinhas |
| `corridas/…/concentracoes_por_amostra.csv` | analito × amostra: verdade, medido, erro |

    python -m validation.artifact_bench --export-dir "<pasta da validação>"

Esta pasta é apagada e refeita a cada exportação.
""")
    print(f'{len(results)} runs exported to {root}')


def real_run(study, mode):
    """``(samples_info, rows)`` of a real study (``real_samples``) annotated
    by ``mode``, kept until the code it depends on changes."""
    from validation import real_samples as rs
    path = os.path.join(CACHE_DIR, f'artifact_real_{study}_{mode}_'
                                   f'{_key(study, 0, 0, "real", 0, mode)}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        si, rows, _setup = rs.analyse(study, annotation=mode, log=lambda *a: None)
    with open(path, 'wb') as f:
        pickle.dump((si, rows), f, protocol=pickle.HIGHEST_PROTOCOL)
    return si, rows


def real_summary(si, rows):
    """What a real study's table holds, with no truth to score it on:
    analytes and dots, cells with a number, the median CV between
    replicates, and how far each analyst-flagged sample ('(bad)' in its
    name) stands from the rest of its group (median |log2| over analytes)."""
    incl = [i for i, s in enumerate(si) if str(s.get('type', 'Sample')).upper() != 'BLK'
            and not s.get('excluded')]
    group = {i: si[i].get('label') for i in incl}
    flagged = [i for i in incl if 'bad' in str(si[i].get('name', '')).lower()]
    q = [r for r in rows if not r.get('_not_quantified') and not r.get('_is_is')]
    dots = {c: sum(int(r.get('_confidence') or 0) == c for r in q) for c in (3, 2, 1)}
    cells, cvs, dev = 0, [], {i: [] for i in flagged}
    for r in q:
        v = list(r.get('_concs_per_sample') or [])
        val = {i: float(v[i]) for i in incl if i < len(v) and v[i] is not None
               and np.isfinite(v[i]) and v[i] > 0}
        cells += len(val)
        for g in set(group.values()):
            xs = [val[i] for i in incl if group[i] == g and i in val]
            if len(xs) >= 3:
                cvs.append(float(np.std(xs, ddof=1) / np.mean(xs)))
        for i in flagged:
            rest = [val[j] for j in incl if group[j] == group[i] and j != i and j in val]
            if i in val and rest:
                dev[i].append(abs(np.log2(val[i] / float(np.median(rest)))))
    return {'n': len(q), 'names': {str(r.get('metabolite')) for r in q},
            'dots': dots, 'cells': cells, 'n_cells': len(q) * len(incl),
            'cv': float(np.median(cvs)) if cvs else float('nan'),
            'flagged': {si[i].get('name'): (float(np.median(d)) if d else float('nan'), len(d))
                        for i, d in dev.items()}}


def real_compare(study, modes):
    out = {m: real_summary(*real_run(study, m)) for m in modes}
    print(f'\n=== {study} ===')
    print(f"{'':12s} {'analitos':>8s} {'***':>4s} {'**':>4s} {'*':>4s} "
          f"{'celulas':>12s} {'CV med':>7s}")
    for m, s in out.items():
        print(f"{m:12s} {s['n']:8d} {s['dots'][3]:4d} {s['dots'][2]:4d} {s['dots'][1]:4d} "
              f"{s['cells']:5d}/{s['n_cells']:<6d} {100 * s['cv']:6.0f}%")
        for name, (d, n) in s['flagged'].items():
            print(f"{'':12s} {name}: |log2| mediano ao grupo {d:.2f} ({n} analitos)")
    if len(modes) == 2:
        a, b = (out[m]['names'] for m in modes)
        print(f'so em {modes[0]}: {sorted(a - b)}')
        print(f'so em {modes[1]}: {sorted(b - a)}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--real', nargs='+', default=None,
                    help='real studies of real_samples (mice, er, liver, rawmg) '
                         'instead of the bench')
    ap.add_argument('--sizes', nargs='+', default=['18'])
    ap.add_argument('--fields', type=int, nargs='+', default=[600])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--artifact', nargs='+', default=['none', 'phase', 'shim'],
                    choices=('none', 'phase', 'shim', 'both'))
    ap.add_argument('--n-bad', type=int, nargs='+', default=[1],
                    help='spoiled samples per group (1-3)')
    ap.add_argument('--modes', nargs='+', default=['mean', 'per_sample'],
                    choices=('mean', 'per_sample'))
    ap.add_argument('--export-dir', default=None)
    a = ap.parse_args()
    if a.real:
        for study in a.real:
            real_compare(study, a.modes)
        return
    from validation import export_validation as ev
    results, lines = {}, []
    for size in a.sizes:
        for fk in a.fields:
            for seed in a.seeds:
                per = ev.sample_truth(fk, seed, int(size), 'padrao')
                for art in a.artifact:
                    for n_bad in ([0] if art == 'none' else a.n_bad):
                        bad = set(bad_samples(n_bad))
                        for mode in a.modes:
                            s, rows, truth, prep = run(size, gb.FIELDS[fk], seed,
                                                       art, n_bad, mode)
                            eb, eo = bad_sample_error(rows, truth, per, bad)
                            el = len(eligible_real(prep, truth))
                            results[size, fk, seed, art, n_bad, mode] = (
                                s, rows, truth, prep, per,
                                {'eligible': el, 'err_bad': eb, 'err_ok': eo})
                            lab = f'{size} s{seed} {art}{n_bad} {mode}'
                            lines.append((lab, s))
                            print(f'{lab}: elegiveis reais {el}/{len(truth)}, '
                                  f'erro amostras ruins {100 * eb:.0f}% / '
                                  f'limpas {100 * eo:.0f}%', flush=True)
    print()
    tables(lines)
    if a.export_dir:
        export(a.export_dir, results)


if __name__ == '__main__':
    main()
