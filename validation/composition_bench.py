# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""How much the method's figures move when the analytes in the tube change,
and how the number of signals grows with the number of analytes.

Tubes of 18, 50, 100 and 150 analytes (75 when there is time), each in
three replicates whose compositions are drawn at random and independently
(``gissmo_bench.random_tube``: any of the 234 compounds GISSMO simulates,
0.05-1 mM log-uniform). Every replicate is built, detected, annotated and
quantified from scratch with the application's defaults
(``validation.bench_runs``), at 600 MHz.

    python -m validation.composition_bench overnight          # runs, then the report
    python -m validation.composition_bench run --sizes 200    # more sizes
    python -m validation.composition_bench report             # every size run

The report (``validation/out/composition/<label>_report.md``):

* per tube size, the mean and the standard deviation over the three
  compositions of what the quality table reports -- the spread is the
  method's precision against a change of analytes;
* the linearity of the detected signals (and of the signals the final
  model explains) against the number of analytes: least squares, the
  lack-of-fit F test against the replicates' pure error, and the inverse
  prediction of the number of analytes from a spectrum's signal count with
  its 95% interval (Miller JN, Miller JC. Statistics and Chemometrics for
  Analytical Chemistry, 6th ed. Pearson, 2010, sections 5.6 and 5.12;
  Draper NR, Smith H. Applied Regression Analysis, 3rd ed. Wiley, 1998,
  section 2.1).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.bench_runs import OUT_DEFAULT, _path
from validation.replicates import mean_sd, reported, summarize
from validation.validation_folder import subfolder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'composition')
FIELD = 600
SIZES = (150, 100, 50, 18)      # longest first: the slowest runs start first
LATE_SIZE = 75
SEEDS = (100, 101, 102)
LABEL = 'comp'


@contextlib.contextmanager
def keep_awake():
    """Keep Windows from sleeping while the runs go (SetThreadExecutionState,
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released on exit. Elsewhere, nothing."""
    k32 = None
    if sys.platform == 'win32':
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetThreadExecutionState(0x80000000 | 0x00000001)
    try:
        yield
    finally:
        if k32 is not None:
            k32.SetThreadExecutionState(0x80000000)


def run_sizes(sizes, *, label=LABEL, seeds=SEEDS, jobs=8):
    from validation import bench_runs
    ns = argparse.Namespace(
        out=OUT_DEFAULT, label=label, tubes=[f'r{n}' for n in sizes],
        shifts=[0.0], seeds=list(seeds), field=FIELD, jobs=jobs, legacy=False,
        sample_shift=0.0, ref_shift=0.0, align_first=None, mstar_min_good=None,
        set=[])
    bench_runs.run(ns)


def run_metrics(rows, truth, n_peaks, coverage) -> dict:
    q = [r for r in rows if not r.get('_not_quantified') and not r.get('_is_is')]
    real = [r for r in q if r['hmdb_id'] in truth]
    errs = []
    for r in real:
        for g, vals in (r.get('_concs_by_group') or {}).items():
            want = truth[r['hmdb_id']].get(g)
            if want and vals:
                errs.append(abs(float(np.median(vals)) / float(want) - 1.0))
    e = np.array(errs) if errs else np.array([np.nan])

    def level(k):
        rr = [r for r in q if int(r.get('_confidence') or 0) >= k]
        t = sum(r['hmdb_id'] in truth for r in rr)
        return t, len(rr) - t
    two, three = level(2), level(3)
    cov = coverage or {}
    return {
        'analytes': len(truth), 'peaks': int(n_peaks),
        'explained': int(cov.get('explained', 0)),
        'reported': len(q), 'found': len(real), 'false': len(q) - len(real),
        'recall': len(real) / len(truth),
        'prec2': two[0] / sum(two) if sum(two) else np.nan,
        'prec3': three[0] / sum(three) if sum(three) else np.nan,
        'real2': two[0], 'false2': two[1], 'real3': three[0], 'false3': three[1],
        'err_mean': float(np.nanmean(e)), 'err_median': float(np.nanmedian(e)),
        'err_p90': float(np.nanpercentile(e, 90)),
    }


def sizes_run(label=LABEL) -> list:
    """Every tube size with at least one replicate of ``label``."""
    import glob
    import re
    pat = re.compile(rf'^{re.escape(label)}_r(\d+)_{FIELD}_0_\d+\.pkl$')
    return sorted({int(m.group(1)) for f in glob.glob(os.path.join(OUT_DEFAULT, f'{label}_r*'))
                   if (m := pat.match(os.path.basename(f)))})


def collect(sizes, *, label=LABEL, seeds=SEEDS) -> dict:
    out = {}
    for n in sizes:
        for s in seeds:
            p = _path(OUT_DEFAULT, label, f'r{n}', 0.0, s, FIELD)
            if not os.path.exists(p):
                continue
            with open(p, 'rb') as f:
                d = pickle.load(f)
            npk = d.get('n_peaks')
            if npk is None:
                npk = (d.get('coverage') or {}).get('peaks', 0)
            out.setdefault(n, []).append(run_metrics(d['rows'], d['truth'], npk,
                                                     d.get('coverage')))
    return out


def linear_fit(x, y) -> dict:
    """Least squares y = a + b x with the lack-of-fit F test against the
    replicates' pure error and what inverse prediction needs."""
    from scipy import stats
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    ss_res = float(res @ res)
    syx = np.sqrt(ss_res / (n - 2))
    r2 = 1.0 - ss_res / float(((y - y.mean()) ** 2).sum())
    levels = np.unique(x)
    ss_pe = float(sum(((y[x == v] - y[x == v].mean()) ** 2).sum() for v in levels))
    k = levels.size
    lof = None
    if n > k > 2 and ss_pe > 0:
        f = ((ss_res - ss_pe) / (k - 2)) / (ss_pe / (n - k))
        lof = (float(f), float(stats.f.sf(f, k - 2, n - k)))
    return {'a': float(a), 'b': float(b), 'r2': float(r2), 'syx': float(syx),
            'n': n, 'xbar': float(x.mean()), 'ybar': float(y.mean()),
            'sxx': float(((x - x.mean()) ** 2).sum()), 'lof': lof,
            't': float(stats.t.ppf(0.975, n - 2))}


def linear_range(x, y, alpha=0.05):
    """``(top level, fit)``: the straight line over the lowest levels,
    dropping the highest while the lack-of-fit test rejects it (a detector
    that saturates bends the curve at the top)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    levels = sorted(set(x))
    while True:
        top = levels[-1]
        m = x <= top
        f = linear_fit(x[m], y[m])
        if f['lof'] is None or f['lof'][1] >= alpha or len(levels) <= 3:
            return int(top), f
        levels = levels[:-1]


def saturation_fit(x, y) -> dict:
    """Detected peaks against analytes as a line, a parabola and a curve
    with a ceiling, y = ymax (1 - exp(-n / tau)), compared by AIC (least
    squares form, n ln(SS/n) + 2k; Burnham KP, Anderson DR. Model Selection
    and Multimodel Inference, 2nd ed. Springer, 2002)."""
    from scipy.optimize import curve_fit
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size

    def aic(ss, k):
        return n * np.log(ss / n) + 2 * k
    out = {}
    for deg, name in ((1, 'reta'), (2, 'parábola')):
        c = np.polyfit(x, y, deg)
        r = y - np.polyval(c, x)
        out[name] = {'aic': aic(float(r @ r), deg + 1), 'sd': float(np.sqrt(r @ r / (n - deg - 1)))}
    def sat(v, ymax, tau):
        return ymax * (1.0 - np.exp(-v / tau))
    try:
        (ymax, tau), cov = curve_fit(sat, x, y, p0=(1.5 * y.max(), x.max()), maxfev=20000)
        r = y - sat(x, ymax, tau)
        se = np.sqrt(np.diag(cov))
        out['teto'] = {'aic': aic(float(r @ r), 2), 'sd': float(np.sqrt(r @ r / (n - 2))),
                       'ymax': float(ymax), 'ymax_se': float(se[0]),
                       'tau': float(tau), 'tau_se': float(se[1])}
    except RuntimeError:
        pass
    return out


def inverse(fit, y0, m=1):
    """``(x0, half-width of the 95% interval)`` of the number of analytes
    a spectrum with ``y0`` signals holds (Miller & Miller, eq. 5.9)."""
    b = fit['b']
    x0 = (y0 - fit['a']) / b
    s = (fit['syx'] / abs(b)) * np.sqrt(1.0 / m + 1.0 / fit['n']
                                       + (y0 - fit['ybar']) ** 2 / (b * b * fit['sxx']))
    return x0, fit['t'] * s


def _ms(vals, pct=False, digits=1):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not v.size:
        return '—'
    k = 100.0 if pct else 1.0
    if v.size == 1:
        return f'{k * v[0]:.{digits}f}'
    return f'{k * v.mean():.{digits}f} ± {k * v.std(ddof=1):.{digits}f}'


def report(sizes, *, label=LABEL, seeds=SEEDS, probe_peaks=(277,)) -> str:
    data = collect(sizes, label=label, seeds=seeds)
    order = sorted(data)
    L = [f'# Variação de analitos — {FIELD} MHz, réplicas {list(seeds)}',
         f'gerado em {datetime.datetime.now().isoformat(timespec="minutes")}; '
         'média ± desvio-padrão entre as composições', '']
    L += ['## 1. Identificação', '',
          '| analitos | réplicas | picos detectados | encontrados (reais) | recuperação | '
          'falsos | ≥●● precisão | ●●● precisão |', '|---|---|---|---|---|---|---|---|']
    for n in order:
        R = data[n]
        L.append(f'| {n} | {len(R)} | {_ms([r["peaks"] for r in R], digits=0)} | '
                 f'{_ms([r["found"] for r in R])} | {_ms([r["recall"] for r in R], True)}% | '
                 f'{_ms([r["false"] for r in R])} | {_ms([r["prec2"] for r in R], True)}% | '
                 f'{_ms([r["prec3"] for r in R], True)}% |')
    L += ['', '## 2. Quantificação (erro |medido/verdadeiro − 1| por analito real e grupo)', '',
          '| analitos | erro médio | erro mediano | p90 | CV do erro mediano entre composições |',
          '|---|---|---|---|---|']
    for n in order:
        R = data[n]
        med = np.array([r['err_median'] for r in R])
        cv = (f'{100 * med.std(ddof=1) / med.mean():.0f}%'
              if med.size > 1 and med.mean() > 0 else '—')
        L.append(f'| {n} | {_ms([r["err_mean"] for r in R], True)}% | '
                 f'{_ms([r["err_median"] for r in R], True)}% | '
                 f'{_ms([r["err_p90"] for r in R], True)}% | {cv} |')
    xs = [r['analytes'] for n in order for r in data[n]]
    L += ['', '## 3. Linearidade: sinais × analitos no tubo', '',
          '| y | inclinação (sinais/analito) | intercepto | R² | s(y/x) | falta de ajuste F (p) |',
          '|---|---|---|---|---|---|']
    fits = {}
    for key, lab in (('peaks', 'picos detectados'), ('explained', 'picos explicados pelo modelo')):
        ys = [r[key] for n in order for r in data[n]]
        if len(set(xs)) < 2:
            continue
        f = linear_fit(xs, ys)
        fits[key] = f
        lof = f'{f["lof"][0]:.2f} ({f["lof"][1]:.3f})' if f['lof'] else '—'
        L.append(f'| {lab} | {f["b"]:.2f} | {f["a"]:.1f} | {f["r2"]:.4f} | '
                 f'{f["syx"]:.1f} | {lof} |')
    if 'peaks' in fits:
        ys = [r['peaks'] for n in order for r in data[n]]
        top, f = linear_range(xs, ys)
        L += ['', f'### Faixa linear: até {top} analitos', '',
              f'Níveis do topo retirados enquanto a falta de ajuste é significativa '
              f'(p < 0,05): picos = {f["a"]:.1f} + {f["b"]:.2f} × analitos, '
              f'R² {f["r2"]:.4f}, s(y/x) {f["syx"]:.1f}'
              + (f', falta de ajuste p = {f["lof"][1]:.3f}' if f['lof'] else '') + '.',
              '', '### Previsão inversa: analitos a partir dos picos detectados (IC 95%)', '',
              '| picos detectados | analitos previstos |', '|---|---|']
        for y0 in sorted(set([int(round(f['a'] + f['b'] * n)) for n in order if n <= top]
                             + [int(v) for v in probe_peaks])):
            x0, h = inverse(f, y0)
            L.append(f'| {y0} | {x0:.0f} ± {h:.0f} |')
        sat = saturation_fit(xs, ys)
        best = min(sat, key=lambda k: sat[k]['aic'])
        L += ['', '### Saturação: reta × parábola × curva com teto (AIC, menor é melhor)', '',
              '| modelo | AIC | s(resíduo) | teto de picos |', '|---|---|---|---|']
        for k, v in sat.items():
            teto = (f'{v["ymax"]:.0f} ± {v["ymax_se"]:.0f} (τ = {v["tau"]:.0f} ± {v["tau_se"]:.0f} analitos)'
                    if 'ymax' in v else '—')
            L.append(f'| {k}{" ←" if k == best else ""} | {v["aic"]:.1f} | {v["sd"]:.1f} | {teto} |')
        L += ['', 'Pontos (analitos, picos detectados, picos explicados): '
              + '; '.join(f'({r["analytes"]}, {r["peaks"]}, {r["explained"]})'
                          for n in order for r in data[n])]
    text = '\n'.join(L) + '\n'
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{label}_report.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    with open(os.path.join(OUT, f'{label}_runs.csv'), 'w', encoding='utf-8') as fh:
        keys = list(next(iter(data.values()))[0].keys()) if data else []
        fh.write(','.join(keys) + '\n')
        for n in order:
            for r in data[n]:
                fh.write(','.join(str(r[k]) for k in keys) + '\n')
    return text


BANDS_MM = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.0))


def _run_file(label, n, seed):
    return _path(OUT_DEFAULT, label, f'r{n}', 0.0, seed, FIELD)


def peak_points(runs) -> list:
    """``[{analytes, seed, peaks, runs}]`` of every (label, seeds) in ``runs``
    whose run exists: detection alone sets the count, so runs of engines that
    differ only after detection share one curve."""
    out = []
    for label, seeds, tag in runs:
        for n in sizes_run(label):
            for s in seeds:
                p = _run_file(label, n, s)
                if not os.path.exists(p):
                    continue
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                npk = d.get('n_peaks')
                if npk is None:
                    npk = (d.get('coverage') or {}).get('peaks', 0)
                out.append({'analytes': n, 'seed': s, 'peaks': int(npk), 'runs': tag})
    return out


def peak_curves(points, step=2):
    """``(params, curve)``: the straight line over its linear range and the
    curve with a ceiling over every point, as rows a chart draws."""
    x = [p['analytes'] for p in points]
    y = [p['peaks'] for p in points]
    top, lin = linear_range(x, y)
    sat = saturation_fit(x, y)
    params = [{'model': 'linear', 'range_analytes': f'{min(x)}-{top}', 'a': round(lin['a'], 2),
               'b': round(lin['b'], 3), 'r2': round(lin['r2'], 4), 'syx': round(lin['syx'], 1),
               'lack_of_fit_p': round(lin['lof'][1], 3) if lin['lof'] else '',
               'ymax': '', 'ymax_se': '', 'tau': '', 'tau_se': '', 'aic': ''}]
    if 'teto' in sat:
        t = sat['teto']
        params.append({'model': 'ceiling', 'range_analytes': f'{min(x)}-{max(x)}', 'a': '', 'b': '',
                       'r2': '', 'syx': round(t['sd'], 1), 'lack_of_fit_p': '',
                       'ymax': round(t['ymax'], 1), 'ymax_se': round(t['ymax_se'], 1),
                       'tau': round(t['tau'], 1), 'tau_se': round(t['tau_se'], 1),
                       'aic': round(t['aic'], 1)})
    curve = []
    for v in range(0, max(x) + step, step):
        row = {'analytes': v, 'linear': round(lin['a'] + lin['b'] * v, 1) if v <= top else ''}
        if 'teto' in sat:
            row['ceiling'] = round(sat['teto']['ymax'] * (1 - np.exp(-v / sat['teto']['tau'])), 1)
        curve.append(row)
    inv = []
    for y0 in range(100, int(max(y)) + 25, 25):
        x0, h = inverse(lin, y0)
        r = {'peaks': y0, 'analytes_linear': round(x0, 1) if x0 <= top else '',
             'ci95_linear': round(h, 1) if x0 <= top else ''}
        if 'teto' in sat and y0 < sat['teto']['ymax']:
            r['analytes_ceiling'] = round(-sat['teto']['tau'] * np.log(1 - y0 / sat['teto']['ymax']), 1)
        else:
            r['analytes_ceiling'] = ''
        inv.append(r)
    return params, curve, inv


def band_figures(rows, truth) -> dict:
    """One run, per band of the true amount (group A): analytes in the band,
    recovery, and the median and p90 error of the found ones (|measured /
    true - 1| per group, median over samples)."""
    rep = {r['hmdb_id']: r for r in reported(rows)}
    acc = {b: [0, 0, []] for b in BANDS_MM}
    for mid, t in truth.items():
        c = float(t.get('A') or 0.0)
        band = next((b for b in BANDS_MM if b[0] <= c < b[1]
                     or (b == BANDS_MM[-1] and c == b[1])), None)
        if band is None:
            continue
        acc[band][0] += 1
        r = rep.get(mid)
        if r is None:
            continue
        acc[band][1] += 1
        for g, vals in (r.get('_concs_by_group') or {}).items():
            want = t.get(g)
            if want and vals:
                acc[band][2].append(abs(float(np.median(vals)) / float(want) - 1.0))
    return {b: {'analytes': k, 'recovery_pct': 100 * f / k,
                'error_median_pct': 100 * float(np.median(e)) if e else np.nan,
                'error_p90_pct': 100 * float(np.percentile(e, 90)) if e else np.nan}
            for b, (k, f, e) in acc.items() if k}


def band_rows(runs) -> list:
    """``runs`` (one ``band_figures`` per tube) as the mean per tube and the
    SD between tubes, band by band."""
    out = []
    for b in BANDS_MM:
        per = [r[b] for r in runs if b in r]
        if per:
            out.append({'band_mM': f'{b[0]:g}-{b[1]:g}',
                        **summarize(per, ('analytes', 'recovery_pct', 'error_median_pct', 'error_p90_pct'))})
    return out


def by_concentration(label, sizes, seeds) -> list:
    """Per tube size, ``band_rows`` of its tubes."""
    out = []
    for n in sizes:
        runs = []
        for s in seeds:
            p = _run_file(label, n, s)
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    d = pickle.load(f)
                runs.append(band_figures(d['rows'], d['truth']))
        out += [{'analytes_in_tube': n, **r} for r in band_rows(runs)]
    return out


# Earlier compositions of identical detection that join the peak curve, per field.
OLD_PEAK_RUNS = {600: (('comp', (100, 101, 102)),), 500: (('comp500', (100, 101, 102)),)}


def export(dest, *, label, sizes, seeds, old=None):
    """``dest``/02_Composicoes_600MHz (03_Composicoes_500MHz at 500), made anew: every run as a project the
    application opens (already detected, annotated and quantified), its
    truth per sample and its tables, the per-run summary, and the two
    curves' data -- detected peaks against analytes (with the earlier
    detection runs of ``old``) and recovery and error by true amount."""
    import csv
    import shutil
    import subprocess
    from app_paths import APP_VERSION
    from data import project_file
    from validation import export_validation as ev
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    from validation.export_bench_projects import project_state
    old = OLD_PEAK_RUNS.get(FIELD, ()) if old is None else old
    root = subfolder(dest, f'composition_{FIELD}')
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)

    def write(name, rows):
        with open(os.path.join(root, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    names = ev._sample_names()
    summary, analytes_all, samples_all = [], [], []
    for n in sizes:
        for s in seeds:
            p = _run_file(label, n, s)
            if not os.path.exists(p):
                continue
            with open(p, 'rb') as f:
                d = pickle.load(f)
            prep, truth = gb.prepared(gb.FIELDS[FIELD], s, f'r{n}')
            mb.MHZ = gb.FIELDS[FIELD]
            run = os.path.join(root, f'tubo{n:03d}_composicao{s}')
            os.makedirs(run)
            project_file.save(os.path.join(run, f'Composicao_{n}_s{s}.nmrproj'),
                              project_state(prep, d['rows'], gb.PPM, gb.FIELDS[FIELD]),
                              keep_backup=False)
            per = ev.sample_truth(FIELD, s, f'r{n}', 'padrao')
            ev.write_csv(os.path.join(run, 'verdade_por_amostra.csv'),
                         ['hmdb_id', 'nome', 'verdade_grupo_A_mM', 'verdade_grupo_B_mM']
                         + [f'{x}_mM' for x in names],
                         ([mid, t['name'], t['A'], t['B']] + [f'{v:.6g}' for v in per.get(mid, [])]
                          for mid, t in truth.items()))
            a_rows, s_rows = ev.run_tables(d['rows'], d['truth'], per, names)
            analytes_all += [[n, s, *r] for r in a_rows]
            samples_all += [[n, s, *r] for r in s_rows]
            ev.write_csv(os.path.join(run, 'resultados_por_analito.csv'), ev.A_HEAD[4:], a_rows)
            ev.write_csv(os.path.join(run, 'concentracoes_por_amostra.csv'), ev.S_HEAD[4:], s_rows)
            npk = d.get('n_peaks') or (d.get('coverage') or {}).get('peaks', 0)
            m = run_metrics(d['rows'], d['truth'], npk, d.get('coverage'))
            summary.append({'analytes_in_tube': n, 'composition': s, **{
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
    write('resumo_por_corrida.csv', summary)
    points = peak_points([(lab, sd, f'{lab} {min(sd)}-{max(sd)}') for lab, sd in old]
                         + [(label, seeds, f'{label} {min(seeds)}-{max(seeds)}')])
    write('picos_por_analitos.csv', points)
    params, curve, inv = peak_curves(points)
    write('curva_picos_parametros.csv', params)
    write('curva_picos_ajuste.csv', curve)
    write('previsao_analitos_por_picos.csv', inv)
    conc = by_concentration(label, sizes, seeds)
    write('recuperacao_por_concentracao.csv', conc)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    with open(os.path.join(root, 'versao.txt'), 'w', encoding='utf-8') as f:
        f.write(f'software: {APP_VERSION}\ncommit: {commit}\ncorridas: {len(summary)}\n')
    with open(os.path.join(root, 'LEIA-ME.md'), 'w', encoding='utf-8') as f:
        f.write(LEIAME.format(version=APP_VERSION, field=FIELD, sizes=', '.join(map(str, sizes)),
                              seeds=', '.join(map(str, seeds)), runs=len(summary),
                              seeds_cmd=' '.join(map(str, seeds)), sizes_cmd=' '.join(map(str, sizes)),
                              old=', '.join(f'{lab} (composições {", ".join(map(str, sd))})'
                                            for lab, sd in old)))
    workbook = os.path.join(root, f"Composicoes_{FIELD}MHz_nMRPro-H_v{APP_VERSION.split(' v')[1].split()[0]}.xlsx")
    write_workbook(workbook, summary=summary, points=points, params=params, curve=curve,
                   inv=inv, conc=conc, analytes=analytes_all, samples=samples_all,
                   about=[('software', APP_VERSION), ('commit', commit),
                          ('campo', f'{FIELD} MHz'), ('tubos (analitos)', ', '.join(map(str, sizes))),
                          ('composições (sementes)', ', '.join(map(str, seeds))),
                          ('corridas', len(summary)),
                          ('amostras por corrida', '12, dois grupos de 6, variação de 12% entre amostras'),
                          ('erro', '|medido / verdadeiro − 1|; por analito e grupo: mediana das amostras'),
                          ('pasta das corridas', 'tuboNNN_composicaoS/ (um .nmrproj por corrida)')])
    print(f'{len(summary)} runs exported to {root}; workbook {workbook}')


def write_workbook(path, *, summary, points, params, curve, inv, conc, analytes, samples, about):
    """Every table of the composition runs in one workbook, numbers as
    numbers: the per-size means and standard deviations, every run, both
    curves' data, recovery by concentration, and every analyte and sample
    of every run."""
    from validation.export_validation import A_HEAD, S_HEAD, _xlsx

    def sheet(title, rows):
        head = list(rows[0])
        return (title, head, [[r.get(h) for h in head] for r in rows])
    per_size = []
    for n in sorted({r['analytes_in_tube'] for r in summary}):
        R = [r for r in summary if r['analytes_in_tube'] == n]
        row = {'analitos_no_tubo': n, 'composicoes': len(R)}
        levels = [{'reais_3': r['real3'], 'falsos_3': r['false3'],
                   'reais_2': r['real2'] - r['real3'], 'falsos_2': r['false2'] - r['false3'],
                   'reais_1': r['found'] - r['real2'], 'falsos_1': r['false'] - r['false2']}
                  for r in R]
        for key, lab, k in (('peaks', 'picos', 1), ('found', 'encontrados', 1),
                            ('false', 'falsos', 1), ('recall', 'recuperacao_pct', 100),
                            ('prec2', 'precisao_2_bolinhas_pct', 100),
                            ('prec3', 'precisao_3_bolinhas_pct', 100),
                            ('err_median', 'erro_mediano_pct', 100),
                            ('err_p90', 'erro_p90_pct', 100)):
            row[f'{lab}_media'], row[f'{lab}_dp'] = mean_sd(
                [k * float(r[key]) for r in R if r.get(key) not in (None, '')], 2)
        for key in levels[0]:
            row[f'{key}_media'], row[f'{key}_dp'] = mean_sd([x[key] for x in levels], 2)
        per_size.append(row)
    run_head = ['analitos_no_tubo', 'composicao']
    _xlsx(path, [
        ('Leia-me', ['item', 'valor'], [list(a) for a in about]),
        sheet('Resumo_por_tamanho', per_size),
        sheet('Resumo_por_corrida', summary),
        sheet('Picos_x_analitos', points),
        sheet('Curva_parametros', params),
        sheet('Curva_ajuste', curve),
        sheet('Previsao_analitos', inv),
        sheet('Recuperacao_concentracao', conc),
        ('Por_analito', run_head + A_HEAD[4:], analytes),
        ('Por_amostra', run_head + S_HEAD[4:], samples),
    ])


LEIAME = """# Variação de composição — {version}

Tubos simulados com o GISSMO a {field} MHz, de {sizes} analitos, cada um em composições sorteadas de forma independente (sementes {seeds}: compostos ao acaso entre os 234 que o GISSMO simula, 0,05–1 mM log-uniforme, 12 amostras em dois grupos iguais, variação de 12% entre amostras). Cada corrida foi detectada, anotada e quantificada do zero com os padrões do software. Estas composições nunca foram usadas para ajustar o motor. {runs} corridas.

## Abrir no software

Cada pasta `tuboNNN_composicaoS/` tem `Composicao_NNN_sS.nmrproj`: abre já com detecção, anotação e quantificação feitas. `verdade_por_amostra.csv` diz o que está no tubo e em que concentração, amostra a amostra.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `tuboNNN_composicaoS/resultados_por_analito.csv` | cada analito presente e cada reportado: verdade, medido, erro, bolinhas, via, SNR, motivo sem número |
| `tuboNNN_composicaoS/concentracoes_por_amostra.csv` | analitos reportados amostra a amostra: verdade, medido, erro |
| `resumo_por_corrida.csv` | por corrida: picos detectados, achados, falsos, recuperação, precisão ●●/●●●, erro |
| `picos_por_analitos.csv` | picos detectados × analitos no tubo, destas corridas e das anteriores de detecção idêntica ({old}) |
| `curva_picos_parametros.csv` | reta na faixa linear (falta de ajuste) e curva com teto y = ymax(1 − e^(−n/τ)) |
| `curva_picos_ajuste.csv` | as duas curvas, ponto a ponto |
| `previsao_analitos_por_picos.csv` | analitos previstos a partir dos picos detectados (reta: IC 95%, Miller & Miller 2010 eq. 5.9; teto: n = −τ ln(1 − picos/ymax)) |
| `recuperacao_por_concentracao.csv` | por tamanho e faixa de concentração verdadeira, medido tubo a tubo: analitos na faixa, recuperação, erro mediano e p90, cada um como média por tubo (`_mean`) e desvio-padrão entre os tubos (`_sd`) |

## Como reproduzir

```
python -m validation.composition_bench run --label comp076 --field {field} --sizes {sizes_cmd} --seeds {seeds_cmd} --jobs 8
python -m validation.composition_bench export --label comp076 --field {field} --sizes {sizes_cmd} --seeds {seeds_cmd} --dest "<pasta da validação>"
```

Esta pasta é apagada e refeita a cada exportação.
"""


def main():
    global FIELD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('cmd', choices=('overnight', 'run', 'report', 'export'))
    ap.add_argument('--label', default=LABEL)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--field', type=int, default=FIELD, choices=(500, 600))
    ap.add_argument('--sizes', type=int, nargs='+', default=None,
                    help='tube sizes for run (default: the overnight set)')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS),
                    help='compositions drawn (random_tube seeds); 100-102 tuned ADR-085')
    ap.add_argument('--dest', help='export: the validation folder (writes <dest>/02_Composicoes_600MHz or 03_Composicoes_500MHz)')
    ap.add_argument('--deadline', default='06:00',
                    help='the tube of 75 runs only if the others end before this')
    a = ap.parse_args()
    FIELD = a.field
    sizes = list(a.sizes or SIZES)
    if a.cmd == 'export':
        export(a.dest, label=a.label, sizes=sizes, seeds=a.seeds)
        return
    if a.cmd in ('overnight', 'run'):
        with keep_awake():
            t0 = datetime.datetime.now()
            print(f'{t0:%H:%M} start: tubes {sizes} x seeds {a.seeds}', flush=True)
            run_sizes(sizes, label=a.label, seeds=a.seeds, jobs=a.jobs)
            print(report(sizes_run(a.label), label=a.label, seeds=a.seeds), flush=True)
            now = datetime.datetime.now()
            hh, mm = (int(v) for v in a.deadline.split(':'))
            deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if deadline < t0:
                deadline += datetime.timedelta(days=1)
            if a.cmd == 'overnight' and now < deadline:
                print(f'{now:%H:%M} before {a.deadline}: tube of {LATE_SIZE}', flush=True)
                run_sizes([LATE_SIZE], label=a.label, seeds=a.seeds,
                          jobs=min(a.jobs, len(a.seeds)))
            elif a.cmd == 'overnight':
                print(f'{now:%H:%M} past {a.deadline}: no tube of {LATE_SIZE}', flush=True)
            print(f'{datetime.datetime.now():%H:%M} done', flush=True)
    print(report(sizes_run(a.label), label=a.label, seeds=a.seeds))


if __name__ == '__main__':
    main()
