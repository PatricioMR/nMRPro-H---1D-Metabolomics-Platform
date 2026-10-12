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
        sample_shift=0.0, align_first=None, mstar_min_good=None, set=[])
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


def main():
    global FIELD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('cmd', choices=('overnight', 'run', 'report'))
    ap.add_argument('--label', default=LABEL)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--field', type=int, default=FIELD, choices=(500, 600))
    ap.add_argument('--sizes', type=int, nargs='+', default=None,
                    help='tube sizes for run (default: the overnight set)')
    ap.add_argument('--deadline', default='06:00',
                    help='the tube of 75 runs only if the others end before this')
    a = ap.parse_args()
    FIELD = a.field
    sizes = list(a.sizes or SIZES)
    if a.cmd in ('overnight', 'run'):
        with keep_awake():
            t0 = datetime.datetime.now()
            print(f'{t0:%H:%M} start: tubes {sizes} x seeds {list(SEEDS)}', flush=True)
            run_sizes(sizes, label=a.label, jobs=a.jobs)
            print(report(sizes_run(a.label), label=a.label), flush=True)
            now = datetime.datetime.now()
            hh, mm = (int(v) for v in a.deadline.split(':'))
            deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if deadline < t0:
                deadline += datetime.timedelta(days=1)
            if a.cmd == 'overnight' and now < deadline:
                print(f'{now:%H:%M} before {a.deadline}: tube of {LATE_SIZE}', flush=True)
                run_sizes([LATE_SIZE], label=a.label, jobs=min(a.jobs, len(SEEDS)))
            elif a.cmd == 'overnight':
                print(f'{now:%H:%M} past {a.deadline}: no tube of {LATE_SIZE}', flush=True)
            print(f'{datetime.datetime.now():%H:%M} done', flush=True)
    print(report(sizes_run(a.label), label=a.label))


if __name__ == '__main__':
    main()
