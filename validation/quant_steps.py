# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""What each group of quantification methods adds, step by step.

The quantification options are three groups (gui.stages.quantification_stage):
A, windows resolved by shapes (contested windows, shared windows, complex
multiplets m*); B, the joint fit of overlapping analytes (with candidate
columns and columns at the matched peak); C, the residual cascade. The base
is window integration with the neighbour-leak correction, always on. This
runs the base, then adds A, B and C in turn (the steps), and then takes the
complete engine with one control switched off at a time (amount consistency
between an analyte's signals, withdrawal of off-drift singlets, singlet
re-selection). Controls stay on during the steps.

Random compositions 200-204 at 600 MHz (independent of any tuning), tube of
100 analytes, plus a blank tube (DSS, methanol and noise only) per seed as a
negative control. Detection and annotation are cached and shared by every
variant; only quantification is re-run.

    python -m validation.quant_steps run --jobs 8
    python -m validation.quant_steps export --dest "<validation folder>"
    python -m validation.quant_steps figure --dest "<validation folder>"   (redraw only)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.ablation import _set_defaults
from validation.bench_runs import OUT_DEFAULT, _apply_sets, _below_normal, _errors
from validation.composition_bench import band_figures, band_rows as band_summary
from validation.replicates import mean_sd, reported, run_figures, summarize
from validation.validation_folder import subfolder

OUT = os.path.join(OUT_DEFAULT, 'quant_steps')
FIELD = 600
TUBES = ('r100', 'r0')
SEEDS = (200, 201, 202, 203, 204)

_A_OFF = {'split_shared': False, 'resolve_shared': False, 'quantify_shapes': False}
# (id, topic, label, module settings, analysis defaults)
VARIANTS = (
    ('base', 'steps', 'Base: window integration with neighbour correction', (),
     {**_A_OFF, 'joint_fit': False, 'cascade': False}),
    ('plus_a', 'steps', '+ A: windows resolved by shapes', (),
     {'joint_fit': False, 'cascade': False}),
    ('plus_b', 'steps', '+ B: joint fit of overlapping analytes', (), {'cascade': False}),
    ('full', 'steps', '+ C: residual cascade (complete engine)', (), {}),
    ('no_amount', 'controls', 'Complete engine without amount consistency',
     ('processing.quant_engine.AMOUNT_RATIO=1e12',), {}),
    ('no_offdrift', 'controls', 'Complete engine without off-drift singlet withdrawal',
     ('processing.quant_engine.OFF_DRIFT_SINGLET_WITHDRAWN=False',), {}),
    ('no_reselect', 'controls', 'Complete engine without singlet re-selection',
     ('gui.stages._quant_adapter.SINGLET_RESELECT=False',), {}),
)


def _path(vid, tube, seed):
    return os.path.join(OUT, f'{vid}_{tube}_{FIELD}_s{seed}.pkl')


def _prepare(job):
    tube, seed = job
    from validation import gissmo_bench as gb
    gb.prepared(gb.FIELDS[FIELD], seed, tube)
    return job


def _one(job):
    vid, tube, seed = job
    _vid, _topic, _label, sets, defaults = next(v for v in VARIANTS if v[0] == vid)
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    from validation.fresh_run import quantify_prepared
    _apply_sets(sets)
    _set_defaults(defaults)
    t0 = time.time()
    prep, truth = gb.prepared(gb.FIELDS[FIELD], seed, tube)
    mb.MHZ = gb.FIELDS[FIELD]
    with contextlib.redirect_stdout(io.StringIO()):
        rows = quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM,
                                 standard_nh=mb.STANDARD_NH,
                                 standard_conc=mb.STANDARD_MM, standard_name='DSS')
    path = _path(vid, tube, seed)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump({'rows': rows, 'truth': truth, 'n_peaks': len(prep['peaks'])}, f)
    os.replace(path + '.tmp', path)
    return job, time.time() - t0


def run(a):
    os.makedirs(OUT, exist_ok=True)
    threads = max(1, (os.cpu_count() or 1) // max(a.jobs, 1))
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = str(threads)
    # detection and annotation once per tube, before the variants share them
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_below_normal,
                             max_tasks_per_child=1) as ex:
        for f in as_completed([ex.submit(_prepare, (t, s)) for t in TUBES for s in SEEDS]):
            print('prepared', f.result(), flush=True)
    jobs = [(v[0], t, s) for t in TUBES for v in VARIANTS for s in SEEDS
            if not os.path.exists(_path(v[0], t, s))]
    t0 = time.time()
    # one job per process: a variant's settings must not outlive its job
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_below_normal,
                             max_tasks_per_child=1) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            (vid, t, s), secs = f.result()
            print(f'{i}/{len(jobs)} {vid} {t} seed {s} in {secs:.0f} s '
                  f'({time.time() - t0:.0f} s elapsed)', flush=True)


def run_row(vid, tube, seed):
    """One run's figures (validation.replicates.run_figures; error per
    analyte and group) and the dose-response of the found ones (log10
    measured against log10 true, per analyte and group)."""
    p = _path(vid, tube, seed)
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, 'rb'))
    truth = d['truth']
    f = run_figures(d['rows'], truth, _errors(d['rows'], truth))
    t_, m_ = [], []
    for x in reported(d['rows']):
        if x['hmdb_id'] not in truth:
            continue
        for g, vals in (x.get('_concs_by_group') or {}).items():
            want = truth[x['hmdb_id']].get(g)
            med = float(np.median(vals)) if vals else np.nan
            if want and np.isfinite(med) and med > 0:
                t_.append(np.log10(float(want)))
                m_.append(np.log10(med))
    slope = intercept = r2 = bias = np.nan
    if len(t_) > 2:
        slope, intercept = np.polyfit(t_, m_, 1)
        res = np.array(m_) - (intercept + slope * np.array(t_))
        r2 = 1 - float(res @ res) / float(((np.array(m_) - np.mean(m_)) ** 2).sum())
        bias = float(np.median(10 ** (np.array(m_) - np.array(t_)) - 1))
    r = lambda v, k=1: round(float(v), k) if np.isfinite(v) else ''
    return {'variant': vid, 'tube': tube, 'composition': seed, 'analytes_in_tube': f['in_tube'],
            **{k: (r(v) if isinstance(v, float) else v) for k, v in f.items() if k != 'in_tube'},
            'dose_slope': r(slope, 4), 'dose_intercept': r(intercept, 4), 'dose_r2': r(r2, 4),
            'dose_bias_pct': r(100 * bias)}


SUM_KEYS = ('found', 'false', 'recovery_pct', 'precision_pct', 'error_median_pct', 'error_p90_pct',
            'real3', 'false3', 'real2', 'false2', 'real1', 'false1',
            'dose_slope', 'dose_r2', 'dose_bias_pct')


def summary_rows(runs):
    """Per variant on the tube of 100: the mean per tube and the SD over the
    compositions (validation.replicates.summarize), each level's precision
    pooled over the tubes, and the false analytes of the blank tube."""
    out = []
    for vid, topic, label, _s, _d in VARIANTS:
        R = [r for r in runs if r['variant'] == vid and r['tube'] == 'r100']
        B = [r for r in runs if r['variant'] == vid and r['tube'] == 'r0']
        if not R:
            continue
        row = {'topic': topic, 'variant': vid, 'label': label, **summarize(R, SUM_KEYS, 2)}
        for lvl in (3, 2, 1):
            re_, fa = sum(r[f'real{lvl}'] for r in R), sum(r[f'false{lvl}'] for r in R)
            row[f'precision{lvl}_pooled_pct'] = round(100 * re_ / (re_ + fa), 1) if re_ + fa else ''
        row['blank_false_mean'], row['blank_false_sd'] = mean_sd([r['false'] for r in B], 2)
        out.append(row)
    return out


def band_rows(vid):
    """Recovery and error by band of the true amount on the tube of 100,
    tube by tube, as the mean per tube and the SD between tubes."""
    runs = []
    for s in SEEDS:
        p = _path(vid, 'r100', s)
        if os.path.exists(p):
            d = pickle.load(open(p, 'rb'))
            runs.append(band_figures(d['rows'], d['truth']))
    return [{'variant': vid, **r} for r in band_summary(runs)]


def level_share_rows(runs):
    """How the real and the false analytes of each variant spread over the
    confidence levels, tube by tube (share of the real ones at each level,
    share of the false ones), as the mean per tube and the SD between
    tubes; each level's precision pooled over the tubes."""
    out = []
    for vid, topic, label, _s, _d in VARIANTS:
        R = [r for r in runs if r['variant'] == vid and r['tube'] == 'r100']
        if not R:
            continue
        shares = []
        for r in R:
            s = {}
            for kind, total in (('real', r['found']), ('false', r['false'])):
                for lvl in (3, 2, 1):
                    s[f'{kind}_at_{lvl}dots_pct'] = 100 * r[f'{kind}{lvl}'] / total if total else ''
            shares.append(s)
        row = {'topic': topic, 'variant': vid, 'label': label,
               **summarize(shares, list(shares[0]))}
        for lvl in (3, 2, 1):
            re_, fa = sum(r[f'real{lvl}'] for r in R), sum(r[f'false{lvl}'] for r in R)
            row[f'precision_{lvl}dots_pooled_pct'] = round(100 * re_ / (re_ + fa), 1) if re_ + fa else ''
        out.append(row)
    return out


STEP_PAIRS = (('base', 'plus_a'), ('plus_a', 'plus_b'), ('plus_b', 'full'))


def _group_errors(x, t):
    return [abs(float(np.median(v)) / float(t[g]) - 1.0)
            for g, v in (x.get('_concs_by_group') or {}).items() if t.get(g) and v]


def paired_rows():
    """Each step against the one before on the tube of 100, composition by
    composition: the real analytes both report (their error before and
    after: the step re-measures them), the ones the step drops (and the
    error they had before) and the ones it adds; false analytes dropped and
    added. Each figure is the mean per tube and the SD between tubes."""
    out = []
    for before, after in STEP_PAIRS:
        per = []
        for s in SEEDS:
            pb, pa = _path(before, 'r100', s), _path(after, 'r100', s)
            if not (os.path.exists(pb) and os.path.exists(pa)):
                continue
            db, da = pickle.load(open(pb, 'rb')), pickle.load(open(pa, 'rb'))
            truth = db['truth']
            rb = {x['hmdb_id']: x for x in reported(db['rows'])}
            ra = {x['hmdb_id']: x for x in reported(da['rows'])}
            n = {k: 0 for k in ('kept', 'dropped', 'added', 'false_kept', 'false_dropped', 'false_added')}
            e_b, e_a, e_drop, e_add = [], [], [], []
            for mid in rb.keys() | ra.keys():
                real = mid in truth
                if mid in rb and mid in ra:
                    n['kept' if real else 'false_kept'] += 1
                    if real:
                        e_b += _group_errors(rb[mid], truth[mid])
                        e_a += _group_errors(ra[mid], truth[mid])
                elif mid in rb:
                    n['dropped' if real else 'false_dropped'] += 1
                    if real:
                        e_drop += _group_errors(rb[mid], truth[mid])
                else:
                    n['added' if real else 'false_added'] += 1
                    if real:
                        e_add += _group_errors(ra[mid], truth[mid])
            q = lambda e, p: 100 * float(np.percentile(e, p)) if e else ''
            per.append({**n, 'kept_error_median_before_pct': q(e_b, 50), 'kept_error_median_after_pct': q(e_a, 50),
                        'kept_error_p90_before_pct': q(e_b, 90), 'kept_error_p90_after_pct': q(e_a, 90),
                        'dropped_error_median_before_pct': q(e_drop, 50), 'added_error_median_pct': q(e_add, 50)})
        if per:
            out.append({'step': f'{before} -> {after}', **summarize(per, list(per[0]))})
    return out


def export(a):
    """``dest``/05_Etapas_quantificacao, made anew: every run's tables and
    project-independent results, the per-run and per-variant summaries
    (mean per tube and SD between tubes), recovery and error by
    concentration, each step paired with the one before, the share of each
    confidence level, the figure and one workbook."""
    from app_paths import APP_VERSION
    from validation import export_validation as ev
    from validation.export_validation import _xlsx
    root = subfolder(a.dest, 'steps')
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    names = ev._sample_names()
    runs, truths = [], {}
    for vid, topic, label, _s, _d in VARIANTS:
        for tube in TUBES:
            for s in SEEDS:
                r = run_row(vid, tube, s)
                if r is None:
                    continue
                runs.append({'topic': topic, 'label': label, **r})
                d = pickle.load(open(_path(vid, tube, s), 'rb'))
                if (tube, s) not in truths:
                    truths[tube, s] = ev.sample_truth(FIELD, s, tube, 'padrao')
                a_rows, s_rows = ev.run_tables(d['rows'], d['truth'], truths[tube, s], names)
                folder = os.path.join(root, 'corridas', vid, f'{tube}_composicao{s}')
                ev.write_csv(os.path.join(folder, 'resultados_por_analito.csv'), ev.A_HEAD[4:], a_rows)
                ev.write_csv(os.path.join(folder, 'concentracoes_por_amostra.csv'), ev.S_HEAD[4:], s_rows)
    summary = summary_rows(runs)
    bands = [row for v in VARIANTS for row in band_rows(v[0])]
    paired = paired_rows()
    shares = level_share_rows(runs)

    def write(name, rows):
        with open(os.path.join(root, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write('resumo_por_variante.csv', summary)
    write('resumo_por_corrida.csv', runs)
    write('recuperacao_por_concentracao.csv', bands)
    write('pareado_entre_etapas.csv', paired)
    write('percentual_por_nivel.csv', shares)
    sheet = lambda title, rows: (title, list(rows[0]), [list(r.values()) for r in rows])
    version = APP_VERSION.split(' v')[1].split()[0]
    _xlsx(os.path.join(root, f'Etapas_quantificacao_nMRPro-H_v{version}.xlsx'), [
        ('Leia-me', ['item', 'valor'], [
            ['software', APP_VERSION], ['campo', f'{FIELD} MHz'],
            ['tubos', '100 analitos (composições 200-204) + branco (DSS, metanol, ruído)'],
            ['passos', 'base -> + A (janelas por formas) -> + B (ajuste conjunto) -> + C (cascata)'],
            ['controles', 'ligados nos passos; tópico próprio: cada um desligado no motor completo'],
            ['números', 'cada tubo medido sozinho; _mean = média por tubo, _sd = desvio-padrão entre os tubos; nunca somados'],
            ['dose-resposta', 'reta log10(medido) x log10(verdadeiro) por analito e grupo; viés = mediana de medido/verdadeiro - 1'],
            ['pareado', 'cada etapa contra a anterior, tubo a tubo: erro dos mesmos analitos antes/depois, os que a etapa retira e os que acrescenta'],
            ['percentual por nível', 'dos verdadeiros (e dos falsos) de cada tubo, quantos % em ●●●, ●● e ●; precisão de cada nível agrupada nos tubos']]),
        sheet('Resumo_por_variante', summary), sheet('Resumo_por_corrida', runs),
        sheet('Recuperacao_concentracao', bands), sheet('Pareado_entre_etapas', paired),
        sheet('Percentual_por_nivel', shares)])
    step_figure(summary, os.path.join(root, f'Etapas_quantificacao_nMRPro-H_v{version}.png'))
    print(f'{len(runs)} runs exported to {root}')


# Ordinal ramps (dark -> light = three dots -> one dot): blue for real, orange for false
REAL_RAMP = ('#1c5cab', '#3987e5', '#86b6ef')
FALSE_RAMP = ('#b5461a', '#eb6834', '#f4a885')
STEP_TICKS = {'base': 'Base\nwindows', 'plus_a': '+ A\nshapes', 'plus_b': '+ B\njoint fit', 'full': '+ C\ncascade'}


def step_figure(summary, path):
    """The steps as one figure: real analytes per tube rise from the axis and
    false ones hang below it, each bar split by confidence level with the dots
    and count written in the segment, and the median and p90 quantification
    error under each step. The y axis is a signed square root, for reading:
    it opens the first units, where the thin segments are, without cutting the
    axis (labels carry the numbers). Same numbers as the summary table (means
    per tube)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [next(r for r in summary if r['variant'] == v) for v in STEP_TICKS
            if any(r['variant'] == v for r in summary)]
    g = lambda r, k: float(r[k])
    ink, width, dpi = '#1a1a19', 0.7, 300
    fwd = lambda v: np.sign(v) * np.sqrt(np.abs(v))
    inv = lambda u: np.sign(u) * np.square(u)
    fig, ax = plt.subplots(figsize=(8.5, 9.5), dpi=dpi)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.98, bottom=0.2)
    ax.set_yscale('function', functions=(fwd, inv))
    x = np.arange(len(rows))
    stacks = {}
    for kind, ramp, sign in (('real', REAL_RAMP, 1), ('false', FALSE_RAMP, -1)):
        vals = np.array([[g(r, f'{kind}{lvl}_mean') for r in rows] for lvl in (3, 2, 1)])
        sds = [[r[f'{kind}{lvl}_sd'] for r in rows] for lvl in (3, 2, 1)]
        cum = np.vstack([np.zeros(len(rows)), np.cumsum(vals, axis=0)])
        for k in range(3):
            ax.bar(x, sign * vals[k], bottom=sign * cum[k], width=width, color=ramp[k], edgecolor=ink, linewidth=0.8)
        stacks[kind] = (vals, cum, sign, sds)
    top_r = stacks['real'][1][-1].max() * 1.2
    top_f = stacks['false'][1][-1].max() * 1.35
    ax.set_ylim(-top_f, top_r)
    ticks = [t for t in (-20, -10, -5, -1, 0, 1, 5, 10, 20, 40, 60, 80) if -top_f <= t <= top_r]
    ax.set_yticks(ticks, [f'{abs(t):g}' for t in ticks])
    ax.axhline(0, color=ink, linewidth=1.0)
    ax.grid(axis='y', color='#e4e3dd', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.tick_params(labelsize=13)
    ax.set_xticks(x, [STEP_TICKS[r['variant']] for r in rows])
    side_label = lambda text, lo, hi: ax.annotate(text, (-0.1, inv((fwd(lo) + fwd(hi)) / 2)), xycoords=('axes fraction', 'data'),
                                                  rotation=90, ha='center', va='center', fontsize=13)
    side_label('Real analytes per tube', 0, top_r)
    side_label('False analytes per tube', -top_f, 0)

    fig.canvas.draw()
    px_per_pt = dpi / 72
    to_px = lambda y: ax.transData.transform((0, y))[1]
    beside = {}
    for kind, (vals, cum, sign, sds) in stacks.items():
        for k, lvl in enumerate((3, 2, 1)):
            for i in range(len(rows)):
                if vals[k][i] <= 0:
                    continue
                y0, y1 = sign * cum[k][i], sign * cum[k + 1][i]
                yc = inv((fwd(y0) + fwd(y1)) / 2)
                text = f'{"●" * lvl} {vals[k][i]:.1f} ± {float(sds[k][i] or 0):.1f}'
                if abs(to_px(y1) - to_px(y0)) / px_per_pt >= 12:
                    ax.text(i, yc, text, ha='center', va='center', fontsize=9, color=ink)
                else:
                    beside.setdefault((i, sign), []).append((yc, text))
    # a segment too thin for its label: beside its bar, away from the axis, 9 pt apart
    for (i, sign), labels in beside.items():
        last = to_px(0) - sign * 3 * px_per_pt
        for y, text in sorted(labels, key=lambda t: to_px(t[0]), reverse=sign < 0):
            ypx = max(to_px(y), last + 12 * px_per_pt) if sign > 0 else min(to_px(y), last - 12 * px_per_pt)
            last = ypx
            ax.annotate(text, (i + width / 2, y), xytext=(i + width / 2 + 0.06, ax.transData.inverted().transform((0, ypx))[1]),
                        fontsize=9, va='center', arrowprops=dict(arrowstyle='-', color=ink, linewidth=0.6, shrinkA=0, shrinkB=0))
    for i, r in enumerate(rows):
        t = stacks['real'][1][-1][i]
        ax.annotate(f'{t:.1f} ± {g(r, "found_sd"):.1f}', (i, t), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=12)
        t = stacks['false'][1][-1][i]
        ax.annotate(f'{t:.1f} ± {g(r, "false_sd"):.1f}', (i, -t), xytext=(0, -3), textcoords='offset points', ha='center', va='top', fontsize=12)
    for i, r in enumerate(rows):
        ax.annotate(f'quant. error\nmedian {g(r, "error_median_pct_mean"):.0f} ± {g(r, "error_median_pct_sd"):.0f}%\n'
                    f'p90 {g(r, "error_p90_pct_mean"):.0f} ± {g(r, "error_p90_pct_sd"):.0f}%',
                    (i, 0), xycoords=('data', 'axes fraction'), xytext=(0, -48), textcoords='offset points',
                    ha='center', va='top', fontsize=11.5, color='#4a4a47')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def figure(a):
    """Redraw the figure from an export's summary table."""
    from app_paths import APP_VERSION
    version = APP_VERSION.split(' v')[1].split()[0]
    root = subfolder(a.dest, 'steps')
    with open(os.path.join(root, 'resumo_por_variante.csv'), encoding='utf-8') as f:
        summary = list(csv.DictReader(f))
    path = os.path.join(root, f'Etapas_quantificacao_nMRPro-H_v{version}.png')
    step_figure(summary, path)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('cmd', choices=('run', 'export', 'figure'))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--dest', help='export: the validation folder')
    a = ap.parse_args()
    {'run': run, 'export': export, 'figure': figure}[a.cmd](a)


if __name__ == '__main__':
    main()
