# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""What STOCSY is worth on the simulated bench.

    Cloarec, O. et al. (2005). Statistical total correlation spectroscopy: an
    exploratory approach for latent biomarker identification from metabolic
    1H NMR data sets. Analytical Chemistry 77(5):1282-1289.

Two uses of the co-variation across samples, each against the engine without it:

* the lift of the confidence dots (processing.stocsy_evidence, ADR-097). It
  only relabels analytes the engine already reports, so one engine run gives
  the dots before and after; the figure is the real and false analytes at each
  dot level, both ways.
* the STOCSY corroboration of the election (validation.ablation, variant
  ann_stocsy): the whole engine against the engine without it, paired seed by
  seed.

The standard GISSMO tubes (18/50/100 analytes) at 600 and 500 MHz, seeds 0-5,
in the standard regime (amounts vary 12% between samples) and in the Extract
regime (45% and a 30% dilution, reference only): 72 runs, plus the election
pairs (600 MHz, standard regime, seeds 0-2).

    python -m validation.stocsy_bench run --jobs 6
    python -m validation.stocsy_bench csv --export-dir "<validation folder>"
        # also <folder>/11_STOCSY: every run's tables, as a real study's

Runs land in out/bench_runs/stocsy (the election pairs in stocsy_election), so
an interrupted batch resumes where it stopped.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation import ablation
from validation.bench_runs import OUT_DEFAULT, _below_normal
from validation.replicates import mean_sd, paired_change, pm, reported
from validation.validation_folder import subfolder

OUT = os.path.join(OUT_DEFAULT, 'stocsy')
ELECTION_OUT = os.path.join(OUT_DEFAULT, 'stocsy_election')
FIELDS = (600, 500)
REGIMES = {'padrao': (0.12, 0.0), 'extrato': (0.45, 0.30)}
TUBES = (18, 50, 100)
SEEDS = (0, 1, 2, 3, 4, 5)
ELECTION_VARIANTS = ('full', 'ann_stocsy')
LEVELS = (3, 2, 1)
# the engine before this session, kept beside the current one for the election
OLD_ENGINE = 'v0.80 (15/10)'
NEW_ENGINE = 'atual'

# What the rules for raising a dot gave on seeds 0-2 (600 MHz, the engine's
# analytes with the dots they already had): ADR-097, promoted real/false.
SELECTION = (
    ('A: um parceiro (r >= 0,8, p < 0,05)', 'rejeitada (77% de precisão)',
     '10/5', '16/6', '7/3', '19/5', '6/1'),
    ('D: um parceiro forte (r >= 0,9)', 'rejeitada',
     '3/0', '13/3', '6/2', '15/3', '3/1'),
    ('F: >= 2 pares', 'ADOTADA (excluindo a cascata: 24/1 de ● para ●● e 6/0 de ●● para ●●●)',
     '1/0', '6/2', '3/0', '11/0', '5/1'),
)


def _path(field, regime, tube, seed):
    return os.path.join(OUT, f'{field}_{regime}_{tube}_s{seed}.pkl')


def _level(x) -> int:
    return min(max(int(x or 1), 1), 3)


def slim_rows(rows, truth) -> list:
    """The reported analytes with the dots the engine gave before the lift and
    after it, and whether each is in the tube."""
    out = []
    for r in reported(rows):
        lf = r.get('_stocsy_lift')
        after = int(r.get('_confidence') or 0)
        out.append({'name': r.get('metabolite'), 'hmdb_id': r.get('hmdb_id'),
                    'real': r['hmdb_id'] in truth, 'cascade': bool(r.get('_by_cascade')),
                    'before': int(lf['from']) if lf else after, 'after': after,
                    'pairs': int(lf['pairs']) if lf else 0})
    return out


def slim_of(run) -> list:
    return run['slim'] if run.get('rows') is None else slim_rows(run['rows'], run['truth'])


def figures(slim, n_truth) -> dict:
    """One run: the analytes found and false, and at each dot level the real
    and false ones before and after the lift, the same for two dots or more,
    the precision of each, and the change."""
    n_real = sum(1 for r in slim if r['real'])
    out = {'in_tube': n_truth, 'found': n_real, 'false': len(slim) - n_real}
    for tag in ('before', 'after'):
        groups = {str(lvl): [r for r in slim if _level(r[tag]) == lvl] for lvl in LEVELS}
        groups['2up'] = [r for r in slim if _level(r[tag]) >= 2]
        for name, at in groups.items():
            real = sum(1 for r in at if r['real'])
            out[f'real{name}_{tag}'], out[f'false{name}_{tag}'] = real, len(at) - real
            if name in ('3', '2', '2up'):
                out[f'prec{name}_{tag}'] = 100 * real / len(at) if at else np.nan
    for name in ('3', '2', '1', '2up'):
        for kind in ('real', 'false'):
            out[f'd_{kind}{name}'] = out[f'{kind}{name}_after'] - out[f'{kind}{name}_before']
    moved = [r for r in slim if _level(r['after']) > _level(r['before'])]
    out['up_real'] = sum(1 for r in moved if r['real'])
    out['up_false'] = len(moved) - out['up_real']
    return out


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _done(path) -> bool:
    """A run of this module's own (the imported ones hold no engine rows)."""
    return os.path.exists(path) and _load(path).get('rows') is not None


def _comparable(slim):
    """The content of a run, without the order of its rows (the engine does not
    return them in the same order on every start; the figures do not depend on it)."""
    return sorted(((str(r['name']), r['real'], r['cascade'], r['before'], r['after'], r['pairs'])
                   for r in slim))


def _lift_one(job):
    field, regime, tube, seed = job
    from app_paths import APP_VERSION
    from processing import stocsy_evidence
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    from validation.fresh_run import quantify_prepared
    stocsy_evidence.STOCSY_LIFT = True
    gb.AMOUNT_SD, gb.DILUTION_SD = REGIMES[regime]
    t0 = time.time()
    prep, truth = gb.prepared(gb.FIELDS[field], seed, tube)
    mb.MHZ = gb.FIELDS[field]
    with contextlib.redirect_stdout(io.StringIO()):
        rows = quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM,
                                 standard_nh=mb.STANDARD_NH,
                                 standard_conc=mb.STANDARD_MM, standard_name='DSS')
    path = _path(field, regime, tube, seed)
    same = None
    if os.path.exists(path):                       # an imported run: the same result?
        same = _comparable(_load(path)['slim']) == _comparable(slim_rows(rows, truth))
    with open(path + '.tmp', 'wb') as f:
        pickle.dump({'rows': rows, 'truth': truth, 'n_peaks': len(prep['peaks']),
                     'version': APP_VERSION, 'same_as_imported': same}, f)
    os.replace(path + '.tmp', path)
    return job, time.time() - t0, same


def _election_one(job):
    ablation.OUT = ELECTION_OUT
    os.makedirs(ELECTION_OUT, exist_ok=True)
    (vid, tube, seed), secs = ablation._one(job)
    return (vid, tube, seed), secs, None


def _jobs(a) -> list:
    """What is left to run, the small tubes first; the imported runs, redone
    for their per-sample tables, after the missing ones."""
    missing, redo = [], []
    for tube in a.tubes:
        for field in a.fields:
            for regime in a.regimes:
                for seed in a.seeds:
                    p = _path(field, regime, tube, seed)
                    if _done(p):
                        continue
                    (redo if os.path.exists(p) else missing).append(
                        (_lift_one, (field, regime, tube, seed)))
    jobs = missing + ([] if a.keep_imported else redo)
    if a.election:
        for tube in reversed(ablation.TUBES):
            for vid in ELECTION_VARIANTS:
                for seed in ablation.SEEDS:
                    if not os.path.exists(os.path.join(
                            ELECTION_OUT, f'{vid}_{tube}_{ablation.FIELD}_s{seed}.pkl')):
                        jobs.append((_election_one, (vid, tube, seed)))
    return jobs


def run(a):
    os.makedirs(OUT, exist_ok=True)
    jobs = _jobs(a)
    if not jobs:
        print('nothing to run')
        return
    threads = max(1, (os.cpu_count() or 1) // max(a.jobs, 1))
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = str(threads)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_below_normal,
                             max_tasks_per_child=1) as ex:
        futs = [ex.submit(fn, j) for fn, j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            job, secs, same = f.result()
            note = '' if same is None else (' = the imported run' if same else ' DIFFERS from the imported run')
            print(f'{i}/{len(jobs)} {job} in {secs:.0f} s ({time.time() - t0:.0f} s elapsed){note}',
                  flush=True)


def import_legacy(a):
    """Runs made before this module (a scratch lift bench: one pickle per
    regime, tube and seed, 600 MHz) as this module's own, without engine rows."""
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for regime in REGIMES:
        for tube in TUBES:
            for seed in SEEDS:
                src = os.path.join(a.source, f'lift_{regime}_{tube}_{seed}.pkl')
                dst = _path(600, regime, tube, seed)
                if os.path.exists(src) and not os.path.exists(dst):
                    d = _load(src)
                    with open(dst, 'wb') as f:
                        pickle.dump({'rows': None, 'slim': d['rows'], 'n_truth': d['n_truth']}, f)
                    n += 1
    print(f'{n} runs imported')


def _runs(field, regime, tube) -> list:
    """``(seed, figures, slim, run)`` of every seed run for the cell."""
    out = []
    for s in SEEDS:
        p = _path(field, regime, tube, s)
        if os.path.exists(p):
            d = _load(p)
            slim = slim_of(d)
            out.append((s, figures(slim, len(d['truth']) if d.get('truth') else d['n_truth']), slim, d))
    return out


def _stat(runs, key) -> tuple:
    return mean_sd([f[key] for _s, f, _l, _d in runs])


def _cell(runs, key) -> str:
    return pm(*_stat(runs, key))


def summary_rows() -> list:
    out = []
    for regime in REGIMES:
        for field in FIELDS:
            for tube in TUBES:
                runs = _runs(field, regime, tube)
                if not runs:
                    continue
                row = {'campo_MHz': field, 'regime': regime, 'tubo': tube, 'corridas': len(runs),
                       'achados': _cell(runs, 'found'), 'falsos': _cell(runs, 'false')}
                for name, label in (('3', '●●●'), ('2', '●●'), ('1', '●'), ('2up', '≥●●')):
                    for kind, kl in (('real', 'reais'), ('false', 'falsos')):
                        for tag, tl in (('before', 'antes'), ('after', 'depois')):
                            row[f'{label} {kl} {tl}'] = _cell(runs, f'{kind}{name}_{tag}')
                row['subiram reais'] = _cell(runs, 'up_real')
                row['subiram falsos'] = _cell(runs, 'up_false')
                for name, label in (('3', '●●●'), ('2', '●●'), ('2up', '≥●●')):
                    for tag, tl in (('before', 'antes'), ('after', 'depois')):
                        row[f'precisão {label} % {tl}'] = _cell(runs, f'prec{name}_{tag}')
                out.append(row)
    return out


def run_rows() -> list:
    out = []
    for regime in REGIMES:
        for field in FIELDS:
            for tube in TUBES:
                for s, f, _slim, d in _runs(field, regime, tube):
                    out.append({'campo_MHz': field, 'regime': regime, 'tubo': tube, 'semente': s,
                                'tabelas_por_amostra': 'sim' if d.get('rows') is not None else 'não (importada)',
                                **{k: ('' if isinstance(v, float) and not np.isfinite(v) else v)
                                   for k, v in f.items()}})
    return out


def promoted_rows() -> list:
    out = []
    for regime in REGIMES:
        for field in FIELDS:
            for tube in TUBES:
                for s, _f, slim, _d in _runs(field, regime, tube):
                    for r in slim:
                        if _level(r['after']) > _level(r['before']):
                            out.append({'campo_MHz': field, 'regime': regime, 'tubo': tube, 'semente': s,
                                        'analito': r['name'], 'está no tubo': 'sim' if r['real'] else 'não',
                                        'de': _level(r['before']), 'para': _level(r['after']),
                                        'pares co-variantes': r['pairs']})
    return out


def map_rows() -> list:
    out = []
    for regime in REGIMES:
        for field in FIELDS:
            for tube in TUBES:
                row = {'campo_MHz': field, 'regime': regime, 'tubo': tube}
                for s in SEEDS:
                    p = _path(field, regime, tube, s)
                    row[f'semente {s}'] = ('—' if not os.path.exists(p) else
                                           'feita' if _done(p) else 'importada')
                out.append(row)
    return out


def _election_runs(out, vid, tube) -> list:
    saved, ablation.OUT = ablation.OUT, out
    try:
        return ablation.runs_of(vid, str(tube))
    finally:
        ablation.OUT = saved


def election_rows() -> list:
    """The whole engine against the engine without STOCSY in the election, per
    tube, for the engine as it was (v0.80) and as it is."""
    keys = ('found', 'false', 'precision_pct', 'error_median_pct')
    out = []
    for engine, folder in ((OLD_ENGINE, os.path.join(OUT_DEFAULT, 'ablation')), (NEW_ENGINE, ELECTION_OUT)):
        for tube in ablation.TUBES:
            on = _election_runs(folder, 'full', tube)
            off = _election_runs(folder, 'ann_stocsy', tube)
            seeds = sorted({r['seed'] for r in on} & {r['seed'] for r in off})
            if not seeds:
                continue
            on, off = ([r for r in rs if r['seed'] in seeds] for rs in (on, off))
            change = paired_change(on, off, keys)
            row = {'motor': engine, 'tubo': int(tube), 'sementes': len(seeds)}
            for k, label in zip(keys, ('achados', 'falsos', 'precisão %', 'erro mediano %')):
                row[f'{label} com STOCSY'] = pm(*mean_sd([r[k] for r in on]))
                row[f'{label} sem STOCSY'] = pm(*mean_sd([r[k] for r in off]))
                row[f'{label}: com − sem'] = pm(change[f'{k}_change_mean'], change[f'{k}_change_sd'])
            out.append(row)
    return out


def method_rows() -> list:
    return [
        ('Simulador', 'GISSMO (data/gissmo_spin_systems.json) a 600 e 500 MHz; tubos de 18, 50 e 100 analitos sorteados da base'),
        ('Sementes', ', '.join(map(str, SEEDS)) + ': média e desvio-padrão entre as sementes por tubo, nunca somados'),
        ('Amostras', '12, dois grupos de 6'),
        ('Regime padrão', 'a quantidade de cada analito varia 12% entre amostras (o regime da validação)'),
        ('Regime extrato', 'variação de 45% e diluição de 30% entre amostras; só referência'),
        ('Motor', 'validation.fresh_run: os padrões que um usuário novo roda, padrão interno DSS'),
        ('Regra testada', 'um analito sobe uma bolinha (no máximo ●●●) quando >= 2 pares de seus sinais co-variam (PQN, r >= 0,80, p < 0,05); nunca desce; a cascata não sobe (processing.stocsy_evidence, ADR-097)'),
        ('Antes / depois', 'a mesma corrida: o motor guarda as bolinhas que daria sem a regra'),
        ('Reais / falsos', 'analitos reportados que estão / não estão no tubo'),
        ('Subiram', 'analitos que subiram de bolinha, reais e falsos'),
        ('Eleição', 'motor inteiro contra o motor sem a corroboração por STOCSY na eleição (variante ann_stocsy da ablação), pareado semente a semente; v0.80 = ablação de 15/10, atual = refeita com o código de hoje'),
        ('Não simulado', 'pH, força iônica, matriz: cada composto no seu deslocamento da base'),
    ]


def _write_workbook(path):
    from validation.export_validation import _xlsx
    clean = lambda rows: [[('' if isinstance(v, float) and not np.isfinite(v) else v) for v in r.values()]
                          for r in rows]
    summary, runs, promoted, mapa, election = (
        summary_rows(), run_rows(), promoted_rows(), map_rows(), election_rows())
    sheets = [('Método', ['item', 'valor'], method_rows())]
    for title, rows in (('Resumo', summary), ('Eleição', election), ('Mapa', mapa),
                        ('Corridas', runs), ('Promovidos', promoted)):
        if rows:
            sheets.append((title, list(rows[0]), clean(rows)))
    sheets.append(('Regras testadas', ['regra', 'veredito', 'tubo 100 padrão', 'tubo 100 extrato',
                                       'tubo 50 padrão', 'tubo 50 extrato', 'tubo 18 extrato'],
                   [list(r) for r in SELECTION]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _xlsx(path, sheets)
    return len(runs), len(promoted)


def export_runs(dest, workbook):
    """``dest``/11_STOCSY, made anew: every run's analytes (and samples, in the
    standard regime) in the validation export's own tables, plus the dots
    before and after, the workbook, the version and a LEIA-ME."""
    from app_paths import APP_VERSION
    from validation import export_validation as ev
    root = subfolder(dest, 'stocsy')
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    names = ev._sample_names()
    n_runs = 0
    for regime in REGIMES:
        for field in FIELDS:
            for tube in TUBES:
                truth_per = {}
                for s, _f, slim, d in _runs(field, regime, tube):
                    if d.get('rows') is None:
                        continue
                    by_id = {r['hmdb_id']: r for r in slim}
                    run = os.path.join(root, 'corridas', f'{field}MHz', regime, f'tubo{tube:03d}_semente{s}')
                    per = {}
                    if regime == 'padrao':
                        per = truth_per.setdefault(s, ev.sample_truth(field, s, tube, regime))
                    a_rows, s_rows = ev.run_tables(d['rows'], d['truth'], per, names)
                    a_rows = [r + ([by_id[r[0]]['before'], by_id[r[0]]['after'], by_id[r[0]]['pairs']]
                                   if r[0] in by_id else ['', '', '']) for r in a_rows]
                    ev.write_csv(os.path.join(run, 'resultados_por_analito.csv'),
                                 ev.A_HEAD[4:] + ['bolinhas_sem_STOCSY', 'bolinhas_com_STOCSY',
                                                  'pares_co_variantes'], a_rows)
                    if regime == 'padrao':
                        ev.write_csv(os.path.join(run, 'concentracoes_por_amostra.csv'),
                                     ev.S_HEAD[4:], s_rows)
                    n_runs += 1
    shutil.copy(workbook, os.path.join(root, os.path.basename(workbook)))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    with open(os.path.join(root, 'versao.txt'), 'w', encoding='utf-8') as f:
        f.write(f'software: {APP_VERSION}\ncommit: {commit}\ncorridas: {n_runs}\n')
    with open(os.path.join(root, 'LEIA-ME.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# STOCSY na bancada simulada — {APP_VERSION}

O que o STOCSY vale no motor, medido nos tubos simulados (GISSMO, 600 e 500 MHz, tubos de {', '.join(map(str, TUBES))} analitos, sementes {', '.join(map(str, SEEDS))}, 12 amostras em dois grupos).

Dois usos, cada um contra o motor sem ele:

1. **Bolinhas** (`processing/stocsy_evidence.py`, ADR-097): o analito sobe uma bolinha (no máximo ●●●) quando pelo menos 2 pares de sinais seus co-variam entre as amostras (PQN, r ≥ 0,80, p < 0,05). Nunca desce; a cascata não sobe. A mesma corrida dá as bolinhas antes e depois.
2. **Eleição** (variante `ann_stocsy` da ablação): o motor inteiro contra o motor sem a corroboração por STOCSY na escolha do analito de cada pico.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `{os.path.basename(workbook)}` | `Resumo` (média ± desvio-padrão entre as sementes, por campo, regime e tubo), `Eleição`, `Mapa` (o que foi rodado), `Corridas` (uma linha por corrida), `Promovidos` (cada analito que subiu), `Regras testadas`, `Método` |
| `corridas/<campo>MHz/<regime>/tuboNNN_sementeS/resultados_por_analito.csv` | cada analito presente e cada reportado, com as bolinhas sem e com STOCSY e os pares que co-variam |
| `corridas/.../concentracoes_por_amostra.csv` | cada analito reportado, amostra a amostra (só no regime padrão: é o único cuja verdade por amostra se refaz) |
| `versao.txt` | versão, commit e número de corridas |

O regime extrato é só referência. Esta pasta é apagada e refeita a cada exportação.

## Como reproduzir

```
python -m validation.stocsy_bench run --jobs 6
python -m validation.stocsy_bench csv --export-dir "<pasta da validação>"
```
""")
    print(f'{n_runs} runs exported to {root}')


def write_csv(a):
    workbook = os.path.join(a.out, 'STOCSY_NMRPro-H.xlsx')
    n_runs, n_up = _write_workbook(workbook)
    print(f'{n_runs} runs, {n_up} analytes raised -> {workbook}')
    if a.export_dir:
        export_runs(a.export_dir, workbook)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('what', choices=('run', 'csv', 'import'))
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--fields', type=int, nargs='*', default=list(FIELDS))
    ap.add_argument('--regimes', nargs='*', default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument('--tubes', type=int, nargs='*', default=list(TUBES))
    ap.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    ap.add_argument('--no-election', dest='election', action='store_false')
    ap.add_argument('--keep-imported', action='store_true',
                    help='do not redo the imported runs (they hold no per-sample tables)')
    ap.add_argument('--source', help='import: the folder of the scratch lift bench pickles')
    ap.add_argument('--out', default=os.path.join(OUT_DEFAULT, 'stocsy_tables'))
    ap.add_argument('--export-dir', help='validation folder: writes <folder>/11_STOCSY anew')
    a = ap.parse_args()
    {'run': run, 'csv': write_csv, 'import': import_legacy}[a.what](a)


if __name__ == '__main__':
    main()
