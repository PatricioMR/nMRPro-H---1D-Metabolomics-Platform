# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Each engine mechanism switched off alone, against the whole engine.

The standard GISSMO tubes (600 MHz, 18/50/100 analytes, seeds 0-2), every
variant one mechanism off; the table answers what each one is worth: analytes
found, strangers reported and the median error of the real ones. Variants
that change detection or annotation prepare the tube again (their settings
enter gissmo_bench's cache key); the others quantify the cached preparation.

    python -m validation.ablation run --jobs 8
    python -m validation.ablation csv          # CSVs + Ablation_NMRPro-H.xlsx
    python -m validation.ablation csv --export-dir "<validation folder>"
        # also <folder>/Ablacao: every run's tables, as a real study's

Runs land in out/bench_runs/ablation as bench_runs' pickles, so an
interrupted batch resumes where it stopped.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.bench_runs import OUT_DEFAULT, _apply_sets, _below_normal, _errors, _prep_sets
from validation.replicates import paired_change, reported as _reported, run_figures, summarize
from validation.validation_folder import subfolder

OUT = os.path.join(OUT_DEFAULT, 'ablation')
TUBES = ('18', '50', '100')
SEEDS = (0, 1, 2)
FIELD = 600

# (id, stage, mechanism, module settings, analysis defaults). Defaults are
# controllers.analysis_defaults entries; DETECTION ones re-prepare the tube.
VARIANTS = (
    ('full', 'Whole engine', 'Every mechanism on', (), {}),
    ('det_snip', 'Detection', 'SNIP baseline model before detection',
     (), {'baseline_model': False}),
    ('det_group', 'Detection', 'Per-sample detection with group presence',
     (), {'per_sample': False}),
    ('ann_spin', 'Annotation', 'Complex multiplets judged by their spin system at the field',
     ('processing.spin_patterns.SPIN_PATTERNS=False',), {}),
    ('ann_covary', 'Annotation', 'Lines of one multiplet must co-vary across samples',
     (), {}),
    ('ann_stocsy', 'Annotation', 'STOCSY corroboration in the election',
     (), {}),
    ('q_reselect', 'Quantification', "Singlet's peak chosen again against the study drift",
     ('gui.stages._quant_adapter.SINGLET_RESELECT=False',), {}),
    ('q_shared', 'Quantification', 'Contested windows balanced together',
     (), {'split_shared': False}),
    ('q_shapes', 'Quantification', 'Complex multiplets quantified by their shape',
     (), {'quantify_shapes': False}),
    ('q_joint', 'Quantification', 'Joint fit of overlapping analytes',
     (), {'joint_fit': False}),
    ('q_candidates', 'Quantification', 'Unreported candidates as columns of the joint fit',
     ('processing.quant_engine.JOINT_FIT_CANDIDATES=False',), {}),
    ('q_owndrift', 'Quantification', "Columns placed at the analyte's matched peak",
     ('processing.quant_engine.JOINT_FIT_OWN_DRIFT=False',), {}),
    ('q_offdrift', 'Quantification', 'Off-drift singlets withdrawn',
     ('processing.quant_engine.OFF_DRIFT_SINGLET_WITHDRAWN=False',), {}),
    ('q_cascade', 'Quantification', 'Residual cascade (analytes the election missed)',
     (), {'cascade': False}),
)


# The engine in the order it runs: (stage, process, what it does to the data,
# reference, ablation variant or "").
PROCESSES = (
    ('Detection', 'Baseline model', 'SNIP removes the broad component before peaks are sought; quantification still integrates the unmodelled data.', 'Ryan et al., Nucl Instrum Methods B 1988;34:396', 'det_snip'),
    ('Detection', 'Noise estimate', 'Noise sigma from the signal-free 12-10 ppm region, or the quietest segment where the spectrum has none.', '', ''),
    ('Detection', 'Peak detection', 'Peaks sought in every sample (SNR >= 6, 4 mppm apart) and kept when present in at least 3 samples of a group.', '', 'det_group'),
    ('Detection', 'Rigid alignment', 'Each sample shifted as a whole onto the most representative one; the shifts are kept for every later stage.', '', ''),
    ('Detection', 'STOCSY matrix', 'Correlation of every detected peak with every other across samples.', 'Cloarec et al., Anal Chem 2005;77:1282', ''),
    ('Annotation', 'Database search', 'Every reference signal within 0.05 ppm of a detected peak becomes a candidate (BMRB Mammalian v05).', '', ''),
    ('Annotation', 'Multiplet matching', 'Expected lines snapped to detected peaks line by line; J measured and compared with the reference (Delta J).', '', ''),
    ('Annotation', 'Spin-system patterns', 'Complex multiplets (m*) judged by their spin system solved at the study field.', 'Dashti et al., Anal Chem 2017;89:12201', 'ann_spin'),
    ('Annotation', 'Line co-variation', 'Lines of one multiplet must rise and fall together across samples.', '', 'ann_covary'),
    ('Annotation', 'Study drift', 'Where the study references stand: the mode of the confirmed multiplets, the reference panel only where it agrees.', 'Parzen, Ann Math Stat 1962;33:1065', ''),
    ('Annotation', 'Quality verdicts', 'Good, undecided or Bad per signal: incomplete multiplets, lines that do not co-vary, heights against multiplicity.', '', ''),
    ('Annotation', 'Election', 'Each peak goes to one analyte, ranked on evidence tier, Delta J, Delta ppm, fingerprint, STOCSY corroboration and drift coherence.', '', 'ann_stocsy'),
    ('Quantification', 'Singlet reselection', "A singlet's peak chosen again among nearby peaks against the study drift.", '', 'q_reselect'),
    ('Quantification', 'Window integration', 'Each signal integrated over its window against the internal standard (DSS), proton-corrected.', '', ''),
    ('Quantification', 'Contested windows', 'Windows two analytes claim solved together as one system of equations.', '', 'q_shared'),
    ('Quantification', 'Shape quantification', 'Complex multiplets quantified by their whole shape, not one line.', '', 'q_shapes'),
    ('Quantification', 'Joint fit', 'Overlapping analytes fitted together by non-negative least squares; each signal slides up to two linewidths.', 'Lawson & Hanson, Solving Least Squares Problems 1974', 'q_joint'),
    ('Quantification', 'Candidate columns', 'Candidates the election did not report enter the joint fit as columns whose amounts are not reported.', 'Tardivel et al. (ASICS), Metabolomics 2017;13:109', 'q_candidates'),
    ('Quantification', 'Own-peak columns', "An analyte's columns stand at the peak the election matched, not at listing plus common drift.", '', 'q_owndrift'),
    ('Quantification', 'Amount consistency', 'Signals of one analyte must agree on one amount; singlets off the study drift are withdrawn.', '', 'q_offdrift'),
    ('Quantification', 'Residual cascade', 'Analytes the election missed are sought in what the model leaves unexplained (greedy pursuit + NNLS, scored).', 'Pati et al., Asilomar 1993 (OMP)', 'q_cascade'),
    ('Reporting', 'Confidence dots', 'One to three dots from identity and quantity evidence.', '', ''),
)


# Mechanisms no setting turns off: the function that measures them answers
# what the engine reads as "not measured" (None for the lines' co-variation,
# an empty matrix for STOCSY -- fresh_run's own value with too few samples),
# wherever a module imported it by name.
PATCHES = {
    'ann_covary': {'lines_covary': lambda *a, **k: None,
                   'lines_covary_on_trace': lambda *a, **k: None},
    'ann_stocsy': {'compute_peak_correlation_matrix': lambda *a, **k: {}},
}


def _patch(vid) -> str:
    from processing import annotation_scoring, stocsy
    patches = PATCHES.get(vid) or {}
    originals = {name: getattr(annotation_scoring, name, None) or getattr(stocsy, name)
                 for name in patches}
    for mod in list(sys.modules.values()):
        for name, fn in patches.items():
            if getattr(mod, name, None) is originals[name]:
                setattr(mod, name, fn)
    return ';'.join(f'patch={n}' for n in sorted(patches))


def _path(vid, tube, seed):
    return os.path.join(OUT, f'{vid}_{tube}_{FIELD}_s{seed}.pkl')


def _set_defaults(defaults) -> str:
    """Put ``defaults`` into the analysis defaults in place (fresh_run holds
    the same dicts) and return the part that changes the preparation."""
    from controllers import analysis_defaults as ad
    prep = []
    for k, v in defaults.items():
        for name in ('DETECTION', 'ANNOTATION', 'QUANTIFICATION'):
            d = getattr(ad, name)
            if k in d:
                d[k] = v
                if name != 'QUANTIFICATION':
                    prep.append(f'{k}={v!r}')
                break
        else:
            raise KeyError(f'not an analysis default: {k}')
    return ';'.join(prep)


def _one(job):
    vid, tube, seed = job
    _vid, _stage, _what, sets, defaults = next(v for v in VARIANTS if v[0] == vid)
    from validation import gissmo_bench as gb
    from validation import mixture_bench as mb
    from validation.fresh_run import quantify_prepared
    _apply_sets(sets)
    gb.PREP_SETTINGS = ';'.join(p for p in (_prep_sets(sets), _set_defaults(defaults),
                                            _patch(vid)) if p)
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
    wanted = set(a.variants or [v[0] for v in VARIANTS])
    # the slow tube first, so the pool's tail is short
    jobs = [(v[0], t, s) for t in reversed(TUBES) for v in VARIANTS for s in SEEDS
            if v[0] in wanted and not os.path.exists(_path(v[0], t, s))]
    if not jobs:
        print('nothing to run')
        return
    threads = max(1, (os.cpu_count() or 1) // max(a.jobs, 1))
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = str(threads)
    t0 = time.time()
    # one job per process: a variant's settings must not outlive its job
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_below_normal,
                             max_tasks_per_child=1) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            (vid, t, s), secs = f.result()
            print(f'{i}/{len(jobs)} {vid} tube {t} seed {s} in {secs:.0f} s '
                  f'({time.time() - t0:.0f} s elapsed)', flush=True)


KEYS = ('in_tube', 'found', 'false', 'recovery_pct', 'precision_pct', 'error_median_pct',
        'error_p90_pct', 'real3', 'false3', 'real2', 'false2', 'real1', 'false1')


def runs_of(vid, tube) -> list:
    """Each seed's figures for ``vid`` on ``tube`` (seed order)."""
    out = []
    for s in SEEDS:
        p = _path(vid, tube, s)
        if os.path.exists(p):
            d = pickle.load(open(p, 'rb'))
            out.append({'seed': s, **run_figures(d['rows'], d['truth'], _errors(d['rows'], d['truth']))})
    return out


def summary(vid, tube):
    """Mean and SD per tube over the seeds run: found, false, error and
    real/false at each confidence level."""
    runs = runs_of(vid, tube)
    return summarize(runs, KEYS) if runs else None


# Reference only (never a criterion): every compound off its listing by
# sigma 6 mppm and each signal by sigma 4 mppm, random compositions 103-105.
EXTRACT = ('o2b', ('r50', 'r75'), (103, 104, 105))


def extract_rows():
    """The shifted-reference (Extract) runs per tube: mean and SD over the
    compositions."""
    label, tubes, seeds = EXTRACT
    out = []
    for t in tubes:
        runs = []
        for s in seeds:
            p = os.path.join(OUT_DEFAULT, f'{label}_{t}_{FIELD}_4_rs6_{s}.pkl')
            if os.path.exists(p):
                d = pickle.load(open(p, 'rb'))
                runs.append(run_figures(d['rows'], d['truth'], _errors(d['rows'], d['truth'])))
        if runs:
            out.append({'tube': t, 'version': 'v0.75-0.8 engine',
                        **summarize(runs, ('in_tube', 'found', 'false', 'recovery_pct',
                                           'error_median_pct', 'error_p90_pct'))})
    return out


def measured_rows(vid='full'):
    """Every real analyte the variant reported: true against measured, the
    median over each group's samples."""
    out = []
    for t in TUBES:
        for s in SEEDS:
            p = _path(vid, t, s)
            if not os.path.exists(p):
                continue
            d = pickle.load(open(p, 'rb'))
            for x in _reported(d['rows']):
                want = d['truth'].get(x['hmdb_id'])
                if not want:
                    continue
                for g, vals in (x.get('_concs_by_group') or {}).items():
                    if want.get(g) and vals:
                        out.append({'tube': t, 'seed': s,
                                    'analyte': x.get('metabolite') or x['hmdb_id'],
                                    'group': g, 'true_mM': round(float(want[g]), 4),
                                    'measured_mM': round(float(np.median(vals)), 4),
                                    'dots': int(x.get('_confidence') or 0)})
    return out


def _write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def impact_rows(tube='100'):
    """Each ablation against the whole engine on ``tube``, paired seed by
    seed: the mean change per tube and its SD."""
    full = runs_of('full', tube)
    out = []
    for vid, stage, what, _s, _d in VARIANTS:
        if vid == 'full':
            continue
        runs = runs_of(vid, tube)
        seeds = {r['seed'] for r in runs} & {r['seed'] for r in full}
        if not seeds:
            continue
        pick = lambda rs: [r for r in rs if r['seed'] in seeds]
        out.append({'variant': vid, 'stage': stage, 'mechanism': what, 'seeds': len(seeds),
                    **paired_change(pick(runs), pick(full), ('found', 'false', 'error_median_pct'))})
    return out


def method_rows():
    return [
        ('Simulator', 'GISSMO spin systems (data/gissmo_spin_systems.json), rendered at 600 MHz'),
        ('Tubes', ', '.join(TUBES) + ' analytes drawn from the reference database'),
        ('Seeds', ', '.join(str(s) for s in SEEDS) + ': figures per tube, mean and SD over the seeds; changes paired seed by seed'),
        ('Samples', '12, two groups of 6; concentration varies 12% between samples'),
        ('Engine', 'validation.fresh_run: the defaults a new user runs, internal standard DSS'),
        ('Found', 'reported analytes that are in the tube'),
        ('False', 'reported analytes that are not in the tube'),
        ('Error', '|measured / true - 1| per real analyte and group, median over samples'),
        ('Ablation', 'one mechanism off per variant, every other setting as the whole engine'),
        ('Not simulated', 'pH, ionic strength, matrix: every compound at its database shift'),
        ('Spin patterns', 'the engine and the simulator share the GISSMO spin systems for m* shapes'),
    ]


def _write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export_runs(dest, workbook):
    """``dest``/04_Ablacao, made anew: every run's analytes and samples in the
    validation export's own tables (export_validation.run_tables), the
    summary, the workbook, the version and a LEIA-ME."""
    import shutil
    import subprocess
    from app_paths import APP_VERSION
    from validation import export_validation as ev
    root = subfolder(dest, 'ablation')
    if os.path.isdir(root):
        shutil.rmtree(root)
    names = ev._sample_names()
    truths = {}
    n_runs = 0
    for vid, _stage, _what, _s, _d in VARIANTS:
        for t in TUBES:
            for s in SEEDS:
                p = _path(vid, t, s)
                if not os.path.exists(p):
                    continue
                d = pickle.load(open(p, 'rb'))
                if (t, s) not in truths:
                    truths[t, s] = ev.sample_truth(FIELD, s, int(t), 'padrao')
                a_rows, s_rows = ev.run_tables(d['rows'], d['truth'], truths[t, s], names)
                run = os.path.join(root, 'corridas', vid, f'tubo{int(t):03d}_semente{s}')
                ev.write_csv(os.path.join(run, 'resultados_por_analito.csv'), ev.A_HEAD[4:], a_rows)
                ev.write_csv(os.path.join(run, 'concentracoes_por_amostra.csv'), ev.S_HEAD[4:], s_rows)
                n_runs += 1
    shutil.copy(os.path.join(OUT_DEFAULT, 'engine_anatomy', 'ablation.csv'),
                os.path.join(root, 'resumo_ablacao.csv'))
    shutil.copy(workbook, os.path.join(root, os.path.basename(workbook)))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    with open(os.path.join(root, 'versao.txt'), 'w', encoding='utf-8') as f:
        f.write(f'software: {APP_VERSION}\ncommit: {commit}\n'
                f'corridas: {n_runs}\n')
    variants = '\n'.join(f'| `{v}` | {st} | {w} |' for v, st, w, _s, _d in VARIANTS)
    with open(os.path.join(root, 'LEIA-ME.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# Ablação do motor do {APP_VERSION}

Cada variante desliga **um** mecanismo do motor e roda de novo os mesmos tubos simulados do regime padrão da validação (GISSMO, {FIELD} MHz, tubos de {', '.join(TUBES)} analitos, sementes {', '.join(map(str, SEEDS))}, 12 amostras em dois grupos, concentração variando 12% entre amostras). A variante `full` é o motor inteiro. Os espectros e a verdade por amostra são os de `../01_Tabela_PDF/espectros/padrao/{FIELD}MHz` e `../01_Tabela_PDF/verdade/padrao/{FIELD}MHz`.

| variante | etapa | mecanismo desligado |
|---|---|---|
{variants}

## Arquivos

| arquivo | conteúdo |
|---|---|
| `corridas/<variante>/tuboNNN_sementeS/resultados_por_analito.csv` | cada analito presente e cada reportado: verdade A/B, medido (mediana do grupo), erro, bolinhas, via (eleição ou cascata), SNR e o motivo quando não recebeu número |
| `corridas/<variante>/tuboNNN_sementeS/concentracoes_por_amostra.csv` | cada analito reportado, amostra a amostra: verdade, medido e erro (medido/verdade − 1; vazio quando o analito não está no tubo) |
| `resumo_ablacao.csv` | achados, falsos, erro e reais/falsos por bolinha, por variante e tubo: média por tubo e desvio-padrão entre as sementes (`_mean`, `_sd`) |
| `{os.path.basename(workbook)}` | as mesmas tabelas em abas, com a lista dos processos do motor |
| `versao.txt` | versão, commit e número de corridas |

## Como reproduzir

```
python -m validation.ablation run --jobs 8
python -m validation.ablation csv --export-dir "<pasta da validação>"
```

Esta pasta é apagada e refeita a cada exportação.
""")
    print(f'{n_runs} runs exported to {root}')


def write_csv(a):
    """The results, once: a CSV per table (the dashboard's datasets) and the
    same tables as sheets of one workbook."""
    from validation.export_validation import _xlsx
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for vid, stage, what, _s, _d in VARIANTS:
        for t in TUBES:
            r = summary(vid, t)
            if r:
                rows.append({'variant': vid, 'stage': stage, 'mechanism': what,
                             'tube': t, **r})
    procs = [{'step': i, 'stage': p[0], 'process': p[1], 'what_it_does': p[2],
              'reference': p[3], 'ablation': p[4]} for i, p in enumerate(PROCESSES, 1)]
    mv = measured_rows()
    tables = {'processes': procs, 'ablation': rows, 'impact_tube100': impact_rows(),
              'measured_vs_true_standard': mv, 'extract_reference': extract_rows()}
    for name, t in tables.items():
        _write(os.path.join(a.out, f'{name}.csv'), t)
    sheets = [('Method', ['item', 'value'], method_rows())]
    sheets += [(name[:31], list(t[0]), [list(r.values()) for r in t]) for name, t in tables.items()]
    workbook = os.path.join(a.out, 'Ablation_NMRPro-H.xlsx')
    _xlsx(workbook, sheets)
    if a.export_dir:
        export_runs(a.export_dir, workbook)
    print(', '.join(f'{k} {len(v)}' for k, v in tables.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('what', choices=('run', 'csv'))
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--variants', nargs='*')
    ap.add_argument('--out', default=os.path.join(OUT_DEFAULT, 'engine_anatomy'))
    ap.add_argument('--export-dir', help='validation folder: writes <folder>/Ablacao anew')
    a = ap.parse_args()
    {'run': run, 'csv': write_csv}[a.what](a)


if __name__ == '__main__':
    main()
