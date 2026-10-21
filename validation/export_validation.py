# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Export the validation behind the report's quality table, so it can be
audited outside the code: every spectrum, the truth it was built from, what
the software reported for it, and the tables.

    python -m validation.export_validation "G:/My Drive/.../nMRPro-H_Validation"   # writes 01_Tabela_PDF

Reads the GISSMO bench's caches (``validation/out/cache``): the tubes of the
quality table (500 and 600 MHz, 18/50/100 analytes, seeds 0-2, every compound
moving as a block) and, when present, the stress runs with every chemical
environment scattered 5 mppm on its own (600 MHz, the ``validation.bench_runs``
label given by ``--stress-label``, run with the application's defaults). Writes:

    espectros/<regime>/<campo>/tubo<n>_semente<s>.csv   ppm + 12 samples
    verdade/<regime>/<campo>/tubo<n>_semente<s>.csv     true mM per group and per sample
    resultados_por_analito.csv                          every real analyte and every reported one
    concentracoes_por_amostra.csv                       true vs reported, sample by sample
    resumo.csv                                          per regime/field/tube: mean per tube, SD over the seeds
    Validacao_nMRPro-H_<versao>.xlsx                    all of the above as sheets
    dados/quality_reference.json, versao.txt

The true concentration of each sample is the group's times a log-normal
factor (CV 12 %) drawn by the generator; it is recovered by replaying the
generator's random stream (``sample_truth``), checked against the cached
spectrum by ``--check``.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pickle
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app_paths import APP_VERSION
from validation import gissmo_bench as gb
from validation import mixture_bench as mb
from validation.quality_reference import _row_error
from validation.replicates import reported, run_figures, summarize
from validation.validation_folder import subfolder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, 'validation', 'out', 'cache')
RUNS = os.path.join(REPO, 'validation', 'out', 'bench_runs')
TUBES = (18, 50, 100)
SEEDS = (0, 1, 2)
FIELDS = {500: 500.13, 600: 600.13}
REGIMES = {'padrao': 0.0, 'disperso_5mppm': 5.0}


def _sample_names():
    count: dict = {}
    out = []
    for info in mb.SAMPLES:
        g = info['label']
        count[g] = count.get(g, 0) + 1
        out.append(f'{g}{count[g]}')
    return out


def _quant_hash_of_quality_rows():
    """The quantification hash the quality table's rows were cached under:
    the one whose 18 row files all exist (newest first)."""
    import glob
    hashes = {}
    for p in glob.glob(os.path.join(CACHE, f'gissmo_rows_*_{gb._data_tag()}_*.pkl')):
        h = os.path.basename(p).rsplit('_', 1)[1][:-4]
        hashes.setdefault(h, []).append(os.path.getmtime(p))
    full = [h for h, t in hashes.items() if len(t) >= len(TUBES) * len(SEEDS) * len(FIELDS)]
    return max(full, key=lambda h: max(hashes[h])) if full else None


def load_run(regime, field, n, seed, qhash, stress_label='atual'):
    """``(X, truth, rows)`` of one run, or None when it is not cached."""
    tag = '' if REGIMES[regime] == 0 else f'_ss{REGIMES[regime]:.0f}'
    tube = os.path.join(CACHE, f'gissmo_{n}_{field}{tag}_s{seed}_{gb._data_tag()}.pkl')
    if regime == 'padrao':
        rows_p = os.path.join(CACHE, f'gissmo_rows_{n}_{field}_s{seed}_'
                                     f'{gb._data_tag()}_{qhash}.pkl')
        if not (os.path.exists(tube) and os.path.exists(rows_p)):
            return None
        rows = pickle.load(open(rows_p, 'rb'))
    else:
        rows_p = os.path.join(RUNS, f'{stress_label}_{n}_{field}_{REGIMES[regime]:g}_{seed}.pkl')
        if not (os.path.exists(tube) and os.path.exists(rows_p)):
            return None
        rows = pickle.load(open(rows_p, 'rb'))['rows']
    X, truth = pickle.load(open(tube, 'rb'))
    return X, truth, rows


def sample_truth(field, seed, n, regime):
    """``{hmdb_id: [true mM per sample]}`` -- the generator's draws replayed
    (gissmo_bench.build: per sample a drift, the noise, the standard, then
    one log-normal factor per compound, then the foreign compounds)."""
    import sqlite3
    gb.SIGNAL_SHIFT = REGIMES[regime] / 1000.0
    comp = gb._composition(str(n), seed)
    con = sqlite3.connect(mb.DB_DEFAULT)
    ids = [(mb._db_signals(con, name)[0], conc, factor) for name, conc, factor in comp]
    rng = np.random.default_rng(seed)
    out = {mid: [] for mid, _c, _f in ids}
    for info in mb.SAMPLES:
        grp = info['label']
        rng.normal(0.0, 0.0015)
        rng.normal(0.0, 1.0, gb.PPM.size)
        rng.normal(0, 0.0003)
        for mid, conc, factor in ids:
            base = conc if grp == 'A' else conc * factor
            out[mid].append(base * float(np.exp(rng.normal(0.0, 0.12))))
        for _f in mb.FOREIGN:
            rng.normal(0.0, 0.12)
    gb.SIGNAL_SHIFT = 0.0
    return out


def check_replay(field=600, seed=0, n=18):
    """Rebuild one tube with the replayed per-sample truth and compare with
    the cached spectrum: the replay is right only if they are identical."""
    gb.SIGNAL_SHIFT = 0.0
    X0, _t = pickle.load(open(os.path.join(
        CACHE, f'gissmo_{n}_{field}_s{seed}_{gb._data_tag()}.pkl'), 'rb'))
    X1, _t1 = gb.build(FIELDS[field], seed, gb._composition(str(n), seed))
    # Not bit for bit: a tube built under another BLAS thread count sums in
    # another order (1e-13 relative); a draw out of step moves it by the
    # signal itself.
    return bool(np.allclose(X0, X1, rtol=0.0, atol=1e-9 * float(np.abs(X0).max())))


def _err(meas, true):
    return (meas / true - 1.0) if meas is not None and true else None


def run_tables(rows, truth, per, names):
    """One run's two tables, without the run's own columns: every real
    analyte and every reported one (``A_HEAD[4:]``), and the reported ones
    sample by sample (``S_HEAD[4:]``). ``per`` is ``sample_truth``."""
    analytes, samples = [], []
    by_id = {r['hmdb_id']: r for r in rows if not r.get('_is_is')}
    for mid in sorted(set(truth) | {m for m, r in by_id.items()
                                    if not r.get('_not_quantified')}):
        r = by_id.get(mid)
        t = truth.get(mid)
        q = r is not None and not r.get('_not_quantified')
        g = (r or {}).get('_concs_by_group') or {}
        mA = float(np.median(g['A'])) if q and g.get('A') else None
        mB = float(np.median(g['B'])) if q and g.get('B') else None
        eA = _err(mA, (t or {}).get('A'))
        eB = _err(mB, (t or {}).get('B'))
        e = [x for x in (eA, eB) if x is not None]
        analytes.append([
            mid, (t or {}).get('name') or (r or {}).get('metabolite'),
            'sim' if t else 'não', 'sim' if q else 'não',
            int(r.get('_confidence') or 0) if q else '',
            (r or {}).get('status', '') if q else '',
            (t or {}).get('A', ''), (t or {}).get('B', ''),
            mA if mA is not None else '', mB if mB is not None else '',
            eA if eA is not None else '', eB if eB is not None else '',
            float(np.median(e)) if e else '',
            (('cascata' if r.get('_by_cascade') else 'eleição') if q else ''),
            (r or {}).get('snr', '') if q else '',
            ('' if q else ((r or {}).get('_status_reasons') or
                           ['sem linha: todos os sinais marcados Bad'])[0]),
        ])
        if q:
            cps = r.get('_concs_per_sample') or []
            for i, s in enumerate(names):
                tv = per.get(mid, [None] * len(names))[i] if t else None
                mv = cps[i] if i < len(cps) else None
                ev = _err(mv, tv) if mv is not None and np.isfinite(mv) else None
                samples.append([mid, (t or {}).get('name') or r.get('metabolite'),
                                s, s[0], tv if tv is not None else 0.0,
                                mv if mv is not None else '',
                                ev if ev is not None else ''])
    return analytes, samples


A_HEAD = ['regime', 'campo_MHz', 'tubo_analitos', 'semente', 'hmdb_id', 'nome',
          'presente_no_tubo', 'quantificado', 'confianca_bolinhas', 'status',
          'verdade_A_mM', 'verdade_B_mM', 'medido_A_mM_mediana',
          'medido_B_mM_mediana', 'erro_A', 'erro_B', 'erro_mediano',
          'via', 'snr', 'motivo_sem_numero']
S_HEAD = ['regime', 'campo_MHz', 'tubo_analitos', 'semente', 'hmdb_id', 'nome',
          'amostra', 'grupo', 'verdade_mM', 'medido_mM', 'erro']


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# per tube, then the mean and the SD over the seeds (validation.replicates)
R_KEYS = ('recovery_pct', 'false', 'precision_pct', 'error_median_pct', 'error_p90_pct',
          'real3', 'false3', 'real2', 'false2', 'real1', 'false1')
R_NAMES = {'recovery_pct': 'encontrados_%', 'false': 'falsos', 'precision_pct': 'precisao_%',
           'error_median_pct': 'erro_mediano_%', 'error_p90_pct': 'erro_p90_%',
           'real3': '3_reais', 'false3': '3_falsos', 'real2': '2_reais', 'false2': '2_falsos',
           'real1': '1_reais', 'false1': '1_falsos'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('out', help='the validation folder (writes <out>/01_Tabela_PDF)')
    ap.add_argument('--stress-label', default='atual',
                    help='bench_runs label of the scattered runs (same engine '
                         'and defaults as the quality table)')
    ap.add_argument('--check', action='store_true',
                    help='verify the per-sample truth replay on one tube first')
    a = ap.parse_args()
    a.out = subfolder(a.out, 'quality')
    if a.check:
        ok = check_replay()
        print('replay of the generator reproduces the cached tube:', ok)
        if not ok:
            raise SystemExit('per-sample truth would be wrong; stopping')
    qhash = _quant_hash_of_quality_rows()
    names = _sample_names()
    analyte_rows, sample_rows, summary = [], [], []
    for regime in REGIMES:
        for field in FIELDS:
            for n in TUBES:
                runs = []
                for seed in SEEDS:
                    got = load_run(regime, field, n, seed, qhash, a.stress_label)
                    if got is None:
                        continue
                    X, truth, rows = got
                    runs.append(run_figures(rows, truth, [
                        e for r in reported(rows) if r['hmdb_id'] in truth
                        for e in [_row_error(r, truth)] if e is not None]))
                    tag = f'{field}MHz'
                    write_csv(os.path.join(a.out, 'espectros', regime, tag,
                                           f'tubo{n:03d}_semente{seed}.csv'),
                              ['ppm'] + names,
                              ([f'{p:.6f}'] + [f'{v:.6g}' for v in X[i]]
                               for i, p in enumerate(gb.PPM)))
                    per = sample_truth(field, seed, n, regime)
                    write_csv(os.path.join(a.out, 'verdade', regime, tag,
                                           f'tubo{n:03d}_semente{seed}.csv'),
                              ['hmdb_id', 'nome', 'verdade_grupo_A_mM',
                               'verdade_grupo_B_mM'] + [f'{s}_mM' for s in names],
                              ([mid, t['name'], t['A'], t['B']]
                               + [f'{v:.6g}' for v in per.get(mid, [])]
                               for mid, t in truth.items()))
                    a_rows, s_rows = run_tables(rows, truth, per, names)
                    analyte_rows += [[regime, field, n, seed, *r] for r in a_rows]
                    sample_rows += [[regime, field, n, seed, *r] for r in s_rows]
                    print(f'{regime} {field} MHz tubo {n} semente {seed}', flush=True)
                if runs:
                    m = summarize(runs, R_KEYS, 2)
                    pooled = [round(100 * sum(r[f'real{l}'] for r in runs)
                                    / max(1, sum(r[f'real{l}'] + r[f'false{l}'] for r in runs)), 1)
                              for l in (3, 2, 1)]
                    summary.append([regime, field, n, len(runs)]
                                   + [m[f'{k}_{x}'] for k in R_KEYS for x in ('mean', 'sd')]
                                   + pooled)
    R_HEAD = (['regime', 'campo_MHz', 'tubo_analitos', 'sementes']
              + [f'{R_NAMES[k]}_{x}' for k in R_KEYS for x in ('media', 'dp')]
              + ['3_precisao_agrupada_%', '2_precisao_agrupada_%', '1_precisao_agrupada_%'])
    write_csv(os.path.join(a.out, 'resultados_por_analito.csv'), A_HEAD, analyte_rows)
    write_csv(os.path.join(a.out, 'concentracoes_por_amostra.csv'), S_HEAD, sample_rows)
    write_csv(os.path.join(a.out, 'resumo.csv'), R_HEAD, summary)
    os.makedirs(os.path.join(a.out, 'dados'), exist_ok=True)
    shutil.copy(os.path.join(REPO, 'data', 'quality_reference.json'),
                os.path.join(a.out, 'dados', 'quality_reference.json'))
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    with open(os.path.join(a.out, 'versao.txt'), 'w', encoding='utf-8') as f:
        f.write(f'software: {APP_VERSION}\ncommit: {commit}\n'
                f'base de referencia: {os.path.basename(mb.DB_DEFAULT)}\n'
                f'cache de quantificacao: {qhash}\n'
                f'exportado em: {datetime.datetime.now().isoformat(timespec="minutes")}\n')
    _xlsx(os.path.join(a.out, f'Validacao_nMRPro-H_v{APP_VERSION.split(" v")[1].split()[0]}.xlsx'),
          [('Resumo', R_HEAD, summary),
           ('Por_analito', A_HEAD, analyte_rows),
           ('Nao_encontrados', A_HEAD,
            [r for r in analyte_rows if r[6] == 'sim' and r[7] == 'não']),
           ('Falsos_reportados', A_HEAD,
            [r for r in analyte_rows if r[6] == 'não']),
           ('Por_amostra', S_HEAD, sample_rows),
           ('Desenho', ['parametro', 'valor'], _design())])
    print('escrito em', a.out)


def _design():
    return [
        ['espectros', 'GISSMO: sistemas de spin (deslocamentos + acoplamentos J) resolvidos '
                      'no campo do estudo (Dashti et al., Anal Chem 2017), somados linearmente '
                      '(MetAssimulo, Muncey et al. 2010)'],
        ['campos (MHz)', '500,13 e 600,13'],
        ['tubos', '18 analitos do estudo de referência; 50 e 100 = os 18 + sorteados da base '
                  '(Mammalian v05) a 0,05-1 mM'],
        ['sementes', '0, 1, 2 (composição e ruído mudam com a semente)'],
        ['amostras', '12 (grupos A e B, 6 cada)'],
        ['variação entre amostras', 'fator log-normal CV 12% por composto e amostra'],
        ['contraste entre grupos', 'B = A x fator do composto (ver verdade_grupo_B)'],
        ['largura de linha', f'{mb.LINEWIDTH_HZ} Hz'],
        ['deriva comum', f'{1000 * mb.DRIFT_PPM:.0f} mppm, + N(0; 1,5 mppm) por amostra'],
        ['ruído', f'gaussiano; dupleto do lactato a SNR {mb.NOISE_LACTATE_SNR:.0f}'],
        ['padrão interno', f'DSS {mb.STANDARD_MM} mM ({mb.STANDARD_NH} H) a 0 ppm'],
        ['composto fora da biblioteca', 'metanol (testa sinal sem dono)'],
        ['regime padrao', 'cada composto desloca-se em bloco (só a deriva comum)'],
        ['regime disperso_5mppm', 'cada ambiente químico com desvio próprio N(0; 5 mppm), '
                                  'como num extrato (pH/matriz); 600 MHz'],
        ['software', 'padrões do utilizador novo (controllers/analysis_defaults.py)'],
        ['erro', 'mediana medida no grupo / verdade do grupo - 1'],
        ['confiança', '3 = ●●●, 2 = ●●, 1 = ●'],
    ]


def _xlsx(path, sheets):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    wb.remove(wb.active)
    for title, head, rows in sheets:
        ws = wb.create_sheet(title)
        ws.append(head)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([x if x != '' else None for x in r])
        ws.freeze_panes = 'A2'
    wb.save(path)


if __name__ == '__main__':
    main()
