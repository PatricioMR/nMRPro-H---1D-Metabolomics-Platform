# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The quantification quality (processing.quant_quality, ADR-105) on every
composition of the composition bench: how many numbers of each tier each tube
gets, and how far each tier's numbers are from the truth.

Reads what the composition bench already saved -- each tube's project
(rows, spectra, samples) and its ``verdade_por_amostra.csv`` -- in
``02_Composicoes_600MHz`` and ``03_Composicoes_500MHz``. Nothing is
quantified again: the tier is read from the rows as the card and the report
read it. Each analyte in each group is one number; its error is the median
of the group's concentrations over the truth, less one.

Writes ``13_Qualidade_quantificacao`` (replaced whole): a workbook whose
counts, medians and spreads are formulas over its data sheet, the data as
CSV, and a LEIA-ME.

    python -m validation.quant_quality_bench --dest "<validation folder>"
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.validation_folder import subfolder

LEVELS = ('Good', 'Warn', 'Poor')
FIELDS = (('composition_600', 600), ('composition_500', 500))


def _truth(path) -> dict:
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        out[r['hmdb_id']] = {'name': r['nome'],
                             'A': float(r['verdade_grupo_A_mM']),
                             'B': float(r['verdade_grupo_B_mM'])}
    return out


def tube_units(project, truth, field, tube, composition) -> list:
    """Every reported number of one tube, per group: its tier and reasons,
    the truth and the median measured."""
    from data import project_file
    from processing import quant_quality
    st = project_file.read(project)
    rows = (st.get('qnt') or {}).get('quant_results') or []
    ppm = np.asarray(st['ppm'], dtype=float)
    X = np.asarray(st['all_intensities'], dtype=float)
    groups = {}
    for i, s in enumerate(st.get('samples_info') or []):
        lbl = str(s.get('label'))
        if str(s.get('type', lbl)).upper() != 'BLK' and not s.get('excluded'):
            groups.setdefault(lbl, []).append(i)
    traces = {g: X[:, idx].mean(axis=1) for g, idx in groups.items()}
    out = []
    for r in rows:
        if r.get('_not_quantified') or r.get('_is_is'):
            continue
        mid = str(r.get('hmdb_id'))
        for g in groups:
            q = quant_quality.assess(r, g, ppm=ppm, trace=traces[g])
            if q['level'] is None:
                continue
            vals = (r.get('_concs_by_group') or {}).get(g) or []
            med = float(np.median(vals)) if vals else float('nan')
            out.append({'field_mhz': field, 'tube': tube, 'composition': composition,
                        'group': g, 'hmdb_id': mid, 'analyte': r.get('metabolite', mid),
                        'in_tube': 'sim' if mid in truth else 'não',
                        'quality': q['level'], 'reason': '; '.join(q['reasons']),
                        'true_mM': (truth.get(mid) or {}).get(g, ''),
                        'measured_mM': med if np.isfinite(med) else ''})
    return out


def collect(dest) -> list:
    units = []
    for key, field in FIELDS:
        root = subfolder(dest, key)
        for d in sorted(glob.glob(os.path.join(root, 'tubo*_composicao*'))):
            m = re.search(r'tubo(\d+)_composicao(\d+)', os.path.basename(d))
            projects = [p for p in glob.glob(os.path.join(d, '*.nmrproj'))]
            truth_csv = os.path.join(d, 'verdade_por_amostra.csv')
            if not m or not projects or not os.path.isfile(truth_csv):
                continue
            units += tube_units(projects[0], _truth(truth_csv), field,
                                int(m.group(1)), int(m.group(2)))
    return units


# ── the workbook ─────────────────────────────────────────────────────────────

def _style(ws, widths):
    from openpyxl.styles import Font
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows():
        for c in row:
            c.font = Font(name='Arial', size=10, bold=c.row == 1)


def write_workbook(units, path, version):
    from openpyxl import Workbook
    from openpyxl.worksheet.formula import ArrayFormula
    wb = Workbook()

    # Data: one row per number; error and |error| are formulas
    ws = wb.active
    ws.title = 'Unidades'
    head = ['Campo (MHz)', 'Tubo', 'Composição', 'Grupo', 'HMDB', 'Analito', 'No tubo',
            'Qualidade', 'Motivo', 'Verdade (mM)', 'Medido (mM)', 'Erro', '|Erro|']
    ws.append(head)
    for i, u in enumerate(units, start=2):
        ws.append([u['field_mhz'], u['tube'], u['composition'], u['group'], u['hmdb_id'],
                   u['analyte'], u['in_tube'], u['quality'], u['reason'],
                   u['true_mM'] if u['true_mM'] != '' else None,
                   u['measured_mM'] if u['measured_mM'] != '' else None,
                   f'=IF(AND(ISNUMBER(J{i}),J{i}>0,ISNUMBER(K{i})),K{i}/J{i}-1,"")',
                   f'=IF(L{i}="","",ABS(L{i}))'])
        ws[f'L{i}'].number_format = '0.0%'
        ws[f'M{i}'].number_format = '0.0%'
    n = len(units) + 1
    _style(ws, {'A': 11, 'B': 6, 'C': 11, 'D': 7, 'E': 13, 'F': 34, 'G': 8, 'H': 10,
                'I': 60, 'J': 12, 'K': 12, 'L': 9, 'M': 9})
    ws.freeze_panes = 'A2'
    R = lambda c: f'Unidades!${c}$2:${c}${n}'

    # Per composition: counts by tier (real / false) and median |error| by tier
    wc = wb.create_sheet('Por composição')
    combos = sorted({(u['field_mhz'], u['tube'], u['composition']) for u in units})
    head = ['Campo (MHz)', 'Tubo', 'Composição']
    head += [f'{lv} reais' for lv in LEVELS] + [f'{lv} falsos' for lv in LEVELS]
    head += [f'|Erro| mediano {lv}' for lv in LEVELS]
    wc.append(head)
    for i, (f, t, c) in enumerate(combos, start=2):
        wc.append([f, t, c])
        cond = f'{R("A")},$A{i},{R("B")},$B{i},{R("C")},$C{i}'
        for k, lv in enumerate(LEVELS):
            col = chr(ord('D') + k)
            wc[f'{col}{i}'] = f'=COUNTIFS({cond},{R("G")},"sim",{R("H")},"{lv}")'
            col = chr(ord('G') + k)
            wc[f'{col}{i}'] = f'=COUNTIFS({cond},{R("G")},"não",{R("H")},"{lv}")'
            col = chr(ord('J') + k)
            sel = (f'({R("A")}=$A{i})*({R("B")}=$B{i})*({R("C")}=$C{i})'
                   f'*({R("G")}="sim")*({R("H")}="{lv}")*ISNUMBER({R("M")})')
            wc[f'{col}{i}'] = ArrayFormula(f'{col}{i}',
                                           f'=IFERROR(MEDIAN(IF({sel},{R("M")})),"")')
            wc[f'{col}{i}'].number_format = '0.0%'
    nc = len(combos) + 1
    _style(wc, {c: 13 for c in 'ABCDEFGHIJKL'})
    wc.freeze_panes = 'D2'
    C = lambda col: f"'Por composição'!${col}$2:${col}${nc}"

    # Per tube: counts as mean ± SD over the compositions; |error| pooled
    wt = wb.create_sheet('Por tubo')
    tubes = sorted({(u['field_mhz'], u['tube']) for u in units})
    head = ['Campo (MHz)', 'Tubo', 'Composições']
    for lv in LEVELS:
        head += [f'{lv} reais (média)', f'{lv} reais (DP)']
    for lv in LEVELS:
        head += [f'{lv} falsos (média)']
    for lv in LEVELS:
        head += [f'|Erro| mediano {lv}', f'|Erro| p90 {lv}']
    wt.append(head)
    for i, (f, t) in enumerate(tubes, start=2):
        wt.append([f, t])
        wt[f'C{i}'] = f'=COUNTIFS({C("A")},$A{i},{C("B")},$B{i})'
        col = 4
        for k, lv in enumerate(LEVELS):
            src = chr(ord('D') + k)
            a = wt.cell(row=i, column=col)
            a.value = f'=AVERAGEIFS({C(src)},{C("A")},$A{i},{C("B")},$B{i})'
            a.number_format = '0.0'
            sd = wt.cell(row=i, column=col + 1)
            ref = sd.coordinate
            sd.value = ArrayFormula(ref, f'=IFERROR(STDEV(IF(({C("A")}=$A{i})*({C("B")}=$B{i}),'
                                         f'{C(src)})),"")')
            sd.number_format = '0.0'
            col += 2
        for k, lv in enumerate(LEVELS):
            src = chr(ord('G') + k)
            a = wt.cell(row=i, column=col)
            a.value = f'=AVERAGEIFS({C(src)},{C("A")},$A{i},{C("B")},$B{i})'
            a.number_format = '0.0'
            col += 1
        for lv in LEVELS:
            sel = (f'({R("A")}=$A{i})*({R("B")}=$B{i})*({R("G")}="sim")'
                   f'*({R("H")}="{lv}")*ISNUMBER({R("M")})')
            for kind in ('MEDIAN', 'P90'):
                cell = wt.cell(row=i, column=col)
                body = (f'MEDIAN(IF({sel},{R("M")}))' if kind == 'MEDIAN'
                        else f'PERCENTILE(IF({sel},{R("M")}),0.9)')
                cell.value = ArrayFormula(cell.coordinate, f'=IFERROR({body},"")')
                cell.number_format = '0.0%'
                col += 1
    _style(wt, {chr(ord('A') + k): 14 for k in range(21)})
    wt.freeze_panes = 'C2'

    # Read me
    wr = wb.create_sheet('Leia-me', 0)
    lines = [
        ['Qualidade da quantificação por tubo e composição'],
        [f'Gerado por validation/quant_quality_bench.py, nMRPro-H {version}.'],
        ['Fonte: os projetos já salvos da bancada de composições (02_Composicoes_600MHz, '
         '03_Composicoes_500MHz; motor da exportação, v0.83) e a verdade de cada tubo '
         '(verdade_por_amostra.csv). Nada foi quantificado de novo.'],
        ['Unidade: cada analito quantificado em cada grupo (A, B). Erro = mediana das '
         'concentrações do grupo / verdade - 1 (só analitos que estão no tubo).'],
        [''],
        ['Qualidade (processing/quant_quality.py, ADR-105), a pior verificação decide:'],
        ['Poor: achado em menos amostras do que o grupo tem; os próprios sinais divergem acima '
         'de max_disagreement_pct (45 %); CV >= 40 %.'],
        ['Warn: CV >= 20 %; o ajuste segura mais de 1,25x a área do traço sob o sinal; '
         'amplitude presa no teto (limite superior).'],
        ['Good: nenhuma das anteriores.'],
        [''],
        ['Abas: Por tubo (contagens como média e DP sobre as composições; |erro| mediano e p90 '
         'agrupando as composições), Por composição, Unidades (os dados; erro por fórmula).'],
        ['"reais" = analitos que estão no tubo; "falsos" = analitos quantificados que não '
         'estão no tubo (sem verdade, sem erro).'],
    ]
    for ln in lines:
        wr.append(ln)
    _style(wr, {'A': 140})
    wb.save(path)


def _soffice():
    for c in (shutil.which('soffice'), shutil.which('soffice.exe'),
              r'C:\Program Files\LibreOffice\program\soffice.exe'):
        if c and os.path.isfile(c):
            return c
    return None


def recalc(path) -> bool:
    """Compute the workbook's formulas with LibreOffice (headless convert),
    so it opens with its values everywhere, previews included; the formulas
    stay. False when LibreOffice is not installed (Excel computes them on
    opening)."""
    import subprocess
    import tempfile
    exe = _soffice()
    if exe is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([exe, '--headless', '--calc', '--convert-to', 'xlsx',
                        '--outdir', tmp, path], check=True, capture_output=True)
        shutil.copyfile(os.path.join(tmp, os.path.basename(path)), path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--dest', required=True, help='the validation folder')
    a = ap.parse_args()
    from app_paths import APP_VERSION
    version = APP_VERSION.split(' v')[1].split()[0]
    units = collect(a.dest)
    out = subfolder(a.dest, 'quant_quality')
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    with open(os.path.join(out, 'unidades.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(units[0]))
        w.writeheader()
        w.writerows(units)
    book = os.path.join(out, f'Qualidade_quantificacao_nMRPro-H_v{version}.xlsx')
    write_workbook(units, book, version)
    if not recalc(book):
        print('LibreOffice not found: the formulas compute when the workbook is opened')
    with open(os.path.join(out, 'LEIA-ME.md'), 'w', encoding='utf-8') as fh:
        fh.write('# Qualidade da quantificação\n\n'
                 f'`{os.path.basename(book)}`: quantos números de cada qualidade (Good, Warn, '
                 'Poor) cada tubo tem e o erro de cada qualidade contra a verdade, nas '
                 'composições da bancada (600 e 500 MHz). Detalhes na aba Leia-me.\n\n'
                 f'Gerado por `python -m validation.quant_quality_bench --dest <pasta>` '
                 f'(nMRPro-H {version}); lê os projetos já salvos, não quantifica de novo.\n')
    print(book, len(units), 'numbers')


if __name__ == '__main__':
    main()
