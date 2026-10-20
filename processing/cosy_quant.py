# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/cosy_quant.py
"""
Quantificação relativa de metabolitos a partir de volumes de cross-peaks COSY.

O análogo 2D da área de um pico 1D é o VOLUME do cross-peak: a soma das
intensidades numa caixa em torno de (F2, F1), acima do nível de ruído.

ATENÇÃO (integridade científica): o volume de um cross-peak COSY é
SEMI-QUANTITATIVO — depende da constante de acoplamento J, do tempo de mistura
e de T2/eficiência de transferência. É rigoroso comparar o MESMO cross-peak do
MESMO metabolito entre amostras adquiridas igualmente; comparar metabolitos
diferentes em absoluto exigiria fatores de resposta por cross-peak (não
disponíveis). Por isso o output é ABUNDÂNCIA RELATIVA, não concentração absoluta.
"""
import numpy as np


def integrate_crosspeak(f2_ppm, f1_ppm, data, f2c, f1c,
                        half_f2=0.02, half_f1=0.02, floor=0.0):
    """
    Integra o volume de um cross-peak na caixa [f2c±half_f2, f1c±half_f1].

    Args:
        f2_ppm, f1_ppm: eixos (ppm)
        data:           matriz (n_f1, n_f2)
        f2c, f1c:       centro do cross-peak (ppm)
        half_f2, half_f1: meia-largura da caixa (ppm)
        floor:          só soma intensidades acima deste valor (nível de ruído)

    Returns:
        volume (float) — Σ (intensidade − floor) para pontos acima de floor,
        escalado pela área do pixel (ppm²) para ser comparável entre eixos.
    """
    jmask = np.abs(f2_ppm - f2c) <= half_f2
    imask = np.abs(f1_ppm - f1c) <= half_f1
    if not np.any(jmask) or not np.any(imask):
        return 0.0
    sub = data[np.ix_(imask, jmask)]
    pos = sub[sub > floor] - floor
    if pos.size == 0:
        return 0.0
    # Área do pixel em ppm² → volume comparável independentemente da resolução
    d_f2 = abs(float(f2_ppm[1] - f2_ppm[0])) if len(f2_ppm) > 1 else 1.0
    d_f1 = abs(float(f1_ppm[1] - f1_ppm[0])) if len(f1_ppm) > 1 else 1.0
    return float(pos.sum() * d_f2 * d_f1)


def quantify_metabolites(results, cosy_data, half_f2=0.02, half_f1=0.02,
                         noise_mult=3.0, aggregate='median'):
    """
    Atribui um volume a cada metabolito anotado integrando os seus cross-peaks.

    Args:
        results:    saída de annotate_cosy (lista de dicts com 'matches')
        cosy_data:  {'f2_ppm','f1_ppm','data', ...}
        half_f2, half_f1: meia-largura da caixa de integração (ppm)
        noise_mult: nível de ruído (floor) = mediana(|data|) × este factor
        aggregate:  'median' | 'sum' | 'mean' — como combinar os cross-peaks
                    de um metabolito num único volume

    Returns:
        lista (cópia de results) com campos adicionais por metabolito:
            'volume'            — volume agregado (au)
            'crosspeak_volumes' — volume de cada cross-peak emparelhado
            'rel_abundance'     — volume / Σ volumes (fracção)
        ordenada por volume desc.
    """
    f2_ppm = cosy_data['f2_ppm']
    f1_ppm = cosy_data['f1_ppm']
    data   = cosy_data['data']

    noise = float(np.median(np.abs(data)))
    floor = noise * float(noise_mult)

    _agg = {'median': np.median, 'sum': np.sum, 'mean': np.mean}.get(
        aggregate, np.median)

    out = []
    for r in results:
        vols = []
        for m in r.get('matches', []):
            v = integrate_crosspeak(
                f2_ppm, f1_ppm, data, m['f2'], m['f1'],
                half_f2=half_f2, half_f1=half_f1, floor=floor)
            vols.append(v)
        volume = float(_agg(vols)) if vols else 0.0
        rec = dict(r)
        rec['crosspeak_volumes'] = vols
        rec['volume'] = volume
        out.append(rec)

    grand = sum(o['volume'] for o in out) or 1.0
    for o in out:
        o['rel_abundance'] = o['volume'] / grand

    out.sort(key=lambda o: -o['volume'])
    return out
