# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/cosy_peaks.py
"""
Detecção de cross-peaks em espectros COSY ¹H-¹H homonucleares.

Método (padrão da literatura para picking 2D, e.g. Bingol & Brüschweiler,
Anal. Chem. 2014 — COLMAR):

  1. Estimar o ruído pela mediana de |data| (a maioria dos pontos é ruído).
  2. Threshold = noise × mult.
  3. Encontrar máximos locais 2D acima do threshold (maximum_filter).
  4. Excluir a banda diagonal (|F2-F1| < diag_width): picos diagonais não
     dão informação de conectividade.
  5. Opcional: validar por simetria — um cross-peak real (a,b) tem um parceiro
     simétrico (b,a). Ruído-t1 não é simétrico. Filtra falsos positivos.

Devolve uma lista de cross-peaks acima da diagonal (F1 > F2) para evitar
duplicar cada par; cada peak é um dict {f2, f1, intensity}.
"""
import numpy as np


def estimate_noise(data):
    """Estimativa robusta do desvio do ruído (mediana de |data|)."""
    noise = float(np.median(np.abs(data)))
    dmax = float(np.percentile(np.abs(data), 99.9))
    if noise < dmax * 1e-4:
        noise = dmax * 1e-4
    return noise


def pick_cosy_crosspeaks(f2_ppm, f1_ppm, data, noise_mult=8.0,
                         diag_width=0.05, min_sep=0.02,
                         symmetric=True, sym_tol=0.03,
                         exclude_water=None):
    """
    Detecta cross-peaks num espectro COSY.

    Args:
        f2_ppm:     np.ndarray (n_f2,)  eixo directo (horizontal)
        f1_ppm:     np.ndarray (n_f1,)  eixo indirecto (vertical)
        data:       np.ndarray (n_f1, n_f2)
        noise_mult: threshold = noise × este factor
        diag_width: largura (ppm) da banda diagonal a excluir
        min_sep:    separação mínima (ppm) entre máximos (clustering)
        symmetric:  manter apenas cross-peaks com parceiro simétrico
        sym_tol:    tolerância (ppm) para encontrar o parceiro simétrico
        exclude_water: (lo, hi) ppm — exclui peaks com F2 ou F1 nessa zona

    Returns:
        list[dict]  cada um com {f2, f1, intensity} (apenas F1 > F2, fora da
                    diagonal). Vazio se nada detectado.
    """
    from scipy.ndimage import maximum_filter

    data = np.asarray(data, dtype=float)
    n_f1, n_f2 = data.shape

    noise = estimate_noise(data)
    threshold = noise * float(noise_mult)

    # Tamanho da janela do maximum_filter a partir de min_sep
    step_f2 = abs(float(f2_ppm[1] - f2_ppm[0])) if n_f2 > 1 else 1e-6
    step_f1 = abs(float(f1_ppm[1] - f1_ppm[0])) if n_f1 > 1 else 1e-6
    win_f2 = max(1, int(round(min_sep / step_f2)))
    win_f1 = max(1, int(round(min_sep / step_f1)))
    size = (2 * win_f1 + 1, 2 * win_f2 + 1)

    # Máximos locais: ponto == máximo da vizinhança E acima do threshold
    local_max = (data == maximum_filter(data, size=size)) & (data > threshold)
    i_idx, j_idx = np.where(local_max)   # i = F1 (linha), j = F2 (coluna)

    candidates = []
    for i, j in zip(i_idx, j_idx):
        f1 = float(f1_ppm[i])
        f2 = float(f2_ppm[j])
        # Excluir banda diagonal
        if abs(f2 - f1) < diag_width:
            continue
        if exclude_water is not None:
            lo, hi = exclude_water
            if (lo <= f2 <= hi) or (lo <= f1 <= hi):
                continue
        candidates.append({'f2': f2, 'f1': f1,
                           'intensity': float(data[i, j])})

    if not candidates:
        return []

    # Manter só a metade acima da diagonal (F1 > F2) — cada par aparece 2×
    upper = [c for c in candidates if c['f1'] > c['f2']]

    if not symmetric:
        return sorted(upper, key=lambda c: -c['intensity'])

    # Validação por simetria: (f2,f1) precisa de parceiro perto de (f1,f2),
    # i.e. um candidato com f2≈c.f1 e f1≈c.f2 (na metade inferior)
    lower = [c for c in candidates if c['f1'] < c['f2']]
    validated = []
    for c in upper:
        for d in lower:
            if (abs(d['f2'] - c['f1']) < sym_tol and
                    abs(d['f1'] - c['f2']) < sym_tol):
                validated.append(c)
                break

    return sorted(validated, key=lambda c: -c['intensity'])
