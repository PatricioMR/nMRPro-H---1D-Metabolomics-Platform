# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/cosy_calibration.py
"""
Calibração (referenciação) dos eixos de um espectro COSY ¹H-¹H.

Convenção universal em metabolómica ¹H: o padrão interno TSP / DSS-d4 define
0.00 ppm. Num COSY homonuclear, o TSP é um singleto → aparece como um pico
DIAGONAL em (0,0). Calibrar = deslocar ambos os eixos (F2 e F1) para que esse
pico de referência caia exactamente em 0 ppm.

Os eixos podem ter offsets ligeiramente diferentes, por isso F2 e F1 são
calibrados independentemente.
"""
import numpy as np


def detect_reference(f2_ppm, f1_ppm, data, ref=0.0, window=0.1):
    """
    Localiza o pico de referência (TSP/DSS) perto de (ref, ref) na diagonal.

    Returns:
        (f2_obs, f1_obs) — ppm observado do máximo dentro da caixa, ou
        (None, None) se não houver dados na janela.
    """
    jmask = np.abs(f2_ppm - ref) <= window
    imask = np.abs(f1_ppm - ref) <= window
    if not np.any(jmask) or not np.any(imask):
        return None, None
    sub = data[np.ix_(imask, jmask)]
    # Índice do máximo dentro da sub-matriz
    li = int(np.argmax(sub))
    si, sj = np.unravel_index(li, sub.shape)
    f1_obs = float(f1_ppm[np.where(imask)[0][si]])
    f2_obs = float(f2_ppm[np.where(jmask)[0][sj]])
    return f2_obs, f1_obs


def calibrate_cosy(f2_ppm, f1_ppm, data, ref=0.0, window=0.1):
    """
    Calcula os offsets de calibração para referenciar TSP/DSS a `ref` ppm.

    Returns:
        dict {
          'df2', 'df1'         : offsets a somar a cada eixo (ppm),
          'f2_obs', 'f1_obs'   : posição observada do pico de referência,
          'f2_ppm', 'f1_ppm'   : eixos já calibrados (np.ndarray),
        }
        ou None se a referência não for encontrada.
    """
    f2_obs, f1_obs = detect_reference(f2_ppm, f1_ppm, data, ref, window)
    if f2_obs is None:
        return None
    df2 = ref - f2_obs
    df1 = ref - f1_obs
    return {
        'df2': df2, 'df1': df1,
        'f2_obs': f2_obs, 'f1_obs': f1_obs,
        'f2_ppm': np.asarray(f2_ppm) + df2,
        'f1_ppm': np.asarray(f1_ppm) + df1,
    }
