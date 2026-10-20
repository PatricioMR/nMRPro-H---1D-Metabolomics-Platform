# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# processing/cosy_annotation.py
"""
Anotação de cross-peaks COSY ¹H-¹H por co-ocorrência de shifts (método COLMAR).

Princípio (Bingol & Brüschweiler, Anal. Chem. 2014; Robinette et al. 2008):
não é preciso uma base de dados COSY. Um cross-peak em (F2=a, F1=b) implica
dois protões acoplados, um a δ=a e outro a δ=b. Um metabolito é candidato se a
sua lista de shifts ¹H contém ressonâncias perto de a E de b.

Refinamento por J (conectividade): a tabela shifts_1h guarda o J de cada linha.
Dois shifts realmente acoplados partilham um valor de J. Quando ambos os shifts
do cross-peak partilham um J comum, a evidência de conectividade é mais forte —
marca-se o match como 'J-confirmed'.

Ranking por COBERTURA: a fracção de cross-peaks observados que o metabolito
explica, ponderada pela proximidade e por J-confirmation.
"""
import sqlite3
import numpy as np


def load_shift_library(db_path):
    """
    Carrega a biblioteca de shifts ¹H de uma DB SQLite.

    Suporta o esquema shifts_1h (ppm, multiplicity, j_coupling, protons).

    Returns:
        dict  hmdb_id → {
            'name': str,
            'shifts': [ {ppm, mult, j (float|None), protons} , ... ]
        }
        Apenas metabolitos com ≥2 shifts (necessário para ≥1 cross-peak).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Verificar que a tabela existe
    tables = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if 'shifts_1h' not in tables:
        conn.close()
        raise ValueError(
            f"A DB {db_path} não tem tabela 'shifts_1h' "
            "(necessária para anotação COSY).")

    cur.execute("""
        SELECT s.hmdb_id, met.name, s.ppm, s.multiplicity,
               s.j_coupling, s.protons
        FROM shifts_1h s
        JOIN metabolites met ON s.hmdb_id = met.hmdb_id
        WHERE s.ppm IS NOT NULL
        ORDER BY s.hmdb_id, s.ppm
    """)
    rows = cur.fetchall()
    conn.close()

    lib = {}
    for r in rows:
        hid = r['hmdb_id']
        if hid not in lib:
            lib[hid] = {'name': r['name'], 'shifts': []}
        try:
            j = float(r['j_coupling']) if r['j_coupling'] not in (None, '') else None
        except (ValueError, TypeError):
            j = None
        lib[hid]['shifts'].append({
            'ppm': float(r['ppm']),
            'mult': str(r['multiplicity'] or ''),
            'j': j,
            'protons': r['protons'],
        })

    # Manter apenas metabolitos com ≥2 shifts distintos em ppm
    out = {}
    for hid, entry in lib.items():
        ppms = sorted({round(s['ppm'], 4) for s in entry['shifts']})
        if len(ppms) >= 2:
            out[hid] = entry
    return out


def _share_j(shifts_a, shifts_b, j_tol=0.5):
    """True se algum shift de A e algum de B partilham um valor de J (±j_tol Hz)."""
    ja = [s['j'] for s in shifts_a if s['j'] is not None]
    jb = [s['j'] for s in shifts_b if s['j'] is not None]
    for x in ja:
        for y in jb:
            if abs(x - y) <= j_tol:
                return True
    return False


def expected_crosspeaks(shifts, j_tol=0.5, fallback=True):
    """
    Prevê os cross-peaks COSY de um metabolito a partir da sua lista de shifts.

    Em COSY, um cross-peak aparece entre dois protões acoplados por J. Dois
    protões acoplados reportam o MESMO valor de J — por isso pares de shifts
    distintos que partilham um J (±j_tol) são cross-peaks previstos. Singletos
    (sem J) não produzem cross-peaks e são correctamente ignorados.

    fallback=True: metabolitos sem qualquer informação de J usam todos os pares
    de shifts distintos (menor confiança, mas continuam anotáveis).

    Returns:
        list[ (ppm_a, ppm_b, j_confirmed) ]  com ppm_a < ppm_b.

    NOTA (limitação): partilhar J é condição necessária mas não suficiente de
    acoplamento — dois pares independentes podem ter o mesmo J por acaso. Sem
    grafo de conectividade explícito isto sobre-prevê ligeiramente; j_tol controla.
    """
    from collections import defaultdict
    j_by_ppm = defaultdict(set)
    all_ppms = set()
    for s in shifts:
        p = round(s['ppm'], 4)
        all_ppms.add(p)
        if s['j'] is not None:
            j_by_ppm[p].add(s['j'])

    coupled_ppms = sorted(j_by_ppm.keys())
    pairs = []
    seen = set()
    for i in range(len(coupled_ppms)):
        for k in range(i + 1, len(coupled_ppms)):
            a, b = coupled_ppms[i], coupled_ppms[k]
            if abs(a - b) < 1e-6:
                continue
            if any(abs(x - y) <= j_tol
                   for x in j_by_ppm[a] for y in j_by_ppm[b]):
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    pairs.append((a, b, True))

    # Fallback: sem nenhum J → todos os pares de ppm distintos
    if not pairs and fallback:
        dist = sorted(all_ppms)
        for i in range(len(dist)):
            for k in range(i + 1, len(dist)):
                a, b = dist[i], dist[k]
                if abs(a - b) >= 1e-6:
                    pairs.append((a, b, False))
    return pairs


def _find_observed(crosspeaks, a, b, tol_f2, tol_f1):
    """Procura um cross-peak observado perto do par esperado (a,b), em qualquer
    orientação (F2,F1). Devolve (cp, dist) do mais próximo, ou (None, None)."""
    best, best_d = None, None
    for cp in crosspeaks:
        f2, f1 = cp['f2'], cp['f1']
        for (xa, xb) in ((a, b), (b, a)):
            if abs(f2 - xa) <= tol_f2 and abs(f1 - xb) <= tol_f1:
                d = float(np.hypot(f2 - xa, f1 - xb))
                if best_d is None or d < best_d:
                    best, best_d = cp, d
    return best, best_d


def annotate_cosy(crosspeaks, db_path, tol_f2=0.03, tol_f1=0.03,
                  j_tol=0.5, require_j=False, min_crosspeaks=1):
    """
    Anota cross-peaks COSY contra a biblioteca de shifts ¹H (método COLMAR).

    Para cada metabolito prevê o seu fingerprint COSY (pares de protões
    acoplados por J) e mede que FRACÇÃO desse fingerprint aparece no espectro.
    Coverage é assim relativa ao próprio metabolito (não ao total da mistura),
    o que dá valores significativos para compostos realmente presentes.

    Args:
        crosspeaks:     list[{f2, f1, intensity}]  (cross-peaks observados)
        db_path:        DB SQLite com tabela shifts_1h
        tol_f2, tol_f1: tolerância (ppm) em cada eixo
        j_tol:          tolerância (Hz) para emparelhar shifts por J
        require_j:      se True, ignora o fallback sem-J (só fingerprints por J)
        min_crosspeaks: nº mínimo de cross-peaks do fingerprint que têm de ser
                        encontrados para o metabolito ser reportado

    Returns:
        list[dict] ordenada por score desc:
            {
              'hmdb_id', 'name',
              'n_matched'    : pares do fingerprint encontrados,
              'n_total'      : pares do fingerprint previstos (do metabolito),
              'coverage'     : n_matched / n_total,
              'n_j_confirmed': pares encontrados que são por-J,
              'score'        : coverage ponderada por proximidade e J,
              'matches'      : [ {f2,f1,ppm_a,ppm_b,j_confirmed,dist} ]
            }
    """
    if not crosspeaks:
        return []

    lib = load_shift_library(db_path)
    results = []
    max_tol = max(tol_f2, tol_f1)

    for hid, entry in lib.items():
        expected = expected_crosspeaks(
            entry['shifts'], j_tol=j_tol, fallback=not require_j)
        if not expected:
            continue

        matches = []
        for (a, b, j_conf) in expected:
            cp, dist = _find_observed(crosspeaks, a, b, tol_f2, tol_f1)
            if cp is None:
                continue
            matches.append({
                'f2': cp['f2'], 'f1': cp['f1'],
                'ppm_a': a, 'ppm_b': b,
                'j_confirmed': j_conf, 'dist': dist,
            })

        n_matched = len(matches)
        if n_matched < min_crosspeaks:
            continue

        n_total = len(expected)
        n_j = sum(1 for m in matches if m['j_confirmed'])
        coverage = n_matched / n_total

        prox = np.mean([1.0 - min(m['dist'] / (max_tol * 1.5), 1.0)
                        for m in matches]) if matches else 0.0
        j_bonus = 1.0 + 0.5 * (n_j / n_matched)
        score = coverage * float(prox) * j_bonus

        results.append({
            'hmdb_id': hid, 'name': entry['name'],
            'n_matched': n_matched, 'n_total': n_total,
            'coverage': coverage, 'n_j_confirmed': n_j,
            'score': float(score), 'matches': matches,
        })

    results.sort(key=lambda r: -r['score'])
    return results
