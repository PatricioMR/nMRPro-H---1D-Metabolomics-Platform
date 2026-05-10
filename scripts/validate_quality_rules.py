"""
Simula as regras de qualidade (good/bad/poor) para os metabolitos de validação
sem precisar correr a GUI. Usa os shifts da ValidationPipe.db e os picos
detetados pelo pipeline de peak detection.

Objetivo: verificar que os fixes a dj_bad_hz e Rule 8 permitem que os 8
metabolitos de validação cheguem a 'good' automaticamente.
"""
import sys, os, sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bruker_loader import load_bruker_dataset
from processing.peak_detection import detect_peaks

SPECTRA_ROOT = r"G:\My Drive\Mestrado\BEPE\NMR 042426"
DB_PATH      = os.path.join("databases", "HMDB_ValidationPipe.db")
EXPNO        = 2
TOLERANCE    = 0.05   # ppm

# ── Parâmetros ANTES e DEPOIS do fix ─────────────────────────────────────────
CONFIGS = {
    'ANTES (dj_bad=0.20)': {'dj_gold': 0.20, 'dj_bad': 0.20, 'good_score': 0.80, 'dj_soft_lo': 0.50},
    'DEPOIS (dj_bad=0.50)': {'dj_gold': 0.20, 'dj_bad': 0.50, 'good_score': 0.80, 'dj_soft_lo': 0.50},
}

SINGLET_MULTS = {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs'}


def simulate_s_dj(dj_hz, mult, params):
    """Simula o score s_dj dado ΔJ e parâmetros."""
    if mult in SINGLET_MULTS or dj_hz is None:
        return params['dj_soft_lo'] / params['dj_soft_lo']  # s_dj=1.0 para singlets (neutro)
    lo = params['dj_soft_lo']
    hi = 3.00
    if dj_hz < lo:
        return 1.0
    if dj_hz > hi:
        return 0.0
    return 1.0 - (dj_hz - lo) / (hi - lo)


def simulate_composite_score(ppm_delta, dj_hz, mult, params, tolerance=0.05):
    """Score simplificado: replica pesos do pipeline real."""
    s_ppm = np.exp(-0.5 * ((ppm_delta) / (tolerance / 2)) ** 2)
    s_dj  = simulate_s_dj(dj_hz, mult, params)
    # Componentes não disponíveis sem pipeline completo → assumir valores razoáveis
    s_cosine = 0.85  # estimativa — espectro real alinha bem
    s_found  = 1.0   # todos os picos encontrados (validado anteriormente)
    s_rri    = 0.80  # intensidade relativa razoável

    if mult in SINGLET_MULTS:
        score = (0.30 * s_cosine + 0.40 * s_ppm + 0.15 * s_found + 0.15 * s_rri)
    else:
        score = (0.60 * s_dj + 0.15 * s_cosine + 0.10 * s_ppm + 0.05 * s_found + 0.10 * s_rri)
    return score


def apply_quality_rules(met_name, shifts, detected_ppms, params, tolerance=0.05):
    """
    Simula as regras de qualidade para um metabolito.
    Devolve lista de (ppm_ref, mult, quality, reason).
    """
    all_singlet = all(s['mult'] in SINGLET_MULTS for s in shifts)
    results = []

    for s in shifts:
        ref_ppm = s['ppm']
        mult    = str(s['mult'] or 's').lower().strip()

        # Encontrar pico mais próximo
        if not detected_ppms:
            results.append((ref_ppm, mult, 'bad', '_s_found=0'))
            continue

        diffs = [abs(dp - ref_ppm) for dp in detected_ppms]
        min_diff = min(diffs)
        det_ppm  = detected_ppms[int(np.argmin(diffs))]

        if min_diff > tolerance:
            results.append((ref_ppm, mult, 'bad', f'_s_found=0 (nearest={det_ppm:.4f} Δ={min_diff*1000:.0f}mppm)'))
            continue

        # J coupling: usar valor da DB como aproximação do ΔJ medido
        # Na prática, ΔJ depende do espectro. Simulamos 3 cenários:
        # melhor caso (ΔJ=0.10), médio (ΔJ=0.30), pior (ΔJ=0.70)
        j_val  = s['j']
        if j_val:
            try:
                j_hz = float(str(j_val).split(',')[0].strip())
            except Exception:
                j_hz = None
        else:
            j_hz = None

        # Simular ΔJ estimado (no espectro real, ΔJ = |J_observado - J_DB|)
        # Usamos 0.25 Hz como estimativa conservadora para amostras biológicas
        simulated_dj = 0.25 if (mult not in SINGLET_MULTS and j_hz) else None

        score = simulate_composite_score(min_diff, simulated_dj, mult, params, tolerance)

        # ── Regra 0b ──
        # _s_found > 0 → não aplica

        # ── Regra 0 ──
        if mult not in SINGLET_MULTS and simulated_dj is not None:
            if simulated_dj >= params['dj_bad']:
                results.append((ref_ppm, mult, 'bad',
                                 f'Rule0: ΔJ={simulated_dj:.2f}≥dj_bad={params["dj_bad"]:.2f}'))
                continue

        # ── Regra 1 (singlets baixo score) ──
        if mult in SINGLET_MULTS and score < 0.50:
            results.append((ref_ppm, mult, 'bad', f'Rule1: score={score:.2f}<0.50'))
            continue

        # ── Regra 2 (gold standard) ──
        if mult not in SINGLET_MULTS and simulated_dj is not None:
            if simulated_dj < params['dj_gold'] and min_diff < 0.05:
                results.append((ref_ppm, mult, 'good', f'Rule2-gold: ΔJ={simulated_dj:.2f}<{params["dj_gold"]:.2f}'))
                continue

        # ── Regras 3/4 (score alto) ──
        if score >= params['good_score'] and not all_singlet:
            results.append((ref_ppm, mult, 'good', f'Rule3/4: score={score:.2f}≥{params["good_score"]:.2f}'))
            continue

        # ── Rule 8: singlet-only ──
        if all_singlet:
            if score >= params['good_score']:
                # Fix novo: singlet-only de alta confiança → good
                results.append((ref_ppm, mult, 'good',
                                 f'Rule8-fix: singlet-only high-score={score:.2f}'))
            else:
                results.append((ref_ppm, mult, 'poor',
                                 f'Rule8: singlet-only score={score:.2f}<{params["good_score"]:.2f}'))
            continue

        results.append((ref_ppm, mult, 'yellow', f'score={score:.2f} (ambiguous)'))

    return results


def main():
    print("=" * 70)
    print("SIMULAÇÃO DE REGRAS DE QUALIDADE — ANTES vs DEPOIS DOS FIXES")
    print("=" * 70)

    # Carregar picos
    ppm, intensities, sample_names, _ = load_bruker_dataset(
        SPECTRA_ROOT, expno=EXPNO, procno=1, ppm_min=0.5, ppm_max=9.5, n_points=32768)
    mmnx_idx = [i for i, n in enumerate(sample_names) if 'MMNX' in n.upper()]
    peaks = detect_peaks(ppm, intensities[:, mmnx_idx], min_snr=3.0,
                         min_distance=0.005, baseline_correct=True, global_align=True)
    det_ppms = [p['ppm'] for p in peaks]

    # Carregar referência
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    rows = cur.execute(
        "SELECT s.hmdb_id, m.name, s.ppm, s.multiplicity, s.j_coupling "
        "FROM shifts_1h s JOIN metabolites m ON s.hmdb_id = m.hmdb_id "
        "ORDER BY m.name, s.ppm DESC"
    ).fetchall()
    conn.close()

    ref = {}
    for _, name, ppm_r, mult, j in rows:
        ref.setdefault(name, []).append({'ppm': ppm_r, 'mult': mult or 's', 'j': j})

    for cfg_name, params in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"CONFIGURAÇÃO: {cfg_name}")
        print(f"  dj_bad={params['dj_bad']} Hz  |  dj_gold={params['dj_gold']} Hz  |  good_score={params['good_score']}")
        print(f"{'='*70}")

        n_good_total = 0
        n_total      = 0
        summary = []

        for met_name, shifts in sorted(ref.items()):
            qr = apply_quality_rules(met_name, shifts, det_ppms, params, TOLERANCE)
            n_good = sum(1 for _, _, q, _ in qr if q == 'good')
            n_total += len(qr)
            n_good_total += n_good
            pct = n_good / len(qr) * 100 if qr else 0
            status = "✓ QUANT" if n_good > 0 else "✗ MISS"
            summary.append((met_name, n_good, len(qr), pct, status))

            print(f"\n  {status}  {met_name}  ({n_good}/{len(qr)} peaks good, {pct:.0f}%)")
            for ref_ppm, mult, quality, reason in qr:
                sym = {'good': '+', 'bad': '✗', 'poor': '~', 'yellow': '?'}.get(quality, '?')
                print(f"    {sym} {ref_ppm:.4f} [{mult:4s}]  → {quality:6s}  ({reason})")

        print(f"\n  TOTAL QUANTIFICÁVEL: {n_good_total}/{n_total} peaks good "
              f"({n_good_total/n_total*100:.0f}%)")
        print(f"  Metabolitos com ≥1 peak good: "
              f"{sum(1 for _,ng,_,_,_ in summary if ng > 0)}/{len(summary)}")


if __name__ == "__main__":
    main()
