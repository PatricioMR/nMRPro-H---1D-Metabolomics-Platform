"""
Validação com HMDB_Total_v01.db contra espectros MMNX Exp 2.
Schema: shifts_1h usa bmrb_id (não hmdb_id). Caffeine não existe.
Compara peak detection + regras de qualidade ANTES e DEPOIS dos fixes.
"""
import sys, os, sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bruker_loader import load_bruker_dataset
from processing.peak_detection import detect_peaks

SPECTRA_ROOT = r"G:\My Drive\Mestrado\BEPE\NMR 042426"
DB_PATH      = os.path.join("databases", "HMDB_Total_v01.db")
EXPNO        = 2
TOLERANCE    = 0.05

CONFIGS = {
    'ANTES (dj_bad=0.20)': {'dj_gold': 0.20, 'dj_bad': 0.20, 'good_score': 0.80},
    'DEPOIS (dj_bad=0.50)': {'dj_gold': 0.20, 'dj_bad': 0.50, 'good_score': 0.80},
}

SINGLET_MULTS = {'s', 'singlet', '', 'm', 'multiplet', 'br s', 'brs'}

# Metabolitos esperados (presentes na amostra MMNX)
EXPECTED = {
    'Taurine', 'D-Glucose', 'Itaconic acid', 'L-Arabinose',
    'L-Cysteine', 'L-Glutamic acid', 'Mannitol'
}


def load_reference(db_path):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    rows = cur.execute(
        "SELECT s.bmrb_id, m.name, s.ppm, s.multiplicity, s.j_coupling "
        "FROM shifts_1h s JOIN metabolites m ON s.bmrb_id = m.bmrb_id "
        "ORDER BY m.name, s.ppm DESC"
    ).fetchall()
    conn.close()
    ref = {}
    for bmrb, name, ppm, mult, j in rows:
        ref.setdefault(name, []).append({'bmrb': bmrb, 'ppm': ppm, 'mult': mult or 's', 'j': j})
    return ref


def simulate_s_dj(dj_hz, mult):
    if mult in SINGLET_MULTS or dj_hz is None:
        return 1.0
    lo, hi = 0.50, 3.00
    if dj_hz < lo:
        return 1.0
    if dj_hz > hi:
        return 0.0
    return 1.0 - (dj_hz - lo) / (hi - lo)


def simulate_score(ppm_delta, dj_hz, mult, tolerance=0.05):
    s_ppm    = np.exp(-0.5 * (ppm_delta / (tolerance / 2)) ** 2)
    s_dj     = simulate_s_dj(dj_hz, mult)
    s_cosine = 0.85
    s_found  = 1.0
    s_rri    = 0.80
    if mult in SINGLET_MULTS:
        return 0.30 * s_cosine + 0.40 * s_ppm + 0.15 * s_found + 0.15 * s_rri
    return 0.60 * s_dj + 0.15 * s_cosine + 0.10 * s_ppm + 0.05 * s_found + 0.10 * s_rri


def apply_rules(met_name, shifts, det_ppms, params, all_singlet):
    results = []
    for s in shifts:
        ref_ppm = s['ppm']
        mult    = str(s['mult'] or 's').lower().strip()

        if not det_ppms:
            results.append((ref_ppm, mult, 'bad', 'no peaks'))
            continue

        diffs   = [abs(dp - ref_ppm) for dp in det_ppms]
        min_d   = min(diffs)

        if min_d > TOLERANCE:
            near = det_ppms[int(np.argmin(diffs))]
            results.append((ref_ppm, mult, 'bad',
                             f'not found (nearest {near:.4f} Δ={min_d*1000:.0f}mppm)'))
            continue

        # simulated ΔJ — 0.25 Hz conservador para amostra biológica
        sim_dj = 0.25 if mult not in SINGLET_MULTS else None
        score  = simulate_score(min_d, sim_dj, mult)

        # Rule 0
        if mult not in SINGLET_MULTS and sim_dj is not None:
            if sim_dj >= params['dj_bad']:
                results.append((ref_ppm, mult, 'bad',
                                 f'Rule0: dj={sim_dj:.2f}>=dj_bad={params["dj_bad"]:.2f}'))
                continue

        # Rule 1 (singlet baixo score)
        if mult in SINGLET_MULTS and score < 0.50:
            results.append((ref_ppm, mult, 'bad', f'Rule1: score={score:.2f}'))
            continue

        # Rule 2 gold
        if mult not in SINGLET_MULTS and sim_dj is not None:
            if sim_dj < params['dj_gold'] and min_d < 0.05:
                results.append((ref_ppm, mult, 'good',
                                 f'Rule2-gold: dj={sim_dj:.2f}'))
                continue

        # Rules 3/4
        if score >= params['good_score'] and not all_singlet:
            results.append((ref_ppm, mult, 'good', f'Rule3/4: score={score:.2f}'))
            continue

        # Rule 8 singlet-only
        if all_singlet:
            if score >= params['good_score']:
                results.append((ref_ppm, mult, 'good',
                                 f'Rule8-fix: score={score:.2f}'))
            else:
                results.append((ref_ppm, mult, 'poor',
                                 f'Rule8: score={score:.2f}'))
            continue

        results.append((ref_ppm, mult, 'yellow', f'score={score:.2f}'))

    return results


def main():
    print("=" * 70)
    print("VALIDAÇÃO — HMDB_Total_v01.db × MMNX Exp 2")
    print("=" * 70)

    # Carregar espectros
    ppm, intensities, sample_names, _ = load_bruker_dataset(
        SPECTRA_ROOT, expno=EXPNO, procno=1,
        ppm_min=0.5, ppm_max=9.5, n_points=32768)
    mmnx_idx = [i for i, n in enumerate(sample_names) if 'MMNX' in n.upper()]
    peaks = detect_peaks(ppm, intensities[:, mmnx_idx], min_snr=3.0,
                         min_distance=0.005, baseline_correct=True, global_align=True)
    det_ppms = [p['ppm'] for p in peaks]
    snr_map  = {p['ppm']: p.get('snr', 0) for p in peaks}
    print(f"Picos detetados: {len(det_ppms)}")

    # Carregar referência
    ref = load_reference(DB_PATH)
    print(f"DB: {len(ref)} metabolitos, "
          f"{sum(len(v) for v in ref.values())} shifts totais\n")

    # Só metabolitos esperados nas amostras
    ref_expected = {k: v for k, v in ref.items() if k in EXPECTED}
    missing_from_db = EXPECTED - set(ref.keys())
    if missing_from_db:
        print(f"AVISO — Não encontrados na DB: {missing_from_db}\n")

    for cfg_name, params in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"{cfg_name}")
        print(f"  dj_bad={params['dj_bad']} Hz  |  dj_gold={params['dj_gold']} Hz  |  good_score={params['good_score']}")
        print(f"{'='*70}")

        total_shifts = total_good = 0
        summary = []

        for met_name in sorted(EXPECTED):
            if met_name not in ref:
                print(f"\n  ✗ AUSENTE DA DB  {met_name}")
                summary.append((met_name, 0, 0, 0, 'ABSENT'))
                continue

            shifts     = ref[met_name]
            all_sing   = all(s['mult'] in SINGLET_MULTS for s in shifts)
            qr         = apply_rules(met_name, shifts, det_ppms, params, all_sing)
            n_good     = sum(1 for _, _, q, _ in qr if q == 'good')
            pct        = n_good / len(qr) * 100 if qr else 0
            status     = '✓ QUANT' if n_good > 0 else '✗ MISS'
            summary.append((met_name, n_good, len(qr), pct, status))
            total_shifts += len(qr)
            total_good   += n_good

            print(f"\n  {status}  {met_name}  ({n_good}/{len(qr)} shifts good, {pct:.0f}%)")
            for ref_ppm, mult, quality, reason in qr:
                sym = {'+': '+', 'good': '+', 'bad': 'x', 'poor': '~', 'yellow': '?'}.get(quality, '?')
                if quality == 'good':
                    sym = '+'
                elif quality == 'bad':
                    sym = 'x'
                near_ppms = sorted(det_ppms, key=lambda d: abs(d - ref_ppm))[:1]
                near = f"det={near_ppms[0]:.4f} SNR={snr_map.get(near_ppms[0],0):.0f}" if near_ppms else ""
                print(f"    {sym} {ref_ppm:.4f} [{mult:4s}]  {near:<30} -> {quality:6s}  ({reason})")

        print(f"\n  TOTAL: {total_good}/{total_shifts} shifts good")
        print(f"  Metabolitos quantificáveis: "
              f"{sum(1 for _,ng,nt,_,s in summary if ng>0 and s!='ABSENT')}/"
              f"{sum(1 for _,_,_,_,s in summary if s!='ABSENT')}")

    # Coverage vs ValidationPipe
    print(f"\n{'='*70}")
    print("NOTA: DB coverage vs ValidationPipe.db")
    print(f"{'='*70}")
    vp_conn = sqlite3.connect("databases/HMDB_ValidationPipe.db")
    vp_cur  = vp_conn.cursor()
    for met in sorted(EXPECTED):
        vp_rows = vp_cur.execute(
            "SELECT COUNT(*) FROM shifts_1h s "
            "JOIN metabolites m ON s.hmdb_id=m.hmdb_id WHERE m.name=?", (met,)
        ).fetchone()[0]
        tot_rows = sum(len(v) for k, v in ref.items() if k == met)
        print(f"  {met:<25}: ValidationPipe={vp_rows:2d} shifts | Total={tot_rows:2d} shifts")
    vp_conn.close()


if __name__ == "__main__":
    main()
