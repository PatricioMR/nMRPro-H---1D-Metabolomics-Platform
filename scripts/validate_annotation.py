"""
Validation script: loads MMNX Exp 2 Bruker spectra, detects peaks,
searches HMDB_ValidationPipe.db, and reports hit/miss per metabolite.

Usage:
    python scripts/validate_annotation.py
"""
import sys
import os
import sqlite3
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bruker_loader import load_bruker_dataset
from processing.peak_detection import detect_peaks

# ── Config ────────────────────────────────────────────────────────────────────

SPECTRA_ROOT = r"G:\My Drive\Mestrado\BEPE\NMR 042426"
DB_PATH      = os.path.join("databases", "HMDB_ValidationPipe.db")
EXPNO        = 2
PROCNO       = 1
TOLERANCE    = 0.05   # ppm — same as app default
MIN_SNR      = 3.0

# Tolerâncias progressivas para diagnóstico
TOL_LEVELS   = [0.03, 0.05, 0.08, 0.10, 0.15]

# ── Load reference shifts from DB ─────────────────────────────────────────────

def load_reference(db_path):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    rows = cur.execute(
        "SELECT s.hmdb_id, m.name, s.ppm, s.multiplicity, s.j_coupling, s.intensity "
        "FROM shifts_1h s JOIN metabolites m ON s.hmdb_id = m.hmdb_id "
        "ORDER BY m.name, s.ppm DESC"
    ).fetchall()
    conn.close()
    ref = {}  # name -> list of {ppm, mult, j, intensity}
    for hmdb_id, name, ppm, mult, j, intensity in rows:
        ref.setdefault(name, []).append({
            'hmdb_id': hmdb_id, 'ppm': ppm,
            'mult': mult, 'j': j, 'intensity': intensity
        })
    return ref


# ── Match detected peaks against reference ────────────────────────────────────

def best_match_distance(detected_ppms, ref_shifts, tolerance):
    """
    For each reference shift, find closest detected peak within tolerance.
    Returns (n_matched, n_total, matched_details).
    """
    matched = []
    missed  = []
    for s in ref_shifts:
        diffs = [abs(dp - s['ppm']) for dp in detected_ppms]
        if not diffs:
            missed.append(s)
            continue
        min_diff = min(diffs)
        closest  = detected_ppms[int(np.argmin(diffs))]
        if min_diff <= tolerance:
            matched.append({'ref_ppm': s['ppm'], 'det_ppm': closest,
                            'delta': min_diff, 'mult': s['mult'],
                            'j': s['j'], 'rel_int': s['intensity']})
        else:
            missed.append(s)
    return matched, missed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NMR VALIDATION — MMNX Exp 2  ×  HMDB_ValidationPipe.db")
    print("=" * 70)

    # 1. Load spectra
    print(f"\n[1] Loading Bruker spectra from {SPECTRA_ROOT} (expno={EXPNO})")
    ppm, intensities, sample_names, _ = load_bruker_dataset(
        SPECTRA_ROOT, expno=EXPNO, procno=PROCNO,
        ppm_min=0.5, ppm_max=9.5, n_points=32768
    )
    # Keep only MMNX samples
    mmnx_idx = [i for i, n in enumerate(sample_names) if 'MMNX' in n.upper()]
    if not mmnx_idx:
        mmnx_idx = list(range(len(sample_names)))
    mmnx_names      = [sample_names[i] for i in mmnx_idx]
    mmnx_intensities = intensities[:, mmnx_idx]
    print(f"   Samples: {mmnx_names}")

    # 2. Detect peaks on mean MMNX spectrum
    print(f"\n[2] Detecting peaks (SNR ≥ {MIN_SNR})")
    peaks = detect_peaks(
        ppm, mmnx_intensities,
        min_snr=MIN_SNR,
        min_distance=0.005,
        baseline_correct=True,
        baseline_method='airPLS',
        global_align=True
    )
    det_ppms = [p['ppm'] for p in peaks]
    print(f"   Detected: {len(det_ppms)} peaks")
    print(f"   PPM range covered: {min(det_ppms):.3f} – {max(det_ppms):.3f}")

    # 3. Load reference
    ref = load_reference(DB_PATH)
    print(f"\n[3] Reference: {len(ref)} metabolites, "
          f"{sum(len(v) for v in ref.values())} shifts")

    # 4. Match at default tolerance + diagnostic sweep
    print(f"\n[4] Matching at tolerance = {TOLERANCE} ppm")
    print("-" * 70)

    results = {}
    for name, shifts in sorted(ref.items()):
        matched, missed = best_match_distance(det_ppms, shifts, TOLERANCE)
        cov = len(matched) / len(shifts) * 100
        results[name] = {'matched': matched, 'missed': missed,
                         'n_total': len(shifts), 'coverage': cov}

        status = "✓ HIT" if cov >= 50 else ("~ PARTIAL" if cov > 0 else "✗ MISS")
        print(f"\n  {status}  {name}  ({len(matched)}/{len(shifts)} shifts, {cov:.0f}% coverage)")
        for m in matched:
            print(f"    + ref {m['ref_ppm']:.4f} → det {m['det_ppm']:.4f}  "
                  f"Δ={m['delta']*1000:.1f}mppm  [{m['mult']}]")
        for s in missed:
            # Find nearest detected peak for diagnostic
            if det_ppms:
                diffs   = [abs(dp - s['ppm']) for dp in det_ppms]
                nearest = det_ppms[int(np.argmin(diffs))]
                gap     = min(diffs)
                print(f"    - MISS ref {s['ppm']:.4f}  [{s['mult']}]  "
                      f"nearest_det={nearest:.4f} (Δ={gap*1000:.1f}mppm)")
            else:
                print(f"    - MISS ref {s['ppm']:.4f}  [{s['mult']}]  no peaks detected")

    # 5. Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Metabolite':<25} {'Shifts':>6} {'Matched':>7} {'Coverage':>9} {'Status'}")
    print("  " + "-" * 60)
    total_shifts   = 0
    total_matched  = 0
    for name, r in sorted(results.items()):
        cov    = r['coverage']
        status = "HIT" if cov >= 50 else ("PARTIAL" if cov > 0 else "MISS")
        print(f"  {name:<25} {r['n_total']:>6} {len(r['matched']):>7} {cov:>8.0f}%  {status}")
        total_shifts  += r['n_total']
        total_matched += len(r['matched'])

    overall = total_matched / total_shifts * 100 if total_shifts else 0
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<25} {total_shifts:>6} {total_matched:>7} {overall:>8.0f}%")

    # 6. Tolerance sweep — shows how many more hits open up with relaxed tol
    print("\n[5] Tolerance sweep (all metabolites)")
    print(f"  {'Tolerance':>10}  {'Matched shifts':>15}  {'Coverage':>10}")
    for tol in TOL_LEVELS:
        nm = sum(
            len(best_match_distance(det_ppms, shifts, tol)[0])
            for shifts in ref.values()
        )
        print(f"  {tol:>10.3f}  {nm:>15}  {nm/total_shifts*100:>9.1f}%")

    # 7. Unmatched detected peaks (potential unknown signals)
    ref_all_ppms = [s['ppm'] for shifts in ref.values() for s in shifts]
    orphan_peaks = []
    for dp in det_ppms:
        if all(abs(dp - rp) > TOLERANCE for rp in ref_all_ppms):
            orphan_peaks.append(dp)

    print(f"\n[6] Detected peaks NOT in reference ({len(orphan_peaks)} orphans):")
    snr_map = {p['ppm']: p.get('snr', 0) for p in peaks}
    for op in sorted(orphan_peaks, key=lambda x: -snr_map.get(x, 0))[:20]:
        print(f"    {op:.4f} ppm  SNR={snr_map.get(op, 0):.1f}")
    if len(orphan_peaks) > 20:
        print(f"    ... and {len(orphan_peaks)-20} more")

    print("\nDone.")


if __name__ == "__main__":
    main()
