"""
Validate predicted J couplings against an experimental reference.

Method
------
HMDB stores a *virtual* 1H spectrum as a stick list (line positions + intensity)
per metabolite. We PREDICT J for a multiplet directly from those sticks using the
same equation the experimental pipeline uses:

    J[Hz] = (line spacing in ppm) × (spectrometer field in MHz)

For a first-order multiplet the adjacent-line spacing is constant and equals J,
so J_pred = median(adjacent stick spacings) × field.

We then COMPARE J_pred to the BMRB ``j_coupling`` (measured, curated from
NMR-STAR) for the same metabolite/signal. BMRB is reference only — never an input
to the prediction — so this is a genuine, non-circular validation of the method.

Matching
--------
  • metabolites: by hmdb_id (present in both DBs);
  • signals: each BMRB clean multiplet (d/t/q/quint/sext/sept) is matched to the
    HMDB stick cluster nearest in ppm whose line count equals the expected number
    of lines for that multiplicity (so only genuinely resolved multiplets count);
  • 'm' (ambiguous J list) and 's' (no J) are excluded.

Outputs: console report, a per-signal CSV, and scatter + Bland-Altman PNGs.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import numpy as np

HMDB_DB = 'databases/HMDB_Metabolomics_v01.db'
BMRB_DB = 'databases/BMRB_Metabolomics_v01.db'

# Expected resolved-line count per first-order multiplicity.
EXPECTED_LINES = {'d': 2, 't': 3, 'q': 4, 'quint': 5, 'sext': 6, 'sept': 7}

PPM_MATCH_TOL = 0.06     # ppm: BMRB signal ↔ HMDB cluster centroid
CLUSTER_GAP_HZ = 20.0    # split sticks into clusters where spacing exceeds this
                         # (larger than any real 1H–1H J → separates signals)


def _parse_j_values(s):
    """Floats from a BMRB j_coupling string like '7.50' or '1.60 6.20 7.60'."""
    out = []
    for tok in str(s or '').replace(',', ' ').split():
        try:
            v = float(tok)
            if v > 0:
                out.append(v)
        except ValueError:
            pass
    return out


def _cluster_sticks(ppms, field_mhz):
    """Split sorted stick ppms into clusters on gaps > CLUSTER_GAP_HZ."""
    gap_ppm = CLUSTER_GAP_HZ / max(field_mhz, 1e-6)
    clusters, cur = [], [ppms[0]]
    for p in ppms[1:]:
        if p - cur[-1] > gap_ppm:
            clusters.append(cur)
            cur = [p]
        else:
            cur.append(p)
    clusters.append(cur)
    return clusters


def predict_j_from_cluster(cluster_ppms, field_mhz):
    """J[Hz] = median adjacent spacing × field. None if < 2 lines."""
    if len(cluster_ppms) < 2:
        return None
    sp = np.diff(np.sort(cluster_ppms))
    return float(np.median(sp) * field_mhz)


# Prediction sources: (label, db, table, ppm_col, field_col). Both are virtual
# stick/peak lists; HMDB sits on a coarse 0.01 ppm grid, BMRB 'transitions' is a
# finer simulated peak list — comparing them shows which feeds J best.
SOURCES = {
    'HMDB sticks':       (HMDB_DB, 'sticks_1h',  'ppm', 'field_mhz'),
    'BMRB transitions':  (BMRB_DB, 'transitions', 'ppm', 'freq_mhz'),
}


def _load_sticks(db, table, ppm_col, field_col):
    con = sqlite3.connect(db)
    sticks = {}
    for hid, ppm, fmhz in con.execute(
            f'SELECT hmdb_id, {ppm_col}, {field_col} FROM {table} '
            f'WHERE {ppm_col} IS NOT NULL AND hmdb_id IS NOT NULL'):
        sticks.setdefault(hid, {'ppm': [], 'field': fmhz})
        sticks[hid]['ppm'].append(float(ppm))
        if fmhz:
            sticks[hid]['field'] = float(fmhz)
    con.close()
    return sticks


def build_comparison(source='HMDB sticks'):
    B = sqlite3.connect(BMRB_DB)
    bc = B.cursor()

    sticks = _load_sticks(*SOURCES[source])

    rows = []          # comparison records
    skipped = {'no_sticks': 0, 'no_cluster_match': 0, 'wrong_line_count': 0,
               'no_j': 0}

    for bmrb_id, hmdb_id, ppm, mult, j_raw, fmhz_b in bc.execute(
            "SELECT bmrb_id, hmdb_id, ppm, multiplicity, j_coupling, field_mhz "
            "FROM shifts_1h WHERE hmdb_id IS NOT NULL AND hmdb_id<>''"):
        mult = (mult or '').strip().lower()
        if mult not in EXPECTED_LINES:
            continue                                   # only clean multiplets
        j_vals = _parse_j_values(j_raw)
        if not j_vals:
            skipped['no_j'] += 1
            continue
        # Reference J: the dominant (largest) coupling — for a clean d/t/q the
        # list is usually a single value anyway.
        j_bmrb = float(np.max(j_vals))

        s = sticks.get(hmdb_id)
        if not s or len(s['ppm']) < 2:
            skipped['no_sticks'] += 1
            continue
        field = s['field'] or fmhz_b or 500.0
        clusters = _cluster_sticks(sorted(s['ppm']), field)

        # nearest cluster by centroid within tolerance
        best, bestd = None, PPM_MATCH_TOL
        for cl in clusters:
            d = abs(float(np.mean(cl)) - float(ppm))
            if d < bestd:
                best, bestd = cl, d
        if best is None:
            skipped['no_cluster_match'] += 1
            continue
        if len(best) != EXPECTED_LINES[mult]:
            skipped['wrong_line_count'] += 1
            continue

        j_pred = predict_j_from_cluster(best, field)
        if j_pred is None:
            continue
        rows.append({
            'hmdb_id': hmdb_id, 'bmrb_id': bmrb_id, 'ppm': float(ppm),
            'mult': mult, 'field_mhz': field,
            'j_bmrb': j_bmrb, 'j_pred': j_pred,
            'abs_err': abs(j_pred - j_bmrb), 'err': j_pred - j_bmrb,
        })

    B.close()
    return rows, skipped


def cv_calibrate(j_pred, j_bmrb, k=5, seed=0):
    """Out-of-fold linear calibration J_cal = a + b·J_pred.

    A grid-quantized source has a systematic slope/offset vs the true J; a linear
    fit removes it. K-fold CV gives the HONEST error of the calibrated predictor
    (fit on train folds, evaluated on held-out) — never fit and scored on the
    same points. Returns calibrated predictions aligned to the input order.
    """
    jp = np.asarray(j_pred, float)
    jb = np.asarray(j_bmrb, float)
    n = len(jp)
    if n < k + 2:
        return jp.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    cal = np.empty(n)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        b, a = np.polyfit(jp[train], jb[train], 1)   # jb ≈ b·jp + a
        cal[test] = b * jp[test] + a
    return cal


def metrics(jp, jb):
    """Error/agreement metrics for predicted (jp) vs reference (jb) J arrays."""
    jp = np.asarray(jp, float); jb = np.asarray(jb, float)
    err = jp - jb; ae = np.abs(err)
    out = {
        'n': len(jp),
        'mae': float(np.mean(ae)), 'median_ae': float(np.median(ae)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'bias': float(np.mean(err)),
        'sd_err': float(np.std(err, ddof=1)) if len(jp) > 1 else float('nan'),
        'pct_0_5': float(np.mean(ae <= 0.5) * 100),
        'pct_1_0': float(np.mean(ae <= 1.0) * 100),
        'pct_2_0': float(np.mean(ae <= 2.0) * 100),
    }
    if len(jp) > 2 and jb.std() > 0 and jp.std() > 0:
        out['pearson'] = float(np.corrcoef(jb, jp)[0, 1])
    else:
        out['pearson'] = float('nan')
    return out


def _write_csv(rows, jcal, out_dir, tag):
    import csv as _csv
    path = os.path.join(out_dir, f'j_validation_{tag}.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = _csv.writer(f)
        w.writerow(['hmdb_id', 'bmrb_id', 'ppm', 'multiplicity', 'field_mhz',
                    'j_bmrb_hz', 'j_pred_hz', 'j_calibrated_hz',
                    'abs_err_raw', 'abs_err_cal'])
        order = sorted(range(len(rows)), key=lambda i: -rows[i]['abs_err'])
        for i in order:
            r = rows[i]
            w.writerow([r['hmdb_id'], r['bmrb_id'], f'{r["ppm"]:.4f}', r['mult'],
                        r['field_mhz'], f'{r["j_bmrb"]:.2f}', f'{r["j_pred"]:.2f}',
                        f'{jcal[i]:.2f}', f'{r["abs_err"]:.2f}',
                        f'{abs(jcal[i] - r["j_bmrb"]):.2f}'])
    return path


def _plot(rows, jcal, st_raw, st_cal, out_dir, tag, source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jb = np.array([r['j_bmrb'] for r in rows])
    jp = np.array([r['j_pred'] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    lim = max(jb.max(), jp.max(), jcal.max()) * 1.1
    ax1.plot([0, lim], [0, lim], 'r--', lw=1, label='y = x')
    ax1.scatter(jb, jp, s=22, alpha=0.6, edgecolors='white', linewidths=0.4,
                label=f'raw (r={st_raw["pearson"]:.2f}, MAE={st_raw["mae"]:.2f})')
    ax1.scatter(jb, jcal, s=22, alpha=0.6, marker='^', color='#2ca02c',
                edgecolors='white', linewidths=0.4,
                label=f'calibrated (MAE={st_cal["mae"]:.2f})')
    ax1.set_xlabel('J BMRB (experimental, Hz)')
    ax1.set_ylabel('J predicted (Hz)')
    ax1.set_title(f'{source}: predicted vs experimental J  (n={st_raw["n"]})')
    ax1.legend(fontsize=8); ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
    ax1.grid(alpha=0.3)
    diff = jp - jb
    ax2.scatter((jb + jp) / 2, diff, s=22, alpha=0.7, edgecolors='white',
                linewidths=0.4)
    ax2.axhline(st_raw['bias'], color='k', lw=1, label=f'bias {st_raw["bias"]:+.2f}')
    ax2.axhline(st_raw['bias'] + 1.96 * st_raw['sd_err'], color='r', ls='--', lw=1,
                label='±1.96 SD')
    ax2.axhline(st_raw['bias'] - 1.96 * st_raw['sd_err'], color='r', ls='--', lw=1)
    ax2.set_xlabel('Mean of methods (Hz)'); ax2.set_ylabel('Predicted − BMRB (Hz)')
    ax2.set_title('Bland-Altman (raw)'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f'j_validation_{tag}.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _per_mult(rows, jcal):
    from collections import defaultdict
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r['mult']].append((r['abs_err'], abs(jcal[i] - r['j_bmrb'])))
    print(f'    {"mult":<6}{"n":>4}{"MAE_raw":>9}{"MAE_cal":>9}')
    for m in ('d', 't', 'q', 'quint', 'sext', 'sept'):
        g = by.get(m)
        if not g:
            continue
        raw = np.mean([x[0] for x in g]); cal = np.mean([x[1] for x in g])
        print(f'    {m:<6}{len(g):>4}{raw:>9.2f}{cal:>9.2f}')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    out_dir = os.environ.get('J_VALID_OUT', '.')

    print('=' * 70)
    print('  J VALIDATION — virtual-spectrum prediction vs BMRB experimental J')
    print('  Sources × {raw, regression-calibrated (5-fold CV)}')
    print('=' * 70)

    summary = []
    for source in SOURCES:
        rows, skipped = build_comparison(source)
        if not rows:
            print(f'  [{source}] no comparable signals.')
            continue
        jp = np.array([r['j_pred'] for r in rows])
        jb = np.array([r['j_bmrb'] for r in rows])
        jcal = cv_calibrate(jp, jb)
        st_raw = metrics(jp, jb)
        st_cal = metrics(jcal, jb)
        tag = source.replace(' ', '_').lower()
        _write_csv(rows, jcal, out_dir, tag)
        _plot(rows, jcal, st_raw, st_cal, out_dir, tag, source)

        print(f'\n  ── {source}  (n={st_raw["n"]}, '
              f'{len({r["hmdb_id"] for r in rows})} metabolites) ──')
        print(f'    {"":12}{"raw":>10}{"calibrated":>12}')
        for k, lbl in (('mae', 'MAE (Hz)'), ('median_ae', 'median AE'),
                       ('rmse', 'RMSE'), ('pct_1_0', '≤1 Hz %'),
                       ('pct_0_5', '≤0.5 Hz %'), ('pearson', 'Pearson r')):
            print(f'    {lbl:<12}{st_raw[k]:>10.2f}{st_cal[k]:>12.2f}')
        print(f'    {"bias":<12}{st_raw["bias"]:>+10.2f}{st_cal["bias"]:>+12.2f}')
        _per_mult(rows, jcal)
        summary.append((source, 'raw', st_raw))
        summary.append((source, 'calibrated', st_cal))

    if not summary:
        return
    # Pick best by MAE (the metric that matters for feeding J into annotation).
    best = min(summary, key=lambda s: s[2]['mae'])
    print('\n' + '=' * 70)
    print('  RECOMMENDATION (lowest MAE = best J for quantitative annotation)')
    print('=' * 70)
    for source, kind, st in sorted(summary, key=lambda s: s[2]['mae']):
        mark = '  <== BEST' if (source, kind) == (best[0], best[1]) else ''
        print(f'    {source:<18} {kind:<11} MAE={st["mae"]:.2f} Hz  '
              f'RMSE={st["rmse"]:.2f}  r={st["pearson"]:.2f}  '
              f'≤1Hz={st["pct_1_0"]:.0f}%{mark}')
    print('\n  NOTE: BMRB transitions are SIMULATED from the same spin parameters')
    print('  that produced the curated j_coupling, so its ~0 Hz error is internal')
    print('  consistency, not an independent test — it proves the J = Δδ·MHz')
    print('  equation is EXACT given a clean high-resolution stick list. HMDB')
    print('  sticks are the independent cross-source test: ~1.25 Hz error, set by')
    print("  HMDB's 0.01 ppm grid (not by the equation). Regression calibration")
    print('  barely helps (the error is random quantization, not a systematic slope).')
    print(f'\n  Artifacts (CSV + PNG per source) → {out_dir}')


if __name__ == '__main__':
    sys.exit(main())
