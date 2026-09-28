"""What each confidence level has been measured to mean, written for the
report.

Runs the crowded reference bench (``validation.crowded_bench``: the 18
analytes of the reference study, then filled to 50 and 100 with analytes
drawn from the reference database at 0.05-1 mM) on three seeds, with the
application's defaults, and records per tube and per confidence level how
many rows were real analytes, how many were not, and how far the real
ones' amounts were from the truth. The report prints these beside every
row's dots: a level is a measured precision, not a fraction of three.

    python -m validation.quality_reference

Writes ``data/quality_reference.json``, keyed by spectrometer field. The
bench synthesises its spectra at ``mixture_bench.MHZ``; a field enters
the file only once it has been measured.
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

from app_paths import APP_VERSION
from validation import mixture_bench as mb
from validation.crowded_bench import CONC_RANGE_MM, prepared
from validation.fresh_run import quantify_prepared

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'data', 'quality_reference.json')
TUBES = (18, 50, 100)
SEEDS = (0, 1, 2)


def _row_error(r, truth) -> float | None:
    """Relative error of a real analyte's amount: the median over the
    groups of (median measured / true - 1)."""
    t = truth.get(r['hmdb_id'])
    if not t:
        return None
    e = [float(np.median(v)) / t[g] - 1.0
         for g, v in (r.get('_concs_by_group') or {}).items()
         if t.get(g) and v]
    return float(np.median(e)) if e else None


def tube_stats(n) -> dict:
    """Pooled over SEEDS: per level the real and false rows and the real
    rows' errors; overall recall and precision at two dots or more."""
    levels = {lv: {'real': 0, 'false': 0, 'errors': []} for lv in (1, 2, 3)}
    found = truth_n = 0
    all_err = []
    for seed in SEEDS:
        prep, truth = prepared(n, seed)
        rows = quantify_prepared(prep, mb.SAMPLES,
                                 standard_ppm=mb.STANDARD_PPM,
                                 standard_nh=mb.STANDARD_NH,
                                 standard_conc=mb.STANDARD_MM,
                                 standard_name='DSS')
        truth_n += len(truth)
        for r in rows:
            if r.get('_not_quantified') or r.get('_is_is'):
                continue
            lv = min(max(int(r.get('_confidence') or 1), 1), 3)
            if r['hmdb_id'] in truth:
                levels[lv]['real'] += 1
                found += 1
                e = _row_error(r, truth)
                if e is not None:
                    levels[lv]['errors'].append(e)
                    all_err.append(e)
            else:
                levels[lv]['false'] += 1
    out = {'analytes': n, 'runs': len(SEEDS), 'levels': {}}
    for lv, d in levels.items():
        tot = d['real'] + d['false']
        err = np.abs(d['errors'])
        out['levels'][str(lv)] = {
            'real': d['real'], 'false': d['false'],
            'precision': d['real'] / tot if tot else None,
            'error_median': float(np.median(err)) if err.size else None,
            'error_p90': float(np.percentile(err, 90)) if err.size else None,
        }
    tp = levels[2]['real'] + levels[3]['real']
    fp = levels[2]['false'] + levels[3]['false']
    all_abs = np.abs(all_err)
    out.update({
        'found': found / truth_n,
        'recall_2plus': tp / truth_n,
        'precision_2plus': tp / (tp + fp) if tp + fp else None,
        'false_per_run': (sum(d['false'] for d in levels.values())
                          / len(SEEDS)),
        'error_median': float(np.median(all_abs)) if all_abs.size else None,
        'error_p90': (float(np.percentile(all_abs, 90))
                      if all_abs.size else None),
    })
    return out


def main():
    field = str(int(mb.MHZ))
    doc = {}
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            doc = json.load(f)
    doc[field] = {
        'software': APP_VERSION,
        'date': datetime.date.today().isoformat(),
        'database': os.path.basename(mb.DB_DEFAULT),
        'design': {
            'field_mhz': mb.MHZ, 'linewidth_hz': mb.LINEWIDTH_HZ,
            'samples': len(mb.SAMPLES), 'groups': 2,
            'noise': f'lactate doublet at SNR {mb.NOISE_LACTATE_SNR:.0f}',
            'concentration_range_mm': list(CONC_RANGE_MM),
            'reference_analytes': len(mb.PANEL), 'seeds': len(SEEDS),
            'internal_standard': f'DSS {mb.STANDARD_MM} mM',
        },
        'tubes': {str(n): tube_stats(n) for n in TUBES},
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    print('->', OUT)


if __name__ == '__main__':
    main()
