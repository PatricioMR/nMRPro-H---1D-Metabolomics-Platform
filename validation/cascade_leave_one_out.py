"""Leave-one-out on the bench: how well does the cascade QUANTIFY, apart
from what it selects (ADR-028)?

For every analyte the bench put in, the others are held at the amounts the
election measured, that one is hidden, and the cascade is asked for it.
What comes out is compared with the truth and with the election's own
number. What this measured: an analyte's free share -- how much of its
listing falls outside what the quantified analytes account for -- is what
predicts the error (60% or more of it free: within 5%, worst 17%; less:
within 26%, worst 65%, and the ratio between groups moves as much). That
is why MIN_FREE_TO_QUANTIFY exists.

    python -m validation.cascade_leave_one_out
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from validation import mixture_bench as mb
from gui.stages._quant_adapter import (quantify_project, standard_areas_from_peak,
                                       project_field_mhz)
from gui.stages._annotation_helpers import ambiguous_peak_ranks
from processing.quant_cascade import cascade_missed
from processing.quant_concentration import standard_per_proton
from processing.annotation_scoring import study_drift

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    X, sigma, truth, foreign = mb.build_study(mb.DB_DEFAULT, seed=0)
    peaks = mb.detect_peaks(mb.PPM, X, noise_std=sigma, min_snr=6.0, min_distance=0.003,
                            baseline_correct=False, global_align=False, samples_info=mb.SAMPLES,
                            per_sample_detection=True, detect_shoulders=False,
                            min_group_presence=3, excluded_regions=mb.WATER)
    mets, params, _ = mb.annotate(mb.PPM, X, peaks, sigma, mb.DB_DEFAULT, patterns=False)
    amb = ambiguous_peak_ranks(mets, manual_quality={})
    ref = min(peaks, key=lambda p: abs(float(p['ppm']) - mb.STANDARD_PPM))
    rows = quantify_project(
        metabolites=mets, peaks=peaks, ppm=mb.PPM, intensities=X, samples_info=mb.SAMPLES,
        ref_peak=ref, ref_nh=mb.STANDARD_NH, ref_conc=mb.STANDARD_MM, ref_name='DSS',
        noise_sigma=sigma, ambiguous_ranks=amb, use_pqn=False, local_baseline=False,
        quantify_shapes=True, deconvolve=False, resolve_shared=True, cascade=False,
        annotation_params=params)
    lw = mb.measure_linewidth_ppm(mb.PPM, X, peaks) or (mb.LINEWIDTH_HZ / mb.MHZ)
    std_areas = standard_areas_from_peak(ref, mb.PPM, X, local_baseline=False,
                                         linewidth_ppm=lw, other_peaks=peaks)
std = standard_per_proton(std_areas, mb.STANDARD_NH)
incl = [s['index'] for s in mb.SAMPLES]
grp = {}
for s in mb.SAMPLES:
    grp.setdefault(s['label'], []).append(s['index'])
field = project_field_mhz(mets)
drift = float(study_drift(mets) or 0.0)

fixed_all = {}
elected = {}
for r in rows:
    if r.get('_not_quantified') or r.get('_is_is'):
        continue
    nh = float(r.get('_nh') or 0)
    ar = r.get('_areas_per_sample')
    ar = dict(enumerate(ar)) if isinstance(ar, list) else (ar or {})
    if not nh:
        continue
    fixed_all[r['hmdb_id']] = {int(s): float(a) / nh for s, a in ar.items()
                               if a is not None and np.isfinite(a) and a > 0}
    elected[r['hmdb_id']] = r


def _fmt(v):
    import math
    return '  --' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * min(v, 9.99):4.0f}%'


def conc_of(per_proton):
    return {g: float(np.median([per_proton[i] / std[i] * mb.STANDARD_MM
                                for i in idx if i in per_proton and std.get(i)]))
            for g, idx in grp.items() if any(i in per_proton for i in idx)}


print(f'{"analyte":28s} {"truth A":>8s} {"elect A":>8s} {"casc A":>8s} '
      f'{"e%":>6s} {"c%":>6s} | {"truth B":>8s} {"casc B":>8s} {"c%":>6s} | ratio')
lines, errs_c, errs_e, ratios, free_err, rel_err = [], [], [], [], [], []
for mid, tr in truth.items():
    fixed = {k: v for k, v in fixed_all.items() if k != mid}
    out = cascade_missed(mets, mb.PPM, X, incl, fixed_per_proton=fixed,
                         noise_sigma=sigma, linewidth_ppm=lw, drift_ppm=drift,
                         field_mhz=field, excluded_regions=mb.WATER)
    res = out.get(mid)
    e = elected.get(mid)
    eA = (e.get('_concs_by_group', {}).get('A') if e else None)
    eA = float(np.median(eA)) if eA else float('nan')
    if res is None:
        print(f'{tr["name"][:28]:28s} {tr["A"]:8.2f} {eA:8.2f} {"--":>8s}  '
              f'(cascade did not take it in)')
        continue
    c = conc_of(res['per_proton'])
    cA, cB = c.get('A', float("nan")), c.get('B', float("nan"))
    ec = (cA - tr['A']) / tr['A'] * 100
    ecB = (cB - tr['B']) / tr['B'] * 100
    ee = (eA - tr['A']) / tr['A'] * 100 if np.isfinite(eA) else float('nan')
    # the ratio between the two groups: trueness aside, does it track?
    rt = (cB / cA) / (tr['B'] / tr['A']) if cA else float('nan')
    tag = (f" free{100 * res.get('free_share', 1):3.0f}%"
           f" agree{_fmt(res.get('disagreement'))}"
           + (f" cap{100 * res.get('capped_share', 0):.0f}%"
              if res.get('capped_share') else ''))
    rel_err.append((res.get('disagreement', float('nan')), abs(ec), abs(ecB)))
    free_err.append((res.get('free_share', 1.0), abs(ec), abs(ecB), tr['name']))
    print(f'{tr["name"][:28]:28s} {tr["A"]:8.2f} {eA:8.2f} {cA:8.2f} '
          f'{ee:+6.0f} {ec:+6.0f} | {tr["B"]:8.2f} {cB:8.2f} {ecB:+6.0f} | '
          f'{rt:5.2f}{tag}')
    errs_c += [abs(ec), abs(ecB)]
    ratios.append(rt)
    if np.isfinite(ee):
        errs_e.append(abs(ee))
if errs_c:
    print(f'\ncascade  median |err| {np.median(errs_c):.0f}%  p90 {np.percentile(errs_c, 90):.0f}%'
          f'  bias {np.mean([e for e in errs_c]):.0f}%')
    print(f'election median |err| {np.median(errs_e):.0f}%  (only where it quantified)')
    lo = [e for f, a, b, _n in free_err if f < 0.6 for e in (a, b)]
    hi = [e for f, a, b, _n in free_err if f >= 0.6 for e in (a, b)]
    print(f'free share >= 60%: n={len(hi)//2:2d}  median |err| {np.median(hi):.0f}%  max {max(hi):.0f}%')
    print(f'free share <  60%: n={len(lo)//2:2d}  median |err| {np.median(lo) if lo else float("nan"):.0f}%  max {max(lo) if lo else float("nan"):.0f}%')
    import math
    unchecked = [e for r, a, b in rel_err if r is None or math.isnan(r) for e in (a, b)]
    print(f'  no second signal to check: n={len(unchecked)//2:2d} '
          f'median |err| {np.median(unchecked) if unchecked else float("nan"):5.1f}%')
    for bar in (0.10, 0.20, 0.30, 0.50):
        ok = [e for r, a, b in rel_err if r is not None and not math.isnan(r) and r <= bar for e in (a, b)]
        no = [e for r, a, b in rel_err if r is not None and not math.isnan(r) and r > bar for e in (a, b)]
        print(f'  predicted +-{100*bar:3.0f}%: kept n={len(ok)//2:2d} median |err| '
              f'{np.median(ok) if ok else float("nan"):5.1f}% max {max(ok) if ok else float("nan"):5.1f}%'
              f'   | dropped n={len(no)//2:2d} median {np.median(no) if no else float("nan"):5.1f}%')
    print(f'group ratio B/A recovered: median {np.median(ratios):.3f}  '
          f'spread {np.percentile(ratios, 10):.2f}-{np.percentile(ratios, 90):.2f}')
