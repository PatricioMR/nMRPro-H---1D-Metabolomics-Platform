"""How far a real study's signals stand from the reference, measured on the
compounds a tissue extract unmistakably holds -- the regime the bench has to
reproduce before what improves the bench can improve the study.

From a saved project, annotated afresh with the code as it is, for every
signal the annotation accepted (Good) of a compound in ``UNMISTAKABLE``:

    offset     apex minus listing minus the study's common drift (mppm)
    by group   the apex in each group's mean trace, and the spread between them
    linewidth  per sample, the full width at half height of the internal
               standard

    python -m validation.extract_regime "G:/.../MRP_MiceISS_v07.nmrproj"
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Compounds every tissue or cell extract holds and whose signals the
# reference lists cleanly: the engine's own REFERENCE_PANEL and the ones the
# Mice diagnosis saw by eye (vault Diagnostico-Mice-v06, section 4).
UNMISTAKABLE = frozenset({
    'l-lactic acid', 'l-alanine', 'l-valine', 'l-leucine', 'l-isoleucine',
    'l-threonine', 'l-glutamic acid', 'l-glutamine', 'glycine', 'taurine',
    'choline', 'betaine', 'myo-inositol', 'creatine', 'creatinine',
    'acetic acid', 'sodium formate', 'formic acid', 'succinic acid',
    'citrate', 'citric acid', 'urocanate', 'urocanic acid', 'inosine',
    'l-serine', 'uridine', 'hypoxanthine', 'l-tyrosine', 'l-phenylalanine',
    'methanol', 'l-proline', 'l-lysine', 'l-methionine', 'l-histidine'})


def annotate_project(path):
    from data import project_file
    from processing.baseline_model import BaselineCache, BaselineModel
    from processing.spectral_regions import parse_regions
    from validation.app_annotation import annotate
    from app_paths import databases_dir, REFERENCE_DATABASES
    st = project_file.read(path)
    ppm = np.asarray(st['ppm'], float)
    raw = np.asarray(st['all_intensities'], float)
    _bl = st.get('baseline_model')
    model = BaselineModel.from_dict(_bl) if _bl else None
    corr = np.asarray(BaselineCache().corrected(
        ppm, raw, model, already_corrected=bool(st.get('baseline_corrected'))), float)
    pd, ann = st['pd'], st['ann']
    excl = parse_regions(pd.get('excluded_regions_applied')
                         or pd.get('excluded_regions') or '') or ()
    db = ann.get('db_path') or os.path.join(databases_dir(), REFERENCE_DATABASES[0][1])
    with contextlib.redirect_stdout(io.StringIO()):
        mets, _params = annotate(
            list(pd['peaks']), ppm, corr, st['samples_info'], db_path=db,
            field_mhz=float(ann.get('spectrometer_mhz') or 600.0),
            tolerance_ppm=float(ann.get('tolerance') or 0.05),
            noise_std=float(pd['noise_std']),
            min_snr=float(pd.get('min_snr_var') or 6.0),
            min_distance_ppm=float(pd.get('min_width_var') or 0.004),
            excluded_regions=excl)
    return st, ppm, corr, mets


def _apex(ppm, y, centre, half):
    m = (ppm >= centre - half) & (ppm <= centre + half)
    if m.sum() < 3:
        return None
    i = np.flatnonzero(m)[np.argmax(y[m])]
    return float(ppm[i])


def fwhm_hz(ppm, y, centre, field, half=0.01):
    m = (ppm >= centre - half) & (ppm <= centre + half)
    x, v = ppm[m], y[m]
    if v.size < 5:
        return float('nan')
    top = v.max()
    above = x[v >= top / 2]
    return float((above.max() - above.min()) * field) if above.size > 1 else float('nan')


def measure(path, log=print):
    from processing.annotation_scoring import study_drift
    st, ppm, corr, mets = annotate_project(path)
    field = float(st['ann'].get('spectrometer_mhz') or 600.0)
    drift = study_drift(mets) or 0.0
    si = st['samples_info']
    groups = {}
    for s in si:
        if str(s.get('type', 'Sample')).upper() in ('BLK', 'BLANK'):
            continue
        groups.setdefault(s.get('label'), []).append(int(s['index']))
    means = {g: corr[:, idx].mean(axis=1) for g, idx in groups.items()}
    half = 0.5 * 4.3e-3
    rows = []
    for met in mets:
        if str(met.get('name') or '').strip().lower() not in UNMISTAKABLE:
            continue
        for mp in met.get('matched_peaks') or ():
            if mp.get('is_sub_peak') or mp.get('annotation_quality') != 'good':
                continue
            if mp.get('peak_ppm') is None or mp.get('db_ppm') is None:
                continue
            off = 1000 * (float(mp['peak_ppm']) - float(mp['db_ppm']) - drift)
            apices = {g: _apex(ppm, y, float(mp['peak_ppm']), half) for g, y in means.items()}
            ok = [v for v in apices.values() if v is not None]
            spread = 1000 * (max(ok) - min(ok)) if len(ok) > 1 else float('nan')
            rows.append((met['name'], float(mp['db_ppm']), mp.get('type'), off, spread))
    log(f'{os.path.basename(path)}: common drift {1000 * drift:+.1f} mppm, '
        f'{len(rows)} Good signals of {len({r[0] for r in rows})} unmistakable compounds')
    log('| compound | listed | mult | offset from listing+drift (mppm) | between groups (mppm) |')
    log('|---|---|---|---|---|')
    for r in sorted(rows, key=lambda r: -abs(r[3])):
        log(f'| {r[0]} | {r[1]:.3f} | {r[2]} | {r[3]:+.1f} | {r[4]:.1f} |')
    off = np.array([r[3] for r in rows])
    spr = np.array([r[4] for r in rows])
    if off.size:
        a = np.abs(off)
        log(f'\n|offset|: median {np.median(a):.1f}, p75 {np.percentile(a, 75):.1f}, '
            f'p90 {np.percentile(a, 90):.1f}, max {a.max():.1f} mppm; '
            f'signed SD {off.std(ddof=1) if off.size > 1 else float("nan"):.1f}')
        log(f'between groups: median {np.nanmedian(spr):.1f}, p90 {np.nanpercentile(spr, 90):.1f} mppm')
        # how much of the offset belongs to the compound and how much to the
        # signal: letters only (a pattern's apex is a fitted centre)
        by = {}
        for name, _p, mult, o, _s in rows:
            if str(mult) in ('s', 'd', 't', 'q', 'dd', 'dt', 'td'):
                by.setdefault(name, []).append(o)
        multi = {k: v for k, v in by.items() if len(v) >= 2}
        within = [x - np.mean(v) for v in multi.values() for x in v]
        dof = sum(len(v) - 1 for v in multi.values())
        means = [np.mean(v) for v in by.values()]
        if dof > 0 and len(means) > 1:
            sw = float(np.sqrt(np.sum(np.square(within)) / dof))
            log(f'letters: within a compound SD {sw:.1f} mppm ({len(multi)} compounds, '
                f'{dof} dof); between compound means SD {np.std(means, ddof=1):.1f} '
                f'({len(means)} compounds)')
    # linewidth per sample on the internal standard
    ref = float(st['qnt'].get('ref_ppm') or 0.0)
    lw = [fwhm_hz(ppm, corr[:, int(s['index'])], ref, field) for s in si]
    lw = np.array([v for v in lw if np.isfinite(v)])
    if lw.size:
        log(f'linewidth of the standard: median {np.median(lw):.2f} Hz, '
            f'range {lw.min():.2f}-{lw.max():.2f} Hz over {lw.size} samples')
    return rows, drift, lw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('projects', nargs='+')
    a = ap.parse_args()
    for p in a.projects:
        measure(p)
        print()


if __name__ == '__main__':
    main()
