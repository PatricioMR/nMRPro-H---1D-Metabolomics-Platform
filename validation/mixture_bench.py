"""A study whose answer is known, built from the reference database's own
measured shapes, put through the decision chain that runs.

On a real extract nobody knows the answer: "alpha-D-glucose 1-phosphate,
4 mM" may be diet, an unknown pathway, or methanol from the extraction
read wrongly, and looking at the table cannot tell. Here the answer is
chosen first. A panel of compounds -- what a skin extract with some blood
holds -- is laid down at concentrations picked in advance, each signal as
the reference database measured it on the pure compound (its profile, or a
Lorentzian for a singlet), broadened to a study's linewidth, drifted, with
the study's noise; a standard of known concentration sits at 0 ppm; a
compound the database does not know (methanol, from the extraction) is
added so that whatever claims its singlet is a counted false positive.

Then detection, the annotation decisions in the order the application makes
them, the election and quantification run over the WHOLE database, and
three numbers come out: how many of the panel were found (recall), how many
analytes were reported that are not there (false positives), and how far
each concentration is from the one that went in. Run with and without a
rule, the difference is what the rule did.

The GUI's orchestration is a Tk mixin and is not here; every function that
DECIDES anything is, called as the application calls it (see
tests/integration/test_end_to_end_study.py for the same chain on synthetic
Lorentzians). What this cannot test is the database's fidelity to a real
sample -- the shapes are the database's own -- only what the pipeline does
with shapes it should recognise.

    python -m validation.mixture_bench [--db PATH] [--seed N] [--patterns]
                                        [--no-shapes] [--shared]

Measured patterns (``m*``, ADR-013) are a method still being built and are
left out by default, as an unresolved 'm' is: a signal the reference DB
measured as a pattern is skipped unless ``--patterns`` puts it in. The
numbers the bench reports by default are the letter-multiplet and singlet
pipeline's alone. ``--no-shapes`` (patterns matched on the peak list, not
fitted to the trace) only means something with ``--patterns``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from processing.annotation_consistency import (  # noqa: E402
    consistent_peaks, reference_per_proton)
from processing.annotation_detectability import (  # noqa: E402
    fingerprint_contradictions, search_half_width)
from processing.annotation_quality import (  # noqa: E402
    promote_singlets, quality_verdict)
from processing.annotation_scoring import (  # noqa: E402
    PATTERN_MULT, SCORING_DEFAULTS, apply_per_sample_consensus, dj_score,
    annotate_drift_spread, expected_line_count, is_singlet_like, lines_covary,
    lines_covary_on_trace, lines_explained, measure_multiplet,
    measure_pattern_signal, measured_letter, pascal_agreement,
    pattern_governs, pattern_offsets_hz, ppm_score, score_signal,
    synthetic_lines)
from processing.multiplet_line_fit import fit_across_samples  # noqa: E402
from processing.multiplet_line_selector import multiplet_line_peaks  # noqa: E402
from processing.pattern_shape import (  # noqa: E402
    lines_model, parse_profile, profile_model)
from processing.peak_detection import detect_peaks  # noqa: E402
from processing.per_sample_signal import signal_across_samples  # noqa: E402
from processing.quant_detection_limit import project_drift  # noqa: E402
from processing.quantification import eligible_metabolite_ids  # noqa: E402
from processing.spectral_regions import (  # noqa: E402
    DEFAULT_EXCLUDED_REGIONS, measure_linewidth_ppm)
from gui.stages._annotation_helpers import (  # noqa: E402
    _multi_peak_coherence, _stocsy_coherence, _stocsy_partner_count,
    ambiguous_peak_ranks)
from gui.stages._annotation_search import _measured_columns_sql  # noqa: E402
from gui.stages._quant_adapter import quantify_project  # noqa: E402
from processing.stocsy import compute_peak_correlation_matrix  # noqa: E402

DB_DEFAULT = os.path.join(REPO, 'databases',
                          'BMRB_Metabolomics_Mammalian_v03.db')
OUT_DIR = os.path.join(REPO, 'validation', 'out')

# ── the study ────────────────────────────────────────────────────────────────
MHZ = 500.0                        # the field most of the reference spectra are at
PPM = np.linspace(-0.5, 10.0, 70001)     # 0.15 mHz, 0.075 Hz per point
LINEWIDTH_HZ = 2.6                 # the skin study's; kept in Hz (the bench runs at MHZ)
STANDARD_PPM, STANDARD_NH, STANDARD_MM = 0.0, 9, 0.5
AREA_UNIT = 1.0e4                  # intensity x ppm per proton per mM
DRIFT_PPM = 0.008                  # every analyte, the study's median
NOISE_LACTATE_SNR = 500.0          # what lactate's doublet reads on the study
TOLERANCE = 0.05
WATER = list(DEFAULT_EXCLUDED_REGIONS)

# What a skin extract with some blood holds, in mM, and how each varies
# between the two groups (a factor on group B). Chosen, not measured: the
# point is that they are known.
PANEL = [
    ('L-lactic acid',              4.0, 1.0),
    ('Creatine',                   1.5, 0.8),
    ('Taurine',                    2.0, 1.2),
    ('Glycine',                    0.8, 1.0),
    ('L-alanine',                  0.7, 1.0),
    ('L-glutamic acid',            0.6, 1.3),
    ('L-glutamine',                0.5, 1.0),
    ('Myo-inositol',               0.4, 1.0),
    ('Choline',                    0.15, 1.0),
    ('Trans 4 hydroxy l proline',  0.3, 1.0),
    ('L-valine',                   0.25, 1.0),
    ('L-leucine',                  0.2, 1.0),
    ('L-isoleucine',               0.12, 1.0),
    ('Acetic Acid',                0.3, 1.0),
    ('Inosine',                    0.4, 0.6),
    ('Betaine',                    0.3, 1.0),
    ('L-Glutathione reduced',      0.5, 1.0),
    ('Creatinine',                 0.1, 1.0),
]
# Not in the database: whatever claims its singlet is a false positive.
FOREIGN = [('Methanol', 5.0, [(3.36, 's', 3)])]

SAMPLES = ([{'index': i, 'label': 'A', 'type': 'Sample'} for i in range(6)]
           + [{'index': 6 + i, 'label': 'B', 'type': 'Sample'}
              for i in range(6)])


def _lorentz_area(x, centre, area, fwhm_hz):
    hw = fwhm_hz / 2.0 / MHZ
    return area / (np.pi * hw) * hw ** 2 / ((x - centre) ** 2 + hw ** 2)


def _db_signals(con, name):
    cols = {r[1] for r in con.execute('pragma table_info(shifts_1h)')}
    m = _measured_columns_sql(cols)
    mid = con.execute('select bmrb_id from metabolites where name=?',
                      (name,)).fetchone()
    if not mid:
        raise SystemExit(f'{name}: not in the database')
    rows = con.execute(
        f"select {m['ppm']} as ppm, s.multiplicity, s.protons, "
        f"{m['profile']} as pattern_profile, {m['profile_fwhm']} as "
        f"profile_fwhm_hz, {m['profile_nh']} as profile_nh "
        "from shifts_1h s where s.bmrb_id=? order by ppm", (mid[0],)
    ).fetchall()
    return mid[0], [dict(zip(('ppm', 'mult', 'protons', 'profile',
                              'profile_fwhm', 'profile_nh'), r)) for r in rows]


def compound_trace(x, signals, conc, *, shift):
    """One compound at ``conc`` mM: each profiled signal as its measured
    profile broadened to the study's width, scaled to its LISTED protons --
    stoichiometry is the truth, and the reference spectrum's own artefacts
    (lactate's quartet attenuated by the water suppression) must not be;
    a signal without a profile as a Lorentzian of its listed protons."""
    y = np.zeros_like(x)
    for s in signals:
        centre = float(s['ppm']) + shift
        prof = parse_profile(s['profile'])
        if prof is not None and s['protons']:
            extra = max(0.0, LINEWIDTH_HZ - float(s['profile_fwhm'] or 0.0))
            m = profile_model(x, prof, centre, extra, MHZ)
            order = np.argsort(x)
            area_m = float(np.trapezoid(m[order], x=x[order]))
            if area_m > 0:
                y += m * (conc * float(s['protons']) * AREA_UNIT / area_m)
        elif s['protons']:
            y += _lorentz_area(x, centre, conc * float(s['protons']) * AREA_UNIT,
                               LINEWIDTH_HZ)
    return y


def build_study(db_path, *, seed=0):
    con = sqlite3.connect(db_path)
    rng = np.random.default_rng(seed)
    truth, comps = {}, []
    for name, conc, factor in PANEL:
        mid, sigs = _db_signals(con, name)
        comps.append((name, mid, sigs, conc, factor))
        truth[mid] = {'name': name, 'A': conc, 'B': conc * factor}
    foreign = []
    for name, conc, sigs in FOREIGN:
        foreign.append((name, [{'ppm': p, 'mult': mu, 'protons': nh,
                                'profile': None, 'profile_fwhm': None,
                                'profile_nh': None} for p, mu, nh in sigs],
                        conc))
    # noise from what lactate reads on the study
    lac = next(c for c in comps if c[0] == 'L-lactic acid')
    y_lac = compound_trace(PPM, lac[2], lac[3], shift=0.0)
    sigma = float(y_lac.max()) / NOISE_LACTATE_SNR
    cols = []
    for info in SAMPLES:
        grp = info['label']
        shift = DRIFT_PPM + rng.normal(0.0, 0.0015)
        y = rng.normal(0.0, sigma, PPM.size)
        y += _lorentz_area(PPM, STANDARD_PPM + rng.normal(0, 0.0003),
                           STANDARD_MM * STANDARD_NH * AREA_UNIT, LINEWIDTH_HZ)
        for name, mid, sigs, conc, factor in comps:
            c = truth[mid][grp] * float(np.exp(rng.normal(0.0, 0.12)))
            y += compound_trace(PPM, sigs, c, shift=shift)
        for name, sigs, conc in foreign:
            y += compound_trace(PPM, sigs, conc * float(np.exp(rng.normal(0.0, 0.12))),
                                shift=shift)
        cols.append(y)
    X = np.column_stack(cols)
    return X, sigma, truth, {n: c for n, _s, c in foreign}


# ── the decisions, in the application's order ────────────────────────────────

def annotate(ppm, X, peaks, noise, db_path, *, shapes=True, patterns=False):
    params = dict(SCORING_DEFAULTS)
    params['dppm_gold'] = TOLERANCE
    tol = TOLERANCE
    incl = [s['index'] for s in SAMPLES]
    y_mean = X[:, incl].mean(axis=1)
    lw_ppm = measure_linewidth_ppm(ppm, y_mean, peaks) or (LINEWIDTH_HZ / MHZ)
    lw_hz = lw_ppm * MHZ
    floor = noise * 6.0
    pk_by_rank = {p['rank']: p for p in peaks}

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cols = {r[1] for r in con.execute('pragma table_info(shifts_1h)')}
    m = _measured_columns_sql(cols)
    entries = [dict(r) for r in con.execute(
        f"select s.bmrb_id as hmdb_id, met.name, {m['ppm']} as ppm_center, "
        f"{m['ppm']} as ppm, s.multiplicity as type, {m['j']} as j_coupling, "
        f"{m['j_sigma']} as j_sigma, {m['pattern']} as pattern_lines, "
        f"{m['profile']} as pattern_profile, {m['profile_fwhm']} as "
        f"profile_fwhm_hz, {m['profile_nh']} as profile_nh, s.protons "
        "from shifts_1h s join metabolites met on met.bmrb_id = s.bmrb_id")]
    by_met: dict = {}
    for e in entries:
        by_met.setdefault(e['hmdb_id'], {'name': e['name'], 'signals': []})
        by_met[e['hmdb_id']]['signals'].append(e)

    ref_pp = reference_per_proton(entries, peaks, tolerance=tol,
                                  met_key=lambda e: e['hmdb_id'])
    # co-variation across samples, as the application computes it after
    # detection: the evidence that separates two analytes whose contested
    # signal is identical
    pk_corr = compute_peak_correlation_matrix(peaks, included=incl)
    all_peaks_by_id: dict = {}
    for e in entries:
        all_peaks_by_id.setdefault(e['hmdb_id'], []).append(
            (float(e['ppm_center']), 1.0))
    n_samp = len(incl)
    mets = []
    for mid, met in by_met.items():
        mps = []
        for e in met['signals']:
            db_ppm = float(e['ppm_center'])
            mult = str(e.get('type') or 's').lower().strip()
            j_raw = str(e.get('j_coupling') or '').strip()
            try:
                j_hz = float(j_raw.split(',')[0]) if j_raw else 0.0
            except ValueError:
                j_hz = 0.0
            e = dict(e)
            letter = (measured_letter(pattern_offsets_hz(e))
                      if e.get('pattern_lines') else None)
            if letter is not None and letter[0] != mult:
                mult, j_hz = letter[0], float(letter[1])
                e['type'] = mult
                e['j_coupling'] = f'{j_hz:.2f}'
            is_pat = bool(e.get('pattern_lines')) and pattern_governs(
                mult, e.get('pattern_lines'))
            if is_pat and not patterns:
                continue          # a future method; out with the 'm's
            if is_pat:
                mult = PATTERN_MULT
                e['type'] = PATTERN_MULT
            if mult in ('m', 'multiplet', 'br s', 'brs'):
                continue
            in_tol = [p for p in peaks if abs(p['ppm'] - db_ppm) <= tol]
            if not in_tol:
                continue
            anchor = min(consistent_peaks(in_tol, e, ref_pp.get(mid)),
                         key=lambda p: abs(p['ppm'] - db_ppm))
            n_exp = expected_line_count(mult, fallback=2)
            mp = {**e, 'db_ppm': db_ppm, 'multiplicity': mult,
                  'peak_rank': anchor['rank'], 'peak_ppm': float(anchor['ppm']),
                  'ppm_diff': abs(float(anchor['ppm']) - db_ppm),
                  'study_linewidth_hz': lw_hz, 'pattern_floor': floor,
                  'pattern_sigma': noise, 'pattern_tolerance_ppm': tol,
                  'pattern_field_mhz': MHZ}
            if is_singlet_like(mult):
                mp['predicted_lines'] = [{'ppm': float(anchor['ppm']),
                                          'rel_intensity': 1.0}]
            elif mult == PATTERN_MULT:
                meas = measure_pattern_signal(
                    mp, peaks, field_mhz=MHZ, dj_gold_hz=params['dj_gold_hz'],
                    anchor_ppm=float(anchor['ppm']), merge_hz=0.0,
                    fwhm_hz=lw_hz, floor=floor,
                    trace=(ppm, y_mean) if shapes else None,
                    sigma=noise, tolerance_ppm=tol)
                if not meas.measured:
                    continue
                mp.update({'predicted_lines': meas.predicted_lines,
                           '_n_found_lines': meas.n_found,
                           '_n_exp_lines': meas.n_expected,
                           'delta_j_hz': meas.delta_j_hz,
                           'pattern_residual_hz': meas.pattern_residual_hz,
                           'intensity_agreement': meas.pattern_intensity_agreement})
                if meas.pattern_snr is not None:
                    mp['pattern_fit'] = {'snr': meas.pattern_snr,
                                         'explained': meas.pattern_explained,
                                         'r2': meas.pattern_r2,
                                         'amplitude': meas.pattern_amplitude}
                if meas.origin_ppm is not None and meas.centre_ppm is not None:
                    mp['ppm_diff'] = abs(meas.origin_ppm - db_ppm)
                    mp['peak_ppm'] = meas.centre_ppm
                    mp['pattern_centre_offset_ppm'] = meas.centre_ppm - meas.origin_ppm
            else:
                pred = synthetic_lines(float(anchor['ppm']), j_hz, n_exp, MHZ) \
                    if j_hz else [{'ppm': float(anchor['ppm']),
                                   'rel_intensity': 1.0}]
                meas = measure_multiplet(
                    pred, peaks, mult=mult, j_coupling=(f'{j_hz}' if j_hz else None),
                    db_ppm=db_ppm, centre_hint=float(anchor['ppm']),
                    n_expected=n_exp, tolerance=tol, field_mhz=MHZ,
                    linewidth_hz=lw_hz)
                mp.update({'predicted_lines': meas.predicted_lines or pred,
                           '_n_found_lines': meas.n_found,
                           '_n_exp_lines': meas.n_expected,
                           'delta_j_hz': meas.delta_j_hz,
                           'observed_j_hz': meas.observed_j_hz})
                if meas.centre_ppm is not None:
                    mp['peak_ppm'] = float(meas.centre_ppm)
                    mp['ppm_diff'] = abs(meas.centre_ppm - db_ppm)
                # per sample: the lines in every sample, the consensus, and
                # how much of their height the multiplet accounts for
                line_pks = multiplet_line_peaks(mp, pk_by_rank)
                if line_pks and meas.measured:
                    exp = [float(l['ppm']) for l in mp['predicted_lines']
                           if 'ppm' in l]
                    fits = fit_across_samples(
                        exp, ppm, X, included=incl, field_mhz=MHZ,
                        dj_bad_hz=params['dj_bad_hz'],
                        dj_gold_hz=params['dj_gold_hz'],
                        max_drift_ppm=tol, noise_floor=floor)
                    out = signal_across_samples(
                        line_pks, fits=fits, included=incl, db_ppm=db_ppm,
                        db_j_hz=j_hz or None, field_mhz=MHZ, n_lines=len(exp))
                    ags, exs = [], []
                    for f in fits.values():
                        if not f.get('fitted'):
                            continue
                        a = pascal_agreement(f.get('ranks'), f.get('heights'), mult)
                        x_ = lines_explained(f.get('ranks'), f.get('heights'), mult)
                        if a is not None:
                            ags.append(a)
                        if x_ is not None:
                            exs.append(x_)
                    mp['intensity_agreement'] = float(np.median(ags)) if ags else None
                    mp['lines_explained'] = float(np.median(exs)) if exs else None
                    mp['lines_covary'] = lines_covary(fits, noise=noise)
                    mp['per_sample_consensus'] = out
                    apply_per_sample_consensus(mp, params=params)
            # co-variation off the traces, for letters and patterns alike
            _lp = [float(l['ppm']) for l in (mp.get('predicted_lines') or ())
                   if 'ppm' in l]
            if len(_lp) >= 2 and mp.get('lines_covary') is None:
                mp['lines_covary'] = lines_covary_on_trace(
                    ppm, X, _lp, included=incl, half_ppm=max(lw_ppm / 2, 0.001),
                    noise=noise)
            s_ppm = ppm_score(mp['ppm_diff'], tol / 2.0)
            dj = mp.get('delta_j_hz')
            s_dj = dj_score(dj, params['dj_gold_hz'], params['dj_bad_hz'],
                            params['dj_neutral'])
            score_signal(mp, s_ppm=s_ppm, s_dj=s_dj,
                         dj_conf=0.0 if dj is None else 1.0,
                         coverage=1.0, params=params)
            # the application's multi-peak coherence and STOCSY terms on
            # the composite, and the partner count the election ranks on
            coh, _cov, _rm = _multi_peak_coherence(mid, all_peaks_by_id,
                                                   peaks, tol)
            comp = float(mp.get('composite_score') or 0.0)
            comp *= 0.05 + 0.95 * float(coh)
            st = _stocsy_coherence(mid, all_peaks_by_id, peaks, tol, pk_corr)
            if st is not None:
                w = 1.0 - float(np.exp(-0.25 * max(0, n_samp - 2)))
                comp *= float(np.exp(0.10 * max(0.0, st) * w))
            mp['composite_score'] = comp
            mp['stocsy_partners'] = _stocsy_partner_count(
                mid, mp.get('peak_rank'), all_peaks_by_id, peaks, tol, pk_corr)
            q, why = quality_verdict(mp, params=params)
            mp['annotation_quality'] = q
            mp['_status_reasons'] = [why] if why else []
            mps.append(mp)
        if not mps:
            continue
        mets.append({'id': mid, 'name': met['name'], 'matched_peaks': mps,
                     'db_peaks': [{'ppm_center': float(s['ppm_center']),
                                   'type': s.get('type'),
                                   'protons': s.get('protons'),
                                   'pattern_lines': s.get('pattern_lines'),
                                   'j_coupling': s.get('j_coupling'),
                                   'pattern_profile': s.get('pattern_profile'),
                                   'profile_fwhm_hz': s.get('profile_fwhm_hz')}
                                  for s in met['signals']]})
    promote_singlets(mets, peaks, params=params)
    annotate_drift_spread(mets)
    drift = project_drift(mets)
    half = search_half_width(tolerance_ppm=tol, drift_ppm=drift,
                             linewidth_ppm=lw_ppm)
    stats: dict = {}
    fingerprint_contradictions(mets, ppm, y_mean, peaks, tolerance_ppm=tol,
                               search_ppm=half, excluded_regions=WATER,
                               stats=stats, noise=noise, linewidth_ppm=lw_ppm)
    return mets, params, stats


def run(db_path, *, seed=0, shapes=True, shared=False, patterns=False,
        cascade=False, ppm_band=None, real_annotation=False):
    X, sigma, truth, foreign = build_study(db_path, seed=seed)
    peaks = detect_peaks(PPM, X, noise_std=sigma, min_snr=6.0,
                         min_distance=0.003, baseline_correct=False,
                         global_align=False, samples_info=SAMPLES,
                         per_sample_detection=True, detect_shoulders=False,
                         min_group_presence=3, excluded_regions=WATER)
    if real_annotation:
        # The application's own path: its search, its grouping, its eleven
        # passes. Slower, and the only version of these numbers that is a
        # statement about the software rather than about this file.
        from validation.app_annotation import annotate as app_annotate
        mets, params = app_annotate(
            peaks, PPM, X, SAMPLES, db_path=db_path, field_mhz=MHZ,
            tolerance_ppm=TOLERANCE, noise_std=sigma, min_snr=6.0,
            min_distance_ppm=0.003)
        fp_stats = {}
    else:
        mets, params, fp_stats = annotate(PPM, X, peaks, sigma, db_path,
                                          shapes=shapes, patterns=patterns)
    eligible = eligible_metabolite_ids(mets)
    # a peak whose candidates tie on every criterion has no winner, as the
    # results window decides before quantification
    ambiguous = ambiguous_peak_ranks(mets, manual_quality={})
    ref = min(peaks, key=lambda p: abs(float(p['ppm']) - STANDARD_PPM))
    rows = quantify_project(
        metabolites=mets, peaks=peaks, ppm=PPM, intensities=X,
        samples_info=SAMPLES, ref_peak=ref, ref_nh=STANDARD_NH,
        ref_conc=STANDARD_MM, ref_name='DSS', noise_sigma=sigma,
        ambiguous_ranks=ambiguous,
        use_pqn=False, local_baseline=False, quantify_shapes=shapes,
        # as the application runs: the shared-window balance is parked
        # (app_paths.DECONVOLUTION_ENABLED); on it gave lactate 40% of its
        # own doublet, the rest to eight false claimants of the window
        deconvolve=False,
        resolve_shared=shared,
        cascade=cascade,
        ppm_band_ppm=ppm_band,
        annotation_params=params)
    rep = report(rows, mets, truth, foreign, peaks, fp_stats)
    # the study itself, for anything that wants to look at a row
    rep.update({'rows': rows, 'ppm': PPM, 'X': X, 'samples_info': SAMPLES})
    return rep


def report(rows, mets, truth, foreign, peaks, fp_stats):
    quantified = {r['hmdb_id']: r for r in rows
                  if not r.get('_not_quantified') and not r.get('_is_is')}
    all_rows = {r['hmdb_id']: r for r in rows}
    by_met = {m['id']: m for m in mets}
    found, missed, errors = [], [], []
    for mid, t in truth.items():
        r = quantified.get(mid)
        if r is None:
            row = all_rows.get(mid)
            why = '; '.join(str(x)[:90] for x in
                            (row or {}).get('_status_reasons') or ())                 if row else 'no candidate at all'
            sigs = [(f"{float(mp.get('db_ppm', 0)):.3f} {mp.get('multiplicity')} "
                     f"{mp.get('annotation_quality') or 'undecided'}"
                     + (f" ({mp.get('_status_reasons')[0][:50]})"
                        if mp.get('_status_reasons') else ''))
                    for mp in (by_met.get(mid) or {}).get('matched_peaks', [])]
            missed.append((t['name'], why, sigs))
            continue
        by = r.get('_concs_by_group') or {}
        got_a = float(np.median(by.get('A') or [np.nan]))
        got_b = float(np.median(by.get('B') or [np.nan]))
        err_a = (got_a - t['A']) / t['A'] * 100 if np.isfinite(got_a) else np.nan
        err_b = (got_b - t['B']) / t['B'] * 100 if np.isfinite(got_b) else np.nan
        win = r.get('_multiplet_window') or [0, 0]
        d = r.get('_deconv') or {}
        rem = d.get('remainder') or {}
        tag = (f" shared[{','.join(c['label'][:12] + ' sep=' + str(round(c.get('separability') or 0, 2)) for c in d.get('components', ()) if c.get('met_id') != r.get('hmdb_id'))}"
               f" rem={100 * float(np.median(list(rem.values()))):.0f}%]"
               if d else '')
        found.append((t['name'], t['A'], got_a, err_a, t['B'], got_b, err_b,
                      r.get('_integ_mode'), r.get('_multiplicity'),
                      f"{min(win):.3f}-{max(win):.3f} nH={r.get('_nh')} "
                      f"cap={r.get('_captured_fraction')}{tag}"))
        errors += [e for e in (err_a, err_b) if np.isfinite(e)]
    false_pos = [(r['metabolite'], float(r.get('conc_mm') or 0),
                  float(r.get('ppm') or 0))
                 for mid, r in quantified.items() if mid not in truth]
    false_pos.sort(key=lambda t: -t[1])
    return {'found': found, 'missed': missed, 'false_positives': false_pos,
            'errors_pct': errors, 'n_peaks': len(peaks),
            'n_ambiguous': sum(1 for r in rows
                               if r.get('_rejection') == 'ambiguous_rank'),
            'fingerprint': fp_stats, 'n_rows': len(rows),
            'n_quantified': len(quantified)}


def print_report(rep, *, label):
    print(f'\n=== {label} ===')
    print(f"peaks {rep['n_peaks']}  |  rows {rep['n_rows']}  |  "
          f"quantified {rep['n_quantified']}")
    n_true = len(rep['found']) + len(rep['missed'])
    print(f"recall {len(rep['found'])}/{n_true}   false positives "
          f"{len(rep['false_positives'])}   ambiguous {rep.get('n_ambiguous')}")
    errs = np.array(rep['errors_pct'])
    if errs.size:
        print(f"concentration error: median {np.median(np.abs(errs)):.0f}%  "
              f"p90 {np.percentile(np.abs(errs), 90):.0f}%  "
              f"bias {np.median(errs):+.0f}%")
    print('  found:')
    for name, ta, ga, ea, tb, gb, eb, mode, mult, win in rep['found']:
        print(f'    {name[:28]:28} A {ta:5.2f} -> {ga:6.2f} ({ea:+5.0f}%)   '
              f'B {tb:5.2f} -> {gb:6.2f} ({eb:+5.0f}%)   {mode} {mult}  {win}')
    if rep['missed']:
        print('  missed:')
        for name, why, sigs in rep['missed']:
            print(f'    {name[:28]:28} {why}')
            for sg in sigs:
                print(f'        {sg}')
    if rep['false_positives']:
        print('  false positives (mM):')
        for name, c, p in rep['false_positives'][:15]:
            print(f'    {name[:32]:32} {c:6.2f} mM at {p:.3f}')
    fp = rep['fingerprint']
    print(f"  fingerprint gate: contradicted {fp.get('contradicted')}  "
          f"marked bad {fp.get('analytes_marked')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DB_DEFAULT)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patterns', action='store_true',
                    help='annotate measured patterns (m*) -- a method in '
                         'progress, off by default')
    ap.add_argument('--no-shapes', action='store_true',
                    help='with --patterns: match on the peak list, no '
                         'trace fit')
    ap.add_argument('--shared', action='store_true',
                    help='resolve shared windows by shape')
    ap.add_argument('--cascade', action='store_true',
                    help='add what the cascade finds on the residual (ADR-028)')
    ap.add_argument('--real-annotation', action='store_true',
                    help="annotate with the application's own search and "
                         "passes instead of this file's own (ADR-030/032)")
    ap.add_argument('--ppm-band', type=float, default=None,
                    help='width the election buckets delta-ppm by, measured '
                         'from the study drift (default: PPM_BAND_PPM)')
    args = ap.parse_args()
    t0 = time.time()
    rep = run(args.db, seed=args.seed, shapes=not args.no_shapes,
              shared=args.shared, patterns=args.patterns, cascade=args.cascade,
              ppm_band=args.ppm_band, real_annotation=args.real_annotation)
    print_report(rep, label=f"{os.path.basename(args.db)}  patterns="
                            f"{args.patterns}  shapes={not args.no_shapes}  "
                            f"shared={args.shared}  cascade={args.cascade}  "
                            f"real_annotation={args.real_annotation}  "
                            f"seed={args.seed}")
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = (('patterns_' + ('shapes' if not args.no_shapes else 'noshapes'))
           if args.patterns else 'letters') + (
        '_shared' if args.shared else '') + ('_cascade' if args.cascade else '')
    with open(os.path.join(OUT_DIR, f'bench_{tag}_seed{args.seed}.json'),
              'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in rep.items() if k not in ('fingerprint', 'rows', 'ppm', 'X', 'samples_info')}, f,
                  indent=1, default=float)
    print(f'({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
