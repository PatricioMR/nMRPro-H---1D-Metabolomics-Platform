# processing/identity_score.py
"""
One identity score for every analyte the engine gave an amount, from all
the evidence the run holds about it -- annotation's and quantification's
together -- in place of the last word of a chain of gates.

Mixture analysis that holds up in crowded spectra generates candidates
generously and decides identity on the joint evidence (ASICS: Tardivel PJC
et al., Metabolomics 2017;13:109; BATMAN: Hao J et al., Bioinformatics
2012;28:2088; COLMAR1d: Anal Chem 2024;96:17174). The engine's gates each
judge one thing in turn; a real analyte whose position the reference got
wrong fails the position gate however well the rest of its evidence holds,
and a stranger that passes every gate by a little is reported. The score
weighs them at once.

The model is logistic, fitted on the GISSMO bench's known answers by
``python -m validation.identity_score_probe fit`` (regimes: tight and
extract-like reference offsets, with and without the samples' biological
variation, tubes of 18-100 analytes) and judged leaving one regime out
(AUC 0.85). Held out, on the extract-like tubes of 50 and 75, scoring the
loose engine's candidates found 232 / 220 real analytes where the engine
found 209 / 188 at the same number of strangers.
"""
from __future__ import annotations

import numpy as np

AMOUNT_REJECTIONS = ('fit_misplaced', 'cascade_crowded', 'singlet_off_drift')


def _f(v, default=0.0):
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def pqn_dilution(M) -> np.ndarray:
    """``M`` peaks x samples. Each sample's dilution by the probabilistic
    quotient (Dieterle F, Ross A, Schlotterbeck G, Senn H. Anal Chem
    2006;78:4281-4290): the median, over the peaks positive in every
    sample, of the sample's area over the peak's median area. Ones with
    fewer than five such peaks."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return np.ones(M.shape[1] if M.ndim == 2 else 0)
    ok = np.all(M > 0, axis=1)
    if ok.sum() < 5:
        return np.ones(M.shape[1])
    ref = np.median(M[ok], axis=1)
    f = np.median(M[ok] / ref[:, None], axis=0)
    return np.where(f > 0, f, 1.0)


def analyte_covariation(met, area_by_rank, dilution):
    """``(median r, min r)`` between the analyte's matched peaks across the
    samples -- areas over the dilution, logged; STOCSY read on peaks
    (Cloarec O et al., Anal Chem 2005;77:1282). None with fewer than two
    peaks positive everywhere."""
    ranks = sorted({mp.get('peak_rank') for mp in met.get('matched_peaks') or ()
                    if not mp.get('is_sub_peak') and mp.get('peak_rank') is not None
                    and mp.get('annotation_quality') != 'bad'})
    rows = [area_by_rank[r] for r in ranks if r in area_by_rank]
    rows = [r for r in rows if r.size == dilution.size and np.all(r > 0)]
    if len(rows) < 2:
        return None
    L = np.log(np.array(rows) / dilution[None, :])
    if np.any(L.std(axis=1) <= 0):
        return None
    C = np.corrcoef(L)
    iu = np.triu_indices(len(rows), 1)
    return float(np.median(C[iu])), float(np.min(C[iu]))


def peak_context(peaks, included=None) -> tuple:
    """``(area_by_rank, dilution)`` for ``analyte_covariation``, over the
    ``included`` samples."""
    area = {}
    for p in peaks or ():
        a = p.get('area_per_sample')
        if p.get('rank') is None or a is None:
            continue
        a = np.asarray(a, dtype=float)
        if included is not None:
            a = a[[i for i in included if 0 <= i < a.size]]
        area[int(p['rank'])] = a
    sizes = {a.size for a in area.values()}
    if len(sizes) != 1:
        return {}, np.ones(0)
    return area, pqn_dilution(np.array(list(area.values())))


def peak_owners(metabolites) -> dict:
    """``{peak rank: how many metabolites matched it}``."""
    owners = {}
    for met in metabolites or ():
        for mp in met.get('matched_peaks') or ():
            if mp.get('peak_rank') is not None and not mp.get('is_sub_peak'):
                owners[mp['peak_rank']] = owners.get(mp['peak_rank'], 0) + 1
    return owners


def row_features(r) -> list:
    """What quantification recorded about an analyte."""
    cas = r.get('_cascade') or {}
    cv = cas.get('covar') or {}
    fp = r.get('_fingerprint_at_amount') or ()
    verd = [str(v[3]).split(' ')[0] for v in fp if len(v) > 3]
    ax = r.get('_confidence_axes') or {}
    rmin = _f(cv.get('r_min'), np.nan)
    sig = [v[3] for v in cas.get('signals') or ()]
    return [
        np.log1p(max(_f(r.get('snr')), 0.0)),
        _f(r.get('_ann_score')),
        float(bool(r.get('_is_singlet'))),
        float(bool(r.get('_by_cascade'))),
        _f(r.get('_identity_level')),
        float(verd.count('present')),
        float(verd.count('contradicted')),
        float(verd.count('faint')),
        _f(ax.get('shape'), -1.0),
        float(bool(ax.get('shared'))),
        _f(r.get('_shape_explained'), -1.0),
        _f(r.get('_share_kept'), -1.0),
        _f(r.get('_neighbour_share'), -1.0),
        _f(r.get('_captured_fraction'), -1.0),
        float(r.get('_rejection') == 'fit_misplaced'),
        float(r.get('_rejection') == 'cascade_crowded'),
        float(r.get('_rejection') == 'singlet_off_drift'),
        np.log1p(_f(cas.get('apex_snr'))),
        np.log1p(float(np.sqrt(np.sum(np.square(sig)))) if sig else 0.0),
        min(_f(cas.get('disagreement'), 3.0), 3.0),
        _f(cas.get('free_share'), 1.0),
        float(np.isfinite(rmin)),
        rmin if np.isfinite(rmin) else 0.0,
        np.log1p(_f(r.get('n_peaks'))),
    ]


def annotation_features(met, area_by_rank, dilution, owners) -> list:
    """What annotation found of an analyte's listing: how much and how well
    it matched, its coupling, its position against the study's drift, how
    its signals co-vary across the samples, how many rivals its peaks
    have."""
    if not met:
        return [0.0] * 20
    mps = [mp for mp in met.get('matched_peaks') or () if not mp.get('is_sub_peak')]
    listed = len(met.get('db_peaks') or ()) or 1
    by_sig = {}
    for mp in mps:
        by_sig.setdefault(round(_f(mp.get('db_ppm')), 4), []).append(mp)
    rank = {'good': 3, None: 2, 'bad': 0}
    best = [max(v, key=lambda m: rank.get(m.get('annotation_quality'), 1))
            for v in by_sig.values()]
    q = [m.get('annotation_quality') for m in best]
    reasons = [str((m.get('_status_reasons') or [''])[0]) for m in best
               if m.get('annotation_quality') == 'bad']
    mult = [str(m.get('type') or m.get('multiplicity') or '') for m in best]
    snr = [_f(m.get('obs_snr')) for m in best]
    dj = [_f(m.get('delta_j_hz')) for m in best if m.get('delta_j_hz') is not None]
    off = [abs(_f(m.get('ppm_off_drift'))) for m in best if m.get('ppm_off_drift') is not None]
    nh_all = sum(_f(dp.get('protons'), 1.0) for dp in met.get('db_peaks') or ()) or 1.0
    nh_found = sum(_f(m.get('protons'), 1.0) for m in best
                   if m.get('annotation_quality') != 'bad')
    cv = analyte_covariation(met, area_by_rank, dilution) if dilution.size else None
    ranks = {m.get('peak_rank') for m in best if m.get('peak_rank') is not None}
    shared = sum(1 for r in ranks if owners.get(r, 0) > 1)
    return [
        np.log1p(listed), np.log1p(len(best)), len(best) / listed,
        float(q.count('good')), q.count('good') / listed,
        float(sum('Incomplete' in r for r in reasons)),
        float(sum('co-vary' in r for r in reasons)),
        float(sum('Incomplete' not in r and 'co-vary' not in r for r in reasons)),
        float(all(x in ('s', '') for x in mult)),
        np.log1p(max(max(snr), 0.0) if snr else 0.0), nh_found / nh_all,
        min(dj) if dj else 0.0, float(bool(dj)),
        1000 * min(off) if off else 50.0, 1000 * float(np.median(off)) if off else 50.0,
        cv[0] if cv else 0.0, cv[1] if cv else 0.0, float(cv is not None),
        float(bool(met.get('_fingerprint_contradictions'))), shared / max(len(ranks), 1),
    ]


def features(row, met, area_by_rank, dilution, owners) -> np.ndarray:
    x = row_features(row) + annotation_features(met, area_by_rank, dilution, owners)
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


# Fitted by ``python -m validation.identity_score_probe fit``; see the module
# docstring. Standardisation then logistic weights, intercept apart.
SCORE_MEAN = (3.3883, 0.21388, 0.21277, 0.65432, 1.5535, 0.85369, 0, 0.13002, -0.3659, 0.092461, -0.84486, -0.76534, -0.64661, -0.76111, 0.14106, 0.15104, 0.013134, 2.3555, 2.5326, 1.8296, 0.92161, 0.41424, 0.042391, 1.0827, 1.5263, 1.4826, 0.9543, 1.088, 0.26432, 0.92199, 0.42474, 0.5264, 0.17625, 5.3, 0.56123, 0.29036, 0.76333, 7.7556, 17.632, 0.073103, -0.015562, 0.55529, 0.083005, 0.95973)
SCORE_SCALE = (1.7256, 0.33587, 0.40926, 0.47559, 1.0102, 1.6878, 1e-09, 1.0887, 1.4067, 0.28968, 0.4993, 0.63138, 0.49065, 0.61929, 0.34808, 0.35809, 0.11385, 1.8585, 1.9663, 1.2256, 0.14536, 0.49259, 0.29791, 0.64445, 0.46452, 0.44396, 0.11139, 1.2218, 0.29177, 1.143, 0.72448, 0.90341, 0.38104, 1.07, 0.36049, 1.3018, 0.42504, 8.4801, 11.433, 0.27531, 0.31758, 0.49693, 0.27589, 0.11835)
SCORE_COEF = (0.11224, 1.2061, 0.76285, 1.0932, 0.55848, 0.13291, 0, 0.066585, -0.020036, 0.03088, 0.1175, 0.43393, 0.17646, 0.52756, -0.73719, 0.69068, -0.25996, -0.47563, 0.83416, -0.41195, 0.18038, 0.30211, 0.39379, 0.61128, -0.71047, 0.63016, 0.076017, 0.011934, -0.031573, -0.20719, -0.14024, -0.077242, -0.20773, 0.07609, 0.20464, 0.017669, 0.092547, -0.039756, -0.59105, 0.29119, 0.062436, -0.11155, -0.066201, -0.25064)
SCORE_INTERCEPT = -0.56762


def score(x) -> float:
    """The probability the model gives that the analyte is in the sample."""
    if SCORE_COEF is None:
        return float('nan')
    z = (np.asarray(x, dtype=float) - np.asarray(SCORE_MEAN)) / np.asarray(SCORE_SCALE)
    t = float(np.dot(z, np.asarray(SCORE_COEF)) + SCORE_INTERCEPT)
    return float(1.0 / (1.0 + np.exp(-t)))


# The score a row needs to be reported, when the engine decides by it. None:
# the engine's gates decide alone (and the cascade and the complex multiplets
# keep their own bars). 0.3 reported as many strangers as the gates on the
# extract-like tubes, held out (thresholds 0.27 and 0.31). On by the user's
# decision (ADR-086), measured on new seeds: extract-like tubes of 50 and 75
# +28 real / +19 strangers; standard tubes 18 51/20 -> 53/21, 50 136/51 ->
# 122/65, 100 249/53 -> 246/81; three and two dots as clean as before, the
# rest at one dot.
IDENTITY_SCORE_MIN = 0.3
IDENTITY_UNLIKELY = 'identity_unlikely'


# Whether, when the score decides, the cascade's bar and the complex
# multiplets' rule open so more candidates reach it. On the standard tubes
# opening them brought strangers the score could not hold (tube 50: 34
# strangers and 1 real through the cascade; found 136 -> 122, strangers
# 51 -> 65; tube 100 strangers 53 -> 81).
IDENTITY_SCORE_OPENS_GATES = True


def deciding() -> bool:
    """Whether the score decides who is reported."""
    return IDENTITY_SCORE_MIN is not None


def opens_gates() -> bool:
    """Whether the gates before the score let more through."""
    return deciding() and IDENTITY_SCORE_OPENS_GATES


def _positive_concs(row) -> list:
    c = row.get('_concs_per_sample') or {}
    vals = c.values() if isinstance(c, dict) else c
    return [float(v) for v in vals if v is not None and np.isfinite(v) and v > 0]


def apply(rows, metabolites, peaks, included=None, protected=()) -> list:
    """Score every row that has an amount, and let the score decide: a row
    reported below IDENTITY_SCORE_MIN is withdrawn, a row the gates refused
    that holds an amount (AMOUNT_REJECTIONS) is reported at one dot when it
    reaches it. Every scored row carries ``_identity_score``.

    The analyst's word stands over the score: an analyte marked Good or
    accepted in the review (``protected``, or ``_analyst_decision``) is
    scored but never withdrawn."""
    if not deciding():
        return rows
    from processing.quant_engine import STATUS_NONE, STATUS_WEAK
    area, dil = peak_context(peaks, included)
    owners = peak_owners(metabolites)
    mets = {str(m.get('id')): m for m in metabolites or ()}
    bar = float(IDENTITY_SCORE_MIN)
    for row in rows:
        if row.get('_is_is'):
            continue
        reported = not row.get('_not_quantified')
        refused = row.get('_rejection') in AMOUNT_REJECTIONS
        if not (reported or refused) or not _positive_concs(row):
            continue
        s = score(features(row, mets.get(str(row.get('hmdb_id'))), area, dil, owners))
        row['_identity_score'] = s
        mine = (str(row.get('hmdb_id')) in {str(m) for m in protected}
                or row.get('_analyst_decision') == 'accepted')
        if reported and s < bar and not mine:
            row['_not_quantified'] = True
            row['_rejection'] = IDENTITY_UNLIKELY
            row['_confidence'] = 0
            row['status'] = STATUS_NONE
            row['_status_reasons'] = [
                f'Not quantified: the evidence about this analyte taken together '
                f'(identity score {s:.2f}, below {bar:.2f}) says it is more '
                f'likely a stranger than an analyte in the sample.'
            ] + list(row.get('_status_reasons') or ())
        elif refused and s >= bar:
            vals = _positive_concs(row)
            was = row.get('_rejection')
            row['_not_quantified'] = False
            row['_rejection'] = None
            row['_confidence'] = 1
            row['status'] = STATUS_WEAK
            row['conc_mm'] = float(np.mean(vals))
            row['_status_reasons'] = [
                f'Reported on the evidence taken together (identity score '
                f'{s:.2f}) although a gate had refused it ({was}). One dot.'
                + (' Its amount is the integral of its window: an upper bound '
                   'where the window holds neighbours.' if was == 'fit_misplaced' else '')
            ] + list(row.get('_status_reasons') or ())
    return rows
