"""Is the evidence annotation already holds about each candidate enough to
tell real analytes from strangers better than the engine's chain of gates?

For every metabolite annotation proposed (a matched signal), features read
from the annotation alone -- how much of its listing was matched and at
what quality, coupling, position against the study's drift, signal-to-noise,
how its signals co-vary across the samples (probabilistic-quotient
normalised, covariation_probe) -- labelled by the tube's truth. A logistic
model is fitted leaving one regime out and judged on it: real / strangers
at the threshold that reports as many strangers as the engine did, against
the engine's own real / strangers (bench_runs rows of ``ENGINE_LABELS``).

    python -m validation.analyte_classifier_probe
"""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from validation.bench_runs import OUT_DEFAULT, _path
from processing.identity_score import annotation_features, peak_context, peak_owners
from validation.identity_score_probe import fit_logistic, predict

# (name, tubes, signal shift, compound shift, amount sd, dilution sd, seeds, engine label)
REGIMES = (
    ('clean', ('r50', 'r75'), 0.0, 0.0, 0.12, 0.0, (100, 101, 102), 'rec1'),
    ('extract', ('r50', 'r75'), 4.0, 6.0, 0.12, 0.0, (100, 101, 102), 'x2'),
    ('extract+bio', ('r50', 'r75'), 4.0, 6.0, 0.5, 0.3, (100, 101, 102), 'x3b'),
)
CACHE = os.path.join(OUT_DEFAULT, 'analyte_features.pkl')

FEATURES = ('listed', 'matched', 'match share', 'good', 'good share', 'bad incomplete',
            'bad covary', 'bad other', 'singlet only', 'best snr', 'listed intensity found',
            'dj best', 'dj measured', 'off drift min', 'off drift med', 'covar med', 'covar min',
            'covar measured', 'contradicted', 'shared peaks')


def _one(job):
    name, t, s, sh, rs, asd, dsd = job
    from validation import gissmo_bench as gb
    gb.SIGNAL_SHIFT, gb.AMOUNT_SD, gb.DILUTION_SD = sh / 1000.0, asd, dsd
    prep, truth = gb.prepared(gb.FIELDS[600], s, t, ref_shift=rs / 1000.0)
    area, dil = peak_context(prep['peaks'])
    owners = peak_owners(prep['mets'])
    out = []
    for met in prep['mets']:
        if not met.get('matched_peaks'):
            continue
        out.append({'regime': name, 'run': (t, s), 'mid': str(met.get('id')),
                    'real': str(met.get('id')) in truth,
                    'x': annotation_features(met, area, dil, owners)})
    return out, len(truth)


def collect(jobs=6):
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, 'rb'))
    work = [(n, t, s, sh, rs, a, d) for n, tubes, sh, rs, a, d, seeds, _l in REGIMES
            for t in tubes for s in seeds]
    recs, n_truth = [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for job, (out, nt) in zip(work, ex.map(_one, work)):
            recs += out
            n_truth[(job[0], job[1], job[2])] = nt
    pickle.dump((recs, n_truth), open(CACHE, 'wb'))
    return recs, n_truth


def engine_counts(label, tubes, sh, rs, seeds):
    real = fake = 0
    for t in tubes:
        for s in seeds:
            p = _path(OUT_DEFAULT, label, t, sh, s, 600, 0.0, rs)
            if not os.path.exists(p):
                continue
            d = pickle.load(open(p, 'rb'))
            for r in d['rows']:
                if r.get('_not_quantified') or r.get('_is_is'):
                    continue
                if r['hmdb_id'] in d['truth']:
                    real += 1
                else:
                    fake += 1
    return real, fake


def main():
    recs, n_truth = collect()
    X = np.nan_to_num(np.array([r['x'] for r in recs], float), nan=0.0, posinf=0.0, neginf=0.0)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = np.array([r['real'] for r in recs], float)
    score = np.zeros(len(recs))
    names = [g[0] for g in REGIMES]
    for g in names:
        test = np.array([r['regime'] == g for r in recs])
        w = fit_logistic(Z[~test], y[~test])
        score[test] = predict(w, Z[test])
    print('| regime | truth | engine real / strangers | classifier real at the same strangers | classifier at strangers x1.5 |')
    print('|---|---|---|---|---|')
    for name, tubes, sh, rs, a, d, seeds, label in REGIMES:
        sel = [i for i, r in enumerate(recs) if r['regime'] == name]
        er, ef = engine_counts(label, tubes, sh, rs, seeds)
        order = sorted(sel, key=lambda i: -score[i])
        def at(nf):
            rr = ff = 0
            for i in order:
                if recs[i]['real']:
                    rr += 1
                else:
                    if ff >= nf:
                        break
                    ff += 1
            return rr
        tot = sum(v for k, v in n_truth.items() if k[0] == name)
        print(f'| {name} | {tot} | {er} / {ef} | {at(ef)} | {at(int(1.5 * ef))} |')
    w = fit_logistic(Z, y)
    print('\nweights (all regimes):', ', '.join(f'{n} {v:+.2f}' for n, v in
                                                sorted(zip(FEATURES, w[1:]), key=lambda t: -abs(t[1]))))


if __name__ == '__main__':
    main()
