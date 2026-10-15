"""Can one identity score, learned on the bench's known answers, decide
better than the engine's chain of gates who is reported?

Offline, on rows bench_runs already computed: every analyte the engine
gave an amount -- reported, or refused with one (joint fit refused, the
cascade's crowded, the singlet off the drift) -- with the evidence the row
carries, labelled real or stranger by the tube's truth. A logistic model
(``sklearn`` not needed: Newton on the log-likelihood, L2) is fitted leaving
one regime out at a time and judged on the regime it never saw: how many
real / strangers each threshold reports against what the engine reported.

    python -m validation.identity_score_probe          # held-out tables
    python -m validation.identity_score_probe fit      # constants for the engine
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from processing.identity_score import (
    annotation_features, peak_context, peak_owners, row_features)
from validation.bench_runs import OUT_DEFAULT, _path

# (regime name, label, tubes, shift, ref_shift, seeds)
REGIMES = (
    ('clean r50/r75', 'rec1', ('r50', 'r75'), 0.0, 0.0, (100, 101, 102)),
    ('extract sig7', 'x0', ('r50', 'r75'), 7.0, 0.0, (100, 101, 102)),
    ('extract rs6+4', 'x2L', ('r50', 'r75'), 4.0, 6.0, (100, 101, 102)),
    ('extract rs6+4 bio', 'x3L', ('r50', 'r75'), 4.0, 6.0, (100, 101, 102)),
    ('standard 18', 't18on', ('18',), 0.0, 0.0, (0, 1, 2)),
    ('standard 50/100', 'std0', ('50', '100'), 0.0, 0.0, (1, 2)),
)
# Amount and dilution spread of the regimes that were run with them
# (gissmo_bench.AMOUNT_SD, DILUTION_SD), to rebuild their annotation.
BIOLOGY = {'extract rs6+4 bio': (0.5, 0.3)}
# Annotation's evidence about each analyte joins the row's
# (analyte_classifier_probe.met_features); False: the row's alone.
WITH_ANNOTATION = True
# The engine as it is, for the regimes run loose: the table compares the
# score over the loose run with these at the same number of strangers.
BASELINES = {'extract rs6+4': 'x2c', 'extract rs6+4 bio': 'x3c'}
AMOUNT_REJECTIONS = {'fit_misplaced', 'cascade_crowded', 'singlet_off_drift'}


def _annotation_features(name, t, s, sh, rs):
    """``{mid: annotation features}`` from the tube's annotation."""
    from validation import gissmo_bench as gb
    gb.SIGNAL_SHIFT = sh / 1000.0
    gb.AMOUNT_SD, gb.DILUTION_SD = BIOLOGY.get(name, (0.12, 0.0))
    try:
        prep, _truth = gb.prepared(gb.FIELDS[600], s, t, ref_shift=rs / 1000.0)
    finally:
        gb.SIGNAL_SHIFT, gb.AMOUNT_SD, gb.DILUTION_SD = 0.0, 0.12, 0.0
    area, dil = peak_context(prep['peaks'])
    owners = peak_owners(prep['mets'])
    return {str(m.get('id')): annotation_features(m, area, dil, owners)
            for m in prep['mets'] if m.get('matched_peaks')}


def collect():
    out = []
    cache = os.path.join(OUT_DEFAULT, 'identity_annotation_features_v2.pkl')
    ann = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    for name, label, tubes, sh, rs, seeds in REGIMES:
        for t in tubes:
            for s in seeds:
                p = _path(OUT_DEFAULT, label, t, sh, s, 600, 0.0, rs)
                if not os.path.exists(p):
                    continue
                d = pickle.load(open(p, 'rb'))
                truth = d['truth']
                key = (name, t, s)
                if WITH_ANNOTATION and key not in ann:
                    ann[key] = _annotation_features(name, t, s, sh, rs)
                    pickle.dump(ann, open(cache, 'wb'))
                af = ann.get(key, {})
                for r in d['rows']:
                    if r.get('_is_is'):
                        continue
                    rep = not r.get('_not_quantified')
                    if not rep and r.get('_rejection') not in AMOUNT_REJECTIONS:
                        continue
                    out.append({'regime': name, 'run': (t, s), 'mid': r['hmdb_id'],
                                'real': r['hmdb_id'] in truth, 'reported': rep,
                                'dots': int(r.get('_confidence') or 0) if rep else 0,
                                'x': row_features(r) + (list(af.get(r['hmdb_id'], [0.0] * 20))
                                                    if WITH_ANNOTATION else []),
                                'n_truth': len(truth)})
    return out


def fit_logistic(X, y, l2=1.0, iters=50):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
        H = (Xb.T * (p * (1 - p))) @ Xb + l2 * np.diag(np.r_[0.0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return w


def predict(w, X):
    return 1.0 / (1.0 + np.exp(-np.column_stack([np.ones(len(X)), X]) @ w))


def fit(recs):
    """Fit on every regime; print the constants for processing.identity_score."""
    X = np.nan_to_num(np.array([r['x'] for r in recs], float), nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logistic((X - mu) / sd, np.array([r['real'] for r in recs], float))
    fmt = lambda v: '(' + ', '.join(f'{x:.5g}' for x in v) + ')'
    print(f'SCORE_MEAN = {fmt(mu)}')
    print(f'SCORE_SCALE = {fmt(sd)}')
    print(f'SCORE_COEF = {fmt(w[1:])}')
    print(f'SCORE_INTERCEPT = {w[0]:.5g}')


def main():
    recs = collect()
    if len(sys.argv) > 1 and sys.argv[1] == 'fit':
        fit(recs)
        return
    regimes = sorted({r['regime'] for r in recs})
    X = np.nan_to_num(np.array([r['x'] for r in recs], float), nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    y = np.array([r['real'] for r in recs], float)
    score = np.zeros(len(recs))
    for g in regimes:
        test = np.array([r['regime'] == g for r in recs])
        w = fit_logistic(Z[~test], y[~test])
        score[test] = predict(w, Z[test])
    print('Held-out regime: engine as it is against the score at thresholds')
    print('| regime | engine real / strangers | ' + ' | '.join(
        f'score >= {t}' for t in (0.3, 0.4, 0.5, 0.6)) + ' |')
    print('|---|---|' + '---|' * 4)
    for g in regimes:
        sel = [i for i, r in enumerate(recs) if r['regime'] == g]
        er = sum(1 for i in sel if recs[i]['reported'] and recs[i]['real'])
        ef = sum(1 for i in sel if recs[i]['reported'] and not recs[i]['real'])
        cells = []
        for t in (0.3, 0.4, 0.5, 0.6):
            sr = sum(1 for i in sel if score[i] >= t and recs[i]['real'])
            sf = sum(1 for i in sel if score[i] >= t and not recs[i]['real'])
            cells.append(f'{sr} / {sf}')
        print(f'| {g} | {er} / {ef} | ' + ' | '.join(cells) + ' |')
    print('\nLoose gates + score, against the engine as it is (held-out regime)')
    print("| regime | engine real / strangers | loose engine real / strangers | loose + score, real at the engine's strangers | at 1.5x |")
    print('|---|---|---|---|---|')
    for g, base in BASELINES.items():
        reg = next(r for r in REGIMES if r[0] == g)
        er = ef = 0
        for t in reg[2]:
            for s_ in reg[5]:
                p = _path(OUT_DEFAULT, base, t, reg[3], s_, 600, 0.0, reg[4])
                if not os.path.exists(p):
                    continue
                d = pickle.load(open(p, 'rb'))
                for r in d['rows']:
                    if r.get('_not_quantified') or r.get('_is_is'):
                        continue
                    if r['hmdb_id'] in d['truth']:
                        er += 1
                    else:
                        ef += 1
        sel = [i for i, r in enumerate(recs) if r['regime'] == g]
        lr = sum(1 for i in sel if recs[i]['reported'] and recs[i]['real'])
        lf = sum(1 for i in sel if recs[i]['reported'] and not recs[i]['real'])
        order = sorted(sel, key=lambda i: -score[i])
        def at(nf):
            rr = ff = 0
            for i in order:
                if recs[i]['real']:
                    rr += 1
                elif ff >= nf:
                    break
                else:
                    ff += 1
            return rr
        thr = sorted((score[i] for i in sel if not recs[i]['real']), reverse=True)
        t_eq = thr[ef - 1] if 0 < ef <= len(thr) else float('nan')
        print(f'| {g} | {er} / {ef} | {lr} / {lf} | {at(ef)} (score >= {t_eq:.2f}) | {at(int(1.5 * ef))} |')
    from itertools import groupby
    try:
        from sklearn.metrics import roc_auc_score
        print('\nAUC held-out:', round(roc_auc_score(y, score), 3))
    except ImportError:
        pass


if __name__ == '__main__':
    main()
