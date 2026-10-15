"""How well do an analyte's own signals move together across the samples,
for every candidate annotation proposed -- real or stranger?

For each annotated metabolite with two or more matched peaks (not Bad),
the peaks' areas in every sample (detection's ``area_per_sample``), each
sample divided by its probabilistic-quotient dilution (Dieterle F et al.,
Anal Chem 2006;78:4281-4290: the median, over every peak, of the sample's
area over the peak's median area), logged, and correlated pairwise across
the samples: STOCSY's evidence (Cloarec O et al., Anal Chem 2005;77:1282)
read on peaks. Reported per regime: real against stranger, the median of
the pairwise r and its AUC.

    python -m validation.covariation_probe
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from processing.identity_score import analyte_covariation, pqn_dilution

# (name, tube sizes, signal shift mppm, compound shift mppm, amount sd, dilution sd)
REGIMES = (
    ('clean', ('r50', 'r75'), 0.0, 0.0, 0.12, 0.0),
    ('extract', ('r50', 'r75'), 4.0, 6.0, 0.12, 0.0),
    ('extract + biology', ('r50', 'r75'), 4.0, 6.0, 0.5, 0.3),
)
SEEDS = (100, 101, 102)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not pos.size or not neg.size:
        return float('nan')
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main():
    from validation import gissmo_bench as gb
    print('| regime | real candidates (scored) | strangers (scored) | median r real | median r stranger | AUC median r | AUC min r |')
    print('|---|---|---|---|---|---|---|')
    for name, tubes, sh, rs, asd, dsd in REGIMES:
        gb.SIGNAL_SHIFT = sh / 1000.0
        gb.AMOUNT_SD, gb.DILUTION_SD = asd, dsd
        R, F = [], []
        for t in tubes:
            for s in SEEDS:
                prep, truth = gb.prepared(gb.FIELDS[600], s, t, ref_shift=rs / 1000.0)
                area = {int(p['rank']): np.asarray(p['area_per_sample'], float)
                        for p in prep['peaks'] if p.get('rank') is not None}
                A = np.array(list(area.values()))
                dil = pqn_dilution(A)
                for met in prep['mets']:
                    c = analyte_covariation(met, area, dil)
                    if c is None:
                        continue
                    (R if str(met.get('id')) in truth else F).append(c)
        R, F = np.array(R), np.array(F)
        print(f'| {name} | {len(R)} | {len(F)} | {np.median(R[:, 0]):.2f} | {np.median(F[:, 0]):.2f} '
              f'| {auc(R[:, 0], F[:, 0]):.2f} | {auc(R[:, 1], F[:, 1]):.2f} |')
    gb.SIGNAL_SHIFT, gb.AMOUNT_SD, gb.DILUTION_SD = 0.0, 0.12, 0.0


if __name__ == '__main__':
    main()
