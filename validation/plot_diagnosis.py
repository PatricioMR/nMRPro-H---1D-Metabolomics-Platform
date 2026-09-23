"""Draw what the software measured against what the study really holds.

Two panels: every measurement on an identity line with the +-10% and +-20%
bands, and the signed error of each analyte with its confidence. The same
figure will serve a real study once the truth comes from spiked standards
instead of from ``mixture_bench``.

    python -m validation.plot_diagnosis validation/out/MixtureBench_v01.nmrproj

Colours are the confidence the results table shows, so a point that sits
far from the line and reads three dots is a thing to go and look at.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os

import numpy as np

REAL, WARM, FAINT = '#2a7f62', '#c8922b', '#b5443a'
GREY = '#8a8a8a'
DOT_COLOUR = {3: REAL, 2: WARM, 1: FAINT}


def measurements(rows, truth):
    """``(put_in_mM, measured_mM, dots, name)`` for every analyte x group.

    Only the analytes the study really holds: an analyte that is not in
    ``truth`` has no true amount to be plotted against.
    """
    out = []
    for r in rows:
        if r.get('_not_quantified') or r.get('_is_is'):
            continue
        if r['hmdb_id'] not in truth:
            continue
        for group, vals in (r.get('_concs_by_group') or {}).items():
            want = truth[r['hmdb_id']].get(group)
            if not want:
                continue
            out.append((float(want), float(np.median(vals)),
                        int(r.get('_confidence') or 0), r['metabolite']))
    return out


def draw(points, path, *, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5))
    if title:
        fig.suptitle(title, fontsize=11)

    put = [p[0] for p in points]
    lo, hi = min(put) * 0.6, max(put) * 1.5
    left.plot([lo, hi], [lo, hi], color=GREY, lw=1)
    for factor, style in ((1.10, '--'), (1.20, ':')):
        left.plot([lo, hi], [lo * factor, hi * factor], style, color=GREY, lw=0.8)
        left.plot([lo, hi], [lo / factor, hi / factor], style, color=GREY, lw=0.8)
    for want, got, dots, _name in points:
        left.scatter(want, got, s=42, zorder=3, edgecolor='white',
                     linewidth=0.6, color=DOT_COLOUR.get(dots, GREY))
    left.set_xscale('log')
    left.set_yscale('log')
    left.set_xlim(lo, hi)
    left.set_ylim(lo, hi)
    left.set_xlabel('concentration put in the study (mM)')
    left.set_ylabel('concentration the software measured (mM)')
    left.set_title(f'(a) {len(points)} measurements (analyte x group); '
                   'line: identity, -- ±10%, ·· ±20%', loc='left', fontsize=9)
    left.legend(handles=[Patch(color=DOT_COLOUR[k],
                               label='●' * k + '○' * (3 - k)) for k in (3, 2, 1)],
                fontsize=8, frameon=False, loc='upper left',
                title='confidence', title_fontsize=8)

    per: dict = {}
    for want, got, dots, name in points:
        per.setdefault(name, []).append((got / want - 1.0, dots, want))
    order = sorted(per, key=lambda n: per[n][0][2])   # index 0 at the bottom
    for i, name in enumerate(order):
        errs = [e for e, _d, _w in per[name]]
        right.barh(i, 100 * float(np.mean(errs)), height=0.62,
                   color=DOT_COLOUR.get(per[name][0][1], GREY))
        right.plot([100 * min(errs), 100 * max(errs)], [i, i], color='k',
                   lw=1, alpha=0.5)
    right.axvline(0, color='k', lw=1)
    for band in (10, 20):
        right.axvspan(-band, band, color=GREY, zorder=0,
                      alpha=0.16 if band == 10 else 0.10)
    right.set_yticks(np.arange(len(order)))
    right.set_yticklabels(order, fontsize=8)
    right.set_xlabel('error in concentration (%) — bar: mean of the groups, '
                     'whisker: their range')
    right.set_xlim(-30, 30)
    right.set_title('(b) Error per analyte, most concentrated at the top',
                    loc='left', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94 if title else 1.0])
    fig.savefig(path, bbox_inches='tight')
    return path


def main():
    from validation import mixture_bench as mb
    from validation.diagnose_project import run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path', help='the .nmrproj to diagnose and draw')
    ap.add_argument('--out', default=None, help='where to write the PNG')
    ap.add_argument('--db', default=None,
                    help='the reference the annotation searches')
    ap.add_argument('--title', default='What the software measured against '
                                       'what the study holds')
    a = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        _x, _s, truth, _f = mb.build_study(mb.DB_DEFAULT, seed=0)
        _st, _mets, rows = run(a.path, db_path=a.db, log=lambda *args: None)
    points = measurements(rows, truth)
    if not points:
        raise SystemExit('nothing was quantified that the study is known to '
                         'hold: there is nothing to draw')
    out = a.out or os.path.join(os.path.dirname(a.path) or '.',
                                'diagnosis.png')
    print(draw(points, out, title=a.title))
    got = len({n for _w, _g, _d, n in points})
    print(f'  {len(points)} measurements over {got} analytes of {len(truth)}')


if __name__ == '__main__':
    main()
