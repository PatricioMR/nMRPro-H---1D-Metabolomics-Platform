# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Does the window hold the whole signal, foot to foot?

The study is a sum of known compounds, so the one signal a window was
built on can be rebuilt alone -- not the analyte's whole trace, which in
the aliphatic region carries its other signals a few hundredths away and
would make a window that holds all of one doublet look like half of two.
Integrating the window on that single signal says how much of it the
window holds; integrating the same window on the whole study says how
much of what it holds is that signal's.

    python -m validation.window_coverage [path.nmrproj]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from processing.quant_integration import integrate_sample
from validation import mixture_bench as mb
from validation.diagnose_project import DEFAULT, run


def one_signal_trace(ppm, sigs, db_ppm, conc):
    """The trace of the single listed signal nearest ``db_ppm``, alone."""
    s = min(sigs, key=lambda q: abs(float(q['ppm']) - float(db_ppm)))
    return mb.compound_trace(ppm, [s], conc, shift=mb.DRIFT_PPM), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default=DEFAULT)
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        _X0, _s, truth, _f = mb.build_study(mb.DB_DEFAULT, seed=0)
    st, _mets, rows = run(a.path, log=lambda *x: None)
    ppm = np.asarray(st['ppm'], dtype=float)
    X = np.asarray(st['all_intensities'], dtype=float)
    incl = [s['index'] for s in st['samples_info']]
    mean = X[:, incl].mean(axis=1)

    con = sqlite3.connect(mb.DB_DEFAULT)
    sigs_of = {}
    for name, conc, _factor in mb.PANEL:
        mid, sigs = mb._db_signals(con, name)
        sigs_of[mid] = (sigs, conc)

    print(f"{'analyte':22s} {'signal':>7s} {'window':>15s} {'width':>7s} "
          f"{'holds':>6s} {'own':>5s} {'error':>7s}")
    print('-' * 78)
    held, purity = [], []
    for r in rows:
        mid = r.get('hmdb_id')
        if (r.get('_not_quantified') or r.get('_is_is') or mid not in sigs_of):
            continue
        mw = r.get('_multiplet_window')
        hit = (r.get('_hits') or [{}])[0] or {}
        db_ppm = hit.get('db_ppm')
        if not mw or mw[0] is None or db_ppm is None:
            continue
        lo, hi = min(map(float, mw)), max(map(float, mw))
        sigs, conc = sigs_of[mid]
        y, s = one_signal_trace(ppm, sigs, db_ppm, conc)
        whole = float(np.trapezoid(y, x=ppm))          # the signal, all of it
        inside = float(integrate_sample(ppm, y, lo, hi, local_baseline=False))
        here = float(integrate_sample(ppm, mean, lo, hi, local_baseline=False))
        f_held = inside / whole if whole > 0 else float('nan')
        f_pure = inside / here if here > 0 else float('nan')
        t = truth[mid]
        g = r.get('_concs_by_group') or {}
        got = float(np.median(g.get('A') or [np.nan]))
        err = (got / t['A'] - 1) * 100 if t['A'] else float('nan')
        held.append(f_held)
        purity.append(f_pure)
        print(f"{r['metabolite'][:22]:22s} {float(db_ppm):7.3f} "
              f"{lo:7.3f}-{hi:6.3f} {1000*(hi-lo):6.1f}m "
              f"{100*f_held:5.0f}% {100*f_pure:4.0f}% {err:+6.0f}%")
    held = np.array([h for h in held if np.isfinite(h)])
    purity = np.array([p for p in purity if np.isfinite(p)])
    print(f"\n  of the signal it is on, the window holds: "
          f"median {100*np.median(held):.0f}%, worst {100*held.min():.0f}%")
    print(f"  of what the window holds, that signal's own: "
          f"median {100*np.median(purity):.0f}%, worst {100*purity.min():.0f}%")


if __name__ == '__main__':
    main()
