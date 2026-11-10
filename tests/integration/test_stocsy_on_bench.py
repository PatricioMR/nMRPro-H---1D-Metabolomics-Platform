# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_stocsy_on_bench.py
"""STOCSY on a spectrum whose answer is known: the synthetic bench (GISSMO
profiles of four metabolites, 12 samples, each analyte varying 40% between
samples). Detection, controller and statistics all run for real."""
import contextlib
import io
import os

import numpy as np
import pytest

from validation import mixture_bench as mb

pytestmark = pytest.mark.skipif(not os.path.exists(mb.DB_DEFAULT),
                                reason='reference database not present')

PANEL = [('Creatine', 1.5, 1.0), ('L-lactic acid', 4.0, 1.0),
         ('Taurine', 2.0, 1.0), ('Glycine', 0.8, 1.0)]


@pytest.fixture(scope='module')
def study():
    from controllers.stocsy_controller import StocsyAnalysis
    from processing.peak_detection import (
        detect_peaks, estimate_noise_minimum_segment)
    X, *_ = mb.build_study(mb.DB_DEFAULT, seed=0, panel=PANEL, bio_cv=0.4)
    incl = [s['index'] for s in mb.SAMPLES]
    sigma = float(estimate_noise_minimum_segment(X[:, incl].mean(axis=1)))
    with contextlib.redirect_stdout(io.StringIO()):
        peaks = detect_peaks(
            mb.PPM, X, noise_std=sigma, min_snr=6.0, min_distance=0.003,
            baseline_correct=False, global_align=False,
            samples_info=mb.SAMPLES, per_sample_detection=True,
            detect_shoulders=False, min_group_presence=3,
            excluded_regions=mb.WATER)
    return StocsyAnalysis(mb.PPM, X, peaks, mb.SAMPLES,
                          excluded_regions=mb.WATER)


def _near(an, ppm):
    return min(an.peaks.values(), key=lambda p: abs(p['ppm'] - ppm))['rank']


def test_creatines_two_singlets_are_each_others_partner(study):
    seed, other = _near(study, 3.03), _near(study, 3.93)
    top = study.partners(seed)[0]
    assert top['rank'] == other
    assert top['r'] > 0.95 and top['significant'] and top['p'] < 1e-4
    assert top['ci_low'] > 0.8


def test_other_molecules_in_the_region_do_not_pass_the_bar(study):
    seed = _near(study, 3.03)
    others = [d for d in study.partners(seed)
              if d['rank'] != _near(study, 3.93)]
    assert others
    assert not any(d['significant'] and d['r'] > 0.8 for d in others)


def test_the_trace_lights_up_the_partner_and_not_the_other_molecules(study):
    seed = _near(study, 3.03)
    tr = study.trace(seed)
    ppm = study.ppm
    at = lambda c: np.nanmax(np.abs(tr['r'][np.abs(ppm - c) <= 0.006]))
    # Point by point, not on areas: the 1.5 mppm jitter between samples (a
    # quarter of the linewidth) costs a few hundredths that the peak areas do not.
    assert at(3.928) > 0.85                     # creatine's other singlet
    assert at(1.326) < 0.8                      # lactate
    assert np.isnan(tr['r'][(ppm >= 4.5) & (ppm <= 5.0)]).all()   # water


def test_a_lactate_doublet_line_is_its_partner(study):
    a, b = _near(study, 1.333), _near(study, 1.319)
    assert a != b
    top = study.partners(a, region=(1.6, 1.0))[0]
    assert top['rank'] == b and top['r'] > 0.85 and top['significant']


def test_the_window_draws_it(study):
    tk = pytest.importorskip('tkinter')
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    root.withdraw()
    try:
        from controllers.stocsy_analytes import AnalyteBrowser
        from gui.widgets.stocsy_window import StocsyWindow
        seed, partner = _near(study, 3.03), _near(study, 3.93)
        w = StocsyWindow(root, study, browser=AnalyteBrowser(study), view_region=(4.3, 2.9),
                         initial={'ref_rank': seed, 'sought_rank': partner})
        assert w.ref.ranks == [seed] and w.sought.ranks == [partner]
        row = next(r for r in w.cand_rows if r['kind'] == 'peak' and r['rank'] == partner)
        assert row['r'] > 0.9 and row['p'] < 0.05                  # the same molecule co-varies
        assert w.win_tree.exists(f'p{partner}')
        w.win.destroy()
    finally:
        root.destroy()
