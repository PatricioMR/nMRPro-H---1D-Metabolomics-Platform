# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_signal_lines_pipeline.py
"""Lines chosen by the analyst, through the whole pipeline: a synthetic study
of four metabolites is detected and annotated for real, the lines of
lactate's doublet are chosen in the editor the STOCSY window uses, and the
quantification measures what was chosen -- and nothing else moves."""
import contextlib
import copy
import functools
import io
import os

import pytest

from controllers.signal_lines import SignalLineEditor
from processing.multiplet_line_selector import multiplet_line_peaks
from validation import fresh_run
from validation import mixture_bench as mb

pytestmark = [pytest.mark.skipif(not os.path.exists(mb.DB_DEFAULT),
                                 reason='reference database not present'),
              pytest.mark.timeout(300)]       # detection and annotation of the study run once here

PANEL = [('Creatine', 1.5, 1.0), ('L-lactic acid', 4.0, 1.0),
         ('Taurine', 2.0, 1.0), ('Glycine', 0.8, 1.0)]


@pytest.fixture(scope='module')
def prep():
    X, *_ = mb.build_study(mb.DB_DEFAULT, seed=0, panel=PANEL, bio_cv=0.4)
    with contextlib.redirect_stdout(io.StringIO()):
        return fresh_run.prepare(mb.PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT,
                                 field_mhz=mb.MHZ, log=lambda *a: None)


def _quantify(prep, marks, monkeypatch):
    monkeypatch.setattr(fresh_run, 'quantify_project', functools.partial(
        fresh_run.quantify_project, quality_marks=marks))
    rows = fresh_run.quantify_prepared(
        prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM, standard_nh=mb.STANDARD_NH,
        standard_conc=mb.STANDARD_MM, standard_name='DSS')
    monkeypatch.undo()
    return rows


def _table(rows):
    return {r['metabolite']: (r.get('_integ_mode'), r.get('_multiplicity'),
                              tuple(sorted(round(x, 4) for x in r.get('_line_ppms') or ())),
                              round(float(r.get('conc_mm') or 0), 4), r.get('_confidence'))
            for r in rows if not r.get('_is_is') and not r.get('_not_quantified')}


def _doublet(p):
    lac = next(m for m in p['mets'] if m['name'] == 'L-lactic acid')
    k = next(i for i, mp in enumerate(lac['matched_peaks']) if mp['type'] == 'd')
    pk = {int(x['rank']): x for x in p['peaks']}
    lines = sorted(multiplet_line_peaks(lac['matched_peaks'][k], pk), key=lambda x: -x['ppm'])
    return lac, k, lines


def test_choosing_the_annotations_own_lines_is_marking_the_signal_good(prep, monkeypatch):
    """The same lines, chosen by hand: the anchor and the position stay, so
    the quantification is the one a Good mark alone gives -- every analyte."""
    lac, k, lines = _doublet(prep)
    good = _table(_quantify(prep, {(lac['id'], lac['matched_peaks'][k]['peak_rank']): 'good'},
                            monkeypatch))
    p2, marks = copy.deepcopy(prep), {}
    lac2, k2, lines2 = _doublet(p2)
    SignalLineEditor(p2['mets'], marks).apply(lac2['id'], k2, lines2)
    assert _table(_quantify(p2, marks, monkeypatch)) == good


def test_the_quantification_measures_the_lines_chosen(prep, monkeypatch):
    p2, marks = copy.deepcopy(prep), {}
    lac, k, lines = _doublet(p2)
    SignalLineEditor(p2['mets'], marks).apply(lac['id'], k, lines)
    row = next(r for r in _quantify(p2, marks, monkeypatch) if r['metabolite'] == 'L-lactic acid')
    assert row['_integ_mode'] == 'Multiplet' and row['_multiplicity'] == 'd'
    assert sorted(round(x, 4) for x in row['_line_ppms']) == sorted(round(x['ppm'], 4) for x in lines)


def test_one_line_of_a_doublet_is_not_scaled_to_the_whole_signal(prep, monkeypatch):
    """Half the pattern is under what the quantification scales: the doublet
    is refused on it, and says why -- it is not quantified on one line."""
    p2, marks = copy.deepcopy(prep), {}
    lac, k, lines = _doublet(p2)
    SignalLineEditor(p2['mets'], marks).apply(lac['id'], k, lines[:1])
    row = next(r for r in _quantify(p2, marks, monkeypatch) if r['metabolite'] == 'L-lactic acid')
    assert row['_integ_mode'] != 'Multiplet' or row['_multiplicity'] != 'd'
    assert any('partial multiplet' in s for s in row.get('_status_reasons') or ())


def test_the_window_reads_a_clean_signal_on_the_quantifications_scale(prep, monkeypatch):
    """The STOCSY window measures a signal with the quantification's own
    function: creatine's singlet, alone in its region, gives the area the
    quantification reported (detection's foot-to-foot areas gave about half).
    One line of lactate's doublet is refused, and says why."""
    from controllers.stocsy_analytes import AnalyteBrowser
    from controllers.stocsy_controller import StocsyAnalysis
    rows = _quantify(prep, {}, monkeypatch)
    an = StocsyAnalysis(prep['ppm'], prep['corr'], prep['peaks'], mb.SAMPLES,
                        excluded_regions=prep['excluded'], noise_std=prep['noise_std'])
    b = AnalyteBrowser(an, prep['mets'], rows)
    cre = next(m['id'] for m in prep['mets'] if m['name'] == 'Creatine')
    ref = b.quant_target(cre)
    i = next(i for i, s in enumerate(b.signals(cre)) if abs(s['ppm'] - 3.03) < 0.01)
    row = b.compare(ref, [b.signal_target(cre, i)])[0]
    s = b.signals(cre)[i]
    assert row['ratio'] * ref.protons / s['protons'] == pytest.approx(1.0, abs=0.1)
    lac = next(m['id'] for m in prep['mets'] if m['name'] == 'L-lactic acid')
    d = next(s for s in b.signals(lac) if s['type'] == 'd')
    half = b.lines_target(d, d['lines'][:1])
    assert half.vector is None and 'under the 75%' in half.why


def test_restoring_the_lines_gives_back_the_quantification_before(prep, monkeypatch):
    before = _table(_quantify(prep, {}, monkeypatch))
    p2, marks = copy.deepcopy(prep), {}
    lac, k, lines = _doublet(p2)
    ed = SignalLineEditor(p2['mets'], marks)
    ed.apply(lac['id'], k, lines[:1])
    ed.restore(lac['id'], k)
    assert marks == {} and _table(_quantify(p2, marks, monkeypatch)) == before
