# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/integration/test_signal_links_pipeline.py
"""The links through the whole pipeline: a synthetic study detected, annotated and
quantified for real. They speak only about the rows the quantification made: the
same analytes, the same amounts, whatever the switch; quantified twice, the same
table and the same annotation; saved and reopened, the same marks."""
import contextlib
import copy
import io
import os

import pytest

from data import project_file
from processing import signal_links as sl
from validation import fresh_run
from validation import mixture_bench as mb

pytestmark = [pytest.mark.skipif(not os.path.exists(mb.DB_DEFAULT),
                                 reason='reference database not present'),
              pytest.mark.timeout(300)]

PANEL = [('Creatine', 1.5, 1.0), ('L-lactic acid', 4.0, 1.0), ('Taurine', 2.0, 1.0),
         ('Glycine', 0.8, 1.0), ('L-threonine', 1.2, 1.0), ('L-alanine', 2.5, 1.0),
         ('L-valine', 1.0, 1.0), ('Choline', 1.0, 1.0), ('Betaine', 1.5, 1.0),
         ('L-glutamine', 2.0, 1.0), ('L-histidine', 0.8, 1.0)]


@pytest.fixture(scope='module')
def prep():
    X, *_ = mb.build_study(mb.DB_DEFAULT, seed=0, panel=PANEL, bio_cv=0.4)
    with contextlib.redirect_stdout(io.StringIO()):
        return fresh_run.prepare(mb.PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT,
                                 field_mhz=mb.MHZ, log=lambda *a: None)


def _quantify(prep):
    return fresh_run.quantify_prepared(
        prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM, standard_nh=mb.STANDARD_NH,
        standard_conc=mb.STANDARD_MM, standard_name='DSS')


def _rows(rows):
    return {r['metabolite']: r for r in rows if not r.get('_is_is') and not r.get('_not_quantified')}


def _amounts(rows):
    return {k: (r.get('_integ_mode'), r.get('_multiplicity'), round(float(r.get('conc_mm') or 0), 6),
                tuple(round(float(x), 5) for x in r.get('_line_ppms') or ()))
            for k, r in _rows(rows).items()}


def _marked(mets):
    return [(m['name'], mp['db_ppm']) for m in mets for mp in m['matched_peaks'] if sl._MARK_KEY in mp]


def test_the_switch_changes_dots_and_marks_and_no_amount(prep, monkeypatch):
    p = copy.deepcopy(prep)
    on = _quantify(p)
    monkeypatch.setattr(sl, 'SIGNAL_LINKS', False)
    off = _quantify(copy.deepcopy(prep))
    assert _amounts(on) == _amounts(off)
    a, b = _rows(on), _rows(off)
    raised = [k for k in a if a[k]['_confidence'] != b[k]['_confidence']]
    assert raised and all(a[k]['_confidence'] == 3 and b[k]['_confidence'] == 2 for k in raised)
    assert all(a[k]['_signal_links_from'] == 2 and a[k].get(sl.LINKS_KEY) for k in raised)
    assert not any(sl.LINKS_KEY in r for r in off)


def test_quantifying_twice_gives_the_same_table_and_annotation(prep):
    p = copy.deepcopy(prep)
    first = _rows(_quantify(p))
    mets = copy.deepcopy(p['mets'])
    second = _rows(_quantify(p))
    assert {k: r['_confidence'] for k, r in first.items()} == {k: r['_confidence'] for k, r in second.items()}
    assert {k: r.get(sl.LINKS_KEY) for k, r in first.items()} == {k: r.get(sl.LINKS_KEY) for k, r in second.items()}
    assert p['mets'] == mets


def test_forgetting_gives_back_the_annotation_of_before_the_quantification(prep):
    p = copy.deepcopy(prep)
    before = copy.deepcopy(p['mets'])
    _quantify(p)
    sl.forget_marks(p['mets'])
    assert p['mets'] == before


def test_what_it_marked_is_saved_and_reopened_and_can_still_be_undone(prep):
    p = copy.deepcopy(prep)
    before = copy.deepcopy(p['mets'])
    _quantify(p)
    assert _marked(p['mets'])
    state = project_file.decode(project_file.encode(project_file.snapshot({'metabolites': p['mets']})))
    assert state['metabolites'] == p['mets'] and _marked(state['metabolites']) == _marked(p['mets'])
    sl.forget_marks(state['metabolites'])
    assert state['metabolites'] == before
