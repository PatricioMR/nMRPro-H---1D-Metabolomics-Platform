# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Run the analysis again and the table is the same.

On the reference study, end to end through the application's own
annotation and quantification: the peaks handed to annotation in another
order, and the metabolites handed to quantification in another order, must
change nothing. Before the inputs were read in a canonical order, the
reversed peaks changed 186 of 293 annotated metabolites and a shuffled list
changed which analytes were quantified at two dots.
"""
import contextlib
import io
import random

import pytest

pytestmark = pytest.mark.timeout(600)


def _signature_mets(mets):
    def one(m):
        return repr((m['id'], round(float(m.get('best_score') or 0), 9),
                     round(float(m.get('composite_score') or 0), 9),
                     round(float(m.get('cosine') or 0), 9),
                     sorted(repr((mp.get('peak_rank'), mp.get('atom_name') or '',
                                  mp.get('annotation_quality')))
                            for mp in m['matched_peaks'])))
    return [one(m) for m in mets]


def _signature_rows(rows):
    return sorted(repr((r['metabolite'], int(r.get('_confidence') or 0),
                        None if r.get('_not_quantified')
                        else round(float(r.get('conc_mm') or 0), 9)))
                  for r in rows)


@pytest.fixture(scope='module')
def study():
    from validation import mixture_bench as mb
    from validation.fresh_run import prepare
    with contextlib.redirect_stdout(io.StringIO()):
        X, _s, _t, _f = mb.build_study(mb.DB_DEFAULT, seed=0)
        prep = prepare(mb.PPM, X, mb.SAMPLES, db_path=mb.DB_DEFAULT,
                       field_mhz=mb.MHZ, log=lambda *a: None)
    return mb, prep


def _quantify(mb, prep):
    from validation.fresh_run import quantify_prepared
    with contextlib.redirect_stdout(io.StringIO()):
        return quantify_prepared(prep, mb.SAMPLES, standard_ppm=mb.STANDARD_PPM,
                                 standard_nh=mb.STANDARD_NH,
                                 standard_conc=mb.STANDARD_MM,
                                 standard_name='DSS')


def test_quantification_does_not_depend_on_the_metabolite_order(study):
    mb, prep = study
    shuffled = dict(prep)
    shuffled['mets'] = list(prep['mets'])
    random.Random(7).shuffle(shuffled['mets'])
    assert _signature_rows(_quantify(mb, prep)) == _signature_rows(
        _quantify(mb, shuffled))


def test_quantifying_leaves_the_annotation_as_it_was(study):
    """Quantification wrote its working values (drift spread, fingerprint
    verdicts) onto the annotation's metabolites, so a second run started
    from the first one's: a Mice ISS project gave 45, then 39 every time."""
    import copy
    mb, prep = study
    before = copy.deepcopy(prep['mets'])
    first = _quantify(mb, prep)
    assert repr(prep['mets']) == repr(before)
    assert _signature_rows(_quantify(mb, prep)) == _signature_rows(first)


def test_annotation_does_not_depend_on_the_peak_order(study):
    mb, prep = study
    from validation.app_annotation import annotate
    kw = dict(db_path=mb.DB_DEFAULT, field_mhz=mb.MHZ,
              tolerance_ppm=prep['settings']['tolerance_ppm'],
              noise_std=prep['noise_std'], min_snr=prep['settings']['min_snr'],
              min_distance_ppm=prep['settings']['min_distance_ppm'],
              excluded_regions=prep['excluded'])
    with contextlib.redirect_stdout(io.StringIO()):
        a, _ = annotate(prep['peaks'], prep['ppm'], prep['raw'], mb.SAMPLES, **kw)
        b, _ = annotate(list(prep['peaks'])[::-1], prep['ppm'], prep['raw'],
                        mb.SAMPLES, **kw)
    assert _signature_mets(a) == _signature_mets(b)
