# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The spectrometer field reaches the annotation and survives a save.

Every J the annotation predicts is converted from Hz to ppm with this
number. The Bruker loader read it to build the ppm axis and then dropped
it, the interface fell back to 600 MHz, and the project never saved it: the
500 MHz reference study, annotated at 600, found 10 of 18 analytes instead
of 16 with a p90 error of 222% instead of 14%.
"""
import pathlib
import re

import pytest

from data.bruker_loader import read_spectrometer_mhz

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _dataset(tmp_path, fields):
    """A dataset of one spectrum per field, laid out sample/1/pdata/1."""
    for i, sf in enumerate(fields):
        proc = tmp_path / f'S{i:02d}' / '1' / 'pdata' / '1'
        proc.mkdir(parents=True)
        (proc / '1r').write_bytes(b'\x00' * 16)
        lines = ['##TITLE= test', '##$SI= 4']
        if sf is not None:
            lines.append(f'##$SF= {sf}')
        (proc / 'procs').write_text('\n'.join(lines) + '\n',
                                    encoding='latin-1')
    return tmp_path


def test_the_field_is_read_from_the_datas_own_procs(tmp_path):
    assert read_spectrometer_mhz(
        _dataset(tmp_path, [500.13, 500.13, 500.13])) == pytest.approx(500.13)


def test_spectra_at_different_fields_are_an_error_not_an_average(tmp_path):
    """They cannot share one ppm-to-Hz conversion."""
    with pytest.raises(ValueError, match='campos diferentes'):
        read_spectrometer_mhz(_dataset(tmp_path, [500.13, 600.13]))


def test_a_rounding_difference_is_not_a_different_field(tmp_path):
    assert read_spectrometer_mhz(
        _dataset(tmp_path, [500.130, 500.132])) == pytest.approx(500.131)


def test_no_field_in_the_data_says_so(tmp_path):
    assert read_spectrometer_mhz(_dataset(tmp_path, [None, None])) is None


def test_the_project_saves_the_field_and_restores_it_to_the_same_variable():
    src = (ROOT / 'gui' / 'project_manager.py').read_text(encoding='utf-8')
    assert re.search(r"'spectrometer_mhz':\s*_safe_var\(ann, 'spectrometer_var'\)",
                     src), 'not saved'
    assert re.search(r"_set_var\(ann, 'spectrometer_var',\s*"
                     r"ann_state\.get\('spectrometer_mhz'\)\)", src), \
        'saved but not restored'


def test_the_application_hands_the_loaded_field_to_the_annotation():
    src = (ROOT / 'gui' / 'app.py').read_text(encoding='utf-8')
    assert 'read_spectrometer_mhz(' in src
    assert 'self.set_spectrometer_mhz(' in src


def test_the_synthetic_study_opens_at_the_field_it_was_built_at():
    from validation import mixture_bench as mb
    from validation.make_bench_project import build
    state, *_ = build()
    assert float(state['ann']['spectrometer_mhz']) == mb.MHZ
