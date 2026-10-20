# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The Bruker import puts each spectrum's reference standard at 0 ppm.

TopSpin referenced the Mice ISS spectra (600 MHz) between -0.088 and +0.078
ppm; detection's rigid alignment reaches 0.03 ppm, so without referencing at
import a creatine singlet stayed spread over 69 mppm across the study.
"""
import os

import numpy as np
import pytest

from data.bruker_loader import load_bruker_dataset, reference_summary
from processing.peak_alignment import reference_offset

SF = 600.0


def _lorentz(x, centre, fwhm_ppm, height):
    hw = fwhm_ppm / 2.0
    return height * hw ** 2 / ((x - centre) ** 2 + hw ** 2)


def _spectrum(tsp_at, si=16384, offset=16.7, sw_ppm=24.0, seed=0, tsp=True):
    x = np.linspace(offset, offset - sw_ppm, si)
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1e3, si)
    y += _lorentz(x, 3.03 + tsp_at, 3.0 / SF, 2e6)        # creatine-like
    if tsp:
        y += _lorentz(x, tsp_at, 2.8 / SF, 1e7)
    return x, y


class TestReferenceOffset:
    def test_finds_the_standard_between_points(self):
        x, y = _spectrum(0.0583)
        off, apex, snr = reference_offset(x, y)
        step = abs(x[1] - x[0])
        assert off == pytest.approx(0.0583, abs=step / 4)
        assert snr > 1000

    def test_a_spectrum_without_its_standard_is_not_moved(self):
        x, y = _spectrum(0.0, tsp=False)
        off, _apex, snr = reference_offset(x, y)
        assert off is None
        assert snr < 6

    def test_a_slope_from_outside_the_window_is_not_a_standard(self):
        x, y = _spectrum(0.0, tsp=False)
        y = y + _lorentz(x, 0.35, 5.0 / SF, 1e8)   # a tall signal past the edge
        off, _apex, _snr = reference_offset(x, y)
        assert off is None


def _write_bruker(root, name, x, y, expno=2):
    d = os.path.join(root, name, str(expno), 'pdata', '1')
    os.makedirs(d)
    sw_hz = (x[0] - x[-1]) * SF
    with open(os.path.join(d, 'procs'), 'w') as f:
        f.write(f'##$SI= {x.size}\n##$SW_p= {sw_hz}\n##$OFFSET= {x[0]}\n'
                f'##$SF= {SF}\n##$BYTORDP= 0\n##$NC_proc= 0\n')
    y.astype('<i4').tofile(os.path.join(d, '1r'))


def test_import_puts_every_standard_at_zero_and_says_what_it_did(tmp_path):
    root = str(tmp_path)
    offsets = {'S_A': 0.0583, 'S_B': -0.0880, 'S_C': 0.0}
    for i, (n, o) in enumerate(offsets.items()):
        _write_bruker(root, n, *_spectrum(o, seed=i, tsp=(n != 'S_C')))
    report = {}
    ppm, X, names, _labels = load_bruker_dataset(
        root, ppm_min=-0.5, ppm_max=10.0, n_points=32768, expno=2,
        reference_ppm=0.0, reference_report=report)
    step = abs(ppm[1] - ppm[0])
    near0 = np.abs(ppm) < 0.15
    for j, n in enumerate(names):
        if n == 'S_C':
            assert report[n]['offset_ppm'] is None
            continue
        assert report[n]['offset_ppm'] == pytest.approx(offsets[n], abs=0.0005)
        assert abs(ppm[near0][np.argmax(X[near0, j])]) <= 2 * step
        # every signal moved with it, not only the standard
        m = (ppm > 2.9) & (ppm < 3.2)
        assert ppm[m][np.argmax(X[m, j])] == pytest.approx(3.03, abs=0.001)
    text, missing = reference_summary(report)
    assert missing == ['S_C']
    assert '2 spectra' in text and 'without a standard' in text


def test_import_without_referencing_leaves_topspin_axis(tmp_path):
    root = str(tmp_path)
    _write_bruker(root, 'S_A', *_spectrum(0.0583))
    ppm, X, _n, _l = load_bruker_dataset(root, ppm_min=-0.5, ppm_max=10.0,
                                         n_points=32768, expno=2)
    near0 = np.abs(ppm) < 0.15
    assert ppm[near0][np.argmax(X[near0, 0])] == pytest.approx(0.0583, abs=0.001)
