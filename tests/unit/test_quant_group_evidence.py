# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The elected signal on each group's mean: its SNR and its shape agreement,
and which group the figure and the numbers then share."""
import numpy as np
import pytest

from processing.quant_snr import best_group, group_evidence

PPM = np.linspace(1.40, 1.60, 8000)
LW = 0.003
G = LW / 2


def _lorentz(c, a):
    return a * G ** 2 / ((PPM - c) ** 2 + G ** 2)


def _doublet(a):
    return _lorentz(1.470, a) + _lorentz(1.482, a)


class TestGroupEvidence:
    def test_one_entry_per_group_read_on_its_mean(self):
        X = np.column_stack([_doublet(1), _doublet(1), _doublet(3), _doublet(3)])
        ev = group_evidence(PPM, X, {'A': [0, 1], 'B': [2, 3]},
                            lo=1.46, hi=1.492, noise=0.01,
                            lines=[1.470, 1.482], weights=[1, 1], hwhm=G)
        assert set(ev) == {'A', 'B'}
        assert ev['B']['snr'] == pytest.approx(3 * ev['A']['snr'], rel=0.02)
        assert ev['A']['cosine'] == pytest.approx(ev['B']['cosine'], abs=1e-6)
        assert ev['A']['cosine'] > 0.95

    def test_the_cosine_is_the_signals_not_the_anchors(self):
        """A doublet judged as a doublet, not as its higher line: a trace
        holding one line where two are predicted agrees less."""
        two = np.column_stack([_doublet(1)])
        one = np.column_stack([_lorentz(1.470, 1)])
        kw = dict(lo=1.46, hi=1.492, noise=0.01, lines=[1.470, 1.482],
                  weights=[1, 1], hwhm=G)
        assert group_evidence(PPM, two, {'A': [0]}, **kw)['A']['cosine'] > \
            group_evidence(PPM, one, {'A': [0]}, **kw)['A']['cosine']

    def test_a_group_with_no_sample_is_absent(self):
        X = np.column_stack([_doublet(1)])
        ev = group_evidence(PPM, X, {'A': [0], 'B': [7]}, lo=1.46, hi=1.492,
                            noise=0.01)
        assert set(ev) == {'A'}

    def test_without_lines_there_is_an_snr_and_no_cosine(self):
        X = np.column_stack([_doublet(1)])
        ev = group_evidence(PPM, X, {'A': [0]}, lo=1.46, hi=1.492, noise=0.01)
        assert np.isfinite(ev['A']['snr']) and ev['A']['cosine'] is None


class TestBestGroup:
    def test_the_best_cosine_wins_and_snr_breaks_the_tie(self):
        ev = {'A': {'snr': 50.0, 'cosine': 0.80},
              'B': {'snr': 20.0, 'cosine': 0.95},
              'C': {'snr': 30.0, 'cosine': 0.93}}
        # B and C are within the annotation's tie band; C is not louder
        assert best_group(ev) == 'C'
        assert best_group(ev, cosine_tie=0.01) == 'B'

    def test_without_cosines_the_loudest_wins(self):
        ev = {'A': {'snr': 5.0, 'cosine': None}, 'B': {'snr': 9.0, 'cosine': None}}
        assert best_group(ev) == 'B'

    def test_nothing_measured_is_nobody(self):
        assert best_group({}) is None
        assert best_group({'A': {'snr': float('nan'), 'cosine': 0.9}}) is None

