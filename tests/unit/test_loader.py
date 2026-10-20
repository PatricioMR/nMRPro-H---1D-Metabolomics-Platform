# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_loader.py
"""Tests for data/loader.py using the real plasma CSV."""
import numpy as np
import pytest
from data.loader import load_all_samples, get_sample_info, detect_separator


class TestDetectSeparator:
    def test_comma(self, tmp_path):
        f = tmp_path / 'test.csv'
        f.write_text('Sample,S1,S2\nLabel,A,B\n1.0,2.0,3.0\n')
        assert detect_separator(str(f)) == ','

    def test_tab(self, tmp_path):
        f = tmp_path / 'test.tsv'
        f.write_text('Sample\tS1\tS2\nLabel\tA\tB\n1.0\t2.0\t3.0\n')
        assert detect_separator(str(f)) == '\t'


class TestLoadAllSamples:
    def test_loads_real_csv(self, loaded_spectrum):
        d = loaded_spectrum
        assert d['ppm'] is not None
        assert d['intensities'] is not None
        assert len(d['ppm']) > 100, 'Expected at least 100 PPM points'
        assert d['intensities'].shape[1] == 12, 'Expected 12 samples'

    def test_ppm_range(self, loaded_spectrum):
        ppm = loaded_spectrum['ppm']
        # NMR spectra can include noise/baseline outside the typical 0-10 ppm window
        assert ppm.max() <= 25.0
        assert ppm.min() >= -15.0

    def test_intensities_shape_matches_ppm(self, loaded_spectrum):
        ppm   = loaded_spectrum['ppm']
        inten = loaded_spectrum['intensities']
        assert inten.shape[0] == len(ppm)

    def test_sample_labels(self, loaded_spectrum):
        labels = loaded_spectrum['labels']
        assert 'BLK' in labels, 'Expected BLK samples'
        assert '300uL' in labels or '200uL' in labels

    def test_no_nan_in_intensities(self, loaded_spectrum):
        assert not np.isnan(loaded_spectrum['intensities']).any()

    def test_raises_on_short_file(self, tmp_path):
        f = tmp_path / 'short.csv'
        f.write_text('Sample,S1\n')
        with pytest.raises(ValueError, match='too short'):
            load_all_samples(str(f))


class TestGetSampleInfo:
    def test_returns_list(self, plasma_csv_path):
        info = get_sample_info(plasma_csv_path)
        assert isinstance(info, list)
        assert len(info) == 12

    def test_has_required_keys(self, plasma_csv_path):
        info = get_sample_info(plasma_csv_path)
        for s in info:
            assert 'index' in s
            assert 'name' in s
            assert 'label' in s

    def test_blk_samples_present(self, plasma_csv_path):
        info = get_sample_info(plasma_csv_path)
        blk = [s for s in info if s['label'] == 'BLK']
        assert len(blk) == 3, 'Expected 3 BLK samples'
