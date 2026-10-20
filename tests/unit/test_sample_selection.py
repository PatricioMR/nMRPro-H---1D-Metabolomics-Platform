# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Which samples a processing step may read.

The rule lived inside detect_peaks, so detection honoured it and nothing else
did: annotation averaged every column, blanks included, and the MCR fitted its
components across them too.
"""
import pytest

from processing.sample_selection import included_sample_indices, is_included


def _s(**kw):
    d = {'type': 'Sample'}
    d.update(kw)
    return d


class TestIsIncluded:
    def test_a_plain_sample_is_in(self):
        assert is_included(_s())

    def test_a_blank_is_out(self):
        assert not is_included(_s(type='BLK'))

    def test_blank_is_matched_case_insensitively(self):
        assert not is_included(_s(type='blk'))

    def test_an_excluded_sample_is_out(self):
        assert not is_included(_s(excluded=True))

    def test_missing_type_defaults_to_sample(self):
        assert is_included({'name': 'S1'})

    def test_none_is_a_sample(self):
        """A row with no metadata is a sample until something says otherwise."""
        assert is_included(None)


class TestIndices:
    def test_drops_blanks_and_excluded(self):
        info = [_s(type='BLK'), _s(), _s(excluded=True), _s()]
        assert included_sample_indices(info) == [1, 3]

    def test_counts_from_n_samples_not_from_the_metadata(self):
        """The matrix has the columns; samples_info may be shorter."""
        assert included_sample_indices([_s()], n_samples=3) == [0, 1, 2]

    def test_metadata_longer_than_the_matrix_is_clipped(self):
        info = [_s(), _s(), _s()]
        assert included_sample_indices(info, n_samples=2) == [0, 1]

    def test_all_blanks_falls_back_to_everything(self):
        """A study of nothing but blanks is a labelling mistake; refusing to
        process it would hide that behind an empty result."""
        info = [_s(type='BLK'), _s(type='BLK')]
        assert included_sample_indices(info) == [0, 1]

    def test_no_metadata_at_all(self):
        assert included_sample_indices(None, n_samples=3) == [0, 1, 2]
        assert included_sample_indices([], n_samples=0) == []


def test_matches_what_detection_used_to_do_inline():
    """The literal rule detect_peaks carried before it was extracted."""
    info = [_s(type='BLK'), _s(), _s(excluded=True), _s(type='Sample')]
    expected = [i for i, x in enumerate(info)
                if str(x.get('type', 'Sample')).upper() != 'BLK'
                and not x.get('excluded', False)]
    assert included_sample_indices(info, len(info)) == expected
