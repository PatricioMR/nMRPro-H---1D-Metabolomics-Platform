# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The line the deconvolution review heads its list with."""


def test_the_reviews_line_says_what_it_lists():
    from processing.deconv_review import counts, counts_text, kind_of
    rows = [{'hmdb_id': 'a', '_by_cascade': True},
            {'hmdb_id': 'b', '_joint_fit': True},
            {'hmdb_id': 'c', '_deconv': {'window': (1.0, 1.1)}},
            {'hmdb_id': 'd'},                                  # plain integral
            {'hmdb_id': 'e', '_is_is': True, '_joint_fit': True},
            {'hmdb_id': 'f', '_not_quantified': True, '_rejection': 'fit_misplaced'}]
    c = counts([{'kind': k} for k in map(kind_of, rows) if k is not None])
    assert (c['shared'], c['joint'], c['cascade'], c['candidate'], c['rejected']) == (1, 1, 1, 1, 0)
    assert counts_text(c) == ('1 shared window(s) · 1 joint fit · 1 cascade · '
                              '1 with no number')
    c['rejected'] = 2
    assert counts_text(c).endswith(' · 2 rejected')
