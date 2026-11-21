# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""What the Quantification panel says the deconvolution review holds."""


def test_row_counts_and_its_line_say_what_the_review_will_list():
    from processing.deconv_review import counts_text, row_counts
    rows = [{'hmdb_id': 'a', '_by_cascade': True},
            {'hmdb_id': 'b', '_joint_fit': True},
            {'hmdb_id': 'c', '_deconv': {'window': (1.0, 1.1)}},
            {'hmdb_id': 'd'},                                  # plain integral
            {'hmdb_id': 'e', '_is_is': True, '_joint_fit': True},
            {'hmdb_id': 'f', '_not_quantified': True, '_rejection': 'fit_misplaced'}]
    c = row_counts(rows)
    assert (c['shared'], c['joint'], c['cascade'], c['candidate'], c['rejected']) == (1, 1, 1, 1, 0)
    assert counts_text(c) == ('1 shared window(s) · 1 joint fit · 1 cascade · '
                              '1 with no number')
    c['rejected'] = 2
    assert counts_text(c).endswith(' · 2 rejected')
