# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The quantification table's tooltip: one short line per reason, the
deciding one first (gui.stages._quant_reasons)."""
from gui.stages._quant_reasons import MAX_LINES, short_reason, short_reasons

LOST = ('Not quantified: lost the competition — every peak this analyte '
        'claims was won by another with better evidence.')
JOINT = ('Measured by fitting its shape together with 220 other analyte(s) '
         'and 29 unassigned peak(s) whose signals overlap it, instead of '
         'integrating a window that holds theirs too. The integrated amount '
         'is kept on the row. 56% of the region is explained by nothing.')
SHARED = ('Shared window resolved by shape with Betaine aldehyde s, Taurine '
          't, Betaine s, Beta-gentiobiose s: this area is what this '
          "analyte's own shape accounts for.")


def test_the_deciding_reason_comes_first_and_short():
    lines = short_reasons([JOINT, LOST])
    assert lines == ['Lost its peaks to analytes with better evidence.',
                     'Joint fit with 220 overlapping analytes.']


def test_numbers_and_names_are_kept_and_long_lists_cut():
    assert short_reason('Lost peak #12 to L-glutamic acid on delta-J.') == \
        'Lost peak #12 to L-glutamic acid (delta-J).'
    assert short_reason(SHARED) == ('Shared window, split by shape with '
                                    'Betaine aldehyde s, Taurine t, Betaine s +1.')


def test_details_are_left_to_the_full_text():
    assert short_reason('The remainder was offered to L-fucose d, X d; none '
                        'met the three conditions.') is None
    assert short_reason('Pantolactone s was placed where annotation expects '
                        "it: detection had given it this signal's own line.") is None


def test_an_unknown_reason_is_cut_at_its_first_clause():
    assert short_reason('Something new: with a long explanation') == 'Something new'


def test_a_row_without_a_number_says_so_when_no_reason_does():
    lines = short_reasons([JOINT], not_quantified=True)
    assert lines[0] == 'Not quantified: no amount in any sample.'
    assert short_reasons([LOST], not_quantified=True)[0].startswith('Lost its')


def test_at_most_a_few_lines():
    many = [f'Lost peak #{i} to X{i} on delta-ppm.' for i in range(10)]
    lines = short_reasons(many)
    assert len(lines) == MAX_LINES and lines[-1] == '(+7 more)'


def test_a_refused_fit_names_the_signal_that_was_held():
    """Ethanol on the Mice study: its triplet in place, its quartet in the
    crowded 3.6 ppm region held at the edge -- 'could not place its shape
    on its peak' read as if the triplet were missing."""
    held = ('Not quantified: its signals overlap others, and the joint fit '
            'that measures overlapping analytes could not place its shape on '
            'its signal. The amount it gives is that of a shape beside the '
            'peak. Here 1 of 2 of its signals stood at the edge of how far '
            'the fit may move them (listed at 3.660 ppm): the shape is not '
            'where the peak is.')
    assert short_reason(held) == ('Joint fit held its signal at 3.660 ppm at '
                                  'the edge of its reach: mark it Good to '
                                  'quantify it anyway (one dot).')
    centred = held.split(' Here ')[0] + (' Here the fit centres on a signal '
                                         'at 3.330-3.360 ppm, not on the '
                                         'lines the election found.')
    assert short_reason(centred) == ('Joint fit centred on its signal at '
                                     '3.330-3.360 ppm, not the elected one.')
