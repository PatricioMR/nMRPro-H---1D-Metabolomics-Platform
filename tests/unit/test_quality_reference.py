"""The report quotes what each confidence level was measured to mean, at the
project's field or the nearest measured one, and says which."""
import pytest

from data.quality_reference import for_field, level_precision


def _tube(p3, p2, p1):
    lv = lambda p: {'real': 10, 'false': 1, 'precision': p,  # noqa: E731
                    'error_median': 0.1, 'error_p90': 0.3}
    return {'analytes': 18, 'runs': 3,
            'levels': {'3': lv(p3), '2': lv(p2), '1': lv(p1)},
            'found': 0.9, 'recall_2plus': 0.7, 'precision_2plus': 0.98,
            'false_per_run': 6.0, 'error_median': 0.07, 'error_p90': 0.2}


DOC = {'500': {'software': 'NMRPro-H v0.52', 'date': '2026-09-28',
               'database': 'BMRB_Metabolomics_Mammalian_v04.db',
               'design': {'field_mhz': 500.0, 'samples': 12, 'groups': 2},
               'tubes': {'18': _tube(1.0, 1.0, 0.4),
                         '100': _tube(1.0, 0.95, 0.3)}}}


def test_the_projects_own_field_is_exact():
    f, entry, exact = for_field(500, DOC)
    assert f == 500 and exact and entry is DOC['500']


def test_an_unmeasured_field_gets_the_nearest_and_says_so():
    f, _entry, exact = for_field(600, DOC)
    assert f == 500 and not exact


def test_nothing_measured_says_nothing():
    assert for_field(500, {}) == (None, None, False)


def test_a_level_is_quoted_at_its_lowest_measured_precision():
    _f, entry, _x = for_field(500, DOC)
    assert level_precision(entry, 2) == pytest.approx(0.95)
    assert level_precision(None, 2) is None


class TestReport:
    def test_the_field_note_is_honest_about_the_direction(self):
        from gui.widgets.pdf_validation import field_note
        assert 'the field this validation was measured at' in field_note(500, 500, True)
        assert 'conservative' in field_note(600, 500, False)
        assert 'not been measured' in field_note(600, 500, False)
        assert 'optimistic' in field_note(400, 500, False)

    def test_the_section_builds_and_names_its_field(self):
        from gui.widgets.pdf_report import _make_styles
        from gui.widgets.pdf_validation import validation_section
        from reportlab.platypus import Paragraph
        out = validation_section(600, _make_styles(), 500.0, doc=DOC)
        assert out
        title = out[0]
        assert isinstance(title, Paragraph) and '500 MHz' in title.text

    def test_no_measurement_no_section(self):
        from gui.widgets.pdf_report import _make_styles
        from gui.widgets.pdf_validation import validation_section
        assert validation_section(500, _make_styles(), 500.0, doc={}) == []

    def test_a_row_quotes_its_measured_precision_not_a_fraction_of_three(self):
        from gui.widgets.pdf_report import _quality_info
        label, _c = _quality_info({'metabolite': 'x', 'conc_mm': 1.0,
                                   '_confidence': 1}, None, {1: 0.3})
        assert '30%' in label and '33%' not in label
        bare, _c = _quality_info({'metabolite': 'x', 'conc_mm': 1.0,
                                  '_confidence': 3}, None, None)
        assert bare == 'CONF ●●●'
