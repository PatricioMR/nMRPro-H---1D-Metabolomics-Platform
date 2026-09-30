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

    def test_the_report_names_the_generator_the_validation_used(self):
        """Measured on GISSMO spectra, the page must not say the mixtures
        were built from the database's own line shapes."""
        from gui.widgets.pdf_validation import spectra_note
        g = spectra_note({'generator': 'gissmo'})
        assert 'GISSMO' in g and 'Dashti' in g and 'own measured line' not in g
        assert 'own measured line shapes' in spectra_note({})
        assert 'own measured line shapes' in spectra_note(
            {'generator': 'profiles'})

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


class TestCrowding:
    """A real sample's number of compounds is unknown; its peaks are not.
    The report quotes the mixture as crowded as the project."""

    ENTRY = {'tubes': {'18': {**_tube(1.0, 0.65, 0.5), 'peaks': 90.0},
                       '50': {**_tube(1.0, 0.90, 0.4), 'peaks': 200.0},
                       '100': {**_tube(1.0, 0.96, 0.3), 'peaks': 300.0}}}

    def test_the_nearest_mixture_by_peaks_is_quoted(self):
        from data.quality_reference import tube_for_peaks
        assert tube_for_peaks(self.ENTRY, 294) == ('100', False)
        assert tube_for_peaks(self.ENTRY, 180) == ('50', False)
        assert tube_for_peaks(self.ENTRY, 60) == ('18', False)

    def test_more_crowded_than_anything_measured_says_so(self):
        from data.quality_reference import tube_for_peaks
        assert tube_for_peaks(self.ENTRY, 450) == ('100', True)

    def test_without_a_count_the_most_crowded_mixture(self):
        from data.quality_reference import tube_for_peaks
        assert tube_for_peaks(self.ENTRY, None) == ('100', False)
        old = {'tubes': {'18': _tube(1, 1, 1), '100': _tube(1, 1, 1)}}
        assert tube_for_peaks(old, 294) == ('100', False)
        assert tube_for_peaks({}, 294) == (None, False)

    def test_the_level_quotes_its_mixture_not_the_worst(self):
        assert level_precision(self.ENTRY, 2, '100') == pytest.approx(0.96)
        assert level_precision(self.ENTRY, 2) == pytest.approx(0.65)

    def test_the_report_says_which_mixture_and_why(self):
        from gui.widgets.pdf_validation import crowding_note, precision_footnote
        n = crowding_note(self.ENTRY, 294)
        assert '294 detected peaks' in n and 'mixture of 100' in n
        assert 'more crowded' in crowding_note(self.ENTRY, 450)
        f = precision_footnote(500, self.ENTRY, '100', False, 294)
        assert '100 compounds' in f and '294 peaks' in f
        assert 'may be lower' in precision_footnote(500, self.ENTRY, '100',
                                                    True, 450)
