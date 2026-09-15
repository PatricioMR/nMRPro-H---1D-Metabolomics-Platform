"""The quantification report draws a shape-quantified analyte as the copy of
the shape that was integrated, names the method with its reference, and
lists the references of the methods it used."""
import sys

import numpy as np
import pytest

from processing.pattern_shape import encode_profile, lines_model


def _report():
    # imported inside the tests: the unit suite's conftest stubs matplotlib
    # at collection, and a module-level import here would run before it
    from gui.widgets import pdf_report
    return pdf_report


def _real_matplotlib() -> bool:
    mpl = sys.modules.get('matplotlib')
    return mpl is not None and hasattr(mpl, 'ticker') and hasattr(
        mpl.ticker, 'FormatStrFormatter') and callable(
        getattr(mpl, 'get_backend', None))

FIELD = 600.0
PPM = np.linspace(3.4, 3.0, 4000)
LINES = [(-9.0, 0.7), (-1.0, 1.0), (0.5, 0.9), (8.0, 0.8)]


def _shape_row(explained=0.95):
    offs = np.arange(-20.0, 20.0, 0.1)
    prof = encode_profile(offs, lines_model(offs / FIELD, LINES, 0.0, 1.2, FIELD))
    return {'metabolite': 'X', 'hmdb_id': 'X', 'ppm': 3.2,
            '_integ_mode': 'Shape', '_multiplet_window': [3.18, 3.22],
            '_shape_explained': explained, '_noise_sigma': 1.0,
            '_shape_fit': {'origin_ppm': 3.2, 'field_mhz': FIELD,
                           'study_linewidth_hz': 2.6,
                           'pattern_profile': prof, 'profile_fwhm_hz': 1.2,
                           'pattern_lines': LINES}}


def test_the_method_row_names_the_shape_fit_and_its_reference():
    _method_label = _report()._method_label
    label = _method_label(_shape_row(0.95))
    assert 'Shape fit' in label and 'Weljie' in label and '95%' in label
    assert 'upper bound' not in label
    assert 'upper bound' in _method_label(_shape_row(0.5))
    assert _method_label({'_integ_mode': 'Multiplet'}) == 'Peak integration'


def test_the_references_section_lists_only_what_was_used():
    rep = _report()
    _references_section, _methods_used = rep._references_section, rep._methods_used
    METHOD_REFERENCES = rep.METHOD_REFERENCES
    styles = rep._make_styles()
    rows = [_shape_row(), {'_integ_mode': 'Multiplet'}]
    flow = _references_section(rows, styles)
    text = ' '.join(getattr(f, 'text', '') for f in flow)
    assert 'Weljie' in text and 'Anal. Chem. 2006' in text
    assert 'Tauler' not in text
    assert _methods_used(_shape_row()) == ['Shape']
    assert set(METHOD_REFERENCES) >= {'Standard', 'Shape', 'MCR', 'JG-Deconv'}


def test_the_crop_draws_the_fitted_copy():
    if not _real_matplotlib():
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    _peak_crop_fig = _report()._peak_crop_fig
    y = 300.0 * lines_model(PPM, LINES, 3.2, 2.6, FIELD)
    X = np.column_stack([y + np.random.default_rng(0).normal(0, 1.0, PPM.size)
                         for _ in range(3)])
    fig, _r2 = _peak_crop_fig(PPM, X, _shape_row(), [], {})
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert any(lb.startswith('Shape fit') for lb in labels)
    assert ax.collections, 'the fitted copy is filled'


def test_the_integration_row_names_method_multiplicity_and_protons():
    name = _report()._multiplicity_name
    assert name('d') == 'Doublet'
    assert name('m*') == 'Complex multiplet (measured shape)'
    assert name('dd') == 'Doublet of doublets'
    assert name(None) == '—'
    assert name('xyz') == 'xyz'
