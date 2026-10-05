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
    assert name('m*') == 'Complex multiplet'
    assert name('dd') == 'Doublet of doublets'
    assert name(None) == '—'
    assert name('xyz') == 'xyz'


def _resolved_row():
    from processing.quant_shared import component_from_evidence, resolve_window
    lw = 2.6 / FIELD
    tau = component_from_evidence(
        'TAU', {'multiplicity': 't',
                'predicted_ppms': [3.2 - 6.6 / FIELD, 3.2, 3.2 + 6.6 / FIELD]},
        label='taurine t', linewidth_ppm=lw, nh=2, owner=True)
    bet = component_from_evidence(
        'BET', {'multiplicity': 's', 'peak_ppm': 3.204}, label='betaine s',
        linewidth_ppm=lw, nh=9, per_proton_hint=1.0)
    y = 100 * tau.model(PPM) + 60 * bet.model(PPM)
    X = np.column_stack([y, y * 1.1])
    out = resolve_window([tau, bet], PPM, X)
    return X, {'metabolite': 'Taurine', 'hmdb_id': 'TAU', 'ppm': 3.2,
               '_integ_mode': 'Multiplet', '_multiplet_window': [3.18, 3.22],
               '_deconv': {'window': out['window'],
                           'remainder': out['remainder'],
                           'components': out['components'],
                           'amplitudes': out['amplitudes'], 'skipped': []}}


def test_a_resolved_window_names_its_method_and_partners():
    rep = _report()
    X, row = _resolved_row()
    label = rep._method_label(row)
    assert 'resolved by shape' in label and 'betaine s' in label
    assert 'remainder' in label
    assert rep._methods_used(row) == ['Standard', 'Resolved']
    text = ' '.join(getattr(f, 'text', '') for f in
                    rep._references_section([row], rep._make_styles()))
    assert 'Weljie' in text and 'Lawson' in text


def test_the_crop_draws_every_component_of_a_resolved_window():
    if not _real_matplotlib():
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    X, row = _resolved_row()
    fig, _r2 = _report()._peak_crop_fig(PPM, X, row, [], {})
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert 'taurine t' in labels and 'betaine s' in labels
    assert ax.collections, "the owner's component is filled"


def test_the_crop_leaves_the_gap_around_a_foreign_peak_unpainted():
    """A window cut around a foreign peak between two lines is painted on
    its pieces, the gap open, the foreign peak marked."""
    if not _real_matplotlib():
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    lw = 2.6 / FIELD
    g = lw / 2
    lines = [3.2 - 6.6 / FIELD, 3.2, 3.2 + 6.6 / FIELD]
    y = sum(a * g ** 2 / ((PPM - c) ** 2 + g ** 2)
            for c, a in zip(lines, (1, 2, 1)))
    X = np.column_stack([y, y])
    cut = (lines[0] + lines[1]) / 2
    row = {'metabolite': 'Taurine', 'hmdb_id': 'TAU', 'ppm': 3.2,
           '_integ_mode': 'Multiplet', '_multiplet_window': [3.18, 3.22],
           '_integ_segments': [[3.18, cut - lw / 2], [cut + lw / 2, 3.22]],
           '_foreign_inside': [cut], '_line_ppms': lines}
    fig, _r2 = _report()._peak_crop_fig(PPM, X, row, [], {})
    ax = fig.axes[0]
    assert ax.collections, 'the pieces are filled'
    verts = np.vstack([p.vertices for p in ax.collections[0].get_paths()])
    inside_gap = (verts[:, 0] > cut - lw / 2 + 1e-6) & \
                 (verts[:, 0] < cut + lw / 2 - 1e-6) & (verts[:, 1] > 0)
    assert not inside_gap.any(), 'nothing painted in the gap'
    marks = {round(float(ln.get_xdata()[0]), 6): ln.get_color()
             for ln in ax.get_lines() if len(ln.get_xdata()) == 2}
    assert marks.get(round(cut, 6)) == '#b0b0b0', 'the foreign peak, grey'
    for c in lines:
        assert marks.get(round(c, 6)) == '#e63946', 'each own line, red'


def test_the_crop_draws_one_mean_per_group_and_fills_the_best():
    """The traces are the groups' means of what was integrated -- blanks
    out, no global mean -- and the fill sits on the group whose mean stands
    highest in the window, the trace the annotation window shows."""
    if not _real_matplotlib():
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    lw = 2.6 / FIELD
    g = lw / 2
    y = g ** 2 / ((PPM - 3.2) ** 2 + g ** 2)
    X = np.column_stack([y, y * 3.0, np.zeros_like(y)])    # A, B, blank
    si = [{'label': 'A', 'type': 'Sample'}, {'label': 'B', 'type': 'Sample'},
          {'label': 'BLK', 'type': 'BLK'}]
    row = {'metabolite': 'X', 'hmdb_id': 'X', 'ppm': 3.2,
           '_integ_mode': 'Multiplet', '_multiplet_window': [3.18, 3.22]}
    fig, _r2 = _report()._peak_crop_fig(PPM, X, row, si, {'A': '#111111',
                                                          'B': '#222222'})
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert 'A' in labels and 'B' in labels and 'BLK' not in labels
    top = max(float(np.max(ln.get_ydata())) for ln in ax.get_lines()
              if ln.get_label() == 'B')
    filled = np.vstack([p.vertices for p in ax.collections[0].get_paths()])
    assert float(filled[:, 1].max()) == pytest.approx(top, rel=0.02), \
        'the fill follows B, three times A'


def _polygon_area(collection):
    """Shoelace over every filled path, so the area painted can be held
    against the area reported."""
    total = 0.0
    for path in collection.get_paths():
        v = path.vertices
        x, y = v[:, 0], v[:, 1]
        total += 0.5 * abs(float(np.dot(x, np.roll(y, 1))
                                 - np.dot(y, np.roll(x, 1))))
    return total


def test_the_shaded_area_is_the_area_the_number_came_from():
    """The report's figure is evidence: the region painted on the group's
    mean trace must integrate to that group's reported area, before the
    capture correction. Same window, same baseline, same trace."""
    if not _real_matplotlib():
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    from gui.stages._quant_adapter import quantify_project
    ppm = np.linspace(1.40, 1.60, 6000)
    g = 0.0015
    y = sum(g ** 2 / ((ppm - c) ** 2 + g ** 2) for c in (1.470, 1.482))
    X = np.column_stack([y, y * 1.2, y * 3.0, y * 3.3])
    si = [{'label': 'A', 'type': 'Sample'}, {'label': 'A', 'type': 'Sample'},
          {'label': 'B', 'type': 'Sample'}, {'label': 'B', 'type': 'Sample'}]
    peaks = [{'rank': 1, 'ppm': 1.470, 'start_ppm': 1.465, 'end_ppm': 1.476,
              'snr': 50.0},
             {'rank': 2, 'ppm': 1.482, 'start_ppm': 1.476, 'end_ppm': 1.487,
              'snr': 50.0}]
    mp = {'db_ppm': 1.476, 'ppm': 1.476, 'peak_rank': 1, 'multiplicity': 'd',
          'type': 'd', 'delta_j_hz': 0.05, 'ppm_diff': 0.0, '_n_found_lines': 2,
          '_n_exp_lines': 2, 'protons': 3, 'annotation_quality': 'good',
          'predicted_lines': [{'ppm': 1.470}, {'ppm': 1.482}]}
    rows = quantify_project(
        metabolites=[{'id': 'D', 'name': 'Doublet', 'matched_peaks': [mp]}],
        peaks=peaks, ppm=ppm, intensities=X, samples_info=si,
        noise_sigma=0.001, use_pqn=False,
        deconvolve=False)   # as the app runs (app_paths.DECONVOLUTION_ENABLED)
    row = next(r for r in rows if r.get('hmdb_id') == 'D')
    assert row['_not_quantified'] is False
    grp = row['_snr_group']
    fig, _r2 = _report()._peak_crop_fig(ppm, X, row, si, {})
    ax = fig.axes[0]
    painted = _polygon_area(ax.collections[0])
    # the group's reported area, with the capture correction undone: what
    # the integration actually summed on that group's samples
    idx = [i for i, s_ in enumerate(si) if s_['label'] == grp]
    reported = np.mean([row['_areas_per_sample'][i] for i in idx])
    reported *= float(row['_captured_fraction'] or 1.0)
    assert painted == pytest.approx(reported, rel=0.02)


def _fitted_row(mode):
    x = np.linspace(3.23, 3.17, 200)
    unit = 1.0 / (1.0 + ((x - 3.2) / 0.002) ** 2)
    row = {'metabolite': 'F', 'hmdb_id': 'F', 'ppm': 3.2,
           '_integ_mode': mode, '_multiplet_window': [3.18, 3.22],
           '_local_baseline': True,
           '_fit_curve': {'ppm': x.tolist(),
                          'unit_by_group': {'All': unit.tolist()},
                          'amp_by_group': {'All': 50.0},
                          'n_by_group': {'All': [2, 2]},
                          'focus': [3.2, 3.2], 'focus_matches_window': True,
                          'at_limit': [], 'n_signals': 1,
                          'baseline_by_group': {'All': [5.0] * x.size},
                          'label': 'Cascade fit'}}
    if mode == 'Cascade':
        row['_by_cascade'] = True
    else:
        row['_joint_fit'] = {'group': ['F'], 'free': 0}
    return row


def test_a_fitted_amount_is_not_called_an_integration():
    """The report says where the NUMBER came from: an amount a fit
    replaced did not come from integrating a window."""
    rep = _report()
    assert rep._methods_used(_fitted_row('Cascade')) == ['Cascade']
    assert rep._methods_used(_fitted_row('Multiplet')) == ['JointFit']
    assert 'local baseline' in rep._method_label(_fitted_row('Cascade'))
    assert 'local baseline' not in rep._method_label(
        {'_integ_mode': 'Multiplet'})


def test_the_crop_draws_the_fit_on_the_baseline_it_stood_on():
    if not _real_matplotlib():
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    row = _fitted_row('Cascade')
    x = np.asarray(row['_fit_curve']['ppm'])
    y = 5.0 + 50.0 * np.asarray(row['_fit_curve']['unit_by_group']['All'])
    ppm = np.linspace(3.4, 3.0, 4000)
    Y = np.interp(ppm, x[::-1], y[::-1], left=5.0, right=5.0)
    si = [{'index': i, 'name': f'S{i}', 'label': 'All'} for i in range(2)]
    fig, _r2 = _report()._peak_crop_fig(ppm, np.column_stack([Y, Y]), row,
                                        si, {})
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert 'Cascade fit' in labels and 'baseline' in labels
    base = next(ln for ln in ax.get_lines() if ln.get_label() == 'baseline')
    assert np.allclose(base.get_ydata(), 5.0)


def _resolved_two_groups():
    """A resolved window on four samples in two groups, the second three
    times the first: the amplitude the report draws has to be its own
    group's, as the table's area is."""
    from processing.quant_shared import component_from_evidence, resolve_window
    lw = 2.6 / FIELD
    tau = component_from_evidence(
        'TAU', {'multiplicity': 't',
                'predicted_ppms': [3.2 - 6.6 / FIELD, 3.2, 3.2 + 6.6 / FIELD]},
        label='taurine t', linewidth_ppm=lw, nh=2, owner=True,
        shift_ppm=0.0006, width_scale=1.3)
    bet = component_from_evidence(
        'BET', {'multiplicity': 's', 'peak_ppm': 3.204}, label='betaine s',
        linewidth_ppm=lw, nh=9, per_proton_hint=1.0)
    y = 100 * tau.curve(PPM) + 60 * bet.curve(PPM)
    X = np.column_stack([y, y * 1.1, y * 3.0, y * 3.3])
    si = [{'index': i, 'label': 'A' if i < 2 else 'B', 'type': 'Sample'}
          for i in range(4)]
    out = resolve_window([tau, bet], PPM, X)
    row = {'metabolite': 'Taurine', 'hmdb_id': 'TAU', 'ppm': 3.2,
           '_integ_mode': 'Multiplet', '_multiplet_window': [3.18, 3.22],
           '_areas_per_sample': [out['areas']['TAU'][i] for i in range(4)],
           '_deconv': {'window': out['window'], 'remainder': out['remainder'],
                       'components': out['components'],
                       'amplitudes': out['amplitudes'], 'skipped': []}}
    return X, si, row


def test_the_resolved_component_is_the_curve_the_fit_used_on_its_groups_amplitude():
    """Painted area == the group's reported area. Drawn from the reference's
    model before the fit moved and widened it, and at the median amplitude of
    every sample, the figure was 20-50% off the number it stood beside."""
    if not _real_matplotlib():
        pytest.skip('matplotlib is stubbed in this session')
    import matplotlib
    matplotlib.use('Agg')
    X, si, row = _resolved_two_groups()
    fig, _r2 = _report()._peak_crop_fig(PPM, X, row, si, {})
    painted = _polygon_area(fig.axes[0].collections[0])
    reported = np.mean([row['_areas_per_sample'][i] for i in (2, 3)])
    assert painted == pytest.approx(reported, rel=0.03)


def test_the_group_amplitude_is_the_mean_over_its_positive_samples():
    group_amplitude = _report()._group_amplitude
    si = [{'label': 'A', 'type': 'Sample'}, {'label': 'A', 'type': 'Sample'},
          {'label': 'B', 'type': 'Sample'}, {'label': 'B', 'type': 'Sample'}]
    amps = {0: 10.0, 1: 0.0, 2: 30.0, 3: 50.0}
    assert group_amplitude(amps, si, 'A') == 10.0
    assert group_amplitude(amps, si, 'B') == 40.0
    assert group_amplitude(amps, si, None) == pytest.approx(20.0)


def _pqn_stage(factors, norm='IS+PQN'):
    import types

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    qs = types.SimpleNamespace(
        norm_var=_V(norm), integ_var=_V('Multiplet'), proton_corr_var=_V(True),
        ref_conc_var=_V(0.5), ref_nh_var=_V(9), ref_peak={'ppm': 0.0},
        _quant_results=[{'_pqn_factors': factors}])
    app = types.SimpleNamespace(samples_info=[{'label': 'A'}] * 3
                                + [{'label': 'B'}] * 3)
    return qs, app


def _cells(tbl):
    return [[getattr(c, 'text', c) for c in row] for row in tbl._cellvalues]


def test_the_report_says_the_pqn_factor_by_group_and_flags_a_gap():
    qs, app = _pqn_stage({0: 1.0, 1: 1.0, 2: 1.0, 3: 0.6, 4: 0.6, 5: 0.6})
    cells = _cells(_report()._params_table(qs, app))
    by = next(r for r in cells if r[0] == 'PQN factor by group')
    assert 'A: 1.00' in by[1] and 'B: 0.60' in by[1]
    note = next(r for r in cells if r[0] == 'PQN notice')
    assert 'taking a difference between groups' in note[1]


def test_the_report_gives_the_factors_without_a_notice_when_the_groups_agree():
    qs, app = _pqn_stage({i: 1.0 for i in range(6)})
    cells = _cells(_report()._params_table(qs, app))
    assert any(r[0] == 'PQN factor by group' for r in cells)
    assert not any(r[0] == 'PQN notice' for r in cells)


def test_without_pqn_the_report_adds_nothing():
    qs, app = _pqn_stage({0: 1.0, 3: 0.5}, norm='IS')
    cells = _cells(_report()._params_table(qs, app))
    assert not any(str(r[0]).startswith('PQN') for r in cells)
