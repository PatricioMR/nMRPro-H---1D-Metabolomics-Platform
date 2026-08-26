"""A peak resolved by MCR must be shown as the component MCR resolved.

When MCR separates an overlap, the candidate's score stops being measured on the
raw signal — `_compute_scores` takes ΔJ from the isolated component instead. If
the panel kept drawing only the merged trace, the analyst would be judging
something the score never looked at.

The display gate is the same one the scoring uses to trust that component
(MCR_MIN_OVERLAP_K / MCR_MIN_SHAPE_R), deliberately not the region-level
`mcr_confirmed` verdict.
"""
import numpy as np
import pytest

from gui.utils.mcr_utils import MCR_MIN_OVERLAP_K, MCR_MIN_SHAPE_R
from gui.widgets.annotation_results_window import AnnotationResultsWindow as ARW


def _window(peaks):
    w = ARW.__new__(ARW)
    w.peaks = peaks
    return w


def _mcr_peak(rank=3, n_pts=64, k=2, shape_r=(0.95, 0.88), c_mean=(2.0, 5.0),
              cand_keys=(('HMDB0000161', 1.47), ('HMDB0000190', 1.48)),
              with_c=True):
    """A detected peak carrying a stored MCR result with k components.

    assignments[i] describes component i; entry 0 is the candidate index and
    entry 1 the Pearson r of that component against the candidate's profile.
    """
    ppm = np.linspace(1.40, 1.55, n_pts)
    S = np.zeros((n_pts, k))
    for c in range(k):
        S[:, c] = np.exp(-0.5 * ((ppm - (1.45 + 0.02 * c)) / 0.004) ** 2)
        S[:, c] /= np.linalg.norm(S[:, c])        # MCR returns unit columns
    cands = [{'hmdb_id': hid, 'db_ppm': dp} for hid, dp in cand_keys]
    # component 0 → candidate 1, component 1 → candidate 0
    order = [1, 0][:k] if k >= 2 else [0]
    assignments = [(order[i], shape_r[i], 0.0, 1.0, 7.2, 7.2, 0.9)
                   for i in range(k)]
    ann = {'S': S, 'ppm_region': ppm, 'assignments': assignments,
           'candidates': cands, 'K': k}
    if with_c:
        # (n_samples, K)
        ann['C'] = np.vstack([np.array(c_mean[:k]) for _ in range(4)])
    return {'rank': rank, 'ppm': 1.47, 'mcr_annotation': ann}


def _mp(db_ppm=1.47, rank=3, **extra):
    """A matched_peak exactly as _group_by_metabolite builds it.

    Note what is NOT here: hmdb_id, bmrb_id, name. The compound identity lives
    on the metabolite, and passing a peak that carries it would test a shape the
    application never produces.
    """
    d = {'peak_rank': rank, 'db_ppm': db_ppm}
    d.update(extra)
    return d


def _met(mid='HMDB0000161', name='Taurine'):
    return {'id': mid, 'name': name}


# ── candidate → component mapping ─────────────────────────────────────────────

def test_returns_the_component_assigned_to_this_candidate():
    pk = _mcr_peak()
    w = _window([pk])
    got = w._mcr_component_for(_mp(), _met())
    assert got is not None
    ppm, profile = got
    assert len(ppm) == len(profile)
    # candidate 0 is described by assignments[1] → component 1
    expected = pk['mcr_annotation']['S'][:, 1] * 5.0     # C mean of component 1
    assert np.allclose(profile, expected)


def test_other_candidate_gets_its_own_component():
    pk = _mcr_peak()
    w = _window([pk])
    _, profile = w._mcr_component_for(_mp(1.48), _met('HMDB0000190'))
    assert np.allclose(profile, pk['mcr_annotation']['S'][:, 0] * 2.0)


def test_matching_uses_bmrb_id_when_there_is_no_hmdb_id():
    pk = _mcr_peak()
    pk['mcr_annotation']['candidates'] = [{'bmrb_id': 'bmse000028', 'db_ppm': 1.47},
                                          {'bmrb_id': 'bmse000282', 'db_ppm': 1.48}]
    w = _window([pk])
    assert w._mcr_component_for(_mp(), _met('bmse000028')) is not None


# ── the gate ──────────────────────────────────────────────────────────────────

def test_single_component_means_nothing_was_deconvolved():
    """K=1: the 'resolved component' is the peak itself."""
    pk = _mcr_peak(k=1, shape_r=(0.99,), c_mean=(3.0,),
                   cand_keys=(('HMDB0000161', 1.47),))
    assert MCR_MIN_OVERLAP_K == 2
    assert _window([pk])._mcr_component_for(_mp(), _met()) is None


def test_shape_mismatch_is_not_shown():
    """A component that does not look like this compound is not its signal."""
    pk = _mcr_peak(shape_r=(0.95, 0.40))     # component 1 → candidate 0
    assert _window([pk])._mcr_component_for(_mp(), _met()) is None


def test_shown_even_when_the_region_verdict_is_inconclusive():
    """Regression: Taurine at 3.41 ppm showed 'MCR: ? R²=0.941 · r=0.82'.

    The region R² fell below its 0.95 threshold, so mcr_confirmed was None and
    the old gate hid a component that had been cleanly resolved (r=0.82).
    """
    pk = _mcr_peak(shape_r=(0.95, 0.82))
    got = _window([pk])._mcr_component_for(_mp(mcr_confirmed=None), _met())
    assert got is not None
    assert MCR_MIN_SHAPE_R <= 0.82


@pytest.mark.parametrize('verdict', [True, False, None])
def test_verdict_does_not_drive_the_display(verdict):
    pk = _mcr_peak()
    assert _window([pk])._mcr_component_for(_mp(mcr_confirmed=verdict), _met()) is not None


# ── amplitude ─────────────────────────────────────────────────────────────────

def test_concentration_restores_data_units():
    """S columns are unit-normalised; C[:, k] carries the actual amount."""
    pk = _mcr_peak()
    _, profile = _window([pk])._mcr_component_for(_mp(), _met())
    raw_col = pk['mcr_annotation']['S'][:, 1]
    assert np.max(profile) == pytest.approx(np.max(raw_col) * 5.0)


def test_missing_concentration_falls_back_to_the_raw_column():
    pk = _mcr_peak(with_c=False)
    _, profile = _window([pk])._mcr_component_for(_mp(), _met())
    assert np.allclose(profile, pk['mcr_annotation']['S'][:, 1])


def test_degenerate_concentration_does_not_zero_the_component():
    pk = _mcr_peak(c_mean=(0.0, 0.0))
    _, profile = _window([pk])._mcr_component_for(_mp(), _met())
    assert float(np.max(profile)) > 0


# ── malformed or absent results ───────────────────────────────────────────────

def test_no_stored_mcr_result_is_handled():
    w = _window([{'rank': 3, 'ppm': 1.47}])
    assert w._mcr_component_for(_mp(), _met()) is None


def test_candidate_absent_from_the_stored_result():
    """A metabolite that was not part of that MCR region has no component."""
    assert _window([_mcr_peak()])._mcr_component_for(
        _mp(3.77), _met('HMDB0009999')) is None


def test_peak_without_a_rank_is_handled():
    assert _window([_mcr_peak()])._mcr_component_for(_mp(rank=None), _met()) is None


def test_component_index_out_of_range_does_not_raise():
    """Guards a stored result whose assignments outnumber its components."""
    pk = _mcr_peak(k=2)
    pk['mcr_annotation']['S'] = pk['mcr_annotation']['S'][:, :1]
    assert _window([pk])._mcr_component_for(_mp(), _met()) is None


def test_missing_k_is_treated_as_no_overlap():
    pk = _mcr_peak()
    del pk['mcr_annotation']['K']
    assert _window([pk])._mcr_component_for(_mp(), _met()) is None


# ── one definition of "trustworthy component" ─────────────────────────────────

def test_scoring_and_view_share_the_constants():
    """The ΔJ substitution and the display must not drift apart."""
    import inspect

    import gui.stages._annotation_search as search_mod
    src = inspect.getsource(search_mod)
    assert 'K >= MCR_MIN_OVERLAP_K' in src
    assert 'c_assign[1] >= MCR_MIN_SHAPE_R' in src

    view = inspect.getsource(ARW._mcr_component_for)
    assert 'MCR_MIN_OVERLAP_K' in view and 'MCR_MIN_SHAPE_R' in view


def test_both_plot_paths_consult_the_mcr_component():
    """The fingerprint and the per-DB-peak zoom must agree on what is shown."""
    import inspect

    for fn in ('_plot_fingerprint', '_plot_db_peak_zoom'):
        src = inspect.getsource(getattr(ARW, fn))
        assert '_mcr_component_for' in src, f'{fn} ignores MCR components'


def test_saved_deconvolution_takes_precedence_over_mcr():
    """The analyst's own deconvolution is an explicit decision; MCR fills in."""
    import inspect

    src = inspect.getsource(ARW._plot_db_peak_zoom)
    assert 'if not _use_deconv_display and mp_ref is not None:' in src


# ── the chooser must never open blank ─────────────────────────────────────────

def test_tier_chooser_defaults_when_nothing_is_stored():
    from processing.signal_quality import (
        ANNOTATABLE_TIERS, SIGNAL_TIERS, normalize_tiers)

    for stored in (None, [], {}):
        assert set(normalize_tiers(stored)) == set(ANNOTATABLE_TIERS)
    assert set(normalize_tiers(SIGNAL_TIERS)) == set(SIGNAL_TIERS)


def test_pre_annotation_defaults_carry_the_tier_key():
    """A missing key left every grade box unticked.

    The defaults used to be a literal inside gui.app's dialog. Both entry points
    now build from the scoring core, so the key has to live there.
    """
    from processing.annotation_scoring import SCORING_DEFAULTS
    from processing.signal_quality import SIGNAL_AVERAGE, SIGNAL_GOOD
    from gui.widgets.annotation_params_dialog import current_params

    assert SCORING_DEFAULTS['annotatable_tiers'] == [SIGNAL_GOOD, SIGNAL_AVERAGE]
    assert current_params(None)['annotatable_tiers'] == [SIGNAL_GOOD, SIGNAL_AVERAGE]
    assert current_params({'dj_bad_hz': 0.6})['annotatable_tiers']

def test_identity_comes_from_the_metabolite_not_the_peak():
    """Regression: the lookup keyed on ids the matched_peak never carries.

    _group_by_metabolite stores the HMDB/BMRB accession on the metabolite and
    gives each matched_peak only positions and scores. Keying the candidate
    search on mp['hmdb_id'] therefore matched nothing, on every peak, and the
    component was never drawn — while unit tests that fed an id-bearing peak
    passed happily.
    """
    pk = _mcr_peak()
    w = _window([pk])
    peak_as_built = _mp()
    assert not (set(peak_as_built) & {'hmdb_id', 'bmrb_id', 'name'})
    assert w._mcr_component_for(peak_as_built, _met()) is not None
    # and without the metabolite there is nothing to key on
    assert w._mcr_component_for(peak_as_built, None) is None


def test_matched_peak_dict_still_lacks_identity_fields():
    """If this ever changes, the helper above can be simplified — but until
    then, the metabolite is the only source of identity."""
    import inspect

    src = inspect.getsource(ARW._group_by_metabolite)
    body = src[src.index("matched_peaks'].append({"):]
    body = body[:body.index('})')]
    for field in ("'hmdb_id'", "'bmrb_id'", "'name'"):
        assert field not in body, f'matched_peaks now carries {field}'


def test_plot_paths_pass_the_metabolite_through():
    import inspect

    for fn in ('_plot_fingerprint', '_plot_db_peak_zoom'):
        src = inspect.getsource(getattr(ARW, fn))
        assert '_mcr_component_for(mp' in src and ', met)' in src, \
            f'{fn} calls the helper without the metabolite'


# ── a singlet cannot be identified by component shape ─────────────────────────

def test_singlet_correlation_is_vacuous_after_alignment():
    """The measurement behind the policy: with +-0.05 ppm of alignment freedom,
    any isolated line matches any other isolated line."""
    ppm = np.linspace(3.40, 3.48, 800)

    def lor(centre, w=0.0015):
        y = w ** 2 / ((ppm - centre) ** 2 + w ** 2)
        return y / np.linalg.norm(y)

    def r(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    line = lor(3.4280)
    assert r(lor(3.4445), line) < 0.0        # unaligned: no correlation at all
    assert r(lor(3.4280), line) > 0.999      # aligned: perfect, whatever it is
    # so a component of two lines still scores well above the 0.70 gate
    two_lobes = lor(3.4400) * 0.55 + lor(3.4280)
    assert r(lor(3.4280), two_lobes) > MCR_MIN_SHAPE_R


def test_search_marks_candidates_informative_only():
    """Two ways an MCR result stops being evidence: a singlet (shape says
    nothing after alignment) and an alignment that landed on another signal."""
    import inspect

    import gui.stages._annotation_search as search_mod
    src = inspect.getsource(search_mod)
    assert "c['mcr_informative_only'] = bool(_c_is_singlet or _c_far)" in src
    assert '_c_disagree > MCR_SHIFT_AGREEMENT_PPM' in src
    assert "c['mcr_confirmed']      = (None if c['mcr_informative_only']" in src
    assert "not c['mcr_informative_only']" in src, 'no MCR ΔJ for those peaks'


def test_shift_is_judged_by_agreement_not_by_size():
    """Regression: an absolute cap rejected the cases where MCR agreed exactly
    with the annotation.

    Inosine 5'-monophosphate 6.1150 [D]: the annotation placed the DB peak at
    6.1024, an offset of -0.0130 ppm, and MCR aligned by -0.0130 ppm — the same
    signal, from both. A 0.01 ppm cap on |shift| called that too far.
    """
    import inspect

    from gui.utils.mcr_utils import MCR_SHIFT_AGREEMENT_PPM
    import gui.stages._annotation_search as search_mod

    src = inspect.getsource(search_mod)
    assert 'MCR_MAX_SHIFT_PPM' not in src, 'the absolute cap is gone'
    assert "float(c.get('peak_ppm'))" in src and "float(c.get('db_ppm'))" in src

    own_delta = 6.1024 - 6.1150
    mcr_shift = -0.0130
    assert abs(mcr_shift - own_delta) <= MCR_SHIFT_AGREEMENT_PPM, \
        'agreeing with the match must pass'
    # a component resolved somewhere else does not
    assert abs(0.0120 - own_delta) > MCR_SHIFT_AGREEMENT_PPM


def test_matched_peaks_carry_the_new_mcr_fields():
    """Otherwise the card and the plot cannot see them."""
    import inspect

    src = inspect.getsource(ARW._group_by_metabolite)
    for field in ("'mcr_shift_ppm'", "'mcr_informative_only'", "'mcr_info_reason'"):
        assert field in src, f'matched_peaks drops {field}'


def test_component_and_reference_are_already_superimposed():
    """The panel draws the DB curve at the OBSERVED position, and MCR aligns its
    profile onto that same signal before correlating — so the curve the r was
    computed against is the one already on screen.

    Regression: a separately shifted copy was drawn on top of that, landing a
    second offset away from both curves (Xanthine 7.7160: DB drawn at 7.6867,
    shifted again by -0.0293 to 7.657, floating alone).
    """
    import inspect

    src = inspect.getsource(ARW._plot_db_peak_zoom)
    assert 'x_db + _sh' not in src, 'DB curve is already at the observed position'
    assert '_mcr_lbl_extra' in src, 'the shift and r belong on the DB curve label'
    assert 'MCR:' in src


def test_hz_is_reserved_for_couplings():
    """An alignment distance is a position, not a coupling. Reported in Hz beside
    the "J = 5.25 Hz" box it reads as the resolved component's own J, which is a
    different quantity entirely — so alignment stays in ppm."""
    import inspect

    src = inspect.getsource(ARW._plot_db_peak_zoom)
    assert 'aligned' in src
    assert "self._get_field_mhz():.1f} Hz" not in src

    import gui.stages._annotation_search as search_mod
    reason = inspect.getsource(search_mod)
    reason = reason[reason.index("resolved a signal"):][:200]
    assert 'Hz' not in reason


def test_no_warning_glyph_on_an_informative_reading():
    """Informative-only is not a warning; the text says what it is."""
    import inspect

    src = inspect.getsource(ARW._plot_db_peak_zoom)
    assert 'ⓘ' not in src


def test_card_reports_the_reason_and_keeps_a_symbol():
    import inspect

    src = inspect.getsource(ARW._plot_db_peak_zoom)
    assert "mp_ref.get('mcr_informative_only')" in src
    assert "mp_ref.get('mcr_info_reason')" in src
    assert "_sym_pk, _color_pk = '—'" in src, 'the row must keep a marker'


def test_labels_name_what_they_measure():
    import inspect

    src = inspect.getsource(ARW._plot_db_peak_zoom)
    assert "f'aligned {_sh:+.4f} ppm'" in src
    assert 'shape r=' in src
    assert "f'shift {float(_shift_pk):+.4f} ppm'" not in src


def test_tier_chooser_defaults_when_nothing_is_stored():
    from processing.signal_quality import (
        ANNOTATABLE_TIERS, SIGNAL_TIERS, normalize_tiers)

    for stored in (None, [], {}):
        assert set(normalize_tiers(stored)) == set(ANNOTATABLE_TIERS)
    assert set(normalize_tiers(SIGNAL_TIERS)) == set(SIGNAL_TIERS)


def test_identity_comes_from_the_metabolite_not_the_peak():
    """Regression: the lookup keyed on ids the matched_peak never carries.

    _group_by_metabolite stores the HMDB/BMRB accession on the metabolite and
    gives each matched_peak only positions and scores. Keying the candidate
    search on mp['hmdb_id'] therefore matched nothing, on every peak, and the
    component was never drawn — while unit tests that fed an id-bearing peak
    passed happily.
    """
    pk = _mcr_peak()
    w = _window([pk])
    peak_as_built = _mp()
    assert not (set(peak_as_built) & {'hmdb_id', 'bmrb_id', 'name'})
    assert w._mcr_component_for(peak_as_built, _met()) is not None
    # and without the metabolite there is nothing to key on
    assert w._mcr_component_for(peak_as_built, None) is None


def test_matched_peak_dict_still_lacks_identity_fields():
    """If this ever changes, the helper above can be simplified — but until
    then, the metabolite is the only source of identity."""
    import inspect

    src = inspect.getsource(ARW._group_by_metabolite)
    body = src[src.index("matched_peaks'].append({"):]
    body = body[:body.index('})')]
    for field in ("'hmdb_id'", "'bmrb_id'", "'name'"):
        assert field not in body, f'matched_peaks now carries {field}'


def test_plot_paths_pass_the_metabolite_through():
    import inspect

    for fn in ('_plot_fingerprint', '_plot_db_peak_zoom'):
        src = inspect.getsource(getattr(ARW, fn))
        assert '_mcr_component_for(mp' in src and ', met)' in src, \
            f'{fn} calls the helper without the metabolite'


# ── a singlet cannot be identified by component shape ─────────────────────────

def test_singlet_correlation_is_vacuous_after_alignment():
    """The measurement behind the policy: with +-0.05 ppm of alignment freedom,
    any isolated line matches any other isolated line."""
    ppm = np.linspace(3.40, 3.48, 800)

    def lor(centre, w=0.0015):
        y = w ** 2 / ((ppm - centre) ** 2 + w ** 2)
        return y / np.linalg.norm(y)

    def r(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    line = lor(3.4280)
    assert r(lor(3.4445), line) < 0.0        # unaligned: no correlation at all
    assert r(lor(3.4280), line) > 0.999      # aligned: perfect, whatever it is
    # so a component of two lines still scores well above the 0.70 gate
    two_lobes = lor(3.4400) * 0.55 + lor(3.4280)
    assert r(lor(3.4280), two_lobes) > MCR_MIN_SHAPE_R
