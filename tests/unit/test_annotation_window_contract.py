"""Structural contracts for AnnotationResultsWindow.

The results window catches exceptions broadly while building rows, so a broken
method signature does not raise — it produces an empty window. These tests
check the shapes that no runtime test would catch.
"""
import inspect

import pytest

from gui.widgets.annotation_results_window import AnnotationResultsWindow as ARW


def test_is_valid_candidate_is_static():
    """Called as self._is_valid_candidate(mp): without @staticmethod the window
    instance lands in the mp slot, every candidate check raises, and the tree
    silently renders empty."""
    assert isinstance(inspect.getattr_static(ARW, '_is_valid_candidate'),
                      staticmethod)


@pytest.mark.parametrize('mp,expected', [
    ({'ppm_diff': 0.01, 'type': 's'}, True),
    ({'ppm_diff': 0.01, 'type': 's', 'annotation_quality': 'bad'}, False),
    ({'ppm_diff': 0.20, 'type': 's'}, False),
    ({'ppm_diff': 0.01, 'type': 'd', 'delta_j_hz': 0.1}, True),
    ({'ppm_diff': 0.01, 'type': 'd', 'delta_j_hz': 0.9}, False),
])
def test_is_valid_candidate_verdicts(mp, expected):
    assert ARW._is_valid_candidate(mp) is expected


def test_anchor_confirmation_methods_are_gone():
    """Confirmation of a singlet by a companion multiplet was removed as too
    variable; nothing may quietly reintroduce it."""
    for name in ('_propagate_j_anchor_quality', '_propagate_anchor_position',
                 '_promote_anchor_confirmed_peaks'):
        assert not hasattr(ARW, name), f'{name} is back'


def test_scoring_pipeline_methods_all_exist():
    """Every method the build chain calls must be present and callable."""
    for name in ('_compute_scores',
                 '_enrich_scores',
                 '_auto_mark_good',
                 '_promote_singlets',
                 '_sync_anchors_to_peaks',
                 '_build_peak_assignments', '_populate'):
        assert callable(getattr(ARW, name, None)), f'{name} missing'


# ── stage construction ────────────────────────────────────────────────────────

def test_annotation_stage_init_ends_by_building_its_widgets():
    """A method inserted mid-__init__ once swallowed the create_widgets() call
    into its own body, so the stage built no UI and its menu never appeared.
    Nothing raised — the widgets simply were not there."""
    import ast
    import inspect

    from gui.stages.annotation_stage import AnnotationStage

    src = inspect.getsource(AnnotationStage.__init__)
    tree = ast.parse(src.lstrip().replace('\n    ', '\n'))
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'create_widgets' in calls


def test_no_unreachable_code_after_return_in_stage_methods():
    """Guards the same splice error anywhere in the annotation stage."""
    import ast
    import inspect

    from gui.stages import annotation_stage

    tree = ast.parse(inspect.getsource(annotation_stage))
    offenders = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            for i, stmt in enumerate(fn.body[:-1]):
                if isinstance(stmt, ast.Return):
                    offenders.append(f'{cls.name}.{fn.name}')
    assert offenders == []


# ── The window shows results; it does not produce them ───────────────────────

def test_window_reuses_the_search_measurement():
    """Line counts and ΔJ come from the annotation search.

    Whether a multiplet's expected lines are present decides if the analyte is a
    candidate at all, so it is measured while candidates are built. The window
    re-measures only when an input has changed since — ppm calibration moving
    the axis, or a parameter changing the matching tolerance.
    """
    src = inspect.getsource(ARW._compute_scores)
    assert 'remeasure=' in src and '_lines_dirty' in src

    from processing.annotation_scoring import resolve_dj
    core = inspect.getsource(resolve_dj)
    assert 'if (not remeasure' in core, 'the stored measurement must be reused'
    guard = core[core.index('remeasure'):]
    assert 'measure_multiplet' in guard, 'the re-measure must sit behind the guard'


def test_stale_line_flag_is_raised_where_inputs_change():
    for name in ('_calibrate_ppm_from_anchors', '_recompute_and_refresh'):
        assert '_lines_dirty = True' in inspect.getsource(getattr(ARW, name)), name


def test_stale_line_flag_is_cleared_by_a_scoring_pass():
    assert '_lines_dirty = False' in inspect.getsource(ARW._enrich_scores)


def test_no_scoring_formula_lives_in_the_window():
    """Every score the window shows comes from processing.annotation_scoring."""
    for name in ('_compute_scores', '_set_analyte_score'):
        src = inspect.getsource(getattr(ARW, name))
        assert 'dynamic_composite' not in src, name


def test_only_one_j_estimator_survives():
    """Five ways of measuring J once coexisted:

      _observe_j_hz               separation closest to the DB J (circular)
      _multiplet_score line mean  mean of whatever spacings came back
      _enrich_delta_j_from_spectrum  index-aware span/idx_span
      measure_multiplet           snap, count, median of spacings
      mcr_delta_j_hz              measured on the deconvolved component

    Only the last two remain: measure_multiplet, run once in the annotation
    search, and the MCR value that legitimately replaces it when a component
    has been deconvolved.
    """
    assert not hasattr(ARW, '_enrich_delta_j_from_spectrum')
    import gui.stages._annotation_search as srch
    assert not hasattr(srch, '_observe_j_hz')


def test_line_fabrication_is_gone():
    """_auto_resolve_unmatched_lines invented a multiplet when one was missing.

    It took the N tallest peaks inside ±tolerance (±0.05 ppm = ±30 Hz at
    600 MHz) and declared them the multiplet's lines — no equal-spacing check,
    no intensity-pattern check, nothing tying them to one spin system. When
    find_peaks returned nothing it fell back to the tallest DATA POINTS, which
    are adjacent samples on a single peak, and treated them as separate lines.
    It then assigned theoretical Pascal intensities to whatever it had grabbed
    and overwrote peak_ppm and ppm_diff — a primary decision criterion — with
    the centroid of the invented lines.

    CLAUDE.md: a peak not detected with confidence is reported as not detected,
    never interpolated silently.
    """
    assert not hasattr(ARW, '_auto_resolve_unmatched_lines')
    src = inspect.getsource(ARW)
    assert '_auto_resolved' not in src, 'the fabrication flag is back'


def test_dj_rejection_lives_in_one_place():
    """Score and quality are two faces of one decision.

    _enrich_scores zeroed the score when ΔJ >= dj_bad_hz while
    _filter_incompatible_candidates separately marked the same peak 'bad'. One
    rule, two methods, and nothing kept them in step.
    """
    assert not hasattr(ARW, '_filter_incompatible_candidates')
    src = inspect.getsource(ARW._enrich_scores)
    assert 'dj_bad_hz' in src
    assert "annotation_quality'] = 'bad'" in src
    assert "annotation_quality') is None" in src, 'manual marks must be preserved'


def test_redundant_scoring_pass_is_gone():
    """_adjust_scores_for_competition had shrunk to the loop _enrich_scores
    already ends with — a second pass recomputing what had just been computed."""
    assert not hasattr(ARW, '_adjust_scores_for_competition')


def test_identity_confirmation_flag_is_gone():
    """j_met_confirmed fed a second 'good' gate that promoted nothing.

    Its threshold (ΔJ < 2.0 Hz) contradicted dj_bad_hz (0.50): 65 of 173
    multiplets in a real dataset came out marked 'bad' AND confirmed at once.
    Once the threshold was made consistent, the gate it fed became unreachable —
    the full-multiplet gate runs first and is strictly weaker (ΔJ < 0.50
    contains ΔJ < 0.20), so it had already promoted everything the gold gate
    could. Measured on that dataset: the gold gate promoted 0 peaks the
    full-multiplet gate had not.
    """
    assert not hasattr(ARW, '_propagate_multiplet_confirmation')
    assert 'j_met_confirmed' not in inspect.getsource(ARW), 'the flag is back'


def test_auto_mark_good_delegates_the_verdict():
    """The rules live in processing.annotation_scoring.quality_verdict; the
    window applies them and protects manual marks."""
    src = inspect.getsource(ARW._auto_mark_good)
    assert 'quality_verdict(' in src
    assert '_quality_marks' in src, 'manual marks must still be protected'
    assert '0.80' not in src, 'a hardcoded threshold is back'


def test_singlet_only_branch_is_gone():
    """It sat after the singlet early-return, so it could never run.

    "all signals are singlets" implies the current one is a singlet, and a
    singlet returns before that point. The branch carried the only use of the
    fingerprint-coverage threshold, which was therefore unreachable too.
    """
    from processing.annotation_scoring import PARAMETER_SPEC, SCORING_DEFAULTS
    blob = inspect.getsource(ARW)
    assert '_singlet_only_am' not in blob and '_fp_gate' not in blob
    assert 'fp_coverage_pct' not in SCORING_DEFAULTS
    assert 'fp_coverage_pct' not in {k for _, _, k, _ in PARAMETER_SPEC}

def test_promotion_rules_do_not_rewrite_the_score():
    """A decision rule reads the evidence; it does not rewrite it.

    All three promotion rules used to raise composite_score to good_score_thresh
    so the number would match the green mark. In a real dataset 92 of 540
    matched peaks carried a score of exactly 0.75 written by a rule rather than
    computed — and that value is not inert: it is the last element of the shared
    ranking key and the tie-break the quantification reads. The demotion rule
    reset the quality but never the score, so a downgraded peak kept a score it
    had not earned.
    """
    for name in ('_promote_singlets',):
        src = inspect.getsource(getattr(ARW, name))
        assert "composite_score'] = " not in src, f'{name} writes the score'


def test_ppm_gates_read_the_parameter_not_a_literal():
    """One threshold, one owner.

    0.05 appeared as a literal in _is_valid_candidate's default and in the
    anchor check, alongside _params['dppm_gold'] which holds the same value.
    Three copies of one threshold drift apart — that is how the singlet rules
    ended up 60x apart from each other.
    """
    src = inspect.getsource(ARW._is_valid_candidate)
    assert 'DPPM_GOLD_DEFAULT' in src or 'dppm_gold' in src

    from gui.widgets.annotation_results_window import DPPM_GOLD_DEFAULT
    from processing.annotation_scoring import SCORING_DEFAULTS
    assert DPPM_GOLD_DEFAULT == SCORING_DEFAULTS['dppm_gold']


def test_old_confidence_classification_is_gone():
    """The A–X classification was superseded and was never displayed.

    Its tags were configured but never applied to a row, the local that read the
    class was assigned and never used, and nothing else in the codebase read the
    key — it was computed, written into every saved project, and thrown away.
    Its C and D bands were the score bands renamed, with their own copies of the
    good/bad thresholds.
    """
    assert not hasattr(ARW, '_classify_confidence_classes')
    src = inspect.getsource(ARW)
    assert 'confidence_class' not in src
    with pytest.raises(ImportError):
        from gui.notation.confidence_classifier import classify_annotation  # noqa


def test_nnls_validation_is_gated_in_one_place():
    """It ran on three paths and was commented out on the one that builds the
    window — the inconsistency that made it look enabled when it was not.

    It reconstructs the whole spectrum from every good candidate and solves a
    non-negative least-squares problem over a (points x metabolites) matrix, on
    the Tk thread, while the window is being built.
    """
    from gui.widgets import annotation_results_window as arw
    assert arw.NNLS_VALIDATION_ENABLED is False
    src = inspect.getsource(ARW._nnls_spectral_validation)
    assert 'if not NNLS_VALIDATION_ENABLED:' in src
    assert '# self._nnls_spectral_validation()' not in inspect.getsource(arw), \
        'a call site is commented out again instead of using the flag'


def test_no_method_references_an_undefined_name():
    """Guards against a deletion taking a definition with it.

    Removing the below-detection-limit handling cut the block that built the
    fingerprint plot's legend handles, and the resulting NameError only surfaced
    when a metabolite was selected — after the window had already been built.
    """
    import builtins
    import symtable
    import textwrap

    from gui.widgets import annotation_results_window as arw

    known = set(dir(arw)) | set(dir(builtins))
    undefined = []
    for name, fn in vars(ARW).items():
        if not callable(fn) or isinstance(fn, type):
            continue
        try:
            src = textwrap.dedent(inspect.getsource(fn))
            table = symtable.symtable(src, '<f>', 'exec').get_children()[0]
        except Exception:
            continue
        for sym in table.get_symbols():
            if (sym.is_referenced() and not sym.is_assigned()
                    and not sym.is_parameter()
                    and sym.get_name() not in known):
                undefined.append(f'{name}: {sym.get_name()}')
    assert not undefined, 'undefined names: ' + ', '.join(undefined)
