"""The suppressed region is not a source of analytes.

On the reference study ATP's H37 triplet — a real database signal at 4.639 ppm,
1 H — matched the residual water and came back as the study's most abundant
metabolite at 288 mM with an SNR of 31520, a hundred times the next row. The
database has 82 signals between 4.5 and 4.9 ppm, so it was not going to be the
only one.
"""
import numpy as np
import pytest

from processing.spectral_regions import (
    DEFAULT_EXCLUDED_REGIONS, WATER_REGION, format_regions,
    SPLIT_WARNING_FRAC, in_excluded_region, measure_linewidth_ppm,
    parse_regions, split_excluded, unresolvable_split_warning)


def test_the_water_region_is_excluded_by_default():
    """Not opt-in. An analyst who never opens the field must not be the one
    person quantifying the solvent."""
    assert WATER_REGION in DEFAULT_EXCLUDED_REGIONS
    assert in_excluded_region(4.639, DEFAULT_EXCLUDED_REGIONS)


def test_the_metabolite_region_is_untouched():
    for ppm in (0.0, 1.33, 3.05, 4.10, 5.25, 8.46):
        assert not in_excluded_region(ppm, DEFAULT_EXCLUDED_REGIONS)


def test_the_window_covers_the_shoulder_presaturation_flattens():
    """A narrower window leaves the wings, and the wings are what a database
    signal matches — 4.639 is a wing, not the centre."""
    lo, hi = WATER_REGION
    assert lo <= 4.60 and hi >= 4.90


class TestParsing:
    def test_a_pair(self):
        assert parse_regions('4.50-5.00') == [(4.5, 5.0)]

    def test_several(self):
        assert parse_regions('4.50-5.00, 0.00-0.10') == [(4.5, 5.0), (0.0, 0.1)]

    def test_semicolons_and_spaces(self):
        assert parse_regions(' 4.5 - 5.0 ; 0-0.1 ') == [(4.5, 5.0), (0.0, 0.1)]

    def test_reversed_bounds_are_ordered(self):
        assert parse_regions('5.00-4.50') == [(4.5, 5.0)]

    def test_empty_is_no_regions(self):
        assert parse_regions('') == []
        assert parse_regions(None) == []

    @pytest.mark.parametrize('bad', ['4.50', '4.5-5.0-6.0', 'water'])
    def test_malformed_raises_rather_than_being_skipped(self, bad):
        """An analyst who believes the water is excluded and is not would get
        the artefact back with nothing said. That is the failure this module
        exists to prevent, so it must not be reachable by a typo."""
        with pytest.raises(ValueError):
            parse_regions(bad)

    def test_it_round_trips(self):
        assert parse_regions(format_regions([(4.5, 5.0)])) == [(4.5, 5.0)]


class TestSplitting:
    def _peaks(self):
        return [{'rank': 1, 'ppm': 1.33}, {'rank': 2, 'ppm': 4.65},
                {'rank': 3, 'ppm': 8.46}, {'rank': 4, 'ppm': 4.80}]

    def test_peaks_inside_are_separated_out(self):
        kept, dropped = split_excluded(self._peaks(), DEFAULT_EXCLUDED_REGIONS)
        assert [p['ppm'] for p in kept] == [1.33, 8.46]
        assert [p['ppm'] for p in dropped] == [4.65, 4.80]

    def test_the_dropped_ones_come_back_so_they_can_be_reported(self):
        """A peak that disappears with no count attached leaves the analyst
        unable to tell an exclusion from a detection failure, and those call
        for opposite responses."""
        _, dropped = split_excluded(self._peaks(), DEFAULT_EXCLUDED_REGIONS)
        assert len(dropped) == 2

    def test_no_regions_drops_nothing(self):
        kept, dropped = split_excluded(self._peaks(), [])
        assert len(kept) == 4 and dropped == []


def test_detection_excludes_the_region_and_reports_what_it_dropped():
    """End to end through detect_peaks, because the exclusion belongs there —
    annotation, quantification and the Signal List all read that one list and
    must not be able to disagree about which peaks the study contains."""
    from processing.peak_detection import detect_peaks
    ppm = np.linspace(0.0, 10.0, 8001)

    def gauss(c, h, w=0.004):
        return h * np.exp(-0.5 * ((ppm - c) / w) ** 2)

    y = (gauss(1.33, 100.0) + gauss(4.70, 10000.0) + gauss(8.46, 80.0))
    out = detect_peaks(ppm, np.column_stack([y, y]), noise_std=1.0,
                       min_snr=5.0, baseline_correct=False, global_align=False,
                       return_details=True,
                       excluded_regions=DEFAULT_EXCLUDED_REGIONS)
    found = [p['ppm'] for p in out['peaks']]
    assert not any(4.5 <= v <= 5.0 for v in found), found
    assert out['excluded_peaks']
    assert out['excluded_regions'] == DEFAULT_EXCLUDED_REGIONS


def test_ranks_stay_contiguous_after_the_exclusion():
    """Dropped before ranking, so nothing downstream sees a hole where the
    solvent used to be."""
    from processing.peak_detection import detect_peaks
    ppm = np.linspace(0.0, 10.0, 8001)

    def gauss(c, h, w=0.004):
        return h * np.exp(-0.5 * ((ppm - c) / w) ** 2)

    y = gauss(1.33, 100.0) + gauss(4.70, 10000.0) + gauss(8.46, 80.0)
    peaks = detect_peaks(ppm, np.column_stack([y, y]), noise_std=1.0,
                         min_snr=5.0, baseline_correct=False,
                         global_align=False,
                         excluded_regions=DEFAULT_EXCLUDED_REGIONS)
    assert [p['rank'] for p in peaks] == list(range(1, len(peaks) + 1))


# ── One definition, one path ─────────────────────────────────────────────────

def test_the_default_is_defined_in_exactly_one_place():
    """A second copy of the window is a second answer waiting to disagree with
    the first, and the analyst would have no way to tell which one ran."""
    import pathlib
    import re
    hits = []
    for f in pathlib.Path('.').rglob('*.py'):
        if '__pycache__' in str(f) or f.as_posix().startswith('tests/'):
            continue
        for i, line in enumerate(
                f.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            if re.match(r'\s*(WATER_REGION|DEFAULT_EXCLUDED_REGIONS)\s*=', line):
                hits.append(f'{f.as_posix()}:{i}')
    assert hits == ['processing/spectral_regions.py:35',
                    'processing/spectral_regions.py:37'], hits


def test_detection_uses_the_field_and_nothing_else():
    """The range that runs must be the range on screen. Reading the constant
    here instead would ignore whatever the analyst typed."""
    import inspect

    from gui.stages import peak_detection_stage as pds
    src = inspect.getsource(pds.PeakDetectionStage.detect_peaks)
    assert 'parse_regions(self.excluded_regions_var.get())' in src
    assert 'DEFAULT_EXCLUDED_REGIONS' not in src


def test_the_constant_only_seeds_the_field():
    """Its single job is the starting value. Any other use would be a second
    path to the same decision. Counted by name node, not by text, so the
    import line cannot pass for a use."""
    import ast
    import inspect

    from gui.stages import peak_detection_stage as pds
    tree = ast.parse(inspect.getsource(pds))
    uses = [n for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id == 'DEFAULT_EXCLUDED_REGIONS']
    assert len(uses) == 1


def test_what_was_excluded_is_reported_from_what_ran():
    """The analyst can edit the field after detecting, and the peak list being
    quantified is the one the earlier value produced. Reporting the field would
    label a run with a setting it never used."""
    import inspect

    from gui.stages import _quant_computation as qc
    src = inspect.getsource(qc._QuantCompMixin._run)
    assert "getattr(self.peak_stage, '_excluded_regions', None)" in src
    assert 'excluded_regions_var' not in src


def test_the_polling_loop_does_not_rebind_the_overlays_widgets():
    """A nested function that assigns a name anywhere makes that name local to
    the whole function, so a read before the assignment raises rather than
    reaching the enclosing scope.

    Reusing `_lbl` for a piece of label text did exactly that: detection died
    with UnboundLocalError on the first progress tick, before the assignment it
    was shadowed by could run — and the failure surfaced as detection never
    finishing rather than as anything about a name.
    """
    import ast
    import inspect
    import textwrap

    from gui.stages import peak_detection_stage as pds

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(pds.PeakDetectionStage.detect_peaks)))
    poll = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == '_poll')
    assigned = {n.id for n in ast.walk(poll)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    # Widgets the overlay owns and the loop only ever reads.
    for widget in ('_lbl', '_pct_lbl', '_progress_queue', '_result'):
        assert widget not in assigned, (
            f'{widget} is assigned inside _poll, which shadows the enclosing '
            f'binding and breaks every read above the assignment')


# ── A splitting narrower than the linewidth is not a splitting ───────────────

class TestUnresolvableSplitWarning:
    """Two Lorentzians are resolved when their separation approaches their
    width. Below that they merge into one contour, and the maxima a detector
    finds inside it are noise: the same resonance peaks here in one sample and
    there in another, and a per-sample union turns those into master peaks.

    Measured on a 21-sample mouse serum study whose linewidth, read off the
    trace, is 3.22 Hz. L-alanine's methyl doublet had two artefacts on its
    flanks — 1.4758 on the rising edge, 1.4877 in the valley between the real
    lines — each carrying about 1% of a real line's area. Annotation matched
    alanine to those two instead of to the doublet, and the window built from
    their feet clipped a real line out, reporting alanine at 0.045 mM.

        min_distance   artefacts   real lines   flank artefacts, whole study
          1.80 Hz          2           2          66 of 533  (12.4%)
          2.40 Hz          2           2          53 of 441  (12.0%)
          3.00 Hz          0           2          37 of 361  (10.2%)
          4.14 Hz          0           2          22 of 301  ( 7.3%)
    """

    LW = 3.22 / 600.71          # the study's linewidth, measured on the trace

    def _warn(self, hz):
        return unresolvable_split_warning(self.LW, hz / 600.71, 600.71)

    def test_it_warns_at_the_setting_that_produced_the_artefacts(self):
        assert self._warn(1.80)

    def test_it_warns_at_the_next_setting_up_that_still_did(self):
        assert self._warn(2.40)

    def test_it_is_quiet_at_the_setting_that_removed_them(self):
        assert self._warn(3.00) is None

    def test_it_is_quiet_above_that(self):
        assert self._warn(4.14) is None

    def test_the_boundary_lands_near_one_full_linewidth(self):
        """Where the physics puts it: two Lorentzians separate into two maxima
        at roughly one FWHM. A threshold far from that would be a fudge."""
        assert 0.7 < SPLIT_WARNING_FRAC <= 1.0

    def test_the_message_names_both_numbers(self):
        """A warning that does not say what to compare against cannot be acted
        on, and the analyst has no other way to see the linewidth."""
        msg = self._warn(1.80)
        assert '1.80' in msg and '3.22' in msg

    def test_the_scale_is_the_studys_own_linewidth_not_a_constant(self):
        """Linewidth is a property of the shim and of T2. A study shimmed
        twice as well must be allowed twice the resolution."""
        sharp = 1.0 / 600.71
        assert unresolvable_split_warning(sharp, 1.80 / 600.71, 600.71) is None
        assert self._warn(1.80)

    def test_no_linewidth_means_no_claim(self):
        assert unresolvable_split_warning(None, 0.003) is None

    def test_it_only_warns_and_never_changes_the_setting(self):
        """Which splittings a study can resolve is the analyst's call. This
        makes the number they are working against visible, nothing more."""
        import inspect

        from gui.stages import peak_detection_stage as pds
        src = inspect.getsource(pds.PeakDetectionStage.detect_peaks)
        i = src.index('unresolvable_split_warning')
        after = src[i:i + 700]
        assert 'min_width_var.set' not in after
        assert 'self._status(' in after


class TestLinewidthIsMeasuredOnTheTrace:
    """Every width the detector records moves with the setting being judged.

    On one dataset, sweeping min_distance from 1.80 to 4.14 Hz, the median
    width_ppm ran 4.40 -> 7.81 Hz and the median FWHM 1.70 -> 2.77 Hz on the
    same spectra — because splitting one line into fragments makes each
    fragment narrower, and merging neighbours makes each valley-to-valley
    width wider. A threshold built on either advises a value that, once
    applied, moves the measurement again. Read off the trace it held at
    3.37 / 3.22 / 3.22 / 3.22 Hz.
    """

    def _spectrum(self, fwhm_ppm=0.0056):
        import numpy as np
        ppm = np.linspace(0.0, 10.0, 20001)
        hw = fwhm_ppm / 2.0
        y = np.zeros_like(ppm)
        for c in (1.5, 3.0, 5.0, 7.5):
            y += 1000.0 * hw ** 2 / ((ppm - c) ** 2 + hw ** 2)
        peaks = [{'ppm': c, 'height': 1000.0} for c in (1.5, 3.0, 5.0, 7.5)]
        return ppm, y, peaks

    def test_it_recovers_the_width_it_was_given(self):
        ppm, y, peaks = self._spectrum(0.0056)
        assert measure_linewidth_ppm(ppm, y, peaks) == pytest.approx(
            0.0056, rel=0.05)

    def test_it_does_not_read_the_peaks_recorded_boundaries(self):
        """That is the whole point: those move with min_distance."""
        ppm, y, peaks = self._spectrum(0.0056)
        for p in peaks:
            p['width_ppm'] = 0.05          # nonsense, must be ignored
            p['fwhm_pts'] = 3
        assert measure_linewidth_ppm(ppm, y, peaks) == pytest.approx(
            0.0056, rel=0.05)

    def test_a_two_dimensional_matrix_is_averaged(self):
        import numpy as np
        ppm, y, peaks = self._spectrum(0.0056)
        X = np.column_stack([y, y * 2.0])
        assert measure_linewidth_ppm(ppm, X, peaks) == pytest.approx(
            0.0056, rel=0.05)

    def test_no_peaks_means_no_measurement(self):
        ppm, y, _ = self._spectrum()
        assert measure_linewidth_ppm(ppm, y, []) is None

    def test_a_line_that_never_falls_to_half_is_not_measured(self):
        """A walk that runs to the edge measured the neighbourhood, not the
        line, and a number from it would be an invented linewidth."""
        import numpy as np
        ppm = np.linspace(0.0, 10.0, 2001)
        assert measure_linewidth_ppm(
            ppm, np.ones_like(ppm), [{'ppm': 5.0, 'height': 1.0}]) is None
