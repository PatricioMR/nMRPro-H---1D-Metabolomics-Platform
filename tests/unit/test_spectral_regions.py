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
    in_excluded_region, parse_regions, split_excluded)


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
