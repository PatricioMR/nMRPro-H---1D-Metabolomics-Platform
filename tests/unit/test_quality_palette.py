# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The Peak List and the annotation window must read as one confidence scale."""
import pytest

from processing.signal_quality import (
    SIGNAL_AVERAGE, SIGNAL_GOOD, SIGNAL_POOR, SIGNAL_TIERS)
from gui.utils.quality_palette import (
    ANNOTATION_QUALITY_TIER,
    QUALITY_BG,
    QUALITY_FG,
    QUALITY_LABEL,
    tier_from_annotation_quality,
)


@pytest.mark.parametrize('mapping', [QUALITY_BG, QUALITY_FG, QUALITY_LABEL])
def test_every_tier_has_an_entry(mapping):
    assert set(mapping) == set(SIGNAL_TIERS)


def test_colours_are_distinct_per_tier():
    assert len(set(QUALITY_BG.values())) == len(SIGNAL_TIERS)
    assert len(set(QUALITY_FG.values())) == len(SIGNAL_TIERS)


@pytest.mark.parametrize('quality,tier', [
    ('good', SIGNAL_GOOD),
    ('bad',  SIGNAL_POOR),
    (None,   SIGNAL_AVERAGE),
])
def test_annotation_verdict_maps_to_tier(quality, tier):
    assert tier_from_annotation_quality(quality) == tier
    assert ANNOTATION_QUALITY_TIER[quality] == tier


def test_no_verdict_reads_as_average_not_poor():
    """An unreached verdict is 'unconfirmed', never 'rejected'."""
    assert tier_from_annotation_quality(None) == SIGNAL_AVERAGE


def test_unknown_verdict_falls_back_to_average():
    assert tier_from_annotation_quality('undefined') == SIGNAL_AVERAGE
    assert tier_from_annotation_quality('') == SIGNAL_AVERAGE


def test_peak_list_shares_the_palette():
    """The Peak List must not keep a second copy of the colours."""
    from gui.widgets.peak_list_window import STATUS_COLORS, STATUS_FG
    assert STATUS_COLORS == QUALITY_BG
    assert STATUS_FG == QUALITY_FG


def test_peak_list_undefined_status_is_the_average_tier():
    from gui.widgets.peak_list_window import STATUS_UNDEFINED
    assert STATUS_UNDEFINED == SIGNAL_AVERAGE


def test_labels_carry_a_glyph_and_the_tier_name():
    for tier, label in QUALITY_LABEL.items():
        assert tier in label
        assert label.split()[0] not in SIGNAL_TIERS   # a glyph, not a word


def test_deconvolution_panel_shares_the_vocabulary():
    """The Deconvolution Tool must not keep a third copy of the tiers."""
    from gui.notation.deconvolution.deconv_list_panel import (
        STATUS_COLORS, STATUS_GOOD, STATUS_POOR, STATUS_SYMBOL,
        STATUS_UNDEFINED)
    assert (STATUS_GOOD, STATUS_UNDEFINED, STATUS_POOR) == SIGNAL_TIERS
    assert STATUS_COLORS == QUALITY_FG
    assert STATUS_SYMBOL == QUALITY_LABEL


def test_legacy_undefined_still_counts_as_a_status():
    """Sub-peaks saved under the old vocabulary must not lose their mark."""
    from gui.notation.deconvolution.deconv_list_panel import LEGACY_UNDEFINED
    assert LEGACY_UNDEFINED == 'Undefined'
    assert LEGACY_UNDEFINED not in SIGNAL_TIERS


def test_no_tier_label_is_a_placeholder_dash():
    """A row must read Good/Average/Poor — never a bare dash."""
    assert all('—' not in label for label in QUALITY_LABEL.values())


# ── grade chooser (Annotation Parameters) ─────────────────────────────────────

def test_annotation_stage_exposes_the_configured_grades():
    """The search reads the selection through this one accessor."""
    from processing.signal_quality import ANNOTATABLE_TIERS, normalize_tiers
    from gui.stages.annotation_stage import AnnotationStage

    stage = AnnotationStage.__new__(AnnotationStage)
    stage._annotation_params = None
    assert stage.annotatable_tiers() == ANNOTATABLE_TIERS

    stage._annotation_params = {'annotatable_tiers': ['Good']}
    assert stage.annotatable_tiers() == normalize_tiers(['Good'])

    stage._annotation_params = {'annotatable_tiers': []}
    assert stage.annotatable_tiers() == ANNOTATABLE_TIERS


def test_read_tier_choice_returns_canonical_order():
    from gui.utils.quality_palette import read_tier_choice

    class _Var:
        def __init__(self, v): self._v = v
        def get(self): return self._v

    variables = {SIGNAL_POOR: _Var(True), SIGNAL_GOOD: _Var(True),
                 SIGNAL_AVERAGE: _Var(False)}
    assert read_tier_choice(variables) == [SIGNAL_GOOD, SIGNAL_POOR]
