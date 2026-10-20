# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""One description of the annotation parameters, one dialog.

The menu and the results window each carried their own copy of the parameter
list and their own defaults. They had already drifted — a threshold removed from
one lingered in the other — and four of the knobs no longer reached any code.
"""
import inspect

import pytest

from gui.widgets.annotation_params_dialog import current_params
from processing.annotation_scoring import (
    LEGACY_PARAM_KEYS, PARAMETER_SPEC, SCORING_DEFAULTS, migrate_params)


class TestSpec:
    def test_every_listed_parameter_has_a_default(self):
        for _, label, key, _ in PARAMETER_SPEC:
            assert key in SCORING_DEFAULTS, f'{label} ({key}) has no default'

    def test_every_listed_parameter_is_read_by_the_code(self):
        """A knob nothing reads is worse than no knob: it implies control."""
        import processing.annotation_scoring as core
        from gui.widgets import annotation_results_window as arw
        import gui.stages._annotation_search as srch
        blob = ''.join(inspect.getsource(m) for m in (core, arw, srch))
        for _, label, key, _ in PARAMETER_SPEC:
            if key in ('annotatable_tiers',):
                continue
            assert blob.count(key) > 1, f'{label} ({key}) is only ever declared'

    def test_tolerance_is_not_duplicated_here(self):
        """It is entered on the Annotation panel and owns the Δppm gate."""
        keys = {k for _, _, k, _ in PARAMETER_SPEC}
        assert 'dppm_gold' not in keys
        assert 'dppm_gold_hz' not in keys

    def test_groups_follow_the_decision_hierarchy(self):
        groups = [g for g, _, _, _ in PARAMETER_SPEC]
        assert groups.index('Coupling (ΔJ)') < groups.index('Decision bands')


class TestDefaults:
    def test_signal_grades_are_present(self):
        """A missing key opened the dialog with every grade box unticked."""
        assert current_params(None)['annotatable_tiers']

    def test_stored_values_win_over_defaults(self):
        assert current_params({'dj_bad_hz': 0.9})['dj_bad_hz'] == 0.9

    def test_retired_weights_are_gone(self):
        for dead in ('w_reverse_ns', 'w_found_s', 'w_rri_s', 'w_reverse_s'):
            assert dead not in SCORING_DEFAULTS


class TestMigration:
    def test_reused_weights_are_carried_to_their_new_names(self):
        """w_found_ns and w_rri_ns became the analyte-level coverage and
        corroboration weights; their old labels described neither."""
        out = migrate_params({'w_found_ns': 0.07, 'w_rri_ns': 0.11})
        assert out == {'w_coverage': 0.07, 'w_corroboration': 0.11}

    def test_dead_keys_are_dropped_not_carried(self):
        assert migrate_params({'w_reverse_s': 0.1, 'dppm_gold_hz': 0.5}) == {}

    def test_current_keys_pass_through(self):
        assert migrate_params({'dj_gold_hz': 0.3}) == {'dj_gold_hz': 0.3}

    def test_a_saved_project_still_loads(self):
        old = {'w_found_ns': 0.06, 'w_rri_s': 0.2, 'good_score_thresh': 0.8,
               'dppm_gold_hz': 0.5}
        p = current_params(old)
        assert p['w_coverage'] == 0.06
        assert p['good_score_thresh'] == 0.8
        assert 'w_rri_s' not in p and 'dppm_gold_hz' not in p

    def test_every_legacy_key_maps_to_a_real_one_or_to_none(self):
        for old, new in LEGACY_PARAM_KEYS.items():
            assert new is None or new in SCORING_DEFAULTS, old


class TestProductIdentity:
    """The product was renamed away from NMRPipe (Delaglio et al., 1995)."""

    def test_app_name_and_version_are_one_source(self):
        import app_paths
        assert app_paths.APP_NAME == 'NMRPro-H'
        assert app_paths.APP_VERSION.startswith('NMRPro-H v')
        assert app_paths.__version__ in app_paths.APP_VERSION

    def test_user_data_survives_the_rename(self):
        """%APPDATA%/<name> holds databases the analyst edited; a fresh
        directory seeded from the bundled defaults would lose them."""
        import inspect

        import app_paths
        assert 'nMRPipe' in app_paths.LEGACY_APP_NAMES
        src = inspect.getsource(app_paths.user_data_dir)
        assert 'LEGACY_APP_NAMES' in src and 'shutil.move' in src

    def test_the_old_name_is_gone_from_the_shipped_code(self):
        import glob
        import io
        import os
        offenders = []
        for path in glob.glob('gui/**/*.py', recursive=True) + ['main.py']:
            text = io.open(path, encoding='utf-8', errors='ignore').read()
            if 'nMRPipe' in text or 'NMRPipe' in text:
                offenders.append(os.path.basename(path))
        assert not offenders, f'old product name left in: {offenders}'
