"""A new analysis starts from one set of defaults, read by the interface and
by the bench alike.

When the defaults lived only in the widgets, the validation bench measured
settings no new user would have had: the reference study's results were
taken with the baseline model off, complex multiplets and shared windows
on, while a new user opening the application got the opposite on all three
-- 11 of 18 analytes with 22 false positives instead of 13 with 6.
"""
import pathlib
import re

import pytest

from controllers.analysis_defaults import ANNOTATION, DETECTION, QUANTIFICATION

STAGES = pathlib.Path(__file__).resolve().parents[2] / 'gui' / 'stages'

# (stage file, widget variable, dict name, key)
WIRED = [
    ('peak_detection_stage.py', 'baseline_correct_var', 'DETECTION', 'baseline_model'),
    ('peak_detection_stage.py', 'min_snr_var', 'DETECTION', 'min_snr'),
    ('peak_detection_stage.py', 'min_width_var', 'DETECTION', 'min_distance_ppm'),
    ('peak_detection_stage.py', 'per_sample_var', 'DETECTION', 'per_sample'),
    ('peak_detection_stage.py', 'min_group_presence_var', 'DETECTION',
     'min_group_presence'),
    ('peak_detection_stage.py', 'detect_shoulders_var', 'DETECTION',
     'detect_shoulders'),
    ('annotation_stage.py', 'tolerance_var', 'ANNOTATION', 'tolerance_ppm'),
    ('quantification_stage.py', 'shape_var', 'QUANTIFICATION', 'quantify_shapes'),
    ('quantification_stage.py', 'shared_var', 'QUANTIFICATION', 'resolve_shared'),
    ('quantification_stage.py', 'cascade_var', 'QUANTIFICATION', 'cascade'),
    ('quantification_stage.py', 'singlet_snr_var', 'QUANTIFICATION',
     'singlet_min_snr'),
    ('quantification_stage.py', 'max_disagreement_var', 'QUANTIFICATION',
     'max_disagreement_pct'),
    ('quantification_stage.py', 'local_baseline_var', 'QUANTIFICATION',
     'local_baseline'),
    ('quantification_stage.py', 'norm_var', 'QUANTIFICATION', 'normalisation'),
]


@pytest.mark.parametrize('fname,var,table,key', WIRED)
def test_the_widget_starts_from_the_shared_default(fname, var, table, key):
    src = (STAGES / fname).read_text(encoding='utf-8')
    m = re.search(rf'self\.{var}\s*=\s*tk\.\w+Var\(\s*value=([^)]*)\)', src)
    assert m, f'{var} not found in {fname}'
    assert f"{table}['{key}']" in m.group(1), (
        f'{var} starts from {m.group(1)!r}, not from {table}[{key!r}]')


def test_the_noise_region_is_the_shared_one():
    src = (STAGES / 'peak_detection_stage.py').read_text(encoding='utf-8')
    assert "self.default_noise_region = tuple(DETECTION['noise_region'])" in src


def test_every_key_the_bench_reads_exists():
    from validation.fresh_run import defaults
    d = defaults()
    for key in ('baseline_model', 'noise_region', 'min_snr', 'min_distance_ppm',
                'tolerance_ppm', 'quantify_shapes', 'resolve_shared', 'cascade',
                'singlet_min_snr', 'max_disagreement_pct'):
        assert key in d


def test_a_misspelt_override_is_an_error_not_a_silent_default():
    from validation.fresh_run import defaults
    with pytest.raises(KeyError):
        defaults({'baseline': False})


def test_the_measured_defaults():
    """What the reference study's results were measured with."""
    assert DETECTION['baseline_model'] is False
    assert QUANTIFICATION['quantify_shapes'] is True
    assert QUANTIFICATION['resolve_shared'] is True
    assert QUANTIFICATION['cascade'] is True
    assert QUANTIFICATION['max_disagreement_pct'] == 30.0
    assert ANNOTATION['tolerance_ppm'] == 0.05
