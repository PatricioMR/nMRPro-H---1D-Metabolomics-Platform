"""A peak is present in a sample when it was detected there, not when the
region happens to be loud.

The filter exists to drop what per-sample detection picks up in a lone sample —
its own comment names the case: "a noise apex next to a real multiplet that then
gets annotated". It asked the spectrum instead, `intensity[idx, s] >= threshold`,
and on the flank of a strong line that is true in every sample. So the fragments
it was written to remove passed it at any threshold.

Measured on a 21-sample mouse serum study. L-alanine's methyl doublet at
1.4787 / 1.4905 (detected in 16 of 16 samples) had two fragments beside it at
1.4758 and 1.4877, detected in 1 and 2 samples. Annotation matched alanine to
the fragments — they sat 0.003 ppm nearer the database position — and the
integration window built from their feet clipped a real line out. Alanine was
reported at 0.045 mM, about a tenth of what the region holds. Raising
min_group_presence from 3 to 12 changed nothing, because the fragments passed
on intensity.

Across that project, 129 of 394 peaks were detected in fewer than three samples
of any one group.
"""
import numpy as np
import pytest

from processing.peak_detection import detect_peaks

PPM = np.linspace(0.0, 10.0, 8001)


def _gauss(centre, height, width=0.004):
    return height * np.exp(-0.5 * ((PPM - centre) / width) ** 2)


def _samples_info(n_per_group=4):
    return ([{'label': 'A', 'type': 'Sample'}] * n_per_group
            + [{'label': 'B', 'type': 'Sample'}] * n_per_group)


def _study(n_per_group=4, seed=0):
    """One strong line at 3.00 ppm in every sample, plus noise."""
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(2 * n_per_group):
        cols.append(_gauss(3.000, 5000.0) + rng.normal(0, 1.0, PPM.size))
    return np.column_stack(cols)


def _detect(X, **kw):
    args = dict(noise_std=1.0, min_snr=6.0, min_distance=0.004,
                baseline_correct=False, global_align=False,
                samples_info=_samples_info(X.shape[1] // 2),
                per_sample_detection=True, detect_shoulders=False,
                min_group_presence=3)
    args.update(kw)
    return detect_peaks(PPM, X, **args)


def test_a_peak_detected_in_one_sample_is_dropped():
    """The whole point of the filter. A spike present in a single sample is
    not an analyte."""
    X = _study()
    X[:, 0] += _gauss(6.500, 4000.0)          # only sample 0
    found = [p['ppm'] for p in _detect(X)]
    assert not any(abs(v - 6.500) < 0.01 for v in found), found


def test_a_peak_detected_across_a_group_is_kept():
    X = _study()
    for s in range(4):                         # all of group A
        X[:, s] += _gauss(6.500, 4000.0)
    found = [p['ppm'] for p in _detect(X)]
    assert any(abs(v - 6.500) < 0.01 for v in found), found


def test_the_loud_flank_of_a_strong_line_does_not_pass_for_presence():
    """The defect. A fragment on the flank sits where the intensity is far
    above the noise in every sample, so an intensity test counts it present
    everywhere however few samples actually peaked there."""
    X = _study()
    # A spike on the rising flank of the 3.000 line, in one sample only.
    X[:, 0] += _gauss(2.988, 900.0, width=0.0015)
    found = [p['ppm'] for p in _detect(X)]
    assert any(abs(v - 3.000) < 0.01 for v in found), 'the real line went'
    assert not any(2.983 < v < 2.995 for v in found), found


def test_raising_the_threshold_used_to_change_nothing_and_now_does():
    """On the reference study the fragments survived min_group_presence 12 of
    16, which is what showed the filter was measuring something else."""
    X = _study()
    for s in range(2):                         # two samples of group A
        X[:, s] += _gauss(6.500, 4000.0)
    kept_at_2 = [p['ppm'] for p in _detect(X, min_group_presence=2)]
    kept_at_4 = [p['ppm'] for p in _detect(X, min_group_presence=4)]
    assert any(abs(v - 6.500) < 0.01 for v in kept_at_2)
    assert not any(abs(v - 6.500) < 0.01 for v in kept_at_4)


def test_a_group_smaller_than_the_threshold_needs_all_of_its_samples():
    """Otherwise a threshold above the group size would drop everything."""
    X = _study(n_per_group=2)
    for s in range(2):
        X[:, s] += _gauss(6.500, 4000.0)
    found = [p['ppm'] for p in _detect(X, min_group_presence=5)]
    assert any(abs(v - 6.500) < 0.01 for v in found), found


def test_the_filter_reads_the_presence_the_union_measured():
    """Two answers to "is this peak in this sample" is how the fragments got
    through: the union knew they were in one sample, the filter asked the
    spectrum and heard sixteen."""
    import inspect
    src = inspect.getsource(detect_peaks)
    i = src.index('def _detected_in(')
    body = src[i:i + 1400]
    assert "peak.get('aligned_per_sample')" in body
    # The intensity test survives only as the no-per-sample-detection fallback.
    assert 'if pres:' in body


def test_without_per_sample_detection_it_still_has_an_answer():
    """The single-pass path never builds a presence vector, and a filter that
    dropped everything there would be worse than one that measures loosely."""
    X = _study()
    for s in range(4):
        X[:, s] += _gauss(6.500, 4000.0)
    found = [p['ppm'] for p in _detect(X, per_sample_detection=False)]
    assert any(abs(v - 6.500) < 0.01 for v in found), found
