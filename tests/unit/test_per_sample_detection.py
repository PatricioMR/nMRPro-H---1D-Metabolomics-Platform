"""Tests for per-sample detection + union merge (processing.peak_detection)."""
import numpy as np

from processing.peak_detection import (
    detect_peaks, _detect_single, _consensus_apices,
    _quiet_segment_stats,
)


def _gauss(x, c, a, w):
    return a * np.exp(-((x - c) ** 2) / (2.0 * w ** 2))


def test_union_count_matches_consolidated_clean_spectrum():
    """Union mode must not over-split: on a clean multi-peak spectrum the master
    peak count equals the consolidated path (global min-distance enforced)."""
    rng = np.random.default_rng(0)
    ppm = np.linspace(0.0, 2.0, 8000)
    centres = [0.3, 0.5, 0.9, 1.2, 1.5, 1.7]
    base = sum(_gauss(ppm, c, 1.0, 0.004) for c in centres)
    arr = np.column_stack([base + rng.normal(0, 0.01, 8000) for _ in range(6)])
    cons = detect_peaks(ppm, arr, min_snr=4, baseline_correct=False,
                        global_align=False, per_sample_detection=False)
    uni = detect_peaks(ppm, arr, min_snr=4, baseline_correct=False,
                       global_align=False, per_sample_detection=True)
    assert len(uni) == len(cons) == len(centres)


def test_consensus_merges_close_apices():
    # three samples, apex near index 100 (±3), one stray apex far away
    per_sample = [[100], [102], [98, 500]]
    cons, presence = _consensus_apices(per_sample, n_points=1000, merge_tol_pts=5)
    assert len(cons) == 2
    # the cluster near 100 has all three samples present
    near = min(range(len(cons)), key=lambda k: abs(cons[k] - 100))
    assert presence[near] == [True, True, True]
    far = 1 - near
    assert presence[far] == [False, False, True]


def test_analyte_in_one_sample_survives_union():
    """Analyte present in only sample 0 must survive and be flagged absent in
    the others (aligned_per_sample = [T,F,F,F])."""
    rng = np.random.default_rng(0)
    ppm = np.linspace(0.0, 1.0, 4000)
    peak = _gauss(ppm, 0.500, 1.0, 0.004)
    cols = [peak + rng.normal(0, 0.01, ppm.size)]
    cols += [rng.normal(0, 0.01, ppm.size) for _ in range(3)]
    arr = np.column_stack(cols)

    result = detect_peaks(
        ppm, arr, min_snr=4.0,
        baseline_correct=False, global_align=False,
        per_sample_detection=True,
    )
    target = [p for p in result if abs(p['ppm'] - 0.500) < 0.01]
    assert target, "single-sample analyte vanished"
    apk = target[0]['aligned_per_sample']
    assert apk[0] is True
    assert apk[1:] == [False, False, False]


def test_same_peak_shifted_merges_to_one_master():
    rng = np.random.default_rng(1)
    ppm = np.linspace(0.0, 1.0, 4000)
    # Small residual cross-sample jitter (bulk shift is removed by global align
    # upstream). Spread must stay below merge_tol (~distance/2) to be one line.
    shifts = [-0.0015, -0.0005, 0.0005, 0.0015]
    cols = [_gauss(ppm, 0.500 + d, 1.0, 0.004) + rng.normal(0, 0.008, ppm.size)
            for d in shifts]
    arr = np.column_stack(cols)

    result = detect_peaks(
        ppm, arr, min_snr=4.0, per_peak_max_shift_ppm=0.02,
        baseline_correct=False, global_align=False,
        per_sample_detection=True,
    )
    near = [p for p in result if abs(p['ppm'] - 0.500) < 0.015]
    assert len(near) == 1, f"shifted peak not merged: {[p['ppm'] for p in near]}"
    assert all(near[0]['aligned_per_sample'])


def test_schema_matches_consolidated():
    """Per-sample peaks carry the same keys as consolidated peaks → downstream
    (annotation/quant/export) is unaffected."""
    rng = np.random.default_rng(2)
    ppm = np.linspace(0.0, 1.0, 4000)
    arr = np.column_stack([_gauss(ppm, 0.5, 1.0, 0.004) + rng.normal(0, 0.01, ppm.size)
                           for _ in range(3)])
    common = dict(min_snr=4.0, baseline_correct=False, global_align=False)
    cons = detect_peaks(ppm, arr, per_sample_detection=False, **common)
    per  = detect_peaks(ppm, arr, per_sample_detection=True,  **common)
    assert cons and per
    required = {'ppm', 'height', 'area', 'start_idx', 'end_idx', 'snr', 'rank',
                'area_per_sample', 'ppm_per_sample', 'aligned_per_sample',
                'n_cluster_peaks', 'cluster_rank', 'cluster_lo', 'cluster_hi'}
    assert required <= set(per[0].keys())
    assert required <= set(cons[0].keys())


def test_noise_only_sample_contributes_no_presence():
    rng = np.random.default_rng(3)
    ppm = np.linspace(0.0, 1.0, 4000)
    peak = _gauss(ppm, 0.5, 1.0, 0.004)
    arr = np.column_stack([
        peak + rng.normal(0, 0.01, ppm.size),
        peak + rng.normal(0, 0.01, ppm.size),
        rng.normal(0, 0.01, ppm.size),     # noise-only
    ])
    result = detect_peaks(
        ppm, arr, min_snr=4.0,
        baseline_correct=False, global_align=False,
        per_sample_detection=True,
    )
    target = [p for p in result if abs(p['ppm'] - 0.5) < 0.01][0]
    assert target['aligned_per_sample'][2] is False
