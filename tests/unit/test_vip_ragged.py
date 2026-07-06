"""Regression: _compute_vip_scores must handle per-sample vectors of differing
lengths (ragged) without the numpy 'inhomogeneous shape' error that previously
skipped ALL VIP scores."""
import types
import numpy as np

from gui.stages._quant_computation import _QuantCompMixin


def _make(results, groups, n_samples):
    f = _QuantCompMixin.__new__(_QuantCompMixin)
    f.app = types.SimpleNamespace(samples_info=[{} for _ in range(n_samples)])
    f._quant_results = results
    f._last_blk_indices = []
    f._last_groups = groups
    return f


def test_ragged_concs_do_not_skip_vip():
    # Five analytes whose _concs_per_sample have lengths 6,4,6,7,6.
    rng = np.random.default_rng(0)
    results = [{'_concs_per_sample': list(rng.normal(5, 1, n))}
               for n in (6, 4, 6, 7, 6)]
    f = _make(results, {'A': [0, 1, 2], 'B': [3, 4, 5]}, 6)
    f._compute_vip_scores()
    vips = [r['vip_score'] for r in results]
    # At least one finite VIP → computation ran (old bug made them all skipped).
    assert any(np.isfinite(v) for v in vips)


def test_uniform_length_still_works():
    rng = np.random.default_rng(1)
    results = [{'_concs_per_sample': list(rng.normal(5, 1, 6))} for _ in range(5)]
    f = _make(results, {'A': [0, 1, 2], 'B': [3, 4, 5]}, 6)
    f._compute_vip_scores()
    assert all('vip_score' in r for r in results)


def test_single_group_sets_nan():
    rng = np.random.default_rng(2)
    results = [{'_concs_per_sample': list(rng.normal(5, 1, 6))} for _ in range(4)]
    f = _make(results, {'A': [0, 1, 2, 3]}, 6)
    f._compute_vip_scores()
    assert all(np.isnan(r['vip_score']) for r in results)
