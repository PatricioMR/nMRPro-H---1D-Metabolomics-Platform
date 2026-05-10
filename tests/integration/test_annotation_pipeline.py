# tests/integration/test_annotation_pipeline.py
"""
End-to-end annotation pipeline test using real CSV + real plasma DB.

Flow: load CSV → detect peaks → query DB → score results → validate output
"""
import numpy as np
import pytest

from data.loader import load_all_samples, get_sample_info
from gui.stages.annotation_stage import _ppm_score, _mult_compat_check
from gui.notation.multiplet_scorer import score_multiplet_match


# ── Helpers ────────────────────────────────────────────────────────────────────

def _non_blk_mean(intensities, samples_info):
    """Mean spectrum excluding BLK samples."""
    non_blk = [s['index'] for s in samples_info if s['label'] != 'BLK']
    if not non_blk:
        non_blk = list(range(intensities.shape[1]))
    return intensities[:, non_blk].mean(axis=1)


def _simple_peak_detect(ppm, mean_spec, min_snr=5.0, noise_region=(9.5, 10.5)):
    """Very simple peak finder for testing purposes."""
    from scipy.signal import find_peaks as _fp
    mask  = (ppm >= min(noise_region)) & (ppm <= max(noise_region))
    noise = float(np.std(mean_spec[mask])) if mask.any() else 1.0
    if noise == 0:
        noise = 1.0
    heights, _ = _fp(mean_spec, height=noise * min_snr, distance=5)
    peaks = []
    for rank, idx in enumerate(heights, start=1):
        peaks.append({
            'rank': rank,
            'index': int(idx),
            'ppm': float(ppm[idx]),
            'height': float(mean_spec[idx]),
            'snr': float(mean_spec[idx] / noise),
            'start_idx': max(0, int(idx) - 30),
            'end_idx':   min(len(ppm) - 1, int(idx) + 30),
            'start_ppm': float(ppm[max(0, int(idx) - 30)]),
            'end_ppm':   float(ppm[min(len(ppm) - 1, int(idx) + 30)]),
            'width_ppm': 0.01,
            'status': 'Good',
        })
    return peaks


def _query_db(db_path, ppm_value, tolerance=0.08):
    """Query HMDB/BMRB db for a single ppm value. Returns list of result dicts."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    tables = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if 'multiplets' in tables:
        cur.execute(
            "SELECT m.*, met.name, ABS(m.ppm_center - ?) as ppm_diff "
            "FROM multiplets m "
            "JOIN metabolites met ON m.hmdb_id = met.hmdb_id "
            "WHERE ABS(m.ppm_center - ?) <= ? "
            "ORDER BY ppm_diff LIMIT 10",
            (ppm_value, ppm_value, tolerance))
    else:
        cur.execute(
            "SELECT s.bmrb_id as hmdb_id, met.name, s.ppm as ppm_center, "
            "s.ppm as ppm, s.multiplicity as type, s.j_coupling, "
            "ABS(s.ppm - ?) as ppm_diff "
            "FROM shifts_1h s "
            "JOIN metabolites met ON s.bmrb_id = met.bmrb_id "
            "WHERE ABS(s.ppm - ?) <= ? "
            "ORDER BY ppm_diff LIMIT 10",
            (ppm_value, ppm_value, tolerance))

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCSVLoad:
    def test_real_csv_loads(self, loaded_spectrum):
        assert loaded_spectrum['ppm'] is not None
        assert len(loaded_spectrum['ppm']) > 500

    def test_mean_spectrum_finite(self, loaded_spectrum):
        d    = loaded_spectrum
        info = [{'index': i, 'label': l}
                for i, l in enumerate(d['labels'])]
        mean = _non_blk_mean(d['intensities'], info)
        assert np.isfinite(mean).all()
        assert mean.max() > 0


class TestPeakDetection:
    def test_detects_peaks_in_plasma(self, loaded_spectrum):
        d    = loaded_spectrum
        info = [{'index': i, 'label': l}
                for i, l in enumerate(d['labels'])]
        mean  = _non_blk_mean(d['intensities'], info)
        peaks = _simple_peak_detect(d['ppm'], mean)
        assert len(peaks) >= 5, f'Expected ≥5 peaks, got {len(peaks)}'

    def test_peaks_within_ppm_range(self, loaded_spectrum):
        d    = loaded_spectrum
        info = [{'index': i, 'label': l}
                for i, l in enumerate(d['labels'])]
        mean  = _non_blk_mean(d['intensities'], info)
        peaks = _simple_peak_detect(d['ppm'], mean)
        for p in peaks:
            assert -1.0 <= p['ppm'] <= 12.0, f"Peak ppm out of range: {p['ppm']}"


class TestDBQueries:
    PLASMA_PPMS = [3.03, 1.33, 3.55, 2.03]   # creatine, lactate, glycine, NAA

    @pytest.mark.parametrize('ppm', PLASMA_PPMS)
    def test_db_returns_results(self, plasma_db_path, ppm):
        rows = _query_db(plasma_db_path, ppm, tolerance=0.1)
        assert len(rows) > 0, f'No DB results near {ppm} ppm'

    def test_result_has_name(self, plasma_db_path):
        rows = _query_db(plasma_db_path, 3.03)
        assert all('name' in r for r in rows)

    def test_result_has_ppm_center(self, plasma_db_path):
        rows = _query_db(plasma_db_path, 3.03)
        assert all('ppm_center' in r for r in rows)


class TestScoringPipeline:
    def test_ppm_score_at_match(self):
        score = _ppm_score(3.03, 3.03, 0.05)
        assert score == pytest.approx(1.0)

    def test_ppm_score_decreases_with_distance(self):
        s0 = _ppm_score(3.03, 3.03, 0.05)
        s1 = _ppm_score(3.04, 3.03, 0.05)
        s2 = _ppm_score(3.06, 3.03, 0.05)
        assert s0 > s1 > s2

    def test_full_annotation_pipeline(self, loaded_spectrum, plasma_db_path):
        """Full flow: load → detect → query → score. Must produce ≥1 scored result."""
        d    = loaded_spectrum
        info = [{'index': i, 'label': l}
                for i, l in enumerate(d['labels'])]
        mean  = _non_blk_mean(d['intensities'], info)
        peaks = _simple_peak_detect(d['ppm'], mean, min_snr=10.0)

        scored = []
        for peak in peaks[:10]:   # limit to first 10 peaks for speed
            rows = _query_db(plasma_db_path, peak['ppm'], tolerance=0.08)
            for row in rows:
                row['peak_ppm']  = peak['ppm']
                row['peak_rank'] = peak['rank']
                ps = _ppm_score(peak['ppm'], row['ppm_center'], 0.05)
                ms, _, _ = score_multiplet_match(
                    row, peaks, tolerance=0.05, spectrometer_mhz=600.0,
                    ppm_array=d['ppm'], intensity_array=mean)
                row['composite_score'] = 0.4 * ps + 0.6 * ms
                scored.append(row)

        assert len(scored) > 0, 'No results scored in annotation pipeline'
        best = max(scored, key=lambda r: r['composite_score'])
        assert best['composite_score'] > 0.0
