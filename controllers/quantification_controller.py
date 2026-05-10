# controllers/quantification_controller.py
"""
QuantificationController — selects the best peak per metabolite and
calculates a reliability score (0–100).

Reliability score
-----------------
  peak_uniqueness  (0–50) : penalises overlap at the chosen peak.
                            Uses overlap_count from the database (HMDB) or,
                            when not available (BMRB/CSV), counts how many
                            OTHER metabolites from the annotation results
                            are within < tolerance of that peak.

  corroboration    (0–30) : bonus if the same metabolite appears in
                            ANOTHER peak (different peak_rank) with
                            uniqueness > 0 (i.e., not completely
                            overlapped).

  intensity        (0–20) : proportional to the relative intensity of
                            the peak in the mean spectrum.

Peak selection for quantification
----------------------------------
  For each hmdb_id, the controller chooses the hit with the lowest
  overlap_count (calculated or from the database). In case of a tie,
  it prefers the hit with the highest intensity in the spectrum.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional


class QuantificationController:

    OVERLAP_PENALTY     = 8    # points deducted per competing metabolite
    MAX_UNIQUENESS      = 50   # ceiling of the uniqueness component
    CORROBORATION_BONUS = 30   # bonus for independent corroboration
    MAX_INTENSITY       = 20   # ceiling of the intensity component

    def __init__(self, ppm: np.ndarray, intensity_mean: np.ndarray,
                 tolerance: float = 0.05):
        """
        Args:
            ppm:            1-D array of ppm values (full spectrum)
            intensity_mean: 1-D array of mean sample intensity
            tolerance:      tolerance in ppm used during annotation
        """
        self.ppm       = ppm
        self.intensity = intensity_mean
        self.tolerance = tolerance
        self.max_intensity = float(np.max(intensity_mean)) if len(intensity_mean) else 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Main API
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, annotation_results: List[Dict]) -> List[Dict]:
        """
        Receives the flat list of annotation results
        (output of annotation_stage._last_results_args['results'])
        and returns a list of dicts — one per metabolite — with:

          hmdb_id              — metabolite identifier
          name                 — name
          best_hit             — annotation dict for the chosen peak
          best_peak_ppm        — ppm of the peak chosen for quantification
          overlap_count        — number of other metabolites at that peak
          overlapping_names    — list of names of overlapping metabolites
          corroborated         — bool: appears in another independent peak?
          corroboration_peaks  — list of ppm values of corroborating peaks
          peak_uniqueness      — score 0–50
          intensity_score      — score 0–20
          confidence_score     — total score 0–100
          confidence_label     — 'High' / 'Medium' / 'Low'
          all_hits             — all hits for this metabolite (for integration)
        """
        if not annotation_results:
            return []

        tol = self.tolerance

        # ── Step 1: calculate overlap from the results themselves ────────────
        # Group all hits by peak_ppm (rounded to tolerance)
        # to determine which metabolites share each peak.
        hits_by_peak: Dict[int, List[Dict]] = {}   # bucket_key → list of hits
        for r in annotation_results:
            key = self._bucket(r.get('peak_ppm', r.get('ppm', 0)), tol)
            hits_by_peak.setdefault(key, []).append(r)

        # ── Step 2: enrich each hit with calculated overlap ──────────────────
        # If the database already provided overlap_count, keep it; otherwise
        # count the number of distinct hmdb_ids in that bucket minus 1.
        for r in annotation_results:
            if 'overlap_count' not in r or r.get('overlap_count') is None:
                key   = self._bucket(r.get('peak_ppm', r.get('ppm', 0)), tol)
                peers = hits_by_peak.get(key, [])
                ids_in_bucket = {
                    (p.get('hmdb_id') or p.get('bmrb_id', ''))
                    for p in peers
                }
                mid = r.get('hmdb_id') or r.get('bmrb_id', '')
                r['overlap_count'] = max(0, len(ids_in_bucket) - 1)
                r['_peers_in_bucket'] = [
                    p.get('name', '')
                    for p in peers
                    if (p.get('hmdb_id') or p.get('bmrb_id', '')) != mid
                ]

        # ── Step 3: group by metabolite ───────────────────────────────────────
        met_map: Dict[str, Dict] = {}
        for r in annotation_results:
            mid  = r.get('hmdb_id') or r.get('bmrb_id', 'unknown')
            met_map.setdefault(mid, {
                'hmdb_id': mid,
                'name':    r.get('name', mid),
                'hits':    [],
            })
            met_map[mid]['hits'].append(r)

        # ── Step 4: for each metabolite, choose the best peak ────────────────
        results = []
        for mid, info in met_map.items():
            best_hit = self._select_best_hit(info['hits'])
            best_ppm = best_hit.get('peak_ppm', best_hit.get('ppm', 0))

            overlap_count     = int(best_hit.get('overlap_count', 0))
            overlapping_names = list(best_hit.get('_peers_in_bucket', []))

            # If names did not come with the hit, try to reconstruct from the bucket
            if not overlapping_names and overlap_count > 0:
                key   = self._bucket(best_ppm, tol)
                peers = hits_by_peak.get(key, [])
                overlapping_names = [
                    p.get('name', '')
                    for p in peers
                    if (p.get('hmdb_id') or p.get('bmrb_id', '')) != mid
                ]

            corroborated, corroboration_peaks = self._calc_corroboration(
                mid, best_hit, info['hits'])

            uniqueness_score    = self._calc_uniqueness(overlap_count)
            intensity_score     = self._calc_intensity_score(best_ppm)
            corroboration_score = self.CORROBORATION_BONUS if corroborated else 0

            total = min(100, uniqueness_score + corroboration_score + intensity_score)

            results.append({
                'hmdb_id':             mid,
                'name':                info['name'],
                'best_hit':            best_hit,
                'best_peak_ppm':       round(best_ppm, 4),
                'overlap_count':       overlap_count,
                'overlapping_names':   overlapping_names,
                'corroborated':        corroborated,
                'corroboration_peaks': [round(p, 4) for p in corroboration_peaks],
                'peak_uniqueness':     uniqueness_score,
                'intensity_score':     intensity_score,
                'confidence_score':    total,
                'confidence_label':    self._label(total),
                'all_hits':            info['hits'],
            })

        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _bucket(ppm_val: float, tolerance: float) -> int:
        """Convert a ppm value into an integer bucket for grouping."""
        return int(round(ppm_val / tolerance))

    def _select_best_hit(self, hits: List[Dict]) -> Dict:
        """
        Choose the hit with the lowest overlap_count.
        Tie → highest intensity in the spectrum.
        """
        def _key(h):
            oc        = int(h.get('overlap_count', 0))
            ppm       = h.get('peak_ppm', h.get('ppm', 0))
            intensity = self._get_intensity_at_ppm(ppm)
            return (oc, -intensity)   # lowest overlap, highest intensity

        return min(hits, key=_key)

    def _calc_corroboration(
        self,
        mid: str,
        best_hit: Dict,
        all_hits: List[Dict],
    ) -> Tuple[bool, List[float]]:
        """
        Check whether the metabolite appears in another peak (different peak_rank
        from the chosen peak) with overlap_count == 0.
        """
        best_rank = best_hit.get('peak_rank')
        best_ppm  = best_hit.get('peak_ppm', best_hit.get('ppm', 0))

        other_unique = [
            h for h in all_hits
            if h.get('peak_rank') != best_rank
            and abs(h.get('peak_ppm', h.get('ppm', 0)) - best_ppm) > self.tolerance
            and int(h.get('overlap_count', 1)) == 0
        ]
        ppms = [h.get('peak_ppm', h.get('ppm', 0)) for h in other_unique]
        return (len(other_unique) > 0), ppms

    def _calc_uniqueness(self, overlap_count: int) -> int:
        """Score 0–50: 50 for a unique peak, decreases by 8 per competing metabolite."""
        return max(0, self.MAX_UNIQUENESS - overlap_count * self.OVERLAP_PENALTY)

    def _calc_intensity_score(self, peak_ppm: float) -> int:
        """Score 0–20 proportional to the relative intensity in the mean spectrum."""
        intensity = self._get_intensity_at_ppm(peak_ppm)
        if self.max_intensity <= 0:
            return 0
        ratio = intensity / self.max_intensity
        return int(round(self.MAX_INTENSITY * min(ratio, 1.0)))

    def _get_intensity_at_ppm(self, ppm_val: float,
                               half_win: float = 0.01) -> float:
        """Maximum intensity in a narrow window around ppm_val."""
        lo = ppm_val - half_win
        hi = ppm_val + half_win
        mask = (self.ppm >= lo) & (self.ppm <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.max(self.intensity[mask]))

    @staticmethod
    def _label(score: int) -> str:
        if score >= 70:
            return 'High'
        if score >= 40:
            return 'Medium'
        return 'Low'