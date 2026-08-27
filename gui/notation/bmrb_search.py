# gui/notation/bmrb_search.py
"""
BMRB API search module.
"""
try:
    import requests
    import urllib3
except ImportError:
    requests = None
    urllib3 = None
import time
from typing import List, Dict

# Suppress SSL warnings
if urllib3 is not None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BMRBSearcher:
    """Handles BMRB API searches."""

    def __init__(self):
        self.base_url = "https://bmrbpub.pdbj.org"
        self.search_url = f"{self.base_url}/search/rdf"
        self.cache = {}
        print("✅ BMRBSearcher initialized")

    def search_by_peak(self, peak_ppm: float, tolerance: float = 0.05) -> List[Dict]:
        """Search BMRB for a single peak."""
        cache_key = f"{peak_ppm:.4f}_{tolerance}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = f"""
        PREFIX bmrb: <http://bmrbpub.pdbj.org/rdf/bms#>
        SELECT DISTINCT ?entry ?name ?shift ?atom
        WHERE {{
          ?entry a bmrb:Entry .
          ?entry bmrb:entity_name ?name .
          ?entry bmrb:chemical_shift ?cs .
          ?cs bmrb:chemical_shift_value ?shift .
          ?cs bmrb:atom_name ?atom .
          FILTER (?shift > {peak_ppm - tolerance} && ?shift < {peak_ppm + tolerance})
        }}
        LIMIT 10
        """

        try:
            if requests is None:
                return self._get_sample_data(peak_ppm, tolerance)
            response = requests.post(
                self.search_url,
                data={"query": query, "format": "application/json"},
                timeout=15,
                verify=False
            )

            if response.status_code == 200:
                results = self._parse_response(response.json())
                self.cache[cache_key] = results
                return results
            else:
                print(f"BMRB API error: {response.status_code}")
        except Exception as e:
            print(f"BMRB search failed: {e}")

        return self._get_sample_data(peak_ppm, tolerance)

    def _parse_response(self, data) -> List[Dict]:
        """Parse SPARQL JSON response."""
        results = []
        if 'results' in data and 'bindings' in data['results']:
            for binding in data['results']['bindings']:
                results.append({
                    'bmrb_id': binding.get('entry', {}).get('value', '').split('/')[-1],
                    'name': binding.get('name', {}).get('value', 'Unknown'),
                    'ppm': float(binding.get('shift', {}).get('value', 0)),
                    'atom': binding.get('atom', {}).get('value', ''),
                    'database': 'BMRB'
                })
        return results

    def _get_sample_data(self, peak_ppm, tolerance):
        """Return sample data for testing."""
        samples = [
            {'name': 'Lactate', 'ppm': 1.33, 'bmrb_id': 'bmse000400'},
            {'name': 'Alanine', 'ppm': 1.48, 'bmrb_id': 'bmse000401'},
            {'name': 'Glucose', 'ppm': 3.48, 'bmrb_id': 'bmse000402'},
            {'name': 'Creatine', 'ppm': 3.03, 'bmrb_id': 'bmse000403'},
            {'name': 'Choline', 'ppm': 3.20, 'bmrb_id': 'bmse000404'},
        ]

        matches = []
        for s in samples:
            if abs(s['ppm'] - peak_ppm) <= tolerance:
                s_copy = s.copy()
                s_copy['database'] = 'BMRB (sample)'
                matches.append(s_copy)

        return matches

    def search_multiple_peaks(self, peaks: List[Dict], tolerance: float = 0.05) -> List[Dict]:
        """Search BMRB for multiple peaks."""
        all_results = []
        for i, peak in enumerate(peaks):
            print(f"Searching peak {i + 1}/{len(peaks)}: {peak['ppm']:.4f} ppm")
            results = self.search_by_peak(peak['ppm'], tolerance)
            for r in results:
                r['peak_ppm'] = peak['ppm']
                r['peak_rank'] = peak.get('rank', i + 1)
            all_results.extend(results)
            time.sleep(0.2)
        return all_results