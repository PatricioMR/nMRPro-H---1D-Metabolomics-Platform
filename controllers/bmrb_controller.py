# controllers/bmrb_controller.py
"""
BMRB API controller for searching and annotating deconvolved peaks.
"""
import requests
import json
import ssl
import certifi
import urllib3
from typing import List, Dict, Optional, Tuple
import time

# Suppress SSL warnings (optional)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BMRBController:
    """Handles BMRB API searches for metabolite identification."""

    def __init__(self, use_ssl_verification=False):
        """
        Initialize BMRB controller.

        Args:
            use_ssl_verification: Set to False if SSL cert issues persist
        """
        self.base_url = "https://bmrbpub.pdbj.org"
        self.search_url = f"{self.base_url}/search/rdf"
        self.metabolomics_url = f"{self.base_url}/xml/bms"
        self.verify_ssl = use_ssl_verification

        # Cache for results
        self.cache = {}

        # Setup SSL if verification is enabled
        if use_ssl_verification:
            try:
                self.ssl_context = ssl.create_default_context(cafile=certifi.where())
                self.session = requests.Session()
                self.session.verify = certifi.where()
            except:
                print("SSL setup failed, falling back to no verification")
                self.verify_ssl = False
                self.session = requests.Session()
                self.session.verify = False
        else:
            self.session = requests.Session()
            self.session.verify = False

    def search_by_peak(self, peak_ppm: float, tolerance: float = 0.05) -> List[Dict]:
        """
        Search BMRB for a single peak.

        Args:
            peak_ppm: Peak position in ppm
            tolerance: Search tolerance in ppm

        Returns:
            List of matching metabolites
        """
        cache_key = f"peak_{peak_ppm:.4f}_{tolerance}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # SPARQL query for peak search
        query = f"""
        PREFIX bmrb: <http://bmrbpub.pdbj.org/rdf/bms#>
        SELECT DISTINCT ?entry ?name ?shift ?atom ?entry_id
        WHERE {{
          ?entry a bmrb:Entry .
          ?entry bmrb:entry_id ?entry_id .
          ?entry bmrb:entity_name ?name .
          ?entry bmrb:chemical_shift ?cs .
          ?cs bmrb:chemical_shift_value ?shift .
          ?cs bmrb:atom_name ?atom .
          FILTER (?shift > {peak_ppm - tolerance} && ?shift < {peak_ppm + tolerance})
        }}
        ORDER BY ?entry
        LIMIT 20
        """

        try:
            response = self.session.post(
                self.search_url,
                data={"query": query, "format": "application/json"},
                timeout=15
            )

            if response.status_code == 200:
                results = self._parse_sparql_response(response.json())
                self.cache[cache_key] = results
                return results
            else:
                print(f"BMRB API error: {response.status_code}")
                return self._get_fallback_data(peak_ppm, tolerance)

        except Exception as e:
            print(f"BMRB search failed: {e}")
            return self._get_fallback_data(peak_ppm, tolerance)

    def search_by_peak_interval(self, peak_start: float, peak_end: float, tolerance: float = 0.05) -> List[Dict]:
        """
        Search BMRB for a peak interval (after deconvolution).

        Args:
            peak_start: Start of peak interval in ppm
            peak_end: End of peak interval in ppm
            tolerance: Additional search tolerance

        Returns:
            List of matching metabolites
        """
        center = (peak_start + peak_end) / 2
        half_width = (peak_end - peak_start) / 2
        search_tolerance = half_width + tolerance

        return self.search_by_peak(center, search_tolerance)

    def search_multiple_peaks(self, peaks: List[Dict], tolerance: float = 0.05) -> List[Dict]:
        """
        Search BMRB for multiple deconvolved peaks.

        Args:
            peaks: List of peak dictionaries with 'ppm' key
            tolerance: Search tolerance

        Returns:
            List of all matches with peak references
        """
        all_results = []

        for i, peak in enumerate(peaks):
            # Progress indication
            print(f"Searching peak {i + 1}/{len(peaks)}: {peak['ppm']:.4f} ppm")

            results = self.search_by_peak(peak['ppm'], tolerance)

            # Tag results with the originating peak
            for r in results:
                r['source_peak_ppm'] = peak['ppm']
                r['source_peak_rank'] = peak.get('rank', i + 1)
                r['source_peak_area'] = peak.get('area', 0)

            all_results.extend(results)

            # Small delay to avoid rate limiting
            time.sleep(0.2)

        return all_results

    def get_metabolite_details(self, entry_id: str) -> Optional[Dict]:
        """
        Fetch detailed information for a BMRB metabolomics entry.

        Args:
            entry_id: BMRB entry ID (e.g., 'bmse000400')

        Returns:
            Dictionary with entry details or None
        """
        # Try experimental data first
        url = f"{self.metabolomics_url}/{entry_id}-noatom.xml"

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                # Parse XML (simplified - you might want proper XML parsing)
                return {
                    'entry_id': entry_id,
                    'xml_data': response.text[:1000],  # Truncated for example
                    'source': 'experimental'
                }
        except:
            pass

        # Try theoretical data as fallback
        url = f"{self.metabolomics_url}/bmst{entry_id[4:]}-noatom.xml"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return {
                    'entry_id': entry_id,
                    'xml_data': response.text[:1000],
                    'source': 'theoretical'
                }
        except:
            pass

        return None

    def _parse_sparql_response(self, data) -> List[Dict]:
        """Parse SPARQL JSON response into list of dictionaries."""
        results = []

        if 'results' in data and 'bindings' in data['results']:
            for binding in data['results']['bindings']:
                result = {
                    'bmrb_id': binding.get('entry_id', {}).get('value', '').split('/')[-1],
                    'name': binding.get('name', {}).get('value', 'Unknown'),
                    'ppm': float(binding.get('shift', {}).get('value', 0)),
                    'atom': binding.get('atom', {}).get('value', ''),
                    'database': 'BMRB',
                    'match_type': 'bmrb_online',
                    'confidence': 'high' if 'entry_id' in binding else 'medium'
                }
                results.append(result)

        return results

    def _get_fallback_data(self, peak_ppm: float, tolerance: float) -> List[Dict]:
        """Return sample data when API fails (for testing)."""
        # Common plasma metabolites with their BMRB IDs
        sample_metabolites = [
            {'name': 'Lactate', 'ppm': 1.33, 'bmrb_id': 'bmse000400', 'atom': 'CH3'},
            {'name': 'Alanine', 'ppm': 1.48, 'bmrb_id': 'bmse000401', 'atom': 'CH3'},
            {'name': 'Glucose', 'ppm': 3.48, 'bmrb_id': 'bmse000402', 'atom': 'H1'},
            {'name': 'Creatine', 'ppm': 3.03, 'bmrb_id': 'bmse000403', 'atom': 'CH3'},
            {'name': 'Choline', 'ppm': 3.20, 'bmrb_id': 'bmse000404', 'atom': 'NCH3'},
            {'name': 'Glutamate', 'ppm': 2.35, 'bmrb_id': 'bmse000405', 'atom': 'HG'},
            {'name': 'Glutamine', 'ppm': 2.45, 'bmrb_id': 'bmse000406', 'atom': 'HG'},
            {'name': 'Valine', 'ppm': 0.98, 'bmrb_id': 'bmse000407', 'atom': 'CH3'},
            {'name': 'Isoleucine', 'ppm': 0.94, 'bmrb_id': 'bmse000408', 'atom': 'CH3'},
            {'name': 'Acetate', 'ppm': 1.91, 'bmrb_id': 'bmse000409', 'atom': 'CH3'},
        ]

        matches = []
        for met in sample_metabolites:
            if abs(met['ppm'] - peak_ppm) <= tolerance:
                match = met.copy()
                match['database'] = 'BMRB (sample)'
                match['match_type'] = 'bmrb_sample'
                matches.append(match)

        return matches