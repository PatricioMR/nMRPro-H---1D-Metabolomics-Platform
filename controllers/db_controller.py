# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# controllers/db_controller.py
"""
Controller for database operations with support for multiple databases.
"""
import sqlite3
import os
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path


class DatabaseController:
    """Handles multiple database operations."""

    # Configurações dos bancos de dados disponíveis
    AVAILABLE_DATABASES = {
        'hmdb': {
            'name': 'HMDB (Human Metabolome Database)',
            'description': 'Complete human metabolome database with >220,000 metabolites',
            'url': 'https://hmdb.ca/downloads',
            'format': 'sqlite',
            'default_file': 'hmdb.db'
        },
        'jmorp_nmr': {
            'name': 'jMorp NMR Plasma (Japanese reference)',
            'description': '37 NMR-identified metabolites in plasma from >5000 healthy Japanese',
            'url': 'https://jmorp.megabank.tohoku.ac.jp/downloads/',
            'format': 'csv',
            'citation': 'Tadaka et al. Nucleic Acids Res. 2018 [citation:3]'
        },
        'jmorp_ms': {
            'name': 'jMorp MS Plasma (Japanese reference)',
            'description': '257 LC-MS characterized metabolites in plasma',
            'url': 'https://jmorp.megabank.tohoku.ac.jp/downloads/',
            'format': 'csv',
            'citation': 'Tadaka et al. Nucleic Acids Res. 2018 [citation:7]'
        },
        'srm1950': {
            'name': 'SRM 1950 (NIST Plasma Reference)',
            'description': '1,058 high-confidence metabolites in NIST plasma standard',
            'url': 'https://markerdb.ca',
            'format': 'json',
            'citation': 'Mandal et al. Anal. Chem. 2024 [citation:6]'
        }
    }

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.current_db = None
        self.current_db_path = None
        self.db_type = None


    def connect(self, db_path, db_type='hmdb'):
        """
        Connect to a database.

        Args:
            db_path: path to database file or directory
            db_type: type of database ('hmdb', 'jmorp_nmr', 'jmorp_ms', 'srm1950')
        """
        self.current_db_path = db_path
        self.db_type = db_type

        if db_type == 'hmdb':
            return self._connect_sqlite(db_path)
        elif db_type.startswith('jmorp'):
            return self._connect_jmorp(db_path, db_type)
        elif db_type == 'srm1950':
            return self._connect_srm1950(db_path)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def _connect_sqlite(self, db_path):
        """Connect to SQLite database (HMDB format)."""
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.current_db = 'hmdb'
        print(f"✅ Connected to HMDB SQLite database: {db_path}")
        return self.conn

    def _connect_jmorp(self, data_dir, db_type):
        """
        Connect to jMorp data directory (CSV files).
        Expected structure:
        - NMR_concentrations.csv
        - MS_peak_intensities.csv
        - metabolite_info.csv
        """
        self.data_dir = data_dir
        self.jmorp_data = {}

        # Load appropriate data based on type
        if db_type == 'jmorp_nmr':
            nmr_file = os.path.join(data_dir, 'NMR_concentrations.csv')
            if os.path.exists(nmr_file):
                self.jmorp_data['nmr'] = pd.read_csv(nmr_file)
                print(f"✅ Loaded jMorp NMR data: {len(self.jmorp_data['nmr'])} metabolites")

        if db_type == 'jmorp_ms' or db_type == 'jmorp':
            ms_file = os.path.join(data_dir, 'MS_peak_intensities.csv')
            if os.path.exists(ms_file):
                self.jmorp_data['ms'] = pd.read_csv(ms_file)
                print(f"✅ Loaded jMorp MS data: {len(self.jmorp_data['ms'])} metabolites")

        # Load metabolite info
        info_file = os.path.join(data_dir, 'metabolite_info.csv')
        if os.path.exists(info_file):
            self.jmorp_data['info'] = pd.read_csv(info_file)

        self.current_db = db_type
        return self.jmorp_data

    def _connect_srm1950(self, json_file):
        """Connect to SRM 1950 JSON data."""
        with open(json_file, 'r') as f:
            self.srm_data = json.load(f)
        print(f"✅ Loaded SRM 1950 data: {len(self.srm_data)} metabolites")
        self.current_db = 'srm1950'
        return self.srm_data

    def search_by_interval(self, ppm_start, ppm_end, tolerance=0.05):
        """
        Search for metabolites matching ppm interval.
        Dispatches to appropriate search method based on current database.
        """
        if self.current_db == 'hmdb':
            return self._search_hmdb(ppm_start, ppm_end, tolerance)
        elif self.current_db in ['jmorp_nmr', 'jmorp_ms', 'jmorp']:
            return self._search_jmorp(ppm_start, ppm_end, tolerance)
        elif self.current_db == 'srm1950':
            return self._search_srm1950(ppm_start, ppm_end, tolerance)
        else:
            raise ValueError(f"No search method for database: {self.current_db}")

    def _search_hmdb(self, ppm_start, ppm_end, tolerance):
        """Search HMDB SQLite database."""
        search_start = min(ppm_start, ppm_end) - tolerance
        search_end = max(ppm_start, ppm_end) + tolerance

        self.cursor.execute("""
            SELECT 
                m.*,
                met.name,
                (
                    SELECT COUNT(DISTINCT m2.hmdb_id) 
                    FROM multiplets m2 
                    WHERE m2.hmdb_id != m.hmdb_id 
                        AND ABS(m2.ppm_center - m.ppm_center) < 0.1
                ) as overlap_count
            FROM multiplets m
            JOIN metabolites met ON m.hmdb_id = met.hmdb_id
            WHERE m.ppm_start <= ? AND m.ppm_end >= ?
            ORDER BY ABS(m.ppm_center - ?)
            LIMIT 10
        """, (search_end, search_start, (ppm_start + ppm_end) / 2))

        results = [dict(row) for row in self.cursor.fetchall()]
        for r in results:
            r['match_type'] = 'multiplet'
            r['database'] = 'HMDB'

        return {'multiplets': results, 'peaks': []}

    def _search_jmorp(self, ppm_start, ppm_end, tolerance):
        """Search jMorp database."""
        results = []
        center = (ppm_start + ppm_end) / 2

        # Search in NMR data if available
        if 'nmr' in self.jmorp_data:
            df = self.jmorp_data['nmr']
            # Assume columns: name, ppm, concentration_mean, concentration_sd
            for _, row in df.iterrows():
                if abs(row['ppm'] - center) <= tolerance:
                    results.append({
                        'name': row['name'],
                        'ppm': row['ppm'],
                        'concentration': row.get('concentration_mean', 'N/A'),
                        'concentration_sd': row.get('concentration_sd', ''),
                        'match_type': 'nmr',
                        'database': 'jMorp NMR',
                        'overlap_count': 0,  # jMorp doesn't have overlap info
                        'source': 'Japanese healthy plasma reference [citation:3]'
                    })

        # Search in MS data if available
        if 'ms' in self.jmorp_data:
            df = self.jmorp_data['ms']
            # Assume columns: name, mz, rt, peak_intensity_mean
            for _, row in df.iterrows():
                # Convert m/z to approximate ppm (simplified)
                # This would need proper conversion based on your instrument
                if 'estimated_ppm' in row:
                    if abs(row['estimated_ppm'] - center) <= tolerance:
                        results.append({
                            'name': row['name'],
                            'ppm': row.get('estimated_ppm', 'N/A'),
                            'mz': row.get('mz', 'N/A'),
                            'peak_intensity': row.get('peak_intensity_mean', 'N/A'),
                            'match_type': 'ms',
                            'database': 'jMorp MS',
                            'overlap_count': 0,
                            'source': 'Japanese healthy plasma reference [citation:7]'
                        })

        return {'multiplets': results, 'peaks': []}

    def _search_srm1950(self, ppm_start, ppm_end, tolerance):
        """Search SRM 1950 reference data."""
        results = []
        center = (ppm_start + ppm_end) / 2

        for met in self.srm_data:
            if 'ppm' in met and abs(met['ppm'] - center) <= tolerance:
                met_copy = met.copy()
                met_copy['match_type'] = 'reference'
                met_copy['database'] = 'SRM 1950'
                met_copy['overlap_count'] = 0
                met_copy['source'] = 'NIST plasma reference standard [citation:6]'
                results.append(met_copy)

        return {'multiplets': results, 'peaks': []}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None