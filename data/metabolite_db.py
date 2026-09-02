# data/metabolite_db.py
import csv
import json
import os
import numpy as np


class MetaboliteDatabase:
    """Manage local metabolite database and queries."""

    def __init__(self, db_path=None):
        """
        Initialize metabolite database.

        Args:
            db_path: path to CSV file with metabolite data
        """
        if db_path is None:
            # Default path relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, 'data', 'metabolites.csv')

        self.db_path = db_path
        self.metabolites = []
        self.load_local_db()

    def load_local_db(self):
        """Load local metabolite database from CSV."""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.metabolites = list(reader)

                # Convert ppm strings to float
                for m in self.metabolites:
                    if 'ppm' in m:
                        try:
                            m['ppm_float'] = float(m['ppm'])
                        except ValueError:
                            m['ppm_float'] = None

            print(f"Loaded {len(self.metabolites)} metabolites from {self.db_path}")

        except FileNotFoundError:
            print(f"Database not found at {self.db_path}. Creating default database.")
            self.create_default_db()
        except Exception as e:
            print(f"Error loading database: {e}")
            self.create_default_db()

    def create_default_db(self):
        """Create default metabolite database with common metabolites."""
        self.metabolites = [
            {'name': 'Lactate', 'ppm': '1.33', 'formula': 'C3H6O3', 'hmdb_id': 'HMDB0000190', 'group': 'Energy'},
            {'name': 'Alanine', 'ppm': '1.48', 'formula': 'C3H7NO2', 'hmdb_id': 'HMDB0000161', 'group': 'Amino acid'},
            {'name': 'NAA', 'ppm': '2.02', 'formula': 'C6H9NO5', 'hmdb_id': 'HMDB0000812', 'group': 'Neuronal'},
            {'name': 'Glutamate', 'ppm': '2.35', 'formula': 'C5H9NO4', 'hmdb_id': 'HMDB0000148',
             'group': 'Neurotransmitter'},
            {'name': 'Glutamine', 'ppm': '2.45', 'formula': 'C5H10N2O3', 'hmdb_id': 'HMDB0000641',
             'group': 'Amino acid'},
            {'name': 'Creatine', 'ppm': '3.03', 'formula': 'C4H9N3O2', 'hmdb_id': 'HMDB0000064', 'group': 'Energy'},
            {'name': 'Choline', 'ppm': '3.20', 'formula': 'C5H14NO', 'hmdb_id': 'HMDB0000097', 'group': 'Membrane'},
            {'name': 'Glucose', 'ppm': '3.48', 'formula': 'C6H12O6', 'hmdb_id': 'HMDB0000122', 'group': 'Carbohydrate'},
            {'name': 'Myo-inositol', 'ppm': '3.55', 'formula': 'C6H12O6', 'hmdb_id': 'HMDB0000211',
             'group': 'Osmolyte'},
            {'name': 'Taurine', 'ppm': '3.42', 'formula': 'C2H7NO3S', 'hmdb_id': 'HMDB0000251', 'group': 'Amino acid'},
            {'name': 'GABA', 'ppm': '2.28', 'formula': 'C4H9NO2', 'hmdb_id': 'HMDB0000112',
             'group': 'Neurotransmitter'},
            {'name': 'Acetate', 'ppm': '1.91', 'formula': 'C2H4O2', 'hmdb_id': 'HMDB0000042', 'group': 'Organic acid'},
            {'name': 'Formate', 'ppm': '8.45', 'formula': 'CH2O2', 'hmdb_id': 'HMDB0000142', 'group': 'Organic acid'},
        ]

        # Convert ppm to float
        for m in self.metabolites:
            m['ppm_float'] = float(m['ppm'])

        self.save_local_db()
        print(f"Created default database with {len(self.metabolites)} metabolites.")

    def save_local_db(self):
        """Save metabolite database to CSV."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with open(self.db_path, 'w', newline='', encoding='utf-8') as f:
                if self.metabolites:
                    fieldnames = self.metabolites[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.metabolites)

            print(f"Database saved to {self.db_path}")

        except Exception as e:
            print(f"Error saving database: {e}")



