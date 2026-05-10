# data/hmdb_local.py
import sqlite3
import os


class HMDBLocalDB:
    """
    Local SQLite database wrapper.
    Compatible with any .db file that follows the schema:
        metabolites (hmdb_id, name)
        multiplets  (id, hmdb_id, ppm_center, protons, type, j_coupling, ppm_start, ppm_end)
        peaks       (id, hmdb_id, ppm, height)
    """

    # Default path — override by passing db_path to __init__ or by setting
    # the environment variable NMR_DB_PATH.
    DEFAULT_PATH = os.environ.get(
        "NMR_DB_PATH",
        os.path.join(os.path.expanduser("~"), "OneDrive", "Documentos", "hmdb.db")
    )

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or self.DEFAULT_PATH
        self.conn = None
        print(f"📂 Database configured: {self.db_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"❌ Database not found: {self.db_path}")
        size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        print(f"✅ Database found — {size_mb:.2f} MB")

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def search_by_interval(self, ppm_start: float, ppm_end: float,
                            tolerance: float = 0.05) -> dict:
        """
        Search metabolites whose multiplets/peaks intersect the given interval.

        Args:
            ppm_start: start of the peak interval (ppm)
            ppm_end:   end of the peak interval (ppm)
            tolerance: additional tolerance (ppm)

        Returns:
            {'multiplets': [...], 'peaks': [...]}
        """
        self.connect()
        cursor = self.conn.cursor()

        search_start = min(ppm_start, ppm_end) - tolerance
        search_end   = max(ppm_start, ppm_end) + tolerance

        cursor.execute("""
            SELECT m.*, met.name,
                   (SELECT COUNT(*) FROM multiplets m2
                    WHERE m2.hmdb_id != m.hmdb_id
                      AND m2.ppm_start <= ? AND m2.ppm_end >= ?
                   ) as overlap_count
            FROM multiplets m
            JOIN metabolites met ON m.hmdb_id = met.hmdb_id
            WHERE m.ppm_start <= ? AND m.ppm_end >= ?
            ORDER BY m.ppm_center
        """, (search_end, search_start, search_end, search_start))
        multiplets = [dict(r) for r in cursor.fetchall()]

        cursor.execute("""
            SELECT p.*, met.name,
                   (SELECT COUNT(*) FROM peaks p2
                    WHERE p2.hmdb_id != p.hmdb_id
                      AND ABS(p2.ppm - p.ppm) <= ?
                   ) as overlap_count
            FROM peaks p
            JOIN metabolites met ON p.hmdb_id = met.hmdb_id
            WHERE p.ppm BETWEEN ? AND ?
            ORDER BY p.ppm
        """, (tolerance, search_start, search_end))
        peaks = [dict(r) for r in cursor.fetchall()]

        return {'multiplets': multiplets, 'peaks': peaks}
