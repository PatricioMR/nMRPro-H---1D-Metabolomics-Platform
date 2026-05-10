"""
Updates a SQLite database (metabolites) with common names, kegg_id and drugbank_id from a mapping CSV.
Usage: python update_db_from_csv.py
"""

import sqlite3
import csv
import tkinter as tk
from tkinter import filedialog, messagebox

def add_columns_if_not_exist(conn, table, columns):
    """Check whether columns exist in the table; add them if not."""
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    existing = [row[1] for row in cur.fetchall()]
    for col in columns:
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
            print(f"Column '{col}' added to table {table}.")
    conn.commit()

def update_metabolites_from_csv(db_path, csv_path):
    """Read CSV and update the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure extra columns exist
    add_columns_if_not_exist(conn, 'metabolites', ['kegg_id', 'drugbank_id'])

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    updated = 0
    skipped = 0
    for row in rows:
        hmdb_id = row.get('hmdb_id')
        if not hmdb_id:
            continue
        name = row.get('name', '').strip()
        kegg_id = row.get('kegg_id', '').strip() or None
        drugbank_id = row.get('drugbank_id', '').strip() or None

        # Check whether the hmdb_id exists in the database
        cur.execute("SELECT name FROM metabolites WHERE hmdb_id = ?", (hmdb_id,))
        if cur.fetchone():
            cur.execute("""
                UPDATE metabolites
                SET name = ?,
                    kegg_id = COALESCE(kegg_id, ?),
                    drugbank_id = COALESCE(drugbank_id, ?)
                WHERE hmdb_id = ?
            """, (name, kegg_id, drugbank_id, hmdb_id))
            updated += 1
        else:
            skipped += 1

    conn.commit()
    conn.close()
    return updated, skipped

def main():
    root = tk.Tk()
    root.withdraw()

    db_path = filedialog.askopenfilename(
        title="Select the SQLite database (e.g. hmdb.db)",
        filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
    )
    if not db_path:
        messagebox.showinfo("Cancelled", "No database selected.")
        return

    csv_path = filedialog.askopenfilename(
        title="Select the CSV file with names and IDs",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not csv_path:
        messagebox.showinfo("Cancelled", "No CSV selected.")
        return

    try:
        updated, skipped = update_metabolites_from_csv(db_path, csv_path)
        messagebox.showinfo("Done",
            f"Updated: {updated} metabolites\n"
            f"HMDB IDs not found in database: {skipped}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

if __name__ == '__main__':
    main()