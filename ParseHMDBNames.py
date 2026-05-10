"""
Converts the HMDB XML file (hmdb_metabolites.xml) to a CSV.
Usage: python hmdb_xml_to_csv_gui.py
A window will open to select the XML file and the CSV output destination.
"""

import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import lxml.etree as ET

NS = {'hmdb': 'http://www.hmdb.ca'}

def parse_metabolite(elem):
    """Extract fields from a <metabolite> element."""
    data = {}

    # Identifiers
    data['hmdb_id'] = elem.findtext('hmdb:accession', '', namespaces=NS)
    data['kegg_id'] = elem.findtext('hmdb:kegg_id', '', namespaces=NS)
    data['drugbank_id'] = elem.findtext('hmdb:drugbank_id', '', namespaces=NS)
    data['pubchem_id'] = elem.findtext('hmdb:pubchem_id', '', namespaces=NS)

    # Basic information
    data['name'] = elem.findtext('hmdb:name', '', namespaces=NS)
    data['chemical_formula'] = elem.findtext('hmdb:chemical_formula', '', namespaces=NS)
    data['inchi'] = elem.findtext('hmdb:inchi', '', namespaces=NS)
    data['smiles'] = elem.findtext('hmdb:smiles', '', namespaces=NS)
    data['description'] = elem.findtext('hmdb:description', '', namespaces=NS)

    # Synonyms (list)
    synonyms = []
    syn_elts = elem.findall('hmdb:synonyms/hmdb:synonym', namespaces=NS)
    for s in syn_elts:
        if s.text:
            synonyms.append(s.text.strip())
    data['synonyms'] = '|'.join(synonyms)  # pipe-separated

    return data

def convert_xml_to_csv(xml_path, csv_path):
    """Convert the XML to CSV."""
    fieldnames = [
        'hmdb_id', 'kegg_id', 'drugbank_id', 'pubchem_id',
        'name', 'chemical_formula', 'inchi', 'smiles', 'description',
        'synonyms'
    ]

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Use iterparse to read the XML efficiently
            context = ET.iterparse(xml_path, tag='{http://www.hmdb.ca}metabolite')
            count = 0
            for event, elem in context:
                data = parse_metabolite(elem)
                writer.writerow(data)
                count += 1
                # Free memory of the processed element
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
        return count
    except Exception as e:
        raise RuntimeError(f"Error during conversion: {e}")

def main():
    # Initialise the tkinter window and hide it
    root = tk.Tk()
    root.withdraw()

    # Select input XML file
    xml_path = filedialog.askopenfilename(
        title="Select the HMDB XML file",
        filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
    )
    if not xml_path:
        messagebox.showinfo("Cancelled", "No file selected.")
        return

    # Select output CSV file
    csv_path = filedialog.asksaveasfilename(
        title="Save CSV as",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not csv_path:
        messagebox.showinfo("Cancelled", "No output location selected.")
        return

    # Confirm with the user
    if not messagebox.askyesno("Confirm", f"Convert\n{xml_path}\nto\n{csv_path}?"):
        return

    try:
        count = convert_xml_to_csv(xml_path, csv_path)
        messagebox.showinfo("Done", f"Conversion complete.\n{count} metabolites saved to:\n{csv_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    main()