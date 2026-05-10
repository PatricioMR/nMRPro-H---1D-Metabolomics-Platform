import os, re, sqlite3

PEAK_DIR = r'C:\Users\migue\OneDrive\Documentos\hmdb_nmr_peak_lists'
DB_PATH = r'C:\Users\migue\PycharmProjects\NMRAnalysis_Gui\databases\HMDB_ValidationPipe.db'

# Ascorbate HMDB ID to try adding
ASCORBATE_HMDB = 'HMDB0000044'


def parse_peak_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse peaks (handles both "Table of Peaks" and "List of Peaks")
    peaks = {}
    peak_section = re.search(
        r'(?:Table|List) of Peaks\n.*?\n(.*?)(?:\n\n|\n(?:Table|List))', content, re.DOTALL)
    if peak_section:
        for line in peak_section.group(1).strip().split('\n'):
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    ppm = float(parts[1])
                    intensity = float(parts[3])
                    if ppm not in peaks or intensity > peaks[ppm]:
                        peaks[ppm] = intensity
                except:
                    pass

    # Parse multiplets
    multiplets = []
    seen_ppm = set()
    mult_section = re.search(
        r'(?:Table|List) of Multiplets\n(.*?)(?:\n\n|\n(?:Table|List)|$)', content, re.DOTALL)
    if mult_section:
        lines = mult_section.group(1).strip().split('\n')
        header = lines[0] if lines else ''
        has_j = 'J' in header
        for line in lines[1:]:
            parts = line.strip().split('\t')
            try:
                if has_j and len(parts) >= 7:
                    ppm = float(parts[1])
                    mult_type = parts[3].strip()
                    j_raw = parts[4].strip()
                    atom = parts[5].strip()
                elif not has_j and len(parts) >= 6:
                    ppm = float(parts[1])
                    mult_type = parts[3].strip()
                    j_raw = '-'
                    atom = parts[4].strip()
                else:
                    continue

                # deduplicate by ppm
                if ppm in seen_ppm:
                    continue
                seen_ppm.add(ppm)

                j_val = None
                if j_raw and j_raw != '-':
                    try:
                        j_val = float(j_raw.split()[0])
                    except:
                        pass

                intensity = None
                if peaks:
                    closest = min(peaks.keys(), key=lambda x: abs(x - ppm))
                    intensity = peaks[closest]

                multiplets.append({
                    'ppm': ppm,
                    'multiplicity': mult_type,
                    'j_coupling': j_val,
                    'atom_name': atom,
                    'intensity': intensity,
                })
            except:
                pass

    return multiplets


def find_file_by_spectrum(hmdb_id, spectrum_id):
    prefix = f'{hmdb_id}_nmroned_{spectrum_id}_'
    for f in os.listdir(PEAK_DIR):
        if f.startswith(prefix):
            return os.path.join(PEAK_DIR, f)
    return None


def find_file_by_hmdb(hmdb_id):
    """Return first nmroned file for this HMDB ID."""
    prefix = f'{hmdb_id}_nmroned_'
    for f in sorted(os.listdir(PEAK_DIR)):
        if f.startswith(prefix):
            # extract spectrum_id from filename
            parts = f.replace('.txt', '').split('_')
            spec_id = parts[2] if len(parts) >= 3 else None
            return os.path.join(PEAK_DIR, f), spec_id
    return None, None


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('SELECT hmdb_id, name, spectrum_id FROM metabolites')
metabolites = cur.fetchall()

for hmdb_id, name, spectrum_id in metabolites:
    filepath = find_file_by_spectrum(hmdb_id, spectrum_id)
    new_spec_id = spectrum_id

    if not filepath:
        # Try finding any file for this HMDB ID
        filepath, new_spec_id = find_file_by_hmdb(hmdb_id)
        if filepath:
            print(f'[WARN] {name}: spectrum_id {spectrum_id} not found, using {new_spec_id}')
        else:
            print(f'[SKIP] {name}: no file found')
            continue

    multiplets = parse_peak_file(filepath)
    if not multiplets:
        print(f'[SKIP] {name}: 0 multiplets parsed')
        continue

    print(f'[OK] {name}: {len(multiplets)} multiplets (spectrum={new_spec_id})')

    # Delete existing shifts
    cur.execute('DELETE FROM shifts_1h WHERE hmdb_id = ?', (hmdb_id,))

    # Update spectrum_id if changed
    if new_spec_id != spectrum_id:
        cur.execute('UPDATE metabolites SET spectrum_id = ? WHERE hmdb_id = ?',
                    (new_spec_id, hmdb_id))

    # Insert new shifts
    for m in multiplets:
        cur.execute(
            '''INSERT INTO shifts_1h (hmdb_id, ppm, multiplicity, intensity, mult_labeled,
               bmrb_id, atom_name, j_coupling)
               VALUES (?, ?, ?, ?, 0, ?, ?, ?)''',
            (hmdb_id, m['ppm'], m['multiplicity'], m['intensity'],
             hmdb_id, m['atom_name'], m['j_coupling'])
        )

# Try to add Ascorbate
print(f'\n--- Checking Ascorbate ({ASCORBATE_HMDB}) ---')
cur.execute('SELECT 1 FROM metabolites WHERE hmdb_id = ?', (ASCORBATE_HMDB,))
if cur.fetchone():
    print('Ascorbate already in DB')
else:
    filepath, spec_id = find_file_by_hmdb(ASCORBATE_HMDB)
    if not filepath:
        print('No file found for Ascorbate in peak list folder')
    else:
        multiplets = parse_peak_file(filepath)
        print(f'Ascorbate: {len(multiplets)} multiplets from spectrum {spec_id}')
        if multiplets:
            cur.execute(
                '''INSERT OR IGNORE INTO metabolites
                   (hmdb_id, name, spectrum_id, frequency_mhz, solvent, ph, temperature_c,
                    assessment, score, peak_count, formula, inchi, smiles, kegg_id, drugbank_id,
                    pubchem_id, bmrb_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL, ?, NULL)''',
                (ASCORBATE_HMDB, 'L-Ascorbic acid', spec_id, 500.0, 'Water', 7.0, 25.0,
                 len(multiplets), '5785')
            )
            for m in multiplets:
                cur.execute(
                    '''INSERT INTO shifts_1h (hmdb_id, ppm, multiplicity, intensity, mult_labeled,
                       bmrb_id, atom_name, j_coupling)
                       VALUES (?, ?, ?, ?, 0, ?, ?, ?)''',
                    (ASCORBATE_HMDB, m['ppm'], m['multiplicity'], m['intensity'],
                     ASCORBATE_HMDB, m['atom_name'], m['j_coupling'])
                )
            print('Ascorbate added.')

conn.commit()
conn.close()
print('\nDone.')
