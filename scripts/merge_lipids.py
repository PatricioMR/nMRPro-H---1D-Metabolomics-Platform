"""
Funde grupos de ácidos gordos indistinguíveis por RMN-1H no HMDB_Total_v01.db.
- Cria analitos fundidos MCFA (C6-C11) e LCFA (C12-C22)
- Remove cis-Vaccenic acid (duplicado exato do Vaccenic acid)
- Remove os 13 constituintes dos grupos fundidos
"""
import sqlite3
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DB = Path('databases/HMDB_Total_v01.db')
BACKUP = Path('databases/HMDB_Total_v01_pre_merge.db')

# ── Grupos a fundir ────────────────────────────────────────────────────────────
GROUPS = [
    {
        'id':      'MERGED_FA_MCFA',
        'name':    'Fatty Acids (MCFA, C6-C11)',
        'formula': None,
        'members': [
            'HMDB0000535',  # Caproic acid    C6
            'HMDB0000482',  # Caprylic acid   C8
            'HMDB0000666',  # Heptanoic acid  C7
            'HMDB0000847',  # Pelargonic acid C9
            'HMDB0000947',  # Undecanoic acid C11
        ],
        # Shifts representativos (média dos grupos coincidentes)
        'shifts': [
            (0.855, 'CH3',  'triplet'),
            (1.535, 'CH2',  'multiplet'),
            (2.158, 'aCH2', 'triplet'),
        ],
    },
    {
        'id':      'MERGED_FA_LCFA',
        'name':    'Fatty Acids (LCFA, C12-C22)',
        'formula': None,
        'members': [
            'HMDB0000638',  # Dodecanoic (Lauric) C12
            'HMDB0000806',  # Myristic acid       C14
            'HMDB0000220',  # Palmitic acid       C16
            'HMDB0000944',  # Behenic acid        C22
            'HMDB0000826',  # Pentadecanoic acid  C15
            'HMDB0000772',  # Nonadecanoic acid   C19
            'HMDB0002212',  # Arachidic acid      C20
            'HMDB0000573',  # Elaidic acid        C18:1t
        ],
        'shifts': [
            (0.888, 'CH3',  'triplet'),
            (1.640, 'CH2',  'multiplet'),
            (2.355, 'aCH2', 'triplet'),
        ],
    },
]

# Duplicado exato — apenas apagar
DUPLICATE_TO_DELETE = 'HMDB0240219'  # cis-Vaccenic acid (= Vaccenic acid)


def get_sample_avg(cur, bmrb_ids: list[str]) -> dict:
    """Calcula médias de pH/temp/field dos membros para atribuir ao analito fundido."""
    ph_vals, temp_vals, field_vals, solvents = [], [], [], []
    for bid in bmrb_ids:
        cur.execute('SELECT ph, temperature, solvent, field_strength FROM samples WHERE bmrb_id=?', (bid,))
        row = cur.fetchone()
        if row:
            if row[0]: ph_vals.append(row[0])
            if row[1]: temp_vals.append(row[1])
            if row[3]: field_vals.append(row[3])
            if row[2]: solvents.append(row[2])
    return {
        'ph':           round(sum(ph_vals) / len(ph_vals), 2) if ph_vals else None,
        'temperature':  round(sum(temp_vals) / len(temp_vals), 1) if temp_vals else None,
        'field_strength': int(round(sum(field_vals) / len(field_vals))) if field_vals else None,
        'solvent':      max(set(solvents), key=solvents.count) if solvents else None,
    }


def delete_cascade(cur, bmrb_id: str, name: str):
    for tbl in ('shifts_1h', 'transitions', 'samples'):
        cur.execute(f'DELETE FROM {tbl} WHERE bmrb_id=?', (bmrb_id,))
        n = cur.rowcount
        if n:
            print(f'    {tbl}: -{n}')
    cur.execute('DELETE FROM metabolites WHERE bmrb_id=?', (bmrb_id,))
    print(f'  ✓ Removido: {name} ({bmrb_id})')


def main():
    # Backup
    shutil.copy2(DB, BACKUP)
    print(f'Backup: {BACKUP}')

    conn = sqlite3.connect(DB)
    conn.execute('PRAGMA foreign_keys = ON')
    cur = conn.cursor()

    # Contagens antes
    for tbl in ('metabolites', 'shifts_1h', 'transitions', 'samples'):
        cur.execute(f'SELECT COUNT(*) FROM {tbl}')
        print(f'{tbl} antes: {cur.fetchone()[0]}')
    print()

    for group in GROUPS:
        merged_id = group['id']
        name      = group['name']
        members   = group['members']

        # Resolve bmrb_ids a partir dos hmdb_ids
        bmrb_ids = []
        member_names = []
        for hmdb_id in members:
            cur.execute('SELECT bmrb_id, name FROM metabolites WHERE hmdb_id=?', (hmdb_id,))
            row = cur.fetchone()
            if row:
                bmrb_ids.append(row[0])
                member_names.append(row[1])
            else:
                print(f'  AVISO: {hmdb_id} não encontrado no DB')

        print(f'=== Criar: {name} ({len(bmrb_ids)} membros) ===')
        for n in member_names:
            print(f'  - {n}')

        # INSERT metabolite fundido
        sample = get_sample_avg(cur, bmrb_ids)
        cur.execute(
            'INSERT OR REPLACE INTO metabolites (bmrb_id, name, formula, hmdb_id) VALUES (?,?,?,?)',
            (merged_id, name, group['formula'], merged_id)
        )
        print(f'  → INSERT metabolites: {merged_id}')

        # INSERT shifts_1h
        for ppm, atom, mult in group['shifts']:
            cur.execute(
                'INSERT INTO shifts_1h (bmrb_id, atom_name, ppm, multiplicity) VALUES (?,?,?,?)',
                (merged_id, atom, ppm, mult)
            )
        print(f'  → INSERT shifts_1h: {len(group["shifts"])} peaks')

        # INSERT samples (médias do grupo)
        cur.execute(
            'INSERT OR REPLACE INTO samples (bmrb_id, ph, temperature, solvent, field_strength) VALUES (?,?,?,?,?)',
            (merged_id, sample['ph'], sample['temperature'], sample['solvent'], sample['field_strength'])
        )
        print(f'  → INSERT samples: pH={sample["ph"]} T={sample["temperature"]} field={sample["field_strength"]}')

        # DELETE membros em cascata
        print(f'  Removendo membros:')
        for bmrb_id, mname in zip(bmrb_ids, member_names):
            delete_cascade(cur, bmrb_id, mname)
        print()

    # Apagar duplicado exato cis-Vaccenic
    print('=== Remover duplicado: cis-Vaccenic acid ===')
    cur.execute('SELECT bmrb_id, name FROM metabolites WHERE hmdb_id=?', (DUPLICATE_TO_DELETE,))
    row = cur.fetchone()
    if row:
        delete_cascade(cur, row[0], row[1])
    else:
        print(f'  AVISO: {DUPLICATE_TO_DELETE} não encontrado')
    print()

    conn.commit()

    # Contagens depois
    print('── Contagens finais ──')
    for tbl in ('metabolites', 'shifts_1h', 'transitions', 'samples'):
        cur.execute(f'SELECT COUNT(*) FROM {tbl}')
        print(f'{tbl} depois: {cur.fetchone()[0]}')

    # Verificação: analitos fundidos presentes
    print()
    print('── Verificação analitos fundidos ──')
    for group in GROUPS:
        cur.execute('SELECT name FROM metabolites WHERE bmrb_id=?', (group['id'],))
        m = cur.fetchone()
        cur.execute('SELECT ppm, atom_name FROM shifts_1h WHERE bmrb_id=? ORDER BY ppm', (group['id'],))
        shifts = cur.fetchall()
        print(f'  {m[0] if m else "NÃO ENCONTRADO"}: {shifts}')

    conn.close()
    print('\nConcluído.')


if __name__ == '__main__':
    main()
