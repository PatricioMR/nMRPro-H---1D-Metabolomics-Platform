"""
Classifica compostos do HMDB_Total_v01.db como fármaco ou metabolito
usando ATC codes do PubChem. Salva resultados em databases/drug_check_results.csv.
"""
import sqlite3
import requests
import time
import csv
import json
import os
from pathlib import Path

DB_PATH = Path("databases/HMDB_Total_v01.db")
RESULTS_PATH = Path("databases/drug_check_results.csv")
PROGRESS_PATH = Path("databases/drug_check_progress.json")

ENDOGENOUS_ATCS = {
    # Soluções IV de nutrientes — são metabólitos usados medicinalmente
    'V06', 'B05', 'A12', 'A11',
}

session = requests.Session()
session.headers.update({
    'User-Agent': 'NMRAnalysis/1.0 (research; contact: migueelpatricio97@gmail.com)'
})


def is_endogenous_atc(atc_code: str) -> bool:
    """Retorna True se o código ATC é apenas de nutrição/suplemento (não fármaco)."""
    return any(atc_code.upper().startswith(prefix) for prefix in ENDOGENOUS_ATCS)


def get_cid_by_name(name: str) -> int | None:
    try:
        r = session.get(
            f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON',
            timeout=15
        )
        if not r.ok:
            return None
        cids = r.json().get('IdentifierList', {}).get('CID', [])
        return cids[0] if cids else None
    except Exception:
        return None


def get_atc_codes(cid: int) -> list[str]:
    try:
        r = session.get(
            f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=ATC+Code',
            timeout=15
        )
        if not r.ok or len(r.text) < 50:
            return []
        data = r.json()
        codes = []
        # Navigate: Record > Section[] > Section[] > Information[] > Value > StringWithMarkup
        sections = data.get('Record', {}).get('Section', [])
        for s in sections:
            for ss in s.get('Section', []):
                for info in ss.get('Information', []):
                    val = info.get('Value', {})
                    for swm in val.get('StringWithMarkup', []):
                        txt = swm.get('String', '')
                        if txt:
                            codes.append(txt)
        return codes
    except Exception:
        return []


def classify(name: str, hmdb_id: str) -> dict:
    result = {
        'hmdb_id': hmdb_id,
        'name': name,
        'pubchem_cid': None,
        'atc_codes': '',
        'is_drug': False,
        'reason': '',
    }

    cid = get_cid_by_name(name)
    time.sleep(0.25)

    if cid is None:
        result['reason'] = 'not_found_pubchem'
        return result

    result['pubchem_cid'] = cid
    atc_codes = get_atc_codes(cid)
    time.sleep(0.25)

    if not atc_codes:
        result['reason'] = 'no_atc'
        return result

    result['atc_codes'] = '; '.join(atc_codes)

    # Filtra ATC de nutrição pura — esses são metabólitos com uso IV
    drug_atcs = [c for c in atc_codes if not is_endogenous_atc(c)]

    if drug_atcs:
        result['is_drug'] = True
        result['reason'] = f'ATC: {"; ".join(drug_atcs)}'
    else:
        result['reason'] = f'ATC apenas nutrição: {"; ".join(atc_codes)}'

    return result


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def save_progress(progress: dict):
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT hmdb_id, name FROM metabolites ORDER BY hmdb_id')
    compounds = cur.fetchall()
    conn.close()

    print(f"Total compostos: {len(compounds)}")

    progress = load_progress()
    results = []

    # Carrega resultados já salvos
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                progress[row['hmdb_id']] = True

    done = set(progress.keys())
    remaining = [(hid, name) for hid, name in compounds if hid not in done]
    print(f"Já processados: {len(done)}, restantes: {len(remaining)}")

    with open(RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['hmdb_id', 'name', 'pubchem_cid', 'atc_codes', 'is_drug', 'reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done:
            writer.writeheader()

        for i, (hmdb_id, name) in enumerate(remaining):
            print(f"[{i+1}/{len(remaining)}] {name} ({hmdb_id})...", end=' ', flush=True)
            result = classify(name, hmdb_id)
            tag = 'FARMACO' if result['is_drug'] else 'metabolito'
            print(f"{tag} | {result['reason'][:60]}")

            writer.writerow(result)
            f.flush()
            progress[hmdb_id] = True
            save_progress(progress)

    # Sumário
    all_results = []
    with open(RESULTS_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_results = list(reader)

    drugs = [r for r in all_results if r['is_drug'] == 'True']
    not_found = [r for r in all_results if r['reason'] == 'not_found_pubchem']
    print(f"\n=== SUMÁRIO ===")
    print(f"Total: {len(all_results)}")
    print(f"Fármacos identificados: {len(drugs)}")
    print(f"Não encontrados no PubChem: {len(not_found)}")
    print(f"Metabólitos (manter): {len(all_results) - len(drugs)}")
    print(f"\nResultados em: {RESULTS_PATH}")

    if drugs:
        print(f"\nFármacos a remover:")
        for d in drugs:
            print(f"  - {d['name']} ({d['hmdb_id']}) | {d['reason'][:70]}")


if __name__ == '__main__':
    main()
