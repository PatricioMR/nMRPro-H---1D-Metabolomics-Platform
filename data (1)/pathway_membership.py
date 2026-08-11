# data/pathway_membership.py
"""
Metabolite → pathway membership lookup.

Data source priority:
  1. Local SQLite DB (databases/hmdb_pathways.db) — instant, offline.
     Build once with: python data/build_pathway_db.py
  2. HMDB REST API fallback — for IDs not in local DB (rate-limited).

Read-only at runtime (parameterised queries only). Extracted from the old
gui/widgets/pathway_window.py so it can be reused by the Statistics stage
Pathway mode without a Toplevel popup.
"""
import re
import json
import time
import sqlite3
from pathlib import Path

from app_paths import databases_dir

# ── Constants ─────────────────────────────────────────────────────────────────

_LOCAL_DB  = Path(databases_dir()) / 'hmdb_pathways.db'
HMDB_URL   = 'https://hmdb.ca/metabolites/{}.json'
SMPDB_URL  = 'https://smpdb.ca/view/{}'
RATE_DELAY = 0.35
TIMEOUT    = 8

# Background size = all logged IDs in the local pathway DB (used by ORA).
N_BACKGROUND = 380

# Whole-word / substring disease and drug markers.
_DISEASE_KW = {'deficiency', 'syndrome', 'disease', 'disorder', 'defect',
               'type i', 'type ii', 'type iii', 'intolerance',
               'oncometabolite', 'inborn', 'tumour', 'tumor', 'cancer',
               'carcinoma'}
# Morphological suffixes of inborn-errors-of-metabolism (…-emia, …-uria, …-osis,
# …-dystonia/dystrophy). Excludes normal metabolic tokens with the same endings.
_DISEASE_SUFFIX = ('emia', 'aemia', 'uria', 'osis', 'dystonia', 'dystrophy')
_DISEASE_SUFFIX_SAFE = {
    'biosynthesis', 'glycolysis', 'metabolism', 'oxidation', 'homeostasis',
    'synthesis', 'catabolism', 'lipolysis', 'proteolysis', 'mitosis',
    'exocytosis', 'endocytosis', 'hydrolysis', 'gluconeogenesis',
    'ketogenesis', 'lipogenesis', 'phosphorylation',
}
_DRUG_KW    = {'action pathway', 'metabolism pathway', 'antihistamine',
               'pharmacokinetics', 'agonist', 'antagonist', 'antidepressant',
               'antipsychotic', 'statin', 'antibiotic', 'antifungal',
               'antiviral', 'inhibitor action', 'receptor action',
               'drug metabolism', 'xenobiotic', 'pharmaceutical',
               'pharmacodynamic', 'pharmacological', 'medication',
               'biotransformation', 'bioavailability'}


# ── ID helpers ────────────────────────────────────────────────────────────────

def norm_hmdb(raw: str):
    m = re.search(r'\d+', raw or '')
    return f'HMDB{int(m.group()):07d}' if m else None


def is_hmdb(raw: str) -> bool:
    return bool(raw) and str(raw).upper().startswith('HMDB')


def _name_is_disease(nd: str) -> bool:
    if any(k in nd for k in _DISEASE_KW):
        return True
    for tok in re.findall(r'[a-z]+', nd):
        if tok.endswith(_DISEASE_SUFFIX) and tok not in _DISEASE_SUFFIX_SAFE:
            return True
    return False


def _name_is_drug(nd: str, raw_name: str) -> bool:
    if raw_name.strip().endswith('.'):   # SMPDB drug entries like "Bivalirudin."
        return True
    return any(k in nd for k in _DRUG_KW)


def pathway_category(name: str, description: str = '', stored_cat: str = '') -> str:
    """
    Classify a pathway as metabolic / drug / disease.

    The local DB's stored 'metabolic' tag is unreliable — SMPDB files many
    inborn-error-of-metabolism (disease) and drug-metabolism pathways under
    'metabolic'. So name/description heuristics override a stored 'metabolic'
    (and empty) tag, while an explicit stored 'drug'/'disease' is trusted.
    """
    if stored_cat in ('drug', 'disease'):
        return stored_cat
    nd = (name + ' ' + description).lower()
    if _name_is_disease(nd):
        return 'disease'
    if _name_is_drug(nd, name):
        return 'drug'
    return 'metabolic'


def local_db_available() -> bool:
    return _LOCAL_DB.exists() and _LOCAL_DB.stat().st_size > 4096


# ── Local DB ──────────────────────────────────────────────────────────────────

def _migrate_local_db(con: sqlite3.Connection):
    """Add category column if missing, then backfill using keyword heuristic."""
    cols = {r[1] for r in con.execute('PRAGMA table_info(pathways)').fetchall()}
    if 'category' not in cols:
        con.execute("ALTER TABLE pathways ADD COLUMN category TEXT DEFAULT ''")
        con.commit()
    rows = con.execute(
        "SELECT id, name, description FROM pathways "
        "WHERE category = '' OR category IS NULL").fetchall()
    if rows:
        updates = []
        for pid, name, desc in rows:
            updates.append((pathway_category(name, desc or ''), pid))
        con.executemany("UPDATE pathways SET category = ? WHERE id = ?", updates)
        con.commit()


def query_local(hmdb_id: str):
    """
    Query local DB for pathways of hmdb_id.
    Returns list of {name, smpdb_id, description, category}, or None if the ID
    is not in the local DB (caller should fall back to the API).
    """
    if not local_db_available():
        return None
    try:
        con = sqlite3.connect(str(_LOCAL_DB))
        _migrate_local_db(con)
        con.row_factory = sqlite3.Row
        cur = con.execute('SELECT 1 FROM fetch_log WHERE hmdb_id = ?', (hmdb_id,))
        if cur.fetchone() is None:
            con.close()
            return None
        try:
            cur = con.execute(
                '''SELECT p.name, p.smpdb_id, p.description, p.category
                   FROM pathways p
                   JOIN metabolite_pathways mp ON mp.pathway_id = p.id
                   WHERE mp.hmdb_id = ?''', (hmdb_id,))
        except Exception:
            cur = con.execute(
                '''SELECT p.name, p.smpdb_id, p.description, '' AS category
                   FROM pathways p
                   JOIN metabolite_pathways mp ON mp.pathway_id = p.id
                   WHERE mp.hmdb_id = ?''', (hmdb_id,))
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return None


def pathway_background_sizes(pw_names):
    """
    Return {pathway_name: distinct metabolite count in background} from the
    local DB. Empty dict when no local DB. Used by ORA for the contingency
    table (K = pathway size within the background).
    """
    sizes = {}
    if not local_db_available():
        return sizes
    try:
        con = sqlite3.connect(str(_LOCAL_DB))
        for pw_name in pw_names:
            cur = con.execute(
                '''SELECT COUNT(DISTINCT mp.hmdb_id)
                   FROM metabolite_pathways mp
                   JOIN pathways p ON p.id = mp.pathway_id
                   WHERE p.name = ?''', (pw_name,))
            sizes[pw_name] = cur.fetchone()[0] or 0
        con.close()
    except Exception:
        pass
    return sizes


# ── HMDB API fallback ─────────────────────────────────────────────────────────

def _fetch_api(hmdb_id: str):
    """Returns (status, pathways). status in ok/no_pathways/not_found/error."""
    import urllib.request
    import urllib.error
    url = HMDB_URL.format(hmdb_id)
    req = urllib.request.Request(
        url, headers={'Accept': 'application/json',
                      'User-Agent': 'NMRAnalysis/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        pws = ((data.get('biological_properties') or {}).get('pathways') or [])
        return ('ok' if pws else 'no_pathways'), pws
    except urllib.error.HTTPError as e:
        return ('not_found' if e.code == 404 else 'error'), []
    except Exception:
        return 'error', []


# ── Public: build membership table ────────────────────────────────────────────

def build_membership(quant_results, *, use_api=True,
                     progress_cb=None, cancel_flag=None):
    """
    Build the pathway membership table for a list of quantification results.

    quant_results : list of dicts, each with 'metabolite' and an HMDB id under
                    'hmdb_id' or 'id'.
    use_api       : fall back to the HMDB REST API for IDs not cached locally.
    progress_cb   : optional callable(done, total, n_pathways) for UI feedback.
    cancel_flag   : optional; pass a list like [False] and set [0]=True to abort,
                    or a zero-arg callable returning truthy to abort.

    Returns
    -------
    dict
        {pathway_name: {'smpdb_id', 'description', 'category', 'mets': [names]}}
    list
        Names of metabolites skipped because they had no HMDB id.
    """
    def _cancelled():
        if cancel_flag is None:
            return False
        try:
            return (bool(cancel_flag[0]) if hasattr(cancel_flag, '__getitem__')
                    else bool(cancel_flag()))
        except Exception:
            return False

    entries, skipped = [], []
    for r in quant_results:
        name   = r.get('metabolite') or r.get('name') or '?'
        raw_id = str(r.get('hmdb_id') or r.get('id') or '')
        if is_hmdb(raw_id):
            entries.append({'name': name, 'hmdb_id': raw_id})
        else:
            skipped.append(name)

    table = {}
    total = len(entries)
    for i, entry in enumerate(entries):
        if _cancelled():
            break
        hmdb_id = norm_hmdb(entry['hmdb_id'])
        if not hmdb_id:
            skipped.append(entry['name'])
            continue

        pathways = query_local(hmdb_id)
        if pathways is None:
            if use_api:
                status, pathways = _fetch_api(hmdb_id)
                if status == 'error':
                    pathways = []
                time.sleep(RATE_DELAY)
            else:
                pathways = []

        for pw in (pathways or []):
            pw_name = pw.get('name') or '(unnamed)'
            e = table.setdefault(pw_name, {
                'smpdb_id':    pw.get('smpdb_id', ''),
                'description': pw.get('description', ''),
                'category':    pw.get('category', ''),
                'mets':        [],
            })
            if entry['name'] not in e['mets']:
                e['mets'].append(entry['name'])

        if progress_cb is not None:
            try:
                progress_cb(i + 1, total, len(table))
            except Exception:
                pass

    return table, skipped


def filter_by_category(pathway_table, category):
    """Filter a membership table by category ('metabolic'/'drug'/'disease'/'all')."""
    if category == 'all':
        return dict(pathway_table)
    return {k: v for k, v in pathway_table.items()
            if pathway_category(k, v.get('description', ''),
                                v.get('category', '')) == category}
