"""Mirror the raw 1D 1H spectra of every BMRB entry the reference DB uses.

The reference database was built from the peak lists BMRB publishes, and those
are rounded to three decimals -- half a hertz at 500 MHz. The entries also
carry the spectra the lists were read from (``_Experiment.Raw_data_flag yes``,
Bruker directories under ``nmr/setNN/``), and a coupling measured from a
spectrum is known to hundredths of a hertz. This fetches them.

    python hmdb_api_download/download_bmrb_fids.py [--limit N] [--entries bmse000208,...]

Reads the local NMR-STAR files in BMRB_ENTRIES, finds every 1D 1H experiment
with raw data, and mirrors its directory from the BMRB FTP (over HTTPS, via
the directory index) into FIDS/<bmse>/<setNN>/. Resumable: a directory whose
``pdata/1/1r`` (or ``fid``) is already present is skipped. Failures go to
_fid_errors.log and never stop the run.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
BMRB_ENTRIES = r"C:\Users\migue\OneDrive\Documentos\bmrb_entries"
FIDS = os.path.join(os.path.dirname(ROOT), 'databases', 'bmrb_fids')
DB = os.path.join(os.path.dirname(ROOT), 'databases', 'BMRB_Metabolomics_v01.db')
BASE = 'https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/'
ERRLOG = os.path.join(ROOT, '_fid_errors.log')
UA = {'User-Agent': 'NMRPro-H reference-db build (contact: repo owner)'}

sys.path.insert(0, ROOT)
from build_bmrb_db import loop  # noqa: E402  (the NMR-STAR loop parser)


def spectrometer_fields(star_text) -> dict:
    """``{save-frame label: field in MHz}`` for every spectrometer listed."""
    out = {}
    for m in re.finditer(r'save_(\S+)\s*\n(.*?)\n\s*save_\s*\n', star_text,
                         re.S):
        fs = re.search(r'_NMR_spectrometer\.Field_strength\s+([\d.]+)',
                       m.group(2))
        if fs:
            out[m.group(1)] = float(fs.group(1))
    return out


def sample_conditions(star_text) -> dict:
    """pH, temperature and the like, as the entry declares them. The
    coupling of a freely rotating methyl does not move with any of these;
    the coupling of a rotamer-dependent alpha/beta pair moves with the pH,
    and the apparent splitting of a second-order system moves with the
    field. A measurement without its conditions cannot say which it is."""
    out = {}
    for r in loop(star_text, '_Sample_condition_variable'):
        t = (r.get('Type') or '').strip("'\"").lower()
        try:
            out[t] = float(r.get('Val'))
        except (TypeError, ValueError):
            pass
    comps = loop(star_text, '_Sample_component')
    buffers = [c.get('Mol_common_name', '').strip("'\"") for c in comps
               if (c.get('Type') or '').lower() == 'buffer']
    refs = [c.get('Mol_common_name', '').strip("'\"") for c in comps
            if (c.get('Type') or '').lower() == 'reference']
    out['buffer'] = ', '.join(b for b in buffers if b) or None
    out['reference'] = ', '.join(r for r in refs if r) or None
    return out


def one_d_proton_dirs(star_text) -> list[dict]:
    """Every 1D 1H experiment that has raw data, in the order the entry lists
    them: ``{'name', 'path', 'spectrometer', 'field_mhz'}``."""
    exps = loop(star_text, '_Experiment')
    files = loop(star_text, '_Experiment_file')
    fields = spectrometer_fields(star_text)
    out = []
    for e in exps:
        name = (e.get('Name') or '').strip("'\"")
        if not re.match(r'1D\s*1H\b', name, re.I):
            continue
        if (e.get('Raw_data_flag') or '').lower() != 'yes':
            continue
        eid = e.get('ID')
        label = (e.get('NMR_spectrometer_label') or '').lstrip('$')
        for f in files:
            if f.get('Experiment_ID') == eid and f.get('Type') == 'text/directory':
                path = (f.get('Directory_path') or '').strip("'\"").strip('/')
                if path:
                    out.append({'name': name, 'path': path,
                                'spectrometer': label,
                                'field_mhz': fields.get(label)})
    return out


def _listing(url) -> list[str]:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        page = r.read().decode('utf-8', 'replace')
    names = []
    for href in re.findall(r'href="([^"]+)"', page):
        href = html.unescape(href)
        if href.startswith('?') or href.startswith('/') or href == '../':
            continue
        names.append(href)
    return names


def mirror(url, dest, *, depth=0, max_depth=6, stats=None) -> None:
    """Recursively copy an FTP-over-HTTP directory index to ``dest``."""
    if depth > max_depth:
        return
    os.makedirs(dest, exist_ok=True)
    for name in _listing(url):
        if name.endswith('/'):
            mirror(url + name, os.path.join(dest, name.rstrip('/')),
                   depth=depth + 1, max_depth=max_depth, stats=stats)
            continue
        target = os.path.join(dest, name)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        req = urllib.request.Request(url + name, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(target, 'wb') as f:
            f.write(r.read())
        if stats is not None:
            stats['files'] = stats.get('files', 0) + 1


def entries_in_db() -> list[str]:
    con = sqlite3.connect(DB)
    ids = [r[0] for r in con.execute(
        "select coalesce(bmse_id, bmrb_id) from metabolites")]
    return sorted(i for i in ids if i and i.startswith('bmse'))


def already_have(dest) -> bool:
    for root, _dirs, files in os.walk(dest):
        if '1r' in files or 'fid' in files:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--entries', default='')
    ap.add_argument('--first-only', action='store_true',
                    help='one 1D 1H experiment per entry, not every one')
    args = ap.parse_args()
    ids = ([e.strip() for e in args.entries.split(',') if e.strip()]
           if args.entries else entries_in_db())
    if args.limit:
        ids = ids[:args.limit]
    done = skipped = failed = 0
    t0 = time.time()
    for bmse in ids:
        star = os.path.join(BMRB_ENTRIES, bmse, bmse + '.str')
        if not os.path.exists(star):
            continue
        text = open(star, encoding='utf-8', errors='replace').read()
        dirs = one_d_proton_dirs(text)
        if args.first_only:
            dirs = dirs[:1]
        for d in dirs:
            name, path, spec = d['name'], d['path'], d['spectrometer']
            dest = os.path.join(FIDS, bmse, path.replace('/', os.sep))
            if already_have(dest):
                skipped += 1
                continue
            try:
                mirror(BASE + bmse + '/' + path + '/', dest)
                done += 1
                print(f'{bmse} {path} ({name}, {spec})')
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                failed += 1
                with open(ERRLOG, 'a', encoding='utf-8') as f:
                    f.write(f'{bmse}\t{path}\t{e}\n')
                print(f'{bmse} {path} FAILED: {e}')
    print(f'done {done}, skipped {skipped}, failed {failed}, '
          f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
