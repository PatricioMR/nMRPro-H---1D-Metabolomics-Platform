"""Mirror the raw 1D 1H spectra of every BMRB entry the reference DB uses.

The reference database was built from the peak lists BMRB publishes, and those
are rounded to three decimals -- half a hertz at 500 MHz. The entries also
carry the spectra the lists were read from, and a coupling measured from a
spectrum is known to hundredths of a hertz. This fetches them.

    python hmdb_api_download/download_bmrb_fids.py [--limit N] [--entries bmse000208,...]

Source: the PDBj mirror of the BMRB FTP (bmrb.io itself did not answer on
2026-09-14; the mirror answers in under a second). Its layout is by compound,
not by entry -- ``standards/<compound>/nmr/<bmse>/`` -- and the map from entry
to compound directory is in the ``standards_tar/*.list`` files. Each entry
directory holds one tarball per experiment; ``1H.tar`` (and any other
``1H*.tar`` that is not a 2D) is a Bruker directory with ``fid``, ``acqus`` and
``pdata/1/1r``: about half a megabyte, 480 entries, a quarter of a gigabyte.

Resumable: an entry whose ``pdata/1/1r`` is already present is skipped.
Failures go to _fid_errors.log and never stop the run. Extracted under
FIDS/<bmse>/<experiment>/.
"""
from __future__ import annotations

import argparse
import html
import io
import os
import re
import sqlite3
import sys
import tarfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
BMRB_ENTRIES = r"C:\Users\migue\OneDrive\Documentos\bmrb_entries"
FIDS = os.path.join(os.path.dirname(ROOT), 'databases', 'bmrb_fids')
DB = os.path.join(os.path.dirname(ROOT), 'databases', 'BMRB_Metabolomics_v01.db')
MIRROR = 'http://bmrb.pdbj.org/ftp/pub/bmrb/metabolomics/'
MAP_CACHE = os.path.join(ROOT, '_bmse_to_compound.tsv')
ERRLOG = os.path.join(ROOT, '_fid_errors.log')
UA = {'User-Agent': 'Mozilla/5.0 (NMRPro-H reference-db build)'}

sys.path.insert(0, ROOT)
from build_bmrb_db import loop  # noqa: E402  (the NMR-STAR loop parser)

# A tarball that is a 1D proton experiment: starts with 1H and is not one of
# the 2D or edited experiments the entries also carry.
_ONE_D_PROTON = re.compile(r'^1H(?!.*(HSQC|HMBC|HMQC|COSY|TOCSY|NOESY|DEPT|13C))'
                           r'.*\.tar$', re.I)


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


def parse_list_files(texts) -> dict:
    """``{bmse: compound_dir}`` from the ``standards_tar/*.list`` files, which
    are lines of ``bmse000208 L_lactic_acid`` (space or tab separated)."""
    out = {}
    for text in texts:
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('bms'):
                out[parts[0]] = parts[1]
    return out


def _get(url, timeout=60, retries=4) -> bytes:
    """One request, retried with backoff. The mirror answers most requests in
    under a second and some directory listings in over a minute; a run that
    gave up on the first slow one lost whole entries to a single listing."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise last


def _listing(url) -> list[str]:
    page = _get(url).decode('utf-8', 'replace')
    return [html.unescape(h) for h in re.findall(r'href="([^"]+)"', page)
            if not h.startswith(('?', '/', '..'))]


def entry_map(refresh=False) -> dict:
    """The entry-to-directory map, fetched once and cached beside this file."""
    if os.path.exists(MAP_CACHE) and not refresh:
        with open(MAP_CACHE, encoding='utf-8') as f:
            return dict(line.rstrip('\n').split('\t', 1) for line in f if '\t' in line)
    names = [n for n in _listing(MIRROR + 'standards_tar/') if n.endswith('.list')]
    texts = []
    for n in names:
        try:
            texts.append(_get(MIRROR + 'standards_tar/' + n).decode('utf-8', 'replace'))
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f'list {n}: {e}')
    m = parse_list_files(texts)
    with open(MAP_CACHE, 'w', encoding='utf-8') as f:
        for k in sorted(m):
            f.write(f'{k}\t{m[k]}\n')
    return m


def entries_in_db() -> list[str]:
    con = sqlite3.connect(DB)
    ids = [r[0] for r in con.execute(
        "select coalesce(bmse_id, bmrb_id) from metabolites")]
    return sorted(i for i in ids if i and i.startswith('bmse'))


def already_have(dest) -> bool:
    for _root, _dirs, files in os.walk(dest):
        if '1r' in files:
            return True
    return False


def fetch_entry(bmse, compound_dir) -> int:
    """Download and extract every 1D proton tarball of every BMRB entry for
    this compound, under the entry the reference DB keys it by.

    A compound often has several entries -- L-lactic acid is bmse000208 at
    500 MHz and bmse000979 at 600 -- and the DB kept one. The others are the
    same molecule at another field, which is exactly the second measurement
    the field-dependence test needs. They land in FIDS/<db entry>/<entry>/
    <experiment>/. Returns how many experiments were fetched.
    """
    base = f'{MIRROR}standards/{compound_dir}/nmr/'
    entries = [n.rstrip('/') for n in _listing(base)
               if n.endswith('/') and n.startswith('bms')]
    n = 0
    for entry in entries:
        ebase = f'{base}{entry}/'
        try:
            tars = [t for t in _listing(ebase) if _ONE_D_PROTON.match(t)]
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
        dest = os.path.join(FIDS, bmse, entry)
        for name in tars:
            if already_have(os.path.join(dest, name[:-4])):
                continue
            data = _get(ebase + name, timeout=180)
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                tf.extractall(dest, filter='data')
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--entries', default='')
    ap.add_argument('--refresh-map', action='store_true')
    ap.add_argument('--workers', type=int, default=6,
                    help='entries fetched at once; the mirror is slow to '
                         'list, not to send, so latency is what overlaps')
    args = ap.parse_args()
    m = entry_map(refresh=args.refresh_map)
    print(f'{len(m)} entries in the mirror map')
    ids = ([e.strip() for e in args.entries.split(',') if e.strip()]
           if args.entries else entries_in_db())
    if args.limit:
        ids = ids[:args.limit]
    done = skipped = failed = unmapped = 0
    t0 = time.time()
    todo = []
    for bmse in ids:
        if already_have(os.path.join(FIDS, bmse)):
            skipped += 1
            continue
        comp = m.get(bmse)
        if not comp:
            unmapped += 1
            with open(ERRLOG, 'a', encoding='utf-8') as f:
                f.write(f'{bmse}\tnot in mirror map\n')
            continue
        todo.append((bmse, comp))

    def _one(item):
        bmse, comp = item
        try:
            return bmse, comp, fetch_entry(bmse, comp), None
        except (urllib.error.URLError, OSError, TimeoutError,
                tarfile.TarError) as e:
            return bmse, comp, 0, e

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (bmse, comp, n, err) in enumerate(pool.map(_one, todo), 1):
            if err is None:
                done += 1
                print(f'[{i}/{len(todo)}] {bmse} {comp}: {n} experiment(s)',
                      flush=True)
            else:
                failed += 1
                with open(ERRLOG, 'a', encoding='utf-8') as f:
                    f.write(f'{bmse}\t{comp}\t{err}\n')
                print(f'[{i}/{len(todo)}] {bmse} FAILED: {err}', flush=True)
    print(f'done {done}, skipped {skipped}, failed {failed}, '
          f'unmapped {unmapped}, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
