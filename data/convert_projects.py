# data/convert_projects.py
"""
Convert projects saved before v0.73 (a bare pickle) to the compressed project
file of data.project_file.

    python -m data.convert_projects FOLDER_OR_FILE [...] [--dry-run]

Every .nmrproj, .nmrproj.autosave and .nmrproj.bak under the paths is looked
at; those already in the new format are left alone. Each one is written beside
itself, read back and compared with the original -- the state exactly, the
spectrum as float32 rounds it -- and only then moved over the original, with
the original's modification time, so "the autosave is newer than the project"
still reads true. A file that does not compare is left as it was and said.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import project_file as pf

SUFFIXES = ('.nmrproj', '.nmrproj.autosave', '.nmrproj.bak')


def project_files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for root, _dirs, files in os.walk(p):
            for f in sorted(files):
                if f.endswith(SUFFIXES):
                    yield os.path.join(root, f)


def _same_state(back, orig, snap):
    rest = {k: v for k, v in back.items() if k != pf.SPECTRUM_KEY
            or snap.spectrum is None}
    if pickle.dumps(rest, protocol=5) == snap.state_bytes:
        return True
    try:
        for k, v in orig.items():
            if k != pf.SPECTRUM_KEY:
                np.testing.assert_equal(back[k], v)
        return set(back) == set(orig)
    except AssertionError:
        return False


def convert(path):
    """Convert one file in place. Returns (old size, new size)."""
    old_size = os.path.getsize(path)
    stat = os.stat(path)
    orig = pf.read(path)
    snap = pf.snapshot(orig)
    blob = pf.encode(snap)
    back = pf.decode(blob)
    if not _same_state(back, orig, snap):
        raise pf.ProjectFileError('the converted state does not compare')
    spec = orig.get(pf.SPECTRUM_KEY)
    if spec is not None and not np.array_equal(
            back[pf.SPECTRUM_KEY], pf.stored_spectrum(spec), equal_nan=True):
        raise pf.ProjectFileError('the converted spectrum does not compare')
    tmp = f'{path}.tmp'
    try:
        with open(tmp, 'wb') as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, 'rb') as f:
            if f.read() != blob:
                raise pf.ProjectFileError('the file written does not read back')
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return old_size, len(blob)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--dry-run', action='store_true',
                    help='list what would be converted, change nothing')
    a = ap.parse_args(argv)
    before = after = 0
    failed = []
    for path in project_files(a.paths):
        if pf.is_current_format(path):
            continue
        size = os.path.getsize(path)
        if a.dry_run:
            print(f'{size / 1e6:7.1f} MB  {path}')
            before += size
            continue
        try:
            old, new = convert(path)
        except Exception as e:
            failed.append((path, e))
            print(f'  NOT converted  {path}: {e}', flush=True)
            continue
        before += old
        after += new
        print(f'{old / 1e6:7.1f} -> {new / 1e6:6.1f} MB  {path}', flush=True)
    if a.dry_run:
        print(f'{before / 1e6:.0f} MB in the old format')
    else:
        print(f'{before / 1e6:.0f} MB -> {after / 1e6:.0f} MB'
              + (f'; {len(failed)} not converted' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
