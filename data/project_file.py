# data/project_file.py
"""
The project file (.nmrproj) on disk: how a session state is written and read.

Format 2 (v0.73 on) is a zip archive, deflated:

  manifest.json     what the archive holds and how each part was stored
  state.pkl         the session state without the spectrum (pickle, protocol 5)
  all_intensities   the spectrum matrix as little-endian float32, one sample
                    after the other, byte-shuffled before deflate

Format 1 (up to v0.72) is a bare pickle of the state and is read as it is.

Why the spectrum goes to float32: it is two thirds of a project and its float64
mantissa below the noise is random, which no lossless coder compresses (the
Liver study: 39 MB raw, 32 MB deflated). float32 rounds each point by at most
6e-8 of its value; on the reference projects that is 2.8 % of the noise sigma
at the tallest peak and far less elsewhere, and the engine run on both gives
the same analytes, confidences and inclusions with concentrations equal to
1e-6 (ADR-084). A file once written reads back to the same numbers every time.

Byte-shuffle is the shuffle filter of HDF5 and Blosc: the n-th byte of every
value is stored together, so the exponent bytes, which repeat, sit side by
side for deflate to find.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import zipfile
from typing import NamedTuple

import numpy as np

FORMAT = 2
SPECTRUM_KEY = 'all_intensities'
_STORED_DTYPE = np.dtype('<f4')
_MANIFEST = 'manifest.json'
_STATE = 'state.pkl'
_LEVEL = 6
# A fixed member date keeps the same state writing the same bytes.
_DATE = (2026, 1, 1, 0, 0, 0)


class ProjectFileError(IOError):
    """The file is not a project this code can read, or did not write back."""


class Snapshot(NamedTuple):
    """A state frozen into bytes. Taken where the state lives (the GUI thread);
    writing it can then happen anywhere while the session goes on changing."""
    state_bytes: bytes
    spectrum: np.ndarray | None     # (n_samples, n_points), stored dtype
    spectrum_shape: tuple | None    # the matrix shape in the session
    digest: str                     # equal digests = equal saved content


def stored_spectrum(intensities) -> np.ndarray:
    """The spectrum as a project file gives it back: rounded to float32,
    returned as float64."""
    return _to_stored(intensities).T.astype(np.float64)


def _to_stored(intensities) -> np.ndarray:
    a = np.asarray(intensities)
    real = np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.integer)
    if a.ndim != 2 or not real:
        raise ProjectFileError(
            f'the spectrum matrix has shape {a.shape} and type {a.dtype}; '
            'a 2-D real matrix was expected')
    with np.errstate(over='ignore'):            # said below, as an error
        s = np.ascontiguousarray(a.T, dtype=_STORED_DTYPE)
    lost = np.isfinite(a.T) & ~np.isfinite(s)
    if lost.any():
        raise ProjectFileError(
            f'{int(lost.sum())} spectrum values are beyond the float32 range '
            f'(largest {np.nanmax(np.abs(a)):.3g}) and cannot be stored')
    return s


def snapshot(state: dict) -> Snapshot:
    spec = state.get(SPECTRUM_KEY)
    rest = {k: v for k, v in state.items()
            if k != SPECTRUM_KEY or spec is None}
    state_bytes = pickle.dumps(rest, protocol=5)
    h = hashlib.blake2b(state_bytes, digest_size=16)
    if spec is None:
        return Snapshot(state_bytes, None, None, h.hexdigest())
    stored = _to_stored(spec)
    h.update(memoryview(stored).cast('B'))
    return Snapshot(state_bytes, stored, tuple(np.shape(spec)), h.hexdigest())


def _shuffle(a: np.ndarray) -> bytes:
    return a.reshape(-1).view(np.uint8).reshape(-1, a.itemsize).T.tobytes()


def _unshuffle(buf: bytes, dtype, shape) -> np.ndarray:
    dtype = np.dtype(dtype)
    planes = np.frombuffer(buf, np.uint8).reshape(dtype.itemsize, -1)
    return np.ascontiguousarray(planes.T).view(dtype).reshape(shape)


def encode(snap: Snapshot) -> bytes:
    manifest = {'format': FORMAT, 'state': _STATE}
    if snap.spectrum is not None:
        manifest['spectrum'] = {
            'key': SPECTRUM_KEY, 'member': SPECTRUM_KEY,
            'shape': list(snap.spectrum_shape),
            'stored_dtype': _STORED_DTYPE.str, 'layout': 'samples-first',
            'filter': 'byte-shuffle'}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=_LEVEL) as z:
        def put(name, data):
            z.writestr(zipfile.ZipInfo(name, _DATE), data,
                       compress_type=zipfile.ZIP_DEFLATED,
                       compresslevel=_LEVEL)
        put(_MANIFEST, json.dumps(manifest, indent=1).encode('utf-8'))
        put(_STATE, snap.state_bytes)
        if snap.spectrum is not None:
            put(SPECTRUM_KEY, _shuffle(snap.spectrum))
    return buf.getvalue()


def decode(blob: bytes) -> dict:
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as e:
        raise ProjectFileError(f'not a project archive: {e}') from e
    with z:
        bad = z.testzip()
        if bad is not None:
            raise ProjectFileError(f'the project file is damaged ({bad})')
        manifest = json.loads(z.read(_MANIFEST))
        if manifest.get('format') != FORMAT:
            raise ProjectFileError(
                f"project format {manifest.get('format')} is newer than this "
                f'version of the software reads ({FORMAT})')
        state = pickle.loads(z.read(manifest['state']))
        spec = manifest.get('spectrum')
        if spec:
            n_points, n_samples = spec['shape']
            stored = _unshuffle(z.read(spec['member']), spec['stored_dtype'],
                                (n_samples, n_points))
            state[spec['key']] = np.ascontiguousarray(
                stored.T, dtype=np.float64)
    return state


def is_current_format(path: str) -> bool:
    return zipfile.is_zipfile(path)


def read(path: str) -> dict:
    """The state saved in ``path``, in either format."""
    with open(path, 'rb') as f:
        blob = f.read()
    if blob[:4] == b'PK\x03\x04':
        return decode(blob)
    try:
        return pickle.loads(blob)
    except Exception as e:
        raise ProjectFileError(
            f'{os.path.basename(path)} is not a project file ({e})') from e


def _check_written(blob: bytes, snap: Snapshot) -> None:
    """The bytes on disk hold exactly the snapshot, and they load."""
    decode(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        same = z.read(_STATE) == snap.state_bytes and (
            snap.spectrum is None
            or z.read(SPECTRUM_KEY) == _shuffle(snap.spectrum))
    if not same:
        raise ProjectFileError('the file written does not read back the same')


def write(path: str, snap: Snapshot, *, keep_backup: bool = True) -> int:
    """Write ``snap`` so a failed write leaves whatever was at ``path`` as it
    was: to a file beside it, read back and checked, then moved over it. With
    ``keep_backup`` the previous file is kept as ``.bak`` for one write.
    Returns the size written."""
    blob = encode(snap)
    tmp = f'{path}.tmp'
    try:
        with open(tmp, 'wb') as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, 'rb') as f:
            _check_written(f.read(), snap)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    if keep_backup and os.path.exists(path):
        try:
            os.replace(path, f'{path}.bak')
        except OSError:
            pass
    os.replace(tmp, path)
    return len(blob)


def save(path: str, state: dict, *, keep_backup: bool = True) -> int:
    return write(path, snapshot(state), keep_backup=keep_backup)
