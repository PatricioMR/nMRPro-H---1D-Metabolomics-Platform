# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Download a release installer and prove it is the file GitHub published.

The packages are not signed with a commercial certificate, so nothing else
vouches for them: the file is hashed while it downloads and kept only when its
SHA-256 equals the digest GitHub computed for the release asset
(``data.update_check``). A partial or altered download is deleted, never run.
"""
import hashlib
import os
import urllib.request

from data.update_check import _ssl_context

_CHUNK = 256 * 1024


class DownloadError(Exception):
    """The installer could not be downloaded, or is not the published file."""


class DownloadCancelled(DownloadError):
    """The analyst stopped the download."""


def download_verified(url, dest, sha256, progress=None, cancelled=None,
                      timeout=30.0):
    """Download *url* to *dest* and check it against *sha256*.

    *progress(done, total)* is called as bytes arrive (total may be None);
    *cancelled()* returning True stops the download. Returns *dest*. The file
    is written beside *dest* with a ``.part`` suffix and renamed only after
    its digest matches, so *dest* never holds an unchecked file.
    """
    if not sha256:
        raise DownloadError('GitHub gave no checksum for this file; it will '
                            'not be installed')
    part = dest + '.part'
    h = hashlib.sha256()
    done = 0
    request = urllib.request.Request(
        url, headers={'User-Agent': 'NMRPro-H-updater'})
    try:
        with urllib.request.urlopen(request, timeout=timeout,
                                    context=_ssl_context()) as reply, \
                open(part, 'wb') as out:
            total = int(reply.headers.get('Content-Length') or 0) or None
            while True:
                if cancelled is not None and cancelled():
                    raise DownloadCancelled('download stopped')
                chunk = reply.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    except DownloadError:
        _remove(part)
        raise
    except Exception as exc:
        _remove(part)
        raise DownloadError(f'download failed: {exc}') from exc
    if h.hexdigest() != sha256.lower():
        _remove(part)
        raise DownloadError('the downloaded file does not match the one '
                            'published on GitHub; it was deleted')
    os.replace(part, dest)
    return dest


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
