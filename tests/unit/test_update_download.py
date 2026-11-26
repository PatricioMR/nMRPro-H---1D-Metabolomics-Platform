# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""An installer is kept only when it is the file GitHub published."""
import hashlib
import pathlib

import pytest

from data.update_download import (
    DownloadCancelled, DownloadError, download_verified)

PAYLOAD = b'NMRPro-H installer ' * 50000          # ~1 MB, several chunks


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'published.bin'
    path.write_bytes(PAYLOAD)
    return path.as_uri(), hashlib.sha256(PAYLOAD).hexdigest()


def test_the_published_file_is_kept_and_progress_is_reported(source, tmp_path):
    url, sha = source
    dest = str(tmp_path / 'Setup.exe')
    seen = []
    assert download_verified(url, dest, sha,
                             progress=lambda d, t: seen.append(d)) == dest
    assert pathlib.Path(dest).read_bytes() == PAYLOAD
    assert seen and seen[-1] == len(PAYLOAD) and seen == sorted(seen)
    assert not pathlib.Path(dest + '.part').exists()


def test_a_file_that_differs_from_the_published_one_is_deleted(source, tmp_path):
    url, _sha = source
    dest = tmp_path / 'Setup.exe'
    with pytest.raises(DownloadError, match='does not match'):
        download_verified(url, str(dest), '0' * 64)
    assert not dest.exists() and not pathlib.Path(str(dest) + '.part').exists()


def test_a_cancelled_download_leaves_nothing(source, tmp_path):
    url, sha = source
    dest = tmp_path / 'Setup.exe'
    with pytest.raises(DownloadCancelled):
        download_verified(url, str(dest), sha, cancelled=lambda: True)
    assert not dest.exists() and not pathlib.Path(str(dest) + '.part').exists()


def test_without_a_checksum_nothing_is_downloaded(source, tmp_path):
    url, _sha = source
    with pytest.raises(DownloadError, match='no checksum'):
        download_verified(url, str(tmp_path / 'x'), None)


def test_an_unreachable_file_is_reported(tmp_path):
    with pytest.raises(DownloadError, match='download failed'):
        download_verified((tmp_path / 'missing').as_uri(),
                          str(tmp_path / 'x'), 'ab')
