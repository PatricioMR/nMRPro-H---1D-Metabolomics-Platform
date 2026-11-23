# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""
Is a newer NMRPro-H published on GitHub?

Asks the public GitHub REST API for the repository's latest release
(``/releases/latest``: drafts and pre-releases excluded, no token needed) and
compares its tag with the running ``__version__``. A commit on ``main`` without
a release never triggers the notice: the analyst downloads the installer
attached to a release, not the source.

Version numbers are decimal, as the tags have always been read: v0.54 → v0.8 →
v0.86 is ascending because 0.8 means 0.80. ``major.minor[.patch]`` with the
minor compared as a decimal fraction, so 0.8 < 0.86 < 0.9 and 1.0 > 0.99.

The version the analyst chose to skip is kept in the per-user data directory so
the notice does not come back until a newer one is out.
"""
import json
import os
import re
import ssl
import urllib.request
from decimal import Decimal

from app_paths import SOURCE_URL, user_data_dir

_REPO = SOURCE_URL.rstrip('/').split('github.com/')[-1]
LATEST_RELEASE_API = f'https://api.github.com/repos/{_REPO}/releases/latest'
RELEASES_PAGE = f'{SOURCE_URL.rstrip("/")}/releases/latest'

_PREFS_FILE = 'update_check.json'
_VERSION_RE = re.compile(r'(\d+)(?:\.(\d+))?(?:\.(\d+))?')

# Set to any non-empty value to skip the check at startup (tests, offline labs).
DISABLE_ENV = 'NMRPRO_NO_UPDATE_CHECK'


class UpdateCheckError(Exception):
    """The latest release could not be read (offline, rate limit, bad reply)."""


def parse_version(text):
    """Sortable key of a version or tag ('v0.86-beta' → (0, 0.86, 0)).

    None when the text holds no version number.
    """
    if not text:
        return None
    match = _VERSION_RE.search(str(text))
    if match is None:
        return None
    major, minor, patch = match.groups()
    return (int(major),
            Decimal(f'0.{minor}') if minor else Decimal(0),
            int(patch) if patch else 0)


def is_newer(remote, local):
    """True only when both parse and *remote* is strictly above *local*."""
    r, l = parse_version(remote), parse_version(local)
    return r is not None and l is not None and r > l


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_latest_release(timeout=6.0):
    """The latest published release as {'tag', 'name', 'url', 'notes'}.

    Raises UpdateCheckError with the reason when GitHub cannot be read.
    """
    request = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={'Accept': 'application/vnd.github+json',
                 'User-Agent': 'NMRPro-H-update-check'})
    try:
        with urllib.request.urlopen(request, timeout=timeout,
                                    context=_ssl_context()) as reply:
            payload = json.loads(reply.read().decode('utf-8'))
    except Exception as exc:
        raise UpdateCheckError(str(exc)) from exc
    tag = payload.get('tag_name') if isinstance(payload, dict) else None
    if not tag:
        raise UpdateCheckError('GitHub reply has no release tag')
    return {
        'tag': tag,
        'name': payload.get('name') or tag,
        'url': payload.get('html_url') or RELEASES_PAGE,
        'notes': payload.get('body') or '',
    }


def newer_release(current_version, release, skipped_tag=None):
    """*release* when it is newer than *current_version* and not skipped."""
    if release is None or not is_newer(release['tag'], current_version):
        return None
    if skipped_tag and release['tag'] == skipped_tag:
        return None
    return release


def _prefs_path():
    return os.path.join(user_data_dir(), _PREFS_FILE)


def load_skipped_tag():
    """The release tag the analyst asked not to be told about again."""
    try:
        with open(_prefs_path(), encoding='utf-8') as fh:
            return json.load(fh).get('skipped_tag')
    except (OSError, ValueError, AttributeError):
        return None


def save_skipped_tag(tag):
    with open(_prefs_path(), 'w', encoding='utf-8') as fh:
        json.dump({'skipped_tag': tag}, fh)
