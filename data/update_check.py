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

A release whose notes carry ``<!-- mandatory -->`` (an HTML comment, so the
release page does not show it) is a mandatory update: it fixes something
critical, cannot be skipped, and is announced at every start.

Each release lists its installers with the SHA-256 GitHub computed for them;
``installer_for`` picks the one for the running system, so the program can
download and install it itself (``data.update_download``,
``controllers.self_update``).
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
MANDATORY_MARK = re.compile(r'<!--\s*mandatory\s*-->', re.IGNORECASE)
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
    body = payload.get('body') or ''
    return {
        'tag': tag,
        'name': payload.get('name') or tag,
        'url': payload.get('html_url') or RELEASES_PAGE,
        'notes': MANDATORY_MARK.sub('', body).strip(),
        'mandatory': bool(MANDATORY_MARK.search(body)),
        'assets': _assets(payload.get('assets')),
    }


def _assets(raw):
    """The release's files: name, download URL, size and SHA-256 (``None``
    when GitHub gave no digest, which forbids installing that file)."""
    out = []
    for a in raw or []:
        if not isinstance(a, dict) or not a.get('name'):
            continue
        digest = str(a.get('digest') or '')
        out.append({
            'name': a['name'],
            'url': a.get('browser_download_url'),
            'size': a.get('size'),
            'sha256': (digest.split(':', 1)[1].lower()
                       if digest.lower().startswith('sha256:') else None),
        })
    return out


def installer_for(assets, system, machine, appimage=False):
    """The release file the running program can install itself, or None.

    *system* is ``platform.system()``, *machine* ``platform.machine()``.
    Windows takes the Setup; Linux only the AppImage, and only when the
    program runs from one (a folder unpacked from the .tar.gz is replaced by
    hand); macOS the disk image of its own chip.
    """
    if system == 'Windows':
        suffix = '-Setup.exe'
    elif system == 'Linux':
        if not appimage:
            return None
        suffix = '-linux-x86_64.AppImage'
    elif system == 'Darwin':
        arch = 'arm64' if machine.lower() in ('arm64', 'aarch64') else 'x86_64'
        suffix = f'-macOS-{arch}.dmg'
    else:
        return None
    for a in assets or []:
        if a['name'].startswith('NMRPro-H-') and a['name'].endswith(suffix):
            return a if a.get('url') and a.get('sha256') else None
    return None


def newer_release(current_version, release, skipped_tag=None):
    """*release* when it is newer than *current_version* and not skipped.
    A mandatory release cannot be skipped."""
    if release is None or not is_newer(release['tag'], current_version):
        return None
    if (skipped_tag and release['tag'] == skipped_tag
            and not release.get('mandatory')):
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
