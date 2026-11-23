# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_update_check.py
"""Reading GitHub's latest release and deciding whether it is newer."""
import io
import json

import pytest

from data import update_check as uc


@pytest.mark.parametrize('remote, local, newer', [
    ('v0.87-beta', '0.86', True),
    ('v0.86-beta', '0.86', False),     # same version, stage suffix ignored
    ('v0.8-beta', '0.86', False),      # 0.8 is 0.80: the tags' own history
    ('v0.86', '0.8', True),
    ('v0.9', '0.86', True),
    ('v1.0', '0.99', True),
    ('v0.86.1', '0.86', True),
    ('v0.54', '0.8', False),
    ('nightly', '0.86', False),        # no number: never claim newer
    ('v0.87', '', False),
])
def test_is_newer(remote, local, newer):
    assert uc.is_newer(remote, local) is newer


def test_parse_version_reads_the_tag_number():
    assert uc.parse_version('v0.86-beta') == uc.parse_version('0.86')
    assert uc.parse_version(None) is None


def test_api_url_points_at_the_repository():
    assert uc.LATEST_RELEASE_API == (
        'https://api.github.com/repos/PatricioMR/'
        'nMRPro-H---1D-Metabolomics-Platform/releases/latest')


_RELEASE = {'tag': 'v0.87-beta', 'name': 'NMRPro-H v0.87 Beta',
            'url': 'https://example.org/r', 'notes': ''}


def test_newer_release_respects_skip():
    assert uc.newer_release('0.86', _RELEASE) is _RELEASE
    assert uc.newer_release('0.86', _RELEASE, skipped_tag='v0.87-beta') is None
    assert uc.newer_release('0.86', _RELEASE, skipped_tag='v0.85') is _RELEASE
    assert uc.newer_release('0.87', _RELEASE) is None
    assert uc.newer_release('0.86', None) is None


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, payload=None, error=None):
    seen = {}

    def fake_urlopen(request, timeout, context):
        seen['url'] = request.full_url
        seen['agent'] = request.get_header('User-agent')
        if error is not None:
            raise error
        return _Reply(json.dumps(payload).encode('utf-8'))

    monkeypatch.setattr(uc.urllib.request, 'urlopen', fake_urlopen)
    return seen


def test_fetch_latest_release_reads_the_reply(monkeypatch):
    seen = _serve(monkeypatch, {
        'tag_name': 'v0.87-beta', 'name': 'NMRPro-H v0.87 Beta',
        'html_url': 'https://github.com/x/releases/tag/v0.87-beta',
        'body': 'Fixes.'})
    release = uc.fetch_latest_release()
    assert release == {'tag': 'v0.87-beta', 'name': 'NMRPro-H v0.87 Beta',
                       'url': 'https://github.com/x/releases/tag/v0.87-beta',
                       'notes': 'Fixes.'}
    assert seen['url'] == uc.LATEST_RELEASE_API
    assert seen['agent']                      # GitHub refuses requests without one


def test_fetch_latest_release_fills_missing_fields(monkeypatch):
    _serve(monkeypatch, {'tag_name': 'v0.87', 'name': None, 'body': None})
    release = uc.fetch_latest_release()
    assert release['name'] == 'v0.87'
    assert release['url'] == uc.RELEASES_PAGE
    assert release['notes'] == ''


@pytest.mark.parametrize('payload, error', [
    (None, OSError('offline')),
    ({'message': 'API rate limit exceeded'}, None),
    (['not', 'a', 'release'], None),
])
def test_fetch_latest_release_reports_failure(monkeypatch, payload, error):
    _serve(monkeypatch, payload, error)
    with pytest.raises(uc.UpdateCheckError):
        uc.fetch_latest_release()


def test_skipped_tag_round_trip(monkeypatch, tmp_path):
    prefs = tmp_path / 'update_check.json'
    monkeypatch.setattr(uc, '_prefs_path', lambda: str(prefs))
    assert uc.load_skipped_tag() is None
    uc.save_skipped_tag('v0.87-beta')
    assert uc.load_skipped_tag() == 'v0.87-beta'
    prefs.write_text('not json', encoding='utf-8')
    assert uc.load_skipped_tag() is None
