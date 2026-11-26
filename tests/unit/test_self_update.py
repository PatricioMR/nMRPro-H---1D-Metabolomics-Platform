# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Where a running copy is installed, and the helper that replaces it.

The Linux and macOS helpers are shell scripts: the tests that run them run on
those systems (the Tests workflow has a job for each).
"""
import os
import platform
import shutil
import subprocess
import sys
import time

import pytest

from controllers import self_update as su


def test_a_copy_run_from_source_does_not_update_itself():
    assert su.current_target(frozen=False, system='Windows') is None


def test_windows_installs_over_the_program_folder():
    t = su.current_target(frozen=True, system='Windows', env={},
                          executable=r'C:\Program Files\NMRPro-H\NMRPro-H.exe')
    assert t == su.Target('windows', r'C:\Program Files\NMRPro-H')


def test_linux_updates_only_an_appimage_it_can_write_beside(tmp_path):
    image = str(tmp_path / 'NMRPro-H-v0.89-linux-x86_64.AppImage')
    t = su.current_target(frozen=True, system='Linux', env={'APPIMAGE': image},
                          executable='/tmp/.mount_x/usr/lib/NMRPro-H/NMRPro-H')
    assert t == su.Target('appimage', image)
    # unpacked from the .tar.gz: no APPIMAGE, updated by hand
    assert su.current_target(frozen=True, system='Linux', env={},
                             executable='/opt/NMRPro-H/NMRPro-H') is None


def test_macos_updates_the_app_bundle(tmp_path):
    app = tmp_path / 'NMRPro-H.app'
    exe = app / 'Contents' / 'MacOS' / 'NMRPro-H'
    t = su.current_target(frozen=True, system='Darwin', env={},
                          executable=str(exe))
    assert t == su.Target('macos', str(app))


def test_windows_runs_the_setup_silently_once_the_program_has_ended():
    argv = su.helper_argv(su.Target('windows', r'C:\P'),
                          r"C:\Temp\it's\NMRPro-H-v0.90-Setup.exe", 4242)
    assert argv[0] == 'powershell'
    script = argv[-1]
    assert script.index('Wait-Process -Id 4242') < script.index('Start-Process')
    assert r"'C:\Temp\it''s\NMRPro-H-v0.90-Setup.exe'" in script   # quoted
    for flag in su.WINDOWS_SETUP_ARGS:
        assert f"'{flag}'" in script


def _ended_pid():
    p = subprocess.Popen([sys.executable, '-c', 'pass'])
    p.wait()
    return p.pid


def _run(argv):
    subprocess.run(argv, check=True, timeout=60)


@pytest.mark.skipif(os.name == 'nt' or not os.path.exists('/bin/sh'),
                    reason='the AppImage helper is a POSIX shell script')
def test_the_appimage_helper_replaces_the_old_file_and_starts_the_new(tmp_path):
    old = tmp_path / 'NMRPro-H-v0.89-linux-x86_64.AppImage'
    old.write_text('old')
    marker = tmp_path / 'started'
    new = tmp_path / 'download' / 'NMRPro-H-v0.90-linux-x86_64.AppImage'
    new.parent.mkdir()
    new.write_text(f'#!/bin/sh\ntouch "{marker}"\n')
    _run(su.helper_argv(su.Target('appimage', str(old)), str(new),
                        _ended_pid(), new_name=new.name))
    installed = tmp_path / new.name
    assert not old.exists() and installed.exists()
    assert os.access(installed, os.X_OK)
    for _ in range(50):                     # started in the background
        if marker.exists():
            break
        time.sleep(0.1)
    assert marker.exists()


@pytest.mark.skipif(platform.system() != 'Darwin' or not shutil.which('hdiutil'),
                    reason='the macOS helper needs hdiutil')
def test_the_macos_helper_replaces_the_bundle_from_the_disk_image(tmp_path):
    def bundle(where, text):
        exe = where / 'NMRPro-H.app' / 'Contents' / 'MacOS'
        exe.mkdir(parents=True)
        (exe / 'NMRPro-H').write_text(text)
        return where / 'NMRPro-H.app'

    apps = tmp_path / 'Applications'
    installed = bundle(apps, 'old')
    stage = tmp_path / 'stage'
    bundle(stage, 'new')
    dmg = tmp_path / 'NMRPro-H-v0.90-macOS-arm64.dmg'
    _run(['hdiutil', 'create', '-quiet', '-volname', 'NMRPro-H',
          '-srcfolder', str(stage), '-format', 'UDZO', str(dmg)])
    _run(su.helper_argv(su.Target('macos', str(installed)), str(dmg),
                        _ended_pid(), relaunch=False))
    assert (installed / 'Contents' / 'MacOS' / 'NMRPro-H').read_text() == 'new'
    assert not (apps / 'NMRPro-H.app.old').exists()
    assert not dmg.exists()
