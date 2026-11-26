# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Install a downloaded release over the running program, then start it again.

The program cannot replace its own files while it runs, so a small helper is
started first, detached: it waits for this process to end, installs, and opens
the new version. What it does per system:

    Windows  runs the Setup silently (it asks for administrator rights; the
             installer reopens the program when done)
    Linux    puts the new AppImage beside the old one, deletes the old one
    macOS    copies the .app out of the disk image over the installed one,
             keeping the old one until the copy succeeded

``current_target`` says whether this copy can be updated that way. A program
run from source, a Linux folder unpacked from the .tar.gz, or a location the
user cannot write to is updated by hand from the release page instead.
"""
import os
import platform
import subprocess
import sys
from dataclasses import dataclass

# Inno Setup command line: no wizard pages, no message boxes, and close a
# program still holding the files.
WINDOWS_SETUP_ARGS = ('/SILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                      '/CLOSEAPPLICATIONS')


@dataclass(frozen=True)
class Target:
    kind: str           # 'windows' | 'appimage' | 'macos'
    path: str           # install folder | AppImage file | .app bundle


def current_target(frozen=None, system=None, env=None, executable=None):
    """Where this running copy is installed, or None when it cannot update
    itself."""
    frozen = getattr(sys, 'frozen', False) if frozen is None else frozen
    system = platform.system() if system is None else system
    env = os.environ if env is None else env
    executable = sys.executable if executable is None else executable
    if not frozen:
        return None
    if system == 'Windows':
        return Target('windows', os.path.dirname(executable))
    if system == 'Linux':
        image = env.get('APPIMAGE')
        if image and os.access(os.path.dirname(image) or '.', os.W_OK):
            return Target('appimage', image)
        return None
    if system == 'Darwin':
        # .../NMRPro-H.app/Contents/MacOS/NMRPro-H
        app = os.path.dirname(os.path.dirname(os.path.dirname(executable)))
        if app.endswith('.app') and os.access(os.path.dirname(app), os.W_OK):
            return Target('macos', app)
    return None


_WAIT = 'while kill -0 "$1" 2>/dev/null; do sleep 0.5; done\n'

_APPIMAGE = _WAIT + r'''
new="$(dirname "$3")/$4"
chmod +x "$2" && mv -f "$2" "$new" || exit 1
[ "$3" = "$new" ] || rm -f "$3"
[ "$5" = relaunch ] && nohup "$new" >/dev/null 2>&1 &
'''

_MACOS = _WAIT + r'''
mnt="$(mktemp -d)"
hdiutil attach -nobrowse -readonly -mountpoint "$mnt" "$2" >/dev/null || exit 1
new="$(ls -d "$mnt"/*.app | head -n 1)"
rm -rf "$3.old"
if mv "$3" "$3.old" && ditto "$new" "$3"; then
    rm -rf "$3.old"
else
    rm -rf "$3"; mv "$3.old" "$3"
fi
hdiutil detach "$mnt" -quiet
xattr -dr com.apple.quarantine "$3" 2>/dev/null
rm -f "$2"
[ "$4" = relaunch ] && open "$3"
'''


def _ps_quote(text):
    return "'" + str(text).replace("'", "''") + "'"


def helper_argv(target, installer, pid, new_name=None, relaunch=True):
    """The command that installs *installer* once process *pid* has ended.

    *new_name* is the AppImage's file name (it carries the version).
    """
    again = 'relaunch' if relaunch else 'stay'
    if target.kind == 'windows':
        args = ','.join(_ps_quote(a) for a in WINDOWS_SETUP_ARGS)
        script = (f'Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue; '
                  f'Start-Process -FilePath {_ps_quote(installer)} '
                  f'-ArgumentList {args}')
        return ['powershell', '-NoProfile', '-NonInteractive',
                '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
                '-Command', script]
    if target.kind == 'appimage':
        return ['/bin/sh', '-c', _APPIMAGE, 'nmrpro-update', str(int(pid)),
                installer, target.path, new_name or os.path.basename(installer),
                again]
    if target.kind == 'macos':
        return ['/bin/sh', '-c', _MACOS, 'nmrpro-update', str(int(pid)),
                installer, target.path, again]
    raise ValueError(f'unknown target kind {target.kind!r}')


def launch(argv):
    """Start *argv* detached, so it outlives this process."""
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == 'nt':
        kwargs['creationflags'] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.CREATE_NO_WINDOW)
    else:
        kwargs['start_new_session'] = True
    return subprocess.Popen(argv, **kwargs)
