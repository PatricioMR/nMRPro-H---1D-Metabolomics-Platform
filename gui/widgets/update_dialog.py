# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The notice that a newer NMRPro-H release is on GitHub, and the update.

"Update now" does two things at once, with no further question: it opens the
release page, so the analyst reads what changed, and it downloads the
installer for this system with a progress bar. The download is checked against
the SHA-256 GitHub published; then the program offers to save the project,
closes, and a helper installs the new version and opens it
(``controllers.self_update``). Where the program cannot update itself (run
from source, a Linux folder from the .tar.gz, a folder it cannot write to)
the release page is all it opens.

A mandatory release (``data.update_check``) is announced in red as a critical
fix and cannot be skipped.
"""
import os
import platform
import queue
import tempfile
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk

from controllers import self_update
from data import update_check
from data.update_download import (
    DownloadCancelled, DownloadError, download_verified)
from gui.utils.window_place import center_on

_NOTES_MAX_CHARS = 4000
_POLL_MS = 100
MANDATORY_TEXT = ('This is a mandatory update. It fixes a critical error, and '
                  'we strongly recommend installing it now.')


class UpdateDialog(tk.Toplevel):
    """Update now · Remind me later · Skip this version (not when mandatory).

    *on_skip* receives the release tag when the analyst skips it.
    *confirm_close()* offers to save the project and returns whether the
    program may close; *shutdown()* closes it. Without them the update stops
    at the download.
    """

    def __init__(self, parent, release, current_version, on_skip=None,
                 confirm_close=None, shutdown=None):
        super().__init__(parent)
        self.title('Mandatory update' if release.get('mandatory')
                   else 'Update available')
        self.transient(parent)
        self.resizable(True, True)
        self._release = release
        self._on_skip = on_skip
        self._confirm_close = confirm_close
        self._shutdown = shutdown
        self._cancel = threading.Event()
        self._events = queue.Queue()
        self._downloaded = None
        self._target = None
        self._asset = None

        self._body = ttk.Frame(self, padding=14)
        self._body.pack(fill=tk.BOTH, expand=True)
        self._build_notice(current_version)

        self.protocol('WM_DELETE_WINDOW', self._close)
        self.bind('<Escape>', lambda _e: self._close())
        center_on(self, parent)
        self._update_btn.focus_set()

    # ── the notice ────────────────────────────────────────────────────────────

    def _build_notice(self, current_version):
        body, release = self._body, self._release
        if release.get('mandatory'):
            tk.Label(body, text='Mandatory update', fg='#b00020',
                     font=('Segoe UI', 12, 'bold')).pack(anchor='w')
            tk.Label(body, text=MANDATORY_TEXT, fg='#b00020', wraplength=520,
                     justify='left', font=('Segoe UI', 10)).pack(
                anchor='w', pady=(2, 6))
        else:
            ttk.Label(body, text='A new version of NMRPro-H is available',
                      font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        ttk.Label(body, text=f'Installed: v{current_version}     '
                             f'Available: {release["name"]}').pack(
            anchor='w', pady=(4, 10))

        notes = release.get('notes', '').strip()
        if notes:
            ttk.Label(body, text="What's new:").pack(anchor='w')
            frame = ttk.Frame(body)
            frame.pack(fill=tk.BOTH, expand=True, pady=(2, 10))
            text = tk.Text(frame, wrap='word', height=12, width=70,
                           font=('Segoe UI', 9), relief='solid', bd=1)
            scroll = ttk.Scrollbar(frame, orient='vertical', command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            if len(notes) > _NOTES_MAX_CHARS:
                notes = notes[:_NOTES_MAX_CHARS] + '\n\n… (see the release page)'
            text.insert('1.0', notes)
            text.configure(state='disabled')
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._buttons = ttk.Frame(body)
        self._buttons.pack(fill=tk.X)
        if not release.get('mandatory'):
            ttk.Button(self._buttons, text='Skip this version',
                       command=self._skip).pack(side=tk.LEFT)
        ttk.Button(self._buttons, text='Remind me later',
                   command=self._close).pack(side=tk.RIGHT)
        self._update_btn = ttk.Button(self._buttons, text='Update now',
                                      command=self._update_now)
        self._update_btn.pack(side=tk.RIGHT, padx=(0, 6))

    # ── update now ────────────────────────────────────────────────────────────

    def _update_now(self):
        webbrowser.open(self._release['url'])
        self._target = self_update.current_target()
        self._asset = update_check.installer_for(
            self._release.get('assets'), platform.system(), platform.machine(),
            appimage=(self._target is not None
                      and self._target.kind == 'appimage'))
        self._buttons.destroy()
        self._progress_frame = ttk.Frame(self._body)
        self._progress_frame.pack(fill=tk.X)
        if self._target is None or self._asset is None:
            self._finish('This copy cannot update itself. The release page '
                         'is open in your browser: download the new version '
                         'there.')
            return
        self._status = ttk.Label(self._progress_frame,
                                 text=f'Downloading {self._asset["name"]}…')
        self._status.pack(anchor='w')
        self._bar = ttk.Progressbar(self._progress_frame, mode='determinate',
                                    maximum=100, length=480)
        self._bar.pack(fill=tk.X, pady=4)
        self._amount = ttk.Label(self._progress_frame, text='',
                                 foreground='#555')
        self._amount.pack(anchor='w')
        self._action = ttk.Button(self._progress_frame, text='Cancel',
                                  command=self._cancel.set)
        self._action.pack(anchor='e', pady=(6, 0))

        folder = tempfile.mkdtemp(prefix='nmrpro-update-')
        dest = os.path.join(folder, self._asset['name'])
        threading.Thread(target=self._download, args=(dest,),
                         name='update-download', daemon=True).start()
        self.after(_POLL_MS, self._poll)

    def _download(self, dest):
        """Worker thread: no Tk here, only the queue."""
        try:
            path = download_verified(
                self._asset['url'], dest, self._asset['sha256'],
                progress=lambda done, total: self._events.put(
                    ('progress', done, total or self._asset.get('size'))),
                cancelled=self._cancel.is_set)
            self._events.put(('done', path))
        except DownloadCancelled:
            self._events.put(('cancelled',))
        except DownloadError as exc:
            self._events.put(('error', str(exc)))

    def _poll(self):
        last = None
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == 'progress':
                    last = event
                else:
                    self._on_result(event)
                    return
        except queue.Empty:
            pass
        if last is not None:
            _k, done, total = last
            if total:
                self._bar['value'] = 100.0 * done / total
                self._amount.config(text=f'{done / 1e6:.1f} of '
                                         f'{total / 1e6:.1f} MB')
            else:
                self._amount.config(text=f'{done / 1e6:.1f} MB')
        if self.winfo_exists():
            self.after(_POLL_MS, self._poll)

    def _on_result(self, event):
        kind = event[0]
        if kind == 'cancelled':
            self._finish('Download cancelled.')
        elif kind == 'error':
            self._finish(f'The update could not be downloaded: {event[1]}. '
                         'The release page is open in your browser.')
        else:
            self._downloaded = event[1]
            self._bar['value'] = 100
            self._install()

    def _install(self):
        if self._confirm_close is None or self._shutdown is None:
            self._finish('Downloaded and verified.')
            return
        self._status.config(text='Downloaded and verified. NMRPro-H will '
                                 'close, install the update and open again.')
        self.update_idletasks()
        if not self._confirm_close():
            self._status.config(text='Installation postponed: the update is '
                                     'downloaded. Install when ready.')
            self._action.config(text='Install', command=self._install)
            return
        self_update.launch(self_update.helper_argv(
            self._target, self._downloaded, os.getpid(),
            new_name=self._asset['name']))
        self._shutdown()

    def _finish(self, message):
        for w in self._progress_frame.winfo_children():
            if w is not getattr(self, '_bar', None):
                w.destroy()
        ttk.Label(self._progress_frame, text=message, wraplength=520,
                  justify='left').pack(anchor='w', pady=(4, 0))
        ttk.Button(self._progress_frame, text='Close',
                   command=self.destroy).pack(anchor='e', pady=(6, 0))

    # ── leaving ───────────────────────────────────────────────────────────────

    def _close(self):
        self._cancel.set()
        self.destroy()

    def _skip(self):
        if self._on_skip is not None:
            self._on_skip(self._release['tag'])
        self.destroy()
