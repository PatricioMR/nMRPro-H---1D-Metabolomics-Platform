# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The notice that a newer NMRPro-H release is on GitHub."""
import tkinter as tk
import webbrowser
from tkinter import ttk

from gui.utils.window_place import center_on

_NOTES_MAX_CHARS = 4000


class UpdateDialog(tk.Toplevel):
    """Download (opens the release page) · Remind me later · Skip this version.

    *on_skip* receives the release tag when the analyst skips it.
    """

    def __init__(self, parent, release, current_version, on_skip=None):
        super().__init__(parent)
        self.title('Update available')
        self.transient(parent)
        self.resizable(True, True)
        self._release = release
        self._on_skip = on_skip

        body = ttk.Frame(self, padding=14)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text='A new version of NMRPro-H is available on GitHub',
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

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text='Skip this version',
                   command=self._skip).pack(side=tk.LEFT)
        ttk.Button(buttons, text='Remind me later',
                   command=self.destroy).pack(side=tk.RIGHT)
        download = ttk.Button(buttons, text='Download',
                              command=self._download)
        download.pack(side=tk.RIGHT, padx=(0, 6))

        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.bind('<Escape>', lambda _e: self.destroy())
        center_on(self, parent)
        download.focus_set()

    def _download(self):
        webbrowser.open(self._release['url'])
        self.destroy()

    def _skip(self):
        if self._on_skip is not None:
            self._on_skip(self._release['tag'])
        self.destroy()
