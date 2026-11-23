# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""Tell the analyst when a newer release is on GitHub.

The request runs on a daemon thread so a slow or absent network never delays
the window; Tk is only touched from the main loop, which polls the result.
At startup a failure is only logged (the app must open offline); from
Help → Check for updates… it is shown, as is "you are up to date".
"""
import os
import queue
import threading
from tkinter import messagebox

from app_paths import __version__
from data import update_check
from gui.widgets.update_dialog import UpdateDialog

_POLL_MS = 250


class _UpdateCheckMixin:

    def _schedule_startup_update_check(self, delay_ms=1500):
        if os.environ.get(update_check.DISABLE_ENV):
            return
        self.root.after(delay_ms, self._start_update_check)

    def check_for_updates(self):
        """Help-menu entry: check now, report every outcome."""
        self._start_update_check(manual=True)

    def _start_update_check(self, manual=False):
        if getattr(self, '_update_check_running', False):
            return
        self._update_check_running = True
        results = queue.Queue(maxsize=1)

        def work():
            try:
                results.put(('ok', update_check.fetch_latest_release()))
            except update_check.UpdateCheckError as exc:
                results.put(('error', exc))

        threading.Thread(target=work, name='update-check', daemon=True).start()
        self.root.after(_POLL_MS, self._poll_update_check, results, manual)

    def _poll_update_check(self, results, manual):
        try:
            status, value = results.get_nowait()
        except queue.Empty:
            self.root.after(_POLL_MS, self._poll_update_check, results, manual)
            return
        self._update_check_running = False

        if status == 'error':
            print(f'[update check] could not reach GitHub: {value}')
            if manual:
                messagebox.showwarning(
                    'Check for updates',
                    f'Could not reach GitHub to check for a new version.\n\n'
                    f'{value}', parent=self.root)
            return

        skipped = None if manual else update_check.load_skipped_tag()
        release = update_check.newer_release(__version__, value, skipped)
        if release is not None:
            UpdateDialog(self.root, release, __version__,
                         on_skip=self._skip_release)
        elif manual:
            messagebox.showinfo(
                'Check for updates',
                f'NMRPro-H v{__version__} is up to date.\n'
                f'Latest release on GitHub: {value["name"]}',
                parent=self.root)

    def _skip_release(self, tag):
        try:
            update_check.save_skipped_tag(tag)
        except OSError as exc:
            print(f'[update check] could not remember skipped {tag}: {exc}')
