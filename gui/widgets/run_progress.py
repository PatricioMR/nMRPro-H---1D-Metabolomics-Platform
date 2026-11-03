# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The progress window of a long run: annotation, quantification.

A run is a chain of phases of very different length, and the analyst has to
see which one is running, how far it is and that the program is alive.
Workers post to it with ``post``, which only stores the state; everything
else here belongs to the Tk thread.

A run that can stop takes ``on_cancel``: closing the window or pressing
Cancel asks it to, and the window says so until the worker has reached a
point where stopping leaves nothing half done.
"""
import time
import tkinter as tk
from tkinter import ttk

TICK_MS = 100
WIDTH = 480
WRAP = WIDTH - 40           # the text wraps inside the window, never past it


def phase_percent(table, phases, key, frac):
    """Overall percent of the run when phase ``key`` is ``frac`` done.
    ``table`` is ``{key: (title, share of the bar)}``."""
    weights = [table[k][1] for k in phases]
    i = list(phases).index(key)
    frac = min(max(float(frac), 0.0), 1.0)
    return 100.0 * (sum(weights[:i]) + frac * weights[i]) / sum(weights)


def bar_percent(table, phases, key, done, total):
    """Overall percent of the run when phase ``key`` has done ``done`` of
    ``total`` items; a phase without a count sits at its start."""
    return phase_percent(table, phases, key, done / total if total else 0.0)


def time_left(elapsed, done, total):
    """Seconds the phase still needs at the rate it has kept so far, or None
    while that rate says nothing yet (under 2 s, or no item done)."""
    if not total or done <= 0 or done >= total or elapsed < 2.0:
        return None
    return elapsed / done * (total - done)


def format_seconds(s):
    s = int(round(s))
    return f'{s} s' if s < 60 else f'{s // 60} min {s % 60:02d} s'


class RunProgress:
    """Modal progress window: step, bar, item count with the time left in the
    step, and the run's elapsed time. It redraws itself every TICK_MS, so a
    long item still shows the clock moving and its estimate growing instead of
    a frozen "about 4 s left"."""

    def __init__(self, root, phases, table, title, subtitle='',
                 on_cancel=None):
        self.root = root
        self.phases = tuple(phases)
        self.table = table
        self.cancelled = False
        self._on_cancel = on_cancel
        self._t0 = time.monotonic()
        self._phase = None
        self._phase_t0 = self._t0
        self._frac = 0.0            # how far the phase on screen has got
        self._pending = None        # written by a worker, read by _tick
        self._taken = None
        self._shown = None
        self._tick_id = None
        self._closed = False

        self.win = tk.Toplevel(root)
        self.win.title(title)
        self.win.resizable(False, False)
        self.win.transient(root)
        self.win.grab_set()
        # without a way to stop it, closing would leave the worker writing to
        # a window that is gone; the run ends by itself
        self.win.protocol('WM_DELETE_WINDOW',
                          self.cancel if on_cancel else (lambda: None))
        # the width is fixed and the height follows the text: a long step or
        # note wraps onto a second line instead of running out of the window
        self.win.minsize(WIDTH, 1)
        self.win.maxsize(WIDTH, 10000)

        if subtitle:
            ttk.Label(self.win, text=subtitle, font=('Arial', 9),
                      foreground='gray', wraplength=WRAP,
                      justify='center').pack(pady=(10, 0))
        self._step_lbl = ttk.Label(self.win, text='Starting…',
                                   font=('Arial', 10, 'bold'),
                                   wraplength=WRAP, justify='center')
        self._step_lbl.pack(pady=(6, 4), padx=20)
        self._bar = ttk.Progressbar(self.win, length=WRAP,
                                    mode='determinate', maximum=100)
        self._bar.pack(padx=20)
        self._detail_lbl = ttk.Label(self.win, text='', font=('Arial', 9),
                                     wraplength=WRAP, justify='center')
        self._detail_lbl.pack(pady=(4, 0), padx=20)
        self._time_lbl = ttk.Label(self.win, text='', font=('Arial', 8),
                                   foreground='gray')
        self._time_lbl.pack(pady=(2, 8))
        self._cancel_btn = None
        if on_cancel:
            self._cancel_btn = ttk.Button(self.win, text='Cancel',
                                          command=self.cancel)
            self._cancel_btn.pack(pady=(0, 8))
        root.update_idletasks()
        self.win.update_idletasks()
        x = root.winfo_rootx() + root.winfo_width() // 2 - WIDTH // 2
        y = (root.winfo_rooty() + root.winfo_height() // 2
             - self.win.winfo_reqheight() // 2)
        self.win.geometry(f'+{max(x, 0)}+{max(y, 0)}')
        self._tick()
        self.win.update()

    def report(self, key, done=0, total=0, unit='', note='', eta=True,
               frac=None):
        """Show phase ``key`` at ``done`` of ``total`` ``unit``. Tk thread.
        ``eta`` off for items of very unequal length, whose rate predicts
        nothing. ``frac`` is how far the whole phase is when the count is
        only one of its steps' (a fit placing signals, then fitting each
        sample); the bar follows it and never goes back within a phase,
        while the text shows the step's own count."""
        if self._closed:
            return
        # a late update queued by an earlier phase must not move the bar back
        if (self._phase is not None
                and self.phases.index(key) < self.phases.index(self._phase)):
            return
        now = time.monotonic()
        if key != self._phase:
            self._phase, self._phase_t0, self._frac = key, now, 0.0
        f = frac if frac is not None else (done / total if total else 0.0)
        self._frac = max(self._frac, min(max(float(f), 0.0), 1.0))
        self._shown = (key, done, total, unit, note, eta, frac)
        n = self.phases.index(key) + 1
        step = f'Step {n} of {len(self.phases)} — {self.table[key][0]}'
        if self.cancelled:
            step = 'Cancelling — stopping at the next safe point…'
        self._step_lbl.config(text=step)
        self._bar['value'] = phase_percent(self.table, self.phases, key,
                                           self._frac)
        parts = []
        if total:
            parts.append(f'{done} of {total} {unit}'.strip())
            left = (time_left(now - self._phase_t0, self._frac, 1.0)
                    if eta else None)
            if left is not None:
                parts.append(f'about {format_seconds(left)} left')
        if note:
            parts.append(note)
        self._detail_lbl.config(text='  ·  '.join(parts))
        self.win.update_idletasks()

    def post(self, key, done=0, total=0, unit='', note='', frac=None):
        """``report`` from any thread: the state is only stored, and the
        window shows the latest at its next tick. Nothing here calls Tk, so a
        worker never waits on the main loop and a loop over thousands of
        items costs nothing."""
        self._pending = (key, done, total, unit, note, True, frac)

    def cancel(self):
        """Ask the run to stop. Tk thread; the worker sees ``cancelled``."""
        if self.cancelled or self._closed:
            return
        self.cancelled = True
        if self._cancel_btn is not None:
            self._cancel_btn.config(state='disabled')
        self._step_lbl.config(text='Cancelling — stopping at the next safe '
                                   'point…')
        if self._on_cancel:
            self._on_cancel()

    def _tick(self):
        if self._closed:
            return
        pending = self._pending
        if pending is not None and pending is not self._taken:
            self._taken = pending
            self.report(*pending)
        elif self._shown is not None:
            self.report(*self._shown)       # the estimate follows the clock
        self._time_lbl.config(
            text=f'Elapsed {format_seconds(time.monotonic() - self._t0)}')
        self._tick_id = self.win.after(TICK_MS, self._tick)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._tick_id is not None:
                self.win.after_cancel(self._tick_id)
            self.win.grab_release()
            self.win.destroy()
        except tk.TclError:
            pass
