# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The one progress window of an annotation run.

A run is a chain of phases of very different length -- the database search,
reading every sample, MCR when asked for, the scoring passes, the window --
and the analyst has to see which one is running, how far it is and that the
program is alive. Workers post to it with ``post``, which hops to the Tk
thread; everything else here belongs to that thread.
"""
import time
import tkinter as tk
from tkinter import ttk

# key -> (title, share of the bar). The shares are the wall times measured on
# the mixture bench study (12 samples, 90 signals, BMRB Mammalian,
# 2026-11-01): search 25 s, per-sample reading 16 s, MCR 48 s, scoring
# passes 11 s; the window itself takes about a second.
PHASES = {
    'search':     ('Searching the database', 25),
    'per_sample': ('Reading each sample', 16),
    'mcr':        ('MCR deconvolution', 48),
    'scoring':    ('Scoring the candidates', 11),
    'interface':  ('Building the results window', 2),
}

TICK_MS = 100


def run_phases(search=True, mcr=False):
    """The phases one run goes through, in order."""
    return ((('search',) if search else ()) + ('per_sample',)
            + (('mcr',) if mcr else ()) + ('scoring', 'interface'))


def bar_percent(phases, key, done, total):
    """Overall percent of the run when phase ``key`` has done ``done`` of
    ``total`` items; a phase without a count sits at its start."""
    weights = [PHASES[k][1] for k in phases]
    i = list(phases).index(key)
    frac = min(max(done / total, 0.0), 1.0) if total else 0.0
    return 100.0 * (sum(weights[:i]) + frac * weights[i]) / sum(weights)


def time_left(elapsed, done, total):
    """Seconds the phase still needs at the rate it has kept so far, or None
    while that rate says nothing yet (under 2 s, or no item done)."""
    if not total or done <= 0 or done >= total or elapsed < 2.0:
        return None
    return elapsed / done * (total - done)


def format_seconds(s):
    s = int(round(s))
    return f'{s} s' if s < 60 else f'{s // 60} min {s % 60:02d} s'


class AnnotationProgress:
    """Modal progress window: step, bar, item count with the time left in the
    step, and the run's elapsed time. It redraws itself every TICK_MS, so a
    long item still shows the clock moving and its estimate growing instead of
    a frozen "about 4 s left"."""

    def __init__(self, root, phases, title='NMRPro-H — Annotating',
                 subtitle=''):
        self.root = root
        self.phases = tuple(phases)
        self._t0 = time.monotonic()
        self._phase = None
        self._phase_t0 = self._t0
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
        # closing it would leave the worker writing to a window that is gone;
        # the run ends by itself
        self.win.protocol('WM_DELETE_WINDOW', lambda: None)
        root.update_idletasks()
        x = root.winfo_rootx() + root.winfo_width() // 2 - 240
        y = root.winfo_rooty() + root.winfo_height() // 2 - 80
        self.win.geometry(f'480x160+{x}+{y}')

        if subtitle:
            ttk.Label(self.win, text=subtitle, font=('Arial', 9),
                      foreground='gray').pack(pady=(10, 0))
        self._step_lbl = ttk.Label(self.win, text='Starting…',
                                   font=('Arial', 10, 'bold'))
        self._step_lbl.pack(pady=(6, 4))
        self._bar = ttk.Progressbar(self.win, length=440, mode='determinate',
                                    maximum=100)
        self._bar.pack(padx=20)
        self._detail_lbl = ttk.Label(self.win, text='', font=('Arial', 9))
        self._detail_lbl.pack(pady=(4, 0))
        self._time_lbl = ttk.Label(self.win, text='', font=('Arial', 8),
                                   foreground='gray')
        self._time_lbl.pack(pady=(2, 8))
        self._tick()
        self.win.update()

    def report(self, key, done=0, total=0, unit='', note='', eta=True):
        """Show phase ``key`` at ``done`` of ``total`` ``unit``. Tk thread.
        ``eta`` off for items of very unequal length, whose rate predicts
        nothing."""
        if self._closed:
            return
        # a late update queued by an earlier phase must not move the bar back
        if (self._phase is not None
                and self.phases.index(key) < self.phases.index(self._phase)):
            return
        self._shown = (key, done, total, unit, note, eta)
        now = time.monotonic()
        if key != self._phase:
            self._phase, self._phase_t0 = key, now
        n = self.phases.index(key) + 1
        self._step_lbl.config(
            text=f'Step {n} of {len(self.phases)} — {PHASES[key][0]}')
        self._bar['value'] = bar_percent(self.phases, key, done, total)
        parts = []
        if total:
            parts.append(f'{done} of {total} {unit}'.strip())
            left = time_left(now - self._phase_t0, done, total) if eta else None
            if left is not None:
                parts.append(f'about {format_seconds(left)} left')
        if note:
            parts.append(note)
        self._detail_lbl.config(text='  ·  '.join(parts))
        self.win.update_idletasks()

    def post(self, key, done=0, total=0, unit='', note=''):
        """``report`` from any thread: the state is only stored, and the
        window shows the latest at its next tick. Nothing here calls Tk, so a
        worker never waits on the main loop and a loop over thousands of
        items costs nothing."""
        self._pending = (key, done, total, unit, note)

    def step_callback(self, key, then=None):
        """A ``progress_cb(step, total, msg)`` for the results window: its
        steps are phase ``key``, and its last call (step == total, the
        interface being built) starts phase ``then``."""
        def _cb(step, total, msg):
            if then is not None and total and step >= total:
                self.report(then, note=msg)
            else:
                self.report(key, step, total, unit='passes', note=msg,
                            eta=False)
        return _cb

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
