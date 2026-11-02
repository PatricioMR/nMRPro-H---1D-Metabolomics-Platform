# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The progress window of an annotation run (gui.widgets.run_progress).

A run is a chain of phases of very different length -- the database search,
reading every sample, MCR when asked for, the scoring passes, the window --
and the analyst has to see which one is running, how far it is and that the
program is alive. Workers post to it with ``post``, which hops to the Tk
thread; everything else here belongs to that thread.
"""
from gui.widgets import run_progress
from gui.widgets.run_progress import (  # noqa: F401  (re-exported)
    RunProgress, format_seconds, time_left)

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


def run_phases(search=True, mcr=False):
    """The phases one run goes through, in order."""
    return ((('search',) if search else ()) + ('per_sample',)
            + (('mcr',) if mcr else ()) + ('scoring', 'interface'))


def bar_percent(phases, key, done, total):
    """Overall percent of the run when phase ``key`` has done ``done`` of
    ``total`` items; a phase without a count sits at its start."""
    return run_progress.bar_percent(PHASES, phases, key, done, total)


class AnnotationProgress(RunProgress):
    """The annotation's phases in the run window. It cannot be cancelled:
    closing it would leave the worker writing to a window that is gone."""

    def __init__(self, root, phases, title='NMRPro-H — Annotating',
                 subtitle=''):
        super().__init__(root, phases, PHASES, title, subtitle)

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
