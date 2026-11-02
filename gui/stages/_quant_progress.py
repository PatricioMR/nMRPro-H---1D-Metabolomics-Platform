# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The quantification run's phases in the progress window, and the bridge
from the engine's progress reports to the window and its Cancel button.

The phases follow the paper: the baseline workflow (election and windows),
Strategy A (shared windows and complex multiplets by shape), Strategy B
(the joint fit), Strategy C (the residual cascade), then concentrations
and confidence. A study with analytes accepted in the deconvolution review
is measured twice (quant_engine._accepted_alone), the second time only for
those analytes.
"""
import threading

# key -> (title, share of the bar). The shares are the wall times measured on
# the Mice ISS study (32 samples, 324 annotated, 2026-11-02): one pass 534 s
# with acceptances, the joint fit 160 s a pass, the cascade about 60.
PHASES = {
    'prepare':  ('Preparing the candidates', 4),
    'windows':  ('Baseline workflow: electing signals, integrating windows', 12),
    'shapes':   ('Strategy A: shared windows and complex multiplets', 12),
    'joint':    ('Strategy B: joint fit of overlapping analytes', 160),
    'recheck':  ('Fingerprint check: measuring again without the '
                 'contradicted', 12),
    'cascade':  ('Strategy C: residual cascade', 60),
    'accepted': ('Measuring the analytes accepted in the review', 250),
    'results':  ('Concentrations, confidence and the table', 8),
}


def run_phases(*, joint=True, cascade=True, accepted=False):
    """The phases one run goes through, in order."""
    return (('prepare', 'windows', 'shapes')
            + (('joint',) if joint else ()) + ('recheck',)
            + (('cascade',) if cascade else ())
            + (('accepted',) if accepted else ()) + ('results',))


class RunCancelled(Exception):
    """The analyst stopped the run; nothing it computed is kept."""


class EngineProgress:
    """The engine's ``progress(phase, done, total, unit, note)``: posts to
    the window and, once Cancel is pressed, raises RunCancelled at the
    engine's next report. Called from the worker thread."""

    def __init__(self, phases):
        self.phases = tuple(phases)
        self.window = None
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def __call__(self, phase, done=0, total=0, unit='', note=''):
        if self._cancel.is_set():
            raise RunCancelled()
        if self.window is not None and phase in self.phases:
            self.window.post(phase, done, total, unit, note)
