# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Annotating must not leave the program "not responding".

After the search, reading every sample (16 s on the 12-sample bench study,
linear in the samples) and MCR (48 s more) ran on the Tk thread with no event
processed, and Windows marked the program frozen. They run on a worker now;
the Tk thread only opens the window when they are done.
"""
import threading
import time

import pytest

tk = pytest.importorskip('tkinter')


@pytest.fixture
def root():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    r.withdraw()
    yield r
    r.destroy()


def _stage(root, calls, work_s):
    from gui.stages._annotation_search import _AnnotationSearchMixin

    class Stage(_AnnotationSearchMixin):
        def _measure_results_per_sample(self, results, mhz, *, tolerance_ppm,
                                        min_snr, progress=None):
            calls['measure'] = (threading.current_thread(), tolerance_ppm,
                                min_snr)
            for i in range(10):
                progress(i, 10, f'signal {i}')
                time.sleep(work_s / 10)
            progress(10, 10, '')

        def _run_mcr_on_results(self, results, mhz, progress=None):
            calls['mcr'] = threading.current_thread()
            return 0

        def _open_results_window(self, results, prog, total_peaks, db_path,
                                 elapsed, then=None):
            calls['open'] = threading.current_thread()
            prog.close()

    s = Stage()
    s.app = type('A', (), {'root': root})()
    s._calls = calls
    return s


def _loop_until(root, start, done, timeout_s=10):
    """Run the Tk main loop, as the application does, start the work from
    inside it, and leave once ``done()`` -- a worker can only post to Tk
    while the main thread is in ``mainloop``."""
    deadline = time.monotonic() + timeout_s

    def _poll():
        if done() or time.monotonic() > deadline:
            root.quit()
        else:
            root.after(10, _poll)
    root.after(0, start)
    root.after(10, _poll)
    root.mainloop()


def _run(root, stage, settings):
    from gui.widgets.annotation_progress import AnnotationProgress, run_phases
    calls = stage._calls
    ticks = []

    def _tick():
        ticks.append(time.monotonic())
        if 'open' not in calls:
            root.after(20, _tick)

    def _start():
        prog = AnnotationProgress(root, run_phases(mcr=settings['run_mcr']))
        root.after(20, _tick)
        stage._finish_in_background([{'peak_rank': 1}], settings, prog, 1,
                                    'x.db', 1.0)
    _loop_until(root, _start, lambda: 'open' in calls)
    return ticks


SETTINGS = {'mhz': 600.0, 'tolerance': 0.02, 'min_snr': 6.0,
            'min_distance': 0.004, 'run_mcr': False}


def test_reading_the_samples_runs_off_the_tk_thread(root):
    calls = {}
    stage = _stage(root, calls, 0.6)
    ticks = _run(root, stage, SETTINGS)
    main = threading.main_thread()
    assert calls['measure'][0] is not main
    assert calls['open'] is main
    # the panel values arrive as arguments, read before the worker started
    assert calls['measure'][1:] == (0.02, 6.0)
    # and the event loop kept turning while the samples were read
    assert len(ticks) >= 10
    assert 'mcr' not in calls


def test_mcr_when_asked_runs_on_the_worker_too(root):
    calls = {}
    stage = _stage(root, calls, 0.1)
    _run(root, stage, dict(SETTINGS, run_mcr=True))
    assert calls['mcr'] is not threading.main_thread()


def test_a_failure_on_the_worker_is_shown_not_swallowed(root, monkeypatch):
    from gui.stages import _annotation_search as mod
    calls, shown = {}, []
    stage = _stage(root, calls, 0.0)

    def _boom(*a, **k):
        raise RuntimeError('fit diverged')
    stage._measure_results_per_sample = _boom
    stage._status_err = lambda m: shown.append(('status', m))
    monkeypatch.setattr(mod.messagebox, 'showerror',
                        lambda title, msg: shown.append(('box', msg)))
    from gui.widgets.annotation_progress import AnnotationProgress, run_phases
    progs = []

    def _start():
        progs.append(AnnotationProgress(root, run_phases()))
        stage._finish_in_background([{'peak_rank': 1}], SETTINGS, progs[0], 1,
                                    'x.db', 1.0)
    _loop_until(root, _start, lambda: len(shown) >= 2, timeout_s=5)
    assert any('fit diverged' in m for _k, m in shown)
    assert progs[0]._closed and 'open' not in calls
