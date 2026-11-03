# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The annotation run's progress window: which step, how far, and only ever
forward."""
import pytest

from gui.widgets.annotation_progress import run_phases

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


def _texts(prog):
    return (prog._step_lbl.cget('text'), prog._detail_lbl.cget('text'))


def test_the_window_names_the_step_and_counts_the_items(root):
    from gui.widgets.annotation_progress import AnnotationProgress
    prog = AnnotationProgress(root, run_phases())
    prog.report('per_sample', 120, 679, unit='signals')
    step, detail = _texts(prog)
    assert step == 'Step 2 of 4 — Reading each sample'
    assert detail.startswith('120 of 679 signals')
    prog.close()
    prog.close()                                   # twice is harmless
    prog.report('scoring', 1, 2)                   # after close: ignored


def test_progress_never_goes_back_a_phase(root):
    from gui.widgets.annotation_progress import AnnotationProgress
    prog = AnnotationProgress(root, run_phases())
    prog.report('scoring', 3, 12)
    before = prog._bar['value']
    prog.report('per_sample', 600, 679)            # a late, queued update
    assert prog._bar['value'] == before
    assert 'Scoring' in _texts(prog)[0]
    prog.close()


def test_the_windows_last_pass_starts_building_the_interface(root):
    from gui.widgets.annotation_progress import AnnotationProgress
    prog = AnnotationProgress(root, ('scoring', 'interface'))
    cb = prog.step_callback('scoring', then='interface')
    cb(7, 12, 'Scoring each signal')
    step, detail = _texts(prog)
    assert step.startswith('Step 1 of 2') and '7 of 12 passes' in detail
    assert 'left' not in detail                    # passes are not equal
    cb(12, 12, '315 metabolites')
    step, detail = _texts(prog)
    assert step == 'Step 2 of 2 — Building the results window'
    assert detail == '315 metabolites'
    prog.close()


def test_a_worker_post_touches_no_tk_and_the_tick_shows_the_latest(root):
    from gui.widgets.annotation_progress import AnnotationProgress
    prog = AnnotationProgress(root, run_phases())
    calls = []
    prog.root = type('R', (), {'after': lambda _s, *a: calls.append(a)})()
    for i in range(1000):
        prog.post('per_sample', i, 1000, unit='signals', note=f'signal {i}')
    prog.post('per_sample', 1000, 1000, unit='signals', note='last')
    assert calls == []                       # a worker never waits on Tk
    prog._tick()
    assert _texts(prog)[1].startswith('1000 of 1000 signals')
    assert _texts(prog)[1].endswith('last')
    prog.close()


def test_a_slow_item_makes_the_estimate_grow_not_freeze(root, monkeypatch):
    from gui.widgets import annotation_progress as mod, run_progress
    now = [100.0]
    monkeypatch.setattr(run_progress.time, "monotonic", lambda: now[0])
    prog = mod.AnnotationProgress(root, run_phases())
    prog.report('per_sample', 0, 100, unit='signals')
    now[0] += 10.0
    prog.post('per_sample', 50, 100, unit='signals', note='L-citrulline')
    prog._tick()
    first = _texts(prog)[1]
    assert 'about 10 s left' in first and 'L-citrulline' in first
    now[0] += 30.0                            # stuck on the same signal
    prog._tick()
    assert 'about 40 s left' in _texts(prog)[1]
    prog.close()


def test_the_bar_never_goes_back_within_a_step(root):
    """A step whose count restarts (a fit placing signals, then fitting each
    sample) keeps its bar where it got to."""
    from gui.widgets.run_progress import RunProgress
    table = {'a': ('A', 1), 'b': ('B', 1)}
    prog = RunProgress(root, ('a', 'b'), table, 'test')
    prog.report('a', 9, 10)
    high = prog._bar['value']
    prog.report('a', 1, 32)
    assert prog._bar['value'] == high
    prog.report('a', 1, 32, frac=0.95)
    assert prog._bar['value'] > high
    prog.report('b')
    assert prog._bar['value'] == 50.0
    prog.close()


def test_everything_stays_inside_the_window_when_the_text_grows(root):
    """A long step and a long note wrap and the window grows with them: the
    Cancel button at the bottom is never cut off."""
    from gui.widgets.run_progress import RunProgress
    root.deiconify()                # a transient of a hidden root is not drawn
    root.geometry('900x600+50+50')
    table = {'a': ('A step whose title is long enough to need a second line '
                   'in the window', 1)}
    prog = RunProgress(root, ('a',), table, 'test', subtitle='sub',
                       on_cancel=lambda: None)
    prog.report('a', 12, 32, 'samples', 'round 2, without 3 the fingerprint '
                'contradicts · joint fit (B) · fitting each sample · and a '
                'note long enough to wrap onto a third line of its own')
    for _ in range(5):
        root.update()
    w = prog.win
    assert w.winfo_height() >= w.winfo_reqheight()
    for child in w.winfo_children():
        assert child.winfo_y() + child.winfo_height() <= w.winfo_height()
        assert child.winfo_x() + child.winfo_width() <= w.winfo_width()
    prog.close()
