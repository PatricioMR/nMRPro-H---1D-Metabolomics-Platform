# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Opt-in profiling of the slow phases, written where the other logs go.

Switched on by the environment variable ``NMRPRO_PROFILE`` or by an empty
file named ``PROFILE`` in the user's ``logs`` directory -- the second is for
an analyst running the packaged app, who has no shell to set a variable in.
Off, the context manager costs one ``os.path.exists``.
"""
import cProfile
import io
import os
import pstats
import time
from contextlib import contextmanager

from app_paths import user_data_dir

FLAG_FILE = 'PROFILE'
TOP_N = 60


def _logs_dir() -> str:
    d = os.path.join(user_data_dir(), 'logs')
    os.makedirs(d, exist_ok=True)
    return d


def enabled() -> bool:
    if os.environ.get('NMRPRO_PROFILE'):
        return True
    try:
        return os.path.exists(os.path.join(_logs_dir(), FLAG_FILE))
    except OSError:
        return False


@contextmanager
def profiled(label: str):
    """Profile the block as ``profile_<label>.txt`` when profiling is on.

    The report lists the callees by cumulative time and then by own time,
    so a phase that is slow because of one deep call and one that is slow
    because of a million cheap ones both show their cause.
    """
    if not enabled():
        yield
        return
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    try:
        yield
    finally:
        prof.disable()
        elapsed = time.perf_counter() - t0
        buf = io.StringIO()
        buf.write(f'{label}: {elapsed:.2f} s wall\n\n')
        for sort in ('cumulative', 'tottime'):
            buf.write(f'=== by {sort} ===\n')
            pstats.Stats(prof, stream=buf).sort_stats(sort).print_stats(TOP_N)
            buf.write('\n')
        path = os.path.join(_logs_dir(), f'profile_{label}.txt')
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(buf.getvalue())
        except OSError:
            pass
