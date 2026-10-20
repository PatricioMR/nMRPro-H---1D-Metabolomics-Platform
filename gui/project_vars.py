# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""Every parameter a stage holds in a tkinter variable, saved and restored.

A project that reopens on other settings than it was saved with gives other
results, and says nothing: a checkbox nobody listed in the saving code comes
back at its default. So the project does not keep a list of the variables it
knows about; it keeps every variable the stages have -- found by looking at
them -- and says out loud which ones it could not put back.

A variable is anything with ``get``, ``set`` and ``trace_add`` (a
``tk.Variable``). One level of dict or list of them is followed
(``{name: var}``, ``[var, var]``), which is how the stages keep the families
of options.
"""
from __future__ import annotations

# Variables that are a view of others: restoring the others restores them, and
# setting them first would send the change back through the trace and onto the
# very variables the project is about to set.
DERIVED = frozenset({'baseline_choice_var'})


def _is_variable(obj) -> bool:
    return (callable(getattr(obj, 'get', None))
            and callable(getattr(obj, 'set', None))
            and callable(getattr(obj, 'trace_add', None)))


def _plain(value):
    """What a variable holds, as a plain value a pickle keeps."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def variables_of(obj) -> dict:
    """``{name: variable}`` of ``obj``: its attributes that are variables, and
    ``attr[key]`` / ``attr[i]`` for the ones in a dict or list under one."""
    out = {}
    for name, val in vars(obj).items():
        if _is_variable(val):
            out[name] = val
        elif isinstance(val, dict):
            for k, v in val.items():
                if _is_variable(v):
                    out[f'{name}[{k!r}]'] = v
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                if _is_variable(v):
                    out[f'{name}[{i}]'] = v
    return out


def collect_vars(obj) -> dict:
    """``{name: value}`` of every variable ``obj`` holds. One that cannot be
    read (its widget is gone) is left out, not recorded as a value."""
    out = {}
    for name, var in variables_of(obj).items():
        try:
            out[name] = _plain(var.get())
        except Exception:
            continue
    return out


def apply_vars(obj, saved: dict) -> dict:
    """Put ``saved`` back on ``obj``'s variables. Returns what could not be:
    ``{'missing': [names with no variable (yet)], 'failed': {name: error}}``.

    A value is set only when it differs, so a setting already right does not
    fire the traces that follow it. Names in ``DERIVED`` are skipped.
    """
    have = variables_of(obj)
    missing, failed = [], {}
    for name, value in (saved or {}).items():
        if name in DERIVED:
            continue
        var = have.get(name)
        if var is None:
            missing.append(name)
            continue
        try:
            if _plain(var.get()) != value:
                var.set(value)
        except Exception as e:
            failed[name] = f'{type(e).__name__}: {e}'
    return {'missing': missing, 'failed': failed}


def diff_vars(before: dict, after: dict) -> dict:
    """``{name: (before, after)}`` for the names in both whose values differ,
    derived variables left out."""
    return {k: (before[k], after[k]) for k in before
            if k in after and k not in DERIVED and before[k] != after[k]}


def engine_fingerprint() -> str | None:
    """A short hash of the code that makes the numbers: the processing, the
    adapter and the defaults. Saved with a project so that opening it on other
    code can say the saved results are an older engine's. None where the
    sources are not there (a packaged build)."""
    import hashlib
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = []
    for rel in ('processing', 'controllers', os.path.join('gui', 'stages')):
        d = os.path.join(root, rel)
        if os.path.isdir(d):
            files += sorted(os.path.join(d, f) for f in os.listdir(d)
                            if f.endswith('.py') and (rel != os.path.join('gui', 'stages')
                                                      or f.startswith('_quant_')))
    if not files:
        return None
    h = hashlib.sha1()
    try:
        for fp in files:
            with open(fp, 'rb') as f:
                h.update(f.read().replace(b'\r\n', b'\n'))
    except OSError:
        return None
    return h.hexdigest()[:12]


def apply_pending(obj) -> None:
    """Set the variables a stage builds late. ``apply_vars`` could not find
    them when the project was opened, so it left them in ``obj._pending_vars``;
    a stage calls this when it has built them."""
    pending = getattr(obj, '_pending_vars', None)
    if not pending:
        return
    res = apply_vars(obj, pending)
    obj._pending_vars = {k: pending[k] for k in res['missing']}
