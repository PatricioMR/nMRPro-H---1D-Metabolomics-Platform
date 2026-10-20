# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""A project saves every variable its stages hold, found by looking at them, and
says which it could not put back. A parameter nobody listed in the saving code
used to come back at its default, and the numbers changed with no warning."""
from gui.project_vars import (
    DERIVED, apply_pending, apply_vars, collect_vars, diff_vars,
    engine_fingerprint, variables_of)


class _Var:
    """What a tk.Variable offers: get, set and trace_add."""

    def __init__(self, value):
        self.value = value
        self.sets = 0

    def get(self):
        return self.value

    def set(self, v):
        self.value = v
        self.sets += 1

    def trace_add(self, *a, **k):
        pass


class _Broken(_Var):
    def set(self, v):
        raise ValueError('not a number')


class _Stage:
    pass


def _stage():
    s = _Stage()
    s.cascade_var = _Var(True)
    s.singlet_var = _Var(5.0)
    s.opts = {'a': _Var(1), 'b': _Var('x'), 'plain': 3}
    s.levels = [_Var(0.1), _Var(0.2)]
    s.title = 'not a variable'
    return s


def test_every_variable_is_found_including_those_in_a_dict_or_a_list():
    assert set(variables_of(_stage())) == {
        'cascade_var', 'singlet_var', "opts['a']", "opts['b']",
        'levels[0]', 'levels[1]'}


def test_the_values_are_collected_by_name():
    assert collect_vars(_stage()) == {
        'cascade_var': True, 'singlet_var': 5.0, "opts['a']": 1,
        "opts['b']": 'x', 'levels[0]': 0.1, 'levels[1]': 0.2}


def test_a_variable_that_cannot_be_read_is_left_out_not_recorded_as_a_value():
    s = _stage()

    class _Gone(_Var):
        def get(self):
            raise RuntimeError('widget destroyed')
    s.gone_var = _Gone(1)
    assert 'gone_var' not in collect_vars(s)


def test_what_was_saved_is_put_back_and_only_what_differs_is_set():
    s = _stage()
    saved = collect_vars(s)
    s.cascade_var.value = False
    s.singlet_var.value = 4.0
    for v in (s.cascade_var, s.singlet_var, s.opts['a'], s.opts['b']):
        v.sets = 0
    res = apply_vars(s, saved)
    assert res == {'missing': [], 'failed': {}}
    assert s.cascade_var.value is True and s.singlet_var.value == 5.0
    assert s.cascade_var.sets == 1 and s.opts['a'].sets == 0, \
        'a setting already right is not set again: its traces would fire'


def test_a_variable_the_stage_does_not_have_is_missing_not_an_error():
    res = apply_vars(_stage(), {'built_later_var': 1})
    assert res == {'missing': ['built_later_var'], 'failed': {}}


def test_a_variable_that_refuses_the_value_is_reported():
    s = _stage()
    s.bad_var = _Broken(1)
    res = apply_vars(s, {'bad_var': 2})
    assert list(res['failed']) == ['bad_var'] and 'not a number' in res['failed']['bad_var']


def test_a_derived_variable_is_not_set_ahead_of_the_ones_it_is_made_from():
    s = _stage()
    s.baseline_choice_var = _Var('None')
    assert 'baseline_choice_var' in DERIVED
    apply_vars(s, {'baseline_choice_var': 'Local per window'})
    assert s.baseline_choice_var.value == 'None'
    assert 'baseline_choice_var' not in diff_vars({'baseline_choice_var': 1},
                                                  {'baseline_choice_var': 2})


def test_a_difference_is_said_by_name_with_both_values():
    before = {'local_baseline_var': False, 'singlet_snr_var': 5.0, 'only_before': 1}
    after = {'local_baseline_var': True, 'singlet_snr_var': 5.0, 'only_after': 2}
    assert diff_vars(before, after) == {'local_baseline_var': (False, True)}


def test_variables_a_stage_builds_late_wait_and_are_set_when_it_does():
    s = _stage()
    res = apply_vars(s, {'late_var': 7})
    s._pending_vars = {k: 7 for k in res['missing']}
    apply_pending(s)
    assert s._pending_vars == {'late_var': 7}, 'still not built'
    s.late_var = _Var(0)
    apply_pending(s)
    assert s.late_var.value == 7 and s._pending_vars == {}


def test_the_engine_fingerprint_is_stable_and_short():
    a, b = engine_fingerprint(), engine_fingerprint()
    assert a == b and a is not None and len(a) == 12
