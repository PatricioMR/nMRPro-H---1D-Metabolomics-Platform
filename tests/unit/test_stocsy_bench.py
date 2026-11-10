# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
# tests/unit/test_stocsy_bench.py
"""The STOCSY bench's figures: dots before and after, what rose, the imported
and the full runs, and the jobs left to run."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pytest

from validation import stocsy_bench as sb
from validation.validation_folder import subfolder

TRUTH = {'a': {}, 'b': {}, 'd': {}, 'e': {}}


def _row(hid, conf, lift=None, **kw):
    r = {'hmdb_id': hid, 'metabolite': hid.upper(), '_confidence': conf}
    if lift:
        r['_stocsy_lift'] = lift
    r.update(kw)
    return r


def _rows():
    return [_row('a', 2, {'from': 1, 'pairs': 3}),       # real, 1 -> 2
            _row('b', 3, {'from': 2, 'pairs': 2}),       # real, 2 -> 3
            _row('c', 1),                                 # false, stays
            _row('d', 3),                                 # real, stays
            _row('x', 2, {'from': 1, 'pairs': 2}),       # false, 1 -> 2
            _row('is', 3, _is_is=True),
            _row('e', 1, _not_quantified=True)]


class TestSlimRows:

    def test_before_is_the_dot_the_lift_started_from_and_after_the_one_it_gave(self):
        slim = {r['hmdb_id']: r for r in sb.slim_rows(_rows(), TRUTH)}
        assert (slim['a']['before'], slim['a']['after'], slim['a']['pairs']) == (1, 2, 3)
        assert (slim['c']['before'], slim['c']['after'], slim['c']['pairs']) == (1, 1, 0)

    def test_the_standard_and_the_refused_are_not_reported(self):
        assert {r['hmdb_id'] for r in sb.slim_rows(_rows(), TRUTH)} == {'a', 'b', 'c', 'd', 'x'}

    def test_real_means_in_the_tube(self):
        slim = {r['hmdb_id']: r['real'] for r in sb.slim_rows(_rows(), TRUTH)}
        assert slim == {'a': True, 'b': True, 'c': False, 'd': True, 'x': False}


class TestComparable:

    def test_the_order_of_the_rows_does_not_make_two_runs_differ(self):
        slim = sb.slim_rows(_rows(), TRUTH)
        assert sb._comparable(slim) == sb._comparable(list(reversed(slim)))

    def test_a_dot_that_changed_does_make_them_differ(self):
        rows = _rows()
        other = [dict(r) for r in rows]
        other[0]['_confidence'] = 3
        assert sb._comparable(sb.slim_rows(rows, TRUTH)) != sb._comparable(sb.slim_rows(other, TRUTH))


class TestFigures:

    def _f(self):
        return sb.figures(sb.slim_rows(_rows(), TRUTH), len(TRUTH))

    def test_found_and_false(self):
        f = self._f()
        assert (f['in_tube'], f['found'], f['false']) == (4, 3, 2)

    def test_each_level_before_and_after(self):
        f = self._f()
        assert (f['real3_before'], f['real3_after']) == (1, 2)
        assert (f['real2_before'], f['real2_after']) == (1, 1)
        assert (f['real1_before'], f['real1_after']) == (1, 0)
        assert (f['false1_before'], f['false1_after']) == (2, 1)
        assert (f['false2_before'], f['false2_after']) == (0, 1)

    def test_two_dots_or_more_and_the_change(self):
        f = self._f()
        assert (f['real2up_before'], f['real2up_after'], f['d_real2up']) == (2, 3, 1)
        assert (f['false2up_before'], f['false2up_after'], f['d_false2up']) == (0, 1, 1)

    def test_what_rose_is_counted_real_and_false_apart(self):
        f = self._f()
        assert (f['up_real'], f['up_false']) == (2, 1)

    def test_precision_per_level_and_nan_when_a_level_is_empty(self):
        f = self._f()
        assert f['prec3_before'] == 100.0 and f['prec3_after'] == 100.0
        assert f['prec2_after'] == 50.0
        assert f['prec2_before'] == 100.0

    def test_a_run_that_reported_nothing_does_not_fail(self):
        f = sb.figures([], 18)
        assert f['found'] == 0 and f['up_real'] == 0 and np.isnan(f['prec3_after'])

    @pytest.mark.parametrize('conf,level', [(0, 1), (None, 1), (1, 1), (3, 3), (5, 3)])
    def test_the_dot_is_clamped_to_one_to_three(self, conf, level):
        assert sb._level(conf) == level


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, 'OUT', str(tmp_path))
    return tmp_path


def _legacy(path, n_truth=4):
    slim = sb.slim_rows(_rows(), TRUTH)
    with open(path, 'wb') as f:
        pickle.dump({'rows': None, 'slim': slim, 'n_truth': n_truth}, f)


def _full(path):
    with open(path, 'wb') as f:
        pickle.dump({'rows': _rows(), 'truth': TRUTH}, f)


class TestStore:

    def test_an_imported_run_is_not_done_but_a_full_one_is(self, out):
        _legacy(sb._path(600, 'padrao', 18, 0))
        _full(sb._path(600, 'padrao', 18, 1))
        assert not sb._done(sb._path(600, 'padrao', 18, 0))
        assert sb._done(sb._path(600, 'padrao', 18, 1))
        assert not sb._done(sb._path(600, 'padrao', 18, 2))

    def test_the_map_says_which_seeds_ran(self, out):
        _legacy(sb._path(600, 'padrao', 18, 0))
        _full(sb._path(600, 'padrao', 18, 1))
        row = next(r for r in sb.map_rows() if (r['campo_MHz'], r['regime'], r['tubo']) == (600, 'padrao', 18))
        assert (row['semente 0'], row['semente 1'], row['semente 2']) == ('importada', 'feita', '—')

    def test_a_cell_is_mean_and_sd_over_the_seeds_not_a_sum(self, out):
        _legacy(sb._path(600, 'padrao', 18, 0))
        _full(sb._path(600, 'padrao', 18, 1))
        row = next(r for r in sb.summary_rows() if (r['campo_MHz'], r['tubo']) == (600, 18))
        assert row['corridas'] == 2
        assert row['achados'] == '3.0 ± 0.0' and row['●●● reais antes'] == '1.0 ± 0.0'
        assert row['●●● reais depois'] == '2.0 ± 0.0'

    def test_import_copies_the_scratch_runs_as_imported(self, out, tmp_path):
        src = tmp_path / 'scratch'
        src.mkdir()
        with open(src / 'lift_padrao_18_0.pkl', 'wb') as f:
            pickle.dump({'rows': sb.slim_rows(_rows(), TRUTH), 'n_truth': 4}, f)
        sb.import_legacy(argparse.Namespace(source=str(src)))
        assert not sb._done(sb._path(600, 'padrao', 18, 0))
        assert sb.run_rows()[0]['tabelas_por_amostra'].startswith('não')


class TestExport:

    def test_a_folder_with_no_full_run_still_gets_its_workbook_and_readme(self, out, tmp_path):
        _legacy(sb._path(600, 'padrao', 18, 0))
        workbook = tmp_path / 'STOCSY_NMRPro-H.xlsx'
        workbook.write_bytes(b'x')
        dest = tmp_path / 'validation'
        sb.export_runs(str(dest), str(workbook))
        root = Path(subfolder(str(dest), 'stocsy'))
        assert (root / 'STOCSY_NMRPro-H.xlsx').exists() and (root / 'LEIA-ME.md').exists()
        assert 'corridas: 0' in (root / 'versao.txt').read_text(encoding='utf-8')

    def test_the_export_replaces_the_folder_whole(self, out, tmp_path):
        workbook = tmp_path / 'STOCSY_NMRPro-H.xlsx'
        workbook.write_bytes(b'x')
        old = Path(subfolder(str(tmp_path / 'validation'), 'stocsy')) / 'old.txt'
        old.parent.mkdir(parents=True)
        old.write_text('stale')
        sb.export_runs(str(tmp_path / 'validation'), str(workbook))
        assert not old.exists()


class TestJobs:

    def _args(self, **kw):
        base = dict(tubes=[18], fields=[600], regimes=['padrao'], seeds=[0, 1, 2],
                    keep_imported=False, election=False)
        return argparse.Namespace(**{**base, **kw})

    def test_missing_runs_first_then_the_imported_ones_redone(self, out):
        _legacy(sb._path(600, 'padrao', 18, 0))
        _full(sb._path(600, 'padrao', 18, 2))
        jobs = [j for _fn, j in sb._jobs(self._args())]
        assert jobs == [(600, 'padrao', 18, 1), (600, 'padrao', 18, 0)]

    def test_keeping_the_imported_leaves_them_out(self, out):
        _legacy(sb._path(600, 'padrao', 18, 0))
        jobs = [j for _fn, j in sb._jobs(self._args(keep_imported=True))]
        assert jobs == [(600, 'padrao', 18, 1), (600, 'padrao', 18, 2)]

    def test_the_election_pairs_are_added_when_asked(self, out, monkeypatch):
        monkeypatch.setattr(sb, 'ELECTION_OUT', str(out / 'election'))
        jobs = sb._jobs(self._args(seeds=[], election=True))
        vids = {(j[0], j[1]) for _fn, j in jobs}
        assert vids == {(v, t) for v in sb.ELECTION_VARIANTS for t in ('100', '50', '18')}
        assert len(jobs) == len(sb.ELECTION_VARIANTS) * 3 * 3
