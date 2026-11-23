# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)
"""The quantification table marks an analyte the analyst verified with ◎,
the first of the Quality dots, at any level of the engine's."""
import pytest

tk = pytest.importorskip('tkinter')


@pytest.fixture(scope='module')
def real_app():
    if not hasattr(tk, 'TclError'):
        pytest.skip('tkinter is stubbed in this session')
    try:
        from gui.app import App
        app = App()
    except tk.TclError as e:
        pytest.skip(f'no display: {e}')
    except Exception as e:
        pytest.skip(f'the application cannot be built here: {e}')
    app.root.withdraw()
    try:
        app._build_ui()
    except Exception as e:
        pytest.skip(f'the application cannot be built here: {e}')
    yield app
    try:
        app.root.destroy()
    except Exception:
        pass


def test_the_verified_rows_carry_the_mark_first_in_quality(real_app):
    q = real_app.stages['quantification']
    if q._tree is None:
        q._build_right_panel()
    q._quant_results = [
        {'hmdb_id': 'A', 'metabolite': 'Alpha', 'ppm': 1.0, 'conc_mm': 1.0,
         'status': '●○○', '_confidence': 1, '_annotation_manual': True},
        {'hmdb_id': 'B', 'metabolite': 'Beta', 'ppm': 2.0, 'conc_mm': 2.0,
         'status': '●●●', '_confidence': 3, '_annotation_manual': True},
        {'hmdb_id': 'C', 'metabolite': 'Gamma', 'ppm': 3.0, 'conc_mm': 3.0,
         'status': '●●○', '_confidence': 2}]
    try:
        q._populate_tree()
        quality = {i: q._tree.set(i, 'status') for i in 'ABC'}
        assert quality == {'A': '◎●○○', 'B': '◎●●●', 'C': '●●○'}
        assert not any(q._tree.item(i)['image'] for i in 'ABC')
    finally:
        q._quant_results = []
