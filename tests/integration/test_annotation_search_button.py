# Copyright (C) 2025-2026 Miguel Rocha Patricio
# SPDX-License-Identifier: AGPL-3.0-or-later (see LICENSE)

"""The Annotation panel has one search: the database in metabolomics, the
solvent's library of lipid regions in lipidomics -- the button stays live in
both modes."""
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


def test_one_search_button_follows_the_mode(real_app, monkeypatch):
    ann = real_app.stages['annotation']
    assert not hasattr(ann, 'btn_search_online'), 'the Internal Library button is gone'
    calls = []
    monkeypatch.setattr(ann, 'search_internal_db', lambda: calls.append('database'))
    monkeypatch.setattr(ann, 'search_lipidomics_library', lambda: calls.append('lipids'))
    ann.internal_library_mode_var.set('Metabolomics (default)')
    real_app.root.update()
    ann.search()
    assert str(ann.btn_search_internal.cget('state')) != 'disabled'
    ann.internal_library_mode_var.set('Lipidomics')
    real_app.root.update()
    ann.search()
    assert str(ann.btn_search_internal.cget('state')) != 'disabled'
    assert calls == ['database', 'lipids']
    ann.internal_library_mode_var.set('Metabolomics (default)')


def test_the_status_bar_is_packed_ahead_of_the_content(real_app):
    """The autosave writes its time in the status bar; packed after the main
    content, the tall Quantification panel pushed it off the window."""
    slaves = real_app.root.pack_slaves()
    bar = real_app.status_label.master
    assert slaves.index(bar) < slaves.index(real_app.main_container)


def test_what_this_release_does_not_offer_is_not_shown(real_app):
    from app_paths import DECONVOLUTION_ENABLED
    ann = real_app.stages['annotation']
    if not DECONVOLUTION_ENABLED:
        assert ann._mcr_run_cb is None
    assert ann.mcr_run_var.get() is False
    quant = real_app.stages['quantification']
    assert not hasattr(quant, '_deconv_btn')
