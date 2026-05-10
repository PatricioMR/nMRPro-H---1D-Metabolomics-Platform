# tests/smoke/test_app_startup.py
"""
Smoke tests: verify core modules import and instantiate without a display.
No real Tkinter window is opened.
"""
import sys
import types
import pytest


# ── Mock Tkinter so tests run headless ────────────────────────────────────────

def _mock_tkinter():
    """Replace tkinter with a minimal stub so non-GUI code can be imported."""
    tk_stub = types.ModuleType('tkinter')

    class _Widget:
        def __init__(self, *a, **kw): pass
        def pack(self, *a, **kw): return self
        def grid(self, *a, **kw): return self
        def config(self, *a, **kw): pass
        configure = config
        def get(self): return ''
        def set(self, v): pass
        def bind(self, *a, **kw): pass
        def after(self, *a, **kw): pass

    for name in ('Tk', 'Frame', 'Label', 'Button', 'Entry', 'Checkbutton',
                 'Listbox', 'Scrollbar', 'Toplevel', 'Canvas', 'Text',
                 'BooleanVar', 'StringVar', 'DoubleVar', 'IntVar',
                 'PanedWindow', 'Menu'):
        setattr(tk_stub, name, _Widget)

    tk_stub.END    = 'end'
    tk_stub.LEFT   = 'left'
    tk_stub.RIGHT  = 'right'
    tk_stub.TOP    = 'top'
    tk_stub.BOTTOM = 'bottom'
    tk_stub.BOTH   = 'both'
    tk_stub.X      = 'x'
    tk_stub.Y      = 'y'
    tk_stub.N = tk_stub.S = tk_stub.E = tk_stub.W = ''
    tk_stub.NW = tk_stub.NE = tk_stub.SW = tk_stub.SE = ''
    tk_stub.CENTER = 'center'
    tk_stub.HORIZONTAL = 'horizontal'
    tk_stub.VERTICAL   = 'vertical'
    tk_stub.BROWSE     = 'browse'
    tk_stub.NORMAL     = 'normal'
    tk_stub.DISABLED   = 'disabled'

    ttk_stub = types.ModuleType('tkinter.ttk')
    for name in ('Frame', 'Label', 'Button', 'Entry', 'Checkbutton',
                 'Combobox', 'LabelFrame', 'Separator', 'Treeview',
                 'Scrollbar', 'Notebook', 'PanedWindow', 'Style'):
        setattr(ttk_stub, name, _Widget)

    mb_stub = types.ModuleType('tkinter.messagebox')
    mb_stub.showinfo    = lambda *a, **kw: None
    mb_stub.showerror   = lambda *a, **kw: None
    mb_stub.showwarning = lambda *a, **kw: None
    mb_stub.askyesno    = lambda *a, **kw: True

    fd_stub = types.ModuleType('tkinter.filedialog')
    fd_stub.askopenfilename  = lambda *a, **kw: ''
    fd_stub.asksaveasfilename = lambda *a, **kw: ''

    sys.modules.setdefault('tkinter',            tk_stub)
    sys.modules.setdefault('tkinter.ttk',        ttk_stub)
    sys.modules.setdefault('tkinter.messagebox', mb_stub)
    sys.modules.setdefault('tkinter.filedialog', fd_stub)


_mock_tkinter()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestImports:
    def test_loader_imports(self):
        from data.loader import load_all_samples, detect_separator
        assert callable(load_all_samples)

    def test_multiplet_scorer_imports(self):
        from gui.notation.multiplet_scorer import score_multiplet_match
        assert callable(score_multiplet_match)

    def test_quantification_math_imports(self):
        from gui.stages.quantification_stage import _trapz, _cv, _pqn
        assert callable(_trapz)
        assert callable(_cv)

    def test_project_manager_imports(self):
        from gui.project_manager import ProjectManager
        assert ProjectManager is not None

    def test_annotation_stage_imports(self):
        from gui.stages.annotation_stage import _ppm_score, _mult_compat_check
        assert callable(_ppm_score)

    def test_db_pick_function_imports(self):
        from gui.stages.annotation_stage import _pick_database
        assert callable(_pick_database)


class TestPureFunctions:
    """Verify key pure functions work without any UI."""

    def test_ppm_score_boundaries(self):
        from gui.stages.annotation_stage import _ppm_score
        assert _ppm_score(1.0, 1.0, 0.05) == pytest.approx(1.0)
        assert _ppm_score(1.0, 2.0, 0.05) < 0.01

    def test_parse_j(self):
        from gui.notation.multiplet_scorer import _parse_j
        assert _parse_j('7.2, 14.4') == [7.2, 14.4]

    def test_cv_nan_on_empty(self):
        import numpy as np
        from gui.stages.quantification_stage import _cv
        assert np.isnan(_cv([]))

    def test_trapz_zero_for_zeros(self):
        import numpy as np
        from gui.stages.quantification_stage import _trapz
        assert _trapz(np.zeros(10), np.linspace(0, 1, 10)) == pytest.approx(0.0)
