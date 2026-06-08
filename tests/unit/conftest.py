# tests/unit/conftest.py
"""
Mock tkinter and matplotlib before importing any GUI module.
This lets unit tests run headless without a display.
"""
import sys
import types


def _install_mocks():
    # ── tkinter ───────────────────────────────────────────────────────────────
    class _W:
        def __init__(self, *a, **kw): pass
        def pack(self, *a, **kw): return self
        def grid(self, *a, **kw): return self
        def config(self, *a, **kw): pass
        configure = config
        def get(self): return ''
        def set(self, v): pass
        def bind(self, *a, **kw): pass
        def after(self, *a, **kw): pass
        def destroy(self, *a, **kw): pass
        def winfo_rootx(self): return 0
        def winfo_rooty(self): return 0
        def winfo_width(self): return 800
        def winfo_height(self): return 600

    tk = types.ModuleType('tkinter')
    for name in ('Tk', 'Frame', 'Label', 'Button', 'Entry', 'Checkbutton',
                 'Listbox', 'Scrollbar', 'Toplevel', 'Canvas', 'Text',
                 'BooleanVar', 'StringVar', 'DoubleVar', 'IntVar',
                 'PanedWindow', 'Menu', 'LabelFrame'):
        setattr(tk, name, _W)
    for attr in ('END', 'LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'BOTH', 'X', 'Y',
                 'N', 'S', 'E', 'W', 'NW', 'NE', 'SW', 'SE', 'CENTER',
                 'HORIZONTAL', 'VERTICAL', 'BROWSE', 'NORMAL', 'DISABLED',
                 'WORD', 'FLAT', 'GROOVE', 'RAISED', 'RIDGE', 'SUNKEN'):
        setattr(tk, attr, attr.lower())
    tk.font = types.ModuleType('tkinter.font')

    ttk = types.ModuleType('tkinter.ttk')
    for name in ('Frame', 'Label', 'Button', 'Entry', 'Checkbutton',
                 'Combobox', 'LabelFrame', 'Separator', 'Treeview',
                 'Scrollbar', 'Notebook', 'PanedWindow', 'Style'):
        setattr(ttk, name, _W)

    mb = types.ModuleType('tkinter.messagebox')
    mb.showinfo = mb.showerror = mb.showwarning = lambda *a, **kw: None
    mb.askyesno = lambda *a, **kw: True

    fd = types.ModuleType('tkinter.filedialog')
    fd.askopenfilename = fd.asksaveasfilename = lambda *a, **kw: ''

    for mod, name in [
        (tk, 'tkinter'), (ttk, 'tkinter.ttk'),
        (mb, 'tkinter.messagebox'), (fd, 'tkinter.filedialog'),
        (tk.font, 'tkinter.font'),
    ]:
        sys.modules.setdefault(name, mod)

    # ── matplotlib (stub enough for module-level imports) ─────────────────────
    mpl = types.ModuleType('matplotlib')
    mpl.use = lambda *a, **kw: None
    mpl.rcParams = {}

    plt = types.ModuleType('matplotlib.pyplot')
    plt.Figure = lambda *a, **kw: types.SimpleNamespace(
        add_subplot=lambda *a, **kw: types.SimpleNamespace(
            plot=lambda *a, **kw: None,
            set_xlabel=lambda *a, **kw: None,
            set_ylabel=lambda *a, **kw: None,
            set_title=lambda *a, **kw: None,
            clear=lambda *a, **kw: None,
            set_xlim=lambda *a, **kw: None,
            set_ylim=lambda *a, **kw: None,
            tick_params=lambda *a, **kw: None,
            legend=lambda *a, **kw: None,
            axvline=lambda *a, **kw: None,
            fill_between=lambda *a, **kw: None,
        ),
        patch=types.SimpleNamespace(set_facecolor=lambda *a, **kw: None),
        tight_layout=lambda *a, **kw: None,
        subplots_adjust=lambda *a, **kw: None,
    )
    plt.subplots = lambda *a, **kw: (plt.Figure(), plt.Figure().add_subplot())
    plt.close    = lambda *a, **kw: None

    gs_mod = types.ModuleType('matplotlib.gridspec')
    gs_mod.GridSpec = lambda *a, **kw: types.SimpleNamespace(
        __getitem__=lambda *a, **kw: None)

    backend_tkagg = types.ModuleType('matplotlib.backends.backend_tkagg')
    backend_tkagg.FigureCanvasTkAgg = lambda *a, **kw: types.SimpleNamespace(
        draw=lambda: None,
        get_tk_widget=lambda: _W(),
        mpl_connect=lambda *a, **kw: None,
    )
    backend_tkagg.NavigationToolbar2Tk = lambda *a, **kw: types.SimpleNamespace(
        update=lambda: None, pack=lambda *a, **kw: None)

    backends = types.ModuleType('matplotlib.backends')

    colors_mod = types.ModuleType('matplotlib.colors')
    colors_mod.to_hex        = lambda c, *a, **kw: '#000000'
    colors_mod.to_rgba       = lambda c, *a, **kw: (0, 0, 0, 1)
    colors_mod.Normalize     = lambda *a, **kw: types.SimpleNamespace(__call__=lambda v: v)
    colors_mod.LinearSegmentedColormap = types.SimpleNamespace(
        from_list=lambda *a, **kw: None)
    colors_mod.Colormap      = object
    colors_mod.ListedColormap = lambda *a, **kw: None

    patches_mod = types.ModuleType('matplotlib.patches')
    patches_mod.Patch        = lambda *a, **kw: types.SimpleNamespace()
    patches_mod.Rectangle    = lambda *a, **kw: types.SimpleNamespace()
    patches_mod.FancyArrowPatch = lambda *a, **kw: types.SimpleNamespace()

    figure_mod = types.ModuleType('matplotlib.figure')
    figure_mod.Figure = lambda *a, **kw: types.SimpleNamespace(
        add_subplot=lambda *a, **kw: types.SimpleNamespace(),
        tight_layout=lambda *a, **kw: None)

    for mod, name in [
        (mpl,           'matplotlib'),
        (plt,           'matplotlib.pyplot'),
        (gs_mod,        'matplotlib.gridspec'),
        (backends,      'matplotlib.backends'),
        (backend_tkagg, 'matplotlib.backends.backend_tkagg'),
        (colors_mod,    'matplotlib.colors'),
        (patches_mod,   'matplotlib.patches'),
        (figure_mod,    'matplotlib.figure'),
    ]:
        sys.modules.setdefault(name, mod)


_install_mocks()
