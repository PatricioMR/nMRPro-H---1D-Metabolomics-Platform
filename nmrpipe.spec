# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for nMRPipe (one-folder Windows build).

Build:   pyinstaller --noconfirm nmrpipe.spec
Output:  dist/nMRPipe/nMRPipe.exe  (+ _internal/, databases/, data/, utils/)

Notes
-----
* One-folder (not one-file): fast startup; the bundled databases/ act as the
  SEED that app_paths.databases_dir() copies to %APPDATA%/nMRPipe on first run.
* Heavy scientific deps (scipy, scikit-learn, matplotlib, reportlab) need their
  data files and some submodules collected explicitly.
"""
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# ── Bundled read-only assets (seed DBs, libraries, logo/icon) ────────────────
datas = [
    ('databases', 'databases'),   # seed metabolite DBs (.db/.csv)
    ('data',      'data'),         # lipidomics / internal CSV libraries
    ('utils',     'utils'),        # Logo_nMRPipe.png, nmrpipe.ico
]
binaries = []
hiddenimports = [
    'PIL._tkinter_finder',
    'scipy.special.cython_special',
]

# Pull in data files + submodules for the heavy packages.
for _pkg in ('scipy', 'sklearn', 'matplotlib', 'reportlab', 'networkx',
             'pandas'):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', '_pytest', 'PyInstaller'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='nMRPipe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,           # keep True for the first debug build; flip to False
    icon='utils/nmrpipe.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='nMRPipe',
)
