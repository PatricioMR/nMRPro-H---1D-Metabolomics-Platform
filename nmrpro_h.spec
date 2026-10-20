# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for NMRPro-H (one-folder Windows build).

Build:   pyinstaller --noconfirm nmrpro_h.spec
Output:  dist/NMRPro-H/NMRPro-H.exe  (+ _internal/, databases/, data/, utils/)

Notes
-----
* One-folder (not one-file): fast startup; the bundled databases/ act as the
  SEED that app_paths.databases_dir() copies to %APPDATA%/NMRPro-H on first run.
* Heavy scientific deps (scipy, scikit-learn, matplotlib, reportlab) need their
  data files and some submodules collected explicitly.
"""
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# ── Bundled read-only assets (seed DBs, libraries, logo/icon) ────────────────
# Only ship the databases the application offers (app_paths.REFERENCE_DATABASES)
# and the pathway table the statistics stage reads -- never old versions,
# the unfiltered build source or .bak/backup_* copies.
import sys
sys.path.insert(0, os.path.abspath('.'))   # the spec runs from the repo root
from app_paths import REFERENCE_DATABASES, SUPPORT_DATABASES
_db_seed = [
    (os.path.join('databases', _f), 'databases')
    for _f in [f for _l, f in REFERENCE_DATABASES] + list(SUPPORT_DATABASES)
]
datas = _db_seed + [
    ('data',      'data'),         # lipidomics / internal CSV libraries
    ('utils',     'utils'),        # Logo_NMRPro-H.png, nmrpro_h.ico
    ('LICENSE',   '.'),            # GNU AGPL-3.0, beside the program
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
    # torch & co. are not used by the application, but collect_all of
    # scipy/sklearn/pandas lists their array_api_compat adapters, and with
    # torch installed PyInstaller imported it to analyse it -- an access
    # violation in torch's DLL loader that stopped the v0.55 build.
    excludes=['pytest', '_pytest', 'PyInstaller',
              'torch', 'torchvision', 'torchaudio', 'tensorflow',
              'tensorboard', 'jax', 'cupy', 'dask'],
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
    name='NMRPro-H',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # release build — no debug console window
    icon='utils/nmrpro_h.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='NMRPro-H',
)
