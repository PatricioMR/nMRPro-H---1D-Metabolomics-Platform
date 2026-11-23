# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for NMRPro-H (one-folder build, Windows / Linux / macOS).

Build:   pyinstaller --noconfirm nmrpro_h.spec
Output:  Windows  dist/NMRPro-H/NMRPro-H.exe   -> build_windows.bat (Inno Setup)
         Linux    dist/NMRPro-H/NMRPro-H       -> installer/build_linux.sh (AppImage)
         macOS    dist/NMRPro-H.app            -> installer/build_macos.sh (.dmg)

PyInstaller does not cross-compile: each system's package is built on that
system (.github/workflows/build-installers.yml runs all three).

Notes
-----
* One-folder (not one-file): fast startup; the bundled databases/ act as the
  SEED that app_paths.databases_dir() copies to the per-user data folder
  (app_paths.user_data_dir()) on first run.
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
from app_paths import (REFERENCE_DATABASES, SUPPORT_DATABASES, APP_NAME,
                       COPYRIGHT, __version__)

# Icon in each system's own format. macOS wants .icns: PyInstaller converts
# the 1024 px PNG with Pillow. Linux executables carry no icon (the AppImage's
# .desktop entry points at the PNG).
if sys.platform == 'win32':
    _icon = 'utils/nmrpro_h.ico'
elif sys.platform == 'darwin':
    _icon = 'utils/Logo_NMRPro-H.png'
else:
    _icon = None
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
    icon=_icon,
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

# macOS: wrap the folder in an application bundle (dist/NMRPro-H.app).
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name=f'{APP_NAME}.app',
        icon=_icon,
        bundle_identifier='io.github.patriciomr.nmrpro-h',
        version=__version__,
        info_plist={
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleShortVersionString': __version__,
            'NSHumanReadableCopyright': COPYRIGHT,
            'NSHighResolutionCapable': True,
        },
    )
