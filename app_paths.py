"""
Resource path resolution for nMRPipe — works both in development and when
frozen by PyInstaller (one-folder).

Two kinds of resource:
  • Read-only bundled assets (logo, lipidomics library, seed databases) live
    next to the code in dev, and inside the PyInstaller bundle when frozen
    (sys._MEIPASS). Use `asset_path(...)` / `bundle_dir()`.
  • Writable, user-editable data (the metabolite databases the user adds/edits)
    must NOT live under Program Files. They live in %APPDATA%/nMRPipe and are
    SEEDED once from the bundled defaults. Use `databases_dir()`.
"""
import os
import sys
import shutil

APP_NAME = "nMRPipe"

# ── Version (single source of truth) ───────────────────────────────────────────
# Bump these two on a release; every display string is derived from APP_VERSION.
__version__   = "0.2"
RELEASE_STAGE = "Alpha"
APP_VERSION   = f"{APP_NAME} v{__version__} {RELEASE_STAGE}"   # "nMRPipe v0.2 Alpha"


def _is_frozen():
    return bool(getattr(sys, "frozen", False))


def bundle_dir():
    """Directory holding read-only bundled assets.

    Frozen: the PyInstaller extraction dir (sys._MEIPASS). Dev: the project root
    (this file lives at the project root).
    """
    if _is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def asset_path(*parts):
    """Absolute path to a bundled, read-only asset (e.g. utils/Logo.png)."""
    return os.path.join(bundle_dir(), *parts)


def user_data_dir():
    """Per-user writable data directory (%APPDATA%/nMRPipe), created if missing.

    Falls back to ~/.nMRPipe when APPDATA is unset (non-Windows / unusual env).
    """
    base = os.environ.get("APPDATA") or os.path.join(
        os.path.expanduser("~"), ".config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def databases_dir():
    """Writable databases directory.

    Dev: the repo's own ``databases/`` folder (edited in place). Frozen: a
    per-user folder under %APPDATA%/nMRPipe, seeded once from the bundled
    defaults — so the user adds/edits .db/.csv there and app updates never
    overwrite them (seeding only when the folder is empty/new).
    """
    if not _is_frozen():
        # Development: use the project's databases/ directly.
        _dev = asset_path("databases")
        os.makedirs(_dev, exist_ok=True)
        return _dev

    target = os.path.join(user_data_dir(), "databases")
    os.makedirs(target, exist_ok=True)
    # Seed from the bundled defaults the first time (target empty).
    try:
        if not any(f.lower().endswith((".db", ".csv"))
                   for f in os.listdir(target)):
            seed = asset_path("databases")
            if os.path.isdir(seed):
                for fname in os.listdir(seed):
                    if fname.lower().endswith((".db", ".csv")):
                        _src = os.path.join(seed, fname)
                        _dst = os.path.join(target, fname)
                        if not os.path.exists(_dst):
                            shutil.copy2(_src, _dst)
    except Exception as exc:
        print(f"[app_paths] database seeding skipped: {exc}")
    return target
