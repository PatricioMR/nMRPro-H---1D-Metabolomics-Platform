"""
Resource path resolution for NMRPro-H — works both in development and when
frozen by PyInstaller (one-folder).

Two kinds of resource:
  • Read-only bundled assets (logo, lipidomics library, seed databases) live
    next to the code in dev, and inside the PyInstaller bundle when frozen
    (sys._MEIPASS). Use `asset_path(...)` / `bundle_dir()`.
  • Writable, user-editable data (the metabolite databases the user adds/edits)
    must NOT live under Program Files. They live in %APPDATA%/NMRPro-H and are
    SEEDED once from the bundled defaults. Use `databases_dir()`.
"""
import os
import sys
import shutil

APP_NAME = "NMRPro-H"

# Previous name. The project shipped as "nMRPipe" until it was renamed to avoid
# colliding with NMRPipe (Delaglio et al., J Biomol NMR 6:277-293, 1995), an
# established and unrelated package. Kept only so an existing installation's
# per-user data — the databases the analyst edited or added — is carried over
# instead of orphaned under a directory the app no longer looks at.
LEGACY_APP_NAMES = ("nMRPipe",)

# ── Version (single source of truth) ───────────────────────────────────────────
# Bump these two on a release; every display string is derived from APP_VERSION.
__version__   = "0.18"
RELEASE_STAGE = "Alpha"
APP_VERSION   = f"{APP_NAME} v{__version__} {RELEASE_STAGE}"   # "NMRPro-H v0.18 Alpha"


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
    """Per-user writable data directory (%APPDATA%/NMRPro-H), created if missing.

    Falls back to ~/.config/NMRPro-H when APPDATA is unset (non-Windows).

    On first run after the rename, an existing directory under one of
    LEGACY_APP_NAMES is adopted: it holds databases the analyst edited or added,
    and a fresh directory seeded from the bundled defaults would silently lose
    them.
    """
    base = os.environ.get("APPDATA") or os.path.join(
        os.path.expanduser("~"), ".config")
    path = os.path.join(base, APP_NAME)
    if not os.path.isdir(path):
        for legacy in LEGACY_APP_NAMES:
            old_path = os.path.join(base, legacy)
            if os.path.isdir(old_path):
                try:
                    shutil.move(old_path, path)
                    print(f"[app_paths] carried over user data from {legacy}")
                    break
                except Exception as exc:      # never block startup over this
                    print(f"[app_paths] could not carry over {legacy}: {exc}")
    os.makedirs(path, exist_ok=True)
    return path


def databases_dir():
    """Writable databases directory.

    Dev: the repo's own ``databases/`` folder (edited in place). Frozen: a
    per-user folder under %APPDATA%/NMRPro-H, seeded once from the bundled
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
