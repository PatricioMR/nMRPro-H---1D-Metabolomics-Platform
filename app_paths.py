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
__version__   = "0.72"
RELEASE_STAGE = "Alpha"
APP_VERSION   = f"{APP_NAME} v{__version__} {RELEASE_STAGE}"

# Deconvolution is a future stage. Everything that resolves overlap between
# analytes -- the JG-Deconv stage, the Full-Spectrum MCR stage, the
# Deconvolution Tool reached from the peak list and the annotation window, and
# the shared-window split in quantification -- stays in the code, tested, and
# hidden behind this one switch. Off, the software is the pipeline that has
# been validated end to end: detect, annotate, elect, integrate, concentrate.
# Turning it back on is this line; nothing else was removed.
DECONVOLUTION_ENABLED = False   # "NMRPro-H v0.38 Alpha"


# The reference databases the application offers for annotation, in the order
# it lists them. Both are split from BMRB_Metabolomics_v05
# (hmdb_api_download/split_dbs.py) and carry every correction of v04 and v05
# (ADR-049, ADR-060): Mammalian holds the compounds ChEBI places in mammals,
# General adds those of unknown origin. Older versions stay in git history;
# an installation that still holds them in its user folder no longer lists
# them. Any other file is reachable through "Browse".
REFERENCE_DATABASES = (
    ("BMRB Mammalian v05", "BMRB_Metabolomics_Mammalian_v05.db"),
    ("BMRB General v05", "BMRB_Metabolomics_General_v05.db"),
)
# Shipped beside them, not offered for annotation: pathway membership for
# the statistics stage.
SUPPORT_DATABASES = ("hmdb_pathways.db",)


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
    per-user folder under %APPDATA%/NMRPro-H, where every bundled .db/.csv
    the folder does not hold yet is copied in -- so a new database version
    reaches an existing installation -- and nothing already there is ever
    overwritten: the user adds and edits databases there.
    """
    if not _is_frozen():
        # Development: use the project's databases/ directly.
        _dev = asset_path("databases")
        os.makedirs(_dev, exist_ok=True)
        return _dev

    target = os.path.join(user_data_dir(), "databases")
    os.makedirs(target, exist_ok=True)
    # Seeded from the bundled defaults with whatever the folder lacks. Only
    # when it was empty, a new version (BMRB v04) never reached an
    # installation that already had a database.
    try:
        seed = asset_path("databases")
        if os.path.isdir(seed):
            for fname in os.listdir(seed):
                if fname.lower().endswith((".db", ".csv")):
                    _dst = os.path.join(target, fname)
                    if not os.path.exists(_dst):
                        shutil.copy2(os.path.join(seed, fname), _dst)
    except Exception as exc:
        print(f"[app_paths] database seeding skipped: {exc}")
    return target


def reference_databases():
    """``[(label, path)]`` of the reference databases present in
    ``databases_dir()``, in ``REFERENCE_DATABASES`` order."""
    folder = databases_dir()
    return [(label, os.path.join(folder, fname))
            for label, fname in REFERENCE_DATABASES
            if os.path.isfile(os.path.join(folder, fname))]
