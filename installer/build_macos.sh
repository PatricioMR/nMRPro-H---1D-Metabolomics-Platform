#!/usr/bin/env bash
# ============================================================================
#  Build the NMRPro-H macOS package: PyInstaller application bundle
#  (dist/NMRPro-H.app) inside a drag-to-Applications disk image (.dmg).
#
#  Run on a Mac, from the project root, inside a venv with requirements.txt
#  installed (python.org or Homebrew Python with Tk):
#      bash installer/build_macos.sh
#
#  The .dmg is built for the machine's architecture (arm64 on Apple Silicon,
#  x86_64 on Intel). It is not signed with a Developer ID nor notarized, so
#  Gatekeeper blocks the first launch; README → Installation says how to open it.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/2] Building application bundle with PyInstaller ==="
python3 -m pip install --upgrade pyinstaller pillow
pyinstaller --noconfirm installer/nmrpro_h.spec

ARCH="$(uname -m)"
OUT="Output"
STAGE="build/dmg"
DMG="$OUT/NMRPro-H-macOS-${ARCH}.dmg"

echo "=== [2/2] Building disk image ==="
mkdir -p "$OUT"
rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
ditto dist/NMRPro-H.app "$STAGE/NMRPro-H.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "NMRPro-H" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
echo "Disk image: $DMG"

echo "=== DONE ==="
