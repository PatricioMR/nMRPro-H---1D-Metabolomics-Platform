#!/usr/bin/env bash
# ============================================================================
#  Build the NMRPro-H Linux package: PyInstaller one-folder app wrapped in an
#  AppImage (one file, runs without installing) plus a .tar.gz of the folder.
#
#  Prereqs (Debian/Ubuntu):  sudo apt install python3-tk python3-venv
#  Run from the project root, inside a venv with requirements.txt installed:
#      bash installer/build_linux.sh
#
#  The AppImage runs on distributions whose glibc is at least the build
#  machine's: build on the oldest system you want to support (CI: Ubuntu 22.04).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/3] Installing/upgrading build tools ==="
python3 -m pip install --upgrade pyinstaller pillow

echo "=== [2/3] Building one-folder app with PyInstaller ==="
pyinstaller --noconfirm installer/nmrpro_h.spec

ARCH="$(uname -m)"
OUT="Output"
NAME="NMRPro-H-linux-${ARCH}"
mkdir -p "$OUT"
tar -C dist -czf "$OUT/${NAME}.tar.gz" NMRPro-H
echo "Portable folder: $OUT/${NAME}.tar.gz"

echo "=== [3/3] Building AppImage ==="
APPDIR="build/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib"
cp -a dist/NMRPro-H "$APPDIR/usr/lib/NMRPro-H"
cp utils/Logo_NMRPro-H.png "$APPDIR/nmrpro-h.png"

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/NMRPro-H/NMRPro-H" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/nmrpro-h.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=NMRPro-H
Comment=1D 1H NMR metabolomics: detection, annotation, quantification
Exec=NMRPro-H
Icon=nmrpro-h
Categories=Science;Chemistry;
Terminal=false
DESKTOP

TOOL="build/appimagetool-${ARCH}.AppImage"
if [ ! -x "$TOOL" ]; then
    curl -fsSL -o "$TOOL" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
    chmod +x "$TOOL"
fi
# Extract-and-run: no FUSE needed on the build machine (CI, WSL, containers).
APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$ARCH" "$TOOL" --no-appstream "$APPDIR" "$OUT/${NAME}.AppImage"
echo "AppImage: $OUT/${NAME}.AppImage"

echo "=== DONE ==="
