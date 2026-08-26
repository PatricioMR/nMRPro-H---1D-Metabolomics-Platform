@echo off
REM ============================================================================
REM  Build NMRPro-H Windows executable (one-folder) + Inno Setup installer.
REM
REM  Prereqs:
REM    * Run from the project root, inside the venv that has the app deps.
REM    * Inno Setup installed (for the installer step) — provides ISCC.exe.
REM      https://jrsoftware.org/isdl.php
REM ============================================================================
setlocal

echo.
echo === [1/3] Installing/upgrading build tools ===
python -m pip install --upgrade pyinstaller pillow || goto :err

echo.
echo === [2/3] Building one-folder app with PyInstaller ===
pyinstaller --noconfirm nmrpro_h.spec || goto :err
echo Built: dist\NMRPro-H\NMRPro-H.exe

echo.
echo === [3/3] Building installer with Inno Setup (ISCC) ===
REM Single source of truth for the version: app_paths.__version__
for /f tokens^=2^ delims^=^" %%v in ('findstr /b "__version__" app_paths.py') do set "APPVER=%%v"
echo App version: %APPVER%
REM Locate ISCC. Inno Setup does not add itself to PATH, and a per-user
REM install lands under %LOCALAPPDATA%, not Program Files.
set "ISCC="
where iscc >nul 2>nul && set "ISCC=iscc"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if defined ISCC (
    "%ISCC%" /DAppVersion=%APPVER% installer\nmrpro_h.iss || goto :err
    echo Installer: Output\NMRPro-H-Setup.exe
) else (
    echo [skip] ISCC not found - install Inno Setup, then run:
    echo        iscc installer\nmrpro_h.iss
)

echo.
echo === DONE ===
goto :eof

:err
echo.
echo *** BUILD FAILED (exit %ERRORLEVEL%) ***
exit /b 1
