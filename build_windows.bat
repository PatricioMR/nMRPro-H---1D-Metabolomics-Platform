@echo off
REM ============================================================================
REM  Build nMRPipe Windows executable (one-folder) + Inno Setup installer.
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
pyinstaller --noconfirm nmrpipe.spec || goto :err
echo Built: dist\nMRPipe\nMRPipe.exe

echo.
echo === [3/3] Building installer with Inno Setup (ISCC) ===
REM Single source of truth for the version: app_paths.__version__
for /f tokens^=2^ delims^=^" %%v in ('findstr /b "__version__" app_paths.py') do set "APPVER=%%v"
echo App version: %APPVER%
where iscc >nul 2>nul
if %ERRORLEVEL%==0 (
    iscc /DAppVersion=%APPVER% installer\nmrpipe.iss || goto :err
    echo Installer: Output\nMRPipe-Setup.exe
) else (
    echo [skip] ISCC not on PATH — install Inno Setup, then run:
    echo        iscc installer\nmrpipe.iss
)

echo.
echo === DONE ===
goto :eof

:err
echo.
echo *** BUILD FAILED (exit %ERRORLEVEL%) ***
exit /b 1
