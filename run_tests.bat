@echo off
echo ============================================================
echo  NMRAnalysis Test Suite
echo ============================================================

cd /d "%~dp0"

echo.
echo [1/3] Smoke tests (imports + pure functions)...
python -m pytest tests/smoke -v --tb=short --timeout=30
if errorlevel 1 goto :fail

echo.
echo [2/3] Unit tests (logic, math, serialization)...
python -m pytest tests/unit -v --tb=short --timeout=60
if errorlevel 1 goto :fail

echo.
echo [3/3] Integration tests (real CSV + DB)...
python -m pytest tests/integration -v --tb=short --timeout=120

echo.
echo ============================================================
echo  All tests completed.
echo ============================================================
goto :end

:fail
echo.
echo ============================================================
echo  TESTS FAILED. Fix errors before proceeding.
echo ============================================================
exit /b 1

:end
