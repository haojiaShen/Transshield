@echo off
setlocal

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found in PATH.
  pause
  exit /b 1
)

cd /d "%REPO_DIR%"
python tools\start_showcase_admin_oneclick.py %*

endlocal
