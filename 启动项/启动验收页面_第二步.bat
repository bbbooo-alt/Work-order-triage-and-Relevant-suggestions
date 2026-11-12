@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Step 2 Verification Page - keep this window open

call "%~dp0_env_check.bat"

echo.
echo Step 2 verification page: http://127.0.0.1:8766/
echo This page is independent from Step 1 (port 8765).
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_step2_server.py"
pause
