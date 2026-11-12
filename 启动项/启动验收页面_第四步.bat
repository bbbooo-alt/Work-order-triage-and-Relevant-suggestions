@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Step 4 Verification Page - keep this window open

call "%~dp0_env_check.bat"

echo.
echo Step 4 verification page: http://127.0.0.1:8768/
echo This page is independent from Steps 1-3 (8765 / 8766 / 8767).
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_step4_server.py"
pause
