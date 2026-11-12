@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Step 1 Verification Page - keep this window open

call "%~dp0_env_check.bat"

echo.
echo Verification page: http://127.0.0.1:8765/
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_server.py"
pause
