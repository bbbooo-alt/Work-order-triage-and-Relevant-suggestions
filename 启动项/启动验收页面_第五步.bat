@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Step 5 Verification Page - keep this window open

call "%~dp0_env_check.bat" ollama

echo.
echo Step 5 verification page: http://127.0.0.1:8769/
echo This page is independent from Steps 1-4 (8765 / 8766 / 8767 / 8768).
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_step5_server.py"
pause
