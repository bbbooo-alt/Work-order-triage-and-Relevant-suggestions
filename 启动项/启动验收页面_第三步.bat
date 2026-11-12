@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Step 3 Verification Page - keep this window open

call "%~dp0_env_check.bat" ollama

echo.
echo Step 3 verification page: http://127.0.0.1:8767/
echo This page is independent from Step 1 (8765) and Step 2 (8766).
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_step3_server.py"
pause
