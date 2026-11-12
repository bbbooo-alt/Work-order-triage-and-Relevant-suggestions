@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title User Workbench - keep this window open

call "%~dp0_env_check.bat" ollama

echo.
echo User workbench: http://127.0.0.1:8772/
echo Submit a ticket to get triage, evidence, advice, engineer recommendation and disposition.
echo Models are warmed up in the background; the header shows when they are ready.
echo Demo only: confirmations and feedback never touch the production assignments table.
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\user_server.py"
pause
