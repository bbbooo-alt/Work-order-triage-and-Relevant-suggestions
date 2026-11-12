@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Unified Console - keep this window open

call "%~dp0_env_check.bat" ollama

echo.
echo Unified console: http://127.0.0.1:8770/
echo Five-step + P0 + end-to-end pipeline in ONE page.
echo Independent verification pages stay on 8765 / 8766 / 8767 / 8768 / 8769.
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\unified_server.py"
pause
