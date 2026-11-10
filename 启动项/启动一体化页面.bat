@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Unified Console - keep this window open

echo Checking local PostgreSQL...
"F:\pgsql\pgsql\bin\pg_isready.exe" -h 127.0.0.1 -p 5432 >nul 2>&1
if errorlevel 1 (
  echo Starting local PostgreSQL...
  "F:\pgsql\pgsql\bin\pg_ctl.exe" -D "F:\pgdata" -l "F:\pgdata\logfile" start >nul
  timeout /t 2 >nul
)

echo Checking local Ollama (embedding + generation)...
python -c "import sys; sys.path.insert(0,'scripts'); import knowledge; print('Ollama ready' if knowledge.ensure_ollama() else 'Ollama NOT available')"

echo.
echo Unified console: http://127.0.0.1:8770/
echo Five-step + P0 + end-to-end pipeline in ONE page.
echo Independent verification pages stay on 8765 / 8766 / 8767 / 8768 / 8769.
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\unified_server.py"
pause
