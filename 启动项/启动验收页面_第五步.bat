@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Step 5 Verification Page - keep this window open

echo Checking local PostgreSQL...
"F:\pgsql\pgsql\bin\pg_isready.exe" -h 127.0.0.1 -p 5432 >nul 2>&1
if errorlevel 1 (
  echo Starting local PostgreSQL...
  "F:\pgsql\pgsql\bin\pg_ctl.exe" -D "F:\pgdata" -l "F:\pgdata\logfile" start >nul
  timeout /t 2 >nul
)

echo Checking local Ollama (embedding service)...
python -c "import sys; sys.path.insert(0,'scripts'); import knowledge; print('Ollama ready' if knowledge.ensure_ollama() else 'Ollama NOT available')"

echo.
echo Step 5 verification page: http://127.0.0.1:8769/
echo This page is independent from Steps 1-4 (8765 / 8766 / 8767 / 8768).
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_step5_server.py"
pause
