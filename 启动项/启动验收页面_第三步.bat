@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Step 3 Verification Page - keep this window open

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
echo Step 3 verification page: http://127.0.0.1:8767/
echo This page is independent from Step 1 (8765) and Step 2 (8766).
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_step3_server.py"
pause
