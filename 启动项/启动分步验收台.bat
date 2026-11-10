@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Step-by-step Testbench - keep this window open

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
echo Step-by-step testbench: http://127.0.0.1:8771/
echo Each button tests ONE point and explains what it tests, how, and the expected result.
echo Structured logs: target -^> method -^> observation -^> criteria -^> conclusion.
echo Independent pages 8765-8769 and unified console 8770 stay unchanged.
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\testbench_server.py"
pause
