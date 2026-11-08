@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Step 1 Verification Page - keep this window open

echo Checking local PostgreSQL...
"F:\pgsql\pgsql\bin\pg_isready.exe" -h 127.0.0.1 -p 5432 >nul 2>&1
if errorlevel 1 (
  echo Starting local PostgreSQL...
  "F:\pgsql\pgsql\bin\pg_ctl.exe" -D "F:\pgdata" -l "F:\pgdata\logfile" start >nul
  timeout /t 2 >nul
)

echo.
echo Verification page: http://127.0.0.1:8765/
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\verify_server.py"
pause
