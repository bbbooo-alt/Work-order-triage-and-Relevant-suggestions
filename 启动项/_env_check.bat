@echo off
rem Shared preflight for all launcher scripts.
rem   call "%~dp0_env_check.bat"          : PostgreSQL only
rem   call "%~dp0_env_check.bat" ollama  : PostgreSQL + Ollama
rem Optional local config (gitignored): local_env.bat with PG_BIN / PG_DATA.
rem If PostgreSQL is already running and pg_isready is on PATH, no config is needed.

if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"

set "PG_READY=pg_isready"
set "PG_CTL=pg_ctl"
if defined PG_BIN (
  set "PG_READY=%PG_BIN%\pg_isready.exe"
  set "PG_CTL=%PG_BIN%\pg_ctl.exe"
)

echo Checking local PostgreSQL...
"%PG_READY%" -h 127.0.0.1 -p 5432 >nul 2>&1
if not errorlevel 1 goto :pg_ok
if not defined PG_DATA (
  echo [WARN] PostgreSQL is not reachable and PG_DATA is not set.
  echo        Start PostgreSQL manually, or copy local_env.example.bat to
  echo        local_env.bat in this folder and set PG_BIN / PG_DATA.
  goto :pg_done
)
echo Starting local PostgreSQL...
"%PG_CTL%" -D "%PG_DATA%" -l "%PG_DATA%\logfile" start >nul
timeout /t 2 >nul
"%PG_READY%" -h 127.0.0.1 -p 5432 >nul 2>&1
if not errorlevel 1 goto :pg_ok
echo [WARN] PostgreSQL is still not reachable. Start it manually, then retry.
goto :pg_done
:pg_ok
echo PostgreSQL ready.
:pg_done

if /i not "%~1"=="ollama" goto :no_ollama
echo Checking local Ollama (embedding + generation)...
python -c "import sys; sys.path.insert(0,'scripts'); import knowledge; print('Ollama ready' if knowledge.ensure_ollama() else 'Ollama NOT available')"
:no_ollama
exit /b 0
