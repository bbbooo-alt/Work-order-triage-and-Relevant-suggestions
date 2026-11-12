@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Step-by-step Testbench - keep this window open

call "%~dp0_env_check.bat" ollama

echo.
echo Step-by-step testbench: http://127.0.0.1:8771/
echo Each button tests ONE point and explains what it tests, how, and the expected result.
echo Structured logs: target -^> method -^> observation -^> criteria -^> conclusion.
echo Independent pages 8765-8769 and unified console 8770 stay unchanged.
echo The browser will open automatically. Close this window to stop the service.
echo.
python "web\testbench_server.py"
pause
