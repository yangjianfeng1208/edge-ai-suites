@echo off
cd /d "%~dp0"
echo Starting VLM Server...
echo.
venv\Scripts\python.exe Qwen_services\vlm_server.py
pause
