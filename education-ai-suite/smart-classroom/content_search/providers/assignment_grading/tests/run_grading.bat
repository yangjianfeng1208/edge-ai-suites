@echo off
REM Assignment Grading System Runner

cd /d "%~dp0"

REM Check if local venv exists
if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
    set VENV_NAME=Local venv
) else (
    REM Fallback to main venv
    set PYTHON_EXE=C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\venv_smartclassroom\Scripts\python.exe
    set VENV_NAME=venv_smartclassroom (fallback)
)

echo ============================================================
echo Assignment Grading System
echo Using: %VENV_NAME%
echo ============================================================
echo.

"%PYTHON_EXE%" grading_prototype.py %*

pause
