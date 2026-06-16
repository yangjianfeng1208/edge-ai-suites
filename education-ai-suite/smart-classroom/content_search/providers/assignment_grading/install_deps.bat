@echo off
REM Safe installation script for assignment_grading dependencies

cd /d "%~dp0"

echo ============================================================
echo Installing Assignment Grading Dependencies
echo ============================================================
echo.
echo IMPORTANT: Please close any running Python processes first!
echo Press Ctrl+C to cancel, or any key to continue...
pause > nul

echo.
echo Step 1: Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo Step 2: Uninstalling old opencv (if any)...
venv\Scripts\pip.exe uninstall -y opencv-python opencv-python-headless

echo.
echo Step 3: Installing dependencies...
venv\Scripts\pip.exe install -r requirements.txt --force-reinstall --no-deps paddleocr paddlepaddle

echo.
echo Step 4: Installing remaining dependencies...
venv\Scripts\pip.exe install -r requirements.txt

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo You can now run: run_grading.bat
echo.
pause
