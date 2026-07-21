@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing PyTorch CPU version...
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo Installing other dependencies...
pip install -r requirements.txt

echo Setup complete!
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
