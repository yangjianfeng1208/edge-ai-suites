@echo off
echo Installing PaddleOCR-VL dependencies to parent venv...
echo.

cd ..
call venv\Scripts\activate.bat

echo Installing PyTorch CPU version...
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo.
echo Installing other dependencies...
pip install openvino>=2025.4.1
pip install transformers==4.54.0
pip install pillow opencv-python numpy
pip install pdf2image
pip install nncf sentencepiece einops protobuf
pip install modelscope huggingface-hub

echo.
echo Done! Now you can run:
echo   cd ocr_services
echo   python download_and_convert_model.py
