# PaddleOCR-VL OCR Service

## Setup

1. Install dependencies:

```bash
setup.bat
```

Or manually:

```bash
python -m venv venv
venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
```

2. Download and convert model:

```bash
venv\Scripts\activate
python download_and_convert_model.py
```

Options:
- `--model-id`: Choose model version (default: PaddlePaddle/PaddleOCR-VL-1.6)
- `--cache-dir`: Cache directory (default: ./_cache)
- `--output-dir`: Output directory (default: ./ov_paddleocr_vl_1_6_model)
- `--llm-int8`: Enable INT8 compression (default: True)
- `--llm-int4`: Enable INT4 compression (smaller, faster)
- `--vision-int8`: Enable INT8 for vision encoder

## Usage

### Test with exam paper

```bash
venv\Scripts\activate
python test_ocr.py
```

### Command line usage

```bash
venv\Scripts\activate
python paddleocr_vl_service.py --model ./ov_paddleocr_vl_1_6_model --image <path> [--output result.txt]
```

### Python API

```python
from paddleocr_vl_service import PaddleOCRVLService

service = PaddleOCRVLService(model_path="./ov_paddleocr_vl_1_6_model")
text = service.ocr_image("exam_paper.pdf", task="ocr")
print(text)
```
