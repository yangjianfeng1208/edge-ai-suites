# Automated Grading System

## Quick Start

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Edit Configuration
```bash
# Edit config.yaml to set:
# - Detection JSON path
# - OCR/VLM service URLs
# - Grading options
```

### 4. Start Services

**OCR Server:**
```bash
python ocr_services/paddleocr_vl_server.py
```
> Note: **For local testing**. Will be replaced by `smart-classroom` production API.

**VLM Server:**
```bash
python Qwen_services/vlm_server.py
```
> Note: **For local testing**. Will be replaced by `smart-classroom` production API.

### 5. Run Grading
```bash
python main.py
```

## Pipeline

```
┌─────────────────┐
│  PDF Exam Paper │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  YOLO Detection     │  ← Detect answer regions
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Extract Regions    │  ← Crop answer boxes
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  OCR Recognition    │  ← PaddleOCR-VL (GPU/CPU)
│  (API call)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Grade Objective    │  ← Rule-based matching
│  (choice + blank)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Grade Subjective   │  ← VLM scoring
│  (API call)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  JSON Results       │
└─────────────────────┘
```

## Configuration

**config.yaml:**
```yaml
grading:
  skip_subjective: false         # Enable/disable VLM grading
  skip_yolo_detection: false     # Skip YOLO, use cached

detection:
  json_path: "./test_data/..."   # Detection results
  yolo_conf: 0.15
  yolo_iou: 0.5

ocr:
  use_cached: false              # Use cached OCR text
  pdf_dpi: 50
  max_pixels: 10000000

ocr_service:
  base_url: "http://127.0.0.1:9901"

vlm_service:
  base_url: "http://127.0.0.1:9900"
```

## Output

```
outputs/
└── {exam_name}/
    ├── ocr_text/
    │   └── {exam_name}_ocr.txt
    ├── objective_grading.json
    ├── processed_answers/
    └── vlm_grading/
```

---

## Appendix: YOLO Training

### Prepare Dataset

1. Annotate exam papers with answer regions
2. Export to YOLO format (images + labels)
3. Organize as:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Train Model

```bash
cd yolo/train_yolo
python train_yolo_hilex.py
```

**Training script does:**
- Fix data.yaml paths
- Load YOLO11n pretrained model
- Train on dataset (100 epochs, 640px, batch 16)
- Save best model to `models/yolo_hilex/`

### Validate Model

```bash
python validate_trained_model.py
```

### Use Trained Model

Update detection JSON to use new model:
```json
{
  "yolo_model": "yolo/train_yolo/models/yolo_hilex/weights/best.pt"
}
```
