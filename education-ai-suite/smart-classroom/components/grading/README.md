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

`main.py` runs the following steps per student paper. Each step writes to its own
`step{N}_*` output directory, and the run ends with a per-step timing summary.

```
┌─────────────────┐
│  PDF Exam Paper │
└────────┬────────┘
         │
         ▼
┌───────────────────────────────┐
│ Step 1: Layout Detection      │  ← Render PDF, detect answer regions
│ (PP-DocLayout, API call)      │    via detection service
│ → step1_layout_detection/     │
└────────┬──────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ Step 2: OCR Recognition       │  ← Region-based OCR (PaddleOCR-VL API)
│ (PaddleOCR-VL, API call)      │    per-page / per-bbox timing
│ → step2_ocr_regions/          │
└────────┬──────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ Step 3: Question Mapping &    │  ← Map questions to OCR regions,
│ Subjective Region Detection   │    locate subjective answer regions
│ → step3_question_mapping/     │    (handles cross-page questions)
└────────┬──────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ Step 4: Grade Objective       │  ← Rule-based matching
│                               │    (OCR text vs answer key)
│ → step4_objective_grading/    │
└────────┬──────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ Step 5: Grade Subjective      │  ← Crop/stitch regions, VLM scoring
│ (Qwen VLM, API call)          │    per-question timing
│ → step5_subjective_grading/   │    (optional: skip_subjective)
└────────┬──────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ Final: Merge Results          │  ← objective + subjective totals
│ → grading_results.json        │
└───────────────────────────────┘
```

**Services used:** Step 1 calls the detection service, Step 2 the OCR server
(`ocr_services/paddleocr_vl_server.py`), Step 5 the VLM server
(`Qwen_services/vlm_server.py`). Step 5 is skipped when `pipeline.skip_subjective`
is `true`; OCR can reuse cached results via `pipeline.skip_ocr`.

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
    └── {student_id}/
        ├── step1_layout_detection/     # per-page detection JSON + visualizations
        ├── step2_ocr_regions/          # per-page OCR JSON + full_document.txt
        ├── step3_question_mapping/     # question_mapping.json, subjective_regions.json
        ├── step4_objective_grading/    # objective_grading.json, objective_questions.txt
        ├── step5_subjective_grading/   # subjective_grading.json, cropped_answers/, vlm_details/
        └── grading_results.json        # merged objective + subjective totals
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
