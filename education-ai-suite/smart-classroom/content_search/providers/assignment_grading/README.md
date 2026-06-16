# Assignment Grading Service

Automated grading system for Math exam assignments using OCR and VLM.

## ⚡ Quick Start

**Prerequisites:**
- VLM service running at `http://127.0.0.1:9900`
- Main smart-classroom venv already set up

**Run Grading (Windows):**
```batch
run_grading.bat
```

Or manually:
```batch
C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\venv_smartclassroom\Scripts\python.exe grading_prototype.py
```

**Prepare Test Data:**
1. Place assignment images (JPG/PNG) in `test_data/math/` folder
2. Configure `answer_key.json` with standard answers
3. Adjust `config.yaml` if needed

**Results:**
- Output files saved to `outputs/` folder
- Each assignment gets a `{filename}_result.json`

## 📁 Project Structure

```
assignment_grading/
├── DESIGN.md                # Detailed design document
├── README.md                # This file
├── run_grading.bat          # Windows launcher script
├── grading_prototype.py     # Main implementation
├── answer_key.json          # Standard answers
├── config.yaml              # Configuration
├── requirements.txt         # Python dependencies
├── models/                  # Local model cache
│   ├── ch_PP-OCRv4_det_infer/
│   ├── ch_PP-OCRv4_rec_infer/
│   └── UVDoc/
├── test_data/math/          # Input: student assignments
│   ├── math_paper_1.jpg
│   └── math_paper_2.jpg
└── outputs/                 # Output: grading results
    ├── math_paper_1_result.json
    └── math_paper_2_result.json
```

## ⚙️ Configuration

**answer_key.json:**
```json
{
  "1": {"type": "choice", "answer": "B", "score": 2},
  "7": {"type": "blank", "answer": "0.5", "tolerance": 0.05, "score": 3},
  "19": {"type": "calculation", "max_score": 10}
}
```

**config.yaml:**
```yaml
subject: "Math"
question_type_map:
  1-6: choice
  7-18: blank
  19-25: calculation

vlm_service:
  base_url: "http://127.0.0.1:9900"
  timeout: 30
  max_retries: 2

ocr_config:
  lang: 'ch'
  use_gpu: false

concurrent_workers: 2
```

## ✅ Implementation Status

- [x] Design document completed (DESIGN.md)
- [x] Core pipeline implemented
- [x] Three-stage grading architecture
  - [x] Image preprocessing (CLAHE/aggressive)
  - [x] Question segmentation (regex-based)
  - [x] Type-specific grading (choice/blank/calculation)
- [x] PaddleOCR integration (Chinese OCR)
- [x] VLM integration (Qwen2.5-VL-3B)
- [x] Concurrent processing (ThreadPoolExecutor)
- [x] JSON output format
- [x] Tested with sample data

## ⚠️ Known Issues

1. **Environment:** Requires main venv_smartclassroom for PaddlePaddle compatibility
2. **OCR Accuracy:** May need better image preprocessing for rotated/blurred images
3. **Answer Extraction:** Regex patterns may need tuning for different question formats

## 🔧 Troubleshooting

**If VLM service not running:**
```bash
python start_services.py --services vlm
```

**If OCR models missing:**
Models are automatically downloaded to `C:\Users\user\.paddleocr\` on first run.

## 📚 References

- Design Doc: [DESIGN.md](DESIGN.md)
- GitHub Issue: https://github.com/open-edge-platform/edge-ai-suites/issues/2781
- Jira Ticket: ITEP-93144
