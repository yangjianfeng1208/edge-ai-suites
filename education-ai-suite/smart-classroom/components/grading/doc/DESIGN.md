# Assignment Grading System Design

**Date:** 2026-06-15  
**Author:** Claude (Brainstorming Session)  
**Requirement:** [GitHub Issue #2781](https://github.com/open-edge-platform/edge-ai-suites/issues/2781), ITEP-93144

## Overview

Design a prototype assignment grading system for Smart Classroom that processes scanned student assignments (Chinese Mandarin, Math subject) using OCR and VLM models on edge devices.

**Key Goals:**
- Process multiple assignments (image format) simulating 4 concurrent scanners
- OCR-based grading for objective questions (choice/blank)
- VLM-based grading for subjective questions (calculation/proof)
- Local on-device processing (no cloud dependency)

## Requirements Summary

**Input:**
- Multiple image files (JPG/PNG) - student assignments
- Each assignment: mix of choice questions, blank questions, calculation questions
- Standard answer key file (for objective questions)

**Output:**
- JSON result files (one per assignment image)
- Each file contains: digitized text (OCR), per-question scores, total score

**Performance:**
- Target: Functional first, optimize later
- 4-way concurrent processing

**Models:**
- OCR: PaddleOCR (Chinese, built-in to grading app)
- VLM: Qwen2.5-VL-3B-Instruct (pre-started service at port 9900)

**Input Format Change:**
- Originally designed for multi-page PDFs
- **Updated to support single-page images (JPG/PNG)**
- Simplified processing (no PDF conversion needed)

## Design Decisions

### Architecture Choice: Simplified Prototype (Single Script)

**Selected Approach:** Single-file implementation (~500 lines)

**Rationale:**
- Fastest time to working prototype (1-2 days)
- Easy to understand and modify
- Suitable for POC and requirements validation
- Architecture refactoring deferred to later phase

**Trade-offs:**
- Lower maintainability (accepted for prototype phase)
- Harder to unit test individual components
- Difficult to extend with new question types
- Migration path: refactor to modular architecture when requirements stabilize

### OCR Integration: Built-in PaddleOCR

**Approach:** Direct PaddleOCR library import within grading app

**Rationale:**
- No dependency on Smart Classroom main app
- Simpler deployment (no separate OCR service)
- Better performance (no HTTP overhead)
- 4 threads share single OCR model instance (~300MB memory)

**Configuration Override:**
- Modify main `config.yaml` to enable Chinese OCR:
  ```yaml
  ocr:
    enabled: true
    provider: paddle
    det_model: ch_PP-OCRv3_det_infer
    rec_model: ch_PP-OCRv4_rec_infer
  ```

### VLM Integration: HTTP Client to Pre-started Service

**Approach:** Call existing VLM service (port 9900) via HTTP

**Rationale:**
- VLM service manually started by user (decoupled from grading app)
- Avoids GPU memory conflict with Summarizer
- Service can be restarted independently
- Simple HTTP client code

**Service Endpoint:** `http://127.0.0.1:9900/v1/chat/completions`

### Question Type Strategy

**Question Type Distribution (Hard-coded for prototype):**
- Questions 1-6: Choice questions (A/B/C/D)
- Questions 7-18: Blank questions (numeric, formulas, expressions)
- Questions 19-25: Calculation/proof questions (show work, 10-15 points each)

**Grading Methods:**
| Question Type | Method | Tools | Accuracy Target |
|---------------|--------|-------|-----------------|
| Choice | OCR + exact match | PaddleOCR | ≥99% |
| Blank (numeric) | OCR + tolerance match | PaddleOCR | ≥95% |
| Blank (formula/expression) | VLM visual verification | Qwen2.5-VL | ≥90% |
| Calculation/Proof | VLM evaluation with math reasoning | Qwen2.5-VL | ≥80% |

**Answer Key Format:**
```json
{
  "1": {"type": "choice", "answer": "B", "score": 2},
  "7": {"type": "blank", "answer": "0.5", "tolerance": 0.05, "score": 3},
  "19": {"type": "essay", "max_score": 10}
}
```

### Question Number Detection

**Challenge:** Multiple question number formats in test papers

**Approach:** Enumerable regex patterns
- Pattern 1: `r'^\d+\.\s'` (e.g., "1. ")
- Pattern 2: `r'^\d+\)\s'` (e.g., "1) ")
- Pattern 3: `r'^（\d+）'` (e.g., "（1）")
- Pattern 4: `r'^\(\d+\)\s'` (e.g., "(1) ")

**Fallback:** If regex fails, log warning and skip question (manual review needed)

### Image Preprocessing

**Scope:** Simple preprocessing (sufficient for good-quality scans)

**Pipeline:**
1. Grayscale conversion
2. Skew correction (based on text line detection)
3. Adaptive thresholding (binarization)

**Deferred Features:**
- Perspective correction (not needed for current test data)
- Four-corner detection
- Advanced noise reduction

### Scoring Strategy

**Objective Questions (Choice/Blank):**
- Use standard answer key for ground truth
- Exact match or tolerance-based comparison
- Binary scoring: correct (full score) or incorrect (0 score)

**Subjective Questions (Calculation/Proof):**
- VLM evaluates based on mathematical reasoning and correctness
- No detailed rubric provided (deferred to later phase)
- VLM outputs score (0-max_score) and brief comment
- Prompt template:
  ```
  This is a Math exam calculation/proof question.
  Question #: {number}
  Student answer (image + OCR text): attached
  
  Evaluate the answer based on:
  - Correct final answer
  - Valid mathematical steps
  - Clear reasoning
  
  Output JSON: {"score": 8, "comment": "Correct answer, clear work shown..."}
  ```

### Performance Considerations

**Current Estimate (Conservative):**
- Image preprocessing: 1 sec per page
- OCR: 0.5-1 sec per page
- Choice/Blank grading: 0.1-0.5 sec per question
- Essay grading: 3-6 sec per question (VLM inference)
- 4-way concurrency: ~4 minutes for 30 assignments

**Optimization Strategy:**
- Phase 1: Implement functional pipeline (accept 4-5 min runtime)
- Phase 2: Profile bottlenecks (likely VLM inference)
- Phase 3: Optimize (parallel VLM calls, batch processing, model optimization)

## Architecture

### File Structure

```
content_search/providers/assignment_grading/
├── grading_prototype.py      # Main script (~500 lines)
├── answer_key.json          # Standard answers
├── config.yaml              # Question type mapping
├── requirements.txt         # Dependencies
├── test_data/              # Input folder
│   └── math/               # Subject-specific folder
│       ├── math_paper_1.jpg
│       ├── math_paper_2.jpg
│       └── ...
└── outputs/                # Output folder
    ├── math_paper_1_result.json
    └── ...
```

### Core Pipeline (Three Stages)

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  ① Preprocess     │ ───▶ │  ② Segment        │ ───▶ │  ③ Grade          │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ Skew correction   │      │ OCR full page     │      │ Choice → OCR      │
│ Adaptive threshold│      │ Regex extract Q#  │      │ Blank → OCR/VLM   │
│ → Clean images    │      │ → Question regions│      │ Essay → VLM       │
└──────────────────┘      └──────────────────┘      └──────────────────┘
    Tool: OpenCV              Tool: PaddleOCR          Tools: OCR + VLM
```

### Main Function Flow

```python
def main():
    # 1. Load configuration
    answer_key = load_answer_key("answer_key.json")
    question_type_map = load_config("config.yaml")
    
    # 2. Initialize models
    ocr = PaddleOCR(lang='ch', use_gpu=False)
    vlm_client = VLMClient("http://127.0.0.1:9900")
    
    # 3. Scan input folder for images
    image_files = glob("test_data/**/*.jpg") + glob("test_data/**/*.png")
    
    # 4. Concurrent processing (4 workers)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(
            lambda f: process_single_assignment(
                f, ocr, vlm_client, answer_key, question_type_map
            ),
            image_files
        )
    
    # 5. Aggregate statistics (optional)
    print_summary(results)
```

### Single Assignment Processing

```python
def process_single_assignment(image_path, ocr, vlm, answer_key, type_map):
    """Process one student assignment through 3-stage pipeline."""
    
    # Stage 1: Image preprocessing
    image = cv2.imread(image_path)
    processed_image = simple_preprocess(image)
    
    # Stage 2: Question segmentation
    ocr_result = ocr.ocr(processed_image)
    questions = extract_questions_from_ocr(ocr_result)
    
    # Stage 3: Per-question grading
    results = []
    for q in questions:
        q_type = type_map.get(str(q.number), "unknown")
        
        if q_type == "choice":
            score = grade_choice_question(q, answer_key, ocr)
        elif q_type == "blank":
            score = grade_blank_question(q, answer_key, ocr, vlm)
        elif q_type == "essay":
            score = grade_essay_question(q, vlm)
        else:
            score = {"score": 0, "error": "Unknown question type"}
        
        results.append({
            "question_number": q.number,
            "type": q_type,
            "score": score
        })
    
    # Output result file
    output_path = f"outputs/{Path(image_path).stem}_result.json"
    save_result_json(output_path, results)
    
    return results
```

## Component Details

### Stage 1: Image Preprocessing

```python
def simple_preprocess(image: np.ndarray) -> np.ndarray:
    """Apply skew correction and adaptive thresholding."""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Skew correction (based on connected components)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    
    # Rotate to correct skew
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), 
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    
    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        rotated, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    return binary
```

### Stage 2: Question Segmentation

```python
def extract_questions_from_ocr(ocr_result: list) -> list:
    """Extract question numbers and regions from OCR output."""
    
    # Multiple regex patterns for question numbers
    patterns = [
        (r'^\d+\.\s', 'standard'),      # "1. "
        (r'^\d+\)\s', 'parenthesis'),   # "1) "
        (r'^（\d+）', 'chinese_paren'), # "（1）"
        (r'^\(\d+\)\s', 'english_paren'),# "(1) "
    ]
    
    questions = []
    for line in ocr_result[0]:  # PaddleOCR returns nested structure
        text = line[1][0]
        bbox = line[0]
        
        for pattern, pattern_name in patterns:
            match = re.match(pattern, text)
            if match:
                q_number = int(re.findall(r'\d+', match.group())[0])
                questions.append({
                    'number': q_number,
                    'bbox': bbox,
                    'text': text,
                    'pattern': pattern_name
                })
                break
    
    return questions
```

### Stage 3: Grading Functions

**Choice Question Grader:**
```python
def grade_choice_question(question, answer_key, ocr):
    """Grade multiple-choice question via OCR."""
    key = answer_key[str(question.number)]
    
    # Extract answer region (assume fixed layout)
    answer_region = crop_answer_region(question.bbox)
    
    # OCR to extract A/B/C/D
    ocr_text = ocr.ocr(answer_region)[0][0][1][0]
    student_answer = extract_letter(ocr_text)  # Regex: [A-D]
    
    correct = (student_answer == key['answer'])
    return {
        'score': key['score'] if correct else 0,
        'correct': correct,
        'student_answer': student_answer,
        'expected': key['answer']
    }
```

**Blank Question Grader:**
```python
def grade_blank_question(question, answer_key, ocr, vlm):
    """Grade fill-in-the-blank question."""
    key = answer_key[str(question.number)]
    answer_region = crop_answer_region(question.bbox)
    
    # Try OCR first
    ocr_text = ocr.ocr(answer_region)[0][0][1][0].strip()
    
    # Numeric answers: tolerance matching
    if 'tolerance' in key:
        try:
            student_val = float(ocr_text)
            expected_val = float(key['answer'])
            tolerance = key.get('tolerance', 0.05)
            correct = abs(student_val - expected_val) <= tolerance
            return {
                'score': key['score'] if correct else 0,
                'correct': correct,
                'student_answer': ocr_text,
                'expected': key['answer']
            }
        except ValueError:
            pass  # Fall through to VLM
    
    # Text/symbol answers: exact match or VLM verification
    if ocr_text == key['answer']:
        return {'score': key['score'], 'correct': True}
    
    # Fallback: VLM visual verification
    result = vlm.verify_answer(
        question_image=answer_region,
        expected=key['answer'],
        student_ocr=ocr_text
    )
    return result
```

**Essay Question Grader:**
```python
def grade_essay_question(question, vlm_client):
    """Grade essay question via VLM."""
    
    # Prepare prompt
    prompt = f"""
You are grading a History exam essay question.

Question Number: {question.number}
Student Answer: (see attached image and OCR text below)

OCR Text:
{question.ocr_text}

Task:
- Evaluate the answer based on general historical knowledge
- Assign a score (0-10)
- Provide a brief comment (1-2 sentences)

Output format (JSON):
{{"score": 8, "comment": "Good understanding of key events..."}}
"""
    
    # Call VLM service
    response = vlm_client.chat(
        prompt=prompt,
        images=[question.answer_image],
        max_tokens=200
    )
    
    # Parse JSON response
    result = json.loads(response)
    return {
        'score': result['score'],
        'comment': result.get('comment', ''),
        'type': 'essay'
    }
```

### VLM Client

```python
class VLMClient:
    """HTTP client for Qwen2.5-VL service."""
    
    def __init__(self, base_url="http://127.0.0.1:9900"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def chat(self, prompt: str, images: list = None, max_tokens=512):
        """Send chat request to VLM service."""
        
        # Encode images to base64
        encoded_images = []
        if images:
            for img in images:
                _, buffer = cv2.imencode('.jpg', img)
                b64 = base64.b64encode(buffer).decode('utf-8')
                encoded_images.append(f"data:image/jpeg;base64,{b64}")
        
        # Build request payload
        payload = {
            "model": "Qwen2.5-VL-3B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {"type": "image_url", "image_url": {"url": img}}
                        for img in encoded_images
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        # Send request
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Extract response text
        result = response.json()
        return result['choices'][0]['message']['content']
```

## Configuration Files

**answer_key.json:**
```json
{
  "1": {"type": "choice", "answer": "B", "score": 2},
  "2": {"type": "choice", "answer": "C", "score": 2},
  "7": {"type": "blank", "answer": "0.5", "tolerance": 0.05, "score": 3},
  "8": {"type": "blank", "answer": "v=s/t", "score": 3},
  "19": {"type": "essay", "max_score": 10},
  "20": {"type": "essay", "max_score": 15}
}
```

**config.yaml:**
```yaml
question_type_map:
  1: choice
  2: choice
  3: choice
  4: choice
  5: choice
  6: choice
  7: blank
  8: blank
  9: blank
  10: blank
  11: blank
  12: blank
  13: blank
  14: blank
  15: blank
  16: blank
  17: blank
  18: blank
  19: essay
  20: essay
  21: essay
  22: essay
  23: essay
  24: essay
  25: essay

vlm_service:
  base_url: "http://127.0.0.1:9900"
  timeout: 30
  max_retries: 2

ocr_config:
  lang: 'ch'
  use_gpu: false
  det_db_thresh: 0.3
  rec_image_shape: '3,48,320'
```

## Output Format

**Per-Assignment Result (JSON):**
```json
{
  "assignment_id": "student_001",
  "total_score": 85,
  "max_score": 100,
  "questions": [
    {
      "question_number": 1,
      "type": "choice",
      "score": 2,
      "max_score": 2,
      "correct": true,
      "student_answer": "B",
      "expected": "B"
    },
    {
      "question_number": 7,
      "type": "blank",
      "score": 3,
      "max_score": 3,
      "correct": true,
      "student_answer": "0.52",
      "expected": "0.5"
    },
    {
      "question_number": 19,
      "type": "essay",
      "score": 8,
      "max_score": 10,
      "comment": "Good understanding of key events, missing some details on social impact."
    }
  ],
  "processing_time_seconds": 31.2,
  "timestamp": "2026-06-15T10:30:45Z"
}
```

## Dependencies

**Python Packages (requirements.txt):**
```
paddleocr>=2.7.0
opencv-python>=4.8.0
numpy>=1.24.0
requests>=2.31.0
Pillow>=10.0.0
pyyaml>=6.0
```

**System Dependencies:**
- PaddlePaddle (CPU version)

**Note:** `pdf2image` and `Poppler` dependencies removed (not needed for image-only input)

## Testing Strategy

**Phase 1: Unit Testing (Deferred)**
- Test each grading function independently
- Mock OCR/VLM responses

**Phase 2: Integration Testing**
- Process sample image assignments (starting with 2 math papers)
- Manually verify grading accuracy
- Validate output JSON format

**Phase 3: Performance Testing**
- Process multiple assignments
- Measure end-to-end time
- Identify bottlenecks (profiling)

## Future Enhancements

**Architecture:**
- Refactor to modular service architecture (Provider pattern)
- Separate stages into independent modules
- Add unit tests for each component

**Features:**
- Advanced image preprocessing (perspective correction, noise reduction)
- RAG-based essay grading (provide historical knowledge base)
- Detailed rubric support for essay questions
- Support for more question types (matching, true/false, etc.)
- Question type auto-detection (remove hard-coded mapping)

**Performance:**
- Parallel VLM calls for essay questions
- Batch VLM inference
- Model optimization (quantization, pruning)
- GPU acceleration for OCR

**Robustness:**
- Error handling and retry logic
- Logging and monitoring
- Partial result recovery (if grading fails mid-process)
- Quality checks (OCR confidence scores, VLM response validation)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OCR accuracy on handwriting | High | Use VLM fallback for low-confidence OCR |
| VLM service unavailable | High | Check health endpoint before processing, retry logic |
| Question number detection fails | Medium | Log warnings, allow manual review |
| Performance below 3-minute target | Low | Accepted for Phase 1, optimize in Phase 2 |
| Memory exhaustion (4 concurrent workers) | Medium | Monitor memory usage, reduce workers if needed |

## Success Criteria

**Functional:**
- ✅ Process image assignments (JPG/PNG) end-to-end
- ✅ Generate JSON output files with scores
- ✅ Choice questions: ≥95% grading accuracy
- ✅ Blank questions: ≥90% grading accuracy
- ✅ Calculation questions: VLM produces valid score (0-max)

**Non-Functional:**
- ⏱️ Functional first, performance optimization later
- 💾 Total memory usage <4GB (OCR + 4 workers)
- 🔧 Code is runnable with minimal setup (pip install + config)

## Conclusion

This design provides a simple, working prototype for assignment grading that:
- Validates the three-stage pipeline approach
- Tests OCR and VLM integration
- Delivers functional grading capability within 1-2 days

The single-file architecture intentionally trades maintainability for speed of implementation. Once requirements are validated through this prototype, the system can be refactored into a production-grade modular architecture.
