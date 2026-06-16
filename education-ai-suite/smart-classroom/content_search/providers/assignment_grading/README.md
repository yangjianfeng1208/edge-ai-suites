# Assignment Grading Service

Automated grading system for Chinese Mandarin history exam assignments using OCR and VLM.

## Quick Start

**Prerequisites:**
- VLM service running at `http://127.0.0.1:9900`
- Python 3.8+

**Setup:**
```bash
cd content_search/providers/assignment_grading
pip install -r requirements.txt
```

**Prepare Test Data:**
1. Place assignment images (JPG/PNG) in `test_data/math/` folder
2. Configure `answer_key.json` with standard answers
3. Adjust `config.yaml` if needed

**Run Grading:**
```bash
python grading_prototype.py
```

**Results:**
- Output files saved to `outputs/` folder
- Each assignment gets a `{filename}_result.json`

## Project Structure

```
assignment_grading/
├── DESIGN.md                # Detailed design document
├── README.md               # This file
├── grading_prototype.py    # Main script (to be implemented)
├── answer_key.json         # Standard answers template
├── config.yaml            # Configuration template
├── requirements.txt       # Python dependencies
├── test_data/            # Input: student assignments (images)
│   └── math/            # Subject-specific folder
│       ├── math_paper_1.jpg
│       └── math_paper_2.jpg
└── outputs/              # Output: grading results (JSON)
    └── (results appear here)
```

## Configuration

**answer_key.json:**
```json
{
  "1": {"type": "choice", "answer": "B", "score": 2},
  "7": {"type": "blank", "answer": "0.5", "tolerance": 0.05, "score": 3},
  "19": {"type": "essay", "max_score": 10}
}
```

**config.yaml:**
```yaml
question_type_map:
  1-6: choice
  7-18: blank
  19-25: essay
```

## Implementation Status

- [x] Design document completed
- [ ] Core pipeline implementation
- [ ] Testing with sample data
- [ ] Performance optimization

## References

- Design Doc: `DESIGN.md`
- GitHub Issue: https://github.com/open-edge-platform/edge-ai-suites/issues/2781
- Jira Ticket: ITEP-93144
