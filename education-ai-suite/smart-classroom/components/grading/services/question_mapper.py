"""Question Mapper - Maps OCR regions to questions based on answer_key.json"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


# Fallback patterns used only when config provides none.
_DEFAULT_QUESTION_PATTERNS = [
    r'第?\s*(\d+)\s*[题\.、]',
    r'(\d+)\s*[\.。）\)、]',
    r'\((\d+)\)',
]
_DEFAULT_SEPARATORS = ['.']


def extract_question_numbers(text: str, patterns: Optional[List[str]] = None) -> List[str]:
    """Extract question numbers from text using configured regex patterns."""
    active = patterns if patterns else _DEFAULT_QUESTION_PATTERNS

    numbers = []
    for pattern in active:
        numbers.extend(re.findall(pattern, text))

    return list(set(numbers))


def match_regions_to_questions(
    ocr_results: Dict[int, List[Dict]],
    answer_key: Dict[str, Any],
    strategy: str = "position",
    patterns: Optional[List[str]] = None,
    separators: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Match OCR regions to questions in answer key"""
    mapping = {}
    seps = separators if separators else _DEFAULT_SEPARATORS

    objective_questions = answer_key.get('objective_questions', {})
    subjective_questions = answer_key.get('subjective_questions', {})

    all_regions = []
    for page_num, regions in ocr_results.items():
        for region in regions:
            all_regions.append({
                **region,
                'page': page_num
            })

    all_regions.sort(key=lambda r: (r['page'], r['bbox'][1], r['bbox'][0]))

    if strategy == "position":
        region_idx = 0

        for q_id in sorted(objective_questions.keys(), key=lambda x: int(x)):
            q_data = objective_questions[q_id]

            if region_idx < len(all_regions):
                region = all_regions[region_idx]
                mapping[q_id] = {
                    "type": "objective",
                    "question_type": q_data.get('type'),
                    "score": q_data.get('score'),
                    "format": q_data.get('format'),
                    "regions": [{
                        "page": region['page'],
                        "region_id": region['region_id'],
                        "bbox": region['bbox'],
                        "content": region['content']
                    }],
                    "merged_content": region['content'],
                    "answer": q_data.get('answer', [])
                }
                region_idx += 1

        for q_id in sorted(subjective_questions.keys(), key=lambda x: int(x)):
            q_data = subjective_questions[q_id]

            question_regions = []
            start_idx = region_idx

            max_regions = 5
            collected = 0

            while region_idx < len(all_regions) and collected < max_regions:
                region = all_regions[region_idx]
                question_regions.append({
                    "page": region['page'],
                    "region_id": region['region_id'],
                    "bbox": region['bbox'],
                    "content": region['content']
                })
                region_idx += 1
                collected += 1

                next_q_id = str(int(q_id) + 1)
                if any(f"{next_q_id}{sep}" in region['content'] for sep in seps):
                    break

            if question_regions:
                merged_content = "\n".join([r['content'] for r in question_regions])
                mapping[q_id] = {
                    "type": "subjective",
                    "question_type": q_data.get('type'),
                    "score": q_data.get('score'),
                    "alias": q_data.get('alias'),
                    "rubric": q_data.get('rubric'),
                    "regions": question_regions,
                    "merged_content": merged_content
                }

    elif strategy == "content":
        for region in all_regions:
            content = region['content']
            found_numbers = extract_question_numbers(content, patterns)

            for q_num in found_numbers:
                q_data_obj = objective_questions.get(q_num)
                q_data_subj = subjective_questions.get(q_num)

                if q_data_obj:
                    if q_num not in mapping:
                        mapping[q_num] = {
                            "type": "objective",
                            "question_type": q_data_obj.get('type'),
                            "score": q_data_obj.get('score'),
                            "format": q_data_obj.get('format'),
                            "regions": [],
                            "answer": q_data_obj.get('answer', [])
                        }

                    mapping[q_num]['regions'].append({
                        "page": region['page'],
                        "region_id": region['region_id'],
                        "bbox": region['bbox'],
                        "content": region['content']
                    })

                elif q_data_subj:
                    if q_num not in mapping:
                        mapping[q_num] = {
                            "type": "subjective",
                            "question_type": q_data_subj.get('type'),
                            "score": q_data_subj.get('score'),
                            "alias": q_data_subj.get('alias'),
                            "rubric": q_data_subj.get('rubric'),
                            "regions": []
                        }

                    mapping[q_num]['regions'].append({
                        "page": region['page'],
                        "region_id": region['region_id'],
                        "bbox": region['bbox'],
                        "content": region['content']
                    })

        for q_id in mapping:
            merged = "\n".join([r['content'] for r in mapping[q_id]['regions']])
            mapping[q_id]['merged_content'] = merged

    return mapping


def _patterns_from_config(config: Optional[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    """Read question-number regex patterns and boundary separators from config."""
    qp = config.get("question_parsing", {}) if isinstance(config, dict) else {}
    if not isinstance(qp, dict):
        qp = {}

    raw_patterns = qp.get("question_patterns", {})
    if isinstance(raw_patterns, dict):
        patterns = [str(v) for v in raw_patterns.values() if str(v).strip()]
    elif isinstance(raw_patterns, list):
        patterns = [str(v) for v in raw_patterns if str(v).strip()]
    else:
        patterns = []

    raw_seps = qp.get("question_number_separators", [])
    separators = [str(s) for s in raw_seps if str(s)] if isinstance(raw_seps, list) else []

    return patterns, separators


def map_questions_to_regions(
    ocr_dir: Path,
    answer_key_path: Path,
    output_path: Path,
    strategy: str = "position",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main function to map questions to OCR regions"""
    patterns, separators = _patterns_from_config(config)

    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    ocr_results = {}

    summary_path = ocr_dir / "ocr_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        pages_info = summary.get('pages', {})
        for page_num_str, page_data in pages_info.items():
            page_num = int(page_num_str)
            ocr_json = Path(page_data['ocr_json'])

            if ocr_json.exists():
                with open(ocr_json, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    ocr_results[page_num] = page_data.get('results', [])
    else:
        for ocr_file in sorted(ocr_dir.glob("page_*_ocr.json")):
            page_num = int(ocr_file.stem.split('_')[1])
            with open(ocr_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                ocr_results[page_num] = page_data.get('results', [])

    mapping = match_regions_to_questions(
        ocr_results,
        answer_key,
        strategy,
        patterns=patterns or None,
        separators=separators or None,
    )

    output_data = {
        "source": {
            "ocr_dir": str(ocr_dir),
            "answer_key": str(answer_key_path)
        },
        "strategy": strategy,
        "total_questions": len(mapping),
        "objective_questions": sum(1 for q in mapping.values() if q['type'] == 'objective'),
        "subjective_questions": sum(1 for q in mapping.values() if q['type'] == 'subjective'),
        "questions": mapping
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_data

