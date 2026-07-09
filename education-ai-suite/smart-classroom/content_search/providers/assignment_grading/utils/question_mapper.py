"""
Question Mapper - Maps OCR regions to questions based on answer_key.json

This module provides intelligent matching between detected OCR regions and
questions defined in the answer key, enabling automated question-answer mapping.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_question_numbers(text: str) -> List[str]:
    """
    Extract question numbers from text

    Supports formats like:
    - "1.", "2.", "3."
    - "第1题", "第2题"
    - "1）", "2）"
    - "(1)", "(2)"

    Args:
        text: Text to extract question numbers from

    Returns:
        List of question numbers as strings
    """
    patterns = [
        r'第?\s*(\d+)\s*[题\.、]',  # 第1题, 1题, 1.
        r'(\d+)\s*[\.。）\)、]',     # 1., 1), 1、
        r'\((\d+)\)',                # (1)
        r'（(\d+）)',                # （1）
    ]

    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)

    return list(set(numbers))  # Remove duplicates


def match_regions_to_questions(
    ocr_results: Dict[int, List[Dict]],
    answer_key: Dict[str, Any],
    strategy: str = "position"
) -> Dict[str, Dict]:
    """
    Match OCR regions to questions in answer key

    Args:
        ocr_results: OCR results per page {page_num: [region_results]}
        answer_key: Answer key with objective_questions and subjective_questions
        strategy: Matching strategy - "position", "content", or "hybrid"

    Returns:
        Dictionary mapping question IDs to their regions and content
        {
            "1": {
                "type": "objective",
                "question_type": "choice",
                "score": 4,
                "regions": [
                    {
                        "page": 1,
                        "region_id": "page1_region5",
                        "bbox": [...],
                        "content": "A"
                    }
                ],
                "merged_content": "A",
                "answer": ["A"]  # Standard answer (if objective)
            },
            ...
        }
    """
    mapping = {}

    # Get all questions from answer key
    objective_questions = answer_key.get('objective_questions', {})
    subjective_questions = answer_key.get('subjective_questions', {})

    # Flatten all OCR regions with page info
    all_regions = []
    for page_num, regions in ocr_results.items():
        for region in regions:
            all_regions.append({
                **region,
                'page': page_num
            })

    # Sort regions by position (top to bottom, left to right)
    all_regions.sort(key=lambda r: (r['page'], r['bbox'][1], r['bbox'][0]))

    # Strategy 1: Position-based matching
    # Assume regions appear in question order
    if strategy == "position":
        region_idx = 0

        # Process objective questions first
        for q_id in sorted(objective_questions.keys(), key=lambda x: int(x)):
            q_data = objective_questions[q_id]

            # For simple questions, assign next available region
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

        # Process subjective questions
        for q_id in sorted(subjective_questions.keys(), key=lambda x: int(x)):
            q_data = subjective_questions[q_id]

            # Subjective questions may span multiple regions
            # Collect regions until we find next question marker or end
            question_regions = []
            start_idx = region_idx

            # Heuristic: collect 1-5 regions per subjective question
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

                # Stop if we see next question number
                content_lower = region['content'].lower()
                next_q_id = str(int(q_id) + 1)
                if f"题{next_q_id}" in content_lower or f"{next_q_id}." in content_lower:
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

    # Strategy 2: Content-based matching
    # Look for question numbers in OCR content
    elif strategy == "content":
        # Build reverse index: question number -> regions containing it
        for region in all_regions:
            content = region['content']
            found_numbers = extract_question_numbers(content)

            for q_num in found_numbers:
                # Check if this is objective or subjective
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

        # Merge content for each question
        for q_id in mapping:
            merged = "\n".join([r['content'] for r in mapping[q_id]['regions']])
            mapping[q_id]['merged_content'] = merged

    return mapping


def map_questions_to_regions(
    ocr_dir: Path,
    answer_key_path: Path,
    output_path: Path,
    strategy: str = "position"
) -> Dict[str, Any]:
    """
    Main function to map questions to OCR regions

    Args:
        ocr_dir: Directory containing OCR results (step2_ocr_regions/)
        answer_key_path: Path to answer_key.json
        output_path: Path to save mapping JSON
        strategy: Matching strategy

    Returns:
        Mapping result dictionary
    """
    # Load answer key
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    # Load OCR results from all pages
    ocr_results = {}

    # Read summary to get page list
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
        # Fallback: scan directory for page_*_ocr.json files
        for ocr_file in sorted(ocr_dir.glob("page_*_ocr.json")):
            page_num = int(ocr_file.stem.split('_')[1])
            with open(ocr_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                ocr_results[page_num] = page_data.get('results', [])

    # Perform mapping
    mapping = match_regions_to_questions(ocr_results, answer_key, strategy)

    # Build output structure
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

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_data
