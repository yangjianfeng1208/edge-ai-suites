"""
Extract Objective Question Answers from Question Mapping

Extracts each objective question's content and attempts to identify the answer.
Outputs a readable text file for verification and debugging.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def extract_choice_answer(content: str, regions: List[Dict]) -> Tuple[Optional[str], float, str]:
    """
    Extract choice answer (A/B/C/D) from content

    Args:
        content: Merged content string
        regions: List of regions for this question

    Returns:
        (answer, confidence, method)
        - answer: 'A', 'B', 'C', 'D' or None
        - confidence: 0.0-1.0
        - method: description of extraction method
    """
    candidates = []

    # Method 1: Find in merged content with brackets
    # Patterns: （A）, (B), 【C】, [D]
    pattern1 = r'[（\(\[【]\s*([A-D])\s*[）\)\]】]'
    matches = list(re.finditer(pattern1, content))
    if matches:
        # Take the last match (in case student changed answer)
        answer = matches[-1].group(1)
        confidence = 0.8 if len(matches) == 1 else 0.6  # Lower if multiple
        candidates.append((answer, confidence, f"Method1: Bracket pattern, {len(matches)} match(es)"))

    # Method 2: Find standalone A/B/C/D
    pattern2 = r'\b([A-D])\b'
    matches = list(re.finditer(pattern2, content))
    if matches:
        answer = matches[-1].group(1)
        confidence = 0.5
        candidates.append((answer, confidence, "Method2: Standalone letter"))

    # Method 3: Check individual regions for pure choice
    for i, region in enumerate(regions):
        region_content = region['content'].strip()

        # Pure choice format: exactly "A" or "(A)" or "（A）"
        if re.match(r'^[（\(\[【]?[A-D][）\)\]】]?$', region_content):
            answer = re.search(r'[A-D]', region_content).group(0)
            confidence = 0.9  # High confidence for pure choice
            candidates.append((answer, confidence, f"Method3: Pure choice in region {i+1}"))

    # Select best candidate
    if not candidates:
        return None, 0.0, "No answer found"

    best = max(candidates, key=lambda x: x[1])
    return best


def extract_blank_answer(content: str, regions: List[Dict]) -> Tuple[Optional[str], float, str]:
    """
    Extract blank answer from content

    Args:
        content: Merged content string
        regions: List of regions for this question

    Returns:
        (answer, confidence, method)
    """
    candidates = []

    # Method 1: Look for LaTeX content (common in math answers)
    latex_pattern = r'\\[()\w\{\}^_\-+=\*\/\d\s]+'
    latex_matches = re.findall(latex_pattern, content)
    if latex_matches:
        # Take the last/longest one
        answer = max(latex_matches, key=len)
        confidence = 0.7
        candidates.append((answer, confidence, "Method1: LaTeX formula"))

    # Method 2: Look for short text answers (< 30 chars)
    # Usually in regions with short content
    for i, region in enumerate(regions):
        region_content = region['content'].strip()

        # Skip question number regions
        if re.match(r'^\d+[\.、）)]', region_content):
            continue

        # Short answer
        if 1 < len(region_content) < 30:
            confidence = 0.6 if len(region_content) < 10 else 0.4
            candidates.append((region_content, confidence, f"Method2: Short text in region {i+1}"))

    # Method 3: Extract from merged content after question number
    # Pattern: "7. xxxxx" -> take xxxxx
    match = re.search(r'^\d+[\.、）)]\s*(.+?)$', content, re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        if len(answer) < 50:
            confidence = 0.5
            candidates.append((answer, confidence, "Method3: Content after question number"))

    if not candidates:
        return None, 0.0, "No answer found"

    best = max(candidates, key=lambda x: x[1])
    return best


def extract_objective_answers(
    question_mapping_path: Path,
    answer_key_path: Path,
    output_txt_path: Path
) -> Dict[str, Any]:
    """
    Extract objective question answers and output to readable text file

    Args:
        question_mapping_path: Path to question_mapping.json
        answer_key_path: Path to answer_key.json
        output_txt_path: Path to output text file

    Returns:
        Dictionary with extraction results
    """
    print(f"\n{'='*80}")
    print("Extracting Objective Question Answers")
    print(f"{'='*80}")

    # Load question mapping
    with open(question_mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    questions = mapping_data.get('questions', {})

    # Load answer key
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    objective_questions = answer_key.get('objective_questions', {})

    print(f"  Total questions in mapping: {len(questions)}")
    print(f"  Objective questions in answer key: {len(objective_questions)}")

    # Extract answers
    results = {}
    extraction_stats = {
        'total': 0,
        'found': 0,
        'not_found': 0,
        'choice': 0,
        'blank': 0
    }

    # Output file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Objective Question Extraction Results\n")
        f.write("="*80 + "\n\n")

        # Process each objective question
        for q_id in sorted(objective_questions.keys(), key=lambda x: int(x)):
            q_info = objective_questions[q_id]
            q_type = q_info.get('type')

            extraction_stats['total'] += 1

            # Get mapping data
            if q_id not in questions:
                f.write(f"Question {q_id}: NOT FOUND IN MAPPING\n")
                f.write("-"*80 + "\n\n")
                extraction_stats['not_found'] += 1
                continue

            q_mapping = questions[q_id]

            # Question header
            f.write(f"Question {q_id} ({q_type})\n")
            f.write("-"*80 + "\n")

            # Standard answer
            standard_answer = q_info.get('answer', [])
            f.write(f"Standard Answer: {standard_answer}\n\n")

            # Regions info
            regions = q_mapping.get('regions', [])
            f.write(f"Regions ({len(regions)} total):\n")
            for i, region in enumerate(regions, 1):
                f.write(f"  [{i}] {region.get('region_id', 'unknown')}\n")
                if 'type' in region:
                    f.write(f"      Type: {region['type']}\n")
                f.write(f"      BBox: {region.get('bbox', 'N/A')}\n")
                f.write(f"      Content: \"{region.get('content', '')}\"\n")
            f.write("\n")

            # Merged content
            merged_content = q_mapping.get('merged_content', '')
            f.write(f"Merged Content:\n")
            f.write(f"  \"{merged_content}\"\n\n")

            # Extract answer
            if q_type == 'choice':
                extraction_stats['choice'] += 1
                answer, confidence, method = extract_choice_answer(merged_content, regions)
            elif q_type == 'blank':
                extraction_stats['blank'] += 1
                answer, confidence, method = extract_blank_answer(merged_content, regions)
            else:
                answer, confidence, method = None, 0.0, f"Unknown type: {q_type}"

            # Result
            if answer:
                extraction_stats['found'] += 1
                f.write(f"✓ Extracted Answer: {answer}\n")
            else:
                extraction_stats['not_found'] += 1
                f.write(f"✗ Extracted Answer: None\n")

            f.write(f"  Confidence: {confidence:.2f}\n")
            f.write(f"  Method: {method}\n")

            # Comparison
            if answer and standard_answer:
                is_correct = answer in standard_answer
                f.write(f"  Match: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")

            results[q_id] = {
                'question_type': q_type,
                'regions_count': len(regions),
                'merged_content': merged_content,
                'extracted_answer': answer,
                'confidence': confidence,
                'method': method,
                'standard_answer': standard_answer
            }

            f.write("\n" + "="*80 + "\n\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("Extraction Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Total Questions: {extraction_stats['total']}\n")
        f.write(f"  Choice Questions: {extraction_stats['choice']}\n")
        f.write(f"  Blank Questions: {extraction_stats['blank']}\n")
        f.write(f"\n")
        f.write(f"Answers Found: {extraction_stats['found']}/{extraction_stats['total']}\n")
        f.write(f"Answers Not Found: {extraction_stats['not_found']}/{extraction_stats['total']}\n")
        f.write(f"Success Rate: {extraction_stats['found']/extraction_stats['total']*100:.1f}%\n")

    print(f"\n  Extraction Results:")
    print(f"    Found: {extraction_stats['found']}/{extraction_stats['total']}")
    print(f"    Not Found: {extraction_stats['not_found']}/{extraction_stats['total']}")
    print(f"    Success Rate: {extraction_stats['found']/extraction_stats['total']*100:.1f}%")
    print(f"\n  Output saved: {output_txt_path}")

    return {
        'results': results,
        'stats': extraction_stats
    }
