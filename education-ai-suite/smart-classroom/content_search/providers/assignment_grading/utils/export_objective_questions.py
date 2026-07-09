"""
Export Objective Questions

Extracts and displays each objective question from OCR results.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_questions_from_ocr(
    ocr_dir: Path,
    answer_key_path: Path,
    output_txt: Path
):
    """
    Extract and display each objective question

    Args:
        ocr_dir: Directory with OCR results
        answer_key_path: Path to answer_key.json
        output_txt: Output text file
    """
    print(f"\n{'='*80}")
    print("Extracting Objective Questions from OCR")
    print(f"{'='*80}")

    # Load answer key
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    objective_questions = answer_key.get('objective_questions', {})

    if not objective_questions:
        print("  No objective questions found in answer key")
        return

    print(f"  Objective questions in answer key: {len(objective_questions)}")

    # Load OCR results
    ocr_files = sorted(ocr_dir.glob("page_*_ocr.json"))
    print(f"  OCR files found: {len(ocr_files)}")

    # Collect all OCR regions
    all_regions = []
    for ocr_file in ocr_files:
        page_num = int(ocr_file.stem.split('_')[1])

        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        for result in ocr_data.get('results', []):
            all_regions.append({
                'page': page_num,
                'type': result.get('type', 'unknown'),
                'content': result.get('content', '').strip(),
                'bbox': result.get('bbox', [])
            })

    print(f"  Total OCR regions: {len(all_regions)}")

    # Find which questions are group type
    group_questions = set()
    for q_id, q_info in objective_questions.items():
        if q_info.get('type') == 'group':
            group_questions.add(q_id)

    print(f"  Group questions: {group_questions}")

    # Define instruction section titles (blacklist)
    INSTRUCTION_TITLES = [
        '考试说明',
        '注意事项',
        '考生注意',
        '答题要求',
        '考场规则',
        '作答须知',
        '填写说明',
        'instructions',
        'directions',
        'notice'
    ]

    # Find question regions (skip instruction sections)
    question_regions = []
    in_instruction_section = False
    current_group_question = None  # Track which group question we're in

    for region in all_regions:
        content = region['content'].strip()
        region_type = region.get('type', 'text')

        # Check if this is a section title
        if region_type == 'paragraph_title' or region_type == 'doc_title':
            # Check if it's an instruction section
            is_instruction = any(
                keyword in content
                for keyword in INSTRUCTION_TITLES
            )

            if is_instruction:
                in_instruction_section = True
                print(f"    Entering instruction section: {content}")
            else:
                in_instruction_section = False
                print(f"    Entering content section: {content}")
            continue

        # Skip if we're in instruction section
        if in_instruction_section:
            continue

        # Only collect text regions
        if region_type != 'text':
            continue

        # Check if starts with question number
        # Support formats:
        # - Main question: "1.", "2."
        # - Sub-question: "1）", "(1)", "（1）", "1.(1)"

        # Try sub-question patterns first
        sub_patterns = [
            (r'^(\d+)[）\)]', 'simple'),             # "1）" or "1)"
            (r'^[（\(](\d+)[）\)]', 'parenthesis'),   # "(1)" or "（1）"
            (r'^(\d+)[\.\s]*[（\(](\d+)[）\)]', 'compound'),  # "1.(1)" or "1 (1)"
        ]

        matched = False
        for pattern, ptype in sub_patterns:
            match = re.match(pattern, content)
            if match:
                if ptype == 'simple':
                    # Format: "1）" - associate with current group if in one
                    sub_num = match.group(1)
                    if current_group_question and current_group_question in group_questions:
                        # This is a sub-question of the current group
                        q_num = f"{current_group_question}({sub_num})"
                    else:
                        # Not in a group context, skip
                        matched = True
                        break
                elif ptype == 'parenthesis':
                    # Format: "(1)" - associate with current group if in one
                    sub_num = match.group(1)
                    if current_group_question and current_group_question in group_questions:
                        q_num = f"{current_group_question}({sub_num})"
                    else:
                        # Not in a group context, skip
                        matched = True
                        break
                else:  # compound
                    # Format: "1.(1)" - has both parent and sub
                    parent_num = match.group(1)
                    sub_num = match.group(2)
                    q_num = f"{parent_num}({sub_num})"

                question_regions.append({
                    'question_num': q_num,
                    'page': region['page'],
                    'content': content
                })
                matched = True
                break

        if not matched:
            # Try main question pattern
            match = re.match(r'^(\d+)\.', content)
            if match:
                q_num = match.group(1)
                question_regions.append({
                    'question_num': q_num,
                    'page': region['page'],
                    'content': content
                })

                # Check if this is a group question
                # (we'll need answer_key for this - defer to later)
                current_group_question = q_num
            else:
                # Not a question number - might be end of group
                # Reset if we see a new section
                if region_type in ['paragraph_title', 'doc_title']:
                    current_group_question = None

    print(f"  Questions found in OCR: {len(question_regions)}")

    # Flatten questions (expand groups into sub_questions)
    flat_questions = []
    for q_id in sorted(objective_questions.keys(), key=lambda x: int(x)):
        q_info = objective_questions[q_id]
        q_type = q_info.get('type', 'unknown')

        # Handle group type - expand to sub_questions
        if q_type == 'group' and 'sub_questions' in q_info:
            for sub_id, sub_info in q_info['sub_questions'].items():
                flat_questions.append({
                    'full_id': f"{q_id}{sub_id}",  # e.g., "1(1)", "1(2)"
                    'parent_id': q_id,
                    'sub_id': sub_id,
                    'type': sub_info.get('type', 'unknown'),
                    'answer': sub_info.get('answer', []),
                    'score': sub_info.get('score', 0)
                })
        else:
            # Regular question (not a group)
            flat_questions.append({
                'full_id': q_id,
                'parent_id': None,
                'sub_id': None,
                'type': q_type,
                'answer': q_info.get('answer', []),
                'score': q_info.get('score', 0)
            })

    # Output to file
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Objective Questions Extraction\n")
        f.write("="*80 + "\n\n")

        # Process each flattened question
        for q in flat_questions:
            full_id = q['full_id']
            q_type = q['type']
            standard_answer = q['answer']

            f.write(f"Question {full_id} ({q_type})\n")
            f.write("-"*80 + "\n")
            f.write(f"Standard Answer: {standard_answer}\n\n")

            # Try to find in OCR
            found = False

            for q_region in question_regions:
                q_num = q_region['question_num']
                content = q_region['content']

                # Direct match on question number
                if q_num == full_id:
                    f.write(f"Page: {q_region['page']}\n")
                    f.write(f"Content:\n")
                    f.write(f"  {content}\n\n")

                    # Try to extract answer
                    extracted = extract_answer(content, q_type)
                    if extracted:
                        f.write(f"Extracted Answer: {extracted}\n")
                        match = extracted in standard_answer
                        f.write(f"Match: {'✓ CORRECT' if match else '✗ INCORRECT'}\n")
                    else:
                        f.write(f"Extracted Answer: None\n")

                    found = True
                    break

            if not found:
                f.write(f"Status: NOT FOUND IN OCR\n")

            f.write("\n" + "="*80 + "\n\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Total objective questions: {len(objective_questions)}\n")
        f.write(f"Questions found in OCR: {len(question_regions)}\n")

    print(f"\n  Output saved: {output_txt}")


def extract_answer(content: str, q_type: str) -> str:
    """
    Extract answer from question content

    Args:
        content: Question content string
        q_type: Question type (choice, blank, etc.)

    Returns:
        Extracted answer or None
    """
    if q_type == 'choice':
        # Look for answer in brackets: （A）, (B), ( B ), etc.
        match = re.search(r'[（\(]\s*([A-D])\s*[）\)]', content)
        if match:
            return match.group(1)

    elif q_type == 'blank':
        # Look for underlined answer: \underline{...} or \underline{\text{...}}
        match = re.search(r'\\underline\{(?:\\text\{)?(.+?)\}+', content)
        if match:
            answer = match.group(1)
            # Clean up LaTeX commands
            answer = answer.replace('\\text{', '').replace('}', '')
            return answer.strip()

        # Look for bare answer in the content (for questions without underline)
        # Try to extract Chinese text after 、 or ，
        match = re.search(r'[，、]([^，。]+)[。，（]', content)
        if match:
            answer = match.group(1).strip()
            if len(answer) < 50 and not any(c in answer for c in ['(', ')', '（', '）']):
                return answer

        # Or look for equals sign: = ...
        match = re.search(r'=\s*(.+?)(?:\.|$)', content)
        if match:
            answer = match.group(1).strip()
            if len(answer) < 50:  # Reasonable length
                return answer

    return None
