"""
Grade Objective Questions

Automatically grades objective questions based on extracted answers.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_answer(content: str, q_type: str) -> str:
    """Extract answer from question content"""
    if q_type == 'choice':
        match = re.search(r'[（\(]\s*([A-D])\s*[）\)]', content)
        if match:
            return match.group(1)

    elif q_type == 'blank':
        match = re.search(r'\\underline\{(?:\\text\{)?(.+?)\}+', content)
        if match:
            answer = match.group(1)
            answer = answer.replace('\\text{', '').replace('}', '')
            return answer.strip()

        match = re.search(r'[，、]([^，。]+)[。，（]', content)
        if match:
            answer = match.group(1).strip()
            if len(answer) < 50 and not any(c in answer for c in ['(', ')', '（', '）']):
                return answer

        match = re.search(r'=\s*(.+?)(?:\.|$)', content)
        if match:
            answer = match.group(1).strip()
            if len(answer) < 50:
                return answer

    return None


def check_answer(extracted: str, standard: List[str], match_mode: str = 'any') -> bool:
    """
    Check if extracted answer matches standard answer

    Args:
        extracted: Extracted answer from student work
        standard: List of acceptable answers
        match_mode: 'any' (any match), 'all' (all required), 'set' (set equality)

    Returns:
        True if answer is correct
    """
    if not extracted or not standard:
        return False

    if match_mode == 'any':
        # Any match is acceptable
        return extracted in standard

    elif match_mode == 'all':
        # All standard answers must appear in extracted
        return all(ans in extracted for ans in standard)

    elif match_mode == 'set':
        # Exact set match (order doesn't matter)
        extracted_set = set(extracted.split(','))
        standard_set = set(standard)
        return extracted_set == standard_set

    return False


def grade_objective_questions(
    ocr_dir: Path,
    answer_key_path: Path,
    output_dir: Path
) -> Dict:
    """
    Grade objective questions automatically

    Args:
        ocr_dir: Directory with OCR results
        answer_key_path: Path to answer_key.json
        output_dir: Output directory for grading results

    Returns:
        Grading results dictionary
    """
    print(f"\n{'='*80}")
    print("Grading Objective Questions")
    print(f"{'='*80}")

    # Load answer key
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key_data = json.load(f)

    objective_questions = answer_key_data.get('objective_questions', {})

    if not objective_questions:
        print("  No objective questions found")
        return {'total_score': 0, 'questions': {}}

    # Load OCR results
    ocr_files = sorted(ocr_dir.glob("page_*_ocr.json"))
    all_regions = []

    for ocr_file in ocr_files:
        page_num = int(ocr_file.stem.split('_')[1])
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        for result in ocr_data.get('results', []):
            all_regions.append({
                'page': page_num,
                'type': result.get('type', 'unknown'),
                'content': result.get('content', '').strip()
            })

    # Find group questions
    group_questions = set()
    for q_id, q_info in objective_questions.items():
        if q_info.get('type') == 'group':
            group_questions.add(q_id)

    # Define instruction section blacklist
    INSTRUCTION_TITLES = [
        '考试说明', '注意事项', '考生注意', '答题要求',
        'instructions', 'directions', 'notice'
    ]

    # Extract question regions
    question_regions = []
    in_instruction_section = False
    current_group_question = None

    for region in all_regions:
        content = region['content'].strip()
        region_type = region.get('type', 'text')

        # Check section titles
        if region_type in ['paragraph_title', 'doc_title']:
            is_instruction = any(kw in content for kw in INSTRUCTION_TITLES)
            if is_instruction:
                in_instruction_section = True
            else:
                in_instruction_section = False
            continue

        if in_instruction_section:
            continue

        if region_type != 'text':
            continue

        # Try sub-question patterns
        sub_patterns = [
            (r'^(\d+)[）\)]', 'simple'),
            (r'^[（\(](\d+)[）\)]', 'parenthesis'),
            (r'^(\d+)[\.\s]*[（\(](\d+)[）\)]', 'compound'),
        ]

        matched = False
        for pattern, ptype in sub_patterns:
            match = re.match(pattern, content)
            if match:
                if ptype in ['simple', 'parenthesis']:
                    sub_num = match.group(1)
                    if current_group_question and current_group_question in group_questions:
                        q_num = f"{current_group_question}({sub_num})"
                    else:
                        matched = True
                        break
                else:  # compound
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
            # Main question pattern
            match = re.match(r'^(\d+)\.', content)
            if match:
                q_num = match.group(1)
                question_regions.append({
                    'question_num': q_num,
                    'page': region['page'],
                    'content': content
                })
                current_group_question = q_num
            else:
                if region_type in ['paragraph_title', 'doc_title']:
                    current_group_question = None

    # Flatten questions (expand groups)
    flat_questions = []
    for q_id in sorted(objective_questions.keys(), key=lambda x: int(x)):
        q_info = objective_questions[q_id]
        q_type = q_info.get('type', 'unknown')

        if q_type == 'group' and 'sub_questions' in q_info:
            for sub_id, sub_info in q_info['sub_questions'].items():
                flat_questions.append({
                    'full_id': f"{q_id}{sub_id}",
                    'parent_id': q_id,
                    'sub_id': sub_id,
                    'type': sub_info.get('type', 'unknown'),
                    'answer': sub_info.get('answer', []),
                    'score': sub_info.get('score', 0),
                    'match_mode': sub_info.get('match_mode', 'any')
                })
        else:
            flat_questions.append({
                'full_id': q_id,
                'parent_id': None,
                'sub_id': None,
                'type': q_type,
                'answer': q_info.get('answer', []),
                'score': q_info.get('score', 0),
                'match_mode': q_info.get('match_mode', 'any')
            })

    # Helper functions for table display
    def display_width(s):
        """Calculate display width accounting for wide characters (Chinese, etc.)"""
        width = 0
        for char in str(s):
            if '一' <= char <= '鿿' or '　' <= char <= '〿':
                width += 2
            else:
                width += 1
        return width

    def pad_string(s, target_width):
        """Pad string to target display width"""
        s = str(s)
        current_width = display_width(s)
        if current_width >= target_width:
            return s
        return s + ' ' * (target_width - current_width)

    # Print table header
    print(f"\n{'='*100}")
    print("Objective Question Grading Details")
    print(f"{'='*100}")
    header = f"{pad_string('Q#', 8)}{pad_string('Result', 10)}{pad_string('Mode', 12)}{pad_string('Answer Key', 35)}{pad_string('Student Answer', 35)}"
    print(header)
    print(f"{'-'*100}")

    # Grade each question
    grading_results = {
        'total_questions': len(flat_questions),
        'total_possible_score': sum(q['score'] for q in flat_questions),
        'total_score': 0,
        'questions': {}
    }

    for q in flat_questions:
        full_id = q['full_id']
        q_type = q['type']
        standard_answer = q['answer']
        max_score = q['score']
        match_mode = q['match_mode']

        # Find in OCR
        extracted_answer = None
        ocr_content = None
        page_num = None

        for q_region in question_regions:
            if q_region['question_num'] == full_id:
                ocr_content = q_region['content']
                page_num = q_region['page']
                extracted_answer = extract_answer(ocr_content, q_type)
                break

        # Check answer
        is_correct = False
        if extracted_answer:
            is_correct = check_answer(extracted_answer, standard_answer, match_mode)

        # Calculate score
        earned_score = max_score if is_correct else 0
        grading_results['total_score'] += earned_score

        # Print table row
        student_str = str(extracted_answer) if extracted_answer else 'None'
        if len(student_str) > 33:
            student_str = student_str[:30] + '...'

        if isinstance(standard_answer, list):
            if len(standard_answer) > 1:
                correct_str = f"{standard_answer[0]}..."
            else:
                correct_str = str(standard_answer[0]) if standard_answer else 'N/A'
        else:
            correct_str = str(standard_answer)

        if len(correct_str) > 33:
            correct_str = correct_str[:30] + '...'

        result_str = "[V]" if is_correct else "[X]"

        row = f"{pad_string(full_id, 8)}{pad_string(result_str, 10)}{pad_string(match_mode, 12)}{pad_string(correct_str, 35)}{pad_string(student_str, 35)}"
        print(row)

        # Record result
        grading_results['questions'][full_id] = {
            'type': q_type,
            'standard_answer': standard_answer,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'max_score': max_score,
            'earned_score': earned_score,
            'page': page_num,
            'ocr_content': ocr_content[:100] if ocr_content else None
        }

    # Print table footer
    print(f"{'-'*100}")
    print(f"Total Score: {grading_results['total_score']}/{grading_results['total_possible_score']}")
    print(f"{'='*100}\n")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_output = output_dir / 'objective_grading_results.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(grading_results, f, ensure_ascii=False, indent=2)

    # Text report
    txt_output = output_dir / 'objective_grading_report.txt'
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Objective Questions Grading Report\n")
        f.write("="*80 + "\n\n")

        for q_id in sorted(grading_results['questions'].keys(),
                          key=lambda x: (int(re.match(r'(\d+)', x).group(1)), x)):
            q_result = grading_results['questions'][q_id]

            f.write(f"Question {q_id} ({q_result['type']})\n")
            f.write("-"*80 + "\n")
            f.write(f"Standard Answer: {q_result['standard_answer']}\n")
            f.write(f"Student Answer:  {q_result['extracted_answer'] or 'NOT FOUND'}\n")
            f.write(f"Status: {'✓ CORRECT' if q_result['is_correct'] else '✗ INCORRECT'}\n")
            f.write(f"Score: {q_result['earned_score']}/{q_result['max_score']}\n")
            f.write("\n" + "="*80 + "\n\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Total Questions: {grading_results['total_questions']}\n")
        f.write(f"Total Score: {grading_results['total_score']}/{grading_results['total_possible_score']}\n")
        accuracy = (grading_results['total_score'] / grading_results['total_possible_score'] * 100) if grading_results['total_possible_score'] > 0 else 0
        f.write(f"Accuracy: {accuracy:.1f}%\n")

    print(f"\n  Graded {grading_results['total_questions']} questions")
    print(f"  Total Score: {grading_results['total_score']}/{grading_results['total_possible_score']}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"\n  Results saved:")
    print(f"    JSON: {json_output}")
    print(f"    Report: {txt_output}")

    return grading_results
