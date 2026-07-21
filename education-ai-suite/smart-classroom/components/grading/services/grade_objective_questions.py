"""Grade Objective Questions - Automatically grades objective questions based on extracted answers"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional
from services.text_normalizer import normalize_text, normalize_for_match


def _extract_underline(content: str) -> Optional[str]:
    """Extract the content of the first \\underline{...} with balanced braces.

    Handles nested braces (e.g. \\underline{\\frac{1}{8}}). Returns None if
    there is no underline or the braces are unbalanced.
    """
    idx = content.find(r'\underline')
    if idx < 0:
        return None
    brace = content.find('{', idx)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(content)):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                return content[brace + 1:i]
    return None


def _match_by_containment(content: str, standards: List[str]) -> Optional[str]:
    """Return the first standard answer that appears in the question text.

    Both sides are normalized for matching. No length/numeric guard: this is
    the first-version behavior (highest recall, accepts false positives for
    short numeric answers that also occur in the question stem).
    """
    norm_content = normalize_for_match(content)
    for raw in standards:
        norm_ans = normalize_for_match(raw)
        if norm_ans and norm_ans in norm_content:
            return raw
    return None


def extract_answer(content: str, q_type: str, standards: Optional[List[str]] = None) -> Optional[str]:
    """Extract the student's answer from a question's OCR text.

    choice: match a bracketed option letter (A-D), NFKC handles full-width.
    blank:  prefer the \\underline{...} span (balanced braces); otherwise fall
            back to checking whether a standard answer is contained in the text.
    """
    content = normalize_text(content)

    if q_type == 'choice':
        match = re.search(r'\(\s*([A-D])\s*\)', content)
        if match:
            return match.group(1)
        return None

    if q_type == 'blank':
        underline = _extract_underline(content)
        if underline is not None:
            return underline.strip()
        if standards:
            return _match_by_containment(content, standards)
        return None

    return None


def check_answer(extracted: str, standard: List[str], match_mode: str = 'any') -> bool:
    """Check if extracted answer matches standard answer (normalized compare)."""
    if not extracted or not standard:
        return False

    norm_extracted = normalize_for_match(extracted)
    norm_standard = [normalize_for_match(s) for s in standard]

    if match_mode == 'any':
        return norm_extracted in norm_standard

    elif match_mode == 'all':
        return all(ans in norm_extracted for ans in norm_standard)

    elif match_mode == 'set':
        extracted_set = set(normalize_for_match(x) for x in extracted.split(','))
        standard_set = set(norm_standard)
        return extracted_set == standard_set

    return False


def _save_objective_debug_images(
    question_mapping_path: Optional[Path],
    page_images: Optional[Dict],
    output_dir: Path,
) -> None:
    """Debug-only: crop and save each objective question's region image."""
    if not question_mapping_path or not page_images:
        return
    if not Path(question_mapping_path).exists():
        return

    from PIL import Image

    with open(question_mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    debug_dir = output_dir / "objective_questions"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for q_id, q_data in mapping_data.get('questions', {}).items():
        if q_data.get('type') != 'objective':
            continue

        for idx, region in enumerate(q_data.get('regions', [])):
            page = region.get('page')
            bbox = region.get('bbox')
            if page is None or not bbox or page not in page_images:
                continue

            image = page_images[page]
            pil_img = Image.fromarray(image) if hasattr(image, 'shape') else image
            x1, y1, x2, y2 = bbox
            cropped = pil_img.crop((x1, y1, x2, y2))
            suffix = f"_{idx}" if len(q_data.get('regions', [])) > 1 else ""
            cropped.save(debug_dir / f"question_{q_id}{suffix}.jpg", quality=95)


def grade_objective_questions(
    ocr_dir: Path,
    answer_key_path: Path,
    output_dir: Path,
    config: Dict = None,
    debug_mode: bool = False,
    question_mapping_path: Optional[Path] = None,
    page_images: Optional[Dict] = None,
) -> Dict:
    """Grade objective questions automatically"""
    print(f"\n{'='*80}")
    print("Grading Objective Questions")
    print(f"{'='*80}")

    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key_data = json.load(f)

    objective_questions = answer_key_data.get('objective_questions', {})

    if not objective_questions:
        print("  No objective questions found")
        return {'total_score': 0, 'questions': {}}

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

    group_questions = set()
    for q_id, q_info in objective_questions.items():
        if q_info.get('type') == 'group':
            group_questions.add(q_id)

    INSTRUCTION_TITLES = config.get('question_parsing', {}).get('instruction_section_keywords', [])

    question_regions = []
    in_instruction_section = False
    current_group_question = None

    for region in all_regions:
        content = region['content'].strip()
        region_type = region.get('type', 'text')

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

        sub_patterns = [
            (r'^(\d+)[）)]', 'simple'),
            (r'^[（(](\d+)[）)]', 'parenthesis'),
            (r'^(\d+)[\.\s]*[（(](\d+)[）)]', 'compound'),
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
                else:
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

    def display_width(s):
        """Calculate display width accounting for wide characters"""
        width = 0
        for char in str(s):
            if unicodedata.east_asian_width(char) in ('W', 'F'):
                width += 2
            else:
                width += 1
        return width

    def pad_string(s, target_width):
        s = str(s)
        current_width = display_width(s)
        if current_width >= target_width:
            return s
        return s + ' ' * (target_width - current_width)

    print(f"\n{'='*100}")
    print("Objective Question Grading Details")
    print(f"{'='*100}")
    header = f"{pad_string('Q#', 8)}{pad_string('Result', 10)}{pad_string('Mode', 12)}{pad_string('Answer Key', 35)}{pad_string('Student Answer', 35)}"
    print(header)
    print(f"{'-'*100}")

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

        extracted_answer = None
        ocr_content = None
        page_num = None

        for q_region in question_regions:
            if q_region['question_num'] == full_id:
                ocr_content = q_region['content']
                page_num = q_region['page']
                extracted_answer = extract_answer(ocr_content, q_type, standard_answer)
                break

        is_correct = False
        if extracted_answer:
            is_correct = check_answer(extracted_answer, standard_answer, match_mode)

        earned_score = max_score if is_correct else 0
        grading_results['total_score'] += earned_score

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

        grading_results['questions'][full_id] = {
            'type': q_type,
            'standard_answer': standard_answer,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'max_score': max_score,
            'earned_score': earned_score,
            'page': page_num
        }

    print(f"{'-'*100}")
    print(f"Total Score: {grading_results['total_score']}/{grading_results['total_possible_score']}")
    print(f"{'='*100}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    if debug_mode:
        _save_objective_debug_images(question_mapping_path, page_images, output_dir)

    json_output = output_dir / 'objective_grading_results.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(grading_results, f, ensure_ascii=False, indent=2)

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
            f.write(f"Status: {'CORRECT' if q_result['is_correct'] else 'INCORRECT'}\n")
            f.write(f"Score: {q_result['earned_score']}/{q_result['max_score']}\n")
            f.write("\n" + "="*80 + "\n\n")

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

