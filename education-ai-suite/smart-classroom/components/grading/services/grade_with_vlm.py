import json
from pathlib import Path
import base64
import re
from functools import lru_cache


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_rubric(rubric_dir, question_id):
    rubric_file = rubric_dir / f"{question_id}.json"

    if not rubric_file.exists():
        print(f"  Warning: Rubric not found {rubric_file}")
        return None

    with open(rubric_file, 'r', encoding='utf-8') as f:
        rubric = json.load(f)

    return rubric


_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _normalize_language(language: str) -> str:
    lang = str(language or 'en').strip().lower()
    return 'cn' if lang in ('cn', 'zh', 'zh-cn', 'chinese') else 'en'


@lru_cache(maxsize=None)
def _load_prompt_template(lang: str) -> dict:
    # All prompt wording (including any non-English text) lives in these
    # template files, not in code. Add a new templates/vlm_grading_prompt.<lang>.json
    # to support another language.
    template_path = _TEMPLATES_DIR / f"vlm_grading_prompt.{lang}.json"
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def construct_vlm_prompt(question_id, rubric, answer_image_path, language='en', subject=None):
    if not rubric:
        return {
            'error': 'no_rubric',
            'prompt': None,
            'image': None
        }

    lang = _normalize_language(language)
    text = _load_prompt_template(lang)
    if not subject:
        subject = text.get('default_subject', 'subject')

    max_score = rubric.get('total_score', 0)
    scoring_points = rubric.get('scoring_points', rubric.get('criteria', []))
    scoring_rules = rubric.get('scoring_rules', [])

    # 1. Scenario setup
    prompt = text['scenario'].format(subject=subject, max_score=max_score) + "\n\n"
    prompt += f"{text['question_label']}: {rubric.get('question_text', 'See image')}\n"

    context = rubric.get('context', {})
    if context:
        prompt += f"\n**{text['context_label']}:**\n"
        for value in context.values():
            prompt += f"{value}\n"

    # 2. Reference answer
    standard_answer = rubric.get('standard_answer_summary', '')
    standard_answers = rubric.get('standard_answers', rubric.get('reference_answers', []))
    if standard_answer:
        prompt += f"\n**{text['reference_answer_label']}:**\n{standard_answer}\n"
    elif standard_answers:
        prompt += f"\n**{text['reference_answer_label']}:**\n"
        if isinstance(standard_answers, dict):
            for key, ans in standard_answers.items():
                prompt += f"- {key}: {ans}\n"
        elif isinstance(standard_answers, list):
            for ans in standard_answers[:3]:
                prompt += f"- {ans}\n"

    # 3. Scoring standard (points = additive dimensions, rules = exclusive conditions)
    if scoring_points:
        prompt += f"\n**{text['scoring_points_label'].format(max_score=max_score)}**\n"
        for point in scoring_points:
            desc = point.get('description', point.get('criterion', ''))
            score = point.get('score', point.get('max_score', 0))
            criteria = point.get('criteria', '')
            if criteria:
                prompt += text['scoring_point_line_with_criteria'].format(desc=desc, score=score, criteria=criteria) + "\n"
            else:
                prompt += text['scoring_point_line'].format(desc=desc, score=score) + "\n"

    if scoring_rules:
        prompt += f"\n**{text['scoring_rules_label'].format(max_score=max_score)}**\n"
        for rule in scoring_rules:
            prompt += text['scoring_rule_line'].format(rule=rule) + "\n"

    few_shot = rubric.get('few_shot_examples', [])
    if few_shot:
        prompt += f"\n**{text['examples_label']}:**\n"
        for example in few_shot[:2]:
            ans = example.get('answer', '')
            score = example.get('score', 0)
            prompt += text['example_line'].format(answer=ans[:50], score=score) + "\n"

    # 4. Expected response format (kept parseable regardless of language)
    prompt += f"\n{text['review_instruction']}\n\n{text['output_format_label']}\n"
    if scoring_points:
        prompt += text['output_points_instruction'] + "\n"
    if scoring_rules:
        prompt += text['output_rules_instruction'] + "\n"
    prompt += text['output_total_instruction'].format(max_score=max_score) + "\n"

    image_base64 = encode_image_to_base64(answer_image_path)

    return {
        'prompt': prompt,
        'image_base64': image_base64,
        'image_path': str(answer_image_path),
        'max_score': rubric.get('total_score', 0)
    }


NO_RUBRIC_MAX_SCORE = 10


def construct_vlm_prompt_no_rubric(question_id, answer_image_path, language='en', subject=None, max_score=NO_RUBRIC_MAX_SCORE):
    """Build a VLM grading prompt when the question has no rubric.

    The question is assumed to be worth ``max_score`` points (default 10). The
    caller can rescale the returned score with actual_score * (vlm_score / max_score).
    """
    lang = _normalize_language(language)
    text = _load_prompt_template(lang)
    if not subject:
        subject = text.get('default_subject', 'subject')

    prompt = text['no_rubric_scenario'].format(subject=subject, max_score=max_score) + "\n\n"
    prompt += f"{text['no_rubric_review_instruction'].format(max_score=max_score)}\n\n"
    prompt += f"{text['output_format_label']}\n"
    prompt += text['no_rubric_output_instruction'].format(max_score=max_score) + "\n"

    image_base64 = encode_image_to_base64(answer_image_path)

    return {
        'prompt': prompt,
        'image_base64': image_base64,
        'image_path': str(answer_image_path),
        'max_score': max_score
    }


def call_vlm_api(vlm_input, model='local', api_url='http://127.0.0.1:9900', max_retries=2):
    import time

    print(f"    Calling local VLM: {api_url}")

    payload = {
        "model": "qwen-vl",
        "messages": [
            {
                "role": "system",
                "content": "You are a strict exam grader. Read the student's handwritten answer in the image, check each rubric criterion, and award points. Do not guess steps the student omitted. Be concise: one short line per criterion (points earned + brief reason), no long analysis, no re-derivation. Always end with exactly one line 'Total Score: X points' where X is the final score."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{vlm_input['image_base64']}"
                        }
                    },
                    {
                        "type": "text",
                        "text": vlm_input['prompt']
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1
    }

    import requests
    for attempt in range(max_retries):
        try:
            print(f"    Attempt {attempt + 1}/{max_retries}...")

            start_time = time.time()

            response = requests.post(
                f"{api_url}/v1/chat/completions",
                json=payload,
                timeout=300
            )

            elapsed_time = time.time() - start_time

            if response.status_code != 200:
                print(f"     Error response ({response.status_code}): {response.text[:500]}")
            response.raise_for_status()
            data = response.json()

            vlm_output = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            if len(vlm_output) < 10 and attempt < max_retries - 1:
                print(f"\n     WARNING: Output too short ({len(vlm_output)} chars), retrying with stronger prompt...")
                continue

            print(f"\n    [VLM Raw Output]")
            print(f"    {'-'*60}")
            print(f"    {vlm_output[:500]}..." if len(vlm_output) > 500 else vlm_output)
            print(f"    {'-'*60}")
            print(f"    Inference time: {elapsed_time:.2f}s")
            print(f"    Output length: {len(vlm_output)} chars\n")

            final_score_patterns = [
                ('Total Score[:\uFF1A=\\s\\*]*(\\d+)\\s*points?', 'Total Score'),
                (r'Final score should be\s*(\d+)\s*points?', 'Final score'),
                ('Score[:\uFF1A]\\s*(\\d+)\\s*/\\s*(\\d+)', 'Score X/Y format'),
                ('(?:Final|Overall) score[:\uFF1A]\\s*(\\d+)\\s*points?', 'Final score'),
                (r'(?:can|should)?\s*(?:get|score)\s*(\d+)\s*points?', 'Get X points'),
            ]

            total_score = None
            max_score = vlm_input.get('max_score', 0)
            reason = "See raw output"

            for pattern, desc in final_score_patterns:
                if desc == 'Score X/Y format':
                    last_1000 = vlm_output[-1000:] if len(vlm_output) > 1000 else vlm_output
                    match = re.search(pattern, last_1000)
                    if match:
                        total_score = int(match.group(1))
                        max_score = int(match.group(2))
                        reason_match = re.search('Reason[:\uFF1A]\\s*(.+?)(?:\\n\\n|$)', last_1000[match.end():], re.DOTALL)
                        reason = reason_match.group(1).strip()[:200] if reason_match else "See raw output"
                        print(f"     Matched format: {desc}")
                        break
                else:
                    matches = list(re.finditer(pattern, vlm_output))
                    if matches:
                        last_match = matches[-1]
                        total_score = int(last_match.group(1))
                        print(f"     Matched format: {desc} (found {len(matches)}, using last)")
                        reason = f"Extracted from VLM output: {desc}={total_score} points"
                        break

            if total_score is None:
                print(f"     Score info not found, defaulting to 0")
                result = {
                    'total_score': 0,
                    'max_score': vlm_input.get('max_score', 0),
                    'comment': 'Parse failed',
                    'parse_error': True,
                    'raw_output': vlm_output
                }
                return result

            result = {
                'total_score': total_score,
                'max_score': max_score,
                'comment': reason,
                'raw_output': vlm_output
            }
            print(f"     Parse successful: {total_score}/{max_score} points")

            return result

        except (requests.exceptions.Timeout, TimeoutError) as e:
            print(f"\n     Timeout (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    'total_score': 0,
                    'max_score': vlm_input.get('max_score', 0),
                    'comment': f'VLM call timeout (retried {max_retries} times)',
                    'error': str(e),
                    'model': model,
                    'raw_output': ''
                }
            print(f"    Waiting 10 seconds before retry...")
            import time
            time.sleep(10)

        except Exception as e:
            print(f"\n     VLM call exception: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total_score': 0,
                'max_score': vlm_input.get('max_score', 0),
                'comment': f'VLM call error: {str(e)}',
                'error': str(e),
                'model': model,
                'raw_output': ''
            }

    return {
        'total_score': 0,
        'max_score': vlm_input.get('max_score', 0),
        'comment': 'All retries failed',
        'error': 'All retries failed',
        'model': model,
        'raw_output': ''
    }


def grade_with_vlm(processed_json, rubric_dir, output_json, vlm_model='local', api_url='http://127.0.0.1:9900'):
    import time

    print(f"\n{'='*80}")
    print("VLM Automatic Grading")
    print(f"{'='*80}")
    print(f"VLM API: {api_url}")

    total_start_time = time.time()

    print(f"\n[1/5] Loading exam metadata...")
    exam_meta_path = Path(rubric_dir) / "exam_meta.json"
    objective_questions = []
    if exam_meta_path.exists():
        with open(exam_meta_path, 'r', encoding='utf-8') as f:
            exam_meta = json.load(f)
            objective_questions = exam_meta.get('objective_questions', [])
        print(f"  Objective questions: {len(objective_questions)}")
        print(f"  Objective list: {objective_questions[:10]}{'...' if len(objective_questions) > 10 else ''}")
    else:
        print(f"  exam_meta.json not found, will determine by rubric type field")

    print(f"\n[2/5] Loading student answers...")
    with open(processed_json, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    student_id = processed_data.get('student_id', 'unknown')
    answer_blocks = processed_data.get('answer_blocks', [])

    print(f"  Student ID: {student_id}")
    print(f"  Answer count: {len(answer_blocks)}")

    print(f"\n[3/5] Loading grading rubric...")
    print(f"  Rubric dir: {rubric_dir}")

    print(f"\n[4/5] VLM grading...\n")

    grading_results = []

    for block in answer_blocks:
        question_id = block['question_id']
        image_path = Path(block['image_path'])

        print(f"  Grading {question_id}...")

        if not image_path.exists():
            print(f"    Skipped (image not found)")
            continue

        is_objective = question_id in objective_questions

        if is_objective:
            print(f"    Skipped (objective question, per exam_meta.json)")
            grading_result = {
                'question_id': question_id,
                'page': block['page'],
                'image_path': str(image_path),
                'rubric': None,
                'vlm_score': None,
                'max_score': 0,
                'skipped': True,
                'reason': 'Objective question, use OCR + rule-based grading'
            }
            grading_results.append(grading_result)
            continue

        rubric = load_rubric(Path(rubric_dir), question_id)

        if not rubric:
            print(f"    Skipped (no rubric)")
            continue

        vlm_input = construct_vlm_prompt(question_id, rubric, image_path)

        if vlm_input.get('error'):
            print(f"    Skipped ({vlm_input['error']})")
            continue

        print(f"\n    [Input Image] {image_path}")
        print(f"\n    [Grading Prompt]")
        print(f"    {'-'*60}")
        print(f"    {vlm_input['prompt'][:500]}...")
        print(f"    {'-'*60}")
        print(f"    (Total length: {len(vlm_input['prompt'])} chars)\n")

        vlm_result = call_vlm_api(vlm_input, model=vlm_model, api_url=api_url)

        grading_result = {
            'question_id': question_id,
            'page': block['page'],
            'image_path': str(image_path),
            'rubric': rubric,
            'vlm_score': vlm_result.get('total_score', 0),
            'max_score': vlm_result.get('max_score', 0),
            'comment': vlm_result.get('comment', ''),
            'raw_output': vlm_result.get('raw_output', ''),
            'model': vlm_result.get('model', vlm_model)
        }

        grading_results.append(grading_result)

        print(f"\n    [Grading Result]")
        print(f"    Score: {vlm_result.get('total_score', 0)}/{vlm_result.get('max_score', 0)}")
        if vlm_result.get('comment'):
            print(f"    Comment: {vlm_result.get('comment', '')[:100]}")
        print()

        vlm_detail_dir = Path(output_json).parent / "vlm_details"
        vlm_detail_dir.mkdir(parents=True, exist_ok=True)

        detail_file = vlm_detail_dir / f"{student_id}_{question_id}_vlm_output.txt"
        with open(detail_file, 'w', encoding='utf-8') as f:
            f.write(f"Question ID: {question_id}\n")
            f.write(f"Question: {rubric.get('question_text', '')}\n")
            f.write(f"Max score: {vlm_result.get('max_score', 0)} points\n")
            f.write(f"Score: {vlm_result.get('total_score', 0)} points\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"VLM Full Output:\n")
            f.write(f"{'='*80}\n\n")
            f.write(vlm_result.get('raw_output', ''))
            f.write(f"\n\n{'='*80}\n")
            f.write(f"Extracted Comment:\n")
            f.write(f"{'='*80}\n\n")
            f.write(vlm_result.get('comment', ''))

    print(f"\n[5/5] Saving grading results...")

    total_score = sum(r.get('vlm_score', 0) for r in grading_results if r.get('vlm_score') is not None)
    max_total = sum(r['max_score'] for r in grading_results if r.get('vlm_score') is not None)

    skipped_count = sum(1 for r in grading_results if r.get('skipped'))

    output_data = {
        'student_id': student_id,
        'total_score': total_score,
        'max_total_score': max_total,
        'grading_results': grading_results,
        'vlm_model': vlm_model
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary_file = Path(output_json).parent / "vlm_details" / f"{student_id}_grading_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# {student_id} - Grading Details\n\n")
        f.write(f"**Total Score: {total_score}/{max_total}**\n\n")
        f.write(f"**Grading Model:** {vlm_model}\n\n")
        f.write(f"---\n\n")

        for result in grading_results:
            q_id = result['question_id']
            if result.get('skipped'):
                f.write(f"## {q_id} - Objective Question (Skipped VLM)\n\n")
                f.write(f"**Reason:** {result.get('reason', '')}\n\n")
                f.write(f"---\n\n")
                continue

            f.write(f"## {q_id}\n\n")
            f.write(f"**Question:** {result['rubric'].get('question_text', '')}\n\n")
            f.write(f"**Score:** {result.get('vlm_score', 0)}/{result.get('max_score', 0)}\n\n")
            f.write(f"**Comment:** {result.get('comment', '')}\n\n")
            f.write(f"**Student Answer Image:** [{q_id}.jpg]({Path(result['image_path']).name})\n\n")
            f.write(f"**Detailed Output:** [{q_id}_vlm_output.txt]({student_id}_{q_id}_vlm_output.txt)\n\n")
            f.write(f"---\n\n")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_start_time
    graded_count = sum(1 for r in grading_results if r.get('vlm_score') is not None)
    avg_time_per_question = total_elapsed / graded_count if graded_count > 0 else 0

    print(f"\n{'='*80}")
    print("Grading completed")
    print(f"{'='*80}")
    print(f"Subjective total score: {total_score}/{max_total}")
    print(f"Objective questions skipped: {skipped_count}")
    print(f"Subjective questions graded: {graded_count}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Average time per question: {avg_time_per_question:.1f}s")
    print(f"Output JSON: {output_json}")
    print(f"Detailed report: {summary_file}")

