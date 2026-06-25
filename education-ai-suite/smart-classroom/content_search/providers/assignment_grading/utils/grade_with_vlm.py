import json
from pathlib import Path
import base64
import re


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_rubric(rubric_dir, question_id):
    rubric_file = rubric_dir / f"{question_id}.json"

    if not rubric_file.exists():
        print(f"  警告: 找不到rubric {rubric_file}")
        return None

    with open(rubric_file, 'r', encoding='utf-8') as f:
        rubric = json.load(f)

    return rubric


def construct_vlm_prompt(question_id, rubric, answer_image_path):
    if not rubric:
        return {
            'error': 'no_rubric',
            'prompt': None,
            'image': None
        }

    max_score = rubric.get('total_score', 0)

    prompt = f"批改中考试卷。\n\n题目：{rubric.get('question_text', '见图片')}\n"

    context = rubric.get('context', {})
    if context:
        prompt += "\n**原文片段：**\n"
        for key, value in context.items():
            prompt += f"{value}\n"

    prompt += f"\n**评分标准（总分{max_score}分）：**\n"

    scoring_points = rubric.get('scoring_points', rubric.get('criteria', []))
    for point in scoring_points:
        desc = point.get('description', point.get('criterion', ''))
        score = point.get('score', point.get('max_score', 0))
        criteria = point.get('criteria', '')
        if criteria:
            prompt += f"\n{desc}（{score}分）：{criteria}"
        else:
            prompt += f"\n{desc}：{score}分"

    standard_answer = rubric.get('standard_answer_summary', '')
    if standard_answer:
        prompt += f"\n\n**标准答案示例：**\n{standard_answer}"

    standard_answers = rubric.get('standard_answers', rubric.get('reference_answers', []))
    if standard_answers and not standard_answer:
        prompt += "\n\n**参考答案：**"
        if isinstance(standard_answers, dict):
            for key, ans in standard_answers.items():
                prompt += f"\n- {key}: {ans}"
        elif isinstance(standard_answers, list):
            for ans in standard_answers[:3]:
                prompt += f"\n- {ans}"

    few_shot = rubric.get('few_shot_examples', [])
    if few_shot:
        prompt += "\n\n**评分示例：**"
        for example in few_shot[:2]:
            ans = example.get('answer', '')
            score = example.get('score', 0)
            prompt += f"\n学生答：{ans[:50]}... → 得分：{score}分"

    prompt += f"\n\n查看图片中学生的手写答案并打分。\n\n**重要：无论你的分析过程多长，最后必须输出：**\n总分：X分\n\n（其中X是最终得分，满分{max_score}分）\n"

    image_base64 = encode_image_to_base64(answer_image_path)

    return {
        'prompt': prompt,
        'image_base64': image_base64,
        'image_path': str(answer_image_path),
        'max_score': rubric.get('total_score', 0)
    }


def call_vlm_api(vlm_input, model='local', api_url='http://127.0.0.1:9900', max_retries=2):
    print(f"    调用本地VLM: {api_url}")

    payload = {
        "model": "qwen-vl",
        "messages": [
            {
                "role": "system",
                "content": "你是一位严格的阅卷教师。必须严格按照评分标准逐项检查。不要根据最终答案猜测过程，不要补全学生省略的步骤。直接输出评分结果，不要输出思考过程。"
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
        "temperature": 0.0
    }

    import requests
    for attempt in range(max_retries):
        try:
            print(f"    尝试 {attempt + 1}/{max_retries}...")
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                json=payload,
                timeout=300
            )

            if response.status_code != 200:
                print(f"     错误响应 ({response.status_code}): {response.text[:500]}")
            response.raise_for_status()
            data = response.json()

            vlm_output = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            print(f"\n    【VLM原始输出】")
            print(f"    {'-'*60}")
            print(f"    {vlm_output[:500]}..." if len(vlm_output) > 500 else vlm_output)
            print(f"    {'-'*60}\n")

            final_score_patterns = [
                (r'总分[：:=\s]*(\d+)\s*分', '总分'),
                (r'最终答案应该是\s*(\d+)\s*分', '最终答案'),
                (r'得分[：:]\s*(\d+)\s*/\s*(\d+)', '得分X/Y格式'),
                (r'(?:最终|综合)得分[：:]\s*(\d+)\s*分', '最终得分'),
                (r'(?:可以|应该)?得\s*(\d+)\s*分', '得X分'),
            ]

            total_score = None
            max_score = vlm_input.get('max_score', 0)
            reason = "见原始输出"

            for pattern, desc in final_score_patterns:
                if desc == '得分X/Y格式':
                    last_1000 = vlm_output[-1000:] if len(vlm_output) > 1000 else vlm_output
                    match = re.search(pattern, last_1000)
                    if match:
                        total_score = int(match.group(1))
                        max_score = int(match.group(2))
                        reason_match = re.search(r'理由[：:]\s*(.+?)(?:\n\n|$)', last_1000[match.end():], re.DOTALL)
                        reason = reason_match.group(1).strip()[:200] if reason_match else "见原始输出"
                        print(f"     匹配到格式: {desc}")
                        break
                else:
                    matches = list(re.finditer(pattern, vlm_output))
                    if matches:
                        last_match = matches[-1]
                        total_score = int(last_match.group(1))
                        print(f"     匹配到格式: {desc} (共{len(matches)}处，取最后)")
                        reason = f"从VLM输出提取：{desc}={total_score}分"
                        break

            if total_score is None:
                print(f"     未找到得分信息，默认给0分")
                result = {
                    'total_score': 0,
                    'max_score': vlm_input.get('max_score', 0),
                    'comment': '解析失败',
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
            print(f"     解析成功：{total_score}/{max_score}分")

            return result

        except (requests.exceptions.Timeout, TimeoutError) as e:
            print(f"\n     超时 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    'total_score': 0,
                    'max_score': vlm_input.get('max_score', 0),
                    'comment': f'VLM调用超时（已重试{max_retries}次）',
                    'error': str(e),
                    'model': model,
                    'raw_output': ''
                }
            print(f"    等待10秒后重试...")
            import time
            time.sleep(10)

        except Exception as e:
            print(f"\n     VLM调用异常: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total_score': 0,
                'max_score': vlm_input.get('max_score', 0),
                'comment': f'VLM调用错误: {str(e)}',
                'error': str(e),
                'model': model,
                'raw_output': ''
            }

    return {
        'total_score': 0,
        'max_score': vlm_input.get('max_score', 0),
        'comment': '所有重试均失败',
        'error': 'All retries failed',
        'model': model,
        'raw_output': ''
    }


def grade_with_vlm(processed_json, rubric_dir, output_json, vlm_model='local', api_url='http://127.0.0.1:9900'):
    print(f"\n{'='*80}")
    print("VLM自动评分")
    print(f"{'='*80}")
    print(f"VLM API: {api_url}")

    print(f"\n[1/5] 加载试卷元数据...")
    exam_meta_path = Path(rubric_dir) / "exam_meta.json"
    objective_questions = []
    if exam_meta_path.exists():
        with open(exam_meta_path, 'r', encoding='utf-8') as f:
            exam_meta = json.load(f)
            objective_questions = exam_meta.get('objective_questions', [])
        print(f"  客观题: {len(objective_questions)}题")
        print(f"  客观题列表: {objective_questions[:10]}{'...' if len(objective_questions) > 10 else ''}")
    else:
        print(f"  未找到exam_meta.json，将根据rubric的type字段判断")

    print(f"\n[2/5] 加载学生答案...")
    with open(processed_json, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    student_id = processed_data.get('student_id', 'unknown')
    answer_blocks = processed_data.get('answer_blocks', [])

    print(f"  学生ID: {student_id}")
    print(f"  答题数: {len(answer_blocks)}")

    print(f"\n[3/5] 加载评分标准...")
    print(f"  Rubric目录: {rubric_dir}")

    print(f"\n[4/5] VLM评分...\n")

    grading_results = []

    for block in answer_blocks:
        question_id = block['question_id']
        image_path = Path(block['image_path'])

        print(f"  评分 {question_id}...")

        if not image_path.exists():
            print(f"    跳过（图片不存在）")
            continue

        is_objective = question_id in objective_questions

        if is_objective:
            print(f"    跳过（客观题，根据exam_meta.json）")
            grading_result = {
                'question_id': question_id,
                'page': block['page'],
                'image_path': str(image_path),
                'rubric': None,
                'vlm_score': None,
                'max_score': 0,
                'skipped': True,
                'reason': '客观题，使用OCR+规则匹配评分'
            }
            grading_results.append(grading_result)
            continue

        rubric = load_rubric(Path(rubric_dir), question_id)

        if not rubric:
            print(f"    跳过（无rubric）")
            continue

        vlm_input = construct_vlm_prompt(question_id, rubric, image_path)

        if vlm_input.get('error'):
            print(f"    跳过（{vlm_input['error']}）")
            continue

        print(f"\n    【输入图片】{image_path}")
        print(f"\n    【评分Prompt】")
        print(f"    {'-'*60}")
        print(f"    {vlm_input['prompt'][:500]}...")
        print(f"    {'-'*60}")
        print(f"    (总长度: {len(vlm_input['prompt'])}字符)\n")

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

        print(f"\n    【评分结果】")
        print(f"    得分: {vlm_result.get('total_score', 0)}/{vlm_result.get('max_score', 0)}")
        if vlm_result.get('comment'):
            print(f"    评语: {vlm_result.get('comment', '')[:100]}")
        print()

        vlm_detail_dir = Path(output_json).parent / "vlm_details"
        vlm_detail_dir.mkdir(parents=True, exist_ok=True)

        detail_file = vlm_detail_dir / f"{student_id}_{question_id}_vlm_output.txt"
        with open(detail_file, 'w', encoding='utf-8') as f:
            f.write(f"题号: {question_id}\n")
            f.write(f"题目: {rubric.get('question_text', '')}\n")
            f.write(f"满分: {vlm_result.get('max_score', 0)}分\n")
            f.write(f"得分: {vlm_result.get('total_score', 0)}分\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"VLM完整输出:\n")
            f.write(f"{'='*80}\n\n")
            f.write(vlm_result.get('raw_output', ''))
            f.write(f"\n\n{'='*80}\n")
            f.write(f"提取的评语:\n")
            f.write(f"{'='*80}\n\n")
            f.write(vlm_result.get('comment', ''))

    print(f"\n[5/5] 保存评分结果...")

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
        f.write(f"# {student_id} - 评分详情\n\n")
        f.write(f"**总分：{total_score}/{max_total}**\n\n")
        f.write(f"**评分模型：** {vlm_model}\n\n")
        f.write(f"---\n\n")

        for result in grading_results:
            q_id = result['question_id']
            if result.get('skipped'):
                f.write(f"## {q_id} - 客观题（跳过VLM）\n\n")
                f.write(f"**原因：** {result.get('reason', '')}\n\n")
                f.write(f"---\n\n")
                continue

            f.write(f"## {q_id}\n\n")
            f.write(f"**题目：** {result['rubric'].get('question_text', '')}\n\n")
            f.write(f"**得分：** {result.get('vlm_score', 0)}/{result.get('max_score', 0)}\n\n")
            f.write(f"**评语：** {result.get('comment', '')}\n\n")
            f.write(f"**学生答案图片：** [{q_id}.jpg]({Path(result['image_path']).name})\n\n")
            f.write(f"**详细输出：** [{q_id}_vlm_output.txt]({student_id}_{q_id}_vlm_output.txt)\n\n")
            f.write(f"---\n\n")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print("评分完成")
    print(f"{'='*80}")
    print(f"主观题总分: {total_score}/{max_total}")
    print(f"客观题跳过: {skipped_count}题")
    print(f"输出JSON: {output_json}")
    print(f"详细报告: {summary_file}")
