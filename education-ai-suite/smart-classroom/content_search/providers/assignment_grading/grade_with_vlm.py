import json
from pathlib import Path
import base64

BASE_DIR = Path(__file__).parent


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_rubric(rubric_dir, question_id):
    """
    加载评分标准
    假设rubric文件命名为: Q1.json, Q2.json 等
    """
    rubric_file = rubric_dir / f"{question_id}.json"

    if not rubric_file.exists():
        print(f"  警告: 找不到rubric {rubric_file}")
        return None

    with open(rubric_file, 'r', encoding='utf-8') as f:
        rubric = json.load(f)

    return rubric


def construct_vlm_prompt(question_id, rubric, answer_image_path):
    """
    构造VLM评分的prompt
    """
    if not rubric:
        return {
            'error': 'no_rubric',
            'prompt': None,
            'image': None
        }

    max_score = rubric.get('total_score', 0)

    prompt = f"""批改中考试卷。

题目：{rubric.get('question_text', '见图片')}
"""

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
        for ans in standard_answers[:3]:
            prompt += f"\n- {ans}"

    few_shot = rubric.get('few_shot_examples', [])
    if few_shot:
        prompt += "\n\n**评分示例：**"
        for example in few_shot[:2]:
            ans = example.get('answer', '')
            score = example.get('score', 0)
            prompt += f"\n学生答：{ans[:50]}... → 得分：{score}分"

    prompt += f"""

查看图片中学生的手写答案并打分。

直接输出（不要任何其他内容）：
得分：X/{max_score}
理由：简要说明
"""

    image_base64 = encode_image_to_base64(answer_image_path)

    return {
        'prompt': prompt,
        'image_base64': image_base64,
        'image_path': str(answer_image_path),
        'max_score': rubric.get('total_score', 0)
    }


def call_vlm_api(vlm_input, model='local', api_url='http://127.0.0.1:9900'):
    """
    调用本地VLM API进行评分
    """
    import requests

    if model == 'mock':
        result = {
            'total_score': 0,
            'max_score': 0,
            'breakdown': [],
            'comment': '(Mock模式，示例输出)',
            'model': model,
            'mock': True
        }
        return result

    print(f"    调用本地VLM: {api_url}")

    payload = {
        "model": "qwen-vl",
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的语文教师。请直接输出评分结果，不要输出思考过程。严格按照指定格式输出。"
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
        "max_tokens": 3000,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        vlm_output = data.get('choices', [{}])[0].get('message', {}).get('content', '')

        print(f"\n    【VLM原始输出】")
        print(f"    {'-'*60}")
        print(f"    {vlm_output[:500]}..." if len(vlm_output) > 500 else vlm_output)
        print(f"    {'-'*60}\n")

        import re

        last_500_chars = vlm_output[-500:] if len(vlm_output) > 500 else vlm_output

        score_match = re.search(r'得分[：:]\s*(\d+)\s*/\s*(\d+)', last_500_chars)
        reason_match = re.search(r'理由[：:]\s*(.+?)(?:\n\n|$)', last_500_chars, re.DOTALL)

        if score_match:
            total_score = int(score_match.group(1))
            max_score = int(score_match.group(2))
            reason = reason_match.group(1).strip()[:200] if reason_match else "见原始输出"
        else:
            final_score_patterns = [
                r'(?:最终|总)?得分[：:]\s*(\d+)\s*分',
                r'应?得[：:]\s*(\d+)\s*分',
                r'给[：:]\s*(\d+)\s*分',
                r'评为[：:]\s*(\d+)\s*分',
                r'(?:可以|应该)?得\s*(\d+)\s*分'
            ]

            total_score = None
            for pattern in final_score_patterns:
                matches = re.findall(pattern, vlm_output)
                if matches:
                    total_score = int(matches[-1])
                    break

            if total_score is None:
                print(f"    ⚠️ 未找到得分信息，默认给0分")
                result = {
                    'total_score': 0,
                    'max_score': vlm_input.get('max_score', 0),
                    'comment': '解析失败',
                    'parse_error': True,
                    'raw_output': vlm_output
                }
                return result

            max_score = vlm_input.get('max_score', 0)
            reason = f"从VLM输出提取：最终得{total_score}分"

        result = {
            'total_score': total_score,
            'max_score': max_score,
            'comment': reason,
            'raw_output': vlm_output
        }
        print(f"    ✅ 解析成功：{total_score}/{max_score}分")

        result['model'] = model
        result['api_url'] = api_url

        return result

    except Exception as e:
        print(f"\n    ❌ VLM调用异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            'total_score': 0,
            'max_score': 0,
            'breakdown': [],
            'comment': f'VLM调用错误: {str(e)}',
            'error': str(e),
            'model': model
        }


def grade_with_vlm(processed_json, rubric_dir, output_json, vlm_model='local', api_url='http://127.0.0.1:9900'):
    """
    使用VLM评分
    """
    print(f"\n{'='*80}")
    print("VLM自动评分")
    print(f"{'='*80}")
    print(f"VLM API: {api_url}")

    print(f"\n[1/4] 加载学生答案...")
    with open(processed_json, 'r', encoding='utf-8') as f:
        answer_data = json.load(f)

    student_id = answer_data['student_id']
    answer_blocks = answer_data['answer_blocks']

    print(f"  学生ID: {student_id}")
    print(f"  答题数: {len(answer_blocks)}")

    print(f"\n[2/4] 加载评分标准...")
    rubric_dir = Path(rubric_dir)
    print(f"  Rubric目录: {rubric_dir}")

    print(f"\n[3/4] VLM评分...")

    grading_results = []

    for block in answer_blocks:
        question_id = block['question_id']
        image_path = block['image_path']

        print(f"\n  评分 {question_id}...")

        rubric = load_rubric(rubric_dir, question_id)

        if not rubric:
            print(f"    跳过（无rubric）")
            continue

        question_type = rubric.get('question_type', '')

        objective_types = [
            'choice', 'multiple_choice',
            'fill_in_blank_recitation',
            'literary_common_sense'
        ]

        is_objective = any(t in question_type for t in objective_types)

        if is_objective:
            print(f"    跳过（客观题，不需要VLM）")
            grading_result = {
                'question_id': question_id,
                'page': block['page'],
                'image_path': image_path,
                'rubric': rubric,
                'vlm_score': None,
                'max_score': rubric.get('total_score', 0),
                'skipped': True,
                'reason': '客观题，使用OCR+规则匹配评分'
            }
            grading_results.append(grading_result)
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
            'image_path': image_path,
            'rubric': rubric,
            'vlm_score': vlm_result.get('total_score', 0),
            'max_score': vlm_result.get('max_score', 0),
            'comment': vlm_result.get('comment', ''),
            'raw_output': vlm_result.get('raw_output', ''),
            'model': vlm_result.get('model', vlm_model)
        }

        grading_results.append(grading_result)

        print(f"\n    【评分结果】")
        print(f"    得分: {vlm_result['total_score']}/{vlm_result['max_score']}")
        if vlm_result.get('comment'):
            print(f"    评语: {vlm_result['comment'][:100]}")
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

    print(f"\n[4/4] 保存评分结果...")

    total_score = sum(r['vlm_score'] for r in grading_results if r.get('vlm_score') is not None)
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
    print(f"学生ID: {student_id}")
    print(f"VLM评分: {total_score}/{max_total} (主观题)")
    print(f"VLM评分题数: {len(grading_results) - skipped_count}")
    print(f"跳过客观题: {skipped_count}")
    print(f"总题数: {len(grading_results)}")
    print(f"输出JSON: {output_json}")
    print(f"\n说明: 客观题需要用OCR+规则匹配单独评分")

    return output_data


def main():
    PROCESSED_JSON = BASE_DIR / "outputs/processed_answers/xiaoming_processed.json"
    RUBRIC_DIR = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/rubric_guided_scoring"
    OUTPUT_JSON = BASE_DIR / "outputs/vlm_grading/xiaoming_grading.json"

    if not PROCESSED_JSON.exists():
        print(f"错误: 处理后的答案不存在: {PROCESSED_JSON}")
        print("请先运行: python process_adjusted_detections.py")
        return

    grade_with_vlm(
        processed_json=PROCESSED_JSON,
        rubric_dir=RUBRIC_DIR,
        output_json=OUTPUT_JSON,
        vlm_model='local',
        api_url='http://127.0.0.1:9900'
    )


if __name__ == '__main__':
    main()
