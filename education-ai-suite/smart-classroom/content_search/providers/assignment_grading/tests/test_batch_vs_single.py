"""
对比测试：整页评分 vs 单题评分
比较性能和准确性
"""

import requests
import base64
import re
import time
from typing import Dict, List, Tuple


SERVER_URL = "http://127.0.0.1:9900"


def extract_score(text: str, max_score: int = 10) -> Tuple[int, str]:
    """从文本中提取分数和理由"""
    score = None

    # 提取分数
    patterns = [
        r'(\d+)\s*分',
        r'得\s*(\d+)',
        r'给\s*(\d+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            s = int(m)
            if 0 <= s <= max_score:
                score = s
                break
        if score:
            break

    # 推断分数
    if score is None:
        if '满分' in text or '完全正确' in text:
            score = max_score
        elif '0分' in text or '空白' in text or '未作答' in text:
            score = 0
        elif '正确' in text and '错误' not in text:
            score = int(max_score * 0.9)
        else:
            score = int(max_score * 0.7)

    # 提取理由
    reason = ""
    keywords = ['但', '错误', '缺少', '遗漏', '正确', '完整', '空白', '未', '增根', '验根']
    for line in text.split('\n'):
        for kw in keywords:
            if kw in line:
                reason = line.strip()[:60]
                break
        if reason:
            break

    if not reason:
        reason = "详见分析"

    return score, reason


def test_batch_grading(image_path: str) -> Dict:
    """方法1：整页评分（一次请求评3题）"""
    print("\n" + "=" * 60)
    print("方法1：整页评分（一次请求）")
    print("=" * 60)

    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    prompt = """这是初中数学试卷的一页。请评分（每题10分）。

对每道题，先判断：
1. **学生是否作答**：空白/未作答=0分
2. 若已作答，判断答案正确性
3. 评估步骤完整性
4. 给出得分和理由

重要：未作答必须给0分。"""

    payload = {
        "model": "Qwen3.5-9B-int4-ov",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    start_time = time.time()
    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=180)
    elapsed = time.time() - start_time

    if response.status_code != 200:
        print(f"[ERROR] 请求失败: {response.status_code}")
        return None

    data = response.json()
    content = data['choices'][0]['message']['content']
    tokens = data['usage']['completion_tokens']
    finish_reason = data['choices'][0]['finish_reason']

    print(f"[TIME] 请求时间: {elapsed:.2f}秒")
    print(f"[TOKENS] 生成tokens: {tokens}")
    print(f"[FINISH] finish_reason: {finish_reason}")

    # 显示VLM完整回复
    print("\n" + "-" * 60)
    print("VLM完整回复:")
    print("-" * 60)
    print(content)
    print("-" * 60)

    # 提取每题的结果
    questions = {}
    lines = content.split('\n')
    current_q = None

    for line in lines:
        match = re.search(r'(?:第|题目|Q)?\s*(\d+)\s*[题：（]', line)
        if match:
            current_q = match.group(1)
            if current_q not in questions:
                questions[current_q] = []
        if current_q:
            questions[current_q].append(line)

    results = {}
    for q_num in sorted(questions.keys()):
        q_text = '\n'.join(questions[q_num])
        score, reason = extract_score(q_text)
        results[q_num] = {
            'score': score,
            'reason': reason,
            'full_text': q_text
        }

    print(f"\n[RESULTS] 提取结果:")
    print(f"   识别题目数: {len(results)}")
    for q_num, res in sorted(results.items()):
        print(f"   Q{q_num}: {res['score']}/10 - {res['reason']}")

    return {
        'method': 'batch',
        'time': elapsed,
        'tokens': tokens,
        'results': results,
        'full_response': content
    }


def test_single_grading(question_images: Dict[str, str]) -> Dict:
    """方法2：单题评分（每题单独请求）"""
    print("\n" + "=" * 60)
    print("方法2：单题评分（3次独立请求）")
    print("=" * 60)

    results = {}
    total_time = 0
    total_tokens = 0

    for q_num, image_path in sorted(question_images.items()):
        print(f"\n评分Q{q_num}...")
        print(f"  图片: {image_path}")

        # 读取单题图片
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        prompt = f"""这是第{q_num}题的答题区域，满分10分。

首先判断：学生是否作答？
- 如果答题区域空白/未作答，直接给0分
- 如果已作答，再评分

给出分数和理由。"""

        payload = {
            "model": "Qwen3.5-9B-int4-ov",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        start_time = time.time()
        response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
        elapsed = time.time() - start_time

        if response.status_code != 200:
            print(f"  [ERROR] 请求失败: {response.status_code}")
            continue

        data = response.json()
        content = data['choices'][0]['message']['content']
        tokens = data['usage']['completion_tokens']

        total_time += elapsed
        total_tokens += tokens

        score, reason = extract_score(content)

        results[q_num] = {
            'score': score,
            'reason': reason,
            'full_text': content,
            'time': elapsed,
            'tokens': tokens
        }

        print(f"  [TIME] {elapsed:.2f}秒, [TOKENS] {tokens} tokens")
        print(f"\n  VLM回复:")
        print("  " + "-" * 56)
        # 显示前500字符
        preview = content[:500] + "..." if len(content) > 500 else content
        for line in preview.split('\n'):
            print(f"  {line}")
        print("  " + "-" * 56)
        print(f"  [EXTRACT] 提取: {score}/10 - {reason}")

    print(f"\n总计:")
    print(f"  [TIME] 总时间: {total_time:.2f}秒")
    print(f"  [TOKENS] 总tokens: {total_tokens}")

    return {
        'method': 'single',
        'time': total_time,
        'tokens': total_tokens,
        'results': results
    }


def compare_results(batch_result: Dict, single_result: Dict):
    """对比两种方法的结果"""
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)

    # 性能对比
    print("\n[PERFORMANCE] 性能对比:")
    print(f"{'指标':<20} {'整页评分':<15} {'单题评分':<15} {'差异':<15}")
    print("-" * 65)

    time_diff = single_result['time'] - batch_result['time']
    time_ratio = single_result['time'] / batch_result['time']
    print(f"{'总耗时':<20} {batch_result['time']:.2f}秒{'':<8} {single_result['time']:.2f}秒{'':<8} +{time_diff:.2f}秒 ({time_ratio:.1f}x)")

    tokens_diff = single_result['tokens'] - batch_result['tokens']
    tokens_ratio = single_result['tokens'] / batch_result['tokens']
    print(f"{'总tokens':<20} {batch_result['tokens']:<15} {single_result['tokens']:<15} +{tokens_diff} ({tokens_ratio:.1f}x)")

    # 分数对比
    print("\n[SCORE] 评分对比:")
    print(f"{'题号':<10} {'整页评分':<15} {'单题评分':<15} {'是否一致':<10}")
    print("-" * 50)

    all_questions = set(batch_result['results'].keys()) | set(single_result['results'].keys())
    match_count = 0
    total_count = 0
    consistency = 0

    for q_num in sorted(all_questions):
        batch_score = batch_result['results'].get(q_num, {}).get('score', 'N/A')
        single_score = single_result['results'].get(q_num, {}).get('score', 'N/A')

        if batch_score != 'N/A' and single_score != 'N/A':
            total_count += 1
            match = "[OK]" if batch_score == single_score else "[X]"
            if batch_score == single_score:
                match_count += 1
        else:
            match = "-"

        print(f"{'Q' + q_num:<10} {str(batch_score) + '/10':<15} {str(single_score) + '/10':<15} {match:<10}")

    if total_count > 0:
        consistency = match_count / total_count * 100
        print(f"\n一致性: {match_count}/{total_count} ({consistency:.0f}%)")

    # 详细评分对比（显示VLM原始评价）
    print("\n[DETAIL] 详细评分对比:")
    for q_num in sorted(all_questions):
        batch_data = batch_result['results'].get(q_num, {})
        single_data = single_result['results'].get(q_num, {})

        print(f"\n{'='*60}")
        print(f"Q{q_num}:")
        print(f"{'='*60}")

        print(f"\n整页评分: {batch_data.get('score', 'N/A')}/10")
        batch_text = batch_data.get('full_text', 'N/A')
        if batch_text != 'N/A':
            preview = batch_text[:300] + "..." if len(batch_text) > 300 else batch_text
            print(f"VLM评价: {preview}")

        print(f"\n单题评分: {single_data.get('score', 'N/A')}/10")
        single_text = single_data.get('full_text', 'N/A')
        if single_text != 'N/A':
            preview = single_text[:300] + "..." if len(single_text) > 300 else single_text
            print(f"VLM评价: {preview}")

    # 结论
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)

    if batch_result['time'] < single_result['time']:
        time_save = single_result['time'] - batch_result['time']
        print(f"[+] 整页评分更快，节省 {time_save:.2f}秒 ({time_ratio:.1f}x加速)")
    else:
        print(f"[-] 单题评分更快")

    if batch_result['tokens'] < single_result['tokens']:
        token_save = single_result['tokens'] - batch_result['tokens']
        print(f"[+] 整页评分更省tokens，节省 {token_save} tokens")
    else:
        print(f"[-] 单题评分更省tokens")

    if consistency >= 80:
        print(f"[+] 评分一致性高 ({consistency:.0f}%)")
    else:
        print(f"[!] 评分一致性较低 ({consistency:.0f}%)，需注意")


if __name__ == '__main__':
    print("对比测试：整页评分 vs 单题评分")

    # 检查服务器
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health.status_code == 200:
            info = health.json()
            print(f"[OK] VLM服务器: {info['model']} on {info['device']}")
        else:
            raise Exception()
    except:
        print("[ERROR] VLM服务器未启动")
        exit(1)

    # 测试图片
    batch_image = "outputs/math_paper_3_preprocessed.jpg"

    # 单题图片
    question_images = {
        '19': 'outputs/q19_full_question.jpg',
        '20': 'outputs/q20_full_question.jpg',
        '21': 'outputs/q21_full_question.jpg',
    }

    # 检查文件存在
    import os
    if not os.path.exists(batch_image):
        print(f"[ERROR] 整页图片不存在: {batch_image}")
        exit(1)

    for q_num, img_path in question_images.items():
        if not os.path.exists(img_path):
            print(f"[ERROR] 单题图片不存在: {img_path}")
            exit(1)

    print(f"\n测试图片:")
    print(f"  整页: {batch_image}")
    for q_num, img_path in sorted(question_images.items()):
        print(f"  Q{q_num}: {img_path}")

    # 方法1：整页评分
    batch_result = test_batch_grading(batch_image)

    # 方法2：单题评分
    single_result = test_single_grading(question_images)

    # 对比分析
    if batch_result and single_result:
        compare_results(batch_result, single_result)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
