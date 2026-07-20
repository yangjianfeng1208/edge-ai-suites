"""
VLM JSON格式评分 - 快速直接
"""

import requests
import base64
import re
import time

SERVER_URL = "http://127.0.0.1:9900"

image_path = "outputs/q20_full_question.jpg"

with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

prompt = """第20题，满分10分。请评分。"""

print("[test v2.0]策略：自然分析，智能提取\n")
print("正在请求VLM...\n")

start_time = time.time()

payload = {
    "model": "Qwen3.5-9B-int8-ov",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ],
    "max_tokens": 1500,
    "temperature": 0.3
}

response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)

elapsed_time = time.time() - start_time

if response.status_code != 200:
    print(f"错误: {response.status_code}")
    exit(1)

data = response.json()
content = data['choices'][0]['message']['content']
tokens = data['usage']['completion_tokens']
finish_reason = data['choices'][0]['finish_reason']

print("=" * 60)
print("VLM原始回复:")
print("=" * 60)
print(content)
print("=" * 60)
print(f"请求时长: {elapsed_time:.2f}秒")
print(f"生成tokens: {tokens}, finish_reason: {finish_reason}\n")

print("=" * 60)
print("提取分数:")
print("=" * 60)

patterns = [
    (r'得分[：:]\s*(\d+)\s*/\s*10', '得分格式'),
    (r'得分[：:]\s*(\d+)\s*分', '得分格式2'),
    (r'评分[：:]\s*(\d+)\s*分', '评分格式'),
    (r'给\s*(\d+)\s*分', '给分格式'),
    (r'扣\s*(\d+)\s*分', '扣分格式', True),
    (r'(\d+)\s*/\s*10', 'X/10格式'),
    (r'\*\*(\d+)\s*分\*\*', '加粗分数'),
]

score = None
reason = ""
for item in patterns:
    if len(item) == 3:
        pattern, name, is_deduct = item
    else:
        pattern, name = item
        is_deduct = False

    match = re.search(pattern, content)
    if match:
        s = int(match.group(1))
        if is_deduct:
            s = 10 - s
        if 0 <= s <= 10:
            score = s
            print(f"提取分数: {score}/10 (使用{name})")
            break

if score is None:
    lines = content.split('\n')
    for line in lines:
        if '满分' in line or '10分' in line or '正确' in line:
            print(f"可能相关行: {line[:80]}")
    print("\n未能自动提取分数，需要检查VLM回复")

print("=" * 60)
