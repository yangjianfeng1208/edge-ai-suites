"""
调试VLM评分 - 看原始输出
"""

import requests
import base64

SERVER_URL = "http://127.0.0.1:9900"

print("测试Q19单题评分\n")

# 读取Q19图片
with open('outputs/q19_full_question.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 最简单的提示
prompt = "看图片中的第19题，满分10分。你给多少分？为什么？"

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

print(f"提示词: {prompt}\n")
print("=" * 60)

response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)

if response.status_code == 200:
    data = response.json()
    content = data['choices'][0]['message']['content']

    print("VLM原始回复:")
    print("=" * 60)
    print(content)
    print("=" * 60)

    print(f"\n请您判断:")
    print("1. VLM的评价是否正确？")
    print("2. 给出的分数是多少？")
    print("3. 正确答案应该是多少分？")
else:
    print(f"错误: {response.status_code}")
