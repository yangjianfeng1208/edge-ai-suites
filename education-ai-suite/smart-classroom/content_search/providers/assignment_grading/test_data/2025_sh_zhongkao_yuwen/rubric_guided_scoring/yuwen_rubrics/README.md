# 2025 上海中考语文 Rubric-Guided Scoring 资产

## 目录结构

```
yuwen_rubrics/
├── exam_meta.json                       # 整卷元信息
├── objective_answers.json               # 12 道客观题标准答案（OCR + 规则匹配）
├── Q5_translation.json                  # 文言文翻译（3 分）
├── Q8_classical_analysis.json           # 文言文分析（8 分）
├── Q10_reason_analysis.json             # 小说原因分析（4 分）
├── Q11_psychological_description.json   # 心理描写作用（5 分）
├── Q12_imagery_appreciation.json        # 画面感赏析（5 分）
├── Q16_news_language_comparison.json    # 新闻语言对比（6 分）
├── Q19_inquiry_insights.json            # 开放式启发感悟（6 分）
├── Q20_poem_revision.json               # 诗歌修改分析（10 分，rubric 不完整）
├── Q21_composition.json                 # 作文（60 分，整体评分）
└── README.md                            # 本文件
```

## 题目分类

### 客观题（37 分，走 OCR + 规则匹配，不调 LLM）
Q1 默写、Q2 作者、Q3 字词、Q4 词类活用、Q6 品质、Q7 乙文理解、Q9 人物关系、Q13 关联词、Q14 新闻分析、Q15 读者需求、Q17 资料选择、Q18 排序

### 主观题（113 分，走 VLM + Rubric-Guided Scoring）
| 题号 | 分值 | 类型 | rubric 完整度 |
|------|------|------|--------------|
| Q5  | 3   | 文言翻译 | ✅ 完整 |
| Q8  | 8   | 文本分析 | ✅ 完整 |
| Q10 | 4   | 原因填空 | ✅ 完整 |
| Q11 | 5   | 文学分析 | ✅ 完整 |
| Q12 | 5   | 文学赏析 | ✅ 完整 |
| Q16 | 6   | 比较分析 | ✅ 完整 |
| Q19 | 6   | 开放感悟 | ✅ 完整 |
| Q20 | 10  | 诗歌赏析 | ⚠️ 待补全（缺原诗与改诗文本） |
| Q21 | 60  | 作文 | ✅ 完整（整体评分制） |

## 数据结构说明

每道主观题 JSON 包含以下核心字段：

```json
{
  "question_id": "Q_n",
  "section": "所属大题",
  "type": "题型分类",
  "total_score": 总分,
  "question_text": "题干",
  "context": { "...相关原文/材料..." },
  "standard_answer_summary": "参考答案要点",
  "scoring_points": [
    {
      "id": "p1",
      "score": 该得分点分值,
      "description": "得分点描述",
      "criteria": "判分标准",
      "hit_signals": ["命中关键词1", "命中关键词2"],
      "must_have": true/false
    }
  ],
  "few_shot_examples": [
    {"answer": "学生答案样本", "score": 应得分, "hits": ["p1","p2"]}
  ]
}
```

## 使用流程（VLM 阅卷）

```python
import json
from pathlib import Path

# 1. 加载 rubric
rubric = json.load(open("yuwen_rubrics/Q11_psychological_description.json"))

# 2. 拼装 prompt
prompt = f"""你是严谨的语文阅卷老师。请按给定 rubric 逐点判分。

=== 题目 ===
{rubric['question_text']}

=== 原文 ===
{rubric['context']['passage_5']}

=== 参考答案 ===
{rubric['standard_answer_summary']}

=== 评分细则 ===
{json.dumps(rubric['scoring_points'], ensure_ascii=False, indent=2)}

=== 评分示例 ===
{json.dumps(rubric['few_shot_examples'], ensure_ascii=False, indent=2)}

=== 学生答案 ===
[IMAGE]

=== 输出格式 ===
{{
  "scoring_points": [{{"id": "p1", "hit": bool, "evidence": "..."}}],
  "total_score": int,
  "confidence": 0.0-1.0,
  "needs_review": bool
}}
"""

# 3. 调 VLM
result = vlm.generate(prompt=prompt, image=student_answer_image, max_new_tokens=300)
```

## 数据源

- [人人文库 2025 上海中考语文真题及答案详解](https://www.renrendoc.com/paper/433079932.html)
- [二一教育官方答案解析](https://zy.21cnjy.com/23234426)
- [Scribd 解析版](https://www.scribd.com/document/1008080430/)

## 待办

1. **Q20 缺数据**：原稿与修改稿的诗歌文本需结合试卷扫描图补全
2. **Q4 考点存疑**：词类活用 vs 主谓间取独，建议拿到官方权威答案后核对
3. **首批锚卷未建立**：建议挑 10-20 份真实学生答卷做人工评分作为校准基线
