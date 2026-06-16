# Rubric-Guided Scoring 一页纸方案（仅解答题）

> 选择题、填空题走 OCR + 规则匹配，本方案只针对**解答题/主观题**用 VLM 评分。让 VLM 不再"凭感觉给分"，而是按结构化评分细则逐点判分。学术界主流范式，可达接近甚至超越人类双评一致性。

---

## 一、核心思想

**离线一次性投入**把每道解答题的标准答案预处理成结构化资产，**在线阅卷**时按题加载，让 VLM 只做"逐点判断"而非"自由打分"。

```
解答题标准答案 → 离线拆解 → 每题 rubric JSON → 在线拼 prompt → VLM 输出结构化分数
```

---

## 二、Rubric 数据结构（每道解答题一份 JSON）

```json
{
  "question_id": "Q23",
  "question_text": "...",
  "standard_answer": "...",
  "total_score": 10,
  "scoring_points": [
    {
      "id": "p1",
      "score": 2,
      "description": "正确写出导数表达式",
      "key_form": "f'(x) = 2x - 3",
      "accept_equiv": ["2x-3", "-3+2x"]
    },
    { "id": "p2", "score": 2, "description": "令导数为 0 求驻点", "key_form": "x = 3/2" },
    { "id": "p3", "score": 3, "description": "验证为极小值",
      "key_phrases": ["极小", "二阶导>0", "左减右增"] },
    { "id": "p4", "score": 3, "description": "正确算出最小值",
      "key_form": "-9/4", "accept_equiv": ["-2.25"] }
  ],
  "alternative_paths": [
    { "name": "配方法", "scoring_points": [...] },
    { "name": "判别式法", "scoring_points": [...] }
  ],
  "few_shot_examples": [
    { "answer": "x=3/2, 最小值=-9/4", "score": 5, "hits": ["p2", "p4"],
      "explanation": "命中驻点和最终值，但缺求导过程和极值验证" },
    { "answer": "完整四步推导", "score": 10, "hits": ["p1","p2","p3","p4"] }
  ]
}
```

---

## 三、离线准备流程（出卷后阅卷前）

只针对解答题。一份试卷通常 4-8 道解答题。

| 步骤 | 工作 | 工具 |
|------|------|------|
| 1 | 拆解原子得分点 | LLM 辅助 + 教师审核 |
| 2 | 枚举主流解法路径 | 教师整理 + LLM 扩展 |
| 3 | 编写 few-shot 学生样本 | LLM 生成 + 教师筛选 |
| 4 | 准备 10-20 份锚卷（仅解答题部分） | 教师人工预判 |
| 5 | 固化为 `rubric_Qx.json` | 一题一文件 |

预估工作量：**每题 10-20 分钟**，一份卷 4-8 道解答题 → 1-2 小时离线投入。

---

## 四、在线阅卷 Prompt 模板

```text
[System - 固定不变，KV Cache 复用]
你是严谨的数学阅卷老师。请严格按给定 rubric 逐点判分。
输出必须是合法 JSON，无任何额外说明。
评分原则：
- 字数不影响给分，只看得分点是否命中
- 即使最终答案错误，过程对的步骤仍给步骤分
- 学生使用 rubric 中列出的 alternative_paths 之一时，按对应路径评分

[User - 每题动态拼接]
=== 题目 ===
{question_text}

=== 标准答案 ===
{standard_answer}

=== 评分细则 ===
{rubric_json}

=== 评分示例 ===
{few_shot_examples}

=== 学生答案 ===
[IMAGE - 该题答题区域裁剪图]

=== 输出格式 ===
{
  "solution_path": "<主路径或 alternative 名称>",
  "scoring_points": [{"id": "p1", "hit": bool, "evidence": "..."}],
  "total_score": int,
  "confidence": 0.0-1.0,
  "needs_review": bool
}
```

---

## 五、整体流水线（解答题部分）

```
学生答卷扫描
    ↓
OpenCV + PaddleOCR 题号定位 → 切出每道解答题答题区域
    ↓
按题号加载对应 rubric_Qx.json
    ↓
拼 prompt → 本地 VLM（Qwen3.5 / InternVL3.5）推理
    ↓
Self-Consistency 采样 N=3，多数投票
    ↓
后处理
├─ 置信度 ≥ 阈值 + 三次结果一致 → 自动通过
├─ 三次结果分歧 ≥ 2 分 → 人工复核队列
└─ 低置信度 → 人工复核队列
    ↓
锚卷校准：每批穿插 3-5 份锚卷，监测系统性偏差
```

---

## 六、关键工程要点

1. **题号路由必须确定性**：用 OCR + 坐标定位题号，不让 VLM 判断"这是第几题"
2. **KV Cache 复用**：System Prompt 固定，OpenVINO GenAI 支持 prefix caching，省 50% 推理时间
3. **原子化得分点**：每条独立 0/1 判断，避免 VLM 自由发挥
4. **多路径支持**：数学题常有多解，rubric 中预置 alternative_paths
5. **Self-Consistency 投票**：解答题采样 N=3，分歧大进人工
6. **置信度路由**：低置信度自动标记，不强求 100% 自动化

---

## 七、目标质量指标（仅解答题）

| 指标 | 目标值 | 参考依据 |
|------|--------|----------|
| 与人工评分 QWK | ≥ 0.85 | 接近人类双评水平 |
| 自动通过率 | ≥ 70% | 其余进人工复核 |
| 单题端到端延迟 | 2-5 秒（B580 + 9B VLM） | 物理底线 |
| 锚卷漂移 | 偏差 ≤ 0.5 分 | 系统性公平 |

---

## 八、对比：传统 vs Rubric-Guided

| 维度 | 传统 Prompt 直接打分 | Rubric-Guided Scoring |
|------|----------------------|------------------------|
| 准确性 | 70-80% | 85-95% |
| 一致性 | 同题多次结果差异大 | 稳定可复现 |
| 可解释性 | 黑盒 | 每个得分点都有 evidence |
| 公正性 | 受答案长度、字迹影响 | rubric 中可显式约束 |
| 可维护性 | 改 prompt 全卷重测 | 改某题不影响其他 |
| 讲评生成 | 需额外调用 | 直接复用 evidence |

---

## 九、立即可做的三件事

1. **挑 1 道解答题**做完整 rubric 设计（含原子得分点 + 多路径 + few-shot）
2. **写 prompt 模板代码**，跑通"加载 rubric → 拼 prompt → VLM 推理 → 解析 JSON"
3. **准备 5-10 份该题的锚卷**（不同分数档），比对 VLM 与人工评分，建立基线
