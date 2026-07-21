# 客观题答案提取与归一化方案设计

## 背景

`grade_objective_questions.py` 的 `extract_answer` 在填空题(blank)上大量出错:

- `\underline{...}` 提取用非贪婪 `(.+?)\}+`,遇到 LaTeX 花括号会截断
  (`m < -\frac{1}{8}` → `m < -\frac{1`)
- 没有 `\underline` 时的两个 fallback 正则从题干任意位置乱抓文字,
  抓到中文题干(`y随x的增大而减小`、`已知小明手里有 1`)
- 题号匹配 `sub_patterns` 里混入 mojibake 乱码字符(`锛塡`/`锛圽`,
  本应是全角 `）（`)

选择题(choice)靠 NFKC 把全角括号转半角侥幸能过,但填空题是重灾区。

## 核心思路:两层策略(第一版)

答案判定从"精确切出学生答案"(易切错)改为分层:

1. **underline 优先** — 有 `\underline{...}` 时,配平花括号提取其内容
   (学生填空位置,最可信)
2. **归一化包含兜底** — 没有 underline 时,判断"标准答案是否出现在
   这道题的归一化文本中"

> **第一版决定:不加短答案护栏。** 兜底对所有答案(含 `1.2`/`1800`
> 这类纯数字)一律启用,追求最高命中率,接受纯数字被题干数字误命中的
> 风险。护栏留待后续版本(见"已知局限")。

## 归一化函数 `normalize_for_match`

对"待比较字符串"(提取值 / 标准答案 / 题目文本)统一施加,使两边可比。

顺序:

1. `NFKC`(复用现有 `normalize_text`)—— 全角→半角、上标数字归一
2. 去 LaTeX 数学定界符:`\(` `\)` `\[` `\]` `$$` `$`
3. 去 LaTeX 排版噪声命令(**只删命令本身,保留内容**):
   `\quad` `\qquad` `\,` `\;` `\!` `\ ` `\text{...}`→内容
   `\left` `\right`(保留其后的括号)
4. 去掉**所有空白**(空格/制表/换行)——答案里的 ` ` 不稳定

> **第一版决定:不动标点。** 不去除末尾 `.` `。` `,` `，`,保留标点原样。
> `1.2` 的小数点在字符串中间,本就不受影响。

归一化只用于**比较**,不改写原始 OCR/答案(保留原文用于展示)。

伪码:

```python
_LATEX_DELIMS = [r'\(', r'\)', r'\[', r'\]', '$$', '$']
_LATEX_NOISE  = [r'\quad', r'\qquad', r'\,', r'\;', r'\!', r'\left', r'\right']

def normalize_for_match(s: str) -> str:
    if not s:
        return ''
    s = normalize_text(s)                       # NFKC
    for d in _LATEX_DELIMS:
        s = s.replace(d, '')
    # \text{X} -> X
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    for c in _LATEX_NOISE:
        s = s.replace(c, '')
    s = re.sub(r'\s+', '', s)                    # 去所有空白
    return s.strip()
```

## `\underline{}` 配平提取

替换现有非贪婪正则,按花括号计数配平,支持嵌套 `\frac{1}{8}`:

```python
def _extract_underline(content: str) -> Optional[str]:
    idx = content.find(r'\underline')
    if idx < 0:
        return None
    brace = content.find('{', idx)
    if brace < 0:
        return None
    depth, i = 0, brace
    while i < len(content):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                return content[brace + 1:i]
        i += 1
    return None    # 花括号不配平 -> 视为无 underline
```

## 包含兜底(第一版:无护栏)

归一化后逐一判断标准答案是否出现在题目文本中,命中即返回。
不做长度 / 纯数字过滤。

```python
def _match_by_containment(norm_content: str, standards: list[str]) -> Optional[str]:
    for raw in standards:
        norm_ans = normalize_for_match(raw)
        if norm_ans and norm_ans in norm_content:
            return raw    # 返回原始标准答案,便于展示
    return None
```

## 新的 `extract_answer` / 判定流程

choice 分支基本不变(NFKC 后匹配 `(A)`)。blank 改为:

```
1. underline = _extract_underline(content)
   若有 -> 提取值 = normalize_for_match(underline)
           与 normalize_for_match(每个标准答案) 逐一比较,命中即对
2. 无 underline -> _match_by_containment(normalize_for_match(content), answers)
3. 都没命中 -> None(判错,但不再假阳性乱匹配)
```

判定统一走归一化比较,`check_answer` 的 `any/all/set` 模式保留,
但比较前对两边都 `normalize_for_match`。

## 各题预期结果(修复后)

| Q | 标准答案 | 修复后 |
|---|---|---|
| 7 | `ab(a+b)` | ✅ underline 去 `\quad` 命中 |
| 9 | `m < -\frac{1}{8}` | ✅ 配平花括号,去空格命中 |
| 12 | `y = 3x^{2} - 2` | ✅ 配平,去空格命中 |
| 18 | `36°`/`108°` | ✅ 配平命中(其一) |
| 10/11/16 | LaTeX 公式 | ✅ 归一化包含命中 |
| 13 | `\frac{1}{2}` | ✅ 有辨识度,包含命中(若 OCR 还原) |
| 14 | `1.2` | ⚠️ 边界,见局限 |
| 15 | `1800` | ✅ len=4 有辨识度 |

## 已知局限(不是本函数能修的)

1. **纯数字短答案**(如 `1.2`)即使加护栏,若题干恰好含相同数字仍会
   误判为对。彻底解法需要答案区域定位(step3 已有 bbox),按位置只在
   "答案框"内查找,而不是整题文本。本方案不涉及,属后续优化。
2. **OCR 未还原学生手写答案**时,提取必然为 None(判错)。这是 OCR
   数据质量问题,本方案把"抓错答案的假阳性"降级为"找不到的真阴性"。
3. NFKC 会合并部分符号(上标 `²`→`2`),对数学表达一般安全,极端
   公式可能误伤,概率低。

## 改动范围

- `text_normalizer.py`:新增 `normalize_for_match`
- `grade_objective_questions.py`:
  - 修 `sub_patterns` 乱码字符(`锛塡`→`）`,`锛圽`→`（`)
  - `extract_answer` blank 分支重写(underline 配平 + 删除乱抓 fallback)
  - `check_answer` 比较前归一化
  - 新增 `_extract_underline` / `_match_by_containment`
