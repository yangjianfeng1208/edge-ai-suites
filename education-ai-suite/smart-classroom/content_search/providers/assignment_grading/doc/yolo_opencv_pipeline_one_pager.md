# YOLO + OpenCV 主观题答题区域提取方案

> **目标**：从扫描的整张试卷中，自动定位并裁剪每道主观题的学生答题区域，输出给 VLM 评分。
> **适用范围**：仅主观题（解答题、作文、简答等）；客观题走 OCR + 规则。
> **部署目标**：Intel B580 GPU + OpenVINO，单张试卷端到端 < 3 秒。

---

## 一、完整 Pipeline

```
扫描整张试卷 (300 DPI, ~2480×3508)
        ↓
[1] 预处理：去噪 → 二值化 → 矫正倾斜
        ↓
[2] 对齐：ArUco 检测 → 单应性变换 → 对齐到模板坐标系
        ↓
[3] YOLO 推理：检测所有答题区域 bbox
   输出：[(class, x, y, w, h, conf), ...]
        ↓
[4] 题号匹配：bbox 坐标查表 → 对应到 Q11 / Q12 / ...
        ↓
[5] OpenCV 裁剪：按 bbox + padding 切图
        ↓
[6] 空白检测：墨水占比 < 0.5% → 直接给 0 分，跳过 VLM
        ↓
[7] 送 VLM 评分：配合该题 rubric JSON
```

---

## 二、各阶段关键设计

### 阶段 1-2：预处理与对齐（OpenCV）

| 步骤 | 方法 | 输出 |
|---|---|---|
| 去噪 | `cv2.fastNlMeansDenoising` | 干净灰度图 |
| 二值化 | `cv2.adaptiveThreshold` | 黑白图 |
| 对齐 | ArUco 4 角标记 + `cv2.findHomography` | 对齐到模板的标准坐标系 |
| 回退 | 黑方块 → ORB/SIFT 特征点 → SuperPoint | 多级容错 |

### 阶段 3：YOLO 检测

- **模型**：YOLOv8m（B580 推理友好，精度够用）
- **输入尺寸**：1280×1280（A4 纸细节需要高分辨率）
- **类别**（5 类）：
  - `choice_question`（选择题答题区）
  - `fill_blank`（填空答题区）
  - `solve_question`（解答题答题区）← **重点**
  - `question_number`（题号印刷区，辅助匹配）
  - `anchor_marker`（ArUco / 黑方块，辅助对齐）

### 阶段 4：题号匹配（两种方案）

| 方案 | 做法 | 可靠性 | 适用场景 |
|---|---|---|---|
| **A. 坐标查表（推荐）** | 模板固定，预定义每题坐标范围 | ★★★★★ | 标准模板试卷 |
| B. 空间关系匹配 | `question_number` bbox 距离最近的答题区 | ★★★ | 模板可能变化时兜底 |

**生产环境用 A，B 当备用**。

### 阶段 5：裁剪（OpenCV）

```python
pad = 20  # 留白，避免手写超出 bbox 被切掉
y1 = max(0, y - pad)
y2 = min(H, y + h + pad)
x1 = max(0, x - pad)
x2 = min(W, x + w + pad)
cropped = image[y1:y2, x1:x2]
```

### 阶段 6：空白检测（省 LLM 调用）

```python
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
ink_ratio = binary.sum() / 255 / binary.size
if ink_ratio < 0.005:
    return {"score": 0, "reason": "空白未作答"}
```

一个班约 10-20% 主观题空白率，能省 10-20% VLM 调用。

### 阶段 7：送 VLM

裁剪图 + 该题 rubric JSON → InternVL3.5-8B / Qwen2.5-VL → 评分输出。

---

## 三、关键工程注意点

1. **YOLO bbox 只框答题空白区，不框题干**——否则 VLM 会把题干当作答案。
2. **bbox 边缘加 padding**——学生写字常超格，留 20 像素余量。
3. **MVP 不做得分点级别检测**——只到题级别即可，得分点判断交给 VLM + rubric。
4. **对齐失败要硬性兜底**——置信度低时降级到坐标查表 + 人工复核标记。
5. **batch 推理**——同一批扫描卷一起送 YOLO，吞吐量翻倍。

---

## 四、性能预期（B580 + OpenVINO INT8）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 预处理 + 对齐 | 100-200 ms | OpenCV CPU |
| YOLO 推理 | 50-100 ms | OpenVINO INT8 |
| 裁剪 + 空白检测 | < 50 ms | 纯像素操作 |
| **本方案总计** | **< 400 ms / 张** | 不含 VLM |
| VLM 评分（5 题）| 5-15 秒 | 主要瓶颈在 VLM |

---

## 五、训练数据需求（YOLO）

- **总量**：500-1000 张标注图
- **构成**：80% 合成（基于空白模板叠加手写样本）+ 20% 真实学生卷
- **标注工具**：Roboflow（推荐）/ LabelImg / CVAT
- **数据增强**：旋转 ±5°、亮度 ±20%、模糊、JPEG 压缩噪声

---

## 六、与整体阅卷系统的关系

```
[本方案 = 区域提取层]
        ↓ 输出裁剪图
[Rubric-Guided Scoring 评分层] ← 配合 rubric JSON
        ↓ 输出评分
[人工复核层] ← 低置信度 / 分歧大的样本
```

本方案只解决 **"把学生写的内容切出来"** 这一步；评分准确性由 Rubric-Guided Scoring 方案保证。

---

## 参考资料

- [IJERT 2025: YOLO + OpenCV OMR 论文](https://www.ijert.org/research/a-web-based-automated-omr-evaluation-system-using-yolo-and-image-processing-techniques-IJERTV14IS120698.pdf)（98-99% 准确率，2-4 秒/张）
- [OMRChecker GitHub](https://github.com/Udayraj123/OMRChecker)
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
