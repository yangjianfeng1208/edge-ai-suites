## 场景与需求

requirement: https://github.com/open-edge-platform/edge-ai-suites/issues/2781

```bash
# What
Add Mandarin language OCR support to the Smart Classroom application using open-source OCR models, and enable an end-to-end workflow that:

1. ingests multiple scanned student assignments (simulating up to 4 high-speed document cameras),
2. digitizes the content via OCR,
3. grades the assignment using an on-device edge LLM/VLM (7B–12B class) grounded by a local knowledge base (RAG),
4. writes one local output file per submitted assignment containing both the digitized input and grading results.
# Key Experience Indicator (KEI)
The app shall consume no more than 3 minutes to process 30 documents with an average of 3 pages each.
which is 1s per page
```
### 1. 输入 | 输出
**输入**
桌面高拍仪扫描学生试卷，最多4台扫描仪同时工作，每个学生的试卷为一个document，每一个doc大概3页左右，这里假设扫描后的文档为pdf

**输出**
每个学生的卷面成绩

### 2. 应用场景 | 主要需求
**假设场景**
1. 使用场景应该是针对非正式考试（随堂测试、练习卷）
2. 支持的学科假设包括文+理
3. 假设试卷为标准测试卷，一般都为普通的8开纸张试卷(4 pages)，包含客观题与主观题

**猜测需求为**
1. 对准确性的需求可能大于对处理速度的需求，避免人工重新审核
2. 对于阅卷效果，客观题应该尽量判断准确，而主观题则尽量要求评分统一

### 4. 方案选择
**整卷进模型评分**
1. 直接ocr转换全卷为纯文本 -> LLM
2. 整张卷子分页截图 -> (4 pages) -> 每页依次进VLM

**拆分后分别评分**
1. 识别出主观题与客观题，客观题ocr处理，主观题截图进vlm
2. 主观题每一题设置单独的评分标准，得分点

**整卷评分存在的issues**
1. 纯文本整张试卷 -> 主观题无法识别图片，公式以及结构复杂的答题区
2. 大语言模型的幻觉，判错题
3. 大语言模型的评分标准不统一，无提示词约束，或者不同的提示词，结果差别很大

### 5. proposal
1. 总体的判题准确性，引入Rubric，评分量表，即每题的标准答案以及得分点，降低不同大小，不同精度的模型的输出一致性
2. 解决客观题判题速度，paddleocr识别，对客观题直接正则对比
3. 解决主观题包含图片，公式，答题区域结构负责的问题，使用vlm进行判题
4. 解决主观题幻觉问题，单题单次进vlm，prompt携带该题的rubric
5. 解决识别主观题答题区域的问题，训练yolo模型识别答题区 (手动标注模板卷不可行)

### 6. 评分量表 Rubric

Rubric: https://zhuanlan.zhihu.com/p/2044494521933852860

在教育评估中，rubric 通常指一套评分指南：
由教师提供，包含：题目原文、评分点（每点分值）、评分标准、参考答案、评分示例。大语言模型基于Rubric对学生答案逐项评判，输出得分和理由，确保评分标准统一且可追溯，便于教师复查和质量控制。

### 7. workflow
- step 1. 使用老师的答题样卷作为标准答案生成评分量表(Rubric)
- step 2. 教师在生成的rubric基础上更新，调整自己的评分标准，得分点
- step 3. 扫描试卷
- step 4. 试卷归档转图片，使用paddleocr转文本
- step 5. 使用yolo对图片进行处理识别试卷的答题区
- step 6. 客观题无需使用大语言模型，纯文本正则
- step 7. 主观题逐题截图送入VLM，prompt加入rubric的评分标准

### 8. 初步的pipeline
grading service:
```bash
pdf -> image -> yolo11s -> (maunal adjust) -> opencv_adjust -> paddleocr_vl_1.6 -> Qwen3.5_9b -> score.json
```
`yolo11s` (trained using HiLEx)

`paddleocr_vl_1.6`: mem (1.9->5.4->7.9GB) | A4单页 ~10s(B580) - 1mins+(iGPU/CPU)

`Qwen3.5_9b`: int8_ov | 默认多模态 | mem (10.3GB) | 单题响应 1min+ (B580) | without disable think mode

### 9. 目前存在的挑战
1. paddleOCR无法100%正确
2. Yolo模型需要真实的考卷进行训练，答题区域识别成功率不高
3. 主观题题号的映射
4. 试卷布局问题，比如题目与答题不在同一页，或者同一题的答题区跨页
5. 鉴于目前模型的处理速度，在保证准确性的前提下，处理速度会很慢
