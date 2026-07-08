# Assignment Grading

## 单份试卷的流程图

```
┌───────────────────┐
│  Rubric Generate  │
└───────────────────┘
─────────────────────
┌─────────────────┐
│  PDF Preprocess │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  OCR Recognition    │  ← PaddleOCR-VL (GPU/CPU)
│  (API call)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Grade Objective    │  ← Rule-based matching
│  (choice + blank)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  YOLO Detection     │  ← Detect answer regions
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Extract Regions    │  ← Crop answer boxes
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Grade Subjective   │  ← VLM scoring
│  (API call)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  JSON Results       │
└─────────────────────┘
```
### Step1 Rubric生成
样卷或者空卷生成rubric文件

### Step2 PDF pre-process
这里需要对pdf做预处理，灰度，dpi，缩小尺寸

### Step3 OCR
这里需要使用OCR将整张试卷转化成纯文本

### Step4 Grade Objective
使用正则与文本匹配找到客观题，参考标准答案进行批卷

### Step5 YOLO
这个yolo模型需要训练，用户专门识别答题区域

### Step6 Grade Subjective
使用VLM逐题对主观题进行评分

### Step7 output json

## service如何读文件
### 核心思路
- **输入**: 高拍仪持续扫描学生试卷到 `/scanned_exams/{student}/` 文件夹
- **输出**: Service 批改完成后生成 `/grading_results/{student}.json` 结果文件
- **并发控制**: 使用 `.locks/{student}.lock` 临时文件避免多Worker重复处理
- **状态判断**: `.json`存在=已完成，`.lock`存在=处理中，都不存在=待处理
- **动态发现**: Service 定时扫描输入文件夹，按学号排序，取下一个未处理的学生
- **进度查询**: 读取 `summary.json` 获取实时统计（已完成数）

## 接口设计
后端service需要给前端提供的核心接口
```bash
POST /api/grading/start
Body: {
  "input_folder": "/scanned_exams",
  "output_folder": "/grading_results",
  "rubric_folder": "/config/rubric"
}
```

