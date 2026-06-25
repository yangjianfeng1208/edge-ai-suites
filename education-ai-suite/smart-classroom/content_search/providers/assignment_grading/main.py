from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).parent


def main():
    print("="*80)
    print("自动评分系统")
    print("="*80)

    SKIP_SUBJECTIVE = True
    USE_CACHED_OCR = False
    OCR_DEVICE = "GPU.1"
    SAVE_PDF_IMAGES = False
    PDF_DPI = 50
    OCR_MAX_PIXELS = 10000000
    OCR_MAX_TOKENS = 4096

    # DETECTION_JSON = BASE_DIR / "test_data/2025_sh_zhongkao_math/papers/student1/yolo_detections.json"
    DETECTION_JSON = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/papers/student1/yolo_detections.json"

    if not DETECTION_JSON.exists():
        print(f" 错误: 检测JSON不存在: {DETECTION_JSON}")
        return

    print(f"\n[配置] 加载检测JSON配置...")
    with open(DETECTION_JSON, 'r', encoding='utf-8') as f:
        detection_data = json.load(f)

    PDF_PATH = Path(detection_data['source_pdf'])
    YOLO_MODEL = Path(detection_data['yolo_model'])

    exam_root = None
    for parent in PDF_PATH.parents:
        if (parent / "rubric_guided_scoring").exists():
            exam_root = parent
            break

    if not exam_root:
        print(f"\n 错误: 找不到rubric_guided_scoring目录")
        return

    exam_name = exam_root.name
    OUTPUT_BASE = BASE_DIR / f"outputs/{exam_name}"
    PROCESSED_OUTPUT_DIR = OUTPUT_BASE / "processed_answers"
    RUBRIC_DIR = exam_root / "rubric_guided_scoring"
    GRADING_OUTPUT = OUTPUT_BASE / "vlm_grading" / f"{exam_name}_grading.json"
    VLM_API_URL = "http://127.0.0.1:9900"

    print(f"  PDF路径: {PDF_PATH}")
    print(f"  YOLO模型: {YOLO_MODEL}")
    print(f"  Rubric目录: {RUBRIC_DIR}")
    print(f"  评分输出: {GRADING_OUTPUT}")

    if not PDF_PATH.exists():
        print(f"\n 错误: PDF不存在: {PDF_PATH}")
        return

    if not YOLO_MODEL.exists():
        print(f"\n 错误: YOLO模型不存在: {YOLO_MODEL}")
        print(f"   请运行训练脚本生成模型")
        return

    if True:  
        print(f"\n{'='*80}")
        print("[0/3] YOLO检测答题区域")
        print(f"{'='*80}")

        from ultralytics import YOLO
        from utils.pdf_utils import render_pdf_to_images

        try:
            print(f"\n加载YOLO模型...")
            model = YOLO(str(YOLO_MODEL))

            print(f"渲染PDF...")
            pages = render_pdf_to_images(PDF_PATH, dpi=300)
            print(f"共{len(pages)}页")

            print(f"\n开始检测...")
            all_detections = {}

            for page_data in pages:
                page_num = page_data['page_num']
                image = page_data['image']

                results = model(image, conf=0.15, iou=0.5, verbose=False)
                boxes = results[0].boxes

                page_detections = []

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    detection = {
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': xyxy
                    }

                    page_detections.append(detection)

                all_detections[page_num] = page_detections
                print(f"  第{page_num}页: {len(page_detections)}个区域")

            print(f"\n YOLO检测完成")
            print(f"   总检测数: {sum(len(v) for v in all_detections.values())}")

        except Exception as e:
            print(f"\n YOLO检测失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        pass

    print(f"\n 使用检测结果: {DETECTION_JSON}")

    print(f"\n{'='*80}")
    print("[1/4] 提取答题区域")
    print(f"{'='*80}")

    from utils.process_adjusted_detections import process_adjusted_detections

    try:
        processed_data = process_adjusted_detections(DETECTION_JSON, PROCESSED_OUTPUT_DIR, pdf_path=PDF_PATH)
        processed_json = PROCESSED_OUTPUT_DIR / f"{processed_data['student_id']}_processed.json"
        print(f"\n 答题区域提取完成")
        print(f"   共提取 {processed_data['total_answer_blocks']} 个答题区域")
        print(f"   输出: {processed_json}")
    except Exception as e:
        print(f"\n 提取答题区域失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{'='*80}")
    print("[2/4] OCR识别答卷")
    print(f"{'='*80}")

    OCR_TEXT = OUTPUT_BASE / "ocr_text" / f"{exam_name}_ocr.txt"
    OCR_MODEL_PATH = BASE_DIR / "models/ov_paddleocr-vl-1_6_model"

    if USE_CACHED_OCR and OCR_TEXT.exists():
        print(f"\n Using cached OCR result: {OCR_TEXT}")
    else:
        if not OCR_MODEL_PATH.exists():
            print(f"\n OCR model not found: {OCR_MODEL_PATH}")
            print(f"   Please run: cd ocr_services && venv\\Scripts\\python download_and_convert_model.py")
            print(f"   Skipping OCR step...")
        else:
            try:
                import sys
                sys.path.insert(0, str(BASE_DIR / "ocr_services"))
                from paddleocr_vl_service import PaddleOCRVLService
                from pdf2image import convert_from_path

                print(f"\n Loading PaddleOCR-VL model...")
                print(f"   Device: {OCR_DEVICE}")
                ocr_service = PaddleOCRVLService(model_path=OCR_MODEL_PATH, device=OCR_DEVICE)

                print(f"\n Converting PDF to images...")
                print(f"   DPI: {PDF_DPI}")
                images = convert_from_path(PDF_PATH, dpi=PDF_DPI)
                print(f"   Total pages: {len(images)}")
                if images:
                    print(f"   Image size: {images[0].width}x{images[0].height} pixels")

                print(f"\n Converting to grayscale (exam papers are black & white)...")
                images_gray = [img.convert("L") for img in images]
                print(f"   Converted {len(images_gray)} pages to grayscale")

                if SAVE_PDF_IMAGES:
                    pdf_images_dir = OUTPUT_BASE / "pdf_images"
                    pdf_images_dir.mkdir(parents=True, exist_ok=True)
                    for idx, img_gray in enumerate(images_gray, 1):
                        img_path = pdf_images_dir / f"page_{idx}.jpg"
                        img_gray.save(img_path, "JPEG", quality=85, optimize=True)
                    print(f"   PDF images saved to: {pdf_images_dir}")

                print(f"\n Running OCR on each page...")
                print(f"   Max pixels: {OCR_MAX_PIXELS:,}")
                print(f"   Max tokens: {OCR_MAX_TOKENS}")
                results = []
                for idx, img_gray in enumerate(images_gray, 1):
                    print(f"   Processing page {idx}/{len(images_gray)}...")
                    text = ocr_service.ocr_image(
                        img_gray,
                        task="ocr",
                        max_pixels=OCR_MAX_PIXELS,
                        max_new_tokens=OCR_MAX_TOKENS
                    )
                    page_result = f"{'='*80}\nPage {idx}\n{'='*80}\n{text}\n\n"
                    results.append(page_result)

                full_text = "\n".join(results)

                OCR_TEXT.parent.mkdir(parents=True, exist_ok=True)
                with open(OCR_TEXT, 'w', encoding='utf-8') as f:
                    f.write(full_text)

                perf_stats = ocr_service.get_perf_stats()
                print(f"\n OCR completed")
                print(f"   Model load time: {perf_stats['model_load_time']:.2f}s")
                print(f"   Total OCR time: {sum(perf_stats['page_times']):.2f}s")
                print(f"   Output: {OCR_TEXT}")

            except Exception as e:
                print(f"\n OCR failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n Please check if model exists: {OCR_MODEL_PATH}")
                if not OCR_TEXT.exists():
                    print(f"   No cached OCR result, skipping objective grading...")
                    OCR_TEXT = None

    print(f"\n{'='*80}")
    print("[3/4] 客观题评分（规则匹配）")
    print(f"{'='*80}")

    from utils.parse_objective_answers import parse_objective_answers_from_ocr, grade_objective_questions

    ANSWER_KEY = RUBRIC_DIR / "answer_key.json"
    OBJECTIVE_OUTPUT = OUTPUT_BASE / "objective_grading.json"

    if OCR_TEXT and OCR_TEXT.exists() and ANSWER_KEY.exists():
        try:
            student_answers = parse_objective_answers_from_ocr(OCR_TEXT, ANSWER_KEY)

            with open(ANSWER_KEY, 'r', encoding='utf-8') as f:
                answer_key = json.load(f)

            objective_result = grade_objective_questions(student_answers, answer_key, verbose=True)

            OBJECTIVE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            with open(OBJECTIVE_OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(objective_result, f, ensure_ascii=False, indent=2)

            print(f"\n 客观题评分完成")
            print(f"   得分: {objective_result['total_score']}/{objective_result['max_score']}")
            print(f"   输出: {OBJECTIVE_OUTPUT}")
        except Exception as e:
            print(f"\n 客观题评分失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n 跳过客观题评分（缺少OCR文本或答案key）")
        if not OCR_TEXT.exists():
            print(f"   OCR文本不存在: {OCR_TEXT}")
        if not ANSWER_KEY.exists():
            print(f"   答案key不存在: {ANSWER_KEY}")

    if SKIP_SUBJECTIVE:
        print(f"\n{'='*80}")
        print("[4/4] 主观题评分 - 已跳过")
        print(f"{'='*80}")
        print(f"\n 设置 SKIP_SUBJECTIVE=False 以启用VLM主观题评分")
    else:
        print(f"\n{'='*80}")
        print("[4/4] 主观题评分（VLM）")
        print(f"{'='*80}")

        from utils.grade_with_vlm import grade_with_vlm

        try:
            grade_with_vlm(
                processed_json=processed_json,
                rubric_dir=RUBRIC_DIR,
                output_json=GRADING_OUTPUT,
                vlm_model='qwen-vl',
                api_url=VLM_API_URL
            )
            print(f"\n 主观题评分完成")
            print(f"   输出: {GRADING_OUTPUT}")
            print(f"   详细报告: {GRADING_OUTPUT.parent / 'vlm_details'}")
        except Exception as e:
            print(f"\n 主观题评分失败: {e}")
            import traceback
            traceback.print_exc()
            return

    print(f"\n{'='*80}")
    print("[5/4] 评分汇总")
    print(f"{'='*80}")

    if OCR_TEXT and OCR_TEXT.exists() and ANSWER_KEY.exists():
        print(f"\n客观题结果:")
        print(f"  输出: {OBJECTIVE_OUTPUT}")

    if not SKIP_SUBJECTIVE:
        print(f"\n主观题结果:")
        print(f"  评分JSON: {GRADING_OUTPUT}")
        print(f"  详细报告: {GRADING_OUTPUT.parent / 'vlm_details'}")

    print(f"\n{'='*80}")
    print(" 评分完成")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
