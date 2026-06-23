from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent


def main():
    print("="*80)
    print("自动评分系统")
    print("="*80)

    PDF_PATH = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/papers/xiaoming/yuwen_paper.pdf"
    YOLO_MODEL = BASE_DIR / "models/yolo_hilex/yolo11n_hilex/weights/best_openvino_model"
    YOLO_DETECTION_JSON = BASE_DIR / "outputs/yolo_detections/xiaoming_yolo_detections.json"
    ADJUSTED_DETECTION_JSON = BASE_DIR / "outputs/yolo_detections/adjusted_yolo_detections.json"
    PROCESSED_OUTPUT_DIR = BASE_DIR / "outputs/processed_answers"
    RUBRIC_DIR = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/rubric_guided_scoring"
    GRADING_OUTPUT = BASE_DIR / "outputs/vlm_grading/xiaoming_grading.json"
    VLM_API_URL = "http://127.0.0.1:9900"

    if not PDF_PATH.exists():
        print(f" 错误: PDF不存在: {PDF_PATH}")
        return

    if not YOLO_MODEL.exists():
        print(f" 错误: YOLO模型不存在: {YOLO_MODEL}")
        print(f"   请运行训练脚本生成模型")
        return

    print(f"\n 找到试卷PDF: {PDF_PATH}")
    print(f" 找到YOLO模型: {YOLO_MODEL}")

    print(f"\n{'='*80}")
    print("[0/3] YOLO检测答题区域")
    print(f"{'='*80}")

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

            output_data = {
                'source_pdf': str(PDF_PATH),
                'yolo_model': str(YOLO_MODEL),
                'total_pages': len(pages),
                'page_dimensions': {
                    p['page_num']: {'width': p['width'], 'height': p['height']}
                    for p in pages
                },
                'detections': all_detections
            }

            YOLO_DETECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(YOLO_DETECTION_JSON, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"\n YOLO检测完成")
            print(f"   原始检测结果: {YOLO_DETECTION_JSON}")
            print(f"   总检测数: {sum(len(v) for v in all_detections.values())}")

        except Exception as e:
            print(f"\n YOLO检测失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        pass

    if YOLO_DETECTION_JSON.exists() and ADJUSTED_DETECTION_JSON.exists():
        try:
            from utils.detection_compare import compare_detections
            comparison_dir = BASE_DIR / "outputs/detection_comparison"
            compare_detections(YOLO_DETECTION_JSON, ADJUSTED_DETECTION_JSON, PDF_PATH, comparison_dir)
        except Exception as e:
            print(f"\n  对比图生成失败: {e}")
            import traceback
            traceback.print_exc()

    if ADJUSTED_DETECTION_JSON.exists():
        DETECTION_JSON = ADJUSTED_DETECTION_JSON
        print(f"\n 使用手动调整后的检测结果: {DETECTION_JSON}")
    elif YOLO_DETECTION_JSON.exists():
        DETECTION_JSON = YOLO_DETECTION_JSON
        print(f"\n 使用YOLO原始检测结果: {DETECTION_JSON}")
    else:
        print(f"\n 错误: 没有可用的检测结果")
        return

    print(f"\n{'='*80}")
    print("[1/3] 提取答题区域")
    print(f"{'='*80}")

    from utils.process_adjusted_detections import process_adjusted_detections

    try:
        processed_data = process_adjusted_detections(DETECTION_JSON, PROCESSED_OUTPUT_DIR)
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
    print("[2/3] VLM评分")
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
        print(f"\n 评分完成")
        print(f"   输出: {GRADING_OUTPUT}")
        print(f"   详细报告: {GRADING_OUTPUT.parent / 'vlm_details'}")
    except Exception as e:
        print(f"\n VLM评分失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{'='*80}")
    print(" 自动评分完成")
    print(f"{'='*80}")
    print(f"\n查看结果:")
    print(f"  评分JSON: {GRADING_OUTPUT}")
    print(f"  详细报告: {GRADING_OUTPUT.parent / 'vlm_details'}")


if __name__ == '__main__':
    main()
