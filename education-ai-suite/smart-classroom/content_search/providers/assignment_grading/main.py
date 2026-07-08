from pathlib import Path
import sys
import json
import yaml
from utils.pdf_processor import convert_pdf_to_images, image_to_bytes

BASE_DIR = Path(__file__).parent


def main():
    print("="*80)
    print("Automated Grading System")
    print("="*80)

    config_path = BASE_DIR / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    SKIP_SUBJECTIVE = config['grading']['skip_subjective']
    SKIP_YOLO = config['grading']['skip_yolo_detection']
    USE_CACHED_OCR = config['ocr']['use_cached']
    SAVE_PDF_IMAGES = config['ocr']['save_pdf_images']
    PDF_DPI = config['ocr']['pdf_dpi']
    OCR_MAX_PIXELS = config['ocr']['max_pixels']
    OCR_MAX_TOKENS = config['ocr']['max_tokens']

    JPEG_QUALITY = config['ocr'].get('jpeg_quality', 85)
    RESIZE_RATIO = config['ocr'].get('resize_ratio')
    MAX_IMAGE_WIDTH = config['ocr'].get('max_image_width')
    MAX_IMAGE_HEIGHT = config['ocr'].get('max_image_height')
    MAX_IMAGE_PIXELS = config['ocr'].get('max_image_pixels')
    ENHANCE_CONTRAST = config['ocr'].get('enhance_contrast')
    ENHANCE_SHARPNESS = config['ocr'].get('enhance_sharpness')

    YOLO_CONF = config['detection']['yolo_conf']
    YOLO_IOU = config['detection']['yolo_iou']
    PDF_RENDER_DPI = config['detection']['pdf_render_dpi']

    DETECTION_JSON = BASE_DIR / config['detection']['json_path']

    if not DETECTION_JSON.exists():
        print(f"Error: Detection JSON not found: {DETECTION_JSON}")
        return

    print(f"\nLoading configuration...")
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
        print(f"Error: rubric_guided_scoring directory not found")
        return

    exam_name = exam_root.name
    OUTPUT_BASE = BASE_DIR / f"outputs/{exam_name}"
    PROCESSED_OUTPUT_DIR = OUTPUT_BASE / "processed_answers"
    RUBRIC_DIR = exam_root / "rubric_guided_scoring"
    GRADING_OUTPUT = OUTPUT_BASE / "vlm_grading" / f"{exam_name}_grading.json"
    VLM_API_URL = config['vlm_service']['base_url']
    OCR_API_URL = config['ocr_service']['base_url']

    print(f"  PDF: {PDF_PATH.name}")
    print(f"  YOLO: {YOLO_MODEL.name}")
    print(f"  Output: {OUTPUT_BASE}")

    if not PDF_PATH.exists():
        print(f"\nError: PDF not found: {PDF_PATH}")
        return

    if not YOLO_MODEL.exists():
        print(f"\nError: YOLO model not found: {YOLO_MODEL}")
        print(f"  Please run training script to generate model")
        return

    step = 0
    total = 3
    if not SKIP_YOLO:
        total += 1
    if not SKIP_SUBJECTIVE:
        total += 1

    if not SKIP_YOLO:
        step += 1
        print(f"\n{'='*80}")
        print(f"[Step {step}/{total}] YOLO Detection")
        print(f"{'='*80}")

        from ultralytics import YOLO
        from utils.pdf_utils import render_pdf_to_images

        try:
            print(f"\nLoading YOLO model...")
            model = YOLO(str(YOLO_MODEL))

            print(f"Rendering PDF...")
            pages = render_pdf_to_images(PDF_PATH, dpi=PDF_RENDER_DPI)
            print(f"Total pages: {len(pages)}")

            print(f"\nStarting detection...")
            all_detections = {}

            for page_data in pages:
                page_num = page_data['page_num']
                image = page_data['image']

                results = model(image, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
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
                print(f"  Page {page_num}: {len(page_detections)} regions")

            print(f"\nCompleted: {sum(len(v) for v in all_detections.values())} regions detected")

        except Exception as e:
            print(f"\nError: YOLO detection failed: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"\nUsing cached detection: {DETECTION_JSON}")

    step += 1
    print(f"\n{'='*80}")
    print(f"[Step {step}/{total}] Extract Answer Regions")
    print(f"{'='*80}")

    from utils.process_adjusted_detections import process_adjusted_detections

    try:
        processed_data = process_adjusted_detections(DETECTION_JSON, PROCESSED_OUTPUT_DIR, pdf_path=PDF_PATH)
        processed_json = PROCESSED_OUTPUT_DIR / f"{processed_data['student_id']}_processed.json"
        print(f"\nCompleted: {processed_data['total_answer_blocks']} regions extracted")
        print(f"  Output: {processed_json}")
    except Exception as e:
        print(f"\nError: Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    step += 1
    print(f"\n{'='*80}")
    print(f"[Step {step}/{total}] OCR Recognition")
    print(f"{'='*80}")

    OCR_TEXT = OUTPUT_BASE / "ocr_text" / f"{exam_name}_ocr.txt"

    if USE_CACHED_OCR and OCR_TEXT.exists():
        print(f"\n Using cached OCR result: {OCR_TEXT}")
    else:
        try:
            import requests
            from pdf2image import convert_from_path
            import io

            print(f"\n Checking OCR server health...")
            print(f"   URL: {OCR_API_URL}")
            health_response = requests.get(f"{OCR_API_URL}/health", timeout=5)
            if health_response.status_code != 200:
                raise Exception(f"OCR server not healthy: {health_response.text}")

            health_data = health_response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   Device: {health_data.get('device')}")

            save_dir = OUTPUT_BASE / "pdf_images" if SAVE_PDF_IMAGES else None
            images_gray = convert_pdf_to_images(
                pdf_path=PDF_PATH,
                dpi=PDF_DPI,
                grayscale=True,
                save_dir=save_dir,
                jpeg_quality=JPEG_QUALITY,
                max_pixels=MAX_IMAGE_PIXELS,
                max_width=MAX_IMAGE_WIDTH,
                max_height=MAX_IMAGE_HEIGHT,
                resize_ratio=RESIZE_RATIO,
                enhance_contrast=ENHANCE_CONTRAST,
                enhance_sharpness=ENHANCE_SHARPNESS
            )

            print(f"\n Running OCR via API on each page...")
            print(f"   Max pixels: {OCR_MAX_PIXELS:,}")
            print(f"   Max tokens: {OCR_MAX_TOKENS}")
            results = []
            total_inference_time = 0

            for idx, img_gray in enumerate(images_gray, 1):
                print(f"   Processing page {idx}/{len(images_gray)}...")

                img_buffer = io.BytesIO()
                img_gray.save(img_buffer, format='JPEG', quality=85, optimize=True)
                img_buffer.seek(0)

                files = {'file': (f'page_{idx}.jpg', img_buffer, 'image/jpeg')}
                data = {
                    'task': 'ocr',
                    'max_new_tokens': OCR_MAX_TOKENS,
                    'max_pixels': OCR_MAX_PIXELS
                }

                response = requests.post(
                    f"{OCR_API_URL}/ocr/file",
                    files=files,
                    data=data,
                    timeout=120
                )

                if response.status_code != 200:
                    raise Exception(f"OCR API failed for page {idx}: {response.text}")

                result = response.json()
                if not result.get('success'):
                    raise Exception(f"OCR failed for page {idx}: {result.get('error')}")

                text = result.get('text', '')
                inference_time = result.get('inference_time', 0)
                total_inference_time += inference_time

                print(f"     Inference time: {inference_time:.2f}s")

                page_result = f"{'='*80}\nPage {idx}\n{'='*80}\n{text}\n\n"
                results.append(page_result)

            full_text = "\n".join(results)

            OCR_TEXT.parent.mkdir(parents=True, exist_ok=True)
            with open(OCR_TEXT, 'w', encoding='utf-8') as f:
                f.write(full_text)

            print(f"\nCompleted: {len(images_gray)} pages processed in {total_inference_time:.1f}s")
            print(f"  Avg: {total_inference_time/len(images_gray):.1f}s/page")
            print(f"  Output: {OCR_TEXT}")

        except requests.exceptions.ConnectionError:
            print(f"\nError: OCR server not running at {OCR_API_URL}")
            print(f"  Start with: python ocr_services/paddleocr_vl_server.py")
            if not OCR_TEXT.exists():
                print(f"  No cached OCR, skipping objective grading")
                OCR_TEXT = None
        except Exception as e:
            print(f"\nError: OCR failed: {e}")
            import traceback
            traceback.print_exc()
            if not OCR_TEXT.exists():
                print(f"  No cached OCR, skipping objective grading")
                OCR_TEXT = None

    step += 1
    print(f"\n{'='*80}")
    print(f"[Step {step}/{total}] Grade Objective Questions")
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

            print(f"\nCompleted: {objective_result['total_score']}/{objective_result['max_score']} points")
            print(f"  Output: {OBJECTIVE_OUTPUT}")
        except Exception as e:
            print(f"\nError: Grading failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipped: Missing OCR text or answer key")
        if not OCR_TEXT or not OCR_TEXT.exists():
            print(f"  OCR text not found")
        if not ANSWER_KEY.exists():
            print(f"  Answer key not found: {ANSWER_KEY}")

    if SKIP_SUBJECTIVE:
        print(f"\nSubjective grading: Skipped (disabled in config.yaml)")
    else:
        step += 1
        print(f"\n{'='*80}")
        print(f"[Step {step}/{total}] Grade Subjective Questions (VLM)")
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
            print(f"\nCompleted")
            print(f"  Output: {GRADING_OUTPUT}")
            print(f"  Details: {GRADING_OUTPUT.parent / 'vlm_details'}")
        except Exception as e:
            print(f"\nError: VLM grading failed: {e}")
            import traceback
            traceback.print_exc()
            return

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    if OCR_TEXT and OCR_TEXT.exists() and ANSWER_KEY.exists():
        print(f"\nObjective: {OBJECTIVE_OUTPUT}")

    if not SKIP_SUBJECTIVE:
        print(f"Subjective: {GRADING_OUTPUT}")

    print(f"\n{'='*80}")
    print("Completed")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
