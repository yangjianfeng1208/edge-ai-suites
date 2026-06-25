import json
from pathlib import Path
import cv2
import sys

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))


def process_adjusted_detections(detection_json, output_dir, pdf_path=None):
    print(f"\n{'='*80}")
    print("Processing adjusted YOLO detection results")
    print(f"{'='*80}")

    print(f"\n[1/5] Loading detection results...")
    with open(detection_json, 'r', encoding='utf-8') as f:
        detection_data = json.load(f)

    print(f"  PDF in detection JSON: {detection_data['source_pdf']}")
    print(f"  Total pages: {detection_data['total_pages']}")

    total_detections = sum(len(v) for v in detection_data['detections'].values())
    print(f"  Total detections: {total_detections}")

    print(f"\n[2/5] Rendering PDF...")
    from utils.pdf_utils import render_pdf_to_images

    if pdf_path is None:
        pdf_path = Path(detection_data['source_pdf'])
    else:
        pdf_path = Path(pdf_path)

    print(f"  Actual PDF used: {pdf_path}")
    pages = render_pdf_to_images(pdf_path, dpi=300)
    pages_dict = {p['page_num']: p for p in pages}

    print(f"\n[3/5] Extracting Answer_Block...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_blocks = []

    for page_num_str, detections in detection_data['detections'].items():
        page_num = int(page_num_str)
        page_img = pages_dict[page_num]['image']

        answer_dets = [d for d in detections if d['class_name'] == 'Answer_Block']

        answer_dets_sorted = sorted(answer_dets, key=lambda x: x['bbox'][1])

        for idx, det in enumerate(answer_dets_sorted, start=1):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            answer_img = page_img[y1:y2, x1:x2]

            question_id = f"Q{page_num}_{idx}"
            img_path = output_dir / f"{question_id}.jpg"
            cv2.imwrite(str(img_path), answer_img)

            print(f"  {question_id} (page {page_num}) - bbox={bbox}")

            answer_blocks.append({
                'question_id': question_id,
                'page': page_num,
                'bbox': bbox,
                'confidence': det['confidence'],
                'image_path': str(img_path)
            })

    print(f"\n[4/5] OCR recognition...")
    print("  Skipping OCR (PyTorch environment issue), saving images only")

    for block in answer_blocks:
        block['student_answer'] = ""
        block['ocr_skipped'] = True

    print(f"\n[5/5] Saving results...")

    output_data = {
        'student_id': pdf_path.parent.name,
        'source_pdf': str(pdf_path),
        'total_pages': detection_data['total_pages'],
        'total_answer_blocks': len(answer_blocks),
        'answer_blocks': answer_blocks,
        'all_detections': detection_data['detections']
    }

    output_json = output_dir / f"{output_data['student_id']}_processed.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print("Processing completed")
    print(f"{'='*80}")
    print(f"Student ID: {output_data['student_id']}")
    print(f"Answer regions: {len(answer_blocks)}")
    print(f"Output JSON: {output_json}")
    print(f"\nNext step: Send to VLM grading with rubric")

    return output_data


def main():
    BASE_DIR = Path("C:/Users/user/jianfeng/EDU-AI/PR/edge-ai-my-fork/education-ai-suite/smart-classroom/content_search/providers/assignment_grading")
    DETECTION_JSON = BASE_DIR / "test_data/2025_sh_zhongkao_math/yolo_detections.json"
    OUTPUT_DIR = BASE_DIR / "outputs/processed_answers"

    if not DETECTION_JSON.exists():
        print(f"Error: Detection results not found: {DETECTION_JSON}")
        return

    process_adjusted_detections(DETECTION_JSON, OUTPUT_DIR)


if __name__ == '__main__':
    main()
