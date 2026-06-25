from ultralytics import YOLO
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent


def detect_regions_with_yolo(pdf_path, yolo_model_path, output_json):
    """
    纯YOLO检测：输入PDF，输出检测框JSON
    不包含OCR，不裁剪图片
    """
    print(f"\n{'='*80}")
    print(f"YOLO检测：{pdf_path.name}")
    print(f"{'='*80}")

    print(f"\n[1/3] 加载YOLO模型...")
    model = YOLO(str(yolo_model_path))
    print(f"  类别: {list(model.names.values())}")

    print(f"\n[2/3] 渲染PDF...")
    from utils.pdf_utils import render_pdf_to_images
    pages = render_pdf_to_images(pdf_path, dpi=300)
    print(f"  共{len(pages)}页")

    print(f"\n[3/3] YOLO检测...")

    all_detections = {}

    for page_data in pages:
        page_num = page_data['page_num']
        image = page_data['image']
        height, width = image.shape[:2]

        results = model(image, conf=0.25, iou=0.5, verbose=False)
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

        print(f"  第{page_num}页: {len(page_detections)}个区域 (尺寸: {width}x{height})")

    output_data = {
        'source_pdf': str(pdf_path),
        'yolo_model': str(yolo_model_path),
        'total_pages': len(pages),
        'page_dimensions': {
            p['page_num']: {'width': p['width'], 'height': p['height']}
            for p in pages
        },
        'detections': all_detections
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print("YOLO检测完成")
    print(f"{'='*80}")
    print(f"输出JSON: {output_json}")
    print(f"总检测数: {sum(len(v) for v in all_detections.values())}")

    return output_data


def main():
    PDF_PATH = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/papers/xiaoming/yuwen_paper.pdf"
    YOLO_MODEL = BASE_DIR / "models/yolo_hilex/yolo11n_hilex/weights/best.pt"
    OUTPUT_JSON = BASE_DIR / "outputs/yolo_detections/xiaoming_yolo_detections.json"

    if not YOLO_MODEL.exists():
        print(f"错误: 训练的模型不存在: {YOLO_MODEL}")
        print("使用通用YOLO11n模型（预训练，自动下载）")
        YOLO_MODEL = "yolo11n.pt"

    if not PDF_PATH.exists():
        print(f"错误: PDF不存在: {PDF_PATH}")
        return

    detect_regions_with_yolo(PDF_PATH, YOLO_MODEL, OUTPUT_JSON)

    print("\n示例JSON结构:")
    print("""
    {
      "source_pdf": "路径",
      "yolo_model": "模型路径",
      "total_pages": 4,
      "page_dimensions": {
        "1": {"width": 2480, "height": 3508}
      },
      "detections": {
        "1": [
          {
            "class_id": 0,
            "class_name": "Answer_Block",
            "confidence": 0.85,
            "bbox": [1500, 3000, 3200, 3500]
          }
        ]
      }
    }
    """)


if __name__ == '__main__':
    main()
