from ultralytics import YOLO
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/yolo_hilex/yolo11n_hilex/weights/best.pt"


def main():
    if not MODEL_PATH.exists():
        print(f"模型不存在: {MODEL_PATH}")
        print("请先训练模型: python train_yolo_hilex.py")
        return

    print(f"加载模型: {MODEL_PATH}\n")
    model = YOLO(str(MODEL_PATH))

    print("=" * 60)
    print("方法1: 在HiLEx测试集上评估")
    print("=" * 60)

    from utils.pdf_utils import render_pdf_to_images

    hilex_data_yaml = BASE_DIR / "models/yolo_hilex/data.yaml"

    if hilex_data_yaml.exists():
        print("\n评估中...")
        metrics = model.val(data=str(hilex_data_yaml), split='test')

        print(f"\n测试集结果:")
        print(f"  mAP@50:    {metrics.box.map50:.3f}")
        print(f"  mAP@50-95: {metrics.box.map:.3f}")
        print(f"  Precision: {metrics.box.mp:.3f}")
        print(f"  Recall:    {metrics.box.mr:.3f}")

    print("\n" + "=" * 60)
    print("方法2: 在中考试卷上测试")
    print("=" * 60)

    pdf_path = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/papers/xiaoming/yuwen_paper.pdf"

    print("\n渲染PDF第3页...")
    pages = render_pdf_to_images(pdf_path, dpi=300)
    test_image = pages[2]['image']

    print("推理中...")
    results = model(test_image, conf=0.3, iou=0.5, verbose=False)

    detections = results[0].boxes
    print(f"\n检测到 {len(detections)} 个区域:\n")

    for i, box in enumerate(detections[:10]):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        print(f"  {i+1:2d}. {cls_name:25s} conf={conf:.3f}")

    output_path = BASE_DIR / "validation_result.jpg"
    img_vis = results[0].plot()
    cv2.imwrite(str(output_path), img_vis)

    print(f"\n可视化结果: {output_path}")

    print("\n" + "=" * 60)
    print("检测类别说明:")
    print("=" * 60)
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")


if __name__ == '__main__':
    main()
