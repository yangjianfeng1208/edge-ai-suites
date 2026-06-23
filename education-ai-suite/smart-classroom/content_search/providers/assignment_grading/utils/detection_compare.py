import json
import cv2
import numpy as np
from pathlib import Path


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def match_boxes(yolo_boxes, adjusted_boxes, iou_threshold=0.3):
    matches = []
    used_adjusted = set()

    for i, yolo_box in enumerate(yolo_boxes):
        best_iou = 0
        best_j = -1

        for j, adj_box in enumerate(adjusted_boxes):
            if j in used_adjusted:
                continue
            iou = calculate_iou(yolo_box['bbox'], adj_box['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou > iou_threshold:
            matches.append({
                'yolo_idx': i,
                'adjusted_idx': best_j,
                'iou': best_iou,
                'yolo_box': yolo_box,
                'adjusted_box': adjusted_boxes[best_j]
            })
            used_adjusted.add(best_j)

    unmatched_yolo = [i for i in range(len(yolo_boxes))
                     if i not in [m['yolo_idx'] for m in matches]]
    unmatched_adjusted = [i for i in range(len(adjusted_boxes))
                         if i not in used_adjusted]

    return matches, unmatched_yolo, unmatched_adjusted


def compare_detections(yolo_json, adjusted_json, pdf_path, output_dir):
    from .pdf_utils import render_pdf_to_images

    print(f"\n{'='*80}")
    print("生成检测结果对比图")
    print(f"{'='*80}")

    with open(yolo_json, 'r', encoding='utf-8') as f:
        yolo_data = json.load(f)
    with open(adjusted_json, 'r', encoding='utf-8') as f:
        adjusted_data = json.load(f)

    pages = render_pdf_to_images(pdf_path, dpi=300)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_report = []

    for page_data in pages:
        page_num = page_data['page_num']
        page_img = page_data['image'].copy()

        yolo_dets = [d for d in yolo_data['detections'].get(str(page_num), [])
                    if d['class_name'] == 'Answer_Block']
        adjusted_dets = [d for d in adjusted_data['detections'].get(str(page_num), [])
                        if d['class_name'] == 'Answer_Block']

        matches, unmatched_yolo, unmatched_adjusted = match_boxes(yolo_dets, adjusted_dets)

        page_errors = {
            'page': page_num,
            'yolo_count': len(yolo_dets),
            'adjusted_count': len(adjusted_dets),
            'matched': len(matches),
            'false_positives': len(unmatched_yolo),
            'false_negatives': len(unmatched_adjusted),
            'avg_iou': sum(m['iou'] for m in matches) / len(matches) if matches else 0,
            'matches': matches
        }
        error_report.append(page_errors)

        h, w = page_img.shape[:2]
        yolo_img = page_img.copy()
        adjusted_img = page_img.copy()

        for i, det in enumerate(yolo_dets):
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if i not in unmatched_yolo else (0, 0, 255)
            thickness = 6 if i not in unmatched_yolo else 8
            cv2.rectangle(yolo_img, (x1, y1), (x2, y2), color, thickness)

            match = next((m for m in matches if m['yolo_idx'] == i), None)
            if match:
                label = f"IoU:{match['iou']:.2f}"
                cv2.putText(yolo_img, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(yolo_img, 'FP', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        for i, det in enumerate(adjusted_dets):
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if i not in unmatched_adjusted else (255, 0, 0)
            thickness = 6 if i not in unmatched_adjusted else 8
            cv2.rectangle(adjusted_img, (x1, y1), (x2, y2), color, thickness)

            if i in unmatched_adjusted:
                cv2.putText(adjusted_img, 'FN', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        comparison = np.hstack([yolo_img, adjusted_img])

        title_h = 150
        title_bg = np.ones((title_h, comparison.shape[1], 3), dtype=np.uint8) * 255

        cv2.putText(title_bg, f'Page {page_num} - YOLO: {len(yolo_dets)} boxes',
                   (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
        cv2.putText(title_bg, f'FP: {page_errors["false_positives"]} (Red)',
                   (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.putText(title_bg, f'Adjusted: {len(adjusted_dets)} boxes',
                   (w + 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
        cv2.putText(title_bg, f'FN: {page_errors["false_negatives"]} (Blue) | Avg IoU: {page_errors["avg_iou"]:.3f}',
                   (w + 50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        final_img = np.vstack([title_bg, comparison])

        scale = 0.3
        small_img = cv2.resize(final_img, None, fx=scale, fy=scale)
        output_path = output_dir / f"comparison_page{page_num}.jpg"
        cv2.imwrite(str(output_path), small_img)

        print(f"  第{page_num}页: YOLO {len(yolo_dets)}个 vs 调整后 {len(adjusted_dets)}个")
        print(f"    匹配: {page_errors['matched']}, 误检(FP): {page_errors['false_positives']}, "
              f"漏检(FN): {page_errors['false_negatives']}, 平均IoU: {page_errors['avg_iou']:.3f}")

    total_yolo = sum(e['yolo_count'] for e in error_report)
    total_adjusted = sum(e['adjusted_count'] for e in error_report)
    total_matched = sum(e['matched'] for e in error_report)
    total_fp = sum(e['false_positives'] for e in error_report)
    total_fn = sum(e['false_negatives'] for e in error_report)
    overall_iou = sum(e['avg_iou'] * e['matched'] for e in error_report) / total_matched if total_matched > 0 else 0

    report_file = output_dir / "detection_error_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YOLO检测误差报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"总体统计:\n")
        f.write(f"  YOLO检测总数: {total_yolo}\n")
        f.write(f"  人工标注总数: {total_adjusted}\n")
        f.write(f"  成功匹配: {total_matched}\n")
        f.write(f"  误检(False Positive): {total_fp}\n")
        f.write(f"  漏检(False Negative): {total_fn}\n")
        f.write(f"  平均IoU: {overall_iou:.3f}\n")
        f.write(f"  精确率(Precision): {total_matched/total_yolo if total_yolo > 0 else 0:.3f}\n")
        f.write(f"  召回率(Recall): {total_matched/total_adjusted if total_adjusted > 0 else 0:.3f}\n\n")

        f.write("="*80 + "\n")
        f.write("分页统计:\n")
        f.write("="*80 + "\n\n")

        for err in error_report:
            f.write(f"第{err['page']}页:\n")
            f.write(f"  YOLO: {err['yolo_count']}, 调整后: {err['adjusted_count']}\n")
            f.write(f"  匹配: {err['matched']}, 误检: {err['false_positives']}, 漏检: {err['false_negatives']}\n")
            f.write(f"  平均IoU: {err['avg_iou']:.3f}\n\n")

    print(f"\n对比图生成完成: {output_dir}")
    print(f"\n误差统计:")
    print(f"   总匹配: {total_matched}/{total_adjusted}")
    print(f"   误检: {total_fp}, 漏检: {total_fn}")
    print(f"   平均IoU: {overall_iou:.3f}")
    print(f"   精确率: {total_matched/total_yolo if total_yolo > 0 else 0:.1%}")
    print(f"   召回率: {total_matched/total_adjusted if total_adjusted > 0 else 0:.1%}")
    print(f"   详细报告: {report_file}")

    return {
        'total_yolo': total_yolo,
        'total_adjusted': total_adjusted,
        'total_matched': total_matched,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'overall_iou': overall_iou,
        'precision': total_matched/total_yolo if total_yolo > 0 else 0,
        'recall': total_matched/total_adjusted if total_adjusted > 0 else 0
    }
