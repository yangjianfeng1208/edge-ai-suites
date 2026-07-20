from pathlib import Path
import time
import cv2
import numpy as np
import openvino as ov
from PIL import Image, ImageDraw, ImageFont
import json
import os
import yaml


class LayoutDetectionService:
    """PP-DocLayoutV2 document layout detection service using OpenVINO"""

    LABEL_LIST = [
        "abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
        "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
        "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
        "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"
    ]

    def __init__(self, model_path, device="GPU", precision="fp32", threshold=0.5):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = device
        self.threshold = threshold
        self.perf_stats = {
            'model_load_time': 0,
            'inference_times': []
        }

        print(f"Loading PP-DocLayoutV2 model from {model_path}...")
        print(f"Target device: {self.device}")
        print(f"Precision: {precision}")

        load_start = time.time()
        self._load_model(precision)
        self.perf_stats['model_load_time'] = time.time() - load_start
        print(f"Model loaded successfully on {self.device} in {self.perf_stats['model_load_time']:.2f}s")

    def _load_model(self, precision):
        # Find model file based on precision
        precision_map = {
            "fp16": "pp_doclayoutv2_f16.xml",
            "fp32": "pp_doclayoutv2_f32.xml",
            "combined_fp16": "pp_doclayoutv2_f16_combined.xml",
            "combined_fp32": "pp_doclayoutv2_f32_combined.xml",
        }

        if self.model_path.is_dir():
            model_filename = precision_map.get(precision)
            if model_filename:
                model_file = self.model_path / model_filename
                if not model_file.exists():
                    # Try to find any .xml file
                    xml_files = list(self.model_path.glob("*.xml"))
                    if xml_files:
                        model_file = xml_files[0]
                        print(f"  Using found model: {model_file.name}")
            else:
                xml_files = list(self.model_path.glob("*.xml"))
                if xml_files:
                    model_file = xml_files[0]
                else:
                    raise FileNotFoundError(f"No .xml files found in {self.model_path}")
        else:
            model_file = self.model_path

        # Initialize OpenVINO
        self.core = ov.Core()
        model = self.core.read_model(str(model_file))

        # Merge preprocessing
        prep = ov.preprocess.PrePostProcessor(model)
        prep.input("image").tensor().set_layout(ov.Layout("NCHW"))
        prep.input("image").preprocess().scale([255, 255, 255])

        if self.device == "NPU":
            prep.input("im_shape").model().set_layout(ov.Layout('N...'))
            prep.input("scale_factor").model().set_layout(ov.Layout('N...'))
            prep.input("image").model().set_layout(ov.Layout('NCHW'))

        model = prep.build()

        # Set batch for NPU after build
        if self.device == "NPU":
            for param in model.get_parameters():
                param_name = param.get_friendly_name()
                if param_name == "im_shape":
                    param.set_layout(ov.Layout('NC'))
                elif param_name == "scale_factor":
                    param.set_layout(ov.Layout('NC'))
                elif param_name == "image":
                    param.set_layout(ov.Layout('NCHW'))
            ov.set_batch(model, 1)

        self.compiled_model = self.core.compile_model(model, self.device)

    def _preprocess_image(self, image, target_size=(800, 800)):
        """Preprocess image for layout detection"""
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = target_size

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        input_blob = resized.astype(np.float32)
        input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]

        return input_blob, orig_h, orig_w

    def _postprocess(self, output, orig_h, orig_w):
        """Postprocess model output to extract bounding boxes"""
        out0 = np.array(output[0]) if len(output) > 0 else None
        out1 = np.array(output[1]) if len(output) > 1 else None

        if out0 is None or out0.size == 0:
            return []

        # Handle Paddle NMS output format [N, 6] or [N, 7]
        if out0.ndim == 2 and out0.shape[1] >= 6:
            if out1 is not None and out1.size > 0:
                num = int(out1.reshape(-1)[0])
                num = max(0, min(num, out0.shape[0]))
            else:
                num = out0.shape[0]

            det = out0[:num]
            if det.size == 0:
                return []

            # Extract class, score, coordinates
            # Output row format: [cls_id, score, x1, y1, x2, y2, (optional extra)]
            cls = det[:, 0]
            score = det[:, 1]
            coords = det[:, 2:6].copy()

            # Normalize coordinates if needed
            if np.max(coords) <= 2.0:
                coords[:, 0] *= float(orig_w)
                coords[:, 2] *= float(orig_w)
                coords[:, 1] *= float(orig_h)
                coords[:, 3] *= float(orig_h)

            # Filter by threshold
            mask = score > self.threshold
            cls = cls[mask]
            score = score[mask]
            coords = coords[mask]

            # Build result list
            results = []
            for i in range(len(cls)):
                cls_id = int(cls[i])
                if cls_id >= 0 and cls_id < len(self.LABEL_LIST):
                    xmin, ymin, xmax, ymax = coords[i]
                    xmin = max(0, min(xmin, orig_w))
                    ymin = max(0, min(ymin, orig_h))
                    xmax = max(0, min(xmax, orig_w))
                    ymax = max(0, min(ymax, orig_h))

                    if xmax > xmin and ymax > ymin:
                        results.append({
                            "cls_id": cls_id,
                            "label": self.LABEL_LIST[cls_id],
                            "score": float(score[i]),
                            "coordinate": [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })

            return results

        return []

    def detect(self, image_input):
        """
        Detect layout regions in an image

        Args:
            image_input: PIL Image or file path

        Returns:
            dict with keys: boxes (list of detection dicts), inference_time (float)
        """
        infer_start = time.time()

        # Load image
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_input}")
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("image_input must be a file path or PIL Image")

        orig_h, orig_w = image.shape[:2]

        # Preprocess
        input_blob, orig_h, orig_w = self._preprocess_image(image)

        # Prepare inputs
        target_h, target_w = 800, 800
        input_tensors = self.compiled_model.inputs
        input_data = {}
        for inp in input_tensors:
            inp_name = inp.get_any_name()
            if inp_name == "im_shape":
                input_data[inp_name] = np.array([[target_h, target_w]], dtype=np.float32)
            elif inp_name == "image":
                input_data[inp_name] = input_blob
            elif inp_name == "scale_factor":
                input_data[inp_name] = np.array(
                    [[target_h / orig_h, target_w / orig_w]], dtype=np.float32
                )

        # Inference
        result = self.compiled_model(input_data)

        # Extract outputs
        output = []
        for out in self.compiled_model.outputs:
            output.append(result[out].data)

        # Postprocess
        boxes = self._postprocess(output, orig_h, orig_w)

        infer_time = time.time() - infer_start
        self.perf_stats['inference_times'].append(infer_time)

        return {
            "boxes": boxes,
            "inference_time": infer_time,
            "image_size": (orig_w, orig_h)
        }

    def visualize(self, image_path, boxes, output_path):
        """Draw bounding boxes on image and save"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Try to load font
        try:
            font_size = int(0.018 * img_pil.width) + 2
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ]
            font = None
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        font = ImageFont.truetype(fp, font_size)
                        break
                    except:
                        continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Color palette
        colors = [
            (128, 64, 128), (232, 35, 244), (70, 70, 70), (156, 102, 102),
            (0, 220, 220), (35, 142, 107), (152, 251, 152), (180, 130, 70),
            (0, 0, 255), (142, 0, 0), (230, 0, 0)
        ]

        draw_thickness = max(2, int(max(img_pil.size) * 0.002))

        for i, box in enumerate(boxes):
            label = box["label"]
            score = box["score"]
            xmin, ymin, xmax, ymax = box["coordinate"]

            color = colors[box["cls_id"] % len(colors)]

            # Draw rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=draw_thickness)

            # Draw label
            text = f"{label} {score:.2f}"
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                tw, th = draw.textsize(text, font=font)

            draw.rectangle([(xmin, ymin - th - 2), (xmin + tw + 4, ymin)], fill=color)
            draw.text((xmin + 2, ymin - th - 2), text, fill=(255, 255, 255), font=font)

        img_pil.save(output_path)

    def get_perf_stats(self):
        stats = self.perf_stats.copy()
        if stats['inference_times']:
            stats['avg_inference_time'] = sum(stats['inference_times']) / len(stats['inference_times'])
            stats['total_images'] = len(stats['inference_times'])
        return stats


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    print("=" * 80)
    print("PP-DocLayoutV2 Layout Detection")
    print("=" * 80)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    layout_config = config.get('layout_detection', {})

    # Get model path (relative to config file)
    model_path = layout_config.get('model_path', '../models/PP-DocLayoutV3-openvino')
    if not Path(model_path).is_absolute():
        model_path = config_path.parent / model_path

    # Get input path (relative to config file)
    input_path_str = layout_config.get('input_path')
    if not input_path_str:
        print("Error: input_path not specified in config.yaml")
        return
    input_path = Path(input_path_str)
    if not input_path.is_absolute():
        input_path = config_path.parent / input_path

    device = layout_config.get('device', 'GPU')
    precision = layout_config.get('precision', 'fp32')
    threshold = layout_config.get('threshold', 0.5)
    output_dir = layout_config.get('output_dir', './layout_output')
    visualize = layout_config.get('visualize', False)

    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print(f"  Precision: {precision}")
    print(f"  Threshold: {threshold}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_dir}")
    print(f"  Visualize: {visualize}")
    print()

    # Initialize service
    service = LayoutDetectionService(
        model_path=model_path,
        device=device,
        precision=precision,
        threshold=threshold
    )

    # Collect image files
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    if input_path.is_file():
        if input_path.suffix.lower() not in image_extensions:
            print(f"Error: Not a supported image format: {input_path}")
            return
        image_files = [input_path]
    else:
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
        image_files = sorted(set(image_files))

    if not image_files:
        print(f"Error: No image files found")
        return

    # Create output directory
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = config_path.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(image_files)} image(s)...")
    print("=" * 80)

    # Process each image
    all_results = []
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_file.name}")

        try:
            result = service.detect(img_file)
            boxes = result["boxes"]
            infer_time = result["inference_time"]

            print(f"  Inference time: {infer_time:.3f}s")
            print(f"  Detected {len(boxes)} regions:")

            # Count by label
            label_counts = {}
            for box in boxes:
                label = box["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            for label, count in sorted(label_counts.items()):
                print(f"    - {label}: {count}")

            # Save JSON
            json_file = output_path / f"{img_file.stem}_layout.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "input_path": str(img_file.absolute()),
                    "image_size": result["image_size"],
                    "boxes": boxes,
                    "inference_time": infer_time
                }, f, indent=2, ensure_ascii=False)

            print(f"  Saved JSON: {json_file.name}")

            # Visualize if requested
            if visualize:
                vis_file = output_path / f"{img_file.stem}_layout.jpg"
                service.visualize(img_file, boxes, vis_file)
                print(f"  Saved visualization: {vis_file.name}")

            img_width, img_height = result["image_size"]
            file_size_kb = img_file.stat().st_size / 1024  # Convert bytes to KB
            all_results.append({
                "file": img_file.name,
                "boxes": len(boxes),
                "time": infer_time,
                "width": img_width,
                "height": img_height,
                "pixels": img_width * img_height,
                "size_kb": file_size_kb
            })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    stats = service.get_perf_stats()
    print(f"Model load time: {stats['model_load_time']:.2f}s")
    print(f"Processed images: {len(all_results)}/{len(image_files)}")

    if all_results:
        total_boxes = sum(r['boxes'] for r in all_results)
        print(f"Total regions detected: {total_boxes}")

    print(f"Results saved to: {output_path.absolute()}")

    # Performance Report (Markdown format)
    if all_results:
        print("\n## Performance Report\n")

        avg_width = sum(r['width'] for r in all_results) / len(all_results)
        avg_height = sum(r['height'] for r in all_results) / len(all_results)
        avg_pixels = sum(r['pixels'] for r in all_results) / len(all_results)
        avg_time = sum(r['time'] for r in all_results) / len(all_results)
        avg_size_kb = sum(r['size_kb'] for r in all_results) / len(all_results)
        avg_boxes = sum(r['boxes'] for r in all_results) / len(all_results)
        total_images = len(all_results)

        print("| Device | Precision | Threshold | Images | Avg Size | Avg FileSize | Avg Pixels | Avg Boxes | Avg Time |")
        print("|--------|-----------|-----------|--------|----------|--------------|------------|-----------|----------|")
        print(f"| {device} | {precision} | {threshold:.2f} | {total_images} | "
              f"{avg_width:.0f}x{avg_height:.0f} | {avg_size_kb:.1f} KB | "
              f"{avg_pixels:,.0f} | {avg_boxes:.1f} | {avg_time:.3f}s |")
        print()


if __name__ == "__main__":
    main()
