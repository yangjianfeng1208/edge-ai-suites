from pathlib import Path
import argparse
from PIL import Image
import openvino as ov
from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM
import time


class PaddleOCRVLService:
    def __init__(self, model_path, device="CPU"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = device
        self.core = ov.Core()
        self.perf_stats = {
            'model_load_time': 0,
            'page_times': []
        }

        print(f"Loading PaddleOCR-VL model from {model_path}...")
        print(f"Target device: {self.device}")

        load_start = time.time()
        self.model = OVPaddleOCRVLForCausalLM(
            core=self.core,
            ov_model_path=str(self.model_path),
            device=self.device,
            llm_int4_compress=False,
            llm_int8_compress=True,
            vision_int8_quant=False,
            llm_int8_quant=True,
            llm_infer_list=[],
            vision_infer=[]
        )
        self.perf_stats['model_load_time'] = time.time() - load_start
        print(f"Model loaded successfully on {self.device} in {self.perf_stats['model_load_time']:.2f}s")

    def ocr_image(self, image_input, task="ocr", max_new_tokens=512, max_pixels=None):
        infer_start = time.time()

        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image")

        PROMPTS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPTS.get(task, "OCR:")},
                ],
            }
        ]

        generation_config = {
            "bos_token_id": self.model.tokenizer.bos_token_id,
            "eos_token_id": self.model.tokenizer.eos_token_id,
            "pad_token_id": self.model.tokenizer.pad_token_id,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }

        image_processor_config = None
        if max_pixels is not None:
            image_processor_config = {"max_pixels": max_pixels}

        response, history = self.model.chat(
            messages=messages,
            generation_config=generation_config,
            image_processor_config=image_processor_config
        )

        infer_time = time.time() - infer_start
        self.perf_stats['page_times'].append(infer_time)

        return response

    def get_perf_stats(self):
        return self.perf_stats


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL OCR Service")
    parser.add_argument("--model", required=True, help="Path to OpenVINO model directory")
    parser.add_argument("--image", required=True, help="Path to input image or PDF")
    parser.add_argument("--task", default="ocr", choices=["ocr", "table", "formula", "chart"], help="Recognition task type")
    parser.add_argument("--device", default="CPU", help="OpenVINO device")
    parser.add_argument("--output", help="Output text file path")

    args = parser.parse_args()

    service = PaddleOCRVLService(model_path=args.model, device=args.device)

    input_path = Path(args.image)

    if input_path.suffix.lower() == '.pdf':
        from pdf2image import convert_from_path
        print(f"Converting PDF to images: {input_path}")
        images = convert_from_path(input_path, dpi=300)

        results = []
        for idx, img in enumerate(images, 1):
            print(f"\nProcessing page {idx}/{len(images)}...")
            text = service.ocr_image(img, task=args.task)
            results.append(f"=== Page {idx} ===\n{text}\n")

        full_text = "\n".join(results)
    else:
        print(f"Processing image: {input_path}")
        full_text = service.ocr_image(input_path, task=args.task)

    print("\n" + "="*80)
    print("OCR Result:")
    print("="*80)
    print(full_text)
    print("="*80)

    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    main()
