"""
Generate Rubric from Blank Exam Paper

Input: Blank exam PDF
Output: VLM-generated answers for each page (intermediate results for debug)

Usage:
    python tools/generate_rubric_from_blank.py --pdf path/to/blank_exam.pdf
"""
import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pdf_processor import render_pdf_to_images
from PIL import Image, ImageEnhance
import io
import base64
import numpy as np


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list:
    """Convert PDF pages to images"""
    print(f"\n[Step 1] Converting PDF to images...")
    print(f"  PDF: {pdf_path}")
    print(f"  DPI: {dpi}")

    pages = render_pdf_to_images(pdf_path, dpi=dpi)
    print(f"  Total pages: {len(pages)}")

    images_dir = output_dir / "page_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    saved_images = []
    for page_data in pages:
        page_num = page_data['page_num']
        image = page_data['image']

        pil_image = Image.fromarray(image)

        orig_width, orig_height = pil_image.width, pil_image.height
        orig_size_mb = image.nbytes / (1024 * 1024)

        print(f"  Page {page_num} original: {orig_width}x{orig_height}, {orig_size_mb:.2f}MB, DPI={dpi}")

        pil_image = pil_image.convert('L')

        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)

        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.5)

        max_size = (1200, 1600)
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        image_path = images_dir / f"page_{page_num}.jpg"
        pil_image.save(image_path, "JPEG", quality=65, optimize=True)

        saved_size_kb = Path(image_path).stat().st_size / 1024

        saved_images.append({
            'page_num': page_num,
            'image_path': str(image_path),
            'pil_image': pil_image
        })

        print(f"  Saved: page_{page_num}.jpg ({pil_image.width}x{pil_image.height}, {saved_size_kb:.1f}KB)")

    return saved_images


def image_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')


def call_vlm_for_page(
    pil_image: Image.Image,
    page_num: int,
    vlm_url: str = "http://127.0.0.1:9900"
) -> dict:
    """Call VLM to analyze exam page and generate answers

    Returns:
        dict with 'prompt' and 'answer' keys
    """
    print(f"\n[Step 2.{page_num}] Calling VLM for page {page_num}...")
    print(f"  VLM URL: {vlm_url}")

    image_base64 = image_to_base64(pil_image)

    prompt = f"""分析这份试卷的第{page_num}页，提取以下信息：

1. 所有题目及题号
2. 题目类型（选择题/填空题/主观题）
3. 客观题：给出正确答案
4. 主观题：给出参考答案和评分标准

按以下格式输出：

题目X：[类型]
答案：[答案内容]
分值：[分数]
备注：[评分要点]

请准确、结构化地输出。"""

    payload = {
        "model": "qwen-vl",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }],
        "max_tokens": 4096,
        "temperature": 0.1
    }

    try:
        import time
        start_time = time.time()

        response = requests.post(
            f"{vlm_url}/v1/chat/completions",
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        elapsed_time = time.time() - start_time

        data = response.json()
        vlm_output = data.get('choices', [{}])[0].get('message', {}).get('content', '')

        print(f"  Inference time: {elapsed_time:.2f}s")
        print(f"  VLM response length: {len(vlm_output)} chars")
        print(f"  Preview: {vlm_output[:200]}...")

        return {
            'prompt': prompt,
            'answer': vlm_output
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'prompt': prompt,
            'answer': f"ERROR: {str(e)}"
        }


def save_single_page_answer(page_num: int, prompt: str, answer_text: str, output_dir: Path):
    """Save single page VLM answer immediately"""
    answers_dir = output_dir / "vlm_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    text_path = answers_dir / f"page_{page_num}_answer.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"Page {page_num} - VLM Generated Answer\n")
        f.write("="*80 + "\n\n")
        f.write("PROMPT SENT TO VLM:\n")
        f.write("-"*80 + "\n")
        f.write(prompt)
        f.write("\n" + "-"*80 + "\n\n")
        f.write("VLM RESPONSE:\n")
        f.write("-"*80 + "\n")
        f.write(answer_text)
        f.write("\n" + "-"*80 + "\n")

    print(f"  Saved: page_{page_num}_answer.txt")


def save_vlm_answers(pages_answers: list, output_dir: Path):
    """Save combined VLM answers"""
    answers_dir = output_dir / "vlm_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    combined_path = answers_dir / "all_answers_combined.txt"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write("VLM Generated Answers - All Pages\n")
        f.write("="*80 + "\n\n")
        f.write("PROMPT TEMPLATE:\n")
        f.write("-"*80 + "\n")
        if pages_answers:
            f.write(pages_answers[0]['prompt'])
        f.write("\n" + "-"*80 + "\n\n")

        for page_data in pages_answers:
            f.write(f"\n{'='*80}\n")
            f.write(f"PAGE {page_data['page_num']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(page_data['vlm_answer'])
            f.write("\n\n")

    print(f"  Combined: all_answers_combined.txt")

    return answers_dir


def save_metadata(
    pdf_path: Path,
    pages_count: int,
    output_dir: Path,
    vlm_url: str
):
    """Save metadata about the generation process"""
    metadata = {
        'input_pdf': str(pdf_path),
        'total_pages': pages_count,
        'generated_at': datetime.now().isoformat(),
        'vlm_url': vlm_url,
        'output_directory': str(output_dir)
    }

    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate rubric from blank exam PDF using VLM')
    parser.add_argument('--pdf', type=str, required=True, help='Path to blank exam PDF')
    parser.add_argument('--vlm-url', type=str, default='http://127.0.0.1:9900', help='VLM service URL')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF rendering')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: outputs/rubric_generation_<timestamp>)')

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 1

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"rubric_generation_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Generate Rubric from Blank Exam Paper")
    print("="*80)
    print(f"Input PDF: {pdf_path}")
    print(f"Output Dir: {output_dir}")
    print(f"VLM URL: {args.vlm_url}")
    print("="*80)

    images = pdf_to_images(pdf_path, output_dir, dpi=args.dpi)

    pages_answers = []
    for img_data in images:
        page_num = img_data['page_num']
        pil_image = img_data['pil_image']

        vlm_result = call_vlm_for_page(pil_image, page_num, args.vlm_url)

        save_single_page_answer(page_num, vlm_result['prompt'], vlm_result['answer'], output_dir)

        pages_answers.append({
            'page_num': page_num,
            'image_path': img_data['image_path'],
            'prompt': vlm_result['prompt'],
            'vlm_answer': vlm_result['answer']
        })

    print(f"\n[Step 3] Generating combined answer file...")
    answers_dir = save_vlm_answers(pages_answers, output_dir)

    save_metadata(pdf_path, len(images), output_dir, args.vlm_url)

    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - page_images/: {len(images)} page images")
    print(f"  - vlm_answers/: {len(pages_answers)} answer files")
    print(f"  - vlm_answers/all_answers_combined.txt: Combined answers")
    print(f"  - generation_metadata.json: Process metadata")
    print("\nNext steps:")
    print("  1. Review vlm_answers/all_answers_combined.txt")
    print("  2. Manually format into answer_key.json")
    print("  3. (Future) Use format_rubric.py to auto-convert")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
