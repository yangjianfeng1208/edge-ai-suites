from pathlib import Path
from paddleocr_vl_service import PaddleOCRVLService
import time

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

MODEL_PATH = BASE_DIR / "models/ov_paddleocr-vl-1_6_model"
# TEST_IMAGE = BASE_DIR / "test_data/2025_sh_zhongkao_math/2025_sh_zhongkao_math.pdf"
TEST_IMAGE = BASE_DIR / "test_data/2025_sh_zhongkao_yuwen/papers/xiaoming/yuwen_paper.pdf"

if __name__ == "__main__":
    script_start = time.time()

    print(f"Model path: {MODEL_PATH}")
    print(f"Test image: {TEST_IMAGE}")
    print()

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please convert the model first using the notebook.")
        exit(1)

    if not TEST_IMAGE.exists():
        print(f"Error: Test file not found at {TEST_IMAGE}")
        exit(1)

    service = PaddleOCRVLService(model_path=MODEL_PATH, device="CPU")

    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{TEST_IMAGE.stem}_ocr.txt"

    if TEST_IMAGE.suffix.lower() == '.pdf':
        from pdf2image import convert_from_path

        pdf_convert_start = time.time()
        print(f"Converting PDF to images...")
        images = convert_from_path(TEST_IMAGE, dpi=300)
        pdf_convert_time = time.time() - pdf_convert_start
        print(f"PDF conversion completed in {pdf_convert_time:.2f}s")

        results = []
        for idx, img in enumerate(images, 1):
            print(f"\nProcessing page {idx}/{len(images)}...")
            text = service.ocr_image(img, task="ocr")
            page_result = f"{'='*80}\nPage {idx}\n{'='*80}\n{text}\n\n"
            results.append(page_result)
            print(f"Page {idx} done.")

        full_text = "\n".join(results)
    else:
        pdf_convert_time = 0
        text = service.ocr_image(TEST_IMAGE, task="ocr")
        full_text = f"{'='*80}\nOCR Result\n{'='*80}\n{text}\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)

    total_time = time.time() - script_start
    perf_stats = service.get_perf_stats()
    page_times = perf_stats['page_times']
    model_load_time = perf_stats['model_load_time']
    ocr_time = sum(page_times)

    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    print(f"Device: {service.device}")
    print(f"Model load time: {model_load_time:.2f}s")
    print(f"PDF conversion time: {pdf_convert_time:.2f}s")
    print(f"Total OCR time: {ocr_time:.2f}s")
    print(f"Total execution time: {total_time:.2f}s")
    print()
    print("Per-page OCR time:")
    print(f"{'Page':<10} {'Time (s)':<15}")
    print("-" * 40)
    for page_idx, page_time in enumerate(page_times, 1):
        print(f"{page_idx:<10} {page_time:<15.2f}")

    if len(page_times) > 1:
        avg_time = ocr_time / len(page_times)
        print("-" * 40)
        print(f"{'Average':<10} {avg_time:<15.2f}")

    print()
    print(f"Output saved to: {output_file}")
    print(f"Total pages processed: {len(page_times)}")
    print("="*80)
