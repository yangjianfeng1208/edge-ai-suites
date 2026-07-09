"""
Export OCR Regions to Readable Text

Exports all OCR regions from all pages in a simple format:
{page} {type}: {content}
"""
import json
from pathlib import Path


def export_ocr_regions(ocr_dir: Path, output_txt: Path):
    """
    Export all OCR regions to a simple text file

    Args:
        ocr_dir: Directory containing page_X_ocr.json files
        output_txt: Output text file path
    """
    print(f"\n{'='*80}")
    print("Exporting OCR Regions")
    print(f"{'='*80}")

    # Find all page OCR files
    ocr_files = sorted(ocr_dir.glob("page_*_ocr.json"))

    if not ocr_files:
        print(f"  No OCR files found in {ocr_dir}")
        return

    print(f"  Found {len(ocr_files)} pages")

    total_regions = 0

    # Create output directory if it doesn't exist
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("page | type | content\n")
        f.write("-" * 100 + "\n")

        for ocr_file in ocr_files:
            # Extract page number from filename: page_1_ocr.json -> 1
            page_num = int(ocr_file.stem.split('_')[1])

            # Load OCR results
            with open(ocr_file, 'r', encoding='utf-8') as rf:
                ocr_data = json.load(rf)

            results = ocr_data.get('results', [])

            for result in results:
                region_type = result.get('type', 'unknown')
                content = result.get('content', '').strip()

                # Replace newlines with space for single-line output
                content = content.replace('\n', ' ')

                # Limit content length for readability
                if len(content) > 200:
                    content = content[:200] + "..."

                f.write(f"{page_num} | {region_type} | {content}\n")

                total_regions += 1

        f.write("-" * 100 + "\n")
        f.write(f"Total: {len(ocr_files)} pages, {total_regions} regions\n")

    print(f"\n  Exported {total_regions} regions from {len(ocr_files)} pages")
    print(f"  Output: {output_txt}")
