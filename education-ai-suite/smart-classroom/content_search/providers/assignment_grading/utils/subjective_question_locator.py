"""
Subjective Question Locator

Locates complete answer regions for subjective questions by analyzing
question number markers and calculating bounding boxes.

Key features:
- Detects question number markers (19., 20., 21., etc.)
- Calculates vertical slicing based on question boundaries
- Handles cross-page questions
- Generates visualization images for verification
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def extract_question_number(text: str) -> Optional[str]:
    """
    Extract question number from text

    Supports formats:
    - "19." "20." "21."
    - "第19题" "第20题"
    - "19、" "20、"

    Args:
        text: Text to extract from

    Returns:
        Question number as string, or None if not found
    """
    # Remove whitespace and newlines
    text_clean = text.strip().replace('\n', ' ')

    patterns = [
        r'^(\d+)\s*[\.。、]',           # "19." "20、"
        r'^第\s*(\d+)\s*[题]',          # "第19题"
        r'^\s*(\d+)\s*[）\)]',         # "19）" "20)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean)
        if match:
            return match.group(1)

    return None


def find_question_markers(
    ocr_results: List[Dict],
    page_num: int,
    subjective_question_ids: List[str]
) -> List[Dict]:
    """
    Find question number markers in OCR results

    Args:
        ocr_results: List of OCR results for this page
        page_num: Page number
        subjective_question_ids: List of subjective question IDs to look for

    Returns:
        List of question markers with positions
        [
            {'question_id': '19', 'y_start': 150, 'page': 3, 'region_id': '...'},
            ...
        ]
    """
    markers = []

    for result in ocr_results:
        content = result.get('content', '').strip()
        bbox = result.get('bbox', [])
        region_id = result.get('region_id', '')

        # Extract question number
        q_num = extract_question_number(content)

        # Check if this is a subjective question we're looking for
        if q_num and q_num in subjective_question_ids:
            markers.append({
                'question_id': q_num,
                'y_start': bbox[1],  # Top y coordinate
                'page': page_num,
                'region_id': region_id,
                'content': content[:50]  # First 50 chars for debugging
            })

    return markers


def find_question_end_boundary(
    question_start_y: float,
    page_num: int,
    all_ocr_results: List[Dict],
    page_height: int
) -> float:
    """
    Find the end boundary for a question by scanning downward

    Args:
        question_start_y: Starting y coordinate of the question
        page_num: Current page number
        all_ocr_results: All OCR results for this page
        page_height: Page height in pixels

    Returns:
        Y coordinate where the question should end
    """
    # Get all regions below the question start
    regions_below = [
        r for r in all_ocr_results
        if r.get('bbox', [0, 0, 0, 0])[1] > question_start_y
    ]

    # Sort by y position
    regions_below.sort(key=lambda r: r.get('bbox', [0, 0, 0, 0])[1])

    for region in regions_below:
        content = region.get('content', '').strip()
        region_type = region.get('type', 'text')
        bbox = region.get('bbox', [0, 0, 0, 0])

        # Stop condition 1: Next question number (e.g., "6.", "11.")
        if re.match(r'^\d+\.', content):
            return bbox[1]  # Return y coordinate of next question

        # Stop condition 2: Chapter/section title (detected by PP-DocLayout)
        if region_type == 'paragraph_title':
            return bbox[1]

    # No boundary found, extend to page bottom
    return page_height


def calculate_question_regions(
    markers: List[Dict],
    all_ocr_per_page: Dict[int, List[Dict]],
    page_dimensions: Dict[int, Tuple[int, int]],
    margin_left: int = 50,
    margin_right: int = 50,
    expand_top: int = 0,
    expand_bottom: int = 20
) -> Dict[str, Dict]:
    """
    Calculate bbox for each question based on markers

    Args:
        markers: List of question markers (sorted by y_start)
        all_ocr_per_page: All OCR results per page {page_num: [results]}
        page_dimensions: Page dimensions {page_num: (width, height)}
        margin_left: Left margin in pixels
        margin_right: Right margin in pixels
        expand_top: Expand bbox upward by N pixels
        expand_bottom: Expand bbox downward by N pixels

    Returns:
        Dict mapping question_id to region info
    """
    regions = {}

    # Sort markers by page and y position
    sorted_markers = sorted(markers, key=lambda m: (m['page'], m['y_start']))

    for marker in sorted_markers:
        q_id = marker['question_id']
        page = marker['page']

        # Get page dimensions
        if page in page_dimensions:
            page_width, page_height = page_dimensions[page]
        else:
            page_width, page_height = 6400, 9000  # Default

        # Get all OCR results for this page
        page_ocr_results = all_ocr_per_page.get(page, [])

        # Calculate y range
        y_start = max(0, marker['y_start'] - expand_top)

        # Find end boundary by scanning downward
        y_end = find_question_end_boundary(
            question_start_y=marker['y_start'],
            page_num=page,
            all_ocr_results=page_ocr_results,
            page_height=page_height
        )

        # Apply expand_bottom (but don't exceed page height)
        y_end = min(page_height, y_end + expand_bottom)

        # Calculate x range
        x_start = margin_left
        x_end = page_width - margin_right

        bbox = [x_start, y_start, x_end, y_end]

        regions[q_id] = {
            'page': page,
            'bbox': bbox,
            'marker': marker,
            'is_cross_page': (y_end >= page_height - 10)
        }

    return regions


def detect_cross_page_questions(
    all_markers: Dict[int, List[Dict]],
    subjective_question_ids: List[str]
) -> Dict[str, List[int]]:
    """
    Detect which questions span multiple pages

    Args:
        all_markers: Markers per page {page_num: [markers]}
        subjective_question_ids: All subjective question IDs

    Returns:
        Dict mapping question_id to list of pages it appears on
        {
            '21': [3, 4],  # Question 21 spans pages 3 and 4
            '22': [4]
        }
    """
    question_pages = {}

    for page_num, markers in all_markers.items():
        for marker in markers:
            q_id = marker['question_id']
            if q_id not in question_pages:
                question_pages[q_id] = []
            if page_num not in question_pages[q_id]:
                question_pages[q_id].append(page_num)

    # Sort pages for each question
    for q_id in question_pages:
        question_pages[q_id].sort()

    return question_pages


def merge_cross_page_regions(
    regions_per_page: Dict[int, Dict[str, Dict]],
    question_pages: Dict[str, List[int]]
) -> Dict[str, Dict]:
    """
    Merge regions for cross-page questions

    Args:
        regions_per_page: Regions per page {page_num: {q_id: region}}
        question_pages: Pages each question appears on

    Returns:
        Merged regions with multi-page support
        {
            '19': {
                'pages': [3],
                'bboxes': {3: [x1, y1, x2, y2]}
            },
            '21': {
                'pages': [3, 4],
                'bboxes': {
                    3: [x1, y1, x2, y2],
                    4: [x1, y1, x2, y2]
                },
                'is_cross_page': True
            }
        }
    """
    merged = {}

    for q_id, pages in question_pages.items():
        merged[q_id] = {
            'pages': pages,
            'bboxes': {},
            'is_cross_page': len(pages) > 1
        }

        for page in pages:
            if page in regions_per_page and q_id in regions_per_page[page]:
                region = regions_per_page[page][q_id]
                merged[q_id]['bboxes'][page] = region['bbox']

    return merged


def visualize_subjective_regions(
    page_images: Dict[int, np.ndarray],
    subjective_regions: Dict[str, Dict],
    output_dir: Path,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_width: int = 8
) -> Dict[int, Path]:
    """
    Visualize subjective question regions on page images

    Args:
        page_images: Page images {page_num: image_array}
        subjective_regions: Regions from merge_cross_page_regions()
        output_dir: Output directory for visualization images
        box_color: RGB color for bounding boxes
        box_width: Line width for boxes

    Returns:
        Dict mapping page_num to saved image path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # Group regions by page for efficient drawing
    regions_by_page = {}
    for q_id, region_data in subjective_regions.items():
        for page in region_data['pages']:
            if page not in regions_by_page:
                regions_by_page[page] = []

            bbox = region_data['bboxes'].get(page)
            if bbox:
                regions_by_page[page].append({
                    'question_id': q_id,
                    'bbox': bbox,
                    'is_cross_page': region_data.get('is_cross_page', False)
                })

    # Draw on each page
    for page_num, regions in regions_by_page.items():
        if page_num not in page_images:
            continue

        # Convert to PIL Image
        img_array = page_images[page_num]
        if isinstance(img_array, np.ndarray):
            pil_img = Image.fromarray(img_array)
        else:
            pil_img = img_array

        draw = ImageDraw.Draw(pil_img)

        # Try to load font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()

        # Draw each region
        for region in regions:
            bbox = region['bbox']
            q_id = region['question_id']
            is_cross_page = region['is_cross_page']

            # Draw rectangle
            draw.rectangle(bbox, outline=box_color, width=box_width)

            # Draw label
            label = f"Q{q_id}"
            if is_cross_page:
                label += " (跨页)"

            # Label background
            label_bbox = draw.textbbox((bbox[0], bbox[1] - 70), label, font=font)
            draw.rectangle(label_bbox, fill=box_color)
            draw.text((bbox[0], bbox[1] - 70), label, fill=(255, 255, 255), font=font)

        # Save visualization
        output_path = output_dir / f"page_{page_num}_subjective_regions.jpg"
        pil_img.save(output_path, quality=95)
        saved_paths[page_num] = output_path

        print(f"    Visualization saved: {output_path.name}")

    return saved_paths


def locate_subjective_questions(
    ocr_dir: Path,
    answer_key_path: Path,
    page_images: Dict[int, np.ndarray],
    output_dir: Path,
    margin_left: int = 50,
    margin_right: int = 50,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Main function to locate subjective question regions

    Args:
        ocr_dir: Directory with OCR results (step2_ocr_regions/)
        answer_key_path: Path to answer_key.json
        page_images: Dict of page images {page_num: image_array}
        output_dir: Output directory for results
        margin_left: Left margin for bbox
        margin_right: Right margin for bbox
        visualize: Whether to generate visualization images

    Returns:
        Dict with subjective question regions and metadata
    """
    print(f"\n{'='*80}")
    print("Locating Subjective Question Regions")
    print(f"{'='*80}")

    # Load answer key
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    subjective_questions = answer_key.get('subjective_questions', {})
    subjective_q_ids = list(subjective_questions.keys())

    if not subjective_q_ids:
        print("  No subjective questions found in answer key")
        return {'subjective_regions': {}, 'visualizations': {}}

    print(f"  Target subjective questions: {subjective_q_ids}")

    # Load OCR results
    print(f"\n  Loading OCR results from {ocr_dir.name}...")
    ocr_summary_path = ocr_dir / "ocr_summary.json"

    ocr_per_page = {}
    if ocr_summary_path.exists():
        with open(ocr_summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        for page_str, page_info in summary.get('pages', {}).items():
            page_num = int(page_str)
            ocr_json = Path(page_info['ocr_json'])

            if ocr_json.exists():
                with open(ocr_json, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    ocr_per_page[page_num] = page_data.get('results', [])

    print(f"    Loaded OCR for {len(ocr_per_page)} pages")

    # Find question markers on each page
    print(f"\n  Finding question markers...")
    all_markers = {}

    for page_num, ocr_results in ocr_per_page.items():
        markers = find_question_markers(ocr_results, page_num, subjective_q_ids)
        if markers:
            all_markers[page_num] = markers
            print(f"    Page {page_num}: Found {len(markers)} markers - {[m['question_id'] for m in markers]}")

    # Detect cross-page questions
    question_pages = detect_cross_page_questions(all_markers, subjective_q_ids)

    cross_page_questions = [q_id for q_id, pages in question_pages.items() if len(pages) > 1]
    if cross_page_questions:
        print(f"\n  Cross-page questions detected: {cross_page_questions}")
        for q_id in cross_page_questions:
            print(f"    Q{q_id}: spans pages {question_pages[q_id]}")

    # Prepare page dimensions
    page_dimensions = {}
    for page_num, image in page_images.items():
        if isinstance(image, np.ndarray):
            page_dimensions[page_num] = (image.shape[1], image.shape[0])  # (width, height)
        else:
            page_dimensions[page_num] = (6400, 9000)  # Default

    # Calculate regions (now with boundary detection)
    print(f"\n  Calculating question regions with boundary detection...")

    # Flatten all markers from all pages
    all_markers_flat = []
    for page_markers in all_markers.values():
        all_markers_flat.extend(page_markers)

    regions = calculate_question_regions(
        markers=all_markers_flat,
        all_ocr_per_page=ocr_per_page,
        page_dimensions=page_dimensions,
        margin_left=margin_left,
        margin_right=margin_right
    )

    # Group by page for reporting
    regions_per_page = {}
    for q_id, region in regions.items():
        page_num = region['page']
        if page_num not in regions_per_page:
            regions_per_page[page_num] = {}
        regions_per_page[page_num][q_id] = region

        print(f"    Page {page_num}, Q{q_id}: bbox={region['bbox']}, cross_page={region['is_cross_page']}")

    # Merge cross-page regions
    subjective_regions = merge_cross_page_regions(regions_per_page, question_pages)

    # Visualize if requested
    visualization_paths = {}
    if visualize:
        print(f"\n  Generating visualizations...")
        visualization_paths = visualize_subjective_regions(
            page_images=page_images,
            subjective_regions=subjective_regions,
            output_dir=output_dir / "visualizations"
        )

    # Build output
    output_data = {
        'source': {
            'ocr_dir': str(ocr_dir),
            'answer_key': str(answer_key_path)
        },
        'total_subjective_questions': len(subjective_q_ids),
        'located_questions': len(subjective_regions),
        'cross_page_questions': cross_page_questions,
        'subjective_regions': subjective_regions,
        'visualizations': {str(k): str(v) for k, v in visualization_paths.items()}
    }

    # Save output
    output_path = output_dir / "subjective_regions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved: {output_path}")
    print(f"  Located {len(subjective_regions)}/{len(subjective_q_ids)} subjective questions")

    return output_data
