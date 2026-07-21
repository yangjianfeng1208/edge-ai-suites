"""
Grade Subjective Questions with VLM

Uses the new subjective_regions.json from Step 3 to grade subjective questions.
Handles cross-page questions by stitching multiple page images together.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from PIL import Image
import numpy as np

from services.grade_with_vlm import (
    encode_image_to_base64,
    load_rubric,
    construct_vlm_prompt,
    construct_vlm_prompt_no_rubric,
    call_vlm_api
)


def crop_question_region(
    page_image: np.ndarray,
    bbox: list
) -> Image.Image:
    """
    Crop question region from page image

    Args:
        page_image: Full page image (numpy array or PIL Image)
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Cropped PIL Image
    """
    if isinstance(page_image, np.ndarray):
        pil_image = Image.fromarray(page_image)
    else:
        pil_image = page_image

    x1, y1, x2, y2 = bbox
    cropped = pil_image.crop((x1, y1, x2, y2))

    return cropped


def stitch_cross_page_images(
    images: list,
    direction: str = 'vertical'
) -> Image.Image:
    """
    Stitch multiple images together for cross-page questions

    Args:
        images: List of PIL Images to stitch
        direction: 'vertical' or 'horizontal'

    Returns:
        Stitched PIL Image
    """
    if not images:
        return None

    if len(images) == 1:
        return images[0]

    if direction == 'vertical':
        # Stack vertically
        widths = [img.width for img in images]
        heights = [img.height for img in images]

        max_width = max(widths)
        total_height = sum(heights)

        stitched = Image.new('RGB', (max_width, total_height), (255, 255, 255))

        y_offset = 0
        for img in images:
            stitched.paste(img, (0, y_offset))
            y_offset += img.height

        return stitched
    else:
        # Stack horizontally
        widths = [img.width for img in images]
        heights = [img.height for img in images]

        total_width = sum(widths)
        max_height = max(heights)

        stitched = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        x_offset = 0
        for img in images:
            stitched.paste(img, (x_offset, 0))
            x_offset += img.width

        return stitched


def prepare_question_image(
    region_data: Dict,
    page_images: Dict[int, np.ndarray],
    output_dir: Path,
    question_id: str
) -> Tuple[Path, bool]:
    """
    Prepare question image for VLM grading

    For single-page questions: crop the region
    For cross-page questions: stitch multiple regions

    Args:
        region_data: Region data from subjective_regions.json
        page_images: Dict of page images {page_num: image_array}
        output_dir: Directory to save cropped images
        question_id: Question ID

    Returns:
        Tuple of (image_path, is_cross_page)
    """
    pages = region_data['pages']
    bboxes = region_data['bboxes']
    is_cross_page = region_data.get('is_cross_page', False)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not is_cross_page:
        # Single page question
        page = pages[0]
        bbox = bboxes[str(page)]

        if page not in page_images:
            raise ValueError(f"Page {page} image not found")

        cropped = crop_question_region(page_images[page], bbox)

        # Save
        output_path = output_dir / f"question_{question_id}.jpg"
        cropped.save(output_path, quality=95)

        return output_path, False

    else:
        # Cross-page question - stitch images
        cropped_images = []

        for page in sorted(pages):
            bbox = bboxes[str(page)]

            if page not in page_images:
                print(f"    Warning: Page {page} image not found, skipping")
                continue

            cropped = crop_question_region(page_images[page], bbox)
            cropped_images.append(cropped)

        if not cropped_images:
            raise ValueError(f"No images available for cross-page question {question_id}")

        # Stitch vertically
        stitched = stitch_cross_page_images(cropped_images, direction='vertical')

        # Save
        output_path = output_dir / f"question_{question_id}_cross_page.jpg"
        stitched.save(output_path, quality=95)

        return output_path, True


def grade_subjective_with_vlm(
    subjective_regions_path: Path,
    answer_key_path: Path,
    rubric_dir: Path,
    page_images: Dict[int, np.ndarray],
    output_dir: Path,
    vlm_api_url: str = 'http://127.0.0.1:9900',
    student_id: str = 'student1',
    language: str = 'en',
    subject: str = None,
    use_rubric: bool = True,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Grade subjective questions using VLM

    Args:
        subjective_regions_path: Path to subjective_regions.json
        answer_key_path: Path to answer_key.json
        rubric_dir: Directory containing rubric files
        page_images: Dict of page images {page_num: image_array}
        output_dir: Output directory for results
        vlm_api_url: VLM API URL
        student_id: Student ID
        language: Prompt language, 'cn' or 'en'
        subject: Subject name inserted into the grading prompt (e.g. "Math")
        use_rubric: When False, grade without a rubric assuming a fixed max
            score; the caller can rescale via actual_score * (vlm_score / max)
        debug_mode: When False, the cropped answer images are removed after
            grading (they are only needed transiently for the VLM call)

    Returns:
        Grading results dict
    """
    print(f"\n{'='*80}")
    print("VLM Automatic Grading - Subjective Questions")
    print(f"{'='*80}")
    print(f"VLM API: {vlm_api_url}")

    total_start_time = time.time()

    # Load subjective regions
    print(f"\n[1/5] Loading subjective question regions...")
    with open(subjective_regions_path, 'r', encoding='utf-8') as f:
        subjective_data = json.load(f)

    subjective_regions = subjective_data['subjective_regions']
    cross_page_questions = subjective_data.get('cross_page_questions', [])

    print(f"  Total subjective questions: {len(subjective_regions)}")
    print(f"  Cross-page questions: {cross_page_questions}")

    # Load answer key
    print(f"\n[2/5] Loading answer key...")
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)

    subjective_questions = answer_key.get('subjective_questions', {})

    # Prepare output directories
    cropped_images_dir = output_dir / "cropped_answers"
    vlm_details_dir = output_dir / "vlm_details"
    cropped_images_dir.mkdir(parents=True, exist_ok=True)
    vlm_details_dir.mkdir(parents=True, exist_ok=True)

    # Grade each question
    print(f"\n[3/5] Preparing question images...")
    question_images = {}

    for q_id, region_data in subjective_regions.items():
        print(f"\n  Question {q_id}:")

        try:
            image_path, is_cross_page = prepare_question_image(
                region_data=region_data,
                page_images=page_images,
                output_dir=cropped_images_dir,
                question_id=q_id
            )

            question_images[q_id] = {
                'image_path': image_path,
                'is_cross_page': is_cross_page,
                'pages': region_data['pages']
            }

            status = "(cross-page)" if is_cross_page else f"(page {region_data['pages'][0]})"
            print(f"    Saved: {image_path.name} {status}")

        except Exception as e:
            print(f"    Error preparing image: {e}")
            continue

    print(f"\n  Prepared {len(question_images)}/{len(subjective_regions)} question images")

    # Grade with VLM
    print(f"\n[4/5] Grading with VLM...\n")
    grading_results = []

    for q_id in sorted(question_images.keys(), key=lambda x: int(x)):
        q_info = question_images[q_id]
        image_path = q_info['image_path']

        print(f"  Grading Question {q_id}...")
        print(f"    Image: {image_path.name}")

        if not image_path.exists():
            print(f"    Skipped (image not found)")
            continue

        # Get question info from answer key
        q_data = subjective_questions.get(q_id, {})
        rubric_file = q_data.get('rubric')

        if use_rubric:
            if not rubric_file:
                print(f"    Skipped (no rubric specified in answer key)")
                continue

            # Load rubric
            rubric_path = rubric_dir / rubric_file
            if not rubric_path.exists():
                print(f"    Skipped (rubric not found: {rubric_file})")
                continue

            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric = json.load(f)

            # Construct VLM prompt
            vlm_input = construct_vlm_prompt(q_id, rubric, image_path, language=language, subject=subject)
        else:
            # No rubric: grade against a fixed max score (see construct_vlm_prompt_no_rubric)
            rubric_file = None
            vlm_input = construct_vlm_prompt_no_rubric(q_id, image_path, language=language, subject=subject)

        if vlm_input.get('error'):
            print(f"    Skipped ({vlm_input['error']})")
            continue

        print(f"\n    [Grading Prompt Preview]")
        print(f"    {'-'*60}")
        print(f"    {vlm_input['prompt'][:300]}...")
        print(f"    {'-'*60}\n")

        # Call VLM API
        q_start_time = time.time()
        vlm_result = call_vlm_api(vlm_input, model='qwen-vl', api_url=vlm_api_url)
        q_elapsed = time.time() - q_start_time

        # Store result
        grading_result = {
            'question_id': q_id,
            'alias': q_data.get('alias', q_id),
            'type': q_data.get('type', 'unknown'),
            'pages': q_info['pages'],
            'is_cross_page': q_info['is_cross_page'],
            'image_path': str(image_path),
            'rubric_file': rubric_file,
            'vlm_score': vlm_result.get('total_score', 0),
            'max_score': vlm_result.get('max_score', 0),
            'comment': vlm_result.get('comment', ''),
            'raw_output': vlm_result.get('raw_output', ''),
            'model': vlm_result.get('model', 'qwen-vl'),
            'time_seconds': q_elapsed
        }

        grading_results.append(grading_result)

        print(f"\n    [Grading Result]")
        print(f"    Score: {vlm_result.get('total_score', 0)}/{vlm_result.get('max_score', 0)}")
        if vlm_result.get('comment'):
            print(f"    Comment: {vlm_result.get('comment', '')[:100]}")
        print(f"    Time: {q_elapsed:.1f}s")
        print()

        # Save detailed output
        detail_file = vlm_details_dir / f"{student_id}_Q{q_id}_vlm_output.txt"
        with open(detail_file, 'w', encoding='utf-8') as f:
            f.write(f"Question ID: {q_id}\n")
            f.write(f"Alias: {q_data.get('alias', q_id)}\n")
            f.write(f"Type: {q_data.get('type', 'unknown')}\n")
            f.write(f"Pages: {q_info['pages']}\n")
            f.write(f"Cross-page: {q_info['is_cross_page']}\n")
            f.write(f"Max score: {vlm_result.get('max_score', 0)} points\n")
            f.write(f"Score: {vlm_result.get('total_score', 0)} points\n")
            f.write(f"Time: {q_elapsed:.1f}s\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"Grading Prompt:\n")
            f.write(f"{'='*80}\n\n")
            f.write(vlm_input.get('prompt', ''))
            f.write(f"\n\n{'='*80}\n")
            f.write(f"VLM Full Output:\n")
            f.write(f"{'='*80}\n\n")
            f.write(vlm_result.get('raw_output', ''))
            f.write(f"\n\n{'='*80}\n")
            f.write(f"Extracted Comment:\n")
            f.write(f"{'='*80}\n\n")
            f.write(vlm_result.get('comment', ''))

    # Calculate totals
    print(f"\n[5/5] Saving grading results...")

    total_score = sum(r['vlm_score'] for r in grading_results)
    max_total = sum(r['max_score'] for r in grading_results)

    output_data = {
        'student_id': student_id,
        'total_subjective_score': total_score,
        'max_subjective_score': max_total,
        'questions_graded': len(grading_results),
        'cross_page_questions': cross_page_questions,
        'grading_results': grading_results,
        'vlm_model': 'qwen-vl',
        'vlm_api_url': vlm_api_url
    }

    # Save JSON
    output_json = output_dir / "subjective_grading.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Save summary markdown
    summary_file = vlm_details_dir / f"{student_id}_grading_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# {student_id} - Subjective Questions Grading\n\n")
        f.write(f"**Total Score: {total_score}/{max_total}**\n\n")
        f.write(f"**Questions Graded:** {len(grading_results)}\n\n")
        f.write(f"**Cross-page Questions:** {', '.join(cross_page_questions) if cross_page_questions else 'None'}\n\n")
        f.write(f"---\n\n")

        for result in grading_results:
            q_id = result['question_id']
            f.write(f"## Question {q_id} ({result['alias']})\n\n")
            f.write(f"**Type:** {result['type']}\n\n")
            f.write(f"**Pages:** {result['pages']}\n\n")
            if result['is_cross_page']:
                f.write(f"**Cross-page:** Yes\n\n")
            f.write(f"**Score:** {result['vlm_score']}/{result['max_score']}\n\n")
            f.write(f"**Comment:** {result['comment']}\n\n")
            f.write(f"**Answer Image:** [{Path(result['image_path']).name}](../cropped_answers/{Path(result['image_path']).name})\n\n")
            f.write(f"**Detailed Output:** [{student_id}_Q{q_id}_vlm_output.txt]({student_id}_Q{q_id}_vlm_output.txt)\n\n")
            f.write(f"---\n\n")

    # Cropped answer images are only needed transiently for the VLM call;
    # keep them for inspection only in debug mode.
    if not debug_mode:
        import shutil
        shutil.rmtree(cropped_images_dir, ignore_errors=True)

    # Calculate timing
    total_elapsed = time.time() - total_start_time
    avg_time = total_elapsed / len(grading_results) if grading_results else 0

    print(f"\n{'='*80}")
    print("Subjective Grading Completed")
    print(f"{'='*80}")
    print(f"Total Score: {total_score}/{max_total}")
    print(f"Questions Graded: {len(grading_results)}")
    print(f"  Total Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Average Time per Question: {avg_time:.1f}s")
    print(f"  Per-Question Time:")
    for r in grading_results:
        print(f"    Q{r['question_id']:<4s} {r.get('time_seconds', 0):6.1f}s  "
              f"(score {r['vlm_score']}/{r['max_score']})")
    print(f"Output JSON: {output_json}")
    print(f"Summary Report: {summary_file}")
    if debug_mode:
        print(f"Cropped Images: {cropped_images_dir}")

    return output_data

