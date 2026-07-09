from pathlib import Path
import sys
import json
import argparse
import yaml
import numpy as np
import time
from utils.pdf_processor import convert_pdf_to_images, image_to_bytes

BASE_DIR = Path(__file__).parent


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Automated Grading System')
    parser.add_argument('--rubric', type=str, default=None,
                       help='Path to rubric directory (e.g., test_data/2025_sh_zhongkao_math/rubric_guided_scoring)')
    parser.add_argument('--paper', type=str, default=None,
                       help='Path to student paper directory (e.g., test_data/2025_sh_zhongkao_math/papers/student1)')
    args = parser.parse_args()

    # Start overall timing
    overall_start_time = time.time()
    step_timings = {}

    print("="*80)
    print("Automated Grading System")
    print("="*80)

    # Load main config
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Pipeline configuration
    SKIP_SUBJECTIVE = config['pipeline']['skip_subjective']
    SKIP_OCR = config['pipeline'].get('skip_ocr', False)
    PDF_RENDER_DPI = config['pipeline']['pdf_render_dpi']

    # OCR configuration
    USE_CACHED_OCR = config['ocr']['use_cached']
    SAVE_PDF_IMAGES = config['ocr']['save_pdf_images']
    PDF_DPI = config['ocr']['pdf_dpi']
    OCR_MAX_PIXELS = config['ocr']['max_pixels']
    OCR_MAX_TOKENS = config['ocr']['max_tokens']

    JPEG_QUALITY = config['ocr'].get('jpeg_quality', 85)
    RESIZE_RATIO = config['ocr'].get('resize_ratio')
    MAX_IMAGE_WIDTH = config['ocr'].get('max_image_width')
    MAX_IMAGE_HEIGHT = config['ocr'].get('max_image_height')
    MAX_IMAGE_PIXELS = config['ocr'].get('max_image_pixels')
    ENHANCE_CONTRAST = config['ocr'].get('enhance_contrast')
    ENHANCE_SHARPNESS = config['ocr'].get('enhance_sharpness')

    # Get rubric directory (command line overrides config)
    if args.rubric:
        rubric_path = args.rubric
    else:
        rubric_path = config['pipeline']['rubric_dir']

    RUBRIC_DIR = Path(rubric_path).resolve()

    if not RUBRIC_DIR.exists():
        print(f"Error: Rubric directory not found: {RUBRIC_DIR}")
        return

    # Get papers directory (command line overrides config)
    if args.paper:
        papers_path = args.paper
    else:
        papers_path = config['pipeline']['papers_dir']

    papers_dir = Path(papers_path).resolve()

    if not papers_dir.exists():
        print(f"Error: Papers directory not found: {papers_dir}")
        return

    # Extract exam name from rubric directory's parent
    if RUBRIC_DIR.name == 'rubric_guided_scoring':
        exam_root = RUBRIC_DIR.parent
        exam_name = exam_root.name
    else:
        # If rubric dir is not named 'rubric_guided_scoring', use its name
        exam_name = RUBRIC_DIR.name

    # Find all student directories
    student_dirs = sorted([d for d in papers_dir.iterdir() if d.is_dir()])

    if not student_dirs:
        print(f"Error: No student directories found in {papers_dir}")
        return

    print(f"\nLoading configuration...")
    print(f"  Exam: {exam_name}")
    print(f"  Rubric: {RUBRIC_DIR}")
    print(f"  Papers: {papers_dir}")
    print(f"  Found {len(student_dirs)} student(s): {[d.name for d in student_dirs]}")

    # Process each student
    for student_idx, student_dir in enumerate(student_dirs, 1):
        student_id = student_dir.name

        print(f"\n{'='*80}")
        print(f"Processing Student {student_idx}/{len(student_dirs)}: {student_id}")
        print(f"{'='*80}")

        # Find PDF file in student directory
        pdf_files = list(student_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"Warning: No PDF file found in {student_dir}, skipping")
            continue

        PDF_PATH = pdf_files[0]  # Use first PDF found

        OUTPUT_BASE = BASE_DIR / f"outputs/{exam_name}/{student_id}"
        PROCESSED_OUTPUT_DIR = OUTPUT_BASE / "processed_answers"

        # Output directories with step prefixes
        STEP1_LAYOUT_DIR = OUTPUT_BASE / "step1_layout_detection"
        STEP2_OCR_DIR = OUTPUT_BASE / "step2_ocr_regions"
        STEP3_MAPPING_DIR = OUTPUT_BASE / "step3_question_mapping"
        STEP4_OBJECTIVE_DIR = OUTPUT_BASE / "step4_objective_grading"
        STEP5_SUBJECTIVE_DIR = OUTPUT_BASE / "step5_subjective_grading"

        VLM_API_URL = config['vlm_service']['base_url']
        OCR_API_URL = config['ocr_service']['base_url']

        print(f"\n  PDF: {PDF_PATH.name}")
        print(f"  Output: {OUTPUT_BASE}")

        if not PDF_PATH.exists():
            print(f"\nError: PDF not found: {PDF_PATH}")
            continue

        step = 0
    # Calculate total steps
    # Steps: Layout Detection (1), OCR (1), Subjective Region Detection (1), Objective Grading (1) = 4
    total = 4

    # Add subjective grading step
    if not SKIP_SUBJECTIVE:
        total += 1  # Subjective Grading (optional)

    # PP-DocLayout Detection
    if True:  # Always use layout detection
        step += 1
        step_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[Step {step}/{total}] Layout Detection (PP-DocLayout)")
        print(f"{'='*80}")

        from utils.pdf_processor import render_pdf_to_images
        from utils.detection_client import (
            check_service_health, detect_page_layout,
            merge_overlapping_boxes, draw_detection_boxes
        )

        DETECTION_SERVICE_URL = config['detection_service']['base_url']
        DETECTION_TIMEOUT = config['detection_service']['timeout']
        TARGET_LABELS = config['detection_service']['target_labels']
        MIN_SCORE = config['detection_service']['min_score']
        SORT_BOXES = config['detection_service']['sort_boxes']
        EXPAND_MARGIN = config['detection_service']['expand_margin']
        MERGE_OVERLAPPING = config['detection_service'].get('merge_overlapping', False)
        IOU_THRESHOLD = config['detection_service'].get('iou_threshold', 0.7)
        SAVE_VISUALIZATIONS = config['detection_service'].get('save_visualizations', True)

        try:
            print(f"\nChecking detection service...")
            print(f"  URL: {DETECTION_SERVICE_URL}")

            if not check_service_health(DETECTION_SERVICE_URL):
                print(f"\nError: Detection service not available at {DETECTION_SERVICE_URL}")
                print(f"  Start with: python ocr_services/layout_detection_server.py")
                print(f"\nSkipping detection and using manual inspection...")
                return

            print(f"  Status: Healthy")

            print(f"\nRendering PDF...")
            pages = render_pdf_to_images(PDF_PATH, dpi=PDF_RENDER_DPI)
            print(f"Total pages: {len(pages)}")

            print(f"\nStarting detection...")
            print(f"  Target labels: {TARGET_LABELS}")
            print(f"  Min score: {MIN_SCORE}")
            print(f"  Expand margin: {EXPAND_MARGIN}px")

            all_detections = {}
            STEP1_LAYOUT_DIR.mkdir(parents=True, exist_ok=True)

            for page_data in pages:
                page_num = page_data['page_num']
                image = page_data['image']

                # Detect layout
                boxes = detect_page_layout(
                    page_image=image,
                    service_url=DETECTION_SERVICE_URL,
                    target_labels=TARGET_LABELS,
                    min_score=MIN_SCORE,
                    sort=SORT_BOXES,
                    expand_margin=EXPAND_MARGIN
                )

                # Merge overlapping boxes if enabled
                if MERGE_OVERLAPPING and len(boxes) > 0:
                    boxes = merge_overlapping_boxes(boxes, iou_threshold=IOU_THRESHOLD)

                all_detections[page_num] = boxes
                print(f"  Page {page_num}: {len(boxes)} regions detected")

                # Save JSON for this page
                page_json_path = STEP1_LAYOUT_DIR / f"page_{page_num}_detections.json"
                with open(page_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'page_num': page_num,
                        'total_regions': len(boxes),
                        'config': {
                            'target_labels': TARGET_LABELS,
                            'min_score': MIN_SCORE,
                            'expand_margin': EXPAND_MARGIN
                        },
                        'boxes': boxes
                    }, f, ensure_ascii=False, indent=2)

                # Save visualization if enabled
                if SAVE_VISUALIZATIONS and len(boxes) > 0:
                    viz_path = STEP1_LAYOUT_DIR / f"page_{page_num}_visualization.jpg"
                    draw_detection_boxes(image, boxes, viz_path)
                    print(f"    Saved visualization: {viz_path.name}")

            # Save summary JSON
            summary_json_path = STEP1_LAYOUT_DIR / "detection_summary.json"
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'source_pdf': str(PDF_PATH),
                    'total_pages': len(pages),
                    'total_regions': sum(len(v) for v in all_detections.values()),
                    'config': {
                        'target_labels': TARGET_LABELS,
                        'min_score': MIN_SCORE,
                        'expand_margin': EXPAND_MARGIN,
                        'merge_overlapping': MERGE_OVERLAPPING
                    },
                    'pages': {
                        str(page_num): {
                            'num_regions': len(boxes),
                            'json_path': str(STEP1_LAYOUT_DIR / f"page_{page_num}_detections.json")
                        }
                        for page_num, boxes in all_detections.items()
                    }
                }, f, ensure_ascii=False, indent=2)

            print(f"\nCompleted: {sum(len(v) for v in all_detections.values())} regions detected")
            print(f"  Output directory: {STEP1_LAYOUT_DIR}")
            print(f"  Summary: {summary_json_path.name}")

            # Don't exit - continue to OCR step
            print(f"\nDetection completed. Continuing to OCR...")

            step_timings[f'Step {step} - Layout Detection'] = time.time() - step_start_time

        except Exception as e:
            print(f"\nError: Layout detection failed: {e}")
            import traceback
            traceback.print_exc()
            return

    step += 1
    step_start_time = time.time()
    print(f"\n{'='*80}")
    print(f"[Step {step}/{total}] OCR Recognition")
    print(f"{'='*80}")

    # Check if OCR should be skipped
    if SKIP_OCR:
        print(f"\nOCR skipped (skip_ocr=true in config)")

        # Verify required outputs exist
        if STEP2_OCR_DIR.exists():
            print(f"  Using existing OCR results:")
            print(f"    Region-based: {STEP2_OCR_DIR}")
            step_timings[f'Step {step} - OCR Recognition'] = time.time() - step_start_time
        else:
            print(f"\nWarning: OCR results not found!")
            print(f"    Missing: {STEP2_OCR_DIR}")
            print(f"\n  Please run OCR at least once before skipping.")
            print(f"  Set skip_ocr=false in config.yaml and run again.")
            return

    # Use region-based OCR
    else:
        print(f"\nUsing region-based OCR (two-stage approach)")

        try:
            import requests
            from utils.pdf_processor import render_pdf_to_images

            print(f"\nChecking OCR server health...")
            print(f"  URL: {OCR_API_URL}")
            health_response = requests.get(f"{OCR_API_URL}/health", timeout=5)
            if health_response.status_code != 200:
                raise Exception(f"OCR server not healthy: {health_response.text}")

            health_data = health_response.json()
            print(f"  Status: {health_data.get('status')}")
            print(f"  Device: {health_data.get('device')}")

            # Create output directory
            STEP2_OCR_DIR.mkdir(parents=True, exist_ok=True)

            # Render PDF to images
            print(f"\nRendering PDF...")
            pages = render_pdf_to_images(PDF_PATH, dpi=PDF_RENDER_DPI)
            print(f"Total pages: {len(pages)}")

            print(f"\nProcessing regions with OCR...")
            all_ocr_results = {}
            total_inference_time = 0

            for page_data in pages:
                page_num = page_data['page_num']
                image = page_data['image']

                print(f"\n  [Page {page_num}/{len(pages)}]")

                # Load detection results for this page
                detection_json = STEP1_LAYOUT_DIR / f"page_{page_num}_detections.json"
                if not detection_json.exists():
                    print(f"    Warning: Detection JSON not found, skipping")
                    continue

                with open(detection_json, 'r', encoding='utf-8') as f:
                    detection_data = json.load(f)

                boxes = detection_data.get('boxes', [])
                if not boxes:
                    print(f"    No regions detected, skipping")
                    continue

                print(f"    Found {len(boxes)} regions")

                # Build regions array
                regions = []
                for i, box in enumerate(boxes):
                    regions.append({
                        'bbox': box['coordinate'],
                        'type': box['label'],
                        'region_id': f"page{page_num}_region{i+1}"
                    })

                # Save image temporarily
                from PIL import Image as PILImage
                temp_image_path = STEP2_OCR_DIR / f"temp_page_{page_num}.jpg"
                if isinstance(image, np.ndarray):
                    pil_img = PILImage.fromarray(image)
                else:
                    pil_img = image
                pil_img.save(temp_image_path, quality=95)

                # Call /ocr/regions API
                print(f"    Calling /ocr/regions API...")
                with open(temp_image_path, 'rb') as f:
                    files = {'file': (f'page_{page_num}.jpg', f, 'image/jpeg')}
                    data = {
                        'regions': json.dumps(regions),
                        'max_new_tokens': OCR_MAX_TOKENS,
                        'max_pixels': OCR_MAX_PIXELS
                    }

                    response = requests.post(
                        f"{OCR_API_URL}/ocr/regions",
                        files=files,
                        data=data,
                        timeout=300
                    )

                # Clean up temp file
                temp_image_path.unlink()

                if response.status_code != 200:
                    print(f"    Error: OCR API failed: {response.text}")
                    continue

                result = response.json()
                if not result.get('success'):
                    print(f"    Error: OCR failed: {result.get('error')}")
                    continue

                # Save results
                ocr_results = result.get('results', [])
                inference_time = result.get('total_inference_time', 0)
                total_inference_time += inference_time

                print(f"    Processed {len(ocr_results)} regions in {inference_time:.2f}s")

                # Save page OCR results
                page_ocr_path = STEP2_OCR_DIR / f"page_{page_num}_ocr.json"
                with open(page_ocr_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'page_num': page_num,
                        'num_regions': len(ocr_results),
                        'total_inference_time': inference_time,
                        'results': ocr_results
                    }, f, ensure_ascii=False, indent=2)

                all_ocr_results[page_num] = ocr_results

            # Save summary
            summary_path = STEP2_OCR_DIR / "ocr_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'source_pdf': str(PDF_PATH),
                    'total_pages': len(pages),
                    'total_regions': sum(len(v) for v in all_ocr_results.values()),
                    'total_inference_time': total_inference_time,
                    'pages': {
                        str(page_num): {
                            'num_regions': len(results),
                            'ocr_json': str(STEP2_OCR_DIR / f"page_{page_num}_ocr.json")
                        }
                        for page_num, results in all_ocr_results.items()
                    }
                }, f, ensure_ascii=False, indent=2)

            print(f"\n{'='*80}")
            print(f"Completed: {sum(len(v) for v in all_ocr_results.values())} regions processed")
            print(f"Total time: {total_inference_time:.1f}s")
            print(f"Results saved to: {STEP2_OCR_DIR}")
            print(f"{'='*80}")

            # Create full document plain text (all content concatenated)
            full_text_path = STEP2_OCR_DIR / "full_document.txt"
            with open(full_text_path, 'w', encoding='utf-8') as f:
                for page_num in sorted(all_ocr_results.keys()):
                    for result in all_ocr_results[page_num]:
                        content = result['content'].strip()
                        if content:  # Only write non-empty content
                            f.write(content)
                            f.write('\n')

            print(f"Created full document text: {full_text_path}")

            step_timings[f'Step {step} - OCR Recognition'] = time.time() - step_start_time

        except requests.exceptions.ConnectionError:
            print(f"\nError: OCR server not running at {OCR_API_URL}")
            print(f"  Start with: python ocr_services/paddleocr_vl_server.py")
            return
        except Exception as e:
            print(f"\nError: Region-based OCR failed: {e}")
            import traceback
            traceback.print_exc()
            return

    step += 1
    step_start_time = time.time()
    print(f"\n{'='*80}")
    print(f"[Step {step}/{total}] Subjective Answer Region Detection")
    print(f"{'='*80}")

    from utils.question_mapper import map_questions_to_regions
    from utils.subjective_question_locator import locate_subjective_questions

    ANSWER_KEY = RUBRIC_DIR / "answer_key.json"
    QUESTION_MAPPING_OUTPUT = STEP3_MAPPING_DIR / "question_mapping.json"

    if ANSWER_KEY.exists() and STEP2_OCR_DIR.exists():
        try:
            # Part 1: Map all questions to OCR regions
            print(f"\n[Part 1] Mapping questions to OCR regions...")
            print(f"  Answer key: {ANSWER_KEY.name}")
            print(f"  OCR directory: {STEP2_OCR_DIR.name}")

            mapping_result = map_questions_to_regions(
                ocr_dir=STEP2_OCR_DIR,
                answer_key_path=ANSWER_KEY,
                output_path=QUESTION_MAPPING_OUTPUT,
                strategy="position"
            )

            print(f"\nCompleted: {mapping_result['total_questions']} questions mapped")
            print(f"  Objective: {mapping_result['objective_questions']}")
            print(f"  Subjective: {mapping_result['subjective_questions']}")
            print(f"  Output: {QUESTION_MAPPING_OUTPUT}")

            # Part 2: Locate complete subjective question regions
            if mapping_result['subjective_questions'] > 0:
                print(f"\n[Part 2] Locating subjective question answer regions...")

                # Load page images for visualization
                from utils.pdf_processor import render_pdf_to_images
                print(f"  Loading page images for visualization...")
                pages = render_pdf_to_images(PDF_PATH, dpi=PDF_RENDER_DPI)
                page_images = {p['page_num']: p['image'] for p in pages}
                print(f"  Loaded {len(page_images)} pages")

                subjective_result = locate_subjective_questions(
                    ocr_dir=STEP2_OCR_DIR,
                    answer_key_path=ANSWER_KEY,
                    page_images=page_images,
                    output_dir=STEP3_MAPPING_DIR,
                    margin_left=50,
                    margin_right=50,
                    visualize=True
                )

                print(f"\nSubjective regions extraction completed")
                print(f"  Located: {subjective_result['located_questions']}/{subjective_result['total_subjective_questions']} questions")
                if subjective_result['cross_page_questions']:
                    print(f"  Cross-page questions: {subjective_result['cross_page_questions']}")
                print(f"  Visualizations: {STEP3_MAPPING_DIR / 'visualizations'}")

            step_timings[f'Step {step} - Subjective Region Detection'] = time.time() - step_start_time

        except Exception as e:
            print(f"\nError: Question mapping failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipped: Missing answer key or OCR results")
        if not ANSWER_KEY.exists():
            print(f"  Answer key not found: {ANSWER_KEY}")
        if not STEP2_OCR_DIR.exists():
            print(f"  OCR directory not found: {STEP2_OCR_DIR}")

    # Initialize grading results
    objective_results = None
    subjective_results = None

    step += 1
    step_start_time = time.time()
    print(f"\n{'='*80}")
    print(f"[Step {step}/{total}] Grade Objective Questions")
    print(f"{'='*80}")

    from utils.export_objective_questions import extract_questions_from_ocr
    from utils.grade_objective_questions import grade_objective_questions

    OBJECTIVE_OUTPUT = STEP4_OBJECTIVE_DIR / "objective_grading.json"
    OBJECTIVE_QUESTIONS_TXT = STEP4_OBJECTIVE_DIR / "objective_questions.txt"
    QUESTION_MAPPING_JSON = STEP3_MAPPING_DIR / "question_mapping.json"

    # Extract and grade objective questions
    if STEP2_OCR_DIR.exists() and ANSWER_KEY.exists():
        try:
            print(f"\nExtracting objective questions from OCR...")

            # First, extract questions for review
            extract_questions_from_ocr(
                ocr_dir=STEP2_OCR_DIR,
                answer_key_path=ANSWER_KEY,
                output_txt=OBJECTIVE_QUESTIONS_TXT,
                config=config
            )

            print(f"\n  Objective questions extracted: {OBJECTIVE_QUESTIONS_TXT}")

            # Then, grade them automatically
            print(f"\nGrading objective questions...")

            objective_results = grade_objective_questions(
                ocr_dir=STEP2_OCR_DIR,
                answer_key_path=ANSWER_KEY,
                output_dir=STEP4_OBJECTIVE_DIR,
                config=config
            )

            step_timings[f'Step {step} - Objective Grading'] = time.time() - step_start_time

        except Exception as e:
            print(f"\nError: Objective grading failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipped: Missing question mapping or answer key")
        if not QUESTION_MAPPING_JSON.exists():
            print(f"  Question mapping not found: {QUESTION_MAPPING_JSON}")
        if not ANSWER_KEY.exists():
            print(f"  Answer key not found: {ANSWER_KEY}")

    if SKIP_SUBJECTIVE:
        print(f"\nSubjective grading: Skipped (disabled in config.yaml)")
    else:
        step += 1
        step_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[Step {step}/{total}] Grade Subjective Questions (VLM)")
        print(f"{'='*80}")

        from utils.grade_subjective_with_vlm import grade_subjective_with_vlm

        SUBJECTIVE_REGIONS_JSON = STEP3_MAPPING_DIR / "subjective_regions.json"

        if not SUBJECTIVE_REGIONS_JSON.exists():
            print(f"\nError: Subjective regions not found: {SUBJECTIVE_REGIONS_JSON}")
            print(f"  Please ensure Step 3 completed successfully")
        else:
            try:
                # Load page images for cropping
                from utils.pdf_processor import render_pdf_to_images
                print(f"\nLoading page images...")
                pages = render_pdf_to_images(PDF_PATH, dpi=PDF_RENDER_DPI)
                page_images = {p['page_num']: p['image'] for p in pages}
                print(f"  Loaded {len(page_images)} pages")

                subjective_results = grade_subjective_with_vlm(
                    subjective_regions_path=SUBJECTIVE_REGIONS_JSON,
                    answer_key_path=ANSWER_KEY,
                    rubric_dir=RUBRIC_DIR,
                    page_images=page_images,
                    output_dir=STEP5_SUBJECTIVE_DIR,
                    vlm_api_url=VLM_API_URL,
                    student_id=student_id
                )

                step_timings[f'Step {step} - Subjective Grading'] = time.time() - step_start_time

            except Exception as e:
                print(f"\nError: VLM grading failed: {e}")
                import traceback
                traceback.print_exc()

        # Merge grading results
        print(f"\n{'='*80}")
        print("Generating Final Grading Report")
        print(f"{'='*80}")

        final_results = {
            'student_id': student_id,
            'exam_name': exam_name,
            'objective': {
                'total_score': objective_results.get('total_score', 0) if objective_results else 0,
                'total_possible_score': objective_results.get('total_possible_score', 0) if objective_results else 0,
                'questions': objective_results.get('questions', {}) if objective_results else {}
            },
            'subjective': {
                'total_score': subjective_results.get('total_subjective_score', 0) if subjective_results else 0,
                'total_possible_score': subjective_results.get('max_subjective_score', 0) if subjective_results else 0,
                'questions': subjective_results.get('grading_results', {}) if subjective_results else {}
            },
            'summary': {
                'objective_score': objective_results.get('total_score', 0) if objective_results else 0,
                'objective_max': objective_results.get('total_possible_score', 0) if objective_results else 0,
                'subjective_score': subjective_results.get('total_subjective_score', 0) if subjective_results else 0,
                'subjective_max': subjective_results.get('max_subjective_score', 0) if subjective_results else 0,
                'total_score': (objective_results.get('total_score', 0) if objective_results else 0) +
                              (subjective_results.get('total_subjective_score', 0) if subjective_results else 0),
                'total_max': (objective_results.get('total_possible_score', 0) if objective_results else 0) +
                            (subjective_results.get('max_subjective_score', 0) if subjective_results else 0)
            }
        }

        # Save final results
        final_results_path = OUTPUT_BASE / "grading_results.json"
        with open(final_results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"\n  Final Results:")
        print(f"    Objective: {final_results['summary']['objective_score']}/{final_results['summary']['objective_max']}")
        print(f"    Subjective: {final_results['summary']['subjective_score']}/{final_results['summary']['subjective_max']}")
        print(f"    Total: {final_results['summary']['total_score']}/{final_results['summary']['total_max']}")
        print(f"\n  Saved to: {final_results_path}")

        print(f"\n  Completed grading for {student_id}")

    # Overall summary
    total_elapsed = time.time() - overall_start_time

    print(f"\n{'='*80}")
    print("Grading Summary")
    print(f"{'='*80}")
    print(f"\n  Exam: {exam_name}")
    print(f"  Total students processed: {len(student_dirs)}")
    print(f"  Output directory: {BASE_DIR / f'outputs/{exam_name}'}")

    total_minutes = int(total_elapsed // 60)
    total_seconds = total_elapsed % 60
    print(f"\n  Total elapsed time: {total_minutes:2d}m {total_seconds:05.2f}s")

    print(f"\n{'='*80}")
    print("All students completed")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
