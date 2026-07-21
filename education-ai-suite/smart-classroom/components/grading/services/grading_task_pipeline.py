from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import requests
import yaml
from PIL import Image as PILImage

from services.detection_client import (
    check_service_health,
    detect_page_layout,
    draw_detection_boxes,
    merge_overlapping_boxes,
)
from services.grade_objective_questions import grade_objective_questions
from services.grade_subjective_with_vlm import grade_subjective_with_vlm
from services.pdf_processor import render_pdf_to_images
from services.question_mapper import map_questions_to_regions
from services.subjective_question_locator import locate_subjective_questions


CheckpointCallback = Callable[[str], bool]
ProgressCallback = Callable[[str, int], None]
LogCallback = Callable[[str], None]


def _smart_classroom_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _grading_component_root() -> Path:
    # components/grading/services/grading_task_pipeline.py -> components/grading
    return Path(__file__).resolve().parents[1]


def _load_runtime_config() -> dict[str, Any]:
    """Root smart-classroom config: service basics (grading.provider, etc.)."""
    config_path = _smart_classroom_root() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("invalid root config format")
    return raw


def _load_detail_config() -> dict[str, Any]:
    """Grading component config: pipeline / ocr / detection_service / question_parsing."""
    config_path = _grading_component_root() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"grading detail config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("invalid grading detail config format")
    return raw


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _build_subjective_scoring_points(
    score: int,
    scoring_standard: dict[str, Any],
) -> list[dict[str, Any]]:
    default_points = scoring_standard.get("default_scoring_points", [])
    if isinstance(default_points, list) and default_points:
        points: list[dict[str, Any]] = []
        for idx, item in enumerate(default_points):
            if not isinstance(item, dict):
                continue
            weight = item.get("weight")
            point_score = _to_int(item.get("score"), 0)
            if point_score <= 0 and isinstance(weight, (int, float)):
                point_score = max(1, int(round(float(weight) * score)))
            points.append(
                {
                    "id": str(item.get("id") or f"criterion_{idx + 1}"),
                    "score": point_score,
                    "description": str(item.get("description") or "scoring point"),
                }
            )
        if points:
            return points

    return [
        {
            "id": "answer",
            "score": max(1, score // 2),
            "description": "Final answer correctness",
        },
        {
            "id": "process",
            "score": max(1, score - max(1, score // 2)),
            "description": "Reasoning quality and completeness",
        },
    ]


def _build_subjective_scoring_rules(
    rubric_cfg: dict[str, Any],
    scoring_standard: dict[str, Any],
) -> list[str]:
    # `scoring_criteria` (when present, either on rubric_cfg or scoring_standard
    # -- the same list is sometimes duplicated across both) is a list of
    # conditional grading rules (e.g. "no score if only the answer is given",
    # "partial credit if steps are correct but the result is wrong")
    # describing alternative outcomes, not independent additive scoring
    # dimensions. They belong here, in scoring_rules, so the VLM picks the
    # single applicable rule instead of summing every line as its own point.
    rules = (
        _to_text_list(rubric_cfg.get("scoring_criteria"))
        + _to_text_list(scoring_standard.get("scoring_criteria"))
        + _to_text_list(scoring_standard.get("scoring_principles"))
    )
    seen: set[str] = set()
    deduped: list[str] = []
    for rule in rules:
        if rule not in seen:
            seen.add(rule)
            deduped.append(rule)
    return deduped


def _convert_single_rubric_json_to_dir(
    rubric_json_path: Path,
    task_id: str,
) -> Path:
    raw = json.loads(rubric_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("rubric json must be an object")

    questions_raw = raw.get("questions", {})
    if not isinstance(questions_raw, dict) or not questions_raw:
        raise ValueError("rubric json must contain non-empty questions")

    runtime_dir = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "jobs"
        / "runtime_rubrics"
        / task_id
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)

    objective_questions: dict[str, Any] = {}
    subjective_questions: dict[str, Any] = {}

    for qid_raw, q_payload in questions_raw.items():
        qid = str(qid_raw).strip()
        if not qid:
            continue
        if not isinstance(q_payload, dict):
            continue

        rubric_cfg = q_payload.get("rubric", q_payload)
        if not isinstance(rubric_cfg, dict):
            continue

        catalog = str(rubric_cfg.get("catalog", "")).strip().lower()
        q_type = str(rubric_cfg.get("type", "")).strip() or "unknown"
        score = _to_int(rubric_cfg.get("score"), 0)
        answers = _to_text_list(rubric_cfg.get("answer"))
        alias = str(rubric_cfg.get("alias") or f"Q{qid}")

        if catalog == "objective":
            objective_questions[qid] = {
                "alias": alias,
                "type": q_type,
                "score": score,
                "format": str(rubric_cfg.get("format") or "string"),
                "match_mode": str(rubric_cfg.get("match_mode") or "any"),
                "answer": answers,
            }
            continue

        if catalog != "subjective":
            continue

        rubric_file = f"question{qid}.json"
        subjective_questions[qid] = {
            "alias": alias,
            "type": q_type,
            "score": score,
            "rubric": rubric_file,
        }

        scoring_standard = q_payload.get("scoring_standard", {})
        if not isinstance(scoring_standard, dict):
            scoring_standard = {}

        detail_rubric = {
            "question_id": alias,
            "type": q_type,
            "total_score": score,
            "question_text": str(rubric_cfg.get("question_text") or f"Question {qid}"),
            "standard_answers": answers,
            "scoring_points": _build_subjective_scoring_points(score, scoring_standard),
            "scoring_rules": _build_subjective_scoring_rules(rubric_cfg, scoring_standard),
        }
        (runtime_dir / rubric_file).write_text(
            json.dumps(detail_rubric, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    answer_key = {
        "metadata": raw.get("metadata", {}),
        "objective_questions": objective_questions,
        "subjective_questions": subjective_questions,
    }
    (runtime_dir / "answer_key.json").write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return runtime_dir


def _resolve_rubric_dir(rubric_path: str, task_id: str) -> Path:
    p = Path(rubric_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"rubric path not found: {p}")

    if p.is_dir():
        candidate = p / "answer_key.json"
        if candidate.exists():
            return p
        raise ValueError(f"rubric dir must contain answer_key.json: {p}")

    if p.suffix.lower() == ".json" and p.name != "answer_key.json":
        return _convert_single_rubric_json_to_dir(p, task_id)

    if p.name == "answer_key.json":
        return p.parent

    candidate = p.parent / "answer_key.json"
    if candidate.exists():
        return p.parent

    raise ValueError(
        "rubric_path must be a rubric directory containing answer_key.json, "
        "a path to answer_key.json, or a single rubric json file"
    )


def _build_task_paths(
    task_id: str,
    rubric_dir: Path,
    student_id: str | None,
    exam_id: str | None,
) -> dict[str, Path]:
    base_name = exam_id or rubric_dir.name
    learner = student_id or task_id
    base = Path(__file__).resolve().parents[1] / "outputs" / base_name / learner
    return {
        "base": base,
        "step1": base / "step1_layout_detection",
        "step2": base / "step2_ocr_regions",
        "step3": base / "step3_question_mapping",
        "step4": base / "step4_objective_grading",
        "step5": base / "step5_subjective_grading",
        "final": base / "grading_results.json",
    }


def _draw_question_mapping(
    question_mapping_output: Path,
    page_images: dict[int, Any],
    step3_dir: Path,
) -> None:
    """Debug-only: draw every mapped question's regions (objective + subjective)
    onto each page image, one image per page."""
    from PIL import ImageDraw, ImageFont

    if not question_mapping_output.exists():
        return

    data = json.loads(question_mapping_output.read_text(encoding="utf-8"))
    questions = data.get("questions", {})

    boxes_by_page: dict[int, list[dict[str, Any]]] = {}
    for q_id, q_data in questions.items():
        q_type = q_data.get("type", "unknown")
        for region in q_data.get("regions", []):
            page = int(region.get("page"))
            bbox = region.get("bbox")
            if not bbox:
                continue
            boxes_by_page.setdefault(page, []).append(
                {"question_id": q_id, "type": q_type, "bbox": bbox}
            )

    colors = {"objective": (0, 128, 255), "subjective": (255, 0, 128)}
    default_color = (255, 0, 0)

    for page_num, boxes in boxes_by_page.items():
        if page_num not in page_images:
            continue

        image = page_images[page_num]
        pil_img = PILImage.fromarray(image).convert("RGB") if hasattr(image, "shape") else image.convert("RGB")

        draw = ImageDraw.Draw(pil_img)
        font_size = max(16, int(pil_img.width * 0.018))
        line_width = max(3, int(pil_img.width * 0.003))
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()

        for box in boxes:
            color = colors.get(box["type"], default_color)
            x1, y1, x2, y2 = box["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            label = f"Q{box['question_id']} ({box['type']})"
            draw.rectangle([x1, y1 - font_size - 6, x1 + len(label) * font_size, y1], fill=color)
            draw.text((x1 + 2, y1 - font_size - 4), label, fill=(255, 255, 255), font=font)

        step3_dir.mkdir(parents=True, exist_ok=True)
        pil_img.save(step3_dir / f"page_{page_num}_question_mapping.jpg", quality=95)


def _run_layout_detection(
    pdf_path: Path,
    step1_dir: Path,
    config: dict[str, Any],
    detection_url: str,
    pdf_render_dpi: int,
    debug_mode: bool = False,
) -> list[dict[str, Any]]:
    detection_cfg = config.get("detection_service", {})
    if not isinstance(detection_cfg, dict):
        detection_cfg = {}

    target_labels = detection_cfg.get("target_labels", ["text", "table", "title"])
    if not isinstance(target_labels, list) or not target_labels:
        target_labels = ["text", "table", "title"]

    min_score = float(detection_cfg.get("min_score", 0.3))
    sort_boxes = bool(detection_cfg.get("sort_boxes", True))
    expand_margin = int(detection_cfg.get("expand_margin", 0))
    merge_enabled = bool(detection_cfg.get("merge_overlapping", False))
    iou_threshold = float(detection_cfg.get("iou_threshold", 0.7))

    if not check_service_health(detection_url):
        raise RuntimeError(f"layout detection service unhealthy: {detection_url}")

    pages = render_pdf_to_images(pdf_path, dpi=pdf_render_dpi)
    step1_dir.mkdir(parents=True, exist_ok=True)

    all_detections: dict[int, list[dict[str, Any]]] = {}
    for page_data in pages:
        page_num = int(page_data["page_num"])
        image = page_data["image"]
        boxes = detect_page_layout(
            page_image=image,
            service_url=detection_url,
            target_labels=target_labels,
            min_score=min_score,
            sort=sort_boxes,
            expand_margin=expand_margin,
        )

        if merge_enabled and boxes:
            boxes = merge_overlapping_boxes(boxes, iou_threshold=iou_threshold)

        all_detections[page_num] = boxes

        if debug_mode and boxes:
            draw_detection_boxes(
                image=image,
                boxes=boxes,
                output_path=step1_dir / f"page_{page_num}_detections.jpg",
            )

        (step1_dir / f"page_{page_num}_detections.json").write_text(
            json.dumps(
                {
                    "page_num": page_num,
                    "total_regions": len(boxes),
                    "boxes": boxes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    (step1_dir / "detection_summary.json").write_text(
        json.dumps(
            {
                "source_pdf": str(pdf_path),
                "total_pages": len(pages),
                "total_regions": sum(len(v) for v in all_detections.values()),
                "pages": {
                    str(page): {
                        "num_regions": len(boxes),
                        "json_path": str(step1_dir / f"page_{page}_detections.json"),
                    }
                    for page, boxes in all_detections.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return pages


def _preprocess_page_for_ocr(
    pil_img: "PILImage.Image",
    preprocess: dict[str, Any] | None,
) -> tuple["PILImage.Image", float]:
    """Apply OCR-time preprocessing to a full page image.

    Resizing (resize_ratio / max_image_width / max_image_height /
    max_image_pixels) shrinks the page and returns the applied scale factor so
    the caller can scale bboxes to match and map results back afterward.
    Enhancement (contrast / sharpness) and jpeg_quality do not affect geometry.
    Returns (processed_image, scale) where scale multiplies original coords.
    """
    from PIL import ImageEnhance

    if not preprocess:
        return pil_img, 1.0

    scale = 1.0
    ratio = preprocess.get("resize_ratio")
    if ratio and ratio != 1.0:
        scale *= float(ratio)

    w, h = pil_img.width * scale, pil_img.height * scale
    max_w = preprocess.get("max_image_width")
    if max_w and w > max_w:
        scale *= max_w / w
        w, h = pil_img.width * scale, pil_img.height * scale
    max_h = preprocess.get("max_image_height")
    if max_h and h > max_h:
        scale *= max_h / h
        w, h = pil_img.width * scale, pil_img.height * scale
    max_px = preprocess.get("max_image_pixels")
    if max_px and (w * h) > max_px:
        scale *= (max_px / (w * h)) ** 0.5

    if scale != 1.0:
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

    contrast = preprocess.get("enhance_contrast")
    if contrast and contrast != 1.0:
        pil_img = ImageEnhance.Contrast(pil_img).enhance(float(contrast))
    sharpness = preprocess.get("enhance_sharpness")
    if sharpness and sharpness != 1.0:
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(float(sharpness))

    return pil_img, scale


def _run_region_ocr(
    pages: list[dict[str, Any]],
    step1_dir: Path,
    step2_dir: Path,
    ocr_url: str,
    max_tokens: int,
    max_pixels: int,
    debug_mode: bool = False,
    preprocess: dict[str, Any] | None = None,
) -> None:
    health = requests.get(f"{ocr_url}/health", timeout=5)
    if health.status_code != 200:
        raise RuntimeError(f"ocr service unhealthy: {ocr_url} status={health.status_code}")

    step2_dir.mkdir(parents=True, exist_ok=True)

    all_ocr_results: dict[int, list[dict[str, Any]]] = {}
    total_inference_time = 0.0

    for page_data in pages:
        page_num = int(page_data["page_num"])
        image = page_data["image"]
        detection_json = step1_dir / f"page_{page_num}_detections.json"
        if not detection_json.exists():
            continue

        detection_data = json.loads(detection_json.read_text(encoding="utf-8"))
        boxes = detection_data.get("boxes", [])
        if not isinstance(boxes, list) or not boxes:
            continue

        if hasattr(image, "shape"):
            pil_img = PILImage.fromarray(image)
        else:
            pil_img = image

        pil_img, scale = _preprocess_page_for_ocr(pil_img, preprocess)

        # bboxes come from step1 (original-resolution page); scale them to the
        # possibly-resized page that is actually sent to OCR.
        regions = [
            {
                "bbox": [c * scale for c in box.get("coordinate", [])],
                "type": box.get("label", "text"),
                "region_id": f"page{page_num}_region{i+1}",
            }
            for i, box in enumerate(boxes)
        ]

        temp_image_path = step2_dir / f"temp_page_{page_num}.jpg"
        jpeg_quality = int(preprocess.get("jpeg_quality", 95)) if preprocess else 95
        pil_img.save(temp_image_path, quality=jpeg_quality)

        with temp_image_path.open("rb") as f:
            files = {"file": (f"page_{page_num}.jpg", f, "image/jpeg")}
            data = {
                "regions": json.dumps(regions),
                "max_new_tokens": max_tokens,
                "max_pixels": max_pixels,
            }
            response = requests.post(
                f"{ocr_url}/ocr/regions",
                files=files,
                data=data,
                timeout=300,
            )

        temp_image_path.unlink(missing_ok=True)

        if response.status_code != 200:
            raise RuntimeError(f"ocr api failed on page {page_num}: {response.text}")

        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"ocr failed on page {page_num}: {result.get('error')}")

        ocr_results = result.get("results", [])
        inference_time = float(result.get("total_inference_time", 0.0))
        total_inference_time += inference_time

        # Map bboxes back to original-image coordinates so downstream steps
        # (question mapping, region cropping) operate on the full-res page.
        if scale != 1.0:
            for r in ocr_results:
                bbox = r.get("bbox")
                if isinstance(bbox, list):
                    r["bbox"] = [c / scale for c in bbox]

        page_ocr_path = step2_dir / f"page_{page_num}_ocr.json"
        page_ocr_path.write_text(
            json.dumps(
                {
                    "page_num": page_num,
                    "num_regions": len(ocr_results),
                    "total_inference_time": inference_time,
                    "results": ocr_results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        all_ocr_results[page_num] = ocr_results

    (step2_dir / "ocr_summary.json").write_text(
        json.dumps(
            {
                "total_pages": len(pages),
                "total_regions": sum(len(v) for v in all_ocr_results.values()),
                "total_inference_time": total_inference_time,
                "pages": {
                    str(page_num): {
                        "num_regions": len(results),
                        "ocr_json": str(step2_dir / f"page_{page_num}_ocr.json"),
                    }
                    for page_num, results in all_ocr_results.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if debug_mode:
        text_lines: list[str] = []
        for page_num in sorted(all_ocr_results.keys()):
            text_lines.append(f"===== Page {page_num} =====")
            for result in all_ocr_results[page_num]:
                content = str(result.get("content", "")).strip()
                if content:
                    text_lines.append(content)
            text_lines.append("")
        (step2_dir / "all_pages_text.txt").write_text(
            "\n".join(text_lines), encoding="utf-8"
        )


def run_grading_pipeline(
    task_id: str,
    request_payload: dict[str, Any],
    update_progress: ProgressCallback,
    check_checkpoint: CheckpointCallback,
    log_event: LogCallback | None = None,
) -> dict[str, Any]:
    def _log(message: str) -> None:
        if log_event is not None:
            log_event(message)

    # Per-step timing. _step_start logs "step X started" and returns the start
    # time; _step_done logs completion with elapsed seconds appended.
    def _step_start(step: str) -> float:
        _log(f"step {step} started")
        return time.perf_counter()

    def _step_done(step: str, started: float, extra: str = "") -> None:
        elapsed = time.perf_counter() - started
        suffix = f" {extra}" if extra else ""
        _log(f"step {step} completed elapsed={elapsed:.2f}s{suffix}")

    paper_path = Path(str(request_payload["paper_path"])).resolve()
    rubric_dir = _resolve_rubric_dir(str(request_payload["rubric_path"]), task_id=task_id)
    _log(f"resolved inputs paper_path={paper_path} rubric_dir={rubric_dir}")

    # Basic/service config (provider URLs) from the root smart-classroom config.
    root_config = _load_runtime_config()
    grading_cfg = root_config.get("grading", {})
    if not isinstance(grading_cfg, dict):
        grading_cfg = {}

    provider_cfg = grading_cfg.get("provider", {})
    if not isinstance(provider_cfg, dict):
        provider_cfg = {}

    # Detail config (pipeline / ocr / detection_service / question_parsing) from
    # the grading component config.yaml.
    detail_config = _load_detail_config()

    pipeline_cfg = detail_config.get("pipeline", {})
    if not isinstance(pipeline_cfg, dict):
        pipeline_cfg = {}

    ocr_cfg = detail_config.get("ocr", {})
    if not isinstance(ocr_cfg, dict):
        ocr_cfg = {}

    options = request_payload.get("options", {})
    if not isinstance(options, dict):
        options = {}

    skip_subjective = bool(options.get("skip_subjective", pipeline_cfg.get("skip_subjective", False)))
    subjective_use_rubric = bool(options.get("subjective_use_rubric", pipeline_cfg.get("subjective_use_rubric", True)))
    debug_mode = bool(options.get("debug_mode", pipeline_cfg.get("debug_mode", False)))
    pdf_render_dpi = int(options.get("pdf_render_dpi", pipeline_cfg.get("pdf_render_dpi", 300)))
    ocr_max_tokens = int(options.get("ocr_max_tokens", ocr_cfg.get("max_tokens", 1280)))
    ocr_max_pixels = int(options.get("ocr_max_pixels", ocr_cfg.get("max_pixels", 10000000)))

    # Image preprocessing applied to the page sent to OCR (bboxes are scaled
    # to match; results are mapped back to original-image coordinates).
    def _opt_num(key: str) -> float | None:
        val = options.get(key, ocr_cfg.get(key))
        return float(val) if val is not None else None

    ocr_preprocess = {
        "jpeg_quality": int(options.get("jpeg_quality", ocr_cfg.get("jpeg_quality", 85))),
        "resize_ratio": _opt_num("resize_ratio"),
        "max_image_width": _opt_num("max_image_width"),
        "max_image_height": _opt_num("max_image_height"),
        "max_image_pixels": _opt_num("max_image_pixels"),
        "enhance_contrast": _opt_num("enhance_contrast"),
        "enhance_sharpness": _opt_num("enhance_sharpness"),
    }

    layout_url = str(options.get("layout_detection_url") or provider_cfg.get("layout_detection", "http://127.0.0.1:9902"))
    ocr_url = str(options.get("ocr_api_url") or provider_cfg.get("ocr_provider", "http://127.0.0.1:9901"))
    vlm_url = str(options.get("vlm_api_url") or provider_cfg.get("vlm_provider", "http://127.0.0.1:9900"))
    grading_language = str(options.get("language") or grading_cfg.get("language", "en"))
    grading_subject = options.get("subject") or request_payload.get("exam_id") or None
    _log(
        "providers "
        f"layout_detection={layout_url} ocr_provider={ocr_url} vlm_provider={vlm_url} "
        f"language={grading_language}"
    )

    task_paths = _build_task_paths(
        task_id=task_id,
        rubric_dir=rubric_dir,
        student_id=request_payload.get("student_id"),
        exam_id=request_payload.get("exam_id"),
    )
    for key in ["base", "step1", "step2", "step3", "step4", "step5"]:
        task_paths[key].mkdir(parents=True, exist_ok=True)
    _log(f"task output base_dir={task_paths['base']}")

    answer_key_path = rubric_dir / "answer_key.json"

    update_progress("validate_inputs", 10)
    _t = _step_start("validate_inputs")

    if not paper_path.exists():
        raise FileNotFoundError(f"paper file not found: {paper_path}")
    if not answer_key_path.exists():
        raise FileNotFoundError(f"answer key not found in rubric dir: {answer_key_path}")
    _step_done("validate_inputs", _t)

    update_progress("layout_detection", 20)
    _t = _step_start("layout_detection")
    pages = _run_layout_detection(
        pdf_path=paper_path,
        step1_dir=task_paths["step1"],
        config=detail_config,
        detection_url=layout_url,
        pdf_render_dpi=pdf_render_dpi,
        debug_mode=debug_mode,
    )
    _step_done("layout_detection", _t, f"pages={len(pages)}")

    if check_checkpoint("after_layout_detection"):
        _log("checkpoint stop after_layout_detection")
        return {"stopped": True}

    update_progress("ocr_inference", 40)
    _t = _step_start("ocr_inference")
    _run_region_ocr(
        pages=pages,
        step1_dir=task_paths["step1"],
        step2_dir=task_paths["step2"],
        ocr_url=ocr_url,
        max_tokens=ocr_max_tokens,
        max_pixels=ocr_max_pixels,
        debug_mode=debug_mode,
        preprocess=ocr_preprocess,
    )
    _step_done("ocr_inference", _t)

    if check_checkpoint("after_ocr_inference"):
        _log("checkpoint stop after_ocr_inference")
        return {"stopped": True}

    update_progress("question_mapping", 55)
    _t = _step_start("question_mapping")
    question_mapping_output = task_paths["step3"] / "question_mapping.json"
    mapping_result = map_questions_to_regions(
        ocr_dir=task_paths["step2"],
        answer_key_path=answer_key_path,
        output_path=question_mapping_output,
        strategy="position",
        config=detail_config,
    )
    _step_done(
        "question_mapping",
        _t,
        f"total={mapping_result.get('total_questions', 0)} "
        f"objective={mapping_result.get('objective_questions', 0)} "
        f"subjective={mapping_result.get('subjective_questions', 0)}",
    )

    page_images = {int(p["page_num"]): p["image"] for p in pages}

    if debug_mode:
        _draw_question_mapping(
            question_mapping_output=question_mapping_output,
            page_images=page_images,
            step3_dir=task_paths["step3"],
        )

    subjective_regions_output = task_paths["step3"] / "subjective_regions.json"
    subjective_region_result: dict[str, Any] | None = None
    if int(mapping_result.get("subjective_questions", 0)) > 0:
        _t = _step_start("subjective_regioning")
        subjective_region_result = locate_subjective_questions(
            ocr_dir=task_paths["step2"],
            answer_key_path=answer_key_path,
            page_images=page_images,
            output_dir=task_paths["step3"],
            margin_left=50,
            margin_right=50,
            visualize=debug_mode,
            config=detail_config,
        )
        _step_done("subjective_regioning", _t)

    if check_checkpoint("after_question_mapping"):
        _log("checkpoint stop after_question_mapping")
        return {"stopped": True}

    update_progress("objective_grading", 70)
    _t = _step_start("objective_grading")
    objective_results = grade_objective_questions(
        ocr_dir=task_paths["step2"],
        answer_key_path=answer_key_path,
        output_dir=task_paths["step4"],
        config=detail_config,
        debug_mode=debug_mode,
        question_mapping_path=question_mapping_output,
        page_images=page_images,
    )
    _step_done(
        "objective_grading",
        _t,
        f"score={objective_results.get('total_score', 0)}"
        f"/{objective_results.get('total_possible_score', 0)}",
    )

    if check_checkpoint("after_objective_grading"):
        _log("checkpoint stop after_objective_grading")
        return {"stopped": True}

    subjective_results: dict[str, Any] | None = None
    if not skip_subjective and subjective_regions_output.exists():
        update_progress("subjective_grading", 85)
        _t = _step_start("subjective_grading")
        subjective_results = grade_subjective_with_vlm(
            subjective_regions_path=subjective_regions_output,
            answer_key_path=answer_key_path,
            rubric_dir=rubric_dir,
            page_images=page_images,
            output_dir=task_paths["step5"],
            vlm_api_url=vlm_url,
            student_id=str(request_payload.get("student_id") or task_id),
            language=grading_language,
            subject=grading_subject,
            use_rubric=subjective_use_rubric,
            debug_mode=debug_mode,
        )
        _step_done(
            "subjective_grading",
            _t,
            f"score={subjective_results.get('total_subjective_score', 0)}"
            f"/{subjective_results.get('max_subjective_score', 0)}",
        )
        if check_checkpoint("after_subjective_grading"):
            _log("checkpoint stop after_subjective_grading")
            return {"stopped": True}
    elif skip_subjective:
        _log("step subjective_grading skipped by options")
    else:
        _log("step subjective_grading skipped because subjective_regions.json missing")

    update_progress("merge_results", 95)
    _t = _step_start("merge_results")

    summary = {
        "objective_score": objective_results.get("total_score", 0) if objective_results else 0,
        "objective_max": objective_results.get("total_possible_score", 0) if objective_results else 0,
        "subjective_score": subjective_results.get("total_subjective_score", 0) if subjective_results else 0,
        "subjective_max": subjective_results.get("max_subjective_score", 0) if subjective_results else 0,
    }
    summary["total_score"] = int(summary["objective_score"]) + int(summary["subjective_score"])
    summary["total_max"] = int(summary["objective_max"]) + int(summary["subjective_max"])

    result_payload: dict[str, Any] = {
        "task_id": task_id,
        "status": "COMPLETED",
        "message": "grading pipeline completed",
        "input": {
            "paper_path": str(paper_path),
            "rubric_dir": str(rubric_dir),
            "answer_key_path": str(answer_key_path),
            "student_id": request_payload.get("student_id"),
            "exam_id": request_payload.get("exam_id"),
            "options": request_payload.get("options", {}),
        },
        "providers": {
            "layout_detection": layout_url,
            "ocr_provider": ocr_url,
            "vlm_provider": vlm_url,
        },
        "mapping": {
            "total_questions": mapping_result.get("total_questions", 0),
            "objective_questions": mapping_result.get("objective_questions", 0),
            "subjective_questions": mapping_result.get("subjective_questions", 0),
        },
        "subjective_regioning": subjective_region_result,
        "objective": objective_results,
        "subjective": subjective_results,
        "summary": summary,
        "artifacts": {
            "base_dir": str(task_paths["base"]),
            "step1_layout_detection": str(task_paths["step1"]),
            "step2_ocr_regions": str(task_paths["step2"]),
            "step3_question_mapping": str(task_paths["step3"]),
            "step4_objective_grading": str(task_paths["step4"]),
            "step5_subjective_grading": str(task_paths["step5"]),
        },
        "steps": [
            "validate_inputs",
            "layout_detection",
            "ocr_inference",
            "question_mapping",
            "objective_grading",
            "subjective_grading",
            "merge_results",
        ],
    }

    task_paths["final"].write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _step_done("merge_results", _t, f"final_result={task_paths['final']}")

    return {
        "stopped": False,
        "result_path": str(task_paths["final"]),
        "summary": summary,
    }
