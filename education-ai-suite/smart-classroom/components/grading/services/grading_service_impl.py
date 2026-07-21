from __future__ import annotations

import json
import time
import traceback
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

from services.job_store import JsonJobStore
from services.grading_task_pipeline import run_grading_pipeline
from services.rubric_generator import generate_rubrics_file as _generate_rubrics_file


def get_health(language: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "grading",
        "language": language,
    }


def generate_rubrics_file(
    input_path: str,
    output_path: str,
    question_key: str = "rubric",
    subjective_template_path: str | None = None,
) -> dict[str, Any]:
    default_template = (
        Path(__file__).resolve().parent / "templates" / "subjective_default.en.json"
    )

    return _generate_rubrics_file(
        input_path=input_path,
        output_path=output_path,
        question_key=question_key,
        subjective_template_path=subjective_template_path,
        default_subjective_template_path=str(default_template),
    )


_JOB_STORE = JsonJobStore(
    Path(__file__).resolve().parents[1] / "outputs" / "jobs" / "job_store.json"
)

_ACTIVE_TASK_STATUSES = {"PENDING", "RUNNING", "PAUSING", "PAUSED", "CANCELLING"}
_TERMINAL_TASK_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}
_RUBRIC_TASK_TYPES = {"rubric.generate", "rubrics_generate"}
_GRADING_TASK_TYPES = {"grading.run", "grading_task", "grading.batch"}
_SUPPORTED_TASK_TYPES = {"rubric.generate", "grading.run"}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_logs_dir() -> Path:
    logs_dir = Path(__file__).resolve().parents[1] / "outputs" / "jobs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _task_log_path(task_id: str, task_type: str) -> Path:
    safe_type = task_type.replace(".", "_")
    return _task_logs_dir() / f"{safe_type}_{task_id}.log"


def _append_task_log(task_id: str, task_type: str, message: str) -> None:
    log_path = _task_log_path(task_id, task_type)
    line = f"[{_now_utc_iso()}] {message}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _append_task_exception(task_id: str, task_type: str, exc: Exception) -> None:
    _append_task_log(task_id, task_type, f"ERROR: {exc}")
    for line in traceback.format_exc().strip().splitlines():
        _append_task_log(task_id, task_type, line)


def _build_folder_submission_key(paper_path: str, student_id: str | None) -> str:
    # Temporary strategy: folder name is treated as unique submission identity.
    if student_id and str(student_id).strip():
        return str(student_id).strip()
    return Path(str(paper_path)).resolve().parent.name


def _should_reuse_existing_task(existing: dict[str, Any], options: dict[str, Any] | None) -> bool:
    opts = options if isinstance(options, dict) else {}
    if bool(opts.get("force_regrade", False)):
        return False

    status = str(existing.get("status", ""))
    return status in _ACTIVE_TASK_STATUSES


def _list_pdf_files_under_dir(paper_dir: Path) -> list[Path]:
    pdfs = sorted(p for p in paper_dir.rglob("*.pdf") if p.is_file())
    return pdfs


def _run_grading_batch_task(batch_task_id: str, request_payload: dict[str, Any]) -> None:
    _append_task_log(batch_task_id, "grading.batch", "batch task started")
    try:
        paper_dir = Path(str(request_payload["paper_path"])).resolve()
        rubric_path = str(request_payload["rubric_path"])
        exam_id = request_payload.get("exam_id")
        options = request_payload.get("options", {})
        if not isinstance(options, dict):
            options = {}

        if not paper_dir.exists() or not paper_dir.is_dir():
            raise FileNotFoundError(f"paper directory not found: {paper_dir}")

        pdf_files = _list_pdf_files_under_dir(paper_dir)
        if not pdf_files:
            raise ValueError(f"no pdf files found under directory: {paper_dir}")

        _JOB_STORE.update_job(
            batch_task_id,
            status="RUNNING",
            current_step="sequential_processing",
            progress=5,
        )
        _append_task_log(batch_task_id, "grading.batch", f"found pdf files count={len(pdf_files)}")

        child_tasks: list[dict[str, Any]] = []
        child_options = options.get("child_options", options)
        if not isinstance(child_options, dict):
            child_options = {}

        total = len(pdf_files)
        done = 0
        succeeded = 0
        failed = 0
        cancelled = 0

        for idx, pdf in enumerate(pdf_files, start=1):
            student_folder = pdf.parent.name
            child_info: dict[str, Any] = {
                "student_id": student_folder,
                "paper_path": str(pdf),
                "status": "RUNNING",
            }
            child_tasks.append(child_info)

            current_progress = 5 + int(((idx - 1) / total) * 90)
            _JOB_STORE.update_job(
                batch_task_id,
                status="RUNNING",
                current_step=f"sequential_processing:{student_folder}",
                progress=current_progress,
                result={
                    "total_children": total,
                    "done_children": done,
                    "succeeded_children": succeeded,
                    "failed_children": failed,
                    "cancelled_children": cancelled,
                    "children": child_tasks,
                },
            )
            _append_task_log(
                batch_task_id,
                "grading.batch",
                f"start student={student_folder} ({idx}/{total})",
            )

            try:
                pipeline_result = run_grading_pipeline(
                    task_id=batch_task_id,
                    request_payload={
                        "paper_path": str(pdf),
                        "rubric_path": rubric_path,
                        "student_id": student_folder,
                        "exam_id": exam_id,
                        "options": child_options,
                    },
                    update_progress=lambda _step, _progress: None,
                    check_checkpoint=lambda _checkpoint: False,
                    log_event=lambda message: _append_task_log(
                        batch_task_id,
                        "grading.batch",
                        f"student={student_folder} {message}",
                    ),
                )

                if pipeline_result.get("stopped"):
                    cancelled += 1
                    child_info["status"] = "CANCELLED"
                else:
                    succeeded += 1
                    child_info["status"] = "COMPLETED"
                    child_info["result_path"] = str(pipeline_result.get("result_path", ""))
                    child_info["summary"] = pipeline_result.get("summary", {})
            except Exception as exc:
                failed += 1
                child_info["status"] = "FAILED"
                child_info["error_message"] = str(exc)
                _append_task_log(
                    batch_task_id,
                    "grading.batch",
                    f"student={student_folder} failed error={exc}",
                )
                for line in traceback.format_exc().strip().splitlines():
                    _append_task_log(batch_task_id, "grading.batch", line)

            done += 1
            current_progress = 5 + int((done / total) * 90)
            _JOB_STORE.update_job(
                batch_task_id,
                status="RUNNING",
                current_step="sequential_processing",
                progress=current_progress,
                result={
                    "total_children": total,
                    "done_children": done,
                    "succeeded_children": succeeded,
                    "failed_children": failed,
                    "cancelled_children": cancelled,
                    "children": child_tasks,
                },
            )

        final_result = {
            "total_children": total,
            "succeeded_children": succeeded,
            "failed_children": failed,
            "cancelled_children": cancelled,
            "children": child_tasks,
            "log_path": str(_task_log_path(batch_task_id, "grading.batch")),
        }

        final_status = "COMPLETED" if failed == 0 else "FAILED"
        _JOB_STORE.update_job(
            batch_task_id,
            status=final_status,
            current_step="completed" if final_status == "COMPLETED" else "failed",
            progress=100,
            result=final_result,
            error_message=None if final_status == "COMPLETED" else "one or more child tasks failed",
        )
        _append_task_log(
            batch_task_id,
            "grading.batch",
            f"batch finished status={final_status} total={total} success={succeeded} failed={failed}",
        )
    except Exception as exc:
        _append_task_exception(batch_task_id, "grading.batch", exc)
        _JOB_STORE.update_job(
            batch_task_id,
            status="FAILED",
            current_step="failed",
            progress=100,
            error_message=str(exc),
        )


def create_grading_batch_task(
    paper_dir_path: str,
    rubric_path: str,
    exam_id: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paper_dir = Path(str(paper_dir_path)).resolve()
    submission_key = f"batch:{paper_dir.name}"
    options_obj = options if isinstance(options, dict) else {}

    existing = _JOB_STORE.find_latest_job(
        task_type="grading.batch",
        request_field="submission_key",
        request_value=submission_key,
    )
    if existing is not None and _should_reuse_existing_task(existing, options_obj):
        return existing

    payload = {
        "paper_path": str(paper_dir),
        "rubric_path": rubric_path,
        "student_id": None,
        "exam_id": exam_id,
        "submission_key": submission_key,
        "options": options_obj,
    }
    task = _JOB_STORE.create_job(task_type="grading.batch", request_payload=payload)
    log_path = _task_log_path(task["job_id"], "grading.batch")
    log_path.write_text("", encoding="utf-8")
    task = _JOB_STORE.update_job(task["job_id"], log_path=str(log_path))
    _append_task_log(task["job_id"], "grading.batch", "batch task created")

    worker = Thread(target=_run_grading_batch_task, args=(task["job_id"], payload), daemon=True)
    worker.start()
    return task


def _run_rubrics_job(job_id: str, request_payload: dict[str, Any]) -> None:
    _append_task_log(job_id, "rubric.generate", "task started")
    _JOB_STORE.update_job(
        job_id,
        status="RUNNING",
        current_step="rubrics_generation",
        progress=10,
    )
    try:
        _append_task_log(job_id, "rubric.generate", "running rubrics generation")
        result = generate_rubrics_file(
            input_path=request_payload["input_path"],
            output_path=request_payload["output_path"],
            question_key=request_payload.get("question_key", "rubric"),
            subjective_template_path=request_payload.get("subjective_template_path"),
        )
        _JOB_STORE.update_job(
            job_id,
            status="COMPLETED",
            current_step="completed",
            progress=100,
            result=result,
            error_message=None,
        )
        _append_task_log(job_id, "rubric.generate", "task completed")
    except Exception as exc:
        _append_task_exception(job_id, "rubric.generate", exc)
        _JOB_STORE.update_job(
            job_id,
            status="FAILED",
            current_step="failed",
            progress=100,
            error_message=str(exc),
        )


def create_grading_job(
    input_path: str,
    output_path: str,
    question_key: str = "rubric",
    subjective_template_path: str | None = None,
) -> dict[str, Any]:
    payload = {
        "input_path": input_path,
        "output_path": output_path,
        "question_key": question_key,
        "subjective_template_path": subjective_template_path,
    }
    job = _JOB_STORE.create_job(task_type="rubric.generate", request_payload=payload)
    log_path = _task_log_path(job["job_id"], "rubric.generate")
    log_path.write_text("", encoding="utf-8")
    job = _JOB_STORE.update_job(job["job_id"], log_path=str(log_path))
    _append_task_log(job["job_id"], "rubric.generate", "task created")
    worker = Thread(target=_run_rubrics_job, args=(job["job_id"], payload), daemon=True)
    worker.start()
    return job


def get_grading_job_status(job_id: str) -> dict[str, Any]:
    job = _JOB_STORE.get_job(job_id)
    if job.get("task_type") not in _RUBRIC_TASK_TYPES:
        raise ValueError(f"task is not a rubric.generate job: {job_id}")
    return job


def get_grading_job_result(job_id: str) -> dict[str, Any]:
    job = get_grading_job_status(job_id)
    status = job.get("status")
    if status != "COMPLETED":
        raise RuntimeError(f"job not completed, current status: {status}")

    result = job.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("job completed but result is missing")
    return {"job_id": job_id, "status": status, "result": result}


def _simulate_external_call(seconds: float = 1.2) -> None:
    # External OCR/VLM calls are treated as atomic units.
    time.sleep(seconds)


def _handle_task_control_checkpoint(task_id: str, checkpoint_step: str) -> bool:
    task = _JOB_STORE.get_job(task_id)
    task_type = str(task.get("task_type", "grading.run"))
    action = task.get("control_action")

    if action == "cancel":
        _append_task_log(task_id, task_type, f"checkpoint={checkpoint_step} action=cancel applied")
        _JOB_STORE.set_control_action(task_id, None)
        _JOB_STORE.update_job(
            task_id,
            status="CANCELLED",
            current_step="cancelled",
            checkpoint_step=checkpoint_step,
            progress=100,
            error_message=None,
            result=None,
        )
        return True

    if action == "pause":
        _append_task_log(task_id, task_type, f"checkpoint={checkpoint_step} action=pause applied")
        _JOB_STORE.set_control_action(task_id, None)
        _JOB_STORE.update_job(
            task_id,
            status="PAUSED",
            current_step=f"paused:{checkpoint_step}",
            checkpoint_step=checkpoint_step,
        )

        while True:
            latest = _JOB_STORE.get_job(task_id)
            latest_status = latest.get("status")
            latest_action = latest.get("control_action")

            if latest_action == "cancel":
                _append_task_log(task_id, task_type, f"paused checkpoint={checkpoint_step} action=cancel applied")
                _JOB_STORE.set_control_action(task_id, None)
                _JOB_STORE.update_job(
                    task_id,
                    status="CANCELLED",
                    current_step="cancelled",
                    checkpoint_step=checkpoint_step,
                    progress=100,
                    error_message=None,
                    result=None,
                )
                return True

            if latest_status == "RUNNING":
                _append_task_log(task_id, task_type, f"resumed from checkpoint={checkpoint_step}")
                _JOB_STORE.update_job(task_id, current_step=f"resumed:{checkpoint_step}")
                return False

            if latest_status in _TERMINAL_TASK_STATUSES:
                return True

            time.sleep(0.2)

    return False


def _run_grading_task(task_id: str, request_payload: dict[str, Any]) -> None:
    _append_task_log(task_id, "grading.run", "task started")
    try:
        def _progress(step: str, progress: int) -> None:
            _append_task_log(task_id, "grading.run", f"progress step={step} value={progress}")
            _JOB_STORE.update_job(
                task_id,
                status="RUNNING",
                current_step=step,
                progress=progress,
            )

        pipeline_result = run_grading_pipeline(
            task_id=task_id,
            request_payload=request_payload,
            update_progress=_progress,
            check_checkpoint=lambda checkpoint: _handle_task_control_checkpoint(task_id, checkpoint),
            log_event=lambda message: _append_task_log(task_id, "grading.run", message),
        )

        if pipeline_result.get("stopped"):
            _append_task_log(task_id, "grading.run", "task stopped at checkpoint")
            return

        _JOB_STORE.update_job(
            task_id,
            status="COMPLETED",
            current_step="completed",
            progress=100,
            result={
                "result_path": str(pipeline_result["result_path"]),
                "summary": pipeline_result["summary"],
                "log_path": str(_task_log_path(task_id, "grading.run")),
            },
            error_message=None,
        )
        _append_task_log(task_id, "grading.run", "task completed")
    except Exception as exc:
        _append_task_exception(task_id, "grading.run", exc)
        _JOB_STORE.update_job(
            task_id,
            status="FAILED",
            current_step="failed",
            progress=100,
            error_message=str(exc),
        )


def create_grading_task(
    paper_path: str,
    rubric_path: str,
    student_id: str | None = None,
    exam_id: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    submission_key = _build_folder_submission_key(paper_path=paper_path, student_id=student_id)
    options_obj = options if isinstance(options, dict) else {}

    existing = _JOB_STORE.find_latest_job(
        task_type="grading.run",
        request_field="submission_key",
        request_value=submission_key,
    )
    if existing is not None and _should_reuse_existing_task(existing, options_obj):
        return existing

    payload = {
        "paper_path": paper_path,
        "rubric_path": rubric_path,
        "student_id": student_id,
        "exam_id": exam_id,
        "submission_key": submission_key,
        "options": options_obj,
    }
    task = _JOB_STORE.create_job(task_type="grading.run", request_payload=payload)
    log_path = _task_log_path(task["job_id"], "grading.run")
    log_path.write_text("", encoding="utf-8")
    task = _JOB_STORE.update_job(task["job_id"], log_path=str(log_path))
    _append_task_log(task["job_id"], "grading.run", "task created")
    worker = Thread(target=_run_grading_task, args=(task["job_id"], payload), daemon=True)
    worker.start()
    return task


def get_grading_task_status(task_id: str) -> dict[str, Any]:
    task = _JOB_STORE.get_job(task_id)
    if task.get("task_type") not in _GRADING_TASK_TYPES:
        raise ValueError(f"task is not a grading.run task: {task_id}")
    return task


def get_grading_task_result(task_id: str) -> dict[str, Any]:
    task = get_grading_task_status(task_id)
    status = task.get("status")
    if status != "COMPLETED":
        raise RuntimeError(f"task not completed, current status: {status}")

    result = task.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("task completed but result is missing")

    return {
        "task_id": task_id,
        "status": status,
        "result": result,
    }


def request_grading_task_pause(task_id: str) -> dict[str, Any]:
    task = get_grading_task_status(task_id)
    status = task.get("status")

    if status == "RUNNING":
        _JOB_STORE.set_control_action(task_id, "pause")
        return _JOB_STORE.update_job(task_id, status="PAUSING", current_step="pause_requested")
    if status == "PAUSING":
        return _JOB_STORE.get_job(task_id)
    if status == "PAUSED":
        return _JOB_STORE.get_job(task_id)
    if status == "PENDING":
        _JOB_STORE.set_control_action(task_id, "pause")
        return _JOB_STORE.update_job(task_id, status="PAUSING", current_step="pause_requested")

    raise RuntimeError(f"pause not allowed in current status: {status}")


def request_grading_task_resume(task_id: str) -> dict[str, Any]:
    task = get_grading_task_status(task_id)
    status = task.get("status")

    if status == "PAUSED":
        _JOB_STORE.set_control_action(task_id, None)
        return _JOB_STORE.update_job(task_id, status="RUNNING", current_step="resume_requested")
    if status == "RUNNING":
        return _JOB_STORE.get_job(task_id)
    if status == "PAUSING":
        raise RuntimeError("task is pausing, retry resume after it reaches PAUSED")

    raise RuntimeError(f"resume not allowed in current status: {status}")


def request_grading_task_cancel(task_id: str) -> dict[str, Any]:
    task = get_grading_task_status(task_id)
    status = task.get("status")

    if status in {"RUNNING", "PAUSING", "PAUSED", "PENDING"}:
        _JOB_STORE.set_control_action(task_id, "cancel")
        return _JOB_STORE.update_job(task_id, status="CANCELLING", current_step="cancel_requested")
    if status in _TERMINAL_TASK_STATUSES:
        return _JOB_STORE.get_job(task_id)
    if status == "CANCELLING":
        return _JOB_STORE.get_job(task_id)

    raise RuntimeError(f"cancel not allowed in current status: {status}")


def create_task(task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task_type}")

    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    if task_type == "rubric.generate":
        if not payload.get("input_path") or not payload.get("output_path"):
            raise ValueError("rubric.generate payload requires input_path and output_path")
        return create_grading_job(
            input_path=str(payload.get("input_path", "")),
            output_path=str(payload.get("output_path", "")),
            question_key=str(payload.get("question_key", "rubric")),
            subjective_template_path=payload.get("subjective_template_path"),
        )

    if task_type == "grading.run":
        if not payload.get("paper_path") or not payload.get("rubric_path"):
            raise ValueError("grading.run payload requires paper_path and rubric_path")

        paper_path_obj = Path(str(payload.get("paper_path", ""))).resolve()
        if paper_path_obj.exists() and paper_path_obj.is_dir():
            return create_grading_batch_task(
                paper_dir_path=str(paper_path_obj),
                rubric_path=str(payload.get("rubric_path", "")),
                exam_id=payload.get("exam_id"),
                options=payload.get("options", {}),
            )

        return create_grading_task(
            paper_path=str(payload.get("paper_path", "")),
            rubric_path=str(payload.get("rubric_path", "")),
            student_id=payload.get("student_id"),
            exam_id=payload.get("exam_id"),
            options=payload.get("options", {}),
        )

    raise ValueError(f"unsupported task_type: {task_type}")


def get_task_status(task_id: str) -> dict[str, Any]:
    return _JOB_STORE.get_job(task_id)


def get_task_result(task_id: str) -> dict[str, Any]:
    task = get_task_status(task_id)
    status = task.get("status")
    if status != "COMPLETED":
        raise RuntimeError(f"task not completed, current status: {status}")

    result = task.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("task completed but result is missing")

    return {
        "task_id": task_id,
        "task_type": str(task.get("task_type", "")),
        "status": status,
        "result": result,
    }


def request_task_pause(task_id: str) -> dict[str, Any]:
    task = get_task_status(task_id)
    if task.get("task_type") not in {"grading.run", "grading_task"}:
        raise RuntimeError("pause is supported only for grading.run tasks")
    return request_grading_task_pause(task_id)


def request_task_resume(task_id: str) -> dict[str, Any]:
    task = get_task_status(task_id)
    if task.get("task_type") not in {"grading.run", "grading_task"}:
        raise RuntimeError("resume is supported only for grading.run tasks")
    return request_grading_task_resume(task_id)


def request_task_cancel(task_id: str) -> dict[str, Any]:
    task = get_task_status(task_id)
    if task.get("task_type") not in {"grading.run", "grading_task"}:
        raise RuntimeError("cancel is supported only for grading.run tasks")
    return request_grading_task_cancel(task_id)
