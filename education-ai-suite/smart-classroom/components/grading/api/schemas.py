from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    language: str


class RubricGenerateRequest(BaseModel):
    input_path: str
    output_path: str
    question_key: str = "rubric"
    subjective_template_path: str | None = None


class RubricGenerateResponse(BaseModel):
    status: str
    output_path: str
    total_questions: int
    metadata_included: bool
    template_applied: bool


class GradingJobCreateRequest(BaseModel):
    input_path: str
    output_path: str
    question_key: str = "rubric"
    subjective_template_path: str | None = None


class GradingJobCreateResponse(BaseModel):
    job_id: str
    status: str
    current_step: str
    progress: int
    created_at: str


class GradingJobStatusResponse(BaseModel):
    job_id: str
    task_type: str
    status: str
    current_step: str
    progress: int
    error_message: str | None = None
    created_at: str
    updated_at: str


class GradingJobResultResponse(BaseModel):
    job_id: str
    status: str
    result: dict


class GradingTaskCreateRequest(BaseModel):
    task_type: str = "grading.run"
    payload: dict | None = None
    paper_path: str | None = None
    rubric_path: str | None = None
    student_id: str | None = None
    exam_id: str | None = None
    options: dict = {}


class GradingTaskCreateResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    current_step: str
    progress: int
    created_at: str
    log_path: str | None = None


class GradingTaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    current_step: str
    progress: int
    error_message: str | None = None
    created_at: str
    updated_at: str
    log_path: str | None = None


class GradingTaskResultResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    result: dict
    log_path: str | None = None


class GradingTaskControlResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    current_step: str
    progress: int
    control_action: str | None = None
    updated_at: str
    log_path: str | None = None


class UnifiedTaskCreateRequest(BaseModel):
    task_type: str
    payload: dict


class UnifiedTaskCreateResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    current_step: str
    progress: int
    created_at: str


class UnifiedTaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    current_step: str
    progress: int
    error_message: str | None = None
    created_at: str
    updated_at: str


class UnifiedTaskResultResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    result: dict


class UnifiedTaskControlResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    current_step: str
    progress: int
    control_action: str | None = None
    updated_at: str
