# task_service.py

import traceback
import asyncio
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
from utils.database import SessionLocal
from utils.crud_task import task_crud
from utils.schemas_task import TaskStatus
from utils.search_service import search_service 
from utils.core_models import AITask

class TaskService:
    @staticmethod
    async def handle_file_upload(
        db: Session, 
        minio_payload: dict, 
        background_tasks: BackgroundTasks,
        should_ingest: bool = False
    ):
        try:
            task = task_crud.create_task(
                db, 
                task_type="file_search", 
                payload=minio_payload, 
                status=TaskStatus.PROCESSING
            )

            if should_ingest:
                background_tasks.add_task(TaskService.execute_worker_logic, str(task.id))
            else:
                task.status = "COMPLETED"
                task.result = {"message": "Upload only, no ingest requested"}
                db.commit()

            return {"task_id": str(task.id), "status": task.status}
        except Exception as e:
            traceback.print_exc()
            raise e

    @staticmethod
    async def handle_file_ingest(
        db: Session,
        file_key: str,
        background_tasks: BackgroundTasks
    ):
        try:
            task = task_crud.create_task(
                db, 
                task_type="file_ingest_only",
                payload={"file_key": file_key},
                status=TaskStatus.PROCESSING
            )

            background_tasks.add_task(TaskService.execute_worker_logic, str(task.id))

            return {"task_id": str(task.id), "status": task.status}

        except Exception as e:
            traceback.print_exc()
            raise e

    @staticmethod
    def execute_worker_logic(task_id: str):
        print(f"[BACKGROUND] Starting Ingest for Task {task_id}", flush=True)
        with SessionLocal() as db:
            task = db.query(AITask).filter(AITask.id == task_id).first()
            if not task: return
            try:
                file_key = task.payload.get('file_key') or task.payload.get('video_key')
                ai_result = asyncio.run(search_service.trigger_ingest(file_key))
                task.status = "COMPLETED"
                task.result = ai_result
                db.commit()
                print(f"[OK] Task {task_id} ingest completed", flush=True)

            except Exception as e:
                task.status = "FAILED"
                task.result = {"error": str(e)}
                db.commit()
                print(f"[FAILED] Task {task_id} failed: {e}", flush=True)

task_service = TaskService()