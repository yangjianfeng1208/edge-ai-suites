from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from utils.database import get_db
from utils.crud_task import task_crud
from uuid import UUID
from utils.core_models import AITask
from utils.core_responses import resp_200

from providers.chromadb_wrapper.chroma_client import ChromaClientWrapper

chroma_db = ChromaClientWrapper()

router = APIRouter()
@router.get("/list-ids")
async def list_ids(collection_name: str):
    # 直接调用 Wrapper 的方法
    results = chroma_db.query_all(collection_name=collection_name)
    return {"ids": [item['id'] for item in results]}