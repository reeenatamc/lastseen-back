from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.dependencies import DB, OptionalUserId
from app.models.analysis import Analysis, AnalysisStatus
from app.workers.tasks import celery_app, process_chat_upload

router = APIRouter()

ALLOWED_MIME_TYPES = {"text/plain", "application/zip", "application/json"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

Platform = Literal["whatsapp", "telegram", "imessage"]


class UploadResponse(BaseModel):
    analysis_id: int | None = None
    task_id: str | None = None
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    result: dict | None = None


@router.post("/", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_chat(
    db: DB,
    user_id: OptionalUserId,
    file: UploadFile = File(...),
    platform: Platform = Form("whatsapp"),
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type or 'unknown'}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 50 MB limit",
        )

    decoded = content.decode("utf-8", errors="replace")

    if user_id is not None:
        analysis = Analysis(
            user_id=user_id,
            platform=platform,
            original_filename=file.filename or "upload",
            status=AnalysisStatus.pending,
        )
        db.add(analysis)
        await db.commit()
        await db.refresh(analysis)

        process_chat_upload.delay(
            analysis_id=analysis.id,
            content=decoded,
            platform=platform,
        )
        return UploadResponse(analysis_id=analysis.id, status=AnalysisStatus.pending)

    task = process_chat_upload.delay(
        analysis_id=None,
        content=decoded,
        platform=platform,
    )
    return UploadResponse(task_id=task.id, status="queued")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Poll endpoint for guest users who don't have an analysis_id."""
    task = celery_app.AsyncResult(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        state=task.state,
        result=task.result if task.successful() else None,
    )
