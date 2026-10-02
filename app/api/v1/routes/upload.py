from datetime import date
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.models.user import User
from app.core.dependencies import DB, OptionalUserId
from app.core.ratelimit import UPLOAD_LIMIT, limiter
from app.models.analysis import Analysis, AnalysisStatus
from app.services.credits import refund_analysis_credit_async, spend_credit
from app.services.access import LOCKED_SECTIONS, build_preview, build_teaser
from app.workers.tasks import celery_app, process_chat_upload

router = APIRouter()

ALLOWED_MIME_TYPES = {"text/plain", "application/zip", "application/json"}
# 20 MB: a multi-year chat export is a few MB of text; more is abuse, and the whole
# file is held in memory (and again as the task payload) while it is processed
MAX_FILE_SIZE = 20 * 1024 * 1024
_READ_CHUNK = 1024 * 1024  # read in 1 MB pieces so an oversized body is cut early

Platform = Literal["whatsapp", "telegram", "imessage"]
Language = Literal["auto", "es", "en"]

async def read_capped(file: UploadFile, limit: int) -> bytes:
    """Read an upload in chunks, raising 413 as soon as it exceeds `limit` bytes."""
    buffer = bytearray()
    while chunk := await file.read(_READ_CHUNK):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds {limit // (1024 * 1024)} MB limit",
            )
    return bytes(buffer)


class UploadResponse(BaseModel):
    analysis_id: int | None = None
    task_id: str | None = None
    status: str
    tier: Literal["preview", "full"]


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    result: dict | None = None


def _enqueue(
    *, analysis_id: int | None, tier: str, content: str, platform: str, language: str,
    date_from: str | None, date_to: str | None,
):
    """Queue the pipeline task without letting chat content reach the worker logs.

    Celery prints the task kwargs in its "received" line. The explicit reprs
    replace them with fixed text that carries no chat content.
    """
    return process_chat_upload.apply_async(
        kwargs={
            "analysis_id": analysis_id,
            "content": content,
            "platform": platform,
            "language": language,
            "date_from": date_from,
            "date_to": date_to,
            "tier": tier,
        },
        argsrepr="()",
        kwargsrepr=f"{{analysis_id: {analysis_id}, tier: {tier}, content: <redacted>}}",
    )


@router.post("/", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(UPLOAD_LIMIT)  # keyed by user when the JWT is valid, by client IP otherwise
async def upload_chat(
    request: Request,
    db: DB,
    user_id: OptionalUserId,
    file: UploadFile = File(...),
    platform: Platform = Form("whatsapp"),
    language: Language = Form("auto"),
    date_from: date | None = Form(None),
    date_to: date | None = Form(None),
):
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="date_from must be on or before date_to",
        )
    # Celery serializes task args as JSON, so dates travel as ISO strings
    date_from_iso = date_from.isoformat() if date_from else None
    date_to_iso = date_to.isoformat() if date_to else None

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type or 'unknown'}",
        )

    content = await read_capped(file, MAX_FILE_SIZE)
    decoded = content.decode("utf-8", errors="replace")

    if user_id is not None:
        user = await db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        # Premium users and users who pay a credit get the full analysis; the
        # charge and the Analysis row are committed together.
        credits_spent = 0
        if not user.is_premium:
            credits_spent = 1 if await spend_credit(db, user_id) else 0
        unlocked = user.is_premium or credits_spent == 1
        tier = "full" if unlocked else "preview"

        analysis = Analysis(
            user_id=user_id,
            platform=platform,
            original_filename=file.filename or "upload",
            status=AnalysisStatus.pending,
            unlocked=unlocked,
            credits_spent=credits_spent,
        )
        db.add(analysis)
        await db.commit()
        await db.refresh(analysis)

        try:
            _enqueue(
                analysis_id=analysis.id, tier=tier, content=decoded, platform=platform,
                language=language, date_from=date_from_iso, date_to=date_to_iso,
            )
        except Exception:
            # Charged but never queued: give the credit back and fail the analysis
            await refund_analysis_credit_async(db, analysis.id)
            analysis.status = AnalysisStatus.failed
            analysis.error = "enqueue_failed"
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="queue_unavailable"
            )
        return UploadResponse(analysis_id=analysis.id, status=AnalysisStatus.pending, tier=tier)

    # Guests only ever get the free preview (no LLM analyzers)
    task = _enqueue(
        analysis_id=None, tier="preview", content=decoded, platform=platform,
        language=language, date_from=date_from_iso, date_to=date_to_iso,
    )
    return UploadResponse(task_id=task.id, status="queued", tier="preview")


# Result fields guests may see in full; `analysis` is replaced by its preview
_GUEST_RESULT_KEYS = ("platform", "total_messages", "participants", "date_filter", "tier")


def _guest_result(output: object) -> dict | None:
    """Reduce a finished task result to what a guest may see (never the full analysis)."""
    if not isinstance(output, dict):
        return None
    analysis = output.get("analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    result = {key: output[key] for key in _GUEST_RESULT_KEYS if key in output}
    result["analysis"] = build_preview(analysis)
    result["access"] = {
        "level": "preview",
        "locked": LOCKED_SECTIONS,
        "teaser": build_teaser(analysis),
    }
    return result


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Poll endpoint for guest users who don't have an analysis_id."""
    task = celery_app.AsyncResult(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        state=task.state,
        result=_guest_result(task.result) if task.successful() else None,
    )
