from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from app.core.dependencies import DB, CurrentUserId
from app.models.analysis import Analysis, AnalysisStatus
from app.services.access import LOCKED_SECTIONS, build_preview, build_teaser

router = APIRouter()


# --- Schemas ---

class AnalysisSummary(BaseModel):
    id: int
    platform: str
    original_filename: str
    status: AnalysisStatus
    created_at: datetime
    unlocked: bool

    model_config = {"from_attributes": True}


class AnalysisDetail(AnalysisSummary):
    result: dict | None
    error: str | None
    updated_at: datetime
    # None while there is no result; {"level": "full"} or {"level": "preview", ...} otherwise
    access: dict | None = None


class AnalysisStatusResponse(BaseModel):
    id: int
    status: AnalysisStatus
    error: str | None
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Helpers ---

async def _get_owned_analysis(db, analysis_id: int, user_id: int) -> Analysis:
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user_id)
    )
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


def _detail(analysis: Analysis, full: bool) -> AnalysisDetail:
    """Build the response, hiding locked sections unless `full` is True."""
    detail = AnalysisDetail.model_validate(analysis)
    if analysis.result is None:
        return detail
    if full:
        detail.access = {"level": "full"}
    else:
        detail.result = build_preview(analysis.result)
        detail.access = {
            "level": "preview",
            "locked": LOCKED_SECTIONS,
            "teaser": build_teaser(analysis.result),
        }
    return detail


# --- Endpoints ---

@router.get("/", response_model=list[AnalysisSummary])
async def list_analyses(db: DB, user_id: CurrentUserId):
    result = await db.execute(
        select(Analysis)
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{analysis_id}", response_model=AnalysisDetail)
async def get_analysis(analysis_id: int, db: DB, user_id: CurrentUserId):
    analysis = await _get_owned_analysis(db, analysis_id, user_id)
    return _detail(analysis, analysis.unlocked)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: int, db: DB, user_id: CurrentUserId):
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == user_id,
        )
    )
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(analysis_id: int, db: DB, user_id: CurrentUserId):
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == user_id,
        )
    )
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    await db.delete(analysis)
    await db.commit()
