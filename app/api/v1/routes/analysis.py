from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from app.core.dependencies import DB, CurrentUserId
from app.models.analysis import Analysis, AnalysisStatus

router = APIRouter()


# --- Schemas ---

class AnalysisSummary(BaseModel):
    id: int
    platform: str
    original_filename: str
    status: AnalysisStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalysisDetail(AnalysisSummary):
    result: dict | None
    error: str | None
    updated_at: datetime


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
