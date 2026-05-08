from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "lastseen",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(bind=True, name="process_chat_upload")
def process_chat_upload(
    self,
    *,
    analysis_id: int | None,
    content: str,
    platform: str,
) -> dict:
    from app.workers.pipeline import run_pipeline

    return run_pipeline(analysis_id=analysis_id, content=content, platform=platform)
