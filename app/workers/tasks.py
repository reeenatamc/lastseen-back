from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from app.core.config import settings

_RESULT_TTL = 3600  # 1 h: enough for the result page to read it;
                    # nothing lingers in Redis.

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
    # Acknowledge only after the task completes so that if the worker
    # crashes mid-task the message is requeued instead of being lost.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Process one task at a time to avoid OOM from multiple ML model copies.
    worker_prefetch_multiplier=1,
    # Results hold participant names and the narrative; drop them from Redis
    # once the result page has had time to read them.
    result_expires=_RESULT_TTL,
)

_SOFT_LIMIT = 900   # 15 min: raises SoftTimeLimitExceeded → clean _mark_failed.
                    # Bumped from 5 min because pysentimiento (es) runs BOTH a
                    # sentiment and an emotion BETO model in series, which on
                    # CPU is ~3-4× slower than the distilbert-multilingual
                    # fallback. A 2000-sample run can take ~10-12 min.
_HARD_LIMIT = 1200  # 20 min: SIGKILL if soft limit ignored. Soft is often
                    # swallowed inside transformers/datasets multiprocessing,
                    # so hard is the real backstop.


@celery_app.task(
    bind=True,
    name="process_chat_upload",
    # The success log line prints the return value (names, narrative summary).
    # A zero repr size keeps it out of the worker logs.
    resultrepr_maxsize=0,
    soft_time_limit=_SOFT_LIMIT,
    time_limit=_HARD_LIMIT,
    max_retries=0,
)
def process_chat_upload(
    self,
    *,
    analysis_id: int | None,
    content: str,
    platform: str,
    language: str = "auto",
    date_from: str | None = None,
    date_to: str | None = None,
    tier: str = "full",
) -> dict:
    from app.workers.pipeline import run_pipeline

    try:
        return run_pipeline(
            analysis_id=analysis_id,
            content=content,
            platform=platform,
            language=language,
            date_from=date_from,
            date_to=date_to,
            tier=tier,
        )
    except SoftTimeLimitExceeded:
        from app.workers.pipeline import _mark_failed
        if analysis_id is not None:
            _mark_failed(analysis_id, "timeout")
        raise
