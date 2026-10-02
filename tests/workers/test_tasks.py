from app.workers.tasks import _RESULT_TTL, celery_app, process_chat_upload


def test_result_expires_uses_ttl():
    assert _RESULT_TTL == 3600
    assert celery_app.conf.result_expires == _RESULT_TTL


def test_task_result_not_echoed_in_worker_log():
    # The "succeeded in Xs: <result>" line truncates the repr to this size.
    assert process_chat_upload.resultrepr_maxsize == 0
