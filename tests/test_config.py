import pytest

from app.core.config import derive_admin_session_key, settings, validate_production_settings


def _cfg(**kw):
    return settings.model_copy(update=kw)


def test_development_accepts_short_secret():
    validate_production_settings(_cfg(ENVIRONMENT="development", SECRET_KEY="short"))


def test_production_rejects_short_secret():
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        validate_production_settings(_cfg(ENVIRONMENT="production", SECRET_KEY="x" * 31))


def test_production_accepts_long_secret():
    validate_production_settings(_cfg(ENVIRONMENT="production", SECRET_KEY="x" * 32))


def test_admin_session_key_is_derived_not_the_secret():
    key = derive_admin_session_key("s" * 40)
    assert key != "s" * 40
    assert key == derive_admin_session_key("s" * 40)
    assert key != derive_admin_session_key("t" * 40)


def test_jwt_algorithm_is_pinned():
    assert settings.ALGORITHM == "HS256"
