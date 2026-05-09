from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    PROJECT_NAME: str = "LastSeen"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str

    REDIS_URL: str = "redis://localhost:6380/0"
    CELERY_BROKER_URL: str = "redis://localhost:6380/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6380/0"

    ANTHROPIC_API_KEY: str | None = None
    NARRATIVE_MODEL: str = "claude-haiku-4-5"

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "changeme"

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]


settings = Settings()
