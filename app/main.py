from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqladmin import Admin
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request

import app.models  # noqa: F401 — registers all SQLAlchemy models
from app.admin.views import AnalysisAdmin, UserAdmin
from app.core.config import settings
from app.core.database import engine
from app.api.v1 import router as api_v1_router


class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        if (
            form.get("username") == settings.ADMIN_USERNAME
            and form.get("password") == settings.ADMIN_PASSWORD
        ):
            request.session["authenticated"] = True
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("authenticated", False)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix=settings.API_V1_STR)

admin = Admin(
    app,
    engine,
    authentication_backend=AdminAuth(secret_key=settings.SECRET_KEY),
    title="LASTSEEN",
    base_url="/admin",
    templates_dir="app/admin/templates",
)
admin.add_view(UserAdmin)
admin.add_view(AnalysisAdmin)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
