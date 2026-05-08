from fastapi import APIRouter

from app.api.v1.routes import analysis, auth, payments, upload

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(upload.router, prefix="/upload", tags=["upload"])
router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
router.include_router(payments.router, prefix="/payments", tags=["payments"])
