from fastapi import APIRouter, HTTPException, Request, status

router = APIRouter()


@router.post("/checkout")
async def create_checkout_session():
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)


@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(request: Request):
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)


@router.get("/subscription")
async def get_subscription():
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
