from fastapi import APIRouter
from .mcqa import router as mcqa_router

router = APIRouter()
router.include_router(mcqa_router, tags=["Stem MCQA"])