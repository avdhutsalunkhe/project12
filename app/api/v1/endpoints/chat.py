"""
POST /chat — conversational assessment recommendation endpoint.
"""

from fastapi import APIRouter

from app.logging_config import get_logger
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import BaseResponse
from app.services.chat_service import chat_service

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    description=(
        "Submit a conversational message (with full history) and receive "
        "assessment recommendations from the SHL catalog."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        logger.info("Incoming /chat request — %d messages", len(request.messages))
        result = await chat_service.process_message(request)
        return result
    except Exception as e:
        logger.exception("Error processing chat request")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Internal server error while processing chat.")
