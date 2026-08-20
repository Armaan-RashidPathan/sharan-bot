"""
Pydantic request/response models for the FastAPI backend.

Kept separate from main.py so the API's data contracts (what a client sends
and receives) are easy to find and change independently of routing logic.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, description="The user's question about personal finance.")


class Citation(BaseModel):
    """One retrieved transcript chunk's citation info, attached to an answer."""

    title: str
    start_time: int
    source_url: str


class AskResponse(BaseModel):
    """Response body for POST /ask."""

    answer: str
    sources: list[Citation]
