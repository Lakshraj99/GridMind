"""Application-layer request and health schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AlertUpdate(BaseModel):
    status: Literal["open", "acknowledged", "resolved", "suppressed"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, object] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
