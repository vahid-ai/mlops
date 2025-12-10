"""Schemas for logs and features."""
from pydantic import BaseModel

class SessionLog(BaseModel):
    session_id: str
    user_id: str
    device_id: str
    ts: int
