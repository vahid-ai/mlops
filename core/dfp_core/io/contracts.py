"""Contracts between services."""
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    user_id: str
    device_id: str
    features: dict[str, float]

class InferenceResponse(BaseModel):
    score: float
    run_id: str
