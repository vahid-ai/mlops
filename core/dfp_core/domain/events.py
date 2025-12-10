from pydantic import BaseModel
from datetime import datetime

class Event(BaseModel):
    user_id: str
    device_id: str
    event_type: str
    event_ts: datetime
