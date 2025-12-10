from pydantic import BaseModel

class Device(BaseModel):
    device_id: str
    model: str
    os_version: str
