from pydantic import BaseModel, HttpUrl

class MLflowConfig(BaseModel):
    tracking_uri: HttpUrl
    experiment_name: str = "dfp-default"
    artifact_location: str | None = None
