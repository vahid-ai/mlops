from pydantic import BaseModel

class DataConfig(BaseModel):
    lakefs_repo: str
    branch: str = "main"
    training_table: str
    validation_table: str
    test_table: str
