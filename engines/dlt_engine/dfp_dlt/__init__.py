"""dlt engine for data ingestion pipelines.

Supports Avro output format for Spark compatibility.
"""

from .kaggle_source import kaggle_dataset_source
from .kronodroid_pipeline import (
    run_kronodroid_pipeline,
    run_kronodroid_to_avro,
)
from .minio_destination import (
    MinioConfig,
    LakeFSConfig,
    get_avro_loader_config,
    get_minio_destination,
    get_lakefs_destination,
)

__all__ = [
    "kaggle_dataset_source",
    "run_kronodroid_pipeline",
    "run_kronodroid_to_avro",
    "MinioConfig",
    "LakeFSConfig",
    "get_avro_loader_config",
    "get_minio_destination",
    "get_lakefs_destination",
]
