"""Shared configuration defaults for Kubeflow pipelines/components.

Keep this module dependency-light. Pipeline definitions import these constants at
compile time, so avoid importing heavy runtime dependencies here.
"""

from __future__ import annotations

# Namespaces
DEFAULT_DATA_NAMESPACE = "dfp"
DEFAULT_KFP_NAMESPACE = "kubeflow"

# Kubernetes names
DEFAULT_SPARK_NAMESPACE = DEFAULT_DATA_NAMESPACE
DEFAULT_SPARK_SERVICE_ACCOUNT = "spark"

# Secrets (namespace-scoped; see Taskfile.yml `secrets:sync`)
DEFAULT_MINIO_SECRET_NAME = "minio-credentials"
DEFAULT_LAKEFS_SECRET_NAME = "lakefs-credentials"

# ConfigMaps
DEFAULT_FEAST_CONFIGMAP_NAME = "feast-config"
DEFAULT_FEAST_MOUNT_PATH = "/feast"

# Service endpoints (cross-namespace safe: use `<svc>.<ns>`)
DEFAULT_MINIO_ENDPOINT = f"http://minio.{DEFAULT_DATA_NAMESPACE}:9000"
DEFAULT_LAKEFS_ENDPOINT = f"http://lakefs.{DEFAULT_DATA_NAMESPACE}:8000"
DEFAULT_MLFLOW_TRACKING_URI = f"http://mlflow.{DEFAULT_DATA_NAMESPACE}:5000"

# Kronodroid defaults
DEFAULT_MINIO_BUCKET = "dlt-data"
DEFAULT_MINIO_PREFIX = "kronodroid_raw"
DEFAULT_LAKEFS_REPOSITORY = "kronodroid"
DEFAULT_LAKEFS_BRANCH = "main"

DEFAULT_ICEBERG_CATALOG = "lakefs"
DEFAULT_ICEBERG_DATABASE = "kronodroid"
DEFAULT_ICEBERG_STAGING_DATABASE = "stg_kronodroid"
DEFAULT_ICEBERG_SOURCE_TABLE = "fct_training_dataset"

# Component images
DEFAULT_LAKEFS_COMPONENT_IMAGE = "dfp-lakefs-component:v2"
DEFAULT_SPARK_DRIVER_COMPONENT_BASE_IMAGE = "python:3.11-slim"
DEFAULT_AUTOENCODER_TRAIN_IMAGE = "dfp-autoencoder-train:v8"
DEFAULT_SPARK_IMAGE = "dfp-spark:latest"

# Secret keys -> env vars for KFP task injection
#
# Note: These are the *secret data keys* (left-hand side) that must exist in the
# Kubernetes Secret object.
MINIO_SECRET_KEY_TO_ENV = {
    "MINIO_ACCESS_KEY_ID": "MINIO_ACCESS_KEY_ID",
    "MINIO_SECRET_ACCESS_KEY": "MINIO_SECRET_ACCESS_KEY",
}

LAKEFS_SECRET_KEY_TO_ENV = {
    "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
    "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
}
