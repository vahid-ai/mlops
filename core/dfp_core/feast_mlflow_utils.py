"""Helpers for tying Feast/LakeFS/Iceberg datasets to MLflow lineage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_files(paths: list[str]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        file_path = Path(path)
        hasher.update(str(file_path).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def log_feature_snapshot(
    *,
    run_id: str,
    lakefs_repository: str,
    lakefs_ref: str,
    iceberg_catalog: str,
    iceberg_database: str,
    iceberg_table: str,
    iceberg_snapshot_id: str | None = None,
    feast_project: str | None = None,
    feast_feature_view: str | None = None,
    feast_defs_sha256: str | None = None,
) -> None:
    """Log a minimal, queryable lineage footprint to an MLflow run."""
    import mlflow

    payload: dict[str, Any] = {
        "data.lakefs_repository": lakefs_repository,
        "data.lakefs_ref": lakefs_ref,
        "data.iceberg_catalog": iceberg_catalog,
        "data.iceberg_database": iceberg_database,
        "data.iceberg_table": iceberg_table,
        "data.iceberg_snapshot_id": iceberg_snapshot_id or "",
        "feast.project": feast_project or "",
        "feast.feature_view": feast_feature_view or "",
        "feast.defs_sha256": feast_defs_sha256 or "",
    }
    for k, v in payload.items():
        mlflow.log_param(k, v)
