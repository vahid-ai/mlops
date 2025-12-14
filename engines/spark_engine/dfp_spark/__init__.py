"""Spark execution engine for DFP.

This module provides Spark utilities for:
- Iceberg table management backed by LakeFS
- Reading Avro data from MinIO
- Writing feature tables for Feast consumption
"""

from dfp_spark.iceberg_catalog import (
    LakeFSIcebergCatalog,
    commit_iceberg_changes,
    get_iceberg_table_path,
)
from dfp_spark.session import (
    SparkConfig,
    get_default_spark_session,
    get_spark_session,
    get_spark_session_with_minio,
    read_avro_from_minio,
    write_iceberg_table,
)

__all__ = [
    "SparkConfig",
    "get_spark_session",
    "get_spark_session_with_minio",
    "get_default_spark_session",
    "read_avro_from_minio",
    "write_iceberg_table",
    "LakeFSIcebergCatalog",
    "get_iceberg_table_path",
    "commit_iceberg_changes",
]
