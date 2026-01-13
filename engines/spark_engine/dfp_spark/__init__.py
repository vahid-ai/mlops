"""Spark execution engine for DFP.

This module provides Spark session configuration and jobs for:
- Reading raw Parquet data from MinIO
- Transforming data and writing Iceberg tables
- Using LakeFS Iceberg REST catalog for version control
"""

from engines.spark_engine.dfp_spark.session import (
    MinioConfig,
    LakeFSConfig,
    SparkIcebergConfig,
    get_spark_session,
    get_spark_session_for_k8s,
)
from engines.spark_engine.dfp_spark.kronodroid_iceberg_job import (
    read_raw_parquet,
    create_stg_emulator,
    create_stg_real_device,
    create_stg_combined,
    create_fct_malware_samples,
    create_dim_malware_families,
    create_fct_training_dataset,
    write_iceberg_table,
    ensure_databases,
    main as run_kronodroid_iceberg_job,
)

__all__ = [
    # Configuration
    "MinioConfig",
    "LakeFSConfig",
    "SparkIcebergConfig",
    # Session builders
    "get_spark_session",
    "get_spark_session_for_k8s",
    # Kronodroid job
    "run_kronodroid_iceberg_job",
    # Data reading
    "read_raw_parquet",
    # Transformations
    "create_stg_emulator",
    "create_stg_real_device",
    "create_stg_combined",
    "create_fct_malware_samples",
    "create_dim_malware_families",
    "create_fct_training_dataset",
    # Iceberg writing
    "write_iceberg_table",
    "ensure_databases",
]
