"""Create SparkSession with Iceberg + LakeFS catalog configured.

This module provides a configured SparkSession that:
- Uses Apache Iceberg for table format with Avro storage
- Connects to LakeFS as the S3-compatible storage backend
- Configures the Iceberg catalog to track tables in LakeFS

Data Flow:
    Raw Avro (MinIO) -> Spark -> Iceberg Tables (LakeFS) -> Feast
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field


class SparkConfig(BaseModel):
    """Configuration for Spark session with Iceberg and LakeFS."""

    app_name: str = Field(default="dfp", description="Spark application name")

    # LakeFS configuration
    lakefs_endpoint: str = Field(
        default_factory=lambda: os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
    )
    lakefs_access_key: str = Field(
        default_factory=lambda: os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    )
    lakefs_secret_key: str = Field(
        default_factory=lambda: os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")
    )
    lakefs_repository: str = Field(
        default_factory=lambda: os.getenv("LAKEFS_REPOSITORY", "kronodroid")
    )
    lakefs_branch: str = Field(
        default_factory=lambda: os.getenv("LAKEFS_BRANCH", "main")
    )

    # MinIO configuration (for raw data ingestion)
    minio_endpoint: str = Field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT_URL", "http://localhost:19000")
    )
    minio_access_key: str = Field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin")
    )
    minio_secret_key: str = Field(
        default_factory=lambda: os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin")
    )
    minio_bucket: str = Field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "dlt-data")
    )

    # Iceberg configuration
    iceberg_warehouse: str = Field(
        default="s3a://warehouse",
        description="S3A path for Iceberg warehouse (on LakeFS)",
    )
    iceberg_catalog_name: str = Field(
        default="lakefs_catalog",
        description="Name of the Iceberg catalog",
    )

    # Spark resource configuration
    driver_memory: str = Field(default="2g")
    executor_memory: str = Field(default="2g")
    executor_cores: int = Field(default=2)

    model_config = {"extra": "allow"}


def _strip_protocol(endpoint: str) -> str:
    """Remove http:// or https:// from endpoint for S3A config."""
    return endpoint.replace("http://", "").replace("https://", "")


def get_spark_session(
    app_name: str = "dfp",
    config: Optional[SparkConfig] = None,
):
    """Create a SparkSession configured for Iceberg + LakeFS.

    Args:
        app_name: Spark application name
        config: Optional SparkConfig, uses defaults/env vars if not provided

    Returns:
        Configured SparkSession with Iceberg catalog
    """
    from pyspark.sql import SparkSession

    if config is None:
        config = SparkConfig(app_name=app_name)

    # Iceberg Spark runtime JAR coordinates
    iceberg_spark_jar = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0"
    aws_bundle_jar = "org.apache.iceberg:iceberg-aws-bundle:1.5.0"
    avro_jar = "org.apache.spark:spark-avro_2.12:3.5.0"

    # Build warehouse path with LakeFS repo/branch
    warehouse_path = f"s3a://{config.lakefs_repository}/{config.lakefs_branch}/iceberg"

    builder = (
        SparkSession.builder.appName(config.app_name)
        # Spark packages for Iceberg + Avro
        .config(
            "spark.jars.packages",
            f"{iceberg_spark_jar},{aws_bundle_jar},{avro_jar}",
        )
        # Iceberg SQL catalog extensions
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        # Configure Iceberg catalog backed by LakeFS (via S3A)
        .config(
            f"spark.sql.catalog.{config.iceberg_catalog_name}",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config(
            f"spark.sql.catalog.{config.iceberg_catalog_name}.type",
            "hadoop",
        )
        .config(
            f"spark.sql.catalog.{config.iceberg_catalog_name}.warehouse",
            warehouse_path,
        )
        # S3A configuration for LakeFS (Iceberg warehouse)
        .config("spark.hadoop.fs.s3a.endpoint", config.lakefs_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", config.lakefs_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", config.lakefs_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Default file format for Iceberg tables
        .config("spark.sql.catalog.lakefs_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        # Avro as default write format for Iceberg
        .config("spark.sql.iceberg.write.format.default", "avro")
        # Resource configuration
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.executor.memory", config.executor_memory)
        .config("spark.executor.cores", str(config.executor_cores))
        # Enable Hive support for dbt-spark compatibility
        .enableHiveSupport()
    )

    return builder.getOrCreate()


def get_spark_session_with_minio(
    app_name: str = "dfp",
    config: Optional[SparkConfig] = None,
):
    """Create a SparkSession configured for both MinIO (raw) and LakeFS (Iceberg).

    This session can read raw Avro data from MinIO and write Iceberg tables to LakeFS.
    Uses different S3A configurations via path prefixes.

    Args:
        app_name: Spark application name
        config: Optional SparkConfig

    Returns:
        Configured SparkSession
    """
    from pyspark.sql import SparkSession

    if config is None:
        config = SparkConfig(app_name=app_name)

    # Get base session
    spark = get_spark_session(app_name, config)

    # Add MinIO configuration for reading raw data
    # Use bucket-specific endpoint configuration
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

    # Configure MinIO bucket access (for raw Avro data)
    minio_endpoint = _strip_protocol(config.minio_endpoint)
    hadoop_conf.set(f"fs.s3a.bucket.{config.minio_bucket}.endpoint", config.minio_endpoint)
    hadoop_conf.set(f"fs.s3a.bucket.{config.minio_bucket}.access.key", config.minio_access_key)
    hadoop_conf.set(f"fs.s3a.bucket.{config.minio_bucket}.secret.key", config.minio_secret_key)

    return spark


@lru_cache(maxsize=1)
def get_default_spark_session():
    """Get or create the default SparkSession (cached).

    Returns:
        Cached SparkSession instance
    """
    return get_spark_session()


def read_avro_from_minio(
    spark,
    path: str,
    config: Optional[SparkConfig] = None,
):
    """Read Avro files from MinIO.

    Args:
        spark: SparkSession
        path: Path within the MinIO bucket (e.g., 'kronodroid_raw/emulator')
        config: Optional SparkConfig

    Returns:
        DataFrame with Avro data
    """
    if config is None:
        config = SparkConfig()

    full_path = f"s3a://{config.minio_bucket}/{path}"

    return spark.read.format("avro").load(full_path)


def write_iceberg_table(
    df,
    table_name: str,
    mode: str = "overwrite",
    partition_by: Optional[list[str]] = None,
    config: Optional[SparkConfig] = None,
):
    """Write a DataFrame to an Iceberg table on LakeFS.

    Args:
        df: Spark DataFrame to write
        table_name: Table name (will be prefixed with catalog.db)
        mode: Write mode ('overwrite', 'append')
        partition_by: Optional list of partition columns
        config: Optional SparkConfig

    Returns:
        None
    """
    if config is None:
        config = SparkConfig()

    full_table_name = f"{config.iceberg_catalog_name}.dfp.{table_name}"

    writer = df.writeTo(full_table_name)

    if partition_by:
        writer = writer.partitionedBy(*partition_by)

    if mode == "overwrite":
        writer.createOrReplace()
    else:
        writer.append()


def create_iceberg_database(spark, database: str = "dfp", config: Optional[SparkConfig] = None):
    """Create an Iceberg database if it doesn't exist.

    Args:
        spark: SparkSession
        database: Database name
        config: Optional SparkConfig
    """
    if config is None:
        config = SparkConfig()

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {config.iceberg_catalog_name}.{database}")
