"""Create SparkSession with Iceberg + LakeFS catalog configured.

This module provides a SparkSession builder that configures:
- LakeFS Iceberg REST catalog for metadata operations
- S3A filesystem for MinIO (raw data) and LakeFS S3 gateway (Iceberg warehouse)
- Per-bucket S3A credentials and endpoints
- Required Iceberg packages and extensions

Environment variables used:
- MINIO_ENDPOINT_URL: MinIO endpoint (default: http://localhost:19000)
- MINIO_ACCESS_KEY_ID: MinIO access key (default: minioadmin)
- MINIO_SECRET_ACCESS_KEY: MinIO secret key (default: minioadmin)
- MINIO_BUCKET_NAME: MinIO bucket for raw data (default: dlt-data)
- LAKEFS_ENDPOINT_URL: LakeFS endpoint (default: http://localhost:8000)
- LAKEFS_ACCESS_KEY_ID: LakeFS access key
- LAKEFS_SECRET_ACCESS_KEY: LakeFS secret key
- LAKEFS_REPOSITORY: LakeFS repository name (default: kronodroid)
- LAKEFS_BRANCH: LakeFS branch (default: main)
"""

import os
from dataclasses import dataclass, field
from typing import Any

from pyspark.sql import SparkSession


@dataclass
class MinioConfig:
    """MinIO connection configuration."""

    endpoint_url: str = "http://localhost:19000"
    access_key_id: str = "minioadmin"
    secret_access_key: str = "minioadmin"
    bucket_name: str = "dlt-data"
    region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> "MinioConfig":
        """Create config from environment variables."""
        return cls(
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://localhost:19000"),
            access_key_id=os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin"),
            secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin"),
            bucket_name=os.getenv("MINIO_BUCKET_NAME", "dlt-data"),
            region=os.getenv("MINIO_REGION", "us-east-1"),
        )

    @property
    def s3a_endpoint(self) -> str:
        """Get S3A-compatible endpoint (strip http:// prefix if needed)."""
        endpoint = self.endpoint_url
        if endpoint.startswith("http://"):
            return endpoint[7:]
        if endpoint.startswith("https://"):
            return endpoint[8:]
        return endpoint


@dataclass
class LakeFSConfig:
    """LakeFS connection configuration."""

    endpoint_url: str = "http://localhost:8000"
    access_key_id: str = ""
    secret_access_key: str = ""
    repository: str = "kronodroid"
    branch: str = "main"

    @classmethod
    def from_env(cls) -> "LakeFSConfig":
        """Create config from environment variables."""
        return cls(
            endpoint_url=os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000"),
            access_key_id=os.getenv("LAKEFS_ACCESS_KEY_ID", ""),
            secret_access_key=os.getenv("LAKEFS_SECRET_ACCESS_KEY", ""),
            repository=os.getenv("LAKEFS_REPOSITORY", "kronodroid"),
            branch=os.getenv("LAKEFS_BRANCH", "main"),
        )

    @property
    def s3a_endpoint(self) -> str:
        """Get S3A-compatible endpoint (strip http:// prefix if needed)."""
        endpoint = self.endpoint_url
        if endpoint.startswith("http://"):
            return endpoint[7:]
        if endpoint.startswith("https://"):
            return endpoint[8:]
        return endpoint

    @property
    def iceberg_rest_uri(self) -> str:
        """Get the LakeFS Iceberg REST catalog URI."""
        base = self.endpoint_url.rstrip("/")
        return f"{base}/api/v1/iceberg"

    @property
    def warehouse_path(self) -> str:
        """Get the S3A warehouse path for Iceberg tables."""
        return f"s3a://{self.repository}/{self.branch}/iceberg"


@dataclass
class SparkIcebergConfig:
    """Combined Spark/Iceberg configuration."""

    minio: MinioConfig = field(default_factory=MinioConfig.from_env)
    lakefs: LakeFSConfig = field(default_factory=LakeFSConfig.from_env)
    catalog_name: str = "lakefs"

    # Maven packages for Iceberg + S3A + Avro
    packages: list[str] = field(default_factory=lambda: [
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2",
        "org.apache.iceberg:iceberg-aws:1.5.2",
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    ])

    @classmethod
    def from_env(cls) -> "SparkIcebergConfig":
        """Create config from environment variables."""
        return cls(
            minio=MinioConfig.from_env(),
            lakefs=LakeFSConfig.from_env(),
            catalog_name=os.getenv("SPARK_ICEBERG_CATALOG", "lakefs"),
        )


def get_spark_session(
    app_name: str = "dfp-iceberg",
    config: SparkIcebergConfig | None = None,
    local_mode: bool = False,
) -> SparkSession:
    """Create a SparkSession configured for Iceberg + LakeFS.

    Args:
        app_name: Spark application name
        config: Spark/Iceberg configuration. If None, loads from environment.
        local_mode: If True, run Spark in local mode (for testing)

    Returns:
        Configured SparkSession with Iceberg catalog and S3A filesystem
    """
    if config is None:
        config = SparkIcebergConfig.from_env()

    builder = SparkSession.builder.appName(app_name)

    if local_mode:
        builder = builder.master("local[*]")

    # Add Maven packages
    packages_str = ",".join(config.packages)
    builder = builder.config("spark.jars.packages", packages_str)

    # Iceberg extensions
    builder = builder.config(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    )

    # Configure the LakeFS Iceberg REST catalog
    cat = config.catalog_name
    builder = (
        builder.config(f"spark.sql.catalog.{cat}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{cat}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config(f"spark.sql.catalog.{cat}.uri", config.lakefs.iceberg_rest_uri)
        .config(f"spark.sql.catalog.{cat}.warehouse", config.lakefs.warehouse_path)
        # LakeFS REST catalog authentication
        .config(f"spark.sql.catalog.{cat}.credential", config.lakefs.access_key_id)
        .config(f"spark.sql.catalog.{cat}.token", config.lakefs.secret_access_key)
    )

    # S3A filesystem configuration - default (for LakeFS S3 gateway)
    # LakeFS acts as an S3-compatible endpoint for reading/writing Iceberg data files
    builder = (
        builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    )

    # Per-bucket configuration for MinIO (raw data bucket)
    minio_bucket = config.minio.bucket_name
    builder = (
        builder.config(
            f"spark.hadoop.fs.s3a.bucket.{minio_bucket}.endpoint",
            config.minio.endpoint_url,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{minio_bucket}.access.key",
            config.minio.access_key_id,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{minio_bucket}.secret.key",
            config.minio.secret_access_key,
        )
    )

    # Per-bucket configuration for LakeFS repository (Iceberg warehouse)
    lakefs_repo = config.lakefs.repository
    builder = (
        builder.config(
            f"spark.hadoop.fs.s3a.bucket.{lakefs_repo}.endpoint",
            config.lakefs.endpoint_url,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{lakefs_repo}.access.key",
            config.lakefs.access_key_id,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{lakefs_repo}.secret.key",
            config.lakefs.secret_access_key,
        )
    )

    # Default Iceberg table properties (Avro format)
    builder = (
        builder.config("spark.sql.iceberg.write.format.default", "avro")
        .config("spark.sql.iceberg.write.avro.compression-codec", "snappy")
    )

    return builder.getOrCreate()


def get_spark_session_for_k8s(
    app_name: str = "dfp-iceberg",
    config: SparkIcebergConfig | None = None,
    k8s_namespace: str = "default",
    service_account: str = "spark",
    driver_cores: int = 1,
    driver_memory: str = "2g",
    executor_cores: int = 2,
    executor_memory: str = "2g",
    executor_instances: int = 2,
) -> SparkSession:
    """Create a SparkSession configured for Kubernetes deployment.

    This is used when the Spark job is submitted via SparkOperator.
    Most K8s-specific configs are set by the SparkApplication CRD,
    so this mainly adds the Iceberg/S3A configuration.

    Args:
        app_name: Spark application name
        config: Spark/Iceberg configuration
        k8s_namespace: Kubernetes namespace
        service_account: Kubernetes service account for Spark
        driver_cores: Number of driver cores
        driver_memory: Driver memory
        executor_cores: Number of executor cores
        executor_memory: Executor memory
        executor_instances: Number of executor instances

    Returns:
        Configured SparkSession for Kubernetes
    """
    if config is None:
        config = SparkIcebergConfig.from_env()

    # Start with the base Iceberg configuration
    builder = SparkSession.builder.appName(app_name)

    # K8s-specific settings (some may be overridden by SparkApplication)
    builder = (
        builder.config("spark.kubernetes.namespace", k8s_namespace)
        .config("spark.kubernetes.authenticate.driver.serviceAccountName", service_account)
        .config("spark.driver.cores", str(driver_cores))
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.cores", str(executor_cores))
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.instances", str(executor_instances))
    )

    # Add Maven packages
    packages_str = ",".join(config.packages)
    builder = builder.config("spark.jars.packages", packages_str)

    # Iceberg extensions
    builder = builder.config(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
    )

    # Configure the LakeFS Iceberg REST catalog
    cat = config.catalog_name
    builder = (
        builder.config(f"spark.sql.catalog.{cat}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{cat}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config(f"spark.sql.catalog.{cat}.uri", config.lakefs.iceberg_rest_uri)
        .config(f"spark.sql.catalog.{cat}.warehouse", config.lakefs.warehouse_path)
        .config(f"spark.sql.catalog.{cat}.credential", config.lakefs.access_key_id)
        .config(f"spark.sql.catalog.{cat}.token", config.lakefs.secret_access_key)
    )

    # S3A filesystem configuration
    builder = (
        builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    )

    # Per-bucket configuration for MinIO
    minio_bucket = config.minio.bucket_name
    builder = (
        builder.config(
            f"spark.hadoop.fs.s3a.bucket.{minio_bucket}.endpoint",
            config.minio.endpoint_url,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{minio_bucket}.access.key",
            config.minio.access_key_id,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{minio_bucket}.secret.key",
            config.minio.secret_access_key,
        )
    )

    # Per-bucket configuration for LakeFS
    lakefs_repo = config.lakefs.repository
    builder = (
        builder.config(
            f"spark.hadoop.fs.s3a.bucket.{lakefs_repo}.endpoint",
            config.lakefs.endpoint_url,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{lakefs_repo}.access.key",
            config.lakefs.access_key_id,
        )
        .config(
            f"spark.hadoop.fs.s3a.bucket.{lakefs_repo}.secret.key",
            config.lakefs.secret_access_key,
        )
    )

    # Default Iceberg table properties
    builder = (
        builder.config("spark.sql.iceberg.write.format.default", "avro")
        .config("spark.sql.iceberg.write.avro.compression-codec", "snappy")
    )

    return builder.getOrCreate()
