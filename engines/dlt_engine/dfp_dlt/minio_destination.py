"""MinIO/S3 destination configuration for dlt pipelines."""

import os
from dataclasses import dataclass, field
from typing import Any

import dlt
from dlt.destinations import filesystem


@dataclass
class MinioConfig:
    """Configuration for MinIO connection."""

    endpoint_url: str = "http://localhost:19000"
    bucket_name: str = "dlt-data"
    access_key_id: str = "minioadmin"
    secret_access_key: str = "minioadmin"
    region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> "MinioConfig":
        """Create config from environment variables."""
        return cls(
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://localhost:19000"),
            bucket_name=os.getenv("MINIO_BUCKET_NAME", "dlt-data"),
            access_key_id=os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin"),
            secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin"),
            region=os.getenv("MINIO_REGION", "us-east-1"),
        )


@dataclass
class LakeFSConfig:
    """Configuration for LakeFS connection (S3 gateway mode)."""

    endpoint_url: str = "http://localhost:8000"
    repository: str = "kronodroid"
    branch: str = "main"
    access_key_id: str = ""
    secret_access_key: str = ""

    @classmethod
    def from_env(cls) -> "LakeFSConfig":
        """Create config from environment variables."""
        return cls(
            endpoint_url=os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000"),
            repository=os.getenv("LAKEFS_REPOSITORY", "kronodroid"),
            branch=os.getenv("LAKEFS_BRANCH", "main"),
            access_key_id=os.getenv("LAKEFS_ACCESS_KEY_ID", ""),
            secret_access_key=os.getenv("LAKEFS_SECRET_ACCESS_KEY", ""),
        )

    @property
    def bucket_url(self) -> str:
        """Get the S3-compatible bucket URL for LakeFS."""
        return f"s3://{self.repository}/{self.branch}"


def get_minio_destination(config: MinioConfig | None = None) -> Any:
    """Get a dlt filesystem destination configured for MinIO.

    Args:
        config: MinIO configuration. If None, loads from environment.

    Returns:
        Configured dlt filesystem destination
    """
    if config is None:
        config = MinioConfig.from_env()

    return filesystem(
        bucket_url=f"s3://{config.bucket_name}",
        credentials={
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.secret_access_key,
            "endpoint_url": config.endpoint_url,
            "region_name": config.region,
        },
    )


def get_lakefs_destination(config: LakeFSConfig | None = None) -> Any:
    """Get a dlt filesystem destination configured for LakeFS S3 gateway.

    Args:
        config: LakeFS configuration. If None, loads from environment.

    Returns:
        Configured dlt filesystem destination
    """
    if config is None:
        config = LakeFSConfig.from_env()

    return filesystem(
        bucket_url=config.bucket_url,
        credentials={
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.secret_access_key,
            "endpoint_url": config.endpoint_url,
        },
    )


def ensure_minio_bucket(config: MinioConfig | None = None) -> None:
    """Ensure the MinIO bucket exists, creating it if necessary.

    Args:
        config: MinIO configuration. If None, loads from environment.
    """
    import boto3
    from botocore.exceptions import ClientError

    if config is None:
        config = MinioConfig.from_env()

    s3_client = boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name=config.region,
    )

    try:
        s3_client.head_bucket(Bucket=config.bucket_name)
    except ClientError:
        s3_client.create_bucket(Bucket=config.bucket_name)
        print(f"Created bucket: {config.bucket_name}")
