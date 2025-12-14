"""MinIO/S3 destination configuration for dlt pipelines.

Configures dlt to write Avro format files to MinIO for Spark consumption.
"""

import os
from dataclasses import dataclass
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


def get_minio_destination(
    config: MinioConfig | None = None,
    file_format: str = "avro",
) -> Any:
    """Get a dlt filesystem destination configured for MinIO.

    Args:
        config: MinIO configuration. If None, loads from environment.
        file_format: Output file format ('avro', 'parquet', 'jsonl')

    Returns:
        Configured dlt filesystem destination
    """
    if config is None:
        config = MinioConfig.from_env()

    # Configure filesystem destination with Avro format for Spark compatibility
    dest = filesystem(
        bucket_url=f"s3://{config.bucket_name}",
        credentials={
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.secret_access_key,
            "endpoint_url": config.endpoint_url,
            "region_name": config.region,
        },
    )

    # Set the loader file format via config
    # Note: dlt uses 'avro' format for Avro output
    dest.config.layout = "{table_name}/{file_id}.{ext}"
    
    return dest


def get_lakefs_destination(
    config: LakeFSConfig | None = None,
    file_format: str = "avro",
) -> Any:
    """Get a dlt filesystem destination configured for LakeFS S3 gateway.

    Args:
        config: LakeFS configuration. If None, loads from environment.
        file_format: Output file format ('avro', 'parquet', 'jsonl')

    Returns:
        Configured dlt filesystem destination
    """
    if config is None:
        config = LakeFSConfig.from_env()

    dest = filesystem(
        bucket_url=config.bucket_url,
        credentials={
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.secret_access_key,
            "endpoint_url": config.endpoint_url,
        },
    )

    dest.config.layout = "{table_name}/{file_id}.{ext}"
    
    return dest


def get_avro_loader_config() -> dict:
    """Get dlt loader configuration for Avro output.

    Returns:
        Dict with loader configuration
    """
    return {
        "loader_file_format": "avro",
        # Avro-specific settings
        "avro": {
            "codec": "snappy",  # Compression codec
        },
    }


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


def ensure_lakefs_repository(config: LakeFSConfig | None = None) -> None:
    """Ensure the LakeFS repository and branch exist, creating them if necessary.

    LakeFS repositories must be created via the LakeFS API, not the S3 gateway.
    This function creates the repository with MinIO as the backing storage.

    Args:
        config: LakeFS configuration. If None, loads from environment.
    """
    import requests

    if config is None:
        config = LakeFSConfig.from_env()

    # LakeFS API base URL (remove trailing slash if present)
    api_base = config.endpoint_url.rstrip("/")
    auth = (config.access_key_id, config.secret_access_key)

    # Check if repository exists
    repo_url = f"{api_base}/api/v1/repositories/{config.repository}"
    resp = requests.get(repo_url, auth=auth)

    if resp.status_code == 404:
        # Repository doesn't exist, create it
        # Get MinIO endpoint from environment for storage namespace
        minio_endpoint = os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000")
        # Use internal minio service name if running in k8s
        if "localhost" in minio_endpoint:
            # For kind cluster, use the internal service name
            storage_namespace = f"s3://lakefs-data/{config.repository}"
        else:
            storage_namespace = f"s3://lakefs-data/{config.repository}"

        create_url = f"{api_base}/api/v1/repositories"
        create_data = {
            "name": config.repository,
            "storage_namespace": storage_namespace,
            "default_branch": "main",
        }

        print(f"Creating LakeFS repository: {config.repository}")
        create_resp = requests.post(create_url, json=create_data, auth=auth)

        if create_resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to create LakeFS repository: {create_resp.status_code} - {create_resp.text}"
            )
        print(f"Created LakeFS repository: {config.repository}")

    elif resp.status_code != 200:
        raise RuntimeError(
            f"Failed to check LakeFS repository: {resp.status_code} - {resp.text}"
        )
    else:
        print(f"LakeFS repository exists: {config.repository}")

    # Now ensure the branch exists (if not main)
    if config.branch != "main":
        branch_url = f"{api_base}/api/v1/repositories/{config.repository}/branches/{config.branch}"
        branch_resp = requests.get(branch_url, auth=auth)

        if branch_resp.status_code == 404:
            # Branch doesn't exist, create it from main
            create_branch_url = f"{api_base}/api/v1/repositories/{config.repository}/branches"
            branch_data = {
                "name": config.branch,
                "source": "main",
            }

            print(f"Creating LakeFS branch: {config.branch}")
            create_branch_resp = requests.post(
                create_branch_url, json=branch_data, auth=auth
            )

            if create_branch_resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"Failed to create LakeFS branch: {create_branch_resp.status_code} - {create_branch_resp.text}"
                )
            print(f"Created LakeFS branch: {config.branch}")
        elif branch_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to check LakeFS branch: {branch_resp.status_code} - {branch_resp.text}"
            )
        else:
            print(f"LakeFS branch exists: {config.branch}")
