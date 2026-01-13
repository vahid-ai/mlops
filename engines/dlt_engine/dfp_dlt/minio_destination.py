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
        # MinIO generally requires path-style addressing.
        config_kwargs={"s3": {"addressing_style": "path"}},
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
        # LakeFS S3 gateway is S3-compatible but works most reliably with path-style addressing.
        config_kwargs={"s3": {"addressing_style": "path"}},
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

    # LakeFS repositories typically point at an S3 bucket/prefix via `storage_namespace`.
    # In the local Kind setup, LakeFS uses MinIO as the blockstore backend, and the backing
    # bucket may not exist yet. If it doesn't, S3 writes via the LakeFS gateway fail with:
    #   "We encountered an internal error, please try again."
    #
    # Create the bucket up-front so downstream dlt loads can succeed.
    minio_endpoint_url = os.getenv("MINIO_ENDPOINT_URL")
    if minio_endpoint_url:
        try:
            # Keep this in sync with the default `storage_namespace` used below.
            backing_bucket = os.getenv("LAKEFS_STORAGE_BUCKET", "lakefs-data")
            minio_config = MinioConfig.from_env()
            minio_config.bucket_name = backing_bucket
            ensure_minio_bucket(minio_config)
        except Exception as e:
            # Don't hard-fail here: users may be running LakeFS against real S3 or a differently
            # provisioned blockstore. We'll surface any real issues when we attempt to write.
            print(f"Warning: Could not ensure LakeFS backing bucket exists: {e}")

    # LakeFS API base URL (remove trailing slash if present)
    api_base = config.endpoint_url.rstrip("/")
    auth = (config.access_key_id, config.secret_access_key)

    # Check if repository exists
    repo_url = f"{api_base}/api/v1/repositories/{config.repository}"
    resp = requests.get(repo_url, auth=auth)

    if resp.status_code == 404:
        # Repository doesn't exist, create it
        backing_bucket = os.getenv("LAKEFS_STORAGE_BUCKET", "lakefs-data")
        storage_namespace = f"s3://{backing_bucket}/{config.repository}"

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
