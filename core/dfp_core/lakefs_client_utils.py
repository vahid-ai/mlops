"""Thin LakeFS client wrapper with dlt/dbt integration support."""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class LakeFSConfig:
    """LakeFS connection configuration."""

    endpoint: str
    access_key_id: str
    secret_access_key: str
    repository: str = "kronodroid"
    branch: str = "main"

    @classmethod
    def from_env(cls) -> "LakeFSConfig":
        """Create config from environment variables."""
        return cls(
            endpoint=os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000"),
            access_key_id=os.getenv("LAKEFS_ACCESS_KEY_ID", ""),
            secret_access_key=os.getenv("LAKEFS_SECRET_ACCESS_KEY", ""),
            repository=os.getenv("LAKEFS_REPOSITORY", "kronodroid"),
            branch=os.getenv("LAKEFS_BRANCH", "main"),
        )

    @property
    def s3_path(self) -> str:
        """Get S3-compatible path for this repository/branch."""
        return f"s3://{self.repository}/{self.branch}"


def get_client(endpoint: str, access_key: str, secret_key: str) -> dict[str, Any]:
    """Get a LakeFS client configuration dict.

    Args:
        endpoint: LakeFS server endpoint URL
        access_key: LakeFS access key ID
        secret_key: LakeFS secret access key

    Returns:
        Configuration dictionary for LakeFS client
    """
    return {"endpoint": endpoint, "access_key": access_key, "secret_key": "****"}


def get_lakefs_client(config: LakeFSConfig | None = None):
    """Get a configured LakeFS Python client.

    Args:
        config: LakeFS configuration. If None, loads from environment.

    Returns:
        Configured lakefs_sdk.client.LakeFSClient instance
    """
    import lakefs_sdk
    from lakefs_sdk.client import LakeFSClient

    if config is None:
        config = LakeFSConfig.from_env()

    configuration = lakefs_sdk.Configuration(
        host=config.endpoint,
        username=config.access_key_id,
        password=config.secret_access_key,
    )

    return LakeFSClient(configuration)


def ensure_repository(
    client,
    repository: str,
    storage_namespace: str | None = None,
    default_branch: str = "main",
) -> bool:
    """Ensure a LakeFS repository exists, creating it if necessary.

    Args:
        client: LakeFS client instance (LakeFSClient from lakefs_sdk)
        repository: Repository name
        storage_namespace: S3 storage namespace (defaults to s3://{repository})
        default_branch: Default branch name

    Returns:
        True if repository was created, False if it already existed
    """
    from lakefs_sdk.exceptions import NotFoundException
    from lakefs_sdk.models import RepositoryCreation

    if storage_namespace is None:
        storage_namespace = f"s3://{repository}"

    try:
        client.repositories_api.get_repository(repository)
        return False
    except NotFoundException:
        repo_creation = RepositoryCreation(
            name=repository,
            storage_namespace=storage_namespace,
            default_branch=default_branch,
        )
        client.repositories_api.create_repository(repo_creation)
        return True


def ensure_branch(
    client,
    repository: str,
    branch: str,
    source_branch: str = "main",
) -> bool:
    """Ensure a LakeFS branch exists, creating it if necessary.

    Args:
        client: LakeFS client instance (LakeFSClient from lakefs_sdk)
        repository: Repository name
        branch: Branch name to ensure
        source_branch: Source branch to create from

    Returns:
        True if branch was created, False if it already existed
    """
    from lakefs_sdk.exceptions import NotFoundException
    from lakefs_sdk.models import BranchCreation

    try:
        client.branches_api.get_branch(repository, branch)
        return False
    except NotFoundException:
        # Get source branch ref
        source = client.branches_api.get_branch(repository, source_branch)
        branch_creation = BranchCreation(
            name=branch,
            source=source.commit_id,
        )
        client.branches_api.create_branch(repository, branch_creation)
        return True


def commit_changes(
    client,
    repository: str,
    branch: str,
    message: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """Commit staged changes in a LakeFS branch.

    Args:
        client: LakeFS client instance (LakeFSClient from lakefs_sdk)
        repository: Repository name
        branch: Branch name
        message: Commit message
        metadata: Optional commit metadata

    Returns:
        Commit ID
    """
    from lakefs_sdk.models import CommitCreation

    commit_creation = CommitCreation(
        message=message,
        metadata=metadata or {},
    )

    result = client.commits_api.commit(repository, branch, commit_creation)
    return result.id


def get_s3_credentials(config: LakeFSConfig | None = None) -> dict[str, str]:
    """Get S3-compatible credentials for LakeFS.

    These credentials can be used with boto3, dlt, or any S3-compatible client.

    Args:
        config: LakeFS configuration. If None, loads from environment.

    Returns:
        Dictionary with AWS-style credentials
    """
    if config is None:
        config = LakeFSConfig.from_env()

    return {
        "aws_access_key_id": config.access_key_id,
        "aws_secret_access_key": config.secret_access_key,
        "endpoint_url": config.endpoint,
    }


def create_dlt_branch(
    pipeline_name: str,
    run_id: str,
    config: LakeFSConfig | None = None,
) -> str:
    """Create a LakeFS branch for a dlt pipeline run.

    This enables data versioning by creating a branch for each pipeline run.

    Args:
        pipeline_name: Name of the dlt pipeline
        run_id: Unique run identifier
        config: LakeFS configuration

    Returns:
        Branch name that was created
    """
    if config is None:
        config = LakeFSConfig.from_env()

    client = get_lakefs_client(config)
    branch_name = f"dlt/{pipeline_name}/{run_id}"

    ensure_branch(client, config.repository, branch_name, config.branch)

    return branch_name


def merge_dlt_branch(
    branch_name: str,
    target_branch: str = "main",
    config: LakeFSConfig | None = None,
) -> str:
    """Merge a dlt pipeline branch back to the target branch.

    Args:
        branch_name: Branch to merge
        target_branch: Target branch to merge into
        config: LakeFS configuration

    Returns:
        Merge commit ID
    """
    from lakefs_sdk.models import Merge

    if config is None:
        config = LakeFSConfig.from_env()

    client = get_lakefs_client(config)

    merge = Merge(
        message=f"Merge {branch_name} into {target_branch}",
    )

    result = client.refs_api.merge_into_branch(
        config.repository,
        branch_name,
        target_branch,
        merge,
    )

    return result.reference
