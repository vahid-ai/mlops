"""LakeFS Iceberg catalog helpers.

Utilities for managing Iceberg tables stored on LakeFS with version control.
"""

import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class LakeFSCommit:
    """Represents a LakeFS commit."""

    id: str
    message: str
    committer: str
    creation_date: str
    metadata: dict


class LakeFSIcebergCatalog:
    """Helper class for managing Iceberg tables on LakeFS.

    Provides utilities for:
    - Tracking Iceberg table versions with LakeFS commits
    - Creating branches for experimentation
    - Merging feature branches back to main
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        repository: Optional[str] = None,
    ):
        self.endpoint = (
            endpoint or os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
        ).rstrip("/")
        self.access_key = access_key or os.getenv("LAKEFS_ACCESS_KEY_ID", "")
        self.secret_key = secret_key or os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")
        self.repository = repository or os.getenv("LAKEFS_REPOSITORY", "kronodroid")
        self._auth = (self.access_key, self.secret_key)

    def _api_url(self, path: str) -> str:
        """Build API URL."""
        return f"{self.endpoint}/api/v1{path}"

    def create_branch(self, branch_name: str, source_branch: str = "main") -> bool:
        """Create a new branch from source branch.

        Args:
            branch_name: Name of new branch
            source_branch: Branch to create from

        Returns:
            True if successful
        """
        url = self._api_url(f"/repositories/{self.repository}/branches")
        data = {"name": branch_name, "source": source_branch}

        resp = requests.post(url, json=data, auth=self._auth)
        return resp.status_code in (200, 201)

    def commit(
        self,
        branch: str,
        message: str,
        metadata: Optional[dict] = None,
    ) -> Optional[LakeFSCommit]:
        """Commit changes on a branch.

        Args:
            branch: Branch name
            message: Commit message
            metadata: Optional metadata dict

        Returns:
            LakeFSCommit if successful, None otherwise
        """
        url = self._api_url(f"/repositories/{self.repository}/branches/{branch}/commits")
        data = {"message": message, "metadata": metadata or {}}

        resp = requests.post(url, json=data, auth=self._auth)

        if resp.status_code in (200, 201):
            commit_data = resp.json()
            return LakeFSCommit(
                id=commit_data.get("id", ""),
                message=commit_data.get("message", ""),
                committer=commit_data.get("committer", ""),
                creation_date=commit_data.get("creation_date", ""),
                metadata=commit_data.get("metadata", {}),
            )
        return None

    def get_commit(self, branch: str, ref: str = "HEAD") -> Optional[LakeFSCommit]:
        """Get commit info.

        Args:
            branch: Branch name
            ref: Commit ref (default HEAD)

        Returns:
            LakeFSCommit if found
        """
        url = self._api_url(f"/repositories/{self.repository}/refs/{branch}/commits")
        params = {"amount": 1}

        resp = requests.get(url, params=params, auth=self._auth)

        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                commit_data = results[0]
                return LakeFSCommit(
                    id=commit_data.get("id", ""),
                    message=commit_data.get("message", ""),
                    committer=commit_data.get("committer", ""),
                    creation_date=commit_data.get("creation_date", ""),
                    metadata=commit_data.get("metadata", {}),
                )
        return None

    def merge(
        self,
        source_branch: str,
        destination_branch: str,
        message: Optional[str] = None,
    ) -> bool:
        """Merge source branch into destination.

        Args:
            source_branch: Branch to merge from
            destination_branch: Branch to merge into
            message: Optional merge commit message

        Returns:
            True if successful
        """
        url = self._api_url(
            f"/repositories/{self.repository}/refs/{source_branch}/merge/{destination_branch}"
        )
        data = {"message": message or f"Merge {source_branch} into {destination_branch}"}

        resp = requests.post(url, json=data, auth=self._auth)
        return resp.status_code in (200, 201)

    def list_tables(self, branch: str, prefix: str = "iceberg/dfp/") -> list[str]:
        """List Iceberg tables on a branch.

        Args:
            branch: Branch name
            prefix: Path prefix for tables

        Returns:
            List of table names
        """
        url = self._api_url(f"/repositories/{self.repository}/refs/{branch}/objects/ls")
        params = {"prefix": prefix, "delimiter": "/"}

        resp = requests.get(url, params=params, auth=self._auth)

        if resp.status_code == 200:
            results = resp.json().get("results", [])
            return [
                r.get("path", "").replace(prefix, "").rstrip("/")
                for r in results
                if r.get("path_type") == "common_prefix"
            ]
        return []

    def get_table_s3a_path(
        self,
        table_name: str,
        branch: Optional[str] = None,
    ) -> str:
        """Get S3A path for an Iceberg table.

        Args:
            table_name: Table name
            branch: Optional branch (uses env default)

        Returns:
            S3A path string
        """
        branch = branch or os.getenv("LAKEFS_BRANCH", "main")
        return f"s3a://{self.repository}/{branch}/iceberg/dfp/{table_name}"


def get_iceberg_table_path(
    table_name: str,
    repository: Optional[str] = None,
    branch: Optional[str] = None,
) -> str:
    """Get the S3A path for an Iceberg table on LakeFS.

    Args:
        table_name: Name of the table
        repository: LakeFS repository (default from env)
        branch: LakeFS branch (default from env)

    Returns:
        S3A path for the table
    """
    repo = repository or os.getenv("LAKEFS_REPOSITORY", "kronodroid")
    br = branch or os.getenv("LAKEFS_BRANCH", "main")
    return f"s3a://{repo}/{br}/iceberg/dfp/{table_name}"


def commit_iceberg_changes(
    branch: str,
    message: str,
    metadata: Optional[dict] = None,
) -> Optional[str]:
    """Commit Iceberg table changes to LakeFS.

    Args:
        branch: LakeFS branch
        message: Commit message
        metadata: Optional metadata

    Returns:
        Commit ID if successful
    """
    catalog = LakeFSIcebergCatalog()
    commit = catalog.commit(branch, message, metadata)
    return commit.id if commit else None
