"""Component: LakeFS commit and merge for per-run branches.

This KFP component:
1. Commits all staged changes in the per-run LakeFS branch
2. Merges the per-run branch into the target branch
3. Optionally deletes the per-run branch after merge

Used after SparkOperator jobs to version-control Iceberg table changes.
"""

import os
from datetime import datetime
from typing import NamedTuple

from kfp import dsl


class LakeFSCommitMergeOutput(NamedTuple):
    """Output from the LakeFS commit/merge component."""

    commit_id: str
    merge_commit_id: str
    source_branch: str
    target_branch: str


@dsl.component(
    base_image="dfp-lakefs-component:v2",
)
def lakefs_commit_merge_op(
    lakefs_endpoint: str,
    lakefs_repository: str,
    source_branch: str,
    target_branch: str,
    commit_message: str,
    run_id: str,
    pipeline_name: str,
    delete_source_branch: bool,
) -> NamedTuple(
    "LakeFSCommitMergeOutput",
    [("commit_id", str), ("merge_commit_id", str), ("source_branch", str), ("target_branch", str)],
):
    """Commit changes and merge a LakeFS branch.

    Args:
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        source_branch: Branch to commit and merge (e.g., spark/<run_id>)
        target_branch: Target branch to merge into (e.g., main)
        commit_message: Message for the commit
        run_id: Pipeline run ID (for metadata)
        pipeline_name: Name of the pipeline (for metadata)
        delete_source_branch: Whether to delete source branch after merge

    Returns:
        NamedTuple with commit_id, merge_commit_id, source_branch, target_branch
    """
    import os
    from collections import namedtuple
    from datetime import datetime

    import requests

    Output = namedtuple(
        "LakeFSCommitMergeOutput",
        ["commit_id", "merge_commit_id", "source_branch", "target_branch"],
    )

    # Get credentials from environment (injected by K8s secret)
    access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")

    api_base = lakefs_endpoint.rstrip("/")
    auth = (access_key, secret_key)

    print(f"LakeFS Commit & Merge")
    print(f"  Endpoint: {api_base}")
    print(f"  Repository: {lakefs_repository}")
    print(f"  Source branch: {source_branch}")
    print(f"  Target branch: {target_branch}")

    # Step 1: Commit all staged changes (may need multiple commits for Iceberg metadata)
    # Iceberg writes via S3A gateway can create additional metadata files after initial commit
    commit_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{source_branch}/commits"
    commit_id = "no-changes"
    max_commit_attempts = 5

    for attempt in range(1, max_commit_attempts + 1):
        commit_data = {
            "message": f"{commit_message} (commit {attempt})" if attempt > 1 else commit_message,
            "metadata": {
                "pipeline": pipeline_name,
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "commit_attempt": str(attempt),
            },
        }

        print(f"Commit attempt {attempt} on branch: {source_branch}")
        commit_resp = requests.post(commit_url, json=commit_data, auth=auth)

        if commit_resp.status_code in (200, 201):
            commit_result = commit_resp.json()
            commit_id = commit_result.get("id", "unknown")
            print(f"Committed: {commit_id}")
        elif commit_resp.status_code == 400 and "no changes" in commit_resp.text.lower():
            print(f"No more changes to commit (branch is clean after {attempt - 1} commits)")
            break
        else:
            raise RuntimeError(f"Failed to commit: {commit_resp.status_code} - {commit_resp.text}")

    # Step 2: Commit any uncommitted changes on the TARGET branch
    # This handles cases where dlt or other processes write directly to target branch
    target_commit_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{target_branch}/commits"
    target_commit_data = {
        "message": f"Auto-commit before merge from {source_branch}",
        "metadata": {"pipeline": pipeline_name, "run_id": run_id, "auto_commit": "true"},
    }
    print(f"Checking target branch {target_branch} for uncommitted changes...")
    target_commit_resp = requests.post(target_commit_url, json=target_commit_data, auth=auth)
    if target_commit_resp.status_code in (200, 201):
        target_commit_id = target_commit_resp.json().get("id", "unknown")
        print(f"Committed target branch changes: {target_commit_id}")
    elif "no changes" in target_commit_resp.text.lower():
        print("Target branch is clean")
    else:
        print(f"Warning: Could not commit target branch: {target_commit_resp.text}")

    # Step 3: Merge into target branch (with retry for dirty branch issues)
    import time
    merge_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/refs/{source_branch}/merge/{target_branch}"
    max_merge_attempts = 5
    merge_commit_id = None

    for merge_attempt in range(1, max_merge_attempts + 1):
        merge_data = {
            "message": f"Merge {source_branch} into {target_branch}",
            "metadata": {
                "pipeline": pipeline_name,
                "run_id": run_id,
                "source_commit": commit_id,
            },
        }

        print(f"Merge attempt {merge_attempt}")
        merge_resp = requests.post(merge_url, json=merge_data, auth=auth)

        if merge_resp.status_code in (200, 201):
            merge_result = merge_resp.json()
            merge_commit_id = merge_result.get("reference", "unknown")
            print(f"Merged: {merge_commit_id}")
            break
        elif "no changes" in merge_resp.text.lower() or merge_resp.status_code == 409:
            print("Nothing to merge (branches are identical)")
            merge_commit_id = commit_id
            break
        elif "uncommitted" in merge_resp.text.lower() or "dirty" in merge_resp.text.lower():
            # Try another commit before retrying merge
            print(f"Dirty branch detected, attempting additional commit...")
            time.sleep(1)  # Brief delay for consistency
            extra_commit_data = {
                "message": f"{commit_message} (cleanup commit {merge_attempt})",
                "metadata": {"pipeline": pipeline_name, "run_id": run_id, "cleanup": "true"},
            }
            extra_resp = requests.post(commit_url, json=extra_commit_data, auth=auth)
            if extra_resp.status_code in (200, 201):
                commit_id = extra_resp.json().get("id", commit_id)
                print(f"Extra commit: {commit_id}")
            elif "no changes" in extra_resp.text.lower():
                # No changes but merge still failed - wait and retry
                print("No changes to commit, waiting before retry...")
                time.sleep(2)
        else:
            raise RuntimeError(f"Failed to merge: {merge_resp.status_code} - {merge_resp.text}")

    if merge_commit_id is None:
        raise RuntimeError(f"Failed to merge after {max_merge_attempts} attempts")

    # Step 4: Optionally delete source branch
    if delete_source_branch and source_branch != target_branch:
        delete_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{source_branch}"
        delete_resp = requests.delete(delete_url, auth=auth)

        if delete_resp.status_code in (200, 204):
            print(f"Deleted branch: {source_branch}")
        else:
            print(f"Warning: Failed to delete branch: {delete_resp.status_code}")

    return Output(commit_id, merge_commit_id, source_branch, target_branch)


@dsl.component(
    base_image="dfp-lakefs-component:v2",
)
def lakefs_commit_only_op(
    lakefs_endpoint: str,
    lakefs_repository: str,
    branch: str,
    commit_message: str,
    run_id: str,
    pipeline_name: str,
) -> NamedTuple("LakeFSCommitOutput", [("commit_id", str), ("branch", str)]):
    """Commit changes to a LakeFS branch without merging.

    Args:
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        branch: Branch to commit
        commit_message: Message for the commit
        run_id: Pipeline run ID (for metadata)
        pipeline_name: Name of the pipeline (for metadata)

    Returns:
        NamedTuple with commit_id and branch
    """
    import os
    from collections import namedtuple
    from datetime import datetime

    import requests

    Output = namedtuple("LakeFSCommitOutput", ["commit_id", "branch"])

    access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")

    api_base = lakefs_endpoint.rstrip("/")
    auth = (access_key, secret_key)

    print(f"LakeFS Commit")
    print(f"  Branch: {branch}")

    # Check for uncommitted changes
    diff_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{branch}/diff"
    diff_resp = requests.get(diff_url, auth=auth)

    if diff_resp.status_code != 200:
        raise RuntimeError(f"Failed to get diff: {diff_resp.status_code} - {diff_resp.text}")

    diff_data = diff_resp.json()
    changes = diff_data.get("results", [])

    if not changes:
        print("No uncommitted changes")
        return Output("no-changes", branch)

    print(f"Found {len(changes)} uncommitted changes")

    # Commit
    commit_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{branch}/commits"
    commit_data = {
        "message": commit_message,
        "metadata": {
            "pipeline": pipeline_name,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    commit_resp = requests.post(commit_url, json=commit_data, auth=auth)

    if commit_resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to commit: {commit_resp.status_code} - {commit_resp.text}")

    commit_result = commit_resp.json()
    commit_id = commit_result.get("id", "unknown")
    print(f"Committed: {commit_id}")

    return Output(commit_id, branch)


# Convenience function for testing outside KFP
def commit_and_merge_lakefs_branch(
    lakefs_endpoint: str | None = None,
    lakefs_repository: str = "kronodroid",
    source_branch: str = "spark/test",
    target_branch: str = "main",
    commit_message: str = "Spark Iceberg transformation",
    run_id: str = "test-run",
    pipeline_name: str = "kronodroid-iceberg",
    delete_source_branch: bool = True,
) -> LakeFSCommitMergeOutput:
    """Commit and merge a LakeFS branch (for testing outside KFP)."""
    import requests

    if lakefs_endpoint is None:
        lakefs_endpoint = os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")

    access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")

    api_base = lakefs_endpoint.rstrip("/")
    auth = (access_key, secret_key)

    # Commit
    commit_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{source_branch}/commits"
    commit_data = {
        "message": commit_message,
        "metadata": {
            "pipeline": pipeline_name,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    commit_resp = requests.post(commit_url, json=commit_data, auth=auth)
    if commit_resp.status_code in (200, 201):
        commit_id = commit_resp.json().get("id", "unknown")
    else:
        commit_id = "no-changes"

    # Merge
    merge_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/refs/{source_branch}/merge/{target_branch}"
    merge_data = {"message": f"Merge {source_branch} into {target_branch}"}

    merge_resp = requests.post(merge_url, json=merge_data, auth=auth)
    if merge_resp.status_code in (200, 201):
        merge_commit_id = merge_resp.json().get("reference", "unknown")
    else:
        merge_commit_id = commit_id

    # Delete source branch
    if delete_source_branch:
        delete_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{source_branch}"
        requests.delete(delete_url, auth=auth)

    return LakeFSCommitMergeOutput(
        commit_id=commit_id,
        merge_commit_id=merge_commit_id,
        source_branch=source_branch,
        target_branch=target_branch,
    )
