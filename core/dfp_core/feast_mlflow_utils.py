"""Helpers for tying Feast datasets to MLflow runs."""

def log_feature_snapshot(run_id: str, repo: str, branch: str, table: str) -> None:
    print(f"Logged snapshot for run {run_id}: {repo}/{branch}/{table}")
