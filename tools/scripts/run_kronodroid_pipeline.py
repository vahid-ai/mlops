#!/usr/bin/env python3
"""
Kronodroid Data Pipeline Orchestration Script.

This script orchestrates the full data ingestion and transformation pipeline:
1. Download Kronodroid dataset from Kaggle using dlt
2. Load raw data into MinIO (or LakeFS for versioning)
3. Run dbt transformations to create feature tables
4. Register features with Feast and materialize to online store

Usage:
    # Full pipeline with MinIO
    python run_kronodroid_pipeline.py --destination minio

    # Full pipeline with LakeFS versioning
    python run_kronodroid_pipeline.py --destination lakefs --branch dev

    # Skip ingestion (only run dbt + feast)
    python run_kronodroid_pipeline.py --skip-ingestion

    # Only materialize features to online store
    python run_kronodroid_pipeline.py --materialize-only
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file(env_path: Path = PROJECT_ROOT / ".env"):
    """Load environment variables from .env file."""
    if not env_path.exists():
        print(f"Warning: {env_path} not found")
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def run_dlt_ingestion(destination: str, lakefs_branch: str | None = None) -> bool:
    """Run dlt pipeline to ingest Kronodroid data from Kaggle.

    Args:
        destination: Target destination ('minio' or 'lakefs')
        lakefs_branch: LakeFS branch name (only for lakefs destination)

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 1: Running dlt ingestion from Kaggle")
    print("=" * 60)

    from engines.dlt_engine.dfp_dlt import run_kronodroid_pipeline

    try:
        pipeline = run_kronodroid_pipeline(
            destination=destination,
            lakefs_branch=lakefs_branch,
        )
        print(f"dlt pipeline completed successfully")
        print(f"  - Dataset: {pipeline.dataset_name}")
        print(f"  - Destination: {destination}")
        return True
    except Exception as e:
        print(f"ERROR: dlt ingestion failed: {e}")
        return False


def run_dbt_transformations(target: str = "dev") -> bool:
    """Run dbt transformations to build feature tables.

    Args:
        target: dbt target profile to use

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 2: Running dbt transformations")
    print("=" * 60)

    dbt_project_dir = PROJECT_ROOT / "analytics" / "dbt"
    dbt_data_dir = dbt_project_dir / "data"
    dbt_data_dir.mkdir(parents=True, exist_ok=True)

    # Set DBT_PROFILES_DIR to use project profiles
    profiles_dir = dbt_project_dir / "profiles"
    os.environ["DBT_PROFILES_DIR"] = str(profiles_dir)

    # Set absolute path for DuckDB database
    os.environ["DBT_DUCKDB_PATH"] = str(dbt_data_dir / f"dbt_{target}.duckdb")

    try:
        # Run dbt deps first
        print("Installing dbt dependencies...")
        subprocess.run(
            ["dbt", "deps"],
            cwd=str(dbt_project_dir),
            check=True,
            capture_output=True,
        )

        # Run dbt build (run + test)
        print(f"Running dbt build with target: {target}")
        result = subprocess.run(
            ["dbt", "build", "--target", target, "--select", "tag:kronodroid"],
            cwd=str(dbt_project_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)

        print("dbt transformations completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: dbt transformations failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: dbt command not found. Install with: pip install dbt-core dbt-duckdb")
        return False


def run_feast_apply() -> bool:
    """Apply Feast feature definitions.

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 3: Applying Feast feature definitions")
    print("=" * 60)

    feast_dir = PROJECT_ROOT / "feature_stores" / "feast_store"

    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd=str(feast_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print("Feast feature definitions applied successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: feast apply failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: feast command not found. Install with: pip install feast")
        return False


def run_feast_materialize(days_back: int = 30) -> bool:
    """Materialize features to online store.

    Args:
        days_back: Number of days of features to materialize

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 4: Materializing features to online store")
    print("=" * 60)

    feast_dir = PROJECT_ROOT / "feature_stores" / "feast_store"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    try:
        result = subprocess.run(
            [
                "feast",
                "materialize",
                start_date.isoformat(),
                end_date.isoformat(),
            ],
            cwd=str(feast_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print("Feature materialization completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: feast materialize failed: {e}")
        print(f"stderr: {e.stderr}")
        return False


def commit_to_lakefs(branch: str, message: str) -> bool:
    """Commit changes to LakeFS.

    Args:
        branch: LakeFS branch to commit to
        message: Commit message

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"Committing changes to LakeFS branch: {branch}")
    print("=" * 60)

    from core.dfp_core.lakefs_client_utils import (
        LakeFSConfig,
        commit_changes,
        get_lakefs_client,
    )

    try:
        config = LakeFSConfig.from_env()
        client = get_lakefs_client(config)

        commit_id = commit_changes(
            client,
            config.repository,
            branch,
            message,
            metadata={
                "pipeline": "kronodroid",
                "timestamp": datetime.now().isoformat(),
            },
        )
        print(f"Committed to LakeFS: {commit_id}")
        return True

    except Exception as e:
        print(f"WARNING: LakeFS commit failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Kronodroid data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--destination",
        choices=["minio", "lakefs"],
        default="minio",
        help="Data destination (default: minio)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="LakeFS branch (only for lakefs destination)",
    )
    parser.add_argument(
        "--dbt-target",
        default="dev",
        help="dbt target profile (default: dev)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip dlt ingestion step",
    )
    parser.add_argument(
        "--skip-dbt",
        action="store_true",
        help="Skip dbt transformation step",
    )
    parser.add_argument(
        "--skip-feast",
        action="store_true",
        help="Skip Feast apply step",
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Only run feast materialize",
    )
    parser.add_argument(
        "--materialize-days",
        type=int,
        default=30,
        help="Days of features to materialize (default: 30)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_env_file()

    # Auto-select dbt target based on destination if not explicitly set
    dbt_target = args.dbt_target
    if dbt_target == "dev" and args.destination == "lakefs":
        dbt_target = "lakefs"

    print("=" * 60)
    print("Kronodroid Data Pipeline")
    print("=" * 60)
    print(f"  Destination: {args.destination}")
    if args.destination == "lakefs":
        print(f"  Branch: {args.branch}")
    print(f"  dbt target: {dbt_target}")

    success = True

    if args.materialize_only:
        success = run_feast_materialize(args.materialize_days)
    else:
        # Step 1: dlt ingestion
        if not args.skip_ingestion:
            if not run_dlt_ingestion(args.destination, args.branch):
                success = False
                print("\nPipeline failed at dlt ingestion step")
                sys.exit(1)

        # Step 2: dbt transformations
        if not args.skip_dbt:
            if not run_dbt_transformations(dbt_target):
                success = False
                print("\nPipeline failed at dbt transformation step")
                sys.exit(1)

        # Step 3: Feast apply
        if not args.skip_feast:
            if not run_feast_apply():
                success = False
                print("\nPipeline failed at Feast apply step")
                sys.exit(1)

        # Step 4: Materialize features
        if not run_feast_materialize(args.materialize_days):
            print("\nWARNING: Feature materialization failed (online store may be unavailable)")

        # Step 5: Commit to LakeFS (if using lakefs)
        if args.destination == "lakefs":
            commit_to_lakefs(
                args.branch,
                f"Pipeline run: {datetime.now().isoformat()}",
            )

    print("\n" + "=" * 60)
    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline completed with warnings")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
