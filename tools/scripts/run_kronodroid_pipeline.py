#!/usr/bin/env python3
"""
Kronodroid Data Pipeline Orchestration Script.

This script orchestrates the full data ingestion and transformation pipeline:
1. Download Kronodroid dataset from Kaggle using dlt → Parquet → MinIO
2. Run dbt-spark transformations → Iceberg tables on LakeFS
3. Register features with Feast and materialize to online store
4. Commit changes to LakeFS for version tracking

Data Flow:
    dlt (Kaggle) → Parquet → MinIO → Spark + dbt → Iceberg (LakeFS) → Feast

Usage:
    # Full pipeline
    python run_kronodroid_pipeline.py

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


def run_dlt_ingestion(file_format: str = "parquet") -> bool:
    """Run dlt pipeline to ingest Kronodroid data from Kaggle to MinIO.

    Args:
        file_format: Output format (parquet recommended for Spark)

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 1: Running dlt ingestion from Kaggle → Parquet → MinIO")
    print("=" * 60)

    from engines.dlt_engine.dfp_dlt import run_kronodroid_pipeline

    try:
        pipeline = run_kronodroid_pipeline(
            destination="minio",
            dataset_name="kronodroid_raw",
            file_format=file_format,
        )
        print("dlt pipeline completed successfully")
        print(f"  - Dataset: {pipeline.dataset_name}")
        print(f"  - Format: {file_format}")
        print(f"  - Destination: MinIO")
        return True
    except Exception as e:
        print(f"ERROR: dlt ingestion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_dbt_spark_transformations(target: str = "dev") -> bool:
    """Run dbt-spark transformations to build Iceberg feature tables.

    Args:
        target: dbt target profile to use (dev or thrift)

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 2: Running dbt-spark transformations → Iceberg tables")
    print("=" * 60)

    dbt_project_dir = PROJECT_ROOT / "analytics" / "dbt"

    # Set DBT_PROFILES_DIR to use project profiles
    profiles_dir = dbt_project_dir / "profiles"
    os.environ["DBT_PROFILES_DIR"] = str(profiles_dir)

    # Set LakeFS warehouse path
    lakefs_repo = os.getenv("LAKEFS_REPOSITORY", "kronodroid")
    lakefs_branch = os.getenv("LAKEFS_BRANCH", "main")
    os.environ["LAKEFS_WAREHOUSE"] = f"s3a://{lakefs_repo}/{lakefs_branch}/iceberg"

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

        print("dbt-spark transformations completed successfully")
        print(f"  - Iceberg catalog: lakefs_catalog")
        print(f"  - Database: dfp")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: dbt transformations failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: dbt command not found. Install with: pip install dbt-spark[session]")
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
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: feast command not found. Install with: pip install feast[spark]")
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
    """Commit Iceberg table changes to LakeFS.

    Args:
        branch: LakeFS branch to commit to
        message: Commit message

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print(f"Step 5: Committing Iceberg changes to LakeFS branch: {branch}")
    print("=" * 60)

    from engines.spark_engine.dfp_spark.iceberg_catalog import commit_iceberg_changes

    commit_id = commit_iceberg_changes(
        branch=branch,
        message=message,
        metadata={
            "pipeline": "kronodroid",
            "timestamp": datetime.now().isoformat(),
            "tables": ["fct_training_dataset", "fct_malware_samples", "dim_malware_families"],
        },
    )

    if commit_id:
        print(f"Committed to LakeFS: {commit_id}")
        return True
    else:
        print("WARNING: LakeFS commit failed or no changes to commit")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Kronodroid data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="LakeFS branch for Iceberg tables (default: main)",
    )
    parser.add_argument(
        "--dbt-target",
        default="dev",
        help="dbt target profile: dev (embedded Spark) or thrift (Spark server)",
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
        "--skip-materialize",
        action="store_true",
        help="Skip Feast materialize step",
    )
    parser.add_argument(
        "--skip-commit",
        action="store_true",
        help="Skip LakeFS commit step",
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
    parser.add_argument(
        "--file-format",
        choices=["avro", "parquet"],
        default="parquet",
        help="Raw data format from dlt (default: parquet; 'avro' is mapped to parquet for dlt)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_env_file()

    # Set LakeFS branch in environment for all components
    os.environ["LAKEFS_BRANCH"] = args.branch

    print("=" * 60)
    print("Kronodroid Data Pipeline (Spark + Iceberg + LakeFS)")
    print("=" * 60)
    print(f"  LakeFS branch: {args.branch}")
    print(f"  dbt target: {args.dbt_target}")
    print(f"  Raw format: {args.file_format}")
    print("")
    print("  Flow: Kaggle → dlt → Parquet → MinIO → Spark → Iceberg → LakeFS → Feast")

    success = True

    if args.materialize_only:
        success = run_feast_materialize(args.materialize_days)
    else:
        # Step 1: dlt ingestion to MinIO (Parquet format)
        if not args.skip_ingestion:
            if not run_dlt_ingestion(file_format=args.file_format):
                success = False
                print("\nPipeline failed at dlt ingestion step")
                sys.exit(1)

        # Step 2: dbt-spark transformations → Iceberg tables
        if not args.skip_dbt:
            if not run_dbt_spark_transformations(args.dbt_target):
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
        if not args.skip_materialize:
            if not run_feast_materialize(args.materialize_days):
                print("\nWARNING: Feature materialization failed (online store may be unavailable)")

        # Step 5: Commit to LakeFS
        if not args.skip_commit:
            commit_to_lakefs(
                args.branch,
                f"Pipeline run: Iceberg tables updated {datetime.now().isoformat()}",
            )

    print("\n" + "=" * 60)
    if success:
        print("Pipeline completed successfully!")
        print("")
        print("Iceberg tables written to LakeFS:")
        print(f"  - lakefs_catalog.dfp.fct_training_dataset")
        print(f"  - lakefs_catalog.dfp.fct_malware_samples")
        print(f"  - lakefs_catalog.dfp.dim_malware_families")
    else:
        print("Pipeline completed with warnings")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
