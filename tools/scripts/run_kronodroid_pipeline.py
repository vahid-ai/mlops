#!/usr/bin/env python3
"""
Kronodroid Data Pipeline Orchestration Script.

This script orchestrates the full data ingestion and transformation pipeline:
1. Download Kronodroid dataset from Kaggle using dlt
2. Load raw data into MinIO (or LakeFS for versioning)
3. Run transformations to create feature tables (dbt or Kubeflow SparkOperator)
4. Register features with Feast and materialize to online store

Usage:
    # Full pipeline with MinIO (using dbt)
    python run_kronodroid_pipeline.py --destination minio

    # Full pipeline with LakeFS versioning (using dbt)
    python run_kronodroid_pipeline.py --destination lakefs --branch dev

    # Use Kubeflow SparkOperator for transformations (writes Iceberg/Avro)
    python run_kronodroid_pipeline.py --destination lakefs --transform-engine kubeflow

    # Skip ingestion (only run transformations + feast)
    python run_kronodroid_pipeline.py --skip-ingestion

    # Only materialize features to online store
    python run_kronodroid_pipeline.py --materialize-only

Transform Engines:
    dbt       - Local DuckDB-based transformations (default)
    kubeflow  - Kubeflow SparkOperator with Iceberg/Avro output via LakeFS
"""

import argparse
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import quote

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Type alias for transform engine
TransformEngine = Literal["dbt", "kubeflow"]


def _lakefs_quote_ref(ref: str) -> str:
    """URL-encode a LakeFS ref (branch/commit/tag) for use in path segments."""
    return quote(ref, safe="")


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


def run_dlt_ingestion(
    destination: str,
    lakefs_branch: str | None = None,
    drop_pending: bool = False,
    retry_on_pending: bool = True,
) -> bool:
    """Run dlt pipeline to ingest Kronodroid data from Kaggle.

    Args:
        destination: Target destination ('minio' or 'lakefs')
        lakefs_branch: LakeFS branch name (only for lakefs destination)
        drop_pending: If True, drop pending packages before running
        retry_on_pending: If True, automatically drop pending and retry on failure

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 1: Running dlt ingestion from Kaggle")
    print("=" * 60)

    import dlt
    from engines.dlt_engine.dfp_dlt import run_kronodroid_pipeline

    # Determine pipeline name
    if destination == "lakefs":
        pipeline_name = f"kronodroid_lakefs_{lakefs_branch or 'main'}"
    else:
        pipeline_name = "kronodroid_minio"

    # Drop pending packages if requested
    if drop_pending:
        try:
            pipeline = dlt.pipeline(pipeline_name=pipeline_name)
            if pipeline.has_pending_data:
                print(f"  Dropping pending packages for pipeline: {pipeline_name}")
                pipeline.drop_pending_packages()
        except Exception as e:
            print(f"  Warning: Could not check/drop pending packages: {e}")

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
        error_msg = str(e)
        print(f"ERROR: dlt ingestion failed: {e}")

        # Check if this is a pending packages issue
        if retry_on_pending and "pending" in error_msg.lower():
            print("\n  Detected pending packages from previous failed run.")
            print("  Dropping pending packages and retrying...")

            try:
                pipeline = dlt.pipeline(pipeline_name=pipeline_name)
                pipeline.drop_pending_packages()
                print("  Pending packages dropped. Retrying ingestion...")

                pipeline = run_kronodroid_pipeline(
                    destination=destination,
                    lakefs_branch=lakefs_branch,
                )
                print(f"dlt pipeline completed successfully on retry")
                print(f"  - Dataset: {pipeline.dataset_name}")
                print(f"  - Destination: {destination}")
                return True
            except Exception as retry_e:
                print(f"ERROR: dlt ingestion failed on retry: {retry_e}")
                return False

        # Check if this is a connectivity issue
        if "internal error" in error_msg.lower() or "errno" in error_msg.lower():
            print("\n  This appears to be a connectivity issue with MinIO/LakeFS.")
            print("  Please verify:")
            print("    1. MinIO is running and accessible at MINIO_ENDPOINT_URL")
            print("    2. LakeFS is running and accessible at LAKEFS_ENDPOINT_URL")
            print("    3. Credentials are correct (MINIO_ACCESS_KEY_ID, LAKEFS_ACCESS_KEY_ID)")
            print("\n  You can also try: --drop-pending to clear stale pipeline state")

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


def run_kubeflow_transformations(
    lakefs_branch: str = "main",
    minio_endpoint: str | None = None,
    minio_bucket: str | None = None,
    lakefs_endpoint: str | None = None,
    lakefs_repository: str | None = None,
    spark_image: str = "apache/spark:3.5.0-python3",
    namespace: str = "default",
    service_account: str = "spark",
    minio_secret_name: str = "minio-credentials",
    lakefs_secret_name: str = "lakefs-credentials",
    driver_cores: int = 1,
    driver_memory: str = "2g",
    executor_cores: int = 2,
    executor_instances: int = 2,
    executor_memory: str = "2g",
    timeout_seconds: int = 3600,
    use_kfp_client: bool = False,
    kfp_host: str | None = None,
) -> bool:
    """Run Kubeflow SparkOperator transformations to build Iceberg tables.

    This replaces the dbt step with a Spark job running on Kubernetes that:
    - Reads raw Parquet from MinIO
    - Transforms data (same logic as dbt models)
    - Writes Iceberg tables with Avro format to LakeFS

    Args:
        lakefs_branch: Target LakeFS branch
        minio_endpoint: MinIO endpoint (defaults to env var)
        minio_bucket: MinIO bucket (defaults to env var)
        lakefs_endpoint: LakeFS endpoint (defaults to env var)
        lakefs_repository: LakeFS repository (defaults to env var)
        spark_image: Docker image for Spark executors
        namespace: Kubernetes namespace
        service_account: K8s service account for Spark
        minio_secret_name: K8s secret name for MinIO credentials
        lakefs_secret_name: K8s secret name for LakeFS credentials
        driver_cores: Spark driver cores
        driver_memory: Spark driver memory
        executor_cores: Spark executor cores
        executor_instances: Number of Spark executors
        executor_memory: Spark executor memory
        timeout_seconds: Max time to wait for Spark job
        use_kfp_client: If True, submit via KFP client; if False, submit SparkApplication directly
        kfp_host: Kubeflow Pipelines host (required if use_kfp_client=True)

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 2: Running Kubeflow SparkOperator transformations")
    print("=" * 60)

    # Get configuration from environment if not provided
    minio_endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000")
    minio_bucket = minio_bucket or os.getenv("MINIO_BUCKET_NAME", "dlt-data")
    lakefs_endpoint = lakefs_endpoint or os.getenv("LAKEFS_ENDPOINT_URL", "http://lakefs:8000")
    lakefs_repository = lakefs_repository or os.getenv("LAKEFS_REPOSITORY", "kronodroid")

    print(f"  MinIO endpoint: {minio_endpoint}")
    print(f"  MinIO bucket: {minio_bucket}")
    print(f"  LakeFS endpoint: {lakefs_endpoint}")
    print(f"  LakeFS repository: {lakefs_repository}")
    print(f"  LakeFS branch: {lakefs_branch}")
    print(f"  Transform engine: Kubeflow SparkOperator")

    if use_kfp_client:
        return _run_kubeflow_via_kfp_client(
            kfp_host=kfp_host,
            minio_endpoint=minio_endpoint,
            minio_bucket=minio_bucket,
            lakefs_endpoint=lakefs_endpoint,
            lakefs_repository=lakefs_repository,
            lakefs_branch=lakefs_branch,
            spark_image=spark_image,
            namespace=namespace,
            service_account=service_account,
            minio_secret_name=minio_secret_name,
            lakefs_secret_name=lakefs_secret_name,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            executor_cores=executor_cores,
            executor_instances=executor_instances,
            executor_memory=executor_memory,
            timeout_seconds=timeout_seconds,
        )
    else:
        return _run_kubeflow_via_sparkapplication(
            minio_endpoint=minio_endpoint,
            minio_bucket=minio_bucket,
            lakefs_endpoint=lakefs_endpoint,
            lakefs_repository=lakefs_repository,
            lakefs_branch=lakefs_branch,
            spark_image=spark_image,
            namespace=namespace,
            service_account=service_account,
            minio_secret_name=minio_secret_name,
            lakefs_secret_name=lakefs_secret_name,
            driver_cores=driver_cores,
            driver_memory=driver_memory,
            executor_cores=executor_cores,
            executor_instances=executor_instances,
            executor_memory=executor_memory,
            timeout_seconds=timeout_seconds,
        )


def _run_kubeflow_via_kfp_client(
    kfp_host: str | None,
    minio_endpoint: str,
    minio_bucket: str,
    lakefs_endpoint: str,
    lakefs_repository: str,
    lakefs_branch: str,
    spark_image: str,
    namespace: str,
    service_account: str,
    minio_secret_name: str,
    lakefs_secret_name: str,
    driver_cores: int,
    driver_memory: str,
    executor_cores: int,
    executor_instances: int,
    executor_memory: str,
    timeout_seconds: int,
) -> bool:
    """Submit the Kubeflow pipeline via KFP client."""
    try:
        import kfp
        from orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_iceberg_pipeline import (
            kronodroid_iceberg_pipeline,
        )
    except ImportError as e:
        print(f"ERROR: KFP not installed or pipeline not found: {e}")
        print("Install with: pip install kfp")
        return False

    if not kfp_host:
        kfp_host = os.getenv("KFP_HOST", "http://localhost:8080")

    print(f"  KFP host: {kfp_host}")

    try:
        client = kfp.Client(host=kfp_host)

        run = client.create_run_from_pipeline_func(
            kronodroid_iceberg_pipeline,
            arguments={
                "minio_endpoint": minio_endpoint,
                "minio_bucket": minio_bucket,
                "minio_prefix": "kronodroid_raw",
                "minio_secret_name": minio_secret_name,
                "lakefs_endpoint": lakefs_endpoint,
                "lakefs_repository": lakefs_repository,
                "target_branch": lakefs_branch,
                "lakefs_secret_name": lakefs_secret_name,
                "spark_image": spark_image,
                "namespace": namespace,
                "service_account": service_account,
                "driver_cores": driver_cores,
                "driver_memory": driver_memory,
                "executor_cores": executor_cores,
                "executor_instances": executor_instances,
                "executor_memory": executor_memory,
                "spark_timeout_seconds": timeout_seconds,
            },
        )

        print(f"  Submitted KFP run: {run.run_id}")

        # Wait for completion
        result = client.wait_for_run_completion(run.run_id, timeout=timeout_seconds)

        if result.run.status == "Succeeded":
            print("Kubeflow SparkOperator transformations completed successfully")
            return True
        else:
            print(f"ERROR: KFP run failed with status: {result.run.status}")
            return False

    except Exception as e:
        print(f"ERROR: Kubeflow pipeline failed: {e}")
        return False


def _run_kubeflow_via_sparkapplication(
    minio_endpoint: str,
    minio_bucket: str,
    lakefs_endpoint: str,
    lakefs_repository: str,
    lakefs_branch: str,
    spark_image: str,
    namespace: str,
    service_account: str,
    minio_secret_name: str,
    lakefs_secret_name: str,
    driver_cores: int,
    driver_memory: str,
    executor_cores: int,
    executor_instances: int,
    executor_memory: str,
    timeout_seconds: int,
) -> bool:
    """Submit a SparkApplication directly via kubectl."""
    try:
        from kubernetes import client, config
        import yaml
    except ImportError as e:
        print(f"ERROR: kubernetes client not installed: {e}")
        print("Install with: pip install kubernetes")
        return False

    run_id = str(uuid.uuid4())[:8]
    app_name = f"kronodroid-iceberg-{run_id}"
    per_run_branch = f"spark-{run_id}"

    print(f"  Run ID: {run_id}")
    print(f"  SparkApplication: {app_name}")
    print(f"  Per-run branch: {per_run_branch}")

    # Create per-run LakeFS branch
    if not _create_lakefs_branch(lakefs_endpoint, lakefs_repository, per_run_branch, lakefs_branch):
        return False

    # Build SparkApplication manifest
    iceberg_rest_uri = f"{lakefs_endpoint.rstrip('/')}/api/v1/iceberg"
    catalog_name = "lakefs"

    spark_app = {
        "apiVersion": "sparkoperator.k8s.io/v1beta2",
        "kind": "SparkApplication",
        "metadata": {
            "name": app_name,
            "namespace": namespace,
        },
        "spec": {
            "type": "Python",
            "pythonVersion": "3",
            "mode": "cluster",
            "image": spark_image,
            "imagePullPolicy": "Always",
            "mainApplicationFile": "local:///opt/spark/jobs/kronodroid_iceberg_job.py",
            "arguments": [
                "--minio-bucket", minio_bucket,
                "--minio-prefix", "kronodroid_raw",
                "--lakefs-repository", lakefs_repository,
                "--lakefs-branch", per_run_branch,
                "--catalog-name", catalog_name,
                "--staging-database", "stg_kronodroid",
                "--marts-database", "kronodroid",
            ],
            "sparkVersion": "3.5.0",
            "restartPolicy": {"type": "Never"},
            "driver": {
                "cores": driver_cores,
                "memory": driver_memory,
                "serviceAccount": service_account,
                "envFrom": [
                    {"secretRef": {"name": minio_secret_name}},
                    {"secretRef": {"name": lakefs_secret_name}},
                ],
                "env": [
                    {"name": "MINIO_ENDPOINT_URL", "value": minio_endpoint},
                    {"name": "LAKEFS_ENDPOINT_URL", "value": lakefs_endpoint},
                    {"name": "LAKEFS_REPOSITORY", "value": lakefs_repository},
                    {"name": "LAKEFS_BRANCH", "value": per_run_branch},
                ],
            },
            "executor": {
                "cores": executor_cores,
                "instances": executor_instances,
                "memory": executor_memory,
                "envFrom": [
                    {"secretRef": {"name": minio_secret_name}},
                    {"secretRef": {"name": lakefs_secret_name}},
                ],
                "env": [
                    {"name": "MINIO_ENDPOINT_URL", "value": minio_endpoint},
                    {"name": "LAKEFS_ENDPOINT_URL", "value": lakefs_endpoint},
                    {"name": "LAKEFS_REPOSITORY", "value": lakefs_repository},
                    {"name": "LAKEFS_BRANCH", "value": per_run_branch},
                ],
            },
            "deps": {
                "packages": [
                    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2",
                    "org.apache.iceberg:iceberg-aws:1.5.2",
                    "org.apache.hadoop:hadoop-aws:3.3.4",
                    "com.amazonaws:aws-java-sdk-bundle:1.12.262",
                ],
            },
            "sparkConf": {
                "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
                f"spark.sql.catalog.{catalog_name}": "org.apache.iceberg.spark.SparkCatalog",
                f"spark.sql.catalog.{catalog_name}.catalog-impl": "org.apache.iceberg.rest.RESTCatalog",
                f"spark.sql.catalog.{catalog_name}.uri": iceberg_rest_uri,
                f"spark.sql.catalog.{catalog_name}.warehouse": f"s3a://{lakefs_repository}/{per_run_branch}/iceberg",
                "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                "spark.hadoop.fs.s3a.path.style.access": "true",
                "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
                f"spark.hadoop.fs.s3a.bucket.{minio_bucket}.endpoint": minio_endpoint,
                f"spark.hadoop.fs.s3a.bucket.{lakefs_repository}.endpoint": lakefs_endpoint,
            },
        },
    }

    try:
        # Load kubeconfig
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        api = client.CustomObjectsApi()

        # Submit SparkApplication
        try:
            api.create_namespaced_custom_object(
                group="sparkoperator.k8s.io",
                version="v1beta2",
                namespace=namespace,
                plural="sparkapplications",
                body=spark_app,
            )
            print(f"  Submitted SparkApplication: {app_name}")
        except client.ApiException as e:
            if e.status == 409:
                print(f"  SparkApplication {app_name} exists, deleting and recreating...")
                api.delete_namespaced_custom_object(
                    group="sparkoperator.k8s.io",
                    version="v1beta2",
                    namespace=namespace,
                    plural="sparkapplications",
                    name=app_name,
                )
                time.sleep(3)
                api.create_namespaced_custom_object(
                    group="sparkoperator.k8s.io",
                    version="v1beta2",
                    namespace=namespace,
                    plural="sparkapplications",
                    body=spark_app,
                )
            else:
                raise

        # Wait for completion
        start_time = time.time()
        poll_interval = 15

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"ERROR: SparkApplication did not complete within {timeout_seconds}s")
                return False

            try:
                app = api.get_namespaced_custom_object(
                    group="sparkoperator.k8s.io",
                    version="v1beta2",
                    namespace=namespace,
                    plural="sparkapplications",
                    name=app_name,
                )

                status = app.get("status", {})
                app_state = status.get("applicationState", {}).get("state", "UNKNOWN")
                print(f"  SparkApplication state: {app_state} (elapsed: {int(elapsed)}s)")

                if app_state == "COMPLETED":
                    # Commit and merge the per-run branch
                    if _commit_and_merge_lakefs(
                        lakefs_endpoint, lakefs_repository, per_run_branch, lakefs_branch, run_id
                    ):
                        print("Kubeflow SparkOperator transformations completed successfully")
                        return True
                    else:
                        print("WARNING: LakeFS commit/merge failed, but Spark job succeeded")
                        return True

                elif app_state in ("FAILED", "SUBMISSION_FAILED", "FAILING"):
                    error_msg = status.get("applicationState", {}).get("errorMessage", "Unknown")
                    print(f"ERROR: SparkApplication failed: {error_msg}")
                    return False

            except client.ApiException as e:
                if e.status != 404:
                    raise

            time.sleep(poll_interval)

    except Exception as e:
        print(f"ERROR: SparkApplication submission/monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _create_lakefs_branch(
    endpoint: str, repository: str, branch_name: str, source_branch: str
) -> bool:
    """Create a per-run LakeFS branch."""
    import requests

    access_key = os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")

    api_base = endpoint.rstrip("/")
    auth = (access_key, secret_key)

    # Check if branch exists
    branch_url = f"{api_base}/api/v1/repositories/{repository}/branches/{_lakefs_quote_ref(branch_name)}"
    resp = requests.get(branch_url, auth=auth)

    if resp.status_code == 200:
        print(f"  LakeFS branch already exists: {branch_name}")
        return True

    if resp.status_code == 404:
        # Create branch from source
        create_url = f"{api_base}/api/v1/repositories/{repository}/branches"
        data = {"name": branch_name, "source": source_branch}
        create_resp = requests.post(create_url, json=data, auth=auth)

        if create_resp.status_code in (200, 201):
            print(f"  Created LakeFS branch: {branch_name}")
            return True
        else:
            print(f"ERROR: Failed to create LakeFS branch: {create_resp.text}")
            return False

    print(f"ERROR: Failed to check LakeFS branch: {resp.status_code}")
    return False


def _commit_and_merge_lakefs(
    endpoint: str, repository: str, source_branch: str, target_branch: str, run_id: str
) -> bool:
    """Commit and merge a per-run LakeFS branch."""
    import requests

    access_key = os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")

    api_base = endpoint.rstrip("/")
    auth = (access_key, secret_key)

    # Commit
    commit_url = (
        f"{api_base}/api/v1/repositories/{repository}/branches/{_lakefs_quote_ref(source_branch)}/commits"
    )
    commit_data = {
        "message": f"Kronodroid Iceberg transformation - run {run_id}",
        "metadata": {
            "pipeline": "kronodroid-iceberg",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        },
    }

    commit_resp = requests.post(commit_url, json=commit_data, auth=auth)
    if commit_resp.status_code in (200, 201):
        commit_id = commit_resp.json().get("id", "unknown")
        print(f"  Committed to LakeFS: {commit_id}")
    else:
        print(f"  No changes to commit or commit failed: {commit_resp.status_code}")

    # Merge
    merge_url = (
        f"{api_base}/api/v1/repositories/{repository}/refs/{_lakefs_quote_ref(source_branch)}/merge/{_lakefs_quote_ref(target_branch)}"
    )
    merge_data = {"message": f"Merge {source_branch} into {target_branch}"}

    merge_resp = requests.post(merge_url, json=merge_data, auth=auth)
    if merge_resp.status_code in (200, 201):
        merge_ref = merge_resp.json().get("reference", "unknown")
        print(f"  Merged to {target_branch}: {merge_ref}")
    else:
        print(f"  Merge skipped or failed: {merge_resp.status_code}")

    # Delete source branch
    delete_url = f"{api_base}/api/v1/repositories/{repository}/branches/{_lakefs_quote_ref(source_branch)}"
    requests.delete(delete_url, auth=auth)
    print(f"  Deleted branch: {source_branch}")

    return True


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
        "--drop-pending",
        action="store_true",
        help="Drop pending dlt packages before running (clears stale pipeline state)",
    )
    parser.add_argument(
        "--skip-dbt",
        action="store_true",
        help="Skip dbt transformation step (deprecated, use --skip-transform)",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Skip transformation step (dbt or kubeflow)",
    )
    parser.add_argument(
        "--transform-engine",
        choices=["dbt", "kubeflow"],
        default="dbt",
        help="Transformation engine: 'dbt' (local DuckDB) or 'kubeflow' (SparkOperator with Iceberg/Avro)",
    )
    parser.add_argument(
        "--use-kfp-client",
        action="store_true",
        help="For kubeflow engine: submit via KFP client instead of direct SparkApplication",
    )
    parser.add_argument(
        "--kfp-host",
        default=None,
        help="Kubeflow Pipelines host URL (for --use-kfp-client)",
    )
    parser.add_argument(
        "--spark-image",
        default="apache/spark:3.5.0-python3",
        help="Spark Docker image (for kubeflow engine)",
    )
    parser.add_argument(
        "--k8s-namespace",
        default="default",
        help="Kubernetes namespace for Spark jobs",
    )
    parser.add_argument(
        "--spark-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for Spark job (default: 3600)",
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

    # Handle deprecated --skip-dbt flag
    skip_transform = args.skip_transform or args.skip_dbt

    # Validate kubeflow engine requirements
    transform_engine = args.transform_engine
    if transform_engine == "kubeflow" and args.destination == "minio":
        print("WARNING: kubeflow engine works best with lakefs destination for versioning")
        print("         Consider using: --destination lakefs --branch dev")

    print("=" * 60)
    print("Kronodroid Data Pipeline")
    print("=" * 60)
    print(f"  Destination: {args.destination}")
    if args.destination == "lakefs":
        print(f"  Branch: {args.branch}")
    print(f"  Transform engine: {transform_engine}")
    if transform_engine == "dbt":
        print(f"  dbt target: {dbt_target}")
    else:
        print(f"  Spark image: {args.spark_image}")
        print(f"  K8s namespace: {args.k8s_namespace}")

    success = True

    if args.materialize_only:
        success = run_feast_materialize(args.materialize_days)
    else:
        # Step 1: dlt ingestion
        if not args.skip_ingestion:
            if not run_dlt_ingestion(
                args.destination,
                args.branch,
                drop_pending=args.drop_pending,
                retry_on_pending=True,
            ):
                success = False
                print("\nPipeline failed at dlt ingestion step")
                sys.exit(1)

        # Step 2: Transformations (dbt or kubeflow)
        if not skip_transform:
            if transform_engine == "dbt":
                if not run_dbt_transformations(dbt_target):
                    success = False
                    print("\nPipeline failed at dbt transformation step")
                    sys.exit(1)
            elif transform_engine == "kubeflow":
                if not run_kubeflow_transformations(
                    lakefs_branch=args.branch,
                    spark_image=args.spark_image,
                    namespace=args.k8s_namespace,
                    timeout_seconds=args.spark_timeout,
                    use_kfp_client=args.use_kfp_client,
                    kfp_host=args.kfp_host,
                ):
                    success = False
                    print("\nPipeline failed at Kubeflow SparkOperator transformation step")
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
