#!/usr/bin/env python3
"""
Kronodroid Data Pipeline Orchestration Script.

This script orchestrates the full data ingestion and transformation pipeline:
1. Download Kronodroid dataset from Kaggle using dlt → Parquet → MinIO
2. Run transformations → Iceberg tables on LakeFS (dbt-spark OR Spark Operator)
3. Register features with Feast and materialize to online store
4. Commit changes to LakeFS for version tracking

Data Flow:
    dlt (Kaggle) → Parquet → MinIO → Spark (dbt OR Spark Operator) → Iceberg (LakeFS) → Feast

Usage:
    # Full pipeline
    python run_kronodroid_pipeline.py

    # Skip ingestion (only run dbt + feast)
    python run_kronodroid_pipeline.py --skip-ingestion

    # Only materialize features to online store
    python run_kronodroid_pipeline.py --materialize-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Kubernetes / Spark Operator Helpers (similar to spark_operator_test.ipynb)
# ---------------------------------------------------------------------------


def run_kubectl(
    args: List[str], *, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess:
    """Run a kubectl command and return the result."""
    cmd = ["kubectl"] + args
    try:
        result = subprocess.run(cmd, check=check, capture_output=capture, text=True)
        return result
    except subprocess.CalledProcessError as e:
        if not capture:
            raise
        print(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        raise


def kubectl_get_json(
    resource: str, name: str = "", namespace: str = "dfp"
) -> Optional[Dict]:
    """Get a Kubernetes resource as JSON."""
    try:
        args = ["-n", namespace, "get", resource]
        if name:
            args.append(name)
        args.extend(["-o", "json"])
        result = run_kubectl(args)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError:
        return None
    except json.JSONDecodeError:
        return None


def print_status(status: str, message: str):
    """Print a status message with icon."""
    icons = {
        "success": "[OK]",
        "warning": "[WARN]",
        "error": "[ERROR]",
        "info": "[INFO]",
        "pending": "[...]",
    }
    icon = icons.get(status, "[?]")
    print(f"{icon} {message}")


def get_spark_application_status(app_name: str, namespace: str = "dfp") -> Dict[str, Any]:
    """Get the current status of a SparkApplication."""
    app = kubectl_get_json("sparkapplication", app_name, namespace)

    if not app:
        return {"found": False, "state": "NOT_FOUND"}

    status = app.get("status", {})
    app_state = status.get("applicationState", {})

    return {
        "found": True,
        "state": app_state.get("state", "UNKNOWN"),
        "error_message": app_state.get("errorMessage", ""),
        "driver_info": status.get("driverInfo", {}),
        "executor_state": status.get("executorState", {}),
        "last_submission_attempt_time": status.get("lastSubmissionAttemptTime", ""),
        "termination_time": status.get("terminationTime", ""),
        "spark_application_id": status.get("sparkApplicationId", ""),
    }


def get_driver_logs(app_name: str, namespace: str = "dfp", tail: int = 100) -> str:
    """Get logs from the driver pod."""
    driver_pod = f"{app_name}-driver"

    try:
        result = run_kubectl(
            ["-n", namespace, "logs", driver_pod, f"--tail={tail}"], check=False
        )
        if result.returncode == 0:
            return result.stdout
        else:
            # Try with --previous flag for crashed containers
            result = run_kubectl(
                ["-n", namespace, "logs", driver_pod, "--previous", f"--tail={tail}"],
                check=False,
            )
            if result.returncode == 0:
                return result.stdout
            return ""
    except Exception:
        return ""


def monitor_spark_application_status(
    app_name: str,
    namespace: str = "dfp",
    timeout_seconds: int = 1800,
    poll_interval: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Monitor a SparkApplication until completion or timeout.

    Returns:
        Dict with 'success', 'state', 'elapsed', and optionally 'error' keys.
    """
    terminal_states = {"COMPLETED", "FAILED", "SUBMISSION_FAILED"}
    start_time = time.time()
    last_state = None

    while True:
        elapsed = int(time.time() - start_time)

        if elapsed > timeout_seconds:
            if verbose:
                print_status("warning", f"Timeout after {elapsed}s")
            return {"success": False, "reason": "TIMEOUT", "last_state": last_state}

        status = get_spark_application_status(app_name, namespace)

        if not status["found"]:
            if verbose:
                print_status(
                    "warning", f"[{elapsed}s] SparkApplication not found, waiting..."
                )
            time.sleep(poll_interval)
            continue

        state = status["state"]

        # Print state changes
        if state != last_state:
            timestamp = datetime.now().strftime("%H:%M:%S")
            if verbose:
                print(f"[{timestamp}] [{elapsed}s] State: {state}")

                if status["driver_info"].get("podName"):
                    print(f"           Driver: {status['driver_info']['podName']}")

            last_state = state

        # Check for terminal state
        if state in terminal_states:
            if state == "COMPLETED":
                if verbose:
                    print_status("success", f"SparkApplication completed in {elapsed}s")
                return {"success": True, "state": state, "elapsed": elapsed}
            else:
                if verbose:
                    print_status("error", f"SparkApplication {state} after {elapsed}s")
                    if status["error_message"]:
                        print(f"    Error: {status['error_message']}")
                return {
                    "success": False,
                    "state": state,
                    "error": status["error_message"],
                    "elapsed": elapsed,
                }

        time.sleep(poll_interval)


def list_spark_applications(namespace: str = "dfp") -> List[Dict]:
    """List all SparkApplications in the namespace."""
    apps = kubectl_get_json("sparkapplication", "", namespace)

    if not apps or not apps.get("items"):
        return []

    result = []
    for app in apps.get("items", []):
        name = app.get("metadata", {}).get("name", "unknown")
        state = (
            app.get("status", {}).get("applicationState", {}).get("state", "UNKNOWN")
        )
        created = app.get("metadata", {}).get("creationTimestamp", "N/A")
        result.append({"name": name, "state": state, "created": created})

    return result


def delete_spark_application(app_name: str, namespace: str = "dfp") -> bool:
    """Delete a SparkApplication."""
    try:
        result = run_kubectl(
            ["-n", namespace, "delete", "sparkapplication", app_name], check=False
        )
        return result.returncode == 0
    except Exception:
        return False


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


def check_spark_thrift_server(host: str = "localhost", port: int = 10000) -> bool:
    """Check if Spark Thrift Server is reachable.

    Args:
        host: Thrift server host
        port: Thrift server port

    Returns:
        True if server is reachable
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except (OSError, TimeoutError):
        return False


def run_dbt_spark_transformations(target: str = "dev") -> bool:
    """Run dbt-spark transformations to build Iceberg feature tables.

    Args:
        target: dbt target profile to use (dev or prod)

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 2: Running dbt-spark transformations → Iceberg tables")
    print("=" * 60)

    # Check Spark Thrift Server connectivity
    thrift_host = os.getenv("SPARK_THRIFT_HOST", "localhost")
    thrift_port = int(os.getenv("SPARK_THRIFT_PORT", "10000"))

    print(f"Checking Spark Thrift Server at {thrift_host}:{thrift_port}...")
    if not check_spark_thrift_server(thrift_host, thrift_port):
        print(f"ERROR: Cannot connect to Spark Thrift Server at {thrift_host}:{thrift_port}")
        print("\nTo start Spark Thrift Server in Kind cluster:")
        print("  1. Ensure Kind cluster is running: kind get clusters")
        print("  2. Deploy Spark Thrift Server:")
        print("     kubectl apply -k infra/k8s/kind/addons/spark-thrift/")
        print("  3. Wait for pod to be ready:")
        print("     kubectl -n dfp wait --for=condition=ready pod -l app=spark-thrift-server --timeout=120s")
        print("\nOr set SPARK_THRIFT_HOST and SPARK_THRIFT_PORT env vars for remote server.")
        return False

    print(f"  ✓ Spark Thrift Server is reachable")

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
        print("ERROR: dbt command not found. Install with: pip install dbt-spark")
        return False


def run_kubeflow_spark_operator_transformations(
    *,
    branch: str,
    namespace: str = "dfp",
    spark_image: str = "apache/spark:3.5.7-python3",
    service_account: str = "spark-operator",
    timeout_seconds: int = 60 * 30,
    raw_format: Optional[str] = None,
) -> bool:
    """Run Spark transformations via Spark Operator (SparkApplication).

    This is an alternative to dbt-spark for Step #2. It submits a SparkApplication
    that reads Kronodroid raw data from MinIO and writes Iceberg tables to LakeFS.
    """
    print("\n" + "=" * 60)
    print("Step 2 (alt): Running Spark Operator transformations → Iceberg tables")
    print("=" * 60)

    import shutil

    if not shutil.which("kubectl"):
        print("ERROR: kubectl not found. Install kubectl and configure KUBECONFIG.")
        return False

    from orchestration.kubeflow.dfp_kfp.components.kronodroid_spark_operator_transform_component import (
        KronodroidSparkOperatorConfig,
        run,
    )

    print(f"Using Spark image: {spark_image}")
    print(f"Namespace: {namespace}")
    print(f"Service account: {service_account}")
    print(f"Timeout: {timeout_seconds}s")

    resolved_raw_format = raw_format or os.getenv("RAW_FORMAT", "parquet")

    cfg = KronodroidSparkOperatorConfig(
        namespace=namespace,
        spark_image=spark_image,
        service_account=service_account,
        timeout_seconds=timeout_seconds,
        raw_bucket=os.getenv("MINIO_BUCKET_NAME", "dlt-data"),
        raw_dataset=os.getenv("RAW_DATASET", "kronodroid_raw"),
        raw_format=resolved_raw_format,
        k8s_minio_endpoint_url=os.getenv(
            "K8S_MINIO_ENDPOINT_URL", "http://minio.dfp.svc.cluster.local:9000"
        ),
        k8s_lakefs_endpoint_url=os.getenv(
            "K8S_LAKEFS_ENDPOINT_URL", "http://lakefs.dfp.svc.cluster.local:8000"
        ),
        minio_access_key_id=os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin"),
        minio_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin"),
        lakefs_repository=os.getenv("LAKEFS_REPOSITORY", "kronodroid"),
        lakefs_branch=branch,
        lakefs_access_key_id=os.getenv("LAKEFS_ACCESS_KEY_ID", ""),
        lakefs_secret_access_key=os.getenv("LAKEFS_SECRET_ACCESS_KEY", ""),
    )

    ok = run(cfg)
    if ok:
        print("Spark Operator transformations completed successfully")
    else:
        print("ERROR: Spark Operator transformations failed")
    return ok


def run_feast_apply(check_spark: bool = True) -> bool:
    """Apply Feast feature definitions.

    Args:
        check_spark: If True, check Spark Thrift Server connectivity first

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 3: Applying Feast feature definitions")
    print("=" * 60)

    # Check Spark connectivity for Feast offline store
    if check_spark:
        thrift_host = os.getenv("SPARK_THRIFT_HOST", "localhost")
        thrift_port = int(os.getenv("SPARK_THRIFT_PORT", "10000"))

        print(f"Checking Spark Thrift Server for Feast at {thrift_host}:{thrift_port}...")
        if not check_spark_thrift_server(thrift_host, thrift_port):
            print(f"WARNING: Cannot connect to Spark Thrift Server at {thrift_host}:{thrift_port}")
            print("\nFeast with Spark offline store requires Spark connectivity.")
            print("To start Spark Thrift Server in Kind cluster:")
            print("  kubectl apply -k infra/k8s/kind/addons/spark-thrift/")
            print("  kubectl -n dfp wait --for=condition=ready pod -l app=spark-thrift-server --timeout=120s")
            print("\nOr skip Feast steps with --skip-feast --skip-materialize")
            return False
        print(f"  ✓ Spark Thrift Server is reachable")

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


def run_feast_materialize(days_back: int = 30, check_spark: bool = True) -> bool:
    """Materialize features to online store.

    Args:
        days_back: Number of days of features to materialize
        check_spark: If True, check Spark Thrift Server connectivity first

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 4: Materializing features to online store")
    print("=" * 60)

    # Check Spark connectivity for Feast offline store
    if check_spark:
        thrift_host = os.getenv("SPARK_THRIFT_HOST", "localhost")
        thrift_port = int(os.getenv("SPARK_THRIFT_PORT", "10000"))

        print(f"Checking Spark Thrift Server for Feast at {thrift_host}:{thrift_port}...")
        if not check_spark_thrift_server(thrift_host, thrift_port):
            print(f"WARNING: Cannot connect to Spark Thrift Server at {thrift_host}:{thrift_port}")
            print("\nFeast with Spark offline store requires Spark connectivity.")
            print("To start Spark Thrift Server in Kind cluster:")
            print("  kubectl apply -k infra/k8s/kind/addons/spark-thrift/")
            print("  kubectl -n dfp wait --for=condition=ready pod -l app=spark-thrift-server --timeout=120s")
            return False
        print(f"  ✓ Spark Thrift Server is reachable")

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
        "--transform-runner",
        choices=["dbt", "spark-operator"],
        default="dbt",
        help="Step 2 runner: 'dbt' (dbt-spark via Thrift) or 'spark-operator' (SparkApplication)",
    )
    parser.add_argument(
        "--k8s-namespace",
        default="dfp",
        help="Kubernetes namespace for SparkApplication (default: dfp)",
    )
    parser.add_argument(
        "--spark-image",
        default="apache/spark:3.5.7-python3",
        help="Spark image for SparkApplication (default: apache/spark:3.5.7-python3)",
    )
    parser.add_argument(
        "--spark-service-account",
        default=os.getenv("SPARK_SERVICE_ACCOUNT", "spark-operator"),
        help="ServiceAccount for Spark driver/executor (default: spark-operator)",
    )
    parser.add_argument(
        "--spark-timeout-seconds",
        type=int,
        default=60 * 30,
        help="Timeout for SparkApplication completion (default: 1800)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip dlt ingestion step",
    )
    parser.add_argument(
        "--skip-dbt",
        action="store_true",
        help="Skip transformation step (dbt or SparkApplication)",
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
    parser.add_argument(
        "--skip-spark-check",
        action="store_true",
        help="Skip Spark Thrift Server connectivity check for Feast (use with caution)",
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
    print(f"  transform runner: {args.transform_runner}")
    raw_format = os.getenv("RAW_FORMAT")
    if not raw_format:
        raw_format = "parquet" if args.file_format == "avro" else args.file_format
        os.environ["RAW_FORMAT"] = raw_format

    print(f"  Raw format: {raw_format}")
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
            if args.transform_runner == "dbt":
                if not run_dbt_spark_transformations(args.dbt_target):
                    success = False
                    print("\nPipeline failed at dbt transformation step")
                    sys.exit(1)
            else:
                if not run_kubeflow_spark_operator_transformations(
                    branch=args.branch,
                    namespace=args.k8s_namespace,
                    spark_image=args.spark_image,
                    service_account=args.spark_service_account,
                    timeout_seconds=args.spark_timeout_seconds,
                    raw_format=raw_format,
                ):
                    success = False
                    print("\nPipeline failed at SparkApplication transformation step")
                    sys.exit(1)

        # Step 3: Feast apply
        if not args.skip_feast:
            check_spark = not args.skip_spark_check
            if not run_feast_apply(check_spark=check_spark):
                success = False
                print("\nPipeline failed at Feast apply step")
                sys.exit(1)

        # Step 4: Materialize features
        if not args.skip_materialize:
            check_spark = not args.skip_spark_check
            if not run_feast_materialize(args.materialize_days, check_spark=check_spark):
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
