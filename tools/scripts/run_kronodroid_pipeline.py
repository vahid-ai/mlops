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
from urllib.parse import quote, urlparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Type alias for transform engine
TransformEngine = Literal["dbt", "kubeflow"]


def _lakefs_quote_ref(ref: str) -> str:
    """URL-encode a LakeFS ref (branch/commit/tag) for use in path segments."""
    return quote(ref, safe="")


def _is_localhost_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.hostname in {"localhost", "127.0.0.1"} if parsed.hostname else False


def _k8s_service_url(service: str, namespace: str, port: int) -> str:
    return f"http://{service}.{namespace}.svc.cluster.local:{port}"


def _ensure_k8s_prereqs(
    namespace: str,
    service_account: str,
    minio_secret_name: str,
    lakefs_secret_name: str,
) -> None:
    """Ensure K8s objects needed by Spark-on-K8s exist."""
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    # Configure Kubernetes client (works both in-cluster and from local kubeconfig).
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    core_api = client.CoreV1Api()
    rbac_api = client.RbacAuthorizationV1Api()

    # ServiceAccount
    try:
        core_api.read_namespaced_service_account(service_account, namespace)
    except ApiException as e:
        if e.status != 404:
            raise
        core_api.create_namespaced_service_account(
            namespace,
            client.V1ServiceAccount(metadata=client.V1ObjectMeta(name=service_account)),
        )

    # Role (minimal Spark driver permissions)
    role_name = "spark-role"
    role_rules = [
        client.V1PolicyRule(
            api_groups=[""],
            resources=["pods"],
            verbs=["get", "list", "watch", "create", "delete", "deletecollection", "patch"],
        ),
        client.V1PolicyRule(
            api_groups=[""],
            resources=["services"],
            verbs=["get", "list", "watch", "create", "delete", "deletecollection"],
        ),
        client.V1PolicyRule(
            api_groups=[""],
            resources=["configmaps"],
            verbs=["get", "list", "watch", "create", "delete", "deletecollection", "update"],
        ),
        client.V1PolicyRule(
            api_groups=[""],
            resources=["secrets"],
            verbs=["get", "list", "watch"],
        ),
        client.V1PolicyRule(
            api_groups=[""],
            resources=["persistentvolumeclaims"],
            verbs=["get", "list", "watch", "create", "delete", "deletecollection"],
        ),
    ]
    try:
        rbac_api.read_namespaced_role(role_name, namespace)
    except ApiException as e:
        if e.status != 404:
            raise
        rbac_api.create_namespaced_role(
            namespace,
            client.V1Role(
                metadata=client.V1ObjectMeta(name=role_name),
                rules=role_rules,
            ),
        )

    # RoleBinding
    rb_name = "spark-rolebinding"
    try:
        rbac_api.read_namespaced_role_binding(rb_name, namespace)
    except ApiException as e:
        if e.status != 404:
            raise
        rbac_api.create_namespaced_role_binding(
            namespace,
            client.V1RoleBinding(
                metadata=client.V1ObjectMeta(name=rb_name),
                role_ref=client.V1RoleRef(
                    api_group="rbac.authorization.k8s.io",
                    kind="Role",
                    name=role_name,
                ),
                subjects=[
                    client.RbacV1Subject(
                        kind="ServiceAccount",
                        name=service_account,
                        namespace=namespace,
                    )
                ],
            ),
        )

    def _ensure_secret(secret_name: str, data: dict[str, str]) -> None:
        try:
            core_api.read_namespaced_secret(secret_name, namespace)
        except ApiException as e:
            if e.status != 404:
                raise
            core_api.create_namespaced_secret(
                namespace,
                client.V1Secret(
                    metadata=client.V1ObjectMeta(name=secret_name),
                    type="Opaque",
                    string_data={k: v for k, v in data.items() if v},
                ),
            )

    _ensure_secret(
        minio_secret_name,
        {
            "MINIO_ACCESS_KEY_ID": os.getenv("MINIO_ACCESS_KEY_ID", ""),
            "MINIO_SECRET_ACCESS_KEY": os.getenv("MINIO_SECRET_ACCESS_KEY", ""),
        },
    )
    _ensure_secret(
        lakefs_secret_name,
        {
            "LAKEFS_ACCESS_KEY_ID": os.getenv("LAKEFS_ACCESS_KEY_ID", ""),
            "LAKEFS_SECRET_ACCESS_KEY": os.getenv("LAKEFS_SECRET_ACCESS_KEY", ""),
        },
    )


def _check_sparkoperator_crd() -> tuple[bool, bool]:
    """Check if the SparkApplication CRD exists and supports status subresource.

    Returns:
        Tuple of (crd_exists, supports_status)
    """
    try:
        from kubernetes import client
        from kubernetes.client.rest import ApiException
    except Exception:
        return False, False

    api = client.ApiextensionsV1Api()
    try:
        crd = api.read_custom_resource_definition("sparkapplications.sparkoperator.k8s.io")
    except ApiException:
        return False, False

    supports_status = False
    for version in crd.spec.versions or []:
        if version.name == "v1beta2" and version.subresources and version.subresources.status is not None:
            supports_status = True
            break
    return True, supports_status


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

        # If using LakeFS, commit ingestion results immediately so the branch is not dirty
        # for subsequent transformation steps that may want to merge into it.
        if destination == "lakefs":
            commit_to_lakefs(
                lakefs_branch or "main",
                f"Ingested raw data via dlt: {datetime.now().isoformat()}"
            )

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

                if destination == "lakefs":
                    commit_to_lakefs(
                        lakefs_branch or "main",
                        f"Ingested raw data via dlt (retry): {datetime.now().isoformat()}"
                    )

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
    destination: str = "lakefs",
    lakefs_branch: str = "main",
    minio_endpoint: str | None = None,
    minio_bucket: str | None = None,
    lakefs_endpoint: str | None = None,
    lakefs_repository: str | None = None,
    spark_image: str = "dfp-spark:latest",
    namespace: str = "default",
    service_account: str = "spark",
    minio_secret_name: str = "minio-credentials",
    lakefs_secret_name: str = "lakefs-credentials",
    driver_cores: int = 1,
    driver_memory: str = "4g",
    executor_cores: int = 2,
    executor_instances: int = 2,
    executor_memory: str = "6g",
    timeout_seconds: int = 3600,
    use_kfp_client: bool = False,
    kfp_host: str | None = None,
    test_limit: int = 0,
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
        test_limit: Limit rows per table for faster testing (0 = no limit)

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

    # Preserve host-accessible endpoints for LakeFS API calls (branch/commit/merge).
    lakefs_api_endpoint = lakefs_endpoint

    # Spark runs inside the cluster, so rewrite localhost endpoints to in-cluster Service DNS.
    minio_cluster_endpoint = (
        _k8s_service_url("minio", namespace, 9000) if _is_localhost_url(minio_endpoint) else minio_endpoint
    )
    lakefs_cluster_endpoint = (
        _k8s_service_url("lakefs", namespace, 8000) if _is_localhost_url(lakefs_endpoint) else lakefs_endpoint
    )

    # For `--destination lakefs`, dlt writes raw data to the LakeFS S3 gateway bucket
    # (the repository name), not the MinIO raw bucket. Use that as the Spark input.
    if destination == "lakefs":
        raw_bucket = lakefs_repository
        raw_endpoint = lakefs_cluster_endpoint
        raw_prefix = f"{lakefs_branch}/kronodroid_raw"
    else:
        raw_bucket = minio_bucket
        raw_endpoint = minio_cluster_endpoint
        raw_prefix = "kronodroid_raw"

    print(f"  MinIO endpoint (host): {minio_endpoint}")
    print(f"  MinIO endpoint (cluster): {minio_cluster_endpoint}")
    print(f"  MinIO bucket: {minio_bucket}")
    print(f"  LakeFS endpoint (host): {lakefs_endpoint}")
    print(f"  LakeFS endpoint (cluster): {lakefs_cluster_endpoint}")
    print(f"  LakeFS repository: {lakefs_repository}")
    print(f"  LakeFS branch: {lakefs_branch}")
    print(f"  Raw input endpoint: {raw_endpoint}")
    print(f"  Raw input bucket: {raw_bucket}")
    print(f"  Raw input prefix: {raw_prefix}")
    print(f"  Transform engine: Kubeflow SparkOperator")

    # Ensure K8s objects exist before submitting (SA/RBAC/Secrets).
    _ensure_k8s_prereqs(
        namespace=namespace,
        service_account=service_account,
        minio_secret_name=minio_secret_name,
        lakefs_secret_name=lakefs_secret_name,
    )

    if use_kfp_client:
        return _run_kubeflow_via_kfp_client(
            kfp_host=kfp_host,
            minio_endpoint=raw_endpoint,
            minio_bucket=raw_bucket,
            minio_prefix=raw_prefix,
            lakefs_endpoint=lakefs_cluster_endpoint,
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
            destination=destination,
            minio_endpoint=minio_cluster_endpoint,
            minio_bucket=minio_bucket,
            lakefs_endpoint=lakefs_cluster_endpoint,
            lakefs_api_endpoint=lakefs_api_endpoint,
            lakefs_repository=lakefs_repository,
            lakefs_branch=lakefs_branch,
            raw_endpoint=raw_endpoint,
            raw_bucket=raw_bucket,
            raw_prefix=raw_prefix,
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
            test_limit=test_limit,
        )


def _run_kubeflow_via_kfp_client(
    kfp_host: str | None,
    minio_endpoint: str,
    minio_bucket: str,
    minio_prefix: str,
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
                "minio_prefix": minio_prefix,
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

        if result.state == "SUCCEEDED":
            print("Kubeflow SparkOperator transformations completed successfully")
            return True
        else:
            print(f"ERROR: KFP run failed with status: {result.state}")
            return False

    except Exception as e:
        print(f"ERROR: Kubeflow pipeline failed: {e}")
        return False


def _run_kubeflow_via_sparkapplication(
    destination: str,
    minio_endpoint: str,
    minio_bucket: str,
    lakefs_endpoint: str,
    lakefs_api_endpoint: str,
    lakefs_repository: str,
    lakefs_branch: str,
    raw_endpoint: str,
    raw_bucket: str,
    raw_prefix: str,
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
    test_limit: int = 0,
) -> bool:
    """Submit a SparkApplication directly via kubectl."""
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError as e:
        print(f"ERROR: kubernetes client not installed: {e}")
        print("Install with: pip install kubernetes")
        return False

    # Load kubeconfig early so we can check CRD before creating branches
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    # Check if SparkOperator CRD is installed before creating any resources
    crd_exists, crd_supports_status = _check_sparkoperator_crd()
    if not crd_exists:
        print("ERROR: SparkOperator CRD (sparkapplications.sparkoperator.k8s.io) not found.")
        print("  The Spark Operator must be installed to use the kubeflow transform engine.")
        print("  Install with Helm:")
        print("    helm repo add spark-operator https://kubeflow.github.io/spark-operator")
        print("    helm install spark-operator spark-operator/spark-operator \\")
        print(f"      --namespace {namespace} --create-namespace")
        print("  Or see: https://github.com/kubeflow/spark-operator")
        return False

    if not crd_supports_status:
        print(
            "  Note: SparkApplication CRD has no /status subresource; "
            "SparkOperator can't report application state. Monitoring driver pod instead."
        )

    run_id = str(uuid.uuid4())[:8]
    app_name = f"kronodroid-iceberg-{run_id}"
    per_run_branch = f"spark-{run_id}"

    print(f"  Run ID: {run_id}")
    print(f"  SparkApplication: {app_name}")
    print(f"  Per-run branch: {per_run_branch}")

    # Create per-run LakeFS branch
    if not _create_lakefs_branch(lakefs_api_endpoint, lakefs_repository, per_run_branch, lakefs_branch):
        return False

    # Build SparkApplication manifest
    # Use Hadoop-based Iceberg catalog since LakeFS OSS doesn't include REST catalog
    catalog_name = "lakefs"
    # IMPORTANT: Use the per-run branch for the warehouse path, not the target branch.
    # Iceberg's Hadoop catalog derives table paths from the warehouse, and if the Spark job
    # writes to a different branch than what's in the warehouse path, Iceberg will raise:
    #   "Cannot set a custom location for a path-based table"
    warehouse_path = f"s3a://{lakefs_repository}/{per_run_branch}/iceberg"

    minio_access_key = os.getenv("MINIO_ACCESS_KEY_ID", "")
    minio_secret_key = os.getenv("MINIO_SECRET_ACCESS_KEY", "")
    lakefs_access_key = os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    lakefs_secret_key = os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")

    if destination == "minio" and (not minio_access_key or not minio_secret_key):
        print("ERROR: MINIO_ACCESS_KEY_ID / MINIO_SECRET_ACCESS_KEY must be set for MinIO input.")
        return False
    if not lakefs_access_key or not lakefs_secret_key:
        print("ERROR: LAKEFS_ACCESS_KEY_ID / LAKEFS_SECRET_ACCESS_KEY must be set for LakeFS output.")
        return False

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
            "imagePullPolicy": "IfNotPresent",
            "mainApplicationFile": "local:///opt/spark/work-dir/kronodroid_iceberg_job.py",
            "arguments": [
                "--minio-bucket", raw_bucket,
                "--minio-prefix", raw_prefix,
                "--lakefs-repository", lakefs_repository,
                "--lakefs-branch", per_run_branch,
                "--catalog-name", catalog_name,
                "--staging-database", "stg_kronodroid",
                "--marts-database", "kronodroid",
            ] + (["--test-limit", str(test_limit)] if test_limit > 0 else []),
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
                    {"name": "MINIO_ENDPOINT_URL", "value": raw_endpoint},
                    {"name": "LAKEFS_ENDPOINT_URL", "value": lakefs_endpoint},
                    {"name": "LAKEFS_REPOSITORY", "value": lakefs_repository},
                    {"name": "LAKEFS_BRANCH", "value": per_run_branch},
                    {"name": "AWS_REGION", "value": os.getenv("MINIO_REGION", "us-east-1")},
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
                    {"name": "MINIO_ENDPOINT_URL", "value": raw_endpoint},
                    {"name": "LAKEFS_ENDPOINT_URL", "value": lakefs_endpoint},
                    {"name": "LAKEFS_REPOSITORY", "value": lakefs_repository},
                    {"name": "LAKEFS_BRANCH", "value": per_run_branch},
                    {"name": "AWS_REGION", "value": os.getenv("MINIO_REGION", "us-east-1")},
                ],
            },
            # Note: deps.packages removed - JARs are pre-built into the Docker image
            # to avoid version mismatches between driver and executor pods
            "sparkConf": {
                "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
                f"spark.sql.catalog.{catalog_name}": "org.apache.iceberg.spark.SparkCatalog",
                f"spark.sql.catalog.{catalog_name}.type": "hadoop",
                f"spark.sql.catalog.{catalog_name}.warehouse": warehouse_path,
                "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                "spark.hadoop.fs.s3a.path.style.access": "true",
                "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
                # Raw input bucket endpoint (+ optional per-bucket creds).
                f"spark.hadoop.fs.s3a.bucket.{raw_bucket}.endpoint": raw_endpoint,
                # LakeFS repo endpoint (+ per-bucket creds for Iceberg warehouse).
                f"spark.hadoop.fs.s3a.bucket.{lakefs_repository}.endpoint": lakefs_endpoint,
                f"spark.hadoop.fs.s3a.bucket.{lakefs_repository}.access.key": lakefs_access_key,
                f"spark.hadoop.fs.s3a.bucket.{lakefs_repository}.secret.key": lakefs_secret_key,
                # Memory optimization settings
                "spark.driver.memoryOverheadFactor": "0.2",
                "spark.executor.memoryOverheadFactor": "0.2",
                "spark.sql.shuffle.partitions": "16",
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.memory.fraction": "0.8",
                "spark.memory.storageFraction": "0.3",
            },
        },
    }

    if destination == "minio":
        spark_app["spec"]["sparkConf"].update(
            {
                f"spark.hadoop.fs.s3a.bucket.{raw_bucket}.access.key": minio_access_key,
                f"spark.hadoop.fs.s3a.bucket.{raw_bucket}.secret.key": minio_secret_key,
            }
        )

    try:
        api = client.CustomObjectsApi()
        core_api = client.CoreV1Api()

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
                app_state = status.get("applicationState", {}).get("state")

                if not app_state:
                    driver_pod_name = f"{app_name}-driver"
                    try:
                        pod = core_api.read_namespaced_pod(driver_pod_name, namespace)
                        phase = pod.status.phase or "UNKNOWN"
                        waiting_reason = None
                        waiting_message = None
                        for container_status in pod.status.container_statuses or []:
                            state = container_status.state
                            waiting = state.waiting if state else None
                            if waiting and waiting.reason:
                                waiting_reason = waiting.reason
                                waiting_message = waiting.message
                                break

                        if phase == "Pending" and waiting_reason in ("ErrImagePull", "ImagePullBackOff"):
                            app_state = waiting_reason
                            print(f"ERROR: Spark driver image pull failed: {waiting_reason}")
                            if waiting_message:
                                print(f"  {waiting_message}")
                            print("  If using kind, build and load the image:")
                            print("    docker build -t dfp-spark:latest -f tools/docker/Dockerfile.spark .")
                            print("    kind load docker-image dfp-spark:latest --name dfp-kind")
                            return False

                        if phase == "Pending" and waiting_reason == "CreateContainerConfigError":
                            app_state = waiting_reason
                            print("ERROR: Spark driver failed to start (CreateContainerConfigError).")
                            print(f"  Check secrets in namespace {namespace}: {minio_secret_name}, {lakefs_secret_name}")
                            return False

                        if phase == "Succeeded":
                            app_state = "COMPLETED"
                        elif phase == "Failed":
                            app_state = "FAILED"
                        elif phase == "Running":
                            app_state = "RUNNING"
                        elif phase == "Pending":
                            app_state = "PENDING"
                        else:
                            app_state = phase
                    except ApiException as e:
                        if e.status == 404:
                            app_state = "SUBMITTED"
                        else:
                            raise
                print(f"  SparkApplication state: {app_state} (elapsed: {int(elapsed)}s)")

                if app_state == "COMPLETED":
                    # Commit and merge the per-run branch
                    merge_success = _commit_and_merge_lakefs(
                        lakefs_api_endpoint, lakefs_repository, per_run_branch, lakefs_branch, run_id
                    )
                    if merge_success:
                        print("Kubeflow SparkOperator transformations completed successfully")
                        return True
                    else:
                        print("WARNING: LakeFS commit/merge failed, but Spark job succeeded")
                        print(f"  Data is available on branch: {per_run_branch}")
                        print(f"  You can manually merge with: lakectl merge lakefs://{lakefs_repository}/{per_run_branch} lakefs://{lakefs_repository}/{lakefs_branch}")
                        # Still return True since Spark job succeeded - data is available on branch
                        return True

                elif app_state in ("FAILED", "SUBMISSION_FAILED", "FAILING"):
                    error_msg = status.get("applicationState", {}).get("errorMessage", "Unknown")
                    if error_msg and error_msg != "Unknown":
                        print(f"ERROR: SparkApplication failed: {error_msg}")
                    else:
                        print("ERROR: SparkApplication failed (no operator status available).")
                        print(f"  Check driver pod: kubectl -n {namespace} describe pod {app_name}-driver")
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
    """Commit and merge a per-run LakeFS branch.

    Returns:
        True if successful, False if there was a fatal error.
    """
    import requests

    access_key = os.getenv("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("LAKEFS_SECRET_ACCESS_KEY", "")

    api_base = endpoint.rstrip("/")
    auth = (access_key, secret_key)

    print(f"  LakeFS Commit & Merge:")
    print(f"    Source branch: {source_branch}")
    print(f"    Target branch: {target_branch}")

    # Step 1: Check for uncommitted changes (diff)
    diff_url = f"{api_base}/api/v1/repositories/{repository}/branches/{_lakefs_quote_ref(source_branch)}/diff"
    try:
        diff_resp = requests.get(diff_url, auth=auth, timeout=30)
    except requests.RequestException as e:
        print(f"  ERROR: Failed to check diff: {e}")
        return False

    if diff_resp.status_code != 200:
        print(f"  ERROR: Failed to get diff: {diff_resp.status_code}")
        print(f"    Response: {diff_resp.text[:500]}")
        return False

    diff_data = diff_resp.json()
    changes = diff_data.get("results", [])
    has_more = diff_data.get("pagination", {}).get("has_more", False)

    if not changes:
        print(f"    No uncommitted changes found on branch '{source_branch}'")
        commit_id = "no-changes"
    else:
        change_count = len(changes)
        if has_more:
            print(f"    Found {change_count}+ uncommitted changes")
        else:
            print(f"    Found {change_count} uncommitted changes")

        # Step 2: Commit changes
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

        try:
            commit_resp = requests.post(commit_url, json=commit_data, auth=auth, timeout=30)
        except requests.RequestException as e:
            print(f"  ERROR: Commit request failed: {e}")
            return False

        if commit_resp.status_code in (200, 201):
            commit_id = commit_resp.json().get("id", "unknown")
            print(f"    Committed: {commit_id}")
        else:
            print(f"  ERROR: Commit failed: {commit_resp.status_code}")
            print(f"    Response: {commit_resp.text[:500]}")
            return False

    # Step 3: Check and commit any uncommitted changes on target branch
    # LakeFS requires the target branch to be clean before merging
    target_diff_url = f"{api_base}/api/v1/repositories/{repository}/branches/{_lakefs_quote_ref(target_branch)}/diff"
    try:
        target_diff_resp = requests.get(target_diff_url, auth=auth, timeout=30)
        if target_diff_resp.status_code == 200:
            target_changes = target_diff_resp.json().get("results", [])
            if target_changes:
                print(f"    Target branch '{target_branch}' has {len(target_changes)} uncommitted changes, committing...")
                target_commit_url = (
                    f"{api_base}/api/v1/repositories/{repository}/branches/{_lakefs_quote_ref(target_branch)}/commits"
                )
                target_commit_data = {
                    "message": f"Auto-commit uncommitted changes before merge from {source_branch}",
                    "metadata": {
                        "pipeline": "kronodroid-iceberg",
                        "run_id": run_id,
                        "auto_commit": "true",
                    },
                }
                target_commit_resp = requests.post(target_commit_url, json=target_commit_data, auth=auth, timeout=30)
                if target_commit_resp.status_code in (200, 201):
                    target_commit_id = target_commit_resp.json().get("id", "unknown")
                    print(f"    Auto-committed target branch: {target_commit_id}")
                else:
                    print(f"    Warning: Failed to auto-commit target branch: {target_commit_resp.status_code}")
    except requests.RequestException as e:
        print(f"    Warning: Failed to check target branch diff: {e}")

    # Step 4: Merge into target branch
    merge_url = (
        f"{api_base}/api/v1/repositories/{repository}/refs/{_lakefs_quote_ref(source_branch)}/merge/{_lakefs_quote_ref(target_branch)}"
    )
    merge_data = {
        "message": f"Merge {source_branch} into {target_branch}",
        "metadata": {
            "pipeline": "kronodroid-iceberg",
            "run_id": run_id,
            "source_commit": commit_id,
        },
    }

    try:
        merge_resp = requests.post(merge_url, json=merge_data, auth=auth, timeout=60)
    except requests.RequestException as e:
        print(f"  ERROR: Merge request failed: {e}")
        return False

    if merge_resp.status_code in (200, 201):
        merge_ref = merge_resp.json().get("reference", "unknown")
        print(f"    Merged to {target_branch}: {merge_ref}")
    elif merge_resp.status_code == 409:
        # 409 Conflict - could be "no changes to merge" or actual conflict
        error_text = merge_resp.text.lower()
        if "no changes" in error_text or "already up to date" in error_text:
            print(f"    Nothing to merge (branches are identical)")
        else:
            print(f"  ERROR: Merge conflict: {merge_resp.status_code}")
            print(f"    Response: {merge_resp.text[:500]}")
            return False
    elif merge_resp.status_code == 400:
        # 400 Bad Request - check for dirty branch error
        error_text = merge_resp.text.lower()
        if "uncommitted" in error_text or "dirty" in error_text:
            print(f"  ERROR: Target branch has uncommitted changes that could not be auto-committed")
            print(f"    Please manually commit or reset changes on '{target_branch}' before merging")
            print(f"    Response: {merge_resp.text[:500]}")
        else:
            print(f"  ERROR: Merge failed: {merge_resp.status_code}")
            print(f"    Response: {merge_resp.text[:500]}")
        return False
    else:
        print(f"  ERROR: Merge failed: {merge_resp.status_code}")
        print(f"    Response: {merge_resp.text[:500]}")
        return False

    # Step 5: Delete source branch (cleanup)
    delete_url = f"{api_base}/api/v1/repositories/{repository}/branches/{_lakefs_quote_ref(source_branch)}"
    try:
        delete_resp = requests.delete(delete_url, auth=auth, timeout=30)
        if delete_resp.status_code in (200, 204):
            print(f"    Deleted branch: {source_branch}")
        else:
            print(f"    Warning: Failed to delete branch: {delete_resp.status_code}")
    except requests.RequestException as e:
        print(f"    Warning: Failed to delete branch: {e}")

    return True


def run_feast_apply(
    lakefs_endpoint: str | None = None,
    redis_connection: str | None = None,
    skip_source_validation: bool = False,
) -> bool:
    """Apply Feast feature definitions.

    Args:
        lakefs_endpoint: LakeFS endpoint URL (default: http://localhost:8000)
        redis_connection: Redis connection string (default: redis://localhost:6379)
        skip_source_validation: Skip validation of data sources (useful if tables don't exist yet)

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 3: Applying Feast feature definitions")
    print("=" * 60)

    feast_dir = PROJECT_ROOT / "feature_stores" / "feast_store"

    # Set up environment for Feast
    # Use localhost endpoints for local Feast CLI (port-forwarded from k8s)
    feast_env = os.environ.copy()
    feast_env["LAKEFS_ENDPOINT_URL"] = lakefs_endpoint or os.getenv(
        "LAKEFS_ENDPOINT_URL", "http://localhost:8000"
    )
    feast_env["REDIS_CONNECTION_STRING"] = redis_connection or os.getenv(
        "REDIS_CONNECTION_STRING", "redis://localhost:6379"
    )

    # Ensure LakeFS credentials are set
    if not feast_env.get("LAKEFS_ACCESS_KEY_ID") or not feast_env.get("LAKEFS_SECRET_ACCESS_KEY"):
        print("WARNING: LAKEFS_ACCESS_KEY_ID/LAKEFS_SECRET_ACCESS_KEY not set")
        print("  Feast may not be able to access LakeFS Iceberg tables")

    print(f"  LakeFS endpoint: {feast_env['LAKEFS_ENDPOINT_URL']}")
    print(f"  Redis connection: {feast_env['REDIS_CONNECTION_STRING']}")

    try:
        cmd = ["feast", "apply"]
        if skip_source_validation:
            cmd.append("--skip-source-validation")
            print("  Skipping source validation")

        result = subprocess.run(
            cmd,
            cwd=str(feast_dir),
            check=True,
            capture_output=True,
            text=True,
            env=feast_env,
        )
        print(result.stdout)
        print("Feast feature definitions applied successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: feast apply failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: feast command not found. Install with: pip install feast")
        return False


def run_feast_materialize(
    days_back: int = 30,
    lakefs_endpoint: str | None = None,
    redis_connection: str | None = None,
) -> bool:
    """Materialize features to online store.

    Args:
        days_back: Number of days of features to materialize
        lakefs_endpoint: LakeFS endpoint URL (default: http://localhost:8000)
        redis_connection: Redis connection string (default: redis://localhost:6379)

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("Step 4: Materializing features to online store")
    print("=" * 60)

    feast_dir = PROJECT_ROOT / "feature_stores" / "feast_store"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Set up environment for Feast
    feast_env = os.environ.copy()
    feast_env["LAKEFS_ENDPOINT_URL"] = lakefs_endpoint or os.getenv(
        "LAKEFS_ENDPOINT_URL", "http://localhost:8000"
    )
    feast_env["REDIS_CONNECTION_STRING"] = redis_connection or os.getenv(
        "REDIS_CONNECTION_STRING", "redis://localhost:6379"
    )

    print(f"  LakeFS endpoint: {feast_env['LAKEFS_ENDPOINT_URL']}")
    print(f"  Redis connection: {feast_env['REDIS_CONNECTION_STRING']}")
    print(f"  Date range: {start_date.isoformat()} to {end_date.isoformat()}")

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
            env=feast_env,
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
        default="dfp-spark:latest",
        help="Spark Docker image (for kubeflow engine)",
    )
    parser.add_argument(
        "--k8s-namespace",
        default="dfp",
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
        "--skip-source-validation",
        action="store_true",
        help="Skip Feast source validation (use if Iceberg tables aren't accessible locally)",
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
    # Spark resource options for testing
    parser.add_argument(
        "--driver-memory",
        default="4g",
        help="Spark driver memory (default: 4g)",
    )
    parser.add_argument(
        "--executor-memory",
        default="6g",
        help="Spark executor memory (default: 6g)",
    )
    parser.add_argument(
        "--executor-instances",
        type=int,
        default=2,
        help="Number of Spark executors (default: 2)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with minimal resources and faster timeout",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Limit rows per source table for testing (0 = no limit, e.g., 1000 for quick tests)",
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
                # Apply test mode defaults for faster iteration
                driver_memory = args.driver_memory
                executor_memory = args.executor_memory
                executor_instances = args.executor_instances
                spark_timeout = args.spark_timeout

                test_limit = args.test_limit
                if args.test_mode:
                    print("  Test mode enabled: using minimal resources")
                    driver_memory = "1g"
                    executor_memory = "1g"
                    executor_instances = 1
                    spark_timeout = min(spark_timeout, 600)  # Max 10 min in test mode
                    if test_limit == 0:
                        test_limit = 100  # Default to 100 rows in test mode

                if not run_kubeflow_transformations(
                    destination=args.destination,
                    lakefs_branch=args.branch,
                    spark_image=args.spark_image,
                    namespace=args.k8s_namespace,
                    timeout_seconds=spark_timeout,
                    use_kfp_client=args.use_kfp_client,
                    kfp_host=args.kfp_host,
                    driver_memory=driver_memory,
                    executor_memory=executor_memory,
                    executor_instances=executor_instances,
                    test_limit=test_limit,
                ):
                    success = False
                    print("\nPipeline failed at Kubeflow SparkOperator transformation step")
                    sys.exit(1)

        # Step 3: Feast apply
        if not args.skip_feast:
            if not run_feast_apply(skip_source_validation=args.skip_source_validation):
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
