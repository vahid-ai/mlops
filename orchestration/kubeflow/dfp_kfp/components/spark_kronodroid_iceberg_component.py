"""Component: Spark Kronodroid Iceberg transformation via SparkOperator.

This KFP component:
1. Creates a per-run LakeFS branch (spark-<run_id>)
2. Submits a SparkApplication CRD to run the kronodroid_iceberg_job
3. Monitors the SparkApplication until completion

The SparkApplication runs on Kubernetes via the Spark Operator and writes
Iceberg tables with Avro format to the LakeFS Iceberg REST catalog.
"""

import os
import time
import uuid
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Output, Artifact


# SparkApplication template for the Kronodroid Iceberg job
SPARK_APPLICATION_TEMPLATE = """
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: {spark_image}
  imagePullPolicy: IfNotPresent
  mainApplicationFile: local:///opt/spark/work-dir/kronodroid_iceberg_job.py
  arguments:
    - "--minio-bucket"
    - "{minio_bucket}"
    - "--minio-prefix"
    - "{minio_prefix}"
    - "--lakefs-repository"
    - "{lakefs_repository}"
    - "--lakefs-branch"
    - "{lakefs_branch}"
    - "--catalog-name"
    - "{catalog_name}"
    - "--staging-database"
    - "{staging_database}"
    - "--marts-database"
    - "{marts_database}"
  sparkVersion: "3.5.0"
  restartPolicy:
    type: Never
  driver:
    cores: {driver_cores}
    memory: "{driver_memory}"
    serviceAccount: {service_account}
    env:
      - name: MINIO_ENDPOINT_URL
        value: "{minio_endpoint}"
      - name: MINIO_ACCESS_KEY_ID
        valueFrom:
          secretKeyRef:
            name: {minio_secret_name}
            key: access-key
      - name: MINIO_SECRET_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: {minio_secret_name}
            key: secret-key
      - name: LAKEFS_ENDPOINT_URL
        value: "{lakefs_endpoint}"
      - name: LAKEFS_ACCESS_KEY_ID
        valueFrom:
          secretKeyRef:
            name: {lakefs_secret_name}
            key: access-key
      - name: LAKEFS_SECRET_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: {lakefs_secret_name}
            key: secret-key
      - name: LAKEFS_REPOSITORY
        value: "{lakefs_repository}"
      - name: LAKEFS_BRANCH
        value: "{lakefs_branch}"
  executor:
    cores: {executor_cores}
    instances: {executor_instances}
    memory: "{executor_memory}"
    env:
      - name: MINIO_ENDPOINT_URL
        value: "{minio_endpoint}"
      - name: MINIO_ACCESS_KEY_ID
        valueFrom:
          secretKeyRef:
            name: {minio_secret_name}
            key: access-key
      - name: MINIO_SECRET_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: {minio_secret_name}
            key: secret-key
      - name: LAKEFS_ENDPOINT_URL
        value: "{lakefs_endpoint}"
      - name: LAKEFS_ACCESS_KEY_ID
        valueFrom:
          secretKeyRef:
            name: {lakefs_secret_name}
            key: access-key
      - name: LAKEFS_SECRET_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: {lakefs_secret_name}
            key: secret-key
      - name: LAKEFS_REPOSITORY
        value: "{lakefs_repository}"
      - name: LAKEFS_BRANCH
        value: "{lakefs_branch}"
  deps:
    packages:
      # iceberg-spark-runtime includes avro support
      - org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2
      # AWS/S3 support for LakeFS S3 gateway
      - org.apache.hadoop:hadoop-aws:3.3.4
      - com.amazonaws:aws-java-sdk-bundle:1.12.262
  sparkConf:
    # Use /tmp for Ivy cache (writable in container)
    spark.jars.ivy: /tmp/.ivy2
    spark.sql.extensions: org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
    spark.sql.iceberg.write.format.default: avro
    spark.sql.iceberg.write.avro.compression-codec: snappy
    # Use Hadoop catalog with s3a:// paths for compatibility with training jobs
    spark.sql.catalog.lakefs: org.apache.iceberg.spark.SparkCatalog
    spark.sql.catalog.lakefs.type: hadoop
    spark.sql.catalog.lakefs.warehouse: "s3a://{lakefs_repository}/{target_branch}"
    spark.sql.catalog.lakefs.cache-enabled: "false"
    # S3A filesystem config - lakeFS acts as S3 gateway
    spark.hadoop.fs.s3.impl: org.apache.hadoop.fs.s3a.S3AFileSystem
    spark.hadoop.fs.s3a.impl: org.apache.hadoop.fs.s3a.S3AFileSystem
    spark.hadoop.fs.s3a.path.style.access: "true"
    spark.hadoop.fs.s3a.connection.ssl.enabled: "false"
    # Per-bucket endpoints and credentials for MinIO (dlt-data bucket)
    spark.hadoop.fs.s3a.bucket.{minio_bucket}.endpoint: "{minio_endpoint}"
    spark.hadoop.fs.s3a.bucket.{minio_bucket}.access.key: "{minio_access_key}"
    spark.hadoop.fs.s3a.bucket.{minio_bucket}.secret.key: "{minio_secret_key}"
    # Per-bucket endpoints and credentials for LakeFS (kronodroid bucket)
    spark.hadoop.fs.s3a.bucket.{lakefs_repository}.endpoint: "{lakefs_endpoint}"
    spark.hadoop.fs.s3a.bucket.{lakefs_repository}.access.key: "{lakefs_access_key}"
    spark.hadoop.fs.s3a.bucket.{lakefs_repository}.secret.key: "{lakefs_secret_key}"
"""


class SparkJobOutput(NamedTuple):
    """Output from the Spark job component."""

    lakefs_branch: str
    app_name: str
    status: str


def create_lakefs_branch(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    repository: str,
    branch_name: str,
    source_branch: str = "main",
) -> bool:
    """Create a LakeFS branch for this pipeline run.

    Args:
        endpoint_url: LakeFS endpoint URL
        access_key: LakeFS access key
        secret_key: LakeFS secret key
        repository: LakeFS repository name
        branch_name: Name of the branch to create
        source_branch: Source branch to create from

    Returns:
        True if branch was created, False if it already existed
    """
    import requests

    api_base = endpoint_url.rstrip("/")
    auth = (access_key, secret_key)

    # Check if branch exists
    branch_url = f"{api_base}/api/v1/repositories/{repository}/branches/{branch_name}"
    resp = requests.get(branch_url, auth=auth)

    if resp.status_code == 200:
        print(f"Branch {branch_name} already exists")
        return False

    if resp.status_code == 404:
        # Create branch from source
        create_url = f"{api_base}/api/v1/repositories/{repository}/branches"
        data = {"name": branch_name, "source": source_branch}
        create_resp = requests.post(create_url, json=data, auth=auth)

        if create_resp.status_code in (200, 201):
            print(f"Created branch: {branch_name}")
            return True
        else:
            raise RuntimeError(
                f"Failed to create branch: {create_resp.status_code} - {create_resp.text}"
            )

    raise RuntimeError(f"Failed to check branch: {resp.status_code} - {resp.text}")


def submit_spark_application(
    app_yaml: str,
    namespace: str = "default",
) -> str:
    """Submit a SparkApplication to Kubernetes.

    Args:
        app_yaml: YAML content of the SparkApplication
        namespace: Kubernetes namespace

    Returns:
        Name of the submitted SparkApplication
    """
    from kubernetes import client, config
    import yaml

    # Load kubeconfig (works in-cluster via service account)
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    # Parse the YAML
    app_dict = yaml.safe_load(app_yaml)
    app_name = app_dict["metadata"]["name"]

    # Create the SparkApplication using the custom objects API
    api = client.CustomObjectsApi()

    try:
        api.create_namespaced_custom_object(
            group="sparkoperator.k8s.io",
            version="v1beta2",
            namespace=namespace,
            plural="sparkapplications",
            body=app_dict,
        )
        print(f"Submitted SparkApplication: {app_name}")
    except client.ApiException as e:
        if e.status == 409:
            # Already exists, delete and recreate
            print(f"SparkApplication {app_name} exists, deleting...")
            api.delete_namespaced_custom_object(
                group="sparkoperator.k8s.io",
                version="v1beta2",
                namespace=namespace,
                plural="sparkapplications",
                name=app_name,
            )
            time.sleep(2)
            api.create_namespaced_custom_object(
                group="sparkoperator.k8s.io",
                version="v1beta2",
                namespace=namespace,
                plural="sparkapplications",
                body=app_dict,
            )
            print(f"Re-submitted SparkApplication: {app_name}")
        else:
            raise

    return app_name


def wait_for_spark_application(
    app_name: str,
    namespace: str = "default",
    timeout_seconds: int = 3600,
    poll_interval: int = 30,
) -> str:
    """Wait for a SparkApplication to complete.

    Args:
        app_name: Name of the SparkApplication
        namespace: Kubernetes namespace
        timeout_seconds: Maximum time to wait
        poll_interval: Seconds between status checks

    Returns:
        Final status of the application

    Raises:
        TimeoutError: If the application doesn't complete in time
        RuntimeError: If the application fails
    """
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"SparkApplication {app_name} did not complete within {timeout_seconds}s"
            )

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

            print(f"SparkApplication {app_name} state: {app_state} (elapsed: {int(elapsed)}s)")

            if app_state == "COMPLETED":
                return "COMPLETED"
            elif app_state in ("FAILED", "SUBMISSION_FAILED", "FAILING"):
                error_msg = status.get("applicationState", {}).get("errorMessage", "Unknown error")
                raise RuntimeError(f"SparkApplication failed: {error_msg}")

        except client.ApiException as e:
            if e.status == 404:
                print(f"SparkApplication {app_name} not found, waiting...")
            else:
                raise

        time.sleep(poll_interval)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes", "requests", "pyyaml"],
)
def spark_kronodroid_iceberg_op(
    run_id: str,
    minio_endpoint: str,
    minio_bucket: str,
    minio_prefix: str,
    lakefs_endpoint: str,
    lakefs_repository: str,
    target_branch: str,
    spark_image: str,
    namespace: str,
    service_account: str,
    minio_secret_name: str,
    lakefs_secret_name: str,
    staging_database: str,
    marts_database: str,
    catalog_name: str,
    driver_cores: int,
    driver_memory: str,
    executor_cores: int,
    executor_instances: int,
    executor_memory: str,
    timeout_seconds: int,
) -> NamedTuple("SparkJobOutput", [("lakefs_branch", str), ("app_name", str), ("status", str)]):
    """Run Kronodroid Iceberg transformation via SparkOperator.

    Creates a per-run LakeFS branch, submits the SparkApplication, and waits
    for completion.

    Args:
        run_id: Unique run identifier for this pipeline run
        minio_endpoint: MinIO endpoint URL
        minio_bucket: MinIO bucket for raw data
        minio_prefix: Prefix path for raw data
        lakefs_endpoint: LakeFS endpoint URL
        lakefs_repository: LakeFS repository name
        target_branch: Target LakeFS branch (source for per-run branch)
        spark_image: Docker image for Spark job
        namespace: Kubernetes namespace
        service_account: Kubernetes service account for Spark
        minio_secret_name: K8s secret name for MinIO credentials
        lakefs_secret_name: K8s secret name for LakeFS credentials
        staging_database: Iceberg database for staging tables
        marts_database: Iceberg database for mart tables
        catalog_name: Iceberg catalog name
        driver_cores: Spark driver cores
        driver_memory: Spark driver memory
        executor_cores: Spark executor cores
        executor_instances: Number of Spark executors
        executor_memory: Spark executor memory
        timeout_seconds: Maximum time to wait for job completion

    Returns:
        NamedTuple with lakefs_branch, app_name, and status
    """
    import os
    import requests
    import time
    import yaml
    from kubernetes import client, config

    # Generate per-run branch name (LakeFS doesn't allow slashes in branch names)
    lakefs_branch = f"spark-{run_id}"
    app_name = f"kronodroid-iceberg-{run_id[:8]}"

    print(f"Starting Spark Kronodroid Iceberg job")
    print(f"  Run ID: {run_id}")
    print(f"  LakeFS branch: {lakefs_branch}")
    print(f"  App name: {app_name}")

    # Load credentials from environment (injected by K8s secret via kubernetes.use_secret_as_env)
    lakefs_access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
    lakefs_secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY_ID", "")
    minio_secret_key = os.environ.get("MINIO_SECRET_ACCESS_KEY", "")

    # Create per-run LakeFS branch
    api_base = lakefs_endpoint.rstrip("/")
    auth = (lakefs_access_key, lakefs_secret_key)

    branch_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches/{lakefs_branch}"
    resp = requests.get(branch_url, auth=auth)

    if resp.status_code == 404:
        create_url = f"{api_base}/api/v1/repositories/{lakefs_repository}/branches"
        data = {"name": lakefs_branch, "source": target_branch}
        create_resp = requests.post(create_url, json=data, auth=auth)
        if create_resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create branch: {create_resp.text}")
        print(f"Created LakeFS branch: {lakefs_branch}")
    else:
        print(f"LakeFS branch exists: {lakefs_branch}")

    # Build SparkApplication YAML
    iceberg_rest_uri = f"{lakefs_endpoint.rstrip('/')}/api/v1/iceberg"

    spark_app_yaml = f"""
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: {spark_image}
  imagePullPolicy: IfNotPresent
  mainApplicationFile: local:///opt/spark/work-dir/kronodroid_iceberg_job.py
  arguments:
    - "--minio-bucket"
    - "{minio_bucket}"
    - "--minio-prefix"
    - "{minio_prefix}"
    - "--lakefs-repository"
    - "{lakefs_repository}"
    - "--lakefs-branch"
    - "{lakefs_branch}"
    - "--catalog-name"
    - "{catalog_name}"
    - "--staging-database"
    - "{staging_database}"
    - "--marts-database"
    - "{marts_database}"
  sparkVersion: "3.5.0"
  restartPolicy:
    type: Never
  driver:
    cores: {driver_cores}
    memory: "{driver_memory}"
    serviceAccount: {service_account}
    envFrom:
      - secretRef:
          name: {minio_secret_name}
      - secretRef:
          name: {lakefs_secret_name}
    env:
      - name: MINIO_ENDPOINT_URL
        value: "{minio_endpoint}"
      - name: LAKEFS_ENDPOINT_URL
        value: "{lakefs_endpoint}"
      - name: LAKEFS_REPOSITORY
        value: "{lakefs_repository}"
      - name: LAKEFS_BRANCH
        value: "{lakefs_branch}"
  executor:
    cores: {executor_cores}
    instances: {executor_instances}
    memory: "{executor_memory}"
    envFrom:
      - secretRef:
          name: {minio_secret_name}
      - secretRef:
          name: {lakefs_secret_name}
    env:
      - name: MINIO_ENDPOINT_URL
        value: "{minio_endpoint}"
      - name: LAKEFS_ENDPOINT_URL
        value: "{lakefs_endpoint}"
      - name: LAKEFS_REPOSITORY
        value: "{lakefs_repository}"
      - name: LAKEFS_BRANCH
        value: "{lakefs_branch}"
  deps:
    packages:
      # iceberg-spark-runtime includes avro support
      - org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2
      # AWS/S3 support for LakeFS S3 gateway
      - org.apache.hadoop:hadoop-aws:3.3.4
      - com.amazonaws:aws-java-sdk-bundle:1.12.262
  sparkConf:
    # Use /tmp for Ivy cache (writable in container)
    spark.jars.ivy: /tmp/.ivy2
    spark.sql.extensions: org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
    spark.sql.iceberg.write.format.default: avro
    spark.sql.iceberg.write.avro.compression-codec: snappy
    # Use Hadoop catalog with s3a:// paths for compatibility with training jobs
    spark.sql.catalog.{catalog_name}: org.apache.iceberg.spark.SparkCatalog
    spark.sql.catalog.{catalog_name}.type: hadoop
    spark.sql.catalog.{catalog_name}.warehouse: "s3a://{lakefs_repository}/{target_branch}"
    spark.sql.catalog.{catalog_name}.cache-enabled: "false"
    # S3A filesystem config - lakeFS acts as S3 gateway
    spark.hadoop.fs.s3.impl: org.apache.hadoop.fs.s3a.S3AFileSystem
    spark.hadoop.fs.s3a.impl: org.apache.hadoop.fs.s3a.S3AFileSystem
    spark.hadoop.fs.s3a.path.style.access: "true"
    spark.hadoop.fs.s3a.connection.ssl.enabled: "false"
    # Per-bucket endpoints and credentials for MinIO (dlt-data bucket)
    spark.hadoop.fs.s3a.bucket.{minio_bucket}.endpoint: "{minio_endpoint}"
    spark.hadoop.fs.s3a.bucket.{minio_bucket}.access.key: "{minio_access_key}"
    spark.hadoop.fs.s3a.bucket.{minio_bucket}.secret.key: "{minio_secret_key}"
    # Per-bucket endpoints and credentials for LakeFS (kronodroid bucket)
    spark.hadoop.fs.s3a.bucket.{lakefs_repository}.endpoint: "{lakefs_endpoint}"
    spark.hadoop.fs.s3a.bucket.{lakefs_repository}.access.key: "{lakefs_access_key}"
    spark.hadoop.fs.s3a.bucket.{lakefs_repository}.secret.key: "{lakefs_secret_key}"
"""

    # Load Kubernetes config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    app_dict = yaml.safe_load(spark_app_yaml)

    # Submit SparkApplication
    try:
        api.create_namespaced_custom_object(
            group="sparkoperator.k8s.io",
            version="v1beta2",
            namespace=namespace,
            plural="sparkapplications",
            body=app_dict,
        )
        print(f"Submitted SparkApplication: {app_name}")
    except client.ApiException as e:
        if e.status == 409:
            print(f"SparkApplication {app_name} exists, deleting...")
            api.delete_namespaced_custom_object(
                group="sparkoperator.k8s.io",
                version="v1beta2",
                namespace=namespace,
                plural="sparkapplications",
                name=app_name,
            )
            time.sleep(5)
            api.create_namespaced_custom_object(
                group="sparkoperator.k8s.io",
                version="v1beta2",
                namespace=namespace,
                plural="sparkapplications",
                body=app_dict,
            )
            print(f"Re-submitted SparkApplication: {app_name}")
        else:
            raise

    # Wait for completion
    start_time = time.time()
    poll_interval = 30

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"SparkApplication did not complete in {timeout_seconds}s")

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
            print(f"SparkApplication state: {app_state} (elapsed: {int(elapsed)}s)")

            if app_state == "COMPLETED":
                from collections import namedtuple
                Output = namedtuple("SparkJobOutput", ["lakefs_branch", "app_name", "status"])
                return Output(lakefs_branch, app_name, "COMPLETED")
            elif app_state in ("FAILED", "SUBMISSION_FAILED", "FAILING"):
                error_msg = status.get("applicationState", {}).get("errorMessage", "Unknown")
                raise RuntimeError(f"SparkApplication failed: {error_msg}")

        except client.ApiException as e:
            if e.status != 404:
                raise

        time.sleep(poll_interval)


# Convenience function for non-KFP testing
def run_spark_kronodroid_iceberg(
    run_id: str | None = None,
    minio_endpoint: str | None = None,
    minio_bucket: str = "kronodroid",  # LakeFS repository name (accessed via S3 gateway)
    minio_prefix: str = "kronodroid_raw",
    lakefs_endpoint: str | None = None,
    lakefs_repository: str = "kronodroid",
    target_branch: str = "main",
    spark_image: str = "apache/spark:3.5.0-python3",
    namespace: str = "default",
    service_account: str = "spark",
    minio_secret_name: str = "lakefs-credentials",  # Use LakeFS credentials for S3 gateway
    lakefs_secret_name: str = "lakefs-credentials",
    staging_database: str = "stg_kronodroid",
    marts_database: str = "kronodroid",
    catalog_name: str = "lakefs",
    driver_cores: int = 1,
    driver_memory: str = "2g",
    executor_cores: int = 2,
    executor_instances: int = 2,
    executor_memory: str = "2g",
    timeout_seconds: int = 3600,
) -> SparkJobOutput:
    """Run the Spark Kronodroid Iceberg job (for testing outside KFP).

    All parameters default to environment variables or reasonable defaults.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    if minio_endpoint is None:
        minio_endpoint = os.getenv("MINIO_ENDPOINT_URL", "http://minio:9000")
    if lakefs_endpoint is None:
        lakefs_endpoint = os.getenv("LAKEFS_ENDPOINT_URL", "http://lakefs:8000")

    # This would need the actual implementation
    # For now, just return a placeholder (LakeFS doesn't allow slashes in branch names)
    lakefs_branch = f"spark-{run_id}"
    app_name = f"kronodroid-iceberg-{run_id[:8]}"

    return SparkJobOutput(
        lakefs_branch=lakefs_branch,
        app_name=app_name,
        status="SUBMITTED",
    )
