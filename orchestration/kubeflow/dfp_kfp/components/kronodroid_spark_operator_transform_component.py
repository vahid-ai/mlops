"""Kubeflow/Spark Operator component: Kronodroid transformations.

This submits a SparkApplication that runs the PySpark job in
`spark_jobs/kronodroid_transform_job.py`.

It is designed to be callable:
  - from a Kubeflow Pipeline step (container with `kubectl`)
  - from local tooling (e.g. `tools/scripts/run_kronodroid_pipeline.py`)

Spark Operator Version Compatibility:
  - Operator: ghcr.io/kubeflow/spark-operator:v1beta2-1.4.3-3.5.0
  - Deployed in: infra/k8s/kind/addons/spark-operator/
  - Supports: Spark 3.5.x
  - API version: v1beta2
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class KronodroidSparkOperatorConfig:
    namespace: str = "dfp"
    # Spark image for driver and executor pods
    # Must be compatible with Spark Operator version (v1beta2-1.4.3-3.5.0 supports Spark 3.5.x)
    # Note: Bitnami has changed tag naming over time; `bitnami/spark:3.5.3` may not exist.
    # Use Apache's official image tags by default for reliability.
    # For `type: Python` SparkApplications we need a Spark image with Python installed.
    # The base `apache/spark:<ver>` tag may not include Python; use the `-python3` variant.
    spark_image: str = "apache/spark:3.5.7-python3"
    spark_version: str = "3.5.7"
    service_account: str = "spark-operator"
    timeout_seconds: int = 60 * 30

    # Raw data (MinIO)
    raw_bucket: str = "dlt-data"
    raw_dataset: str = "kronodroid_raw"
    raw_format: str = "parquet"

    # Table names produced by the Kaggle download
    emulator_table: str = "kronodroid_2021_emu_v1"
    real_device_table: str = "kronodroid_2021_real_v1"

    # Iceberg target
    iceberg_catalog: str = "lakefs_catalog"
    iceberg_database: str = "dfp"

    # In-cluster endpoints
    k8s_minio_endpoint_url: str = "http://minio.dfp.svc.cluster.local:9000"
    k8s_lakefs_endpoint_url: str = "http://lakefs.dfp.svc.cluster.local:8000"

    # Credentials (passed into Spark Hadoop conf)
    minio_access_key_id: str = "minioadmin"
    minio_secret_access_key: str = "minioadmin"
    lakefs_repository: str = "kronodroid"
    lakefs_branch: str = "main"
    lakefs_access_key_id: str = ""
    lakefs_secret_access_key: str = ""


def _run(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        check=True,
        capture_output=True,
    )


def _job_file() -> Path:
    return Path(__file__).parent / "spark_jobs" / "kronodroid_transform_job.py"


def _ensure_kubeconfig() -> Optional[Path]:
    # Prefer an explicitly provided kubeconfig, but fall back to the common local default.
    # This makes local/dev runs work without requiring users to export KUBECONFIG.
    kubeconfig = os.environ.get("KUBECONFIG")
    if kubeconfig:
        # KUBECONFIG can be a path list (":"-separated on Unix). We only need one.
        first = kubeconfig.split(os.pathsep)[0]
        return Path(first).expanduser()

    default_local = Path.home() / ".kube" / "config"
    if default_local.exists():
        os.environ["KUBECONFIG"] = str(default_local)
        return default_local

    host = os.getenv("KUBERNETES_SERVICE_HOST")
    port = os.getenv("KUBERNETES_SERVICE_PORT")
    if not host or not port:
        return None

    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    namespace_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")

    if not token_path.exists() or not ca_path.exists():
        return None

    token = token_path.read_text().strip()
    namespace = namespace_path.read_text().strip() if namespace_path.exists() else "default"
    kubeconfig_yaml = f"""apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: {ca_path}
    server: https://{host}:{port}
  name: in-cluster
contexts:
- context:
    cluster: in-cluster
    namespace: {namespace}
    user: in-cluster
  name: in-cluster
current-context: in-cluster
users:
- name: in-cluster
  user:
    token: {token}
"""

    tmp_dir = Path(tempfile.mkdtemp(prefix="kubeconfig-"))
    kubeconfig_path = tmp_dir / "config"
    kubeconfig_path.write_text(kubeconfig_yaml)
    kubeconfig_path.chmod(0o600)
    os.environ["KUBECONFIG"] = str(kubeconfig_path)
    return kubeconfig_path


def _render_configmap_yaml(name: str, namespace: str) -> str:
    job_path = _job_file()
    if not job_path.exists():
        raise FileNotFoundError(f"Spark job file not found: {job_path}")

    result = _run(
        [
            "kubectl",
            "-n",
            namespace,
            "create",
            "configmap",
            name,
            f"--from-file=kronodroid_transform_job.py={str(job_path)}",
            "--dry-run=client",
            "-o",
            "yaml",
        ]
    )
    return result.stdout


def _render_sparkapplication_yaml(app_name: str, cfg: KronodroidSparkOperatorConfig, configmap: str) -> str:
    # Spark dependencies - versions must align with spark_version
    # For Spark 3.5.x, use compatible library versions
    packages = [
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0",  # Iceberg for Spark 3.5
        "org.apache.hadoop:hadoop-aws:3.3.4",                        # AWS S3 support
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",                # AWS SDK
        "org.apache.spark:spark-avro_2.12:3.5.7",                    # Avro support for Spark 3.5
    ]
    packages_yaml = "\n".join([f'      - "{p}"' for p in packages])

    spark_conf = {
        # Spark resolves `deps.packages` via Ivy. In many k8s images the user has HOME=/nonexistent,
        # which causes SparkSubmit to try writing to `/nonexistent/.ivy2` and crash.
        # Force Ivy cache to a writable location.
        "spark.jars.ivy": "/tmp/.ivy2",
        "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        f"spark.sql.catalog.{cfg.iceberg_catalog}": "org.apache.iceberg.spark.SparkCatalog",
        f"spark.sql.catalog.{cfg.iceberg_catalog}.type": "hadoop",
        f"spark.sql.catalog.{cfg.iceberg_catalog}.warehouse": f"s3a://{cfg.lakefs_repository}/{cfg.lakefs_branch}/iceberg",
        "spark.sql.iceberg.write.format.default": "avro",
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        f"spark.hadoop.fs.s3a.bucket.{cfg.raw_bucket}.endpoint": cfg.k8s_minio_endpoint_url,
        f"spark.hadoop.fs.s3a.bucket.{cfg.raw_bucket}.access.key": cfg.minio_access_key_id,
        f"spark.hadoop.fs.s3a.bucket.{cfg.raw_bucket}.secret.key": cfg.minio_secret_access_key,
        f"spark.hadoop.fs.s3a.bucket.{cfg.lakefs_repository}.endpoint": cfg.k8s_lakefs_endpoint_url,
        f"spark.hadoop.fs.s3a.bucket.{cfg.lakefs_repository}.access.key": cfg.lakefs_access_key_id,
        f"spark.hadoop.fs.s3a.bucket.{cfg.lakefs_repository}.secret.key": cfg.lakefs_secret_access_key,
    }

    spark_conf_yaml = "\n".join(
        [f"    {json.dumps(k)}: {json.dumps(str(v))}" for k, v in spark_conf.items()]
    )

    env_lines: list[str] = []
    for name, value in [
        # Ensure HOME is writable in driver/executor containers (pairs with spark.jars.ivy above).
        ("HOME", "/tmp"),
        ("RAW_BUCKET", cfg.raw_bucket),
        ("RAW_DATASET", cfg.raw_dataset),
        ("RAW_FORMAT", cfg.raw_format),
        ("KRONODROID_EMULATOR_TABLE", cfg.emulator_table),
        ("KRONODROID_REAL_DEVICE_TABLE", cfg.real_device_table),
        ("ICEBERG_CATALOG", cfg.iceberg_catalog),
        ("ICEBERG_DATABASE", cfg.iceberg_database),
    ]:
        env_lines.append(f"        - name: {name}")
        env_lines.append(f"          value: {json.dumps(value)}")
    env_yaml = "\n".join(env_lines)

    return f"""apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: {app_name}
  namespace: {cfg.namespace}
spec:
  type: Python
  mode: cluster
  pythonVersion: "3"
  sparkVersion: "{cfg.spark_version}"
  image: "{cfg.spark_image}"
  imagePullPolicy: IfNotPresent
  # Note: Spark Operator commonly uses `/opt/spark/work-dir` internally (e.g., for downloaded jars).
  # Mount the application source elsewhere to avoid mount conflicts/overwrites.
  mainApplicationFile: "local:///opt/spark/app/kronodroid_transform_job.py"
  deps:
    packages:
{packages_yaml}
  sparkConf:
{spark_conf_yaml}
  restartPolicy:
    type: Never
  driver:
    cores: 1
    memory: "2g"
    serviceAccount: {cfg.service_account}
    env:
{env_yaml}
    volumeMounts:
      - name: job-src
        mountPath: /opt/spark/app
  executor:
    instances: 2
    cores: 1
    memory: "2g"
    serviceAccount: {cfg.service_account}
    env:
{env_yaml}
    volumeMounts:
      - name: job-src
        mountPath: /opt/spark/app
  volumes:
    - name: job-src
      configMap:
        name: {configmap}
"""


def _check_spark_operator_version(namespace: str) -> dict:
    """Check if Spark Operator is deployed and get its version."""
    try:
        result = _run(
            [
                "kubectl",
                "-n",
                namespace,
                "get",
                "deployment",
                "spark-operator",
                "-o",
                "json",
            ]
        )
        data = json.loads(result.stdout)
        image = data["spec"]["template"]["spec"]["containers"][0]["image"]

        return {
            "installed": True,
            "image": image,
            "supports_spark_3_5": "3.5" in image,
        }
    except subprocess.CalledProcessError:
        return {
            "installed": False,
            "message": "Spark Operator not found",
        }
    except Exception as e:
        return {
            "installed": False,
            "error": str(e),
        }


def _check_pod_status(namespace: str, application_name: str) -> dict:
    """Check the status of driver pod for common issues."""
    try:
        pods = _run(
            [
                "kubectl",
                "-n",
                namespace,
                "get",
                "pods",
                "-l",
                f"sparkoperator.k8s.io/app-name={application_name},spark-role=driver",
                "-o",
                "json",
            ]
        ).stdout
        driver_items = json.loads(pods).get("items", [])

        if not driver_items:
            return {"status": "NO_POD", "message": "Driver pod not created yet"}

        driver_pod = driver_items[0]
        pod_name = driver_pod["metadata"]["name"]
        pod_status = driver_pod["status"]
        phase = pod_status.get("phase", "Unknown")

        # Check container statuses
        container_statuses = pod_status.get("containerStatuses", [])
        if container_statuses:
            container = container_statuses[0]
            state = container.get("state", {})

            # Check for ImagePullBackOff
            if "waiting" in state:
                reason = state["waiting"].get("reason", "")
                message = state["waiting"].get("message", "")
                if "ImagePull" in reason or "ErrImagePull" in reason:
                    return {
                        "status": "IMAGE_PULL_ERROR",
                        "pod_name": pod_name,
                        "reason": reason,
                        "message": message,
                        "phase": phase,
                    }
                return {
                    "status": "WAITING",
                    "pod_name": pod_name,
                    "reason": reason,
                    "message": message,
                    "phase": phase,
                }

            # Check for termination / crash
            if "terminated" in state:
                return {
                    "status": "TERMINATED",
                    "pod_name": pod_name,
                    "phase": phase,
                    "exit_code": state["terminated"].get("exitCode", -1),
                    "reason": state["terminated"].get("reason", ""),
                    "message": state["terminated"].get("message", ""),
                }

        return {
            "status": "RUNNING" if phase == "Running" else phase.upper(),
            "pod_name": pod_name,
            "phase": phase,
        }

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


def run(cfg: KronodroidSparkOperatorConfig, *, application_name: Optional[str] = None) -> bool:
    """Apply ConfigMap + SparkApplication and wait for completion."""
    if _ensure_kubeconfig() is None:
        print("ERROR: No Kubernetes credentials detected.")
        print("Set KUBECONFIG (or ensure ~/.kube/config exists), or run inside a Kubernetes pod with a service account.")
        return False

    if application_name is None:
        application_name = f"kronodroid-transform-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    configmap_name = f"{application_name}-job"

    print(f"\n📦 Creating SparkApplication: {application_name}")
    print(f"   Namespace: {cfg.namespace}")
    print(f"   Image: {cfg.spark_image}")

    # Check Spark Operator installation
    operator_info = _check_spark_operator_version(cfg.namespace)
    if not operator_info.get("installed"):
        print(f"\n   ⚠️  WARNING: Spark Operator may not be installed in namespace '{cfg.namespace}'")
        print(f"   To install: kubectl apply -k infra/k8s/kind/addons/spark-operator/")
    else:
        print(f"   ✓ Spark Operator: {operator_info.get('image', 'unknown')}")
        if not operator_info.get("supports_spark_3_5"):
            print(f"   ⚠️  WARNING: Operator may not support Spark 3.5.x")

    try:
        # Create ConfigMap with the job script
        print(f"\n1️⃣  Creating ConfigMap: {configmap_name}")
        cm_yaml = _render_configmap_yaml(configmap_name, cfg.namespace)
        _run(["kubectl", "apply", "-f", "-"], input_text=cm_yaml)
        print(f"   ✓ ConfigMap created")

        # Create SparkApplication
        print(f"\n2️⃣  Creating SparkApplication: {application_name}")
        app_yaml = _render_sparkapplication_yaml(application_name, cfg, configmap_name)
        _run(["kubectl", "apply", "-f", "-"], input_text=app_yaml)
        print(f"   ✓ SparkApplication created")

        print(f"\n3️⃣  Waiting for SparkApplication to complete (timeout: {cfg.timeout_seconds}s)")
        print(f"   You can monitor progress with:")
        print(f"     kubectl -n {cfg.namespace} get sparkapplication {application_name}")
        print(f"     kubectl -n {cfg.namespace} logs -f {application_name}-driver")
        print()

        deadline = time.time() + cfg.timeout_seconds
        last_state = None
        check_count = 0

        while time.time() < deadline:
            check_count += 1

            # Get SparkApplication status
            try:
                raw = _run(
                    [
                        "kubectl",
                        "-n",
                        cfg.namespace,
                        "get",
                        "sparkapplication",
                        application_name,
                        "-o",
                        "json",
                    ]
                ).stdout
                data = json.loads(raw)
            except subprocess.CalledProcessError:
                print(f"   ⚠ SparkApplication {application_name} not found, waiting...")
                time.sleep(5)
                continue

            state = (
                data.get("status", {})
                .get("applicationState", {})
                .get("state", "")
                .upper()
            )

            # Print state changes
            if state and state != last_state:
                elapsed = int(time.time() - (deadline - cfg.timeout_seconds))
                print(f"   [{elapsed}s] State: {state}")
                last_state = state

            # Check pod status every 10 checks (50 seconds) or if state is empty
            if check_count % 10 == 0 or not state:
                pod_status = _check_pod_status(cfg.namespace, application_name)

                if pod_status["status"] == "IMAGE_PULL_ERROR":
                    print(f"\n   ❌ ERROR: Failed to pull Spark image!")
                    print(f"      Pod: {pod_status['pod_name']}")
                    print(f"      Reason: {pod_status['reason']}")
                    print(f"      Message: {pod_status.get('message', 'N/A')[:200]}")
                    print(f"\n   💡 Fix: Use a valid Spark image, such as:")
                    print(f"      - apache/spark:3.5.7-python3")
                    print(f"      - apache/spark:3.5.7-java17-python3")
                    print(f"      - gcr.io/spark-operator/spark:v3.5.0")
                    print(f"\n   Or pre-load the image into kind:")
                    print(f"      docker pull {cfg.spark_image}")
                    print(f"      kind load docker-image {cfg.spark_image} --name dfp-kind")
                    return False

                elif pod_status["status"] == "TERMINATED":
                    print(f"\n   ❌ Driver pod terminated (SparkApplication will not succeed)")
                    print(f"      Pod: {pod_status.get('pod_name', 'N/A')}")
                    print(f"      Exit code: {pod_status.get('exit_code', 'N/A')}")
                    if pod_status.get("reason"):
                        print(f"      Reason: {pod_status.get('reason')}")
                    if pod_status.get("message"):
                        print(f"      Message: {str(pod_status.get('message'))[:200]}")

                    # Print logs immediately (best-effort) and stop looping to avoid spamming.
                    try:
                        pod = pod_status.get("pod_name")
                        if pod:
                            print(f"\n   📋 Driver logs (tail):")
                            print("   " + "=" * 70)
                            try:
                                logs = _run(
                                    ["kubectl", "-n", cfg.namespace, "logs", pod, "--tail=200"]
                                ).stdout
                            except subprocess.CalledProcessError:
                                # If the container restarted, `--previous` may have the logs.
                                logs = _run(
                                    ["kubectl", "-n", cfg.namespace, "logs", pod, "--previous", "--tail=200"]
                                ).stdout
                            for line in logs.split("\n"):
                                print(f"   {line}")
                            print("   " + "=" * 70)
                    except Exception as e:
                        print(f"   ⚠ Could not fetch driver logs: {e}")

                    return False

            # Terminal states
            if state in {"COMPLETED", "FAILED", "SUBMISSION_FAILED"}:
                elapsed = int(time.time() - (deadline - cfg.timeout_seconds))

                if state == "COMPLETED":
                    print(f"\n   ✅ SparkApplication completed successfully! ({elapsed}s)")
                    return True

                print(f"\n   ❌ SparkApplication {state}! ({elapsed}s)")

                # Try to get driver logs
                try:
                    print(f"\n   📋 Driver logs:")
                    print("   " + "=" * 70)
                    pods = _run(
                        [
                            "kubectl",
                            "-n",
                            cfg.namespace,
                            "get",
                            "pods",
                            "-l",
                            f"sparkoperator.k8s.io/app-name={application_name},spark-role=driver",
                            "-o",
                            "json",
                        ]
                    ).stdout
                    driver_items = json.loads(pods).get("items", [])
                    if driver_items:
                        driver_pod = driver_items[0]["metadata"]["name"]
                        logs = _run(
                            ["kubectl", "-n", cfg.namespace, "logs", driver_pod, "--tail=100"]
                        ).stdout
                        for line in logs.split('\n'):
                            print(f"   {line}")
                    print("   " + "=" * 70)
                except Exception as e:
                    print(f"   ⚠ Could not retrieve driver logs: {e}")

                return False

            time.sleep(5)

        # Timeout
        elapsed = int(time.time() - (deadline - cfg.timeout_seconds))
        print(f"\n   ⏱️ TIMEOUT: SparkApplication did not complete within {cfg.timeout_seconds}s")
        print(f"   Last known state: {last_state or 'UNKNOWN'}")
        print(f"\n   To check status manually:")
        print(f"     kubectl -n {cfg.namespace} get sparkapplication {application_name}")
        print(f"     kubectl -n {cfg.namespace} describe sparkapplication {application_name}")
        print(f"     kubectl -n {cfg.namespace} logs {application_name}-driver")

        return False

    except subprocess.CalledProcessError as e:
        print(f"\n   ❌ kubectl command failed: {' '.join(e.cmd)}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("\n   ❌ kubectl not found in PATH.")
        print("   Install kubectl in the component image or use a Python client instead.")
        return False
    except Exception as e:
        print(f"\n   ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
