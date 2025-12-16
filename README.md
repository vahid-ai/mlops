# Digital Fingerprinting Project (dfp)

Skeleton monorepo that mirrors the pipeline1.md layout. It separates core ML logic from integration glue so Kubeflow/Airflow, Spark/DataFusion, and Feast/alternative stores can be swapped without touching the core code. LakeFS + Iceberg + Minio serve as the versioned data plane; Feast + MLflow + DataHub provide metadata and experiment traceability.

## Structure
See inline README stubs and doc files under `docs/` for guidance on how each package fits together. Use `uv` for Python dependencies and `pyrefly`/`torchfix` for linting and typing checks.

## How to operate
- Python env: `uv venv && source .venv/bin/activate && uv pip install -e .` (deps in `pyproject.toml`).
- Config: pick presets in `core/conf/config.yaml` (model/data/engine/store/orchestration). Override per-run via Hydra-style flags or env vars.
- Data/metadata plane: `docker-compose up` to start MinIO, LakeFS, MLflow, Redis; point MLflow to `http://localhost:5000`, LakeFS to `http://localhost:8000`, MinIO to `http://localhost:9000` (minioadmin/minioadmin).
- Feature store: configure `feature_stores/feast_store/feature_store.yaml` registry path and Spark catalog settings to align with LakeFS/Iceberg.

## Build and run
- Feature jobs (Spark/DataFusion): run engine scripts such as `python engines/spark_engine/dfp_spark/feature_jobs.py` once Spark/Iceberg configs are set.
- Training: `python -m core.dfp_core.ml.training_pytorch` (placeholder loop) uses configs from `core/conf`; add MLflow/LakeFS logging via `core/dfp_core/feast_mlflow_utils.py`.
- Export: `python -m core.dfp_core.ml.export_executorch` (or `export_tflite.py`, `export_onnx.py`) to emit mobile-friendly artifacts.
- API: start inference API with `uvicorn apps.api.dfp_api.main:app --reload`.
- Orchestration: Kubeflow components/pipeline stubs in `orchestration/kubeflow/dfp_kfp`; Airflow DAG stubs in `orchestration/airflow/dags`. Compile/deploy once tasks are implemented.
- Runtimes/Android: Bazel targets are placeholders in `runtimes/**` and `apps/android/**`; build with `bazel build //runtimes/executorch:all` or app targets after rules are filled.

## Kubernetes (kind)
- Recommended: install pinned tooling with `mise install`, then run `task up` (wraps `tools/scripts/kind_bootstrap.sh`).
- Install `mise`:
  - macOS (Homebrew): `brew install mise`
  - Linux/macOS (script): `curl https://mise.run | sh`
  - Windows (winget): `winget install jdx.mise`
- Install `task` (go-task):
  - macOS (Homebrew): `brew install go-task/tap/go-task`
  - Linux/macOS (script): `sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin`
  - Windows: `choco install go-task` (or `scoop install task`)
- Direct: create a local cluster and deploy the data/metadata plane via `tools/scripts/kind_bootstrap.sh` (uses `infra/k8s/kind/kind-config.yaml` and `infra/k8s/kind/manifests`).
- `task up` starts port-forwards by default (`PORT_FORWARD=1`); disable with `PORT_FORWARD=0 task up`. You can also run `task port-forward`, `task port-forward:status`, and `task port-forward:stop`.
- Services available on host:
  - LakeFS: `http://localhost:8000`
  - MinIO API: `http://localhost:19000` (returns 403 without auth)
  - MinIO Console: `http://localhost:19001`
  - MLflow: `http://localhost:5050`
  - Redis: `127.0.0.1:16379` (use `redis-cli -h 127.0.0.1 -p 16379 ping`)
  - Feast feature server: `http://localhost:16566`
  - Spark Thrift Server: `localhost:10000` (for dbt-spark, requires addon)
  - Spark UI: `http://localhost:4040` (requires addon)
- Optional addons (installed by `task up WITH_SPARK_OPERATOR=1 WITH_FEAST=1` or `task addons:up`):
  - Spark Operator (SparkApplication CRD/controller)
  - Spark Thrift Server (for dbt-spark connectivity)
  - Feast feature server (serving from an embedded `registry.db`)
- Kubeflow Pipelines: `task kfp:up` (requires network to fetch upstream manifests)
- For cloud, reuse the same manifests with overlays to swap NodePort → LoadBalancer/Ingress and point LakeFS/MLflow at managed object storage + Postgres.

### Spark Thrift Server (for dbt-spark)

The Spark Thrift Server provides a HiveServer2-compatible interface for dbt-spark transformations with Iceberg tables.

**Deploy Spark Thrift Server:**
```bash
kubectl apply -k infra/k8s/kind/addons/spark-thrift/
```

**Wait for it to be ready:**
```bash
kubectl -n dfp wait --for=condition=ready pod -l app=spark-thrift-server --timeout=120s
```

**Access endpoints:**
- Thrift Server: `localhost:10000` (for dbt-spark connections)
- Spark UI: `http://localhost:4040`

The Thrift Server is pre-configured with:
- Iceberg Spark extensions
- LakeFS catalog (`lakefs_catalog`) pointing to MinIO
- S3A filesystem for MinIO connectivity

### Managing MinIO Buckets

MinIO credentials: `minioadmin` / `minioadmin`

#### Listing Buckets

**Using MinIO Client (mc)**:
```bash
# Install mc (macOS)
brew install minio/stable/mc

# Start port-forwarding (if not already running)
./tools/scripts/kind_portforward.sh start

# Configure the MinIO client
mc alias set dfp-minio http://localhost:19000 minioadmin minioadmin

# List all buckets
mc ls dfp-minio
```

**Using kubectl exec**:
```bash
kubectl exec -n dfp deployment/minio -- mc alias set local http://localhost:9000 minioadmin minioadmin
kubectl exec -n dfp deployment/minio -- mc ls local
```

**Using Python boto3**:
```python
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:19000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

buckets = s3.list_buckets()
for bucket in buckets['Buckets']:
    print(bucket['Name'])
```

**Using MinIO Console UI**: Navigate to `http://localhost:19001` and login with `minioadmin` / `minioadmin`.

#### Deleting Buckets

**Using MinIO Client (mc)**:
```bash
# Delete an empty bucket
mc rb dfp-minio/bucket-name

# Delete a bucket and ALL its contents (force)
mc rb --force dfp-minio/bucket-name
```

**Using AWS CLI**:
```bash
# Delete an empty bucket
aws s3 rb s3://bucket-name --endpoint-url http://localhost:19000

# Delete a bucket with all contents (force)
aws s3 rb s3://bucket-name --force --endpoint-url http://localhost:19000
```

**Using Python boto3**:
```python
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:19000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

s3_resource = boto3.resource(
    's3',
    endpoint_url='http://localhost:19000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

bucket_name = 'bucket-name'
# Delete all objects first (for non-empty buckets)
bucket = s3_resource.Bucket(bucket_name)
bucket.objects.all().delete()
# Then delete the bucket
s3.delete_bucket(Bucket=bucket_name)
```

⚠️ **Warning**: Deleting buckets is irreversible. If LakeFS or other services are using specific buckets, deleting them will break those services until the buckets are recreated.

## Kronodroid Data Pipeline

The Kronodroid Android malware detection dataset can be ingested from Kaggle and processed through the dlt + dbt + Feast pipeline.

### Prerequisites
1. Start the kind cluster with `task up`
2. Deploy Spark Thrift Server for dbt transformations:
   ```bash
   kubectl apply -k infra/k8s/kind/addons/spark-thrift/
   kubectl -n dfp wait --for=condition=ready pod -l app=spark-thrift-server --timeout=120s
   ```
3. Ensure you have a Kaggle API token in `.env` as `KAGGLE_API_TOKEN`
4. Install Python dependencies: `uv pip install -e .`

### Running the Pipeline

```bash
# Full pipeline: Kaggle → MinIO → dbt → Feast
uv run tools/scripts/run_kronodroid_pipeline.py

# Skip dbt transformations (ingestion only)
uv run tools/scripts/run_kronodroid_pipeline.py --skip-dbt

# Skip ingestion (if data already loaded)
uv run tools/scripts/run_kronodroid_pipeline.py --skip-ingestion

# Only materialize features to Redis
uv run tools/scripts/run_kronodroid_pipeline.py --materialize-only
```

### Pipeline Components

1. **dlt Engine** (`engines/dlt_engine/dfp_dlt/`): Downloads Kronodroid from Kaggle and loads to MinIO as Parquet
2. **dbt-spark** (`analytics/dbt/`): Connects to Spark Thrift Server to transform raw data into Iceberg feature tables
3. **Feast Features** (`feature_stores/feast_store/dfp_feast/kronodroid_features.py`): Feature definitions for ML training and serving

### Data Flow
```
Kaggle API → dlt → Parquet → MinIO (raw)
                                ↓
                    Spark Thrift Server (dbt-spark)
                                ↓
                    Iceberg Tables → LakeFS (versioned)
                                ↓
                          Feast → Redis (online serving)
```

## Tests and CI
- Tests live in `tests/` (unit, integration, e2e). Run with `pytest` after replacing placeholders.
- GitHub Actions workflows in `ci/github/workflows` cover lint, build/test, and KFP compile; Tekton and Dagger stubs also provided.
