# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Fingerprinting Project (dfp) - A modular MLOps monorepo implementing Android malware detection with swappable components. The architecture separates core ML logic from integration glue, enabling flexible swaps of orchestration (Kubeflow ↔ Airflow), compute engines (Spark ↔ DataFusion), and feature stores (Feast ↔ alternatives).

**Key architectural principle**: LakeFS + Iceberg + MinIO serve as the versioned data plane; Feast + MLflow + DataHub provide the metadata/experiment plane that "points into" specific commits/snapshots.

## Core Commands

### Environment Setup
```bash
# Python environment (uses uv)
uv venv
source .venv/bin/activate
uv pip install -e .

# Install tooling via mise
mise install

# Start local data/metadata plane
docker-compose up  # MinIO, LakeFS, MLflow, Redis
```

### Kubernetes (kind) Cluster
```bash
# Create cluster + deploy services (recommended workflow)
task up

# With optional addons
task up WITH_SPARK_OPERATOR=1 WITH_FEAST=1

# Manage cluster
task down           # Delete cluster
task status         # Show deployments/pods/services
task logs APP=minio # Tail logs for specific deployment

# Port-forwarding
task port-forward        # Start port-forwards
task port-forward:stop   # Stop port-forwards
task port-forward:status # Check status

# Optional addons
task addons:up          # Install Spark Operator + Feast
task spark-operator:up  # Install Spark Operator only
task feast:up           # Install Feast feature server only
task kfp:up            # Install Kubeflow Pipelines
```

**Service endpoints** (after `task up` with `PORT_FORWARD=1`):
- LakeFS: http://localhost:8000
- MinIO API: http://localhost:19000
- MinIO Console: http://localhost:19001 (minioadmin/minioadmin)
- MLflow: http://localhost:5050
- Redis: 127.0.0.1:16379
- Feast feature server: http://localhost:16566
- Spark Thrift Server: localhost:10000 (requires addon)
- Spark UI: http://localhost:4040 (requires addon)

### Kronodroid Data Pipeline
```bash
# Full pipeline: Kaggle → MinIO → dbt → Feast
uv run tools/scripts/run_kronodroid_pipeline.py

# Use Spark Operator for transformations (instead of dbt)
uv run tools/scripts/run_kronodroid_pipeline.py --transform-runner spark-operator --skip-spark-check

# Use custom Spark image
uv run tools/scripts/run_kronodroid_pipeline.py --transform-runner spark-operator --spark-image apache/spark:3.5.7-python3

# Skip components
uv run tools/scripts/run_kronodroid_pipeline.py --skip-dbt
uv run tools/scripts/run_kronodroid_pipeline.py --skip-ingestion
uv run tools/scripts/run_kronodroid_pipeline.py --materialize-only

# Skip Spark Thrift Server check (when using spark-operator or if Feast steps are skipped)
uv run tools/scripts/run_kronodroid_pipeline.py --skip-spark-check

# Required: KAGGLE_API_TOKEN in .env
```

### Feature Engineering
```bash
# Spark feature jobs (after Spark/Iceberg configs set)
python engines/spark_engine/dfp_spark/feature_jobs.py

# Feast feature store
cd feature_stores/feast_store
feast apply  # Register feature views
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")  # To Redis
```

### ML Training & Export
```bash
# Training (placeholder loop with MLflow/LakeFS logging)
python -m core.dfp_core.ml.training_pytorch

# Export to mobile-friendly formats
python -m core.dfp_core.ml.export_executorch
python -m core.dfp_core.ml.export_tflite
python -m core.dfp_core.ml.export_onnx
```

### API & Orchestration
```bash
# Inference API
uvicorn apps.api.dfp_api.main:app --reload

# Kubeflow Pipelines (compile & deploy once tasks implemented)
# See orchestration/kubeflow/dfp_kfp/

# Airflow DAGs (stubs)
# See orchestration/airflow/dags/
```

### dbt Transformations
```bash
# dbt-spark connects to Spark Thrift Server for Iceberg transformations
cd analytics/dbt
dbt run    # Execute models
dbt test   # Run tests
dbt docs generate && dbt docs serve  # Documentation
```

### Testing
```bash
pytest                    # Run all tests
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests
pytest tests/e2e/         # End-to-end tests
```

### Linting & Type Checking
```bash
# Use pyrefly and torchfix (per AGENTS.md guidelines)
pyrefly check .
torchfix --check .
```

### Testing & Debugging
```bash
# Test Spark Operator integration (interactive notebook)
cd notebooks
jupyter notebook spark_operator_test.ipynb

# Test MinIO connectivity and management
jupyter notebook minio_utils.ipynb

# Explore Feast feature store
jupyter notebook feast_spark_explorer.ipynb
```

### Bazel (Android runtimes - placeholders)
```bash
# Build Android runtimes/apps (rules need to be filled)
bazel build //runtimes/executorch:all
bazel build //apps/android/runtime_comparison_app:all
```

## Repository Structure

```
dfp-monorepo/
├─ core/dfp_core/           # Core ML logic (engine-agnostic)
│  ├─ config/               # Pydantic schemas, global config
│  ├─ domain/               # Domain types (events, sessions, device info)
│  ├─ features/             # Feature definitions & transformations
│  │  ├─ base.py            # Interfaces: FeatureStore, ExecutionEngine
│  │  ├─ feature_views.py   # Logical feature definitions
│  │  └─ transformations.py # Shared pandas/pyarrow/polars ops
│  ├─ ml/                   # PyTorch models, training, export
│  │  ├─ models.py          # AEs, VAEs, etc.
│  │  ├─ training_pytorch.py # Hydra-powered training
│  │  ├─ export_executorch.py
│  │  ├─ export_tflite.py
│  │  └─ export_onnx.py
│  ├─ io/                   # Schemas & contracts
│  ├─ lakefs_client_utils.py # LakeFS Python client wrapper
│  └─ feast_mlflow_utils.py # MLflow logging for Feast + LakeFS metadata
│
├─ core/conf/               # Hydra config tree (model/data/engine/store/orchestration)
│  └─ config.yaml           # Top-level defaults
│
├─ engines/                 # Swappable compute engines
│  ├─ spark_engine/dfp_spark/
│  │  ├─ session.py         # SparkSession with Iceberg+LakeFS catalog
│  │  ├─ ingestion_jobs.py  # Raw logs → curated tables
│  │  └─ feature_jobs.py    # Spark → Iceberg feature tables
│  ├─ datafusion_engine/dfp_datafusion/
│  └─ dlt_engine/dfp_dlt/   # dlt for Kaggle → MinIO ingestion
│
├─ feature_stores/          # Swappable feature stores
│  ├─ feast_store/
│  │  ├─ feature_store.yaml # Feast project config (Spark offline + Iceberg)
│  │  └─ dfp_feast/
│  │     ├─ entities.py
│  │     ├─ feature_views.py
│  │     └─ kronodroid_features.py # Kronodroid feature definitions
│  └─ alt_store/            # Alternative feature store (slot)
│
├─ orchestration/           # Swappable orchestration layers
│  ├─ kubeflow/dfp_kfp/     # Kubeflow Pipelines components/pipelines
│  ├─ airflow/dags/         # Airflow DAG stubs
│  └─ argo/workflows/       # Argo Workflows (optional)
│
├─ analytics/
│  ├─ dbt/                  # dbt-spark models for Iceberg transformations
│  └─ tinybird/             # Real-time API pipes (optional)
│
├─ runtimes/                # Android runtime bridges (ExecuTorch/TFLite/ONNX)
│  ├─ executorch/           # C++/JNI bridge + Kotlin wrapper
│  ├─ tflite/
│  └─ onnx/
│
├─ apps/
│  ├─ android/              # Android APKs (main app, runtime comparison)
│  └─ api/dfp_api/          # FastAPI inference API
│
├─ infra/
│  ├─ iac/                  # IaC (OpenTofu/Pulumi) for LakeFS/Spark/Feast/MLflow/Kubeflow
│  ├─ k8s/                  # Kubernetes manifests (kind/local + cloud overlays)
│  └─ pyinfra/              # pyinfra scripts for non-K8s hosts
│
├─ tools/
│  ├─ scripts/              # Utility scripts
│  │  ├─ kind_bootstrap.sh  # Create kind cluster + deploy services
│  │  ├─ kind_portforward.sh # Port-forwarding management
│  │  ├─ run_kronodroid_pipeline.py # Kronodroid pipeline runner
│  │  ├─ sync_lakefs_branches.py
│  │  ├─ backfill_features.py
│  │  └─ replay_mlflow_run.py
│  ├─ eval/                 # Android runtime evaluation tools
│  └─ docker/               # Dockerfiles for KFP components
│
├─ tests/                   # unit, integration, e2e tests
├─ notebooks/               # Jupyter notebooks for exploration
├─ ci/                      # CI configs (GitHub Actions, Tekton, Buildkite, Dagger)
└─ docs/                    # architecture.md, data_lineage.md, experiments.md, android_runtime_lab.md
```

## Key Architectural Patterns

### Swappable Components

**Core principle**: Separate "core logic" from "integration glue"

- **Core logic** (`core/dfp_core/`): Feature definitions, transformations, training code, eval logic - **no Kubeflow/Airflow/Spark imports**
- **Integration glue**: Orchestration (Kubeflow/Airflow), compute engines (Spark/DataFusion), feature stores (Feast/alternatives)

### Data Flow: Kronodroid Pipeline

```
Kaggle API → dlt → Parquet → MinIO (raw bucket)
                                ↓
                    Spark Thrift Server (dbt-spark)
                                ↓
                    Iceberg Tables → LakeFS (versioned)
                                ↓
                          Feast → Redis (online serving)
```

### Hydra Configuration

- Override configs per-run via Hydra-style flags or env vars
- Pick presets in `core/conf/config.yaml`: model, data, engine, store, orchestration
- Example: `python -m core.dfp_core.ml.training_pytorch model=ae_large data=android_logs_prod engine=spark`

### Reproducibility with LakeFS + MLflow

Given an experiment and run ID, you can reconstruct:
- Exact model weights (MLflow artifacts)
- Training/validation/test data (LakeFS commit references)
- Feature-engineered features (Feast feature views + Iceberg snapshots)
- Inference results

MLflow logs LakeFS repo/branch/commit metadata alongside artifacts.

## Development Patterns

### Coding Style
- **Package manager**: `uv` (not pip or conda)
- **Data validation**: Use Pydantic and Pandera when appropriate
- **Linting & type checking**: Use `pyrefly` and `torchfix` (not pylint/flake8/mypy)
- **Imports**: Make dfp_spark imports package-relative (e.g., `from .session import create_spark_session`)

### Git Workflow
1. Branch from `main` with descriptive name: `feature/<slug>` or `bugfix/<slug>`
2. Keep commits atomic; prefer conventional commit style (`feat:`, `fix:`, `test:`, `docs:`)
3. Always update `.gitignore` when new components/dependencies/frameworks are introduced

### Configuration Management
- **Spark/Iceberg**: Configure catalog settings in `engines/spark_engine/dfp_spark/session.py` to align with LakeFS
- **Feast**: Registry path and Spark catalog in `feature_stores/feast_store/feature_store.yaml`
- **dbt**: Profiles in `analytics/dbt/profiles/` for Spark Thrift Server connectivity

### LakeFS + Iceberg Integration
- Use LakeFS catalog for Spark/Iceberg tables
- LakeFS S3 gateway endpoint: `http://localhost:8000` (local), credentials in `.env`
- Iceberg catalog: `lakefs_catalog` (pre-configured in Spark Thrift Server)

### MinIO Management
- Credentials: `minioadmin` / `minioadmin`
- List buckets: `mc alias set dfp-minio http://localhost:19000 minioadmin minioadmin && mc ls dfp-minio`
- Delete bucket (force): `mc rb --force dfp-minio/bucket-name`
- ⚠️ Deleting buckets used by LakeFS/services will break them until recreated

## Important Notes

### Spark Operator for Kubeflow
The Spark Operator enables running Spark jobs as Kubernetes-native SparkApplications:
- **Version**: `ghcr.io/kubeflow/spark-operator:v1beta2-1.4.3-3.5.0`
- **Location**: `infra/k8s/kind/addons/spark-operator/`
- **Supports**: Spark 3.5.x
- **API**: `v1beta2` (SparkApplication CRD)

Deploy with:
```bash
kubectl apply -k infra/k8s/kind/addons/spark-operator/
# Or using task
task spark-operator:up
```

**Spark Version Compatibility**:
- Operator `v1beta2-1.4.3-3.5.0` → Use Spark 3.5.x images with Python (e.g., `apache/spark:3.5.7-python3`)
- Default Spark image: `apache/spark:3.5.7-python3`
- Spark dependencies (Iceberg, Hadoop, Avro) must match Spark version

### Spark Thrift Server for dbt-spark and Feast
The Spark Thrift Server provides HiveServer2-compatible interface for:
- **dbt-spark transformations** with Iceberg tables
- **Feast feature registration and materialization** (Feast uses Spark offline store to read from Iceberg)

It's pre-configured with:
- Iceberg Spark extensions
- LakeFS catalog (`lakefs_catalog`) pointing to MinIO
- S3A filesystem for MinIO connectivity
- **Image**: `apache/spark:3.5.7-python3`

Deploy with:
```bash
kubectl apply -k infra/k8s/kind/addons/spark-thrift/
kubectl -n dfp wait --for=condition=ready pod -l app=spark-thrift-server --timeout=120s
```

**Important**: Even when using `spark-operator` for transformations (Step 2), Feast still needs Spark connectivity (Steps 3 & 4) because it's configured with a Spark offline store. You have three options:
1. Deploy Spark Thrift Server (recommended)
2. Skip Feast steps with `--skip-feast --skip-materialize`
3. Use `--skip-spark-check` to bypass the connectivity check (Feast may fail)

### Environment Variables
Key variables for local development (set in `.env`):
- `KAGGLE_API_TOKEN`: For Kronodroid dataset download
- `LAKEFS_ACCESS_KEY_ID`, `LAKEFS_SECRET_ACCESS_KEY`: LakeFS credentials
- `LAKEFS_ENDPOINT_URL`: Default `http://localhost:8000`
- `LAKEFS_REPOSITORY`, `LAKEFS_BRANCH`: LakeFS repo/branch

### Cloud Deployment
For cloud deployments, reuse the same K8s manifests with overlays to:
- Swap NodePort → LoadBalancer/Ingress
- Point LakeFS/MLflow at managed object storage + Postgres
- Use managed Spark/Redis/Feast services

### Taskfile Variables
Override default variables when running `task`:
```bash
CLUSTER_NAME=my-cluster PORT_FORWARD=0 task up
WITH_SPARK_OPERATOR=1 WITH_FEAST=1 task up
```

Defaults:
- `CLUSTER_NAME`: dfp-kind
- `NAMESPACE`: dfp
- `PORT_FORWARD`: 1 (enabled)
- `WITH_SPARK_OPERATOR`, `WITH_FEAST`, `WITH_KFP`: 0 (disabled)
