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
- For cloud, reuse the same manifests with overlays to swap NodePort → LoadBalancer/Ingress and point LakeFS/MLflow at managed object storage + Postgres.

### Full Platform Setup (Recommended)

For a complete from-scratch setup including Kubeflow Pipelines, use:

```bash
task up:full
```

This command performs a complete 8-step setup:

| Step | Description |
|------|-------------|
| 1/8 | Tears down any existing Kind cluster and stops port-forwards |
| 2/8 | Creates Kind cluster and deploys core services (MinIO, LakeFS, MLflow, Redis, Postgres) |
| 3/8 | Installs Kubeflow Pipelines v2.4.0 |
| 4/8 | Installs Spark Operator via Helm |
| 5/8 | Builds and loads the `dfp-spark` Docker image |
| 6/8 | Waits for all pods to be ready |
| 7/8 | Generates fresh LakeFS credentials (saved to `.env`) |
| 8/8 | Starts all port-forwards |

After completion, the following services are available:

| Service | URL |
|---------|-----|
| Kubeflow Pipelines UI | http://localhost:8081 |
| Kubeflow Pipelines API | http://localhost:8080 |
| LakeFS | http://localhost:8000 |
| MinIO Console | http://localhost:19001 |
| MinIO API | http://localhost:19000 |
| MLflow | http://localhost:5050 |
| Redis | localhost:16379 |

LakeFS credentials are automatically generated and saved to `.env`. The setup takes approximately 5-10 minutes depending on network speed for image pulls.

### Other Useful Tasks

```bash
task up              # Quick setup (no Kubeflow Pipelines)
task down            # Delete the Kind cluster
task status          # Show cluster status
task port-forward    # Start port-forwards
task port-forward:status  # Check port-forward status
task port-forward:reset   # Reset all port-forwards
task lakefs:keys     # Reset LakeFS and generate new credentials
task kfp:clear       # Clear all Kubeflow pipelines and runs
task spark:logs      # Tail logs for the latest Spark job
```


## Kronodroid Data Pipeline

The Kronodroid Android malware detection dataset can be ingested from Kaggle and processed through the dlt + dbt + Feast pipeline.

### Prerequisites
1. Start the kind cluster with `task up`
2. Ensure you have a Kaggle API token in `.env` as `KAGGLE_API_TOKEN`
3. Install Python dependencies: `uv sync`
4. For `--destination lakefs`, the default LakeFS storage namespace uses the MinIO bucket `lakefs-data` (override via `LAKEFS_STORAGE_BUCKET`).

### Running the Pipeline

```bash
# Full pipeline: Kaggle → MinIO → dbt → Feast
python tools/scripts/run_kronodroid_pipeline.py --destination minio

# With LakeFS versioning
python tools/scripts/run_kronodroid_pipeline.py --destination lakefs --branch dev

# SparkOperator (kind): build & load the Spark image, then run the kubeflow transform engine
docker build -t dfp-spark:latest -f tools/docker/Dockerfile.spark .
kind load docker-image dfp-spark:latest --name dfp-kind
python tools/scripts/run_kronodroid_pipeline.py --destination lakefs --transform-engine kubeflow --k8s-namespace dfp

# Skip ingestion (if data already loaded)
python tools/scripts/run_kronodroid_pipeline.py --skip-ingestion

# Only materialize features to Redis
python tools/scripts/run_kronodroid_pipeline.py --materialize-only
```

### Pipeline Components

1. **dlt Engine** (`engines/dlt_engine/dfp_dlt/`): Downloads Kronodroid from Kaggle and loads to MinIO/LakeFS
2. **dbt Models** (`analytics/dbt/models/*/kronodroid/`): Transforms raw data into feature tables
3. **Feast Features** (`feature_stores/feast_store/dfp_feast/kronodroid_features.py`): Feature definitions for ML

### Data Flow
```
Kaggle API → dlt → MinIO (raw) → dbt → MinIO (transformed) → Feast → Redis (online)
                     ↓                        ↓
                  LakeFS (versioning)    LakeFS (versioning)
```

## MLflow Model Management

MLflow provides three levels of model storage, each serving a different purpose:

### 1. Run Artifacts (Model Logged)
**Location:** `/#/experiments/{id}/runs/{run_id}` → Artifacts tab

When you call `mlflow.pytorch.log_model(model, "model")`:
- Model files (weights, MLmodel, conda.yaml, etc.) are saved to the artifact store (MinIO)
- Path: `s3://mlflow-artifacts/{experiment_id}/{run_id}/artifacts/model/`
- This is file storage only - no versioning, no stage management, no central catalog
- Any run can have multiple logged models (e.g., "model", "best_model", "checkpoint")

### 2. Experiment Models Tab
**Location:** `/#/experiments/{id}/models`

A convenience view showing:
- All models logged via `log_model()` in runs belonging to this experiment
- Detected via the `mlflow.log-model.history` tag on each run
- Not a registry - just a UI aggregation of logged models across runs

### 3. Global Model Registry
**Location:** `/#/models`

When you call `mlflow.register_model(model_uri, "model_name")`:
- Creates an entry in the Model Registry (separate database table)
- Provides versioning (v1, v2, v3...), stages (None → Staging → Production → Archived), aliases, and lineage tracking
- Central catalog for all production-ready models

### Loading Models

```python
# From run artifacts (by run ID)
model = mlflow.pytorch.load_model("runs:/e3cfb6f0.../model")

# From registry (by name and version)
model = mlflow.pytorch.load_model("models:/kronodroid_autoencoder/1")

# From registry (by stage)
model = mlflow.pytorch.load_model("models:/kronodroid_autoencoder/Production")
```

### Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLflow Model Registry                        │
│                      /#/models                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ kronodroid_autoencoder                                   │    │
│  │   v1 (Production) ──────────┐                           │    │
│  │   v2 (Staging)    ────────┐ │                           │    │
│  │   v3 (None)       ──────┐ │ │                           │    │
│  └─────────────────────────│─│─│───────────────────────────┘    │
└────────────────────────────│─│─│────────────────────────────────┘
                             │ │ │  (references)
                             ▼ ▼ ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Experiment: kronodroid-autoencoder            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Run abc123   │ │ Run def456   │ │ Run ghi789   │             │
│  │  Artifacts:  │ │  Artifacts:  │ │  Artifacts:  │             │
│  │  - model/    │◄┼─│  - model/  │◄┼─│  - model/  │◄(registered)│
│  │  - logs/     │ │ │            │ │ │  - ckpt/   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Tests and CI
- Tests live in `tests/` (unit, integration, e2e). Run with `pytest` after replacing placeholders.
- GitHub Actions workflows in `ci/github/workflows` cover lint, build/test, and KFP compile; Tekton and Dagger stubs also provided.
