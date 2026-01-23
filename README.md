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

## Kubeflow Pipelines

Kubeflow Pipelines (KFP) uses **Argo Workflows** as its execution engine. Understanding this hierarchy helps with debugging and management.

### Architecture

```
Kubeflow Pipeline Run
    └── Argo Workflow (Custom Resource - the orchestrator)
            ├── dag-driver pod (plans execution)
            ├── container-driver pod (prepares component)
            └── container-impl pod (runs your actual code)
```

When you submit a pipeline, KFP creates a **Workflow** CR. The Workflow Controller watches for these resources and manages pod lifecycle - if a pod is deleted, the controller recreates it.

### Managing Pipelines

**List running workflows:**
```bash
kubectl get workflows -n kubeflow
# Or use Taskfile
task kfp:workflows:list
```

**Delete a specific run (stops all its pods):**
```bash
kubectl delete workflow <workflow-name> -n kubeflow
# Or use Taskfile
task kfp:workflows:delete WORKFLOW=<workflow-name>
```

**Delete all workflows:**
```bash
kubectl delete workflows --all -n kubeflow
# Or use Taskfile
task kfp:workflows:delete ALL=1
```

**Clean up completed/failed pods:**
```bash
task kfp:cleanup        # Remove completed/failed pods
task kfp:cleanup ALL=1  # Remove all pipeline pods
```

**View logs from a running pipeline:**
```bash
# Find the impl pod (runs your actual code)
kubectl get pods -n kubeflow | grep impl

# View logs (use -c main for the main container)
kubectl logs -n kubeflow <pod-name> -c main -f
```

### KFP Training Image

For faster pipeline startup (skip runtime pip installs), build the pre-configured training image with all dependencies pre-installed:

```bash
# Build and load into Kind (takes 10-15 minutes on first build)
task kfp-training-image

# Or separately
task kfp-training-image:build  # Build the Docker image
task kfp-training-image:load   # Load into Kind cluster
```

**Check build status:**
```bash
# Check if image exists locally
docker images | grep dfp-kfp-training

# Check if build is currently running
docker ps | grep build

# Check buildx builders
docker buildx ls
```

**Build time:** The first build takes 10-15 minutes because it compiles `grpcio` from source (especially on ARM64). Subsequent builds are faster if Docker layers are cached.

**Fallback behavior:** If the `dfp-kfp-training:latest` image doesn't exist, the pipeline automatically falls back to using `apache/spark:3.5.0-python3` with runtime pip installs. This works but adds 5-10 minutes to pipeline startup time.

**After building:** Once the image is built and loaded, update `orchestration/kubeflow/dfp_kfp/components/train_pytorch_lightning_component.py` to use `base_image="dfp-kfp-training:latest"` and remove `packages_to_install` for faster startup.

### Troubleshooting

If pipelines fail with Gateway Timeout or pods are in CrashLoopBackOff:

```bash
# Check KFP pod status
kubectl get pods -n kubeflow

# Restart crashing pods (they'll be recreated by deployments)
kubectl delete pod -n kubeflow <pod-name>

# Check KFP API health
curl http://localhost:8080/apis/v2beta1/healthz
```

## Tests and CI
- Tests live in `tests/` (unit, integration, e2e). Run with `pytest` after replacing placeholders.
- GitHub Actions workflows in `ci/github/workflows` cover lint, build/test, and KFP compile; Tekton and Dagger stubs also provided.
