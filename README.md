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

## Tests and CI
- Tests live in `tests/` (unit, integration, e2e). Run with `pytest` after replacing placeholders.
- GitHub Actions workflows in `ci/github/workflows` cover lint, build/test, and KFP compile; Tekton and Dagger stubs also provided.
