# MyProject

A flexible monorepo for building and testing ml and data engineering pipelines. The components should be modular and be able to be mixed and matched to test different combinations and configurations. Use feast with spark, iceberg, avro and lakefs and minio to create feature views for training, validation, testing, save them with lakefs and track/tie them to an mlflow run/experiment along with the trained model weights and executorch weights, training metrics and inference results. That way given an experiment and run id, you can go back in time and get the exact weights and training data + feature engineered features used create the model. The key is to treat LakeFS+Iceberg+Minio as your versioned data plane, and Feast+MLflow as the metadata/experiment plane that “points into” specific commits/snapshots. Also use Datahub as a metadata platform. Also use dbt and dlthub.

## Core Commands


## Project Layout

Overall Project and Goals - Roadmap and Reference Example:
	1.	A monorepo layout that explicitly supports swapping:
	•	Kubeflow ↔ Airflow
	•	Spark ↔ DataFusion
	•	Feast ↔ “other feature store”
	2.	How to handle imports/dependencies cleanly (Python vs build tool).
	3.	How to let people/CI run a single pipeline without cloning the whole monorepo (Git/GitHub patterns).

⸻

1. Monorepo principles for swappable components

Key idea: separate “core logic” from “integration glue.”
	•	Core logic: feature definitions, transformations, training code, eval logic.
	•	Integration glue:
	•	Orchestration: Kubeflow DAGs vs Airflow DAGs.
	•	Compute engine: Spark vs DataFusion adapters.
	•	Feature store implementation: Feast vs something else.

So you want:
	•	dfp.core (pure Python, no Kubeflow/Airflow imports).
	•	dfp.orchestration.kubeflow, dfp.orchestration.airflow (thin wrappers).
	•	dfp.engines.spark, dfp.engines.datafusion (engine-specific execution).
	•	dfp.features.feast_store, dfp.features.alt_store (feature store implementations).


2. Monorepo layout with pluggable orchestration, engines, feature stores

Here’s a concrete tree/ example layout for a single monorepo called dfp:

dfp-monorepo/
├─ WORKSPACE                         # Bazel workspace (Android, C++, runtimes, etc.)
├─ BUILD.bazel                       # Optional root BUILD
├─ pyproject.toml                    # Python monorepo (dfp_core, engines, etc.)
├─ README.md
├─ .gitignore
├─ .bazelrc
├─ docker-compose.yml                # Optional: local stack (minio, lakefs, mlflow, redis…)
├─ apps/
│  ├─ android/
│  │  ├─ main_app/                   # Main production Android app using ExecuTorch
│  │  │  ├─ BUILD.bazel
│  │  │  ├─ src/main/AndroidManifest.xml
│  │  │  └─ src/main/java/com/dfp/app/...
│  │  └─ runtime_comparison_app/     # Android APK to compare runtimes (ExecuTorch/TFLite/ONNX)
│  │     ├─ BUILD.bazel
│  │     ├─ src/main/AndroidManifest.xml
│  │     └─ src/main/java/com/dfp/runtimeapp/MainActivity.kt
│  │
│  └─ api/
│     ├─ dfp_api/                  # Backend API (FastAPI/Flask/etc.) for inference & results
│     │  ├─ __init__.py
│     │  ├─ main.py
│     │  ├─ routes/
│     │  └─ models/
│     └─ BUILD.bazel                 # Optional: build backend with Bazel
├─ core/
│  ├─ dfp_core/
│  │  ├─ __init__.py
│  │  ├─ config/                     # Pydantic schemas, global config helpers
│  │  │  ├─ mlflow_config.py
│  │  │  └─ data_config.py
│  │  ├─ domain/                     # Domain types (events, sessions, device info, etc.)
│  │  │  ├─ events.py
│  │  │  └─ device.py
│  │  ├─ features/                   # Engine-agnostic feature definitions
│  │  │  ├─ base.py                  # Interfaces: FeatureStore, ExecutionEngine, etc.
│  │  │  ├─ feature_views.py         # Logical (Python) feature definitions
│  │  │  └─ transformations.py       # Shared pandas/pyarrow/polars ops
│  │  ├─ ml/
│  │  │  ├─ models.py                # PyTorch models (AEs, VAEs, etc.)
│  │  │  ├─ datasets.py              # DataLoaders from parquet/Avro/features
│  │  │  ├─ training_pytorch.py      # Hydra-powered training (Feast + LakeFS + MLflow)
│  │  │  ├─ training_tf.py           # (slot) TensorFlow training (optional)
│  │  │  ├─ export_executorch.py     # Export PyTorch → ExecuTorch (.pte) (or placeholder)
│  │  │  ├─ export_tflite.py         # (slot) Export TF → TFLite
│  │  │  ├─ export_onnx.py           # Export PyTorch/TF → ONNX
│  │  │  └─ eval_server.py           # Common server-side eval/comparison utilities
│  │  ├─ io/
│  │  │  ├─ schema.py                # Schemas for logs/features, etc.
│  │  │  └─ contracts.py             # HTTP/IPC contracts between services
│  │  ├─ lakefs_client_utils.py      # Thin wrapper over official lakeFS Python client
│  │  └─ feast_mlflow_utils.py       # Standardized MLflow logging for Feast + LakeFS metadata
│  │
│  └─ conf/                          # Hydra config tree
│     ├─ config.yaml                 # Top-level defaults: model, data, engine, export, device…
│     ├─ model/
│     │  ├─ ae_small.yaml
│     │  ├─ ae_large.yaml
│     │  └─ vae.yaml
│     ├─ data/
│     │  ├─ android_logs_dev.yaml    # Includes LakeFS repo/branch + table paths
│     │  └─ android_logs_prod.yaml
│     ├─ engine/
│     │  ├─ spark.yaml               # Spark config (Iceberg, LakeFS catalog)
│     │  └─ datafusion.yaml          # DataFusion config (if you swap engines)
│     ├─ feature_store/
│     │  ├─ feast.yaml               # Config to pick Feast store and project
│     │  └─ alt_store.yaml           # Alternative feature store (slot)
│     ├─ export/
│     │  ├─ executorch_android.yaml  # ExecuTorch export parameters (quantization, paths)
│     │  └─ tflite_android.yaml      # TFLite export parameters
│     ├─ device/
│     │  ├─ galaxy_s24.yaml
│     │  └─ android_emulator.yaml
│     └─ orchestration/
│        ├─ kubeflow.yaml            # Pipeline-specific knobs
│        └─ airflow.yaml
├─ engines/
│  ├─ spark_engine/
│  │  ├─ dfp_spark/
│  │  │  ├─ __init__.py
│  │  │  ├─ session.py               # Creates SparkSession with Iceberg+LakeFS catalog
│  │  │  ├─ ingestion_jobs.py        # Raw logs → curated/bronze/silver tables
│  │  │  └─ feature_jobs.py          # Spark jobs → Iceberg feature tables (dfp.features.*)
│  │  └─ BUILD.bazel                 # Optional: Bazelify Spark jobs
│  └─ datafusion_engine/
│     ├─ dfp_datafusion/
│     │  ├─ __init__.py
│     │  ├─ session.py               # DataFusion context
│     │  └─ feature_jobs.py          # DataFusion-based feature builds
│     └─ BUILD.bazel
├─ feature_stores/
│  ├─ feast_store/
│  │  ├─ feature_store.yaml          # Feast project config (Spark offline store + LakeFS-backed Iceberg)
│  │  └─ dfp_feast/
│  │     ├─ __init__.py
│  │     ├─ entities.py              # Feast entities (user, device, session, etc.)
│  │     ├─ feature_views.py         # Feast FVs (e.g., user_daily_features from Iceberg)
│  │     └─ materialization_jobs.py  # Optional: materialize to online store (Redis)
│  └─ alt_store/
│     └─ dfp_alt_store/
│        ├─ __init__.py
│        └─ adapter.py               # Implementation of FeatureStore interface for an alternative FS
├─ runtimes/
│  ├─ executorch/
│  │  ├─ BUILD.bazel
│  │  ├─ native/                     # C++/JNI bridge to ExecuTorch runtime
│  │  │  ├─ execu_bridge.cc
│  │  │  └─ execu_bridge.h
│  │  └─ java/
│  │     └─ com/dfp/runtime/ExecuRunner.kt  # Kotlin wrapper (System.loadLibrary + JNI call)
│  │
│  ├─ tflite/
│  │  ├─ BUILD.bazel
│  │  ├─ native/                     # TFLite C++ runtime / custom ops bridge
│  │  └─ java/
│  │     └─ com/dfp/runtime/TFLiteRunner.kt
│  │
│  └─ onnx/
│     ├─ BUILD.bazel
│     ├─ native/                     # ONNX Runtime Mobile C++ bridge
│     └─ java/
│        └─ com/dfp/runtime/OnnxRunner.kt
├─ orchestration/
│  ├─ kubeflow/
│  │  └─ dfp_kfp/
│  │     ├─ __init__.py
│  │     ├─ components/
│  │     │  ├─ feast_build_training_set_component.py   # Feast → LakeFS → MLflow (train/val/test)
│  │     │  ├─ train_pytorch_executorch_component.py   # Calls dfp_core.ml.training_pytorch (Hydra)
│  │     │  ├─ export_executorch_component.py          # Calls dfp_core.ml.export_executorch
│  │     │  ├─ build_android_apk_component.py          # Bazel build //apps/android/runtime_comparison_app
│  │     │  ├─ eval_on_emulator_component.py           # Runs emulator + instrumentation tests
│  │     │  └─ aggregate_metrics_component.py          # Logs runtime metrics (Android/test) to MLflow
│  │     └─ pipelines/
│  │        ├─ execu_runtime_pipeline.py               # Train → export → build APK → eval → log metrics
│  │        └─ (later) tf_tflite_pipeline.py, onnx_pipeline.py, full comparison pipelines
│  │
│  ├─ airflow/
│  │  └─ dags/
│  │     ├─ dfp_feature_build_dag.py                 # Optional Airflow DAG for Spark feature jobs
│  │     └─ dfp_training_dag.py                      # Optional Airflow-based training DAG
│  │
│  └─ argo/
│     └─ workflows/                                    # Optional: Argo Workflows defs if you go that path
├─ analytics/
│  ├─ dbt/                                             # dbt project (models, seeds, snapshots, tests)
│  │  ├─ dbt_project.yml
│  │  ├─ profiles/ (or use env profiles)
│  │  ├─ models/
│  │  ├─ tests/
│  │  └─ macros/
│  └─ tinybird/                                        # Tinybird pipes for real-time APIs (optional)
│     ├─ pipes/
│     └─ datasources/
├─ infra/
│  ├─ iac/
│  │  ├─ lakefs/                                       # OpenTofu/Pulumi for LakeFS deployment & repos
│  │  ├─ spark-cluster/                                # Spark on K8s / EMR / Dataproc config
│  │  ├─ feast/                                        # Feast infra (Redis, service, etc.)
│  │  ├─ mlflow/                                       # MLflow tracking server & artifact store
│  │  ├─ kubeflow/                                     # Kubeflow Pipelines infra
│  │  └─ base-networking/                              # VPC, subnets, gateways…
│  │
│  ├─ k8s/
│  │  ├─ api/                                          # Manifests/Helm for dfp_api service
│  │  ├─ feast/                                        # Feast deployment on K8s
│  │  ├─ mlflow/                                       # MLflow deployment
│  │  ├─ lakefs/                                       # LakeFS service manifest
│  │  ├─ spark-operator/                               # Spark-on-K8s operator
│  │  └─ kubeflow/                                     # KFP manifests/Helm (if not installed separately)
│  │
│  └─ pyinfra/                                         # (Optional) pyinfra scripts for non-K8s hosts
├─ tools/
│  ├─ eval/
│  │  ├─ BUILD.bazel
│  │  ├─ eval_android.sh                               # Shell: start emulator, install APK, run tests
│  │  └─ eval_server.py                                # Server-side runtime comparison script
│  ├─ scripts/                                         # Ad-hoc CLIs: data loaders, debugging tools
│  │  ├─ sync_lakefs_branches.py
│  │  ├─ backfill_features.py
│  │  └─ replay_mlflow_run.py                          # (slot) Reconstruct run from MLflow+LakeFS
│  └─ docker/                                          # Dockerfiles for KFP components & services
│     ├─ Dockerfile.pytorch_train
│     ├─ Dockerfile.bazel_android
│     ├─ Dockerfile.android_emulator
│     └─ Dockerfile.feast_mlflow
├─ ci/
│  ├─ github/
│  │  └─ workflows/
│  │     ├─ build_and_test.yml                         # Test Python + Bazel + Android
│  │     ├─ kfp_pipeline_compile.yml                   # Compile & upload KFP pipelines
│  │     └─ lint.yml
│  ├─ buildkite/
│  │  ├─ pipeline.yml                                  # Buildkite pipeline definitions
│  │  └─ steps/
│  ├─ tekton/
│  │  ├─ tasks/
│  │  └─ pipelines/
│  └─ dagger/
│     ├─ main.go                                       # Dagger pipeline orchestrating builds/tests
│     └─ model_pipeline.go
├─ tests/
│  ├─ unit/
│  │  ├─ test_models.py
│  │  ├─ test_feature_transforms.py
│  │  └─ test_lakefs_utils.py
│  ├─ integration/
│  │  ├─ test_feast_spark_integration.py
│  │  ├─ test_mlflow_logging.py
│  │  └─ test_execu_android_roundtrip.py
│  └─ e2e/
│     └─ test_full_kfp_execu_pipeline.py
└─ docs/
   ├─ architecture.md                # High-level system diagram + tool taxonomy
   ├─ data_lineage.md                # How LakeFS + Iceberg + Feast tie together
   ├─ experiments.md                 # Rules for MLflow experiments & reproducibility
   └─ android_runtime_lab.md         # How to run the runtime_comparison pipeline end-to-end

## Development Patterns & Constraints

Coding style
• Use Pydantic and Panderas when appropriate in python
• Use uv as python package manager
• Use pyrefly and torchfix for linting, debugging and type checking

## Git Workflow Essentials

1. Branch from `main` with a descriptive name: `feature/<slug>` or `bugfix/<slug>`.
2. Keep commits atomic; prefer checkpoints (`feat: …`, `test: …`).
