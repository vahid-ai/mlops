# Kronodroid PyTorch Lightning Autoencoder (Feast + Spark/Iceberg + LakeFS + MLflow + Kubeflow)

This repo is set up so that:
- **LakeFS + Iceberg + MinIO** is the versioned data plane (Avro data files inside Iceberg tables).
- **Feast + MLflow** is the metadata/experiment plane that points at specific LakeFS/Iceberg versions.
- **Kubeflow Pipelines** orchestrates transformations and training end-to-end.

## What you get

- A Feast FeatureView dedicated to autoencoder training: `kronodroid_autoencoder_features`
- A Lightning training implementation that:
  - Loads train/val/test splits via **Feast** from **LakeFS-pinned** Iceberg tables
  - Trains + validates + tests a PyTorch Lightning autoencoder
  - Logs metrics/artifacts to **MLflow**
  - Registers the model in the **MLflow Model Registry**
  - Tags the registered model version with **dataset lineage** (LakeFS ref + Iceberg snapshot id + dataset ids)
- A Kubeflow pipeline that wires:
  1) SparkOperator Kronodroid Iceberg transform (Avro)
  2) LakeFS commit + merge
  3) Lightning training + MLflow registration

## Key files

- Spark transformations (Avro Iceberg tables): `engines/spark_engine/dfp_spark/kronodroid_iceberg_job.py`
- Feast Kronodroid FeatureViews: `feature_stores/feast_store/dfp_feast/kronodroid_features.py`
- Lightning training implementation: `core/dfp_core/ml/train_kronodroid_autoencoder.py`
- KFP component: `orchestration/kubeflow/dfp_kfp/components/train_kronodroid_autoencoder_component.py`
- KFP pipeline: `orchestration/kubeflow/dfp_kfp/pipelines/kronodroid_autoencoder_pipeline.py`
- Training image Dockerfile: `tools/docker/Dockerfile.pytorch_train`

## Data lineage strategy (how a model maps to exact data)

Each pipeline run produces (or reuses) an Iceberg table `kronodroid.fct_training_dataset` under a **specific LakeFS ref**.

The training step reads features with:
- `LAKEFS_REPOSITORY=<repo>`
- `LAKEFS_BRANCH=<lakefs_ref>` where `lakefs_ref` is set to the **LakeFS merge commit id** output by the pipeline.

The MLflow run + registered model version record:
- `data.lakefs_repository`
- `data.lakefs_ref` (a LakeFS commit hash when the pipeline runs transformations)
- `data.source_table` (`fct_training_dataset`)
- `data.iceberg_snapshot_id` (best-effort from Iceberg metadata)
- `feast.feature_view` and `feast.defs_sha256`
- `dataset.train_id`, `dataset.val_id`, `dataset.test_id` (stable IDs derived from the full dataset spec)

This is the minimal efficient footprint to deterministically reconstruct the exact training/validation/testing datasets later.

## One-time prerequisites

1) Bring up your local stack (examples in `docker-compose.yml` and `tools/scripts/*`):
   - LakeFS + MinIO (+ credentials secrets in your cluster)
   - MLflow tracking server + artifact store (often MinIO/S3)
   - Spark Operator installed in the K8s cluster used by Kubeflow

2) Ensure the Kronodroid Iceberg tables exist (can be done via the existing pipeline):
   - `stg_kronodroid.*` and `kronodroid.fct_training_dataset`

3) Ensure Feast is applied for your environment (so the registry knows the FeatureViews):
   - Use `feature_stores/feast_store/feature_store_spark.yaml`
   - Feature definitions are in `feature_stores/feast_store/dfp_feast/`

## Build the training image (used by KFP component)

The KFP training component expects the image `dfp/kronodroid-train:latest` by default.

Build and push:
```bash
docker build -f tools/docker/Dockerfile.pytorch_train -t dfp/kronodroid-train:latest .
docker tag dfp/kronodroid-train:latest <your-registry>/dfp/kronodroid-train:latest
docker push <your-registry>/dfp/kronodroid-train:latest
```

If you push to a registry, update the image name in:
- `orchestration/kubeflow/dfp_kfp/components/train_kronodroid_autoencoder_component.py`

## Compile and run the Kubeflow pipeline

Compile:
```bash
python -c "from orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_autoencoder_pipeline import compile_kronodroid_autoencoder_pipeline; print(compile_kronodroid_autoencoder_pipeline())"
```

Upload `kronodroid_autoencoder_pipeline.yaml` to Kubeflow Pipelines and run it.

Recommended parameters:
- `run_transform=true` for a fully reproducible run that produces a new LakeFS commit
- `mlflow_tracking_uri=http://mlflow:5000` (or your deployment)
- `max_epochs`, `latent_dim`, `batch_size` for tuning

## Notes / knobs

- Avro is enforced by the Spark job via Iceberg table properties (`write.format.default=avro`).
- To pin to an existing branch instead of a commit hash, set `run_transform=false` and choose `target_branch`.
- For large datasets, keep `max_rows_per_split=0` (unlimited); use a small non-zero value for smoke tests.

