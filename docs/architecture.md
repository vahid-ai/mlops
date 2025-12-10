# Architecture

DFP stitches LakeFS + Iceberg + Minio as the versioned data plane with Feast + MLflow + DataHub as the metadata plane. Core code stays engine-agnostic while orchestration layers plug into Kubeflow or Airflow.
