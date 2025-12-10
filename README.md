# Digital Fingerprinting Project (dfp)

Skeleton monorepo that mirrors the pipeline1.md layout. It separates core ML logic from integration glue so Kubeflow/Airflow, Spark/DataFusion, and Feast/alternative stores can be swapped without touching the core code. LakeFS + Iceberg + Minio serve as the versioned data plane; Feast + MLflow + DataHub provide metadata and experiment traceability.

## Structure
See inline README stubs and doc files under `docs/` for guidance on how each package fits together. Use `uv` for Python dependencies and `pyrefly`/`torchfix` for linting and typing checks.
