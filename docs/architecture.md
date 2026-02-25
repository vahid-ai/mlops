# Architecture

DFP separates the platform into two planes:
- **Versioned data plane**: MinIO + LakeFS + Iceberg (+ Spark for transformations)
- **Metadata/ML control plane**: Feast + MLflow + DataHub (+ Kubeflow for orchestration)

This split is what makes experiments reproducible and operationally scalable.

## Why these tools exist

| Tool | What it does | Why it is needed |
| --- | --- | --- |
| MinIO (object storage) | Stores raw files, table data files, model artifacts, and exports in S3-compatible buckets. | Provides durable, low-cost, scalable storage foundation for all data and artifacts. |
| LakeFS | Adds Git-like branching, commit, merge, and rollback semantics over object storage paths. | Lets teams version entire datasets and recover exact training states. |
| Apache Iceberg | Table format with ACID transactions, schema evolution, partition evolution, and snapshot metadata. | Makes large analytical tables reliable and queryable over object storage without mutable warehouse locks. |
| Spark | Distributed transform engine for ingestion, joins, aggregations, and feature generation on large data. | Executes heavy data engineering workloads efficiently and writes Iceberg tables at scale. |
| Feast | Feature store metadata/control layer for entities, feature views, and offline/online retrieval contracts. | Standardizes how training and serving read identical feature definitions. |
| MLflow | Tracks runs, params, metrics, artifacts, and model versions. | Provides experiment traceability and reproducibility for model development. |
| DataHub | Enterprise metadata catalog for datasets, schemas, lineage, ownership, and governance. | Makes data discoverable, auditable, and governable across tools and teams. |
| Kubeflow Pipelines | DAG orchestration for repeatable ML workflows on Kubernetes. | Coordinates multi-step pipelines, retries, caching, and dependency-aware execution. |
| HPO + model optimization steps | Automated search and post-training optimization (quantization/export/pruning where applicable). | Improves model quality-cost tradeoff and makes models deployable to runtime targets (for example ExecuTorch). |

## Why DataHub with object-backed storage

DataHub is not a replacement for MinIO/LakeFS/Iceberg; it is the metadata layer above them.

- Object storage (MinIO/S3-compatible) stores the physical files.
- LakeFS versions those files as commits/branches.
- Iceberg organizes them as transactional tables.
- DataHub indexes the resulting datasets, schemas, lineage, ownership, and tags.

This gives a clean separation:
- **Storage systems** handle bytes and table state.
- **Metadata systems** handle discoverability, lineage, trust, and governance.

Without DataHub, teams can still run pipelines, but cross-team visibility and governance degrade quickly.

## System integration view

```mermaid
flowchart LR
  subgraph DP[Versioned Data Plane]
    O[MinIO Object Storage]
    L[LakeFS Version Control]
    I[Apache Iceberg Tables]
    S[Spark Transform Engine]
    O --> L --> I
    S --> I
  end

  subgraph MP[Metadata and ML Control Plane]
    F[Feast Feature Registry]
    M[MLflow Tracking and Registry]
    D[DataHub Catalog and Lineage]
    K[Kubeflow Pipelines]
    H[HPO and Model Optimization]
  end

  I --> F
  I --> D
  L --> M
  F --> K
  M --> K
  K --> S
  K --> H
  H --> M
  K --> D
```

## End-to-end pipeline flow

```mermaid
sequenceDiagram
  participant K as Kubeflow Pipeline
  participant SP as Spark
  participant LF as LakeFS
  participant IC as Iceberg
  participant FS as Feast
  participant ML as MLflow
  participant DH as DataHub

  K->>SP: Run ingest + transform jobs
  SP->>IC: Write curated feature tables
  IC->>LF: Persist table files on versioned object paths
  K->>LF: Commit/merge branch for this run
  K->>FS: Register/update feature views
  K->>ML: Start run and log data commit IDs
  K->>K: Execute hyperparameter tuning trials
  K->>ML: Log trial metrics + best params
  K->>K: Run optimization/export (for example ExecuTorch)
  K->>ML: Log model artifacts + optimization metadata
  K->>DH: Publish lineage, ownership, and dataset metadata
```

## Practical contract for reproducibility

For each training run, log at least:
- LakeFS repo, branch, commit ID
- Iceberg table(s) and snapshot ID(s)
- Feast feature service/view version
- Hyperparameters and search space
- Best-trial metrics and selected checkpoint
- Optimization/export settings and output artifact URIs

This is the minimum contract that allows a run to be replayed and audited later.

## Why Kubeflow orchestrates HPO and optimization

Hyperparameter tuning and optimization are not isolated scripts; they depend on data versions, feature definitions, and prior training outputs. Kubeflow makes this dependency graph explicit:
- Retries failed steps without rerunning everything.
- Parallelizes trials while preserving lineage.
- Captures step inputs/outputs for auditability.
- Promotes repeatable production-grade workflows instead of ad-hoc notebooks.
