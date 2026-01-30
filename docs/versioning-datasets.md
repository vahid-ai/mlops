# ML Dataset Versioning Strategy

A layered versioning approach for LakeFS + Iceberg + Feast + MLflow stacks.

## Tool Responsibilities

```
+-------------------+------------------+----------------------------------------+
| Concern           | Primary Tool     | Why                                    |
+-------------------+------------------+----------------------------------------+
| Data versioning   | LakeFS           | Atomic commits, branches for experiments|
| Table time-travel | Iceberg          | Convenience queries (secondary)        |
| Feature definitions| Feast           | Point-in-time joins, online/offline    |
| Experiment lineage| MLflow           | Links everything together              |
+-------------------+------------------+----------------------------------------+
```

---

## Core Principles

### 1. Don't Duplicate Data for Splits

Instead of separate train/val/test tables, use a **split assignment table**:

```
ml_datasets.splits
├── entity_key: string
├── event_timestamp: timestamp
├── split: string ("train" | "val" | "test")
├── split_version: string ("v1_stratified_80_10_10")
└── split_seed: int
```

**Benefits:** Changing splits updates a tiny index table, not TBs of feature data.

### 2. LakeFS Branching Strategy

```
main (production)
 │
 ├── [tag] feature-eng/v1
 ├── [tag] feature-eng/v2
 │
 └── experiments/
      ├── exp-123-new-embeddings    (branch, delete after)
      └── exp-124-different-splits  (branch, delete after)
```

- **Tags** (immutable) = anything you need to trace back to
- **Branches** = experimentation, delete after merge

### 3. Feast Defines Retrieval, Not Storage

```python
# Same feature view for all splits - filter at retrieval time
training_df = store.get_historical_features(
    entity_df=splits_df[splits_df.split == "train"],
    features=["model_v2_features:txn_count_7d", ...],
).to_df()
```

### 4. MLflow Logs Everything

```python
mlflow.log_param("lakefs_commit", "<commit_sha>")
mlflow.log_param("lakefs_tag", "model-v2.3.0-data")
mlflow.log_param("split_version", "v1_stratified_80_10_10")
mlflow.log_param("feast_repo_commit", "<git_sha>")
mlflow.log_param("feast_feature_service", "model_v2_features")
```

---

## Storage Optimization

| Strategy                              | Storage Impact              |
|---------------------------------------|-----------------------------|
| Split index table vs duplicated tables| ~95% reduction for splits   |
| LakeFS branches (copy-on-write)       | Only changed files duplicated|
| Iceberg snapshots                     | Metadata only, shared files |
| Feature views over source tables      | Zero duplication            |

---

## Complete Reproducibility Flow

```
MLflow Model Registry
 └── model: fraud-detector-v2.3
      └── params:
           ├── lakefs_tag: "model-v2.3.0-data"
           ├── split_version: "v1_stratified"
           ├── feast_feature_service: "fraud_model_v2"
           └── feast_repo_commit: "abc123"

                    │
                    ▼ To reproduce:

1. lakefs checkout ml-data --ref model-v2.3.0-data
2. cd feast_repo && git checkout abc123 && feast apply
3. store.get_historical_features(
       entity_df=splits.filter(split_version="v1_stratified", split="train"),
       features=["fraud_model_v2:*"]
   )
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT TIME                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   LakeFS main branch                                            │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐              ┌─────────────────┐              │
│   │ Iceberg     │              │ Split Index     │              │
│   │ Feature     │              │ Table (tiny)    │              │
│   │ Tables      │              └─────────────────┘              │
│   └─────────────┘                      │                        │
│          │                             │                        │
│          └──────────┬──────────────────┘                        │
│                     ▼                                           │
│           ┌──────────────────┐                                  │
│           │ Feast            │◄─── Feature definitions          │
│           │ Feature Service  │     (Git versioned)              │
│           └──────────────────┘                                  │
│                     │                                           │
│                     ▼  Point-in-time join                       │
│           ┌──────────────────┐                                  │
│           │ Training Data    │                                  │
│           │ (ephemeral)      │                                  │
│           └──────────────────┘                                  │
│                     │                                           │
│                     ▼                                           │
│           ┌──────────────────┐                                  │
│           │ MLflow Run       │                                  │
│           │ ├─ lakefs_commit │                                  │
│           │ ├─ split_version │                                  │
│           │ ├─ feast_commit  │                                  │
│           │ └─ model         │                                  │
│           └──────────────────┘                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                   PROMOTION TO PRODUCTION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   lakefs tag create "model-v2.3.0-data" @ commit                │
│                     │                                           │
│                     ▼                                           │
│           ┌──────────────────┐                                  │
│           │ MLflow Model     │                                  │
│           │ Registry         │──► Production                    │
│           │ fraud-detector   │                                  │
│           │ version 23       │                                  │
│           └──────────────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Anti-Patterns to Avoid

| Don't                                                    | Do Instead                              |
|----------------------------------------------------------|-----------------------------------------|
| Create train/val/test as separate Iceberg tables         | Use split index table                   |
| Use LakeFS branches as permanent version markers         | Use LakeFS tags                         |
| Version feature transforms in LakeFS                     | Keep in Git with Feast definitions      |
| Rely on Iceberg snapshots as primary lineage             | Use LakeFS commits (snapshots can expire)|

---

## Feature Engineering Changes

When you want same splits but different feature transformations:

```python
# 1. Create experiment branch
lakefs.branch_create(repository="ml-data", branch="exp-new-normalization", source="main")

# 2. Run feature pipeline on branch
spark.conf.set("spark.sql.catalog.lakefs.ref", "exp-new-normalization")
run_feature_pipeline()

# 3. If successful, merge and tag
lakefs.merge("exp-new-normalization", "main")
lakefs.tag_create("feature-eng-v3")
```

---

## Key Lineage Parameters for MLflow

```python
@dataclass
class DataLineage:
    # LakeFS state
    lakefs_repository: str
    lakefs_ref: str           # branch or tag name
    lakefs_commit: str        # resolved SHA

    # Split definition
    split_version: str
    split_seed: int

    # Feast state
    feast_repo_commit: str
    feast_feature_service: str

    # Dataset statistics
    train_samples: int
    val_samples: int
    test_samples: int
    feature_schema_hash: str  # detect schema drift
```

---

## Quick Reference

**Tag after successful experiment:**
```python
lakefs.tag_create(repository="ml-data", tag="model-v2.3.0-data", ref="main")
```

**Query lineage from model registry:**
```python
run = client.get_run(model_version.run_id)
lakefs_commit = run.data.params["data/lakefs_commit"]
feast_commit = run.data.params["data/feast_repo_commit"]
```
