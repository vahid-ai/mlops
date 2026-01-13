# Spark Operator Manifests

Spark-on-Kubernetes operator configuration for running Spark jobs in the cluster.

## Overview

The Spark Operator enables running Apache Spark applications natively on Kubernetes.
This directory contains:

- Helm values for installing the operator
- SparkApplication templates for the Kronodroid Iceberg job
- RBAC configuration for Spark service accounts
- Secrets templates for MinIO and LakeFS credentials

## Quick Start

### 1. Install the Spark Operator

Using Helm:

```bash
# Add the Spark Operator Helm repo
helm repo add spark-operator https://kubeflow.github.io/spark-operator
helm repo update

# Install the operator
helm install spark-operator spark-operator/spark-operator \
  --namespace spark-operator \
  --create-namespace \
  --values values.yaml
```

### 2. Create Secrets

Before running Spark jobs, create the required secrets:

```bash
# MinIO credentials
kubectl create secret generic minio-credentials \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=minioadmin

# LakeFS credentials
kubectl create secret generic lakefs-credentials \
  --from-literal=access-key=${LAKEFS_ACCESS_KEY_ID} \
  --from-literal=secret-key=${LAKEFS_SECRET_ACCESS_KEY}
```

### 3. Apply RBAC

```bash
kubectl apply -f rbac.yaml
```

### 4. Test with a SparkApplication

```bash
kubectl apply -f sparkapplication-kronodroid.yaml
```

## Files

| File | Description |
|------|-------------|
| `values.yaml` | Helm values for Spark Operator installation |
| `rbac.yaml` | ServiceAccount, Role, and RoleBinding for Spark |
| `sparkapplication-kronodroid.yaml` | Template SparkApplication for Kronodroid Iceberg job |
| `secrets-template.yaml` | Template for creating credential secrets |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌────────────────────────────────┐ │
│  │  Spark Operator  │    │     SparkApplication CRD       │ │
│  │   (Controller)   │◄───│  kronodroid-iceberg-<run_id>   │ │
│  └────────┬─────────┘    └────────────────────────────────┘ │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Spark Cluster                         ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               ││
│  │  │  Driver  │  │ Executor │  │ Executor │               ││
│  │  │   Pod    │  │   Pod    │  │   Pod    │               ││
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘               ││
│  │       │              │              │                    ││
│  └───────┼──────────────┼──────────────┼────────────────────┘│
│          │              │              │                     │
│          ▼              ▼              ▼                     │
│  ┌───────────────┐  ┌─────────────────────────────────────┐ │
│  │    MinIO      │  │           LakeFS                     │ │
│  │ (Raw Parquet) │  │  (Iceberg REST Catalog + S3 GW)     │ │
│  └───────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Spark Resources

The default SparkApplication template uses:

| Resource | Driver | Executor |
|----------|--------|----------|
| Cores | 1 | 2 |
| Memory | 2g | 2g |
| Instances | 1 | 2 |

Adjust these in the SparkApplication spec or pass them as pipeline parameters.

### Environment Variables

The Spark job expects these environment variables (injected via secrets):

| Variable | Secret | Key |
|----------|--------|-----|
| `MINIO_ENDPOINT_URL` | - | Hardcoded in spec |
| `MINIO_ACCESS_KEY_ID` | `minio-credentials` | `access-key` |
| `MINIO_SECRET_ACCESS_KEY` | `minio-credentials` | `secret-key` |
| `LAKEFS_ENDPOINT_URL` | - | Hardcoded in spec |
| `LAKEFS_ACCESS_KEY_ID` | `lakefs-credentials` | `access-key` |
| `LAKEFS_SECRET_ACCESS_KEY` | `lakefs-credentials` | `secret-key` |
| `LAKEFS_REPOSITORY` | - | Hardcoded in spec |
| `LAKEFS_BRANCH` | - | Passed as argument |

### Spark Configuration

Key Spark configs for Iceberg + LakeFS:

```yaml
sparkConf:
  # Iceberg extensions
  spark.sql.extensions: org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
  
  # LakeFS Iceberg REST catalog
  spark.sql.catalog.lakefs: org.apache.iceberg.spark.SparkCatalog
  spark.sql.catalog.lakefs.catalog-impl: org.apache.iceberg.rest.RESTCatalog
  spark.sql.catalog.lakefs.uri: http://lakefs:8000/api/v1/iceberg
  spark.sql.catalog.lakefs.warehouse: s3a://kronodroid/main/iceberg
  
  # S3A filesystem (path-style access for MinIO/LakeFS)
  spark.hadoop.fs.s3a.impl: org.apache.hadoop.fs.s3a.S3AFileSystem
  spark.hadoop.fs.s3a.path.style.access: "true"
  spark.hadoop.fs.s3a.connection.ssl.enabled: "false"
  
  # Per-bucket endpoints
  spark.hadoop.fs.s3a.bucket.dlt-data.endpoint: http://minio:9000
  spark.hadoop.fs.s3a.bucket.kronodroid.endpoint: http://lakefs:8000
```

## Troubleshooting

### Check SparkApplication status

```bash
kubectl get sparkapplications
kubectl describe sparkapplication kronodroid-iceberg-<run_id>
```

### View driver logs

```bash
kubectl logs <driver-pod-name>
```

### Common issues

1. **Pod pending**: Check if there are enough resources in the cluster
2. **S3 access denied**: Verify secrets are correctly mounted
3. **Iceberg catalog errors**: Check LakeFS endpoint is accessible
4. **Job timeout**: Increase executor instances or memory

## Integration with Kubeflow Pipelines

The `spark_kronodroid_iceberg_component.py` KFP component:

1. Creates a per-run LakeFS branch
2. Generates a SparkApplication manifest from template
3. Submits the SparkApplication via K8s API
4. Polls for completion
5. Returns the branch name for downstream commit/merge

See `orchestration/kubeflow/dfp_kfp/pipelines/kronodroid_iceberg_pipeline.py` for the full pipeline definition.
