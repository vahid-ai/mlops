# Notebooks

Jupyter notebooks for exploring and interacting with the MLOps pipeline components.

## Available Notebooks

### `spark_operator_test.ipynb`

Test and debug notebook for Spark Operator with Kubeflow integration.

**Features:**
- Check Spark Operator deployment status
- Verify SparkApplication CRD availability
- Submit test SparkApplications
- Monitor application lifecycle with real-time status
- Retrieve driver and executor logs
- View Spark Operator controller logs
- Verify RBAC configuration
- Clean up test resources
- Comprehensive troubleshooting helpers

**Prerequisites:**
- Kind cluster running with Spark Operator deployed (`task spark-operator:up`)
- kubectl configured to access the cluster
- Python environment with subprocess support

**Usage:**
```bash
# From project root
cd notebooks
jupyter notebook spark_operator_test.ipynb
```

This notebook is perfect for:
- Verifying Spark Operator setup before running production jobs
- Debugging SparkApplication submission issues
- Understanding the SparkApplication lifecycle
- Testing Spark image compatibility

### `feast_spark_explorer.ipynb`

Interactive notebook for querying the Feast feature store with Spark backend.

**Features:**
- Initialize Spark session with Iceberg + LakeFS configuration
- Connect to Feast feature store
- List and display all feature views (regular, batch, on-demand)
- Explore entities and data sources
- Example code for historical feature retrieval

**Prerequisites:**
- Python environment with `feast`, `pyspark`, and project dependencies installed
- For full functionality: running LakeFS, MinIO, and Redis services (see `docker-compose.yml`)

**Usage:**
```bash
# From project root
cd notebooks
jupyter notebook feast_spark_explorer.ipynb
```

### `minio_utils.ipynb`

Utility functions for managing MinIO buckets and objects.

**Features:**
- List all buckets with sizes and object counts
- Create and delete buckets (with force option)
- Upload and download files
- Copy objects between buckets
- Delete objects by prefix
- Bulk operations with dry-run mode
- Comprehensive error handling
- Safety warnings for destructive operations

**Prerequisites:**
- MinIO server running (locally: `task up` or `docker-compose up`)
- boto3 installed
- Port-forwarding enabled (`task port-forward`)

**Usage:**
```bash
cd notebooks
jupyter notebook minio_utils.ipynb
```

**Safety Note:** Always use dry_run=True first for destructive operations!

### `run_kronodroid_pipeline.ipynb`

Notebook runner equivalent of `tools/scripts/run_kronodroid_pipeline.py`.

**Features:**
- Runs the Kronodroid pipeline step-by-step (dlt → transformations → Feast → LakeFS commit)
- Provides a single "Run full pipeline" cell mirroring the script flags
  - Step 2 supports `TRANSFORM_RUNNER = "dbt"` or `TRANSFORM_RUNNER = "spark-operator"`

## Environment Variables

The notebooks expect these environment variables (with defaults for local development):

| Variable | Default | Description |
|----------|---------|-------------|
| `LAKEFS_ENDPOINT_URL` | `http://localhost:8000` | LakeFS S3 gateway endpoint |
| `LAKEFS_ACCESS_KEY_ID` | - | LakeFS access key |
| `LAKEFS_SECRET_ACCESS_KEY` | - | LakeFS secret key |
| `LAKEFS_REPOSITORY` | `kronodroid` | LakeFS repository name |
| `LAKEFS_BRANCH` | `main` | LakeFS branch |
| `REDIS_CONNECTION_STRING` | `redis://localhost:16379` | Redis for online store |
