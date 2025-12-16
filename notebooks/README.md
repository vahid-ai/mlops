# Notebooks

Jupyter notebooks for exploring and interacting with the MLOps pipeline components.

## Available Notebooks

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

### `run_kronodroid_pipeline.ipynb`

Notebook runner equivalent of `tools/scripts/run_kronodroid_pipeline.py`.

**Features:**
- Runs the Kronodroid pipeline step-by-step (dlt → dbt → Feast → LakeFS commit)
- Provides a single "Run full pipeline" cell mirroring the script flags

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
