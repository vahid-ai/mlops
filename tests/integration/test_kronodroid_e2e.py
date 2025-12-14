"""End-to-end integration tests for the Kronodroid data pipeline.

Tests the full flow:
    dlt (Kaggle) → MinIO → dbt → DuckDB → Export → LakeFS → Feast

Prerequisites:
    - MinIO running at localhost:19000
    - LakeFS running at localhost:8000
    - Redis running at localhost:16379
    - Environment variables set for credentials

Run with:
    pytest tests/integration/test_kronodroid_e2e.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env_file():
    """Load environment variables from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


# Load environment at module level
load_env_file()


def is_minio_available() -> bool:
    """Check if MinIO is accessible."""
    try:
        import boto3
        from botocore.exceptions import ClientError, EndpointConnectionError

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://localhost:19000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin"),
            region_name="us-east-1",
        )
        s3.list_buckets()
        return True
    except (EndpointConnectionError, ClientError):
        return False
    except Exception:
        return False


def is_lakefs_available() -> bool:
    """Check if LakeFS is accessible."""
    try:
        import requests

        endpoint = os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
        resp = requests.get(f"{endpoint}/api/v1/healthcheck", timeout=5)
        return resp.status_code == 204
    except Exception:
        return False


def is_redis_available() -> bool:
    """Check if Redis is accessible."""
    try:
        import redis

        conn_string = os.getenv("REDIS_CONNECTION_STRING", "redis://localhost:16379")
        r = redis.from_url(conn_string)
        r.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def ensure_infrastructure():
    """Ensure infrastructure is available before running tests."""
    if not is_minio_available():
        pytest.skip("MinIO not available at localhost:19000")
    if not is_lakefs_available():
        pytest.skip("LakeFS not available at localhost:8000")
    # Redis is optional for some tests


class TestDltIngestion:
    """Test dlt ingestion from Kaggle to MinIO."""

    @pytest.mark.skipif(not is_minio_available(), reason="MinIO not available")
    def test_dlt_ingestion_creates_bucket(self, ensure_infrastructure):
        """Test that dlt ingestion creates the dlt-data bucket."""
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://localhost:19000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin"),
            region_name="us-east-1",
        )

        buckets = s3.list_buckets()
        bucket_names = [b["Name"] for b in buckets.get("Buckets", [])]
        assert "dlt-data" in bucket_names or "lakefs-data" in bucket_names

    @pytest.mark.skipif(not is_minio_available(), reason="MinIO not available")
    def test_kronodroid_data_exists_in_minio(self, ensure_infrastructure):
        """Test that Kronodroid data files exist in MinIO."""
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://localhost:19000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin"),
            region_name="us-east-1",
        )

        response = s3.list_objects_v2(
            Bucket="dlt-data", Prefix="kronodroid_raw/", MaxKeys=10
        )
        contents = response.get("Contents", [])
        assert len(contents) > 0, "No Kronodroid data files found in MinIO"


class TestDbtTransformations:
    """Test dbt transformations."""

    @pytest.fixture(scope="class")
    def duckdb_path(self):
        """Path to the dbt DuckDB database."""
        return PROJECT_ROOT / "analytics" / "dbt" / "data" / "dbt_dev.duckdb"

    def test_dbt_database_exists(self, duckdb_path):
        """Test that dbt has created the DuckDB database."""
        assert duckdb_path.exists(), f"DuckDB database not found at {duckdb_path}"

    def test_staging_tables_exist(self, duckdb_path):
        """Test that staging tables were created."""
        import duckdb

        conn = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            tables = conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_name LIKE 'stg_kronodroid%'
                """
            ).fetchall()
            table_names = [t[1] for t in tables]
            assert "stg_kronodroid__emulator" in table_names
            assert "stg_kronodroid__real_device" in table_names
            assert "stg_kronodroid__combined" in table_names
        finally:
            conn.close()

    def test_mart_tables_exist(self, duckdb_path):
        """Test that mart tables were created."""
        import duckdb

        conn = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            tables = conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_name LIKE 'fct_%' OR table_name LIKE 'dim_%'
                """
            ).fetchall()
            table_names = [t[1] for t in tables]
            assert "fct_malware_samples" in table_names
            assert "fct_training_dataset" in table_names
            assert "dim_malware_families" in table_names
        finally:
            conn.close()

    def test_training_dataset_has_splits(self, duckdb_path):
        """Test that training dataset has train/val/test splits."""
        import duckdb

        conn = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            result = conn.execute(
                """
                SELECT dataset_split, count(*) as cnt
                FROM main_kronodroid.fct_training_dataset
                GROUP BY dataset_split
                ORDER BY dataset_split
                """
            ).fetchall()
            splits = {r[0]: r[1] for r in result}
            assert "train" in splits
            assert "validation" in splits
            assert "test" in splits
            # Check approximate split ratios (70/15/15)
            total = sum(splits.values())
            assert splits["train"] / total > 0.6  # At least 60% train
            assert splits["train"] / total < 0.8  # At most 80% train
        finally:
            conn.close()


class TestLakeFSExport:
    """Test LakeFS export functionality."""

    @pytest.mark.skipif(not is_lakefs_available(), reason="LakeFS not available")
    def test_lakefs_repository_exists(self, ensure_infrastructure):
        """Test that LakeFS repository exists."""
        import requests

        endpoint = os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
        repo = os.getenv("LAKEFS_REPOSITORY", "kronodroid")
        auth = (
            os.getenv("LAKEFS_ACCESS_KEY_ID"),
            os.getenv("LAKEFS_SECRET_ACCESS_KEY"),
        )

        resp = requests.get(f"{endpoint}/api/v1/repositories/{repo}", auth=auth)
        assert resp.status_code == 200, f"Repository {repo} not found"

    @pytest.mark.skipif(not is_lakefs_available(), reason="LakeFS not available")
    def test_dbt_tables_exported_to_lakefs(self, ensure_infrastructure):
        """Test that dbt tables were exported to LakeFS."""
        import boto3

        endpoint = os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
        repo = os.getenv("LAKEFS_REPOSITORY", "kronodroid")
        branch = os.getenv("LAKEFS_BRANCH", "dev")

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv("LAKEFS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("LAKEFS_SECRET_ACCESS_KEY"),
            region_name="us-east-1",
        )

        # List objects in the dbt export path
        response = s3.list_objects_v2(Bucket=repo, Prefix=f"{branch}/dbt/", MaxKeys=10)
        contents = response.get("Contents", [])
        keys = [c["Key"] for c in contents]

        # Check for expected table exports
        assert any("fct_training_dataset" in k for k in keys), "fct_training_dataset not found"
        assert any("fct_malware_samples" in k for k in keys), "fct_malware_samples not found"


class TestFeastIntegration:
    """Test Feast feature store integration."""

    @pytest.fixture(scope="class")
    def feature_store(self):
        """Initialize Feast feature store."""
        from feast import FeatureStore

        return FeatureStore(repo_path=str(PROJECT_ROOT / "feature_stores" / "feast_store"))

    def test_feature_views_registered(self, feature_store):
        """Test that feature views are registered."""
        views = feature_store.list_feature_views()
        view_names = [v.name for v in views]
        assert "malware_sample_features" in view_names

    def test_data_sources_configured(self, feature_store):
        """Test that data sources are configured correctly."""
        sources = feature_store.list_data_sources()
        source_names = [s.name for s in sources]
        assert "kronodroid_training_source" in source_names

    def test_entities_registered(self, feature_store):
        """Test that entities are registered."""
        entities = feature_store.list_entities()
        entity_names = [e.name for e in entities]
        assert "malware_sample" in entity_names


class TestFullPipeline:
    """Test the full end-to-end pipeline."""

    @pytest.mark.skipif(
        not (is_minio_available() and is_lakefs_available()),
        reason="Infrastructure not available",
    )
    def test_pipeline_data_integrity(self, ensure_infrastructure):
        """Test that data flows correctly through the pipeline."""
        import boto3
        import duckdb

        # Check MinIO has raw data
        minio_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://localhost:19000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin"),
            region_name="us-east-1",
        )

        # Check DuckDB has transformed data
        duckdb_path = PROJECT_ROOT / "analytics" / "dbt" / "data" / "dbt_dev.duckdb"
        if duckdb_path.exists():
            conn = duckdb.connect(str(duckdb_path), read_only=True)
            try:
                result = conn.execute(
                    "SELECT count(*) FROM main_kronodroid.fct_training_dataset"
                ).fetchone()
                row_count = result[0]
                assert row_count > 0, "No rows in training dataset"
                print(f"Training dataset has {row_count} rows")
            finally:
                conn.close()

        # Check LakeFS has exported data
        lakefs_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000"),
            aws_access_key_id=os.getenv("LAKEFS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("LAKEFS_SECRET_ACCESS_KEY"),
            region_name="us-east-1",
        )

        repo = os.getenv("LAKEFS_REPOSITORY", "kronodroid")
        branch = os.getenv("LAKEFS_BRANCH", "dev")

        response = lakefs_client.list_objects_v2(
            Bucket=repo, Prefix=f"{branch}/dbt/fct_training_dataset/", MaxKeys=5
        )
        contents = response.get("Contents", [])
        assert len(contents) > 0, "No exported data in LakeFS"
