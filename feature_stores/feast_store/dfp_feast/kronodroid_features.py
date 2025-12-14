"""Feast feature views for Kronodroid Android malware detection.

Data Flow:
    dlt (Kaggle) → Avro → MinIO → Spark → Iceberg (LakeFS) → Feast

The feature sources read from Iceberg tables on LakeFS via SparkSource.
dbt-spark writes Iceberg tables that Feast consumes for feature serving.
"""

import os
from datetime import timedelta

import pandas as pd

from feast import (
    BatchFeatureView,
    Field,
    FeatureView,
    PushSource,
)
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64, String

from .entities import malware_sample

# Configuration from environment
LAKEFS_REPO = os.getenv("LAKEFS_REPOSITORY", "kronodroid")
LAKEFS_BRANCH = os.getenv("LAKEFS_BRANCH", "main")
ICEBERG_CATALOG = os.getenv("ICEBERG_CATALOG", "lakefs_catalog")
ICEBERG_DATABASE = os.getenv("ICEBERG_DATABASE", "dfp")


def get_iceberg_table_name(table: str) -> str:
    """Get fully qualified Iceberg table name.

    Args:
        table: Table name

    Returns:
        Fully qualified table name (catalog.database.table)
    """
    return f"{ICEBERG_CATALOG}.{ICEBERG_DATABASE}.{table}"


# SparkSource for training dataset (Iceberg table from dbt)
kronodroid_training_source = SparkSource(
    name="kronodroid_training_source",
    table=get_iceberg_table_name("fct_training_dataset"),
    timestamp_field="event_timestamp",
    description="Training dataset from dbt mart (Iceberg table on LakeFS)",
)

# SparkSource for malware samples fact table
kronodroid_samples_source = SparkSource(
    name="kronodroid_samples_source",
    table=get_iceberg_table_name("fct_malware_samples"),
    timestamp_field="event_timestamp",
    description="Malware samples from dbt mart (Iceberg table on LakeFS)",
)

# SparkSource for category statistics
kronodroid_categories_source = SparkSource(
    name="kronodroid_categories_source",
    table=get_iceberg_table_name("dim_malware_families"),
    timestamp_field="_dbt_loaded_at",
    description="Malware family statistics (Iceberg table on LakeFS)",
)

# Push source for real-time feature updates
kronodroid_push_source = PushSource(
    name="kronodroid_push_source",
    batch_source=kronodroid_training_source,
)


# Main feature view for malware sample features
# Schema matches fct_training_dataset output from dbt
malware_sample_features = FeatureView(
    name="malware_sample_features",
    entities=[malware_sample],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        # Core identifiers and labels
        Field(name="app_package", dtype=String, description="Android app package name"),
        Field(name="is_malware", dtype=Int64, description="1=malware, 0=benign"),
        Field(name="data_source", dtype=String, description="emulator or real_device"),
        Field(name="dataset_split", dtype=String, description="train/validation/test"),
    ],
    source=kronodroid_training_source,
    online=True,
    tags={"team": "dfp", "dataset": "kronodroid", "storage": "iceberg"},
)


# Batch feature view for offline training (subset of features)
malware_batch_features = BatchFeatureView(
    name="malware_batch_features",
    entities=[malware_sample],
    ttl=timedelta(days=365),
    schema=[
        Field(name="is_malware", dtype=Int64, description="Target label"),
        Field(name="data_source", dtype=String),
        Field(name="dataset_split", dtype=String),
    ],
    source=kronodroid_training_source,
    tags={"team": "dfp", "dataset": "kronodroid", "usage": "training", "storage": "iceberg"},
)


# On-demand feature view for derived features
@on_demand_feature_view(
    sources=[malware_sample_features],
    schema=[
        Field(name="is_emulator_sample", dtype=Int64),
    ],
)
def malware_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from base malware sample features."""
    df = pd.DataFrame()
    df["is_emulator_sample"] = (inputs["data_source"] == "emulator").astype(int)
    return df
