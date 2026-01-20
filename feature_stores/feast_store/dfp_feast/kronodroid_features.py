"""Feast feature views for Kronodroid Android malware detection."""

import os
from datetime import timedelta

import pandas as pd

from feast import (
    BatchFeatureView,
    Entity,
    Field,
    FeatureView,
    PushSource,
)
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String

from .entities import malware_family, malware_sample

# LakeFS Iceberg catalog and table configuration
# Tables located at: lakefs://kronodroid/dev/iceberg/kronodroid/<table_name>
LAKEFS_CATALOG = os.environ.get("LAKEFS_CATALOG", "lakefs")
LAKEFS_DATABASE = os.environ.get("LAKEFS_DATABASE", "kronodroid")

# Spark source for training dataset from LakeFS-tracked Iceberg table
# Path: s3a://kronodroid/dev/iceberg/kronodroid/fct_training_dataset
kronodroid_training_source = SparkSource(
    name="kronodroid_training_source",
    table=f"{LAKEFS_CATALOG}.{LAKEFS_DATABASE}.fct_training_dataset",
    timestamp_field="event_timestamp",
    description="Training dataset from LakeFS-tracked Iceberg table",
)

# Spark source for family statistics from LakeFS-tracked Iceberg table
# Path: s3a://kronodroid/dev/iceberg/kronodroid/dim_malware_families
kronodroid_family_source = SparkSource(
    name="kronodroid_family_source",
    table=f"{LAKEFS_CATALOG}.{LAKEFS_DATABASE}.dim_malware_families",
    timestamp_field="_dbt_loaded_at",
    description="Family statistics from LakeFS-tracked Iceberg table",
)

# Push source for real-time feature updates
kronodroid_push_source = PushSource(
    name="kronodroid_push_source",
    batch_source=kronodroid_training_source,
)


# Main feature view for malware sample features
malware_sample_features = FeatureView(
    name="malware_sample_features",
    entities=[malware_sample],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        # Target and metadata
        Field(name="label", dtype=Int64, description="1=malware, 0=benign"),
        Field(name="malware_family", dtype=String),
        Field(name="first_seen_year", dtype=Int64),
        Field(name="data_source", dtype=String, description="emulator or real_device"),
        Field(name="dataset_split", dtype=String, description="train/validation/test"),
        # Syscall features (dynamic features)
        Field(name="syscall_1_normalized", dtype=Float32),
        Field(name="syscall_2_normalized", dtype=Float32),
        Field(name="syscall_3_normalized", dtype=Float32),
        Field(name="syscall_4_normalized", dtype=Float32),
        Field(name="syscall_5_normalized", dtype=Float32),
        Field(name="syscall_6_normalized", dtype=Float32),
        Field(name="syscall_7_normalized", dtype=Float32),
        Field(name="syscall_8_normalized", dtype=Float32),
        Field(name="syscall_9_normalized", dtype=Float32),
        Field(name="syscall_10_normalized", dtype=Float32),
        Field(name="syscall_11_normalized", dtype=Float32),
        Field(name="syscall_12_normalized", dtype=Float32),
        Field(name="syscall_13_normalized", dtype=Float32),
        Field(name="syscall_14_normalized", dtype=Float32),
        Field(name="syscall_15_normalized", dtype=Float32),
        Field(name="syscall_16_normalized", dtype=Float32),
        Field(name="syscall_17_normalized", dtype=Float32),
        Field(name="syscall_18_normalized", dtype=Float32),
        Field(name="syscall_19_normalized", dtype=Float32),
        Field(name="syscall_20_normalized", dtype=Float32),
        # Aggregated features
        Field(name="syscall_total", dtype=Float64, description="Sum of all syscalls"),
        Field(name="syscall_mean", dtype=Float64, description="Mean of syscalls"),
    ],
    source=kronodroid_training_source,
    online=True,
    tags={"team": "dfp", "dataset": "kronodroid"},
)


# Family statistics feature view
malware_family_features = FeatureView(
    name="malware_family_features",
    entities=[malware_family],
    ttl=timedelta(days=30),
    schema=[
        Field(name="family_name", dtype=String),
        Field(name="is_malware_family", dtype=Int64),
        Field(name="total_samples", dtype=Int64),
        Field(name="unique_samples", dtype=Int64),
        Field(name="emulator_count", dtype=Int64),
        Field(name="real_device_count", dtype=Int64),
        Field(name="earliest_year", dtype=Int64),
        Field(name="latest_year", dtype=Int64),
        Field(name="year_span", dtype=Int64),
    ],
    source=kronodroid_family_source,
    online=True,
    tags={"team": "dfp", "dataset": "kronodroid"},
)


# On-demand feature view for derived features
@on_demand_feature_view(
    sources=[malware_sample_features],
    schema=[
        Field(name="syscall_variance", dtype=Float64),
        Field(name="is_high_activity", dtype=Int64),
    ],
)
def malware_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from base malware sample features."""
    import numpy as np

    # Get syscall columns that exist in the input DataFrame
    syscall_cols = [
        f"syscall_{i}_normalized"
        for i in range(1, 21)
        if f"syscall_{i}_normalized" in inputs.columns
    ]

    if syscall_cols:
        syscall_data = inputs[syscall_cols].values
        variance = np.var(syscall_data, axis=1)
        total = np.sum(syscall_data, axis=1)
        is_high = (total > 100).astype(int)
    else:
        variance = np.zeros(len(inputs))
        is_high = np.zeros(len(inputs), dtype=int)

    return pd.DataFrame({
        "syscall_variance": variance,
        "is_high_activity": is_high,
    })


# Batch feature view for offline training
malware_batch_features = BatchFeatureView(
    name="malware_batch_features",
    entities=[malware_sample],
    ttl=timedelta(days=365),
    schema=[
        Field(name="label", dtype=Int64),
        Field(name="syscall_total", dtype=Float64),
        Field(name="syscall_mean", dtype=Float64),
    ],
    source=kronodroid_training_source,
    tags={"team": "dfp", "dataset": "kronodroid", "usage": "training"},
)


# Autoencoder-friendly numeric feature view (no labels/strings)
kronodroid_autoencoder_features = FeatureView(
    name="kronodroid_autoencoder_features",
    entities=[malware_sample],
    ttl=timedelta(days=365),
    schema=[
        Field(name=f"syscall_{i}_normalized", dtype=Float32)
        for i in range(1, 21)
    ]
    + [
        Field(name="syscall_total", dtype=Float64),
        Field(name="syscall_mean", dtype=Float64),
    ],
    source=kronodroid_training_source,
    online=False,
    tags={"team": "dfp", "dataset": "kronodroid", "usage": "autoencoder"},
)
