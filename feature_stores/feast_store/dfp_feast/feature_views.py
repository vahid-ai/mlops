"""Feast feature view definitions for Kronodroid malware detection.

These feature views reference Iceberg tables written by the Spark pipeline
to the LakeFS Iceberg catalog. The tables are versioned via LakeFS branches.

Tables:
- kronodroid.fct_malware_samples: Malware samples with features
- kronodroid.fct_training_dataset: Training dataset with splits
- kronodroid.dim_malware_families: Dimension table for malware families
"""

import os
from datetime import timedelta

from feast import Entity, Field, FeatureView, FileSource
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.types import Float32, Int64, String, Bool, UnixTimestamp


# -----------------------------------------------------------------------------
# Environment-based configuration
# -----------------------------------------------------------------------------

LAKEFS_REPOSITORY = os.getenv("LAKEFS_REPOSITORY", "kronodroid")
LAKEFS_BRANCH = os.getenv("LAKEFS_BRANCH", "main")
ICEBERG_CATALOG = os.getenv("ICEBERG_CATALOG_NAME", "lakefs")


# -----------------------------------------------------------------------------
# Entities
# -----------------------------------------------------------------------------

sample_entity = Entity(
    name="sample",
    join_keys=["sample_id"],
    description="A malware/benign sample identified by sample_id",
)

malware_family_entity = Entity(
    name="malware_family",
    join_keys=["family_id"],
    description="A malware family or data source category",
)


# -----------------------------------------------------------------------------
# Data Sources (Iceberg tables via LakeFS)
# -----------------------------------------------------------------------------

# Iceberg table paths follow: <catalog>.<database>.<table>
# With LakeFS REST catalog, the branch is encoded in the catalog URI

malware_samples_source = SparkSource(
    name="malware_samples_source",
    description="Iceberg table with malware sample features",
    table=f"{ICEBERG_CATALOG}.kronodroid.fct_malware_samples",
    timestamp_field="event_timestamp",
)

training_dataset_source = SparkSource(
    name="training_dataset_source",
    description="Iceberg table with training dataset splits",
    table=f"{ICEBERG_CATALOG}.kronodroid.fct_training_dataset",
    timestamp_field="event_timestamp",
)

malware_families_source = SparkSource(
    name="malware_families_source",
    description="Iceberg table with malware family statistics",
    table=f"{ICEBERG_CATALOG}.kronodroid.dim_malware_families",
    timestamp_field="_dbt_loaded_at",
)


# -----------------------------------------------------------------------------
# Feature Views
# -----------------------------------------------------------------------------

malware_sample_features = FeatureView(
    name="malware_sample_features",
    entities=[sample_entity],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        Field(name="data_source", dtype=String),
        Field(name="label", dtype=Int64),
        Field(name="family", dtype=String),
        Field(name="year", dtype=Int64),
        Field(name="sha256", dtype=String),
        # The actual feature columns are dynamic (289 syscalls + 200 static)
        # In production, list them explicitly or use a registry
    ],
    source=malware_samples_source,
    online=True,
    description="Features for malware samples from Kronodroid dataset",
    tags={
        "team": "ml",
        "pipeline": "kronodroid-iceberg",
        "lakefs_repository": LAKEFS_REPOSITORY,
    },
)

training_dataset_features = FeatureView(
    name="training_dataset_features",
    entities=[sample_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="data_source", dtype=String),
        Field(name="label", dtype=Int64),
        Field(name="dataset_split", dtype=String),  # train/validation/test
    ],
    source=training_dataset_source,
    online=False,  # Training data doesn't need online serving
    description="Training dataset with deterministic splits",
    tags={
        "team": "ml",
        "pipeline": "kronodroid-iceberg",
        "use_case": "training",
    },
)

malware_family_features = FeatureView(
    name="malware_family_features",
    entities=[malware_family_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="family_name", dtype=String),
        Field(name="is_data_source", dtype=Bool),
        Field(name="total_samples", dtype=Int64),
        Field(name="unique_samples", dtype=Int64),
        Field(name="emulator_count", dtype=Int64),
        Field(name="real_device_count", dtype=Int64),
        Field(name="earliest_year", dtype=Int64),
        Field(name="latest_year", dtype=Int64),
        Field(name="year_span", dtype=Int64),
    ],
    source=malware_families_source,
    online=True,
    description="Aggregated statistics for malware families",
    tags={
        "team": "ml",
        "pipeline": "kronodroid-iceberg",
    },
)


# -----------------------------------------------------------------------------
# Legacy compatibility (users_source from original file)
# -----------------------------------------------------------------------------

# Keeping the original placeholder for backwards compatibility
# Remove this once the codebase is fully migrated to Iceberg sources

users_source = FileSource(
    name="users_legacy",
    path="",  # Placeholder - not used
    timestamp_field="event_timestamp",
)

user_daily_features = FeatureView(
    name="user_daily_features",
    entities=[],
    ttl=None,
    schema=[Field(name="score", dtype=Float32)],
    source=users_source,
    description="Legacy placeholder - use malware_sample_features instead",
)
