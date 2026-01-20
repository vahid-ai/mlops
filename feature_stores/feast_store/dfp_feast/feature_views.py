"""Feast feature view definitions.

This module contains legacy/placeholder feature views. The main Kronodroid
feature views are defined in kronodroid_features.py using FileSource.

For Iceberg-backed feature views with SparkSource, the offline_store in
feature_store.yaml must be configured for Spark. See the Spark offline
store documentation for configuration details.
"""

from datetime import timedelta

from feast import Entity, Field, FeatureView, FileSource
from feast.types import Float32


# -----------------------------------------------------------------------------
# Legacy placeholder (for backwards compatibility)
# -----------------------------------------------------------------------------

# Placeholder source - not used but required for the legacy feature view
# Path must be non-empty for Feast validation even if not actively used
users_source = FileSource(
    name="users_legacy",
    path="data/placeholder/training_dataset.parquet",
    timestamp_field="event_timestamp",
)

# Legacy placeholder feature view
# This is kept for backwards compatibility - use malware_sample_features instead
user_daily_features = FeatureView(
    name="user_daily_features",
    entities=[],
    ttl=None,
    schema=[Field(name="score", dtype=Float32)],
    source=users_source,
    description="Legacy placeholder - use malware_sample_features instead",
)
