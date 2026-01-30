"""Feast feature views for Kronodroid Android malware detection.

These feature views map to the actual Iceberg table schema created by the
Spark transformation job. The table contains Android permission counts,
syscall counts, and malware labels from the Kronodroid dataset.
"""

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
from feast.types import Float64, Int64, String

from .entities import malware_family, malware_sample

# LakeFS Iceberg catalog and table configuration
LAKEFS_CATALOG = os.environ.get("LAKEFS_CATALOG", "lakefs")
LAKEFS_DATABASE = os.environ.get("LAKEFS_DATABASE", "kronodroid")

# Spark source for training dataset from LakeFS-tracked Iceberg table
# Timestamp field is 'feature_timestamp' (timestamptz type)
kronodroid_training_source = SparkSource(
    name="kronodroid_training_source",
    table=f"{LAKEFS_CATALOG}.{LAKEFS_DATABASE}.fct_training_dataset",
    timestamp_field="feature_timestamp",
    description="Training dataset from LakeFS-tracked Iceberg table",
)

# Push source for real-time feature updates
kronodroid_push_source = PushSource(
    name="kronodroid_push_source",
    batch_source=kronodroid_training_source,
)


# Main feature view for malware sample features
# Uses actual column names from the Iceberg table schema
malware_sample_features = FeatureView(
    name="malware_sample_features",
    entities=[malware_sample],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        # Target and metadata (actual column names from schema)
        Field(name="malware", dtype=Int64, description="1=malware, 0=benign"),
        Field(name="mal_family", dtype=String, description="Malware family name"),
        Field(name="data_source", dtype=String, description="emulator or real_device"),
        Field(name="dataset_split", dtype=String, description="train/validation/test"),
        # Key syscall features (actual syscall column names)
        Field(name="read", dtype=Int64, description="read syscall count"),
        Field(name="write", dtype=Int64, description="write syscall count"),
        Field(name="open", dtype=Int64, description="open syscall count"),
        Field(name="close", dtype=Int64, description="close syscall count"),
        Field(name="clone", dtype=Int64, description="clone syscall count"),
        Field(name="socket", dtype=Int64, description="socket syscall count"),
        Field(name="connect", dtype=Int64, description="connect syscall count"),
        Field(name="sendto", dtype=Int64, description="sendto syscall count"),
        Field(name="recvfrom", dtype=Int64, description="recvfrom syscall count"),
        Field(name="mmap2", dtype=Int64, description="mmap2 syscall count"),
        Field(name="mprotect", dtype=Int64, description="mprotect syscall count"),
        Field(name="ioctl", dtype=Int64, description="ioctl syscall count"),
        Field(name="futex", dtype=Int64, description="futex syscall count"),
        Field(name="kill", dtype=Int64, description="kill syscall count"),
        Field(name="bind", dtype=Int64, description="bind syscall count"),
        # Aggregated/derived columns
        Field(name="nr_syscalls", dtype=Int64, description="Total number of syscalls"),
        Field(name="nr_permissions", dtype=Int64, description="Number of permissions requested"),
        Field(name="activities", dtype=Float64, description="Number of activities"),
        Field(name="detection_ratio", dtype=Float64, description="VirusTotal detection ratio"),
        # Permission features
        Field(name="internet", dtype=Int64, description="INTERNET permission"),
        Field(name="read_external_storage", dtype=Int64, description="READ_EXTERNAL_STORAGE permission"),
        Field(name="write_external_storage", dtype=Int64, description="WRITE_EXTERNAL_STORAGE permission"),
        Field(name="read_phone_state", dtype=Int64, description="READ_PHONE_STATE permission"),
        Field(name="access_network_state", dtype=Int64, description="ACCESS_NETWORK_STATE permission"),
        Field(name="wake_lock", dtype=Int64, description="WAKE_LOCK permission"),
        Field(name="receive_boot_completed", dtype=Int64, description="RECEIVE_BOOT_COMPLETED permission"),
        Field(name="camera", dtype=Int64, description="CAMERA permission"),
        Field(name="record_audio", dtype=Int64, description="RECORD_AUDIO permission"),
        Field(name="send_sms", dtype=Int64, description="SEND_SMS permission"),
    ],
    source=kronodroid_training_source,
    online=True,
    tags={"team": "dfp", "dataset": "kronodroid"},
)


# On-demand feature view for derived features
@on_demand_feature_view(
    sources=[malware_sample_features],
    schema=[
        Field(name="syscall_activity_score", dtype=Float64),
        Field(name="is_high_syscall_activity", dtype=Int64),
        Field(name="network_activity_score", dtype=Float64),
        Field(name="permission_risk_score", dtype=Float64),
    ],
)
def malware_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from base malware sample features."""
    import numpy as np

    # Syscall activity score based on key syscalls
    syscall_cols = ["read", "write", "open", "close", "clone", "socket", "connect"]
    existing_syscall_cols = [c for c in syscall_cols if c in inputs.columns]

    if existing_syscall_cols:
        syscall_data = inputs[existing_syscall_cols].fillna(0).values
        syscall_activity = np.sum(syscall_data, axis=1)
        is_high_activity = (syscall_activity > 1000).astype(int)
    else:
        syscall_activity = np.zeros(len(inputs))
        is_high_activity = np.zeros(len(inputs), dtype=int)

    # Network activity score
    network_cols = ["socket", "connect", "sendto", "recvfrom", "bind"]
    existing_network_cols = [c for c in network_cols if c in inputs.columns]

    if existing_network_cols:
        network_activity = inputs[existing_network_cols].fillna(0).sum(axis=1).values
    else:
        network_activity = np.zeros(len(inputs))

    # Permission risk score (dangerous permissions)
    risky_perm_cols = ["send_sms", "record_audio", "camera", "read_phone_state"]
    existing_perm_cols = [c for c in risky_perm_cols if c in inputs.columns]

    if existing_perm_cols:
        perm_risk = inputs[existing_perm_cols].fillna(0).sum(axis=1).values
    else:
        perm_risk = np.zeros(len(inputs))

    return pd.DataFrame({
        "syscall_activity_score": syscall_activity.astype(float),
        "is_high_syscall_activity": is_high_activity,
        "network_activity_score": network_activity.astype(float),
        "permission_risk_score": perm_risk.astype(float),
    })


# Batch feature view for offline training
malware_batch_features = BatchFeatureView(
    name="malware_batch_features",
    entities=[malware_sample],
    ttl=timedelta(days=365),
    schema=[
        Field(name="malware", dtype=Int64, description="Label: 1=malware, 0=benign"),
        Field(name="nr_syscalls", dtype=Int64, description="Total syscall count"),
        Field(name="nr_permissions", dtype=Int64, description="Number of permissions"),
        Field(name="detection_ratio", dtype=Float64, description="VirusTotal detection ratio"),
    ],
    source=kronodroid_training_source,
    tags={"team": "dfp", "dataset": "kronodroid", "usage": "training"},
)
