"""Feast entity definitions for the DFP project."""

from feast import Entity, ValueType

# Kronodroid entity for Android malware detection
malware_sample = Entity(
    name="malware_sample",
    join_keys=["sample_id"],
    value_type=ValueType.STRING,
    description="Android APK sample identified by unique dlt-generated ID",
)
