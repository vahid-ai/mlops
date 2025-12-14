"""Feast entity definitions."""

from feast import Entity

# Original DFP entities
user = Entity(name="user", join_keys=["user_id"])
device = Entity(name="device", join_keys=["device_id"])

# Kronodroid entities for Android malware detection
malware_sample = Entity(
    name="malware_sample",
    join_keys=["sample_id"],
    description="Android APK sample identified by unique ID",
)

malware_family = Entity(
    name="malware_family",
    join_keys=["family_id"],
    description="Android malware family identifier",
)
