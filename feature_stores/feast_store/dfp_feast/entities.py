"""Feast entity definitions."""
from feast import Entity

user = Entity(name="user", join_keys=["user_id"])
device = Entity(name="device", join_keys=["device_id"])
