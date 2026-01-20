"""Feast project configuration for DFP."""

from .entities import device, malware_family, malware_sample, user
from .feature_views import user_daily_features
from .kronodroid_features import (
    kronodroid_family_source,
    kronodroid_training_source,
    kronodroid_autoencoder_features,
    malware_batch_features,
    malware_derived_features,
    malware_family_features,
    malware_sample_features,
)

__all__ = [
    # Entities
    "user",
    "device",
    "malware_sample",
    "malware_family",
    # Feature views
    "user_daily_features",
    "malware_sample_features",
    "malware_family_features",
    "malware_batch_features",
    "malware_derived_features",
    "kronodroid_autoencoder_features",
    # Sources
    "kronodroid_training_source",
    "kronodroid_family_source",
]
