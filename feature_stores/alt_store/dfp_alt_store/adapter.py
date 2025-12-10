"""Implements FeatureStore protocol for an alternative backend."""
from core.dfp_core.features.base import FeatureStore

class AltStore(FeatureStore):
    def get_historical_features(self, entity_df, feature_refs):
        return []

    def materialize(self, start_date: str, end_date: str) -> None:
        print(f"Materializing alt store from {start_date} to {end_date}")
