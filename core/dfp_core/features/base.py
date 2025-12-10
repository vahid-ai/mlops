from typing import Protocol, Any

class ExecutionEngine(Protocol):
    def run_query(self, query: str) -> Any:
        ...

class FeatureStore(Protocol):
    def get_historical_features(self, entity_df: Any, feature_refs: list[str]) -> Any:
        ...

    def materialize(self, start_date: str, end_date: str) -> None:
        ...
