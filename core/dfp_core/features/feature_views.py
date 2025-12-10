"""Logical feature view definitions for Feast and alt stores."""
from dataclasses import dataclass

@dataclass
class FeatureViewSpec:
    name: str
    entities: list[str]
    features: list[str]
    batch_source: str
