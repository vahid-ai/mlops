"""Feast feature view definitions."""
from feast import Field, FeatureView
from feast.types import Float32
from feast.data_source import DataSource

users_source = DataSource(name="users", path="iceberg://users_features")

user_daily_features = FeatureView(
    name="user_daily_features",
    entities=["user"],
    ttl=None,
    schema=[Field(name="score", dtype=Float32)],
    source=users_source,
)
