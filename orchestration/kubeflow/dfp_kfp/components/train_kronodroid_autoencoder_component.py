"""Component: train + validate + test Kronodroid autoencoder and register in MLflow."""

from __future__ import annotations

from typing import NamedTuple

from kfp import dsl


DEFAULT_TRAIN_IMAGE = "dfp/kronodroid-train:latest"


@dsl.component(base_image=DEFAULT_TRAIN_IMAGE)
def train_kronodroid_autoencoder_op(
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_model_name: str,
    feature_store_yaml_path: str,
    lakefs_repository: str,
    lakefs_ref: str,
    iceberg_catalog: str,
    iceberg_database: str,
    source_table: str,
    feast_project: str,
    feast_feature_view: str,
    feature_names_json: str,
    feast_definitions_paths_json: str = "[]",
    max_rows_per_split: int = 0,
    latent_dim: int = 16,
    hidden_dims_json: str = "[128, 64]",
    batch_size: int = 512,
    max_epochs: int = 10,
    seed: int = 1337,
) -> NamedTuple("TrainOutput", [("run_id", str), ("model_name", str), ("model_version", str)]):
    import json
    from collections import namedtuple

    Output = namedtuple("TrainOutput", ["run_id", "model_name", "model_version"])

    feature_names = json.loads(feature_names_json)
    feast_def_paths = json.loads(feast_definitions_paths_json)
    hidden_dims_list = json.loads(hidden_dims_json)

    from core.dfp_core.ml.train_kronodroid_autoencoder import train_and_register

    result = train_and_register(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_model_name=mlflow_model_name,
        feature_store_yaml_path=feature_store_yaml_path,
        lakefs_repository=lakefs_repository,
        lakefs_ref=lakefs_ref,
        iceberg_catalog=iceberg_catalog,
        iceberg_database=iceberg_database,
        source_table=source_table,
        feast_project=feast_project,
        feast_feature_view=feast_feature_view,
        feature_names=list(feature_names),
        feast_definitions_paths=list(feast_def_paths) if feast_def_paths else None,
        max_rows_per_split=(max_rows_per_split or None),
        latent_dim=int(latent_dim),
        hidden_dims=tuple(int(x) for x in hidden_dims_list),
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        seed=int(seed),
    )

    return Output(result["run_id"], result["model_name"], result["model_version"])

