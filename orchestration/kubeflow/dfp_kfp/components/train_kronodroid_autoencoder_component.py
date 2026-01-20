"""Component: train + validate + test Kronodroid autoencoder and register in MLflow."""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "mlflow>=2.0",
        "torch>=2.0",
        "lightning>=2.0",
        "feast[spark]>=0.30",
        "pyarrow>=14.0",
    ],
)
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
    feast_definitions_paths_json: str,
    max_rows_per_split: int,
    latent_dim: int,
    hidden_dims_json: str,
    batch_size: int,
    max_epochs: int,
    seed: int,
) -> str:
    """Train, validate, test, and register a Kronodroid autoencoder model.

    Returns:
        JSON string with run_id, model_name, model_version
    """
    import json
    import os

    feature_names = json.loads(feature_names_json)
    feast_def_paths = json.loads(feast_definitions_paths_json) if feast_definitions_paths_json else []
    hidden_dims_list = json.loads(hidden_dims_json)

    # Import training module
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
        max_rows_per_split=(max_rows_per_split if max_rows_per_split > 0 else None),
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims_list),
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
    )

    # Return JSON string with results
    return json.dumps({
        "run_id": result["run_id"],
        "model_name": result["model_name"],
        "model_version": result["model_version"],
    })
