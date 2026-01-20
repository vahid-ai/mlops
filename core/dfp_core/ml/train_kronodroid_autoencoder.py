"""Train a PyTorch Lightning autoencoder on Kronodroid via Feast + LakeFS + MLflow."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from core.dfp_core.ml.autoencoder_lightning import AutoencoderConfig, make_lightning_module
from core.dfp_core.ml.kronodroid_feast_dataset import (
    KronodroidDatasetSpec,
    load_split_as_pandas,
    pandas_to_tensor_dataset,
    resolve_feature_columns,
    try_get_iceberg_snapshot_id,
)
from core.dfp_core.feast_mlflow_utils import sha256_files


def train_and_register(
    *,
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
    feature_names: list[str],
    feast_definitions_paths: list[str] | None = None,
    max_rows_per_split: int | None = None,
    latent_dim: int = 16,
    hidden_dims: tuple[int, ...] = (128, 64),
    batch_size: int = 512,
    max_epochs: int = 10,
    seed: int = 1337,
) -> dict[str, Any]:
    """Train, validate, test, and register a Kronodroid autoencoder."""
    import json

    try:
        import lightning.pytorch as pl
    except Exception:  # pragma: no cover - optional dependency at dev time
        import pytorch_lightning as pl  # type: ignore[no-redef]

    import mlflow
    from torch.utils.data import DataLoader

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    pl.seed_everything(seed, workers=True)

    feature_refs = [f"{feast_feature_view}:{name}" for name in feature_names]
    input_dim = len(feature_names)

    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )

    # Dataset specs (stable ids used for lineage tags)
    train_spec = KronodroidDatasetSpec(
        lakefs_repository=lakefs_repository,
        lakefs_ref=lakefs_ref,
        iceberg_catalog=iceberg_catalog,
        iceberg_database=iceberg_database,
        source_table=source_table,
        split="train",
        feast_project=feast_project,
        feast_feature_view=feast_feature_view,
        feature_names=tuple(feature_names),
        max_rows=max_rows_per_split,
    )
    val_spec = replace(train_spec, split="validation")
    test_spec = replace(train_spec, split="test")

    snapshot_id = try_get_iceberg_snapshot_id(
        feature_store_yaml_path=feature_store_yaml_path,
        iceberg_catalog=iceberg_catalog,
        iceberg_database=iceberg_database,
        table=source_table,
    )
    feast_defs_sha = sha256_files(feast_definitions_paths) if feast_definitions_paths else None

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlf_logger = pl.loggers.MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_id=run_id,
        )

        mlflow.log_params(
            {
                "data.lakefs_repository": lakefs_repository,
                "data.lakefs_ref": lakefs_ref,
                "data.iceberg_catalog": iceberg_catalog,
                "data.iceberg_database": iceberg_database,
                "data.source_table": source_table,
                "data.iceberg_snapshot_id": snapshot_id or "",
                "feast.project": feast_project,
                "feast.feature_view": feast_feature_view,
                "feast.defs_sha256": feast_defs_sha or "",
                "feast.features": json.dumps(feature_names),
                "dataset.train_id": train_spec.stable_id(),
                "dataset.val_id": val_spec.stable_id(),
                "dataset.test_id": test_spec.stable_id(),
                "model.input_dim": input_dim,
                "model.latent_dim": latent_dim,
                "model.hidden_dims": json.dumps(list(hidden_dims)),
                "train.batch_size": batch_size,
                "train.max_epochs": max_epochs,
                "train.seed": seed,
                "train.max_rows_per_split": max_rows_per_split or 0,
            }
        )

        train_df = load_split_as_pandas(
            feature_store_yaml_path=feature_store_yaml_path,
            lakefs_repository=lakefs_repository,
            lakefs_ref=lakefs_ref,
            iceberg_catalog=iceberg_catalog,
            iceberg_database=iceberg_database,
            source_table=source_table,
            split="train",
            feast_feature_refs=feature_refs,
            max_rows=max_rows_per_split,
            ensure_registry=True,
        )
        val_df = load_split_as_pandas(
            feature_store_yaml_path=feature_store_yaml_path,
            lakefs_repository=lakefs_repository,
            lakefs_ref=lakefs_ref,
            iceberg_catalog=iceberg_catalog,
            iceberg_database=iceberg_database,
            source_table=source_table,
            split="validation",
            feast_feature_refs=feature_refs,
            max_rows=max_rows_per_split,
        )
        test_df = load_split_as_pandas(
            feature_store_yaml_path=feature_store_yaml_path,
            lakefs_repository=lakefs_repository,
            lakefs_ref=lakefs_ref,
            iceberg_catalog=iceberg_catalog,
            iceberg_database=iceberg_database,
            source_table=source_table,
            split="test",
            feast_feature_refs=feature_refs,
            max_rows=max_rows_per_split,
        )

        mlflow.log_metrics(
            {
                "data.train_rows": float(len(train_df)),
                "data.val_rows": float(len(val_df)),
                "data.test_rows": float(len(test_df)),
            }
        )

        resolved_cols = resolve_feature_columns(train_df, feast_feature_view, feature_names)
        train_ds = pandas_to_tensor_dataset(train_df, resolved_cols)
        val_ds = pandas_to_tensor_dataset(val_df, resolved_cols)
        test_ds = pandas_to_tensor_dataset(test_df, resolved_cols)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        module = make_lightning_module(cfg)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            enable_checkpointing=False,
            logger=mlf_logger,
        )

        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(module, dataloaders=test_loader)

        # Register model with lineage baked into the registered version tags.
        mlflow.pytorch.log_model(pytorch_model=module, artifact_path="model")

        model_uri = f"runs:/{run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name=mlflow_model_name)

        client = mlflow.tracking.MlflowClient()
        version = registered.version
        client.set_model_version_tag(mlflow_model_name, version, "data.lakefs_repository", lakefs_repository)
        client.set_model_version_tag(mlflow_model_name, version, "data.lakefs_ref", lakefs_ref)
        client.set_model_version_tag(mlflow_model_name, version, "data.iceberg_catalog", iceberg_catalog)
        client.set_model_version_tag(mlflow_model_name, version, "data.iceberg_database", iceberg_database)
        client.set_model_version_tag(mlflow_model_name, version, "data.source_table", source_table)
        if snapshot_id is not None:
            client.set_model_version_tag(mlflow_model_name, version, "data.iceberg_snapshot_id", snapshot_id)
        client.set_model_version_tag(mlflow_model_name, version, "feast.feature_view", feast_feature_view)
        if feast_defs_sha is not None:
            client.set_model_version_tag(mlflow_model_name, version, "feast.defs_sha256", feast_defs_sha)
        client.set_model_version_tag(mlflow_model_name, version, "dataset.train_id", train_spec.stable_id())
        client.set_model_version_tag(mlflow_model_name, version, "dataset.val_id", val_spec.stable_id())
        client.set_model_version_tag(mlflow_model_name, version, "dataset.test_id", test_spec.stable_id())

        mlflow.log_dict(asdict(train_spec), "lineage/train_dataset_spec.json")
        mlflow.log_dict(asdict(val_spec), "lineage/val_dataset_spec.json")
        mlflow.log_dict(asdict(test_spec), "lineage/test_dataset_spec.json")

        return {
            "run_id": run_id,
            "model_name": mlflow_model_name,
            "model_version": str(version),
        }
