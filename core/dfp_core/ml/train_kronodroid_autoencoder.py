"""Training orchestration for Kronodroid autoencoder.

Integrates:
- Feast for feature retrieval
- PyTorch Lightning for training
- MLflow for experiment tracking and model registry
- LakeFS commit hashes for data lineage
- Iceberg snapshot IDs for data lineage
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlflow
    import mlflow.pytorch
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import MLFlowLogger
    import requests
except ImportError:  # pragma: no cover - tooling only
    mlflow = None
    L = None
    MLFlowLogger = None
    requests = None

from core.dfp_core.ml.lightning_modules import LightningAutoencoder
from core.dfp_core.ml.datasets import (
    create_dataloaders_from_dataframes,
    load_splits_from_iceberg,
)


# Default feature names (22 features)
DEFAULT_FEATURE_NAMES = [
    f"syscall_{i}_normalized" for i in range(1, 21)
] + ["syscall_total", "syscall_mean"]


def get_lakefs_commit_info(
    lakefs_endpoint: str,
    repository: str,
    ref: str,
) -> Dict[str, str]:
    """Get LakeFS commit information for data lineage.

    Args:
        lakefs_endpoint: LakeFS API endpoint URL
        repository: LakeFS repository name
        ref: LakeFS branch or commit reference

    Returns:
        Dictionary with commit_id, repository, and ref
    """
    if requests is None:
        return {"lakefs_ref": ref, "lakefs_repository": repository}

    access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")

    api_base = lakefs_endpoint.rstrip("/")
    auth = (access_key, secret_key) if access_key else None

    # Get commit info for the ref
    url = f"{api_base}/api/v1/repositories/{repository}/refs/{ref}"
    try:
        resp = requests.get(url, auth=auth, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "lakefs_commit_id": data.get("commit_id", "unknown"),
                "lakefs_repository": repository,
                "lakefs_ref": ref,
            }
    except Exception:
        pass

    return {"lakefs_ref": ref, "lakefs_repository": repository}


def get_spark_session(
    catalog: str = "lakefs",
    lakefs_endpoint: str = "http://lakefs:8000",
):
    """Get or create a SparkSession configured for Iceberg/LakeFS.

    Args:
        catalog: Iceberg catalog name
        lakefs_endpoint: LakeFS endpoint URL

    Returns:
        Configured SparkSession
    """
    try:
        from engines.spark_engine.dfp_spark.session import get_spark_session
        return get_spark_session()
    except ImportError:
        # Fallback: create basic Spark session
        from pyspark.sql import SparkSession

        lakefs_access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
        lakefs_secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")

        return (
            SparkSession.builder
            .appName("kronodroid-autoencoder-training")
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
            .config(f"spark.sql.catalog.{catalog}.type", "rest")
            .config(f"spark.sql.catalog.{catalog}.uri", f"{lakefs_endpoint}/api/v1/iceberg")
            .config(f"spark.sql.catalog.{catalog}.credential", f"{lakefs_access_key}:{lakefs_secret_key}")
            .getOrCreate()
        )


def train_and_register(
    # MLflow config
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_model_name: str,
    # Data config
    lakefs_repository: str = "kronodroid",
    lakefs_ref: str = "main",
    iceberg_catalog: str = "lakefs",
    iceberg_database: str = "kronodroid",
    source_table: str = "fct_training_dataset",
    # Feast/Feature config
    feast_project: str = "dfp",
    feast_feature_view: str = "malware_sample_features",
    feature_names: Optional[List[str]] = None,
    # Model config
    latent_dim: int = 16,
    hidden_dims: Tuple[int, ...] = (128, 64),
    # Training config
    batch_size: int = 512,
    max_epochs: int = 10,
    learning_rate: float = 1e-3,
    seed: int = 1337,
    num_workers: int = 4,
    max_rows_per_split: Optional[int] = None,
    # LakeFS config for lineage
    lakefs_endpoint: str = "http://lakefs:8000",
) -> Dict[str, Any]:
    """Train autoencoder and register in MLflow.

    This function:
    1. Loads training data from Iceberg tables
    2. Trains a PyTorch Lightning autoencoder with early stopping
    3. Logs all metrics, hyperparameters, and data lineage to MLflow
    4. Registers the trained model in MLflow Model Registry

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_model_name: Name for model in MLflow registry
        lakefs_repository: LakeFS repository name
        lakefs_ref: LakeFS branch or commit reference
        iceberg_catalog: Iceberg catalog name
        iceberg_database: Iceberg database name
        source_table: Source Iceberg table name
        feast_project: Feast project name (for lineage)
        feast_feature_view: Feast feature view name (for lineage)
        feature_names: List of feature column names
        latent_dim: Autoencoder latent dimension
        hidden_dims: Tuple of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Optimizer learning rate
        seed: Random seed for reproducibility
        num_workers: DataLoader workers
        max_rows_per_split: Optional row limit per split (for testing)
        lakefs_endpoint: LakeFS API endpoint

    Returns:
        Dictionary with run_id, model_name, model_version, metrics, and lineage info
    """
    if mlflow is None or L is None:
        raise RuntimeError("mlflow and lightning must be installed")

    feature_names = feature_names or DEFAULT_FEATURE_NAMES

    # Set seeds for reproducibility
    L.seed_everything(seed)

    # Setup MLflow
    print(f"Setting MLflow tracking URI: {mlflow_tracking_uri} with experiment {mlflow_experiment_name}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # Get LakeFS lineage info
    print(f"Getting LakeFS lineage info for {lakefs_repository}@{lakefs_ref}")
    lakefs_info = get_lakefs_commit_info(lakefs_endpoint, lakefs_repository, lakefs_ref)

    # Get Spark session and load data
    print(f"Getting Spark session for {iceberg_catalog}")
    spark = get_spark_session(catalog=iceberg_catalog, lakefs_endpoint=lakefs_endpoint)

    splits, data_lineage = load_splits_from_iceberg(
        spark_session=spark,
        catalog=iceberg_catalog,
        database=iceberg_database,
        table=source_table,
        feature_columns=feature_names,
        max_rows_per_split=max_rows_per_split,
    )

    # Create DataLoaders
    print(f"Creating DataLoaders for {feature_names}")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders_from_dataframes(
        train_df=splits["train"],
        val_df=splits["validation"],
        test_df=splits["test"],
        feature_columns=feature_names,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=True,
    )

    # Create model
    print(f"Creating model with input dimension {len(feature_names)}")
    input_dim = len(feature_names)
    model = LightningAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
    )

    # Start MLflow run
    print(f"Starting MLflow run")
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log data lineage
        mlflow.log_params({
            **lakefs_info,
            "iceberg_table": data_lineage.get("iceberg_table"),
            "iceberg_snapshot_id": data_lineage.get("iceberg_snapshot_id"),
            "feast_project": feast_project,
            "feast_feature_view": feast_feature_view,
            "feature_names": json.dumps(feature_names),
            "train_samples": data_lineage.get("train_samples"),
            "validation_samples": data_lineage.get("validation_samples"),
            "test_samples": data_lineage.get("test_samples"),
        })

        # Log model hyperparameters
        mlflow.log_params({
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dims": str(hidden_dims),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        })

        # Setup Lightning trainer with MLflow logger
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_id=run_id,
        )

        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-{val_loss:.4f}",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
            ),
        ]

        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=mlflow_logger,
            callbacks=callbacks,
            deterministic=True,
            enable_progress_bar=True,
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Test
        test_results = trainer.test(model, test_loader)
        test_loss = test_results[0]["test_loss"]

        # Log test metrics
        mlflow.log_metric("test_loss", test_loss)

        # Log normalization params as artifact
        if norm_params:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({
                    "mean": norm_params["mean"].tolist(),
                    "std": norm_params["std"].tolist(),
                    "feature_names": feature_names,
                }, f)
                mlflow.log_artifact(f.name, "normalization")
                os.unlink(f.name)

        # Log model to MLflow and register
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=mlflow_model_name,
        )

        # Get model version
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{mlflow_model_name}'")
        latest_version = max([int(v.version) for v in versions]) if versions else 1

        return {
            "run_id": run_id,
            "model_name": mlflow_model_name,
            "model_version": str(latest_version),
            "test_loss": float(test_loss),
            "train_samples": data_lineage.get("train_samples"),
            "validation_samples": data_lineage.get("validation_samples"),
            "test_samples": data_lineage.get("test_samples"),
            "lakefs_commit_id": lakefs_info.get("lakefs_commit_id", ""),
            "iceberg_snapshot_id": data_lineage.get("iceberg_snapshot_id", ""),
        }


if __name__ == "__main__":
    # Example usage for local testing
    import argparse

    parser = argparse.ArgumentParser(description="Train Kronodroid autoencoder")
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5000")
    parser.add_argument("--mlflow-experiment-name", default="kronodroid-autoencoder")
    parser.add_argument("--mlflow-model-name", default="kronodroid_autoencoder")
    parser.add_argument("--lakefs-repository", default="kronodroid")
    parser.add_argument("--lakefs-ref", default="main")
    parser.add_argument("--iceberg-catalog", default="lakefs")
    parser.add_argument("--iceberg-database", default="kronodroid")
    parser.add_argument("--source-table", default="fct_training_dataset")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", default="128,64")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-rows-per-split", type=int, default=0)

    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    result = train_and_register(
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_model_name=args.mlflow_model_name,
        lakefs_repository=args.lakefs_repository,
        lakefs_ref=args.lakefs_ref,
        iceberg_catalog=args.iceberg_catalog,
        iceberg_database=args.iceberg_database,
        source_table=args.source_table,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.seed,
        max_rows_per_split=args.max_rows_per_split if args.max_rows_per_split > 0 else None,
    )

    print(f"Training complete: {json.dumps(result, indent=2)}")
