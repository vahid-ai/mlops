"""Kubeflow component for PyTorch Lightning autoencoder training.

This KFP component:
1. Loads training data from Iceberg tables (backed by LakeFS)
2. Trains a PyTorch Lightning autoencoder with early stopping
3. Logs all metrics, hyperparameters, and data lineage to MLflow
4. Registers the trained model in MLflow Model Registry

Data lineage is tracked via:
- LakeFS commit hash (data version)
- Iceberg snapshot ID (table version)
- Feast feature view name (feature definition)
"""

from typing import NamedTuple

from kfp import dsl


class TrainAutoencoderOutput(NamedTuple):
    """Output from the autoencoder training component."""

    run_id: str
    model_name: str
    model_version: str
    test_loss: float
    lakefs_commit_id: str
    iceberg_snapshot_id: str
    train_samples: int
    validation_samples: int
    test_samples: int


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "mlflow>=2.0",
        "torch>=2.0",
        "lightning>=2.0",
        "pyspark>=3.5",
        "pandas>=2.0",
        "numpy>=1.24",
        "requests>=2.31",
        "pyarrow>=14.0",
    ],
)
def train_kronodroid_autoencoder_op(
    # MLflow config
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_model_name: str,
    # Data config
    lakefs_endpoint: str,
    lakefs_repository: str,
    lakefs_ref: str,
    iceberg_catalog: str,
    iceberg_database: str,
    source_table: str,
    # Feast/Feature config (for lineage tracking)
    feast_project: str,
    feast_feature_view: str,
    feature_names_json: str,
    # Model config
    latent_dim: int,
    hidden_dims_json: str,
    # Training config
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    seed: int,
    max_rows_per_split: int,
    # MLflow artifact storage (S3/MinIO)
    minio_endpoint: str = "http://minio:9000",
) -> NamedTuple(
    "TrainAutoencoderOutput",
    [
        ("run_id", str),
        ("model_name", str),
        ("model_version", str),
        ("test_loss", float),
        ("lakefs_commit_id", str),
        ("iceberg_snapshot_id", str),
        ("train_samples", int),
        ("validation_samples", int),
        ("test_samples", int),
    ],
):
    """Train, validate, test, and register a Kronodroid autoencoder model.

    This component orchestrates the full training workflow:
    1. Connects to LakeFS-backed Iceberg tables via Spark
    2. Loads train/validation/test splits
    3. Trains PyTorch Lightning autoencoder
    4. Logs metrics and data lineage to MLflow
    5. Registers model in MLflow Model Registry

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_model_name: Name for model in MLflow registry
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        lakefs_ref: LakeFS branch or commit reference
        iceberg_catalog: Iceberg catalog name
        iceberg_database: Iceberg database name
        source_table: Source Iceberg table name
        feast_project: Feast project name (for lineage)
        feast_feature_view: Feast feature view name (for lineage)
        feature_names_json: JSON-encoded list of feature column names
        latent_dim: Autoencoder latent dimension
        hidden_dims_json: JSON-encoded list of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Optimizer learning rate
        seed: Random seed for reproducibility
        max_rows_per_split: Row limit per split (0 for unlimited)

    Returns:
        NamedTuple with run info, model info, metrics, and lineage
    """
    import json
    import os
    import tempfile
    from collections import namedtuple
    from typing import Dict, List, Optional, Tuple

    import mlflow
    import mlflow.pytorch
    import lightning as L
    import numpy as np
    import pandas as pd
    import requests
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import MLFlowLogger

    Output = namedtuple(
        "TrainAutoencoderOutput",
        [
            "run_id",
            "model_name",
            "model_version",
            "test_loss",
            "lakefs_commit_id",
            "iceberg_snapshot_id",
            "train_samples",
            "validation_samples",
            "test_samples",
        ],
    )

    # Parse JSON inputs
    feature_names = json.loads(feature_names_json)
    hidden_dims = tuple(json.loads(hidden_dims_json))
    max_rows = max_rows_per_split if max_rows_per_split > 0 else None

    # Set MLflow S3 endpoint for artifact storage
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_endpoint

    print(f"Training Kronodroid Autoencoder")
    print(f"  MLflow URI: {mlflow_tracking_uri}")
    print(f"  Experiment: {mlflow_experiment_name}")
    print(f"  Model: {mlflow_model_name}")
    print(f"  LakeFS: {lakefs_repository}@{lakefs_ref}")
    print(f"  Table: {iceberg_catalog}.{iceberg_database}.{source_table}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Architecture: {len(feature_names)} -> {hidden_dims} -> {latent_dim}")

    # --- LightningAutoencoder class (inline) ---
    class LightningAutoencoder(L.LightningModule):
        def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            hidden_dims: Tuple[int, ...],
            lr: float,
        ):
            super().__init__()
            self.save_hyperparameters()
            self.lr = lr

            # Build encoder
            encoder_layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                ])
                prev_dim = h_dim
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            # Build decoder
            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                ])
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

            self.loss_fn = nn.MSELoss()

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def _shared_step(self, batch, stage):
            x_hat = self(batch)
            loss = self.loss_fn(x_hat, batch)
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        def training_step(self, batch, batch_idx):
            return self._shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self._shared_step(batch, "val")

        def test_step(self, batch, batch_idx):
            return self._shared_step(batch, "test")

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    # --- Dataset class (inline) ---
    class AutoencoderDataset(Dataset):
        def __init__(self, df: pd.DataFrame, columns: List[str], mean=None, std=None):
            data = df[columns].values.astype(np.float32)
            data = np.nan_to_num(data, nan=0.0)

            if mean is None:
                self.mean = data.mean(axis=0)
                self.std = data.std(axis=0) + 1e-8
            else:
                self.mean = mean
                self.std = std

            self.data = torch.from_numpy((data - self.mean) / self.std)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # --- Get LakeFS commit info ---
    def get_lakefs_info(endpoint: str, repo: str, ref: str) -> Dict[str, str]:
        access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")
        api_base = endpoint.rstrip("/")

        try:
            url = f"{api_base}/api/v1/repositories/{repo}/refs/{ref}"
            resp = requests.get(url, auth=(access_key, secret_key), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "lakefs_commit_id": data.get("commit_id", "unknown"),
                    "lakefs_repository": repo,
                    "lakefs_ref": ref,
                }
        except Exception as e:
            print(f"Warning: Could not get LakeFS info: {e}")

        return {"lakefs_commit_id": "unknown", "lakefs_repository": repo, "lakefs_ref": ref}

    # --- Load data from Iceberg ---
    def load_data_from_iceberg(
        catalog: str,
        database: str,
        table: str,
        columns: List[str],
        max_rows: Optional[int],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, any]]:
        from pyspark.sql import SparkSession

        # Get credentials
        lakefs_access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
        lakefs_secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")

        # Create Spark session
        spark = (
            SparkSession.builder
            .appName("kronodroid-autoencoder-training")
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
            .config(f"spark.sql.catalog.{catalog}.type", "rest")
            .config(f"spark.sql.catalog.{catalog}.uri", f"{lakefs_endpoint}/api/v1/iceberg")
            .config(f"spark.sql.catalog.{catalog}.credential", f"{lakefs_access_key}:{lakefs_secret_key}")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )

        full_table = f"{catalog}.{database}.{table}"
        print(f"Reading from: {full_table}")

        df = spark.read.table(full_table)

        # Get snapshot ID
        snapshot_id = "unknown"
        try:
            snapshots = spark.sql(
                f"SELECT snapshot_id FROM {full_table}.snapshots ORDER BY committed_at DESC LIMIT 1"
            ).first()
            if snapshots:
                snapshot_id = str(snapshots["snapshot_id"])
        except Exception as e:
            print(f"Warning: Could not get snapshot ID: {e}")

        # Select columns
        select_cols = ["sample_id", "dataset_split"] + columns
        available = [c for c in select_cols if c in df.columns]
        df = df.select(*available)

        # Split and convert
        splits = {}
        counts = {}
        for split_name in ["train", "validation", "test"]:
            split_df = df.filter(df["dataset_split"] == split_name)
            if max_rows:
                split_df = split_df.limit(max_rows)
            pdf = split_df.toPandas()
            splits[split_name] = pdf
            counts[f"{split_name}_samples"] = len(pdf)
            print(f"  {split_name}: {len(pdf)} samples")

        spark.stop()

        return splits, {"iceberg_snapshot_id": snapshot_id, **counts}

    # --- Main training logic ---

    # Set seed
    L.seed_everything(seed)

    # Get lineage info
    lakefs_info = get_lakefs_info(lakefs_endpoint, lakefs_repository, lakefs_ref)
    print(f"LakeFS commit: {lakefs_info.get('lakefs_commit_id', 'unknown')}")

    # Load data
    splits, data_info = load_data_from_iceberg(
        iceberg_catalog, iceberg_database, source_table, feature_names, max_rows
    )
    print(f"Iceberg snapshot: {data_info.get('iceberg_snapshot_id', 'unknown')}")

    # Create datasets
    train_ds = AutoencoderDataset(splits["train"], feature_names)
    val_ds = AutoencoderDataset(splits["validation"], feature_names, train_ds.mean, train_ds.std)
    test_ds = AutoencoderDataset(splits["test"], feature_names, train_ds.mean, train_ds.std)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create model
    model = LightningAutoencoder(
        input_dim=len(feature_names),
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        lr=learning_rate,
    )

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run: {run_id}")

        # Log lineage
        mlflow.log_params({
            **lakefs_info,
            "iceberg_table": f"{iceberg_catalog}.{iceberg_database}.{source_table}",
            "iceberg_snapshot_id": data_info.get("iceberg_snapshot_id"),
            "feast_project": feast_project,
            "feast_feature_view": feast_feature_view,
            "feature_names": json.dumps(feature_names),
            "train_samples": data_info.get("train_samples"),
            "validation_samples": data_info.get("validation_samples"),
            "test_samples": data_info.get("test_samples"),
        })

        # Log hyperparameters
        mlflow.log_params({
            "input_dim": len(feature_names),
            "latent_dim": latent_dim,
            "hidden_dims": str(hidden_dims),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        })

        # Setup trainer
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_id=run_id,
        )

        callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ]

        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=mlflow_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
        )

        # Train
        print("Starting training...")
        trainer.fit(model, train_loader, val_loader)

        # Test
        print("Running test evaluation...")
        test_results = trainer.test(model, test_loader)
        test_loss = float(test_results[0]["test_loss"])
        print(f"Test loss: {test_loss:.6f}")

        mlflow.log_metric("test_loss", test_loss)

        # Save normalization params
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "mean": train_ds.mean.tolist(),
                "std": train_ds.std.tolist(),
                "feature_names": feature_names,
            }, f)
            mlflow.log_artifact(f.name, "normalization")
            os.unlink(f.name)

        # Register model
        print("Registering model...")
        mlflow.pytorch.log_model(model, "model", registered_model_name=mlflow_model_name)

        # Get version
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{mlflow_model_name}'")
        model_version = str(max([int(v.version) for v in versions])) if versions else "1"
        print(f"Registered as {mlflow_model_name} v{model_version}")

    return Output(
        run_id=run_id,
        model_name=mlflow_model_name,
        model_version=model_version,
        test_loss=test_loss,
        lakefs_commit_id=lakefs_info.get("lakefs_commit_id", ""),
        iceberg_snapshot_id=data_info.get("iceberg_snapshot_id", ""),
        train_samples=data_info.get("train_samples", 0),
        validation_samples=data_info.get("validation_samples", 0),
        test_samples=data_info.get("test_samples", 0),
    )
