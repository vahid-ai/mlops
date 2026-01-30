"""Component: Train Kronodroid Autoencoder with MLflow tracking and data lineage.

This KFP component trains a PyTorch Lightning autoencoder on the Kronodroid syscall
features dataset, with comprehensive monitoring and lineage tracking.

Data Lineage (per versioning-datasets.md):
- Uses `dataset_split` column for train/val/test splits (no data duplication)
- Logs LakeFS commit ID to MLflow for exact data version tracking
- Logs Feast feature view reference for feature engineering lineage
- Creates LakeFS tag on successful training for permanent reference

The training data comes from LakeFS-versioned Iceberg tables accessed via Feast
or direct Spark/Iceberg read as fallback.
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
    base_image="dfp-autoencoder-train:v6",
)
def train_kronodroid_autoencoder_op(
    # MLflow config
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_model_name: str,
    # Data config - LakeFS/Iceberg for lineage
    lakefs_endpoint: str,
    lakefs_repository: str,
    lakefs_ref: str,
    iceberg_catalog: str,
    iceberg_database: str,
    source_table: str,
    # Feast config for data loading
    feast_repo_path: str,
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
    # Logging and monitoring config
    log_level: str = "INFO",
    enable_tensorboard: bool = True,
    log_every_n_steps: int = 10,
    enable_gradient_logging: bool = True,
    enable_resource_monitoring: bool = True,
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

    This component orchestrates the full training workflow following the
    versioning-datasets.md lineage pattern:

    1. Resolves LakeFS ref to exact commit ID for reproducibility
    2. Loads train/validation/test data using dataset_split column
    3. Trains PyTorch Lightning autoencoder with comprehensive monitoring
    4. Logs all lineage parameters to MLflow (LakeFS commit, Feast view, etc.)
    5. Registers model in MLflow Model Registry

    Data Lineage Parameters Logged:
    - lakefs_repository, lakefs_ref, lakefs_commit_id: Exact data version
    - feast_project, feast_feature_view: Feature engineering version
    - iceberg_table: Source table reference
    - train/val/test_samples: Dataset statistics

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_model_name: Name for model in MLflow registry
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        lakefs_ref: LakeFS branch or commit reference
        iceberg_catalog: Iceberg catalog name
        iceberg_database: Iceberg database name
        source_table: Source Iceberg table name (with dataset_split column)
        feast_repo_path: Path to Feast feature_store.yaml
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
        minio_endpoint: MinIO/S3 endpoint for MLflow artifacts
        log_level: Python logging level (DEBUG, INFO, WARNING, ERROR)
        enable_tensorboard: Enable TensorBoard logging alongside MLflow
        log_every_n_steps: Log per-step metrics every N training steps
        enable_gradient_logging: Log gradient statistics during training
        enable_resource_monitoring: Log memory/GPU usage metrics

    Returns:
        NamedTuple with run info, model info, metrics, and lineage
    """
    import json
    import logging
    import os
    import sys
    import tempfile
    import time
    from collections import namedtuple
    from datetime import datetime
    from typing import Any, Dict, List, Optional, Tuple

    import mlflow
    import mlflow.pytorch
    import lightning as L
    import numpy as np
    import pandas as pd
    import psutil
    import requests
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from lightning.pytorch.callbacks import (
        Callback,
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

    # --- Setup logging ---
    def setup_logging(level: str) -> logging.Logger:
        """Configure logging with the specified level."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        logger = logging.getLogger("kronodroid-autoencoder")
        logger.setLevel(numeric_level)

        # Reduce noise from other libraries
        logging.getLogger("pyspark").setLevel(logging.WARNING)
        logging.getLogger("py4j").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        return logger

    logger = setup_logging(log_level)

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

    # Set LakeFS endpoint for Spark/Feast config variable substitution
    os.environ["LAKEFS_ENDPOINT_URL"] = lakefs_endpoint

    logger.info("=" * 60)
    logger.info("Kronodroid Autoencoder Training")
    logger.info("=" * 60)
    logger.info(f"MLflow URI: {mlflow_tracking_uri}")
    logger.info(f"Experiment: {mlflow_experiment_name}")
    logger.info(f"Model: {mlflow_model_name}")
    logger.info(f"LakeFS: {lakefs_repository}@{lakefs_ref}")
    logger.info(f"Table: {iceberg_catalog}.{iceberg_database}.{source_table}")
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Architecture: {len(feature_names)} -> {hidden_dims} -> {latent_dim}")
    logger.info(f"Training: batch_size={batch_size}, max_epochs={max_epochs}, lr={learning_rate}")
    logger.info("=" * 60)

    # --- Custom Callbacks for Monitoring ---

    class ResourceMonitorCallback(Callback):
        """Monitor system resources during training."""

        def __init__(self, log_every_n_steps: int = 10):
            super().__init__()
            self.log_every_n_steps = log_every_n_steps

        def _get_memory_stats(self) -> Dict[str, float]:
            stats = {}
            process = psutil.Process()
            mem_info = process.memory_info()
            stats["memory_rss_mb"] = mem_info.rss / (1024 * 1024)
            stats["memory_percent"] = process.memory_percent()
            sys_mem = psutil.virtual_memory()
            stats["system_memory_percent"] = sys_mem.percent

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    stats[f"gpu_{i}_allocated_mb"] = allocated

            return stats

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx % self.log_every_n_steps == 0:
                stats = self._get_memory_stats()
                for key, value in stats.items():
                    pl_module.log(f"resource/{key}", value, on_step=True, on_epoch=False)

        def on_train_epoch_end(self, trainer, pl_module):
            stats = self._get_memory_stats()
            for key, value in stats.items():
                pl_module.log(f"resource/{key}", value, on_step=False, on_epoch=True)

    class GradientMonitorCallback(Callback):
        """Monitor gradient statistics during training."""

        def __init__(self, log_every_n_steps: int = 10):
            super().__init__()
            self.log_every_n_steps = log_every_n_steps

        def on_after_backward(self, trainer, pl_module):
            if trainer.global_step % self.log_every_n_steps == 0:
                grad_norms = []
                grad_values = []

                for name, param in pl_module.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append(grad_norm)
                        grad_values.extend(param.grad.flatten().tolist()[:100])

                if grad_norms:
                    pl_module.log("gradient/norm_mean", np.mean(grad_norms), on_step=True, on_epoch=False)
                    pl_module.log("gradient/norm_max", np.max(grad_norms), on_step=True, on_epoch=False)

                if grad_values:
                    pl_module.log("gradient/value_std", np.std(grad_values), on_step=True, on_epoch=False)

    class TrainingProgressCallback(Callback):
        """Log training progress with timing and throughput."""

        def __init__(self):
            super().__init__()
            self.epoch_start_time = None
            self.training_start_time = None
            self.epoch_samples = 0
            self.total_samples = 0

        def on_train_start(self, trainer, pl_module):
            self.training_start_time = time.time()
            logger.info("Training started")

        def on_train_epoch_start(self, trainer, pl_module):
            self.epoch_start_time = time.time()
            self.epoch_samples = 0
            logger.info(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)
            self.epoch_samples += batch_size
            self.total_samples += batch_size

        def on_train_epoch_end(self, trainer, pl_module):
            epoch_time = time.time() - self.epoch_start_time
            total_time = time.time() - self.training_start_time
            samples_per_second = self.epoch_samples / epoch_time if epoch_time > 0 else 0

            pl_module.log("timing/epoch_seconds", epoch_time, on_step=False, on_epoch=True)
            pl_module.log("timing/total_seconds", total_time, on_step=False, on_epoch=True)
            pl_module.log("throughput/samples_per_second", samples_per_second, on_step=False, on_epoch=True)

            train_loss = trainer.callback_metrics.get("train_loss", 0)
            val_loss = trainer.callback_metrics.get("val_loss", 0)

            logger.info(
                f"Epoch {trainer.current_epoch + 1} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"time={epoch_time:.1f}s | throughput={samples_per_second:.0f}/s"
            )

        def on_train_end(self, trainer, pl_module):
            total_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {total_time:.1f}s | Total samples: {self.total_samples}")

    # --- LightningAutoencoder class ---
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
            self.mae_fn = nn.L1Loss()

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def encode(self, x):
            return self.encoder(x)

        def _shared_step(self, batch, stage):
            x_hat = self(batch)
            mse_loss = self.loss_fn(x_hat, batch)
            mae_loss = self.mae_fn(x_hat, batch)

            self.log(f"{stage}_loss", mse_loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
            self.log(f"{stage}_mse", mse_loss, on_epoch=True, on_step=False)
            self.log(f"{stage}_mae", mae_loss, on_epoch=True, on_step=False)

            if stage == "val":
                with torch.no_grad():
                    per_feature_mse = ((x_hat - batch) ** 2).mean(dim=0)
                    self.log(f"{stage}_max_feature_error", per_feature_mse.max(), on_epoch=True)
                    self.log(f"{stage}_min_feature_error", per_feature_mse.min(), on_epoch=True)

            return mse_loss

        def training_step(self, batch, batch_idx):
            loss = self._shared_step(batch, "train")
            if batch_idx % log_every_n_steps == 0:
                self.log("train_batch_loss", loss, on_step=True, on_epoch=False)
            return loss

        def validation_step(self, batch, batch_idx):
            return self._shared_step(batch, "val")

        def test_step(self, batch, batch_idx):
            loss = self._shared_step(batch, "test")
            with torch.no_grad():
                x_hat = self(batch)
                sample_errors = ((x_hat - batch) ** 2).mean(dim=1)
                self.log("test_error_mean", sample_errors.mean(), on_epoch=True)
                self.log("test_error_std", sample_errors.std(), on_epoch=True)
                self.log("test_error_max", sample_errors.max(), on_epoch=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

    # --- Dataset class ---
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

    # --- Get LakeFS commit info for lineage ---
    def get_lakefs_commit_id(endpoint: str, repo: str, ref: str) -> str:
        """Resolve LakeFS ref to exact commit ID for reproducibility."""
        access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")
        api_base = endpoint.rstrip("/")

        try:
            url = f"{api_base}/api/v1/repositories/{repo}/refs/{ref}"
            resp = requests.get(url, auth=(access_key, secret_key), timeout=10)
            if resp.status_code == 200:
                return resp.json().get("commit_id", "unknown")
        except Exception as e:
            logger.warning(f"Could not resolve LakeFS ref: {e}")

        return "unknown"

    # --- Load data from Feast ---
    def load_data_from_feast(
        feast_repo_path: str,
        iceberg_table: str,
        feature_columns: List[str],
        max_rows: Optional[int],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, any], List[str]]:
        """Load training data from Feast offline store (Iceberg via Spark).

        Returns:
            Tuple of (splits dict, lineage info dict, actual feature columns used)
        """
        from pathlib import Path
        from pyspark.sql import SparkSession
        from pyspark.sql.types import NumericType
        import yaml
        import re

        feast_config_path = Path(feast_repo_path) / "feature_store.yaml"
        logger.info(f"Loading Feast config from: {feast_config_path}")

        with open(feast_config_path) as f:
            feast_config = yaml.safe_load(f)

        spark_conf = feast_config.get("offline_store", {}).get("spark_conf", {})
        logger.info("Creating Spark session with Iceberg config from Feast")

        # Override JAR packages to use PySpark 3.5 compatible versions
        # The Feast config may have Spark 4.0 JARs which are incompatible
        spark_35_jars = (
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )

        builder = SparkSession.builder.appName("kronodroid-training")
        for key, value in spark_conf.items():
            # Override spark.jars.packages with PySpark 3.5 compatible versions
            if key == "spark.jars.packages":
                logger.info(f"Overriding {key} with PySpark 3.5 compatible JARs")
                value = spark_35_jars
            elif isinstance(value, str) and "${" in value:
                for match in re.finditer(r'\$\{(\w+)\}', value):
                    env_var = match.group(1)
                    env_val = os.environ.get(env_var, "")
                    value = value.replace(f"${{{env_var}}}", env_val)
            builder = builder.config(key, str(value))

        spark_session = builder.getOrCreate()
        logger.info(f"Reading from Iceberg table: {iceberg_table}")

        spark_df = spark_session.read.table(iceberg_table)
        all_columns = spark_df.columns
        logger.info(f"Available columns in table: {all_columns}")

        # Check if specified feature columns exist
        available_features = [c for c in feature_columns if c in all_columns]

        if not available_features:
            # Auto-discover numeric columns as features
            logger.warning("Specified feature columns not found. Auto-discovering numeric columns...")
            exclude_cols = {"sample_id", "dataset_split", "data_source", "event_timestamp",
                          "feature_timestamp", "_dbt_loaded_at", "_source_file", "_ingestion_timestamp",
                          "label", "malware_family", "first_seen_year", "hash_value"}
            numeric_cols = []
            for field in spark_df.schema.fields:
                if field.name not in exclude_cols and isinstance(field.dataType, NumericType):
                    numeric_cols.append(field.name)
            available_features = sorted(numeric_cols)[:50]  # Limit to 50 features
            logger.info(f"Auto-discovered {len(available_features)} numeric feature columns")

        logger.info(f"Using features: {available_features[:10]}... ({len(available_features)} total)")

        select_cols = ["sample_id", "dataset_split"] + available_features
        available = [c for c in select_cols if c in all_columns]
        spark_df = spark_df.select(*available)

        df = spark_df.toPandas()
        logger.info(f"Retrieved {len(df):,} total samples")

        # Split by dataset_split column (following versioning-datasets.md pattern)
        splits = {}
        counts = {}
        for split_name in ["train", "validation", "test"]:
            split_df = df[df["dataset_split"] == split_name].copy()
            if max_rows and len(split_df) > max_rows:
                split_df = split_df.head(max_rows)
            splits[split_name] = split_df
            counts[f"{split_name}_samples"] = len(split_df)
            logger.info(f"  {split_name}: {len(split_df):,} samples")

        # Get Iceberg snapshot ID for lineage
        iceberg_snapshot_id = ""
        try:
            snapshot_df = spark_session.sql(
                f"SELECT snapshot_id FROM {iceberg_table}.snapshots ORDER BY committed_at DESC LIMIT 1"
            )
            snapshot_row = snapshot_df.first()
            if snapshot_row:
                iceberg_snapshot_id = str(snapshot_row["snapshot_id"])
        except Exception as e:
            logger.warning(f"Could not get Iceberg snapshot ID: {e}")

        lineage_info = {
            "iceberg_table": iceberg_table,
            "iceberg_snapshot_id": iceberg_snapshot_id,
            "data_source": "feast_spark_iceberg",
            **counts,
        }

        spark_session.stop()
        return splits, lineage_info, available_features

    # --- Main training logic ---

    training_start_time = time.time()
    logger.info("Initializing training pipeline...")

    # Set seed for reproducibility
    L.seed_everything(seed)

    # Get LakeFS commit ID for lineage (per versioning-datasets.md)
    logger.info("Resolving LakeFS ref to commit ID for lineage...")
    lakefs_commit_id = get_lakefs_commit_id(lakefs_endpoint, lakefs_repository, lakefs_ref)
    logger.info(f"LakeFS commit: {lakefs_commit_id}")

    # Load data from Feast (Iceberg via Spark)
    iceberg_table_full = f"{iceberg_catalog}.{iceberg_database}.{source_table}"
    logger.info("Loading data from Feast offline store (Iceberg via Spark)...")
    data_load_start = time.time()
    splits, data_info, actual_features = load_data_from_feast(
        feast_repo_path=feast_repo_path,
        iceberg_table=iceberg_table_full,
        feature_columns=feature_names,
        max_rows=max_rows,
    )
    data_load_time = time.time() - data_load_start
    logger.info(f"Data loading completed in {data_load_time:.1f}s")

    # Use actual discovered features (may differ from requested feature_names)
    if actual_features != feature_names:
        logger.info(f"Using {len(actual_features)} auto-discovered features instead of {len(feature_names)} specified")
        feature_names = actual_features

    # Create datasets
    logger.info("Creating PyTorch datasets...")
    train_ds = AutoencoderDataset(splits["train"], feature_names)
    val_ds = AutoencoderDataset(splits["validation"], feature_names, train_ds.mean, train_ds.std)
    test_ds = AutoencoderDataset(splits["test"], feature_names, train_ds.mean, train_ds.std)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Create model
    logger.info("Initializing model...")
    model = LightningAutoencoder(
        input_dim=len(feature_names),
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        lr=learning_rate,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Setup MLflow
    logger.info("Setting up MLflow tracking...")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Log comprehensive lineage parameters (per versioning-datasets.md)
        mlflow.log_params({
            # LakeFS data version (primary lineage)
            "data/lakefs_repository": lakefs_repository,
            "data/lakefs_ref": lakefs_ref,
            "data/lakefs_commit": lakefs_commit_id,
            # Iceberg table reference
            "data/iceberg_table": iceberg_table_full,
            "data/iceberg_snapshot_id": data_info.get("iceberg_snapshot_id", ""),
            # Feast feature store reference
            "data/feast_project": feast_project,
            "data/feast_feature_view": feast_feature_view,
            "data/feast_repo_path": feast_repo_path,
            # Split statistics
            "data/train_samples": data_info.get("train_samples"),
            "data/validation_samples": data_info.get("validation_samples"),
            "data/test_samples": data_info.get("test_samples"),
            # Feature names
            "data/feature_names": json.dumps(feature_names),
        })

        # Log model hyperparameters
        mlflow.log_params({
            "model/input_dim": len(feature_names),
            "model/latent_dim": latent_dim,
            "model/hidden_dims": str(hidden_dims),
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
        })

        # Log training config
        mlflow.log_params({
            "training/batch_size": batch_size,
            "training/max_epochs": max_epochs,
            "training/learning_rate": learning_rate,
            "training/seed": seed,
        })

        mlflow.log_metric("data_load_time_seconds", data_load_time)

        # Setup loggers
        loggers = []

        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_id=run_id,
        )
        loggers.append(mlflow_logger)

        tensorboard_dir = None
        if enable_tensorboard:
            tensorboard_dir = tempfile.mkdtemp(prefix="tensorboard_")
            tb_logger = TensorBoardLogger(
                save_dir=tensorboard_dir,
                name="kronodroid_autoencoder",
                version=run_id[:8],
            )
            loggers.append(tb_logger)

        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-{val_loss:.4f}",
                verbose=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            TrainingProgressCallback(),
        ]

        if enable_resource_monitoring:
            callbacks.append(ResourceMonitorCallback(log_every_n_steps=log_every_n_steps))

        if enable_gradient_logging:
            callbacks.append(GradientMonitorCallback(log_every_n_steps=log_every_n_steps))

        # Create trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            logger=loggers,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=log_every_n_steps,
            deterministic=True,
        )

        # Train
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        fit_start_time = time.time()
        trainer.fit(model, train_loader, val_loader)
        fit_time = time.time() - fit_start_time
        logger.info(f"Training completed in {fit_time:.1f}s")

        mlflow.log_metric("fit_time_seconds", fit_time)

        # Test
        logger.info("=" * 60)
        logger.info("RUNNING TEST EVALUATION")
        logger.info("=" * 60)
        test_start_time = time.time()
        test_results = trainer.test(model, test_loader)
        test_time = time.time() - test_start_time

        test_loss = float(test_results[0]["test_loss"])
        logger.info(f"Test Loss (MSE): {test_loss:.6f}")

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_time_seconds", test_time)

        for key, value in test_results[0].items():
            if key != "test_loss":
                mlflow.log_metric(key, float(value))

        # Save normalization params
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "mean": train_ds.mean.tolist(),
                "std": train_ds.std.tolist(),
                "feature_names": feature_names,
            }, f)
            mlflow.log_artifact(f.name, "normalization")
            os.unlink(f.name)

        # Save TensorBoard logs
        if enable_tensorboard and tensorboard_dir:
            mlflow.log_artifacts(tensorboard_dir, "tensorboard")

        # Log training summary
        total_time = time.time() - training_start_time
        mlflow.log_metric("total_time_seconds", total_time)
        mlflow.log_metric("best_val_loss", float(trainer.checkpoint_callback.best_model_score))

        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Test loss: {test_loss:.6f}")
        logger.info(f"Best val loss: {trainer.checkpoint_callback.best_model_score:.6f}")

        # Register model
        logger.info("=" * 60)
        logger.info("REGISTERING MODEL")
        logger.info("=" * 60)
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=mlflow_model_name,
            pip_requirements=["torch>=2.0", "lightning>=2.0"],
        )

        mlflow_client = mlflow.tracking.MlflowClient()
        versions = mlflow_client.search_model_versions(f"name='{mlflow_model_name}'")
        model_version = str(max([int(v.version) for v in versions])) if versions else "1"
        logger.info(f"Registered as {mlflow_model_name} v{model_version}")

        # Cleanup
        if tensorboard_dir:
            import shutil
            shutil.rmtree(tensorboard_dir, ignore_errors=True)

    return Output(
        run_id=run_id,
        model_name=mlflow_model_name,
        model_version=model_version,
        test_loss=test_loss,
        lakefs_commit_id=lakefs_commit_id,
        iceberg_snapshot_id=data_info.get("iceberg_snapshot_id", ""),
        train_samples=data_info.get("train_samples", 0),
        validation_samples=data_info.get("validation_samples", 0),
        test_samples=data_info.get("test_samples", 0),
    )


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests"],
)
def lakefs_tag_model_data_op(
    lakefs_endpoint: str,
    lakefs_repository: str,
    lakefs_commit_id: str,
    model_name: str,
    model_version: str,
) -> NamedTuple("LakeFSTagOutput", [("tag_name", str), ("success", bool)]):
    """Create a LakeFS tag to permanently reference the training data version.

    Following versioning-datasets.md: After successful training, tag the LakeFS
    commit so the exact data version can be referenced for reproducibility.

    Args:
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        lakefs_commit_id: LakeFS commit ID to tag
        model_name: Model name (used in tag name)
        model_version: Model version (used in tag name)

    Returns:
        NamedTuple with tag_name and success status
    """
    import os
    from collections import namedtuple

    import requests

    Output = namedtuple("LakeFSTagOutput", ["tag_name", "success"])

    access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")
    api_base = lakefs_endpoint.rstrip("/")
    auth = (access_key, secret_key)

    # Create tag name following versioning-datasets.md pattern
    tag_name = f"model-{model_name}-v{model_version}-data"

    print(f"Creating LakeFS tag: {tag_name}")
    print(f"  Commit: {lakefs_commit_id}")

    if lakefs_commit_id == "unknown" or not lakefs_commit_id:
        print("Warning: No valid commit ID, skipping tag creation")
        return Output(tag_name="", success=False)

    try:
        url = f"{api_base}/api/v1/repositories/{lakefs_repository}/tags"
        data = {
            "id": tag_name,
            "ref": lakefs_commit_id,
        }
        resp = requests.post(url, json=data, auth=auth)

        if resp.status_code in (200, 201):
            print(f"Created tag: {tag_name}")
            return Output(tag_name=tag_name, success=True)
        elif resp.status_code == 409:
            print(f"Tag already exists: {tag_name}")
            return Output(tag_name=tag_name, success=True)
        else:
            print(f"Failed to create tag: {resp.status_code} - {resp.text}")
            return Output(tag_name="", success=False)

    except Exception as e:
        print(f"Error creating tag: {e}")
        return Output(tag_name="", success=False)
