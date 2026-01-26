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

Logging and Monitoring:
- Python logging with configurable verbosity
- Per-epoch and per-batch metric logging
- Resource monitoring (memory, GPU if available)
- Training progress with ETA and throughput
- Gradient statistics and learning rate tracking
- TensorBoard support (optional)
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


# Pre-built training image with all dependencies installed (fast startup)
# Build with: task kfp-training-image
# Falls back to runtime installs if image not available
DEFAULT_TRAINING_IMAGE = "dfp-kfp-training:latest"


@dsl.component(
    # Using pre-built image with all dependencies - no runtime pip installs needed
    base_image=DEFAULT_TRAINING_IMAGE,
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
    feast_repo_path: str,  # Path to feast feature_store.yaml
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
    log_level: str = "INFO",  # DEBUG, INFO, WARNING, ERROR
    enable_tensorboard: bool = True,
    log_every_n_steps: int = 10,  # Log metrics every N training steps
    enable_gradient_logging: bool = True,  # Log gradient statistics
    enable_resource_monitoring: bool = True,  # Log memory/GPU usage
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
    3. Trains PyTorch Lightning autoencoder with comprehensive monitoring
    4. Logs metrics, data lineage, and training statistics to MLflow
    5. Registers model in MLflow Model Registry

    Monitoring Features:
    - Per-epoch metrics: train_loss, val_loss, learning_rate
    - Per-step metrics (configurable frequency): batch_loss, throughput
    - Resource monitoring: memory_used_mb, gpu_memory_mb (if available)
    - Gradient statistics: gradient_norm, gradient_mean, gradient_std
    - Training progress: epoch_time, total_time, samples_per_second
    - TensorBoard integration for visualization

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
        minio_endpoint: MinIO/S3 endpoint for MLflow artifacts
        log_level: Python logging level (DEBUG, INFO, WARNING, ERROR)
        enable_tensorboard: Enable TensorBoard logging alongside MLflow
        log_every_n_steps: Log per-step metrics every N training steps
        enable_gradient_logging: Log gradient statistics during training
        enable_resource_monitoring: Log memory and GPU usage metrics

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
        RichProgressBar,
    )
    from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

    # --- Setup logging ---
    def setup_logging(level: str) -> logging.Logger:
        """Configure logging with the specified level."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Create component logger
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
    logger.info(f"Training config: batch_size={batch_size}, max_epochs={max_epochs}, lr={learning_rate}")
    logger.info(f"Monitoring: tensorboard={enable_tensorboard}, gradients={enable_gradient_logging}, resources={enable_resource_monitoring}")
    logger.info("=" * 60)

    # --- Custom Callbacks for Monitoring ---

    class ResourceMonitorCallback(Callback):
        """Monitor system resources (CPU, memory, GPU) during training."""

        def __init__(self, log_every_n_steps: int = 10):
            super().__init__()
            self.log_every_n_steps = log_every_n_steps

        def _get_memory_stats(self) -> Dict[str, float]:
            """Get current memory usage statistics."""
            stats = {}

            # CPU/System memory
            process = psutil.Process()
            mem_info = process.memory_info()
            stats["memory_rss_mb"] = mem_info.rss / (1024 * 1024)
            stats["memory_percent"] = process.memory_percent()

            # System-wide memory
            sys_mem = psutil.virtual_memory()
            stats["system_memory_percent"] = sys_mem.percent

            # GPU memory (if available)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                    stats[f"gpu_{i}_allocated_mb"] = allocated
                    stats[f"gpu_{i}_reserved_mb"] = reserved

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
                        grad_values.extend(param.grad.flatten().tolist()[:100])  # Sample

                if grad_norms:
                    pl_module.log("gradient/norm_mean", np.mean(grad_norms), on_step=True, on_epoch=False)
                    pl_module.log("gradient/norm_max", np.max(grad_norms), on_step=True, on_epoch=False)
                    pl_module.log("gradient/norm_min", np.min(grad_norms), on_step=True, on_epoch=False)

                if grad_values:
                    pl_module.log("gradient/value_mean", np.mean(grad_values), on_step=True, on_epoch=False)
                    pl_module.log("gradient/value_std", np.std(grad_values), on_step=True, on_epoch=False)

    class TrainingProgressCallback(Callback):
        """Log detailed training progress with timing and throughput."""

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

            # Log timing metrics
            pl_module.log("timing/epoch_seconds", epoch_time, on_step=False, on_epoch=True)
            pl_module.log("timing/total_seconds", total_time, on_step=False, on_epoch=True)
            pl_module.log("throughput/samples_per_second", samples_per_second, on_step=False, on_epoch=True)
            pl_module.log("throughput/epoch_samples", float(self.epoch_samples), on_step=False, on_epoch=True)

            # Get current metrics
            train_loss = trainer.callback_metrics.get("train_loss", 0)
            val_loss = trainer.callback_metrics.get("val_loss", 0)

            logger.info(
                f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} completed | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"time={epoch_time:.1f}s | throughput={samples_per_second:.0f} samples/s"
            )

        def on_train_end(self, trainer, pl_module):
            total_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {total_time:.1f}s | Total samples processed: {self.total_samples}")

    class MLflowMetricsCallback(Callback):
        """Log additional metrics to MLflow beyond what Lightning logs automatically."""

        def __init__(self, mlflow_client, run_id: str):
            super().__init__()
            self.client = mlflow_client
            self.run_id = run_id

        def on_validation_epoch_end(self, trainer, pl_module):
            # Log best metrics
            if trainer.checkpoint_callback:
                best_score = trainer.checkpoint_callback.best_model_score
                if best_score is not None:
                    self.client.log_metric(self.run_id, "best_val_loss", float(best_score), step=trainer.current_epoch)

            # Log learning rate
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    self.client.log_metric(self.run_id, "learning_rate", param_group["lr"], step=trainer.current_epoch)

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
            self.mae_fn = nn.L1Loss()

            # Track per-feature reconstruction errors for analysis
            self.feature_errors = []

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def encode(self, x):
            """Get latent representation."""
            return self.encoder(x)

        def _shared_step(self, batch, stage):
            x_hat = self(batch)
            mse_loss = self.loss_fn(x_hat, batch)
            mae_loss = self.mae_fn(x_hat, batch)

            # Log multiple metrics
            self.log(f"{stage}_loss", mse_loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
            self.log(f"{stage}_mse", mse_loss, on_epoch=True, on_step=False)
            self.log(f"{stage}_mae", mae_loss, on_epoch=True, on_step=False)

            # Per-feature error (on validation only to avoid overhead)
            if stage == "val":
                with torch.no_grad():
                    per_feature_mse = ((x_hat - batch) ** 2).mean(dim=0)
                    self.log(f"{stage}_max_feature_error", per_feature_mse.max(), on_epoch=True)
                    self.log(f"{stage}_min_feature_error", per_feature_mse.min(), on_epoch=True)

            return mse_loss

        def training_step(self, batch, batch_idx):
            loss = self._shared_step(batch, "train")

            # Log batch-level metrics at intervals
            if batch_idx % log_every_n_steps == 0:
                self.log("train_batch_loss", loss, on_step=True, on_epoch=False)

            return loss

        def validation_step(self, batch, batch_idx):
            return self._shared_step(batch, "val")

        def test_step(self, batch, batch_idx):
            loss = self._shared_step(batch, "test")

            # Compute reconstruction error distribution for test set
            with torch.no_grad():
                x_hat = self(batch)
                sample_errors = ((x_hat - batch) ** 2).mean(dim=1)
                self.log("test_error_mean", sample_errors.mean(), on_epoch=True)
                self.log("test_error_std", sample_errors.std(), on_epoch=True)
                self.log("test_error_max", sample_errors.max(), on_epoch=True)
                self.log("test_error_min", sample_errors.min(), on_epoch=True)

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
            logger.warning(f"Could not get LakeFS info: {e}")

        return {"lakefs_commit_id": "unknown", "lakefs_repository": repo, "lakefs_ref": ref}

    # --- Load data from Iceberg ---
    def load_data_from_iceberg_direct(
        repo_path: str,
        iceberg_table: str,
        feature_columns: List[str],
        max_rows: Optional[int],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, any]]:
        """Load training data directly from Iceberg table using Spark config from Feast.

        This is a fallback method when Feast feature views are not registered.
        It reads the Spark configuration from feature_store.yaml but bypasses
        the Feast registry entirely.

        Args:
            repo_path: Path to Feast feature_store.yaml (for Spark config)
            iceberg_table: Full Iceberg table name (e.g., lakefs.kronodroid.fct_training_dataset)
            feature_columns: List of feature columns to retrieve
            max_rows: Optional limit per split

        Returns:
            Tuple of (splits dict, lineage info dict)
        """
        from pathlib import Path
        from pyspark.sql import SparkSession
        import yaml
        import re

        # Read Feast config for Spark settings
        feast_config_path = Path(repo_path) / "feature_store.yaml"
        with open(feast_config_path) as f:
            feast_config = yaml.safe_load(f)

        spark_conf = feast_config.get("offline_store", {}).get("spark_conf", {})
        logger.info(f"Creating Spark session with Iceberg JARs from Feast config")

        # Build Spark session
        builder = SparkSession.builder.appName("kronodroid-training-direct")
        for key, value in spark_conf.items():
            # Expand environment variables in config values
            if isinstance(value, str) and "${" in value:
                for match in re.finditer(r'\$\{(\w+)\}', value):
                    env_var = match.group(1)
                    env_val = os.environ.get(env_var, "")
                    value = value.replace(f"${{{env_var}}}", env_val)
            builder = builder.config(key, str(value))

        # Inject S3A and S3FileIO credentials from environment variables (set by K8s secrets)
        # These are required for LakeFS S3 gateway authentication
        lakefs_access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
        lakefs_secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")
        if lakefs_access_key and lakefs_secret_key:
            logger.info("Injecting LakeFS credentials from environment into Spark config")
            # S3A credentials for Hadoop filesystem
            builder = builder.config("spark.hadoop.fs.s3a.access.key", lakefs_access_key)
            builder = builder.config("spark.hadoop.fs.s3a.secret.key", lakefs_secret_key)
            # Per-bucket credentials for the LakeFS repository (kronodroid)
            builder = builder.config("spark.hadoop.fs.s3a.bucket.kronodroid.access.key", lakefs_access_key)
            builder = builder.config("spark.hadoop.fs.s3a.bucket.kronodroid.secret.key", lakefs_secret_key)
            # S3FileIO credentials for Iceberg (uses different property names)
            builder = builder.config("spark.sql.catalog.lakefs.s3.access-key-id", lakefs_access_key)
            builder = builder.config("spark.sql.catalog.lakefs.s3.secret-access-key", lakefs_secret_key)
        else:
            logger.warning("LAKEFS_ACCESS_KEY_ID/LAKEFS_SECRET_ACCESS_KEY not set - S3A auth may fail")

        spark_session = builder.getOrCreate()
        logger.info(f"Spark session created, reading from: {iceberg_table}")

        # Read from Iceberg table
        spark_df = spark_session.read.table(iceberg_table)
        select_cols = ["sample_id", "dataset_split"] + feature_columns
        available = [c for c in select_cols if c in spark_df.columns]
        spark_df = spark_df.select(*available)

        df = spark_df.toPandas()
        logger.info(f"Retrieved {len(df):,} total samples via direct Iceberg read")

        # Split by dataset_split column
        splits = {}
        counts = {}
        for split_name in ["train", "validation", "test"]:
            split_df = df[df["dataset_split"] == split_name].copy()
            if max_rows and len(split_df) > max_rows:
                split_df = split_df.head(max_rows)
            splits[split_name] = split_df
            counts[f"{split_name}_samples"] = len(split_df)
            logger.info(f"  Loaded {split_name}: {len(split_df):,} samples")

        lineage_info = {
            "iceberg_table": iceberg_table,
            "data_source": "iceberg_direct",
            "feast_repo_path": repo_path,
            **counts,
        }

        spark_session.stop()
        return splits, lineage_info

    def load_data_from_feast(
        repo_path: str,
        feature_view_name: str,
        feature_columns: List[str],
        max_rows: Optional[int],
        iceberg_table: str = "",
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, any]]:
        """Load training data from Feast feature store.

        Uses Feast's offline store (Spark with Iceberg) to retrieve historical features.
        The Spark configuration including Iceberg JARs is handled by Feast.
        Falls back to direct Iceberg read if Feast registry is unavailable.

        Args:
            repo_path: Path to Feast feature_store.yaml
            feature_view_name: Name of the Feast feature view
            feature_columns: List of feature columns to retrieve
            max_rows: Optional limit per split
            iceberg_table: Fallback Iceberg table name if Feast lookup fails

        Returns:
            Tuple of (splits dict, lineage info dict)
        """
        from datetime import datetime, timedelta
        from pathlib import Path

        # Verify Feast config exists
        feast_config_path = Path(repo_path) / "feature_store.yaml"
        if not feast_config_path.exists():
            raise FileNotFoundError(
                f"Feast config not found at {feast_config_path}. "
                "Ensure the 'feast-config' ConfigMap is mounted at /feast"
            )

        # Try Feast first
        try:
            from feast import FeatureStore

            logger.info(f"Initializing Feast store from: {repo_path}")
            store = FeatureStore(repo_path=repo_path)

            # Get the feature view
            feature_view = store.get_feature_view(feature_view_name)
            logger.info(f"Using feature view: {feature_view.name}")
        except Exception as e:
            # Feast registry not available or feature view not found
            logger.warning(f"Feast feature view lookup failed: {e}")
            if iceberg_table:
                logger.info(f"Falling back to direct Iceberg read from: {iceberg_table}")
                return load_data_from_iceberg_direct(
                    repo_path=repo_path,
                    iceberg_table=iceberg_table,
                    feature_columns=feature_columns,
                    max_rows=max_rows,
                )
            else:
                raise RuntimeError(
                    f"Feast feature view '{feature_view_name}' not found and no fallback Iceberg table provided"
                )

        # Build feature references
        feature_refs = [f"{feature_view_name}:{col}" for col in feature_columns]
        # Also get dataset_split for filtering
        feature_refs.append(f"{feature_view_name}:dataset_split")

        # Create entity dataframe - we need sample_ids to fetch features
        # First, get all sample IDs from the feature view's source
        logger.info("Fetching entity dataframe from Feast offline store...")

        # Use Feast's materialization or direct source query
        # For training, we'll use get_historical_features with a timestamp range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365 * 5)  # Last 5 years of data

        # Get historical features - Feast handles Spark/Iceberg internally
        logger.info(f"Retrieving historical features from {start_time} to {end_time}")

        # For batch retrieval, we need entity_df with sample_id and event_timestamp
        # Let's use the offline store directly to get all data
        try:
            # Method 1: Use Feast's offline store pull
            offline_store = store._get_provider().offline_store
            source = feature_view.batch_source

            # Read from the source directly via Feast's Spark offline store
            logger.info(f"Reading from source: {source.name}")

            # Pull data using Feast's internal Spark session (configured with Iceberg JARs)
            retrieval_job = store._get_provider().offline_store.pull_latest_from_table_or_query(
                config=store.config,
                data_source=source,
                join_key_columns=["sample_id"],
                feature_name_columns=feature_columns + ["dataset_split"],
                timestamp_field=source.timestamp_field,
                created_timestamp_column=None,
                start_date=start_time,
                end_date=end_time,
            )
            df = retrieval_job.to_df()
            logger.info(f"Retrieved {len(df):,} total samples from Feast")

        except Exception as e:
            logger.warning(f"Direct pull failed ({e}), falling back to get_historical_features")

            # Method 2: Fallback - create minimal entity df and fetch features
            # This requires knowing sample IDs ahead of time
            # For now, use Spark directly through Feast's configured session
            from pyspark.sql import SparkSession

            # Get Feast's Spark session (properly configured with Iceberg JARs)
            spark_session = store._get_provider().offline_store._get_spark_session(store.config)

            # Read from the source table
            source_table = source.table
            logger.info(f"Reading from Spark table: {source_table}")

            spark_df = spark_session.read.table(source_table)
            select_cols = ["sample_id", "dataset_split"] + feature_columns
            available = [c for c in select_cols if c in spark_df.columns]
            spark_df = spark_df.select(*available)

            df = spark_df.toPandas()
            logger.info(f"Retrieved {len(df):,} total samples via Spark")

        # Get lineage info
        lineage_info = {
            "feast_feature_view": feature_view_name,
            "feast_project": store.project,
            "feast_repo_path": repo_path,
        }

        # Try to get Iceberg snapshot ID if available
        try:
            if hasattr(source, 'table'):
                lineage_info["iceberg_table"] = source.table
        except Exception:
            pass

        # Split by dataset_split column
        splits = {}
        counts = {}
        for split_name in ["train", "validation", "test"]:
            split_df = df[df["dataset_split"] == split_name].copy()
            if max_rows and len(split_df) > max_rows:
                split_df = split_df.head(max_rows)
            splits[split_name] = split_df
            counts[f"{split_name}_samples"] = len(split_df)
            logger.info(f"  Loaded {split_name}: {len(split_df):,} samples")

        lineage_info.update(counts)

        return splits, lineage_info

    # --- Main training logic ---

    training_start_time = time.time()
    logger.info("Initializing training pipeline...")

    # Set seed for reproducibility
    L.seed_everything(seed)
    logger.debug(f"Random seed set to {seed}")

    # Get lineage info
    logger.info("Fetching data lineage information...")
    lakefs_info = get_lakefs_info(lakefs_endpoint, lakefs_repository, lakefs_ref)
    logger.info(f"LakeFS commit: {lakefs_info.get('lakefs_commit_id', 'unknown')}")

    # Load data from Feast (uses Spark with Iceberg JARs internally)
    # Falls back to direct Iceberg read if Feast registry is unavailable
    # Hadoop catalog uses format: catalog.database.table
    iceberg_table_full = f"{iceberg_catalog}.{iceberg_database}.{source_table}"
    logger.info("Loading data from Feast feature store...")
    data_load_start = time.time()
    splits, data_info = load_data_from_feast(
        repo_path=feast_repo_path,
        feature_view_name=feast_feature_view,
        feature_columns=feature_names,
        max_rows=max_rows,
        iceberg_table=iceberg_table_full,
    )
    data_load_time = time.time() - data_load_start
    logger.info(f"Data loading completed in {data_load_time:.1f}s")
    logger.info(f"Feast feature view: {data_info.get('feast_feature_view', 'unknown')}")

    # Create datasets
    logger.info("Creating PyTorch datasets...")
    train_ds = AutoencoderDataset(splits["train"], feature_names)
    val_ds = AutoencoderDataset(splits["validation"], feature_names, train_ds.mean, train_ds.std)
    test_ds = AutoencoderDataset(splits["test"], feature_names, train_ds.mean, train_ds.std)

    # Log normalization statistics
    logger.debug(f"Feature means: {train_ds.mean[:5]}... (showing first 5)")
    logger.debug(f"Feature stds: {train_ds.std[:5]}... (showing first 5)")

    # Create dataloaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Create model
    logger.info("Initializing model...")
    model = LightningAutoencoder(
        input_dim=len(feature_names),
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        lr=learning_rate,
    )

    # Log model architecture
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
        logger.info(f"MLflow experiment: {mlflow_experiment_name}")

        # Log lineage parameters
        mlflow.log_params({
            **lakefs_info,
            # Iceberg table reference (for lineage)
            "iceberg_table": f"{iceberg_catalog}.{iceberg_database}.{source_table}",
            "iceberg_snapshot_id": data_info.get("iceberg_snapshot_id", ""),
            # Feast feature store reference
            "feast_project": feast_project,
            "feast_feature_view": feast_feature_view,
            "feast_repo_path": feast_repo_path,
            "feature_names": json.dumps(feature_names),
            # Sample counts
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
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })

        # Log monitoring configuration
        mlflow.log_params({
            "log_level": log_level,
            "enable_tensorboard": enable_tensorboard,
            "log_every_n_steps": log_every_n_steps,
            "enable_gradient_logging": enable_gradient_logging,
            "enable_resource_monitoring": enable_resource_monitoring,
        })

        # Log data loading time
        mlflow.log_metric("data_load_time_seconds", data_load_time)

        # Setup loggers
        loggers = []

        # MLflow logger
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_id=run_id,
        )
        loggers.append(mlflow_logger)

        # TensorBoard logger (optional)
        tensorboard_dir = None
        if enable_tensorboard:
            tensorboard_dir = tempfile.mkdtemp(prefix="tensorboard_")
            tb_logger = TensorBoardLogger(
                save_dir=tensorboard_dir,
                name="kronodroid_autoencoder",
                version=run_id[:8],
            )
            loggers.append(tb_logger)
            logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")

        # Setup callbacks
        logger.info("Setting up training callbacks...")
        mlflow_client = mlflow.tracking.MlflowClient()

        callbacks = [
            # Core callbacks
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

            # Custom monitoring callbacks
            TrainingProgressCallback(),
            MLflowMetricsCallback(mlflow_client, run_id),
        ]

        # Add resource monitoring if enabled
        if enable_resource_monitoring:
            callbacks.append(ResourceMonitorCallback(log_every_n_steps=log_every_n_steps))
            logger.info("Resource monitoring enabled")

        # Add gradient monitoring if enabled
        if enable_gradient_logging:
            callbacks.append(GradientMonitorCallback(log_every_n_steps=log_every_n_steps))
            logger.info("Gradient monitoring enabled")

        # Create trainer
        logger.info("Creating Lightning Trainer...")
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

        # Log training time
        mlflow.log_metric("fit_time_seconds", fit_time)

        # Test
        logger.info("=" * 60)
        logger.info("RUNNING TEST EVALUATION")
        logger.info("=" * 60)
        test_start_time = time.time()
        test_results = trainer.test(model, test_loader)
        test_time = time.time() - test_start_time

        test_loss = float(test_results[0]["test_loss"])
        logger.info(f"Test completed in {test_time:.1f}s")
        logger.info(f"Test Loss (MSE): {test_loss:.6f}")

        # Log all test metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_time_seconds", test_time)

        # Log additional test metrics if available
        for key, value in test_results[0].items():
            if key != "test_loss":
                mlflow.log_metric(key, float(value))
                logger.info(f"  {key}: {value:.6f}")

        # Save normalization params
        logger.info("Saving normalization parameters...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "mean": train_ds.mean.tolist(),
                "std": train_ds.std.tolist(),
                "feature_names": feature_names,
            }, f)
            mlflow.log_artifact(f.name, "normalization")
            os.unlink(f.name)

        # Save TensorBoard logs if enabled
        if enable_tensorboard and tensorboard_dir:
            logger.info("Uploading TensorBoard logs to MLflow...")
            mlflow.log_artifacts(tensorboard_dir, "tensorboard")

        # Log training summary
        total_time = time.time() - training_start_time
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"  Data loading: {data_load_time:.1f}s")
        logger.info(f"  Training: {fit_time:.1f}s")
        logger.info(f"  Testing: {test_time:.1f}s")
        logger.info(f"Final test loss: {test_loss:.6f}")
        logger.info(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")

        mlflow.log_metric("total_time_seconds", total_time)
        mlflow.log_metric("best_val_loss", float(trainer.checkpoint_callback.best_model_score))

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

        # Get version
        versions = mlflow_client.search_model_versions(f"name='{mlflow_model_name}'")
        model_version = str(max([int(v.version) for v in versions])) if versions else "1"
        logger.info(f"Registered as {mlflow_model_name} v{model_version}")
        logger.info(f"MLflow run: {mlflow_tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run_id}")

        # Cleanup TensorBoard temp directory
        if tensorboard_dir:
            import shutil
            shutil.rmtree(tensorboard_dir, ignore_errors=True)

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
