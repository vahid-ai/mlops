"""Kronodroid Autoencoder Training Component.

KFP v2 component that trains a PyTorch Lightning autoencoder on Kronodroid 
syscall features with MLflow tracking and data lineage.

The component:
1. Loads data from Iceberg via Spark (using Feast config for Spark settings)
2. Splits data using dataset_split column (train/validation/test)
3. Trains PyTorch Lightning autoencoder with monitoring callbacks
4. Logs lineage to MLflow: LakeFS commit, Iceberg table, split statistics
5. Registers model in MLflow Model Registry
"""

import json
import logging
import os
import sys
import tempfile
import time
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from kfp import dsl


@dsl.component(
    base_image="dfp-kfp-training:latest",
    packages_to_install=[
        "mlflow>=2.9.0",
        "lightning>=2.0",
        "torch>=2.0",
        "pyspark>=3.5.0",
        "pandas>=2.0",
        "numpy>=1.24",
        "psutil>=5.9",
        "requests>=2.28",
        "PyYAML>=6.0",
    ],
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
    # Feast config for Spark settings
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

    This component orchestrates the full training workflow:
    1. Connects to LakeFS-backed Iceberg tables via Spark
    2. Loads train/validation/test splits using dataset_split column
    3. Trains PyTorch Lightning autoencoder with comprehensive monitoring
    4. Logs metrics, data lineage, and training statistics to MLflow
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
        feast_repo_path: Path to Feast feature_store.yaml (for Spark config)
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
        enable_tensorboard: Enable TensorBoard logging
        log_every_n_steps: Log per-step metrics every N steps
        enable_gradient_logging: Log gradient statistics
        enable_resource_monitoring: Log memory and GPU usage

    Returns:
        NamedTuple with run info, model info, metrics, and lineage
    """
    import json
    import logging
    import os
    import re
    import sys
    import tempfile
    import time
    from collections import namedtuple
    from datetime import datetime
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import lightning as L
    import mlflow
    import mlflow.pytorch
    import numpy as np
    import pandas as pd
    import psutil
    import requests
    import torch
    import yaml
    from lightning.pytorch.callbacks import (
        Callback,
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
    from pyspark.sql import SparkSession
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    # --- Setup logging ---
    def setup_logging(level: str) -> logging.Logger:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger = logging.getLogger("kronodroid-autoencoder")
        logger.setLevel(numeric_level)
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
    if not feature_names_json:
        # Default syscall features if none provided
        feature_names = [
            "syscall_1_normalized", "syscall_2_normalized", "syscall_3_normalized",
            "syscall_4_normalized", "syscall_5_normalized", "syscall_6_normalized",
            "syscall_7_normalized", "syscall_8_normalized", "syscall_9_normalized",
            "syscall_10_normalized", "syscall_11_normalized", "syscall_12_normalized",
            "syscall_13_normalized", "syscall_14_normalized", "syscall_15_normalized",
            "syscall_16_normalized", "syscall_17_normalized", "syscall_18_normalized",
            "syscall_19_normalized", "syscall_20_normalized",
            "syscall_total", "syscall_mean",
        ]
    else:
        feature_names = json.loads(feature_names_json)
        
    hidden_dims = tuple(json.loads(hidden_dims_json))
    max_rows = max_rows_per_split if max_rows_per_split > 0 else None

    # Set MLflow S3 endpoint
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
    logger.info(f"Training: batch_size={batch_size}, max_epochs={max_epochs}, lr={learning_rate}")
    logger.info("=" * 60)

    # --- Custom Callbacks ---
    class ResourceMonitorCallback(Callback):
        def __init__(self, log_every_n: int = 10):
            super().__init__()
            self.log_every_n = log_every_n

        def _get_memory_stats(self) -> Dict[str, float]:
            stats = {}
            process = psutil.Process()
            mem_info = process.memory_info()
            stats["memory_rss_mb"] = mem_info.rss / (1024 * 1024)
            stats["memory_percent"] = process.memory_percent()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    stats[f"gpu_{i}_allocated_mb"] = allocated
            return stats

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx % self.log_every_n == 0:
                stats = self._get_memory_stats()
                for key, value in stats.items():
                    pl_module.log(f"resource/{key}", value, on_step=True, on_epoch=False)

    class GradientMonitorCallback(Callback):
        def __init__(self, log_every_n: int = 10):
            super().__init__()
            self.log_every_n = log_every_n

        def on_after_backward(self, trainer, pl_module):
            if trainer.global_step % self.log_every_n == 0:
                grad_norms = []
                for name, param in pl_module.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                if grad_norms:
                    pl_module.log("gradient/norm_mean", np.mean(grad_norms), on_step=True, on_epoch=False)
                    pl_module.log("gradient/norm_max", np.max(grad_norms), on_step=True, on_epoch=False)

    class TrainingProgressCallback(Callback):
        def __init__(self):
            super().__init__()
            self.epoch_start_time = None
            self.training_start_time = None
            self.epoch_samples = 0

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

        def on_train_epoch_end(self, trainer, pl_module):
            epoch_time = time.time() - self.epoch_start_time
            samples_per_second = self.epoch_samples / epoch_time if epoch_time > 0 else 0
            pl_module.log("timing/epoch_seconds", epoch_time, on_step=False, on_epoch=True)
            pl_module.log("throughput/samples_per_second", samples_per_second, on_step=False, on_epoch=True)
            train_loss = trainer.callback_metrics.get("train_loss", 0)
            val_loss = trainer.callback_metrics.get("val_loss", 0)
            logger.info(
                f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} completed | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"time={epoch_time:.1f}s | throughput={samples_per_second:.0f} samples/s"
            )

    # --- LightningAutoencoder ---
    class LightningAutoencoder(L.LightningModule):
        def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Tuple[int, ...], lr: float):
            super().__init__()
            self.save_hyperparameters()
            self.lr = lr

            # Build encoder
            encoder_layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
                prev_dim = h_dim
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            # Build decoder
            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
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
            return mse_loss

        def training_step(self, batch, batch_idx):
            return self._shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self._shared_step(batch, "val")

        def test_step(self, batch, batch_idx):
            loss = self._shared_step(batch, "test")
            with torch.no_grad():
                x_hat = self(batch)
                sample_errors = ((x_hat - batch) ** 2).mean(dim=1)
                self.log("test_error_mean", sample_errors.mean(), on_epoch=True)
                self.log("test_error_std", sample_errors.std(), on_epoch=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

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

    # --- Get LakeFS commit info ---
    def get_lakefs_info(endpoint: str, repo: str, ref: str) -> Dict[str, str]:
        access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")
        api_base = endpoint.rstrip("/")
        if "localhost" in api_base or "127.0.0.1" in api_base:
            api_base = "http://lakefs.dfp.svc.cluster.local:8000"
        elif api_base == "http://lakefs:8000":
            api_base = "http://lakefs.dfp.svc.cluster.local:8000"
            
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
    def load_data_from_iceberg(
        repo_path: str,
        iceberg_table: str,
        feature_columns: List[str],
        max_rows: Optional[int],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, any]]:
        """Load training data from Iceberg table using Spark config.
        
        Attempts to read Spark config from Feast feature_store.yaml if it exists,
        otherwise uses projects defaults.
        """
        feast_config_path = Path(repo_path) / "feature_store.yaml"
        
        if feast_config_path.exists():
            logger.info(f"Loading Feast config from: {feast_config_path}")
            with open(feast_config_path) as f:
                feast_config = yaml.safe_load(f)
            spark_conf = feast_config.get("offline_store", {}).get("spark_conf", {})
        else:
            logger.info(f"Feast config not found at {feast_config_path}, using internal defaults")
            # Default Spark configuration for Iceberg + LakeFS
            spark_conf = {
                "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
                "spark.sql.catalog.lakefs": "org.apache.iceberg.spark.SparkCatalog",
                "spark.sql.catalog.lakefs.type": "hadoop",
                "spark.sql.catalog.lakefs.warehouse": f"s3a://{lakefs_repository}/{lakefs_ref}/iceberg",
                "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                "spark.hadoop.fs.s3a.path.style.access": "true",
                "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
                # Maven packages
                "spark.jars.packages": "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,org.apache.iceberg:iceberg-aws:1.5.2,org.apache.hadoop:hadoop-aws:3.3.4,software.amazon.awssdk:bundle:2.20.160"
            }

        logger.info("Creating Spark session with Iceberg JARs")

        # Get credentials
        lakefs_access_key = os.environ.get("LAKEFS_ACCESS_KEY_ID", "")
        lakefs_secret_key = os.environ.get("LAKEFS_SECRET_ACCESS_KEY", "")

        builder = SparkSession.builder.appName("kronodroid-training")
        for key, value in spark_conf.items():
            if isinstance(value, str) and "${" in value:
                for match in re.finditer(r'\$\{(\w+)\}', value):
                    env_var = match.group(1)
                    env_val = os.environ.get(env_var, "")
                    value = value.replace(f"${{{env_var}}}", env_val)
            builder = builder.config(key, str(value))

        # Inject LakeFS credentials and endpoint
        if lakefs_access_key and lakefs_secret_key:
            logger.info("Injecting LakeFS credentials into Spark config")
            builder = builder.config("spark.hadoop.fs.s3a.access.key", lakefs_access_key)
            builder = builder.config("spark.hadoop.fs.s3a.secret.key", lakefs_secret_key)
            
            # Use provided lakefs_endpoint for s3a
            # Ensure it's reachable from inside the cluster
            s3a_endpoint = lakefs_endpoint
            if "localhost" in s3a_endpoint or "127.0.0.1" in s3a_endpoint:
                s3a_endpoint = "http://lakefs.dfp.svc.cluster.local:8000"
            elif s3a_endpoint == "http://lakefs:8000":
                s3a_endpoint = "http://lakefs.dfp.svc.cluster.local:8000"
                
            builder = builder.config("spark.hadoop.fs.s3a.endpoint", s3a_endpoint)
            builder = builder.config("spark.hadoop.fs.s3a.bucket.kronodroid.endpoint", s3a_endpoint)

        spark_session = builder.getOrCreate()
        logger.info(f"Spark session created, reading from: {iceberg_table}")

        spark_df = spark_session.read.table(iceberg_table)
        select_cols = ["sample_id", "dataset_split"] + feature_columns
        available = [c for c in select_cols if c in spark_df.columns]
        spark_df = spark_df.select(*available)

        df = spark_df.toPandas()
        logger.info(f"Retrieved {len(df):,} total samples")

        # Split by dataset_split column (from versioning-datasets.md pattern)
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

    # --- Main training logic ---
    training_start_time = time.time()
    logger.info("Initializing training pipeline...")

    L.seed_everything(seed)
    logger.debug(f"Random seed set to {seed}")

    # Get lineage info
    logger.info("Fetching data lineage information...")
    lakefs_info = get_lakefs_info(lakefs_endpoint, lakefs_repository, lakefs_ref)
    logger.info(f"LakeFS commit: {lakefs_info.get('lakefs_commit_id', 'unknown')}")

    # Load data from Iceberg
    iceberg_table_full = f"{iceberg_catalog}.{iceberg_database}.{source_table}"
    logger.info("Loading data from Iceberg table...")
    data_load_start = time.time()
    splits, data_info = load_data_from_iceberg(
        repo_path=feast_repo_path,
        iceberg_table=iceberg_table_full,
        feature_columns=feature_names,
        max_rows=max_rows,
    )
    data_load_time = time.time() - data_load_start
    logger.info(f"Data loading completed in {data_load_time:.1f}s")

    # Create datasets
    logger.info("Creating PyTorch datasets...")
    train_ds = AutoencoderDataset(splits["train"], feature_names)
    val_ds = AutoencoderDataset(splits["validation"], feature_names, train_ds.mean, train_ds.std)
    test_ds = AutoencoderDataset(splits["test"], feature_names, train_ds.mean, train_ds.std)

    # Create dataloaders
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

        # Log lineage parameters (following versioning-datasets.md pattern)
        mlflow.log_params({
            **lakefs_info,
            "iceberg_table": f"{iceberg_catalog}.{iceberg_database}.{source_table}",
            "iceberg_snapshot_id": data_info.get("iceberg_snapshot_id", ""),
            "feast_project": feast_project,
            "feast_feature_view": feast_feature_view,
            "feast_repo_path": feast_repo_path,
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
            "total_parameters": total_params,
        })

        mlflow.log_metric("data_load_time_seconds", data_load_time)

        # Setup loggers
        loggers = [
            MLFlowLogger(
                experiment_name=mlflow_experiment_name,
                tracking_uri=mlflow_tracking_uri,
                run_id=run_id,
            )
        ]

        tensorboard_dir = None
        if enable_tensorboard:
            tensorboard_dir = tempfile.mkdtemp(prefix="tensorboard_")
            loggers.append(TensorBoardLogger(save_dir=tensorboard_dir, name="kronodroid_autoencoder", version=run_id[:8]))
            logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")

        # Setup callbacks
        logger.info("Setting up training callbacks...")
        callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch:02d}-{val_loss:.4f}"),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
            TrainingProgressCallback(),
        ]

        if enable_resource_monitoring:
            callbacks.append(ResourceMonitorCallback(log_every_n=log_every_n_steps))
        if enable_gradient_logging:
            callbacks.append(GradientMonitorCallback(log_every_n=log_every_n_steps))

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
        test_results = trainer.test(model, test_loader)
        test_loss = float(test_results[0]["test_loss"])
        logger.info(f"Test Loss (MSE): {test_loss:.6f}")

        mlflow.log_metric("test_loss", test_loss)

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

        # Upload TensorBoard logs
        if enable_tensorboard and tensorboard_dir:
            mlflow.log_artifacts(tensorboard_dir, "tensorboard")

        # Log summary
        total_time = time.time() - training_start_time
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
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
        lakefs_commit_id=lakefs_info.get("lakefs_commit_id", ""),
        iceberg_snapshot_id=data_info.get("iceberg_snapshot_id", ""),
        train_samples=data_info.get("train_samples", 0),
        validation_samples=data_info.get("validation_samples", 0),
        test_samples=data_info.get("test_samples", 0),
    )
