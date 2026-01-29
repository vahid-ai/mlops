"""Kronodroid Autoencoder Training Pipeline.

Kubeflow Pipeline for training a PyTorch Lightning autoencoder on the Kronodroid 
dataset with proper train/validation/test splits and MLflow lineage tracking.

Usage:
    # Compile the pipeline (from project root)
    uv run python orchestration/kubeflow/dfp_kfp/pipelines/kronodroid_autoencoder_pipeline.py

    # Or use the convenience function
    from orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_autoencoder_pipeline import (
        compile_pipeline,
    )
    compile_pipeline('kronodroid_autoencoder_pipeline.yaml')
"""

import sys
from pathlib import Path

# Add project root to path when running directly (before any orchestration imports)
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kfp import dsl
from kfp import kubernetes

from orchestration.kubeflow.dfp_kfp.components.train_autoencoder_component import (
    train_kronodroid_autoencoder_op,
)


# Default configuration values
DEFAULT_MINIO_ENDPOINT = "http://minio:9000"
DEFAULT_LAKEFS_ENDPOINT = "http://lakefs:8000"
DEFAULT_LAKEFS_REPOSITORY = "kronodroid"
DEFAULT_LAKEFS_REF = "main"
DEFAULT_MLFLOW_TRACKING_URI = "http://mlflow:5000"
DEFAULT_MLFLOW_EXPERIMENT = "kronodroid-autoencoder"
DEFAULT_MLFLOW_MODEL_NAME = "kronodroid_autoencoder"
DEFAULT_FEAST_REPO_PATH = "/feast"

# Default syscall feature names
DEFAULT_FEATURE_NAMES = [
    "syscall_1_normalized", "syscall_2_normalized", "syscall_3_normalized",
    "syscall_4_normalized", "syscall_5_normalized", "syscall_6_normalized",
    "syscall_7_normalized", "syscall_8_normalized", "syscall_9_normalized",
    "syscall_10_normalized", "syscall_11_normalized", "syscall_12_normalized",
    "syscall_13_normalized", "syscall_14_normalized", "syscall_15_normalized",
    "syscall_16_normalized", "syscall_17_normalized", "syscall_18_normalized",
    "syscall_19_normalized", "syscall_20_normalized",
    "syscall_total", "syscall_mean",
]


@dsl.pipeline(
    name="Kronodroid Autoencoder Training Pipeline",
    description="Train a PyTorch Lightning autoencoder on Kronodroid syscall features with MLflow tracking and data lineage",
)
def kronodroid_autoencoder_pipeline(
    # MLflow configuration
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    mlflow_model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    # LakeFS configuration
    lakefs_endpoint: str = DEFAULT_LAKEFS_ENDPOINT,
    lakefs_repository: str = DEFAULT_LAKEFS_REPOSITORY,
    lakefs_ref: str = DEFAULT_LAKEFS_REF,
    # Iceberg configuration
    iceberg_catalog: str = "lakefs",
    iceberg_database: str = "kronodroid",
    source_table: str = "fct_training_dataset",
    # Feast configuration (for Spark config)
    feast_repo_path: str = DEFAULT_FEAST_REPO_PATH,
    feast_project: str = "dfp",
    feast_feature_view: str = "malware_sample_features",
    feature_names_json: str = "",  # Will use default if empty
    # Model configuration
    latent_dim: int = 16,
    hidden_dims_json: str = "[128, 64]",
    # Training configuration
    batch_size: int = 512,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    seed: int = 1337,
    max_rows_per_split: int = 0,  # 0 = unlimited
    # MinIO configuration
    minio_endpoint: str = DEFAULT_MINIO_ENDPOINT,
    # Logging configuration
    log_level: str = "INFO",
    enable_tensorboard: bool = True,
    log_every_n_steps: int = 10,
    enable_gradient_logging: bool = True,
    enable_resource_monitoring: bool = True,
):
    """Run the Kronodroid autoencoder training pipeline.

    This pipeline trains a PyTorch Lightning autoencoder on the Kronodroid
    dataset with:
    - Train/validation/test splits from dataset_split column
    - MLflow tracking with complete data lineage
    - LakeFS commit tracking for reproducibility

    Credentials are injected via K8s secrets (lakefs-credentials, minio-credentials).

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
        feature_names_json: JSON list of feature column names
        latent_dim: Autoencoder latent dimension
        hidden_dims_json: JSON list of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Optimizer learning rate
        seed: Random seed for reproducibility
        max_rows_per_split: Row limit per split (0 = unlimited)
        minio_endpoint: MinIO/S3 endpoint for MLflow artifacts
        log_level: Python logging level
        enable_tensorboard: Enable TensorBoard logging
        log_every_n_steps: Log per-step metrics every N steps
        enable_gradient_logging: Log gradient statistics
        enable_resource_monitoring: Log memory and GPU usage
    """
    import json
    
    # The component handles default feature names if feature_names_json is empty

    # Training task
    train_task = train_kronodroid_autoencoder_op(
        # MLflow config
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_model_name=mlflow_model_name,
        # LakeFS/Iceberg config
        lakefs_endpoint=lakefs_endpoint,
        lakefs_repository=lakefs_repository,
        lakefs_ref=lakefs_ref,
        iceberg_catalog=iceberg_catalog,
        iceberg_database=iceberg_database,
        source_table=source_table,
        # Feast config
        feast_repo_path=feast_repo_path,
        feast_project=feast_project,
        feast_feature_view=feast_feature_view,
        feature_names_json=feature_names_json,
        # Model config
        latent_dim=latent_dim,
        hidden_dims_json=hidden_dims_json,
        # Training config
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        seed=seed,
        max_rows_per_split=max_rows_per_split,
        # MinIO config
        minio_endpoint=minio_endpoint,
        # Logging config
        log_level=log_level,
        enable_tensorboard=enable_tensorboard,
        log_every_n_steps=log_every_n_steps,
        enable_gradient_logging=enable_gradient_logging,
        enable_resource_monitoring=enable_resource_monitoring,
    ).set_memory_limit("2Gi").set_cpu_limit("4")

    # Configure credentials via K8s secrets
    # Use static secret names (not pipeline parameters) to work with current KFP backend
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name="lakefs-credentials",
        secret_key_to_env={
            "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
            "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name="minio-credentials",
        secret_key_to_env={
            "MINIO_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "MINIO_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
    )

    # Set image pull policy to IfNotPresent for local images in Kind
    kubernetes.set_image_pull_policy(train_task, "IfNotPresent")


def compile_pipeline(output_path: str = "kronodroid_autoencoder_pipeline.yaml") -> str:
    """Compile the pipeline to YAML.

    Args:
        output_path: Path for the compiled pipeline YAML

    Returns:
        Path to the compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=kronodroid_autoencoder_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile Kronodroid Autoencoder Training pipeline")
    parser.add_argument(
        "--output",
        default="kronodroid_autoencoder_pipeline.yaml",
        help="Output path for compiled pipeline",
    )

    args = parser.parse_args()
    compile_pipeline(args.output)
