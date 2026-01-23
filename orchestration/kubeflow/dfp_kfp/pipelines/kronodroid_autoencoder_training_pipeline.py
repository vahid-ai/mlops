"""Kronodroid Autoencoder Training Pipeline.

This Kubeflow Pipeline trains a PyTorch Lightning autoencoder on the
Kronodroid dataset with full data lineage tracking:
1. Reads features from LakeFS-tracked Iceberg tables
2. Trains autoencoder with train/validation/test splits
3. Logs metrics and data lineage to MLflow
4. Registers model in MLflow Model Registry

Data lineage is tracked via:
- LakeFS commit hash (data version)
- Iceberg snapshot ID (table version)
- Feast feature view name (feature definition)

Usage:
    # Compile the pipeline
    python -m orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_autoencoder_training_pipeline

    # Or programmatically:
    from kfp import compiler
    compiler.Compiler().compile(
        kronodroid_autoencoder_training_pipeline,
        'kronodroid_autoencoder_training_pipeline.yaml'
    )

    # Submit to KFP
    client = kfp.Client()
    client.create_run_from_pipeline_func(
        kronodroid_autoencoder_training_pipeline,
        arguments={...}
    )
"""

import json
from typing import List

from kfp import dsl
from kfp import kubernetes

from orchestration.kubeflow.dfp_kfp.components.train_pytorch_lightning_component import (
    train_kronodroid_autoencoder_op,
)


# Default configuration values
DEFAULT_MLFLOW_TRACKING_URI = "http://mlflow:5000"
DEFAULT_MLFLOW_EXPERIMENT = "kronodroid-autoencoder"
DEFAULT_MLFLOW_MODEL_NAME = "kronodroid_autoencoder"

DEFAULT_LAKEFS_ENDPOINT = "http://lakefs:8000"
DEFAULT_LAKEFS_REPOSITORY = "kronodroid"
DEFAULT_LAKEFS_REF = "main"
DEFAULT_LAKEFS_SECRET = "lakefs-credentials"

DEFAULT_MINIO_ENDPOINT = "http://minio:9000"
DEFAULT_MINIO_SECRET = "minio-credentials"

DEFAULT_ICEBERG_CATALOG = "lakefs"
DEFAULT_ICEBERG_DATABASE = "kronodroid"
DEFAULT_SOURCE_TABLE = "fct_training_dataset"

DEFAULT_FEAST_PROJECT = "dfp"
DEFAULT_FEAST_FEATURE_VIEW = "malware_sample_features"
DEFAULT_FEAST_REPO_PATH = "/feast"  # Path to Feast feature_store.yaml in container

# Default feature names (22 features)
DEFAULT_FEATURE_NAMES: List[str] = [
    f"syscall_{i}_normalized" for i in range(1, 21)
] + ["syscall_total", "syscall_mean"]

# Default model architecture
DEFAULT_LATENT_DIM = 16
DEFAULT_HIDDEN_DIMS = [128, 64]

# Default training config
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SEED = 1337


@dsl.pipeline(
    name="Kronodroid Autoencoder Training Pipeline",
    description="Train a PyTorch Lightning autoencoder on Kronodroid syscall features with MLflow tracking and data lineage",
)
def kronodroid_autoencoder_training_pipeline(
    # MLflow configuration
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    mlflow_model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    # LakeFS configuration
    lakefs_endpoint: str = DEFAULT_LAKEFS_ENDPOINT,
    lakefs_repository: str = DEFAULT_LAKEFS_REPOSITORY,
    lakefs_ref: str = DEFAULT_LAKEFS_REF,
    lakefs_secret_name: str = DEFAULT_LAKEFS_SECRET,
    # MinIO configuration (for MLflow artifacts)
    minio_endpoint: str = DEFAULT_MINIO_ENDPOINT,
    minio_secret_name: str = DEFAULT_MINIO_SECRET,
    # Iceberg configuration
    iceberg_catalog: str = DEFAULT_ICEBERG_CATALOG,
    iceberg_database: str = DEFAULT_ICEBERG_DATABASE,
    source_table: str = DEFAULT_SOURCE_TABLE,
    # Feast/Feature configuration (for data loading and lineage)
    feast_repo_path: str = DEFAULT_FEAST_REPO_PATH,
    feast_project: str = DEFAULT_FEAST_PROJECT,
    feast_feature_view: str = DEFAULT_FEAST_FEATURE_VIEW,
    feature_names_json: str = json.dumps(DEFAULT_FEATURE_NAMES),
    feast_registry_b64: str = "",
    # Model architecture
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims_json: str = json.dumps(DEFAULT_HIDDEN_DIMS),
    # Training configuration
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    max_rows_per_split: int = 0,  # 0 = unlimited
    # Logging and monitoring configuration
    log_level: str = "INFO",  # DEBUG, INFO, WARNING, ERROR
    enable_tensorboard: bool = True,
    log_every_n_steps: int = 10,
    enable_gradient_logging: bool = True,
    enable_resource_monitoring: bool = True,
):
    """Run the Kronodroid autoencoder training pipeline.

    This pipeline trains an autoencoder on the Kronodroid syscall features
    stored in LakeFS-tracked Iceberg tables. It:

    1. Reads training/validation/test splits from fct_training_dataset
    2. Trains a PyTorch Lightning autoencoder with early stopping
    3. Logs all metrics, hyperparameters, and data lineage to MLflow
    4. Registers the trained model in MLflow Model Registry

    Data lineage is tracked via:
    - LakeFS commit hash (exact data version)
    - Iceberg snapshot ID (table version)
    - Feast feature view name (feature definition)
    - Sample counts for each split

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_model_name: Name for model in MLflow registry
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        lakefs_ref: LakeFS branch or commit reference
        lakefs_secret_name: K8s secret with LakeFS credentials
        minio_endpoint: MinIO endpoint for MLflow artifacts
        minio_secret_name: K8s secret with MinIO credentials
        iceberg_catalog: Iceberg catalog name
        iceberg_database: Iceberg database name
        source_table: Source Iceberg table with dataset_split column
        feast_repo_path: Path to Feast feature_store.yaml in container
        feast_project: Feast project name (for lineage tracking)
        feast_feature_view: Feast feature view name (data loading and lineage)
        feature_names_json: JSON-encoded list of feature column names
        latent_dim: Autoencoder latent space dimension
        hidden_dims_json: JSON-encoded list of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Optimizer learning rate
        seed: Random seed for reproducibility
        max_rows_per_split: Row limit per split (0 for unlimited, useful for testing)
        log_level: Python logging level (DEBUG, INFO, WARNING, ERROR)
        enable_tensorboard: Enable TensorBoard logging alongside MLflow
        log_every_n_steps: Log per-step metrics every N training steps
        enable_gradient_logging: Log gradient statistics during training
        enable_resource_monitoring: Log memory and GPU usage metrics
    """
    # Training component
    train_task = train_kronodroid_autoencoder_op(
        # MLflow config
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_model_name=mlflow_model_name,
        # Data config
        lakefs_endpoint=lakefs_endpoint,
        lakefs_repository=lakefs_repository,
        lakefs_ref=lakefs_ref,
        iceberg_catalog=iceberg_catalog,
        iceberg_database=iceberg_database,
        source_table=source_table,
        # Feast config (data loading and lineage)
        feast_repo_path=feast_repo_path,
        feast_project=feast_project,
        feast_feature_view=feast_feature_view,
        feature_names_json=feature_names_json,
        feast_registry_b64=feast_registry_b64,
        # Model config
        latent_dim=latent_dim,
        hidden_dims_json=hidden_dims_json,
        # Training config
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        seed=seed,
        max_rows_per_split=max_rows_per_split,
        # MLflow artifact storage
        minio_endpoint=minio_endpoint,
        # Logging and monitoring
        log_level=log_level,
        enable_tensorboard=enable_tensorboard,
        log_every_n_steps=log_every_n_steps,
        enable_gradient_logging=enable_gradient_logging,
        enable_resource_monitoring=enable_resource_monitoring,
    )

    # Use local image from Kind - don't try to pull from registry
    kubernetes.set_image_pull_policy(train_task, "IfNotPresent")

    # Inject LakeFS credentials from K8s secret
    kubernetes.use_secret_as_env(
        train_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "access-key": "LAKEFS_ACCESS_KEY_ID",
            "secret-key": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # Inject MinIO/S3 credentials for MLflow artifact storage
    kubernetes.use_secret_as_env(
        train_task,
        secret_name=minio_secret_name,
        secret_key_to_env={
            "access-key": "AWS_ACCESS_KEY_ID",
            "secret-key": "AWS_SECRET_ACCESS_KEY",
        },
    )

    # Mount Feast config as a volume at /feast
    # The ConfigMap 'feast-config' must exist in the namespace with key 'feature_store.yaml'
    kubernetes.use_config_map_as_volume(
        train_task,
        config_map_name="feast-config",
        mount_path="/feast",
        optional=False,
    )

    # Note: MLFLOW_S3_ENDPOINT_URL is set to DEFAULT_MINIO_ENDPOINT internally
    # as pipeline parameters cannot be used with set_env_variable in KFP v2

    # Set resource requirements
    train_task.set_memory_request("4Gi")
    train_task.set_memory_limit("8Gi")
    train_task.set_cpu_request("2")
    train_task.set_cpu_limit("4")


@dsl.pipeline(
    name="Kronodroid Full ML Pipeline",
    description="Full pipeline: Spark Iceberg transform -> Autoencoder training with data lineage",
)
def kronodroid_full_ml_pipeline(
    # Spark transformation config
    run_transform: bool = False,
    minio_endpoint: str = DEFAULT_MINIO_ENDPOINT,
    minio_bucket: str = "dlt-data",
    minio_prefix: str = "kronodroid_raw",
    minio_secret_name: str = DEFAULT_MINIO_SECRET,
    spark_image: str = "apache/spark:3.5.0-python3",
    namespace: str = "default",
    service_account: str = "spark",
    # MLflow configuration
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    mlflow_model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    # LakeFS configuration
    lakefs_endpoint: str = DEFAULT_LAKEFS_ENDPOINT,
    lakefs_repository: str = DEFAULT_LAKEFS_REPOSITORY,
    lakefs_ref: str = DEFAULT_LAKEFS_REF,
    lakefs_secret_name: str = DEFAULT_LAKEFS_SECRET,
    # Iceberg configuration
    iceberg_catalog: str = DEFAULT_ICEBERG_CATALOG,
    iceberg_database: str = DEFAULT_ICEBERG_DATABASE,
    source_table: str = DEFAULT_SOURCE_TABLE,
    # Feast/Feature configuration
    feast_repo_path: str = DEFAULT_FEAST_REPO_PATH,
    feast_project: str = DEFAULT_FEAST_PROJECT,
    feast_feature_view: str = DEFAULT_FEAST_FEATURE_VIEW,
    feature_names_json: str = json.dumps(DEFAULT_FEATURE_NAMES),
    # Model architecture
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims_json: str = json.dumps(DEFAULT_HIDDEN_DIMS),
    # Training configuration
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    max_rows_per_split: int = 0,
    # Logging and monitoring configuration
    log_level: str = "INFO",
    enable_tensorboard: bool = True,
    log_every_n_steps: int = 10,
    enable_gradient_logging: bool = True,
    enable_resource_monitoring: bool = True,
):
    """Full Kronodroid ML pipeline: transform (optional) -> train -> register.

    This pipeline optionally runs the Spark transformation step first,
    then trains the autoencoder on the resulting Iceberg tables.

    Args:
        run_transform: Whether to run Spark transformation first
        minio_endpoint: MinIO endpoint URL
        minio_bucket: MinIO bucket for raw data
        minio_prefix: Path prefix for raw data
        minio_secret_name: K8s secret with MinIO credentials
        spark_image: Spark executor image
        namespace: K8s namespace for Spark
        service_account: K8s service account for Spark
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_model_name: Name for model in MLflow registry
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        lakefs_ref: LakeFS branch or commit reference
        lakefs_secret_name: K8s secret with LakeFS credentials
        iceberg_catalog: Iceberg catalog name
        iceberg_database: Iceberg database name
        source_table: Source Iceberg table
        feast_repo_path: Path to Feast feature_store.yaml in container
        feast_project: Feast project name
        feast_feature_view: Feast feature view name (data loading and lineage)
        feature_names_json: JSON-encoded list of feature names
        latent_dim: Autoencoder latent dimension
        hidden_dims_json: JSON-encoded hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum epochs
        learning_rate: Learning rate
        seed: Random seed
        max_rows_per_split: Row limit per split
    """
    # Step 1: Optional Spark transformation
    # This would use the existing spark_kronodroid_iceberg_op
    # if run_transform:
    #     from orchestration.kubeflow.dfp_kfp.components.spark_kronodroid_iceberg_component import (
    #         spark_kronodroid_iceberg_op,
    #     )
    #     spark_task = spark_kronodroid_iceberg_op(...)
    #     # ... configure and run

    # Step 2: Training
    train_task = train_kronodroid_autoencoder_op(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_model_name=mlflow_model_name,
        lakefs_endpoint=lakefs_endpoint,
        lakefs_repository=lakefs_repository,
        lakefs_ref=lakefs_ref,
        iceberg_catalog=iceberg_catalog,
        iceberg_database=iceberg_database,
        source_table=source_table,
        feast_repo_path=feast_repo_path,
        feast_project=feast_project,
        feast_feature_view=feast_feature_view,
        feature_names_json=feature_names_json,
        latent_dim=latent_dim,
        hidden_dims_json=hidden_dims_json,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        seed=seed,
        max_rows_per_split=max_rows_per_split,
        minio_endpoint=minio_endpoint,
        # Logging and monitoring
        log_level=log_level,
        enable_tensorboard=enable_tensorboard,
        log_every_n_steps=log_every_n_steps,
        enable_gradient_logging=enable_gradient_logging,
        enable_resource_monitoring=enable_resource_monitoring,
    )

    # Use local image from Kind - don't try to pull from registry
    kubernetes.set_image_pull_policy(train_task, "IfNotPresent")

    # Configure credentials
    kubernetes.use_secret_as_env(
        train_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "access-key": "LAKEFS_ACCESS_KEY_ID",
            "secret-key": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )
    kubernetes.use_secret_as_env(
        train_task,
        secret_name=minio_secret_name,
        secret_key_to_env={
            "access-key": "AWS_ACCESS_KEY_ID",
            "secret-key": "AWS_SECRET_ACCESS_KEY",
        },
    )

    # Mount Feast config as a volume at /feast
    kubernetes.use_config_map_as_volume(
        train_task,
        config_map_name="feast-config",
        mount_path="/feast",
        optional=False,
    )

    train_task.set_memory_request("4Gi")
    train_task.set_memory_limit("8Gi")
    train_task.set_cpu_request("2")
    train_task.set_cpu_limit("4")


def compile_training_pipeline(
    output_path: str = "kronodroid_autoencoder_training_pipeline.yaml",
) -> str:
    """Compile the training pipeline to YAML.

    Args:
        output_path: Path for the compiled pipeline YAML

    Returns:
        Path to the compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=kronodroid_autoencoder_training_pipeline,
        package_path=output_path,
    )
    print(f"Training pipeline compiled to: {output_path}")
    return output_path


def compile_full_ml_pipeline(
    output_path: str = "kronodroid_full_ml_pipeline.yaml",
) -> str:
    """Compile the full ML pipeline to YAML.

    Args:
        output_path: Path for the compiled pipeline YAML

    Returns:
        Path to the compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=kronodroid_full_ml_pipeline,
        package_path=output_path,
    )
    print(f"Full ML pipeline compiled to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile Kronodroid autoencoder training pipelines"
    )
    parser.add_argument(
        "--output",
        default="kronodroid_autoencoder_training_pipeline.yaml",
        help="Output path for compiled pipeline",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Compile the full ML pipeline (transform + train)",
    )

    args = parser.parse_args()

    if args.full:
        compile_full_ml_pipeline(args.output)
    else:
        compile_training_pipeline(args.output)
