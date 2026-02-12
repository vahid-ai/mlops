"""Kronodroid Autoencoder Training Pipeline.

This Kubeflow Pipeline trains a PyTorch Lightning autoencoder on the Kronodroid
malware detection dataset with comprehensive data lineage tracking.

Data Versioning Strategy (per versioning-datasets.md):
- Uses `dataset_split` column in source table (no data duplication)
- Logs LakeFS commit ID to MLflow for exact data version tracking
- Creates LakeFS tag on successful training for permanent reference
- Tracks Feast feature view for feature engineering lineage

Pipeline Steps:
1. Train autoencoder on LakeFS-versioned Iceberg table data
2. Register model in MLflow with full lineage
3. (Optional) Tag LakeFS commit for reproducibility

Usage:
    # Compile the pipeline
    from kfp import compiler
    compiler.Compiler().compile(
        kronodroid_autoencoder_training_pipeline,
        'kronodroid_autoencoder_training_pipeline.yaml'
    )

    # Submit to KFP
    client = kfp.Client()
    client.create_run_from_pipeline_func(
        kronodroid_autoencoder_training_pipeline,
        arguments={
            'lakefs_ref': 'main',
            'max_epochs': 20,
        }
    )
"""

import json

from kfp import dsl, kubernetes

from orchestration.kubeflow.dfp_kfp.components.train_kronodroid_autoencoder_component import (
    train_kronodroid_autoencoder_op,
    lakefs_tag_model_data_op,
)
from orchestration.kubeflow.dfp_kfp.config import (
    DEFAULT_FEAST_CONFIGMAP_NAME,
    DEFAULT_FEAST_MOUNT_PATH,
    DEFAULT_ICEBERG_CATALOG,
    DEFAULT_ICEBERG_DATABASE,
    DEFAULT_ICEBERG_SOURCE_TABLE,
    DEFAULT_LAKEFS_BRANCH,
    DEFAULT_LAKEFS_ENDPOINT,
    DEFAULT_LAKEFS_REPOSITORY,
    DEFAULT_LAKEFS_SECRET_NAME,
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_MINIO_ENDPOINT,
    DEFAULT_MINIO_SECRET_NAME,
)
from orchestration.kubeflow.dfp_kfp.k8s_utils import (
    mount_feast_repo,
    use_lakefs_credentials,
    use_minio_credentials,
)


DEFAULT_MLFLOW_EXPERIMENT = "kronodroid-autoencoder"
DEFAULT_MLFLOW_MODEL_NAME = "kronodroid_autoencoder"
DEFAULT_FEAST_PROJECT = "dfp"
DEFAULT_FEAST_FEATURE_VIEW = "malware_sample_features"
DEFAULT_FEAST_REPO_PATH = DEFAULT_FEAST_MOUNT_PATH

# Default feature names (22 syscall features)
DEFAULT_FEATURE_NAMES = [
    "syscall_1_normalized",
    "syscall_2_normalized",
    "syscall_3_normalized",
    "syscall_4_normalized",
    "syscall_5_normalized",
    "syscall_6_normalized",
    "syscall_7_normalized",
    "syscall_8_normalized",
    "syscall_9_normalized",
    "syscall_10_normalized",
    "syscall_11_normalized",
    "syscall_12_normalized",
    "syscall_13_normalized",
    "syscall_14_normalized",
    "syscall_15_normalized",
    "syscall_16_normalized",
    "syscall_17_normalized",
    "syscall_18_normalized",
    "syscall_19_normalized",
    "syscall_20_normalized",
    "syscall_total",
    "syscall_mean",
]


@dsl.pipeline(
    name="Kronodroid Autoencoder Training Pipeline",
    description="Train a PyTorch Lightning autoencoder on Kronodroid syscall features with MLflow tracking and LakeFS data lineage",
)
def kronodroid_autoencoder_training_pipeline(
    # MLflow configuration
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    mlflow_model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    # Data source configuration (LakeFS/Iceberg)
    lakefs_endpoint: str = DEFAULT_LAKEFS_ENDPOINT,
    lakefs_repository: str = DEFAULT_LAKEFS_REPOSITORY,
    lakefs_ref: str = DEFAULT_LAKEFS_BRANCH,
    lakefs_secret_name: str = DEFAULT_LAKEFS_SECRET_NAME,
    iceberg_catalog: str = DEFAULT_ICEBERG_CATALOG,
    iceberg_database: str = DEFAULT_ICEBERG_DATABASE,
    source_table: str = DEFAULT_ICEBERG_SOURCE_TABLE,
    # Feast configuration (for lineage tracking)
    feast_repo_path: str = DEFAULT_FEAST_REPO_PATH,
    feast_project: str = DEFAULT_FEAST_PROJECT,
    feast_feature_view: str = DEFAULT_FEAST_FEATURE_VIEW,
    feature_names_json: str = json.dumps(DEFAULT_FEATURE_NAMES),
    # Model architecture
    latent_dim: int = 16,
    hidden_dims_json: str = "[128, 64]",
    # Training configuration
    batch_size: int = 512,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    seed: int = 1337,
    max_rows_per_split: int = 0,
    # MLflow artifact storage (MinIO)
    minio_endpoint: str = DEFAULT_MINIO_ENDPOINT,
    minio_secret_name: str = DEFAULT_MINIO_SECRET_NAME,
    # Monitoring options
    log_level: str = "INFO",
    enable_tensorboard: bool = True,
    log_every_n_steps: int = 10,
    enable_gradient_logging: bool = True,
    enable_resource_monitoring: bool = True,
    # Pipeline options
    create_lakefs_tag: bool = True,
):
    """Train a Kronodroid autoencoder with full data lineage tracking.

    This pipeline follows the versioning-datasets.md pattern for reproducible ML:

    1. Data Versioning:
       - Uses LakeFS-versioned Iceberg tables as the data source
       - Resolves the lakefs_ref to an exact commit ID
       - Logs the commit ID to MLflow for lineage

    2. Split Strategy:
       - Uses `dataset_split` column in the source table
       - Splits: train/validation/test (no data duplication)

    3. Lineage Tracking:
       - LakeFS commit ID logged to MLflow
       - Feast feature view reference logged
       - Iceberg snapshot ID logged
       - All hyperparameters and metrics logged

    4. Reproducibility:
       - Optional LakeFS tag creation for permanent data reference
       - Tag format: model-{name}-v{version}-data

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_model_name: Name for model in MLflow registry
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        lakefs_ref: LakeFS branch, tag, or commit reference
        lakefs_secret_name: K8s secret with LakeFS credentials
        iceberg_catalog: Iceberg catalog name
        iceberg_database: Iceberg database name
        source_table: Source Iceberg table (must have dataset_split column)
        feast_repo_path: Path to Feast feature_store.yaml in container
        feast_project: Feast project name (for lineage)
        feast_feature_view: Feast feature view name (for lineage)
        feature_names_json: JSON-encoded list of feature column names
        latent_dim: Autoencoder latent space dimension
        hidden_dims_json: JSON-encoded list of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Optimizer learning rate
        seed: Random seed for reproducibility
        max_rows_per_split: Row limit per split (0=unlimited, useful for testing)
        minio_endpoint: MinIO endpoint for MLflow artifacts
        minio_secret_name: K8s secret with MinIO credentials
        log_level: Python logging level (DEBUG, INFO, WARNING, ERROR)
        enable_tensorboard: Enable TensorBoard logging alongside MLflow
        log_every_n_steps: Log per-step metrics every N steps
        enable_gradient_logging: Log gradient statistics during training
        enable_resource_monitoring: Log memory/GPU usage metrics
        create_lakefs_tag: Create LakeFS tag for training data version
    """
    # Step 1: Train the autoencoder
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
        # MLflow artifact storage
        minio_endpoint=minio_endpoint,
        # Monitoring
        log_level=log_level,
        enable_tensorboard=enable_tensorboard,
        log_every_n_steps=log_every_n_steps,
        enable_gradient_logging=enable_gradient_logging,
        enable_resource_monitoring=enable_resource_monitoring,
    )

    # Set resource limits
    train_task.set_memory_limit("8Gi")
    train_task.set_memory_request("4Gi")
    train_task.set_cpu_limit("4")
    train_task.set_cpu_request("2")

    # Use local image (for Kind clusters)
    kubernetes.set_image_pull_policy(task=train_task, policy="IfNotPresent")

    # Inject LakeFS credentials
    use_lakefs_credentials(train_task, secret_name=lakefs_secret_name)

    # Inject MinIO credentials (for MLflow artifact storage)
    use_minio_credentials(train_task, secret_name=minio_secret_name)

    # Mount Feast config
    # KFP kubernetes volume helpers require a static mount path (not a pipeline parameter).
    mount_feast_repo(
        train_task,
        config_map_name=DEFAULT_FEAST_CONFIGMAP_NAME,
        mount_path=DEFAULT_FEAST_MOUNT_PATH,
    )

    # Step 2: Optionally create LakeFS tag for reproducibility
    # Note: Using hardcoded secret name because KFP v2 doesn't support
    # pipeline parameters inside conditional blocks for kubernetes SDK calls
    with dsl.If(create_lakefs_tag == True):
        tag_task = lakefs_tag_model_data_op(
            lakefs_endpoint=lakefs_endpoint,
            lakefs_repository=lakefs_repository,
            lakefs_commit_id=train_task.outputs["lakefs_commit_id"],
            model_name=train_task.outputs["model_name"],
            model_version=train_task.outputs["model_version"],
        )

        tag_task.set_memory_limit("512Mi")
        tag_task.set_memory_request("256Mi")
        tag_task.set_cpu_limit("500m")

        # Inject LakeFS credentials (hardcoded secret name for KFP v2 compatibility)
        use_lakefs_credentials(tag_task, secret_name=DEFAULT_LAKEFS_SECRET_NAME)


@dsl.pipeline(
    name="Kronodroid Full Training Pipeline",
    description="End-to-end: Data transform -> Autoencoder training -> Model registration with lineage",
)
def kronodroid_full_training_pipeline(
    # Data transformation options
    run_data_transform: bool = False,
    # MinIO configuration (for raw data)
    minio_endpoint: str = DEFAULT_MINIO_ENDPOINT,
    minio_bucket: str = "dlt-data",
    minio_prefix: str = "kronodroid_raw",
    minio_secret_name: str = DEFAULT_MINIO_SECRET_NAME,
    # LakeFS configuration
    lakefs_endpoint: str = DEFAULT_LAKEFS_ENDPOINT,
    lakefs_repository: str = DEFAULT_LAKEFS_REPOSITORY,
    lakefs_ref: str = DEFAULT_LAKEFS_BRANCH,
    lakefs_secret_name: str = DEFAULT_LAKEFS_SECRET_NAME,
    # Iceberg configuration
    iceberg_catalog: str = DEFAULT_ICEBERG_CATALOG,
    iceberg_database: str = DEFAULT_ICEBERG_DATABASE,
    source_table: str = DEFAULT_ICEBERG_SOURCE_TABLE,
    # MLflow configuration
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    mlflow_model_name: str = DEFAULT_MLFLOW_MODEL_NAME,
    # Feast configuration
    feast_repo_path: str = DEFAULT_FEAST_REPO_PATH,
    feast_project: str = DEFAULT_FEAST_PROJECT,
    feast_feature_view: str = DEFAULT_FEAST_FEATURE_VIEW,
    feature_names_json: str = json.dumps(DEFAULT_FEATURE_NAMES),
    # Model architecture
    latent_dim: int = 16,
    hidden_dims_json: str = "[128, 64]",
    # Training configuration
    batch_size: int = 512,
    max_epochs: int = 10,
    learning_rate: float = 0.001,
    seed: int = 1337,
    max_rows_per_split: int = 0,
    # Monitoring options
    log_level: str = "INFO",
    enable_tensorboard: bool = True,
    # Pipeline options
    create_lakefs_tag: bool = True,
):
    """Full Kronodroid ML pipeline: data transform -> training -> model registration.

    This pipeline optionally runs the Iceberg transformation step before training,
    providing an end-to-end workflow from raw data to registered model.

    Args:
        run_data_transform: Whether to run Iceberg transformation first
        minio_endpoint: MinIO endpoint (for raw data)
        minio_bucket: MinIO bucket for raw data
        minio_prefix: Prefix path for raw data
        minio_secret_name: K8s secret with MinIO credentials
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        lakefs_ref: LakeFS branch, tag, or commit reference
        lakefs_secret_name: K8s secret with LakeFS credentials
        iceberg_catalog: Iceberg catalog name
        iceberg_database: Iceberg database name
        source_table: Source Iceberg table name
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_model_name: Name for model in MLflow registry
        feast_repo_path: Path to Feast feature_store.yaml
        feast_project: Feast project name
        feast_feature_view: Feast feature view name
        feature_names_json: JSON-encoded list of feature column names
        latent_dim: Autoencoder latent space dimension
        hidden_dims_json: JSON-encoded list of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Optimizer learning rate
        seed: Random seed for reproducibility
        max_rows_per_split: Row limit per split (0=unlimited)
        log_level: Python logging level
        enable_tensorboard: Enable TensorBoard logging
        create_lakefs_tag: Create LakeFS tag for data version
    """
    # Step 1: Optionally run data transformation
    # This would import and run the Iceberg pipeline first
    # Currently stubbed - can be enabled by importing kronodroid_iceberg_pipeline

    # Step 2: Train the autoencoder
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
        log_level=log_level,
        enable_tensorboard=enable_tensorboard,
        log_every_n_steps=10,
        enable_gradient_logging=True,
        enable_resource_monitoring=True,
    )

    train_task.set_memory_limit("8Gi")
    train_task.set_memory_request("4Gi")
    train_task.set_cpu_limit("4")
    train_task.set_cpu_request("2")

    use_lakefs_credentials(train_task, secret_name=lakefs_secret_name)

    use_minio_credentials(train_task, secret_name=minio_secret_name)

    # KFP kubernetes volume helpers require a static mount path (not a pipeline parameter).
    mount_feast_repo(
        train_task,
        config_map_name=DEFAULT_FEAST_CONFIGMAP_NAME,
        mount_path=DEFAULT_FEAST_MOUNT_PATH,
    )

    # Step 3: Create LakeFS tag
    # Note: Using hardcoded secret name because KFP v2 doesn't support
    # pipeline parameters inside conditional blocks for kubernetes SDK calls
    with dsl.If(create_lakefs_tag == True):
        tag_task = lakefs_tag_model_data_op(
            lakefs_endpoint=lakefs_endpoint,
            lakefs_repository=lakefs_repository,
            lakefs_commit_id=train_task.outputs["lakefs_commit_id"],
            model_name=train_task.outputs["model_name"],
            model_version=train_task.outputs["model_version"],
        )

        tag_task.set_memory_limit("512Mi")
        tag_task.set_memory_request("256Mi")
        tag_task.set_cpu_limit("500m")

        use_lakefs_credentials(tag_task, secret_name=DEFAULT_LAKEFS_SECRET_NAME)


def compile_pipeline(output_path: str = "kronodroid_autoencoder_training_pipeline.yaml") -> str:
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
    print(f"Pipeline compiled to: {output_path}")
    return output_path


def compile_full_pipeline(output_path: str = "kronodroid_full_training_pipeline.yaml") -> str:
    """Compile the full training pipeline to YAML.

    Args:
        output_path: Path for the compiled pipeline YAML

    Returns:
        Path to the compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=kronodroid_full_training_pipeline,
        package_path=output_path,
    )
    print(f"Full pipeline compiled to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile Kronodroid Autoencoder Training pipelines")
    parser.add_argument(
        "--output",
        default="kronodroid_autoencoder_training_pipeline.yaml",
        help="Output path for compiled pipeline",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Compile the full training pipeline",
    )

    args = parser.parse_args()

    if args.full:
        compile_full_pipeline(args.output)
    else:
        compile_pipeline(args.output)
