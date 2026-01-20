"""Kronodroid autoencoder training pipeline (Spark -> LakeFS commit -> Feast -> Lightning -> MLflow)."""

import json

from kfp import dsl
from kfp import kubernetes

from orchestration.kubeflow.dfp_kfp.components.lakefs_commit_merge_component import (
    lakefs_commit_merge_op,
)
from orchestration.kubeflow.dfp_kfp.components.spark_kronodroid_iceberg_component import (
    spark_kronodroid_iceberg_op,
)
from orchestration.kubeflow.dfp_kfp.components.train_kronodroid_autoencoder_component import (
    train_kronodroid_autoencoder_op,
)


# Default feature names for the autoencoder
DEFAULT_FEATURE_NAMES = json.dumps(
    [f"syscall_{i}_normalized" for i in range(1, 21)] + ["syscall_total", "syscall_mean"]
)
DEFAULT_FEAST_DEFINITION_PATHS = json.dumps([
    "feature_stores/feast_store/dfp_feast/entities.py",
    "feature_stores/feast_store/dfp_feast/kronodroid_features.py",
])


@dsl.pipeline(
    name="Kronodroid Autoencoder Training Pipeline",
    description="Train a Lightning autoencoder using Feast features from existing LakeFS data, register in MLflow",
)
def kronodroid_autoencoder_training_pipeline(
    # LakeFS data source
    lakefs_endpoint: str = "http://lakefs:8000",
    lakefs_repository: str = "kronodroid",
    lakefs_ref: str = "main",
    lakefs_secret_name: str = "lakefs-credentials",
    iceberg_catalog: str = "lakefs",
    iceberg_database: str = "kronodroid",
    source_table: str = "fct_training_dataset",
    # Feast (data access)
    feature_store_yaml_path: str = "feature_stores/feast_store/feature_store_spark.yaml",
    feast_project: str = "dfp",
    feast_feature_view: str = "kronodroid_autoencoder_features",
    feature_names_json: str = DEFAULT_FEATURE_NAMES,
    feast_definitions_paths_json: str = DEFAULT_FEAST_DEFINITION_PATHS,
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000",
    mlflow_experiment_name: str = "kronodroid-autoencoder",
    mlflow_model_name: str = "kronodroid_autoencoder",
    minio_endpoint: str = "http://minio:9000",
    minio_secret_name: str = "minio-credentials",
    # Training hyperparameters
    max_rows_per_split: int = 0,
    latent_dim: int = 16,
    hidden_dims_json: str = "[128, 64]",
    batch_size: int = 512,
    max_epochs: int = 10,
    seed: int = 1337,
):
    """Training-only pipeline that assumes data is already transformed in LakeFS.

    Use this pipeline when you have already run the Spark transformation step
    and just want to train/retrain the autoencoder model.
    """
    train_task = train_kronodroid_autoencoder_op(
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
        feature_names_json=feature_names_json,
        feast_definitions_paths_json=feast_definitions_paths_json,
        max_rows_per_split=max_rows_per_split,
        latent_dim=latent_dim,
        hidden_dims_json=hidden_dims_json,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
    )

    train_task.set_display_name("Train/Test/Register Autoencoder")

    # LakeFS credentials for Feast (Spark->LakeFS)
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
            "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # MLflow artifact store (MinIO)
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name=minio_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
    )


@dsl.pipeline(
    name="Kronodroid Autoencoder Full Pipeline",
    description="Transform data with Spark, commit to LakeFS, then train autoencoder and register in MLflow",
)
def kronodroid_autoencoder_full_pipeline(
    # Spark/MinIO configuration
    minio_endpoint: str = "http://minio:9000",
    minio_bucket: str = "dlt-data",
    minio_prefix: str = "kronodroid_raw",
    minio_secret_name: str = "minio-credentials",
    # LakeFS configuration
    lakefs_endpoint: str = "http://lakefs:8000",
    lakefs_repository: str = "kronodroid",
    target_branch: str = "main",
    lakefs_secret_name: str = "lakefs-credentials",
    # Spark job configuration
    spark_image: str = "dfp-spark:latest",
    namespace: str = "dfp",
    service_account: str = "spark",
    staging_database: str = "stg_kronodroid",
    marts_database: str = "kronodroid",
    catalog_name: str = "lakefs",
    driver_cores: int = 1,
    driver_memory: str = "2g",
    executor_cores: int = 2,
    executor_instances: int = 2,
    executor_memory: str = "2g",
    spark_timeout_seconds: int = 3600,
    # Feast configuration
    feature_store_yaml_path: str = "feature_stores/feast_store/feature_store_spark.yaml",
    feast_project: str = "dfp",
    feast_feature_view: str = "kronodroid_autoencoder_features",
    feature_names_json: str = DEFAULT_FEATURE_NAMES,
    feast_definitions_paths_json: str = DEFAULT_FEAST_DEFINITION_PATHS,
    # MLflow configuration
    mlflow_tracking_uri: str = "http://mlflow:5000",
    mlflow_experiment_name: str = "kronodroid-autoencoder",
    mlflow_model_name: str = "kronodroid_autoencoder",
    # Training hyperparameters
    max_rows_per_split: int = 0,
    latent_dim: int = 16,
    hidden_dims_json: str = "[128, 64]",
    batch_size: int = 512,
    max_epochs: int = 10,
    seed: int = 1337,
):
    """Full pipeline: Spark transformation -> LakeFS commit -> Autoencoder training.

    Use this pipeline to run the complete workflow from raw data to trained model.
    """
    run_id = "{{workflow.uid}}"

    # Step 1: Spark transformation (Parquet -> Iceberg)
    spark_task = spark_kronodroid_iceberg_op(
        run_id=run_id,
        minio_endpoint=minio_endpoint,
        minio_bucket=minio_bucket,
        minio_prefix=minio_prefix,
        lakefs_endpoint=lakefs_endpoint,
        lakefs_repository=lakefs_repository,
        target_branch=target_branch,
        spark_image=spark_image,
        namespace=namespace,
        service_account=service_account,
        minio_secret_name=minio_secret_name,
        lakefs_secret_name=lakefs_secret_name,
        staging_database=staging_database,
        marts_database=marts_database,
        catalog_name=catalog_name,
        driver_cores=driver_cores,
        driver_memory=driver_memory,
        executor_cores=executor_cores,
        executor_instances=executor_instances,
        executor_memory=executor_memory,
        timeout_seconds=spark_timeout_seconds,
    )
    spark_task.set_display_name("Spark: Transform to Iceberg")

    # Inject LakeFS credentials for branch creation
    kubernetes.use_secret_as_env(
        task=spark_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
            "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # Step 2: Commit and merge LakeFS branch
    commit_merge_task = lakefs_commit_merge_op(
        lakefs_endpoint=lakefs_endpoint,
        lakefs_repository=lakefs_repository,
        source_branch=spark_task.outputs["lakefs_branch"],
        target_branch=target_branch,
        commit_message=f"Kronodroid Iceberg transformation - run {run_id}",
        run_id=run_id,
        pipeline_name="kronodroid-autoencoder",
        delete_source_branch=True,
    )
    commit_merge_task.set_display_name("LakeFS: Commit & Merge")
    commit_merge_task.after(spark_task)

    kubernetes.use_secret_as_env(
        task=commit_merge_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
            "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # Step 3: Train autoencoder
    train_task = train_kronodroid_autoencoder_op(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_model_name=mlflow_model_name,
        feature_store_yaml_path=feature_store_yaml_path,
        lakefs_repository=lakefs_repository,
        lakefs_ref=commit_merge_task.outputs["merge_commit_id"],
        iceberg_catalog=catalog_name,
        iceberg_database=marts_database,
        source_table="fct_training_dataset",
        feast_project=feast_project,
        feast_feature_view=feast_feature_view,
        feature_names_json=feature_names_json,
        feast_definitions_paths_json=feast_definitions_paths_json,
        max_rows_per_split=max_rows_per_split,
        latent_dim=latent_dim,
        hidden_dims_json=hidden_dims_json,
        batch_size=batch_size,
        max_epochs=max_epochs,
        seed=seed,
    )
    train_task.set_display_name("Train/Test/Register Autoencoder")
    train_task.after(commit_merge_task)

    # LakeFS credentials for Feast
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
            "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # MLflow artifact store (MinIO)
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name=minio_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
    )


# Backwards compatibility alias
kronodroid_autoencoder_pipeline = kronodroid_autoencoder_training_pipeline


def compile_kronodroid_autoencoder_pipeline(
    output_path: str = "kronodroid_autoencoder_pipeline.yaml",
    full_pipeline: bool = False,
) -> str:
    """Compile the pipeline to YAML.

    Args:
        output_path: Path for the compiled pipeline YAML
        full_pipeline: If True, compile the full pipeline (with Spark transform);
                      if False, compile training-only pipeline

    Returns:
        Path to the compiled pipeline
    """
    from kfp import compiler

    pipeline_func = (
        kronodroid_autoencoder_full_pipeline if full_pipeline
        else kronodroid_autoencoder_training_pipeline
    )

    compiler.Compiler().compile(pipeline_func, output_path)
    print(f"Pipeline compiled to: {output_path}")
    return output_path
