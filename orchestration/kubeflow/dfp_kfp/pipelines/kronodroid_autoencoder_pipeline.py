"""Kronodroid autoencoder training pipeline (Spark -> LakeFS commit -> Feast -> Lightning -> MLflow)."""

from __future__ import annotations

import json

from kfp import dsl

from orchestration.kubeflow.dfp_kfp.components.lakefs_commit_merge_component import (
    lakefs_commit_merge_op,
)
from orchestration.kubeflow.dfp_kfp.components.spark_kronodroid_iceberg_component import (
    spark_kronodroid_iceberg_op,
)
from orchestration.kubeflow.dfp_kfp.components.train_kronodroid_autoencoder_component import (
    train_kronodroid_autoencoder_op,
)


@dsl.pipeline(
    name="Kronodroid Autoencoder Pipeline",
    description="Build Avro/Iceberg feature tables in LakeFS, then train/test/register a Lightning autoencoder via Feast + MLflow",
)
def kronodroid_autoencoder_pipeline(
    # Data plane (Spark/LakeFS)
    run_transform: bool = True,
    minio_endpoint: str = "http://minio:9000",
    minio_bucket: str = "dlt-data",
    minio_prefix: str = "kronodroid_raw",
    minio_secret_name: str = "minio-credentials",
    lakefs_endpoint: str = "http://lakefs:8000",
    lakefs_repository: str = "kronodroid",
    target_branch: str = "main",
    lakefs_secret_name: str = "lakefs-credentials",
    spark_image: str = "apache/spark:3.5.0-python3",
    namespace: str = "default",
    service_account: str = "spark",
    staging_database: str = "stg_kronodroid",
    marts_database: str = "kronodroid",
    catalog_name: str = "lakefs",
    # Feast (data access)
    feature_store_yaml_path: str = "feature_stores/feast_store/feature_store_spark.yaml",
    feast_project: str = "dfp",
    feast_feature_view: str = "kronodroid_autoencoder_features",
    feature_names_json: str = json.dumps([f"syscall_{i}_normalized" for i in range(1, 21)] + ["syscall_total", "syscall_mean"]),
    feast_definitions_paths_json: str = json.dumps(
        [
            "feature_stores/feast_store/dfp_feast/entities.py",
            "feature_stores/feast_store/dfp_feast/kronodroid_features.py",
        ]
    ),
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000",
    mlflow_experiment_name: str = "kronodroid-autoencoder",
    mlflow_model_name: str = "kronodroid_autoencoder",
    # Training
    max_rows_per_split: int = 0,
    latent_dim: int = 16,
    hidden_dims_json: str = "[128, 64]",
    batch_size: int = 512,
    max_epochs: int = 10,
    seed: int = 1337,
):
    run_id = "{{workflow.uid}}"

    if run_transform:
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
        )

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
        commit_merge_task.after(spark_task)

        lakefs_ref = commit_merge_task.outputs["merge_commit_id"]
    else:
        lakefs_ref = target_branch

    train_task = train_kronodroid_autoencoder_op(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_model_name=mlflow_model_name,
        feature_store_yaml_path=feature_store_yaml_path,
        lakefs_repository=lakefs_repository,
        lakefs_ref=lakefs_ref,
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

    # Credentials for Feast (Spark->LakeFS) and MLflow artifact logging (S3/MinIO-compatible).
    train_task.set_env_variable(name="LAKEFS_ENDPOINT_URL", value=lakefs_endpoint)
    train_task.set_env_variable(
        name="LAKEFS_ACCESS_KEY_ID",
        value_from_secret=dsl.V1EnvVarSource(
            secret_key_ref=dsl.V1SecretKeySelector(name=lakefs_secret_name, key="access-key")
        ),
    )
    train_task.set_env_variable(
        name="LAKEFS_SECRET_ACCESS_KEY",
        value_from_secret=dsl.V1EnvVarSource(
            secret_key_ref=dsl.V1SecretKeySelector(name=lakefs_secret_name, key="secret-key")
        ),
    )

    # MLflow artifact store (MinIO). If your MLflow server uses a different artifact backend,
    # you can remove/override these environment variables.
    train_task.set_env_variable(name="MLFLOW_S3_ENDPOINT_URL", value=minio_endpoint)
    train_task.set_env_variable(
        name="AWS_ACCESS_KEY_ID",
        value_from_secret=dsl.V1EnvVarSource(
            secret_key_ref=dsl.V1SecretKeySelector(name=minio_secret_name, key="access-key")
        ),
    )
    train_task.set_env_variable(
        name="AWS_SECRET_ACCESS_KEY",
        value_from_secret=dsl.V1EnvVarSource(
            secret_key_ref=dsl.V1SecretKeySelector(name=minio_secret_name, key="secret-key")
        ),
    )


def compile_kronodroid_autoencoder_pipeline(output_path: str = "kronodroid_autoencoder_pipeline.yaml") -> str:
    from kfp import compiler

    compiler.Compiler().compile(kronodroid_autoencoder_pipeline, output_path)
    return output_path
