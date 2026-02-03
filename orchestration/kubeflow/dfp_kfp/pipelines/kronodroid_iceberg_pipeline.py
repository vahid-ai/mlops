"""Kronodroid Iceberg Pipeline via SparkOperator.

This Kubeflow Pipeline replaces the dbt transformation step with:
1. SparkOperator job that reads Parquet from MinIO and writes Iceberg tables
2. LakeFS commit + merge to version control the Iceberg tables
3. (Optional) Feast apply/materialize steps

The pipeline uses per-run LakeFS branches for isolation and data versioning.

Usage:
    # Compile the pipeline
    from kfp import compiler
    compiler.Compiler().compile(
        kronodroid_iceberg_pipeline,
        'kronodroid_iceberg_pipeline.yaml'
    )

    # Submit to KFP
    client = kfp.Client()
    client.create_run_from_pipeline_func(
        kronodroid_iceberg_pipeline,
        arguments={...}
    )
"""

import uuid
from typing import NamedTuple

from kfp import dsl, kubernetes
from kfp.dsl import PipelineTask

from orchestration.kubeflow.dfp_kfp.components.spark_kronodroid_iceberg_component import (
    spark_kronodroid_iceberg_op,
)
from orchestration.kubeflow.dfp_kfp.components.lakefs_commit_merge_component import (
    lakefs_commit_merge_op,
)


# Default configuration values
DEFAULT_MINIO_ENDPOINT = "http://minio:9000"
DEFAULT_MINIO_BUCKET = "dlt-data"
DEFAULT_MINIO_PREFIX = "kronodroid_raw"
DEFAULT_LAKEFS_ENDPOINT = "http://lakefs:8000"
DEFAULT_LAKEFS_REPOSITORY = "kronodroid"
DEFAULT_LAKEFS_BRANCH = "main"
DEFAULT_SPARK_IMAGE = "apache/spark:3.5.0-python3"
DEFAULT_NAMESPACE = "default"
DEFAULT_SERVICE_ACCOUNT = "spark"
DEFAULT_MINIO_SECRET = "minio-credentials"
DEFAULT_LAKEFS_SECRET = "lakefs-credentials"


@dsl.pipeline(
    name="Kronodroid Iceberg Pipeline",
    description="Transform raw Parquet data to Iceberg tables with Avro format via SparkOperator, tracked in LakeFS",
)
def kronodroid_iceberg_pipeline(
    # MinIO configuration
    minio_endpoint: str = DEFAULT_MINIO_ENDPOINT,
    minio_bucket: str = DEFAULT_MINIO_BUCKET,
    minio_prefix: str = DEFAULT_MINIO_PREFIX,
    minio_secret_name: str = DEFAULT_MINIO_SECRET,
    # LakeFS configuration
    lakefs_endpoint: str = DEFAULT_LAKEFS_ENDPOINT,
    lakefs_repository: str = DEFAULT_LAKEFS_REPOSITORY,
    target_branch: str = DEFAULT_LAKEFS_BRANCH,
    lakefs_secret_name: str = DEFAULT_LAKEFS_SECRET,
    # Spark configuration
    spark_image: str = DEFAULT_SPARK_IMAGE,
    namespace: str = DEFAULT_NAMESPACE,
    service_account: str = DEFAULT_SERVICE_ACCOUNT,
    driver_cores: int = 1,
    driver_memory: str = "512m",
    executor_cores: int = 2,
    executor_instances: int = 1,
    executor_memory: str = "512m",
    spark_timeout_seconds: int = 3600,
    # Iceberg configuration
    staging_database: str = "stg_kronodroid",
    marts_database: str = "kronodroid",
    catalog_name: str = "lakefs",
    # Pipeline options
    delete_source_branch: bool = True,
    run_feast_apply: bool = False,
):
    """Run the Kronodroid Iceberg transformation pipeline.

    This pipeline:
    1. Creates a per-run LakeFS branch
    2. Runs Spark job to transform Parquet -> Iceberg (Avro)
    3. Commits and merges the branch to target
    4. (Optional) Applies Feast feature definitions

    Args:
        minio_endpoint: MinIO endpoint URL
        minio_bucket: MinIO bucket containing raw Parquet data
        minio_prefix: Path prefix within the bucket
        minio_secret_name: K8s secret with MinIO credentials
        lakefs_endpoint: LakeFS API endpoint URL
        lakefs_repository: LakeFS repository name
        target_branch: Target LakeFS branch (e.g., main)
        lakefs_secret_name: K8s secret with LakeFS credentials
        spark_image: Docker image for Spark executors
        namespace: Kubernetes namespace for Spark jobs
        service_account: K8s service account for Spark
        driver_cores: Spark driver cores
        driver_memory: Spark driver memory
        executor_cores: Spark executor cores
        executor_instances: Number of Spark executors
        executor_memory: Spark executor memory
        spark_timeout_seconds: Max time to wait for Spark job
        staging_database: Iceberg database for staging tables
        marts_database: Iceberg database for mart tables
        catalog_name: Iceberg catalog name
        delete_source_branch: Delete per-run branch after merge
        run_feast_apply: Whether to run Feast apply after transformation
    """
    # Generate a unique run ID for this pipeline execution
    run_id = "{{workflow.uid}}"  # KFP/Argo provides this

    # Step 1: Run Spark transformation job
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
    ).set_memory_limit("4Gi").set_memory_request("2Gi").set_cpu_limit("2000m").set_cpu_request("500m")

    # Configure credentials for Spark task
    kubernetes.use_secret_as_env(
        task=spark_task,
        secret_name=minio_secret_name,
        secret_key_to_env={
            # Prefer the canonical keys used by infra/k8s/spark-operator/README.md
            # (env var names are set by the mapping destination).
            "access-key": "MINIO_ACCESS_KEY_ID",
            "secret-key": "MINIO_SECRET_ACCESS_KEY",
        },
    )
    kubernetes.use_secret_as_env(
        task=spark_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "access-key": "LAKEFS_ACCESS_KEY_ID",
            "secret-key": "LAKEFS_SECRET_ACCESS_KEY",
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
        pipeline_name="kronodroid-iceberg",
        delete_source_branch=delete_source_branch,
    ).set_memory_limit("2Gi").set_memory_request("512Mi").set_cpu_limit("1000m").set_cpu_request("250m")

    # Set dependency
    commit_merge_task.after(spark_task)

    # Configure LakeFS credentials
    kubernetes.use_secret_as_env(
        task=commit_merge_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "access-key": "LAKEFS_ACCESS_KEY_ID",
            "secret-key": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # Step 3: (Optional) Feast apply - would plug into existing component
    # if run_feast_apply:
    #     feast_task = feast_apply_op(...)
    #     feast_task.after(commit_merge_task)


@dsl.pipeline(
    name="Kronodroid Full Pipeline",
    description="Full pipeline: dlt ingest -> Spark Iceberg transform -> LakeFS commit -> Feast",
)
def kronodroid_full_pipeline(
    # dlt/Ingestion configuration
    skip_ingestion: bool = False,
    kaggle_dataset: str = "dhoogla/kronodroid-2021",
    # MinIO configuration
    minio_endpoint: str = DEFAULT_MINIO_ENDPOINT,
    minio_bucket: str = DEFAULT_MINIO_BUCKET,
    minio_prefix: str = DEFAULT_MINIO_PREFIX,
    minio_secret_name: str = DEFAULT_MINIO_SECRET,
    # LakeFS configuration
    lakefs_endpoint: str = DEFAULT_LAKEFS_ENDPOINT,
    lakefs_repository: str = DEFAULT_LAKEFS_REPOSITORY,
    target_branch: str = DEFAULT_LAKEFS_BRANCH,
    lakefs_secret_name: str = DEFAULT_LAKEFS_SECRET,
    # Spark configuration
    spark_image: str = DEFAULT_SPARK_IMAGE,
    namespace: str = DEFAULT_NAMESPACE,
    service_account: str = DEFAULT_SERVICE_ACCOUNT,
    driver_cores: int = 1,
    driver_memory: str = "512m",
    executor_cores: int = 2,
    executor_instances: int = 1,
    executor_memory: str = "512m",
    spark_timeout_seconds: int = 3600,
    # Iceberg configuration
    staging_database: str = "stg_kronodroid",
    marts_database: str = "kronodroid",
    catalog_name: str = "lakefs",
    # Feast configuration
    run_feast_apply: bool = True,
    materialize_days: int = 30,
):
    """Full Kronodroid pipeline: ingest -> transform -> version -> featurize.

    This is the complete pipeline that mirrors run_kronodroid_pipeline.py but
    runs entirely on Kubeflow with SparkOperator for transformations.

    Args:
        skip_ingestion: Skip dlt ingestion step (data already in MinIO)
        kaggle_dataset: Kaggle dataset identifier
        minio_endpoint: MinIO endpoint URL
        minio_bucket: MinIO bucket for raw data
        minio_prefix: Path prefix for raw data
        minio_secret_name: K8s secret with MinIO credentials
        lakefs_endpoint: LakeFS API endpoint
        lakefs_repository: LakeFS repository
        target_branch: Target LakeFS branch
        lakefs_secret_name: K8s secret with LakeFS credentials
        spark_image: Spark executor image
        namespace: K8s namespace
        service_account: K8s service account
        driver_cores: Spark driver cores
        driver_memory: Spark driver memory
        executor_cores: Spark executor cores
        executor_instances: Number of executors
        executor_memory: Executor memory
        spark_timeout_seconds: Spark job timeout
        staging_database: Iceberg staging database
        marts_database: Iceberg marts database
        catalog_name: Iceberg catalog name
        run_feast_apply: Run Feast apply step
        materialize_days: Days of features to materialize
    """
    run_id = "{{workflow.uid}}"

    # Step 1: dlt ingestion (optional)
    # if not skip_ingestion:
    #     ingest_task = dlt_ingest_op(
    #         kaggle_dataset=kaggle_dataset,
    #         minio_endpoint=minio_endpoint,
    #         minio_bucket=minio_bucket,
    #         output_prefix=minio_prefix,
    #     )

    # Step 2: Spark transformation
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
    ).set_memory_limit("4Gi").set_memory_request("2Gi").set_cpu_limit("2000m").set_cpu_request("500m")

    # Configure MinIO credentials for Spark task
    kubernetes.use_secret_as_env(
        task=spark_task,
        secret_name=minio_secret_name,
        secret_key_to_env={
            "MINIO_ACCESS_KEY_ID": "MINIO_ACCESS_KEY_ID",
            "MINIO_SECRET_ACCESS_KEY": "MINIO_SECRET_ACCESS_KEY",
        },
    )
    # Configure LakeFS credentials for Spark task
    kubernetes.use_secret_as_env(
        task=spark_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
            "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # Step 3: LakeFS commit + merge
    commit_merge_task = lakefs_commit_merge_op(
        lakefs_endpoint=lakefs_endpoint,
        lakefs_repository=lakefs_repository,
        source_branch=spark_task.outputs["lakefs_branch"],
        target_branch=target_branch,
        commit_message=f"Kronodroid Iceberg transformation - run {run_id}",
        run_id=run_id,
        pipeline_name="kronodroid-full",
        delete_source_branch=True,
    ).set_memory_limit("2Gi").set_memory_request("512Mi").set_cpu_limit("1000m").set_cpu_request("250m")

    kubernetes.use_secret_as_env(
        task=commit_merge_task,
        secret_name=lakefs_secret_name,
        secret_key_to_env={
            "LAKEFS_ACCESS_KEY_ID": "LAKEFS_ACCESS_KEY_ID",
            "LAKEFS_SECRET_ACCESS_KEY": "LAKEFS_SECRET_ACCESS_KEY",
        },
    )

    # Step 4: Feast apply (optional) - plug into existing stubs
    # if run_feast_apply:
    #     feast_apply_task = feast_apply_op(...)
    #     feast_apply_task.after(commit_merge_task)
    #
    #     feast_materialize_task = feast_materialize_op(
    #         days_back=materialize_days,
    #     )
    #     feast_materialize_task.after(feast_apply_task)


def compile_pipeline(output_path: str = "kronodroid_iceberg_pipeline.yaml") -> str:
    """Compile the pipeline to YAML.

    Args:
        output_path: Path for the compiled pipeline YAML

    Returns:
        Path to the compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=kronodroid_iceberg_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to: {output_path}")
    return output_path


def compile_full_pipeline(
    output_path: str = "kronodroid_full_pipeline.yaml",
) -> str:
    """Compile the full pipeline to YAML.

    Args:
        output_path: Path for the compiled pipeline YAML

    Returns:
        Path to the compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=kronodroid_full_pipeline,
        package_path=output_path,
    )
    print(f"Full pipeline compiled to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile Kronodroid Iceberg pipelines")
    parser.add_argument(
        "--output",
        default="kronodroid_iceberg_pipeline.yaml",
        help="Output path for compiled pipeline",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Compile the full pipeline instead of just the Iceberg transform",
    )

    args = parser.parse_args()

    if args.full:
        compile_full_pipeline(args.output)
    else:
        compile_pipeline(args.output)
