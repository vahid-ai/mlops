"""Kubeflow Pipelines components and pipelines for DFP."""

# Note: Components and pipelines are imported lazily to avoid import errors
# when kfp is not installed in all environments.

__all__ = [
    # Spark Kronodroid Iceberg component
    "spark_kronodroid_iceberg_op",
    "run_spark_kronodroid_iceberg",
    # LakeFS commit/merge component
    "lakefs_commit_merge_op",
    "lakefs_commit_only_op",
    "commit_and_merge_lakefs_branch",
    # Autoencoder training component
    "train_kronodroid_autoencoder_op",
    # Pipelines
    "kronodroid_iceberg_pipeline",
    "kronodroid_full_pipeline",
    "compile_pipeline",
    "compile_full_pipeline",
    # Autoencoder training pipeline
    "kronodroid_autoencoder_pipeline",
    "compile_autoencoder_pipeline",
]


def __getattr__(name: str):
    """Lazy import to avoid kfp dependency when not needed."""
    if name in ("spark_kronodroid_iceberg_op", "run_spark_kronodroid_iceberg"):
        from orchestration.kubeflow.dfp_kfp.components.spark_kronodroid_iceberg_component import (
            spark_kronodroid_iceberg_op,
            run_spark_kronodroid_iceberg,
        )
        return spark_kronodroid_iceberg_op if name == "spark_kronodroid_iceberg_op" else run_spark_kronodroid_iceberg

    if name in ("lakefs_commit_merge_op", "lakefs_commit_only_op", "commit_and_merge_lakefs_branch"):
        from orchestration.kubeflow.dfp_kfp.components.lakefs_commit_merge_component import (
            lakefs_commit_merge_op,
            lakefs_commit_only_op,
            commit_and_merge_lakefs_branch,
        )
        if name == "lakefs_commit_merge_op":
            return lakefs_commit_merge_op
        elif name == "lakefs_commit_only_op":
            return lakefs_commit_only_op
        else:
            return commit_and_merge_lakefs_branch

    if name == "train_kronodroid_autoencoder_op":
        from orchestration.kubeflow.dfp_kfp.components.train_autoencoder_component import (
            train_kronodroid_autoencoder_op,
        )
        return train_kronodroid_autoencoder_op

    if name in ("kronodroid_iceberg_pipeline", "kronodroid_full_pipeline", "compile_pipeline", "compile_full_pipeline"):
        from orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_iceberg_pipeline import (
            kronodroid_iceberg_pipeline,
            kronodroid_full_pipeline,
            compile_pipeline,
            compile_full_pipeline,
        )
        if name == "kronodroid_iceberg_pipeline":
            return kronodroid_iceberg_pipeline
        elif name == "kronodroid_full_pipeline":
            return kronodroid_full_pipeline
        elif name == "compile_pipeline":
            return compile_pipeline
        else:
            return compile_full_pipeline

    if name in ("kronodroid_autoencoder_pipeline", "compile_autoencoder_pipeline"):
        from orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_autoencoder_pipeline import (
            kronodroid_autoencoder_pipeline,
            compile_pipeline as compile_autoencoder_pipeline,
        )
        if name == "kronodroid_autoencoder_pipeline":
            return kronodroid_autoencoder_pipeline
        else:
            return compile_autoencoder_pipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

