import pytest


pytest.importorskip("kfp")


def test_compile_kronodroid_iceberg_pipeline(tmp_path):
    from kfp import compiler

    from orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_iceberg_pipeline import (
        kronodroid_iceberg_pipeline,
    )

    output_path = tmp_path / "kronodroid_iceberg_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=kronodroid_iceberg_pipeline,
        package_path=str(output_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_compile_kronodroid_autoencoder_training_pipeline(tmp_path):
    from kfp import compiler

    from orchestration.kubeflow.dfp_kfp.pipelines.kronodroid_autoencoder_training_pipeline import (
        kronodroid_autoencoder_training_pipeline,
    )

    output_path = tmp_path / "kronodroid_autoencoder_training_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=kronodroid_autoencoder_training_pipeline,
        package_path=str(output_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
