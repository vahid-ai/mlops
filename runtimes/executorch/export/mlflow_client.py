"""MLflow integration for ExecuTorch models.

Provides utilities to fetch PyTorch models from MLflow and register
ExecuTorch exports back to the model registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata about a model fetched from MLflow."""

    run_id: str
    model_name: str
    version: str
    params: dict[str, Any]
    metrics: dict[str, float]
    tags: dict[str, str]
    artifact_uri: str


class MLflowExecuTorchClient:
    """Client for MLflow operations related to ExecuTorch models.

    Handles:
    - Fetching PyTorch models from MLflow model registry
    - Registering ExecuTorch .pte files as artifacts
    - Logging comparison results and validation metrics
    """

    def __init__(self, tracking_uri: str = "http://localhost:5050"):
        """Initialize the MLflow client.

        Args:
            tracking_uri: MLflow tracking server URI.
        """
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.tracking_uri = tracking_uri
        self._mlflow = mlflow

    def fetch_pytorch_model(
        self,
        model_name: str,
        version: str = "latest",
        stage: str | None = None,
    ) -> tuple[nn.Module, ModelMetadata]:
        """Fetch a PyTorch model from MLflow model registry.

        Args:
            model_name: Name of the registered model.
            version: Model version number, or "latest".
            stage: Model stage (e.g., "Production", "Staging"). Overrides version.

        Returns:
            Tuple of (model, metadata).
        """
        # Determine model URI
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Fetching model '{model_name}' stage '{stage}'")
        elif version == "latest":
            model_uri = f"models:/{model_name}/latest"
            logger.info(f"Fetching latest version of model '{model_name}'")
        else:
            model_uri = f"models:/{model_name}/{version}"
            logger.info(f"Fetching model '{model_name}' version {version}")

        # Load model
        model = self._mlflow.pytorch.load_model(model_uri)

        # Get metadata
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"No model found for stage '{stage}'")
            model_version = versions[0]
        elif version == "latest":
            versions = self.client.get_latest_versions(model_name)
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            model_version = versions[0]
        else:
            model_version = self.client.get_model_version(model_name, version)

        run = self.client.get_run(model_version.run_id)

        metadata = ModelMetadata(
            run_id=model_version.run_id,
            model_name=model_name,
            version=model_version.version,
            params=run.data.params,
            metrics=run.data.metrics,
            tags=run.data.tags,
            artifact_uri=run.info.artifact_uri,
        )

        logger.info(
            f"Loaded model from run {metadata.run_id}, version {metadata.version}"
        )

        return model, metadata

    def register_executorch_model(
        self,
        pte_path: str | Path,
        source_run_id: str,
        model_name: str,
        export_metadata: dict[str, Any],
        metadata_path: str | Path | None = None,
    ) -> str:
        """Register an ExecuTorch model as an artifact in MLflow.

        The .pte file is logged as an artifact under the original training run,
        keeping all artifacts (PyTorch model, ExecuTorch model, metrics) together.

        Args:
            pte_path: Path to the .pte file.
            source_run_id: The original training run ID.
            model_name: Name of the model.
            export_metadata: Metadata about the export (backend, quantization, etc.).
            metadata_path: Optional path to metadata JSON file.

        Returns:
            The artifact URI for the registered .pte file.
        """
        pte_path = Path(pte_path)

        with self._mlflow.start_run(run_id=source_run_id):
            # Log the .pte file
            self._mlflow.log_artifact(str(pte_path), artifact_path="executorch")
            logger.info(f"Logged .pte artifact: executorch/{pte_path.name}")

            # Log metadata JSON if provided
            if metadata_path:
                self._mlflow.log_artifact(
                    str(metadata_path), artifact_path="executorch"
                )

            # Log export parameters
            export_params = {
                f"executorch.{k}": str(v) for k, v in export_metadata.items()
            }
            self._mlflow.log_params(export_params)

            # Set tags
            self._mlflow.set_tag("executorch.exported", "true")
            self._mlflow.set_tag("executorch.backend", export_metadata.get("backend"))
            self._mlflow.set_tag(
                "executorch.quantization", export_metadata.get("quantization")
            )

        artifact_uri = f"runs:/{source_run_id}/executorch/{pte_path.name}"
        logger.info(f"Registered ExecuTorch model at: {artifact_uri}")

        return artifact_uri

    def download_executorch_model(
        self,
        run_id: str,
        output_dir: str | Path,
        model_filename: str = "model.pte",
    ) -> Path:
        """Download an ExecuTorch model from MLflow.

        Args:
            run_id: The run ID containing the ExecuTorch model.
            output_dir: Directory to save the downloaded model.
            model_filename: Name of the .pte file in the artifacts.

        Returns:
            Path to the downloaded .pte file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = f"executorch/{model_filename}"
        local_path = self.client.download_artifacts(
            run_id, artifact_path, str(output_dir)
        )

        logger.info(f"Downloaded ExecuTorch model to: {local_path}")
        return Path(local_path)

    def log_validation_results(
        self,
        run_id: str,
        pytorch_results: dict[str, float],
        executorch_results: dict[str, float],
        comparison_metrics: dict[str, float],
        tolerance: float,
    ) -> None:
        """Log validation comparison results to MLflow.

        Args:
            run_id: The run ID to log to.
            pytorch_results: Metrics from PyTorch model.
            executorch_results: Metrics from ExecuTorch model.
            comparison_metrics: Computed comparison metrics.
            tolerance: The tolerance used for comparison.
        """
        with self._mlflow.start_run(run_id=run_id):
            # Log PyTorch baseline metrics
            for name, value in pytorch_results.items():
                self._mlflow.log_metric(f"validation.pytorch.{name}", value)

            # Log ExecuTorch metrics
            for name, value in executorch_results.items():
                self._mlflow.log_metric(f"validation.executorch.{name}", value)

            # Log comparison metrics
            for name, value in comparison_metrics.items():
                self._mlflow.log_metric(f"validation.{name}", value)

            # Log tolerance and overall result
            self._mlflow.log_param("validation.tolerance", tolerance)
            self._mlflow.set_tag(
                "validation.passed", str(comparison_metrics.get("passed", False))
            )

        logger.info(f"Logged validation results to run {run_id}")

    def log_android_benchmark_results(
        self,
        run_id: str,
        benchmark_results: dict[str, Any],
    ) -> None:
        """Log Android benchmark results to MLflow.

        Args:
            run_id: The run ID to log to.
            benchmark_results: Benchmark results from Android test.
        """
        with self._mlflow.start_run(run_id=run_id):
            # Log device info as tags
            device_info = benchmark_results.get("deviceInfo", {})
            for key, value in device_info.items():
                self._mlflow.set_tag(f"android.device.{key}", str(value))

            # Log performance metrics
            self._mlflow.log_metric(
                "android.avg_latency_ms", benchmark_results.get("avgLatencyMs", 0)
            )
            self._mlflow.log_metric(
                "android.total_samples", benchmark_results.get("totalSamples", 0)
            )

            # Log individual sample latencies as artifact
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(benchmark_results, f, indent=2)
                results_path = f.name

            self._mlflow.log_artifact(results_path, artifact_path="android_benchmark")

            self._mlflow.set_tag("android.tested", "true")

        logger.info(f"Logged Android benchmark results to run {run_id}")

    def create_model_comparison_run(
        self,
        experiment_name: str,
        pytorch_run_id: str,
        model_name: str,
        description: str | None = None,
    ) -> str:
        """Create a new run for model comparison experiments.

        Args:
            experiment_name: Name of the experiment.
            pytorch_run_id: Original PyTorch training run ID.
            model_name: Name of the model being compared.
            description: Optional description.

        Returns:
            The new run ID.
        """
        # Get or create experiment
        experiment = self._mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self._mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Create run
        with self._mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{model_name}_validation",
            description=description,
        ) as run:
            self._mlflow.set_tag("source_run_id", pytorch_run_id)
            self._mlflow.set_tag("model_name", model_name)
            self._mlflow.set_tag("run_type", "model_validation")

            return run.info.run_id
