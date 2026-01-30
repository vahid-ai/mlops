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


def _convert_lightning_to_pytorch(lightning_model: nn.Module) -> nn.Module:
    """Convert a Lightning model to a pure PyTorch model.

    Lightning models saved via MLflow can have issues during inference
    due to how the class is pickled. This function extracts the state dict
    and loads it into a pure PyTorch equivalent.

    Args:
        lightning_model: The loaded Lightning model.

    Returns:
        A pure PyTorch model with the same weights.
    """
    # Extract state dict
    state_dict = lightning_model.state_dict()

    # Check if it has encoder/decoder structure (autoencoder)
    has_encoder = any(k.startswith("encoder.") for k in state_dict.keys())
    has_decoder = any(k.startswith("decoder.") for k in state_dict.keys())

    if has_encoder and has_decoder:
        # Infer architecture from state dict
        # Encoder structure: Linear, ReLU, BatchNorm1d, Linear, ReLU, BatchNorm1d, ..., Linear
        # Layer indices: 0=Linear, 1=ReLU, 2=BatchNorm, 3=Linear, 4=ReLU, 5=BatchNorm, 6=Linear (final)
        # Linear layers have 2D weights, BatchNorm has 1D weights

        # Find all Linear layer indices in encoder (2D weight tensors)
        encoder_linear_indices = sorted([
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("encoder.") and k.endswith(".weight") and state_dict[k].dim() == 2
        ])

        # Get dimensions
        input_dim = state_dict["encoder.0.weight"].shape[1]
        # Last encoder linear is the latent projection
        last_encoder_idx = encoder_linear_indices[-1]
        latent_dim = state_dict[f"encoder.{last_encoder_idx}.weight"].shape[0]

        # Hidden dims are from all encoder linears except the last one
        hidden_dims = []
        for idx in encoder_linear_indices[:-1]:
            hidden_dims.append(state_dict[f"encoder.{idx}.weight"].shape[0])

        logger.info(f"Detected autoencoder: input={input_dim}, hidden={hidden_dims}, latent={latent_dim}")

        class PureAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dims):
                super().__init__()
                # Build encoder
                encoder_layers = []
                prev_dim = input_dim
                for h_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, h_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(h_dim),
                    ])
                    prev_dim = h_dim
                encoder_layers.append(nn.Linear(prev_dim, latent_dim))
                self.encoder = nn.Sequential(*encoder_layers)

                # Build decoder
                decoder_layers = []
                prev_dim = latent_dim
                for h_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, h_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(h_dim),
                    ])
                    prev_dim = h_dim
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = PureAutoencoder(input_dim, latent_dim, tuple(hidden_dims))
        # Filter state dict to only encoder/decoder
        filtered_state = {k: v for k, v in state_dict.items() if k.startswith(("encoder", "decoder"))}
        model.load_state_dict(filtered_state)
        return model

    # Fallback: return original model
    logger.warning("Could not convert Lightning model, returning original")
    return lightning_model


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

        # Convert Lightning models to pure PyTorch for compatibility
        model_class_name = type(model).__name__
        if "Lightning" in model_class_name or hasattr(model, "trainer"):
            logger.info(f"Converting Lightning model ({model_class_name}) to pure PyTorch")
            model = _convert_lightning_to_pytorch(model)

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
