"""ExecuTorch model exporter.

Converts PyTorch models to ExecuTorch format (.pte) for mobile deployment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExportMetadata:
    """Metadata about an exported ExecuTorch model."""

    model_name: str
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]
    input_dtypes: list[str]
    output_dtypes: list[str]
    backend: str
    quantization: str
    export_version: str = "1.0.0"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "input_dtypes": self.input_dtypes,
            "output_dtypes": self.output_dtypes,
            "backend": self.backend,
            "quantization": self.quantization,
            "export_version": self.export_version,
            **self.extra,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExportMetadata":
        with open(path) as f:
            data = json.load(f)
        return cls(
            model_name=data["model_name"],
            input_shapes=data["input_shapes"],
            output_shapes=data["output_shapes"],
            input_dtypes=data["input_dtypes"],
            output_dtypes=data["output_dtypes"],
            backend=data["backend"],
            quantization=data["quantization"],
            export_version=data.get("export_version", "1.0.0"),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "model_name",
                    "input_shapes",
                    "output_shapes",
                    "input_dtypes",
                    "output_dtypes",
                    "backend",
                    "quantization",
                    "export_version",
                }
            },
        )


class ExecuTorchExporter:
    """Exports PyTorch models to ExecuTorch format.

    Supports multiple backends (XNNPACK, Vulkan, CoreML) and quantization options.

    Example:
        >>> exporter = ExecuTorchExporter(backend="xnnpack")
        >>> exporter.export(model, example_inputs, "model.pte")
    """

    SUPPORTED_BACKENDS = {"xnnpack", "vulkan", "coreml", "qnn", "portable"}

    def __init__(
        self,
        backend: str = "xnnpack",
        quantization: str = "none",
        optimize: bool = True,
    ):
        """Initialize the exporter.

        Args:
            backend: Target backend for optimization. One of:
                - "xnnpack": Optimized CPU inference (recommended for most cases)
                - "vulkan": GPU inference via Vulkan
                - "coreml": Apple CoreML (iOS/macOS only)
                - "qnn": Qualcomm QNN
                - "portable": No backend-specific optimizations
            quantization: Quantization mode. One of:
                - "none": No quantization (fp32)
                - "dynamic": Dynamic quantization (int8 weights)
                - "static": Static quantization (requires calibration data)
                - "qat": Quantization-aware training (model must be QAT-trained)
            optimize: Whether to apply graph optimizations.
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.quantization = quantization
        self.optimize = optimize

    def _get_partitioner(self):
        """Get the appropriate partitioner for the selected backend."""
        if self.backend == "xnnpack":
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )

            return [XnnpackPartitioner()]
        elif self.backend == "vulkan":
            from executorch.backends.vulkan.partition.vulkan_partitioner import (
                VulkanPartitioner,
            )

            return [VulkanPartitioner()]
        elif self.backend == "coreml":
            from executorch.backends.apple.coreml.partition.coreml_partitioner import (
                CoreMLPartitioner,
            )

            return [CoreMLPartitioner()]
        elif self.backend == "qnn":
            from executorch.backends.qualcomm.partition.qnn_partitioner import (
                QnnPartitioner,
            )

            return [QnnPartitioner()]
        else:
            # Portable - no partitioner
            return []

    def _apply_quantization(
        self, model: nn.Module, example_inputs: tuple
    ) -> nn.Module:
        """Apply quantization to the model if configured."""
        if self.quantization == "none":
            return model

        if self.quantization == "dynamic":
            from torch.ao.quantization import quantize_dynamic

            return quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8,
            )

        elif self.quantization == "static":
            # Static quantization requires calibration
            logger.warning(
                "Static quantization requires calibration data. "
                "Falling back to dynamic quantization."
            )
            from torch.ao.quantization import quantize_dynamic

            return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        elif self.quantization == "qat":
            # QAT model should already be quantized during training
            logger.info("Using QAT model as-is (assumes model was QAT-trained)")
            return model

        else:
            logger.warning(f"Unknown quantization mode: {self.quantization}")
            return model

    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
        output_path: str | Path,
        model_name: str | None = None,
        save_metadata: bool = True,
    ) -> ExportMetadata:
        """Export a PyTorch model to ExecuTorch format.

        Args:
            model: The PyTorch model to export.
            example_inputs: Example inputs for tracing. Should be a tuple of tensors.
            output_path: Path to save the .pte file.
            model_name: Optional name for the model (used in metadata).
            save_metadata: Whether to save metadata JSON alongside the .pte file.

        Returns:
            ExportMetadata with information about the exported model.
        """
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_name = model_name or model.__class__.__name__

        logger.info(f"Exporting model '{model_name}' to ExecuTorch format")
        logger.info(f"Backend: {self.backend}, Quantization: {self.quantization}")

        # Prepare model
        model = model.eval()

        # Apply quantization if configured
        model = self._apply_quantization(model, example_inputs)

        # Step 1: Export with torch.export
        logger.info("Step 1/3: Exporting with torch.export...")
        exported_program = torch.export.export(model, example_inputs)

        # Step 2: Convert to edge format with backend partitioning
        logger.info("Step 2/3: Converting to edge format...")
        partitioners = self._get_partitioner()

        edge_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=partitioners if partitioners else None,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=True,
            ),
        )

        # Step 3: Convert to ExecuTorch program and save
        logger.info("Step 3/3: Generating ExecuTorch program...")
        et_program = edge_program.to_executorch()

        with open(output_path, "wb") as f:
            f.write(et_program.buffer)

        logger.info(f"Saved ExecuTorch model to: {output_path}")

        # Collect metadata
        input_shapes = [list(t.shape) for t in example_inputs]
        input_dtypes = [str(t.dtype).replace("torch.", "") for t in example_inputs]

        # Run inference to get output shapes
        with torch.no_grad():
            outputs = model(*example_inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            output_shapes = [list(o.shape) for o in outputs]
            output_dtypes = [str(o.dtype).replace("torch.", "") for o in outputs]

        metadata = ExportMetadata(
            model_name=model_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            backend=self.backend,
            quantization=self.quantization,
            extra={
                "file_size_bytes": output_path.stat().st_size,
                "optimize": self.optimize,
            },
        )

        if save_metadata:
            metadata_path = output_path.with_suffix(".json")
            metadata.save(metadata_path)
            logger.info(f"Saved metadata to: {metadata_path}")

        return metadata

    def validate(
        self,
        model: nn.Module,
        pte_path: str | Path,
        example_inputs: tuple[torch.Tensor, ...],
        tolerance: float = 1e-4,
    ) -> bool:
        """Validate that ExecuTorch model produces similar outputs to PyTorch.

        Args:
            model: Original PyTorch model.
            pte_path: Path to the exported .pte file.
            example_inputs: Example inputs for validation.
            tolerance: Maximum allowed difference between outputs.

        Returns:
            True if outputs are within tolerance, False otherwise.
        """
        from executorch.runtime import Runtime

        # Get PyTorch output
        model.eval()
        with torch.no_grad():
            pt_output = model(*example_inputs)
            if isinstance(pt_output, torch.Tensor):
                pt_output = pt_output.numpy()

        # Get ExecuTorch output
        runtime = Runtime.get()
        program = runtime.load_program(str(pte_path))
        method = program.load_method("forward")

        inputs = [t.numpy() for t in example_inputs]
        et_output = method.execute(inputs)

        # Compare outputs
        max_diff = abs(pt_output - et_output).max()
        is_valid = max_diff <= tolerance

        if is_valid:
            logger.info(f"Validation passed. Max difference: {max_diff:.6f}")
        else:
            logger.error(
                f"Validation failed. Max difference: {max_diff:.6f} > {tolerance}"
            )

        return is_valid
