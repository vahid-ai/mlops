#!/usr/bin/env python3
"""CLI for ExecuTorch model export and MLflow integration.

Usage:
    # Export a model from MLflow to ExecuTorch
    uv run python -m runtimes.executorch.export.cli export \\
        --model-name kronodroid_autoencoder \\
        --output models/kronodroid_autoencoder.pte

    # Export with quantization
    uv run python -m runtimes.executorch.export.cli export \\
        --model-name kronodroid_autoencoder \\
        --output models/kronodroid_autoencoder_int8.pte \\
        --quantization dynamic_int8

    # List available quantization configs
    uv run python -m runtimes.executorch.export.cli list-quantization

    # Validate exported model
    uv run python -m runtimes.executorch.export.cli validate \\
        --model-name kronodroid_autoencoder \\
        --pte-path models/kronodroid_autoencoder.pte
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from .exporter import ExecuTorchExporter
from .mlflow_client import MLflowExecuTorchClient
from .quantization import get_quantization_config, list_quantization_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_export(args: argparse.Namespace) -> int:
    """Export a PyTorch model to ExecuTorch format."""
    client = MLflowExecuTorchClient(args.mlflow_uri)

    # Fetch model from MLflow
    logger.info(f"Fetching model '{args.model_name}' from MLflow...")
    model, metadata = client.fetch_pytorch_model(
        args.model_name,
        version=args.model_version,
    )

    # Determine input shape
    if args.input_shape:
        input_shape = [int(x) for x in args.input_shape.split(",")]
    else:
        # Try to infer from model params
        input_dim = metadata.params.get("input_dim")
        if input_dim:
            input_shape = [1, int(input_dim)]
        else:
            logger.error(
                "Could not determine input shape. "
                "Please provide --input-shape (e.g., '1,289')"
            )
            return 1

    logger.info(f"Using input shape: {input_shape}")
    example_inputs = (torch.randn(*input_shape),)

    # Get quantization config
    quant_config = get_quantization_config(args.quantization)
    logger.info(f"Quantization: {quant_config.mode.value}")

    # Export
    exporter = ExecuTorchExporter(
        backend=args.backend,
        quantization=quant_config.mode.value,
    )

    export_metadata = exporter.export(
        model=model,
        example_inputs=example_inputs,
        output_path=args.output,
        model_name=args.model_name,
    )

    # Register in MLflow if requested
    if args.register:
        logger.info("Registering ExecuTorch model in MLflow...")
        artifact_uri = client.register_executorch_model(
            pte_path=args.output,
            source_run_id=metadata.run_id,
            model_name=args.model_name,
            export_metadata=export_metadata.to_dict(),
            metadata_path=Path(args.output).with_suffix(".json"),
        )
        logger.info(f"Registered at: {artifact_uri}")

    logger.info("Export complete!")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Size: {export_metadata.extra.get('file_size_bytes', 0):,} bytes")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate an exported ExecuTorch model against PyTorch."""
    client = MLflowExecuTorchClient(args.mlflow_uri)

    # Fetch original model
    logger.info(f"Fetching model '{args.model_name}' from MLflow...")
    model, metadata = client.fetch_pytorch_model(
        args.model_name,
        version=args.model_version,
    )

    # Determine input shape
    if args.input_shape:
        input_shape = [int(x) for x in args.input_shape.split(",")]
    else:
        input_dim = metadata.params.get("input_dim")
        if input_dim:
            input_shape = [1, int(input_dim)]
        else:
            logger.error("Could not determine input shape.")
            return 1

    example_inputs = (torch.randn(*input_shape),)

    # Validate
    exporter = ExecuTorchExporter()
    is_valid = exporter.validate(
        model=model,
        pte_path=args.pte_path,
        example_inputs=example_inputs,
        tolerance=args.tolerance,
    )

    if is_valid:
        logger.info("✓ Validation PASSED")
        return 0
    else:
        logger.error("✗ Validation FAILED")
        return 1


def cmd_list_quantization(args: argparse.Namespace) -> int:
    """List available quantization configurations."""
    configs = list_quantization_configs()

    print("\nAvailable quantization configurations:")
    print("=" * 50)

    for name in configs:
        config = get_quantization_config(name)
        print(f"\n{name}:")
        print(f"  Mode: {config.mode.value}")
        print(f"  Dtype: {config.dtype}")
        print(f"  Backend: {config.backend.value}")
        if config.mode.value == "static":
            print(f"  Calibration samples: {config.calibration_samples}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about an exported model."""
    from .exporter import ExportMetadata

    pte_path = Path(args.pte_path)
    metadata_path = pte_path.with_suffix(".json")

    if not pte_path.exists():
        logger.error(f"Model file not found: {pte_path}")
        return 1

    print(f"\nModel: {pte_path}")
    print(f"Size: {pte_path.stat().st_size:,} bytes")

    if metadata_path.exists():
        metadata = ExportMetadata.load(metadata_path)
        print(f"\nMetadata:")
        print(f"  Name: {metadata.model_name}")
        print(f"  Backend: {metadata.backend}")
        print(f"  Quantization: {metadata.quantization}")
        print(f"  Input shapes: {metadata.input_shapes}")
        print(f"  Output shapes: {metadata.output_shapes}")
        print(f"  Input dtypes: {metadata.input_dtypes}")
        print(f"  Output dtypes: {metadata.output_dtypes}")
    else:
        print(f"\nNo metadata file found at: {metadata_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ExecuTorch model export CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5050",
        help="MLflow tracking server URI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export PyTorch model to ExecuTorch"
    )
    export_parser.add_argument(
        "--model-name", required=True, help="MLflow registered model name"
    )
    export_parser.add_argument(
        "--model-version", default="latest", help="Model version (default: latest)"
    )
    export_parser.add_argument(
        "--output", "-o", required=True, help="Output .pte file path"
    )
    export_parser.add_argument(
        "--backend",
        default="xnnpack",
        choices=["xnnpack", "vulkan", "coreml", "qnn", "portable"],
        help="Target backend (default: xnnpack)",
    )
    export_parser.add_argument(
        "--quantization",
        default="none",
        help="Quantization config name (default: none)",
    )
    export_parser.add_argument(
        "--input-shape",
        help="Input shape as comma-separated values (e.g., '1,289')",
    )
    export_parser.add_argument(
        "--register",
        action="store_true",
        help="Register exported model in MLflow",
    )
    export_parser.set_defaults(func=cmd_export)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate exported model against PyTorch"
    )
    validate_parser.add_argument(
        "--model-name", required=True, help="MLflow registered model name"
    )
    validate_parser.add_argument(
        "--model-version", default="latest", help="Model version"
    )
    validate_parser.add_argument(
        "--pte-path", required=True, help="Path to .pte file to validate"
    )
    validate_parser.add_argument(
        "--input-shape", help="Input shape as comma-separated values"
    )
    validate_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max allowed difference (default: 1e-4)",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # List quantization command
    list_quant_parser = subparsers.add_parser(
        "list-quantization", help="List available quantization configs"
    )
    list_quant_parser.set_defaults(func=cmd_list_quantization)

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show information about an exported model"
    )
    info_parser.add_argument("pte_path", help="Path to .pte file")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
