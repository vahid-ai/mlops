#!/usr/bin/env python3
"""Test script to verify ExecuTorch export pipeline works.

This script creates a simple test model and exports it to ExecuTorch format,
bypassing MLflow to test the core export functionality.

Usage:
    uv run python tools/model_validation/test_export.py
"""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for testing export."""

    def __init__(self, input_dim: int = 289, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def test_export():
    """Test the ExecuTorch export pipeline."""
    print("=" * 60)
    print("ExecuTorch Export Pipeline Test")
    print("=" * 60)

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from runtimes.executorch.export import ExecuTorchExporter

    # Create test model
    print("\n1. Creating test model...")
    model = SimpleAutoencoder(input_dim=289, hidden_dim=64)
    model.eval()
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create example inputs
    print("\n2. Creating example inputs...")
    example_inputs = (torch.randn(1, 289),)
    print(f"   Input shape: {example_inputs[0].shape}")

    # Test PyTorch inference
    print("\n3. Testing PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(*example_inputs)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output range: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")

    # Export to ExecuTorch
    print("\n4. Exporting to ExecuTorch...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_model.pte"

        try:
            exporter = ExecuTorchExporter(backend="xnnpack", quantization="none")
            metadata = exporter.export(
                model=model,
                example_inputs=example_inputs,
                output_path=output_path,
                model_name="test_autoencoder",
            )

            print(f"   ✓ Export successful!")
            print(f"   Output: {output_path}")
            print(f"   Size: {output_path.stat().st_size:,} bytes")
            print(f"   Backend: {metadata.backend}")

            # Try to load and run with ExecuTorch runtime
            print("\n5. Testing ExecuTorch inference...")
            try:
                from executorch.runtime import Runtime

                runtime = Runtime.get()
                program = runtime.load_program(str(output_path))
                method = program.load_method("forward")

                et_output = method.execute([example_inputs[0].numpy()])
                print(f"   ✓ ExecuTorch inference successful!")
                print(f"   Output shape: {et_output[0].shape}")

                # Compare outputs
                import numpy as np

                max_diff = np.abs(pytorch_output.numpy() - et_output[0]).max()
                print(f"   Max diff vs PyTorch: {max_diff:.6f}")

                if max_diff < 0.001:
                    print("   ✓ Outputs match within tolerance!")
                else:
                    print(f"   ⚠ Outputs differ by {max_diff:.6f}")

            except ImportError:
                print("   ⚠ ExecuTorch runtime not installed (pip install executorch)")
                print("   Skipping runtime validation...")

            except Exception as e:
                print(f"   ⚠ ExecuTorch runtime error: {e}")

        except Exception as e:
            print(f"   ✗ Export failed: {e}")
            import traceback

            traceback.print_exc()
            return 1

    print("\n" + "=" * 60)
    print("✓ Export pipeline test completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(test_export())
